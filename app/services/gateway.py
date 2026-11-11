"""Thin, typed gateway over Sarvam AI's APIs.

All Sarvam traffic goes through `SarvamGateway`, which adds retries with
backoff, a concurrency cap, timing metrics and per-session usage counting.
`MockGateway` (services/mock_gateway.py) implements the same interface with
deterministic fake responses for offline development and tests.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from app.config import Settings
from app.core.metrics import metrics
from app.models.session import UsageCounters

log = logging.getLogger("goldenhour.sarvam")

RETRYABLE = {408, 409, 429, 500, 502, 503, 504}
# Chat Completions v1 only accepts these; the string "none" is a 400.
_CHAT_V1_EFFORTS = {"low", "medium", "high"}


def resolve_chat_reasoning_effort(model: str, reasoning_effort: str | None, configured: str) -> str | None:
    """Return the value to send, or None to omit the field.

    Gemma 4 reasons only when the field is set, so None/none omits it.
    Sarvam 105B's v1 endpoint requires low|medium|high; omitting the field
    can spend the completion budget on hidden reasoning before any JSON.
    """
    effort = configured if reasoning_effort == "default" else reasoning_effort
    if effort in (None, "none"):
        if str(model).startswith("sarvam-105b"):
            return configured if configured in _CHAT_V1_EFFORTS else "low"
        return None
    return effort


class SarvamServiceError(Exception):
    def __init__(self, op: str, status: int | None, message: str):
        super().__init__(f"{op}: {message}")
        self.op, self.status, self.message = op, status, message

    @property
    def user_message(self) -> str:
        if self.status in (401, 403):
            return "The AI service rejected the API key. Check SARVAM_API_KEY."
        if self.status == 402:
            return "Sarvam API credits are exhausted. Top up or switch to another team member's key."
        if self.status == 429:
            return "The AI service is rate-limiting us. Please try again in a few seconds."
        if self.status in (400, 422):
            return f"The AI service could not process this input ({self.op})."
        return "The AI service is temporarily unavailable. Please try again."


@dataclass
class STTResult:
    transcript: str
    language_code: str | None
    language_probability: float | None = None


@dataclass
class ChatResult:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str | None = None


class Gateway(Protocol):
    async def transcribe(self, wav: bytes, *, language_code: str | None, keyterms: list[str],
                         usage: UsageCounters, duration_s: float, purpose: str = "narration") -> STTResult: ...

    async def doc_extract(self, image: bytes, *, filename: str, mime: str, schema: dict,
                          usage: UsageCounters) -> dict: ...

    async def chat(self, messages: list[dict], *, model: str, schema: dict | None, usage: UsageCounters,
                   api: str = "v1", reasoning_effort: str | None = "default", max_tokens: int | None = None) -> ChatResult: ...

    async def tts(self, text: str, *, language_code: str, model: str, usage: UsageCounters) -> tuple[bytes, str]: ...

    async def translate(self, text: str, *, source: str, target: str, usage: UsageCounters) -> str: ...


class SarvamGateway:
    def __init__(self, settings: Settings):
        from sarvamai import AsyncSarvamAI

        self.s = settings
        self.client = AsyncSarvamAI(
            api_subscription_key=settings.sarvam_api_key.get_secret_value(),
            timeout=settings.sarvam_timeout_s,
        )
        self.sem = asyncio.Semaphore(settings.sarvam_max_concurrency)

    async def _call(self, op: str, fn, usage: UsageCounters | None):
        from sarvamai.core.api_error import ApiError

        delay = 1.0
        for attempt in range(1, self.s.sarvam_max_retries + 1):
            t0 = time.perf_counter()
            try:
                async with self.sem:
                    result = await fn()
                metrics.record(op, (time.perf_counter() - t0) * 1000)
                if usage:
                    usage.sarvam_calls += 1
                return result
            except ApiError as e:
                status = e.status_code
                body = str(e.body)[:300]
                metrics.record(op, (time.perf_counter() - t0) * 1000, f"{status}: {body}")
                if usage:
                    usage.sarvam_errors += 1
                if status in RETRYABLE and attempt < self.s.sarvam_max_retries:
                    retry_after = (e.headers or {}).get("retry-after")
                    wait = float(retry_after) if retry_after and retry_after.isdigit() else delay
                    log.warning("%s failed with %s (attempt %d); retrying in %.1fs", op, status, attempt, wait)
                    await asyncio.sleep(wait + random.random() * 0.3)
                    delay *= 2
                    continue
                raise SarvamServiceError(op, status, body) from e
            except (httpx.TimeoutException, httpx.TransportError) as e:
                metrics.record(op, (time.perf_counter() - t0) * 1000, type(e).__name__)
                if usage:
                    usage.sarvam_errors += 1
                if attempt < self.s.sarvam_max_retries:
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                raise SarvamServiceError(op, 504, f"network error: {type(e).__name__}") from e
        raise SarvamServiceError(op, None, "retries exhausted")  # pragma: no cover

    # ------------------------------------------------------------------ STT
    async def transcribe(self, wav, *, language_code, keyterms, usage, duration_s, purpose="narration"):
        kwargs: dict[str, Any] = dict(
            file=("chunk.wav", wav, "audio/wav"), model=self.s.stt_model, mode=self.s.stt_mode,
            language_code=language_code or "unknown", input_audio_codec="wav",
        )
        if keyterms:
            kwargs["keyterms"] = keyterms[:40]
        res = await self._call("stt", lambda: self.client.speech_to_text.transcribe(**kwargs), usage)
        usage.stt_seconds += duration_s
        return STTResult(res.transcript or "", res.language_code, res.language_probability)

    # -------------------------------------------------------------- Doc AI
    async def doc_extract(self, image, *, filename, mime, schema, usage):
        start = await self._call("doc_ai_start", lambda: self.client.doc_ai.extract(
            file=[(filename, image, mime)], schema=json.dumps(schema), output_format="json"), usage)
        job_id = start.job_id
        deadline = time.monotonic() + self.s.doc_ai_timeout_s
        status = start.status
        while status not in ("completed", "partially_completed", "failed", "rejected"):
            if time.monotonic() > deadline:
                raise SarvamServiceError("doc_ai", 504, f"job {job_id} still '{status}' after {self.s.doc_ai_timeout_s}s")
            await asyncio.sleep(1.0)
            st = await self._call("doc_ai_status", lambda: self.client.doc_ai.get_status(job_id), None)
            status = st.status
        if status in ("failed", "rejected"):
            raise SarvamServiceError("doc_ai", 422, f"job {job_id} {status}")
        res = await self._call("doc_ai_results", lambda: self.client.doc_ai.get_results(job_id), None)
        usage.doc_pages += 1
        return {"result": dict(res.result or {}), "annotations": dict(getattr(res, "annotations", {}) or {})}

    # ---------------------------------------------------------------- Chat
    async def chat(self, messages, *, model, schema, usage, api="v1", reasoning_effort="default", max_tokens=None):
        kwargs: dict[str, Any] = dict(
            messages=messages, model=model, temperature=self.s.llm_temperature,
            max_tokens=max_tokens or self.s.llm_max_tokens,
        )
        effort = resolve_chat_reasoning_effort(model, reasoning_effort, self.s.llm_reasoning_effort)
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        if schema:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": schema}
        fn = self.client.chat.completions_v2 if api == "v2" else self.client.chat.completions
        res = await self._call(f"chat_{api}", lambda: fn(**kwargs), usage)
        content = (res.choices[0].message.content or "").strip() if res.choices else ""
        finish_reason = res.choices[0].finish_reason if res.choices else None
        in_tok = getattr(res.usage, "prompt_tokens", 0) if res.usage else 0
        out_tok = getattr(res.usage, "completion_tokens", 0) if res.usage else 0
        usage.llm_input_tokens += in_tok or 0
        usage.llm_output_tokens += out_tok or 0
        return ChatResult(content, in_tok or 0, out_tok or 0, finish_reason)

    # ----------------------------------------------------------------- TTS
    async def tts(self, text, *, language_code, model, usage):
        kwargs: dict[str, Any] = dict(text=text, language_code=language_code, model=model,
                                      pace=self.s.tts_pace, output_audio_codec="mp3")
        if self.s.tts_speaker:
            kwargs["speaker"] = self.s.tts_speaker
        res = await self._call("tts", lambda: self.client.text_to_speech.convert(**kwargs), usage)
        usage.tts_chars += len(text)
        return b"".join(base64.b64decode(a) for a in res.audios), "audio/mpeg"

    # ----------------------------------------------------------- Translate
    async def translate(self, text, *, source, target, usage):
        res = await self._call("translate", lambda: self.client.text.translate(
            input=text, source_language_code=source, target_language_code=target,
            model=self.s.translate_model), usage)
        usage.translate_chars += len(text)
        return res.translated_text


def build_gateway(settings: Settings) -> Gateway:
    if settings.mock_mode:
        from app.services.mock_gateway import MockGateway

        return MockGateway()
    return SarvamGateway(settings)
