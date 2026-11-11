"""Regression tests for the live-eval failures:

    empty LLM content (attempt 1; finish_reason=length)
    LLM call failed: chat_v1: network error: ReadTimeout
    falling back to baseline extractor
"""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest

from app.config import Settings
from app.models.session import SessionState, TranscriptSegment
from app.services import gateway as gw_mod
from app.services.extraction import extract_incident
from app.services.gateway import ChatResult, SarvamGateway, SarvamServiceError
from app.services.mock_gateway import MockGateway

TEXT = "I sent Rs 50000 to cbi.verify2@ybl after a CBI officer called from 9876543210."


def make_session() -> SessionState:
    now = datetime.now(timezone.utc)
    s = SessionState(id="s1", expires_at=now + timedelta(hours=1), consent_at=now, language_code="en-IN")
    s.segments.append(TranscriptSegment(id="t1", text=TEXT))
    return s


class ScriptedGateway:
    """Returns/raises the scripted outcomes in order, recording max_tokens per call."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.max_tokens: list[int | None] = []
        self._mock = MockGateway()

    async def chat(self, messages, *, model, schema, usage, api="v1", reasoning_effort="default", max_tokens=None):
        self.max_tokens.append(max_tokens)
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        if o == "ok":
            return await self._mock.chat(messages, model=model, schema=schema, usage=usage, max_tokens=max_tokens)
        return o


@pytest.fixture()
def cfg(tmp_path):
    return Settings(mock_mode=True, env="test", data_dir=tmp_path)


def run(coro):
    return asyncio.run(coro)


def test_length_then_success_retries_with_bigger_budget(cfg):
    gw = ScriptedGateway([ChatResult("", 100, cfg.llm_max_tokens, "length"), "ok"])
    inc, method = run(extract_incident(make_session(), gw, cfg))
    assert method == "llm"
    assert gw.max_tokens == [cfg.llm_max_tokens, cfg.llm_retry_max_tokens]
    assert gw.max_tokens[1] > gw.max_tokens[0]
    assert inc.scam_type.value != "unknown" or inc.transactions


def test_length_twice_falls_back_to_baseline(cfg):
    empty = ChatResult("", 100, 6000, "length")
    gw = ScriptedGateway([empty, empty])
    inc, method = run(extract_incident(make_session(), gw, cfg))
    assert method == "baseline"
    assert any(f.value == "+919876543210" for f in inc.suspect_phones)  # baseline still extracts


def test_truncated_json_retries(cfg):
    gw = ScriptedGateway([ChatResult('{"scam_type": "digital_arr', 100, 6000, "length"), "ok"])
    _, method = run(extract_incident(make_session(), gw, cfg))
    assert method == "llm"
    assert len(gw.max_tokens) == 2


def test_timeout_falls_back_without_crashing(cfg):
    gw = ScriptedGateway([SarvamServiceError("chat_v1", 504, "network error: ReadTimeout")])
    _, method = run(extract_incident(make_session(), gw, cfg))
    assert method == "baseline"
    assert len(gw.max_tokens) == 1  # no pointless second long call after a timeout


def test_auth_errors_still_raise(cfg):
    gw = ScriptedGateway([SarvamServiceError("chat_v1", 401, "bad key")])
    with pytest.raises(SarvamServiceError):
        run(extract_incident(make_session(), gw, cfg))


# ---------------------------------------------------------------- SarvamGateway


class FakeCompletions:
    def __init__(self, behaviours):
        self.behaviours = list(behaviours)
        self.calls: list[dict] = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        b = self.behaviours.pop(0)
        if isinstance(b, Exception):
            raise b
        return b


def make_gateway(cfg, completions) -> SarvamGateway:
    g = object.__new__(SarvamGateway)
    g.s = cfg
    g.sem = asyncio.Semaphore(2)
    g.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return g


def ok_response():
    msg = SimpleNamespace(content='{"a": 1}')
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                           usage=SimpleNamespace(prompt_tokens=5, completion_tokens=7))


@pytest.fixture()
def no_sleep(monkeypatch):
    async def instant(_):
        return None

    monkeypatch.setattr(gw_mod.asyncio, "sleep", instant)


def test_chat_passes_llm_timeout(cfg):
    comp = FakeCompletions([ok_response()])
    g = make_gateway(cfg, comp)
    res = run(g.chat([{"role": "user", "content": "x"}], model="sarvam-105b", schema=None,
                     usage=make_session().usage))
    assert res.content == '{"a": 1}' and res.finish_reason == "stop"
    assert comp.calls[0]["request_options"] == {"timeout_in_seconds": int(cfg.llm_timeout_s)}
    assert cfg.llm_timeout_s > cfg.sarvam_timeout_s  # chat gets longer than the 60s STT/TTS default


def test_timeouts_are_retried_at_most_once(cfg, no_sleep):
    cfg = cfg.model_copy(update={"sarvam_max_retries": 3})
    comp = FakeCompletions([httpx.ReadTimeout("t")] * 3)
    g = make_gateway(cfg, comp)
    with pytest.raises(SarvamServiceError) as ei:
        run(g.chat([{"role": "user", "content": "x"}], model="sarvam-105b", schema=None,
                   usage=make_session().usage))
    assert ei.value.status == 504 and "ReadTimeout" in ei.value.message
    assert len(comp.calls) == gw_mod.MAX_TIMEOUT_ATTEMPTS


def test_timeout_then_success_recovers(cfg, no_sleep):
    comp = FakeCompletions([httpx.ReadTimeout("t"), ok_response()])
    g = make_gateway(cfg, comp)
    res = run(g.chat([{"role": "user", "content": "x"}], model="sarvam-105b", schema=None,
                     usage=make_session().usage))
    assert res.content and len(comp.calls) == 2


def test_connect_errors_keep_full_retry_budget(cfg, no_sleep):
    cfg = cfg.model_copy(update={"sarvam_max_retries": 3})
    comp = FakeCompletions([httpx.ConnectError("c")] * 3)
    g = make_gateway(cfg, comp)
    with pytest.raises(SarvamServiceError):
        run(g.chat([{"role": "user", "content": "x"}], model="sarvam-105b", schema=None,
                   usage=make_session().usage))
    assert len(comp.calls) == 3
