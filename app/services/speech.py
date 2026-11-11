"""Speech output and translation helpers (chunking, language fallbacks)."""
from __future__ import annotations

import re

from app.config import Settings
from app.core.languages import TRANSLATE_LANGS, plan_for
from app.models.session import UsageCounters
from app.services.gateway import Gateway

TTS_CHUNK_CHARS = 1500  # below Bulbul's 2,500-char REST limit
TRANSLATE_CHUNK_CHARS = 1800  # below Sarvam-Translate's 2,000-char limit


def split_text(text: str, limit: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?।॥])\s+|\n+", text.strip())
    chunks, cur = [], ""
    for s in sentences:
        if not s:
            continue
        while len(s) > limit:  # pathological long sentence
            chunks.append(s[:limit])
            s = s[limit:]
        if len(cur) + len(s) + 1 > limit:
            chunks.append(cur.strip())
            cur = s
        else:
            cur = f"{cur} {s}" if cur else s
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


async def translate(text: str, target: str, gw: Gateway, usage: UsageCounters, source: str = "en-IN") -> str:
    if not text or target == source or target not in TRANSLATE_LANGS:
        return text
    parts = [await gw.translate(c, source=source, target=target, usage=usage)
             for c in split_text(text, TRANSLATE_CHUNK_CHARS)]
    return " ".join(parts)


async def translate_protected(text: str, target: str, gw: Gateway, usage: UsageCounters,
                              protect: list[str]) -> str:
    """Translate while keeping identifiers (UTRs, UPI IDs, phone numbers) byte-exact.

    Identifiers are swapped for numbered placeholders before translation and
    restored afterwards. If the translator drops a placeholder, the original
    English sentence is kept for safety rather than risk a mangled number.
    """
    if not text or target == "en-IN":
        return text
    values = sorted({p for p in protect if p}, key=len, reverse=True)
    masked, mapping = text, {}
    for i, v in enumerate(values):
        token = f"[[{i}]]"
        if v in masked:
            masked = masked.replace(v, token)
            mapping[token] = v
    out = await translate(masked, target, gw, usage)
    if any(tok not in out for tok in mapping):
        return text  # safer to show English than a corrupted identifier
    for tok, v in mapping.items():
        out = out.replace(tok, v)
    return out


async def speak(text: str, language: str, gw: Gateway, settings: Settings,
                usage: UsageCounters) -> tuple[bytes, str] | None:
    """Return audio bytes, or None when the language has no voice (text-only output)."""
    plan = plan_for(language, settings.tts_model, settings.tts_fallback_model)
    if not plan.tts_model or not text.strip():
        return None
    audio, mime = b"", "audio/mpeg"
    for chunk in split_text(text, TTS_CHUNK_CHARS):
        part, mime = await gw.tts(chunk, language_code=plan.code, model=plan.tts_model, usage=usage)
        audio += part
    return audio, mime
