"""Language capability table.

Sarvam's services support different language sets, so every output step checks
here before choosing a model. Values reflect docs.sarvam.ai as of Nov 2026 —
re-check before the demo; Sarvam updates these often.
"""
from __future__ import annotations

from dataclasses import dataclass

# Speech-to-text (saaras:v4): 22 scheduled languages + English.
STT_LANGS = {
    "hi-IN", "bn-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN", "en-IN",
    "gu-IN", "as-IN", "ur-IN", "ne-IN", "kok-IN", "ks-IN", "sd-IN", "sa-IN", "sat-IN", "mni-IN",
    "brx-IN", "mai-IN", "doi-IN",
}

# Languages sarvam-105b handles well on the API (10 Indic + English).
LLM_LANGS = {"hi-IN", "bn-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN", "gu-IN", "en-IN"}

# Text-to-speech. bulbul:v4-flash lacks Malayalam and Odia -> use bulbul:v3 for those.
TTS_V4_LANGS = {"hi-IN", "bn-IN", "kn-IN", "mr-IN", "pa-IN", "ta-IN", "te-IN", "gu-IN", "en-IN"}
TTS_V3_LANGS = TTS_V4_LANGS | {"ml-IN", "od-IN"}

# Sarvam-Translate: 22 + English.
TRANSLATE_LANGS = STT_LANGS

NAMES = {
    "hi-IN": "Hindi", "bn-IN": "Bengali", "kn-IN": "Kannada", "ml-IN": "Malayalam", "mr-IN": "Marathi",
    "od-IN": "Odia", "pa-IN": "Punjabi", "ta-IN": "Tamil", "te-IN": "Telugu", "en-IN": "English",
    "gu-IN": "Gujarati", "as-IN": "Assamese", "ur-IN": "Urdu", "ne-IN": "Nepali", "kok-IN": "Konkani",
    "ks-IN": "Kashmiri", "sd-IN": "Sindhi", "sa-IN": "Sanskrit", "sat-IN": "Santali", "mni-IN": "Manipuri",
    "brx-IN": "Bodo", "mai-IN": "Maithili", "doi-IN": "Dogri",
}

# Font family per script for PDF rendering (Noto fonts in app/assets/fonts).
FONT_FOR_LANG = {
    "kn-IN": "Noto Sans Kannada", "hi-IN": "Noto Sans Devanagari", "mr-IN": "Noto Sans Devanagari",
    "ne-IN": "Noto Sans Devanagari", "kok-IN": "Noto Sans Devanagari", "mai-IN": "Noto Sans Devanagari",
    "doi-IN": "Noto Sans Devanagari", "sa-IN": "Noto Sans Devanagari", "brx-IN": "Noto Sans Devanagari",
    "ta-IN": "Noto Sans Tamil", "te-IN": "Noto Sans Telugu", "ml-IN": "Noto Sans Malayalam",
    "bn-IN": "Noto Sans Bengali", "as-IN": "Noto Sans Bengali", "gu-IN": "Noto Sans Gujarati",
    "pa-IN": "Noto Sans Gurmukhi", "od-IN": "Noto Sans Oriya",
}


@dataclass(frozen=True)
class LanguagePlan:
    code: str
    name: str
    stt: bool
    llm_native: bool  # can sarvam-105b write directly in this language?
    tts_model: str | None  # None -> text-only output

    def as_dict(self) -> dict:
        return {
            "code": self.code, "name": self.name, "speech_input": self.stt,
            "llm_native": self.llm_native, "speech_output": self.tts_model is not None,
            "tts_model": self.tts_model,
        }


def normalize(code: str | None) -> str:
    if not code:
        return "en-IN"
    code = code.strip()
    if code in STT_LANGS:
        return code
    short = code.split("-")[0].lower()
    for c in STT_LANGS:
        if c.split("-")[0] == short:
            return c
    return "en-IN"


def plan_for(code: str | None, tts_model: str = "bulbul:v4-flash", tts_fallback: str = "bulbul:v3") -> LanguagePlan:
    code = normalize(code)
    if code in TTS_V4_LANGS:
        tts = tts_model
    elif code in TTS_V3_LANGS:
        tts = tts_fallback
    else:
        tts = None
    return LanguagePlan(code, NAMES.get(code, code), code in STT_LANGS, code in LLM_LANGS, tts)


def all_plans(tts_model: str = "bulbul:v4-flash", tts_fallback: str = "bulbul:v3") -> list[dict]:
    return [plan_for(c, tts_model, tts_fallback).as_dict() for c in sorted(STT_LANGS)]
