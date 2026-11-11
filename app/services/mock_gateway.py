"""Deterministic offline stand-in for Sarvam (MOCK_MODE=true).

Lets the frontend and tests run the full flow without an API key or credits.
The demo story is a Kannada-English "digital arrest" case with two payments.
"""
from __future__ import annotations

import json

from app.core import baseline_extractor
from app.models.incident import Fact, Incident
from app.models.session import UsageCounters
from app.services.audio import silent_wav
from app.services.extraction import SOURCES_END, SOURCES_START
from app.services.gateway import ChatResult, STTResult

NARRATION = [
    "ನಮಸ್ಕಾರ, ನನ್ನ ಹೆಸರು Ramesh. ಇವತ್ತು ಬೆಳಿಗ್ಗೆ 10 ಗಂಟೆಗೆ ಒಬ್ಬರು call ಮಾಡಿದ್ರು, ಅವರು CBI officer ಅಂತ "
    "ಹೇಳಿದ್ರು. ನನ್ನ Aadhaar ಮೇಲೆ money laundering case ಇದೆ, digital arrest ಮಾಡ್ತೀವಿ ಅಂತ ಹೆದರಿಸಿದ್ರು.",
    "ಅವರು video call ನಲ್ಲಿ police uniform ಹಾಕಿಕೊಂಡಿದ್ರು. Verification ಗೆ ಹಣ ಕಳಿಸಿ ಅಂತ ಹೇಳಿದ್ರು. ನಾನು PhonePe "
    "ಇಂದ ₹50,000 cbi.verify2@ybl ಗೆ ಕಳಿಸಿದೆ. ಆಮೇಲೆ ಇನ್ನೊಂದು ₹1,20,000 ಕಳಿಸಿದೆ.",
    "ಅವರ number 9876543210. ನನ್ನ account SBI ದು. ಈಗ ಅವರು phone ತೆಗೀತಿಲ್ಲ.",
]

ANSWERS = [
    "ಎರಡನೇ payment ದು UTR 612345098765",
    "2026-11-11 11:05",
    "ಕೊನೆಯ ನಾಲ್ಕು ಅಂಕಿ 4821",
]

SCREENSHOT = {
    "is_payment_receipt": "yes",
    "status": "Payment Successful",
    "amount": "₹50,000.00",
    "datetime": "11 Nov 2026, 10:42 am",
    "utr": "412398765432",
    "app_txn_id": "T2611111042123456789012",
    "payee_name": "CBI Verification Cell",
    "payee_upi": "cbi.verify2@ybl",
    "payee_account": None,
    "payee_ifsc": None,
    "payer_bank": "State Bank of India",
    "payer_account_last4": "4821",
    "payment_app": "PhonePe",
    "visible_text": "Payment Successful ₹50,000.00 Paid to CBI Verification Cell cbi.verify2@ybl "
                    "11 Nov 2026, 10:42 am UTR: 412398765432 Debited from State Bank of India XX4821",
}


def incident_to_llm_json(inc: Incident) -> dict:
    def f(x: Fact) -> dict:
        return {"value": x.value, "source_id": x.source_id, "quote": x.quote}

    return {
        "scam_type": inc.scam_type.value, "scam_type_evidence": f(inc.scam_type_evidence),
        "first_contact_channel": inc.first_contact_channel.value,
        "impersonated_entity": f(inc.impersonated_entity), "incident_date": f(inc.incident_date),
        "victim_name": f(inc.victim_name), "victim_bank": f(inc.victim_bank),
        "victim_account_last4": f(inc.victim_account_last4),
        "suspect_phones": [f(x) for x in inc.suspect_phones], "suspect_upi_ids": [f(x) for x in inc.suspect_upi_ids],
        "suspect_accounts": [f(x) for x in inc.suspect_accounts], "suspect_links": [f(x) for x in inc.suspect_links],
        "transactions": [{k: f(getattr(t, k)) for k in ("amount", "datetime", "utr", "payee_upi", "payee_account",
                                                        "payee_ifsc", "payment_mode", "payment_app")}
                         for t in inc.transactions],
        "narrative_en": inc.narrative_en or "",
    }


class MockGateway:
    def __init__(self):
        self._narration_i = 0
        self._answer_i = 0

    async def transcribe(self, wav, *, language_code, keyterms, usage, duration_s, purpose="narration"):
        usage.stt_seconds += duration_s
        usage.sarvam_calls += 1
        if purpose == "answer":
            text = ANSWERS[self._answer_i % len(ANSWERS)]
            self._answer_i += 1
        else:
            text = NARRATION[self._narration_i % len(NARRATION)]
            self._narration_i += 1
        return STTResult(text, "kn-IN", 0.97)

    async def doc_extract(self, image, *, filename, mime, schema, usage):
        usage.doc_pages += 1
        usage.sarvam_calls += 1
        return {"result": dict(SCREENSHOT), "annotations": {}}

    async def chat(self, messages, *, model, schema, usage, api="v1", reasoning_effort=None, max_tokens=None):
        usage.sarvam_calls += 1
        name = (schema or {}).get("name")
        if name == "payment_screenshot":
            return ChatResult(json.dumps(SCREENSHOT))
        if name == "incident_extraction":
            user = messages[-1]["content"]
            block = user.split(SOURCES_START, 1)[1].split(SOURCES_END, 1)[0]
            sources = json.loads(block)
            inc = baseline_extractor.extract([(s["id"], s["kind"], s["text"]) for s in sources
                                              if s["kind"] != "answer"])
            inc.narrative_en = ("A caller claiming to be a CBI officer threatened me with a 'digital arrest' and told "
                                "me to send money for verification. I sent the money through UPI.")
            # mimic an LLM picking up the impersonated agency
            for s in sources:
                if "CBI" in s["text"]:
                    i = s["text"].index("CBI")
                    inc.impersonated_entity = Fact(value="CBI", source_id=s["id"], quote=s["text"][i:i + 15])
                    break
            usage.llm_input_tokens += len(user) // 3
            usage.llm_output_tokens += 600
            return ChatResult(json.dumps(incident_to_llm_json(inc), ensure_ascii=False))
        return ChatResult("OK")

    async def tts(self, text, *, language_code, model, usage):
        usage.tts_chars += len(text)
        usage.sarvam_calls += 1
        return silent_wav(0.3), "audio/wav"

    async def translate(self, text, *, source, target, usage):
        usage.translate_chars += len(text)
        usage.sarvam_calls += 1
        return f"[{target}] {text}"


def reset_mock_counters(gw) -> None:
    if isinstance(gw, MockGateway):
        gw._narration_i = 0
        gw._answer_i = 0

