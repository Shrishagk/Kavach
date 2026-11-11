"""Rule-based extractor (no AI).

Used for three things:
* MOCK_MODE stand-in for the LLM, so the whole pipeline runs offline;
* graceful fallback if the LLM call fails or returns unusable JSON;
* a baseline to compare the LLM against in the evaluation harness.

It only finds things with an unambiguous surface form (UTRs, UPI IDs, phone
numbers, amounts with a currency marker), each with its exact quote.
"""
from __future__ import annotations

import re

from app.core import validators as V
from app.core.scam_types import guess_from_text
from app.models.incident import Fact, Incident, Transaction

UPI_FIND = re.compile(r"[a-zA-Z0-9.\-_]{2,64}@[a-zA-Z][a-zA-Z0-9]{1,30}")
PHONE_FIND = re.compile(r"(?:\+\d{1,3}[\s-]?)?(?:\d[\s-]?){10,12}")
UTR_FIND = re.compile(r"(?<![\d])(\d{12})(?![\d])|\b([A-Z]{4}[A-Z0-9]{12})\b")
AMOUNT_FIND = re.compile(
    r"(?:₹|rs\.?|inr|rupees?)\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:lakh|lakhs|lac|crore|thousand|k))?"
    r"|\d[\d,]*(?:\.\d+)?\s*(?:rupees?|rs\b|lakh|lakhs|crore|thousand|ಸಾವಿರ|ಲಕ್ಷ|हजार|हज़ार|लाख)",
    re.IGNORECASE,
)
UTR_CONTEXT = re.compile(r"\b(utr|rrn|transaction id|txn id|reference|ref no|upi ref)\b", re.IGNORECASE)

BANK_WORDS = [
    "SBI", "State Bank", "HDFC", "ICICI", "Axis", "Kotak", "Canara", "Bank of Baroda", "PNB",
    "Punjab National", "Union Bank", "Indian Bank", "Bank of India", "Karnataka Bank", "Federal",
    "IndusInd", "Yes Bank", "IDFC", "Karur Vysya", "South Indian Bank", "Indian Overseas",
]


def _fact(value: str, source_id: str, quote: str, kind: str = "transcript") -> Fact:
    return Fact(value=value, source_kind=kind, source_id=source_id, quote=quote.strip(), confidence=0.6)


def extract(sources: list[tuple[str, str, str]]) -> Incident:
    """sources: list of (source_id, source_kind, text)."""
    inc = Incident()
    seen_upi, seen_phone, seen_utr = set(), set(), set()
    amounts: list[Fact] = []
    utrs: list[Fact] = []
    full_text = " ".join(t for _, _, t in sources)

    for sid, kind, raw in sources:
        text = V.normalize_text(raw)
        for m in UPI_FIND.finditer(text):
            v = V.validate_upi(m.group(0))
            if v.ok and v.normalized not in seen_upi and "." not in v.normalized.split("@")[1]:
                seen_upi.add(v.normalized)
                inc.suspect_upi_ids.append(_fact(v.normalized, sid, m.group(0), kind))
        for m in UTR_FIND.finditer(text):
            val = m.group(1) or m.group(2)
            ctx = text[max(0, m.start() - 40):m.start()]
            # a bare 12-digit number counts as a UTR only with nearby context words
            if m.group(1) and not UTR_CONTEXT.search(ctx):
                continue
            if val not in seen_utr:
                seen_utr.add(val)
                utrs.append(_fact(val, sid, text[max(0, m.start() - 20):m.end()], kind))
        for m in PHONE_FIND.finditer(text):
            chunk = m.group(0)
            if V.only_digits(chunk) in seen_utr:
                continue
            v = V.validate_phone(chunk)
            if v.ok and v.kind != "in_landline" and v.normalized not in seen_phone:
                seen_phone.add(v.normalized)
                inc.suspect_phones.append(_fact(v.normalized, sid, chunk, kind))
        for m in AMOUNT_FIND.finditer(text):
            v = V.parse_amount(m.group(0))
            if v.ok:
                amounts.append(_fact(v.normalized, sid, m.group(0), kind))
        for b in BANK_WORDS:
            if re.search(rf"(\bmy\b|\bmine\b|\bnanna\b|\bmera\b|\bmeri\b|\bmere\b|ನನ್ನ|मेरा|मेरी|मेरे|माझे|माझ्या|என்|నా)[^.]{{0,30}}\b{re.escape(b)}",
                         text, re.IGNORECASE):
                if not inc.victim_bank.present:
                    mm = re.search(re.escape(b), text, re.IGNORECASE)
                    inc.victim_bank = _fact(b, sid, mm.group(0), kind)

    # pair amounts and UTRs in order of appearance (best effort)
    for i in range(max(len(amounts), len(utrs))):
        tx = Transaction()
        if i < len(amounts):
            tx.amount = amounts[i]
        if i < len(utrs):
            tx.utr = utrs[i]
            vu = V.validate_utr(utrs[i].value or "")
            tx.utr_kind = vu.kind
        inc.transactions.append(tx)

    st, kw = guess_from_text(full_text)
    inc.scam_type = st
    if kw:
        for sid, kind, raw in sources:
            if kw in raw.lower():
                idx = raw.lower().index(kw)
                inc.scam_type_evidence = _fact(kw, sid, raw[idx:idx + len(kw)], kind)
                break
    return inc
