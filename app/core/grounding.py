"""Evidence grounding: the anti-hallucination gate.

The LLM proposes facts with a source id and a supporting quote. This module
re-checks every one of them against the stored source text, deterministically:

1. The cited source must exist (transcript segment / answer / screenshot).
2. The quote must actually appear in that source (fuzzy, to tolerate
   punctuation and spacing differences).
3. For identifiers and amounts, the value itself must be recoverable from the
   source: the digits of a UTR must appear in it, a UPI ID must appear in it,
   the amount must parse to the same number.
4. The value must pass its format validator.

Anything that fails is nulled out and recorded as a RejectedValue, which then
turns into a follow-up question instead of a guessed field.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Callable, Iterator

from app.core import validators as V
from app.models.incident import Fact, Incident, RejectedValue
from app.models.session import SessionState

QUOTE_MATCH_THRESHOLD = 0.82

# Which validator applies to which field name (last path component).
FIELD_KIND = {
    "utr": "utr",
    "payee_upi": "upi",
    "suspect_upi_ids": "upi",
    "suspect_phones": "phone",
    "payee_account": "account",
    "suspect_accounts": "account",
    "payee_ifsc": "ifsc",
    "amount": "amount",
    "victim_account_last4": "last4",
}


def _norm(s: str) -> str:
    s = V.normalize_text(s).lower()
    s = re.sub(r"[\"'`“”‘’.,;:!?()\[\]{}|]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def quote_in_source(quote: str, source: str) -> float:
    """Best fuzzy match ratio of `quote` against any same-length window of `source`."""
    q, s = _norm(quote), _norm(source)
    if not q or not s:
        return 0.0
    if q in s:
        return 1.0
    if len(q) > len(s):
        return SequenceMatcher(None, q, s).ratio()
    best = 0.0
    step = max(1, len(q) // 4)
    for i in range(0, len(s) - len(q) + 1, step):
        r = SequenceMatcher(None, q, s[i:i + len(q)]).ratio()
        if r > best:
            best = r
            if best >= 0.99:
                break
    # also check the tail window
    best = max(best, SequenceMatcher(None, q, s[-len(q):]).ratio())
    return best


def value_supported(kind: str | None, value: str, source: str) -> bool:
    if not kind:
        return True  # free-text fields rely on the quote check
    src = V.normalize_text(source)
    if kind in ("utr", "phone", "account", "last4"):
        digits = V.only_digits(value)
        if kind == "phone" and digits.startswith("91") and len(digits) == 12:
            digits = digits[2:]
        if not digits:
            # alphanumeric UTR (NEFT/RTGS): compare case-insensitively without spaces
            return value.replace(" ", "").upper() in src.replace(" ", "").upper()
        if kind == "last4":
            return digits[-4:] in V.only_digits(src)
        return digits in V.only_digits(src) or digits in re.sub(r"[\s\-]", "", src)
    if kind == "upi":
        return value.replace(" ", "").lower() in src.replace(" ", "").lower()
    if kind == "ifsc":
        return value.replace(" ", "").upper() in src.replace(" ", "").upper()
    if kind == "amount":
        target = V.parse_amount(value)
        if not target.ok:
            return False
        for m in V._AMOUNT_RE.finditer(src):
            cand = V.parse_amount(m.group(0))
            if cand.ok and cand.normalized == target.normalized:
                return True
        return False
    return True


def iter_facts(incident: Incident) -> Iterator[tuple[str, Fact, Callable[[], None]]]:
    """Yield (path, fact, remove_fn) for every fact in the incident."""
    for name in ("scam_type_evidence", "impersonated_entity", "incident_date", "victim_name",
                 "victim_bank", "victim_account_last4"):
        fact = getattr(incident, name)

        def clear(n=name):
            setattr(incident, n, Fact())

        yield name, fact, clear
    for list_name in ("suspect_phones", "suspect_upi_ids", "suspect_accounts", "suspect_links"):
        lst: list[Fact] = getattr(incident, list_name)
        for i, fact in enumerate(list(lst)):
            def remove(lst=lst, fact=fact):
                if fact in lst:
                    lst.remove(fact)

            yield f"{list_name}[{i}]", fact, remove
    for i, tx in enumerate(incident.transactions):
        for fname in ("amount", "datetime", "utr", "payee_upi", "payee_account", "payee_ifsc",
                      "payment_mode", "payment_app"):
            fact = getattr(tx, fname)

            def clear(tx=tx, f=fname):
                setattr(tx, f, Fact())

            yield f"transactions[{i}].{fname}", fact, clear


def _kind_for(path: str) -> str | None:
    last = path.split(".")[-1]
    last = re.sub(r"\[\d+\]$", "", last)
    return FIELD_KIND.get(last)


def ground_incident(incident: Incident, session: SessionState) -> list[RejectedValue]:
    rejected: list[RejectedValue] = []
    for path, fact, remove in list(iter_facts(incident)):
        if not fact.present:
            continue
        if fact.source_kind in ("correction", "computed"):
            fact.verified = True
            continue
        reason = check_fact(path, fact, session)
        if reason:
            rejected.append(RejectedValue(field=path, value=fact.value, reason=reason))
            remove()
        else:
            fact.verified = True

    # drop transactions that ended up completely empty
    incident.transactions = [
        t for t in incident.transactions
        if any(getattr(t, f).present for f in ("amount", "utr", "payee_upi", "payee_account", "datetime"))
    ]
    return rejected


def check_fact(path: str, fact: Fact, session: SessionState) -> str | None:
    """Return a rejection reason, or None if the fact is grounded and valid."""
    if not fact.source_id:
        return "no source cited"
    source = session.source_text(fact.source_id)
    if source is None:
        return f"cited source {fact.source_id} does not exist"
    kind = _kind_for(path)

    if fact.source_kind != "screenshot":
        quote = fact.quote or fact.value or ""
        score = quote_in_source(quote, source)
        if score < QUOTE_MATCH_THRESHOLD:
            return f"quote not found in {fact.source_id} (match {score:.2f})"

    if not value_supported(kind, fact.value or "", source):
        # spoken amounts in words (e.g. "fifty thousand") can't be verified digit-wise;
        # keep them only if the quote parses to the same amount.
        if kind == "amount" and fact.quote and V.parse_amount(fact.quote).normalized == V.parse_amount(fact.value or "").normalized:
            fact.note = "amount spoken in words; please confirm"
        else:
            return f"value not present in {fact.source_id}"

    if kind:
        res = V.VALIDATORS[kind](fact.value or "")
        if not res.ok:
            return f"format check failed: {res.reason}"
        fact.value = res.normalized
    return None
