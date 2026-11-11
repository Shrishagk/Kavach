"""Deterministic validators and normalizers for financial identifiers.

These never call an AI model. They are the backstop that keeps a malformed
UTR, phone number or amount out of a police complaint.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

# --------------------------------------------------------------------------
# Text normalization
# --------------------------------------------------------------------------

_DIGIT_WORDS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


def ascii_digits(text: str) -> str:
    """Convert native-script digits (೧, १, ௧ ...) to ASCII 0-9."""
    out = []
    for ch in text:
        if ch.isdigit() and not ("0" <= ch <= "9"):
            try:
                out.append(str(unicodedata.digit(ch)))
                continue
            except (TypeError, ValueError):
                pass
        out.append(ch)
    return "".join(out)


def spoken_digits_to_numbers(text: str) -> str:
    """Collapse runs of spoken English digits ("nine eight four ...") into digit strings.

    Only runs of 4+ digit words are collapsed, so ordinary sentences are untouched.
    """
    tokens = re.split(r"(\s+)", text)
    result: list[str] = []
    run: list[str] = []

    def flush():
        if len(run) >= 4:
            result.append("".join(_DIGIT_WORDS[w.lower()] for w in run))
        else:
            for i, w in enumerate(run):
                result.append(w)
                if i < len(run) - 1:
                    result.append(" ")
        run.clear()

    for tok in tokens:
        if tok.isspace():
            if not run:
                result.append(tok)
            continue
        word = tok.strip(",.").lower()
        if word in _DIGIT_WORDS:
            run.append(tok.strip(",."))
        else:
            if run:
                flush()
                result.append(" ")
            result.append(tok)
    if run:
        flush()
    return "".join(result)


def normalize_text(text: str) -> str:
    return spoken_digits_to_numbers(ascii_digits(text or ""))


def only_digits(text: str) -> str:
    return re.sub(r"\D", "", ascii_digits(text or ""))


# --------------------------------------------------------------------------
# Identifier validation
# --------------------------------------------------------------------------


@dataclass
class Validation:
    ok: bool
    normalized: str | None = None
    kind: str | None = None
    reason: str | None = None


UPI_RE = re.compile(r"^[a-zA-Z0-9.\-_]{2,256}@[a-zA-Z][a-zA-Z0-9.\-]{1,64}$")
IFSC_RE = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")


def validate_utr(value: str) -> Validation:
    raw = ascii_digits(value or "").strip().upper().replace(" ", "").replace("-", "")
    if raw.isdigit() and (len(set(raw)) == 1 or raw in "012345678901234567890" or raw in "98765432109876543210"):
        return Validation(False, None, None, "looks like a placeholder, not a real transaction ID")
    if re.fullmatch(r"\d{12}", raw):
        return Validation(True, raw, "upi_rrn")  # UPI/IMPS RRN (12 digits)
    if re.fullmatch(r"[A-Z]{4}[A-Z0-9]{12}", raw):
        return Validation(True, raw, "neft_utr")  # NEFT UTR (16 chars)
    if re.fullmatch(r"[A-Z]{4}R[A-Z0-9]{17}", raw):
        return Validation(True, raw, "rtgs_utr")  # RTGS UTR (22 chars)
    if re.fullmatch(r"[A-Z]{1,4}\d{10,30}[A-Z0-9]*", raw) and 13 <= len(raw) <= 35:
        return Validation(True, raw, "app_txn_id")  # e.g. PhonePe "T2411..." IDs
    return Validation(False, None, None, "not a recognised UTR/RRN/transaction-ID format")


def validate_upi(value: str) -> Validation:
    v = (value or "").strip().replace(" ", "")
    if UPI_RE.fullmatch(v):
        return Validation(True, v.lower(), "upi_id")
    return Validation(False, None, None, "not a valid UPI ID (expected name@handle)")


def validate_phone(value: str) -> Validation:
    raw = ascii_digits(value or "").strip()
    has_plus = raw.startswith("+")
    digits = only_digits(raw)
    if len(digits) == 10 and digits[0] in "6789":
        return Validation(True, "+91" + digits, "in_mobile")
    if len(digits) == 12 and digits.startswith("91") and digits[2] in "6789":
        return Validation(True, "+" + digits, "in_mobile")
    if len(digits) == 11 and digits.startswith("0") and digits[1] in "6789":
        return Validation(True, "+91" + digits[1:], "in_mobile")
    if len(digits) == 11 and digits.startswith("0"):
        return Validation(True, "+91" + digits[1:], "in_landline")
    if has_plus and 8 <= len(digits) <= 15 and not digits.startswith("91"):
        return Validation(True, "+" + digits, "international")
    if digits.startswith("140") and len(digits) == 10:
        return Validation(True, digits, "in_telemarketing_140")
    return Validation(False, None, None, "not a valid phone number")


def validate_account(value: str) -> Validation:
    digits = only_digits(value)
    if 9 <= len(digits) <= 18:
        return Validation(True, digits, "bank_account")
    return Validation(False, None, None, "bank account numbers have 9-18 digits")


def validate_ifsc(value: str) -> Validation:
    v = (value or "").strip().upper().replace(" ", "")
    if IFSC_RE.fullmatch(v):
        return Validation(True, v, "ifsc")
    return Validation(False, None, None, "IFSC must look like ABCD0123456")


def validate_last4(value: str) -> Validation:
    digits = only_digits(value)
    if len(digits) >= 4:
        return Validation(True, digits[-4:], "last4")
    return Validation(False, None, None, "need the last 4 digits")


# --------------------------------------------------------------------------
# Amounts (Indian numbering: thousand / lakh / crore, in several scripts)
# --------------------------------------------------------------------------

_MULTIPLIERS = {
    # English
    "k": 1_000, "thousand": 1_000, "thousands": 1_000, "lakh": 100_000, "lakhs": 100_000,
    "lac": 100_000, "lacs": 100_000, "crore": 10_000_000, "crores": 10_000_000, "cr": 10_000_000,
    # Hindi / Marathi
    "हजार": 1_000, "हज़ार": 1_000, "लाख": 100_000, "करोड़": 10_000_000, "करोड": 10_000_000,
    # Kannada
    "ಸಾವಿರ": 1_000, "ಲಕ್ಷ": 100_000, "ಕೋಟಿ": 10_000_000,
    # Tamil
    "ஆயிரம்": 1_000, "லட்சம்": 100_000, "கோடி": 10_000_000,
    # Telugu
    "వేలు": 1_000, "వెయ్యి": 1_000, "లక్ష": 100_000, "లక్షలు": 100_000, "కోట్లు": 10_000_000, "కోటి": 10_000_000,
    # Malayalam
    "ആയിരം": 1_000, "ലക്ഷം": 100_000, "കോടി": 10_000_000,
    # Bengali
    "হাজার": 1_000, "লাখ": 100_000, "কোটি": 10_000_000,
}

_AMOUNT_RE = re.compile(
    r"(?:₹|rs\.?|inr|rupees?)?\s*(\d[\d,]*(?:\.\d+)?)\s*(" + "|".join(
        sorted((re.escape(k) for k in _MULTIPLIERS), key=len, reverse=True)
    ) + r")?",
    re.IGNORECASE,
)


def parse_amount(value: str) -> Validation:
    text = normalize_text(value or "").strip()
    m = _AMOUNT_RE.search(text)
    if not m:
        return Validation(False, None, None, "no number found")
    num_s, mult_s = m.group(1), m.group(2)
    try:
        num = Decimal(num_s.replace(",", ""))
    except InvalidOperation:
        return Validation(False, None, None, "unreadable number")
    if mult_s:
        num *= _MULTIPLIERS.get(mult_s.lower(), _MULTIPLIERS.get(mult_s, 1))
    if num <= 0:
        return Validation(False, None, None, "amount must be positive")
    if num > Decimal("100000000000"):
        return Validation(False, None, None, "amount implausibly large")
    return Validation(True, f"{num.quantize(Decimal('0.01'))}", "inr")


_DT_FORMATS = [
    "%d %b %Y, %I:%M %p", "%d %b %Y, %I:%M:%S %p", "%d %b %Y %I:%M %p", "%d %B %Y, %I:%M %p",
    "%d %b %Y, %H:%M", "%d/%m/%Y %H:%M", "%d/%m/%Y, %I:%M %p", "%d-%m-%Y %H:%M", "%d-%m-%Y %I:%M %p",
    "%I:%M %p on %d %b %Y", "%I:%M %p, %d %b %Y", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M",
]


def to_iso_ist(value: str | None) -> str | None:
    """Best-effort parse of receipt-style date strings into ISO 8601 (+05:30)."""
    if not value:
        return None
    from datetime import datetime, timedelta, timezone

    v = re.sub(r"\s+", " ", ascii_digits(value)).strip().replace("am", "AM").replace("pm", "PM")
    try:
        dt = datetime.fromisoformat(v)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone(timedelta(hours=5, minutes=30)))
        return dt.isoformat()
    except ValueError:
        pass
    for fmt in _DT_FORMATS:
        try:
            dt = datetime.strptime(v, fmt).replace(tzinfo=timezone(timedelta(hours=5, minutes=30)))
            return dt.isoformat()
        except ValueError:
            continue
    return None


def format_inr(value: str | float | None) -> str:
    """Format as Indian currency: 1234567.5 -> ₹12,34,567.50"""
    if value in (None, ""):
        return "—"
    try:
        d = Decimal(str(value)).quantize(Decimal("0.01"))
    except InvalidOperation:
        return str(value)
    whole, frac = f"{d:.2f}".split(".")
    neg = whole.startswith("-")
    whole = whole.lstrip("-")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    return f"{'-' if neg else ''}₹{whole}.{frac}"


def spell_digits(value: str) -> str:
    """'9845012345' -> '98450 12345' style grouping for read-aloud scripts."""
    v = (value or "").replace(" ", "")
    if v.startswith("+91"):
        v = v[3:]
    if len(v) <= 5:
        return " ".join(v)
    groups = [v[i:i + 4] for i in range(0, len(v), 4)]
    return ", ".join(" ".join(g) for g in groups)


VALIDATORS = {
    "utr": validate_utr,
    "upi": validate_upi,
    "phone": validate_phone,
    "account": validate_account,
    "ifsc": validate_ifsc,
    "amount": parse_amount,
    "last4": validate_last4,
}
