"""Golden-hour timer and action checklist."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.core.scam_types import ADVICE_EN, GENERIC_ADVICE_EN
from app.models.incident import Incident

IST = timezone(timedelta(hours=5, minutes=30))


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    return dt


def timer(inc: Incident, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    times = [parse_dt(t.datetime.value) for t in inc.transactions]
    times = [t for t in times if t]
    if not times:
        return {"last_payment_at": None, "minutes_since_last_payment": None, "within_golden_hour": None,
                "message_en": "Report immediately: the sooner the bank is told, the better the chance of freezing the money."}
    last = max(times)
    mins = int((now - last).total_seconds() // 60)
    within = 0 <= mins <= 60
    msg = (f"{mins} minutes since the last payment. You are inside the golden hour — call 1930 now."
           if within else
           f"{mins} minutes since the last payment. Report anyway: money can still be frozen if it has not been withdrawn.")
    return {"last_payment_at": last.isoformat(), "minutes_since_last_payment": mins,
            "within_golden_hour": within, "message_en": msg}


def checklist(inc: Incident) -> list[dict]:
    steps = [
        {"id": "call_1930", "title_en": "Call 1930 (National Cybercrime Helpline) and read out your script.", "urgent": True},
        {"id": "call_bank", "title_en": "Call your bank's official fraud/customer-care number (on your card or the bank's website) and ask them to block your account/cards and raise a fraud complaint.", "urgent": True},
        {"id": "file_ncrp", "title_en": "File the complaint at cybercrime.gov.in ('Report Financial Fraud') using your complaint draft, and save the acknowledgement number.", "urgent": True},
        {"id": "email_bank", "title_en": "Send the bank-freeze request email to your bank, adding the acknowledgement number.", "urgent": False},
        {"id": "secure", "title_en": "Change your UPI PIN, net-banking and email passwords. Uninstall any app the caller asked you to install.", "urgent": False},
        {"id": "evidence", "title_en": "Keep all screenshots, SMS, call logs and chat messages. Do not delete the chat.", "urgent": False},
    ]
    tips = ADVICE_EN.get(inc.scam_type, []) + GENERIC_ADVICE_EN
    return steps + [{"id": f"tip_{i}", "title_en": t, "urgent": False, "tip": True} for i, t in enumerate(tips)]
