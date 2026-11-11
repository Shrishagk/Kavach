"""Incident data model.

Every extracted value is a `Fact`: the value plus WHERE it came from
(transcript segment, screenshot or the victim's answer) and the exact quote.
Facts that cannot be traced back to a source are rejected by the grounding
step, so documents never contain invented details.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

SourceKind = Literal["transcript", "screenshot", "answer", "correction", "computed"]


class ScamType(str, Enum):
    DIGITAL_ARREST = "digital_arrest"
    INVESTMENT = "investment_trading"
    KYC_UPDATE = "kyc_update"
    PARCEL_COURIER = "parcel_courier"
    PART_TIME_JOB = "part_time_job_task"
    LOAN_APP = "loan_app"
    SEXTORTION = "sextortion"
    UPI_COLLECT_QR = "upi_collect_or_qr"
    CUSTOMER_CARE = "fake_customer_care"
    LOTTERY_PRIZE = "lottery_prize"
    IMPERSONATION_RELATIVE = "impersonation_known_person"
    OTHER = "other"
    UNKNOWN = "unknown"


class Channel(str, Enum):
    PHONE_CALL = "phone_call"
    VIDEO_CALL = "video_call"
    WHATSAPP = "whatsapp"
    SMS = "sms"
    TELEGRAM = "telegram"
    EMAIL = "email"
    SOCIAL_MEDIA = "social_media"
    WEBSITE_APP = "website_or_app"
    IN_PERSON = "in_person"
    OTHER = "other"
    UNKNOWN = "unknown"


class Fact(BaseModel):
    value: str | None = None
    source_kind: SourceKind | None = None
    source_id: str | None = None  # e.g. "t3", "img1", "a2"
    quote: str | None = None  # exact words / OCR text supporting the value
    confidence: float | None = None
    verified: bool = False  # passed grounding + format validation
    note: str | None = None  # e.g. "spoken as words; please confirm"

    @property
    def present(self) -> bool:
        return self.value not in (None, "")


class Transaction(BaseModel):
    amount: Fact = Field(default_factory=Fact)  # normalized rupees as string, e.g. "50000.00"
    datetime: Fact = Field(default_factory=Fact)  # ISO 8601 when known, else as spoken
    utr: Fact = Field(default_factory=Fact)  # UTR / RRN / transaction reference
    utr_kind: str | None = None  # upi_rrn | neft_utr | rtgs_utr | app_txn_id
    payee_upi: Fact = Field(default_factory=Fact)
    payee_account: Fact = Field(default_factory=Fact)
    payee_ifsc: Fact = Field(default_factory=Fact)
    payment_mode: Fact = Field(default_factory=Fact)  # UPI / IMPS / NEFT / card / wallet
    payment_app: Fact = Field(default_factory=Fact)


class Incident(BaseModel):
    scam_type: ScamType = ScamType.UNKNOWN
    scam_type_evidence: Fact = Field(default_factory=Fact)
    first_contact_channel: Channel = Channel.UNKNOWN
    impersonated_entity: Fact = Field(default_factory=Fact)
    incident_date: Fact = Field(default_factory=Fact)
    victim_name: Fact = Field(default_factory=Fact)
    victim_bank: Fact = Field(default_factory=Fact)
    victim_account_last4: Fact = Field(default_factory=Fact)
    suspect_phones: list[Fact] = Field(default_factory=list)
    suspect_upi_ids: list[Fact] = Field(default_factory=list)
    suspect_accounts: list[Fact] = Field(default_factory=list)
    suspect_links: list[Fact] = Field(default_factory=list)  # URLs, app names, social handles
    transactions: list[Transaction] = Field(default_factory=list)
    narrative_en: str | None = None  # AI summary of the victim's own account (confirmed later)

    def total_amount(self) -> float:
        total = 0.0
        for t in self.transactions:
            try:
                total += float(t.amount.value) if t.amount.value else 0.0
            except ValueError:
                continue
        return round(total, 2)


class RejectedValue(BaseModel):
    field: str
    value: str | None
    reason: str


class Conflict(BaseModel):
    field: str
    values: list[str]
    sources: list[str]
    resolution: str | None = None
