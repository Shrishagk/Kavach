"""Session state persisted (encrypted) between API calls."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from .incident import Conflict, Incident, RejectedValue


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Stage(str, Enum):
    CREATED = "created"
    INTAKE = "intake"
    ANALYZED = "analyzed"
    CONFIRMED = "confirmed"
    DOCUMENTS_READY = "documents_ready"


class TranscriptSegment(BaseModel):
    id: str  # "t1", "t2", ...
    text: str
    language_code: str | None = None
    start_s: float = 0.0
    end_s: float = 0.0
    narration_id: str | None = None


class ScreenshotRecord(BaseModel):
    id: str  # "img1"
    file_id: str
    method: str  # doc_ai | vision_llm | mock
    raw_fields: dict = Field(default_factory=dict)
    ocr_text: str | None = None
    warnings: list[str] = Field(default_factory=list)


class Question(BaseModel):
    id: str  # "q1"
    field: str  # e.g. "transactions[0].utr"
    text: str  # in the victim's language
    text_en: str
    audio_file_id: str | None = None
    answered: bool = False


class Answer(BaseModel):
    id: str  # "a1"
    question_id: str | None
    text: str
    language_code: str | None = None


class UsageCounters(BaseModel):
    stt_seconds: float = 0.0
    tts_chars: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    translate_chars: int = 0
    doc_pages: int = 0
    sarvam_calls: int = 0
    sarvam_errors: int = 0

    def estimated_cost_inr(self) -> float:
        # List prices (Nov 2026, ex-GST). Approximate; for monitoring only.
        cost = (
            self.stt_seconds / 60 * 0.49
            + self.tts_chars / 1000 * 1.0
            + self.llm_input_tokens / 1e6 * 15
            + self.llm_output_tokens / 1e6 * 60
            + self.translate_chars / 1000 * 2
            + self.doc_pages * 0.5
        )
        return round(cost, 2)


class DocumentRecord(BaseModel):
    kind: str  # complaint_pdf | freeze_email | call_script | call_script_audio | checklist
    file_id: str
    filename: str
    mime: str


class SessionState(BaseModel):
    id: str
    created_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime
    stage: Stage = Stage.CREATED
    consent_at: datetime
    language_code: str | None = None  # detected from speech, or hint
    language_hint: str | None = None
    segments: list[TranscriptSegment] = Field(default_factory=list)
    screenshots: list[ScreenshotRecord] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    answers: list[Answer] = Field(default_factory=list)
    incident: Incident = Field(default_factory=Incident)
    rejected: list[RejectedValue] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    corrections: dict[str, str] = Field(default_factory=dict)
    confirmed_at: datetime | None = None
    documents: list[DocumentRecord] = Field(default_factory=list)
    usage: UsageCounters = Field(default_factory=UsageCounters)
    extraction_method: str | None = None  # llm | baseline
    narration_count: int = 0
    analysis_count: int = 0

    def next_id(self, prefix: str, existing: list) -> str:
        return f"{prefix}{len(existing) + 1}"

    def source_text(self, source_id: str) -> str | None:
        for s in self.segments:
            if s.id == source_id:
                return s.text
        for a in self.answers:
            if a.id == source_id:
                return a.text
        for img in self.screenshots:
            if img.id == source_id:
                fields = " ".join(str(v) for k, v in img.raw_fields.items() if v and k != "visible_text")
                return f"{img.ocr_text or ''} {fields}".strip()
        return None
