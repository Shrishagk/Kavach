"""Request/response schemas for the HTTP API."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.models.incident import Conflict, Incident, RejectedValue


class CreateSessionRequest(BaseModel):
    consent: bool = Field(description="Victim agreed to processing of their statement for this report.")
    language_hint: str | None = Field(default=None, examples=["kn-IN"])


class CreateSessionResponse(BaseModel):
    session_id: str
    session_token: str = Field(description="Send as the X-Session-Token header on every later call.")
    expires_at: datetime
    privacy_notice: str


class SegmentOut(BaseModel):
    id: str
    text: str
    language_code: str | None


class QuestionOut(BaseModel):
    id: str
    field: str
    text: str
    text_en: str
    audio_url: str | None
    answered: bool


class ScreenshotOut(BaseModel):
    id: str
    method: str
    fields: dict
    warnings: list[str]


class SessionView(BaseModel):
    session_id: str
    stage: str
    language: dict
    segments: list[SegmentOut]
    screenshots: list[ScreenshotOut]
    incident: Incident
    extraction_method: str | None
    total_amount: float
    questions: list[QuestionOut]
    open_questions: list[QuestionOut]
    conflicts: list[Conflict]
    rejected: list[RejectedValue]
    ready_for_documents: bool
    blockers: list[str]
    timer: dict
    documents: list[dict]
    usage: dict
    expires_at: datetime


class AnswerRequest(BaseModel):
    question_id: str | None = None
    text: str = Field(min_length=1, max_length=2000)


class ConfirmRequest(BaseModel):
    confirmed: bool = True
    corrections: dict[str, str] = Field(default_factory=dict, examples=[{"transactions[1].utr": "612345098765"}])


class ReadbackResponse(BaseModel):
    language_code: str
    text: str
    text_en: str
    audio_url: str | None
