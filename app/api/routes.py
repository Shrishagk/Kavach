"""HTTP API (v1). See README for the end-to-end call sequence."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response

from app.config import Settings
from app.core.languages import all_plans, normalize, plan_for
from app.core.metrics import metrics
from app.core.missing import ready_for_documents
from app.core.storage import SessionNotFound, Storage
from app.documents.generator import generate_all
from app.models.api import (AnswerRequest, ConfirmRequest, CreateSessionRequest, CreateSessionResponse,
                            QuestionOut, ReadbackResponse, ScreenshotOut, SegmentOut, SessionView)
from app.models.session import SessionState, Stage
from app.services import workflow as W
from app.services.gateway import Gateway

router = APIRouter(prefix="/api/v1")

PRIVACY_NOTICE = (
    "GoldenHour turns what you say and the screenshots you share into draft complaint documents. "
    "Your voice recordings are deleted right after they are transcribed. Everything else is encrypted "
    "and deleted automatically after 24 hours, or immediately when you press Delete. GoldenHour does not "
    "file complaints or contact your bank for you, and it is not legal advice."
)

IMAGE_MAGIC = {b"\x89PNG\r\n\x1a\n": "image/png", b"\xff\xd8\xff": "image/jpeg"}


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_storage(request: Request) -> Storage:
    return request.app.state.storage


def get_gateway(request: Request) -> Gateway:
    return request.app.state.gateway


@asynccontextmanager
async def open_session(storage: Storage, session_id: str, token: str | None):
    lock = storage.lock(session_id)
    async with lock:
        try:
            state = storage.load(session_id, token)
        except SessionNotFound:
            raise HTTPException(404, "Session not found or expired.")
        yield state
        storage.save(state)


def sniff_image(data: bytes) -> str | None:
    for magic, mime in IMAGE_MAGIC.items():
        if data.startswith(magic):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


async def read_limited(upload: UploadFile, limit: int) -> bytes:
    data = await upload.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File too large (max {limit // (1024 * 1024)} MB).")
    if not data:
        raise HTTPException(400, "Empty file.")
    return data


def view(state: SessionState, storage: Storage) -> SessionView:
    def q_out(q) -> QuestionOut:
        return QuestionOut(id=q.id, field=q.field, text=q.text, text_en=q.text_en, answered=q.answered,
                           audio_url=storage.sign(state.id, q.audio_file_id) if q.audio_file_id else None)

    questions = [q_out(q) for q in state.questions]
    ready, blockers = ready_for_documents(state.incident)
    payload = W.status_payload(state)
    return SessionView(
        session_id=state.id, stage=state.stage.value, language=payload["language"],
        segments=[SegmentOut(id=s.id, text=s.text, language_code=s.language_code) for s in state.segments],
        screenshots=[ScreenshotOut(id=i.id, method=i.method, warnings=i.warnings,
                                   fields={k: v for k, v in i.raw_fields.items() if k != "visible_text"})
                     for i in state.screenshots],
        incident=state.incident, extraction_method=state.extraction_method, total_amount=state.incident.total_amount(),
        questions=questions, open_questions=[q for q in questions if not q.answered],
        conflicts=state.conflicts, rejected=state.rejected, ready_for_documents=ready, blockers=blockers,
        timer=payload["timer"],
        documents=[{"kind": d.kind, "filename": d.filename, "mime": d.mime,
                    "url": storage.sign(state.id, d.file_id)} for d in state.documents],
        usage={**state.usage.model_dump(), "estimated_cost_inr": state.usage.estimated_cost_inr()},
        expires_at=state.expires_at,
    )


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------

@router.get("/meta/languages", tags=["meta"])
def languages(settings: Settings = Depends(get_settings)):
    return {"languages": all_plans(settings.tts_model, settings.tts_fallback_model)}


@router.get("/admin/metrics", tags=["meta"])
def admin_metrics(settings: Settings = Depends(get_settings), x_admin_token: str | None = Header(default=None)):
    expected = settings.admin_token.get_secret_value() if settings.admin_token else None
    if expected is None and settings.env == "prod":
        raise HTTPException(404)
    if expected is not None and x_admin_token != expected:
        raise HTTPException(401, "Admin token required.")
    return metrics.snapshot()


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

@router.post("/sessions", response_model=CreateSessionResponse, status_code=201, tags=["session"])
def create_session(body: CreateSessionRequest, storage: Storage = Depends(get_storage)):
    if not body.consent:
        raise HTTPException(400, "Consent is required before we can process your statement.")
    hint = normalize(body.language_hint) if body.language_hint else None
    state, token = storage.create_session(hint)
    metrics.inc("sessions_created")
    return CreateSessionResponse(session_id=state.id, session_token=token, expires_at=state.expires_at,
                                 privacy_notice=PRIVACY_NOTICE)


@router.get("/sessions/{session_id}", response_model=SessionView, tags=["session"])
async def get_session(session_id: str, x_session_token: str | None = Header(default=None),
                      storage: Storage = Depends(get_storage)):
    async with open_session(storage, session_id, x_session_token) as state:
        return view(state, storage)


@router.delete("/sessions/{session_id}", status_code=204, tags=["session"])
async def delete_session(session_id: str, x_session_token: str | None = Header(default=None),
                         storage: Storage = Depends(get_storage)):
    try:
        storage.load(session_id, x_session_token)
    except SessionNotFound:
        raise HTTPException(404, "Session not found or expired.")
    storage.delete_session(session_id)
    metrics.inc("sessions_deleted")
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Intake
# ---------------------------------------------------------------------------

@router.post("/sessions/{session_id}/narration", response_model=SessionView, tags=["intake"])
async def upload_narration(session_id: str, audio: UploadFile = File(...),
                           x_session_token: str | None = Header(default=None),
                           storage: Storage = Depends(get_storage), gw: Gateway = Depends(get_gateway),
                           settings: Settings = Depends(get_settings)):
    data = await read_limited(audio, settings.max_audio_bytes)
    async with open_session(storage, session_id, x_session_token) as state:
        if state.narration_count >= settings.max_narrations_per_session:
            raise HTTPException(429, "Too many recordings in this session.")
        await W.add_narration(state, data, gw, settings)
        del data  # raw audio is not retained
        return view(state, storage)


@router.post("/sessions/{session_id}/screenshots", response_model=SessionView, tags=["intake"])
async def upload_screenshot(session_id: str, image: UploadFile = File(...),
                            x_session_token: str | None = Header(default=None),
                            storage: Storage = Depends(get_storage), gw: Gateway = Depends(get_gateway),
                            settings: Settings = Depends(get_settings)):
    data = await read_limited(image, settings.max_image_bytes)
    mime = sniff_image(data)
    if not mime:
        raise HTTPException(415, "Only PNG, JPEG or WebP screenshots are accepted.")
    async with open_session(storage, session_id, x_session_token) as state:
        if len(state.screenshots) >= settings.max_screenshots_per_session:
            raise HTTPException(429, "Too many screenshots in this session.")
        await W.add_screenshot(state, data, mime, gw, settings, storage)
        return view(state, storage)


# ---------------------------------------------------------------------------
# Analysis, follow-up questions, confirmation
# ---------------------------------------------------------------------------

@router.post("/sessions/{session_id}/analyze", response_model=SessionView, tags=["analysis"])
async def analyze(session_id: str, x_session_token: str | None = Header(default=None),
                  storage: Storage = Depends(get_storage), gw: Gateway = Depends(get_gateway),
                  settings: Settings = Depends(get_settings)):
    async with open_session(storage, session_id, x_session_token) as state:
        await W.analyze(state, gw, settings, storage)
        metrics.inc("analyses")
        return view(state, storage)


@router.post("/sessions/{session_id}/answers", response_model=SessionView, tags=["analysis"])
async def answer_text(session_id: str, body: AnswerRequest, x_session_token: str | None = Header(default=None),
                      storage: Storage = Depends(get_storage), gw: Gateway = Depends(get_gateway),
                      settings: Settings = Depends(get_settings)):
    async with open_session(storage, session_id, x_session_token) as state:
        await W.add_answer(state, body.question_id, body.text, None, gw, settings)
        await W.analyze(state, gw, settings, storage)
        return view(state, storage)


@router.post("/sessions/{session_id}/answers/voice", response_model=SessionView, tags=["analysis"])
async def answer_voice(session_id: str, audio: UploadFile = File(...), question_id: str | None = Form(default=None),
                       x_session_token: str | None = Header(default=None),
                       storage: Storage = Depends(get_storage), gw: Gateway = Depends(get_gateway),
                       settings: Settings = Depends(get_settings)):
    data = await read_limited(audio, settings.max_audio_bytes)
    async with open_session(storage, session_id, x_session_token) as state:
        await W.add_answer(state, question_id, None, data, gw, settings)
        await W.analyze(state, gw, settings, storage)
        return view(state, storage)


@router.post("/sessions/{session_id}/readback", response_model=ReadbackResponse, tags=["analysis"])
async def readback(session_id: str, x_session_token: str | None = Header(default=None),
                   storage: Storage = Depends(get_storage), gw: Gateway = Depends(get_gateway),
                   settings: Settings = Depends(get_settings)):
    async with open_session(storage, session_id, x_session_token) as state:
        if state.stage in (Stage.CREATED, Stage.INTAKE):
            raise HTTPException(409, "Run /analyze first.")
        out = await W.readback(state, gw, settings, storage)
        return ReadbackResponse(language_code=out["language_code"], text=out["text"], text_en=out["text_en"],
                                audio_url=storage.sign(state.id, out["audio_file_id"]) if out["audio_file_id"] else None)


@router.post("/sessions/{session_id}/confirm", response_model=SessionView, tags=["analysis"])
async def confirm(session_id: str, body: ConfirmRequest, x_session_token: str | None = Header(default=None),
                  storage: Storage = Depends(get_storage)):
    async with open_session(storage, session_id, x_session_token) as state:
        if state.stage in (Stage.CREATED, Stage.INTAKE):
            raise HTTPException(409, "Run /analyze first.")
        if not body.confirmed:
            raise HTTPException(400, "Confirmation required.")
        errors = W.confirm(state, body.corrections)
        if errors:
            raise HTTPException(422, {"errors": errors})
        return view(state, storage)


@router.post("/sessions/{session_id}/documents", response_model=SessionView, tags=["documents"])
async def documents(session_id: str, x_session_token: str | None = Header(default=None),
                    storage: Storage = Depends(get_storage), gw: Gateway = Depends(get_gateway),
                    settings: Settings = Depends(get_settings)):
    async with open_session(storage, session_id, x_session_token) as state:
        if state.stage not in (Stage.CONFIRMED, Stage.DOCUMENTS_READY):
            raise HTTPException(409, "The victim must confirm the read-back summary first (/confirm).")
        await generate_all(state, gw, settings, storage)
        metrics.inc("documents_generated")
        return view(state, storage)


@router.get("/sessions/{session_id}/files/{file_id}", tags=["documents"])
def get_file(session_id: str, file_id: str, exp: int = Query(...), sig: str = Query(...),
             storage: Storage = Depends(get_storage)):
    if not storage.verify_signature(session_id, file_id, exp, sig):
        raise HTTPException(403, "Link expired or invalid. Refresh the session to get a new link.")
    try:
        data, mime, filename = storage.get_file(session_id, file_id)
    except FileNotFoundError:
        raise HTTPException(404, "File not found.")
    disposition = "inline" if mime.startswith(("audio/", "image/", "application/pdf")) else "attachment"
    return Response(data, media_type=mime, headers={
        "Content-Disposition": f'{disposition}; filename="{filename}"',
        "Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff",
    })


@router.get("/sessions/{session_id}/language", tags=["session"])
async def language(session_id: str, x_session_token: str | None = Header(default=None),
                   storage: Storage = Depends(get_storage), settings: Settings = Depends(get_settings)):
    async with open_session(storage, session_id, x_session_token) as state:
        return plan_for(state.language_code or state.language_hint, settings.tts_model,
                        settings.tts_fallback_model).as_dict()
