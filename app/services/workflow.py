"""The GoldenHour pipeline: intake -> extraction -> grounding -> questions -> confirmation.

Each public function takes the session state, mutates it, and returns a result
for the API layer. Persistence and locking are handled by the caller.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from app.config import Settings
from app.core import validators as V
from app.core.golden_hour import checklist, timer
from app.core.grounding import ground_incident
from app.core.languages import normalize, plan_for
from app.core.missing import find_needs, ready_for_documents
from app.core.scam_types import LABELS_EN, guess_from_text
from app.core.storage import Storage
from app.models.incident import Conflict, Fact, Incident, ScamType, Transaction
from app.models.session import Answer, Question, ScreenshotRecord, SessionState, Stage, TranscriptSegment
from app.services import audio as A
from app.services.extraction import extract_incident
from app.services.gateway import Gateway
from app.services.speech import speak, translate, translate_protected
from app.services.vision import read_screenshot, screenshot_to_transaction

log = logging.getLogger("goldenhour.workflow")

DATA = Path(__file__).resolve().parent.parent / "data" / "banks.json"


class WorkflowError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@lru_cache
def bank_data() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


def keyterms() -> list[str]:
    d = bank_data()
    terms = [b["name"] for b in d["banks"]] + [a for b in d["banks"] for a in b["aliases"]]
    return list(dict.fromkeys(d["keyterms_extra"] + terms))[:40]


def match_bank(text: str) -> str | None:
    t = (text or "").lower()
    for b in bank_data()["banks"]:
        for name in [b["name"], *b["aliases"]]:
            if re.search(rf"\b{re.escape(name.lower())}\b", t):
                return b["name"]
    return None


# --------------------------------------------------------------------------
# Intake
# --------------------------------------------------------------------------

async def add_narration(session: SessionState, data: bytes, gw: Gateway, settings: Settings,
                        purpose: str = "narration") -> list[TranscriptSegment]:
    try:
        chunks, duration = await A.chunk_audio(data, settings.stt_chunk_seconds, settings.max_audio_seconds)
    except A.AudioError as e:
        raise WorkflowError(str(e), 422) from e

    hint = session.language_code or session.language_hint
    narration_id = f"n{session.narration_count + 1}"
    new_segments: list[TranscriptSegment] = []
    langs: Counter[str] = Counter()
    offset = sum(s.end_s - s.start_s for s in session.segments)
    for ch in chunks:
        res = await gw.transcribe(ch.wav, language_code=hint, keyterms=keyterms(), usage=session.usage,
                                  duration_s=ch.end_s - ch.start_s, purpose=purpose)
        text = res.transcript.strip()
        if not text:
            continue
        if res.language_code:
            langs[res.language_code] += int((ch.end_s - ch.start_s) * 10)
        if purpose == "narration":
            seg = TranscriptSegment(id=f"t{len(session.segments) + 1}", text=text, language_code=res.language_code,
                                    start_s=offset + ch.start_s, end_s=offset + ch.end_s, narration_id=narration_id)
            session.segments.append(seg)
        else:
            seg = TranscriptSegment(id="tmp", text=text, language_code=res.language_code)
        new_segments.append(seg)
    # raw audio is never stored; only transcripts are kept
    if purpose == "narration":
        session.narration_count += 1
        if langs and not session.language_code:
            session.language_code = normalize(langs.most_common(1)[0][0])
        session.stage = Stage.INTAKE if session.stage == Stage.CREATED else session.stage
    if not new_segments:
        raise WorkflowError("No speech was recognised. Please try again closer to the microphone.", 422)
    log.info("session %s: %d segments, %.1fs audio", session.id[:8], len(new_segments), duration)
    return new_segments


async def add_screenshot(session: SessionState, data: bytes, mime: str, gw: Gateway, settings: Settings,
                         storage: Storage) -> ScreenshotRecord:
    file_id = storage.put_file(session.id, data, "screenshot", mime, "screenshot")
    fields, method, warnings = await read_screenshot(data, mime, gw, settings, session.usage)
    if settings.mock_mode:
        method = "mock"
    rec = ScreenshotRecord(id=f"img{len(session.screenshots) + 1}", file_id=file_id, method=method,
                           raw_fields=fields, ocr_text=fields.get("visible_text"), warnings=warnings)
    _, tx_warnings = screenshot_to_transaction(rec.id, fields)
    rec.warnings += tx_warnings
    session.screenshots.append(rec)
    if session.stage == Stage.CREATED:
        session.stage = Stage.INTAKE
    return rec


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

def _same(a: Fact, b: Fact, kind: str) -> bool:
    if not (a.present and b.present):
        return False
    if kind == "amount":
        return V.parse_amount(a.value).normalized == V.parse_amount(b.value).normalized
    return V.only_digits(a.value) == V.only_digits(b.value) if kind == "utr" else a.value.lower() == b.value.lower()


def merge_screenshot_transactions(inc: Incident, session: SessionState) -> list[Conflict]:
    conflicts: list[Conflict] = []
    for img in session.screenshots:
        stx, _ = screenshot_to_transaction(img.id, img.raw_fields)
        if not stx:
            continue
        match = None
        for tx in inc.transactions:
            if _same(tx.utr, stx.utr, "utr") or (_same(tx.amount, stx.amount, "amount")
                                                  and (not tx.utr.present or _same(tx.utr, stx.utr, "utr"))):
                match = tx
                break
        if match is None:
            # a single spoken payment with no amount yet is very likely this one
            empties = [t for t in inc.transactions if not t.amount.present]
            if len(empties) == 1:
                match = empties[0]
        if match is None:
            inc.transactions.append(stx)
            continue
        for fname, kind in (("amount", "amount"), ("utr", "utr"), ("payee_upi", "upi"), ("payee_account", "account"),
                            ("payee_ifsc", "ifsc"), ("datetime", None), ("payment_app", None), ("payment_mode", None)):
            spoken, shot = getattr(match, fname), getattr(stx, fname)
            if not shot.present:
                continue
            if not spoken.present:
                setattr(match, fname, shot)
            elif kind in ("amount", "utr", "upi", "account") and not _same(spoken, shot, kind):
                conflicts.append(Conflict(field=f"{fname}", values=[spoken.value, shot.value],
                                          sources=[spoken.source_id or "?", shot.source_id]))
                setattr(match, fname, shot)  # screenshots are more reliable for numbers; victim confirms
        if stx.utr_kind and not match.utr_kind:
            match.utr_kind = stx.utr_kind
    # victim bank / last4 from screenshot
    for img in session.screenshots:
        f = img.raw_fields
        if not inc.victim_bank.present and f.get("payer_bank"):
            bank = match_bank(f["payer_bank"]) or f["payer_bank"]
            inc.victim_bank = Fact(value=bank, source_kind="screenshot", source_id=img.id, quote=f["payer_bank"])
        if not inc.victim_account_last4.present and f.get("payer_account_last4"):
            v = V.validate_last4(f["payer_account_last4"])
            if v.ok:
                inc.victim_account_last4 = Fact(value=v.normalized, source_kind="screenshot", source_id=img.id,
                                                quote=f["payer_account_last4"])
    return conflicts


def _tx_index(field: str) -> tuple[int | None, str]:
    m = re.match(r"transactions\[(\d+)\]\.(\w+)", field)
    return (int(m.group(1)), m.group(2)) if m else (None, field)


def bind_answers(inc: Incident, session: SessionState) -> None:
    """Directly bind a typed/spoken answer to the field it answers, when it parses cleanly."""
    for ans in session.answers:
        q = next((q for q in session.questions if q.id == ans.question_id), None)
        if not q:
            continue
        text = V.normalize_text(ans.text)
        idx, fname = _tx_index(q.field)

        def fact(value: str, quote: str | None = None) -> Fact:
            return Fact(value=value, source_kind="answer", source_id=ans.id, quote=quote or ans.text)

        if idx is not None and idx < len(inc.transactions):
            tx = inc.transactions[idx]
            target = getattr(tx, fname)
            if target.present:
                continue
            if fname == "utr":
                cleaned = re.sub(r"\b(utr|rrn|ref|reference|txn|transaction|id|no|number)\b[:.#]?", " ", text,
                                 flags=re.IGNORECASE)
                candidates = re.findall(r"\d[\d ]{10,40}\d", cleaned) + re.findall(r"\b[A-Za-z0-9]{12,35}\b", cleaned)
                for tok in candidates:
                    v = V.validate_utr(tok)
                    if v.ok:
                        tx.utr, tx.utr_kind = fact(v.normalized), v.kind
                        break
            elif fname == "amount":
                v = V.parse_amount(text)
                if v.ok:
                    tx.amount = fact(v.normalized)
            elif fname == "datetime":
                tx.datetime = fact(V.to_iso_ist(text) or ans.text.strip())
        elif q.field == "transactions":
            v = V.parse_amount(text)
            if v.ok and not inc.transactions:
                inc.transactions.append(Transaction(amount=fact(v.normalized)))
        elif q.field == "victim_bank" and not inc.victim_bank.present:
            bank = match_bank(text)
            if bank:
                inc.victim_bank = fact(bank)
            elif len(ans.text.strip()) <= 40:
                inc.victim_bank = fact(ans.text.strip())
        elif q.field == "victim_account_last4" and not inc.victim_account_last4.present:
            digits = V.only_digits(text)
            if len(digits) == 4:
                inc.victim_account_last4 = fact(digits)
        elif q.field == "suspect_identifiers":
            for m in re.finditer(r"[a-zA-Z0-9.\-_]{2,64}@[a-zA-Z][a-zA-Z0-9]{1,30}", text):
                v = V.validate_upi(m.group(0))
                if v.ok and all(f.value != v.normalized for f in inc.suspect_upi_ids):
                    inc.suspect_upi_ids.append(fact(v.normalized))
            for m in re.finditer(r"\+?\d[\d \-]{8,16}\d", text):
                v = V.validate_phone(m.group(0))
                if v.ok and all(f.value != v.normalized for f in inc.suspect_phones):
                    inc.suspect_phones.append(fact(v.normalized))
        elif q.field == "incident_date" and not inc.incident_date.present:
            inc.incident_date = fact(V.to_iso_ist(text) or ans.text.strip())
        elif q.field == "scam_type" and inc.scam_type == ScamType.UNKNOWN:
            st, _ = guess_from_text(ans.text)
            if st != ScamType.UNKNOWN:
                inc.scam_type = st
                inc.scam_type_evidence = fact(ans.text.strip()[:80])


def resolve_conflicts(conflicts: list[Conflict], session: SessionState, inc: Incident) -> None:
    """A conflict is resolved when the victim's answer contains one of the values."""
    for c in conflicts:
        for q in session.questions:
            if q.field != f"conflict:{c.field}":
                continue
            for a in session.answers:
                if a.question_id != q.id:
                    continue
                said = V.normalize_text(a.text)
                for val in c.values:
                    if V.only_digits(val) and V.only_digits(val) in V.only_digits(said):
                        c.resolution = val
                    elif val.lower() in said.lower():
                        c.resolution = val
        if c.resolution:
            for tx in inc.transactions:
                f = getattr(tx, c.field, None)
                if isinstance(f, Fact) and f.value in c.values:
                    f.value = c.resolution
                    f.note = "confirmed by victim"


def apply_corrections(inc: Incident, corrections: dict[str, str]) -> list[str]:
    """Corrections typed by the victim on the confirmation screen. Returns errors."""
    errors = []
    kinds = {"utr": "utr", "amount": "amount", "payee_upi": "upi", "payee_account": "account",
             "payee_ifsc": "ifsc", "victim_account_last4": "last4"}
    for path, value in corrections.items():
        idx, fname = _tx_index(path)
        kind = kinds.get(fname)
        val = value.strip()
        if kind and val:
            res = V.VALIDATORS[kind](val)
            if not res.ok:
                errors.append(f"{path}: {res.reason}")
                continue
            val = res.normalized
        f = Fact(value=val or None, source_kind="correction", source_id="victim", quote=value, verified=True)
        if idx is not None:
            while idx >= len(inc.transactions):
                inc.transactions.append(Transaction())
            setattr(inc.transactions[idx], fname, f)
            if fname == "utr" and val:
                inc.transactions[idx].utr_kind = V.validate_utr(val).kind
        elif fname in ("victim_bank", "victim_account_last4", "victim_name", "incident_date", "impersonated_entity"):
            setattr(inc, fname, f)
        elif fname == "scam_type":
            try:
                inc.scam_type = ScamType(val)
            except ValueError:
                errors.append(f"{path}: unknown scam type")
        else:
            errors.append(f"{path}: cannot be corrected here")
    return errors


def finalize(inc: Incident, session: SessionState) -> None:
    # payee UPI IDs / accounts are also suspect identifiers
    for tx in inc.transactions:
        if tx.payee_upi.present and all(f.value != tx.payee_upi.value for f in inc.suspect_upi_ids):
            inc.suspect_upi_ids.append(tx.payee_upi.model_copy())
        if tx.payee_account.present and all(f.value != tx.payee_account.value for f in inc.suspect_accounts):
            inc.suspect_accounts.append(tx.payee_account.model_copy())
        if tx.utr.present and not tx.utr_kind:
            tx.utr_kind = V.validate_utr(tx.utr.value).kind
    # de-duplicate lists by value
    for name in ("suspect_phones", "suspect_upi_ids", "suspect_accounts", "suspect_links"):
        seen, out = set(), []
        for f in getattr(inc, name):
            if f.value not in seen:
                seen.add(f.value)
                out.append(f)
        setattr(inc, name, out)
    if inc.victim_bank.present:
        canonical = match_bank(inc.victim_bank.value)
        if canonical:
            inc.victim_bank.value = canonical
    if inc.scam_type == ScamType.UNKNOWN:
        st, kw = guess_from_text(" ".join(s.text for s in session.segments))
        if st != ScamType.UNKNOWN:
            inc.scam_type = st


async def analyze(session: SessionState, gw: Gateway, settings: Settings, storage: Storage) -> dict:
    if not session.segments and not session.screenshots and not session.answers:
        raise WorkflowError("Record your account of what happened or upload a screenshot first.")
    if session.analysis_count >= settings.max_analyses_per_session:
        raise WorkflowError("Analysis limit reached for this session.", 429)
    session.analysis_count += 1

    inc, method = await extract_incident(session, gw, settings)
    rejected = ground_incident(inc, session)
    conflicts = merge_screenshot_transactions(inc, session)
    bind_answers(inc, session)
    resolve_conflicts(conflicts, session, inc)
    if session.corrections:
        apply_corrections(inc, session.corrections)
    rejected += ground_incident(inc, session)  # screenshot/answer facts are re-checked too
    finalize(inc, session)

    session.incident = inc
    session.extraction_method = method
    session.rejected = rejected
    session.conflicts = conflicts

    answered = {q.field for q in session.questions if q.answered}
    answered |= {f.split(":", 1)[1] for f in answered if f.startswith("conflict:")}
    needs = find_needs(inc, [c for c in conflicts if not c.resolution], answered)
    for n in needs:
        if n.field in {c.field for c in conflicts}:
            n.field = f"conflict:{n.field}"
    open_fields = {q.field for q in session.questions if not q.answered}
    new_needs = [n for n in needs if n.field not in open_fields][: settings.max_questions_per_round]
    lang = session.language_code or normalize(session.language_hint)
    for n in new_needs:
        q = Question(id=f"q{len(session.questions) + 1}", field=n.field, text_en=n.question_en,
                     text=await translate(n.question_en, lang, gw, session.usage))
        spoken = await speak(q.text, lang, gw, settings, session.usage)
        if spoken:
            q.audio_file_id = storage.put_file(session.id, spoken[0], "question_audio", spoken[1], f"{q.id}.mp3")
        session.questions.append(q)

    if session.stage in (Stage.CREATED, Stage.INTAKE):
        session.stage = Stage.ANALYZED
    ready, blockers = ready_for_documents(inc)
    return {"method": method, "ready_for_documents": ready, "blockers": blockers}


async def add_answer(session: SessionState, question_id: str | None, text: str | None, audio: bytes | None,
                     gw: Gateway, settings: Settings) -> Answer:
    q = next((q for q in session.questions if q.id == question_id), None) if question_id else None
    if question_id and not q:
        raise WorkflowError("Unknown question id.", 404)
    if audio:
        segs = await add_narration(session, audio, gw, settings, purpose="answer")
        text = " ".join(s.text for s in segs)
    if not text or not text.strip():
        raise WorkflowError("Empty answer.")
    if len(text) > 2000:
        raise WorkflowError("Answer too long (max 2000 characters).")
    ans = Answer(id=f"a{len(session.answers) + 1}", question_id=question_id, text=text.strip(),
                 language_code=session.language_code)
    session.answers.append(ans)
    if q:
        q.answered = True
    return ans


# --------------------------------------------------------------------------
# Read-back summary and confirmation
# --------------------------------------------------------------------------

def _human_dt(value: str | None) -> str | None:
    from app.core.golden_hour import parse_dt

    dt = parse_dt(value)
    if not dt:
        return value
    if "T" in (value or ""):
        return dt.strftime("%d %b %Y, %I:%M %p")
    return dt.strftime("%d %b %Y")


def summary_text_en(inc: Incident, for_speech: bool = False) -> str:
    def num(v: str | None) -> str:
        return V.spell_digits(v) if (for_speech and v) else (v or "")

    lines = ["Here is what I understood."]
    total = inc.total_amount()
    n = len(inc.transactions)
    if n:
        lines.append(f"You lost {V.format_inr(total)} in {n} payment{'s' if n > 1 else ''}.")
    for i, tx in enumerate(inc.transactions, 1):
        parts = [f"Payment {i}: {V.format_inr(tx.amount.value) if tx.amount.present else 'amount not known yet'}"]
        if tx.datetime.present:
            parts.append(f"on {_human_dt(tx.datetime.value)}")
        if tx.payee_upi.present:
            parts.append(f"to UPI ID {tx.payee_upi.value}")
        elif tx.payee_account.present:
            parts.append(f"to account {num(tx.payee_account.value)}")
        parts.append(f"transaction ID {num(tx.utr.value)}" if tx.utr.present else "transaction ID not known yet")
        lines.append(", ".join(parts) + ".")
    if inc.impersonated_entity.present:
        lines.append(f"The scammer pretended to be {inc.impersonated_entity.value}.")
    if inc.scam_type not in (ScamType.UNKNOWN,):
        lines.append(f"Type of fraud: {LABELS_EN[inc.scam_type]}.")
    if inc.suspect_phones:
        lines.append("Scammer phone: " + ", ".join(num(p.value) for p in inc.suspect_phones) + ".")
    if inc.victim_bank.present:
        acct = f", account ending {num(inc.victim_account_last4.value)}" if inc.victim_account_last4.present else ""
        lines.append(f"Your bank: {inc.victim_bank.value}{acct}.")
    lines.append("Is this correct? Confirm, or tell me what to change.")
    return " ".join(lines)


async def readback(session: SessionState, gw: Gateway, settings: Settings, storage: Storage) -> dict:
    from app.documents.generator import identifiers

    lang = session.language_code or normalize(session.language_hint)
    protect = identifiers(session.incident)
    text_en = summary_text_en(session.incident)
    text = await translate_protected(text_en, lang, gw, session.usage, protect)
    spoken_protect = protect + [V.spell_digits(v) for v in protect]
    speech_src = await translate_protected(summary_text_en(session.incident, for_speech=True), lang, gw,
                                           session.usage, spoken_protect)
    audio = await speak(speech_src, lang, gw, settings, session.usage)
    file_id = storage.put_file(session.id, audio[0], "readback_audio", audio[1], "summary.mp3") if audio else None
    return {"language_code": lang, "text": text, "text_en": text_en, "audio_file_id": file_id}


def confirm(session: SessionState, corrections: dict[str, str]) -> list[str]:
    errors = apply_corrections(session.incident, corrections)
    if errors:
        return errors
    session.corrections.update(corrections)
    ok, blockers = ready_for_documents(session.incident)
    if not ok:
        return [f"Still needed: {b}" for b in blockers]
    from app.models.session import utcnow

    session.confirmed_at = utcnow()
    session.stage = Stage.CONFIRMED
    return []


def status_payload(session: SessionState) -> dict:
    return {
        "timer": timer(session.incident),
        "checklist": checklist(session.incident),
        "language": plan_for(session.language_code or session.language_hint).as_dict(),
    }
