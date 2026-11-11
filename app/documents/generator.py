"""Generate the four GoldenHour outputs from a confirmed incident.

1. Complaint draft PDF (English + victim's language), with an evidence annexure
2. Bank-freeze request email (plain text, English — banks process English)
3. 1930 call script (victim's language + English) and its audio
4. Golden-hour checklist (JSON for the app's timer screen)

Documents are drafts for the victim to submit; nothing is filed automatically.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from app.config import Settings
from app.core import validators as V
from app.core.golden_hour import IST, checklist, parse_dt, timer
from app.core.grounding import iter_facts
from app.core.languages import FONT_FOR_LANG, NAMES, normalize
from app.core.scam_types import LABELS_EN, ncrp_subcategory
from app.core.storage import Storage
from app.models.incident import Incident, ScamType
from app.models.session import DocumentRecord, SessionState, Stage
from app.services.gateway import Gateway
from app.services.speech import speak, translate_protected

TEMPLATES = Path(__file__).resolve().parent / "templates"
FONTS = Path(__file__).resolve().parent.parent / "assets" / "fonts"
env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))

CHANNEL_LABELS = {
    "phone_call": "Phone call", "video_call": "Video call", "whatsapp": "WhatsApp", "sms": "SMS",
    "telegram": "Telegram", "email": "Email", "social_media": "Social media", "website_or_app": "Website / app",
    "in_person": "In person", "other": "Other", "unknown": "Not stated",
}


def _fmt_dt(value: str | None) -> str | None:
    if not value:
        return None
    dt = parse_dt(value)
    if not dt:
        return value  # keep exactly as the victim said it
    return dt.astimezone(IST).strftime("%d %b %Y, %I:%M %p" if "T" in value else "%d %b %Y")


def identifiers(inc: Incident) -> list[str]:
    vals = [f.value for _, f, _ in iter_facts(inc) if f.present]
    vals += [V.format_inr(t.amount.value) for t in inc.transactions if t.amount.present]
    return [v for v in vals if v and any(ch.isdigit() or ch == "@" for ch in v)]


def _font_list(lang: str) -> tuple[list[tuple[str, str, str]], str]:
    fams = {"Noto Sans": "NotoSans"}
    local = FONT_FOR_LANG.get(lang)
    if local:
        fams[local] = local.replace(" ", "")
    fonts = [(fam, (FONTS / f"{stem}-Regular.ttf").as_uri(), (FONTS / f"{stem}-Bold.ttf").as_uri())
             for fam, stem in fams.items()]
    return fonts, Markup(f'"{local}"' if local else '"Noto Sans"')


def complaint_context(session: SessionState, narrative_local: str | None) -> dict:
    inc = session.incident
    lang = normalize(session.language_code or session.language_hint)
    fonts, local_css = _font_list(lang)
    txs = []
    for t in inc.transactions:
        payee = t.payee_upi.value or t.payee_account.value
        if t.payee_account.present and t.payee_ifsc.present:
            payee = f"{t.payee_account.value} (IFSC {t.payee_ifsc.value})"
        mode = " / ".join(x for x in (t.payment_mode.value, t.payment_app.value) if x) or "—"
        txs.append({"amount": V.format_inr(t.amount.value) if t.amount.present else "not stated",
                    "datetime": _fmt_dt(t.datetime.value), "utr": t.utr.value, "payee": payee, "mode": mode})
    evidence = []
    for path, f, _ in iter_facts(inc):
        if f.present:
            src = {"transcript": f"Victim's statement ({f.source_id})", "answer": f"Victim's answer ({f.source_id})",
                   "screenshot": f"Payment screenshot ({f.source_id})", "correction": "Corrected by victim on review",
                   "computed": "Derived"}.get(f.source_kind or "", f.source_id or "—")
            evidence.append({"field": path, "value": f.value, "source": src, "quote": (f.quote or "")[:220]})
    return {
        "lang_short": lang.split("-")[0], "fonts": fonts, "local_font_css": local_css,
        "language_name": NAMES.get(lang, lang), "generated_at": datetime.now(IST).strftime("%d %b %Y, %I:%M %p"),
        "ref": f"GH-{session.id[:8].upper()}",
        "ncrp_subcategory": ncrp_subcategory([t.payment_mode.value or "" for t in inc.transactions]
                                             + (["upi"] if any(t.payee_upi.present for t in inc.transactions) else [])),
        "scam_label": LABELS_EN[inc.scam_type], "total": V.format_inr(inc.total_amount()), "txs": txs,
        "incident_date": _fmt_dt(inc.incident_date.value) or (txs[0]["datetime"] if txs and txs[0]["datetime"] else None),
        "channel": CHANNEL_LABELS.get(inc.first_contact_channel.value, "Not stated"),
        "impersonated": inc.impersonated_entity.value, "victim_bank": inc.victim_bank.value,
        "last4": inc.victim_account_last4.value,
        "suspect_phones": ", ".join(f.value for f in inc.suspect_phones),
        "suspect_upis": ", ".join(f.value for f in inc.suspect_upi_ids),
        "suspect_accounts": ", ".join(f.value for f in inc.suspect_accounts),
        "suspect_links": ", ".join(f.value for f in inc.suspect_links),
        "narrative_en": inc.narrative_en or "(No description generated — please describe the incident in your own words on the portal.)",
        "narrative_local": narrative_local,
        "confirmed_at": session.confirmed_at.astimezone(IST).strftime("%d %b %Y, %I:%M %p") if session.confirmed_at else "—",
        "evidence": evidence, "rejected": session.rejected, "screenshots": len(session.screenshots),
    }


def render_pdf(context: dict) -> bytes:
    html = env.get_template("complaint.html").render(**context)
    try:
        from weasyprint import HTML
        return HTML(string=html, base_url=str(TEMPLATES)).write_pdf()
    except (ImportError, OSError):
        # WeasyPrint needs GTK/Pango shared libraries, which are often absent on
        # Windows. Keep PDF generation available with this pure-Python fallback.
        return _render_pdf_reportlab(context)


def _render_pdf_reportlab(context: dict) -> bytes:
    from io import BytesIO

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    font_path = FONTS / "NotoSans-Regular.ttf"
    bold_path = FONTS / "NotoSans-Bold.ttf"
    font_name, bold_name = "GoldenHourNoto", "GoldenHourNotoBold"
    if font_path.exists() and bold_path.exists():
        pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
        pdfmetrics.registerFont(TTFont(bold_name, str(bold_path)))
    else:
        font_name, bold_name = "Helvetica", "Helvetica-Bold"

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
                            topMargin=16*mm, bottomMargin=16*mm, title="Cybercrime Complaint Draft")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="GHBody", parent=styles["BodyText"], fontName=font_name,
                              fontSize=8.5, leading=12, alignment=TA_LEFT, spaceAfter=4))
    styles.add(ParagraphStyle(name="GHHeading", parent=styles["Heading2"], fontName=bold_name,
                              fontSize=11, leading=14, textColor=colors.HexColor("#17365D"), spaceBefore=8))
    styles.add(ParagraphStyle(name="GHTitle", parent=styles["Title"], fontName=bold_name,
                              fontSize=16, leading=20, textColor=colors.HexColor("#17365D")))

    def para(value: object, style: str = "GHBody") -> Paragraph:
        return Paragraph(escape(str(value or "—")), styles[style])

    from xml.sax.saxutils import escape

    story = [para("Cybercrime Complaint Draft", "GHTitle"),
             para(f"Reference: {context.get('ref')}  |  Generated: {context.get('generated_at')}"),
             para("Please review and submit this draft through the appropriate official channel."),
             para("Incident summary", "GHHeading")]
    summary = [
        ("Scam type", context.get("scam_label")), ("Amount lost", context.get("total")),
        ("Incident date", context.get("incident_date")), ("Contact channel", context.get("channel")),
        ("Impersonated entity", context.get("impersonated")), ("Victim bank", context.get("victim_bank")),
        ("Account ending", context.get("last4")), ("Suspect phone numbers", context.get("suspect_phones")),
        ("Suspect UPI IDs", context.get("suspect_upis")), ("Suspect accounts", context.get("suspect_accounts")),
        ("Suspect links", context.get("suspect_links")),
    ]
    rows = [[para(k), para(v)] for k, v in summary if v]
    table = Table(rows, colWidths=[42*mm, 132*mm], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), bold_name), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D5DDE5")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F2F5F8")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.extend([table, para("Narrative", "GHHeading"), para(context.get("narrative_en"))])
    if context.get("narrative_local"):
        story.extend([para(f"Victim language ({context.get('language_name')})", "GHHeading"),
                      para(context["narrative_local"])])
    if context.get("txs"):
        story.append(para("Transactions", "GHHeading"))
        for i, tx in enumerate(context["txs"], 1):
            story.append(para(f"{i}. Amount: {tx.get('amount')}; Date/time: {tx.get('datetime')}; "
                              f"UTR: {tx.get('utr')}; Paid to: {tx.get('payee')}; Mode: {tx.get('mode')}"))
    if context.get("evidence"):
        story.append(para("Evidence annexure", "GHHeading"))
        for item in context["evidence"]:
            story.append(para(f"{item.get('field')}: {item.get('value')} — "
                              f"{item.get('source')}. {item.get('quote')}"))
    doc.build(story)
    return buffer.getvalue()


def freeze_email(session: SessionState) -> str:
    inc = session.incident
    bank = inc.victim_bank.value or "[your bank]"
    acct = f"ending {inc.victim_account_last4.value}" if inc.victim_account_last4.present else "ending [last 4 digits]"
    lines = [
        f"Subject: URGENT – Cyber fraud – request to block account and hold/freeze beneficiary funds – A/c {acct}",
        "",
        f"To: Fraud / Customer Care Team, {bank}",
        "",
        "Dear Sir/Madam,",
        "",
        f"I am the holder of account {acct} with {bank}. I have been the victim of an online financial fraud "
        f"({LABELS_EN[inc.scam_type].lower()}). I request you to urgently:",
        "  1. register this as a fraud complaint and block further debits from my account/cards/UPI;",
        "  2. raise a request with the beneficiary bank(s) to hold/freeze the amounts below; and",
        "  3. share the complaint reference number with me.",
        "",
        "Fraudulent transactions:",
    ]
    for i, t in enumerate(inc.transactions, 1):
        lines.append(f"  {i}. Amount: {V.format_inr(t.amount.value) if t.amount.present else '[amount]'}"
                     f" | Date/time: {_fmt_dt(t.datetime.value) or '[date & time]'}"
                     f" | UTR/Txn ID: {t.utr.value or '[UTR]'}"
                     f" | Paid to: {t.payee_upi.value or t.payee_account.value or '[beneficiary]'}")
    lines += [
        f"  Total: {V.format_inr(inc.total_amount())}",
        "",
        "Suspect details: " + (", ".join(f.value for f in inc.suspect_phones + inc.suspect_upi_ids + inc.suspect_accounts) or "—"),
        "",
        "I have reported / am reporting this on the National Cybercrime Helpline 1930 and at cybercrime.gov.in. "
        "Acknowledgement number: [add after filing].",
        "",
        "Regards,",
        "[Your name]",
        "[Registered mobile number]",
    ]
    return "\n".join(lines)


def call_script_en(inc: Incident) -> str:
    parts = ["Hello, I want to report an online financial fraud."]
    if inc.scam_type != ScamType.UNKNOWN:
        parts.append(f"It was a {LABELS_EN[inc.scam_type].lower()}.")
    parts.append(f"I lost {V.format_inr(inc.total_amount())} in {len(inc.transactions)} transaction(s).")
    for i, t in enumerate(inc.transactions, 1):
        s = f"Transaction {i}: {V.format_inr(t.amount.value) if t.amount.present else 'amount unknown'}"
        if t.datetime.present:
            s += f", on {_fmt_dt(t.datetime.value)}"
        if t.utr.present:
            s += f", UTR {t.utr.value}"
        if t.payee_upi.present:
            s += f", paid to UPI ID {t.payee_upi.value}"
        parts.append(s + ".")
    if inc.victim_bank.present:
        parts.append(f"The money was debited from my {inc.victim_bank.value} account"
                     + (f" ending {inc.victim_account_last4.value}." if inc.victim_account_last4.present else "."))
    if inc.suspect_phones:
        parts.append("The fraudster's phone number: " + ", ".join(f.value for f in inc.suspect_phones) + ".")
    parts.append("Please help me block these transactions. What is my acknowledgement number?")
    return " ".join(parts)


async def generate_all(session: SessionState, gw: Gateway, settings: Settings, storage: Storage) -> list[DocumentRecord]:
    inc = session.incident
    lang = normalize(session.language_code or session.language_hint)
    protect = identifiers(inc)
    docs: list[DocumentRecord] = []

    narrative_local = None
    if inc.narrative_en and lang != "en-IN":
        narrative_local = await translate_protected(inc.narrative_en, lang, gw, session.usage, protect)

    pdf = await asyncio.to_thread(render_pdf, complaint_context(session, narrative_local))
    fid = storage.put_file(session.id, pdf, "complaint_pdf", "application/pdf", "goldenhour-complaint-draft.pdf")
    docs.append(DocumentRecord(kind="complaint_pdf", file_id=fid, filename="goldenhour-complaint-draft.pdf",
                               mime="application/pdf"))

    email = freeze_email(session)
    fid = storage.put_file(session.id, email.encode(), "freeze_email", "text/plain; charset=utf-8", "bank-freeze-request.txt")
    docs.append(DocumentRecord(kind="freeze_email", file_id=fid, filename="bank-freeze-request.txt",
                               mime="text/plain; charset=utf-8"))

    script_en = call_script_en(inc)
    script_local = await translate_protected(script_en, lang, gw, session.usage, protect) if lang != "en-IN" else script_en
    script_txt = (f"{NAMES.get(lang, lang)}:\n{script_local}\n\nEnglish:\n{script_en}\n"
                  if lang != "en-IN" else script_en)
    fid = storage.put_file(session.id, script_txt.encode(), "call_script", "text/plain; charset=utf-8", "1930-call-script.txt")
    docs.append(DocumentRecord(kind="call_script", file_id=fid, filename="1930-call-script.txt",
                               mime="text/plain; charset=utf-8"))

    # Audio: spaced-out digits so the voice reads identifiers one digit at a time.
    spoken_src = script_local
    for v in protect:
        if v and sum(ch.isdigit() for ch in v) >= 6:
            spoken_src = spoken_src.replace(v, V.spell_digits(v))
    audio = await speak(spoken_src, lang, gw, settings, session.usage)
    if audio:
        ext = "mp3" if audio[1] == "audio/mpeg" else "wav"
        fid = storage.put_file(session.id, audio[0], "call_script_audio", audio[1], f"1930-call-script.{ext}")
        docs.append(DocumentRecord(kind="call_script_audio", file_id=fid, filename=f"1930-call-script.{ext}", mime=audio[1]))

    # Checklist, translated in one call (one step per line); English kept alongside.
    steps = checklist(inc)
    if lang != "en-IN":
        joined = "\n".join(st["title_en"] for st in steps)
        try:
            translated = (await gw.translate(joined, source="en-IN", target=lang, usage=session.usage)).split("\n")
        except Exception:  # noqa: BLE001 - checklist translation is best-effort
            translated = []
        if len(translated) == len(steps):
            for st, tr in zip(steps, translated):
                st["title"] = tr.strip()
    for st in steps:
        st.setdefault("title", st["title_en"])
    payload = {"timer": timer(inc), "steps": steps}
    fid = storage.put_file(session.id, json.dumps(payload, ensure_ascii=False).encode(), "checklist",
                           "application/json", "golden-hour-checklist.json")
    docs.append(DocumentRecord(kind="checklist", file_id=fid, filename="golden-hour-checklist.json",
                               mime="application/json"))

    session.documents = docs
    session.stage = Stage.DOCUMENTS_READY
    return docs

