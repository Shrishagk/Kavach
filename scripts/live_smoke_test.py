"""Check every Sarvam API GoldenHour uses, with your real key. Run on your laptop:

    python scripts/live_smoke_test.py

Costs roughly ₹1–2 of credits. Each check is independent, so one failure does
not hide the others. Outputs (audio, test receipt) go to ./smoke_out/.
"""
from __future__ import annotations

import asyncio
import io
import json
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.core.grounding import ground_incident  # noqa: E402
from app.models.session import SessionState, TranscriptSegment, UsageCounters  # noqa: E402
from app.services import audio as A  # noqa: E402
from app.services.extraction import extract_incident  # noqa: E402
from app.services.gateway import SarvamGateway, SarvamServiceError  # noqa: E402
from app.services.vision import read_screenshot  # noqa: E402

OUT = Path("smoke_out")
HINTS = {
    401: "Key rejected. Check SARVAM_API_KEY in .env (no quotes, no spaces).",
    403: "Key not allowed for this API. Check your Sarvam dashboard plan/permissions.",
    402: "Out of credits. Use another team member's key.",
    404: "Endpoint/model not found. The model name in .env may have changed — check docs.sarvam.ai.",
    422: "Request shape rejected. Run `pip install -U sarvamai` and check the changelog.",
    429: "Rate limited. Wait a minute and rerun.",
}
results: list[tuple[str, bool, float, str]] = []


async def check(name: str, coro):
    t0 = time.perf_counter()
    try:
        detail = await coro
        results.append((name, True, time.perf_counter() - t0, detail or ""))
    except SarvamServiceError as e:
        results.append((name, False, time.perf_counter() - t0,
                        f"HTTP {e.status}: {e.message[:160]}  → {HINTS.get(e.status, 'see error')}"))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        results.append((name, False, time.perf_counter() - t0, f"{type(e).__name__}: {e}"))


def receipt_png() -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (720, 900), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(str(Path(__file__).resolve().parent.parent / "app/assets/fonts/NotoSans-Regular.ttf"), 30)
        big = ImageFont.truetype(str(Path(__file__).resolve().parent.parent / "app/assets/fonts/NotoSans-Bold.ttf"), 48)
    except OSError:
        font = big = ImageFont.load_default()
    lines = [("SYNTHETIC TEST RECEIPT", font), ("Payment Successful", font), ("Rs 2,500.00", big),
             ("Paid to Test Merchant", font), ("test.merchant@ybl", font), ("12 Nov 2026, 10:15 am", font),
             ("UPI transaction ID: 498765432101", font), ("Debited from: State Bank of India XX1234", font)]
    y = 60
    for text, f in lines:
        d.text((50, y), text, fill="black", font=f)
        y += 90 if f is big else 60
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


async def main():
    s = get_settings()
    if s.mock_mode:
        sys.exit("MOCK_MODE is true in .env — set MOCK_MODE=false to test the real API.")
    OUT.mkdir(exist_ok=True)
    gw = SarvamGateway(s)
    usage = UsageCounters()
    tts_audio: dict[str, bytes] = {}

    async def t_translate():
        out = await gw.translate("Please read out the transaction ID.", source="en-IN", target="kn-IN", usage=usage)
        return out

    async def t_tts():
        audio, mime = await gw.tts("ನಮಸ್ಕಾರ. ನಾನು ಐವತ್ತು ಸಾವಿರ ರೂಪಾಯಿ PhonePe ಮೂಲಕ ಕಳಿಸಿದೆ.", language_code="kn-IN",
                                   model=s.tts_model, usage=usage)
        (OUT / "tts_kn.mp3").write_bytes(audio)
        tts_audio["kn"] = audio
        return f"{len(audio)} bytes → smoke_out/tts_kn.mp3 ({s.tts_model})"

    async def t_tts_fallback():
        audio, _ = await gw.tts("നമസ്കാരം", language_code="ml-IN", model=s.tts_fallback_model, usage=usage)
        return f"{len(audio)} bytes ({s.tts_fallback_model}, Malayalam)"

    async def t_stt():
        if "kn" not in tts_audio:
            raise RuntimeError("skipped: TTS failed, no audio to transcribe")
        chunks, dur = await A.chunk_audio(tts_audio["kn"], s.stt_chunk_seconds, 300)
        res = await gw.transcribe(chunks[0].wav, language_code=None, keyterms=["PhonePe"], usage=usage, duration_s=dur)
        return f"[{res.language_code}] {res.transcript}"

    async def t_llm():
        now = datetime.now(timezone.utc)
        sess = SessionState(id="smoke", expires_at=now + timedelta(hours=1), consent_at=now, language_code="kn-IN")
        sess.segments.append(TranscriptSegment(id="t1", text="ಒಬ್ಬರು CBI officer ಅಂತ call ಮಾಡಿದ್ರು. ನಾನು ₹50,000 "
                                               "cbi.verify2@ybl ಗೆ ಕಳಿಸಿದೆ. UTR 412398765432. ನನ್ನ account SBI ದು."))
        inc, method = await extract_incident(sess, gw, s)
        rejected = ground_incident(inc, sess)
        if method != "llm":
            raise RuntimeError("LLM output unusable; pipeline fell back to the regex baseline")
        tx = inc.transactions[0] if inc.transactions else None
        return (f"{s.llm_model}: scam={inc.scam_type.value}, amount={tx and tx.amount.value}, utr={tx and tx.utr.value}, "
                f"upi={[f.value for f in inc.suspect_upi_ids]}, bank={inc.victim_bank.value}, rejected={len(rejected)}")

    png = receipt_png()
    (OUT / "test_receipt.png").write_bytes(png)

    async def t_doc_ai():
        s.vision_mode = "doc_ai"
        fields, method, _ = await read_screenshot(png, "image/png", gw, s, usage)
        return f"{method}: amount={fields.get('amount')}, utr={fields.get('utr')}, upi={fields.get('payee_upi')}"

    async def t_vision_llm():
        s.vision_mode = "llm"
        fields, method, _ = await read_screenshot(png, "image/png", gw, s, usage)
        return f"{method} ({s.vision_llm_model}): amount={fields.get('amount')}, utr={fields.get('utr')}"

    await check("Translate (sarvam-translate)", t_translate())
    await check("Text-to-speech (Kannada)", t_tts())
    await check("Text-to-speech fallback (Malayalam)", t_tts_fallback())
    await check("Speech-to-text (Saaras, round-trip)", t_stt())
    await check("LLM extraction + grounding", t_llm())
    await check("Screenshot via Document AI", t_doc_ai())
    await check("Screenshot via vision LLM (fallback)", t_vision_llm())

    print("\n" + "=" * 78)
    for name, ok, secs, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<40} {secs:5.1f}s\n      {detail}")
    print("=" * 78)
    print(f"Approx. cost of this run: ₹{usage.estimated_cost_inr()}  |  calls: {usage.sarvam_calls}, errors: {usage.sarvam_errors}")
    doc_ok = next(r[1] for r in results if r[0].startswith("Screenshot via Document"))
    vis_ok = next(r[1] for r in results if r[0].startswith("Screenshot via vision"))
    if not doc_ok and vis_ok:
        print("Tip: Document AI failed but the vision model works → set VISION_MODE=llm in .env for the demo.")
    (OUT / "results.json").write_text(json.dumps([r[:2] + (round(r[2], 2), r[3]) for r in results], indent=1,
                                                 ensure_ascii=False), encoding="utf-8")
    sys.exit(0 if all(r[1] for r in results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
