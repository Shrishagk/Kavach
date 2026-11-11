"""Field-level evaluation of the extraction + grounding pipeline.

    python scripts/run_eval.py                 # text mode (no STT); uses MOCK_MODE from .env
    python scripts/run_eval.py --voice         # live only: Bulbul speaks each case, Saaras transcribes it
    python scripts/run_eval.py --limit 3       # save credits

Reports precision/recall per field and, most importantly, how many values in
the final output were NOT in the gold set (hallucinations that got through).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.core.grounding import ground_incident  # noqa: E402
from app.models.session import SessionState, TranscriptSegment  # noqa: E402
from app.services import audio as A  # noqa: E402
from app.services.extraction import extract_incident  # noqa: E402
from app.core import baseline_extractor  # noqa: E402
from app.services.gateway import build_gateway  # noqa: E402
from app.services.speech import speak  # noqa: E402
from app.services.workflow import finalize  # noqa: E402

FIELDS = ("amounts", "utrs", "upi_ids", "phones", "victim_bank", "scam_type")


def predicted(inc) -> dict:
    return {
        "amounts": sorted({t.amount.value for t in inc.transactions if t.amount.present}),
        "utrs": sorted({t.utr.value for t in inc.transactions if t.utr.present}),
        "upi_ids": sorted({f.value for f in inc.suspect_upi_ids}),
        "phones": sorted({f.value for f in inc.suspect_phones}),
        "victim_bank": inc.victim_bank.value,
        "scam_type": inc.scam_type.value,
    }


async def run_case(case: dict, gw, settings, voice: bool) -> tuple[dict, dict, int, str]:
    now = datetime.now(timezone.utc)
    s = SessionState(id=case["id"], expires_at=now + timedelta(hours=1), consent_at=now, language_code=case["language"])
    for i, text in enumerate(case["segments"], 1):
        if voice:
            spoken = await speak(text, case["language"], gw, settings, s.usage)
            if spoken:
                chunks, _ = await A.chunk_audio(spoken[0], settings.stt_chunk_seconds, 300)
                parts = [(await gw.transcribe(c.wav, language_code=case["language"], keyterms=[], usage=s.usage,
                                              duration_s=c.end_s - c.start_s)).transcript for c in chunks]
                text = " ".join(parts)
        s.segments.append(TranscriptSegment(id=f"t{i}", text=text))
    try:
        inc, method = await extract_incident(s, gw, settings)
    except Exception as e:
        # An evaluation should report every case even if an external provider
        # or SDK has an unanticipated failure.  Production requests still use
        # their normal error mapping; this is only the batch-run safety net.
        print(f"  {case['id']}: extraction crashed ({type(e).__name__}); using baseline extractor")
        inc = baseline_extractor.extract([(seg.id, "transcript", seg.text) for seg in s.segments])
        method = "baseline-error"
    rejected = ground_incident(inc, s)
    finalize(inc, s)
    return predicted(inc), case["gold"] | {"scam_type": case["scam_type"]}, len(rejected), method


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cases", default=str(Path(__file__).resolve().parent.parent / "eval" / "cases.jsonl"))
    args = ap.parse_args()
    settings = get_settings()
    if args.voice and settings.mock_mode:
        sys.exit("--voice needs the real API (MOCK_MODE=false).")
    gw = build_gateway(settings)
    cases = [json.loads(line) for line in open(args.cases, encoding="utf-8") if line.strip()]
    if args.limit:
        cases = cases[: args.limit]

    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    rows = []
    for case in cases:
        pred, gold, n_rej, method = await run_case(case, gw, settings, args.voice)
        for f in FIELDS:
            p = set(pred[f]) if isinstance(pred[f], list) else ({pred[f]} if pred[f] else set())
            g = set(gold.get(f) or []) if isinstance(gold.get(f), list) else ({gold[f]} if gold.get(f) else set())
            tp[f] += len(p & g)
            fp[f] += len(p - g)
            fn[f] += len(g - p)
        ok = all((set(pred[f]) if isinstance(pred[f], list) else pred[f]) ==
                 (set(gold.get(f) or []) if isinstance(gold.get(f), list) else gold.get(f)) for f in FIELDS)
        rows.append((case["id"], "PASS" if ok else "diff", n_rej, "mock-baseline" if settings.mock_mode else method))
        if not ok:
            for f in FIELDS:
                if (set(pred[f]) if isinstance(pred[f], list) else pred[f]) != (set(gold.get(f) or []) if isinstance(gold.get(f), list) else gold.get(f)):
                    print(f"  {case['id']}.{f}: predicted={pred[f]} gold={gold.get(f)}")

    print(f"\nMode: {'voice (TTS->STT->LLM)' if args.voice else 'text'} | mock={settings.mock_mode} | cases={len(cases)}\n")
    print(f"{'field':<12} {'precision':>9} {'recall':>7} {'false+':>7}")
    for f in FIELDS:
        prec = tp[f] / (tp[f] + fp[f]) if tp[f] + fp[f] else 1.0
        rec = tp[f] / (tp[f] + fn[f]) if tp[f] + fn[f] else 1.0
        print(f"{f:<12} {prec:>9.2f} {rec:>7.2f} {fp[f]:>7}")
    print("\ncase                      result  rejected  method")
    for r in rows:
        print(f"{r[0]:<25} {r[1]:<7} {r[2]:>8}  {r[3]}")
    total_fp = sum(fp[f] for f in ("amounts", "utrs", "upi_ids", "phones"))
    print(f"\nIdentifier values in the output that are not in the gold answer (false positives): {total_fp}")


if __name__ == "__main__":
    asyncio.run(main())
