# GoldenHour — backend

Voice-first, multilingual cyber-fraud reporting assistant. A victim speaks in their own language
(22 Indian languages + English, code-mixed is fine) and uploads payment screenshots. GoldenHour turns
that into **evidence-linked** drafts: a complaint for cybercrime.gov.in, a bank-freeze request email,
a 1930 call script (text + audio) and a golden-hour checklist.

It is **not** a chatbot. It is a fixed pipeline where AI does narrow jobs and deterministic code
checks every result:

```
 voice note ──► ffmpeg (16 kHz WAV, ≤25 s chunks at pauses) ──► Saaras v4 STT (codemix, keyterms)
 screenshot ──► Sarvam Document AI extract (fallback: image LLM)  ──► validators ──┐
                                                                                   ▼
 transcripts + answers ──► Sarvam-105B (strict JSON schema: value + source id + quote)
                                   │
                                   ▼
                    GROUNDING GATE (no AI): quote must exist in the cited source,
                    digits/UPI/amount must be recoverable from it, format validators pass
                                   │  rejected values → follow-up questions, never guesses
                                   ▼
        merge screenshot facts (conflicts surfaced) ─► bind answers ─► missing-field policy
                                   │
               questions (Sarvam-Translate + Bulbul voice) ◄─┘
                                   ▼
          read-back (translated with identifiers protected) ─► victim CONFIRMS / corrects
                                   ▼
     complaint PDF (bilingual + evidence annexure) · freeze email · 1930 script + audio · checklist
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# also needs ffmpeg on PATH; WeasyPrint needs Pango (macOS: brew install pango; Ubuntu: apt install libpango-1.0-0)
cp .env.example .env                                      # then edit .env
python -c "import secrets; print(secrets.token_urlsafe(48))"   # paste into SESSION_SECRET
```

**Without a key** (frontend work, demos offline): set `MOCK_MODE=true`. Every Sarvam call returns a
deterministic Kannada-English "digital arrest" story, so the whole flow works without spending credits.

**With your key**: set `SARVAM_API_KEY=...` and `MOCK_MODE=false`, then:

```bash
python scripts/live_smoke_test.py      # checks every Sarvam API (~₹1–2), tells you what to fix
uvicorn app.main:app --reload          # API at http://localhost:8000, docs at /docs
pytest -q                              # 19 tests, run in mock mode
python scripts/run_eval.py             # field-level accuracy on 12 synthetic cases
python scripts/run_eval.py --voice --limit 4   # live: Bulbul speaks each case, Saaras transcribes it
```

## API flow (for the frontend)

All session calls need the header `X-Session-Token: <session_token>`. File URLs in responses are
short-lived signed links (no header needed — use directly in `<audio src>` / downloads).

| Step | Call | Notes |
|---|---|---|
| 1 | `POST /api/v1/sessions` `{"consent": true, "language_hint": "kn-IN"}` | returns `session_id`, `session_token`, privacy notice |
| 2 | `POST /sessions/{id}/narration` (multipart `audio`) | webm/ogg/m4a/mp3/wav, ≤5 min; call again for more |
| 3 | `POST /sessions/{id}/screenshots` (multipart `image`) | PNG/JPEG/WebP, ≤6 per session |
| 4 | `POST /sessions/{id}/analyze` | returns `incident`, `open_questions` (each with `audio_url`), `conflicts`, `rejected`, `timer` |
| 5 | `POST /sessions/{id}/answers` `{"question_id","text"}` or `/answers/voice` (multipart `audio`, `question_id`) | re-analyzes automatically |
| 6 | `POST /sessions/{id}/readback` | summary text in victim's language + English + `audio_url` |
| 7 | `POST /sessions/{id}/confirm` `{"confirmed": true, "corrections": {"transactions[1].utr": "..."}}` | corrections are validated |
| 8 | `POST /sessions/{id}/documents` | `documents[]`: complaint_pdf, freeze_email, call_script, call_script_audio, checklist |
| — | `GET /sessions/{id}` · `DELETE /sessions/{id}` · `GET /api/v1/meta/languages` · `GET /api/v1/admin/metrics` | |

Every response is the same `SessionView`, so the UI can re-render from it after each step.

## Design decisions judges may ask about

- **No invented details.** The LLM must cite a source id and verbatim quote for every value. `core/grounding.py`
  re-checks each one without AI; anything unsupported is dropped and asked as a question. The PDF's
  Annexure A lists where every value came from. Tested in `tests/test_grounding.py` (including a prompt-injection case).
- **Screenshots beat speech for numbers**, but disagreements are shown to the victim as a question, never silently resolved.
- **Identifiers survive translation.** UTRs, UPI IDs and numbers are swapped for placeholders before Sarvam-Translate
  and restored after; if a placeholder is lost, the English sentence is kept instead of a corrupted number.
- **Honest language coverage.** Speech input: 22 languages + English. Spoken replies: 11 (Bulbul v4-flash, with v3 for
  Malayalam/Odia). Others get text output. `GET /api/v1/meta/languages` reports this per language.
- **Graceful degradation.** Sarvam errors retry with backoff; if the LLM fails, a regex baseline extractor keeps the
  flow working (`extraction_method` in the response says which was used); Document AI falls back to an image LLM.
- **Privacy.** Consent first. Raw voice is never stored. Session data and files are encrypted (Fernet) and auto-deleted
  after 24 h or on DELETE. Only the hash of the session token is stored. Logs mask digits, UPI IDs and IFSCs.
  Only the last 4 digits of the victim's own account are collected.
- **Cost control.** Per-IP rate limit, per-session caps, usage counters with an estimated ₹ cost per session.
- **Scope.** GoldenHour drafts; it never files complaints, contacts banks or gives legal advice.

## Project layout

```
app/
  main.py               FastAPI app, middleware (rate limit, security headers), error mapping
  config.py             settings from .env
  api/routes.py         HTTP endpoints
  core/                 deterministic logic: validators, grounding, missing-field policy, scam taxonomy,
                        golden-hour checklist, encrypted storage, PII-masking logs, metrics, languages
  services/             Sarvam gateway (+ mock), audio chunking, extraction (LLM), vision, speech, workflow
  documents/            PDF/email/script generation (Jinja2 + WeasyPrint, Noto fonts bundled)
  data/banks.json       bank names/aliases/IFSC prefixes (fraud contacts left blank on purpose — fill from official sites)
scripts/                live_smoke_test.py, run_eval.py
eval/cases.jsonl        12 synthetic cases, 7 languages, gold answers
tests/                  pytest suite (mock mode)
```

## Before the demo — verify

- Run `scripts/live_smoke_test.py` with each teammate's key; if Document AI fails, set `VISION_MODE=llm`.
- Model names change: if a check returns 404, update the model in `.env` (see docs.sarvam.ai).
- Fill `fraud_phone` / `fraud_email` in `app/data/banks.json` only from banks' official websites.
- The NCRP sub-category in the PDF is a suggestion; the portal's labels may differ.
- All test data (names, numbers, UPI IDs) is synthetic.
