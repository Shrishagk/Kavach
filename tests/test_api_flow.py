"""End-to-end flow in MOCK_MODE: the demo story from narration to documents."""
from tests.conftest import PNG_1PX, make_audio


def hdr(tok):
    return {"X-Session-Token": tok}


def start(client):
    r = client.post("/api/v1/sessions", json={"consent": True, "language_hint": "kn"})
    assert r.status_code == 201, r.text
    return r.json()["session_id"], r.json()["session_token"]


def test_consent_required(client):
    assert client.post("/api/v1/sessions", json={"consent": False}).status_code == 400


def test_full_flow(client):
    sid, tok = start(client)
    base = f"/api/v1/sessions/{sid}"

    # 1. victim speaks (3 recordings in the mock story)
    for _ in range(3):
        r = client.post(f"{base}/narration", headers=hdr(tok), files={"audio": ("v.webm", make_audio(4), "audio/webm")})
        assert r.status_code == 200, r.text
    v = r.json()
    assert v["language"]["code"] == "kn-IN" and len(v["segments"]) == 3

    # 2. payment screenshot
    r = client.post(f"{base}/screenshots", headers=hdr(tok), files={"image": ("s.png", PNG_1PX, "image/png")})
    assert r.status_code == 200, r.text

    # 3. analyze
    r = client.post(f"{base}/analyze", headers=hdr(tok))
    assert r.status_code == 200, r.text
    v = r.json()
    inc = v["incident"]
    assert inc["scam_type"] == "digital_arrest"
    assert len(inc["transactions"]) == 2, inc["transactions"]
    t1, t2 = inc["transactions"]
    assert t1["amount"]["value"] == "50000.00" and t1["utr"]["value"] == "412398765432"
    assert t1["utr"]["source_kind"] == "screenshot" and t1["payee_upi"]["value"] == "cbi.verify2@ybl"
    assert t2["amount"]["value"] == "120000.00" and t2["utr"]["value"] is None
    assert v["total_amount"] == 170000.0
    assert inc["victim_bank"]["value"] in ("SBI", "State Bank of India")
    assert any(p["value"] == "+919876543210" for p in inc["suspect_phones"])
    fields = [q["field"] for q in v["open_questions"]]
    assert fields[0] == "transactions[1].utr", fields
    assert len(fields) <= 3
    assert all(q["audio_url"] for q in v["open_questions"])

    # 4. answer by voice (mock STT returns the UTR answer), then by text
    q_utr = v["open_questions"][0]["id"]
    r = client.post(f"{base}/answers/voice", headers=hdr(tok), data={"question_id": q_utr},
                    files={"audio": ("a.webm", make_audio(2), "audio/webm")})
    assert r.status_code == 200, r.text
    v = r.json()
    assert v["incident"]["transactions"][1]["utr"]["value"] == "612345098765"
    assert v["incident"]["transactions"][1]["utr"]["source_kind"] == "answer"
    for q in v["open_questions"]:
        if q["field"] == "transactions[1].datetime":
            r = client.post(f"{base}/answers", headers=hdr(tok), json={"question_id": q["id"], "text": "11 Nov 2026, 11:05 am"})
            assert r.status_code == 200
    v = r.json()
    assert v["incident"]["transactions"][1]["datetime"]["value"].startswith("2026-11-11T11:05")

    # 5. read-back + confirm with a correction
    r = client.post(f"{base}/readback", headers=hdr(tok))
    assert r.status_code == 200 and "₹1,70,000.00" in r.json()["text_en"]
    r = client.post(f"{base}/documents", headers=hdr(tok))
    assert r.status_code == 409  # cannot generate before confirmation
    r = client.post(f"{base}/confirm", headers=hdr(tok), json={"confirmed": True,
                    "corrections": {"victim_account_last4": "4821"}})
    assert r.status_code == 200, r.text
    assert r.json()["stage"] == "confirmed"
    bad = client.post(f"{base}/confirm", headers=hdr(tok), json={"corrections": {"transactions[0].utr": "123"}})
    assert bad.status_code == 422

    # 6. documents
    r = client.post(f"{base}/documents", headers=hdr(tok))
    assert r.status_code == 200, r.text
    docs = {d["kind"]: d for d in r.json()["documents"]}
    assert {"complaint_pdf", "freeze_email", "call_script", "call_script_audio", "checklist"} <= set(docs)
    pdf = client.get(docs["complaint_pdf"]["url"])
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    email = client.get(docs["freeze_email"]["url"]).text
    assert "412398765432" in email and "612345098765" in email and "4821" in email
    script = client.get(docs["call_script"]["url"]).text
    assert "Kannada" in script and "cbi.verify2@ybl" in script
    checklist = client.get(docs["checklist"]["url"]).json()
    assert checklist["steps"][0]["id"] == "call_1930"

    # 7. usage is tracked, and delete wipes everything
    assert r.json()["usage"]["sarvam_calls"] > 0
    assert client.delete(base, headers=hdr(tok)).status_code == 204
    assert client.get(base, headers=hdr(tok)).status_code == 404
    assert client.get(docs["complaint_pdf"]["url"]).status_code == 404


def test_security_controls(client):
    sid, tok = start(client)
    base = f"/api/v1/sessions/{sid}"
    assert client.get(base).status_code == 404  # no token
    assert client.get(base, headers=hdr("wrong")).status_code == 404
    r = client.post(f"{base}/screenshots", headers=hdr(tok), files={"image": ("x.png", b"%PDF-1.4 not an image", "image/png")})
    assert r.status_code == 415
    r = client.get(f"{base}/files/abc?exp=9999999999&sig=deadbeef")
    assert r.status_code == 403
    assert client.post(f"{base}/analyze", headers=hdr(tok)).status_code == 400  # nothing to analyze yet
    r = client.post(f"{base}/narration", headers=hdr(tok), files={"audio": ("x.webm", b"not audio at all" * 100, "audio/webm")})
    assert r.status_code == 422


def test_languages_endpoint(client):
    langs = {l["code"]: l for l in client.get("/api/v1/meta/languages").json()["languages"]}
    assert len(langs) == 23
    assert langs["kn-IN"]["speech_output"] and langs["ml-IN"]["tts_model"] == "bulbul:v3"
    assert not langs["sat-IN"]["speech_output"]
