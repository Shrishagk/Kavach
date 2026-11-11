from datetime import datetime, timedelta, timezone

from app.models.incident import Fact, Incident, Transaction
from app.models.session import ScreenshotRecord, SessionState
from app.services.workflow import bind_answers, merge_screenshot_transactions
from app.models.session import Answer, Question


def make_session():
    now = datetime.now(timezone.utc)
    return SessionState(id="m", expires_at=now + timedelta(hours=1), consent_at=now)


def shot(amount, utr):
    return ScreenshotRecord(id="img1", file_id="f", method="mock", raw_fields={
        "is_payment_receipt": "yes", "status": "Success", "amount": amount, "utr": utr,
        "visible_text": f"Paid {amount} UTR {utr}"}, ocr_text=f"Paid {amount} UTR {utr}")


def test_same_utr_different_amount_is_a_conflict():
    s = make_session()
    s.screenshots.append(shot("₹5,000.00", "412398765432"))
    inc = Incident(transactions=[Transaction(amount=Fact(value="50000.00", source_id="t1"),
                                             utr=Fact(value="412398765432", source_id="t1"))])
    conflicts = merge_screenshot_transactions(inc, s)
    assert len(inc.transactions) == 1
    assert conflicts and conflicts[0].field == "amount"
    assert set(conflicts[0].values) == {"50000.00", "5000.00"}


def test_unmatched_screenshot_adds_transaction():
    s = make_session()
    s.screenshots.append(shot("₹9,999.00", "498765432109"))
    inc = Incident(transactions=[Transaction(amount=Fact(value="50000.00", source_id="t1"),
                                             utr=Fact(value="412398765432", source_id="t1"))])
    merge_screenshot_transactions(inc, s)
    assert len(inc.transactions) == 2


def test_answer_binding_ignores_label_words():
    s = make_session()
    s.questions.append(Question(id="q1", field="transactions[0].utr", text="?", text_en="?", answered=True))
    s.answers.append(Answer(id="a1", question_id="q1", text="UTR no: 6123 4509 8765"))
    inc = Incident(transactions=[Transaction(amount=Fact(value="1000.00", source_id="t1"))])
    bind_answers(inc, s)
    assert inc.transactions[0].utr.value == "612345098765"
