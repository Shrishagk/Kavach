"""Decide which details are still missing and phrase follow-up questions.

Deterministic on purpose: the order of questions is a policy decision
(freeze-critical first), not something to leave to a model. Questions are
written in English here and translated by Sarvam-Translate into the victim's
language by the workflow.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.validators import format_inr
from app.models.incident import Conflict, Incident, ScamType


@dataclass
class Need:
    field: str
    priority: int  # lower = ask first
    question_en: str


def find_needs(inc: Incident, conflicts: list[Conflict], answered_fields: set[str]) -> list[Need]:
    needs: list[Need] = []

    for c in conflicts:
        if c.resolution is None:
            vals = " or ".join(c.values)
            needs.append(Need(c.field, 0, f"I found two different values for the same detail: {vals}. Which one is correct?"))

    if not inc.transactions:
        needs.append(Need("transactions", 1,
                          "Did you send any money to the scammer? If yes, how much, and through which app or bank?"))
    for i, tx in enumerate(inc.transactions):
        amt = format_inr(tx.amount.value) if tx.amount.present else None
        label = f"the {amt} payment" if amt else f"payment number {i + 1}"
        if not tx.amount.present:
            needs.append(Need(f"transactions[{i}].amount", 1, f"How much money was sent in {label}? Please say the amount in rupees."))
        if not tx.utr.present:
            needs.append(Need(f"transactions[{i}].utr", 2,
                              f"Please read out the UTR or transaction ID for {label}. It is in your UPI app's payment "
                              "history or the bank SMS. You can also upload a screenshot instead."))
        if not tx.datetime.present:
            needs.append(Need(f"transactions[{i}].datetime", 4, f"At what date and time did you make {label}?"))

    if not inc.victim_bank.present:
        needs.append(Need("victim_bank", 3, "Which bank account was the money taken from? Please say the bank's name."))
    elif not inc.victim_account_last4.present:
        needs.append(Need("victim_account_last4", 5,
                          "What are the last 4 digits of that bank account? Only the last 4 digits, please."))

    if not (inc.suspect_phones or inc.suspect_upi_ids or inc.suspect_accounts
            or any(t.payee_upi.present or t.payee_account.present for t in inc.transactions)):
        needs.append(Need("suspect_identifiers", 4,
                          "Do you have the scammer's phone number, or the UPI ID or account you paid to?"))

    if inc.scam_type in (ScamType.UNKNOWN,):
        needs.append(Need("scam_type", 6, "How did this start? Did someone call you, message you, or send a link?"))

    if not inc.incident_date.present and not any(t.datetime.present for t in inc.transactions):
        needs.append(Need("incident_date", 6, "On what date did this happen?"))

    needs = [n for n in needs if n.field not in answered_fields]
    needs.sort(key=lambda n: n.priority)
    return needs


def ready_for_documents(inc: Incident) -> tuple[bool, list[str]]:
    """Minimum bar for generating documents. Everything else is optional."""
    blockers = []
    if not inc.transactions or not any(t.amount.present for t in inc.transactions):
        blockers.append("at least one payment amount")
    return (not blockers, blockers)
