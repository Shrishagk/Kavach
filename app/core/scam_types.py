"""Scam taxonomy: labels, keyword hints and type-specific advice."""
from __future__ import annotations

from app.models.incident import ScamType

LABELS_EN = {
    ScamType.DIGITAL_ARREST: "Digital arrest / fake police or agency threat",
    ScamType.INVESTMENT: "Investment or stock-trading fraud",
    ScamType.KYC_UPDATE: "Fake KYC / account-block update",
    ScamType.PARCEL_COURIER: "Fake parcel / courier / customs case",
    ScamType.PART_TIME_JOB: "Part-time job / online task fraud",
    ScamType.LOAN_APP: "Instant loan app fraud or harassment",
    ScamType.SEXTORTION: "Sextortion / video-call blackmail",
    ScamType.UPI_COLLECT_QR: "UPI collect request / QR code fraud",
    ScamType.CUSTOMER_CARE: "Fake customer-care number",
    ScamType.LOTTERY_PRIZE: "Lottery / prize / gift fraud",
    ScamType.IMPERSONATION_RELATIVE: "Impersonation of a relative, friend or boss",
    ScamType.OTHER: "Other online financial fraud",
    ScamType.UNKNOWN: "Not yet identified",
}

# Lower-case keyword hints (English + common romanised Hindi/Kannada words).
# Used only by the offline baseline extractor and as a cross-check, never as the final word.
KEYWORDS: dict[ScamType, list[str]] = {
    ScamType.DIGITAL_ARREST: ["digital arrest", "cbi", "ed officer", "enforcement directorate", "narcotics",
                              "money laundering", "arrest warrant", "police officer", "video call police",
                              "customs officer", "trai", "skype", "ncb", "court order"],
    ScamType.INVESTMENT: ["investment", "trading", "stock tips", "ipo", "crypto", "returns", "profit",
                          "telegram group", "share market"],
    ScamType.KYC_UPDATE: ["kyc", "account will be blocked", "pan update", "aadhaar update", "account blocked"],
    ScamType.PARCEL_COURIER: ["parcel", "courier", "fedex", "dhl", "package", "drugs in parcel"],
    ScamType.PART_TIME_JOB: ["part time job", "part-time", "task", "like videos", "rating task", "work from home"],
    ScamType.LOAN_APP: ["loan app", "instant loan", "recovery agent", "morphed"],
    ScamType.SEXTORTION: ["nude", "obscene video", "video call recorded", "blackmail"],
    ScamType.UPI_COLLECT_QR: ["collect request", "scan qr", "qr code", "enter pin to receive", "receive money"],
    ScamType.CUSTOMER_CARE: ["customer care", "helpline number", "refund", "anydesk", "teamviewer", "screen share"],
    ScamType.LOTTERY_PRIZE: ["lottery", "prize", "won", "lucky draw", "gift"],
    ScamType.IMPERSONATION_RELATIVE: ["my son", "my daughter", "uncle", "boss", "friend in hospital", "accident"],
}

ADVICE_EN: dict[ScamType, list[str]] = {
    ScamType.DIGITAL_ARREST: [
        "No police, CBI, ED or customs officer arrests anyone or collects money over a video call.",
        "Disconnect, and do not transfer more money even if they threaten you.",
    ],
    ScamType.INVESTMENT: [
        "Do not pay any 'withdrawal fee' or 'tax' to release profits — that is the next stage of the same fraud.",
        "Save screenshots of the trading app/website and the chat group before they disappear.",
    ],
    ScamType.KYC_UPDATE: [
        "Banks do not ask for OTP, PIN or remote-access apps for KYC. Change your net-banking password now.",
    ],
    ScamType.CUSTOMER_CARE: [
        "If you installed a screen-sharing app (AnyDesk, TeamViewer, etc.), uninstall it and change passwords.",
    ],
    ScamType.UPI_COLLECT_QR: [
        "You never need to enter your UPI PIN to receive money. Check your UPI app for pending collect requests.",
    ],
    ScamType.SEXTORTION: [
        "Do not pay. Paying usually leads to more demands. Keep the evidence and report it.",
    ],
    ScamType.LOAN_APP: [
        "Uninstall the app after taking screenshots, and revoke its contact/photo permissions.",
    ],
}

GENERIC_ADVICE_EN = [
    "Never share OTP, UPI PIN, card CVV or passwords with anyone.",
    "Beware of 'recovery agents' who promise to get your money back for a fee — that is another scam.",
]


def guess_from_text(text: str) -> tuple[ScamType, str | None]:
    t = (text or "").lower()
    best, best_hits, best_kw = ScamType.UNKNOWN, 0, None
    for st, kws in KEYWORDS.items():
        hits = [k for k in kws if k in t]
        if len(hits) > best_hits:
            best, best_hits, best_kw = st, len(hits), hits[0]
    return best, best_kw


def ncrp_subcategory(payment_modes: list[str]) -> str:
    """Suggested NCRP 'Online Financial Fraud' sub-category from the payment mode.

    The portal's exact labels can change; the victim should pick the closest one.
    """
    modes = " ".join(payment_modes).lower()
    if "upi" in modes:
        return "UPI Related Frauds"
    if "card" in modes:
        return "Debit/Credit Card Fraud"
    if "wallet" in modes:
        return "E-Wallet Related Fraud"
    if any(m in modes for m in ("neft", "imps", "rtgs", "net banking", "internet banking")):
        return "Internet Banking Related Fraud"
    return "Online Financial Fraud (choose the closest sub-category)"
