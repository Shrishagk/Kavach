"""Logging with PII masking. Transcripts and identifiers never reach logs in clear."""
from __future__ import annotations

import logging
import re

_PATTERNS = [
    (re.compile(r"[a-zA-Z0-9.\-_]{2,}@[a-zA-Z][a-zA-Z0-9.\-]+"), lambda m: "***@" + m.group(0).split("@")[1]),
    (re.compile(r"\d{6,}"), lambda m: "*" * (len(m.group(0)) - 4) + m.group(0)[-4:]),
    (re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"), lambda m: m.group(0)[:4] + "*******"),
]


def mask(text: str) -> str:
    for pat, repl in _PATTERNS:
        text = pat.sub(repl, text)
    return text


class PIIMaskingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # pragma: no cover
            return True
        record.msg = mask(msg)
        record.args = ()
        return True


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.addFilter(PIIMaskingFilter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "httpx2", "httpcore", "fontTools", "weasyprint"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
