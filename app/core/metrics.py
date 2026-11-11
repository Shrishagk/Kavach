"""In-process metrics for monitoring Sarvam calls (latency, errors) — no PII."""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class _Stat:
    calls: int = 0
    errors: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0
    last_error: str | None = None


@dataclass
class Metrics:
    started_at: float = field(default_factory=time.time)
    stats: dict[str, _Stat] = field(default_factory=lambda: defaultdict(_Stat))
    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _lock: Lock = field(default_factory=Lock)

    def record(self, op: str, ms: float, error: str | None = None) -> None:
        with self._lock:
            s = self.stats[op]
            s.calls += 1
            s.total_ms += ms
            s.max_ms = max(s.max_ms, ms)
            if error:
                s.errors += 1
                s.last_error = error[:200]

    def inc(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.counters[name] += n

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "uptime_s": int(time.time() - self.started_at),
                "sarvam": {
                    op: {"calls": s.calls, "errors": s.errors,
                         "avg_ms": round(s.total_ms / s.calls, 1) if s.calls else 0,
                         "max_ms": round(s.max_ms, 1), "last_error": s.last_error}
                    for op, s in self.stats.items()
                },
                "counters": dict(self.counters),
            }


metrics = Metrics()
