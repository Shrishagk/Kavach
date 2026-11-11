"""GoldenHour backend — FastAPI application entry point.

Run:  uvicorn app.main:app --reload
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import Settings, get_settings
from app.core.logging_utils import setup_logging
from app.core.metrics import metrics
from app.core.storage import Storage
from app.services.gateway import SarvamServiceError, build_gateway
from app.services.workflow import WorkflowError

log = logging.getLogger("goldenhour")


class RateLimiter:
    """Per-IP sliding window. Protects API credits from abuse; in-memory (single instance)."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        q = self.hits[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            return False
        q.append(now)
        return True


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging("DEBUG" if settings.env == "dev" else "INFO")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async def purge_loop():
            while True:
                try:
                    n = app.state.storage.purge_expired()
                    if n:
                        log.info("purged %d expired sessions", n)
                except Exception:  # noqa: BLE001
                    log.exception("purge failed")
                await asyncio.sleep(600)

        task = asyncio.create_task(purge_loop())
        log.info("GoldenHour backend started (mock_mode=%s, env=%s)", settings.mock_mode, settings.env)
        yield
        task.cancel()

    app = FastAPI(
        title="GoldenHour API",
        version="1.0.0",
        description="Voice-first multilingual cyber-fraud reporting assistant. Converts a victim's spoken "
                    "account and payment screenshots into evidence-linked complaint drafts.",
        lifespan=lifespan,
        docs_url="/docs" if settings.env != "prod" else None,
        redoc_url=None,
    )
    app.state.settings = settings
    app.state.storage = Storage(settings)
    app.state.gateway = build_gateway(settings)
    limiter = RateLimiter(settings.rate_limit_per_minute)

    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type", "X-Session-Token", "X-Admin-Token"],
    )

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.url.path.startswith("/api/") and request.method != "OPTIONS":
            ip = request.client.host if request.client else "unknown"
            if not limiter.allow(ip):
                metrics.inc("rate_limited")
                return JSONResponse({"detail": "Too many requests. Please wait a minute."}, status_code=429)
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Response-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.0f}"
        return response

    @app.exception_handler(WorkflowError)
    async def workflow_error(_: Request, exc: WorkflowError):
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)

    @app.exception_handler(SarvamServiceError)
    async def sarvam_error(_: Request, exc: SarvamServiceError):
        log.error("Sarvam error op=%s status=%s", exc.op, exc.status)
        status = 503 if exc.status in (None, 429, 500, 502, 503, 504) else 502
        return JSONResponse({"detail": exc.user_message, "service": exc.op, "upstream_status": exc.status},
                            status_code=status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        return JSONResponse({"detail": "Invalid request", "errors": exc.errors()}, status_code=422)

    @app.get("/health", tags=["meta"])
    def health():
        return {"status": "ok", "mock_mode": settings.mock_mode, "version": app.version}

    app.include_router(router)
    return app


app = create_app()
