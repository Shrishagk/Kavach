"""Application settings, loaded from environment variables / .env."""
from __future__ import annotations

import base64
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Runtime mode ---------------------------------------------------
    env: Literal["dev", "prod", "test"] = "dev"
    mock_mode: bool = Field(
        default=False,
        description="Use deterministic fake Sarvam responses (no key, no credits).",
    )

    # --- Sarvam ------------------------------------------------------------
    sarvam_api_key: SecretStr | None = None
    sarvam_timeout_s: float = 60.0
    sarvam_max_retries: int = 3
    sarvam_max_concurrency: int = 3  # parallel calls per request (rate limit ~60/min)

    stt_model: str = "saaras:v4"
    stt_mode: str = "codemix"
    stt_chunk_seconds: float = 25.0  # REST STT accepts <=30s per request
    max_audio_seconds: float = 300.0

    llm_model: str = "sarvam-105b"
    llm_reasoning_effort: Literal["low", "medium", "high"] = "low"
    llm_max_tokens: int = 6000
    llm_temperature: float = 0.1

    vision_mode: Literal["auto", "doc_ai", "llm"] = "auto"
    vision_llm_model: str = "gemma4"  # image-capable model on Sarvam's v2 endpoint
    doc_ai_timeout_s: float = 30.0

    tts_model: str = "bulbul:v4-flash"
    tts_fallback_model: str = "bulbul:v3"  # used for languages v4-flash lacks (ml, od)
    tts_speaker: str | None = None  # None = model default voice
    tts_pace: float = 0.95

    translate_model: str = "sarvam-translate:v1"

    # --- Security / storage ------------------------------------------------
    session_secret: SecretStr = SecretStr("dev-only-change-me")
    data_dir: Path = BASE_DIR / "var"
    session_ttl_hours: int = 24
    signed_url_ttl_s: int = 900
    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    admin_token: SecretStr | None = None

    # --- Abuse / cost limits -------------------------------------------------
    rate_limit_per_minute: int = 40
    max_audio_bytes: int = 15 * 1024 * 1024
    max_image_bytes: int = 6 * 1024 * 1024
    max_screenshots_per_session: int = 6
    max_narrations_per_session: int = 8
    max_analyses_per_session: int = 12
    max_questions_per_round: int = 3

    @model_validator(mode="after")
    def _check(self) -> "Settings":
        if not self.mock_mode and self.env != "test" and not self.sarvam_api_key:
            raise ValueError(
                "SARVAM_API_KEY is not set. Put it in .env, or set MOCK_MODE=true to run without it."
            )
        if self.env == "prod" and self.session_secret.get_secret_value() == "dev-only-change-me":
            raise ValueError("Set SESSION_SECRET to a long random value in production.")
        return self

    # Derived helpers -----------------------------------------------------
    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def fernet_key(self) -> bytes:
        digest = hashlib.sha256(b"goldenhour-fernet:" + self.session_secret.get_secret_value().encode()).digest()
        return base64.urlsafe_b64encode(digest)

    def signing_key(self) -> bytes:
        return hashlib.sha256(b"goldenhour-sign:" + self.session_secret.get_secret_value().encode()).digest()


@lru_cache
def get_settings() -> Settings:
    return Settings()
