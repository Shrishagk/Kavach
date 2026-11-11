"""Encrypted session + file storage with automatic expiry.

* Session state is a JSON blob encrypted with Fernet (AES-128-CBC + HMAC).
* Files (screenshots, generated PDFs, TTS audio) are encrypted on disk.
* Raw narration audio is never stored: it is deleted right after transcription.
* Everything is purged when the session expires (default 24h) or on DELETE.
* Clients prove ownership with a random session token; only its hash is kept.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from app.config import Settings
from app.models.session import SessionState


class SessionNotFound(Exception):
    pass


class Storage:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = Path(settings.data_dir)
        self.files_dir = self.root / "files"
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "goldenhour.db"
        self.fernet = Fernet(settings.fernet_key())
        self._locks: dict[str, asyncio.Lock] = {}
        self._init_db()

    # ------------------------------------------------------------------ db
    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("""CREATE TABLE IF NOT EXISTS sessions(
                id TEXT PRIMARY KEY, token_hash TEXT NOT NULL, expires_at REAL NOT NULL, blob BLOB NOT NULL)""")
            c.execute("""CREATE TABLE IF NOT EXISTS files(
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL, kind TEXT, mime TEXT,
                filename TEXT, path TEXT NOT NULL, created_at REAL NOT NULL)""")
            c.execute("CREATE INDEX IF NOT EXISTS files_session ON files(session_id)")

    def lock(self, session_id: str) -> asyncio.Lock:
        return self._locks.setdefault(session_id, asyncio.Lock())

    # ------------------------------------------------------------ sessions
    @staticmethod
    def _hash_token(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def create_session(self, language_hint: str | None) -> tuple[SessionState, str]:
        now = datetime.now(timezone.utc)
        sid = uuid.uuid4().hex
        token = secrets.token_urlsafe(32)
        state = SessionState(
            id=sid, created_at=now, consent_at=now, language_hint=language_hint,
            expires_at=now + timedelta(hours=self.settings.session_ttl_hours),
        )
        with self._conn() as c:
            c.execute("INSERT INTO sessions VALUES (?,?,?,?)",
                      (sid, self._hash_token(token), state.expires_at.timestamp(), self._encrypt_state(state)))
        return state, token

    def _encrypt_state(self, state: SessionState) -> bytes:
        return self.fernet.encrypt(state.model_dump_json().encode())

    def load(self, session_id: str, token: str | None) -> SessionState:
        with self._conn() as c:
            row = c.execute("SELECT token_hash, expires_at, blob FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not row:
            raise SessionNotFound()
        token_hash, expires_at, blob = row
        if not token or not hmac.compare_digest(token_hash, self._hash_token(token)):
            raise SessionNotFound()  # same error as missing: don't reveal existence
        if expires_at < time.time():
            self.delete_session(session_id)
            raise SessionNotFound()
        try:
            return SessionState.model_validate_json(self.fernet.decrypt(blob))
        except InvalidToken as e:  # secret rotated
            raise SessionNotFound() from e

    def save(self, state: SessionState) -> None:
        with self._conn() as c:
            c.execute("UPDATE sessions SET blob=? WHERE id=?", (self._encrypt_state(state), state.id))

    def delete_session(self, session_id: str) -> None:
        with self._conn() as c:
            paths = [r[0] for r in c.execute("SELECT path FROM files WHERE session_id=?", (session_id,))]
            c.execute("DELETE FROM files WHERE session_id=?", (session_id,))
            c.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        for p in paths:
            Path(p).unlink(missing_ok=True)
        self._locks.pop(session_id, None)

    def purge_expired(self) -> int:
        with self._conn() as c:
            ids = [r[0] for r in c.execute("SELECT id FROM sessions WHERE expires_at < ?", (time.time(),))]
        for sid in ids:
            self.delete_session(sid)
        return len(ids)

    # --------------------------------------------------------------- files
    def put_file(self, session_id: str, data: bytes, kind: str, mime: str, filename: str) -> str:
        fid = uuid.uuid4().hex
        path = self.files_dir / f"{fid}.bin"
        path.write_bytes(self.fernet.encrypt(data))
        with self._conn() as c:
            c.execute("INSERT INTO files VALUES (?,?,?,?,?,?,?)",
                      (fid, session_id, kind, mime, filename, str(path), time.time()))
        return fid

    def get_file(self, session_id: str, file_id: str) -> tuple[bytes, str, str]:
        with self._conn() as c:
            row = c.execute("SELECT path, mime, filename FROM files WHERE id=? AND session_id=?",
                            (file_id, session_id)).fetchone()
        if not row:
            raise FileNotFoundError(file_id)
        path, mime, filename = row
        return self.fernet.decrypt(Path(path).read_bytes()), mime, filename

    # --------------------------------------------------------- signed URLs
    def sign(self, session_id: str, file_id: str, ttl: int | None = None) -> str:
        exp = int(time.time()) + (ttl or self.settings.signed_url_ttl_s)
        msg = f"{session_id}:{file_id}:{exp}".encode()
        sig = hmac.new(self.settings.signing_key(), msg, hashlib.sha256).hexdigest()[:32]
        return f"/api/v1/sessions/{session_id}/files/{file_id}?exp={exp}&sig={sig}"

    def verify_signature(self, session_id: str, file_id: str, exp: int, sig: str) -> bool:
        if exp < time.time():
            return False
        msg = f"{session_id}:{file_id}:{exp}".encode()
        expected = hmac.new(self.settings.signing_key(), msg, hashlib.sha256).hexdigest()[:32]
        return hmac.compare_digest(expected, sig)
