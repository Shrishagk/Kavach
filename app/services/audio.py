"""Audio normalization and silence-aware chunking (ffmpeg).

Sarvam's REST speech-to-text accepts up to 30 s per request, so narrations are
converted to 16 kHz mono WAV and cut into <=25 s chunks, preferring cut points
inside pauses so words are not split in half.
"""
from __future__ import annotations

import asyncio
import io
import re
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path


class AudioError(Exception):
    pass


def _ffmpeg() -> str:
    """Use a system ffmpeg when present, otherwise imageio-ffmpeg's bundled binary."""
    import shutil

    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as e:
        raise AudioError("ffmpeg is required for audio conversion; install ffmpeg or run pip install -r requirements.txt") from e


@dataclass
class Chunk:
    index: int
    start_s: float
    end_s: float
    wav: bytes


async def _run(*args: str, stdin: bytes | None = None) -> tuple[bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        *args, stdin=asyncio.subprocess.PIPE if stdin else None,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate(stdin)
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg failed: {err.decode(errors='ignore')[-300:]}")
    return out, err


async def to_wav_16k_mono(data: bytes) -> bytes:
    """Decode any browser/phone format (webm, ogg, m4a, mp3, wav...) to 16 kHz mono PCM WAV.

    Also strips metadata. Reading from a temp file (not stdin) lets ffmpeg seek in MP4/M4A.
    """
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "in", Path(d) / "out.wav"
        src.write_bytes(data)
        # write to a file (not a pipe) so the WAV header carries the real length
        await _run(_ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(src),
                   "-map_metadata", "-1", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst))
        out = dst.read_bytes() if dst.exists() else b""
    if len(out) < 1000:
        raise AudioError("no audio could be decoded")
    return out


def wav_duration(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes)) as w:
        return w.getnframes() / float(w.getframerate())


async def detect_silences(wav_bytes: bytes, noise_db: int = -32, min_silence: float = 0.35) -> list[tuple[float, float]]:
    _, err = await _run(_ffmpeg(), "-nostdin", "-hide_banner", "-i", "pipe:0", "-af",
                        f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-", stdin=wav_bytes)
    text = err.decode(errors="ignore")
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", text)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", text)]
    return list(zip(starts, ends))


def plan_cuts(duration: float, silences: list[tuple[float, float]], max_len: float) -> list[tuple[float, float]]:
    """Greedy: from the current start, cut at the latest silence midpoint within max_len."""
    mids = [(s + e) / 2 for s, e in silences]
    cuts, start = [], 0.0
    while duration - start > max_len:
        window = [m for m in mids if start + 3.0 < m <= start + max_len]
        cut = window[-1] if window else start + max_len
        cuts.append((start, cut))
        start = cut
    cuts.append((start, duration))
    return [(round(a, 3), round(b, 3)) for a, b in cuts if b - a > 0.2]


def slice_wav(wav_bytes: bytes, start: float, end: float) -> bytes:
    with wave.open(io.BytesIO(wav_bytes)) as w:
        rate, width, ch = w.getframerate(), w.getsampwidth(), w.getnchannels()
        w.setpos(int(start * rate))
        frames = w.readframes(int((end - start) * rate))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as o:
        o.setnchannels(ch)
        o.setsampwidth(width)
        o.setframerate(rate)
        o.writeframes(frames)
    return buf.getvalue()


async def chunk_audio(data: bytes, max_chunk_s: float, max_total_s: float) -> tuple[list[Chunk], float]:
    wav_bytes = await to_wav_16k_mono(data)
    duration = wav_duration(wav_bytes)
    if duration > max_total_s:
        raise AudioError(f"recording is {duration:.0f}s; the limit is {max_total_s:.0f}s")
    if duration < 0.5:
        raise AudioError("recording is too short")
    silences = await detect_silences(wav_bytes) if duration > max_chunk_s else []
    cuts = plan_cuts(duration, silences, max_chunk_s)
    chunks = [Chunk(i, a, b, slice_wav(wav_bytes, a, b)) for i, (a, b) in enumerate(cuts)]
    return chunks, duration


def silent_wav(seconds: float = 0.5, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as o:
        o.setnchannels(1)
        o.setsampwidth(2)
        o.setframerate(rate)
        o.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()
