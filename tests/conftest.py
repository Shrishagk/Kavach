import os
import shutil
import subprocess
import tempfile

os.environ.setdefault("MOCK_MODE", "true")
os.environ.setdefault("ENV", "test")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="gh-test-"))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402


@pytest.fixture()
def settings(tmp_path):
    return Settings(mock_mode=True, env="test", data_dir=tmp_path, rate_limit_per_minute=1000)


@pytest.fixture()
def client(settings):
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


def make_audio(seconds: float = 3.0, fmt: str = "webm") -> bytes:
    """Tone bursts separated by silence, encoded like a browser recording."""
    # Prefer a system FFmpeg when available, but use imageio-ffmpeg's bundled
    # executable so the test suite also works on machines without FFmpeg on PATH.
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise RuntimeError(
                "Generating test audio requires FFmpeg. Install FFmpeg or run "
                "'pip install imageio-ffmpeg'."
            ) from exc
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, f"a.{fmt}")
        expr = "sin(2*PI*440*t)*lt(mod(t\\,4)\\,3)"
        subprocess.run([ffmpeg, "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i",
                        f"aevalsrc={expr}:s=16000:d={seconds}", "-c:a", "libopus" if fmt == "webm" else "pcm_s16le",
                        out], check=True)
        return open(out, "rb").read()


PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082")
