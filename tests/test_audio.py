import asyncio

from app.services.audio import chunk_audio, plan_cuts
from tests.conftest import make_audio


def test_plan_cuts_prefers_silence():
    cuts = plan_cuts(60.0, [(11.0, 12.0), (23.0, 24.0), (40.0, 41.0), (47.0, 48.0)], 25.0)
    assert all(b - a <= 25.0 for a, b in cuts)
    assert cuts[0] == (0.0, 23.5)


def test_chunk_real_audio_under_limit():
    data = make_audio(62.0)
    chunks, duration = asyncio.run(chunk_audio(data, 25.0, 300.0))
    assert 61 < duration < 63
    assert len(chunks) >= 3
    assert all(c.end_s - c.start_s <= 25.01 for c in chunks)
