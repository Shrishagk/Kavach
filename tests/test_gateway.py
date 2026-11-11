from app.services.gateway import resolve_chat_reasoning_effort


def test_105b_never_sends_none():
    assert resolve_chat_reasoning_effort("sarvam-105b", None, "low") == "low"
    assert resolve_chat_reasoning_effort("sarvam-105b", "none", "medium") == "medium"
    assert resolve_chat_reasoning_effort("sarvam-105b", "default", "high") == "high"


def test_105b_passes_through_valid_effort():
    assert resolve_chat_reasoning_effort("sarvam-105b", "medium", "low") == "medium"


def test_gemma_omits_when_disabled():
    assert resolve_chat_reasoning_effort("gemma4", None, "low") is None
    assert resolve_chat_reasoning_effort("gemma4", "none", "low") is None
    assert resolve_chat_reasoning_effort("gemma4", "default", "low") == "low"
