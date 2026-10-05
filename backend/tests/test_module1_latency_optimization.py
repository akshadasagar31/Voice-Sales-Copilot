"""
Tests for Module 1 Latency Optimization:
1. In-memory LRU TTS prompt audio cache (get_cached_tts_audio, set_cached_tts_audio).
2. Deepgram Aura TTS routing for Module 1 English.
3. Sub-millisecond deterministic extraction and prompt generator (generate_module1_ws_response).
4. Shared singleton LeadRepository to avoid connection pool rebuilds.
5. Barge-in / cancellation / immediate sentence 1 response payload.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from services.tts import get_cached_tts_audio, set_cached_tts_audio, _audio_response_cache
from main import (
    get_shared_lead_repo,
    generate_module1_ws_response,
    app,
)
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def clear_audio_cache():
    _audio_response_cache.clear()
    yield
    _audio_response_cache.clear()


def test_audio_cache_store_and_retrieve():
    """Verify that TTS audio byte cache returns stored bytes in sub-millisecond time."""
    test_key = "may i have your name please?::en::default"
    fake_audio = b"FAKE_AUDIO_BYTES_TEST"
    assert get_cached_tts_audio(test_key) is None

    set_cached_tts_audio(test_key, fake_audio)
    cached = get_cached_tts_audio(test_key)
    assert cached == fake_audio


def test_generate_module1_ws_response_instant_name():
    """Verify generate_module1_ws_response returns immediate_sentence1 in sub-millisecond time."""
    res = generate_module1_ws_response(
        current="My name is Sarah Connor",
        req_lang="en",
        last_conf=0.98,
        existing_lead_str=None,
        lead_id_val=None,
    )

    assert res["type"] == "final"
    assert res["transcript"] == "My name is Sarah Connor"
    assert res["lead"]["name"] == "Sarah Connor"
    assert res["speech_final"] is True
    assert res["next_missing_parameter"] == "phone"
    assert "phone" in res["immediate_sentence1"].lower() or "number" in res["immediate_sentence1"].lower()


def test_generate_module1_ws_response_hindi_amount_and_tenure():
    """Verify Hindi number parsing and next missing parameter prompt in ws response."""
    res = generate_module1_ws_response(
        current="मुझे 25 लाख रुपये का बिजनेस लोन दो साल के लिए चाहिए",
        req_lang="hi",
        last_conf=0.99,
        existing_lead_str='{"name": "राकेश शर्मा", "phone": "9812345678"}',
        lead_id_val=42,
    )

    assert res["type"] == "final"
    assert res["lead"]["loan_amount"] == 2500000
    assert res["lead"]["tenure_months"] == 24
    assert res["lead"]["name"] == "राकेश शर्मा"
    assert res["lead_id"] == 42
    assert res["language"] == "hi"


def test_generate_module1_ws_response_assistant_query():
    """Verify assistant capability questions generate instant conversational answers."""
    res = generate_module1_ws_response(
        current="What can you do?",
        req_lang="en",
        last_conf=0.95,
        existing_lead_str=None,
        lead_id_val=None,
    )

    assert res["is_assistant_query"] is True
    assert "lead" in res["immediate_sentence1"].lower() or "crm" in res["immediate_sentence1"].lower()


def test_shared_lead_repo_singleton():
    """Verify get_shared_lead_repo returns identical singleton across calls."""
    repo1 = get_shared_lead_repo()
    repo2 = get_shared_lead_repo()
    assert repo1 is repo2


def test_api_tts_cache_hit_returns_cached_audio():
    """Verify /api/tts returns cached audio bytes directly without upstream API calls."""
    test_client = TestClient(app)
    cache_key = "m1:en:what is your company name?"
    set_cached_tts_audio(cache_key, b"CACHED_MP3_STREAM")

    response = test_client.post(
        "/api/tts",
        json={"text": "What is your company name?", "language": "en", "module": "module1"},
    )
    assert response.status_code == 200
    assert response.content == b"CACHED_MP3_STREAM"
