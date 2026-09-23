import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
import base64

from main import app
from services.sarvam_tts import (
    SarvamTTSService,
    SarvamTTSError,
    SarvamTTSConfigurationError,
    SarvamTTSAPIError,
    clean_english_financial_text,
)


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Unit Tests: clean_english_financial_text
# ---------------------------------------------------------------------------

def test_clean_english_financial_text_strips_markdown():
    raw = "The **HDFC Bank** personal loan requires a `CIBIL score` of *750*."
    cleaned = clean_english_financial_text(raw)
    assert "**" not in cleaned
    assert "*" not in cleaned
    assert "`" not in cleaned
    assert "HDFC Bank personal loan requires a of 750." in cleaned or "HDFC Bank" in cleaned


def test_clean_english_financial_text_expands_symbols():
    raw = "The interest rate is 10.5% p.a. with minimum loan amount of ₹50,000 or Rs. 100,000."
    cleaned = clean_english_financial_text(raw)
    assert "percent" in cleaned
    assert "per annum" in cleaned
    assert "rupees" in cleaned
    assert "₹" not in cleaned
    assert "Rs." not in cleaned


def test_clean_english_financial_text_empty():
    assert clean_english_financial_text("") == ""
    assert clean_english_financial_text("   ") == ""


# ---------------------------------------------------------------------------
# Unit Tests: SarvamTTSService for English (en-IN)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sarvam_tts_service_english_synthesis():
    service = SarvamTTSService(api_key="test_sarvam_key")
    mock_wav_bytes = b"RIFFfake_english_wav_bytes"
    b64_audio = base64.b64encode(mock_wav_bytes).decode("utf-8")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"audios": [b64_audio]}

    with patch("httpx.AsyncClient.post", return_value=mock_response) as mock_post:
        audio = await service.synthesize_speech(
            "Hello, your loan is approved at 10.5% interest.",
            language_code="en-IN",
        )
        assert audio == mock_wav_bytes
        assert mock_post.called
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["speaker"] == "simran"
        assert call_kwargs["json"]["language_code"] == "en-IN"
        assert call_kwargs["json"]["model"] == "bulbul:v3"
        assert "percent" in call_kwargs["json"]["text"]


# ---------------------------------------------------------------------------
# Integration Tests: POST /api/tts Module 2 English Routing
# ---------------------------------------------------------------------------

def test_api_tts_routes_module2_english_to_sarvam(client):
    """Verify that Module 2 English requests route to Sarvam Bulbul v3 en-IN with simran."""
    mock_wav_bytes = b"RIFFfake_english_wav_bytes"
    with patch.object(
        SarvamTTSService,
        "synthesize_speech",
        new_callable=AsyncMock,
        return_value=mock_wav_bytes,
    ) as mock_sarvam:
        response = client.post(
            "/api/tts",
            json={
                "text": "What is the minimum CIBIL score for an HDFC personal loan?",
                "language": "en",
                "module": "module2",
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.headers["x-tts-provider"] == "sarvam"
        assert response.headers["x-tts-language"] == "en-IN"
        assert response.headers["x-tts-voice"] == "simran"
        assert response.content == mock_wav_bytes
        mock_sarvam.assert_called_once()
        args, kwargs = mock_sarvam.call_args
        assert kwargs["language_code"] == "en-IN"
        assert kwargs["speaker"] == "simran"


def test_api_tts_module2_english_falls_back_to_deepgram_on_sarvam_error(client):
    """Verify that if Sarvam TTS fails or runs out of credits for Module 2 English, it automatically falls back to Deepgram."""
    mock_mp3_bytes = b"ID3\x03\x00\x00\x00\x00\x00#TSSEfake_deepgram_english_mp3"
    with patch.object(
        SarvamTTSService,
        "synthesize_speech",
        new_callable=AsyncMock,
        side_effect=SarvamTTSAPIError(402, "No credits available."),
    ), patch(
        "services.tts.DeepgramTTSService.synthesize_speech",
        new_callable=AsyncMock,
        return_value=mock_mp3_bytes,
    ) as mock_dg:
        response = client.post(
            "/api/tts",
            json={
                "text": "What is the minimum CIBIL score for an HDFC personal loan?",
                "language": "en",
                "module": "module2",
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
        assert response.headers["x-tts-provider"] == "deepgram"
        assert response.content == mock_mp3_bytes
        mock_dg.assert_called_once()


def test_api_tts_module1_english_routes_to_sarvam_simran(client):
    """Verify that Module 1 English TTS routes to Sarvam Bulbul v3 en-IN with simran."""
    mock_wav_bytes = b"RIFFfake_english_wav_bytes"
    with patch.object(
        SarvamTTSService,
        "synthesize_speech",
        new_callable=AsyncMock,
        return_value=mock_wav_bytes,
    ) as mock_sarvam:
        # Module 1 request without module="module2"
        response = client.post(
            "/api/tts",
            json={
                "text": "Lead saved successfully for Rajesh Kumar.",
                "language": "en",
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.headers["x-tts-provider"] == "sarvam"
        assert response.headers["x-tts-language"] == "en-IN"
        assert response.headers["x-tts-voice"] == "simran"
        assert response.content == mock_wav_bytes
        mock_sarvam.assert_called_once()
        args, kwargs = mock_sarvam.call_args
        assert kwargs["language_code"] == "en-IN"
        assert kwargs["speaker"] == "simran"

