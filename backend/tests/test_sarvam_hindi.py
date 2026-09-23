import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
import httpx

from main import app
from services.sarvam_tts import (
    SarvamTTSService,
    SarvamTTSError,
    SarvamTTSConfigurationError,
    SarvamTTSAPIError,
    clean_hindi_financial_text,
)
from services.sarvam_stt import (
    SarvamSTTService,
    SarvamSTTError,
    SarvamSTTConfigurationError,
    SarvamSTTAPIError,
)


@pytest.fixture
def client():
    return TestClient(app)


# ============================================================================
# 1. Tests for clean_hindi_financial_text
# ============================================================================

def test_clean_hindi_financial_text_expands_symbols():
    raw = "HDFC Bank में **personal loan** ₹5,00,000 पर ब्याज दर 10.5% p.a. है और मासिक EMI ₹10,500 होगी।"
    cleaned = clean_hindi_financial_text(raw)

    assert "**" not in cleaned
    assert "5,00,000 रुपये" in cleaned
    assert "10.5 प्रतिशत" in cleaned
    assert "प्रति वर्ष" in cleaned
    assert "10,500 रुपये" in cleaned
    assert "HDFC Bank" in cleaned
    assert "personal loan" in cleaned
    assert "EMI" in cleaned


def test_clean_hindi_financial_text_empty_and_whitespace():
    assert clean_hindi_financial_text("") == ""
    assert clean_hindi_financial_text("   ") == ""


# ============================================================================
# 2. Tests for SarvamTTSService Hindi synthesis
# ============================================================================

@pytest.mark.asyncio
async def test_sarvam_tts_hindi_synthesis_simran():
    service = SarvamTTSService(api_key="test_sarvam_key")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "audios": ["UklGRgAAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="]  # valid base64 WAV header
    }

    with patch("services.sarvam_tts.get_shared_sarvam_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_get_client.return_value = mock_client

        audio_bytes = await service.synthesize_speech(
            "नमस्ते! CIBIL स्कोर 750 होना चाहिए।",
            language_code="hi-IN",
            speaker="simran",
            model="bulbul:v3",
        )

        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) > 0
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args.kwargs
        assert call_kwargs["json"]["language_code"] == "hi-IN"
        assert call_kwargs["json"]["speaker"] == "simran"
        assert call_kwargs["json"]["model"] == "bulbul:v3"


# ============================================================================
# 3. Tests for SarvamSTTService saaras:v4
# ============================================================================

@pytest.mark.asyncio
async def test_sarvam_stt_success():
    service = SarvamSTTService(api_key="test_sarvam_key")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "transcript": "नमस्ते, मेरा सिबिल स्कोर 750 है। मुझे एचडीएफसी बैंक से पर्सनल लोन चाहिए।",
        "language_code": "hi-IN",
    }

    with patch("services.sarvam_stt.get_shared_sarvam_stt_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_get_client.return_value = mock_client

        dummy_wav = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        result = await service.transcribe_audio(
            dummy_wav,
            content_type="audio/wav",
            language_code="hi-IN",
            model="saaras:v4",
        )

        assert result["transcript"] == "नमस्ते, मेरा सिबिल स्कोर 750 है। मुझे एचडीएफसी बैंक से पर्सनल लोन चाहिए।"
        assert result["detected_language"] == "hi"
        assert result["stt_provider"] == "sarvam"
        assert result["model"] == "saaras:v4"
        mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_sarvam_stt_missing_key_raises():
    service = SarvamSTTService(api_key="")
    with pytest.raises(SarvamSTTConfigurationError):
        await service.transcribe_audio(b"audio_bytes")


# ============================================================================
# 4. Tests for FastAPI /api/tts Module 2 Hindi Routing
# ============================================================================

def test_api_tts_routes_module2_hindi_to_sarvam(client):
    dummy_wav = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"

    with patch.object(
        SarvamTTSService,
        "synthesize_speech",
        new_callable=AsyncMock,
        return_value=dummy_wav,
    ) as mock_sarvam:
        response = client.post(
            "/api/tts",
            json={
                "text": "एचडीएफसी बैंक में पर्सनल लोन के लिए न्यूनतम सिबिल स्कोर 750 होना चाहिए।",
                "language": "hi",
                "module": "module2",
            },
        )

        assert response.status_code == 200
        assert response.headers["x-tts-provider"] == "sarvam"
        assert response.headers["x-tts-language"] == "hi-IN"
        assert response.headers["x-tts-voice"] == "simran"
        mock_sarvam.assert_called_once()


def test_api_tts_module2_hindi_falls_back_to_deepgram_on_sarvam_error(client):
    """Verify that if Sarvam TTS fails or runs out of credits for Module 2 Hindi, it automatically falls back to Deepgram."""
    mock_mp3_bytes = b"ID3\x03\x00\x00\x00\x00\x00#TSSEfake_deepgram_hindi_mp3"
    with patch.object(
        SarvamTTSService,
        "synthesize_speech",
        side_effect=SarvamTTSAPIError(402, "No credits available."),
    ), patch(
        "services.tts.DeepgramTTSService.synthesize_speech",
        new_callable=AsyncMock,
        return_value=mock_mp3_bytes,
    ) as mock_deepgram:
        response = client.post(
            "/api/tts",
            json={
                "text": "एचडीएफसी बैंक में पर्सनल लोन के लिए न्यूनतम सिबिल स्कोर 750 होना चाहिए।",
                "language": "hi",
                "module": "module2",
            },
        )

        assert response.status_code == 200
        assert response.headers["x-tts-provider"] == "deepgram"
        assert response.content == mock_mp3_bytes
        mock_deepgram.assert_called_once()


# ============================================================================
# 5. Tests for FastAPI /api/voice-entry Module 2 Hindi Routing
# ============================================================================

def test_api_voice_entry_routes_module2_hindi_to_deepgram_stt(client):
    dummy_wav = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"

    mock_result = {
        "success": True,
        "transcript": "नमस्ते, मेरा सिबिल स्कोर 750 है।",
        "confidence": 0.98,
        "detected_language": "hi",
        "duration": 2.5,
        "model": "nova-3",
    }

    with patch(
        "services.stt.DeepgramSTTService.transcribe_audio",
        new_callable=AsyncMock,
        return_value=mock_result,
    ) as mock_stt:
        response = client.post(
            "/api/voice-entry",
            files={"file": ("test.wav", dummy_wav, "audio/wav")},
            data={"language": "hi", "module": "module2"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["transcript"] == "नमस्ते, मेरा सिबिल स्कोर 750 है।"
        assert data["detected_language"] == "hi"
        mock_stt.assert_called_once()
        call_kwargs = mock_stt.call_args[1]
        assert call_kwargs["language"] == "hi"
        assert call_kwargs["model"] == "nova-3"

