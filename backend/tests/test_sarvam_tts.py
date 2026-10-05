import pytest
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient
import httpx
import base64

from main import app
from services.sarvam_tts import (
    SarvamTTSService,
    SarvamTTSError,
    SarvamTTSConfigurationError,
    SarvamTTSAPIError,
    clean_marathi_financial_text,
)


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Unit Tests: clean_marathi_financial_text
# ---------------------------------------------------------------------------

def test_clean_marathi_financial_text_preserves_english_terms():
    raw = "**एचडीएफसी बँक** personal loan साठी किमान CIBIL score ७५० आवश्यक आहे."
    cleaned = clean_marathi_financial_text(raw)
    assert "एचडीएफसी बँक" in cleaned
    assert "personal loan" in cleaned
    assert "CIBIL score" in cleaned
    assert "**" not in cleaned


def test_clean_marathi_financial_text_expands_symbols():
    raw = "व्याजदर १०.५% p.a. असून किमान कर्ज ₹५०,००० आहे."
    cleaned = clean_marathi_financial_text(raw)
    assert "टक्के" in cleaned
    assert "प्रति वर्ष" in cleaned
    assert "रुपये" in cleaned
    assert "₹" not in cleaned


# ---------------------------------------------------------------------------
# Unit Tests: SarvamTTSService
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sarvam_tts_service_successful_synthesis():
    service = SarvamTTSService(api_key="test_sarvam_key")
    mock_wav_bytes = b"RIFFfake_wav_data"
    b64_audio = base64.b64encode(mock_wav_bytes).decode("utf-8")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"audios": [b64_audio]}

    with patch("httpx.AsyncClient.post", return_value=mock_response) as mock_post:
        audio = await service.synthesize_speech("नमस्कार", speaker="priya")
        assert audio == mock_wav_bytes
        assert mock_post.called
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["speaker"] == "priya"
        assert call_kwargs["json"]["language_code"] == "mr-IN"
        assert call_kwargs["headers"]["api-subscription-key"] == "test_sarvam_key"


@pytest.mark.asyncio
async def test_sarvam_tts_missing_api_key_raises():
    service = SarvamTTSService(api_key="")
    with pytest.raises(SarvamTTSConfigurationError):
        await service.synthesize_speech("नमस्कार")


@pytest.mark.asyncio
async def test_sarvam_tts_empty_text_raises():
    service = SarvamTTSService(api_key="test_key")
    with pytest.raises(ValueError):
        await service.synthesize_speech("")


# ---------------------------------------------------------------------------
# Integration Tests: POST /api/tts with module="module2"
# ---------------------------------------------------------------------------

def test_api_tts_routes_module2_marathi_to_sarvam(client):
    mock_wav_bytes = b"RIFFfake_wav_data"
    with patch.object(
        SarvamTTSService,
        "synthesize_speech",
        return_value=mock_wav_bytes,
    ) as mock_sarvam:
        response = client.post(
            "/api/tts",
            json={
                "text": "एचडीएफसी बँकेच्या वैयक्तिक कर्जासाठी किमान CIBIL score ७५० आवश्यक आहे.",
                "language": "mr",
                "module": "module2",
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.headers["x-tts-provider"] == "sarvam"
        assert response.headers["x-tts-voice"] == "ritu"
        assert response.headers["x-tts-language"] == "mr-IN"
        assert response.content == mock_wav_bytes
        mock_sarvam.assert_called_once()


def test_api_tts_routes_module2_hindi_to_sarvam(client):
    mock_wav_bytes = b"RIFFfake_wav_data"
    with patch.object(
        SarvamTTSService,
        "synthesize_speech",
        return_value=mock_wav_bytes,
    ) as mock_sarvam:
        response = client.post(
            "/api/tts",
            json={
                "text": "एचडीएफसी बैंक के पर्सनल लोन के लिए न्यूनतम सिबिल स्कोर 750 आवश्यक है।",
                "language": "hi",
                "module": "module2",
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.headers["x-tts-provider"] == "sarvam"
        assert response.headers["x-tts-voice"] == "priya"
        assert response.headers["x-tts-language"] == "hi-IN"
        assert response.content == mock_wav_bytes
        mock_sarvam.assert_called_once()


def test_api_tts_module2_marathi_falls_back_to_browser_on_sarvam_error(client):
    """Verify that if Sarvam TTS fails or runs out of credits for Module 2 Marathi, it falls back to native browser speech synthesis."""
    with patch.object(
        SarvamTTSService,
        "synthesize_speech",
        side_effect=SarvamTTSAPIError(402, "No credits available."),
    ):
        response = client.post(
            "/api/tts",
            json={
                "text": "एचडीएफसी बँकेच्या वैयक्तिक कर्जासाठी किमान CIBIL score ७५० आवश्यक आहे.",
                "language": "mr",
                "module": "module2",
            },
        )
        assert response.status_code == 200
        assert response.headers["x-tts-fallback"] == "browser-speech-synthesis"
        data = response.json()
        assert data["fallback_to_browser"] is True
        assert data["language"] == "mr-IN"
        assert "एचडीएफसी" in data["text"]


# ---------------------------------------------------------------------------
# Integration Tests: POST /api/tts for Module 1 (Voice-to-CRM)
# ---------------------------------------------------------------------------

def test_api_tts_routes_module1_marathi_to_sarvam_ritu(client):
    """Verify that Module 1 Marathi requests route to Sarvam Bulbul v3 mr-IN with ritu."""
    mock_wav_bytes = b"RIFFfake_module1_marathi_wav"
    with patch.object(
        SarvamTTSService,
        "synthesize_speech",
        return_value=mock_wav_bytes,
    ) as mock_sarvam:
        response = client.post(
            "/api/tts",
            json={
                "text": "धन्यवाद राजेश शर्मा जी! आपले सर्व आवश्यक तपशील नोंदवले गेले आहेत.",
                "language": "mr",
                "module": "module1",
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.headers["x-tts-provider"] == "sarvam"
        assert response.headers["x-tts-voice"] == "ritu"
        assert response.headers["x-tts-language"] == "mr-IN"
        assert response.content == mock_wav_bytes
        mock_sarvam.assert_called_once()
        args, kwargs = mock_sarvam.call_args
        assert kwargs["language_code"] == "mr-IN"
        assert kwargs["speaker"] == "ritu"


def test_api_tts_routes_module1_hindi_to_sarvam_priya(client):
    """Verify that Module 1 Hindi requests route to Sarvam Bulbul v3 hi-IN with priya."""
    mock_wav_bytes = b"RIFFfake_module1_hindi_wav"
    with patch.object(
        SarvamTTSService,
        "synthesize_speech",
        return_value=mock_wav_bytes,
    ) as mock_sarvam:
        response = client.post(
            "/api/tts",
            json={
                "text": "धन्यवाद राजेश शर्मा जी! आपकी सभी आवश्यक जानकारी दर्ज कर ली गई है।",
                "language": "hi",
                "module": "module1",
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.headers["x-tts-provider"] == "sarvam"
        assert response.headers["x-tts-voice"] == "priya"
        assert response.headers["x-tts-language"] == "hi-IN"
        assert response.content == mock_wav_bytes
        mock_sarvam.assert_called_once()
        args, kwargs = mock_sarvam.call_args
        assert kwargs["language_code"] == "hi-IN"
        assert kwargs["speaker"] == "priya"


@pytest.mark.parametrize(
    ("language", "text", "expected_language", "expected_voice"),
    [
        ("hi-IN", "\u092e\u0941\u091d\u0947 \u092a\u0930\u094d\u0938\u0928\u0932 \u0932\u094b\u0928 \u091a\u093e\u0939\u093f\u090f\u0964", "hi-IN", "priya"),
        ("mr-IN", "\u092e\u0932\u093e \u0935\u0948\u092f\u0915\u094d\u0924\u093f\u0915 \u0915\u0930\u094d\u091c \u0939\u0935\u0947 \u0906\u0939\u0947\u0964", "mr-IN", "ritu"),
    ],
)
def test_api_tts_module1_normalizes_indic_language_variants(client, language, text, expected_language, expected_voice):
    """Indic Module 1 requests always use the matching Sarvam voice and locale."""
    mock_wav_bytes = b"RIFFindic_variant_wav"
    with patch.object(SarvamTTSService, "synthesize_speech", return_value=mock_wav_bytes) as mock_sarvam, patch.object(
        __import__("services.tts", fromlist=["DeepgramTTSService"]).DeepgramTTSService,
        "synthesize_speech",
    ) as mock_deepgram:
        response = client.post("/api/tts", json={"text": text, "language": language, "module": "module1"})

    assert response.status_code == 200
    assert response.headers["x-tts-provider"] == "sarvam"
    assert response.headers["x-tts-voice"] == expected_voice
    assert response.headers["x-tts-language"] == expected_language
    assert mock_sarvam.call_args.kwargs["language_code"] == expected_language
    assert mock_sarvam.call_args.kwargs["speaker"] == expected_voice
    mock_deepgram.assert_not_called()


@pytest.mark.parametrize(
    ("language", "text", "expected_language", "expected_voice"),
    [
        ("hi-IN", "\u092e\u0941\u091d\u0947 \u0932\u094b\u0928 \u091a\u093e\u0939\u093f\u090f\u0964", "hi-IN", "priya"),
        ("mr-IN", "\u092e\u0932\u093e \u0915\u0930\u094d\u091c \u0939\u0935\u0947 \u0906\u0939\u0947\u0964", "mr-IN", "ritu"),
    ],
)
def test_api_tts_module1_indic_sarvam_failure_uses_browser_fallback(client, language, text, expected_language, expected_voice):
    """Indic Module 1 content never falls through to English Deepgram TTS."""
    with patch.object(SarvamTTSService, "synthesize_speech", side_effect=SarvamTTSAPIError(402, "quota")), patch.object(
        __import__("services.tts", fromlist=["DeepgramTTSService"]).DeepgramTTSService,
        "synthesize_speech",
    ) as mock_deepgram:
        response = client.post("/api/tts", json={"text": text, "language": language, "module": "module1"})

    assert response.status_code == 200
    assert response.headers["x-tts-fallback"] == "browser-speech-synthesis"
    body = response.json()
    assert body["language"] == expected_language
    assert body["speaker"] == expected_voice
    mock_deepgram.assert_not_called()


def test_api_tts_routes_module1_english_to_sarvam_simran(client):
    """Verify that Module 1 English requests route to Sarvam Bulbul v3 en-IN with simran when Sarvam is active."""
    mock_wav_bytes = b"RIFFfake_module1_english_wav"
    with patch(
        "main.get_model_manager"
    ) as mock_mgr, patch.object(
        SarvamTTSService,
        "synthesize_speech",
        return_value=mock_wav_bytes,
    ) as mock_sarvam:
        mock_mgr.return_value.get_active_model.return_value = {
            "provider": "Sarvam AI",
            "model_id": "bulbul:v3",
            "is_active": True,
        }
        response = client.post(
            "/api/tts",
            json={
                "text": "Thank you, Rajesh Sharma! All your details have been recorded.",
                "language": "en",
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.headers["x-tts-provider"] == "sarvam"
        assert response.headers["x-tts-voice"] == "simran"
        assert response.headers["x-tts-language"] == "en-IN"
        assert response.content == mock_wav_bytes
        mock_sarvam.assert_called_once()
        args, kwargs = mock_sarvam.call_args
        assert kwargs["language_code"] == "en-IN"
        assert kwargs["speaker"] == "simran"
