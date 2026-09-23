import os
import sys
from unittest.mock import patch, AsyncMock, MagicMock
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app
from services.stt import DeepgramSTTService, DeepgramConfigurationError, DeepgramAPIError


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_audio_bytes():
    # Simulated webm/opus audio bytes
    return b"\x1a\x45\xdf\xa3" + b"sample-voice-note-audio-bytes-content-12345"


def test_voice_entry_success(client, sample_audio_bytes):
    """Verifies successful audio transcription via POST /api/voice-entry with Deepgram mocked."""
    mock_result = {
        "success": True,
        "transcript": "Had a great discovery call with Dr. Rachel Green. She loved our HIPAA compliance architecture.",
        "confidence": 0.9824,
        "words_count": 16,
        "duration": 4.85,
        "model": "nova-2",
    }

    with patch.object(DeepgramSTTService, "transcribe_audio", new_callable=AsyncMock) as mock_transcribe, \
         patch.dict(os.environ, {"DEEPGRAM_API_KEY": "mock-deepgram-key"}):

        mock_transcribe.return_value = mock_result

        response = client.post(
            "/api/voice-entry",
            files={"file": ("voice_note.webm", sample_audio_bytes, "audio/webm")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "Dr. Rachel Green" in data["transcript"]
    assert data["confidence"] == 0.9824
    assert data["model"] == "nova-2"
    assert data["filename"] == "voice_note.webm"
    assert mock_transcribe.called


def test_voice_entry_with_audio_file_field(client, sample_audio_bytes):
    """Verifies that the endpoint accepts 'audio_file' as an alternative form field name."""
    mock_result = {
        "success": True,
        "transcript": "Checking pricing and volume discounts for enterprise tier.",
        "confidence": 0.95,
        "words_count": 9,
        "duration": 3.2,
        "model": "nova-2",
    }

    with patch.object(DeepgramSTTService, "transcribe_audio", new_callable=AsyncMock) as mock_transcribe, \
         patch.dict(os.environ, {"DEEPGRAM_API_KEY": "mock-deepgram-key"}):

        mock_transcribe.return_value = mock_result

        response = client.post(
            "/api/voice-entry",
            files={"audio_file": ("recording.wav", sample_audio_bytes, "audio/wav")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "pricing" in data["transcript"]
    assert data["filename"] == "recording.wav"


def test_voice_entry_missing_file(client):
    """Verifies rejection with HTTP 400 when no audio file is provided."""
    response = client.post("/api/voice-entry", data={})
    assert response.status_code == 400
    assert "No audio file" in response.json()["detail"]


def test_voice_entry_empty_audio_file(client):
    """Verifies rejection with HTTP 400 when an empty (0-byte) file is uploaded."""
    response = client.post(
        "/api/voice-entry",
        files={"file": ("empty.webm", b"", "audio/webm")},
    )
    assert response.status_code == 400
    assert "empty (0 bytes)" in response.json()["detail"]


def test_voice_entry_missing_api_key_returns_503(client, sample_audio_bytes):
    """Verifies that missing DEEPGRAM_API_KEY returns HTTP 503 Service Unavailable."""
    with patch.dict(os.environ, {"DEEPGRAM_API_KEY": ""}, clear=False):
        response = client.post(
            "/api/voice-entry",
            files={"file": ("voice_note.webm", sample_audio_bytes, "audio/webm")},
        )

    assert response.status_code == 503
    assert "DEEPGRAM_API_KEY is not configured" in response.json()["detail"]


def test_voice_entry_deepgram_api_error_returns_502(client, sample_audio_bytes):
    """Verifies that an error returned from Deepgram API maps to HTTP 502 Bad Gateway."""
    with patch.object(
        DeepgramSTTService,
        "transcribe_audio",
        side_effect=DeepgramAPIError(401, "Invalid credentials")
    ), patch.dict(os.environ, {"DEEPGRAM_API_KEY": "invalid-key"}):

        response = client.post(
            "/api/voice-entry",
            files={"file": ("voice_note.webm", sample_audio_bytes, "audio/webm")},
        )

    assert response.status_code == 502
    assert "Deepgram API error" in response.json()["detail"]


@pytest.mark.asyncio
async def test_deepgram_service_transcribe_parses_response_correctly(sample_audio_bytes):
    """Direct unit test of DeepgramSTTService parsing Deepgram JSON output."""
    mock_deepgram_response = {
        "metadata": {"duration": 5.42},
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": "Follow up scheduled for Thursday at 2 PM.",
                            "confidence": 0.9912,
                            "words": [{"word": "Follow"}, {"word": "up"}],
                        }
                    ]
                }
            ]
        },
    }

    mock_http_response = MagicMock()
    mock_http_response.status_code = 200
    mock_http_response.json.return_value = mock_deepgram_response

    mock_client_instance = AsyncMock()
    mock_client_instance.post.return_value = mock_http_response

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client_cls.return_value.__aenter__.return_value = mock_client_instance

        service = DeepgramSTTService(api_key="test-key-123")
        res = await service.transcribe_audio(sample_audio_bytes, content_type="audio/webm")

    assert res["success"] is True
    assert res["transcript"] == "Follow up scheduled for Thursday at 2 PM."
    assert res["confidence"] == 0.9912
    assert res["words_count"] == 2
    assert res["duration"] == 5.42
    assert res["model"] == "nova-2"


@pytest.mark.asyncio
async def test_deepgram_service_retries_transient_503_and_succeeds(sample_audio_bytes):
    """Verify DeepgramSTTService retries on HTTP 503 and succeeds on subsequent attempt."""
    mock_503_response = MagicMock()
    mock_503_response.status_code = 503
    mock_503_response.headers = {"dg-request-id": "req-503-abc", "content-type": "application/json"}
    mock_503_response.json.return_value = {"err_msg": "Service Temporarily Unavailable"}
    mock_503_response.text = '{"err_msg": "Service Temporarily Unavailable"}'

    mock_200_response = MagicMock()
    mock_200_response.status_code = 200
    mock_200_response.headers = {"dg-request-id": "req-200-def", "content-type": "application/json"}
    mock_200_response.content = b'{"results":{}}'
    mock_200_response.json.return_value = {
        "metadata": {"duration": 2.0},
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {"transcript": "Hello, recovery succeeded.", "confidence": 0.97, "words": []}
                    ]
                }
            ]
        },
    }

    mock_client_instance = AsyncMock()
    mock_client_instance.post.side_effect = [mock_503_response, mock_200_response]

    with patch("httpx.AsyncClient") as mock_client_cls, \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:

        mock_client_cls.return_value.__aenter__.return_value = mock_client_instance

        service = DeepgramSTTService(api_key="secret-key-xyz")
        res = await service.transcribe_audio(sample_audio_bytes)

    assert res["success"] is True
    assert res["transcript"] == "Hello, recovery succeeded."
    assert mock_client_instance.post.call_count == 2
    assert mock_sleep.called
    assert mock_sleep.call_args[0][0] == 0.5


@pytest.mark.asyncio
async def test_deepgram_service_exhausts_retries_on_persistent_503(sample_audio_bytes):
    """Verify DeepgramSTTService retries max 2 times (3 attempts total) on persistent 503 and raises DeepgramAPIError."""
    mock_503_response = MagicMock()
    mock_503_response.status_code = 503
    mock_503_response.headers = {"dg-request-id": "req-503-fail", "content-type": "application/json"}
    mock_503_response.json.return_value = {"err_msg": "Service Unavailable - Server Overload"}
    mock_503_response.text = '{"err_msg": "Service Unavailable - Server Overload"}'

    mock_client_instance = AsyncMock()
    mock_client_instance.post.return_value = mock_503_response

    with patch("httpx.AsyncClient") as mock_client_cls, \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:

        mock_client_cls.return_value.__aenter__.return_value = mock_client_instance

        service = DeepgramSTTService(api_key="secret-key-xyz")
        with pytest.raises(DeepgramAPIError) as exc_info:
            await service.transcribe_audio(sample_audio_bytes)

    assert exc_info.value.status_code == 503
    assert "Service Unavailable" in exc_info.value.message
    # 1 initial + max 2 retries = 3 total attempts
    assert mock_client_instance.post.call_count == 3
    assert mock_sleep.call_count == 2


def test_extract_safe_response_detail_never_leaks_api_key():
    """Verify _extract_safe_response_detail redacts the API key if present in the error payload."""
    from services.stt import _extract_safe_response_detail

    mock_resp = MagicMock()
    secret_key = "660af0f04c838e51539fc1614dc911d56ced14ae"
    mock_resp.json.return_value = {"err_msg": f"Auth token {secret_key} rejected by server"}
    mock_resp.text = f"Auth token {secret_key} rejected by server"

    safe_text = _extract_safe_response_detail(mock_resp, api_key=secret_key)
    assert secret_key not in safe_text
    assert "[REDACTED_API_KEY]" in safe_text


def test_financial_keyterms_contain_required_domain_terms():
    """Verify that domain-specific financial keyterms are registered for Nova-3 keyterm boosting."""
    from services.stt import FINANCIAL_KEYTERMS
    assert "CIBIL" in FINANCIAL_KEYTERMS
    assert "EMI" in FINANCIAL_KEYTERMS
    assert "personal loan" in FINANCIAL_KEYTERMS
    assert "lakh" in FINANCIAL_KEYTERMS
    assert "HDFC" in FINANCIAL_KEYTERMS
    assert "सिबिल" in FINANCIAL_KEYTERMS
    assert "ईएमआई" in FINANCIAL_KEYTERMS


def test_build_deepgram_ws_url_parameters():
    """Verify build_deepgram_ws_url constructs valid Nova-3 streaming parameters with keyterm and linear16."""
    from services.stt import build_deepgram_ws_url
    url = build_deepgram_ws_url(model="nova-3", sample_rate=48000, language="en")
    assert url.startswith("wss://api.deepgram.com/v1/listen?")
    assert "model=nova-3" in url
    assert "encoding=linear16" in url
    assert "sample_rate=48000" in url
    assert "interim_results=true" in url
    assert "keyterm=CIBIL" in url
    assert "keyterm=EMI" in url


def test_ws_voice_stt_requires_api_key(client):
    """Verify /ws/voice-stt rejects connection with error when DEEPGRAM_API_KEY is unset."""
    with patch.dict(os.environ, {"DEEPGRAM_API_KEY": ""}):
        with client.websocket_connect("/ws/voice-stt") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "DEEPGRAM_API_KEY" in msg["message"]


def test_financial_keyterms_contain_marathi_specific_terms():
    """Verify that domain-specific Marathi financial keyterms are registered."""
    from services.stt import FINANCIAL_KEYTERMS
    assert "CIBIL स्कोर" in FINANCIAL_KEYTERMS
    assert "मासिक EMI" in FINANCIAL_KEYTERMS
    assert "दरमहा EMI" in FINANCIAL_KEYTERMS
    assert "HDFC बँक" in FINANCIAL_KEYTERMS
    assert "SBI बँक" in FINANCIAL_KEYTERMS
    assert "लाख" in FINANCIAL_KEYTERMS
    assert "गृहकर्ज" in FINANCIAL_KEYTERMS
    assert "कागदपत्रे" in FINANCIAL_KEYTERMS
    assert "दरमहा" in FINANCIAL_KEYTERMS


def test_build_deepgram_ws_url_marathi_and_multi():
    """Verify build_deepgram_ws_url correctly passes language=mr and language=multi for auto."""
    from services.stt import build_deepgram_ws_url
    url_mr = build_deepgram_ws_url(model="nova-3", language="mr")
    assert "language=mr" in url_mr

    url_auto = build_deepgram_ws_url(model="nova-3", language=None)
    assert "language=multi" in url_auto


def test_normalize_stt_transcript_marathi_and_multilingual():
    """Verify normalize_stt_transcript normalizes whitespace and punctuation without rewriting, correcting, or replacing words."""
    from services.stt import normalize_stt_transcript

    assert normalize_stt_transcript("  मला   HDFC Bank कडून कर्ज हवे आहे  ") == "मला HDFC Bank कडून कर्ज हवे आहे"
    assert normalize_stt_transcript("What is the   minimum CIBIL score for personal loans ?") == "What is the minimum CIBIL score for personal loans?"
    assert normalize_stt_transcript("पर्सनल लोन के लिए   न्यूनतम सिबिल स्कोर ,  कितना होना चाहिए ?") == "पर्सनल लोन के लिए न्यूनतम सिबिल स्कोर, कितना होना चाहिए?"
    assert normalize_stt_transcript("This has slang like gonna and wanna and typo xyz123 .") == "This has slang like gonna and wanna and typo xyz123."
    assert normalize_stt_transcript("") == ""
    assert normalize_stt_transcript(None) == ""

