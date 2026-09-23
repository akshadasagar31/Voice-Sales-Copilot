import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import httpx

from main import app
from services.tts import (
    DeepgramTTSService,
    DeepgramTTSError,
    DeepgramTTSConfigurationError,
    DeepgramTTSAPIError,
    generate_concise_response,
)
from services.lead_extractor import Lead


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Unit Tests: DeepgramTTSService
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tts_service_successful_synthesis():
    """Verify that valid text sends expected payload and returns audio bytes."""
    service = DeepgramTTSService(api_key="test_deepgram_key")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"FAKE_MP3_AUDIO_BYTES_TEST"

    with patch("httpx.AsyncClient.post", return_value=mock_response) as mock_post:
        audio = await service.synthesize_speech("Lead saved successfully.", model="aura-asteria-en")

        assert audio == b"FAKE_MP3_AUDIO_BYTES_TEST"
        assert mock_post.called
        call_args, call_kwargs = mock_post.call_args
        assert "aura-asteria-en" in call_args[0]
        assert call_kwargs["json"] == {"text": "Lead saved successfully."}
        assert call_kwargs["headers"]["Authorization"] == "Token test_deepgram_key"


@pytest.mark.asyncio
async def test_tts_service_missing_api_key_raises_configuration_error():
    """Verify that missing API key raises DeepgramTTSConfigurationError."""
    with patch.dict("os.environ", {}, clear=True):
        service = DeepgramTTSService(api_key="")
        with pytest.raises(DeepgramTTSConfigurationError) as exc_info:
            await service.synthesize_speech("Hello world")
        assert "DEEPGRAM_API_KEY is not configured" in str(exc_info.value)


@pytest.mark.asyncio
async def test_tts_service_empty_text_raises_value_error():
    """Verify that empty or whitespace text raises ValueError."""
    service = DeepgramTTSService(api_key="test_key")

    with pytest.raises(ValueError) as exc_info1:
        await service.synthesize_speech("")
    assert "cannot be empty" in str(exc_info1.value)

    with pytest.raises(ValueError) as exc_info2:
        await service.synthesize_speech("   \n\t  ")
    assert "cannot be empty" in str(exc_info2.value)


@pytest.mark.asyncio
async def test_tts_service_deepgram_api_error():
    """Verify that Deepgram non-200 responses raise DeepgramTTSAPIError."""
    service = DeepgramTTSService(api_key="test_key")

    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = '{"err_msg": "Invalid model specified"}'
    mock_response.json.return_value = {"err_msg": "Invalid model specified"}

    with patch("httpx.AsyncClient.post", return_value=mock_response):
        with pytest.raises(DeepgramTTSAPIError) as exc_info:
            await service.synthesize_speech("Test speech", model="invalid-model")
        assert exc_info.value.status_code == 400
        assert "Invalid model specified" in str(exc_info.value)


@pytest.mark.asyncio
async def test_tts_service_network_error():
    """Verify that network connection failures raise DeepgramTTSAPIError with status 503."""
    service = DeepgramTTSService(api_key="test_key")

    with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("Connection refused")):
        with pytest.raises(DeepgramTTSAPIError) as exc_info:
            await service.synthesize_speech("Test speech")
        assert exc_info.value.status_code == 503
        assert "Unable to reach Deepgram TTS service" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Unit Tests: generate_concise_response Helper
# ---------------------------------------------------------------------------

def test_generate_concise_response_full_lead():
    """Verify spoken text construction for a fully populated lead."""
    lead = Lead(
        name="Vikram Seth",
        company="Reliance Retail",
        loan_type="Commercial Loan",
        loan_amount=2500000.0,
        tenure_months=60,
    )
    speech = generate_concise_response(lead)
    assert "Vikram Seth" in speech
    assert "Reliance Retail" in speech
    assert "Commercial Loan" in speech
    assert "$2.5 million" in speech
    assert "60 months" in speech


def test_generate_concise_response_partial_lead():
    """Verify spoken text construction for a partially populated lead."""
    lead = Lead(
        name="Ananya Rao",
        loan_type="Personal Loan",
        loan_amount=50000.0,
    )
    speech = generate_concise_response(lead)
    assert "Ananya Rao" in speech
    assert "$50,000" in speech
    assert "Personal Loan" in speech


def test_generate_concise_response_dict_input():
    """Verify spoken text construction using a dictionary."""
    lead_dict = {
        "name": "Sarah Miller",
        "company": "Apex Corp",
        "loan_type": "Equipment Financing",
        "loan_amount": 100000,
    }
    speech = generate_concise_response(lead_dict)
    assert "Sarah Miller" in speech
    assert "Apex Corp" in speech
    assert "$100,000" in speech


def test_generate_concise_response_empty():
    """Verify spoken text construction when all fields are empty."""
    speech = generate_concise_response({})
    assert "successfully verified and saved to the CRM database" in speech


def test_generate_concise_response_update():
    """Verify spoken text for updated lead clearly states update action."""
    lead = Lead(name="Sunita Rao", company="TechCorp", loan_amount=500000.0)
    speech = generate_concise_response(lead, is_update=True)
    assert "Updated Sunita Rao's lead details in the CRM." in speech


def test_generate_concise_response_hindi():
    """Verify natural Hindi spoken confirmation for create and update."""
    lead = Lead(name="राजेश कुमार", company="एक्मे कॉर्प")
    # Creation
    speech_create = generate_concise_response(lead, language="hi", is_update=False)
    assert "राजेश कुमार" in speech_create
    assert "सहेज लिया गया है" in speech_create

    # Update
    speech_update = generate_concise_response(lead, language="hi", is_update=True)
    assert "राजेश कुमार" in speech_update
    assert "अपडेट कर दिया गया है" in speech_update


def test_generate_concise_response_marathi():
    """Verify natural Marathi spoken confirmation for create and update."""
    lead = Lead(name="राहुल पाटील", company="टेक महिंद्रा")
    # Creation
    speech_create = generate_concise_response(lead, language="mr", is_update=False)
    assert "राहुल पाटील" in speech_create
    assert "नोंदवले गेले आहेत" in speech_create

    # Update
    speech_update = generate_concise_response(lead, language="mr", is_update=True)
    assert "राहुल पाटील" in speech_update
    assert "अपडेट केले आहेत" in speech_update



# ---------------------------------------------------------------------------
# API Integration Tests: POST /api/tts
# ---------------------------------------------------------------------------

def test_api_tts_endpoint_success(client):
    """Verify POST /api/tts returns 200 with audio/mpeg content when falling back to Deepgram."""
    from services.sarvam_tts import SarvamTTSService, SarvamTTSAPIError
    mock_audio_bytes = b"MOCK_STREAMING_AUDIO_MPEG_BINARY"

    with patch.object(
        SarvamTTSService,
        "synthesize_speech",
        side_effect=SarvamTTSAPIError(500, "Simulated Sarvam fallback"),
    ), patch.object(
        DeepgramTTSService,
        "synthesize_speech",
        return_value=mock_audio_bytes,
    ) as mock_synth:
        response = client.post(
            "/api/tts",
            json={"text": "Lead for Vikram Seth has been saved to the CRM."},
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
        assert response.content == mock_audio_bytes
        assert "inline; filename=tts_response.mp3" in response.headers["content-disposition"]
        mock_synth.assert_called_once()


def test_api_tts_endpoint_empty_text_returns_400(client):
    """Verify POST /api/tts returns 400 when text is empty or whitespace."""
    response = client.post("/api/tts", json={"text": "    "})
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]


def test_api_tts_endpoint_missing_api_key_returns_503(client):
    """Verify POST /api/tts returns 503 when DEEPGRAM_API_KEY is missing."""
    from services.sarvam_tts import SarvamTTSService, SarvamTTSAPIError
    with patch.object(
        SarvamTTSService,
        "synthesize_speech",
        side_effect=SarvamTTSAPIError(500, "Simulated Sarvam fallback"),
    ), patch.object(
        DeepgramTTSService,
        "synthesize_speech",
        side_effect=DeepgramTTSConfigurationError("DEEPGRAM_API_KEY is not configured in backend/.env."),
    ):
        response = client.post("/api/tts", json={"text": "Synthesize this speech"})
        assert response.status_code == 503
        assert "DEEPGRAM_API_KEY is not configured" in response.json()["detail"]


def test_api_tts_endpoint_deepgram_error_returns_502(client):
    """Verify POST /api/tts returns 502 when Deepgram upstream returns an error."""
    from services.sarvam_tts import SarvamTTSService, SarvamTTSAPIError
    with patch.object(
        SarvamTTSService,
        "synthesize_speech",
        side_effect=SarvamTTSAPIError(500, "Simulated Sarvam fallback"),
    ), patch.object(
        DeepgramTTSService,
        "synthesize_speech",
        side_effect=DeepgramTTSAPIError(500, "Internal Server Error from Deepgram"),
    ):
        response = client.post("/api/tts", json={"text": "Synthesize this speech"})
        assert response.status_code == 502
        assert "Deepgram TTS API error" in response.json()["detail"]

