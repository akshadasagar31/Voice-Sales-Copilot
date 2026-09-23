import os
import sys
from unittest.mock import patch, AsyncMock
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app
from services.stt import DeepgramSTTService


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_wav_bytes():
    return b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"


def test_module2_stt_marathi_routing(client, sample_wav_bytes):
    """Verifies that Module 2 with language=mr routes directly to Deepgram Nova-3 with mr."""
    mock_result = {
        "success": True,
        "transcript": "एचडीएफसी बँकेकडून 10 लाख रुपयांच्या पर्सनल लोनसाठी सिबिल स्कोर किती लागतो आणि दरमहा ईएमआय किती येईल?",
        "confidence": 0.98,
        "detected_language": "mr",
        "duration": 4.5,
        "model": "nova-3",
    }

    with patch.object(
        DeepgramSTTService, "transcribe_audio", new_callable=AsyncMock, return_value=mock_result
    ) as mock_dg:
        response = client.post(
            "/api/voice-entry",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
            data={"module": "module2", "language": "mr"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "सिबिल स्कोर" in data["transcript"]
        assert "ईएमआय" in data["transcript"]
        assert "10 लाख" in data["transcript"]
        mock_dg.assert_called_once()
        call_kwargs = mock_dg.call_args[1]
        assert call_kwargs["language"] == "mr"
        assert call_kwargs["model"] == "nova-3"


def test_module2_stt_english_routing(client, sample_wav_bytes):
    """Verifies that Module 2 with language=en routes directly to Deepgram Nova-3 with en-IN."""
    mock_result = {
        "success": True,
        "transcript": "What is the minimum CIBIL score required for a personal loan of 15 lakh with HDFC Bank and what is the monthly EMI?",
        "confidence": 0.98,
        "detected_language": "en",
        "duration": 4.0,
        "model": "nova-3",
    }

    with patch.object(
        DeepgramSTTService, "transcribe_audio", new_callable=AsyncMock, return_value=mock_result
    ) as mock_dg:
        response = client.post(
            "/api/voice-entry",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
            data={"module": "module2", "language": "en"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "CIBIL" in data["transcript"]
        assert "EMI" in data["transcript"]
        assert "HDFC Bank" in data["transcript"]
        mock_dg.assert_called_once()
        call_kwargs = mock_dg.call_args[1]
        assert call_kwargs["language"] == "en-IN"
        assert call_kwargs["model"] == "nova-3"


def test_module2_stt_hindi_routing(client, sample_wav_bytes):
    """Verifies that Module 2 with language=hi routes directly to Deepgram Nova-3 with hi."""
    mock_result = {
        "success": True,
        "transcript": "एचडीएफसी बैंक से 15 लाख के पर्सनल लोन के लिए सिबिल स्कोर कितना होना चाहिए?",
        "confidence": 0.98,
        "detected_language": "hi",
        "duration": 3.8,
        "model": "nova-3",
    }

    with patch.object(
        DeepgramSTTService, "transcribe_audio", new_callable=AsyncMock, return_value=mock_result
    ) as mock_dg:
        response = client.post(
            "/api/voice-entry",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
            data={"module": "module2", "language": "hi"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "एचडीएफसी बैंक" in data["transcript"]
        mock_dg.assert_called_once()
        call_kwargs = mock_dg.call_args[1]
        assert call_kwargs["language"] == "hi"
        assert call_kwargs["model"] == "nova-3"


def test_module2_stt_auto_routing(client, sample_wav_bytes):
    """Verifies that Module 2 with auto language routes directly to Deepgram Nova-3 with language='multi'."""
    mock_result = {
        "success": True,
        "transcript": "HDFC Bank me 10 lakh personal loan ke liye CIBIL score kitna hoga?",
        "confidence": 0.98,
        "detected_language": "hi",
        "duration": 3.5,
        "model": "nova-3",
    }

    with patch.object(
        DeepgramSTTService, "transcribe_audio", new_callable=AsyncMock, return_value=mock_result
    ) as mock_dg:
        response = client.post(
            "/api/voice-entry",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
            data={"module": "module2", "language": "auto"},
        )

        assert response.status_code == 200
        data = response.json()
        mock_dg.assert_called_once()
        call_kwargs = mock_dg.call_args[1]
        assert call_kwargs["language"] == "multi"
        assert call_kwargs["model"] == "nova-3"
