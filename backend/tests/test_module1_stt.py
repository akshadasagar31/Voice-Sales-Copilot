import os
import sys
import json
from unittest.mock import patch, AsyncMock
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app
from services.sarvam_stt import SarvamSTTService, SarvamSTTAPIError
from services.stt import DeepgramSTTService


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_wav_bytes():
    return b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"


def test_module1_stt_english_routing(client, sample_wav_bytes):
    """Verifies that Module 1 with language=en routes to Deepgram Nova-3 with en-IN."""
    mock_result = {
        "success": True,
        "transcript": "My name is Rajesh Sharma. Phone number is 9876543210. Need 5 lakh loan.",
        "confidence": 0.98,
        "detected_language": "en",
        "duration": 4.5,
        "words_count": 13,
        "stt_provider": "deepgram",
        "model": "nova-3",
    }

    with patch.object(
        DeepgramSTTService, "transcribe_audio", new_callable=AsyncMock, return_value=mock_result
    ) as mock_dg:
        response = client.post(
            "/api/voice-entry",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
            data={"module": "module1", "language": "en"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["transcript"] == "My name is Rajesh Sharma. Phone number is 9876543210. Need 5 lakh loan."
        assert data["stt_provider"] == "deepgram"
        assert data["model"] == "nova-3"
        assert "9876543210" in data["transcript"]
        assert "Rajesh Sharma" in data["transcript"]

        # Verify called with language='en-IN' and model='nova-3'
        mock_dg.assert_called_once()
        call_kwargs = mock_dg.call_args[1]
        assert call_kwargs["language"] == "en-IN"
        assert call_kwargs["model"] == "nova-3"


def test_module1_stt_hindi_routing(client, sample_wav_bytes):
    """Verifies that Module 1 with language=hi routes to Deepgram Nova-3 with hi."""
    mock_result = {
        "success": True,
        "transcript": "मेरा नाम राजेश शर्मा है। फोन नंबर 9876543210 है। 5 लाख का लोन चाहिए।",
        "confidence": 0.98,
        "detected_language": "hi",
        "duration": 4.0,
        "words_count": 14,
        "stt_provider": "deepgram",
        "model": "nova-3",
    }

    with patch.object(
        DeepgramSTTService, "transcribe_audio", new_callable=AsyncMock, return_value=mock_result
    ) as mock_dg:
        response = client.post(
            "/api/voice-entry",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
            data={"module": "module1", "language": "hi"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "9876543210" in data["transcript"]
        assert data["stt_provider"] == "deepgram"
        assert data["model"] == "nova-3"
        mock_dg.assert_called_once()
        call_kwargs = mock_dg.call_args[1]
        assert call_kwargs["language"] == "hi"
        assert call_kwargs["model"] == "nova-3"


def test_module1_stt_marathi_routing(client, sample_wav_bytes):
    """Verifies that Module 1 with language=mr routes to Deepgram Nova-3 with mr."""
    mock_result = {
        "success": True,
        "transcript": "माझे नाव राजेश शर्मा आहे. फोन नंबर 9876543210 आहे.",
        "confidence": 0.98,
        "detected_language": "mr",
        "duration": 3.8,
        "words_count": 9,
        "stt_provider": "deepgram",
        "model": "nova-3",
    }

    with patch.object(
        DeepgramSTTService, "transcribe_audio", new_callable=AsyncMock, return_value=mock_result
    ) as mock_dg:
        response = client.post(
            "/api/voice-entry",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
            data={"module": "module1", "language": "mr"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "9876543210" in data["transcript"]
        assert data["stt_provider"] == "deepgram"
        assert data["model"] == "nova-3"
        mock_dg.assert_called_once()
        call_kwargs = mock_dg.call_args[1]
        assert call_kwargs["language"] == "mr"
        assert call_kwargs["model"] == "nova-3"


def test_module1_stt_auto_routing(client, sample_wav_bytes):
    """Verifies that Module 1 with language=auto (or omitted) routes to Deepgram Nova-3 with multi."""
    mock_result = {
        "success": True,
        "transcript": "Hello mera naam Rajesh Sharma hai phone 9876543210.",
        "confidence": 0.98,
        "detected_language": "hi",
        "duration": 3.5,
        "words_count": 8,
        "stt_provider": "deepgram",
        "model": "nova-3",
    }

    with patch.object(
        DeepgramSTTService, "transcribe_audio", new_callable=AsyncMock, return_value=mock_result
    ) as mock_dg:
        response = client.post(
            "/api/voice-entry",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
            data={"module": "module1", "language": "auto"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["transcript"] == "Hello mera naam Rajesh Sharma hai phone 9876543210."
        assert data["stt_provider"] == "deepgram"
        assert data["model"] == "nova-3"
        mock_dg.assert_called_once()
        call_kwargs = mock_dg.call_args[1]
        assert call_kwargs["language"] == "multi"
        assert call_kwargs["model"] == "nova-3"


def test_module1_stt_fallback_to_deepgram(client, sample_wav_bytes):
    """Verifies that if Sarvam STT encounters an API error, it gracefully falls back to Deepgram Nova-3."""
    mock_dg_result = {
        "success": True,
        "transcript": "My name is Rajesh Sharma phone number 9876543210.",
        "confidence": 0.95,
        "words_count": 9,
        "duration": 3.2,
        "model": "nova-3",
    }

    with patch.object(
        SarvamSTTService, "transcribe_audio", new_callable=AsyncMock, side_effect=SarvamSTTAPIError(503, "Service down")
    ), patch.object(
        DeepgramSTTService, "transcribe_audio", new_callable=AsyncMock, return_value=mock_dg_result
    ) as mock_dg:
        response = client.post(
            "/api/voice-entry",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
            data={"module": "module1", "language": "en"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["transcript"] == "My name is Rajesh Sharma phone number 9876543210."
        mock_dg.assert_called_once()


def test_module1_language_detection_english_never_defaults_to_hindi(client, sample_wav_bytes):
    """
    Verifies that English speech (with Indian entity names and lakh) is detected as English ('en')
    and NEVER defaults to Hindi ('hi'), even if the acoustic classifier raw returned 'hi' or 'multi'.
    """
    mock_dg_result = {
        "success": True,
        "transcript": "My name is Rajesh, phone 9876543210, personal loan 5 lakh.",
        "confidence": 0.96,
        "detected_language": "hi",  # Deepgram acoustic classifier misclassified Indian accent as hi
        "words_count": 10,
        "duration": 3.5,
        "model": "nova-3",
    }

    with patch.object(
        SarvamSTTService, "transcribe_audio", new_callable=AsyncMock, side_effect=SarvamSTTAPIError(402, "No credits")
    ), patch.object(
        DeepgramSTTService, "transcribe_audio", new_callable=AsyncMock, return_value=mock_dg_result
    ):
        response = client.post(
            "/api/voice-entry",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
            data={"module": "module1", "language": "auto"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["detected_language"] == "en"
        assert "Rajesh" in data["transcript"]


def test_module1_language_detection_hindi_response(client, sample_wav_bytes):
    """Verifies that Hindi speech is accurately detected as 'hi'."""
    mock_dg_result = {
        "success": True,
        "transcript": "मेरा नाम राजेश है और मुझे 5 लाख का पर्सनल लोन चाहिए।",
        "confidence": 0.97,
        "detected_language": "multi",
        "words_count": 12,
        "duration": 4.0,
        "model": "nova-3",
    }

    with patch.object(
        SarvamSTTService, "transcribe_audio", new_callable=AsyncMock, side_effect=SarvamSTTAPIError(402, "No credits")
    ), patch.object(
        DeepgramSTTService, "transcribe_audio", new_callable=AsyncMock, return_value=mock_dg_result
    ):
        response = client.post(
            "/api/voice-entry",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
            data={"module": "module1", "language": "auto"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["detected_language"] == "hi"


def test_module1_language_detection_marathi_response(client, sample_wav_bytes):
    """Verifies that Marathi speech is accurately detected as 'mr'."""
    mock_dg_result = {
        "success": True,
        "transcript": "माझे नाव राहुल आहे आणि मला 5 लाख रुपयांचे वैयक्तिक कर्ज हवे आहे.",
        "confidence": 0.98,
        "detected_language": "multi",
        "words_count": 13,
        "duration": 4.2,
        "model": "nova-3",
    }

    with patch.object(
        SarvamSTTService, "transcribe_audio", new_callable=AsyncMock, side_effect=SarvamSTTAPIError(402, "No credits")
    ), patch.object(
        DeepgramSTTService, "transcribe_audio", new_callable=AsyncMock, return_value=mock_dg_result
    ):
        response = client.post(
            "/api/voice-entry",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
            data={"module": "module1", "language": "auto"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["detected_language"] == "mr"


def test_module1_language_detection_mixed_speech(client, sample_wav_bytes):
    """
    Verifies that mixed speech (code-switching) detects 'mixed' as the spoken language,
    and returns an English response prompt according to the specification.
    """
    # 1. English + Hindi code-mixing -> spoken: mixed, response: English
    mock_hi_mix = {
        "success": True,
        "transcript": "My name is Rajesh and mujhe personal loan chahiye",
        "confidence": 0.95,
        "detected_language": "multi",
        "words_count": 9,
        "duration": 3.0,
        "model": "nova-3",
    }

    with patch.object(
        SarvamSTTService, "transcribe_audio", new_callable=AsyncMock, side_effect=SarvamSTTAPIError(402, "No credits")
    ), patch.object(
        DeepgramSTTService, "transcribe_audio", new_callable=AsyncMock, return_value=mock_hi_mix
    ):
        response = client.post(
            "/api/voice-entry",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
            data={"module": "module1", "language": "auto"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["detected_language"] == "mixed"
        assert data["language"] == "en"
        # English response prompt
        assert "What" in data["immediate_sentence1"] or "Thank you" in data["immediate_sentence1"] or "phone" in data["immediate_sentence1"].lower()

    # 2. English + Marathi code-mixing -> spoken: mixed, response: English
    mock_mr_mix = {
        "success": True,
        "transcript": "My name is Rahul and mala personal loan pahije",
        "confidence": 0.95,
        "detected_language": "multi",
        "words_count": 9,
        "duration": 3.0,
        "model": "nova-3",
    }

    with patch.object(
        SarvamSTTService, "transcribe_audio", new_callable=AsyncMock, side_effect=SarvamSTTAPIError(402, "No credits")
    ), patch.object(
        DeepgramSTTService, "transcribe_audio", new_callable=AsyncMock, return_value=mock_mr_mix
    ):
        response = client.post(
            "/api/voice-entry",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
            data={"module": "module1", "language": "auto"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["detected_language"] == "mixed"
        assert data["language"] == "en"
        assert "What" in data["immediate_sentence1"] or "Thank you" in data["immediate_sentence1"] or "phone" in data["immediate_sentence1"].lower()


def test_module1_new_lead_reset_starts_from_name(client, sample_wav_bytes):
    """
    Verifies that when a new lead starts (e.g. transcript says 'new lead' or 'start fresh'),
    the previous active lead/session is reset, starting fresh from 'name'.
    """
    mock_dg = {
        "success": True,
        "transcript": "Start a new lead please",
        "confidence": 0.98,
        "words_count": 5,
        "duration": 2.0,
        "model": "nova-3",
    }

    # Pass an existing lead with already populated fields
    existing = {"name": "Old Customer", "phone": "9999999999", "company": "Old Corp"}

    with patch.object(
        DeepgramSTTService, "transcribe_audio", new_callable=AsyncMock, return_value=mock_dg
    ):
        response = client.post(
            "/api/voice-entry",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
            data={"module": "module1", "language": "en", "existing_lead": json.dumps(existing), "lead_id": 123},
        )
        assert response.status_code == 200
        data = response.json()
        # Active session must be reset
        assert data.get("is_new_lead") is True
        assert data["next_missing_parameter"] == "name"
        assert "name" in data["immediate_sentence1"].lower() or "May I have your name" in data["immediate_sentence1"]


def test_module1_extract_multiple_fields_single_sentence(client, sample_wav_bytes):
    """
    Verifies extracting multiple fields from one sentence:
    e.g. Name, Company, Phone, Loan Type, Loan Amount, Tenure.
    And verifies it never asks again for collected fields.
    """
    mock_dg = {
        "success": True,
        "transcript": "My name is Rajesh Kumar, working at TCS, phone number 9876543210, need personal loan of 5 lakhs for 2 years.",
        "confidence": 0.99,
        "words_count": 21,
        "duration": 5.0,
        "model": "nova-3",
    }

    with patch.object(
        DeepgramSTTService, "transcribe_audio", new_callable=AsyncMock, return_value=mock_dg
    ):
        response = client.post(
            "/api/voice-entry",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
            data={"module": "module1", "language": "en"},
        )
        assert response.status_code == 200
        data = response.json()
        lead = data["lead"]
        assert lead["name"] == "Rajesh Kumar"
        assert lead["company"] == "TCS"
        assert lead["phone"] == "9876543210"
        assert lead["loan_type"] == "Personal Loan"
        assert lead["loan_amount"] == 500000.0
        assert lead["tenure_months"] == 24
        # All required fields captured -> is_complete is True and next_missing_parameter is None
        assert data["is_complete"] is True
        assert data["next_missing_parameter"] is None
        assert "recorded" in data["immediate_sentence1"].lower() or "thank you" in data["immediate_sentence1"].lower()


def test_module1_multi_turn_partial_fields_asks_only_next_missing_and_never_repeats(client, sample_wav_bytes):
    """
    Validates:
    1. Multiple fields extracted in Turn 1 (name + phone).
    2. Next question asks ONLY for company (skipping name and phone).
    3. Turn 2 receives existing_lead and extracts company + loan_type + loan_amount.
    4. Next question asks ONLY for tenure_months (never repeating any collected field).
    5. Turn 3 provides tenure_months, completing the lead.
    """
    # TURN 1: User provides Name and Phone
    t1_mock = {
        "transcript": "My name is Priya and my phone number is 9876543210",
        "detected_language": "en",
        "language": "en",
        "confidence": 0.99,
        "words_count": 10,
        "duration": 3.0,
        "model": "nova-3",
    }
    with patch.object(DeepgramSTTService, "transcribe_audio", new_callable=AsyncMock, return_value=t1_mock):
        r1 = client.post(
            "/api/voice-entry",
            files={"file": ("t1.wav", sample_wav_bytes, "audio/wav")},
            data={"module": "module1", "language": "en"},
        )
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1["lead"]["name"] == "Priya"
        assert d1["lead"]["phone"] == "9876543210"
        # Company is the next missing required field
        assert d1["next_missing_parameter"] == "company"
        assert d1["is_complete"] is False
        # Asks ONLY for company
        assert "company" in d1["immediate_sentence1"].lower() or "organization" in d1["immediate_sentence1"].lower()
        # Never repeats name or phone in question
        assert "name" not in d1["immediate_sentence1"].lower() or "thank you, priya" in d1["immediate_sentence1"].lower()
        assert "phone" not in d1["immediate_sentence1"].lower()

    # TURN 2: User provides Company, Loan Type, and Loan Amount (3 fields in 1 sentence)
    t2_mock = {
        "transcript": "I work at Infosys and I need 20 lakhs business loan",
        "detected_language": "en",
        "language": "en",
        "confidence": 0.99,
        "words_count": 10,
        "duration": 3.0,
        "model": "nova-3",
    }
    existing_t2 = d1["lead"]
    lead_id_t1 = d1.get("lead_id")
    with patch.object(DeepgramSTTService, "transcribe_audio", new_callable=AsyncMock, return_value=t2_mock):
        r2 = client.post(
            "/api/voice-entry",
            files={"file": ("t2.wav", sample_wav_bytes, "audio/wav")},
            data={
                "module": "module1",
                "language": "en",
                "existing_lead": json.dumps(existing_t2),
                "lead_id": str(lead_id_t1) if lead_id_t1 else "",
            },
        )
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["lead"]["name"] == "Priya"
        assert d2["lead"]["phone"] == "9876543210"
        assert d2["lead"]["company"] == "Infosys"
        assert d2["lead"]["loan_type"] == "Business Loan"
        assert d2["lead"]["loan_amount"] == 2000000.0
        # Only tenure_months is missing now
        assert d2["next_missing_parameter"] == "tenure_months"
        assert d2["is_complete"] is False
        # Asks ONLY for tenure duration
        assert "tenure" in d2["immediate_sentence1"].lower() or "duration" in d2["immediate_sentence1"].lower()
        # Never asks for company, phone, loan_type, loan_amount
        assert "company" not in d2["immediate_sentence1"].lower()
        assert "phone" not in d2["immediate_sentence1"].lower()

    # TURN 3: User provides Tenure
    t3_mock = {
        "transcript": "for 3 years",
        "detected_language": "en",
        "language": "en",
        "confidence": 0.99,
        "words_count": 3,
        "duration": 1.5,
        "model": "nova-3",
    }
    existing_t3 = d2["lead"]
    with patch.object(DeepgramSTTService, "transcribe_audio", new_callable=AsyncMock, return_value=t3_mock):
        r3 = client.post(
            "/api/voice-entry",
            files={"file": ("t3.wav", sample_wav_bytes, "audio/wav")},
            data={
                "module": "module1",
                "language": "en",
                "existing_lead": json.dumps(existing_t3),
                "lead_id": str(lead_id_t1) if lead_id_t1 else "",
            },
        )
        assert r3.status_code == 200
        d3 = r3.json()
        assert d3["lead"]["name"] == "Priya"
        assert d3["lead"]["phone"] == "9876543210"
        assert d3["lead"]["company"] == "Infosys"
        assert d3["lead"]["loan_type"] == "Business Loan"
        assert d3["lead"]["loan_amount"] == 2000000.0
        assert d3["lead"]["tenure_months"] == 36
        # All 6 required fields collected!
        assert d3["next_missing_parameter"] is None
        assert d3["is_complete"] is True
        assert "recorded" in d3["immediate_sentence1"].lower() or "thank you" in d3["immediate_sentence1"].lower()


def test_module1_assistant_name_query(client, sample_wav_bytes):
    """Verifies that 'What is your name?' gets natural assistant response, not lead-field prompt or name extraction."""
    mock_stt = {
        "success": True,
        "transcript": "What is your name?",
        "language": "en",
        "confidence": 0.98,
        "words_count": 4,
        "duration": 1.2,
        "model": "nova-3",
    }
    with patch.object(DeepgramSTTService, "transcribe_audio", new_callable=AsyncMock, return_value=mock_stt):
        resp = client.post(
            "/api/voice-entry",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
            data={"module": "module1", "language": "en"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("is_assistant_query") is True
        assert "VoiceCopilot" in data["immediate_sentence1"]
        assert "VoiceCopilot" in data["assistant_response"]
        # Must NOT treat 'What is your name' as prospect's name
        lead = data.get("lead") or {}
        assert lead.get("name") is None or lead.get("name") != "What is your name"


def test_module1_assistant_who_are_you_query(client, sample_wav_bytes):
    """Verifies that 'Who are you?' gets natural assistant identity response."""
    mock_stt = {
        "success": True,
        "transcript": "Who are you?",
        "language": "en",
        "confidence": 0.99,
        "words_count": 3,
        "duration": 1.0,
        "model": "nova-3",
    }
    with patch.object(DeepgramSTTService, "transcribe_audio", new_callable=AsyncMock, return_value=mock_stt):
        resp = client.post(
            "/api/voice-entry",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
            data={"module": "module1", "language": "en"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("is_assistant_query") is True
        assert "VoiceCopilot" in data["immediate_sentence1"]
        assert "sales copilot" in data["immediate_sentence1"].lower() or "assistant" in data["immediate_sentence1"].lower()


def test_module1_assistant_what_can_you_do_query(client, sample_wav_bytes):
    """Verifies that 'What can you do?' gets natural assistant capabilities response."""
    mock_stt = {
        "success": True,
        "transcript": "What can you do?",
        "language": "en",
        "confidence": 0.99,
        "words_count": 4,
        "duration": 1.1,
        "model": "nova-3",
    }
    with patch.object(DeepgramSTTService, "transcribe_audio", new_callable=AsyncMock, return_value=mock_stt):
        resp = client.post(
            "/api/voice-entry",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
            data={"module": "module1", "language": "en"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("is_assistant_query") is True
        assert "collect" in data["immediate_sentence1"].lower() or "leads" in data["immediate_sentence1"].lower()


def test_module1_assistant_queries_hindi_and_marathi(client, sample_wav_bytes):
    """Verifies Hindi and Marathi assistant queries return localized answers."""
    # Hindi: आपका नाम क्या है?
    mock_hi = {
        "success": True,
        "transcript": "आपका नाम क्या है?",
        "language": "hi",
        "confidence": 0.98,
        "words_count": 4,
        "duration": 1.2,
        "model": "nova-3",
    }
    with patch.object(DeepgramSTTService, "transcribe_audio", new_callable=AsyncMock, return_value=mock_hi):
        resp = client.post(
            "/api/voice-entry",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
            data={"module": "module1", "language": "hi"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("is_assistant_query") is True
        assert "वॉयस कोपायलट" in data["immediate_sentence1"]

    # Marathi: तुम्ही काय करू शकता?
    mock_mr = {
        "success": True,
        "transcript": "तुम्ही काय करू शकता?",
        "language": "mr",
        "confidence": 0.98,
        "words_count": 4,
        "duration": 1.2,
        "model": "nova-3",
    }
    with patch.object(DeepgramSTTService, "transcribe_audio", new_callable=AsyncMock, return_value=mock_mr):
        resp = client.post(
            "/api/voice-entry",
            files={"file": ("test.wav", sample_wav_bytes, "audio/wav")},
            data={"module": "module1", "language": "mr"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("is_assistant_query") is True
        assert "मदत" in data["immediate_sentence1"] or "सीआरएम" in data["immediate_sentence1"]


