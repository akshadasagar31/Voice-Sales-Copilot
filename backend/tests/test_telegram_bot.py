# ============================================================================
# UNIT & INTEGRATION TESTS: TELEGRAM BOT INTEGRATION (VoiceCopilotBot)
# ============================================================================
# Verifies end-to-end functionality of Module 2 Telegram Bot:
# 1. Text Message -> RAG -> immediate text reply -> TTS voice reply.
# 2. Voice Note (.ogg) -> Sarvam STT -> RAG -> text reply -> TTS voice reply.
# 3. STT fallback to Deepgram Nova-3.
# 4. TTS fallback to Deepgram Aura.
# 5. Telegram sendVoice fallback to sendAudio.
# 6. Webhook FastAPI endpoints (/api/telegram/webhook, /api/telegram/status).
# ============================================================================

import os
import sys
import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app
from services.telegram_bot import (
    VoiceCopilotBot,
    TelegramBotConfigurationError,
    TelegramBotAPIError,
)
from services.rag import RAGService
from services.sarvam_stt import SarvamSTTService
from services.stt import DeepgramSTTService
from services.sarvam_tts import SarvamTTSService
from services.tts import DeepgramTTSService


@pytest.fixture
def test_client():
    return TestClient(app)


from services.lead_extractor import LeadExtractorService
from services.lead_repository import LeadRepository


@pytest.fixture
def mock_bot():
    """Returns a VoiceCopilotBot instance with mock services and fake token."""
    mock_rag = MagicMock(spec=RAGService)
    mock_sarvam_stt = MagicMock(spec=SarvamSTTService)
    mock_deepgram_stt = MagicMock(spec=DeepgramSTTService)
    mock_sarvam_tts = MagicMock(spec=SarvamTTSService)
    mock_deepgram_tts = MagicMock(spec=DeepgramTTSService)
    mock_lead_extractor = MagicMock(spec=LeadExtractorService)
    mock_lead_repo = MagicMock(spec=LeadRepository)
    mock_http = AsyncMock()

    bot = VoiceCopilotBot(
        bot_token="test_token_123456",
        rag_service=mock_rag,
        sarvam_stt=mock_sarvam_stt,
        deepgram_stt=mock_deepgram_stt,
        sarvam_tts=mock_sarvam_tts,
        deepgram_tts=mock_deepgram_tts,
        lead_extractor=mock_lead_extractor,
        lead_repository=mock_lead_repo,
        client=mock_http,
    )
    return bot


# ---------------------------------------------------------------------------
# 1. Bot Configuration & Initialization Tests
# ---------------------------------------------------------------------------

def test_bot_is_configured_property():
    bot_with_token = VoiceCopilotBot(bot_token="12345:ABCDEF")
    assert bot_with_token.is_configured is True
    assert "12345:ABCDEF" in bot_with_token.api_url

    bot_empty = VoiceCopilotBot(bot_token="")
    assert bot_empty.is_configured is False
    with pytest.raises(TelegramBotConfigurationError):
        _ = bot_empty.api_url


# ---------------------------------------------------------------------------
# 2. Command Handling (/start, /help, /reset)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_process_command_start(mock_bot):
    mock_bot.send_message = AsyncMock(return_value={"ok": True})

    res = await mock_bot.process_text_message(chat_id=98765, text="/start")
    assert res["status"] == "command_handled"
    assert res["command"] == "/start"
    mock_bot.send_message.assert_awaited_once()
    sent_text = mock_bot.send_message.call_args[0][1]
    assert "Voice Sales Copilot" in sent_text


@pytest.mark.asyncio
async def test_process_command_reset(mock_bot):
    mock_bot.send_message = AsyncMock(return_value={"ok": True})
    mock_bot.get_lead_session(111)["lead"] = {"name": "Test User"}

    res = await mock_bot.process_text_message(chat_id=111, text="/reset")
    assert res["status"] == "command_handled"
    assert res["command"] == "/reset"
    # Verify session was cleared
    assert mock_bot.get_lead_session(111)["lead"] == {}


# ---------------------------------------------------------------------------
# 3. Text Message -> Module 1 Lead Extraction & CRM Multi-Turn
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_process_text_message_flow(mock_bot):
    # Setup mock lead extraction
    mock_bot._lead_extractor.extract_lead.return_value = {
        "status": "success",
        "lead": {
            "name": "Rajesh Kumar",
            "loan_type": "Personal Loan",
            "phone": None,
            "loan_amount": None,
            "email": None,
            "company": None,
            "role": None,
            "tenure_months": None,
            "notes": None,
        },
        "next_missing_parameter": "phone",
        "is_complete": False,
        "message": "Thank you, Rajesh Kumar. What is your contact phone number?",
    }
    # Setup mock Sarvam TTS
    fake_wav = b"RIFF" + b"\x00" * 40
    mock_bot._sarvam_tts.synthesize_speech = AsyncMock(return_value=fake_wav)

    mock_bot.send_chat_action = AsyncMock(return_value=True)
    mock_bot.send_message = AsyncMock(return_value={"ok": True})
    mock_bot.send_voice = AsyncMock(return_value={"ok": True})

    result = await mock_bot.process_text_message(
        chat_id=12345,
        text="My name is Rajesh Kumar and I am looking for a Personal Loan.",
    )

    assert result["status"] == "success"
    assert result["lead"]["name"] == "Rajesh Kumar"
    assert result["next_missing_parameter"] == "phone"
    assert result["is_complete"] is False
    assert result["has_voice"] is True

    # Verify Module 1 LeadExtractor was called
    mock_bot._lead_extractor.extract_lead.assert_called_once_with(
        transcript="My name is Rajesh Kumar and I am looking for a Personal Loan.",
        existing_lead={},
        language="en",
    )

    # Incomplete lead -> create_lead NOT called yet
    mock_bot._lead_repository.create_lead.assert_not_called()

    # Verify ONE copilot container was sent via send_voice containing voice + plain text caption
    mock_bot.send_voice.assert_awaited_once()
    assert "What is your contact phone number?" in mock_bot.send_voice.call_args.kwargs.get("caption")
    mock_bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_multi_turn_lead_collection_and_crm_save(mock_bot):
    """
    Verifies multi-turn memory:
    Turn 1: Customer provides Name. Bot remembers and asks for missing phone.
    Turn 2: Customer provides Phone, Loan Type, and Amount.
    Bot merges previously provided details (NEVER re-asking for Name),
    saves the complete lead to PostgreSQL CRM, and returns a short Thank You message.
    """
    fake_wav = b"RIFF" + b"\x00" * 40
    mock_bot._sarvam_tts.synthesize_speech = AsyncMock(return_value=fake_wav)
    mock_bot.send_chat_action = AsyncMock(return_value=True)
    mock_bot.send_voice = AsyncMock(return_value={"ok": True})
    mock_bot.send_message = AsyncMock(return_value={"ok": True})

    # Turn 1: Name provided
    mock_bot._lead_extractor.extract_lead.side_effect = [
        # Turn 1 response
        {
            "status": "success",
            "lead": {
                "name": "Amit Verma",
                "phone": None,
                "loan_type": None,
                "loan_amount": None,
            },
            "next_missing_parameter": "phone",
            "is_complete": False,
            "message": "Thank you, Amit Verma. What is your contact phone number?",
        },
        # Turn 2 response (all 4 required fields complete)
        {
            "status": "success",
            "lead": {
                "name": "Amit Verma",
                "phone": "9876543210",
                "loan_type": "Personal Loan",
                "loan_amount": 500000.0,
            },
            "next_missing_parameter": None,
            "is_complete": True,
            "message": "Thank you, Amit Verma! All your details have been recorded. Our team will contact you shortly.",
        },
    ]

    mock_bot._lead_repository.create_lead.return_value = {
        "id": 101,
        "name": "Amit Verma",
        "phone": "9876543210",
        "loan_type": "Personal Loan",
        "loan_amount": 500000.0,
    }

    # Turn 1 execution
    turn1_res = await mock_bot.process_text_message(chat_id=777, text="Hi, my name is Amit Verma")
    assert turn1_res["status"] == "success"
    assert turn1_res["lead"]["name"] == "Amit Verma"
    assert turn1_res["next_missing_parameter"] == "phone"
    assert turn1_res["is_complete"] is False
    mock_bot._lead_repository.create_lead.assert_not_called()

    # Turn 2 execution: provides remaining details
    turn2_res = await mock_bot.process_text_message(
        chat_id=777,
        text="My number is 9876543210 and I need a personal loan of 5 lakhs.",
    )

    # Verify Turn 2 passed existing_lead containing Amit Verma (remembered across turns)
    turn2_call_args = mock_bot._lead_extractor.extract_lead.call_args_list[1]
    assert turn2_call_args.kwargs["existing_lead"]["name"] == "Amit Verma"

    # Verify complete lead was committed to CRM database
    assert turn2_res["is_complete"] is True
    assert turn2_res["lead_id"] == 101
    mock_bot._lead_repository.create_lead.assert_called_once()
    saved_lead = mock_bot._lead_repository.create_lead.call_args[0][0]
    assert saved_lead.name == "Amit Verma"
    assert saved_lead.phone == "9876543210"
    assert saved_lead.loan_type == "Personal Loan"
    assert saved_lead.loan_amount == 500000.0

    # Verify short Thank You confirmation was sent
    assert "Thank you, Amit Verma" in turn2_res["answer"]
    assert "recorded" in turn2_res["answer"]


# ---------------------------------------------------------------------------
# 4. Voice Note -> STT -> Module 1 Lead Extraction -> Voice Reply
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_process_voice_message_flow(mock_bot):
    # Mock file download from Telegram
    mock_bot.download_file = AsyncMock(return_value=(b"FAKE_OGG_AUDIO_BYTES", "ogg"))

    # Mock Sarvam STT
    mock_bot._sarvam_stt.transcribe_audio = AsyncMock(return_value={
        "transcript": "नमस्कार, माझं नाव राहुल पाटील आहे आणि मला 10 लाख रुपयांचं व्यवसाय कर्ज हवं आहे.",
        "detected_language": "mr",
    })

    # Mock Lead Extractor
    mock_bot._lead_extractor.extract_lead.return_value = {
        "status": "success",
        "lead": {
            "name": "राहुल पाटील",
            "loan_type": "व्यवसाय कर्ज",
            "loan_amount": 1000000.0,
            "phone": None,
        },
        "next_missing_parameter": "phone",
        "is_complete": False,
        "message": "धन्यवाद राहुल पाटील जी. कृपया आपला संपर्क फोन नंबर सांगा?",
    }

    # Mock TTS
    fake_wav = b"RIFF" + b"\x00" * 40
    mock_bot._sarvam_tts.synthesize_speech = AsyncMock(return_value=fake_wav)

    mock_bot.send_chat_action = AsyncMock(return_value=True)
    mock_bot.send_message = AsyncMock(return_value={"ok": True})
    mock_bot.send_voice = AsyncMock(return_value={"ok": True})

    voice_payload = {
        "file_id": "telegram_voice_file_abc123",
        "mime_type": "audio/ogg",
        "duration": 5,
    }

    result = await mock_bot.process_voice_message(chat_id=54321, voice_data=voice_payload)

    assert result["status"] == "success"
    assert result["query_type"] == "voice"
    assert "राहुल पाटील" in result["lead"]["name"]
    assert result["next_missing_parameter"] == "phone"
    assert result["language"] == "mr"
    assert result["has_voice"] is True

    # Verify download called
    mock_bot.download_file.assert_awaited_once_with("telegram_voice_file_abc123")

    # Verify Sarvam STT called with unknown auto-detect
    mock_bot._sarvam_stt.transcribe_audio.assert_awaited_once_with(
        audio_bytes=b"FAKE_OGG_AUDIO_BYTES",
        content_type="audio/ogg",
        language_code="unknown",
        model="saaras:v4",
        filename="telegram_voice.ogg",
    )

    # Verify ONE copilot container was sent via send_voice with clean caption
    mock_bot.send_voice.assert_awaited_once()
    sent_caption = mock_bot.send_voice.call_args.kwargs.get("caption")
    assert sent_caption is not None
    assert "राहुल पाटील" in sent_caption
    mock_bot.send_message.assert_not_called()


# ---------------------------------------------------------------------------
# 5. STT Fallback to Deepgram
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stt_fallback_to_deepgram(mock_bot):
    mock_bot.download_file = AsyncMock(return_value=(b"FAKE_AUDIO", "ogg"))

    # Sarvam fails
    mock_bot._sarvam_stt.transcribe_audio = AsyncMock(side_effect=Exception("Sarvam API 500 error"))
    # Deepgram succeeds
    mock_bot._deepgram_stt.transcribe_audio = AsyncMock(return_value={
        "transcript": "My name is John Doe and phone is 9876543210",
        "language": "en",
    })

    mock_bot._lead_extractor.extract_lead.return_value = {
        "status": "success",
        "lead": {"name": "John Doe", "phone": "9876543210", "loan_type": None, "loan_amount": None},
        "next_missing_parameter": "loan_type",
        "is_complete": False,
        "message": "Thank you, John Doe. What type of loan are you looking for?",
    }
    mock_bot._sarvam_tts.synthesize_speech = AsyncMock(return_value=b"RIFF_AUDIO")

    mock_bot.send_chat_action = AsyncMock(return_value=True)
    mock_bot.send_message = AsyncMock(return_value={"ok": True})
    mock_bot.send_voice = AsyncMock(return_value={"ok": True})

    result = await mock_bot.process_voice_message(chat_id=111, voice_data={"file_id": "f_123"})
    assert result["status"] == "success"
    assert result["query"] == "My name is John Doe and phone is 9876543210"
    mock_bot._deepgram_stt.transcribe_audio.assert_awaited_once()


# ---------------------------------------------------------------------------
# 6. TTS Fallback to Deepgram
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tts_fallback_to_deepgram(mock_bot):
    # Sarvam TTS fails
    mock_bot._sarvam_tts.synthesize_speech = AsyncMock(side_effect=Exception("Sarvam quota exceeded"))
    # Deepgram TTS succeeds
    mock_bot._deepgram_tts.synthesize_speech = AsyncMock(return_value=b"DEEPGRAM_WAV_BYTES")

    audio = await mock_bot._synthesize_voice_answer("Test answer text", "en")
    assert audio == b"DEEPGRAM_WAV_BYTES"
    mock_bot._deepgram_tts.synthesize_speech.assert_awaited_once_with(
        "Test answer text",
        language="en",
    )


# ---------------------------------------------------------------------------
# 7. Telegram sendVoice Fallback to sendAudio
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_voice_fallback_to_send_audio(mock_bot):
    # First call to /sendVoice fails with 400
    mock_resp_fail = MagicMock()
    mock_resp_fail.status_code = 400
    mock_resp_fail.text = "Bad Request: wrong voice format"

    # Second call to /sendAudio succeeds
    mock_resp_ok = MagicMock()
    mock_resp_ok.status_code = 200
    mock_resp_ok.json.return_value = {"ok": True, "result": {"message_id": 999}}

    mock_bot.client.post = AsyncMock(side_effect=[mock_resp_fail, mock_resp_ok])

    res = await mock_bot.send_voice(chat_id=123, audio_bytes=b"RIFF_AUDIO")
    assert res["ok"] is True
    assert mock_bot.client.post.await_count == 2
    second_call_url = mock_bot.client.post.call_args_list[1][0][0]
    assert "sendAudio" in second_call_url


# ---------------------------------------------------------------------------
# 8. Webhook Dispatcher & FastAPI Endpoints
# ---------------------------------------------------------------------------

def test_telegram_status_endpoint(test_client):
    response = test_client.get("/api/telegram/status")
    assert response.status_code == 200
    data = response.json()
    assert data["bot_name"] == "VoiceCopilotBot"
    assert data["webhook_endpoint"] == "/api/telegram/webhook"
    assert "voice_notes" in data["capabilities"]


def test_telegram_webhook_endpoint_text(test_client):
    mock_result = {
        "status": "success",
        "chat_id": 12345,
        "query_type": "text",
        "answer": "Test answer",
    }
    with patch("services.telegram_bot.VoiceCopilotBot.handle_webhook_update", new_callable=AsyncMock) as mock_handle:
        mock_handle.return_value = mock_result

        payload = {
            "update_id": 10001,
            "message": {
                "message_id": 1,
                "chat": {"id": 12345, "type": "private"},
                "text": "What is the policy for personal loans?",
            },
        }
        response = test_client.post("/api/telegram/webhook", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["result"]["status"] == "success"
        mock_handle.assert_awaited_once_with(payload)


def test_telegram_webhook_endpoint_voice(test_client):
    mock_result = {
        "status": "success",
        "chat_id": 12345,
        "query_type": "voice",
        "transcript": "Test speech",
        "answer": "Test answer",
    }
    with patch("services.telegram_bot.VoiceCopilotBot.handle_webhook_update", new_callable=AsyncMock) as mock_handle:
        mock_handle.return_value = mock_result

        payload = {
            "update_id": 10002,
            "message": {
                "message_id": 2,
                "chat": {"id": 12345, "type": "private"},
                "voice": {
                    "file_id": "test_voice_file_id_xyz",
                    "mime_type": "audio/ogg",
                    "duration": 4,
                },
            },
        }
        response = test_client.post("/api/telegram/webhook", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["result"]["query_type"] == "voice"


@pytest.mark.asyncio
async def test_telegram_polling_worker(mock_bot):
    """Verifies that poll_once fetches updates and dispatches them with updated offset."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "ok": True,
        "result": [
            {
                "update_id": 5001,
                "message": {
                    "message_id": 1,
                    "chat": {"id": 8888, "type": "private"},
                    "text": "Hello bot",
                },
            }
        ],
    }
    mock_bot.client.get = AsyncMock(return_value=mock_resp)
    mock_bot.handle_webhook_update = AsyncMock(return_value={"status": "success"})

    count = await mock_bot.poll_once()
    assert count == 1
    assert mock_bot._last_update_id == 5001
    await asyncio.sleep(0.01)  # allow spawned task to run
    mock_bot.handle_webhook_update.assert_awaited_once()


@pytest.mark.asyncio
async def test_single_copilot_container_delivery(mock_bot):
    """Verifies that the bot delivers ONE copilot container (voice note + plain text answer caption)."""
    call_order = []

    mock_bot.download_file = AsyncMock(return_value=(b"FAKE_OGG", "ogg"))
    mock_bot._sarvam_stt.transcribe_audio = AsyncMock(return_value={"transcript": "My name is Priya"})
    mock_bot._lead_extractor.extract_lead.return_value = {
        "status": "success",
        "lead": {"name": "Priya"},
        "next_missing_parameter": "phone",
        "is_complete": False,
        "message": "Thank you, Priya. What is your contact phone number?",
    }

    async def mock_send_message(*args, **kwargs):
        call_order.append("send_message")
        return {"ok": True}

    async def mock_synthesize(*args, **kwargs):
        call_order.append("synthesize_speech")
        return b"RIFF_AUDIO"

    async def mock_send_voice(*args, **kwargs):
        call_order.append("send_voice")
        return {"ok": True}

    mock_bot.send_message = AsyncMock(side_effect=mock_send_message)
    mock_bot._sarvam_tts.synthesize_speech = AsyncMock(side_effect=mock_synthesize)
    mock_bot.send_voice = AsyncMock(side_effect=mock_send_voice)

    result = await mock_bot.process_voice_message(chat_id=1234, voice_data={"file_id": "f_99"})
    assert result["status"] == "success"
    # Verify strict container delivery: synthesizes audio then delivers ONE send_voice container with caption
    assert call_order == ["synthesize_speech", "send_voice"]
    mock_bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_copilot_container_voice_fallback_to_text(mock_bot):
    """Verifies that if voice synthesis returns None, bot falls back to single text message container."""
    mock_bot._lead_extractor.extract_lead.return_value = {
        "status": "success",
        "lead": {"name": "Alex"},
        "next_missing_parameter": "phone",
        "is_complete": False,
        "message": "Fallback text answer.",
    }
    mock_bot.send_chat_action = AsyncMock(return_value=True)
    mock_bot.send_message = AsyncMock(return_value={"ok": True})
    mock_bot.send_voice = AsyncMock(return_value={"ok": True})
    mock_bot._synthesize_voice_answer = AsyncMock(return_value=None)

    result = await mock_bot.process_text_message(chat_id=999, text="My name is Alex")
    assert result["status"] == "success"
    assert result["has_voice"] is False
    mock_bot.send_voice.assert_not_called()
    mock_bot.send_message.assert_awaited_once_with(999, "Fallback text answer.", parse_mode=None)


# ---------------------------------------------------------------------------
# 10. Mobile Response Formatting Unit Tests
# ---------------------------------------------------------------------------

def test_mobile_response_formatting_bullet_points():
    raw_answer = (
        "Here are the eligibility criteria:\n"
        "* **Age:** Minimum 21 years and maximum 60 years.\n"
        "* **Income:** Net monthly salary of at least ₹25,000.\n"
        "* **CIBIL Score:** 720 or higher is required."
    )
    formatted = VoiceCopilotBot._clean_plain_text(raw_answer)

    assert "*" not in formatted
    assert "**" not in formatted
    assert "• Age: Minimum 21 years and maximum 60 years." in formatted
    assert "• Income: Net monthly salary of at least ₹25,000." in formatted
    assert "• CIBIL Score: 720 or higher is required." in formatted
    assert "Here are the eligibility criteria:\n\n• Age:" in formatted


def test_mobile_response_formatting_inline_numbers():
    raw_answer = (
        "To apply for a personal loan: "
        "1) Check your credit score, "
        "2) Submit KYC and income proof, "
        "3) Complete digital verification."
    )
    formatted = VoiceCopilotBot._clean_plain_text(raw_answer)

    assert "1. Check your credit score" in formatted
    assert "2. Submit KYC and income proof" in formatted
    assert "3. Complete digital verification." in formatted


def test_mobile_response_formatting_technical_labels_removed():
    raw_answer = (
        "Playbook Answer: Here is the info.\n"
        "According to Sarvam AI and Pinecone database:\n"
        "1. First step\n"
        "2. Second step\n"
        "Source: hdfc_personal_loan.pdf\n"
        "Confidence: 0.95"
    )
    formatted = VoiceCopilotBot._clean_plain_text(raw_answer)

    assert "Playbook Answer:" not in formatted
    assert "Sarvam" not in formatted
    assert "Pinecone" not in formatted
    assert "Source:" not in formatted
    assert "hdfc_personal_loan.pdf" not in formatted
    assert "Confidence:" not in formatted
    assert "1. First step" in formatted
    assert "2. Second step" in formatted


def test_mobile_response_formatting_caption_length_boundary():
    long_answer = "This is a key point about personal loan eligibility. " * 30
    formatted = VoiceCopilotBot._clean_plain_text(long_answer)

    assert len(formatted) <= 1024
    assert not formatted.endswith("  ")


# ---------------------------------------------------------------------------
# 11. Module 2 Knowledge Intent Routing Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_knowledge_question_services_provided_routes_to_module2(mock_bot):
    """
    Verifies that 'What services do you provide?' routes to Module 2 RAG,
    calls RAGService.answer_question, delivers the answer, and NEVER calls LeadExtractor
    or asks for Name.
    """
    mock_bot._rag_service.answer_question.return_value = {
        "question": "What services do you provide?",
        "answer": "We offer personal loans, home loans, and business loans with flexible repayment options and competitive interest rates.",
        "sources": ["services_overview.pdf"],
        "fallback_used": False,
    }
    fake_wav = b"RIFF" + b"\x00" * 40
    mock_bot._sarvam_tts.synthesize_speech = AsyncMock(return_value=fake_wav)
    mock_bot.send_chat_action = AsyncMock(return_value=True)
    mock_bot.send_voice = AsyncMock(return_value={"ok": True})
    mock_bot.send_message = AsyncMock(return_value={"ok": True})

    res = await mock_bot.process_text_message(
        chat_id=12345,
        text="What services do you provide?",
    )

    assert res["status"] == "success"
    assert res["module"] == "module2"
    assert "personal loans, home loans, and business loans" in res["answer"]
    assert "name" not in res["answer"].lower() or "full name" not in res["answer"].lower()

    # RAG service must be called
    mock_bot._rag_service.answer_question.assert_called_once()
    assert mock_bot._rag_service.answer_question.call_args.kwargs["question"] == "What services do you provide?"

    # LeadExtractor must NEVER be called for knowledge questions
    mock_bot._lead_extractor.extract_lead.assert_not_called()
    mock_bot._lead_repository.create_lead.assert_not_called()

    # Verify voice response delivery
    mock_bot.send_voice.assert_awaited_once()


@pytest.mark.asyncio
async def test_knowledge_question_personal_loan_routes_to_module2(mock_bot):
    """
    Verifies that 'What is a personal loan?' routes to Module 2 RAG,
    calls RAGService.answer_question, delivers the answer, and NEVER calls LeadExtractor
    or asks for Name.
    """
    mock_bot._rag_service.answer_question.return_value = {
        "question": "What is a personal loan?",
        "answer": "A personal loan is an unsecured credit option provided by financial institutions based on your income and credit score, requiring no collateral.",
        "sources": ["loan_playbook.pdf"],
        "fallback_used": False,
    }
    fake_wav = b"RIFF" + b"\x00" * 40
    mock_bot._sarvam_tts.synthesize_speech = AsyncMock(return_value=fake_wav)
    mock_bot.send_chat_action = AsyncMock(return_value=True)
    mock_bot.send_voice = AsyncMock(return_value={"ok": True})
    mock_bot.send_message = AsyncMock(return_value={"ok": True})

    res = await mock_bot.process_text_message(
        chat_id=12345,
        text="What is a personal loan?",
    )

    assert res["status"] == "success"
    assert res["module"] == "module2"
    assert "unsecured credit option" in res["answer"]
    assert "full name" not in res["answer"].lower()
    assert "may i have your name" not in res["answer"].lower()

    # RAG service must be called
    mock_bot._rag_service.answer_question.assert_called_once()
    assert mock_bot._rag_service.answer_question.call_args.kwargs["question"] == "What is a personal loan?"

    # LeadExtractor must NEVER be called
    mock_bot._lead_extractor.extract_lead.assert_not_called()
    mock_bot._lead_repository.create_lead.assert_not_called()


@pytest.mark.asyncio
async def test_voice_knowledge_question_routes_to_module2(mock_bot):
    """
    Verifies that a voice note with a knowledge question routes to Module 2 RAG,
    synthesizes a voice response, and never asks for lead parameters.
    """
    mock_bot.download_file = AsyncMock(return_value=(b"FAKE_AUDIO", "ogg"))
    mock_bot._sarvam_stt.transcribe_audio = AsyncMock(return_value={
        "transcript": "What is a personal loan?",
        "detected_language": "en",
    })
    mock_bot._rag_service.answer_question.return_value = {
        "question": "What is a personal loan?",
        "answer": "A personal loan is an unsecured loan that you can use for any financial need without pledging collateral.",
        "sources": ["faq.pdf"],
    }
    fake_wav = b"RIFF" + b"\x00" * 40
    mock_bot._sarvam_tts.synthesize_speech = AsyncMock(return_value=fake_wav)
    mock_bot.send_chat_action = AsyncMock(return_value=True)
    mock_bot.send_voice = AsyncMock(return_value={"ok": True})

    res = await mock_bot.process_voice_message(chat_id=8899, voice_data={"file_id": "f_v_1"})

    assert res["status"] == "success"
    assert res["module"] == "module2"
    assert "unsecured loan" in res["answer"]
    mock_bot._rag_service.answer_question.assert_called_once()
    mock_bot._lead_extractor.extract_lead.assert_not_called()
    mock_bot.send_voice.assert_awaited_once()


