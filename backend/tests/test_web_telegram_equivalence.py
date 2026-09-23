# ============================================================================
# EQUIVALENCE TEST SUITE: WEB MODULE 1 vs. TELEGRAM MODULE 1
# ============================================================================
# Verifies that Web Module 1 and Telegram Module 1 use the EXACT SAME:
# 1. Lead extraction logic (LeadExtractorService)
# 2. Lead Pydantic model and schema
# 3. Missing-field determination sequence (REQUIRED_LEAD_FIELDS = name, phone, loan_type, loan_amount)
# 4. Spoken question prompts for missing fields across English, Hindi, and Marathi
# 5. Completion conditions (is_complete = True when all 4 required fields collected)
# 6. Final thank-you confirmation message
# 7. PostgreSQL CRM database persistence (LeadRepository.create_lead)
# ============================================================================

import os
import sys
import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.lead_extractor import (
    LeadExtractorService,
    Lead,
    REQUIRED_LEAD_FIELDS,
    get_next_missing_parameter,
    get_missing_parameter_prompt,
)
from services.lead_repository import LeadRepository
from services.telegram_bot import VoiceCopilotBot


class MockHTTPClient:
    """Mock HTTP client simulating OpenRouter responses for lead extraction."""
    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0

    def post(self, url, json=None, headers=None, **kwargs):
        import json as _json_module
        resp_mock = MagicMock()
        resp_mock.status_code = 200
        if self.responses and self.call_count < len(self.responses):
            ret = self.responses[self.call_count]
            self.call_count += 1
            resp_mock.json.return_value = {
                "choices": [{"message": {"content": _json_module.dumps(ret)}}]
            }
        else:
            resp_mock.json.return_value = {
                "choices": [{"message": {"content": "{}"}}]
            }
        return resp_mock


# ---------------------------------------------------------------------------
# 1. Multi-Turn English Lead Flow Equivalence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multi_turn_english_equivalence():
    """
    Verifies that sequential turns (Name -> Phone -> Company -> Loan Type -> Loan Amount -> Tenure)
    produce the EXACT same lead state, missing fields, questions, and completion flow
    on both Web Module 1 and Telegram Module 1.
    """
    llm_turns = [
        {"name": "Rajesh Sharma", "phone": None, "company": None, "loan_type": None, "loan_amount": None, "tenure_months": None},
        {"name": "Rajesh Sharma", "phone": "9876543210", "company": None, "loan_type": None, "loan_amount": None, "tenure_months": None},
        {"name": "Rajesh Sharma", "phone": "9876543210", "company": "Infosys", "loan_type": None, "loan_amount": None, "tenure_months": None},
        {"name": "Rajesh Sharma", "phone": "9876543210", "company": "Infosys", "loan_type": "Personal Loan", "loan_amount": None, "tenure_months": None},
        {"name": "Rajesh Sharma", "phone": "9876543210", "company": "Infosys", "loan_type": "Personal Loan", "loan_amount": 500000.0, "tenure_months": None},
        {"name": "Rajesh Sharma", "phone": "9876543210", "company": "Infosys", "loan_type": "Personal Loan", "loan_amount": 500000.0, "tenure_months": 24},
    ]

    # Two identical extractors: one for Web, one for Telegram
    web_extractor = LeadExtractorService(api_key="test_key", http_client=MockHTTPClient(list(llm_turns)))
    tg_extractor = LeadExtractorService(api_key="test_key", http_client=MockHTTPClient(list(llm_turns)))

    mock_repo = MagicMock(spec=LeadRepository)
    mock_repo.create_lead.side_effect = lambda l: {"id": 101, **l.model_dump()}

    bot = VoiceCopilotBot(
        bot_token="test_token",
        lead_extractor=tg_extractor,
        lead_repository=mock_repo,
        client=AsyncMock(),
    )
    bot._synthesize_voice_answer = AsyncMock(return_value=b"AUDIO")
    bot.send_voice = AsyncMock(return_value={"ok": True})
    bot.send_message = AsyncMock(return_value={"ok": True})
    bot.send_chat_action = AsyncMock(return_value=True)

    chat_id = 998877
    web_existing = None

    test_inputs = [
        "My name is Rajesh Sharma",
        "9876543210",
        "I work at Infosys",
        "I need a Personal Loan",
        "500000",
        "2 years",
    ]

    for turn_idx, user_input in enumerate(test_inputs):
        # 1. Web Module 1 execution
        web_res = web_extractor.extract_lead(
            transcript=user_input,
            existing_lead=web_existing,
            language="en",
        )
        web_existing = web_res["lead"]

        # 2. Telegram Module 1 execution
        tg_res = await bot.process_text_message(chat_id=chat_id, text=user_input)

        # 3. Equivalence assertions for this turn
        assert web_res["lead"]["name"] == tg_res["lead"]["name"]
        assert web_res["lead"]["phone"] == tg_res["lead"]["phone"]
        assert web_res["lead"]["company"] == tg_res["lead"]["company"]
        assert web_res["lead"]["loan_type"] == tg_res["lead"]["loan_type"]
        assert web_res["lead"]["loan_amount"] == tg_res["lead"]["loan_amount"]
        assert web_res["lead"]["tenure_months"] == tg_res["lead"]["tenure_months"]
        assert web_res["next_missing_parameter"] == tg_res["next_missing_parameter"]
        assert web_res["is_complete"] == tg_res["is_complete"]

        # Spoken question / message must be 100% identical
        assert web_res["message"] == tg_res["answer"]

        if turn_idx < 5:
            assert tg_res["is_complete"] is False
            mock_repo.create_lead.assert_not_called()
        else:
            # Turn 6: Completion turn
            assert tg_res["is_complete"] is True
            assert web_res["is_complete"] is True
            assert tg_res["next_missing_parameter"] is None
            assert web_res["next_missing_parameter"] is None
            assert "Thank you, Rajesh Sharma!" in tg_res["answer"]
            assert "All your details have been recorded" in tg_res["answer"]
            # Database persistence must be triggered on Telegram completion
            mock_repo.create_lead.assert_called_once()


# ---------------------------------------------------------------------------
# 2. One-Shot Full Lead Extraction Equivalence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_one_shot_full_lead_equivalence():
    """
    Verifies that when all 6 required fields are provided in a single turn,
    both Web and Telegram immediately mark is_complete=True and output the
    exact same thank-you confirmation without asking redundant questions.
    """
    full_lead = {
        "name": "Anita Roy",
        "phone": "9822012345",
        "company": "TCS",
        "loan_type": "Home Loan",
        "loan_amount": 2500000.0,
        "tenure_months": 240,
    }

    web_ext = LeadExtractorService(api_key="test_key", http_client=MockHTTPClient([full_lead]))
    tg_ext = LeadExtractorService(api_key="test_key", http_client=MockHTTPClient([full_lead]))

    mock_repo = MagicMock(spec=LeadRepository)
    mock_repo.create_lead.return_value = {"id": 202, **full_lead}

    bot = VoiceCopilotBot(
        bot_token="test_token",
        lead_extractor=tg_ext,
        lead_repository=mock_repo,
        client=AsyncMock(),
    )
    bot._synthesize_voice_answer = AsyncMock(return_value=b"AUDIO")
    bot.send_voice = AsyncMock(return_value={"ok": True})
    bot.send_chat_action = AsyncMock(return_value=True)

    input_text = "I am Anita Roy, phone 9822012345, working at TCS, looking for 25 lakh Home Loan for 20 years."

    web_res = web_ext.extract_lead(input_text, language="en")
    tg_res = await bot.process_text_message(chat_id=554433, text=input_text)

    # Identical state
    assert web_res["is_complete"] is True
    assert tg_res["is_complete"] is True
    assert web_res["next_missing_parameter"] is None
    for field in ["name", "phone", "company", "loan_type", "loan_amount", "tenure_months"]:
        assert web_res["lead"][field] == tg_res["lead"][field]
    assert web_res["message"] == tg_res["answer"]
    assert "Thank you, Anita Roy!" in tg_res["answer"]
    mock_repo.create_lead.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Multi-Turn Hindi Lead Flow Equivalence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multi_turn_hindi_equivalence():
    """
    Verifies that in Hindi, the exact same Devanagari questions and completion
    confirmation are generated for Web and Telegram across all 6 required fields.
    """
    llm_turns = [
        {"name": "अमित शर्मा", "phone": None, "company": None, "loan_type": None, "loan_amount": None, "tenure_months": None},
        {"name": "अमित शर्मा", "phone": "9811122233", "company": None, "loan_type": None, "loan_amount": None, "tenure_months": None},
        {"name": "अमित शर्मा", "phone": "9811122233", "company": "टाटा मोटर्स", "loan_type": None, "loan_amount": None, "tenure_months": None},
        {"name": "अमित शर्मा", "phone": "9811122233", "company": "टाटा मोटर्स", "loan_type": "पर्सनल लोन", "loan_amount": None, "tenure_months": None},
        {"name": "अमित शर्मा", "phone": "9811122233", "company": "टाटा मोटर्स", "loan_type": "पर्सनल लोन", "loan_amount": 300000.0, "tenure_months": None},
        {"name": "अमित शर्मा", "phone": "9811122233", "company": "टाटा मोटर्स", "loan_type": "पर्सनल लोन", "loan_amount": 300000.0, "tenure_months": 24},
    ]

    web_ext = LeadExtractorService(api_key="test_key", http_client=MockHTTPClient(list(llm_turns)))
    tg_ext = LeadExtractorService(api_key="test_key", http_client=MockHTTPClient(list(llm_turns)))

    mock_repo = MagicMock(spec=LeadRepository)
    mock_repo.create_lead.side_effect = lambda l: {"id": 303, **l.model_dump()}

    bot = VoiceCopilotBot(
        bot_token="test_token",
        lead_extractor=tg_ext,
        lead_repository=mock_repo,
        client=AsyncMock(),
    )
    bot._synthesize_voice_answer = AsyncMock(return_value=b"AUDIO")
    bot.send_voice = AsyncMock(return_value={"ok": True})
    bot.send_chat_action = AsyncMock(return_value=True)

    hindi_inputs = [
        "मेरा नाम अमित शर्मा है",
        "9811122233",
        "मेरी कंपनी टाटा मोटर्स है",
        "मुझे पर्सनल लोन चाहिए",
        "3 लाख",
        "2 साल",
    ]

    web_existing = None
    chat_id = 771122

    for turn_idx, u_in in enumerate(hindi_inputs):
        web_res = web_ext.extract_lead(u_in, existing_lead=web_existing, language="hi")
        web_existing = web_res["lead"]
        tg_res = await bot.process_text_message(chat_id=chat_id, text=u_in)

        assert web_res["lead"]["name"] == tg_res["lead"]["name"]
        assert web_res["next_missing_parameter"] == tg_res["next_missing_parameter"]
        assert web_res["is_complete"] == tg_res["is_complete"]
        assert web_res["message"] == tg_res["answer"]

        if turn_idx == 5:
            assert tg_res["is_complete"] is True
            assert "धन्यवाद अमित शर्मा जी!" in tg_res["answer"]
            assert "आपकी सभी आवश्यक जानकारी दर्ज कर ली गई है" in tg_res["answer"]
            mock_repo.create_lead.assert_called_once()


# ---------------------------------------------------------------------------
# 4. Multi-Turn Marathi Lead Flow Equivalence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multi_turn_marathi_equivalence():
    """
    Verifies that in Marathi, the exact same Devanagari questions and completion
    confirmation are generated for Web and Telegram across all 6 required fields.
    """
    llm_turns = [
        {"name": "सचिन पाटील", "phone": None, "company": None, "loan_type": None, "loan_amount": None, "tenure_months": None},
        {"name": "सचिन पाटील", "phone": "9822233344", "company": None, "loan_type": None, "loan_amount": None, "tenure_months": None},
        {"name": "सचिन पाटील", "phone": "9822233344", "company": "टीसीएस", "loan_type": None, "loan_amount": None, "tenure_months": None},
        {"name": "सचिन पाटील", "phone": "9822233344", "company": "टीसीएस", "loan_type": "गृह कर्ज", "loan_amount": None, "tenure_months": None},
        {"name": "सचिन पाटील", "phone": "9822233344", "company": "टीसीएस", "loan_type": "गृह कर्ज", "loan_amount": 1500000.0, "tenure_months": None},
        {"name": "सचिन पाटील", "phone": "9822233344", "company": "टीसीएस", "loan_type": "गृह कर्ज", "loan_amount": 1500000.0, "tenure_months": 60},
    ]

    web_ext = LeadExtractorService(api_key="test_key", http_client=MockHTTPClient(list(llm_turns)))
    tg_ext = LeadExtractorService(api_key="test_key", http_client=MockHTTPClient(list(llm_turns)))

    mock_repo = MagicMock(spec=LeadRepository)
    mock_repo.create_lead.side_effect = lambda l: {"id": 404, **l.model_dump()}

    bot = VoiceCopilotBot(
        bot_token="test_token",
        lead_extractor=tg_ext,
        lead_repository=mock_repo,
        client=AsyncMock(),
    )
    bot._synthesize_voice_answer = AsyncMock(return_value=b"AUDIO")
    bot.send_voice = AsyncMock(return_value={"ok": True})
    bot.send_chat_action = AsyncMock(return_value=True)

    marathi_inputs = [
        "माझे नाव सचिन पाटील आहे",
        "9822233344",
        "टीसीएस मध्ये काम करतो",
        "मला गृह कर्ज हवे आहे",
        "15 लाख",
        "5 वर्षे",
    ]

    web_existing = None
    chat_id = 882233

    for turn_idx, u_in in enumerate(marathi_inputs):
        web_res = web_ext.extract_lead(u_in, existing_lead=web_existing, language="mr")
        web_existing = web_res["lead"]
        tg_res = await bot.process_text_message(chat_id=chat_id, text=u_in)

        assert web_res["lead"]["name"] == tg_res["lead"]["name"]
        assert web_res["next_missing_parameter"] == tg_res["next_missing_parameter"]
        assert web_res["is_complete"] == tg_res["is_complete"]
        assert web_res["message"] == tg_res["answer"]

        if turn_idx == 5:
            assert tg_res["is_complete"] is True
            assert "धन्यवाद सचिन पाटील जी!" in tg_res["answer"]
            assert "आपले सर्व आवश्यक तपशील नोंदवले गेले आहेत" in tg_res["answer"]
            mock_repo.create_lead.assert_called_once()


# ---------------------------------------------------------------------------
# 5. Telegram Voice Note vs. Web Equivalence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_telegram_voice_note_equivalence_with_web():
    """
    Verifies that when customer sends a voice note on Telegram,
    it transcribes via STT and produces the EXACT same lead extraction
    and question prompt as the Web Module 1 endpoint.
    """
    spoken_lead = {"name": "Kavita Rao", "phone": None, "loan_type": None, "loan_amount": None}

    web_ext = LeadExtractorService(api_key="test_key", http_client=MockHTTPClient([spoken_lead]))
    tg_ext = LeadExtractorService(api_key="test_key", http_client=MockHTTPClient([spoken_lead]))

    bot = VoiceCopilotBot(
        bot_token="test_token",
        lead_extractor=tg_ext,
        lead_repository=MagicMock(spec=LeadRepository),
        client=AsyncMock(),
    )
    bot.download_file = AsyncMock(return_value=(b"FAKE_AUDIO_BYTES", "ogg"))
    bot._sarvam_stt = MagicMock()
    bot._sarvam_stt.transcribe_audio = AsyncMock(return_value={
        "transcript": "Hello, my name is Kavita Rao",
        "detected_language": "en",
    })
    bot._synthesize_voice_answer = AsyncMock(return_value=b"RESPONSE_AUDIO")
    bot.send_voice = AsyncMock(return_value={"ok": True})
    bot.send_chat_action = AsyncMock(return_value=True)

    # 1. Web
    web_res = web_ext.extract_lead("Hello, my name is Kavita Rao", language="en")

    # 2. Telegram voice note
    tg_res = await bot.process_voice_message(chat_id=112233, voice_data={"file_id": "voice_file_001"})

    assert tg_res["status"] == "success"
    assert tg_res["lead"]["name"] == web_res["lead"]["name"] == "Kavita Rao"
    assert tg_res["next_missing_parameter"] == web_res["next_missing_parameter"] == "phone"
    assert tg_res["is_complete"] == web_res["is_complete"] == False
    assert tg_res["answer"] == web_res["message"]
    # Bot delivers voice response
    bot.send_voice.assert_awaited_once()


# ---------------------------------------------------------------------------
# 6. Greeting Turn Preserves Session State Equivalence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_greeting_preserves_lead_state_equivalence():
    """
    Verifies that when a greeting (e.g. 'Hello', 'Namaste') occurs mid-conversation,
    both Web and Telegram preserve the existing collected details and do not reset.
    """
    turn1_lead = {"name": "Sunil Gupta", "phone": None, "loan_type": None, "loan_amount": None}

    web_ext = LeadExtractorService(api_key="test_key", http_client=MockHTTPClient([turn1_lead]))
    tg_ext = LeadExtractorService(api_key="test_key", http_client=MockHTTPClient([turn1_lead]))

    bot = VoiceCopilotBot(
        bot_token="test_token",
        lead_extractor=tg_ext,
        lead_repository=MagicMock(spec=LeadRepository),
        client=AsyncMock(),
    )
    bot._synthesize_voice_answer = AsyncMock(return_value=b"AUDIO")
    bot.send_voice = AsyncMock(return_value={"ok": True})
    bot.send_chat_action = AsyncMock(return_value=True)

    chat_id = 445566

    # Turn 1: Provide name
    web_res_1 = web_ext.extract_lead("I am Sunil Gupta", language="en")
    tg_res_1 = await bot.process_text_message(chat_id=chat_id, text="I am Sunil Gupta")
    assert web_res_1["lead"]["name"] == tg_res_1["lead"]["name"] == "Sunil Gupta"

    # Turn 2: User says "Hello" (pure greeting)
    web_res_2 = web_ext.extract_lead("Hello", existing_lead=web_res_1["lead"], language="en")
    tg_res_2 = await bot.process_text_message(chat_id=chat_id, text="Hello")

    # Both must preserve Sunil Gupta
    assert web_res_2["lead"]["name"] == tg_res_2["lead"]["name"] == "Sunil Gupta"
    assert web_res_2["next_missing_parameter"] == tg_res_2["next_missing_parameter"] == "phone"
    assert web_res_2["is_complete"] == tg_res_2["is_complete"] == False
    assert web_res_2["message"] == tg_res_2["answer"]
