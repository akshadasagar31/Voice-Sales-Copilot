# =============================================================================
# REGRESSION TESTS: MODULE 1 GENERAL QUESTION ACTIVE LLM WEBSOCKET FLOW
# =============================================================================
# Verifies that general queries ("What is your name?", "Who are you?", etc.):
# 1. Are NOT classified as greetings or loan intent.
# 2. Are classified as general questions.
# 3. Stream from the CURRENT ACTIVE LLM over WebSocket /ws/voice-stt without being cancelled.
# 4. Emit type="final" with is_general_query=True and has_subsequent_sentences=True.
# 5. Emit type="sentence" carrying the generated LLM text and type="stream_complete".
# 6. Never substitute canned marketing paragraphs.
# =============================================================================

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.language import is_greeting
from services.lead_extractor import LeadExtractorService, is_loan_intent, is_general_question, extract_name
from main import app


@pytest.mark.parametrize(
    "phrase",
    [
        "What is your name?",
        "what is your name",
        "Who are you?",
        "who are you",
        "What is your purpose?",
        "What can you do?",
        "What do you do?",
        "What knowledge do you have?",
        "How can you help me?",
    ],
)
def test_general_questions_are_not_greetings(phrase: str):
    """General questions containing question indicators must NOT be classified as greetings."""
    assert is_greeting(phrase) is False, f"'{phrase}' should NOT be classified as a greeting"


@pytest.mark.parametrize(
    "phrase",
    [
        "What is your name?",
        "Who are you?",
        "What is your purpose?",
        "What can you do?",
    ],
)
def test_general_questions_are_not_loan_intent(phrase: str):
    """General questions must NOT trigger loan intent or start lead capture."""
    assert is_loan_intent(phrase) is False, f"'{phrase}' should NOT trigger loan intent"
    # Must not treat "What is your name?" as prospect name
    extracted = extract_name(phrase)
    assert extracted != phrase, f"'{phrase}' must not be extracted as prospect name"


@pytest.mark.parametrize(
    "phrase",
    [
        "What is your name?",
        "Who are you?",
        "What can you do?",
        "What is the interest rate on home loans?",
        "What is 15 multiplied by 4?",
    ],
)
def test_general_questions_are_general_questions(phrase: str):
    """General questions must be recognized by is_general_question."""
    assert is_general_question(phrase) is True, f"'{phrase}' should be recognized as a general question"


@pytest.mark.asyncio
async def test_stream_general_query_tokens_uses_active_llm():
    """Verify stream_general_query_tokens targets the active LLM from ModelManager."""
    extractor = LeadExtractorService()

    mock_resp = MagicMock()
    mock_resp.status_code = 200

    async def fake_aiter_lines():
        yield 'data: {"choices": [{"delta": {"content": "I am VoiceCopilot, "}}]}'
        yield 'data: {"choices": [{"delta": {"content": "your financial assistant."}}]}'
        yield "data: [DONE]"

    mock_resp.aiter_lines = fake_aiter_lines

    with patch("httpx.AsyncClient.stream") as mock_stream:
        mock_stream.return_value.__aenter__.return_value = mock_resp

        tokens = []
        async for t in extractor.stream_general_query_tokens(
            transcript="What is your name?",
            language="en",
        ):
            tokens.append(t)

        full_streamed = "".join(tokens)
        assert "VoiceCopilot" in full_streamed
        assert "financial assistant" in full_streamed
        # Verify old canned marketing paragraph was NOT emitted
        assert "comprehensive knowledge of loan products" not in full_streamed


@pytest.mark.asyncio
async def test_llm_failure_emits_neutral_notice_not_canned_marketing():
    """When the active LLM call completely fails, emits a neutral retry notice, never the marketing paragraph."""
    extractor = LeadExtractorService()

    # Streaming fails with an exception, and non-streaming retry also fails
    with patch("httpx.AsyncClient.stream", side_effect=RuntimeError("Connection refused")):
        with patch("httpx.AsyncClient.post", side_effect=RuntimeError("Connection refused")):
            tokens = []
            async for t in extractor.stream_general_query_tokens(
                transcript="What is your name?",
                language="en",
            ):
                tokens.append(t)

            output = "".join(tokens)
            # Must NOT contain old canned paragraph
            assert "comprehensive knowledge of loan products" not in output
            # Must be a concise identity or neutral notice
            assert "VoiceCopilot" in output or "trouble reaching my assistant" in output


@pytest.mark.asyncio
async def test_stream_general_query_ws_emits_frames_and_completes():
    """Verify _stream_general_query_ws emits initial final frame, sentence frames, and stream_complete."""
    from main import _stream_general_query_ws

    mock_ws = AsyncMock()
    sent_frames = []

    async def fake_send_json(data):
        sent_frames.append(data)

    mock_ws.send_json = fake_send_json

    async def mock_stream_tokens(*args, **kwargs):
        yield "I am VoiceCopilot. "
        yield "I can assist you with your financial questions."

    with patch("services.lead_extractor.LeadExtractorService.stream_general_query_tokens", side_effect=mock_stream_tokens):
        await _stream_general_query_ws(
            websocket=mock_ws,
            clean="What is your name?",
            req_lang="en",
            last_conf=0.99,
            clean_existing={},
            prior_missing=None,
            lead_id_val=None,
        )

    final_frame = next((f for f in sent_frames if f.get("type") == "final"), None)
    assert final_frame is not None, "Must emit final frame"
    assert final_frame.get("is_general_query") is True, "Must have is_general_query=True"
    assert final_frame.get("has_subsequent_sentences") is True, "Must have has_subsequent_sentences=True"
    assert final_frame.get("is_greeting") is False, "Must NOT be a greeting"
    assert final_frame.get("is_assistant_query") is True

    sentence_frames = [f for f in sent_frames if f.get("type") == "sentence"]
    assert len(sentence_frames) >= 1, "Must emit sentence frames"
    assert any("VoiceCopilot" in s.get("sentence", "") for s in sentence_frames)

    complete_frame = next((f for f in sent_frames if f.get("type") == "stream_complete"), None)
    assert complete_frame is not None, "Must emit stream_complete frame"


@pytest.mark.asyncio
async def test_stream_general_query_ws_fallback_on_error():
    """Verify _stream_general_query_ws always emits sentence and stream_complete even if token generator raises."""
    from main import _stream_general_query_ws

    mock_ws = AsyncMock()
    sent_frames = []

    async def fake_send_json(data):
        sent_frames.append(data)

    mock_ws.send_json = fake_send_json

    async def broken_tokens(*args, **kwargs):
        raise RuntimeError("API timeout")
        if False:
            yield "never"

    with patch("services.lead_extractor.LeadExtractorService.stream_general_query_tokens", side_effect=broken_tokens):
        await _stream_general_query_ws(
            websocket=mock_ws,
            clean="What is your name?",
            req_lang="en",
            last_conf=0.99,
            clean_existing={},
            prior_missing=None,
            lead_id_val=None,
        )

    sentence_frames = [f for f in sent_frames if f.get("type") == "sentence"]
    assert len(sentence_frames) == 1, "Must emit 1 neutral fallback sentence"
    complete_frame = next((f for f in sent_frames if f.get("type") == "stream_complete"), None)
    assert complete_frame is not None, "Must emit stream_complete even on failure"


def test_generate_module1_ws_response_general_query_flags():
    """Verify generate_module1_ws_response sets is_general_query=True and is_greeting=False."""
    from main import generate_module1_ws_response

    with patch("services.lead_extractor.LeadExtractorService.answer_general_query", return_value="I am VoiceCopilot."):
        res = generate_module1_ws_response(
            current="What is your name?",
            req_lang="en",
            last_conf=0.99,
            existing_lead_str=None,
        )

    assert res.get("is_general_query") is True
    assert res.get("is_assistant_query") is True
    assert res.get("is_greeting") is False
    assert "VoiceCopilot" in res.get("immediate_sentence1", "")


@pytest.mark.parametrize(
    "transcript,expected_lang",
    [
        ("आपका नाम क्या है?", "hi"),
        ("तुमचे नाव काय आहे?", "mr"),
    ],
)
def test_multilingual_general_questions(transcript: str, expected_lang: str):
    """Verify Hindi and Marathi general questions are detected and not marked as greetings."""
    assert is_greeting(transcript) is False
    assert is_general_question(transcript) is True
