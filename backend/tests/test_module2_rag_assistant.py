# ============================================================================
# MODULE 2 REGRESSION TESTS: ASK ASSISTANT / MULTILINGUAL RAG COPILOT
# (backend/tests/test_module2_rag_assistant.py)
# ============================================================================
# Verifies:
# 1. English, Hindi, and Marathi greetings: warm greeting + 2-3 grounded playbook topics.
# 2. Multilingual knowledge base questions: answered strictly from retrieved chunks.
# 3. Unrelated questions (trivia, world facts, math): intercepted and returns exact fallback.
# 4. Insufficient documentation: returns exact localized fallback message.
# 5. Turn-by-turn language isolation: no stale language inheritance across turns.
# 6. Streaming SSE: metadata, tokens, and done events correctly emitted.
# ============================================================================

import json
import pytest
from unittest.mock import MagicMock, patch

from services.rag import (
    RAGService,
    DEFAULT_FALLBACK_MESSAGE,
    generate_grounded_greeting_response,
)
from services.language import (
    FALLBACK_MESSAGES,
    LANG_EN,
    LANG_HI,
    LANG_MR,
)


@pytest.fixture
def mock_retriever():
    """Mock VectorRetriever with representative sales playbook chunks."""
    retriever = MagicMock()
    retriever.retrieve.return_value = {
        "query": "What is the CIBIL score required for a personal loan?",
        "results": [
            {
                "id": "playbook_loan_eligibility.pdf_p1_c0",
                "score": 0.88,
                "text": "Personal Loan Eligibility: Minimum CIBIL score required is 700. Minimum net monthly salary is INR 25,000.",
                "source": "playbook_loan_eligibility.pdf",
                "page": 1,
                "chunk_index": 0,
            },
            {
                "id": "playbook_interest_rates.pdf_p2_c1",
                "score": 0.82,
                "text": "Interest Rates: Personal loans range from 10.5% to 16.0% depending on CIBIL score and employer category.",
                "source": "playbook_interest_rates.pdf",
                "page": 2,
                "chunk_index": 1,
            },
        ],
    }
    return retriever


@pytest.fixture
def mock_http_client():
    """Mock HTTP client returning 200 responses for OpenRouter completions."""
    client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "The minimum CIBIL score required for a personal loan is 700, with interest rates starting at 10.5%."
                }
            }
        ]
    }
    client.post.return_value = mock_resp
    return client


# ----------------------------------------------------------------------------
# 1. Multilingual Greetings Tests (EN, HI, MR)
# ----------------------------------------------------------------------------

def test_greeting_english_returns_warm_greeting_and_grounded_topics(mock_retriever, mock_http_client):
    """Verify English greeting returns natural greeting and grounded topics, NEVER fallback."""
    rag_service = RAGService(
        retriever=mock_retriever,
        api_key="test-api-key",
        http_client=mock_http_client,
    )
    result = rag_service.answer_question("Hello, good morning!")

    assert result["is_greeting"] is True
    assert result["fallback_used"] is False
    assert result["language"] == LANG_EN
    assert "Hello" in result["answer"] or "Welcome" in result["answer"] or "playbooks" in result["answer"]
    # Ensure fallback is never present in greeting response
    assert result["answer"] != FALLBACK_MESSAGES[LANG_EN]
    assert "not contain sufficient information" not in result["answer"].lower()


def test_greeting_hindi_returns_warm_greeting_and_grounded_topics(mock_retriever, mock_http_client):
    """Verify Hindi greeting returns natural greeting and grounded topics in Hindi."""
    rag_service = RAGService(
        retriever=mock_retriever,
        api_key="test-api-key",
        http_client=mock_http_client,
    )
    result = rag_service.answer_question("नमस्ते")

    assert result["is_greeting"] is True
    assert result["fallback_used"] is False
    assert result["language"] == LANG_HI
    assert "नमस्ते" in result["answer"] or "स्वागत" in result["answer"]
    assert "पर्याप्त जानकारी उपलब्ध नहीं है" not in result["answer"]


def test_greeting_marathi_returns_warm_greeting_and_grounded_topics(mock_retriever, mock_http_client):
    """Verify Marathi greeting returns natural greeting and grounded topics in Marathi."""
    rag_service = RAGService(
        retriever=mock_retriever,
        api_key="test-api-key",
        http_client=mock_http_client,
    )
    result = rag_service.answer_question("नमस्कार")

    assert result["is_greeting"] is True
    assert result["fallback_used"] is False
    assert result["language"] == LANG_MR
    assert "नमस्कार" in result["answer"] or "स्वागत" in result["answer"]
    assert "पुरेशी माहिती उपलब्ध नाही" not in result["answer"]


def test_generate_grounded_greeting_response_direct_helper():
    """Verify generate_grounded_greeting_response produces grounded topics for EN, HI, MR."""
    chunks = [{"source": "playbook.pdf", "text": "Interest rates and eligibility"}]
    
    en_resp = generate_grounded_greeting_response(LANG_EN, chunks)
    assert "Welcome to Voice Sales Copilot" in en_resp
    assert "Interest rates" in en_resp

    hi_resp = generate_grounded_greeting_response(LANG_HI, chunks)
    assert "वॉयस सेल्स कोपायलट में आपका स्वागत है" in hi_resp
    assert "ब्याज दरें" in hi_resp

    mr_resp = generate_grounded_greeting_response(LANG_MR, chunks)
    assert "व्हॉइस सेल्स कोपायलटमध्ये आपले स्वागत आहे" in mr_resp
    assert "व्याजदर" in mr_resp


# ----------------------------------------------------------------------------
# 2. Multilingual Knowledge Base Question Answering Tests
# ----------------------------------------------------------------------------

def test_kb_question_english_answers_from_context_chunks(mock_retriever, mock_http_client):
    """Verify English KB question answers strictly from context chunks."""
    rag_service = RAGService(
        retriever=mock_retriever,
        api_key="test-api-key",
        http_client=mock_http_client,
    )
    result = rag_service.answer_question("What is the CIBIL score required for a personal loan?")

    assert result["is_greeting"] is False
    assert result["fallback_used"] is False
    assert result["language"] == LANG_EN
    assert "700" in result["answer"]
    assert len(result["context_used"]) == 2
    assert "playbook_loan_eligibility.pdf" in result["sources"]


def test_kb_question_hindi_answers_from_context_chunks(mock_retriever):
    """Verify Hindi KB question answers in Hindi grounded in retrieved chunks."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "पर्सनल लोन के लिए न्यूनतम 700 सिबिल (CIBIL) स्कोर आवश्यक है।"
                }
            }
        ]
    }
    mock_client.post.return_value = mock_resp

    rag_service = RAGService(
        retriever=mock_retriever,
        api_key="test-api-key",
        http_client=mock_client,
    )
    result = rag_service.answer_question("पर्सनल लोन के लिए सिबिल स्कोर की क्या आवश्यकता है?")

    assert result["is_greeting"] is False
    assert result["fallback_used"] is False
    assert result["language"] == LANG_HI
    assert "700" in result["answer"]
    assert "सिबिल" in result["answer"]


def test_kb_question_marathi_answers_from_context_chunks(mock_retriever):
    """Verify Marathi KB question answers in Marathi grounded in retrieved chunks."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "वैयक्तिक कर्जासाठी किमान 700 सिबिल (CIBIL) स्कोअर असणे आवश्यक आहे."
                }
            }
        ]
    }
    mock_client.post.return_value = mock_resp

    rag_service = RAGService(
        retriever=mock_retriever,
        api_key="test-api-key",
        http_client=mock_client,
    )
    result = rag_service.answer_question("पर्सनल लोनसाठी सिबिल स्कोअर किती असावा?")

    assert result["is_greeting"] is False
    assert result["fallback_used"] is False
    assert result["language"] == LANG_MR
    assert "700" in result["answer"]
    assert "सिबिल" in result["answer"]


# ----------------------------------------------------------------------------
# 3. Unrelated Questions Rejection Tests (Dual-Layer Guardrail)
# ----------------------------------------------------------------------------

def test_unrelated_question_english_returns_exact_fallback():
    """
    Verify unrelated question (e.g. geography trivia) produces low vector similarity (<0.32)
    and returns exact English fallback without calling the LLM.
    """
    mock_low_retriever = MagicMock()
    mock_low_retriever.retrieve.return_value = {
        "query": "What is the capital of France?",
        "results": [
            {
                "id": "chunk_low",
                "score": 0.22,  # Below MIN_RELEVANCE_THRESHOLD (0.32)
                "text": "Personal loan documentation guidelines.",
                "source": "playbook.pdf",
            }
        ],
    }
    mock_http_client = MagicMock()

    rag_service = RAGService(
        retriever=mock_low_retriever,
        api_key="test-api-key",
        http_client=mock_http_client,
    )
    result = rag_service.answer_question("What is the capital of France?")

    assert result["fallback_used"] is True
    assert result["language"] == LANG_EN
    assert result["answer"] == FALLBACK_MESSAGES[LANG_EN]
    # Verify OpenRouter LLM was NOT called due to Layer 1 relevance cutoff
    mock_http_client.post.assert_not_called()


def test_unrelated_question_hindi_returns_exact_fallback():
    """Verify Hindi unrelated question returns exact Hindi fallback."""
    mock_low_retriever = MagicMock()
    mock_low_retriever.retrieve.return_value = {
        "query": "फ्रांस की राजधानी क्या है?",
        "results": [
            {
                "id": "chunk_low",
                "score": 0.18,
                "text": "Personal loan interest rates and processing fee.",
                "source": "playbook.pdf",
            }
        ],
    }
    mock_http_client = MagicMock()

    rag_service = RAGService(
        retriever=mock_low_retriever,
        api_key="test-api-key",
        http_client=mock_http_client,
    )
    result = rag_service.answer_question("फ्रांस की राजधानी क्या है?")

    assert result["fallback_used"] is True
    assert result["language"] == LANG_HI
    assert result["answer"] == FALLBACK_MESSAGES[LANG_HI]
    mock_http_client.post.assert_not_called()


def test_unrelated_question_marathi_returns_exact_fallback():
    """Verify Marathi unrelated question returns exact Marathi fallback."""
    mock_low_retriever = MagicMock()
    mock_low_retriever.retrieve.return_value = {
        "query": "चंद्रावर जाणारा पहिला माणूस कोण?",
        "results": [
            {
                "id": "chunk_low",
                "score": 0.15,
                "text": "Personal loan interest rates and processing fee.",
                "source": "playbook.pdf",
            }
        ],
    }
    mock_http_client = MagicMock()

    rag_service = RAGService(
        retriever=mock_low_retriever,
        api_key="test-api-key",
        http_client=mock_http_client,
    )
    result = rag_service.answer_question("चंद्रावर जाणारा पहिला माणूस कोण?")

    assert result["fallback_used"] is True
    assert result["language"] == LANG_MR
    assert result["answer"] == FALLBACK_MESSAGES[LANG_MR]
    mock_http_client.post.assert_not_called()


# ----------------------------------------------------------------------------
# 4. Insufficient Documentation Handling Tests
# ----------------------------------------------------------------------------

def test_insufficient_documentation_normalizes_to_exact_fallback():
    """
    Verify that if chunks are retrieved (score >= 0.32) but the LLM indicates
    the documentation does not contain sufficient information, it normalizes
    to the exact standard fallback string.
    """
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = {
        "query": "What is the policy for international currency loans?",
        "results": [
            {
                "id": "chunk_1",
                "score": 0.45,
                "text": "Domestic INR personal loans are offered to resident Indian citizens.",
                "source": "playbook.pdf",
            }
        ],
    }

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "The provided documentation does not contain information about international currency loans."
                }
            }
        ]
    }
    mock_client.post.return_value = mock_resp

    rag_service = RAGService(
        retriever=mock_retriever,
        api_key="test-api-key",
        http_client=mock_client,
    )
    result = rag_service.answer_question("What is the policy for international currency loans?")

    assert result["fallback_used"] is True
    assert result["language"] == LANG_EN
    assert result["answer"] == FALLBACK_MESSAGES[LANG_EN]


# ----------------------------------------------------------------------------
# 5. Turn-by-Turn Language Isolation Tests
# ----------------------------------------------------------------------------

def test_turn_by_turn_language_isolation_no_stale_inheritance(mock_retriever):
    """
    Verify that each turn strictly determines its language from the current user speech:
    Turn 1: Hindi query -> language = 'hi'
    Turn 2: English query with language='hi' passed from previous turn -> language = 'en'
    Turn 3: Marathi query with language='en' passed from previous turn -> language = 'mr'
    """
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Grounded answer text."
                }
            }
        ]
    }
    mock_client.post.return_value = mock_resp

    rag_service = RAGService(
        retriever=mock_retriever,
        api_key="test-api-key",
        http_client=mock_client,
    )

    # Turn 1: Hindi
    res1 = rag_service.answer_question(
        question="नमस्ते, पर्सनल लोन के लिए क्या ब्याज दर है?",
        language="auto",
    )
    assert res1["language"] == LANG_HI

    # Turn 2: English question (previous turn language='hi' passed)
    res2 = rag_service.answer_question(
        question="What is the minimum CIBIL score required?",
        language="hi",  # Stale previous turn language
    )
    assert res2["language"] == LANG_EN, "English query must NOT inherit Hindi from previous turn!"

    # Turn 3: Marathi question (previous turn language='en' passed)
    res3 = rag_service.answer_question(
        question="पर्सनल लोनसाठी आवश्यक कागदपत्रे कोणती आहेत?",
        language="en",  # Stale previous turn language
    )
    assert res3["language"] == LANG_MR, "Marathi query must NOT inherit English from previous turn!"


# ----------------------------------------------------------------------------
# 6. Streaming SSE Mode Tests
# ----------------------------------------------------------------------------

def test_stream_answer_chunks_unrelated_query_yields_fallback():
    """Verify stream_answer_chunks immediately streams localized fallback for unrelated queries."""
    mock_low_retriever = MagicMock()
    mock_low_retriever.retrieve.return_value = {
        "query": "What is the capital of Germany?",
        "results": [
            {
                "id": "c1",
                "score": 0.20,
                "text": "Personal loan tenure options.",
                "source": "playbook.pdf",
            }
        ],
    }

    rag_service = RAGService(
        retriever=mock_low_retriever,
        api_key="test-api-key",
    )

    events = list(rag_service.stream_answer_chunks(
        question="What is the capital of Germany?",
        language="en",
    ))

    # Should have metadata event, token event with fallback, and done event
    assert any("event: metadata" in e for e in events)
    assert any("event: token" in e for e in events)
    assert any("event: done" in e for e in events)

    # Check done event content
    done_line = [e for e in events if "event: done" in e][0]
    data_str = done_line.split("data: ")[1].strip()
    done_payload = json.loads(data_str)
    assert done_payload["fallback_used"] is True
    assert done_payload["answer"] == FALLBACK_MESSAGES[LANG_EN]
    assert done_payload["language"] == LANG_EN
