import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app
from services.rag import (
    RAGService,
    format_context,
    DEFAULT_FALLBACK_MESSAGE,
    OpenRouterConfigurationError,
    OpenRouterAPIError,
)
from services.retriever import VectorRetriever


@pytest.fixture
def client():
    return TestClient(app)


def test_format_context():
    """Verify format_context structures document chunks with source and page annotations."""
    chunks = [
        {
            "source": "sales_guide.pdf",
            "page": 1,
            "text": "Chapter 1: Enterprise sales qualification criteria.",
        },
        {
            "source": "pricing.pdf",
            "page": 4,
            "text": "Volume discount tiers for enterprise contracts.",
        },
    ]
    formatted = format_context(chunks)
    assert "--- Document: sales_guide.pdf | Page: 1 [Segment 1] ---" in formatted
    assert "Chapter 1: Enterprise sales qualification criteria." in formatted
    assert "--- Document: pricing.pdf | Page: 4 [Segment 2] ---" in formatted
    assert "Volume discount tiers for enterprise contracts." in formatted


def test_rag_service_generates_grounded_answer():
    """Verify RAGService queries retriever, formats prompt, and parses OpenRouter LLM response."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = {
        "query": "What is the discount policy?",
        "results": [
            {
                "id": "pricing.pdf_p2_c0",
                "score": 0.94,
                "text": "Enterprise discount: 15% discount for 100+ seats, 25% for 500+ seats.",
                "source": "pricing.pdf",
                "page": 2,
                "chunk_index": 0,
            }
        ],
    }

    mock_http_response = MagicMock()
    mock_http_response.status_code = 200
    mock_http_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "For enterprise customers, we provide a 15% discount for 100+ seats and 25% for 500+ seats."
                }
            }
        ]
    }
    mock_http_client = MagicMock()
    mock_http_client.post.return_value = mock_http_response

    rag_service = RAGService(
        retriever=mock_retriever,
        api_key="fake-openrouter-key",
        model="deepseek/deepseek-flash-latest",
        http_client=mock_http_client,
    )

    result = rag_service.answer_question("What is the discount policy?", top_k=1)

    assert result["question"] == "What is the discount policy?"
    assert "15% discount for 100+ seats" in result["answer"]
    assert result["fallback_used"] is False
    assert result["sources"] == ["pricing.pdf"]
    assert len(result["context_used"]) == 1
    assert result["model"] == "deepseek/deepseek-flash-latest"

    # Verify OpenRouter call payload
    mock_http_client.post.assert_called_once()
    call_args = mock_http_client.post.call_args
    url = call_args[0][0]
    payload = call_args[1]["json"]
    headers = call_args[1]["headers"]

    assert "openrouter.ai/api/v1/chat/completions" in url
    assert headers["Authorization"] == "Bearer fake-openrouter-key"
    assert payload["model"] == "deepseek/deepseek-flash-latest"
    assert len(payload["messages"]) == 2
    assert "CRITICAL RULES" in payload["messages"][0]["content"]
    assert "pricing.pdf" in payload["messages"][1]["content"]


def test_rag_service_zero_chunks_returns_fallback_without_calling_llm():
    """Verify that when 0 chunks are retrieved, fallback is returned immediately without calling OpenRouter."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = {"results": []}

    mock_http_client = MagicMock()
    rag_service = RAGService(
        retriever=mock_retriever,
        api_key="fake-key",
        http_client=mock_http_client,
    )

    result = rag_service.answer_question("What is our corporate vacation policy?")

    assert result["answer"] == DEFAULT_FALLBACK_MESSAGE
    assert result["fallback_used"] is True
    assert result["context_used"] == []
    assert result["sources"] == []
    # Assert LLM was never contacted
    mock_http_client.post.assert_not_called()


def test_rag_service_llm_indicates_unavailable_information():
    """Verify fallback is returned when LLM responds that information is not in context."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = {
        "results": [
            {
                "id": "playbook.pdf_p1_c0",
                "score": 0.51,
                "text": "Sales process and pipeline management guidelines.",
                "source": "playbook.pdf",
                "page": 1,
                "chunk_index": 0,
            }
        ]
    }

    mock_http_response = MagicMock()
    mock_http_response.status_code = 200
    mock_http_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": DEFAULT_FALLBACK_MESSAGE
                }
            }
        ]
    }
    mock_http_client = MagicMock()
    mock_http_client.post.return_value = mock_http_response

    rag_service = RAGService(
        retriever=mock_retriever,
        api_key="fake-key",
        http_client=mock_http_client,
    )

    result = rag_service.answer_question("What are the hardware server specs?")

    assert result["answer"] == DEFAULT_FALLBACK_MESSAGE
    assert result["fallback_used"] is True


def test_rag_service_empty_question_raises_value_error():
    """Verify ValueError is raised if question is empty or blank whitespace."""
    rag_service = RAGService()
    with pytest.raises(ValueError, match="cannot be empty"):
        rag_service.answer_question("")

    with pytest.raises(ValueError, match="cannot be empty"):
        rag_service.answer_question("    ")


def test_rag_service_missing_api_key_raises_configuration_error():
    """Verify OpenRouterConfigurationError is raised when API key is missing and context exists."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = {
        "results": [{"text": "pricing info", "source": "pricing.pdf", "page": 1}]
    }

    rag_service = RAGService(retriever=mock_retriever, api_key="")
    with pytest.raises(OpenRouterConfigurationError, match="OPENROUTER_API_KEY is not set"):
        rag_service.answer_question("What is the price?")


def test_rag_service_openrouter_api_error():
    """Verify OpenRouterAPIError is raised when OpenRouter returns non-200 HTTP code."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = {
        "results": [{"text": "some info", "source": "info.pdf", "page": 1}]
    }

    mock_http_response = MagicMock()
    mock_http_response.status_code = 500
    mock_http_response.text = "Internal Server Error"

    mock_http_client = MagicMock()
    mock_http_client.post.return_value = mock_http_response

    rag_service = RAGService(
        retriever=mock_retriever,
        api_key="fake-key",
        http_client=mock_http_client,
    )

    with pytest.raises(OpenRouterAPIError, match="returned error HTTP 500"):
        rag_service.answer_question("Explain our SLA.")


def test_api_ask_endpoint_success(client):
    """Integration test for POST /api/ask with mocked Pinecone and OpenRouter."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = {
        "query": "How do we handle competitor FUD regarding uptime?",
        "results": [
            {
                "id": "playbook.pdf_p2_c0",
                "score": 0.91,
                "text": "Competitor battlecard: Emphasize our 99.99% multi-region SLA backed by financial penalties.",
                "source": "playbook.pdf",
                "page": 2,
                "chunk_index": 0,
            }
        ],
    }

    mock_openrouter_response = MagicMock()
    mock_openrouter_response.status_code = 200
    mock_openrouter_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Highlight our 99.99% multi-region uptime SLA with financial penalties (playbook.pdf, Page 2)."
                }
            }
        ]
    }

    with patch("services.retriever.VectorRetriever", return_value=mock_retriever), \
         patch("httpx.Client.post", return_value=mock_openrouter_response), \
         patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake-openrouter-key"}):

        payload = {
            "question": "How do we handle competitor FUD regarding uptime?",
            "top_k": 3,
            "namespace": "sales_team",
        }
        res = client.post("/api/ask", json=payload)

    assert res.status_code == 200
    data = res.json()
    assert data["question"] == "How do we handle competitor FUD regarding uptime?"
    assert "99.99% multi-region uptime SLA" in data["answer"]
    assert data["fallback_used"] is False
    assert data["sources"] == ["playbook.pdf"]
    assert len(data["context_used"]) == 1


def test_api_ask_endpoint_streaming_success(client):
    """Integration test for POST /api/ask with stream=True returning SSE stream."""
    import httpx

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = {
        "query": "What is our discount?",
        "results": [
            {
                "id": "pricing.pdf_p1_c0",
                "score": 0.95,
                "text": "15% discount for 100+ licenses.",
                "source": "pricing.pdf",
                "page": 1,
            }
        ],
    }

    mock_openrouter_response = httpx.Response(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        content=(
            b'data: {"choices": [{"delta": {"content": "We offer "}}]}\n'
            b'data: {"choices": [{"delta": {"content": "a 15% discount."}}]}\n'
            b"data: [DONE]\n"
        ),
    )

    with patch("services.retriever.VectorRetriever", return_value=mock_retriever), \
         patch("httpx.Client.post", return_value=mock_openrouter_response), \
         patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake-openrouter-key"}):

        payload = {
            "question": "What is our discount?",
            "stream": True,
        }
        res = client.post("/api/ask", json=payload)

    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]
    body = res.text
    assert "event: metadata" in body
    assert "pricing.pdf" in body
    assert "event: token" in body
    assert "We offer " in body
    assert "event: done" in body


def test_api_ask_endpoint_empty_question_validation(client):
    """Verify POST /api/ask rejects blank questions with HTTP 400 Bad Request."""
    res_empty = client.post("/api/ask", json={"question": ""})
    assert res_empty.status_code == 400

    res_whitespace = client.post("/api/ask", json={"question": "   "})
    assert res_whitespace.status_code == 400
    assert "cannot be empty" in res_whitespace.json()["detail"]


def test_api_ask_endpoint_missing_api_key(client):
    """Verify POST /api/ask returns 503 when OPENROUTER_API_KEY is missing and context exists."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = {
        "results": [{"text": "playbook info", "source": "playbook.pdf", "page": 1}]
    }

    with patch("services.retriever.VectorRetriever", return_value=mock_retriever), \
         patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):

        res = client.post("/api/ask", json={"question": "What is our pricing?"})
        assert res.status_code == 503
        assert "OPENROUTER_API_KEY is not set" in res.json()["detail"]


def test_api_ask_endpoint_upstream_error(client):
    """Verify POST /api/ask returns 502 when OpenRouter LLM call fails."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = {
        "results": [{"text": "playbook info", "source": "playbook.pdf", "page": 1}]
    }

    mock_openrouter_response = MagicMock()
    mock_openrouter_response.status_code = 503
    mock_openrouter_response.text = "Service Unavailable"

    with patch("services.retriever.VectorRetriever", return_value=mock_retriever), \
         patch("httpx.Client.post", return_value=mock_openrouter_response), \
         patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake-key"}):

        res = client.post("/api/ask", json={"question": "What is our SLA?"})
        assert res.status_code == 502
        assert "OpenRouter API returned error HTTP 503" in res.json()["detail"]


def test_rag_service_streaming_sse_chunks():
    """Verify RAGService sends stream=True, processes SSE chunks line-by-line, and returns final answer."""
    import httpx

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = {
        "results": [
            {
                "id": "playbook.pdf_p1_c0",
                "score": 0.95,
                "text": "Our enterprise contract includes 99.99% uptime guarantee.",
                "source": "playbook.pdf",
                "page": 1,
            }
        ]
    }

    sse_lines = [
        b'data: {"choices": [{"delta": {"content": "Our "}}]}\n',
        b'data: {"choices": [{"delta": {"content": "enterprise "}}]}\n',
        b'data: {"choices": [{"delta": {"content": "uptime "}}]}\n',
        b'data: {"choices": [{"delta": {"content": "is 99.99%."}}]}\n',
        b"data: [DONE]\n",
    ]

    mock_response = httpx.Response(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        content=b"".join(sse_lines),
    )

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    rag_service = RAGService(
        retriever=mock_retriever,
        api_key="fake-key",
        http_client=mock_client,
    )

    chunks_emitted = []
    def chunk_collector(tok):
        chunks_emitted.append(tok)

    result = rag_service.answer_question(
        "What is our enterprise uptime?",
        on_chunk=chunk_collector,
    )

    assert result["answer"] == "Our enterprise uptime is 99.99%."
    assert result["fallback_used"] is False
    assert result["sources"] == ["playbook.pdf"]
    assert chunks_emitted == ["Our ", "enterprise ", "uptime ", "is 99.99%."]

    # Verify stream=True was sent in payload
    mock_client.post.assert_called_once()
    payload = mock_client.post.call_args[1]["json"]
    assert payload["stream"] is True


def test_rag_service_streaming_error_in_stream():
    """Verify OpenRouterAPIError is raised if error object is encountered inside SSE stream."""
    import httpx

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = {
        "results": [{"text": "doc text", "source": "doc.pdf", "page": 1}]
    }

    sse_lines = [
        b'data: {"error": {"message": "Model is overloaded, please retry"}}\n',
    ]

    mock_response = httpx.Response(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        content=b"".join(sse_lines),
    )

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    rag_service = RAGService(
        retriever=mock_retriever,
        api_key="fake-key",
        http_client=mock_client,
    )

    with pytest.raises(OpenRouterAPIError, match="Model is overloaded"):
        rag_service.answer_question("What is our policy?")


def test_rag_service_streaming_timeout_error():
    """Verify OpenRouterAPIError is raised on request timeout."""
    import httpx

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = {
        "results": [{"text": "doc text", "source": "doc.pdf", "page": 1}]
    }

    mock_client = MagicMock()
    mock_client.post.side_effect = httpx.ReadTimeout("Streaming connection read timed out")

    rag_service = RAGService(
        retriever=mock_retriever,
        api_key="fake-key",
        http_client=mock_client,
    )

    with pytest.raises(OpenRouterAPIError, match="timed out"):
        rag_service.answer_question("What is our SLA?")


def test_rag_service_models_parameter_present():
    """Verify that models parameter is always included in the OpenRouter payload for automatic failover."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = {
        "results": [{"text": "SLA is 99.9%", "source": "sla.pdf", "page": 1}]
    }

    mock_client = MagicMock()
    mock_client.post.return_value = MagicMock(
        status_code=200,
        text='data: {"choices": [{"delta": {"content": "SLA is 99.9%"}}]}\n\ndata: [DONE]\n\n',
        headers={"content-type": "text/event-stream"},
        iter_lines=MagicMock(return_value=[
            b'data: {"choices": [{"delta": {"content": "SLA is 99.9%"}}]}\n',
            b'data: [DONE]\n'
        ])
    )

    rag_service = RAGService(
        retriever=mock_retriever,
        api_key="fake-key",
        model="deepseek/deepseek-chat",
        fallback_model="meta-llama/llama-3.3-70b-instruct",
        http_client=mock_client,
    )

    rag_service.answer_question("What is our SLA?")
    payload = mock_client.post.call_args[1]["json"]
    assert payload["model"] == "deepseek/deepseek-chat"
    assert payload["models"] == ["deepseek/deepseek-chat", "meta-llama/llama-3.3-70b-instruct"]


def test_rag_service_429_automatic_failover():
    """Verify that when primary model returns HTTP 429, RAGService fails over immediately to fallback model."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = {
        "results": [{"text": "Loan approval requires CIBIL > 730.", "source": "policy.pdf", "page": 2}]
    }

    resp_429 = MagicMock(status_code=429, text='{"error": {"message": "Rate limit exceeded", "code": 429}}')
    resp_200 = MagicMock(
        status_code=200,
        text='data: {"choices": [{"delta": {"content": "CIBIL > 730 required."}}]}\n\ndata: [DONE]\n\n',
        headers={"content-type": "text/event-stream"},
        iter_lines=MagicMock(return_value=[
            b'data: {"choices": [{"delta": {"content": "CIBIL > 730 required."}}]}\n',
            b'data: [DONE]\n'
        ])
    )

    mock_client = MagicMock()
    mock_client.post.side_effect = [resp_429, resp_200]

    rag_service = RAGService(
        retriever=mock_retriever,
        api_key="fake-key",
        model="deepseek/deepseek-chat",
        fallback_model="meta-llama/llama-3.3-70b-instruct",
        http_client=mock_client,
    )

    res = rag_service.answer_question("What is the CIBIL requirement?")
    assert "CIBIL > 730 required." in res["answer"]
    assert mock_client.post.call_count == 2
    # Verify second call used fallback model
    second_payload = mock_client.post.call_args_list[1][1]["json"]
    assert second_payload["model"] == "meta-llama/llama-3.3-70b-instruct"


