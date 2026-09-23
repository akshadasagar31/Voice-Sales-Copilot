import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app
from services.retriever import VectorRetriever
from services.vector_store import PineconeService, PineconeQueryError
from services.embedder import TextEmbedder


@pytest.fixture
def client():
    return TestClient(app)


def test_vector_retriever_generates_embedding_and_queries_pinecone():
    """Verify VectorRetriever generates query vector and standardizes Pinecone matches."""
    mock_index = MagicMock()
    mock_index.query.return_value = {
        "matches": [
            {
                "id": "playbook.pdf_p2_c1",
                "score": 0.92,
                "metadata": {
                    "text": "Enterprise cloud security SOC2 Type II certification and 99.99% SLA.",
                    "source": "playbook.pdf",
                    "page": 2,
                    "chunk_index": 1,
                    "character_count": 68,
                },
            },
            {
                "id": "playbook.pdf_p1_c0",
                "score": 0.81,
                "metadata": {
                    "text": "Overview of sales security guarantees.",
                    "source": "playbook.pdf",
                    "page": 1,
                    "chunk_index": 0,
                    "character_count": 38,
                },
            },
        ]
    }

    service = PineconeService(api_key="fake-key", index_name="voice-sales-copilot", index=mock_index)
    retriever = VectorRetriever(pinecone_service=service)

    response = retriever.retrieve(
        query="What is your SOC2 compliance and SLA guarantee?",
        top_k=2,
        namespace="enterprise_team",
        filter_dict={"source": "playbook.pdf"},
    )

    # Validate high-level response
    assert response["query"] == "What is your SOC2 compliance and SLA guarantee?"
    assert response["top_k"] == 2
    assert response["total_results"] == 2
    assert response["namespace"] == "enterprise_team"
    assert "embedding_model" in response
    assert "embedding_provider" in response

    # Validate Pinecone index.query call args
    mock_index.query.assert_called_once()
    call_kwargs = mock_index.query.call_args.kwargs
    assert call_kwargs["top_k"] == 2
    assert call_kwargs["namespace"] == "enterprise_team"
    assert call_kwargs["filter"] == {"source": "playbook.pdf"}
    assert call_kwargs["include_metadata"] is True
    assert len(call_kwargs["vector"]) == 1536

    # Validate structured results
    results = response["results"]
    assert len(results) == 2
    assert results[0]["id"] == "playbook.pdf_p2_c1"
    assert results[0]["score"] == 0.92
    assert "SOC2 Type II" in results[0]["text"]
    assert results[0]["source"] == "playbook.pdf"
    assert results[0]["page"] == 2
    assert results[0]["chunk_index"] == 1


def test_vector_retriever_empty_query_raises_value_error():
    """Verify VectorRetriever raises ValueError when query is empty or whitespace."""
    retriever = VectorRetriever()
    with pytest.raises(ValueError, match="cannot be empty"):
        retriever.retrieve("")

    with pytest.raises(ValueError, match="cannot be empty"):
        retriever.retrieve("    ")


def test_api_retrieve_endpoint_success(client):
    """Integration test for POST /api/retrieve with mocked Pinecone client."""
    mock_index = MagicMock()
    mock_index.query.return_value = {
        "matches": [
            {
                "id": "sales_guide.pdf_p3_c2",
                "score": 0.88,
                "metadata": {
                    "text": "Volume discount tiers: 15% discount for 100+ seats, 25% for 500+ seats.",
                    "source": "sales_guide.pdf",
                    "page": 3,
                    "chunk_index": 2,
                    "character_count": 76,
                },
            }
        ]
    }

    mock_client = MagicMock()
    mock_client.Index.return_value = mock_index

    with patch("services.vector_store.Pinecone", return_value=mock_client), \
         patch.dict(os.environ, {"PINECONE_API_KEY": "fake-pinecone-key"}):

        payload = {
            "query": "What discount do we offer for 100 enterprise seats?",
            "top_k": 3,
            "namespace": "sales_battlecards",
            "filter": {"source": "sales_guide.pdf"},
        }
        res = client.post("/api/retrieve", json=payload)

    assert res.status_code == 200
    data = res.json()
    assert data["query"] == "What discount do we offer for 100 enterprise seats?"
    assert data["top_k"] == 3
    assert data["total_results"] == 1
    assert data["namespace"] == "sales_battlecards"

    result = data["results"][0]
    assert result["id"] == "sales_guide.pdf_p3_c2"
    assert result["score"] == 0.88
    assert "Volume discount tiers" in result["text"]
    assert result["source"] == "sales_guide.pdf"
    assert result["page"] == 3
    assert result["chunk_index"] == 2


def test_api_retrieve_endpoint_empty_query_validation(client):
    """Verify POST /api/retrieve returns 400 Bad Request when query is empty or whitespace."""
    res_empty = client.post("/api/retrieve", json={"query": ""})
    assert res_empty.status_code == 400

    res_whitespace = client.post("/api/retrieve", json={"query": "   "})
    assert res_whitespace.status_code == 400
    assert "cannot be empty" in res_whitespace.json()["detail"]


def test_api_retrieve_endpoint_missing_api_key(client):
    """Verify POST /api/retrieve returns 503 Service Unavailable when PINECONE_API_KEY is not configured."""
    with patch.dict(os.environ, {"PINECONE_API_KEY": ""}):
        res = client.post("/api/retrieve", json={"query": "How to respond to competitor feature claims?"})
        assert res.status_code == 503
        assert "PINECONE_API_KEY is not set" in res.json()["detail"]


def test_api_retrieve_endpoint_pinecone_query_error(client):
    """Verify POST /api/retrieve returns 502 Bad Gateway when Pinecone query fails."""
    mock_index = MagicMock()
    mock_index.query.side_effect = Exception("Connection timeout to Pinecone gateway")

    mock_client = MagicMock()
    mock_client.Index.return_value = mock_index

    with patch("services.vector_store.Pinecone", return_value=mock_client), \
         patch.dict(os.environ, {"PINECONE_API_KEY": "fake-pinecone-key"}):

        res = client.post("/api/retrieve", json={"query": "Cloud uptime guarantees"})
        assert res.status_code == 502
        assert "Failed to query vectors from Pinecone" in res.json()["detail"]
