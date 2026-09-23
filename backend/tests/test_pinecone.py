import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app
from services.vector_store import (
    PineconeService,
    format_chunk_for_pinecone,
    sanitize_vector_id,
    PineconeConfigurationError,
    PineconeUpsertError,
)
from tests.test_pdf_extractor import create_minimal_pdf_bytes


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_pdf(tmp_path):
    pdf_bytes = create_minimal_pdf_bytes([
        "Voice Sales Copilot Enterprise Pricing Battlecard.",
        "Cloud Security SOC2 Compliance and SLA Guarantees.",
    ])
    file_path = tmp_path / "pinecone_test_doc.pdf"
    file_path.write_bytes(pdf_bytes)
    return str(file_path)


def test_sanitize_vector_id():
    """Verify vector ID generation creates clean, deterministic, ASCII-safe identifiers."""
    id1 = sanitize_vector_id("Sales Playbook (v2).pdf", page=1, chunk_index=0)
    assert id1 == "Sales_Playbook__v2_.pdf_p1_c0"

    id_prefixed = sanitize_vector_id("guide.pdf", page=3, chunk_index=7, prefix="tenant_123")
    assert id_prefixed == "tenant_123_guide.pdf_p3_c7"

    # Empty source fallback
    id_empty = sanitize_vector_id("", page=1, chunk_index=2)
    assert id_empty == "doc_p1_c2"


def test_format_chunk_for_pinecone():
    """Verify format_chunk_for_pinecone strictly formats chunk into Pinecone vector record."""
    chunk = {
        "chunk_index": 2,
        "page_number": 4,
        "character_count": 55,
        "text": "Handling price discount pushback during enterprise renewal.",
        "source": "renewal_battlecard.pdf",
        "embedding": [0.05] * 1536,
        "metadata": {"category": "renewal", "author": "RevOps"},
    }

    record = format_chunk_for_pinecone(chunk, doc_id_prefix="org1")

    assert record["id"] == "org1_renewal_battlecard.pdf_p4_c2"
    assert record["values"] == [0.05] * 1536
    assert record["metadata"]["text"] == chunk["text"]
    assert record["metadata"]["source"] == "renewal_battlecard.pdf"
    assert record["metadata"]["page"] == 4
    assert record["metadata"]["chunk_index"] == 2
    assert record["metadata"]["character_count"] == 55
    assert record["metadata"]["category"] == "renewal"
    assert record["metadata"]["author"] == "RevOps"


def test_format_chunk_missing_embedding_raises_value_error():
    """Verify ValueError is raised if chunk embedding is missing or empty."""
    invalid_chunk = {
        "chunk_index": 0,
        "page_number": 1,
        "text": "Text without embedding",
        "source": "doc.pdf",
    }
    with pytest.raises(ValueError, match="missing or invalid 'embedding'"):
        format_chunk_for_pinecone(invalid_chunk)


def test_pinecone_missing_api_key_raises_configuration_error():
    """Verify PineconeConfigurationError is raised when PINECONE_API_KEY is unset."""
    service = PineconeService(api_key="", index_name="voice-sales-copilot")
    with pytest.raises(PineconeConfigurationError, match="PINECONE_API_KEY is not set"):
        service.get_client()


def test_upsert_chunks_empty_list():
    """Verify upserting an empty chunk list returns a success response with count=0."""
    service = PineconeService(api_key="mock-key", index_name="test-index")
    result = service.upsert_chunks([])
    assert result["status"] == "success"
    assert result["upserted_count"] == 0
    assert result["vector_ids"] == []


def test_mocked_pinecone_service_upsert_chunks():
    """Verify PineconeService correctly batches vectors and invokes index.upsert."""
    mock_index = MagicMock()
    mock_client = MagicMock()
    mock_client.Index.return_value = mock_index

    service = PineconeService(
        api_key="mock-api-key",
        index_name="test-sales-index",
        namespace="enterprise-tenant",
        client=mock_client,
        index=mock_index,
    )

    # Prepare 5 dummy chunks
    chunks = [
        {
            "chunk_index": i,
            "page_number": 1,
            "character_count": 25,
            "text": f"Chunk text number {i}",
            "source": "playbook.pdf",
            "embedding": [0.01 * (i + 1)] * 1536,
        }
        for i in range(5)
    ]

    # Upsert with batch_size=2 (should result in ceil(5/2) = 3 batch calls)
    result = service.upsert_chunks(chunks, batch_size=2)

    assert result["status"] == "success"
    assert result["upserted_count"] == 5
    assert result["index_name"] == "test-sales-index"
    assert result["namespace"] == "enterprise-tenant"
    assert len(result["vector_ids"]) == 5

    # Verify 3 batches were sent to mock_index.upsert
    assert mock_index.upsert.call_count == 3
    # Check first batch arguments
    first_call_kwargs = mock_index.upsert.call_args_list[0].kwargs
    assert len(first_call_kwargs["vectors"]) == 2
    assert first_call_kwargs["namespace"] == "enterprise-tenant"
    assert first_call_kwargs["vectors"][0]["id"] == "playbook.pdf_p1_c0"
    assert len(first_call_kwargs["vectors"][0]["values"]) == 1536
    assert first_call_kwargs["vectors"][0]["metadata"]["chunk_index"] == 0


def test_upsert_chunks_auto_embeds_unembedded_chunks():
    """Verify un-embedded chunks automatically receive vector embeddings before Pinecone upsert."""
    mock_index = MagicMock()
    service = PineconeService(
        api_key="mock-api-key",
        index_name="test-index",
        index=mock_index,
    )

    raw_chunks = [
        {
            "chunk_index": 0,
            "page_number": 1,
            "text": "Objection handling pricing value pitch.",
            "source": "guide.pdf",
        }
    ]

    result = service.upsert_chunks(raw_chunks)
    assert result["upserted_count"] == 1
    assert mock_index.upsert.call_count == 1
    upserted_vectors = mock_index.upsert.call_args_list[0].kwargs["vectors"]
    assert len(upserted_vectors[0]["values"]) == 1536


def test_api_upsert_chunks_endpoint_success(client):
    """Integration test for POST /api/upsert-chunks with mocked Pinecone client."""
    mock_index = MagicMock()
    mock_client = MagicMock()
    mock_client.Index.return_value = mock_index

    with patch("services.vector_store.Pinecone", return_value=mock_client):
        payload = {
            "chunks": [
                {
                    "chunk_index": 0,
                    "page_number": 1,
                    "text": "SaaS Cloud Security and SOC2 Type II certification.",
                    "source": "compliance.pdf",
                    "embedding": [0.02] * 1536,
                }
            ],
            "namespace": "test_namespace",
            "index_name": "voice-sales-copilot",
        }

        with patch.dict(os.environ, {"PINECONE_API_KEY": "fake-pinecone-key"}):
            res = client.post("/api/upsert-chunks", json=payload)

        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["upserted_count"] == 1
        assert data["namespace"] == "test_namespace"
        assert len(data["vector_ids"]) == 1
        assert "compliance.pdf_p1_c0" in data["vector_ids"][0]


def test_api_upsert_chunks_endpoint_empty_chunks(client):
    """Verify POST /api/upsert-chunks rejects empty chunk list with 400 Bad Request."""
    res = client.post("/api/upsert-chunks", json={"chunks": []})
    assert res.status_code == 400
    assert "cannot be empty" in res.json()["detail"]


def test_api_upsert_chunks_missing_api_key(client):
    """Verify POST /api/upsert-chunks returns 503 when PINECONE_API_KEY is not configured."""
    with patch.dict(os.environ, {"PINECONE_API_KEY": ""}):
        payload = {
            "chunks": [
                {
                    "chunk_index": 0,
                    "page_number": 1,
                    "text": "Demo chunk text",
                    "source": "demo.pdf",
                    "embedding": [0.01] * 1536,
                }
            ]
        }
        res = client.post("/api/upsert-chunks", json=payload)
        assert res.status_code == 503
        assert "PINECONE_API_KEY is not set" in res.json()["detail"]


def test_api_extract_pdf_with_pinecone_upsert(client, sample_pdf):
    """Integration test: POST /api/extract-pdf?upsert_to_pinecone=true extracts, embeds, and upserts."""
    mock_index = MagicMock()
    mock_client = MagicMock()
    mock_client.Index.return_value = mock_index

    with patch("services.vector_store.Pinecone", return_value=mock_client), \
         patch.dict(os.environ, {"PINECONE_API_KEY": "mock-test-key"}):

        with open(sample_pdf, "rb") as f:
            res = client.post(
                "/api/extract-pdf?upsert_to_pinecone=true&namespace=sales_team",
                files={"file": ("battlecard.pdf", f, "application/pdf")},
            )

        assert res.status_code == 200
        data = res.json()

        # Normal extraction & embedding response fields
        assert data["total_pages"] == 2
        assert data["total_chunks"] >= 2
        assert "embedding_dimension" in data

        # Pinecone upsert summary field
        assert "pinecone_upsert" in data
        upsert_info = data["pinecone_upsert"]
        assert upsert_info["status"] == "success"
        assert upsert_info["upserted_count"] == data["total_chunks"]
        assert upsert_info["namespace"] == "sales_team"
        assert len(upsert_info["vector_ids"]) == data["total_chunks"]

        # Ensure mock index upsert was executed
        assert mock_index.upsert.called


def test_pinecone_service_creates_serverless_index_if_missing():
    """Verify that PineconeService auto-creates serverless index with 1536 dims, cosine metric when missing."""
    from services.vector_store import PineconeService

    mock_client = MagicMock()
    mock_client.has_index.return_value = False
    mock_index = MagicMock()
    mock_client.Index.return_value = mock_index

    service = PineconeService(
        api_key="mock-key",
        index_name="voice-sales-copilot",
        client=mock_client,
        dimension=1536,
        metric="cosine",
    )

    idx = service.get_index()
    assert idx == mock_index
    assert mock_client.create_index.called
    create_call_kwargs = mock_client.create_index.call_args[1]
    assert create_call_kwargs["name"] == "voice-sales-copilot"
    assert create_call_kwargs["dimension"] == 1536
    assert create_call_kwargs["metric"] == "cosine"

