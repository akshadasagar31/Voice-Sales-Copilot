import os
import sys
import tempfile
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app
from tests.test_pdf_extractor import create_minimal_pdf_bytes


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_pdf_bytes():
    return create_minimal_pdf_bytes([
        "Acme Cloud Voice Sales Objection Playbook: High Price objection handling tactics.",
        "Competitor Matrix: Acme Cloud provides 99.99% uptime and sub-second voice intelligence.",
    ])


def test_upload_pdf_with_pinecone_upsert_success(client, sample_pdf_bytes):
    """Verifies uploading a PDF with upsert_to_pinecone=true successfully extracts, chunks, embeds, and upserts."""
    mock_index = MagicMock()
    mock_client = MagicMock()
    mock_client.Index.return_value = mock_index

    with patch("services.vector_store.Pinecone", return_value=mock_client), \
         patch.dict(os.environ, {"PINECONE_API_KEY": "mock-api-key"}):

        response = client.post(
            "/api/extract-pdf?upsert_to_pinecone=true&namespace=sales_knowledge",
            files={"file": ("sales_playbook.pdf", sample_pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "sales_playbook.pdf"
    assert data["total_pages"] == 2
    assert data["total_chunks"] >= 2
    assert "embedding_dimension" in data
    assert "pinecone_upsert" in data
    assert data["pinecone_upsert"]["status"] == "success"
    assert data["pinecone_upsert"]["upserted_count"] == data["total_chunks"]
    assert data["pinecone_upsert"]["namespace"] == "sales_knowledge"
    assert mock_index.upsert.called


def test_upload_pdf_missing_pinecone_key_returns_503(client, sample_pdf_bytes):
    """Verifies that attempting to upsert without PINECONE_API_KEY returns HTTP 503 Service Unavailable."""
    with patch.dict(os.environ, {"PINECONE_API_KEY": ""}, clear=False):
        response = client.post(
            "/api/extract-pdf?upsert_to_pinecone=true",
            files={"file": ("playbook.pdf", sample_pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 503
    assert "PINECONE_API_KEY is not set" in response.json()["detail"]


def test_upload_pdf_without_pinecone_upsert(client, sample_pdf_bytes):
    """Verifies that uploading with upsert_to_pinecone=false processes extraction and embeddings without Pinecone."""
    response = client.post(
        "/api/extract-pdf?upsert_to_pinecone=false",
        files={"file": ("guide.pdf", sample_pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total_pages"] == 2
    assert data["total_chunks"] >= 2
    assert "pinecone_upsert" not in data
    assert "chunks" in data
    assert len(data["chunks"]) == data["total_chunks"]


def test_upload_pdf_invalid_extension(client):
    """Verifies rejection of non-PDF files."""
    response = client.post(
        "/api/extract-pdf",
        files={"file": ("report.docx", b"dummy content", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Only .pdf files are supported" in response.json()["detail"]


def test_upload_pdf_empty_file(client):
    """Verifies rejection of empty (0 bytes) PDF file."""
    response = client.post(
        "/api/extract-pdf",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400
    assert "Uploaded file is empty" in response.json()["detail"]
