import os
import sys
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app
from services.document_repository import DocumentRepository
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


# ---------------------------------------------------------------------------
# Unit Tests: DocumentRepository
# ---------------------------------------------------------------------------

import uuid

def test_document_repository_lifecycle():
    """Verifies table creation, binary PDF storage, metadata update, and retrieval."""
    repo = DocumentRepository()
    repo.ensure_documents_table()

    pdf_content = b"%PDF-1.4 sample binary content for unit test 12345"
    filename = "sales_strategy_playbook.pdf"
    doc_uid = f"doc_test_{uuid.uuid4().hex[:10]}"

    # 1. Create document
    created = repo.create_document(
        filename=filename,
        file_bytes=pdf_content,
        mime_type="application/pdf",
        document_uid=doc_uid,
    )

    assert created["id"] is not None
    assert created["document_uid"] == doc_uid
    assert created["filename"] == filename
    assert created["file_size"] == len(pdf_content)
    assert created["mime_type"] == "application/pdf"
    assert "upload_date" in created

    doc_id = created["id"]

    # 2. Update with extraction and Pinecone vector tracking
    updated = repo.update_document_extraction(
        document_id=doc_id,
        extracted_text="Sample extracted text from pages 1 and 2.",
        total_pages=2,
        total_characters=41,
        total_chunks=3,
        metadata={"embedding_model": "text-embedding-3-small", "dimension": 1536},
        pinecone_namespace="sales_playbooks",
        pinecone_doc_ids=[f"{doc_uid}_chunk_0", f"{doc_uid}_chunk_1", f"{doc_uid}_chunk_2"],
        pinecone_upserted_count=3,
    )

    assert updated["id"] == doc_id
    assert updated["total_pages"] == 2
    assert updated["total_chunks"] == 3
    assert updated["pinecone_namespace"] == "sales_playbooks"
    assert len(updated["pinecone_doc_ids"]) == 3
    assert updated["pinecone_upserted_count"] == 3

    # 3. Retrieve document without binary data
    fetched_meta = repo.get_document(doc_id, include_data=False)
    assert fetched_meta is not None
    assert fetched_meta["id"] == doc_id
    assert "file_data" not in fetched_meta
    assert fetched_meta["extracted_text"] == "Sample extracted text from pages 1 and 2."
    assert fetched_meta["pinecone_doc_ids"] == [f"{doc_uid}_chunk_0", f"{doc_uid}_chunk_1", f"{doc_uid}_chunk_2"]

    # 4. Retrieve document with binary data (BYTEA)
    fetched_full = repo.get_document(doc_id, include_data=True)
    assert fetched_full is not None
    assert fetched_full["file_data"] == pdf_content

    # 5. Retrieve by UID
    fetched_by_uid = repo.get_document_by_uid(doc_uid, include_data=True)
    assert fetched_by_uid is not None
    assert fetched_by_uid["id"] == doc_id
    assert fetched_by_uid["file_data"] == pdf_content

    # 6. List documents
    docs_list = repo.get_documents(limit=10)
    assert len(docs_list) >= 1
    matching = [d for d in docs_list if d["document_uid"] == doc_uid]
    assert len(matching) == 1
    assert matching[0]["filename"] == filename


def test_document_repository_empty_bytes_raises_value_error():
    """Verifies that attempting to create a document with 0 bytes raises ValueError."""
    repo = DocumentRepository()
    with pytest.raises(ValueError, match="File bytes cannot be empty"):
        repo.create_document("empty.pdf", b"")


# ---------------------------------------------------------------------------
# Integration Tests: FastAPI Endpoints
# ---------------------------------------------------------------------------

def test_api_upload_pdf_persists_to_postgresql(client, sample_pdf_bytes):
    """
    Verifies that uploading a PDF via POST /api/extract-pdf:
    1. Stores original binary PDF in PostgreSQL.
    2. Runs extraction, chunking, embedding, and Pinecone upsert.
    3. Updates PostgreSQL with Pinecone vector tracking IDs.
    4. Returns document_id, document_uid, and db_persisted=True.
    """
    mock_index = MagicMock()
    mock_client = MagicMock()
    mock_client.Index.return_value = mock_index

    with patch("services.vector_store.Pinecone", return_value=mock_client), \
         patch.dict(os.environ, {"PINECONE_API_KEY": "mock-api-key"}):

        response = client.post(
            "/api/extract-pdf?upsert_to_pinecone=true&namespace=enterprise_playbooks",
            files={"file": ("q3_sales_guide.pdf", sample_pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 200
    data = response.json()

    # Verify extraction fields
    assert data["filename"] == "q3_sales_guide.pdf"
    assert data["total_pages"] == 2
    assert data["total_chunks"] >= 2
    assert "embedding_dimension" in data

    # Verify Pinecone upsert
    assert "pinecone_upsert" in data
    assert data["pinecone_upsert"]["status"] == "success"
    assert data["pinecone_upsert"]["namespace"] == "enterprise_playbooks"

    # Verify PostgreSQL persistence fields
    assert "document_id" in data
    assert "document_uid" in data
    assert data["db_persisted"] is True
    assert data["file_size"] == len(sample_pdf_bytes)

    doc_id = data["document_id"]
    doc_uid = data["document_uid"]

    # Verify document exists in PostgreSQL with vector IDs
    repo = DocumentRepository()
    db_doc = repo.get_document(doc_id, include_data=True)
    assert db_doc is not None
    assert db_doc["document_uid"] == doc_uid
    assert db_doc["filename"] == "q3_sales_guide.pdf"
    assert db_doc["file_data"] == sample_pdf_bytes
    assert db_doc["total_pages"] == 2
    assert db_doc["pinecone_namespace"] == "enterprise_playbooks"
    assert len(db_doc["pinecone_doc_ids"]) >= 2
    # Verify vector IDs are prefixed with doc_uid for exact tracing
    assert all(doc_uid in vid for vid in db_doc["pinecone_doc_ids"])


def test_api_list_documents_endpoint(client, sample_pdf_bytes):
    """Verifies GET /api/documents returns list of stored documents."""
    response = client.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "count" in data
    assert "documents" in data
    assert isinstance(data["documents"], list)
    if data["count"] > 0:
        first = data["documents"][0]
        assert "id" in first
        assert "document_uid" in first
        assert "filename" in first
        assert "file_size" in first
        assert "file_data" not in first  # Binary data excluded from list view


def test_api_get_document_metadata_endpoint(client, sample_pdf_bytes):
    """Verifies GET /api/documents/{id} returns metadata and Pinecone vector IDs."""
    repo = DocumentRepository()
    doc = repo.create_document("metadata_test.pdf", sample_pdf_bytes, "application/pdf")
    doc_id = doc["id"]

    response = client.get(f"/api/documents/{doc_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == doc_id
    assert data["filename"] == "metadata_test.pdf"
    assert "file_size" in data
    assert "pinecone_doc_ids" in data

    # 404 for non-existent document
    res_404 = client.get("/api/documents/99999999")
    assert res_404.status_code == 404


def test_api_download_document_endpoint(client, sample_pdf_bytes):
    """
    Verifies GET /api/documents/{id}/download streams the exact original binary PDF bytes
    from PostgreSQL with appropriate headers.
    """
    repo = DocumentRepository()
    doc = repo.create_document("downloadable_spec.pdf", sample_pdf_bytes, "application/pdf")
    doc_id = doc["id"]

    response = client.get(f"/api/documents/{doc_id}/download")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert "downloadable_spec.pdf" in response.headers.get("content-disposition", "")
    assert response.content == sample_pdf_bytes

    # 404 for non-existent document
    res_404 = client.get("/api/documents/99999999/download")
    assert res_404.status_code == 404
