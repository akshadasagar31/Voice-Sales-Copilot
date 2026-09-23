import os
import sys
import math
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app
from services.embedder import (
    TextEmbedder,
    embed_chunks,
    generate_local_fast_embedding,
    DEFAULT_DIMENSION,
    DEFAULT_MODEL,
)
from services.pdf_extractor import extract_text_from_pdf
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
    file_path = tmp_path / "embed_test_doc.pdf"
    file_path.write_bytes(pdf_bytes)
    return str(file_path)


def test_generate_local_fast_embedding_properties():
    """Verify dimension, L2 normalization, determinism, and differentiation."""
    dim = 1536
    text1 = "Enterprise SaaS software pricing and SLA negotiation."
    text2 = "Security compliance frameworks including ISO 27001 and SOC2."

    vec1 = generate_local_fast_embedding(text1, dimension=dim)
    vec1_repeat = generate_local_fast_embedding(text1, dimension=dim)
    vec2 = generate_local_fast_embedding(text2, dimension=dim)

    # Dimension check
    assert len(vec1) == dim
    assert len(vec2) == dim

    # Type check: floats
    assert all(isinstance(x, float) for x in vec1)

    # Determinism: identical input yields identical vector
    assert vec1 == vec1_repeat

    # Differentiation: different inputs yield different vectors
    assert vec1 != vec2

    # L2 Unit Normalization (sum of squares is ~1.0)
    norm1 = sum(x**2 for x in vec1)
    norm2 = sum(x**2 for x in vec2)
    assert math.isclose(norm1, 1.0, rel_tol=1e-3)
    assert math.isclose(norm2, 1.0, rel_tol=1e-3)

    # Test custom dimension (e.g., 384)
    custom_dim = 384
    vec_custom = generate_local_fast_embedding(text1, dimension=custom_dim)
    assert len(vec_custom) == custom_dim
    assert math.isclose(sum(x**2 for x in vec_custom), 1.0, rel_tol=1e-3)

    # Test empty or whitespace text
    vec_empty = generate_local_fast_embedding("", dimension=dim)
    assert len(vec_empty) == dim
    assert math.isclose(sum(x**2 for x in vec_empty), 1.0, rel_tol=1e-3)


def test_text_embedder_documents_and_query():
    """Verify TextEmbedder batch embed_documents and embed_query methods."""
    embedder = TextEmbedder(provider="local_fast", dimension=1536)

    # Batch embedding
    texts = [
        "Welcome to the Voice Sales Copilot platform.",
        "Handling customer objections with real-time battlecards.",
        "Third sentence for multi-document batch verification.",
    ]
    doc_embeddings = embedder.embed_documents(texts)
    assert len(doc_embeddings) == 3
    for emb in doc_embeddings:
        assert len(emb) == 1536
        assert math.isclose(sum(x**2 for x in emb), 1.0, rel_tol=1e-3)

    # Empty documents list
    assert embedder.embed_documents([]) == []

    # Query embedding
    query = "What is the discount policy for enterprise customers?"
    query_vec = embedder.embed_query(query)
    assert len(query_vec) == 1536
    assert math.isclose(sum(x**2 for x in query_vec), 1.0, rel_tol=1e-3)


def test_embed_chunks_preserves_metadata():
    """Verify embed_chunks strictly preserves all metadata and appends embeddings."""
    text1 = "Chapter 1: Strategic Pricing for Enterprise Deals."
    text2 = "Chapter 2: Defending Against Competitor Feature FUD."
    mock_chunks = [
        {
            "chunk_index": 0,
            "page_number": 1,
            "character_count": len(text1),
            "text": text1,
            "source": "sales_guide.pdf",
            "metadata": {"custom_tag": "pricing", "author": "Revenue Ops"},
        },
        {
            "chunk_index": 1,
            "page_number": 2,
            "character_count": len(text2),
            "text": text2,
            "source": "sales_guide.pdf",
            "metadata": {"custom_tag": "battlecard", "author": "Product Marketing"},
        },
    ]

    enriched = embed_chunks(mock_chunks)

    assert len(enriched) == 2
    for idx, c in enumerate(enriched):
        # Preserved fields
        assert c["chunk_index"] == idx
        assert c["page_number"] == idx + 1
        assert c["source"] == "sales_guide.pdf"
        assert c["character_count"] == len(c["text"])
        assert c["metadata"]["author"] in ["Revenue Ops", "Product Marketing"]

        # Newly enriched embedding fields
        assert "embedding" in c
        assert isinstance(c["embedding"], list)
        assert len(c["embedding"]) == 1536
        assert c["embedding_dimension"] == 1536
        assert c["embedding_model"] == DEFAULT_MODEL
        assert c["embedding_provider"] == "local_fast"
        assert math.isclose(sum(x**2 for x in c["embedding"]), 1.0, rel_tol=1e-3)

    # Verify empty chunk list returns empty list
    assert embed_chunks([]) == []


def test_openai_fallback_without_api_key():
    """Verify that provider='openai' cleanly falls back to 'local_fast' if OPENAI_API_KEY is empty."""
    embedder = TextEmbedder(provider="openai", api_key="")
    assert embedder.provider == "local_fast"

    vec = embedder.embed_query("Fallback test query")
    assert len(vec) == 1536
    assert math.isclose(sum(x**2 for x in vec), 1.0, rel_tol=1e-3)


def test_openai_provider_with_mocked_client():
    """Verify that TextEmbedder delegates to langchain-openai when configured."""
    mock_embeddings_instance = MagicMock()
    mock_embeddings_instance.embed_documents.return_value = [[0.1] * 1536, [0.2] * 1536]
    mock_embeddings_instance.embed_query.return_value = [0.15] * 1536

    with patch("langchain_openai.OpenAIEmbeddings", return_value=mock_embeddings_instance):
        embedder = TextEmbedder(provider="openai", api_key="sk-fake-key-for-testing", dimension=1536)
        assert embedder.provider == "openai"
        assert embedder._client is not None

        doc_results = embedder.embed_documents(["doc1", "doc2"])
        assert len(doc_results) == 2
        assert len(doc_results[0]) == 1536
        mock_embeddings_instance.embed_documents.assert_called_once_with(["doc1", "doc2"])

        query_result = embedder.embed_query("search query")
        assert len(query_result) == 1536
        mock_embeddings_instance.embed_query.assert_called_once_with("search query")


def test_api_extract_pdf_returns_embeddings(client, sample_pdf):
    """End-to-end integration test: POST /api/extract-pdf returns embedded chunks."""
    with open(sample_pdf, "rb") as f:
        res = client.post(
            "/api/extract-pdf",
            files={"file": ("sales_battlecards.pdf", f, "application/pdf")},
        )

    assert res.status_code == 200
    data = res.json()

    # Top-level embedding metadata
    assert data["embedding_dimension"] in (1024, 1536)
    assert data["embedding_provider"] in ("pinecone_integrated", "local_fast")
    assert "embedding_model" in data
    assert data["total_chunks"] >= 2

    # Chunk-level assertions
    for chunk in data["chunks"]:
        assert "text" in chunk
        assert "chunk_index" in chunk
        if "embedding" in chunk:
            assert isinstance(chunk["embedding"], list)
            assert len(chunk["embedding"]) in (1024, 1536)


def test_api_embed_chunks_standalone_endpoint(client):
    """Integration test for POST /api/embed-chunks with chunks, texts, and query inputs."""
    # 1. Embed raw texts
    res_texts = client.post(
        "/api/embed-chunks",
        json={"texts": ["High-velocity sales script", "Enterprise objection response"]},
    )
    assert res_texts.status_code == 200
    data_texts = res_texts.json()
    assert data_texts["total_embeddings"] == 2
    assert len(data_texts["embeddings"]) == 2
    assert len(data_texts["embeddings"][0]) == 1536

    # 2. Embed single query
    res_query = client.post(
        "/api/embed-chunks",
        json={"query": "How do I pitch value over competitor discount?"},
    )
    assert res_query.status_code == 200
    data_query = res_query.json()
    assert "embedding" in data_query
    assert len(data_query["embedding"]) == 1536

    # 3. Embed chunk dictionary list
    res_chunks = client.post(
        "/api/embed-chunks",
        json={
            "chunks": [
                {
                    "chunk_index": 0,
                    "page_number": 1,
                    "text": "Enterprise cloud compliance details.",
                    "source": "compliance.pdf",
                }
            ]
        },
    )
    assert res_chunks.status_code == 200
    data_chunks = res_chunks.json()
    assert data_chunks["total_chunks"] == 1
    chunk_0 = data_chunks["chunks"][0]
    assert chunk_0["chunk_index"] == 0
    assert chunk_0["page_number"] == 1
    assert len(chunk_0["embedding"]) == 1536
    assert chunk_0["embedding_dimension"] == 1536

    # 4. Invalid empty request body returns 400
    res_bad = client.post("/api/embed-chunks", json={})
    assert res_bad.status_code == 400
