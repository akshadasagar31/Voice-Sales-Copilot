import os
import sys
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app
from services.pdf_extractor import extract_text_from_pdf, PDFExtractionError
from services.chunker import chunk_documents, get_text_splitter


def create_minimal_pdf_bytes(pages_text: list[str]) -> bytes:
    """
    Constructs a minimal, strictly valid multi-page PDF document in raw bytes
    without external third-party PDF generators.
    """
    num_pages = len(pages_text)
    page_obj_ids = []

    current_obj_id = 4
    page_and_content_objs = []

    for text in pages_text:
        page_id = current_obj_id
        content_id = current_obj_id + 1
        current_obj_id += 2

        page_obj_ids.append(page_id)

        stream_content = f"BT\n/F1 14 Tf\n72 712 Td\n({text}) Tj\nET\n".encode("latin-1")
        content_obj = (
            f"{content_id} 0 obj\n"
            f"<< /Length {len(stream_content)} >>\n"
            f"stream\n".encode("latin-1")
            + stream_content
            + b"\nendstream\nendobj\n"
        )

        page_obj = (
            f"{page_id} 0 obj\n"
            f"<< /Type /Page\n"
            f"/Parent 2 0 R\n"
            f"/MediaBox [0 0 612 792]\n"
            f"/Contents {content_id} 0 R\n"
            f"/Resources << /Font << /F1 3 0 R >> >>\n"
            f">>\nendobj\n"
        ).encode("latin-1")

        page_and_content_objs.extend([page_obj, content_obj])

    kids_str = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    pages_obj = (
        f"2 0 obj\n"
        f"<< /Type /Pages\n"
        f"/Kids [{kids_str}]\n"
        f"/Count {num_pages}\n"
        f">>\nendobj\n"
    ).encode("latin-1")

    catalog_obj = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    font_obj = b"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"

    all_objs = [catalog_obj, pages_obj, font_obj] + page_and_content_objs

    pdf_bytes = bytearray(b"%PDF-1.4\n")
    xref_offsets = [0]

    for obj in all_objs:
        xref_offsets.append(len(pdf_bytes))
        pdf_bytes.extend(obj)

    xref_start = len(pdf_bytes)
    total_objs = len(all_objs) + 1
    pdf_bytes.extend(f"xref\n0 {total_objs}\n0000000000 65535 f \n".encode("latin-1"))

    for offset in xref_offsets[1:]:
        pdf_bytes.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))

    pdf_bytes.extend(
        f"trailer\n<< /Size {total_objs} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode(
            "latin-1"
        )
    )

    return bytes(pdf_bytes)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_two_page_pdf(tmp_path):
    pdf_bytes = create_minimal_pdf_bytes([
        "Voice Sales Copilot: Page 1 Playbook Overview",
        "Objection Battlecard: Page 2 Pricing and Discount Guidelines",
    ])
    file_path = tmp_path / "sample_sales_playbook.pdf"
    file_path.write_bytes(pdf_bytes)
    return str(file_path)


def test_health_endpoints(client):
    """Test GET / and GET /health return 200 OK."""
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert res_root.json()["status"] == "online"

    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "online"


def test_unit_extract_text_and_chunks(sample_two_page_pdf):
    """Unit test for extract_text_from_pdf function verifying extraction and chunking."""
    result = extract_text_from_pdf(sample_two_page_pdf, "sample_sales_playbook.pdf")

    assert result["filename"] == "sample_sales_playbook.pdf"
    assert result["total_pages"] == 2
    assert result["total_characters"] > 0
    assert len(result["pages"]) == 2

    # Verify individual pages
    assert result["pages"][0]["page_number"] == 1
    assert "Voice Sales Copilot: Page 1" in result["pages"][0]["text"]

    assert result["pages"][1]["page_number"] == 2
    assert "Objection Battlecard: Page 2" in result["pages"][1]["text"]

    # Verify chunking results
    assert result["total_chunks"] >= 2
    assert result["chunk_size"] == 600
    assert result["chunk_overlap"] == 100
    assert len(result["chunks"]) == result["total_chunks"]

    # Verify chunk metadata preservation
    for chunk in result["chunks"]:
        assert "chunk_index" in chunk
        assert "page_number" in chunk
        assert chunk["page_number"] in [1, 2]
        assert chunk["source"] == "sample_sales_playbook.pdf"
        assert chunk["character_count"] == len(chunk["text"])
        assert chunk["character_count"] <= 600


def test_chunking_size_and_overlap():
    """Unit test verifying RecursiveCharacterTextSplitter with chunk_size=600 and chunk_overlap=100."""
    # Create a long text document (~1500 chars)
    sentence = "Voice sales copilot assists account executives during active customer discovery calls. "
    long_text = sentence * 18  # 87 * 18 = 1566 characters
    assert len(long_text) > 1500

    docs = [Document(page_content=long_text, metadata={"page": 0, "source": "discovery_playbook.pdf"})]
    chunks = chunk_documents(docs, original_filename="discovery_playbook.pdf", chunk_size=600, chunk_overlap=100)

    assert len(chunks) >= 3

    for idx, chunk in enumerate(chunks):
        assert chunk["chunk_index"] == idx
        assert chunk["page_number"] == 1
        assert chunk["source"] == "discovery_playbook.pdf"
        assert chunk["character_count"] <= 600
        assert len(chunk["text"]) == chunk["character_count"]

    # Verify overlap between consecutive chunks
    for i in range(len(chunks) - 1):
        current_chunk_text = chunks[i]["text"]
        next_chunk_text = chunks[i + 1]["text"]
        # The end of current chunk should overlap with the beginning of the next chunk
        overlap_found = any(
            current_chunk_text[j:j+30] in next_chunk_text
            for j in range(len(current_chunk_text) - 120, len(current_chunk_text) - 30)
        )
        assert overlap_found, f"Expected text overlap between chunk {i} and {i+1}"


def test_chunking_multipage_metadata_preservation():
    """Verify that chunks across multiple pages correctly track their respective page numbers."""
    page1_text = "Page 1: Introduction to Voice AI. " * 25  # ~875 characters (will be 2 chunks)
    page2_text = "Page 2: Pricing and Objection Battlecards. " * 20  # ~860 characters (will be 2 chunks)

    docs = [
        Document(page_content=page1_text, metadata={"page": 0, "source": "manual.pdf"}),
        Document(page_content=page2_text, metadata={"page": 1, "source": "manual.pdf"}),
    ]

    chunks = chunk_documents(docs, original_filename="manual.pdf", chunk_size=600, chunk_overlap=100)

    # Page 1 chunks should have page_number == 1
    page1_chunks = [c for c in chunks if c["page_number"] == 1]
    assert len(page1_chunks) >= 2
    for c in page1_chunks:
        assert "Page 1" in c["text"]
        assert c["page_number"] == 1
        assert c["source"] == "manual.pdf"

    # Page 2 chunks should have page_number == 2
    page2_chunks = [c for c in chunks if c["page_number"] == 2]
    assert len(page2_chunks) >= 2
    for c in page2_chunks:
        assert "Page 2" in c["text"]
        assert c["page_number"] == 2
        assert c["source"] == "manual.pdf"


def test_api_upload_extract_and_chunk(client, sample_two_page_pdf):
    """Integration test verifying POST /api/extract-pdf returns both pages and chunks."""
    with open(sample_two_page_pdf, "rb") as f:
        response = client.post(
            "/api/extract-pdf",
            files={"file": ("sample_sales_playbook.pdf", f, "application/pdf")},
        )

    assert response.status_code == 200
    data = response.json()

    assert data["filename"] == "sample_sales_playbook.pdf"
    assert data["total_pages"] == 2
    assert data["total_characters"] > 0
    assert len(data["pages"]) == 2

    # Step 3 chunking assertions
    assert data["total_chunks"] >= 2
    assert data["chunk_size"] == 600
    assert data["chunk_overlap"] == 100
    assert len(data["chunks"]) == data["total_chunks"]

    for chunk in data["chunks"]:
        assert "chunk_index" in chunk
        assert "page_number" in chunk
        assert "character_count" in chunk
        assert "text" in chunk
        assert "source" in chunk
        assert chunk["character_count"] <= 600


def test_api_reject_non_pdf_file(client, tmp_path):
    """Test that uploading non-PDF files is rejected with HTTP 400."""
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("Plain text notes")

    with open(txt_file, "rb") as f:
        response = client.post(
            "/api/extract-pdf",
            files={"file": ("notes.txt", f, "text/plain")},
        )

    assert response.status_code == 400
    assert "Only .pdf files are supported" in response.json()["detail"]


def test_api_reject_empty_file(client, tmp_path):
    """Test that uploading an empty PDF (0 bytes) is rejected with HTTP 400."""
    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.write_bytes(b"")

    with open(empty_pdf, "rb") as f:
        response = client.post(
            "/api/extract-pdf",
            files={"file": ("empty.pdf", f, "application/pdf")},
        )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_unit_extract_non_existent_file():
    """Test that extract_text_from_pdf raises PDFExtractionError on missing file."""
    with pytest.raises(PDFExtractionError) as exc_info:
        extract_text_from_pdf("non_existent_file_path.pdf")
    assert "File not found" in str(exc_info.value)
