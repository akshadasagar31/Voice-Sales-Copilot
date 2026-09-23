import os
import sys
import httpx

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_pdf_extractor import create_minimal_pdf_bytes


def main():
    # Try testing against port 8000 or 8001
    ports = [8000, 8001]
    base_url = None
    live_detected = False

    for port in ports:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
            if r.status_code == 200:
                base_url = f"http://127.0.0.1:{port}"
                live_detected = True
                break
        except Exception:
            continue

    if live_detected:
        print(f"Testing live FastAPI service at {base_url}...")
        client = httpx.Client(base_url=base_url)
    else:
        print("No live server detected on port 8000/8001. Running in-process verification with TestClient(app)...")
        from main import app
        from fastapi.testclient import TestClient
        client = TestClient(app)

    with client:

        # 1. Health check
        print("\n[1] Checking /health endpoint...")
        res = client.get("/health")
        print(f"Status: {res.status_code}")
        print("Response:", res.json())
        assert res.status_code == 200

        # 2. Upload valid 3-page PDF with multi-paragraph text
        print("\n[2] Generating and uploading multi-page PDF to /api/extract-pdf...")
        pdf_pages = [
            "Voice Sales Copilot - Chapter 1: Introduction to AI-Assisted Selling. " * 10,
            "Voice Sales Copilot - Chapter 2: Handling Pricing & Compliance Objections in Real-Time. " * 10,
            "Voice Sales Copilot - Chapter 3: Enterprise Cloud SLA and Security Guarantees. " * 10,
        ]
        pdf_bytes = create_minimal_pdf_bytes(pdf_pages)

        files = {
            "file": ("enterprise_sales_playbook.pdf", pdf_bytes, "application/pdf")
        }
        res_upload = client.post("/api/extract-pdf", files=files)
        print(f"Status: {res_upload.status_code}")
        upload_data = res_upload.json()
        print(f"Extracted metadata: filename='{upload_data.get('filename')}', pages={upload_data.get('total_pages')}, characters={upload_data.get('total_characters')}")
        print(f"Chunking results: total_chunks={upload_data.get('total_chunks')}, chunk_size={upload_data.get('chunk_size')}, chunk_overlap={upload_data.get('chunk_overlap')}")
        
        print("\n  Page breakdown:")
        for p in upload_data.get("pages", []):
            print(f"    - Page {p['page_number']} ({p['character_count']} chars)")

        print(f"\n  Embedding details: provider='{upload_data.get('embedding_provider')}', model='{upload_data.get('embedding_model')}', dimension={upload_data.get('embedding_dimension')}")
        print("\n  Sample chunks with preserved metadata & embeddings:")
        for c in upload_data.get("chunks", [])[:5]:
            snippet = c['text'][:60].replace('\n', ' ')
            emb = c.get('embedding', [])
            emb_preview = f"[{emb[0]:.4f}, {emb[1]:.4f}, ... len={len(emb)}]" if emb else "none"
            print(f"    - Chunk #{c['chunk_index']} | Page {c['page_number']} | Length {c['character_count']} chars | Vector: {emb_preview}")
            print(f"      Text preview: \"{snippet}...\"")

        assert res_upload.status_code == 200
        assert upload_data["total_pages"] == 3
        assert upload_data["total_chunks"] >= 3
        assert len(upload_data["chunks"]) == upload_data["total_chunks"]
        assert upload_data["embedding_dimension"] == 1536
        for c in upload_data["chunks"]:
            assert "embedding" in c
            assert len(c["embedding"]) == 1536

        # 3. Test standalone /api/embed-chunks endpoint
        print("\n[3] Testing standalone /api/embed-chunks endpoint...")
        res_standalone = client.post(
            "/api/embed-chunks",
            json={"texts": ["Sales objection: Pricing is too high.", "Feature comparison matrix."]},
        )
        print(f"Status: {res_standalone.status_code}")
        standalone_data = res_standalone.json()
        print(f"Generated {standalone_data.get('total_embeddings')} embeddings of dim {standalone_data.get('embedding_dimension')}")
        assert res_standalone.status_code == 200
        assert standalone_data["total_embeddings"] == 2
        assert len(standalone_data["embeddings"][0]) == 1536

        # 4. Test /api/retrieve query endpoint validation
        print("\n[4] Testing /api/retrieve query validation...")
        res_ret_bad = client.post("/api/retrieve", json={"query": "   "})
        print(f"Status for empty query: {res_ret_bad.status_code} (expected 400)")
        assert res_ret_bad.status_code == 400

        # 5. Test /api/ask RAG endpoint validation
        print("\n[5] Testing /api/ask RAG query validation...")
        res_ask_bad = client.post("/api/ask", json={"question": "   "})
        print(f"Status for empty question: {res_ask_bad.status_code} (expected 400)")
        assert res_ask_bad.status_code == 400

        # 6. Test non-PDF rejection
        print("\n[6] Testing rejection of non-PDF file...")
        res_txt = client.post(
            "/api/extract-pdf",
            files={"file": ("playbook.txt", b"plain text", "text/plain")}
        )
        print(f"Status: {res_txt.status_code} (expected 400)")
        print("Response detail:", res_txt.json().get("detail"))
        assert res_txt.status_code == 400

        print("\n=== Live API verification passed 100%! ===")


if __name__ == "__main__":
    main()
