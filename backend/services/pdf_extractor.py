# ============================================================================
# PDF INGESTION & EXTRACTION PIPELINE (backend/services/pdf_extractor.py)
# ============================================================================
# WHAT THIS SERVICE DOES:
# Ingests sales playbook PDFs and prepares them for real-time vector search.
#
# STEP-BY-STEP DATA FLOW:
# 1. Loads the uploaded PDF using PyPDFLoader.
# 2. Extracts text from each page with 1-indexed page numbering.
# 3. Splits pages into 600-character overlapping chunks using `chunk_documents`.
# 4. Generates 1536-dimensional dense vector embeddings for each chunk.
# 5. Automatically upserts vectors and text into Pinecone so that Module 2 can
#    immediately answer questions using the newly uploaded information!
# ============================================================================

import os
from typing import Dict, Any, List, Optional
from langchain_community.document_loaders import PyPDFLoader
from services.chunker import chunk_documents, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

class PDFExtractionError(Exception):
    """Custom exception raised when PDF text extraction fails."""
    pass



def extract_text_from_pdf(
    file_path: str,
    original_filename: str = "",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embedder: Optional[Any] = None,
    upsert_to_pinecone: bool = False,
    namespace: Optional[str] = None,
    pinecone_service: Optional[Any] = None,
    doc_id_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extracts text from a PDF file using LangChain's PyPDFLoader, chunks
    the document using RecursiveCharacterTextSplitter while preserving page metadata,
    and prepares chunks for Pinecone Integrated Embeddings (llama-text-embed-v2).

    Args:
        file_path: Absolute or relative path to the local PDF file.
        original_filename: Original name of the uploaded file for reporting.
        chunk_size: Maximum character length per chunk (default 600).
        chunk_overlap: Overlap between consecutive chunks (default 100).
        embedder: Unused legacy parameter maintained for signature compatibility.
        upsert_to_pinecone: If True, automatically upserts chunks into Pinecone.
        namespace: Optional Pinecone namespace for multi-tenancy.
        pinecone_service: Optional custom or mocked PineconeService instance.

    Returns:
        Dictionary containing total pages, total characters, full text,
        page-by-page breakdown, embedding metadata, chunked segments,
        and optional Pinecone upsert summary.

    Raises:
        PDFExtractionError: If the file does not exist, is empty, or cannot be parsed.
    """
    embedding_model = os.getenv("EMBEDDING_MODEL", "llama-text-embed-v2")
    embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "pinecone_integrated")

    if not os.path.exists(file_path):
        raise PDFExtractionError(f"File not found: {file_path}")

    if os.path.getsize(file_path) == 0:
        raise PDFExtractionError("The uploaded PDF file is empty (0 bytes).")

    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    except Exception as e:
        raise PDFExtractionError(f"Failed to extract text from PDF with PyPDFLoader: {str(e)}") from e

    if not docs:
        # Empty document with no readable pages
        return {
            "filename": original_filename or os.path.basename(file_path),
            "total_pages": 0,
            "total_characters": 0,
            "extracted_text": "",
            "pages": [],
            "total_chunks": 0,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
            "embedding_provider": embedding_provider,
            "chunks": [],
        }

    pages_data: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []

    for idx, doc in enumerate(docs):
        # PyPDFLoader usually stores 0-indexed 'page' in metadata
        raw_page_num = doc.metadata.get("page")
        page_number = (raw_page_num + 1) if raw_page_num is not None else (idx + 1)
        page_text = doc.page_content or ""

        pages_data.append({
            "page_number": page_number,
            "character_count": len(page_text),
            "text": page_text,
        })
        full_text_parts.append(page_text)

    combined_text = "\n\n".join(full_text_parts)

    # Perform chunking using RecursiveCharacterTextSplitter
    raw_chunks = chunk_documents(
        docs,
        original_filename=original_filename or os.path.basename(file_path),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    result: Dict[str, Any] = {
        "filename": original_filename or os.path.basename(file_path),
        "total_pages": len(docs),
        "total_characters": len(combined_text),
        "extracted_text": combined_text,
        "pages": pages_data,
        "total_chunks": len(raw_chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "embedding_provider": embedding_provider,
        "chunks": raw_chunks,
    }

    if upsert_to_pinecone and raw_chunks:
        from services.vector_store import PineconeService
        ps = pinecone_service or PineconeService(namespace=namespace)
        upsert_res = ps.upsert_chunks(raw_chunks, namespace=namespace, doc_id_prefix=doc_id_prefix)
        result["pinecone_upsert"] = upsert_res

    if doc_id_prefix:
        result["doc_id_prefix"] = doc_id_prefix

    return result

