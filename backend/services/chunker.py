# ============================================================================
# DOCUMENT CHUNKER SERVICE (backend/services/chunker.py)
# ============================================================================
# WHAT THIS SERVICE DOES:
# Splits large PDF document text into smaller, overlapping chunks suitable for
# vector embeddings and RAG retrieval.
#
# WHY CHUNKING IS NECESSARY:
# 1. Embedding models and vector databases perform best on concise paragraphs (300-800 chars).
# 2. Chunk Overlap (100 characters) ensures that thoughts spanning across chunk boundaries
#    are not cut in half, preserving semantic context.
# 3. Preserves Page Numbers: Every chunk keeps its 1-indexed page number so the AI
#    can cite exact pages (e.g., "Page 4, Sales Playbook").
# ============================================================================

from typing import List, Dict, Any
from langchain_core.documents import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 100


# ----------------------------------------------------------------------------
# FUNCTION: get_text_splitter
# ----------------------------------------------------------------------------
# • WHAT IT DOES: Creates and configures a RecursiveCharacterTextSplitter instance.
# • INPUTS:
#     - chunk_size (int): Max number of characters allowed in each chunk (default: 600).
#     - chunk_overlap (int): Number of overlapping characters between adjacent chunks (default: 100).
# • OUTPUT: A configured RecursiveCharacterTextSplitter object ready to split text.
# • WHY IT IS USED: Uses recursive separators (paragraphs "\n\n", then sentences "\n",
#   then spaces " ") so text is split at natural reading boundaries rather than mid-word.
# • WHERE IT FITS IN THE FLOW:
#     [PDF Upload] -> [pdf_extractor.py] -> [chunk_documents] -> [get_text_splitter]
# ----------------------------------------------------------------------------
def get_text_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """
    Returns a configured RecursiveCharacterTextSplitter instance.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )


# ----------------------------------------------------------------------------
# FUNCTION: chunk_documents
# ----------------------------------------------------------------------------
# • WHAT IT DOES: Splits a list of LangChain Document objects into smaller chunks while
#   tracking and preserving 1-indexed page numbers and source filename metadata.
# • INPUTS:
#     - documents (List[Document]): Raw pages extracted by PyPDFLoader.
#     - original_filename (str): The name of the PDF file uploaded by the user.
#     - chunk_size (int): Max characters per segment (default: 600).
#     - chunk_overlap (int): Overlap characters to prevent lost context (default: 100).
# • OUTPUT: List of dictionaries containing:
#     - chunk_index: Sequential number of the chunk (0, 1, 2, ...).
#     - page_number: The physical page in the PDF where this text appeared.
#     - text: The chunk content string.
#     - source: The original filename.
# • WHY IT IS USED: Search engines and LLMs work best on small targeted paragraphs.
#   Tracking page numbers enables the assistant to cite exact pages to the user.
# • WHERE IT FITS IN THE FLOW:
#     [User uploads PDF]
#            ↓
#     [extract_text_from_pdf in pdf_extractor.py]
#            ↓
#     [chunk_documents in chunker.py]  <-- (WE ARE HERE)
#            ↓
#     [embed_chunks in embedder.py]
#            ↓
#     [upsert_chunks in vector_store.py -> Pinecone Database]
# ----------------------------------------------------------------------------
def chunk_documents(
    documents: List[Document],
    original_filename: str = "",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Chunks a list of LangChain Document objects using RecursiveCharacterTextSplitter
    while strictly preserving page numbers and source metadata.


    Args:
        documents: List of LangChain Document objects (e.g. from PyPDFLoader).
        original_filename: Original filename of the document.
        chunk_size: Maximum character count per chunk (default 600).
        chunk_overlap: Overlapping character count between consecutive chunks (default 100).

    Returns:
        List of dictionaries with text, page_number, chunk_index, character_count, and source.
    """
    if not documents:
        return []

    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(documents)

    chunks: List[Dict[str, Any]] = []

    for idx, doc in enumerate(split_docs):
        # Extract page number: PyPDFLoader uses 0-indexed 'page'
        raw_page = doc.metadata.get("page")
        if raw_page is not None:
            page_number = int(raw_page) + 1
        elif "page_number" in doc.metadata:
            page_number = int(doc.metadata["page_number"])
        else:
            page_number = 1

        source = original_filename or str(doc.metadata.get("source", ""))
        text_content = doc.page_content or ""

        chunks.append({
            "chunk_index": idx,
            "page_number": page_number,
            "character_count": len(text_content),
            "text": text_content,
            "source": source,
            "metadata": {
                **doc.metadata,
                "chunk_index": idx,
                "page_number": page_number,
                "source": source,
            },
        })

    return chunks
