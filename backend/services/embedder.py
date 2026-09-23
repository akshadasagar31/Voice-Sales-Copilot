# ============================================================================
# TEXT EMBEDDING SERVICE (backend/services/embedder.py)
# ============================================================================
# WHAT THIS SERVICE DOES:
# Converts human text into mathematical float arrays (dense vectors of 1536 numbers).
#
# WHY EMBEDDINGS MATTER:
# In vector search, texts with similar meanings (like "interest rate" and "ROI policy")
# will have embeddings that point in similar directions in 1536-dimensional space.
#
# SUPPORTED EMBEDDING PROVIDERS:
# 1. 'local_fast': A deterministic, fast, zero-dependency embedding generator.
#    Uses SHA-256 and lexical feature hashing to produce normalized unit vectors.
#    Runs locally with no API keys, no network calls, and zero cost!
# 2. 'openai': Calls OpenAI's text-embedding-3-small model for production semantic embeddings.
# ============================================================================

import os
import math
import hashlib
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
import numpy as np

# Load environment variables from backend/.env if available
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_PROVIDER = "local_fast"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DIMENSION = 1536



# ----------------------------------------------------------------------------
# FUNCTION: generate_local_fast_embedding
# ----------------------------------------------------------------------------
# • WHAT IT DOES: Generates a deterministic 1536-dimensional unit float vector
#   directly in Python without making any external network or API calls.
# • INPUTS:
#     - text (str): The text string to embed.
#     - dimension (int): Vector length (default: 1536).
# • OUTPUT: Normalized array of 1536 float numbers ($||v|| = 1.0$).
# • WHY IT IS USED: Perfect for local testing and offline development. The same text
#   always yields the exact same vector, and distinct texts produce distinct vectors.
# • WHERE IT FITS IN THE FLOW:
#     [TextEmbedder] -> [generate_local_fast_embedding] -> [Pinecone or VectorRetriever]
# ----------------------------------------------------------------------------
def generate_local_fast_embedding(text: str, dimension: int = DEFAULT_DIMENSION) -> List[float]:
    """
    Generates a deterministic, normalized dense float vector of the specified dimension
    without requiring any external API keys or network connection.


    - Deterministic: The same input text always generates the exact same vector.
    - Normalized: The resulting vector has an L2 norm of approximately 1.0 (unit vector).
    - Differentiating: Distinct texts produce distinct vectors.
    """
    if not text or not text.strip():
        val = 1.0 / math.sqrt(dimension)
        return [round(float(val), 6)] * dimension

    # Deterministically seed NumPy random generator with the text's SHA-256 digest
    text_digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(text_digest[:8], byteorder="big")
    rng = np.random.default_rng(seed)

    # Base Gaussian distribution
    vec = rng.standard_normal(dimension).astype(np.float64)

    # Add word-level frequency hashing to simulate token feature weighting
    words = text.lower().split()
    for w in words[:150]:
        w_hash = hashlib.md5(w.encode("utf-8")).digest()
        idx = int.from_bytes(w_hash[:4], byteorder="big") % dimension
        vec[idx] += 1.25

    # L2 normalize the vector to unit length
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return [round(float(x), 6) for x in vec]


class TextEmbedder:
    """
    Configurable text embedding service supporting both 'openai' and 'local_fast' providers.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
        api_key: Optional[str] = None,
    ):
        env_provider = os.getenv("EMBEDDING_PROVIDER", DEFAULT_PROVIDER)
        if env_provider == "pinecone_integrated":
            env_provider = DEFAULT_PROVIDER
        self.provider = (provider or env_provider).strip().lower()

        env_model = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
        if env_model == "llama-text-embed-v2":
            env_model = DEFAULT_MODEL
        self.model = (model or env_model).strip()

        env_dim = os.getenv("EMBEDDING_DIMENSION", str(DEFAULT_DIMENSION))
        if env_dim == "1024":
            env_dim = str(DEFAULT_DIMENSION)
        self.dimension = int(dimension or env_dim)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "").strip()

        self._client = None
        if self.provider == "openai":
            if not self.api_key:
                logger.warning(
                    "EMBEDDING_PROVIDER is 'openai' but OPENAI_API_KEY is not set. "
                    "Falling back to 'local_fast' embedding provider."
                )
                self.provider = "local_fast"
            else:
                try:
                    from langchain_openai import OpenAIEmbeddings

                    # text-embedding-3 models support explicit dimensions
                    if "text-embedding-3" in self.model:
                        self._client = OpenAIEmbeddings(
                            model=self.model,
                            openai_api_key=self.api_key,
                            dimensions=self.dimension,
                        )
                    else:
                        self._client = OpenAIEmbeddings(
                            model=self.model,
                            openai_api_key=self.api_key,
                        )
                except Exception as e:
                    logger.error(f"Failed to initialize OpenAIEmbeddings: {e}. Falling back to local_fast.")
                    self.provider = "local_fast"

    # ------------------------------------------------------------------------
    # METHOD: embed_documents
    # ------------------------------------------------------------------------
    # • WHAT IT DOES: Converts a list of text strings into a list of 1536-dim vectors.
    # • INPUTS:
    #     - texts (List[str]): Plain text strings from document chunks.
    # • OUTPUT: List of float arrays, one 1536-dim embedding vector per input text.
    # • WHY IT IS USED: Batches embedding requests to OpenAI or local generator.
    # • WHERE IT FITS IN THE FLOW:
    #     [chunk_documents] -> [embed_documents] -> [upsert_chunks to Pinecone]
    # ------------------------------------------------------------------------
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of text strings into dense vector representations.
        """
        if not texts:
            return []

        if self.provider == "openai" and self._client:
            try:
                return self._client.embed_documents(texts)
            except Exception as e:
                logger.error(f"OpenAI embedding call failed: {e}. Falling back to local_fast generation.")
                return [generate_local_fast_embedding(t, self.dimension) for t in texts]

        # Default local_fast provider
        return [generate_local_fast_embedding(t, self.dimension) for t in texts]

    # ------------------------------------------------------------------------
    # METHOD: embed_query
    # ------------------------------------------------------------------------
    # • WHAT IT DOES: Converts a single user query string into a 1536-dim vector.
    # • INPUTS:
    #     - text (str): The search query or objection asked by the prospect.
    # • OUTPUT: Single 1536-dimensional float vector.
    # • WHY IT IS USED: The query vector must live in the exact same vector space
    #   as the document embeddings for cosine similarity search to work.
    # • WHERE IT FITS IN THE FLOW:
    #     [User asks question] -> [embed_query] -> [Pinecone query_vectors]
    # ------------------------------------------------------------------------
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string into a dense vector representation.
        """
        if not text:
            return generate_local_fast_embedding("", self.dimension)

        if self.provider == "openai" and self._client:
            try:
                return self._client.embed_query(text)
            except Exception as e:
                logger.error(f"OpenAI embed_query failed: {e}. Falling back to local_fast generation.")
                return generate_local_fast_embedding(text, self.dimension)

        return generate_local_fast_embedding(text, self.dimension)


# ----------------------------------------------------------------------------
# FUNCTION: embed_chunks
# ----------------------------------------------------------------------------
# • WHAT IT DOES: Iterates through chunk dicts, generates their vector embeddings,
#   and attaches the embedding vector to each chunk dictionary.
# • INPUTS:
#     - chunks (List[Dict[str, Any]]): Chunks from chunk_documents.
#     - embedder (Optional[TextEmbedder]): Embedder instance.
# • OUTPUT: List of chunks enriched with 'embedding', 'embedding_model', and 'embedding_dimension'.
# • WHY IT IS USED: Keeps all chunk data, page numbers, and vector embeddings unified
#   in a single data structure before sending to Pinecone.
# • WHERE IT FITS IN THE FLOW:
#     [chunk_documents] -> [embed_chunks] -> [format_chunk_for_pinecone] -> [upsert]
# ----------------------------------------------------------------------------
def embed_chunks(
    chunks: List[Dict[str, Any]],
    embedder: Optional[TextEmbedder] = None,
) -> List[Dict[str, Any]]:
    """
    Enriches each chunk dictionary with vector embeddings and embedding metadata,

    strictly preserving existing attributes (chunk_index, page_number, source, etc.).

    Args:
        chunks: List of chunk dictionaries produced by chunk_documents.
        embedder: Optional TextEmbedder instance (creates a default instance if omitted).

    Returns:
        List of enriched chunk dictionaries with:
          - 'embedding': List[float]
          - 'embedding_dimension': int
          - 'embedding_model': str
          - 'embedding_provider': str
    """
    if not chunks:
        return []

    if embedder is None:
        embedder = TextEmbedder()

    texts = [str(c.get("text", "")) for c in chunks]
    embeddings = embedder.embed_documents(texts)

    enriched_chunks: List[Dict[str, Any]] = []
    for chunk, emb in zip(chunks, embeddings):
        enriched_chunk = {
            **chunk,
            "embedding": emb,
            "embedding_dimension": len(emb),
            "embedding_model": embedder.model,
            "embedding_provider": embedder.provider,
        }
        # Also ensure metadata dictionary reflects embedding fields if present
        if "metadata" in enriched_chunk and isinstance(enriched_chunk["metadata"], dict):
            enriched_chunk["metadata"] = {
                **enriched_chunk["metadata"],
                "embedding_dimension": len(emb),
                "embedding_model": embedder.model,
                "embedding_provider": embedder.provider,
            }
        enriched_chunks.append(enriched_chunk)

    return enriched_chunks
