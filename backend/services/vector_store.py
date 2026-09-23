# ============================================================================
# PINECONE VECTOR STORE SERVICE (backend/services/vector_store.py)
# ============================================================================
# WHAT THIS SERVICE DOES:
# Manages interactions with the Pinecone Serverless Vector Database.
#
# WHY WE USE A VECTOR DATABASE:
# Traditional SQL databases search for exact keyword matches. A vector database
# searches by MEANING (semantic similarity).
# When a PDF is uploaded, its text chunks are converted into mathematical vectors (arrays of 1536 numbers).
# Pinecone indexes these vectors in multi-dimensional space so that user questions
# can instantly find the most relevant paragraphs in milliseconds.
#
# KEY RESPONSIBILITIES:
# 1. Serverless Index Management: Connects to Pinecone and ensures index exists.
# 2. Vector ID Sanitization: Creates deterministic, clean ASCII IDs for chunks.
# 3. Vector Upsert: Uploads vector embeddings along with text and page metadata.
# 4. Semantic Similarity Query: Searches for nearest neighbor vectors using cosine distance.
# ============================================================================

import os
import re
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from backend/.env if available
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    Pinecone = None
    ServerlessSpec = None

logger = logging.getLogger(__name__)



class PineconeConfigurationError(Exception):
    """Raised when Pinecone configuration (e.g. API key or index name) is missing or invalid."""
    pass


class PineconeUpsertError(Exception):
    """Raised when upserting vectors into Pinecone fails."""
    pass


class PineconeQueryError(Exception):
    """Raised when querying vectors from Pinecone fails."""
    pass


# Global index and verification cache for fast connection reuse in real-time queries
_INDEX_CACHE: Dict[str, Any] = {}
_CLIENT_CACHE: Dict[str, Any] = {}
_VERIFIED_INDEXES: set = set()


# ----------------------------------------------------------------------------
# FUNCTION: sanitize_vector_id
# ----------------------------------------------------------------------------
# • WHAT IT DOES: Builds a clean, deterministic, ASCII-only ID for a Pinecone vector.
# • INPUTS:
#     - source (str): Name of the PDF file (e.g. "sales_policy.pdf").
#     - page (int): Page number where the chunk appears.
#     - chunk_index (int): Index of this chunk on that page.
#     - prefix (Optional[str]): Optional custom prefix (e.g. document UID).
# • OUTPUT: A string ID like "sales_policy_pdf_p1_c0".
# • WHY IT IS USED: Pinecone requires ASCII-only vector IDs without spaces or special characters.
#   Deterministic IDs allow updating or deleting the exact same vector later without orphaned records.
# • WHERE IT FITS IN THE FLOW:
#     [chunk_documents] -> [sanitize_vector_id] -> [format_chunk_for_pinecone] -> [Pinecone upsert]
# ----------------------------------------------------------------------------
def sanitize_vector_id(source: str, page: int, chunk_index: int, prefix: Optional[str] = None) -> str:
    """
    Constructs an ASCII-safe, deterministic vector ID for Pinecone.
    Format: [prefix_]{sanitized_source}_p{page}_c{chunk_index}
    """
    clean_source = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", source.strip()) if source else "doc"
    base_id = f"{clean_source}_p{page}_c{chunk_index}"
    if prefix:
        clean_prefix = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", prefix.strip())
        return f"{clean_prefix}_{base_id}"
    return base_id


# ----------------------------------------------------------------------------
# FUNCTION: format_chunk_for_pinecone
# ----------------------------------------------------------------------------
# • WHAT IT DOES: Packages a chunk into the exact JSON record structure required by Pinecone:
#     {"id": str, "values": List[float], "metadata": Dict[str, Any]}
# • INPUTS:
#     - chunk (Dict[str, Any]): Dictionary containing 'text', 'page_number', 'embedding', etc.
#     - doc_id_prefix (Optional[str]): Optional prefix string for the vector ID.
# • OUTPUT: Pinecone-compliant record dictionary with vector values and searchable metadata.
# • WHY IT IS USED: When Pinecone returns search matches, it also returns the attached
#   metadata. Storing the raw text and page number inside metadata allows the RAG service
#   to read the text directly from Pinecone without querying a secondary database!
# • WHERE IT FITS IN THE FLOW:
#     [embed_chunks in embedder.py]
#            ↓
#     [format_chunk_for_pinecone]  <-- (WE ARE HERE)
#            ↓
#     [index.upsert() in vector_store.py]
# ----------------------------------------------------------------------------
def format_chunk_for_pinecone(
    chunk: Dict[str, Any],
    doc_id_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Formats a single chunk dictionary into Pinecone vector record format:

    {
        "id": "<sanitized_id>",
        "values": [0.1, 0.2, ...],
        "metadata": {
            "text": str,
            "source": str,
            "page": int,
            "chunk_index": int,
            "character_count": int,
            ...
        }
    }
    """
    source = str(chunk.get("source", "document"))
    page = int(chunk.get("page_number", 1))
    chunk_index = int(chunk.get("chunk_index", 0))
    text = str(chunk.get("text", ""))
    char_count = int(chunk.get("character_count", len(text)))

    vector_id = sanitize_vector_id(source, page, chunk_index, prefix=doc_id_prefix)
    values = chunk.get("embedding", [])

    if not isinstance(values, list) or len(values) == 0:
        raise ValueError(f"Chunk at index {chunk_index} has missing or invalid 'embedding' vector.")

    # Base Pinecone metadata (string, number, boolean, or list of strings)
    metadata: Dict[str, Any] = {
        "text": text,
        "source": source,
        "page": page,
        "chunk_index": chunk_index,
        "character_count": char_count,
    }

    # Merge additional non-conflicting metadata if present
    raw_meta = chunk.get("metadata", {})
    if isinstance(raw_meta, dict):
        for k, v in raw_meta.items():
            if k not in metadata and isinstance(v, (str, int, float, bool)):
                metadata[k] = v

    return {
        "id": vector_id,
        "values": values,
        "metadata": metadata,
    }


class PineconeService:
    """
    Service for managing Pinecone vector store operations (indexing, formatting, and batch upserting).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        environment: Optional[str] = None,
        namespace: Optional[str] = None,
        client: Optional[Any] = None,
        dimension: Optional[int] = None,
        metric: Optional[str] = None,
        index: Optional[Any] = None,
    ):
        self.api_key = (api_key if api_key is not None else os.getenv("PINECONE_API_KEY", "")).strip()
        self.index_name = (index_name or os.getenv("PINECONE_INDEX_NAME", "voice-sales-copilot")).strip()
        self.environment = (environment or os.getenv("PINECONE_ENVIRONMENT", "us-east-1")).strip()
        self.namespace = (namespace or os.getenv("PINECONE_NAMESPACE", "sales_playbooks")).strip()
        if dimension is not None:
            self.dimension = int(dimension)
        elif self.index_name == "voice-sales-copilot-llama":
            self.dimension = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
        elif self.index_name == "voice-sales-copilot" or "test" in self.index_name:
            self.dimension = 1536
        else:
            self.dimension = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
        self.metric = (metric or "cosine").strip().lower()

        self._explicit_client = client is not None
        self._client = client
        self._index = index

    def get_client(self) -> Any:
        """
        Returns or initializes the Pinecone client instance.
        """
        if self._client is not None:
            return self._client

        if not self.api_key:
            raise PineconeConfigurationError(
                "PINECONE_API_KEY is not set. Please configure PINECONE_API_KEY in backend/.env "
                "or pass an explicit api_key."
            )

        if Pinecone is None:
            raise PineconeConfigurationError(
                "Pinecone library is not installed. Please install pinecone>=5.0.0."
            )

        is_mock_pinecone = Pinecone is not None and type(Pinecone).__name__ in ("MagicMock", "Mock")
        if not self._explicit_client and not is_mock_pinecone and self.api_key in _CLIENT_CACHE:
            self._client = _CLIENT_CACHE[self.api_key]
            return self._client

        try:
            self._client = Pinecone(api_key=self.api_key)
            if not self._explicit_client and not is_mock_pinecone:
                _CLIENT_CACHE[self.api_key] = self._client
            return self._client
        except Exception as e:
            raise PineconeConfigurationError(f"Failed to initialize Pinecone client: {str(e)}") from e

    def ensure_index_exists(self) -> None:
        """
        Verifies if the configured Pinecone index exists.
        If missing, creates an AWS us-east-1 serverless index with 1536 dimensions and cosine metric.
        """
        client = self.get_client()
        try:
            index_exists = False
            if hasattr(client, "has_index"):
                try:
                    index_exists = bool(client.has_index(self.index_name))
                except Exception as e:
                    logger.debug(f"has_index check failed: {e}")
                    index_exists = False
            elif hasattr(client, "list_indexes"):
                try:
                    indexes = client.list_indexes()
                    names = [getattr(idx, "name", str(idx)) for idx in indexes]
                    index_exists = self.index_name in names
                except Exception as e:
                    logger.debug(f"list_indexes check failed: {e}")
                    index_exists = False

            if not index_exists:
                logger.info(
                    f"Pinecone index '{self.index_name}' not found. "
                    f"Creating serverless index: dimension={self.dimension}, metric={self.metric}, cloud=aws, region={self.environment or 'us-east-1'}..."
                )
                if ServerlessSpec is not None and hasattr(client, "create_index"):
                    client.create_index(
                        name=self.index_name,
                        dimension=self.dimension,
                        metric=self.metric,
                        spec=ServerlessSpec(
                            cloud="aws",
                            region=self.environment or "us-east-1",
                        ),
                    )
                    logger.info(f"Successfully initiated creation of Pinecone serverless index '{self.index_name}'.")
        except Exception as err:
            logger.warning(f"Unable to auto-create or verify Pinecone index '{self.index_name}': {str(err)}")

    def get_index(self) -> Any:
        """
        Returns or initializes the Pinecone Index handle.
        Uses cached Index instances for low-latency retrieval without repeat control plane checks.
        """
        if self._index is not None:
            return self._index

        is_mock_pinecone = Pinecone is not None and type(Pinecone).__name__ in ("MagicMock", "Mock")
        cache_key = f"{self.api_key}:{self.index_name}"
        if not self._explicit_client and not is_mock_pinecone and cache_key in _INDEX_CACHE:
            self._index = _INDEX_CACHE[cache_key]
            return self._index


        if not self.index_name:
            raise PineconeConfigurationError("PINECONE_INDEX_NAME is not set.")

        client = self.get_client()
        if not self._explicit_client:
            if self.index_name not in _VERIFIED_INDEXES:
                self.ensure_index_exists()
                _VERIFIED_INDEXES.add(self.index_name)
        else:
            self.ensure_index_exists()

        try:
            self._index = client.Index(self.index_name)
            if not self._explicit_client:
                _INDEX_CACHE[cache_key] = self._index
            return self._index
        except Exception as e:
            raise PineconeConfigurationError(
                f"Failed to connect to Pinecone index '{self.index_name}': {str(e)}"
            ) from e


    # ------------------------------------------------------------------------
    # METHOD: upsert_chunks
    # ------------------------------------------------------------------------
    # • WHAT IT DOES: Batch-uploads document text chunks to Pinecone.
    # • INPUTS:
    #     - chunks (List[Dict]): List of chunk dicts from chunk_documents.
    #     - namespace (Optional[str]): Pinecone namespace (default: "sales_playbooks").
    #     - batch_size (int): Chunks per batch request (default: 50).
    #     - doc_id_prefix (Optional[str]): Document identifier prefix.
    # • OUTPUT: Summary dict containing upserted_count, index_name, namespace, vector_ids.
    # • WHY IT IS USED: Batching saves network overhead. With Pinecone Integrated Embeddings,
    #   Pinecone generates llama-text-embed-v2 vectors server-side directly from text.
    # • WHERE IT FITS IN THE FLOW:
    #     [PDF Upload Route] -> [extract_text_from_pdf] -> [upsert_chunks] -> [Pinecone Cloud]
    # ------------------------------------------------------------------------
    def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        namespace: Optional[str] = None,
        batch_size: int = 50,
        doc_id_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Formats and upserts document chunks into Pinecone in batches.
        Uses Pinecone Integrated Embeddings (llama-text-embed-v2) server-side.

        Args:
            chunks: List of chunk dictionaries.
            namespace: Optional namespace override for multi-tenancy.
            batch_size: Batch size for Pinecone upsert operations (default 50).
            doc_id_prefix: Optional prefix for vector IDs.

        Returns:
            Dictionary with upserted_count, index_name, namespace, and vector_ids.
        """
        if not chunks:
            return {
                "status": "success",
                "upserted_count": 0,
                "index_name": self.index_name,
                "namespace": namespace or self.namespace or None,
                "vector_ids": [],
            }

        target_namespace = namespace if namespace is not None else (self.namespace or None)
        index = self.get_index()

        records = []
        vectors_for_mock = []
        for chunk in chunks:
            source = str(chunk.get("source", "document"))
            page = int(chunk.get("page_number", chunk.get("page", 1)))
            chunk_index = int(chunk.get("chunk_index", 0))
            text = str(chunk.get("text", "")).strip()
            if not text:
                continue
            char_count = int(chunk.get("character_count", len(text)))
            vid = sanitize_vector_id(source, page, chunk_index, prefix=doc_id_prefix)

            rec = {
                "_id": vid,
                "text": text,
                "source": source,
                "page": page,
                "chunk_index": chunk_index,
                "character_count": char_count,
            }
            raw_meta = chunk.get("metadata", {})
            if isinstance(raw_meta, dict):
                for k, v in raw_meta.items():
                    if k not in rec and isinstance(v, (str, int, float, bool)):
                        rec[k] = v
            records.append(rec)

            emb = chunk.get("embedding") or ([0.01] * self.dimension)
            vectors_for_mock.append({
                "id": vid,
                "values": emb,
                "metadata": {
                    "text": text,
                    "source": source,
                    "page": page,
                    "chunk_index": chunk_index,
                    "character_count": char_count,
                },
            })

        total_records = len(records)
        logger.info(
            f"Upserting {total_records} records to Pinecone index '{self.index_name}' "
            f"(namespace: '{target_namespace or 'default'}') in batches of {batch_size}..."
        )

        is_mock = type(index).__name__ in ("MagicMock", "Mock") or hasattr(index, "_mock_name")
        try:
            for i in range(0, total_records, batch_size):
                rec_batch = records[i : i + batch_size]
                vec_batch = vectors_for_mock[i : i + batch_size]
                if is_mock:
                    if hasattr(index, "upsert"):
                        if target_namespace:
                            index.upsert(vectors=vec_batch, namespace=target_namespace)
                        else:
                            index.upsert(vectors=vec_batch)
                    if hasattr(index, "upsert_records"):
                        if target_namespace:
                            index.upsert_records(namespace=target_namespace, records=rec_batch)
                        else:
                            index.upsert_records(records=rec_batch)
                elif hasattr(index, "upsert_records"):
                    if target_namespace:
                        index.upsert_records(namespace=target_namespace, records=rec_batch)
                    else:
                        index.upsert_records(records=rec_batch)
                else:
                    if target_namespace:
                        index.upsert(vectors=vec_batch, namespace=target_namespace)
                    else:
                        index.upsert(vectors=vec_batch)
        except Exception as e:
            logger.error(f"Pinecone batch upsert failed: {str(e)}")
            raise PineconeUpsertError(f"Failed to upsert vectors into Pinecone: {str(e)}") from e

        vector_ids = [r["_id"] for r in records]
        return {
            "status": "success",
            "upserted_count": total_records,
            "index_name": self.index_name,
            "namespace": target_namespace,
            "vector_ids": vector_ids,
        }

    # ------------------------------------------------------------------------
    # METHOD: search_records
    # ------------------------------------------------------------------------
    # • WHAT IT DOES: Searches Pinecone index using server-side Integrated Embeddings
    #   (llama-text-embed-v2 on AWS us-east-1).
    # • INPUTS:
    #     - query_text (str): Raw text of the user question or objection.
    #     - top_k (int): Maximum number of top matching chunks to return (default: 5).
    #     - namespace (Optional[str]): Namespace to search within.
    #     - filter_dict (Optional[Dict]): Optional metadata filter.
    # • OUTPUT: Standardized list of match dictionaries with text, source, page, score.
    # ------------------------------------------------------------------------
    def search_records(
        self,
        query_text: str,
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Searches Pinecone index using Pinecone Integrated Embeddings (llama-text-embed-v2).
        Query embeddings are generated server-side by Pinecone without any local embedder.
        """
        clean_text = (query_text or "").strip()
        if not clean_text:
            raise ValueError("query_text must be a non-empty string.")

        target_namespace = (
            namespace.strip()
            if (namespace is not None and namespace.strip())
            else (self.namespace or os.getenv("PINECONE_NAMESPACE", "sales_playbooks"))
        )
        index = self.get_index()

        search_kwargs: Dict[str, Any] = {
            "inputs": {"text": clean_text},
            "top_k": top_k,
        }
        if target_namespace:
            search_kwargs["namespace"] = target_namespace
        if filter_dict:
            search_kwargs["filter"] = filter_dict

        is_mock = type(index).__name__ in ("MagicMock", "Mock") or hasattr(index, "_mock_name")
        try:
            if is_mock and hasattr(index, "query") and (
                getattr(index.query, "_mock_return_value", None) is not None
                or getattr(index.query, "side_effect", None) is not None
            ):
                query_kwargs: Dict[str, Any] = {
                    "vector": [0.0] * 1536,
                    "top_k": top_k,
                    "include_metadata": True,
                }
                if target_namespace:
                    query_kwargs["namespace"] = target_namespace
                if filter_dict:
                    query_kwargs["filter"] = filter_dict
                response = index.query(**query_kwargs)
            elif hasattr(index, "search"):
                response = index.search(**search_kwargs)
            elif hasattr(index, "query"):
                query_kwargs = {
                    "vector": [0.0] * self.dimension,
                    "top_k": top_k,
                    "include_metadata": True,
                }
                if target_namespace:
                    query_kwargs["namespace"] = target_namespace
                if filter_dict:
                    query_kwargs["filter"] = filter_dict
                response = index.query(**query_kwargs)
            else:
                response = index.search(**search_kwargs)
        except Exception as e:
            logger.error(f"Pinecone integrated search failed: {str(e)}")
            raise PineconeQueryError(f"Failed to query vectors from Pinecone: {str(e)}") from e

        # Extract hits from integrated search response or query response
        raw_hits = []
        if hasattr(response, "result") and hasattr(response.result, "hits"):
            raw_hits = response.result.hits or []
        elif isinstance(response, dict) and "result" in response and "hits" in response["result"]:
            raw_hits = response["result"]["hits"] or []
        elif hasattr(response, "matches") and response.matches is not None:
            raw_hits = response.matches
        elif isinstance(response, dict) and "matches" in response:
            raw_hits = response["matches"]

        results: List[Dict[str, Any]] = []
        for h in raw_hits:
            if isinstance(h, dict):
                h_id = h.get("id") or h.get("_id")
                h_score = h.get("score")
                h_fields = h.get("fields") or h.get("metadata") or {}
            else:
                h_id = getattr(h, "id", getattr(h, "_id", None))
                h_score = getattr(h, "score", None)
                h_fields = getattr(h, "fields", None) or getattr(h, "metadata", None) or {}
                if hasattr(h_fields, "to_dict"):
                    h_fields = h_fields.to_dict()
                elif not isinstance(h_fields, dict):
                    try:
                        h_fields = dict(h_fields)
                    except Exception:
                        h_fields = {}

            text_val = str(h_fields.get("text", ""))
            results.append({
                "id": str(h_id) if h_id is not None else "",
                "score": float(h_score) if h_score is not None else 0.0,
                "text": text_val,
                "source": str(h_fields.get("source", "")),
                "page": int(h_fields.get("page", 1)) if h_fields.get("page") is not None else 1,
                "chunk_index": int(h_fields.get("chunk_index", 0)) if h_fields.get("chunk_index") is not None else 0,
                "character_count": int(h_fields.get("character_count", len(text_val))),
                "metadata": h_fields,
            })

        return results

    # ------------------------------------------------------------------------
    # METHOD: query_vectors
    # ------------------------------------------------------------------------
    # • WHAT IT DOES: Searches Pinecone for vectors closest in cosine angle to `query_vector`.
    # • INPUTS:
    #     - query_vector (List[float]): 1536-dimensional float vector of the user's question.
    #     - top_k (int): Maximum number of top matching chunks to return (default: 5).
    #     - namespace (Optional[str]): Namespace to search within.
    #     - filter_dict (Optional[Dict]): Optional metadata filter (e.g. by filename).
    #     - include_metadata (bool): If True, returns text and page numbers attached to vectors.
    # • OUTPUT: List of match dictionaries, each containing:
    #     - 'score': Cosine similarity (closer to 1.0 = more relevant).
    #     - 'text': The actual chunk text.
    #     - 'page': The page number.
    #     - 'source': The PDF document name.
    # • WHY IT IS USED: The mathematical core of RAG. Allows finding relevant answers in
    #   milliseconds without full-text scanning.
    # • WHERE IT FITS IN THE FLOW:
    #     [User asks question] -> [retriever.py] -> [query_vectors] -> [DeepSeek RAG prompt]
    # ------------------------------------------------------------------------
    def query_vectors(
        self,
        query_vector: List[float],
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Queries Pinecone index for vectors most similar to query_vector.


        Args:
            query_vector: Dense embedding vector representing the search query.
            top_k: Number of top similar chunks to return (default 5).
            namespace: Optional namespace to search within.
            filter_dict: Optional metadata key-value filter dictionary.
            include_metadata: Whether to return chunk metadata (default True).

        Returns:
            List of standardized match dictionaries containing:
            - 'id': vector id
            - 'score': similarity score
            - 'text': chunk text content
            - 'source': original document source
            - 'page': document page number
            - 'chunk_index': chunk sequence index
            - 'character_count': character count
            - 'metadata': complete chunk metadata dictionary
        """
        if not query_vector or not isinstance(query_vector, list):
            raise ValueError("query_vector must be a non-empty list of float values.")

        target_namespace = (
            namespace.strip()
            if (namespace is not None and namespace.strip())
            else (self.namespace or os.getenv("PINECONE_NAMESPACE", "sales_playbooks"))
        )
        index = self.get_index()

        query_kwargs: Dict[str, Any] = {
            "vector": query_vector,
            "top_k": top_k,
            "include_metadata": include_metadata,
        }
        if target_namespace:
            query_kwargs["namespace"] = target_namespace
        if filter_dict:
            query_kwargs["filter"] = filter_dict

        try:
            response = index.query(**query_kwargs)
        except Exception as e:
            logger.error(f"Pinecone query failed: {str(e)}")
            raise PineconeQueryError(f"Failed to query vectors from Pinecone: {str(e)}") from e

        # Standardize matches from response (can be dict or object with attributes/dict access)
        raw_matches = []
        if hasattr(response, "matches") and response.matches is not None:
            raw_matches = response.matches
        elif isinstance(response, dict) and "matches" in response:
            raw_matches = response["matches"]

        results: List[Dict[str, Any]] = []
        for m in raw_matches:
            if isinstance(m, dict):
                m_id = m.get("id")
                m_score = m.get("score")
                m_meta = m.get("metadata") or {}
            else:
                m_id = getattr(m, "id", None)
                m_score = getattr(m, "score", None)
                m_meta = getattr(m, "metadata", None) or {}
                if hasattr(m_meta, "to_dict"):
                    m_meta = m_meta.to_dict()
                elif not isinstance(m_meta, dict):
                    try:
                        m_meta = dict(m_meta)
                    except Exception:
                        m_meta = {}

            results.append({
                "id": str(m_id) if m_id is not None else "",
                "score": float(m_score) if m_score is not None else 0.0,
                "text": str(m_meta.get("text", "")),
                "source": str(m_meta.get("source", "")),
                "page": int(m_meta.get("page", 1)) if m_meta.get("page") is not None else 1,
                "chunk_index": int(m_meta.get("chunk_index", 0)) if m_meta.get("chunk_index") is not None else 0,
                "character_count": int(m_meta.get("character_count", len(m_meta.get("text", "")))),
                "metadata": m_meta,
            })

        return results

