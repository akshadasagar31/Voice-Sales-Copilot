# ============================================================================
# VECTOR RETRIEVER SERVICE (backend/services/retriever.py)
# ============================================================================
# WHAT THIS SERVICE DOES:
# Connects the user's question to the most relevant sales playbook knowledge.
#
# HOW RETRIEVAL WORKS:
# 1. Takes user question: "What is the minimum CIBIL score for a loan?"
# 2. Vectorizes the question into a 1536-dimensional float vector using TextEmbedder.
# 3. Queries Pinecone vector database to find chunks with high cosine similarity.
# 4. Performs optional hybrid reranking to ensure exact terms match accurately.
# 5. Returns the top-k matching passages with page numbers for DeepSeek LLM context.
# ============================================================================

import logging
from typing import List, Dict, Any, Optional
from services.vector_store import (
    PineconeService,
    PineconeConfigurationError,
    PineconeQueryError,
)
from services.language import (
    translate_indic_query_to_english,
    count_devanagari_chars,
)

logger = logging.getLogger(__name__)



class VectorRetriever:
    """
    Vector similarity search service that utilizes Pinecone's server-side
    Integrated Embeddings (llama-text-embed-v2) to retrieve relevant document chunks.
    No local embedding model is used.
    """

    def __init__(
        self,
        embedder: Optional[Any] = None,
        pinecone_service: Optional[PineconeService] = None,
    ):
        self.pinecone_service = pinecone_service or PineconeService()
        self.embedding_model = "llama-text-embed-v2"
        self.embedding_provider = "pinecone_integrated"

    # ------------------------------------------------------------------------
    # METHOD: retrieve
    # ------------------------------------------------------------------------
    # • WHAT IT DOES: Converts a user query into an embedding, queries Pinecone for the top
    #   matching document chunks, and applies keyword-overlap reranking.
    # • INPUTS:
    #     - query (str): User's sales question or objection text.
    #     - top_k (int): Number of most relevant chunks to return (default: 8).
    #     - namespace (Optional[str]): Namespace in Pinecone to search.
    #     - filter_dict (Optional[Dict]): Metadata filter constraints.
    #     - candidate_k (Optional[int]): Initial pool size for expanded recall before reranking.
    # • OUTPUT: Dictionary containing query, total_results, embedding metadata, and 'results' list.
    # • WHY IT IS USED: Hybrid search: Pure vector similarity sometimes misses exact jargon
    #   or product names. Combining Pinecone vector search with lexical keyword reranking gives
    #   the highest precision for sales and financial policy questions.
    # • WHERE IT FITS IN THE FLOW:
    #     [User asks question] -> [api/ask endpoint] -> [VectorRetriever.retrieve] -> [RAG prompt]
    # ------------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        namespace: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        candidate_k: Optional[int] = None,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes semantic vector search for a given user query:

        1. Validates the query text.
        2. Detects Indic script (Hindi/Marathi) and performs zero-latency cross-lingual translation
           to English search concepts to perfectly match English PDF playbooks in Pinecone.
        3. Queries Pinecone for the most similar chunk vectors using the English search query.
        4. Applies hybrid lexical reranking using merged cross-lingual keywords.
        5. Returns top_k relevant chunks with similarity score, text, and metadata.

        Args:
            query: User search or sales objection query.
            top_k: Maximum number of relevant chunks to retrieve (default 8).
            namespace: Optional Pinecone namespace override.
            filter_dict: Optional metadata filter dict.
            candidate_k: Optional candidate pool size for expanded recall prior to reranking.
            language: Optional target language code ('en', 'hi', 'mr').

        Returns:
            Dictionary containing query metadata and list of retrieved chunk results.
        """
        import os
        clean_query = (query or "").strip()
        if not clean_query:
            raise ValueError("Search query cannot be empty or whitespace.")

        if top_k < 1:
            top_k = 1

        target_namespace = (
            namespace.strip()
            if (namespace is not None and namespace.strip())
            else (self.pinecone_service.namespace or os.getenv("PINECONE_NAMESPACE", "sales_playbooks"))
        )

        fetch_k = candidate_k if candidate_k is not None else top_k

        # Cross-lingual alignment for Hindi / Marathi queries targeting English PDF chunks
        has_devanagari = count_devanagari_chars(clean_query) >= 2 or (language and language.lower().startswith(("hi", "mr")))
        if has_devanagari:
            english_search_query = translate_indic_query_to_english(clean_query, language=language)
            logger.info(f"Cross-lingual query alignment: '{clean_query[:50]}...' -> '{english_search_query}' (namespace={target_namespace})")
        else:
            english_search_query = clean_query
            logger.info(f"Querying Pinecone Integrated Index for: '{clean_query[:80]}...' (top_k={fetch_k}, namespace={target_namespace})")

        # Primary vector search using English-aligned query
        raw_matches = self.pinecone_service.search_records(
            query_text=english_search_query,
            top_k=fetch_k,
            namespace=target_namespace,
            filter_dict=filter_dict,
        )

        # If Indic query, merge any unique matches from original query as well (reciprocal coverage)
        if has_devanagari and english_search_query != clean_query:
            try:
                native_matches = self.pinecone_service.search_records(
                    query_text=clean_query,
                    top_k=fetch_k,
                    namespace=target_namespace,
                    filter_dict=filter_dict,
                )
                seen_ids = {m.get("id") for m in raw_matches if m.get("id")}
                for nm in native_matches:
                    nid = nm.get("id")
                    if nid and nid not in seen_ids:
                        raw_matches.append(nm)
                        seen_ids.add(nid)
            except Exception as ex:
                logger.debug(f"Native match fetch skipped: {ex}")

        # Deduplicate multiple uploads of the exact same PDF chunk to preserve diverse context
        seen_chunk_signatures = set()
        deduped_raw_matches = []
        for m in raw_matches:
            sig = m.get("text", "").strip()[:120]
            if sig and sig not in seen_chunk_signatures:
                seen_chunk_signatures.add(sig)
                deduped_raw_matches.append(m)
            elif not sig:
                deduped_raw_matches.append(m)
        raw_matches = deduped_raw_matches

        # Hybrid lexical reranking: combines Pinecone cosine similarity with query keyword overlap
        if len(raw_matches) > top_k:
            import re
            stop_words = {
                "what", "are", "the", "for", "an", "a", "is", "in", "of", "to", "and", "how",
                "do", "does", "can", "could", "should", "would", "be", "with", "at", "from",
                "by", "about", "as", "into", "like", "through", "after", "over", "between",
                "out", "against", "during", "without", "before", "under", "around", "among",
            }
            # Combine words from both English translation and original query for cross-lingual lexical matching
            q_words = (
                set(re.findall(r"\w+", english_search_query.lower())) |
                set(re.findall(r"\w+", clean_query.lower()))
            ) - stop_words

            scored_matches = []
            for m in raw_matches:
                text_lower = m.get("text", "").lower()
                text_words = set(re.findall(r"\w+", text_lower))
                overlap = (len(q_words & text_words) / len(q_words)) if q_words else 0.0

                kw_boost = 0.0
                for qw in q_words:
                    if len(qw) >= 3 and qw in text_lower:
                        kw_boost += 0.05
                if "mandatory" in q_words and "mandatory" in text_words:
                    kw_boost += 0.25
                if ("document" in q_words or "documents" in q_words) and ("document" in text_lower or "documents" in text_lower):
                    kw_boost += 0.25
                if "age" in q_words and "age" in text_words:
                    kw_boost += 0.20
                if "cibil" in q_words and "cibil" in text_words:
                    kw_boost += 0.20
                if ("rate" in q_words or "interest" in q_words or "roi" in q_words) and ("interest" in text_lower or "roi" in text_lower or "rate" in text_lower):
                    kw_boost += 0.20
                if ("salary" in q_words or "income" in q_words) and ("salary" in text_lower or "income" in text_lower):
                    kw_boost += 0.20

                final_score = float(m.get("score", 0.0)) + (overlap * 0.25) + kw_boost
                m_copy = dict(m)
                m_copy["score"] = final_score
                scored_matches.append((final_score, m_copy))

            scored_matches.sort(key=lambda x: x[0], reverse=True)
            matches = [m for _, m in scored_matches[:top_k]]
        else:
            matches = raw_matches

        return {
            "query": clean_query,
            "top_k": top_k,
            "total_results": len(matches),
            "namespace": target_namespace,
            "embedding_model": self.embedding_model,
            "embedding_provider": self.embedding_provider,
            "results": matches,
        }
