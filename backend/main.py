# ============================================================================
# FASTAPI BACKEND ENTRYPOINT (backend/main.py)
# ============================================================================
# WHAT THIS SERVICE DOES:
# The core Python web server powering all AI, voice, and database capabilities.
# Runs on http://127.0.0.1:8001 and serves the Next.js frontend (port 3000).
#
# MAIN API ROUTES:
# 1. /api/voice-entry: Transcribes audio recordings to text via Deepgram STT.
# 2. /api/extract-lead: Extracts structured Pydantic CRM leads using DeepSeek LLM
#    and streams conversational confirmations via Server-Sent Events (SSE).
# 3. /api/tts: Converts text to speech audio bytes using Deepgram Aura TTS.
# 4. /api/ask: Answers sales playbook questions using Pinecone RAG + DeepSeek.
# 5. /api/upload-pdf: Ingests, chunks, embeds, and indexes PDF playbooks.
# 6. /api/leads: PostgreSQL CRM operations (list, create, update, delete).
# ============================================================================

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Ensure backend/.env is loaded
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
else:
    load_dotenv(override=True)

import io
import re
import wave
import json
import asyncio
import websockets
from pydantic import BaseModel, Field
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Form, WebSocket, WebSocketDisconnect, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse


from services.pdf_extractor import extract_text_from_pdf, PDFExtractionError
from services.embedder import TextEmbedder, embed_chunks
from services.stt import (
    DeepgramSTTService,
    DeepgramConfigurationError,
    DeepgramAPIError,
    build_deepgram_ws_url,
    normalize_stt_transcript,
    get_deepgram_ws_pool,
)
from services.sarvam_stt import (
    SarvamSTTService,
    SarvamSTTError,
    SarvamSTTConfigurationError,
    SarvamSTTAPIError,
)
from services.language import (
    detect_spoken_language,
    get_response_language,
    is_new_lead_intent,
    is_greeting,
    get_greeting_response,
    is_assistant_query,
    get_assistant_query_response,
    LANG_MIXED,
    ENGLISH_GRAMMAR_WORDS,
    ENGLISH_CORE_WORDS,
    MARATHI_WORDS,
    HINDI_WORDS,
    MARATHI_SPECIFIC_CHARS,
    count_devanagari_chars,
    CLARIFICATION_PROMPTS,
    select_best_multilingual_transcript,
)
from services.lead_extractor import (
    Lead,
    LeadExtractionRequest,
    LeadExtractionResponse,
    LeadExtractorService,
    LeadExtractionError,
    LeadValidationError,
    MalformedLLMOutputError,
    OpenRouterConfigurationError as LeadOpenRouterConfigError,
    OpenRouterAPIError as LeadOpenRouterAPIError,
    get_next_missing_parameter,
    get_missing_parameter_prompt,
    extract_phone_number,
    is_valid_phone_number,
    extract_email,
    extract_company,
    extract_loan_type,
    extract_loan_amount,
    extract_tenure_months,
    extract_name,
    is_valid_prospect_name,
    is_valid_company_name,
    is_field_value_valid,
    merge_lead_safely,
    get_clarification_prompt,
    REQUIRED_LEAD_FIELDS,
    is_field_inquiry_or_refusal,
    is_field_refusal,
    is_field_inquiry,
    is_flow_resume_intent,
    is_general_question,
    is_loan_intent,
    is_loan_informational_question,
    classify_turn_intent,
    INTENT_LOAN_APPLICATION,
    INTENT_LOAN_INFO_QUESTION,
    INTENT_GENERIC_QUESTION,
    INTENT_GREETING_OR_INTRO,
    INTENT_FIELD_INQUIRY_OR_REFUSAL,
    INTENT_FLOW_RESUME,
    INTENT_LEAD_DATA,
)
import uuid
from services.lead_repository import (
    LeadRepository,
    DatabaseError,
    DatabaseConfigurationError,
    DatabaseConnectionError,
    DatabaseOperationError,
)
from services.document_repository import DocumentRepository
from services.tts import (
    DeepgramTTSService,
    DeepgramTTSError,
    DeepgramTTSConfigurationError,
    DeepgramTTSAPIError,
    SarvamTTSService,
    SarvamTTSError,
    SarvamTTSConfigurationError,
    SarvamTTSAPIError,
    generate_concise_response,
    get_cached_tts_audio,
    set_cached_tts_audio,
)
from services.sarvam_tts import (
    clean_hindi_financial_text,
    clean_marathi_financial_text,
    clean_english_financial_text,
    get_sarvam_tts_service,
    prewarm_sarvam_client,
)
from services.language import is_greeting, get_greeting_response, detect_language as detect_text_language

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = 48000, channels: int = 1, sampwidth: int = 2) -> bytes:
    """Pack raw 16-bit PCM bytes into standard WAV container in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()

app = FastAPI(
    title="Voice Sales & Knowledge Copilot - Document & Embedding Service",
    description="Module 2: PDF Text Extraction & Dense Vector Embedding Service",
    version="1.1.0",
)

# Configure CORS for Next.js frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=86400,
)


@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint to verify the extraction and embedding backend is running."""
    return {
        "status": "online",
        "service": "Voice Sales Copilot - PDF Extractor & Embedder",
        "version": "1.1.0",
        "module": "Module 2: PDF Extraction & Vector Embeddings",
        "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "pinecone_integrated"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "llama-text-embed-v2"),
        "embedding_dimension": int(os.getenv("EMBEDDING_DIMENSION", "1024")),
    }


# ----------------------------------------------------------------------------
# ROUTE HANDLER: upload_and_extract_pdf (POST /api/upload-pdf, /api/extract-pdf)
# ----------------------------------------------------------------------------
# • WHAT IT DOES: Receives an uploaded sales playbook PDF, extracts text via PyPDFLoader,
#   splits into 600-char chunks, embeds with 1536-dim vectors, saves to PostgreSQL,
#   and optionally upserts vectors into Pinecone.
# • INPUTS:
#     - file (UploadFile): The uploaded PDF file binary.
#     - upsert_to_pinecone (bool): If True, upserts chunks directly into Pinecone.
#     - namespace (Optional[str]): Target Pinecone namespace.
# • OUTPUT: JSON response containing page count, chunk count, text, and Pinecone upsert status.
# • WHY IT IS USED: The automated ingest gateway for Module 3 (Playbook Management).
# • WHERE IT FITS IN THE FLOW:
#     [Sales Rep uploads PDF] -> [POST /api/upload-pdf] -> [Pinecone + PostgreSQL]
# ----------------------------------------------------------------------------
@app.post("/api/extract-pdf", tags=["PDF Extraction"])
@app.post("/api/upload-pdf", tags=["PDF Extraction"], include_in_schema=False)
async def upload_and_extract_pdf(
    file: UploadFile = File(...),
    upsert_to_pinecone: bool = False,
    namespace: Optional[str] = None,
):
    """
    Upload a PDF document, extract text using PyPDFLoader, chunk

    the content using RecursiveCharacterTextSplitter (chunk_size=600, chunk_overlap=100),
    and embed the chunks into dense vector representations.
    Optionally upserts the embedded chunks directly into Pinecone if upsert_to_pinecone=true.

    - Validates file extension is .pdf
    - Safely writes upload to a temporary storage path
    - Extracts text page-by-page with PyPDFLoader
    - Chunks text while preserving page numbers and source metadata
    - Generates dense vector embeddings for each chunk (configurable via .env)
    - Optionally upserts vectors into Pinecone
    - Cleans up temporary files in a finally block
    - Returns structured extraction metadata, raw pages, enriched embedded chunks, and upsert summary
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is missing or invalid.",
        )

    # Validate file extension
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type '{file.filename}'. Only .pdf files are supported.",
        )

    temp_file_path = None
    try:
        # Create a unique temporary file to store uploaded bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            temp_file_path = tmp.name
            content = await file.read()
            if not content:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Uploaded file is empty (0 bytes).",
                )
            tmp.write(content)

        logger.info(f"Extracting text from uploaded PDF: {file.filename} (temp: {temp_file_path})")

        # Persist original binary PDF to PostgreSQL documents table
        doc_repo = DocumentRepository()
        saved_doc = None
        doc_uid = f"doc_{uuid.uuid4().hex[:12]}"
        try:
            saved_doc = doc_repo.create_document(
                filename=file.filename,
                file_bytes=content,
                mime_type=file.content_type or "application/pdf",
                document_uid=doc_uid,
            )
        except Exception as db_err:
            logger.warning(f"PostgreSQL document persistence warning for '{file.filename}': {db_err}")

        # Run extraction through PyPDFLoader with optional Pinecone upsert
        result = extract_text_from_pdf(
            temp_file_path,
            original_filename=file.filename,
            upsert_to_pinecone=upsert_to_pinecone,
            namespace=namespace,
            doc_id_prefix=doc_uid,
        )

        # Update document in PostgreSQL with extracted text and Pinecone vector IDs
        if saved_doc:
            try:
                pinecone_meta = result.get("pinecone_upsert", {})
                pinecone_doc_ids = pinecone_meta.get("vector_ids") or [
                    c.get("id") for c in result.get("chunks", []) if c.get("id")
                ]
                doc_repo.update_document_extraction(
                    document_id=saved_doc["id"],
                    extracted_text=result.get("extracted_text", ""),
                    total_pages=result.get("total_pages", 0),
                    total_characters=result.get("total_characters", 0),
                    total_chunks=result.get("total_chunks", 0),
                    metadata={
                        "chunk_size": result.get("chunk_size", 600),
                        "chunk_overlap": result.get("chunk_overlap", 100),
                        "embedding_model": result.get("embedding_model"),
                        "embedding_dimension": result.get("embedding_dimension"),
                        "embedding_provider": result.get("embedding_provider"),
                    },
                    pinecone_namespace=pinecone_meta.get("namespace") or namespace,
                    pinecone_doc_ids=pinecone_doc_ids,
                    pinecone_upserted_count=pinecone_meta.get("upserted_count", 0),
                )
                result["document_id"] = saved_doc["id"]
                result["document_uid"] = saved_doc["document_uid"]
                result["db_persisted"] = True
                result["file_size"] = len(content)
            except Exception as upd_err:
                logger.warning(f"Failed to update document extraction in DB: {upd_err}")

        return JSONResponse(status_code=status.HTTP_200_OK, content=result)

    except HTTPException:
        raise
    except PDFExtractionError as pe:
        logger.error(f"PDF extraction error for '{file.filename}': {str(pe)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(pe),
        )
    except Exception as e:
        # Check for Pinecone errors if raised during upsert
        from services.vector_store import PineconeConfigurationError, PineconeUpsertError
        if isinstance(e, PineconeConfigurationError):
            logger.error(f"Pinecone configuration error for '{file.filename}': {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e),
            )
        if isinstance(e, PineconeUpsertError):
            logger.error(f"Pinecone upsert error for '{file.filename}': {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(e),
            )

        logger.error(f"Unexpected error processing '{file.filename}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the PDF: {str(e)}",
        )
    finally:
        # Ensure temporary file is always deleted to prevent disk leaks
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError as cleanup_err:
                logger.warning(f"Could not remove temp file {temp_file_path}: {cleanup_err}")


class EmbedChunksRequest(BaseModel):
    chunks: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of document chunk dictionaries to enrich with dense vector embeddings.",
    )
    texts: Optional[List[str]] = Field(
        default=None,
        description="List of raw text strings to embed as dense vectors.",
    )
    query: Optional[str] = Field(
        default=None,
        description="Single query text string to embed as a dense vector.",
    )
    provider: Optional[str] = Field(
        default=None,
        description="Optional provider override ('local_fast' or 'openai').",
    )
    model: Optional[str] = Field(
        default=None,
        description="Optional embedding model override.",
    )
    dimension: Optional[int] = Field(
        default=None,
        description="Optional vector dimension override.",
    )


@app.post("/api/embed-chunks", tags=["Embeddings"])
async def embed_chunks_endpoint(request: EmbedChunksRequest):
    """
    Generate dense vector embeddings for document chunks, raw texts, or query strings.

    - Preserves all chunk metadata (chunk_index, page_number, character_count, source)
    - Returns dense float vector representations with strictly consistent dimensions
    - Supports configurable providers ('local_fast' or 'openai')
    """
    embedder = TextEmbedder(
        provider=request.provider,
        model=request.model,
        dimension=request.dimension,
    )

    response_data: Dict[str, Any] = {
        "embedding_provider": embedder.provider,
        "embedding_model": embedder.model,
        "embedding_dimension": embedder.dimension,
    }

    if request.chunks is not None:
        enriched_chunks = embed_chunks(request.chunks, embedder=embedder)
        response_data["chunks"] = enriched_chunks
        response_data["total_chunks"] = len(enriched_chunks)
    elif request.texts is not None:
        embeddings = embedder.embed_documents(request.texts)
        response_data["embeddings"] = embeddings
        response_data["total_embeddings"] = len(embeddings)
    elif request.query is not None:
        query_embedding = embedder.embed_query(request.query)
        response_data["query"] = request.query
        response_data["embedding"] = query_embedding
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide at least one of 'chunks', 'texts', or 'query' in request body.",
        )

    return JSONResponse(status_code=status.HTTP_200_OK, content=response_data)


class UpsertChunksRequest(BaseModel):
    chunks: List[Dict[str, Any]] = Field(
        ...,
        description="List of document chunk dictionaries to upsert into Pinecone.",
    )
    namespace: Optional[str] = Field(
        default=None,
        description="Optional Pinecone namespace for multi-tenancy.",
    )
    index_name: Optional[str] = Field(
        default=None,
        description="Optional Pinecone index name override.",
    )
    batch_size: Optional[int] = Field(
        default=100,
        description="Batch size for Pinecone vector upsert operations.",
    )
    doc_id_prefix: Optional[str] = Field(
        default=None,
        description="Optional prefix to prepend to chunk vector IDs.",
    )


@app.post("/api/upsert-chunks", tags=["Vector Store"])
async def upsert_chunks_endpoint(request: UpsertChunksRequest):
    """
    Upload and upsert document chunks into Pinecone vector database.

    - Stores vector embedding, text content, and metadata (source, page, chunk_index)
    - Automatically embeds chunks if vector embeddings are missing
    - Batches upsert operations (default 100 vectors per call)
    - Returns upserted count, index name, namespace, and vector IDs
    """
    if not request.chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'chunks' list cannot be empty.",
        )

    try:
        from services.vector_store import (
            PineconeService,
            PineconeConfigurationError,
            PineconeUpsertError,
        )

        service = PineconeService(
            index_name=request.index_name,
            namespace=request.namespace,
        )
        upsert_summary = service.upsert_chunks(
            chunks=request.chunks,
            namespace=request.namespace,
            batch_size=request.batch_size or 100,
            doc_id_prefix=request.doc_id_prefix,
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=upsert_summary)

    except PineconeConfigurationError as pce:
        logger.error(f"Pinecone configuration error: {str(pce)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(pce),
        )
    except PineconeUpsertError as pue:
        logger.error(f"Pinecone upsert error: {str(pue)}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(pue),
        )
    except Exception as e:
        logger.error(f"Unexpected error during Pinecone upsert: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during Pinecone upsert: {str(e)}",
        )


class RetrievalRequest(BaseModel):
    query: str = Field(
        ...,
        description="Search query string or sales objection to retrieve relevant knowledge chunks for.",
    )
    top_k: Optional[int] = Field(
        default=8,
        ge=1,
        le=50,
        description="Number of top similar chunks to return (default 8).",
    )
    namespace: Optional[str] = Field(
        default=None,
        description="Optional Pinecone namespace to search within.",
    )
    filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata key-value filter dictionary.",
    )
    index_name: Optional[str] = Field(
        default=None,
        description="Optional Pinecone index name override.",
    )


@app.post("/api/retrieve", tags=["Vector Store"])
async def retrieve_chunks_endpoint(request: RetrievalRequest):
    """
    Search and retrieve top-k most similar document chunks for a query from Pinecone.

    - Embeds the user query string using the configured embedding model
    - Performs cosine/dense vector similarity search in Pinecone
    - Returns top-k matching chunks with similarity score, text, and metadata
    """
    clean_query = request.query.strip()
    if not clean_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'query' string cannot be empty or whitespace.",
        )

    try:
        from services.retriever import VectorRetriever
        from services.vector_store import (
            PineconeService,
            PineconeConfigurationError,
            PineconeQueryError,
        )

        target_namespace = (
            request.namespace.strip()
            if (request.namespace is not None and request.namespace.strip())
            else os.getenv("PINECONE_NAMESPACE", "sales_playbooks")
        )
        pinecone_service = PineconeService(
            index_name=request.index_name,
            namespace=target_namespace,
        )
        retriever = VectorRetriever(pinecone_service=pinecone_service)
        retrieval_result = retriever.retrieve(
            query=clean_query,
            top_k=request.top_k or 8,
            namespace=target_namespace,
            filter_dict=request.filter,
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=retrieval_result)

    except PineconeConfigurationError as pce:
        logger.error(f"Pinecone configuration error during retrieval: {str(pce)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(pce),
        )
    except PineconeQueryError as pqe:
        logger.error(f"Pinecone query error during retrieval: {str(pqe)}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(pqe),
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        logger.error(f"Unexpected error during vector retrieval: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during vector retrieval: {str(e)}",
        )


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        description="User question to answer using the document knowledge base.",
    )
    top_k: Optional[int] = Field(
        default=8,
        ge=1,
        le=20,
        description="Number of context chunks to retrieve from Pinecone (default 8).",
    )
    namespace: Optional[str] = Field(
        default=None,
        description="Optional Pinecone namespace.",
    )
    filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata key-value filter dictionary.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Optional OpenRouter model override.",
    )
    index_name: Optional[str] = Field(
        default=None,
        description="Optional Pinecone index name override.",
    )
    language: Optional[str] = Field(
        default=None,
        description="Optional language override ('en', 'hi', 'mr'). Automatically detected if omitted.",
    )
    stream: Optional[bool] = Field(
        default=False,
        description="Optional boolean to enable Server-Sent Events (SSE) streaming mode for live tokens.",
    )


# ----------------------------------------------------------------------------
# ROUTE HANDLER: ask_question_endpoint (POST /api/ask)
# ----------------------------------------------------------------------------
# • WHAT IT DOES: Answers sales and credit policy questions using Pinecone RAG + DeepSeek.
#   Supports real-time SSE streaming (`stream: true`) and JSON responses.
# • INPUTS:
#     - request (AskRequest): Payload with:
#         - question: The user's query or objection.
#         - top_k: Number of playbook chunks to retrieve (default: 8).
#         - namespace: Pinecone namespace ("sales_playbooks").
#         - stream: True for real-time SSE token stream.
#         - language: 'en', 'hi', or 'mr'.
# • OUTPUT: StreamingResponse (text/event-stream) or JSONResponse with grounded answer.
# • WHY IT IS USED: The core RAG gateway powering Module 2 Knowledge Assistant.
# • WHERE IT FITS IN THE FLOW:
#     [KnowledgeAssistant.tsx] -> [POST /api/ask] -> [RAGService] -> [SentenceAudioQueue]
# ----------------------------------------------------------------------------
@app.post("/api/ask", tags=["RAG Question Answering"])
async def ask_question_endpoint(request: AskRequest):
    """
    RAG Question Answering:

    - Retrieves top-k relevant chunks from Pinecone using vector similarity
    - Grounds response generation strictly in the retrieved context
    - Queries OpenRouter LLM (default: deepseek/deepseek-flash-latest)
    - Returns exact fallback message if context is insufficient or unavailable
    - Supports real-time SSE streaming mode when stream=True
    """
    clean_question = request.question.strip()
    if not clean_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'question' string cannot be empty or whitespace.",
        )

    try:
        from services.rag import (
            RAGService,
            OpenRouterConfigurationError,
            OpenRouterAPIError,
        )
        from services.retriever import VectorRetriever
        from services.vector_store import (
            PineconeService,
            PineconeConfigurationError,
            PineconeQueryError,
        )

        target_namespace = (
            request.namespace.strip()
            if (request.namespace is not None and request.namespace.strip())
            else os.getenv("PINECONE_NAMESPACE", "sales_playbooks")
        )
        pinecone_service = PineconeService(
            index_name=request.index_name,
            namespace=target_namespace,
        )
        retriever = VectorRetriever(pinecone_service=pinecone_service)
        rag_service = RAGService(retriever=retriever)

        # Resolve dynamic active LLM model if none specified
        active_llm = None
        try:
            from services.model_manager import get_active_model_id
            active_llm = get_active_model_id("llm")
        except Exception:
            pass
        effective_model = request.model or active_llm

        # If streaming mode is requested, return real-time Server-Sent Events (SSE)
        if request.stream:
            return StreamingResponse(
                rag_service.stream_answer_chunks(
                    question=clean_question,
                    top_k=request.top_k or 8,
                    namespace=target_namespace,
                    filter_dict=request.filter,
                    model=effective_model,
                    language=request.language,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        answer_result = rag_service.answer_question(
            question=clean_question,
            top_k=request.top_k or 8,
            namespace=target_namespace,
            filter_dict=request.filter,
            model=effective_model,
            language=request.language,
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=answer_result)

    except OpenRouterConfigurationError as oce:
        logger.error(f"OpenRouter configuration error: {str(oce)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(oce),
        )
    except PineconeConfigurationError as pce:
        logger.error(f"Pinecone configuration error during RAG: {str(pce)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(pce),
        )
    except OpenRouterAPIError as oae:
        logger.error(f"OpenRouter API error: {str(oae)}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(oae),
        )
    except PineconeQueryError as pqe:
        logger.error(f"Pinecone query error during RAG: {str(pqe)}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(pqe),
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        logger.error(f"Unexpected error during RAG question answering: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during RAG question answering: {str(e)}",
        )


# ----------------------------------------------------------------------------
# ROUTE HANDLER: voice_entry (POST /api/voice-entry)
# ----------------------------------------------------------------------------
# • WHAT IT DOES: Receives an uploaded audio file (WAV / WebM) from the frontend
#   and transcribes it into text using Deepgram STT (Nova-2 / Nova-3).
# • INPUTS:
#     - file / audio_file (Optional[UploadFile]): Binary audio from microphone recording.
#     - language (Optional[str]): Optional language hint ('en', 'hi', 'mr').
# • OUTPUT: JSON response containing 'transcript', 'detected_language', 'duration', etc.
# • WHY IT IS USED: The voice input gateway for both Module 1 and Module 2.
# • WHERE IT FITS IN THE FLOW:
#     [Browser Microphone] -> [POST /api/voice-entry] -> [DeepgramSTTService] -> [Transcribed Text]
# ----------------------------------------------------------------------------
@app.post("/api/voice-entry", tags=["Voice-to-CRM"])
async def voice_entry(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    audio_file: Optional[UploadFile] = File(None),
    language: Optional[str] = Form(None),
    module: Optional[str] = Form(None),
    existing_lead: Optional[str] = Form(None),
    lead_id: Optional[int] = Form(None),
    is_interim: Optional[bool] = Form(None),
    is_final: Optional[bool] = Form(None),
):
    """
    Accepts recorded voice note audio (Blob/File), transcribes to text via Sarvam (for Module 2 Hindi)
    or Deepgram Nova-3 (for English, Marathi, Module 1, and fallback).
    """
    # CRITICAL: Ignore interim / partial STT requests immediately
    if is_interim is True or is_final is False:
        logger.info("[voice-entry] Interim STT input ignored.")
        clean_lead = {}
        if existing_lead:
            try:
                clean_lead = json.loads(existing_lead) if isinstance(existing_lead, str) else dict(existing_lead)
            except Exception:
                clean_lead = {}
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "interim_ignored",
                "is_interim": True,
                "transcript": "",
                "lead": clean_lead,
                "lead_id": lead_id,
                "next_missing_parameter": get_next_missing_parameter(clean_lead),
                "is_complete": False,
            },
        )

    upload = file or audio_file
    if not upload or not upload.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No audio file was provided in the upload request.",
        )

    try:
        content = await upload.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded audio file is empty (0 bytes).",
            )

        # Resolve content type reliably
        content_type = upload.content_type
        if not content_type or content_type in ("application/octet-stream", "binary/octet-stream"):
            filename_lower = (upload.filename or "").lower()
            if filename_lower.endswith(".webm"):
                content_type = "audio/webm"
            elif filename_lower.endswith(".wav"):
                content_type = "audio/wav"
            elif filename_lower.endswith(".ogg") or filename_lower.endswith(".opus"):
                content_type = "audio/ogg"
            elif filename_lower.endswith(".mp4") or filename_lower.endswith(".m4a"):
                content_type = "audio/mp4"
            elif filename_lower.endswith(".mp3") or filename_lower.endswith(".mpeg"):
                content_type = "audio/mpeg"
            else:
                content_type = "audio/webm"

        logger.info(
            f"[voice-entry] Received audio upload: filename='{upload.filename}', size={len(content)} bytes, "
            f"upload_content_type='{upload.content_type}', resolved_mime='{content_type}', language='{language}', module='{module}'"
        )

        req_lang = (language or "").strip().lower()
        if req_lang in ("mr", "mr-in", "marathi"):
            req_lang = "mr"
        elif req_lang in ("hi", "hi-in", "hindi"):
            req_lang = "hi"
        elif req_lang in ("en", "en-in", "en-us", "english"):
            req_lang = "en"

        is_hindi = req_lang == "hi"
        is_marathi = req_lang == "mr"
        is_english = req_lang == "en"
        mod_lower = (module or "").strip().lower()
        is_module2 = mod_lower in ("module2", "knowledge_assistant", "rag")
        is_module1 = mod_lower in ("module1", "crm", "voice_copilot", "voice-copilot")

        # For Module 1 (Voice CRM Copilot): route to Deepgram Nova-3 for sub-second, instant voice replies
        # Preserves exact spoken words, names, numbers, Marathi/Hindi/English verbatim without translation, rewriting, or guessing.
        if is_module1:
            dg_lang = None
            if is_english:
                dg_lang = "en-IN"
            elif is_hindi:
                dg_lang = "hi"
            elif is_marathi:
                dg_lang = "mr"
            else:
                dg_lang = None

            # Resolve dynamic active STT model
            active_stt = get_model_manager().get_active_model("stt") or {}
            stt_model_id = (active_stt.get("model_id") or "nova-3").strip()
            stt_provider = (active_stt.get("provider") or "").lower()
            stt_api_key = active_stt.get("api_key")

            if stt_provider in ("sarvam ai", "sarvam") or "saaras" in stt_model_id:
                sarvam_stt = SarvamSTTService(api_key=stt_api_key) if stt_api_key else SarvamSTTService()
                sarvam_lang = "mr-IN" if is_marathi else ("hi-IN" if is_hindi else "en-IN")
                result = await sarvam_stt.transcribe_audio(
                    content,
                    content_type=content_type,
                    language_code=sarvam_lang,
                    model=stt_model_id if "saaras" in stt_model_id else "saaras:v4",
                    filename=upload.filename,
                )
            else:
                stt_service = DeepgramSTTService(api_key=stt_api_key) if stt_api_key else DeepgramSTTService()
                result = await stt_service.transcribe_audio(
                    content,
                    content_type=content_type,
                    model=stt_model_id,
                    language=dg_lang,
                )
            result["filename"] = upload.filename
            # Log raw final transcript before any processing
            raw_transcript = result.get("transcript", "")
            logger.info(
                f"[voice-entry:module1] Raw {stt_model_id} final transcript (dg_lang={dg_lang}): '{raw_transcript}'"
            )
            # Transcript already normalized by transcribe_audio; no double normalization needed
            result["transcript"] = raw_transcript
            result["stt_provider"] = stt_provider or "deepgram"
            result["model"] = "nova-3"
            greeting_flag = is_greeting(raw_transcript)
            result["is_greeting"] = greeting_flag

            # Authoritative language resolution for Module 1 from actual spoken transcript
            spoken_lang = detect_spoken_language(raw_transcript, requested_language=req_lang)
            resp_lang = get_response_language(spoken_lang, requested_language=req_lang, text=raw_transcript)
            result["detected_language"] = spoken_lang
            result["language"] = resp_lang

            if greeting_flag:
                greeting_text = get_greeting_response(resp_lang)
                result["greeting_response"] = greeting_text
                result["immediate_sentence1"] = greeting_text
                result["is_greeting"] = True
            elif raw_transcript:
                try:
                    clean_existing = {}
                    if existing_lead:
                        try:
                            clean_existing = json.loads(existing_lead) if isinstance(existing_lead, str) else dict(existing_lead)
                        except Exception:
                            clean_existing = {}

                    # Check if user spoke intent to start a new lead / reset, or if prior lead was complete
                    is_prior_complete = get_next_missing_parameter(clean_existing) is None and bool(clean_existing)
                    is_reset = is_new_lead_intent(raw_transcript) or is_prior_complete
                    if is_reset:
                        clean_existing = {}
                        lead_id = None
                        result["is_new_lead"] = True

                    prior_missing = get_next_missing_parameter(clean_existing)

                    # 1. Flow Resume: user voluntarily continues a paused flow
                    if prior_missing and is_flow_resume_intent(raw_transcript):
                        cont_prompt = get_missing_parameter_prompt(prior_missing, clean_existing, lang=resp_lang)
                        result["is_resumed"] = True
                        result["immediate_sentence1"] = cont_prompt
                        result["lead"] = clean_existing
                        result["lead_id"] = lead_id
                        result["next_missing_parameter"] = prior_missing
                        result["is_complete"] = False
                        logger.info(
                            f"[voice-entry:module1] Flow resumed on '{prior_missing}': '{raw_transcript}' -> '{cont_prompt}'"
                        )
                        return JSONResponse(status_code=status.HTTP_200_OK, content=result)

                    # 2. Refusal / Pause: user refuses to provide a requested field or says "don't proceed"
                    if prior_missing and is_field_refusal(raw_transcript, prior_missing):
                        extractor = LeadExtractorService()
                        refusal_ack = extractor.handle_field_refusal(
                            raw_transcript, prior_missing, existing_lead=clean_existing, language=resp_lang
                        )
                        result["is_field_refusal"] = True
                        result["is_paused"] = True
                        result["is_field_inquiry"] = True
                        result["is_assistant_query"] = True
                        result["assistant_response"] = refusal_ack
                        result["immediate_sentence1"] = refusal_ack
                        result["lead"] = clean_existing
                        result["lead_id"] = lead_id
                        result["next_missing_parameter"] = prior_missing
                        result["is_complete"] = False
                        logger.info(
                            f"[voice-entry:module1] Field refusal/pause handled on '{prior_missing}': '{raw_transcript}' -> '{refusal_ack}'"
                        )
                        return JSONResponse(status_code=status.HTTP_200_OK, content=result)

                    # 3. Field Inquiry: user asks why the field is required or expresses privacy concern
                    if prior_missing and is_field_inquiry(raw_transcript, prior_missing):
                        extractor = LeadExtractorService()
                        explanation = extractor.explain_field_requirement(
                            raw_transcript, prior_missing, existing_lead=clean_existing, language=resp_lang
                        )
                        result["is_field_inquiry"] = True
                        result["is_assistant_query"] = True
                        result["assistant_response"] = explanation
                        result["immediate_sentence1"] = explanation
                        result["lead"] = clean_existing
                        result["lead_id"] = lead_id
                        result["next_missing_parameter"] = prior_missing
                        result["is_complete"] = False
                        logger.info(
                            f"[voice-entry:module1] Field inquiry handled on '{prior_missing}': '{raw_transcript}' -> '{explanation}'"
                        )
                        return JSONResponse(status_code=status.HTTP_200_OK, content=result)

                    # 4. General / Random Question: send to CURRENT ACTIVE LLM and never hardcode answers
                    if is_general_question(raw_transcript, context_field=prior_missing):
                        extractor = LeadExtractorService()
                        has_active_lead = bool(clean_existing) and any(
                            clean_existing.get(f) is not None and str(clean_existing.get(f)).strip() != ""
                            for f in REQUIRED_LEAD_FIELDS
                        )
                        is_asst_ident = is_assistant_query(raw_transcript)
                        pending_to_resume = prior_missing if (has_active_lead or is_asst_ident) else None
                        raw_llm = extractor.answer_general_query(
                            raw_transcript,
                            language=resp_lang,
                            existing_lead=clean_existing,
                            pending_field=None,
                        )
                        cont_prompt = (
                            get_missing_parameter_prompt(pending_to_resume, clean_existing, lang=resp_lang)
                            if pending_to_resume
                            else ""
                        )
                        assistant_answer = (
                            f"{raw_llm} {cont_prompt}".strip()
                            if (cont_prompt and cont_prompt not in raw_llm)
                            else raw_llm
                        )
                        result["is_assistant_query"] = True
                        result["is_general_query"] = True
                        result["assistant_response"] = raw_llm
                        result["immediate_sentence1"] = assistant_answer
                        result["lead"] = clean_existing
                        result["lead_id"] = lead_id
                        result["next_missing_parameter"] = pending_to_resume
                        result["is_complete"] = False
                        logger.info(
                            f"[voice-entry:module1] General question handled via active LLM: '{raw_transcript}' -> '{assistant_answer}'"
                        )
                        return JSONResponse(status_code=status.HTTP_200_OK, content=result)

                    # 5. Loan / Lead Intent on fresh turn: start existing lead flow and ask first missing field
                    if not clean_existing and is_loan_intent(raw_transcript):
                        immediate_lead = {}
                        phone_found = extract_phone_number(raw_transcript)
                        if phone_found:
                            immediate_lead["phone"] = phone_found
                        email_found = extract_email(raw_transcript)
                        if email_found:
                            immediate_lead["email"] = email_found
                        comp_found = extract_company(raw_transcript)
                        if comp_found:
                            immediate_lead["company"] = comp_found
                        lt_found = extract_loan_type(raw_transcript)
                        if lt_found:
                            immediate_lead["loan_type"] = lt_found
                        amt_found = extract_loan_amount(raw_transcript)
                        if amt_found:
                            immediate_lead["loan_amount"] = amt_found
                        tenure_found = extract_tenure_months(raw_transcript)
                        if tenure_found:
                            immediate_lead["tenure_months"] = tenure_found
                        name_found = extract_name(raw_transcript)
                        if name_found and is_valid_prospect_name(name_found):
                            immediate_lead["name"] = name_found

                        next_missing = get_next_missing_parameter(immediate_lead)
                        prompt_msg = get_missing_parameter_prompt(next_missing, immediate_lead, lang=resp_lang)
                        result["is_new_lead"] = True
                        result["immediate_sentence1"] = prompt_msg
                        result["lead"] = immediate_lead
                        result["lead_id"] = None
                        result["next_missing_parameter"] = next_missing
                        result["is_complete"] = next_missing is None
                        logger.info(
                            f"[voice-entry:module1] Loan intent started on fresh session: '{raw_transcript}' -> '{prompt_msg}'"
                        )
                        return JSONResponse(status_code=status.HTTP_200_OK, content=result)

                    immediate_lead = dict(clean_existing)

                    phone_found = extract_phone_number(raw_transcript)
                    if phone_found:
                        immediate_lead["phone"] = phone_found
                    email_found = extract_email(raw_transcript)
                    if email_found:
                        immediate_lead["email"] = email_found
                    comp_found = extract_company(raw_transcript, context_field=prior_missing)
                    if comp_found:
                        immediate_lead["company"] = comp_found
                    lt_found = extract_loan_type(raw_transcript, context_field=prior_missing)
                    if lt_found:
                        immediate_lead["loan_type"] = lt_found
                    amt_found = extract_loan_amount(raw_transcript, context_field=prior_missing)
                    if amt_found:
                        immediate_lead["loan_amount"] = amt_found
                    tenure_found = extract_tenure_months(raw_transcript, context_field=prior_missing)
                    if tenure_found:
                        immediate_lead["tenure_months"] = tenure_found
                    name_found = extract_name(raw_transcript, context_field=prior_missing)
                    if name_found and is_valid_prospect_name(name_found):
                        immediate_lead["name"] = name_found

                    for k, v in clean_existing.items():
                        if immediate_lead.get(k) is None and v is not None:
                            immediate_lead[k] = v

                    # Check if prior requested parameter was not satisfied by the speech turn
                    was_unclear = False
                    if prior_missing and prior_missing in REQUIRED_LEAD_FIELDS:
                        prior_val = immediate_lead.get(prior_missing)
                        prior_satisfied = prior_val is not None and (not isinstance(prior_val, str) or bool(prior_val.strip()))
                        newly_collected_any = any(
                            immediate_lead.get(f) is not None and clean_existing.get(f) is None
                            for f in REQUIRED_LEAD_FIELDS
                        )
                        if not prior_satisfied and not newly_collected_any:
                            was_unclear = True

                    next_missing = get_next_missing_parameter(immediate_lead)
                    if was_unclear:
                        immediate_sentence1 = get_clarification_prompt(prior_missing, raw_transcript, immediate_lead, lang=resp_lang)
                    else:
                        immediate_sentence1 = get_missing_parameter_prompt(next_missing, immediate_lead, lang=resp_lang)

                    result["immediate_sentence1"] = immediate_sentence1
                    result["lead"] = immediate_lead
                    result["next_missing_parameter"] = next_missing
                    result["is_complete"] = next_missing is None
                    result["is_unclear"] = was_unclear
                    result["clarified_field"] = prior_missing if was_unclear else None

                    # Non-blocking database sync: update lead in background when lead_id exists;
                    # on initial turn, create lead using async thread pool to establish ID without blocking
                    has_data = any(v is not None for v in immediate_lead.values() if v != "")
                    if has_data:
                        try:
                            repo = get_shared_lead_repo()
                            val_lead = Lead.model_validate(immediate_lead)
                            if lead_id is not None:
                                background_tasks.add_task(repo.update_lead, lead_id, val_lead)
                            else:
                                created = await asyncio.to_thread(repo.create_lead, val_lead)
                                if created and "id" in created:
                                    lead_id = created["id"]
                        except Exception as persist_err:
                            logger.warning(f"[voice-entry:persist] {persist_err}")

                    result["lead_id"] = lead_id
                except Exception as extract_err:
                    logger.warning(f"[voice-entry:deterministic_extract] {extract_err}")

            logger.info(
                f"[voice-entry:module1] Deepgram Nova-3 ({dg_lang}) transcript: '{raw_transcript[:80]}' (spoken: {spoken_lang}, response: {resp_lang})"
            )
            return JSONResponse(status_code=status.HTTP_200_OK, content=result)

        # For Module 2 (Knowledge Assistant / RAG): Route directly to Deepgram Nova-3 streaming STT for instant voice response (<1s)
        if is_module2:
            dg_lang = "multi"
            if is_english:
                dg_lang = "en-IN"
            elif is_hindi:
                dg_lang = "hi"
            elif is_marathi:
                dg_lang = "mr"

            # Resolve dynamic active STT model
            active_stt = get_model_manager().get_active_model("stt") or {}
            stt_model_id = (active_stt.get("model_id") or "nova-3").strip()
            stt_provider = (active_stt.get("provider") or "").lower()
            stt_api_key = active_stt.get("api_key")

            if stt_provider in ("sarvam ai", "sarvam") or "saaras" in stt_model_id:
                sarvam_stt = SarvamSTTService(api_key=stt_api_key) if stt_api_key else SarvamSTTService()
                sarvam_lang = "mr-IN" if is_marathi else ("hi-IN" if is_hindi else "en-IN")
                result = await sarvam_stt.transcribe_audio(
                    content,
                    content_type=content_type,
                    language_code=sarvam_lang,
                    model=stt_model_id if "saaras" in stt_model_id else "saaras:v4",
                    filename=upload.filename,
                )
            else:
                deepgram_service = DeepgramSTTService(api_key=stt_api_key) if stt_api_key else DeepgramSTTService()
                result = await deepgram_service.transcribe_audio(
                    content,
                    content_type=content_type,
                    language=dg_lang,
                    model=stt_model_id,
                )
            result["filename"] = upload.filename
            raw_transcript = (result.get("transcript") or "").strip()
            detected = result.get("detected_language") or (dg_lang if dg_lang != "multi" else "en")
            if detected in ("en-in", "en-us", "en-gb", "english"):
                detected = "en"
            result["detected_language"] = detected

            greeting_flag = is_greeting(raw_transcript)
            result["is_greeting"] = greeting_flag
            if greeting_flag:
                result["greeting_response"] = get_greeting_response(detected)

            logger.info(
                f"[voice-entry:module2] Deepgram Nova-3 ({dg_lang}) transcript: '{raw_transcript[:80]}' (lang: {detected})"
            )
            return JSONResponse(status_code=status.HTTP_200_OK, content=result)


        # For Hindi queries when module is unspecified: route to Sarvam saaras:v4 for authentic Devanagari Hindi transcription
        elif is_hindi and not module:
            try:
                sarvam_stt = SarvamSTTService()
                result = await sarvam_stt.transcribe_audio(
                    content,
                    content_type=content_type,
                    language_code="hi-IN",
                    model="saaras:v4",
                    filename=upload.filename,
                )
                result["filename"] = upload.filename
                raw_transcript = result.get("transcript", "")
                greeting_flag = is_greeting(raw_transcript)
                result["is_greeting"] = greeting_flag
                if greeting_flag:
                    result["greeting_response"] = get_greeting_response("hi")
                logger.info(f"[voice-entry] Sarvam saaras:v4 Hindi transcript: '{raw_transcript[:80]}...'")
                return JSONResponse(status_code=status.HTTP_200_OK, content=result)
            except (SarvamSTTConfigurationError, SarvamSTTAPIError, Exception) as sarvam_err:
                logger.warning(
                    f"Sarvam Hindi STT failed ({str(sarvam_err)}). Falling back to Deepgram Nova-3."
                )
                # Fall through to DeepgramSTTService below

        stt_service = DeepgramSTTService()
        # Nova-3 delivers lowest real-world latency, auto-detection, and full EN/HI/MR support
        stt_model = "nova-3"
        result = await stt_service.transcribe_audio(
            content,
            content_type=content_type,
            model=stt_model,
            language=language,
        )
        result["filename"] = upload.filename
        raw_transcript = result.get("transcript", "")
        result["transcript"] = raw_transcript

        if is_module1:
            raw_detected = (result.get("detected_language") or "").strip().lower()
            text_lang = detect_text_language(raw_transcript)
            # Prioritize text-based detection from actual spoken words over acoustic classifier
            if req_lang in ("en", "hi", "mr"):
                detected = req_lang
            elif text_lang in ("en", "hi", "mr"):
                detected = text_lang
            elif raw_detected in ("en-in", "en-us", "en"):
                detected = "en"
            elif raw_detected in ("hi-in", "hi"):
                detected = "hi"
            elif raw_detected in ("mr-in", "mr"):
                detected = "mr"
            else:
                detected = "en"
            result["detected_language"] = detected



        greeting_flag = is_greeting(raw_transcript)
        result["is_greeting"] = greeting_flag
        if greeting_flag:
            detected = result.get("detected_language") or language or "en"
            result["greeting_response"] = get_greeting_response(detected)
        return JSONResponse(status_code=status.HTTP_200_OK, content=result)

    except DeepgramConfigurationError as dce:
        logger.error(f"Deepgram configuration error: {str(dce)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(dce),
        )
    except DeepgramAPIError as dae:
        logger.error(f"Deepgram API error (HTTP {dae.status_code}): {str(dae)}")
        http_code = status.HTTP_503_SERVICE_UNAVAILABLE if dae.status_code == 503 else status.HTTP_502_BAD_GATEWAY
        raise HTTPException(
            status_code=http_code,
            detail=str(dae),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during audio transcription: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during audio transcription: {str(e)}",
        )


_lead_repo: Optional[LeadRepository] = None

def get_shared_lead_repo() -> LeadRepository:
    """Returns singleton LeadRepository instance with pre-warmed connection pool."""
    global _lead_repo
    if _lead_repo is None:
        _lead_repo = LeadRepository()
    return _lead_repo


def generate_module1_ws_response(
    current: str,
    req_lang: str,
    last_conf: float,
    existing_lead_str: Optional[str] = None,
    lead_id_val: Optional[int] = None,
) -> dict:
    """
    Sub-millisecond deterministic extraction and prompt generator for Module 1 WebSocket streaming.
    Returns the immediate sentence 1, lead object, and next missing parameter directly in the final event.
    """
    req_clean = (req_lang or "").strip().lower()
    if req_clean in ("mr", "mr-in", "marathi"):
        req_clean = "mr"
    elif req_clean in ("hi", "hi-in", "hindi"):
        req_clean = "hi"
    elif req_clean in ("en", "en-in", "en-us", "english"):
        req_clean = "en"

    # Evaluate language strictly from current turn transcript, never inheriting previous assistant/user language
    detected_turn_lang = detect_spoken_language(current, requested_language=req_clean, confidence=last_conf)
    lang = detected_turn_lang if detected_turn_lang in ("en", "hi", "mr") else (req_clean if req_clean in ("en", "hi", "mr") else "en")
    resp_lang = get_response_language(lang, requested_language=lang, text=current)
    if lang not in ("en", "hi", "mr"):
        lang = "en"
    if resp_lang not in ("en", "hi", "mr"):
        resp_lang = "en"

    is_greet = is_greeting(current)
    greet_resp = get_greeting_response(resp_lang) if is_greet else ""
    asst_resp = get_assistant_query_response(current, resp_lang, allow_llm=False)

    clean_existing = {}
    if existing_lead_str:
        try:
            clean_existing = json.loads(existing_lead_str) if isinstance(existing_lead_str, str) else dict(existing_lead_str)
        except Exception:
            clean_existing = {}

    is_prior_complete = get_next_missing_parameter(clean_existing) is None and bool(clean_existing)
    is_reset = is_new_lead_intent(current) or is_prior_complete
    if is_reset:
        clean_existing = {}
        lead_id_val = None

    prior_missing = get_next_missing_parameter(clean_existing)

    # 1. Flow Resume: user voluntarily continues a paused flow
    if prior_missing and is_flow_resume_intent(current):
        cont_prompt = get_missing_parameter_prompt(prior_missing, clean_existing, lang=resp_lang)
        return {
            "type": "final",
            "transcript": current,
            "confidence": round(last_conf, 4),
            "detected_language": lang,
            "language": resp_lang,
            "is_greeting": False,
            "greeting_response": "",
            "is_assistant_query": False,
            "is_field_inquiry": False,
            "is_field_refusal": False,
            "is_resumed": True,
            "assistant_response": "",
            "immediate_sentence1": cont_prompt,
            "lead": clean_existing,
            "lead_id": lead_id_val,
            "next_missing_parameter": prior_missing,
            "is_complete": False,
            "is_new_lead": False,
            "is_unclear": False,
        }

    # 2. Refusal / Pause: user refuses to provide a requested field or says "don't proceed"
    if prior_missing and is_field_refusal(current, prior_missing):
        extractor = LeadExtractorService()
        refusal_ack = extractor.handle_field_refusal(
            current, prior_missing, existing_lead=clean_existing, language=resp_lang
        )
        return {
            "type": "final",
            "transcript": current,
            "confidence": round(last_conf, 4),
            "detected_language": lang,
            "language": resp_lang,
            "is_greeting": False,
            "greeting_response": "",
            "is_assistant_query": True,
            "is_field_inquiry": True,
            "is_field_refusal": True,
            "is_paused": True,
            "assistant_response": refusal_ack,
            "immediate_sentence1": refusal_ack,
            "lead": clean_existing,
            "lead_id": lead_id_val,
            "next_missing_parameter": prior_missing,
            "is_complete": False,
            "is_new_lead": False,
            "is_unclear": False,
        }

    # 3. Field Inquiry: user asks why the field is required or expresses privacy concern
    if prior_missing and is_field_inquiry(current, prior_missing):
        extractor = LeadExtractorService()
        field_explanation = extractor.explain_field_requirement(
            current, prior_missing, existing_lead=clean_existing, language=resp_lang
        )
        return {
            "type": "final",
            "transcript": current,
            "confidence": round(last_conf, 4),
            "detected_language": lang,
            "language": resp_lang,
            "is_greeting": False,
            "greeting_response": "",
            "is_assistant_query": True,
            "is_field_inquiry": True,
            "is_field_refusal": False,
            "assistant_response": field_explanation,
            "immediate_sentence1": field_explanation,
            "lead": clean_existing,
            "lead_id": lead_id_val,
            "next_missing_parameter": prior_missing,
            "is_complete": False,
            "is_new_lead": False,
            "is_unclear": False,
        }

    # 4. General / Random Question / Loan Info Question: send to CURRENT ACTIVE LLM and never hardcode answers
    if is_general_question(current, context_field=prior_missing) or is_loan_informational_question(current):
        extractor = LeadExtractorService()
        has_active_lead = bool(clean_existing) and any(
            clean_existing.get(f) is not None and str(clean_existing.get(f)).strip() != ""
            for f in REQUIRED_LEAD_FIELDS
        )
        pending_to_resume = prior_missing if has_active_lead else None
        llm_response = extractor.answer_general_query(
            current,
            language=resp_lang,
            existing_lead=clean_existing if has_active_lead else {},
            pending_field=pending_to_resume,
        )
        cont_prompt = (
            get_missing_parameter_prompt(pending_to_resume, clean_existing, lang=resp_lang)
            if pending_to_resume
            else ""
        )
        full_response = (
            f"{llm_response} {cont_prompt}".strip()
            if (cont_prompt and cont_prompt not in llm_response)
            else llm_response
        )
        return {
            "type": "final",
            "transcript": current,
            "confidence": round(last_conf, 4),
            "detected_language": lang,
            "language": resp_lang,
            "is_greeting": False,
            "greeting_response": "",
            "is_assistant_query": True,
            "is_general_query": True,
            "llm_unavailable": False,
            "is_field_inquiry": False,
            "is_field_refusal": False,
            "is_resumed": False,
            "assistant_response": llm_response,
            "immediate_sentence1": full_response,
            "lead": clean_existing if has_active_lead else {},
            "lead_id": lead_id_val if has_active_lead else None,
            "next_missing_parameter": pending_to_resume,
            "is_complete": False,
            "is_new_lead": False,
            "is_unclear": False,
            "speech_final": True,
        }

    # 5. Loan / Lead Intent on fresh turn: start existing lead flow
    if not clean_existing and is_loan_intent(current):
        immediate_lead = {}
        phone_found = extract_phone_number(current)
        if phone_found and is_valid_phone_number(phone_found):
            immediate_lead["phone"] = phone_found
        email_found = extract_email(current)
        if email_found:
            immediate_lead["email"] = email_found
        comp_found = extract_company(current)
        if comp_found and is_valid_company_name(comp_found):
            immediate_lead["company"] = comp_found
        lt_found = extract_loan_type(current)
        if lt_found:
            immediate_lead["loan_type"] = lt_found
        amt_found = extract_loan_amount(current)
        if amt_found and is_field_value_valid("loan_amount", amt_found):
            immediate_lead["loan_amount"] = amt_found
        tenure_found = extract_tenure_months(current)
        if tenure_found and is_field_value_valid("tenure_months", tenure_found):
            immediate_lead["tenure_months"] = tenure_found
        name_found = extract_name(current)
        if name_found and is_valid_prospect_name(name_found):
            immediate_lead["name"] = name_found

        next_missing = get_next_missing_parameter(immediate_lead)
        prompt_msg = get_missing_parameter_prompt(next_missing, immediate_lead, lang=resp_lang)
        return {
            "type": "final",
            "transcript": current,
            "confidence": round(last_conf, 4),
            "detected_language": lang,
            "language": resp_lang,
            "is_greeting": False,
            "greeting_response": "",
            "is_assistant_query": False,
            "is_field_inquiry": False,
            "is_field_refusal": False,
            "is_resumed": False,
            "assistant_response": "",
            "immediate_sentence1": prompt_msg,
            "lead": immediate_lead,
            "lead_id": None,
            "next_missing_parameter": next_missing,
            "is_complete": next_missing is None,
            "is_new_lead": True,
            "is_unclear": False,
            "speech_final": True,
        }

    # 6. If no active lead exists and neither loan intent nor name was expressed:
    # Do NOT start lead collection.
    # Greet user or answer as general query using active LLM.
    has_active_lead = bool(clean_existing) and any(
        clean_existing.get(f) is not None and str(clean_existing.get(f)).strip() != ""
        for f in REQUIRED_LEAD_FIELDS
    )
    if not has_active_lead and not is_loan_intent(current):
        has_greet_word = bool(
            re.search(
                r"\b(?:hello|hi|hey|greetings|good\s+(?:morning|afternoon|evening|day)|namaste|namaskar|pranam|halo|नमस्ते|नमस्कार|प्रणाम|हाय|हैलो|हॅलो)\b",
                current,
                re.IGNORECASE,
            )
        )
        name_intro = extract_name(current)
        if is_greet or has_greet_word:
            prospect = name_intro if (name_intro and is_valid_prospect_name(name_intro)) else ""
            if resp_lang == "hi":
                greeting_text = f"नमस्ते {prospect} जी! मैं आज आपकी क्या सहायता कर सकता हूँ?" if prospect else get_greeting_response(resp_lang)
            elif resp_lang == "mr":
                greeting_text = f"नमस्कार {prospect} जी! मी आज आपली काय मदत करू शकेन?" if prospect else get_greeting_response(resp_lang)
            else:
                greeting_text = f"Hello {prospect}! How can I help you today?" if prospect else get_greeting_response(resp_lang)

            return {
                "type": "final",
                "transcript": current,
                "confidence": round(last_conf, 4),
                "detected_language": lang,
                "language": resp_lang,
                "is_greeting": True,
                "greeting_response": greeting_text,
                "is_assistant_query": False,
                "is_field_inquiry": False,
                "is_field_refusal": False,
                "is_resumed": False,
                "assistant_response": greeting_text,
                "immediate_sentence1": greeting_text,
                "lead": {},
                "lead_id": None,
                "next_missing_parameter": None,
                "is_complete": False,
                "is_new_lead": False,
                "is_unclear": False,
                "speech_final": True,
            }
        elif not (name_intro and is_valid_prospect_name(name_intro)):
            extractor = LeadExtractorService()
            llm_response = extractor.answer_general_query(
                current,
                language=resp_lang,
                existing_lead={},
                pending_field=None,
            )
            return {
                "type": "final",
                "transcript": current,
                "confidence": round(last_conf, 4),
                "detected_language": lang,
                "language": resp_lang,
                "is_greeting": False,
                "greeting_response": "",
                "is_assistant_query": True,
                "is_field_inquiry": False,
                "is_field_refusal": False,
                "is_resumed": False,
                "assistant_response": llm_response,
                "immediate_sentence1": llm_response,
                "lead": {},
                "lead_id": None,
                "next_missing_parameter": None,
                "is_complete": False,
                "is_new_lead": False,
                "is_unclear": False,
                "speech_final": True,
            }

    immediate_lead = dict(clean_existing)
    phone_found = extract_phone_number(current)
    if phone_found and is_valid_phone_number(phone_found) and not immediate_lead.get("phone"):
        immediate_lead["phone"] = phone_found
    email_found = extract_email(current)
    if email_found and not immediate_lead.get("email"):
        immediate_lead["email"] = email_found
    comp_found = extract_company(current, context_field=prior_missing)
    if comp_found and is_valid_company_name(comp_found) and not immediate_lead.get("company"):
        immediate_lead["company"] = comp_found
    lt_found = extract_loan_type(current, context_field=prior_missing)
    if lt_found and not immediate_lead.get("loan_type"):
        immediate_lead["loan_type"] = lt_found
    amt_found = extract_loan_amount(current, context_field=prior_missing)
    if amt_found and is_field_value_valid("loan_amount", amt_found) and not immediate_lead.get("loan_amount"):
        immediate_lead["loan_amount"] = amt_found
    ten_found = extract_tenure_months(current, context_field=prior_missing)
    if ten_found and is_field_value_valid("tenure_months", ten_found) and not immediate_lead.get("tenure_months"):
        immediate_lead["tenure_months"] = ten_found
    name_found = extract_name(current, context_field=prior_missing)
    if name_found and is_valid_prospect_name(name_found) and not immediate_lead.get("name"):
        immediate_lead["name"] = name_found

    for k, v in clean_existing.items():
        if immediate_lead.get(k) is None and v is not None:
            immediate_lead[k] = v

    was_unclear = False
    if prior_missing:
        prior_val = immediate_lead.get(prior_missing)
        if not is_field_value_valid(prior_missing, prior_val):
            was_unclear = True
            immediate_lead.pop(prior_missing, None)

    if was_unclear:
        next_missing = prior_missing
        if asst_resp:
            cont_prompt = get_clarification_prompt(prior_missing, current, immediate_lead, lang=resp_lang)
            immediate_sentence1 = f"{asst_resp} {cont_prompt}".strip() if (cont_prompt and cont_prompt not in asst_resp) else asst_resp
        elif is_greet:
            immediate_sentence1 = greet_resp
        else:
            immediate_sentence1 = get_clarification_prompt(prior_missing, current, immediate_lead, lang=resp_lang)
    elif asst_resp:
        next_missing = get_next_missing_parameter(immediate_lead)
        cont_prompt = get_missing_parameter_prompt(next_missing, immediate_lead, lang=resp_lang)
        immediate_sentence1 = f"{asst_resp} {cont_prompt}".strip() if (cont_prompt and cont_prompt not in asst_resp) else asst_resp
    elif is_greet:
        next_missing = get_next_missing_parameter(immediate_lead)
        immediate_sentence1 = greet_resp
    else:
        next_missing = get_next_missing_parameter(immediate_lead)
        immediate_sentence1 = get_missing_parameter_prompt(next_missing, immediate_lead, lang=resp_lang)

    # Non-blocking async persist: update existing lead or create new lead in background
    has_data = any(v is not None for v in immediate_lead.values() if v != "")
    if has_data:
        try:
            repo = get_shared_lead_repo()
            val_lead = Lead.model_validate(immediate_lead)
            try:
                loop = asyncio.get_running_loop()
                if lead_id_val is not None:
                    loop.create_task(asyncio.to_thread(repo.update_lead, lead_id_val, val_lead))
                else:
                    loop.create_task(asyncio.to_thread(repo.create_lead, val_lead))
            except RuntimeError:
                pass
        except Exception as persist_err:
            logger.debug(f"[ws:persist] {persist_err}")

    return {
        "type": "final",
        "transcript": current,
        "confidence": round(last_conf, 4),
        "detected_language": lang,
        "language": resp_lang,
        "is_greeting": is_greet,
        "greeting_response": greet_resp,
        "is_assistant_query": bool(asst_resp),
        "assistant_response": asst_resp or "",
        "immediate_sentence1": immediate_sentence1,
        "lead": immediate_lead,
        "lead_id": lead_id_val,
        "next_missing_parameter": next_missing,
        "is_complete": next_missing is None,
        "is_new_lead": is_reset,
        "is_unclear": was_unclear,
        "speech_final": True,
    }


async def _prewarm_tts_for_sentence1(text: str, language: str) -> None:
    """Pre-synthesizes sentence 1 in the background so audio is cached before client HTTP request arrives."""
    try:
        clean = (text or "").strip()
        if not clean:
            return
        lang = (language or "en").strip().lower()
        if lang in ("mr", "mr-in", "marathi"):
            t_lang = "mr"
            speaker = os.getenv("SARVAM_MODULE1_MARATHI_VOICE", "ritu").strip().lower()
            sarvam_lang = "mr-IN"
        elif lang in ("hi", "hi-in", "hindi"):
            t_lang = "hi"
            speaker = os.getenv("SARVAM_MODULE1_HINDI_VOICE", "priya").strip().lower()
            sarvam_lang = "hi-IN"
        else:
            t_lang = "en"
            speaker = os.getenv("SARVAM_MODULE1_ENGLISH_VOICE", "simran").strip().lower()
            sarvam_lang = "en-IN"

        cache_key = f"m1:{t_lang}:{clean.lower()}"
        if get_cached_tts_audio(cache_key) is not None:
            return

        service = get_sarvam_tts_service()
        audio = await service.synthesize_speech(clean, language_code=sarvam_lang, speaker=speaker, model="bulbul:v3")
        if audio:
            set_cached_tts_audio(cache_key, audio)
    except Exception as e:
        logger.debug(f"[_prewarm_tts_for_sentence1] notice: {e}")


async def _prewarm_standard_mod1_prompts() -> None:
    """Pre-warms the Sarvam client and caches common first-turn prompts for instant delivery."""
    try:
        await prewarm_sarvam_client()
        common_prompts = [
            ("en", "Hello! May I have your name, please?"),
            ("hi", "नमस्ते! आपका नाम क्या है?"),
            ("mr", "नमस्कार! आपले नाव काय आहे?"),
        ]
        for lang, text in common_prompts:
            await _prewarm_tts_for_sentence1(text, lang)
    except Exception as e:
        logger.debug(f"[_prewarm_standard_mod1_prompts] notice: {e}")


async def _send_pipelined_mod1_response(websocket: WebSocket, mod1_res: dict) -> None:
    """Sends one deterministic Module 1 response for each finalized user turn."""
    imm = (mod1_res.get("immediate_sentence1") or "").strip()
    if imm:
        try:
            asyncio.create_task(_prewarm_tts_for_sentence1(imm, mod1_res.get("language") or "en"))
        except RuntimeError:
            pass
    mod1_res["has_subsequent_sentences"] = False
    await websocket.send_json(mod1_res)


async def _stream_general_query_ws(
    websocket: WebSocket,
    clean: str,
    req_lang: str,
    last_conf: float,
    clean_existing: dict,
    prior_missing: Optional[str],
    lead_id_val: Optional[int],
) -> None:
    """Asynchronously streams general LLM responses over the active WebSocket without blocking the server."""
    detected_turn_lang = detect_spoken_language(clean, requested_language=req_lang, confidence=last_conf)
    lang = detected_turn_lang if detected_turn_lang in ("en", "hi", "mr") else (req_lang if req_lang in ("en", "hi", "mr") else "en")
    resp_lang = get_response_language(lang, requested_language=lang, text=clean)
    if lang not in ("en", "hi", "mr"):
        lang = "en"
    if resp_lang not in ("en", "hi", "mr"):
        resp_lang = "en"

    # Send initial final STT immediately so frontend stops listening and displays transcript (<1ms)
    await websocket.send_json({
        "type": "final",
        "transcript": clean,
        "confidence": round(last_conf, 4),
        "detected_language": lang,
        "language": resp_lang,
        "is_greeting": False,
        "greeting_response": "",
        "is_assistant_query": True,
        "is_general_query": True,
        "llm_unavailable": False,
        "is_field_inquiry": False,
        "is_field_refusal": False,
        "is_resumed": False,
        "assistant_response": "",
        "immediate_sentence1": "",
        "has_subsequent_sentences": True,
        "lead": clean_existing,
        "lead_id": lead_id_val,
        "next_missing_parameter": prior_missing,
        "is_complete": False,
        "is_new_lead": False,
        "is_unclear": False,
        "speech_final": True,
    })

    # Asynchronously stream tokens from OpenRouter LLM
    extractor = LeadExtractorService()
    buffer = ""
    sentence_punct_pattern = re.compile(r'(?<=[.?!।॥\n])\s+')
    stream_tokens = extractor.stream_general_query_tokens(
        transcript=clean,
        language=resp_lang,
        existing_lead=clean_existing,
        pending_field=prior_missing,
    )

    sent_any_sentence = False
    try:
        async for token in stream_tokens:
            buffer += token
            splits = sentence_punct_pattern.split(buffer)
            if len(splits) > 1:
                for complete_sentence in splits[:-1]:
                    s_clean = complete_sentence.strip()
                    if s_clean:
                        await websocket.send_json({"type": "sentence", "sentence": s_clean})
                        sent_any_sentence = True
                buffer = splits[-1]

        if buffer.strip():
            await websocket.send_json({"type": "sentence", "sentence": buffer.strip()})
            sent_any_sentence = True
    except Exception as stream_err:
        logger.warning(f"[ws/voice-stt] _stream_general_query_ws token stream error: {stream_err}")
    finally:
        if not sent_any_sentence:
            if resp_lang == "hi":
                fallback_msg = "मैं अभी अपने असिस्टेंट से संपर्क नहीं कर पा रहा हूँ। कृपया दोबारा पूछिए।"
            elif resp_lang == "mr":
                fallback_msg = "मी सध्या माझ्या असिस्टंटशी संपर्क साधू शकत नाही. कृपया पुन्हा विचारा."
            else:
                fallback_msg = "I'm having trouble reaching my assistant right now. Please ask that again."
            try:
                await websocket.send_json({"type": "sentence", "sentence": fallback_msg})
            except Exception:
                pass
        try:
            await websocket.send_json({"type": "stream_complete"})
        except Exception:
            pass



async def _dispatch_stt_final_turn(
    websocket: WebSocket,
    clean: str,
    winning_lang: str,
    winning_conf: float,
    is_mod1: bool,
    existing_lead: Optional[str],
    lead_id: Optional[int],
) -> Optional[asyncio.Task]:
    """
    Authoritative final turn dispatcher for both Module 1 and Module 2.
    Routes to lead extraction / general LLM query / Module 2 RAG response in the turn's exact language.
    """
    if not clean:
        await websocket.send_json({
            "type": "final",
            "transcript": "",
            "confidence": 0.0,
            "speech_final": True,
        })
        return None

    if winning_lang == "unclear":
        target_clarify_lang = "en"
        clarification_text = CLARIFICATION_PROMPTS.get(target_clarify_lang, "Could you please repeat that?")
        await websocket.send_json({
            "type": "final",
            "transcript": clean,
            "confidence": round(winning_conf, 4),
            "detected_language": target_clarify_lang,
            "language": target_clarify_lang,
            "is_greeting": False,
            "greeting_response": "",
            "is_assistant_query": False,
            "assistant_response": clarification_text,
            "speech_final": True,
        })
        if is_mod1:
            await websocket.send_json({
                "type": "sentence",
                "sentence": clarification_text,
                "language": target_clarify_lang,
                "is_final": True,
            })
        return None

    if is_mod1:
        clean_existing = {}
        if existing_lead:
            try:
                clean_existing = json.loads(existing_lead) if isinstance(existing_lead, str) else dict(existing_lead)
            except Exception:
                clean_existing = {}
        prior_missing = get_next_missing_parameter(clean_existing)
        has_active_lead = bool(clean_existing) and any(
            clean_existing.get(f) is not None and str(clean_existing.get(f)).strip() != ""
            for f in REQUIRED_LEAD_FIELDS
        )
        pending_to_resume = prior_missing if has_active_lead else None

        is_lead_intent = is_loan_intent(clean)
        name_intro = extract_name(clean)
        has_explicit_name = bool(name_intro and is_valid_prospect_name(name_intro) and re.search(r"\b(?:my\s+name\s+is|this\s+is|i\s+am)\s+[A-Za-z]+", clean, re.IGNORECASE))
        is_greeting_turn = is_greeting(clean)
        is_general_q_turn = is_general_question(clean, context_field=prior_missing)

        if not has_active_lead:
            if is_greeting_turn or (is_general_q_turn and not is_lead_intent and not has_explicit_name):
                return asyncio.create_task(
                    _stream_general_query_ws(
                        websocket=websocket,
                        clean=clean,
                        req_lang=winning_lang,
                        last_conf=winning_conf,
                        clean_existing=clean_existing,
                        prior_missing=None,
                        lead_id_val=lead_id,
                    )
                )
            elif is_lead_intent or has_explicit_name:
                mod1_res = generate_module1_ws_response(clean, winning_lang, winning_conf, existing_lead, lead_id)
                await _send_pipelined_mod1_response(websocket, mod1_res)
                return None
            else:
                return asyncio.create_task(
                    _stream_general_query_ws(
                        websocket=websocket,
                        clean=clean,
                        req_lang=winning_lang,
                        last_conf=winning_conf,
                        clean_existing=clean_existing,
                        prior_missing=None,
                        lead_id_val=lead_id,
                    )
                )
        else:
            if is_greeting_turn or is_general_q_turn:
                return asyncio.create_task(
                    _stream_general_query_ws(
                        websocket=websocket,
                        clean=clean,
                        req_lang=winning_lang,
                        last_conf=winning_conf,
                        clean_existing=clean_existing,
                        prior_missing=pending_to_resume,
                        lead_id_val=lead_id,
                    )
                )
            else:
                mod1_res = generate_module1_ws_response(clean, winning_lang, winning_conf, existing_lead, lead_id)
                await _send_pipelined_mod1_response(websocket, mod1_res)
                return None
    else:
        # Module 2 (Knowledge Assistant)
        resp_lang = get_response_language(winning_lang, requested_language=winning_lang, text=clean)
        is_greet = is_greeting(clean)
        greet_resp = get_greeting_response(resp_lang) if is_greet else ""
        asst_resp = get_assistant_query_response(clean, resp_lang)
        await websocket.send_json({
            "type": "final",
            "transcript": clean,
            "confidence": round(winning_conf, 4),
            "detected_language": winning_lang,
            "language": resp_lang,
            "is_greeting": is_greet,
            "greeting_response": greet_resp,
            "is_assistant_query": bool(asst_resp),
            "assistant_response": asst_resp or "",
            "speech_final": True,
        })
        return None


# select_best_multilingual_transcript is imported from services.language


# ----------------------------------------------------------------------------
# WEBSOCKET HANDLER: websocket_voice_stt (/ws/voice-stt)
# ----------------------------------------------------------------------------
# • WHAT IT DOES: Low-latency live streaming WebSocket proxy for Deepgram Nova-3.
#   Streams 16-bit linear PCM audio chunks from browser directly to Deepgram,
#   emitting interim transcripts in real time and instant final transcript upon speech completion.
# • INPUTS:
#     - WebSocket connection with binary 16-bit PCM frames.
#     - language (query param): Optional language hint ('en', 'hi', 'mr').
#     - sample_rate (query param): Client sample rate (default: 48000).
# ----------------------------------------------------------------------------
@app.websocket("/ws/voice-stt")
async def websocket_voice_stt(
    websocket: WebSocket,
    language: Optional[str] = None,
    sample_rate: int = 48000,
    encoding: Optional[str] = "linear16",
    module: Optional[str] = None,
    existing_lead: Optional[str] = None,
    lead_id: Optional[int] = None,
):
    await websocket.accept()

    active_stt = get_model_manager().get_active_model("stt") or {}
    stt_model_id = (active_stt.get("model_id") or "nova-3").strip()
    stt_provider = (active_stt.get("provider") or "").lower()
    if "sarvam" in stt_provider or "saaras" in stt_model_id.lower():
        logger.info(f"[ws/voice-stt] Selected STT model '{stt_model_id}' is HTTP-only; routing WebSocket stream to nova-3")
        stt_model_id = "nova-3"
    stt_custom_key = active_stt.get("api_key")
    api_key = (stt_custom_key or os.getenv("DEEPGRAM_API_KEY", "")).strip()
    if not api_key:
        await websocket.send_json({"type": "error", "message": "DEEPGRAM_API_KEY not configured in backend/.env"})
        await websocket.close(code=1008)
        return

    req_lang = (language or "").strip().lower()
    if req_lang in ("mr", "mr-in", "marathi"):
        req_lang = "mr"
    elif req_lang in ("hi", "hi-in", "hindi"):
        req_lang = "hi"
    elif req_lang in ("en", "en-in", "en-us", "english"):
        req_lang = "en"
    else:
        req_lang = "auto"

    is_explicit_single = req_lang in ("mr", "hi", "en")
    is_mod1 = (module or "").strip().lower() in ("module1", "crm", "voice_copilot", "voice-copilot")

    headers = {"Authorization": f"Token {api_key}"}

    logger.info(f"[ws/voice-stt] Client connected. Mode: {'single-stream' if is_explicit_single else '3-stream-auto'} (model: {stt_model_id}, lang: {req_lang}), module: {module}")

    try:
        if is_explicit_single:
            # Single-stream mode: strictly route to mr, hi, or en-IN
            if req_lang == "mr":
                dg_lang_ws = "mr"
            elif req_lang == "hi":
                dg_lang_ws = "hi"
            else:
                dg_lang_ws = "en-IN"

            single_url = build_deepgram_ws_url(
                model=stt_model_id,
                sample_rate=sample_rate,
                language=dg_lang_ws,
                encoding=encoding,
            )
            dg_pool = get_deepgram_ws_pool()
            dg_ws = await dg_pool.acquire(single_url, headers)
            dg_broken = False

            try:
                accumulated_finals = []
                latest_interim = ""
                sent_final = False
                last_conf = 1.0
                finalize_requested = False
                final_done_event = asyncio.Event()
                general_query_task: Optional[asyncio.Task] = None

                def get_current_transcript(include_interim: bool = True) -> str:
                    parts = list(accumulated_finals)
                    if include_interim and latest_interim and latest_interim.strip():
                        parts.append(latest_interim.strip())
                    return " ".join(parts).strip()

                async def emit_final_stt(source: str = "direct"):
                    nonlocal sent_final, general_query_task
                    if sent_final:
                        return
                    sent_final = True
                    final_done_event.set()
                    clean = get_current_transcript(include_interim=False) or get_current_transcript(include_interim=True)
                    if clean:
                        clean = normalize_stt_transcript(clean)
                        logger.info(
                            f"[ws/voice-stt] Raw Deepgram final transcript ({source}, req_lang={req_lang}, finalize_requested={finalize_requested}): '{clean}'"
                        )
                        final_lang = detect_spoken_language(clean, requested_language=req_lang, confidence=last_conf)
                        general_query_task = await _dispatch_stt_final_turn(
                            websocket=websocket,
                            clean=clean,
                            winning_lang=final_lang,
                            winning_conf=last_conf,
                            is_mod1=is_mod1,
                            existing_lead=existing_lead,
                            lead_id=lead_id,
                        )
                    else:
                        logger.info(f"[ws/voice-stt] Emitting empty final transcript ({source})")
                        await _dispatch_stt_final_turn(
                            websocket=websocket,
                            clean="",
                            winning_lang="en",
                            winning_conf=0.0,
                            is_mod1=is_mod1,
                            existing_lead=existing_lead,
                            lead_id=lead_id,
                        )

                closed_single = False

                async def client_to_single():
                    nonlocal closed_single, finalize_requested
                    try:
                        while True:
                            msg = await websocket.receive()
                            if "bytes" in msg and msg["bytes"]:
                                await dg_ws.send(msg["bytes"])
                            elif "text" in msg and msg["text"]:
                                try:
                                    payload = json.loads(msg["text"])
                                    msg_t = payload.get("type")
                                    if msg_t in ("Finalize", "CloseStream"):
                                        finalize_requested = True
                                        try:
                                            await dg_ws.send(json.dumps({"type": "Finalize"}))
                                        except Exception:
                                            pass

                                        try:
                                            await asyncio.wait_for(final_done_event.wait(), timeout=0.75)
                                        except (asyncio.TimeoutError, Exception):
                                            pass

                                        if not sent_final:
                                            await emit_final_stt("finalize_deadline")
                                        break
                                except Exception:
                                    pass
                    except (WebSocketDisconnect, asyncio.CancelledError):
                        pass
                    except Exception as e:
                        logger.info(f"[ws/voice-stt] client_to_single exception: {e}")
                    finally:
                        closed_single = True

                async def single_to_client():
                    nonlocal sent_final, last_conf, latest_interim
                    try:
                        while True:
                            try:
                                dg_msg = await dg_ws.recv()
                            except (websockets.exceptions.ConnectionClosed, asyncio.CancelledError):
                                break
                            if isinstance(dg_msg, str):
                                data = json.loads(dg_msg)
                                msg_type = data.get("type")
                                if msg_type == "Results":
                                    alts = (data.get("channel") or {}).get("alternatives") or []
                                    if alts:
                                        alt = alts[0]
                                        tr = (alt.get("transcript") or "").strip()
                                        conf = alt.get("confidence", 0.0)
                                        if conf > 0:
                                            last_conf = conf

                                        if data.get("is_final", False):
                                            if tr:
                                                accumulated_finals.append(tr)
                                                latest_interim = ""
                                            if not finalize_requested:
                                                current = get_current_transcript(include_interim=False)
                                                if current:
                                                    await websocket.send_json({"type": "interim", "transcript": current, "is_final": False})
                                        elif tr:
                                            latest_interim = tr
                                            if not finalize_requested:
                                                current = get_current_transcript(include_interim=True)
                                                if current:
                                                    await websocket.send_json({"type": "interim", "transcript": current, "is_final": False})

                                        if finalize_requested and (data.get("is_final") or data.get("speech_final")) and not sent_final:
                                            await emit_final_stt("deepgram_is_final")

                                        if sent_final:
                                            break
                                elif msg_type == "Metadata":
                                    if finalize_requested and not sent_final:
                                        await emit_final_stt("deepgram_metadata")
                                    break
                    except (WebSocketDisconnect, asyncio.CancelledError):
                        pass
                    except Exception as e:
                        logger.info(f"[ws/voice-stt] single_to_client exception: {e}")
                    finally:
                        if finalize_requested and not sent_final:
                            await emit_final_stt("finally_fallback")

                c_task = asyncio.create_task(client_to_single())
                s_task = asyncio.create_task(single_to_client())
                done, pending = await asyncio.wait([c_task, s_task], return_when=asyncio.FIRST_COMPLETED)
                if s_task in pending and not sent_final:
                    try:
                        await asyncio.wait_for(s_task, timeout=0.5)
                    except (asyncio.TimeoutError, Exception):
                        pass
                for t in pending:
                    if not t.done():
                        t.cancel()

            except Exception as exc:
                dg_broken = True
                raise
            finally:
                if dg_ws:
                    try:
                        while True:
                            await asyncio.wait_for(dg_ws.recv(), timeout=0.03)
                    except (asyncio.TimeoutError, Exception):
                        pass
                state_name = getattr(getattr(dg_ws, "state", None), "name", "")
                is_stale_or_broken = dg_broken or (state_name != "OPEN" if dg_ws else True)
                logger.info(f"[ws/voice-stt] Pool release: state={state_name}, dg_broken={dg_broken}, broken={is_stale_or_broken}")
                await dg_pool.release(single_url, dg_ws, broken=is_stale_or_broken)

        else:
            # -------------------------------------------------------------
            # Multi-Stream Concurrent Auto STT Mode (en-IN, hi, mr)
            # -------------------------------------------------------------
            url_en = build_deepgram_ws_url(model=stt_model_id, sample_rate=sample_rate, language="en-IN", encoding=encoding)
            url_hi = build_deepgram_ws_url(model=stt_model_id, sample_rate=sample_rate, language="hi", encoding=encoding)
            url_mr = build_deepgram_ws_url(model=stt_model_id, sample_rate=sample_rate, language="mr", encoding=encoding)

            dg_pool = get_deepgram_ws_pool()
            ws_en, ws_hi, ws_mr = await asyncio.gather(
                dg_pool.acquire(url_en, headers),
                dg_pool.acquire(url_hi, headers),
                dg_pool.acquire(url_mr, headers),
            )
            sockets = {"en": ws_en, "hi": ws_hi, "mr": ws_mr}
            urls = {"en": url_en, "hi": url_hi, "mr": url_mr}
            broken = {"en": False, "hi": False, "mr": False}

            accumulated: Dict[str, List[str]] = {"en": [], "hi": [], "mr": []}
            confidences: Dict[str, List[float]] = {"en": [], "hi": [], "mr": []}
            latest_interims: Dict[str, str] = {"en": "", "hi": "", "mr": ""}
            streams_finished: Dict[str, bool] = {"en": False, "hi": False, "mr": False}
            grace_timer_task: Optional[asyncio.Task] = None

            sent_final = False
            finalize_requested = False
            final_done_event = asyncio.Event()
            general_query_task: Optional[asyncio.Task] = None

            def get_best_interim():
                for k in ("mr", "hi"):
                    txt = f"{' '.join(accumulated[k])} {latest_interims[k]}".strip()
                    if any("\u0900" <= ch <= "\u097f" for ch in txt):
                        return txt
                txt_en = f"{' '.join(accumulated['en'])} {latest_interims['en']}".strip()
                return txt_en

            async def trigger_grace_timeout():
                try:
                    await asyncio.sleep(0.85)
                    if finalize_requested and not sent_final:
                        await emit_final_multi("grace_timeout")
                except asyncio.CancelledError:
                    pass

            async def client_to_multi():
                nonlocal finalize_requested
                try:
                    while True:
                        msg = await websocket.receive()
                        if "bytes" in msg and msg["bytes"]:
                            b = msg["bytes"]
                            await asyncio.gather(
                                ws_en.send(b),
                                ws_hi.send(b),
                                ws_mr.send(b),
                                return_exceptions=True,
                            )
                        elif "text" in msg and msg["text"]:
                            try:
                                payload = json.loads(msg["text"])
                                if payload.get("type") in ("Finalize", "CloseStream"):
                                    finalize_requested = True
                                    fin_bytes = json.dumps({"type": "Finalize"})
                                    await asyncio.gather(
                                        ws_en.send(fin_bytes),
                                        ws_hi.send(fin_bytes),
                                        ws_mr.send(fin_bytes),
                                        return_exceptions=True,
                                    )
                                    try:
                                        await asyncio.wait_for(final_done_event.wait(), timeout=1.8)
                                    except (asyncio.TimeoutError, Exception):
                                        pass
                                    if not sent_final:
                                        await emit_final_multi("finalize_deadline")
                                    break
                            except Exception:
                                pass
                except (WebSocketDisconnect, asyncio.CancelledError):
                    pass

            async def stream_reader(lang_key: str, ws: Any):
                nonlocal sent_final, grace_timer_task
                try:
                    while True:
                        try:
                            dg_msg = await ws.recv()
                        except (websockets.exceptions.ConnectionClosed, asyncio.CancelledError):
                            broken[lang_key] = True
                            streams_finished[lang_key] = True
                            if finalize_requested and not sent_final:
                                if all(streams_finished[k] or broken[k] for k in ("en", "hi", "mr")):
                                    await emit_final_multi(f"all_streams_closed_{lang_key}")
                            break
                        if isinstance(dg_msg, str):
                            data = json.loads(dg_msg)
                            msg_type = data.get("type")
                            if msg_type == "Results":
                                alts = (data.get("channel") or {}).get("alternatives") or []
                                if alts:
                                    alt = alts[0]
                                    tr = (alt.get("transcript") or "").strip()
                                    conf = alt.get("confidence", 0.0)
                                    if conf > 0:
                                        confidences[lang_key].append(conf)
                                    if data.get("is_final", False):
                                        if tr:
                                            accumulated[lang_key].append(tr)
                                            latest_interims[lang_key] = ""
                                        if not finalize_requested:
                                            disp = get_best_interim()
                                            if disp:
                                                await websocket.send_json({"type": "interim", "transcript": disp, "is_final": False})
                                    elif tr:
                                        latest_interims[lang_key] = tr
                                        if not finalize_requested:
                                            disp = get_best_interim()
                                            if disp:
                                                await websocket.send_json({"type": "interim", "transcript": disp, "is_final": False})
                            elif msg_type == "Metadata":
                                streams_finished[lang_key] = True
                                if finalize_requested and not sent_final:
                                    if all(streams_finished[k] or broken[k] for k in ("en", "hi", "mr")):
                                        await emit_final_multi("all_streams_metadata")
                                    else:
                                        if grace_timer_task is None or grace_timer_task.done():
                                            grace_timer_task = asyncio.create_task(trigger_grace_timeout())
                                break
                except (WebSocketDisconnect, asyncio.CancelledError):
                    pass
                except Exception as e:
                    broken[lang_key] = True
                    streams_finished[lang_key] = True
                    if finalize_requested and not sent_final:
                        if all(streams_finished[k] or broken[k] for k in ("en", "hi", "mr")):
                            await emit_final_multi(f"all_streams_err_{lang_key}")

            async def emit_final_multi(source: str):
                nonlocal sent_final, general_query_task
                if sent_final:
                    return
                sent_final = True
                final_done_event.set()

                candidates = {}
                for k in ("en", "hi", "mr"):
                    parts = list(accumulated[k])
                    if latest_interims[k] and latest_interims[k].strip():
                        parts.append(latest_interims[k].strip())
                    candidates[k] = {
                        "transcript": " ".join(parts).strip(),
                        "confidence": (sum(confidences[k]) / len(confidences[k])) if confidences[k] else 0.0
                    }

                logger.info(f"[ws/voice-stt] Auto STT multi-stream candidates ({source}): {candidates}")
                winning_tr, winning_lang, winning_conf = select_best_multilingual_transcript(candidates)
                clean = normalize_stt_transcript(winning_tr)
                logger.info(f"[ws/voice-stt] Auto STT multi-stream winner ({source}): lang={winning_lang}, conf={winning_conf:.3f}, transcript='{clean}'")

                general_query_task = await _dispatch_stt_final_turn(
                    websocket=websocket,
                    clean=clean,
                    winning_lang=winning_lang,
                    winning_conf=winning_conf,
                    is_mod1=is_mod1,
                    existing_lead=existing_lead,
                    lead_id=lead_id,
                )

            c_task = asyncio.create_task(client_to_multi())
            r_tasks = [
                asyncio.create_task(stream_reader("en", ws_en)),
                asyncio.create_task(stream_reader("hi", ws_hi)),
                asyncio.create_task(stream_reader("mr", ws_mr)),
            ]
            all_tasks = [c_task] + r_tasks
            try:
                done, pending = await asyncio.wait(all_tasks, return_when=asyncio.FIRST_COMPLETED)
                if c_task in done:
                    if not sent_final and finalize_requested:
                        try:
                            await asyncio.wait_for(final_done_event.wait(), timeout=0.6)
                        except (asyncio.TimeoutError, Exception):
                            pass
                    if not sent_final:
                        await emit_final_multi("client_done")
                else:
                    if all(t.done() for t in r_tasks):
                        if not sent_final:
                            await emit_final_multi("all_readers_done")
                    else:
                        try:
                            await asyncio.wait_for(final_done_event.wait(), timeout=2.0)
                        except (asyncio.TimeoutError, Exception):
                            pass
                        if not sent_final:
                            await emit_final_multi("wait_done")
                for t in all_tasks:
                    if not t.done():
                        t.cancel()
            except Exception as multi_exc:
                for k in ("en", "hi", "mr"):
                    broken[k] = True
                raise multi_exc
            finally:
                if grace_timer_task and not grace_timer_task.done():
                    grace_timer_task.cancel()
                for k in ("en", "hi", "mr"):
                    try:
                        s = sockets[k]
                        if s:
                            try:
                                while True:
                                    await asyncio.wait_for(s.recv(), timeout=0.03)
                            except (asyncio.TimeoutError, Exception):
                                pass
                        state_name = getattr(getattr(s, "state", None), "name", "")
                        is_stale_or_broken = broken[k] or (state_name != "OPEN" if s else True)
                        await dg_pool.release(urls[k], s, broken=is_stale_or_broken)
                    except Exception:
                        pass


    except Exception as exc:
        logger.error(f"[ws/voice-stt] WebSocket error: {exc}")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        if general_query_task and not general_query_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(general_query_task), timeout=25.0)
            except Exception as gq_err:
                logger.warning(f"[ws/voice-stt] General query stream task finished: {gq_err}")
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("[ws/voice-stt] Session ended.")


# ----------------------------------------------------------------------------
# ROUTE HANDLER: extract_lead_from_transcript (POST /api/extract-lead)
# ----------------------------------------------------------------------------
# • WHAT IT DOES: Extracts structured Pydantic lead JSON from sales transcripts via DeepSeek.
#   Supports both non-streaming JSONResponse and real-time SSE StreamingResponse.
# • INPUTS:
#     - request (LeadExtractionRequest): JSON payload containing:
#         - transcript: Raw audio text.
#         - existing_lead: Lead data from previous turns to update.
#         - language: 'en', 'hi', 'mr'.
#         - stream: bool (True for real-time SSE streaming).
#         - lead_id: int (PostgreSQL ID to update in-place).
# • OUTPUT: JSONResponse with extracted lead fields, or StreamingResponse (SSE text/event-stream).
# • WHY IT IS USED: The core intelligence gateway for Module 1 Voice-to-CRM.
# • WHERE IT FITS IN THE FLOW:
#     [LiveCallVoiceCopilot.tsx] -> [POST /api/extract-lead] -> [lead_extractor.py] -> [PostgreSQL]
# ----------------------------------------------------------------------------
@app.post(
    "/api/extract-lead",
    tags=["Voice-to-CRM"],
    summary="Extract structured Pydantic lead from speech transcript",
)
async def extract_lead_from_transcript(request: LeadExtractionRequest):
    """
    Module 1 Step 3: OpenRouter LLM -> Pydantic structured lead extraction.

    Flow: Deepgram transcript -> OpenRouter LLM -> Pydantic validation -> structured lead JSON.
    Supports real-time SSE streaming mode when stream=True.

    Extracts:
    - name, phone, email, company, role, loan_type, loan_amount, tenure_months, notes
    Missing fields are strictly null. Pydantic is the authoritative validation layer.
    """
    clean_transcript = (request.transcript or "").strip()
    if not clean_transcript:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transcript cannot be empty or whitespace.",
        )

    # CRITICAL: Bypass lead extraction for interim / partial speech transcripts
    if request.is_interim is True or request.is_final is False:
        logger.info(f"[api/extract-lead] Ignoring interim STT transcript: '{clean_transcript}'")
        clean_existing = request.existing_lead or {}
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "interim_ignored",
                "is_interim": True,
                "message": "",
                "lead": clean_existing,
                "next_missing_parameter": get_next_missing_parameter(clean_existing),
                "is_complete": False,
            },
        )

    try:
        # Dynamic active LLM model resolution
        active_llm = None
        try:
            from services.model_manager import get_active_model_id
            active_llm = get_active_model_id("llm")
        except Exception:
            pass
        effective_model = request.model or active_llm

        extractor = LeadExtractorService()

        if request.stream:
            return StreamingResponse(
                extractor.stream_lead_turn(
                    transcript=clean_transcript,
                    model=effective_model,
                    existing_lead=request.existing_lead,
                    language=request.language,
                    lead_id=request.lead_id,
                    is_interim=request.is_interim,
                    is_final=request.is_final,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        result = extractor.extract_lead(
            clean_transcript,
            model=effective_model,
            existing_lead=request.existing_lead,
            language=request.language,
            is_interim=request.is_interim,
            is_final=request.is_final,
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=result)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except LeadValidationError as lve:
        logger.error(f"Lead validation error: {str(lve)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "Pydantic lead validation failed",
                "message": str(lve),
                "validation_errors": lve.errors,
            },
        )
    except MalformedLLMOutputError as mfe:
        logger.error(f"Malformed LLM output: {str(mfe)}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"OpenRouter LLM returned malformed or non-JSON output: {str(mfe)}",
        )
    except LeadOpenRouterConfigError as oce:
        logger.error(f"OpenRouter configuration error: {str(oce)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(oce),
        )
    except LeadOpenRouterAPIError as oae:
        logger.error(f"OpenRouter API error: {str(oae)}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(oae),
        )
    except Exception as e:
        logger.error(f"Unexpected error during lead extraction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during lead extraction: {str(e)}",
        )


@app.on_event("startup")
def startup_db_check():
    """Ensure PostgreSQL 'leads' and 'documents' tables are created and ready on application startup."""
    try:
        repo = LeadRepository()
        repo.ensure_leads_table()
        logger.info("Startup check: PostgreSQL 'leads' table is ready.")
    except Exception as e:
        logger.warning(f"Startup check: Unable to initialize 'leads' table (will retry on query): {e}")

    try:
        doc_repo = DocumentRepository()
        doc_repo.ensure_documents_table()
        logger.info("Startup check: PostgreSQL 'documents' table is ready.")
    except Exception as e:
        logger.warning(f"Startup check: Unable to initialize 'documents' table (will retry on query): {e}")

    try:
        from services.vector_store import PineconeService
        ps = PineconeService()
        if ps.api_key:
            ps.get_index()
            logger.info("Startup check: Pinecone index client pre-warmed for instant voice search.")
    except Exception as e:
        logger.debug(f"Startup check: Pinecone pre-warm skipped or deferred: {e}")

    try:
        from services.stt import get_shared_stt_client
        from services.tts import get_shared_tts_client
        from services.sarvam_tts import get_shared_sarvam_client, prewarm_sarvam_client
        get_shared_stt_client()
        get_shared_tts_client()
        get_shared_sarvam_client()
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_prewarm_standard_mod1_prompts())
        except Exception:
            pass
        logger.info("Startup check: Deepgram STT, TTS, and Sarvam TTS keep-alive connection pools initialized.")
    except Exception as e:
        logger.debug(f"Startup check: STT/TTS pool initialization skipped: {e}")

    try:
        from services.rag import RAGService
        rs = RAGService()
        rs.get_http_client()
        logger.info("Startup check: OpenRouter RAG keep-alive client initialized.")
    except Exception as e:
        logger.debug(f"Startup check: OpenRouter client pre-warm skipped: {e}")

    try:
        from services.telegram_bot import get_voice_copilot_bot
        bot = get_voice_copilot_bot()
        if bot.is_configured:
            bot.start_polling_task()
            logger.info("Startup check: VoiceCopilotBot polling worker started for real-time Telegram updates.")
    except Exception as e:
        logger.debug(f"Startup check: Telegram bot polling startup skipped: {e}")


@app.post(
    "/api/leads",
    status_code=status.HTTP_201_CREATED,
    tags=["CRM / Leads"],
    summary="Save a validated Pydantic Lead to PostgreSQL",
)
@app.post(
    "/api/save-lead",
    status_code=status.HTTP_201_CREATED,
    tags=["CRM / Leads"],
    include_in_schema=False,
)
async def save_lead_endpoint(lead: Lead):
    """
    Module 1 Step 4: Save validated Pydantic Lead into PostgreSQL.
    Accepts validated Lead, stores in PostgreSQL 'leads' table, and returns the saved record.
    """
    try:
        repo = LeadRepository()
        saved_record = repo.create_lead(lead)
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "status": "success",
                "message": "Lead saved to database successfully",
                "lead": saved_record,
            },
        )
    except DatabaseConnectionError as dce:
        logger.error(f"Database connection error: {str(dce)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"PostgreSQL connection unavailable: {str(dce)}",
        )
    except DatabaseConfigurationError as dcfg:
        logger.error(f"Database configuration error: {str(dcfg)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database configuration error: {str(dcfg)}",
        )
    except DatabaseOperationError as doe:
        logger.error(f"Database operation error: {str(doe)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database operation error: {str(doe)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error saving lead: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error saving lead: {str(e)}",
        )


@app.get(
    "/api/leads",
    tags=["CRM / Leads"],
    summary="Retrieve saved leads from PostgreSQL",
)
async def get_leads_endpoint(limit: int = 50):
    """
    Retrieves recent saved leads from PostgreSQL for the CRM dashboard.
    """
    try:
        repo = LeadRepository()
        leads = repo.get_leads(limit=limit)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "success", "count": len(leads), "leads": leads},
        )
    except DatabaseConnectionError as dce:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"PostgreSQL connection unavailable: {str(dce)}",
        )
    except DatabaseOperationError as doe:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query error: {str(doe)}",
        )


@app.put(
    "/api/leads/{lead_id}",
    status_code=status.HTTP_200_OK,
    tags=["CRM / Leads"],
    summary="Update an existing Pydantic Lead in PostgreSQL",
)
async def update_lead_endpoint(lead_id: int, lead: Lead):
    """
    Module 1 Continuous Voice Assistant: Update existing lead in PostgreSQL.
    Ensures multi-turn conversations update the same record without creating duplicates.
    """
    try:
        repo = LeadRepository()
        updated_record = repo.update_lead(lead_id, lead)
        if not updated_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Lead #{lead_id} not found in database.",
            )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "message": f"Lead #{lead_id} updated successfully",
                "lead": updated_record,
            },
        )
    except HTTPException:
        raise
    except DatabaseConnectionError as dce:
        logger.error(f"Database connection error updating lead #{lead_id}: {str(dce)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"PostgreSQL connection unavailable: {str(dce)}",
        )
    except DatabaseOperationError as doe:
        logger.error(f"Database operation error updating lead #{lead_id}: {str(doe)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database update error: {str(doe)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error updating lead #{lead_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error updating lead: {str(e)}",
        )


class TTSRequest(BaseModel):
    text: str = Field(
        ...,
        description="Text content to synthesize into speech.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Optional voice model override (e.g. Deepgram aura voice or 'bulbul:v3').",
    )
    language: Optional[str] = Field(
        default=None,
        description="Optional language code ('en', 'hi', 'mr').",
    )
    module: Optional[str] = Field(
        default=None,
        description="Optional calling module identifier (e.g. 'module2').",
    )
    speaker: Optional[str] = Field(
        default=None,
        description="Optional voice speaker name ('priya' or 'ritu' for Sarvam).",
    )


# ----------------------------------------------------------------------------
# ROUTE HANDLER: tts_endpoint (POST /api/tts)
# ----------------------------------------------------------------------------
# • WHAT IT DOES: Converts a text sentence into binary audio:
#     - Module 2 (English, Marathi, Hindi): Uses Sarvam AI Bulbul v3.
#       For English: Uses Sarvam Bulbul v3 Indian English ('en-IN') with natural female voice ('simran').
#       For Marathi/Hindi: Uses Sarvam Bulbul v3 ('mr-IN'/'hi-IN').
#       All Module 2 requests include automatic Deepgram Aura fallback on upstream error.
#     - Module 1 (Voice-to-CRM): Uses Deepgram Aura TTS for fast low-latency playback.
# • INPUTS:
#     - request (TTSRequest): Payload with 'text', optional 'model', 'language', 'module', 'speaker'.
# • OUTPUT: Binary Response with media_type="audio/wav" or "audio/mpeg".
# • WHY IT IS USED: Called in parallel for every completed sentence streamed by the LLM.
# • WHERE IT FITS IN THE FLOW:
#     [SentenceAudioQueue.enqueueSentence] -> [POST /api/tts] -> [Browser Web Audio / HTMLAudioElement]
# ----------------------------------------------------------------------------
@app.post(
    "/api/tts",
    tags=["Voice-to-CRM"],
    summary="Synthesize speech from text using Deepgram or Sarvam TTS",
)
async def tts_endpoint(request: TTSRequest):
    """
    Synthesizes speech from text.
    - For Module 2 queries: Uses Sarvam Bulbul v3 (English 'en-IN' with natural female voice 'simran',
      Marathi 'mr-IN', Hindi 'hi-IN') with automatic fallback to Deepgram Aura.
    - For Module 1 (Voice-to-CRM): Uses Deepgram Aura TTS.
    Returns binary audio stream (audio/wav or audio/mpeg).
    """
    import time as _lat_t3
    _lat_t3_start = _lat_t3.perf_counter()
    safe_preview = (request.text or '').strip()[:60].encode("ascii", errors="backslashreplace").decode("ascii")
    print(f"[LATENCY] T6 TTS request START at perf={_lat_t3.perf_counter():.6f}s text='{safe_preview}'", flush=True)
    clean_text = (request.text or "").strip()
    if not clean_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'text' field cannot be empty or whitespace.",
        )

    # Determine target language and caller context
    has_devanagari = count_devanagari_chars(clean_text) > 0
    detected_lang = detect_text_language(clean_text)

    # If the text has Devanagari characters, it is strictly Indic (Hindi or Marathi, never English)
    if has_devanagari:
        req_l = (request.language or "").strip().lower()
        if req_l in ("mr", "mr-in", "marathi"):
            target_lang = "mr"
        elif req_l in ("hi", "hi-in", "hindi"):
            target_lang = "hi"
        else:
            target_lang = detected_lang if detected_lang in ("mr", "hi") else "hi"
    else:
        req_l = (request.language or "").strip().lower()
        if req_l in ("auto", ""):
            target_lang = detected_lang or "en"
        else:
            target_lang = req_l

    is_marathi = target_lang in ("mr", "mr-in", "marathi")
    is_hindi = target_lang in ("hi", "hi-in", "hindi")
    is_english = not (is_marathi or is_hindi)
    is_module2 = request.module in ("module2", "knowledge_assistant", "rag")
    is_module1 = request.module in ("module1", "voice_copilot", "voice-copilot", "crm") or not is_module2

    # Fast-Path: In-memory LRU prompt audio cache for Module 1 (< 1ms delivery for known phrases)
    if is_module1 and not is_module2:
        cache_key = f"m1:{target_lang}:{clean_text.strip().lower()}"
        cached_audio = get_cached_tts_audio(cache_key)
        if cached_audio:
            print(f"[LATENCY] T7 TTS audio READY (cache HIT) at perf={_lat_t3.perf_counter():.6f}s", flush=True)
            media_type = "audio/wav" if cached_audio.startswith(b"RIFF") else "audio/mpeg"
            filename = "tts_response.wav" if media_type == "audio/wav" else "tts_response.mp3"
            return Response(
                content=cached_audio,
                media_type=media_type,
                headers={
                    "Content-Disposition": f"inline; filename={filename}",
                    "X-Audio-Length": str(len(cached_audio)),
                    "X-TTS-Provider": "cache",
                    "X-TTS-Cache": "HIT",
                },
            )

    # Dynamic Active TTS model resolution from ModelManager
    active_tts = get_model_manager().get_active_model("tts") or {}
    active_tts_model = (request.model or active_tts.get("model_id") or active_tts.get("id") or "aura-asteria-en").strip()
    active_tts_provider = (active_tts.get("provider") or "").lower()
    active_tts_key = active_tts.get("api_key")

    # Engine routing:
    # - Hindi and Marathi ALWAYS route to Sarvam Bulbul v3 (Deepgram Aura only supports English).
    # - English routes to Deepgram Aura by default, or Sarvam if explicitly requested.
    if is_hindi or is_marathi:
        use_sarvam = True
    elif request.model and any(sv in request.model.lower() for sv in ("bulbul", "sarvam")):
        use_sarvam = True
    elif active_tts_provider in ("sarvam ai", "sarvam") and "bulbul" in active_tts_model.lower() and not request.model:
        use_sarvam = True
    else:
        use_sarvam = False

    if use_sarvam:
        if is_hindi:
            sarvam_lang = "hi-IN"
            chosen_voice = (
                request.speaker.strip().lower()
                if (request.speaker and request.speaker.strip().lower() not in ("simran", "default"))
                else (
                    "priya" if (is_module1 and not is_module2)
                    else os.getenv("SARVAM_HINDI_VOICE", "priya").strip().lower()
                )
            )
            if not chosen_voice or chosen_voice in ("simran", "default"):
                chosen_voice = "priya"
        elif is_marathi:
            sarvam_lang = "mr-IN"
            chosen_voice = (
                request.speaker.strip().lower()
                if (request.speaker and request.speaker.strip().lower() not in ("simran", "default"))
                else (
                    "ritu" if (is_module1 and not is_module2)
                    else os.getenv("SARVAM_MARATHI_VOICE", "ritu").strip().lower()
                )
            )
            if not chosen_voice or chosen_voice in ("simran", "default"):
                chosen_voice = "ritu"
        else:
            sarvam_lang = "en-IN"
            chosen_voice = (
                request.speaker.strip().lower()
                if request.speaker
                else os.getenv("SARVAM_ENGLISH_VOICE", os.getenv("SARVAM_MODULE1_ENGLISH_VOICE", "simran")).strip().lower()
            )
            if not chosen_voice or chosen_voice == "default":
                chosen_voice = "simran"

        try:
            sarvam_service = get_sarvam_tts_service(api_key=active_tts_key)
            audio_bytes = await sarvam_service.synthesize_speech(
                clean_text,
                language_code=sarvam_lang,
                speaker=chosen_voice,
                model=active_tts_model if "bulbul" in active_tts_model else "bulbul:v3",
            )
            print(f"[LATENCY] T7 TTS audio READY (sarvam) at perf={_lat_t3.perf_counter():.6f}s (took={(_lat_t3.perf_counter()-_lat_t3_start)*1000:.3f}ms)", flush=True)
            if is_module1 and not is_module2:
                set_cached_tts_audio(cache_key, audio_bytes)
            media_type = "audio/wav" if audio_bytes.startswith(b"RIFF") else "audio/mpeg"
            filename = "tts_response.wav" if media_type == "audio/wav" else "tts_response.mp3"
            return Response(
                content=audio_bytes,
                media_type=media_type,
                headers={
                    "Content-Disposition": f"inline; filename={filename}",
                    "X-Audio-Length": str(len(audio_bytes)),
                    "X-TTS-Provider": "sarvam",
                    "X-TTS-Voice": chosen_voice,
                    "X-TTS-Language": sarvam_lang,
                },
            )
        except (SarvamTTSConfigurationError, SarvamTTSAPIError, Exception) as sarvam_err:
            lang_label = "English" if is_english else ("Hindi" if is_hindi else "Marathi")
            module_label = "Module 2" if is_module2 else "Module 1"
            logger.warning(
                f"[{module_label} TTS] Sarvam Bulbul v3 {lang_label} ({sarvam_lang}) unavailable ({str(sarvam_err)})."
            )
            if is_hindi or is_marathi:
                # Return structured JSON instructing client to speak via native browser Web Speech API
                # Never fall back to Romanized Hindi/Marathi or American English phonetic speech!
                return JSONResponse(
                    content={
                        "fallback_to_browser": True,
                        "language": sarvam_lang,
                        "text": clean_text,
                        "speaker": chosen_voice,
                        "message": "Sarvam TTS quota unavailable. Use native browser speech synthesis."
                    },
                    status_code=200,
                    headers={"X-TTS-Fallback": "browser-speech-synthesis"}
                )
            # For English: fall through to Deepgram Aura TTS

    # Guard: Never send Hindi/Marathi or Devanagari text to English Deepgram Aura!
    if is_hindi or is_marathi or has_devanagari:
        fallback_lang = "hi-IN" if is_hindi else "mr-IN"
        fallback_speaker = "priya" if is_hindi else "ritu"
        logger.warning(
            f"Devanagari text routed to Deepgram Aura fallback blocked. Returning browser speech synthesis fallback for {fallback_lang}."
        )
        return JSONResponse(
            content={
                "fallback_to_browser": True,
                "language": fallback_lang,
                "text": clean_text,
                "speaker": fallback_speaker,
                "message": "Devanagari text cannot be synthesized by English Deepgram Aura. Use native browser speech synthesis."
            },
            status_code=200,
            headers={"X-TTS-Fallback": "browser-speech-synthesis"}
        )

    # Default / English Engine: Deepgram Aura TTS (140ms ultra-low latency female voice aura-asteria-en)
    try:
        effective_dg_model = request.model if (request.model and "aura" in request.model.lower()) else (
            active_tts_model if "aura" in active_tts_model.lower() else "aura-asteria-en"
        )
        tts_service = DeepgramTTSService(api_key=active_tts_key) if (active_tts_key and active_tts_provider == "deepgram") else DeepgramTTSService()
        audio_bytes = await tts_service.synthesize_speech(
            clean_text,
            model=effective_dg_model,
            language="en",
            skip_sarvam=True,
        )
        print(f"[LATENCY] T7 TTS audio READY (deepgram) at perf={_lat_t3.perf_counter():.6f}s (took={(_lat_t3.perf_counter()-_lat_t3_start)*1000:.3f}ms)", flush=True)
        if is_module1 and not is_module2:
            set_cached_tts_audio(cache_key, audio_bytes)
        media_type = "audio/wav" if audio_bytes.startswith(b"RIFF") else "audio/mpeg"
        filename = "tts_response.wav" if media_type == "audio/wav" else "tts_response.mp3"
        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "Content-Disposition": f"inline; filename={filename}",
                "X-Audio-Length": str(len(audio_bytes)),
                "X-TTS-Provider": "deepgram",
            },
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except DeepgramTTSConfigurationError as dce:
        logger.error(f"Deepgram TTS configuration error: {str(dce)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(dce),
        )
    except DeepgramTTSAPIError as dae:
        logger.error(f"Deepgram TTS API error: {str(dae)}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(dae),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during TTS speech synthesis: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text-to-speech synthesis failed: {str(e)}",
        )


@app.get(
    "/api/documents",
    tags=["PDF Documents"],
    summary="Retrieve list of uploaded documents stored in PostgreSQL",
)
async def list_documents_endpoint(limit: int = 50, offset: int = 0):
    """
    Returns metadata list of all permanently stored PDF documents in PostgreSQL,
    including total pages, chunk counts, Pinecone namespace, and upload date.
    Excludes binary file data for fast retrieval.
    """
    try:
        doc_repo = DocumentRepository()
        docs = doc_repo.get_documents(limit=limit, offset=offset)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "success", "count": len(docs), "documents": docs},
        )
    except DatabaseConnectionError as dce:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"PostgreSQL connection unavailable: {str(dce)}",
        )
    except DatabaseOperationError as doe:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query error: {str(doe)}",
        )


@app.get(
    "/api/documents/{document_id}",
    tags=["PDF Documents"],
    summary="Get document metadata and Pinecone vector tracing IDs",
)
async def get_document_endpoint(document_id: int):
    """
    Retrieves full metadata for an uploaded PDF, including extracted text,
    page breakdown, and Pinecone vector IDs for vector traceability.
    """
    try:
        doc_repo = DocumentRepository()
        doc = doc_repo.get_document(document_id, include_data=False)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} was not found.",
            )
        return JSONResponse(status_code=status.HTTP_200_OK, content=doc)
    except HTTPException:
        raise
    except DatabaseConnectionError as dce:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"PostgreSQL connection unavailable: {str(dce)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving document: {str(e)}",
        )


@app.get(
    "/api/documents/{document_id}/download",
    tags=["PDF Documents"],
    summary="Download the original binary PDF file from PostgreSQL",
)
async def download_document_endpoint(document_id: int):
    """
    Streams the original uploaded binary PDF file directly from PostgreSQL BYTEA storage.
    Ensures PDF documents remain fully available across server restarts.
    """
    try:
        doc_repo = DocumentRepository()
        doc = doc_repo.get_document(document_id, include_data=True)
        if not doc or not doc.get("file_data"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} was not found or has no binary data.",
            )

        file_bytes = doc["file_data"]
        filename = doc.get("filename") or f"document_{document_id}.pdf"
        mime_type = doc.get("mime_type") or "application/pdf"

        return Response(
            content=file_bytes,
            media_type=mime_type,
            headers={
                "Content-Disposition": f'inline; filename="{filename}"',
                "Content-Length": str(len(file_bytes)),
            },
        )
    except HTTPException:
        raise
    except DatabaseConnectionError as dce:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"PostgreSQL connection unavailable: {str(dce)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error downloading document: {str(e)}",
        )


# ============================================================================
# ROUTE: /api/telegram/webhook (POST) & /api/telegram/status (GET)
# ============================================================================
# Telegram Bot Webhook endpoint for VoiceCopilotBot.
# Connects Telegram messenger directly to Module 2 Knowledge Assistant:
# - Text messages -> RAG -> immediate text reply -> TTS voice reply.
# - Voice notes (.ogg) -> Sarvam STT (saaras:v4) -> RAG -> text reply -> TTS voice reply.
# ============================================================================
@app.post(
    "/api/telegram/webhook",
    tags=["Telegram Bot"],
    summary="Receive Telegram Webhook updates for VoiceCopilotBot",
)
async def telegram_webhook_endpoint(request: Request):
    """
    Receives incoming Telegram updates (text messages, voice notes) and dispatches them
    through Module 2 STT, RAG, and TTS pipeline.
    """
    try:
        body = await request.json()
    except Exception as parse_err:
        logger.warning(f"[Telegram Webhook] Failed to parse incoming JSON: {parse_err}")
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"ok": False, "error": "Invalid JSON"})

    from services.telegram_bot import get_voice_copilot_bot
    bot = get_voice_copilot_bot()
    try:
        result = await bot.handle_webhook_update(body)
        return JSONResponse(status_code=status.HTTP_200_OK, content={"ok": True, "result": result})
    except Exception as e:
        logger.error(f"[Telegram Webhook] Error processing update: {e}", exc_info=True)
        return JSONResponse(status_code=status.HTTP_200_OK, content={"ok": False, "error": str(e)})


@app.get(
    "/api/telegram/status",
    tags=["Telegram Bot"],
    summary="Get VoiceCopilotBot Telegram integration status",
)
async def telegram_status_endpoint():
    """
    Returns whether the Telegram bot token is configured and operational.
    """
    from services.telegram_bot import get_voice_copilot_bot
    bot = get_voice_copilot_bot()
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "bot_name": "VoiceCopilotBot",
            "module": "Module 2 (Knowledge Assistant)",
            "is_configured": bot.is_configured,
            "webhook_endpoint": "/api/telegram/webhook",
            "capabilities": ["text", "voice_notes", "stt", "rag", "tts"],
        },
    )


@app.post(
    "/api/telegram/set-webhook",
    tags=["Telegram Bot"],
    summary="Register Webhook URL with Telegram API",
)
async def telegram_set_webhook_endpoint(request: Request):
    """
    Registers a public webhook URL with Telegram's setWebhook API.
    Body format: {"webhook_url": "https://<your-public-host>/api/telegram/webhook"}
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    webhook_url = body.get("webhook_url", "").strip()
    if not webhook_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing 'webhook_url' in request body.",
        )

    from services.telegram_bot import get_voice_copilot_bot
    bot = get_voice_copilot_bot()
    if not bot.is_configured:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="TELEGRAM_BOT_TOKEN is not configured in backend/.env.",
        )

    url = f"{bot.api_url}/setWebhook"
    try:
        resp = await bot.client.post(url, json={"url": webhook_url})
        if resp.status_code == 200:
            bot.stop_polling_task()
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set webhook: {str(e)}",
        )


# ============================================================================
# MODEL MANAGEMENT API (STT, LLM, TTS)
# ============================================================================

from services.model_manager import (
    get_model_manager,
    get_active_model,
    get_active_model_id,
    get_active_model_key,
    get_active_model_config,
)


class CreateModelRequest(BaseModel):
    category: str = Field(..., description="stt, llm, or tts")
    name: str = Field(..., description="Human-friendly model name or identifier")
    api_key: Optional[str] = Field(default=None, description="API Key for the model")
    provider: Optional[str] = Field(default=None, description="Provider name e.g. OpenRouter, Deepgram, Sarvam AI, Custom")
    model_id: Optional[str] = Field(default=None, description="Exact API model string identifier")
    description: Optional[str] = Field(default=None, description="Optional description of model capabilities")
    configuration: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Configuration parameters JSON")
    is_active: Optional[bool] = Field(default=False, description="Whether to activate immediately")


class UpdateModelRequest(BaseModel):
    name: Optional[str] = None
    api_key: Optional[str] = None
    provider: Optional[str] = None
    model_id: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    configuration: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class ActivateModelRequest(BaseModel):
    category: str = Field(..., description="stt, llm, or tts")
    model_id: str = Field(..., description="ID of the model to activate")


@app.get("/api/models", tags=["Model Management"])
async def list_models_endpoint(category: Optional[str] = None):
    """Lists all registered models (STT, LLM, TTS), optionally filtered by category."""
    manager = get_model_manager()
    models = manager.get_all_models(category=category)
    return {"status": "success", "models": models}


@app.get("/api/models/active", tags=["Model Management"])
async def get_active_models_endpoint():
    """Returns the currently active model for STT, LLM, and TTS."""
    manager = get_model_manager()
    active = manager.get_active_models()
    return {"status": "success", "active": active}


@app.post("/api/models", status_code=status.HTTP_201_CREATED, tags=["Model Management"])
async def create_model_endpoint(req: CreateModelRequest):
    """Registers a new custom model using Category, Model Name, and API Key."""
    manager = get_model_manager()
    try:
        new_model = manager.add_model(
            category=req.category,
            name=req.name,
            api_key=req.api_key,
            provider=req.provider,
            model_id=req.model_id,
            configuration=req.configuration,
            description=req.description,
            set_active=bool(req.is_active),
        )
        return {"status": "success", "model": new_model}
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"[create_model_endpoint] Error creating model: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.put("/api/models/{model_id}", tags=["Model Management"])
async def update_model_endpoint(model_id: str, req: UpdateModelRequest):
    """Updates an existing model configuration or parameters."""
    manager = get_model_manager()
    try:
        update_data = {k: v for k, v in req.dict().items() if v is not None}
        updated = manager.update_model(model_id=model_id, data=update_data)
        return {"status": "success", "model": updated}
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"[update_model_endpoint] Error updating model {model_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.delete("/api/models/{model_id}", tags=["Model Management"])
async def delete_model_endpoint(model_id: str):
    """Permanently deletes any model (built-in or custom). If active, clears active selection."""
    manager = get_model_manager()
    try:
        manager.delete_model(model_id=model_id)
        return {
            "status": "success",
            "message": f"Model '{model_id}' deleted successfully.",
            "deleted_id": model_id
        }
    except ValueError as ve:
        message = str(ve)
        if "does not exist" in message:
            code = status.HTTP_404_NOT_FOUND
        else:
            code = status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=code, detail=message)
    except Exception as e:
        logger.error(f"[delete_model_endpoint] Error deleting model {model_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/models/restore-defaults", tags=["Model Management"])
async def restore_defaults_endpoint():
    """Restores missing default built-in models and restores default active models if unassigned."""
    manager = get_model_manager()
    try:
        models = manager.restore_defaults()
        active = manager.get_active_models()
        return {
            "status": "success",
            "message": "Default models restored successfully.",
            "models": models,
            "active": active
        }
    except Exception as e:
        logger.error(f"[restore_defaults_endpoint] Error restoring defaults: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/models/activate", tags=["Model Management"])
async def activate_model_endpoint(req: ActivateModelRequest):
    """Activates a model for the given category (STT, LLM, or TTS)."""
    manager = get_model_manager()
    try:
        activated = manager.set_active_model(category=req.category, model_id=req.model_id)
        return {"status": "success", "active": activated}
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"[activate_model_endpoint] Error activating model {req.model_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
