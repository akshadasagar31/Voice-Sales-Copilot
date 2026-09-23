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
)
from services.sarvam_stt import (
    SarvamSTTService,
    SarvamSTTError,
    SarvamSTTConfigurationError,
    SarvamSTTAPIError,
)
from services.language import (
    detect_language,
    detect_spoken_language,
    get_response_language,
    is_new_lead_intent,
    is_greeting,
    get_greeting_response,
    is_assistant_query,
    get_assistant_query_response,
    select_best_stt_transcript,
    LANG_MIXED,
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
    extract_email,
    extract_company,
    extract_loan_type,
    extract_loan_amount,
    extract_tenure_months,
    extract_name,
    is_valid_prospect_name,
    merge_lead_safely,
    get_clarification_prompt,
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
)
from services.sarvam_tts import (
    clean_hindi_financial_text,
    clean_marathi_financial_text,
    clean_english_financial_text,
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

        # If streaming mode is requested, return real-time Server-Sent Events (SSE)
        if request.stream:
            return StreamingResponse(
                rag_service.stream_answer_chunks(
                    question=clean_question,
                    top_k=request.top_k or 8,
                    namespace=target_namespace,
                    filter_dict=request.filter,
                    model=request.model,
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
            model=request.model,
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
):
    """
    Accepts recorded voice note audio (Blob/File), transcribes to text via Sarvam (for Module 2 Hindi)
    or Deepgram Nova-3 (for English, Marathi, Module 1, and fallback).
    """
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
        is_hindi = req_lang in ("hi", "hi-in", "hindi")
        is_marathi = req_lang in ("mr", "mr-in", "marathi")
        is_english = req_lang in ("en", "en-in", "en-us", "english")
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
                dg_lang = "multi"

            stt_service = DeepgramSTTService()
            result = await stt_service.transcribe_audio(
                content,
                content_type=content_type,
                model="nova-3",
                language=dg_lang,
            )
            result["filename"] = upload.filename
            raw_transcript = normalize_stt_transcript(result.get("transcript", ""))
            result["transcript"] = raw_transcript
            result["stt_provider"] = "deepgram"
            result["model"] = "nova-3"
            greeting_flag = is_greeting(raw_transcript)
            result["is_greeting"] = greeting_flag

            # Authoritative language resolution for Module 1 from actual spoken transcript
            spoken_lang = detect_spoken_language(raw_transcript)
            resp_lang = get_response_language(spoken_lang)
            result["detected_language"] = spoken_lang
            result["language"] = resp_lang

            if greeting_flag:
                greeting_text = get_greeting_response(resp_lang)
                result["greeting_response"] = greeting_text
                result["immediate_sentence1"] = greeting_text
                result["is_greeting"] = True
            elif is_assistant_query(raw_transcript):
                assistant_answer = get_assistant_query_response(raw_transcript, resp_lang)
                result["is_assistant_query"] = True
                result["assistant_response"] = assistant_answer
                result["immediate_sentence1"] = assistant_answer
                clean_existing = {}
                if existing_lead:
                    try:
                        clean_existing = json.loads(existing_lead) if isinstance(existing_lead, str) else dict(existing_lead)
                    except Exception:
                        clean_existing = {}
                result["lead"] = clean_existing
                result["lead_id"] = lead_id
                logger.info(
                    f"[voice-entry:module1] Assistant query handled: '{raw_transcript}' -> '{assistant_answer}'"
                )
                return JSONResponse(status_code=status.HTTP_200_OK, content=result)
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

                    next_missing = get_next_missing_parameter(immediate_lead)
                    immediate_sentence1 = get_missing_parameter_prompt(next_missing, immediate_lead, lang=resp_lang)

                    result["immediate_sentence1"] = immediate_sentence1
                    result["lead"] = immediate_lead
                    result["next_missing_parameter"] = next_missing
                    result["is_complete"] = next_missing is None

                    # Non-blocking database sync: update lead in background when lead_id exists;
                    # on initial turn, create lead using pooled connection (<1ms) to establish ID
                    has_data = any(v is not None for v in immediate_lead.values() if v != "")
                    if has_data:
                        try:
                            from services.lead_repository import LeadRepository
                            repo = LeadRepository()
                            val_lead = Lead.model_validate(immediate_lead)
                            if lead_id is not None:
                                background_tasks.add_task(repo.update_lead, lead_id, val_lead)
                            else:
                                created = repo.create_lead(val_lead)
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

            deepgram_service = DeepgramSTTService()
            result = await deepgram_service.transcribe_audio(
                content,
                content_type=content_type,
                language=dg_lang,
                model="nova-3",
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
        raw_transcript = normalize_stt_transcript(result.get("transcript", ""))
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
):
    await websocket.accept()

    api_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    if not api_key:
        await websocket.send_json({"type": "error", "message": "DEEPGRAM_API_KEY not configured in backend/.env"})
        await websocket.close(code=1008)
        return

    req_lang = (language or "").strip().lower()
    is_explicit_single = req_lang in ("en", "hi", "mr")
    is_mod1 = (module or "").strip().lower() in ("module1", "crm", "voice_copilot", "voice-copilot")

    headers = {"Authorization": f"Token {api_key}"}

    logger.info(f"[ws/voice-stt] Client connected. Mode: {'single (' + req_lang + ')' if is_explicit_single else 'dual-stream (mr + multi)'}, module: {module}")

    try:
        if is_explicit_single:
            # Single-stream mode when user explicitly chose language
            single_url = build_deepgram_ws_url(
                model="nova-3",
                sample_rate=sample_rate,
                language=req_lang,
                encoding=encoding,
            )
            async with websockets.connect(single_url, additional_headers=headers) as dg_ws:
                accumulated_finals = []
                latest_interim = ""
                sent_final = False
                last_conf = 1.0
                single_pcm = bytearray()

                def get_current_transcript(include_interim: bool = True) -> str:
                    parts = list(accumulated_finals)
                    if include_interim and latest_interim and latest_interim.strip():
                        parts.append(latest_interim.strip())
                    return normalize_stt_transcript(" ".join(parts))

                closed_single = False

                async def client_to_single():
                    nonlocal closed_single
                    try:
                        while True:
                            msg = await websocket.receive()
                            if "bytes" in msg and msg["bytes"]:
                                single_pcm.extend(msg["bytes"])
                                await dg_ws.send(msg["bytes"])
                            elif "text" in msg and msg["text"]:
                                try:
                                    payload = json.loads(msg["text"])
                                    if payload.get("type") == "CloseStream":
                                        if not closed_single:
                                             closed_single = True
                                             await dg_ws.send(json.dumps({"type": "CloseStream"}))
                                        break
                                except Exception:
                                    pass
                    except (WebSocketDisconnect, asyncio.CancelledError):
                        pass
                    except Exception as e:
                        logger.debug(f"[ws/voice-stt] client_to_single exception: {e}")
                    finally:
                        if not closed_single:
                            closed_single = True
                            try:
                                await dg_ws.send(json.dumps({"type": "CloseStream"}))
                            except Exception:
                                pass

                async def single_to_client():
                    nonlocal sent_final, last_conf, latest_interim
                    try:
                        async for dg_msg in dg_ws:
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

                                        is_speech_final = data.get("speech_final", False)
                                        if data.get("is_final", False):
                                            if tr:
                                                accumulated_finals.append(tr)
                                                latest_interim = ""
                                            current = get_current_transcript(include_interim=False)
                                            # If speech_final is true, user finished speaking their utterance
                                            if is_speech_final and current and not sent_final:
                                                lang = detect_spoken_language(current) if req_lang == "auto" else req_lang
                                                resp_lang = get_response_language(lang)
                                                is_greet = is_greeting(current)
                                                greet_resp = get_greeting_response(resp_lang) if is_greet else ""
                                                asst_resp = get_assistant_query_response(current, resp_lang)
                                                await websocket.send_json({
                                                    "type": "final",
                                                    "transcript": current,
                                                    "confidence": round(last_conf, 4),
                                                    "detected_language": lang,
                                                    "language": resp_lang,
                                                    "is_greeting": is_greet,
                                                    "greeting_response": greet_resp,
                                                    "is_assistant_query": bool(asst_resp),
                                                    "assistant_response": asst_resp or "",
                                                    "speech_final": True,
                                                })
                                                sent_final = True
                                            elif current:
                                                await websocket.send_json({"type": "interim", "transcript": current, "is_final": False})
                                        elif tr:
                                            latest_interim = tr
                                            current = get_current_transcript(include_interim=True)
                                            if current:
                                                await websocket.send_json({"type": "interim", "transcript": current, "is_final": False})
                                elif msg_type == "Metadata":
                                    clean = get_current_transcript(include_interim=False) or get_current_transcript(include_interim=True)
                                    if clean and not sent_final:
                                        lang = detect_spoken_language(clean) if req_lang == "auto" else req_lang
                                        resp_lang = get_response_language(lang)
                                        is_greet = is_greeting(clean)
                                        greet_resp = get_greeting_response(resp_lang) if is_greet else ""
                                        asst_resp = get_assistant_query_response(clean, resp_lang)
                                        await websocket.send_json({
                                            "type": "final",
                                            "transcript": clean,
                                            "confidence": round(last_conf, 4),
                                            "detected_language": lang,
                                            "language": resp_lang,
                                            "is_greeting": is_greet,
                                            "greeting_response": greet_resp,
                                            "is_assistant_query": bool(asst_resp),
                                            "assistant_response": asst_resp or "",
                                            "speech_final": True,
                                        })
                                        sent_final = True
                                    await websocket.send_json({"type": "metadata", "metadata": data})
                                    break
                    except (WebSocketDisconnect, asyncio.CancelledError):
                        pass
                    except Exception as e:
                        logger.debug(f"[ws/voice-stt] single_to_client exception: {e}")
                    finally:
                        if not sent_final:
                            clean = get_current_transcript(include_interim=False) or get_current_transcript(include_interim=True)
                            if clean:
                                try:
                                    lang = detect_spoken_language(clean) if req_lang == "auto" else req_lang
                                    resp_lang = get_response_language(lang)
                                    is_greet = is_greeting(clean)
                                    greet_resp = get_greeting_response(resp_lang) if is_greet else ""
                                    asst_resp = get_assistant_query_response(clean, resp_lang)
                                    await websocket.send_json({
                                        "type": "final",
                                        "transcript": clean,
                                        "confidence": round(last_conf, 4),
                                        "detected_language": lang,
                                        "language": resp_lang,
                                        "is_greeting": is_greet,
                                        "greeting_response": greet_resp,
                                        "is_assistant_query": bool(asst_resp),
                                        "assistant_response": asst_resp or "",
                                        "speech_final": True,
                                    })
                                except Exception:
                                    pass

                c_task = asyncio.create_task(client_to_single())
                s_task = asyncio.create_task(single_to_client())
                await c_task
                try:
                    await asyncio.wait_for(s_task, timeout=3.5)
                except asyncio.TimeoutError:
                    s_task.cancel()
                except Exception:
                    pass

        else:
            # Dual-stream mode (Auto-Detect): Stream audio to both Nova-3 Marathi and Nova-3 Multilingual
            url_mr = build_deepgram_ws_url(
                model="nova-3",
                sample_rate=sample_rate,
                language="mr",
                encoding=encoding,
            )
            url_multi = build_deepgram_ws_url(
                model="nova-3",
                sample_rate=sample_rate,
                language="multi",
                encoding=encoding,
            )

            async with websockets.connect(url_mr, additional_headers=headers) as ws_mr, \
                       websockets.connect(url_multi, additional_headers=headers) as ws_multi:

                finals = {"mr": [], "multi": []}
                interims = {"mr": "", "multi": ""}
                confs = {"mr": 1.0, "multi": 1.0}
                sent_final = False
                metadata_payload = None
                dual_pcm = bytearray()

                def build_stream_transcript(name: str) -> str:
                    parts = list(finals[name])
                    if interims[name] and interims[name].strip():
                        parts.append(interims[name].strip())
                    return normalize_stt_transcript(" ".join(parts))

                async def safe_send(ws, payload):
                    try:
                        await ws.send(payload)
                    except Exception:
                        pass

                closed_dual = False

                async def client_to_dual():
                    nonlocal closed_dual
                    try:
                        while True:
                            msg = await websocket.receive()
                            if "bytes" in msg and msg["bytes"]:
                                chunk = msg["bytes"]
                                dual_pcm.extend(chunk)
                                await asyncio.gather(safe_send(ws_mr, chunk), safe_send(ws_multi, chunk))
                            elif "text" in msg and msg["text"]:
                                try:
                                    payload = json.loads(msg["text"])
                                    if payload.get("type") == "CloseStream":
                                        if not closed_dual:
                                            closed_dual = True
                                            close_msg = json.dumps({"type": "CloseStream"})
                                            await asyncio.gather(safe_send(ws_mr, close_msg), safe_send(ws_multi, close_msg))
                                        break
                                except Exception:
                                    pass
                    except (WebSocketDisconnect, asyncio.CancelledError):
                        pass
                    except Exception as e:
                        logger.debug(f"[ws/voice-stt] client_to_dual exception: {e}")
                    finally:
                        if not closed_dual:
                            closed_dual = True
                            close_msg = json.dumps({"type": "CloseStream"})
                            await asyncio.gather(safe_send(ws_mr, close_msg), safe_send(ws_multi, close_msg))

                async def listen_stream(ws, name: str):
                    nonlocal metadata_payload
                    try:
                        async for dg_msg in ws:
                            if isinstance(dg_msg, str):
                                data = json.loads(dg_msg)
                                msg_type = data.get("type")
                                if msg_type == "Results":
                                    alts = (data.get("channel") or {}).get("alternatives") or []
                                    if alts:
                                        alt = alts[0]
                                        tr = (alt.get("transcript") or "").strip()
                                        c = alt.get("confidence", 0.0)
                                        if c > 0:
                                            confs[name] = c
                                        if data.get("is_final", False):
                                            if tr:
                                                finals[name].append(tr)
                                                interims[name] = ""
                                        elif tr:
                                            interims[name] = tr

                                        # Forward real-time interim to client
                                        txt_mr = build_stream_transcript("mr")
                                        txt_multi = build_stream_transcript("multi")
                                        # Prefer Marathi interim if Marathi tokens present, otherwise multi
                                        lang_mr = detect_language(txt_mr)
                                        display_interim = txt_mr if lang_mr == "mr" and txt_mr else (txt_multi or txt_mr)
                                        if display_interim:
                                            try:
                                                await websocket.send_json({"type": "interim", "transcript": display_interim})
                                            except Exception:
                                                pass
                                elif msg_type == "Metadata":
                                    if not metadata_payload:
                                        metadata_payload = data
                                    break
                    except (WebSocketDisconnect, asyncio.CancelledError):
                        pass
                    except Exception as e:
                        logger.debug(f"[ws/voice-stt] listen_stream({name}) exception: {e}")

                c_task = asyncio.create_task(client_to_dual())
                l_mr_task = asyncio.create_task(listen_stream(ws_mr, "mr"))
                l_multi_task = asyncio.create_task(listen_stream(ws_multi, "multi"))

                await c_task
                try:
                    await asyncio.wait_for(asyncio.gather(l_mr_task, l_multi_task), timeout=3.5)
                except asyncio.TimeoutError:
                    l_mr_task.cancel()
                    l_multi_task.cancel()
                except Exception:
                    pass

                clean_mr = build_stream_transcript("mr")
                clean_multi = build_stream_transcript("multi")

                chosen_text, chosen_lang, source = select_best_stt_transcript(clean_mr, clean_multi)
                winning_conf = confs["mr"] if "MR" in source else confs["multi"]



                if chosen_text and not sent_final:
                    spoken_l = detect_spoken_language(chosen_text)
                    resp_l = get_response_language(spoken_l)
                    is_greet = is_greeting(chosen_text)
                    greet_resp = get_greeting_response(resp_l) if is_greet else ""
                    asst_resp = get_assistant_query_response(chosen_text, resp_l)
                    logger.info(
                        f"[ws/voice-stt] Dual-stream resolution -> {source} (lang: {chosen_lang}, conf: {winning_conf:.2f}): '{chosen_text}'"
                    )
                    await websocket.send_json({
                        "type": "final",
                        "transcript": chosen_text,
                        "confidence": round(winning_conf, 4),
                        "detected_language": chosen_lang,
                        "language": resp_l,
                        "is_greeting": is_greet,
                        "greeting_response": greet_resp,
                        "is_assistant_query": bool(asst_resp),
                        "assistant_response": asst_resp or "",
                        "speech_final": True,
                    })
                    sent_final = True

                if metadata_payload:
                    await websocket.send_json({"type": "metadata", "metadata": metadata_payload})

    except Exception as exc:
        logger.error(f"[ws/voice-stt] WebSocket error: {exc}")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
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
        extractor = LeadExtractorService()

        if request.stream:
            return StreamingResponse(
                extractor.stream_lead_turn(
                    transcript=clean_transcript,
                    model=request.model,
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
            model=request.model,
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
        get_shared_stt_client()
        get_shared_tts_client()
        logger.info("Startup check: Deepgram STT and TTS keep-alive connection pools initialized.")
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
    clean_text = (request.text or "").strip()
    if not clean_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'text' field cannot be empty or whitespace.",
        )

    # Determine target language and caller context
    target_lang = (request.language or detect_text_language(clean_text) or "en").strip().lower()
    is_marathi = target_lang in ("mr", "mr-in", "marathi")
    is_hindi = target_lang in ("hi", "hi-in", "hindi")
    is_english = target_lang in ("en", "en-in", "en-us", "en-gb", "english")
    is_module2 = request.module in ("module2", "knowledge_assistant", "rag")

    # Module 2 (Knowledge Assistant / Ask Assistant) & Module 1:
    # Use Sarvam Bulbul v3 first ('en-IN', 'hi-IN', 'mr-IN' with voice 'simran').
    # If Sarvam requires credits, returns HTTP 402/insufficient_quota, or fails due to quota/cost/error,
    # automatically fallback to Deepgram TTS.
    use_sarvam = (
        is_module2
        or (is_english or is_hindi or is_marathi)
        or request.model in ("bulbul:v3", "sarvam")
    )

    is_module1 = request.module in ("module1", "voice_copilot", "voice-copilot", "crm") or not is_module2

    if use_sarvam:
        if is_module1:
            if is_hindi:
                sarvam_lang = "hi-IN"
                chosen_voice = (
                    request.speaker
                    if (request.speaker and request.speaker.lower() not in ("simran", "default"))
                    else os.getenv("SARVAM_MODULE1_HINDI_VOICE", "priya").strip().lower()
                )
            elif is_marathi:
                sarvam_lang = "mr-IN"
                chosen_voice = (
                    request.speaker
                    if (request.speaker and request.speaker.lower() not in ("simran", "default"))
                    else os.getenv("SARVAM_MODULE1_MARATHI_VOICE", "ritu").strip().lower()
                )
            else:
                sarvam_lang = "en-IN"
                chosen_voice = request.speaker or os.getenv("SARVAM_ENGLISH_VOICE", "simran").strip().lower()
        else:
            # Module 2 (Knowledge Assistant / Telegram / RAG) - keep unchanged as simran
            if is_hindi:
                sarvam_lang = "hi-IN"
                chosen_voice = request.speaker or os.getenv("SARVAM_HINDI_VOICE", "simran").strip().lower()
            elif is_marathi:
                sarvam_lang = "mr-IN"
                chosen_voice = request.speaker or os.getenv("SARVAM_MARATHI_VOICE", "simran").strip().lower()
            else:
                sarvam_lang = "en-IN"
                chosen_voice = request.speaker or os.getenv("SARVAM_ENGLISH_VOICE", "simran").strip().lower()

        try:
            sarvam_service = SarvamTTSService()
            audio_bytes = await sarvam_service.synthesize_speech(
                clean_text,
                language_code=sarvam_lang,
                speaker=chosen_voice,
                model="bulbul:v3",
            )
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
                f"[{module_label} TTS] Sarvam Bulbul v3 {lang_label} ({sarvam_lang}) failed or out of credits ({str(sarvam_err)}). "
                "Automatically falling back to Deepgram TTS."
            )
            # Fall through to DeepgramTTSService below

    # Default / Automatic Fallback: Deepgram Aura TTS
    try:
        tts_service = DeepgramTTSService()
        audio_bytes = await tts_service.synthesize_speech(
            clean_text,
            model=request.model if request.model != "bulbul:v3" else None,
            language=request.language,
            skip_sarvam=True,
        )
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








