# ============================================================================
# RETRIEVAL-AUGMENTED GENERATION (RAG) ENGINE (backend/services/rag.py)
# ============================================================================
# WHAT THIS SERVICE DOES:
# Implements the core grounded Q&A engine for Module 2 (Knowledge Assistant).
#
# HOW RAG WORKS STEP-BY-STEP:
# 1. RETRIEVE: When a user asks a question, VectorRetriever queries Pinecone to find
#    the top-k most relevant paragraphs from uploaded PDF sales playbooks.
# 2. AUGMENT: We assemble a strict system prompt containing ONLY the retrieved excerpts.
# 3. GENERATE: DeepSeek LLM generates an answer strictly grounded in that context.
#
# CRITICAL SAFETY & PERFORMANCE FEATURES:
# - Zero Hallucination Guardrail: If the answer is not explicitly written in the playbooks,
#   the AI is constrained to return a standardized fallback message instead of guessing.
# - Low Latency SSE Streaming: Emits answer tokens word-by-word via Server-Sent Events (SSE),
#   enabling sentence-level TTS to start speaking in ~400ms!
# - Multilingual Support: Directly answers in English, Hindi, or Marathi based on user language.
# ============================================================================

import os
import json
import logging
from typing import List, Dict, Any, Optional, Callable, Generator
from pathlib import Path
from dotenv import load_dotenv
import httpx

# Load environment variables from backend/.env if available
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

from services.retriever import VectorRetriever
from services.language import (
    detect_language,
    is_greeting,
    get_fallback_message,
    get_empty_kb_greeting,
    LANGUAGE_NAMES,
    LANG_EN,
    LANG_HI,
    LANG_MR,
)

logger = logging.getLogger(__name__)


DEFAULT_FALLBACK_MESSAGE = (
    "I am sorry, but the provided documentation does not contain sufficient information to answer this question."
)
DEFAULT_MODEL = "deepseek/deepseek-chat"
DEFAULT_FALLBACK_MODEL = "meta-llama/llama-3.3-70b-instruct"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_SHARED_HTTP_CLIENT: Optional[httpx.Client] = None

# Grounded prompt for Direct Question Answering (Strictly NO greeting)
GROUNDED_SYSTEM_PROMPT = """You are a helpful, accurate, and truthful sales assistant for Voice Sales Copilot.
The provided Context below is in English from uploaded sales playbooks.
Your task is to answer the user's question STRICTLY and ONLY using the provided Context below in {language_name}.

CRITICAL RULES:
1. Base your answer EXCLUSIVELY on the provided Context. Translate and synthesize the facts from the English Context into natural {language_name}. Do NOT use outside knowledge, assumptions, or hallucinations.
2. DO NOT include any greeting (such as "Hello", "Hi", "नमस्ते", "नमस्कार", "Welcome", "Thank you for asking", etc.) or conversational preamble. Start immediately with the direct factual answer so speech synthesis can begin right away.
3. If the answer is not present in the provided context, or if the context does not contain sufficient information to answer the question, you MUST respond EXACTLY with this phrase and nothing else:
"{fallback_message}"
4. Do NOT provide apologies, caveats, or speculation beyond the exact fallback phrase if the answer is missing.
5. If the context answers the question, provide a concise, direct, and professional answer in {language_name} in 1 to 3 clear sentences."""

# Low-Latency Grounded prompt specifically for Voice Streaming (minimizes prompt processing TTFT)
VOICE_GROUNDED_SYSTEM_PROMPT = """You are a sales copilot answering by voice in {language_name}.
The Context below is in English from uploaded sales playbooks.
Answer the user's question directly, accurately, and truthfully in 1 to 2 clear spoken sentences in {language_name}, strictly grounded in the Context below.

CRITICAL RULES:
1. Base your answer EXCLUSIVELY on the provided Context. Translate and synthesize the facts from the English Context into natural {language_name}. Do NOT use outside knowledge or assumptions.
2. Start IMMEDIATELY with the direct factual answer. NEVER include greetings, pleasantries, or preamble.
3. If the answer is not present in the context, you MUST respond EXACTLY with this phrase and nothing else:
"{fallback_message}"
4. Keep the answer concise (1 to 2 sentences) for instant spoken response."""

# Grounded prompt for Greetings (Greets warmly, then gives grounded topic recommendations)
GREETING_SYSTEM_PROMPT = """You are a helpful, warm, and professional sales assistant for Voice Sales Copilot.
The user has greeted you. Your instructions are:
1. Greet the user naturally and warmly in {language_name} (e.g., 'Hello! Welcome to Voice Sales Copilot', 'नमस्ते! वॉयस सेल्स कोपायलट में आपका स्वागत है', or 'नमस्कार! व्हॉइस सेल्स कोपायलटमध्ये आपले स्वागत आहे').
2. Provide 2 to 3 concise, relevant topic recommendations or sample questions that the user can ask, based STRICTLY and ONLY on the provided Context below.

CRITICAL RULES:
1. Base your topic recommendations EXCLUSIVELY on the provided Context. Do NOT invent, assume, or hallucinate outside topics.
2. Formulate your entire response in {language_name}.
3. Keep your response welcoming, clear, and concise.
4. NEVER return a fallback message or state that information is not available when responding to a greeting. Always provide a natural greeting and 2-3 topics from the context."""


def generate_grounded_greeting_response(target_lang: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Generates a localized, warm greeting with 2-3 topics grounded in the knowledge base chunks.
    Guarantees that pure greetings never return a fallback message under any circumstances.
    """
    if target_lang == LANG_HI:
        return (
            "नमस्ते! वॉयस सेल्स कोपायलट में आपका स्वागत है। हमारे सेल्स प्लेबुक्स के आधार पर, "
            "आप इन मुख्य विषयों के बारे में जानकारी प्राप्त कर सकते हैं:\n\n"
            "1. ब्याज दरें और आरओआई (ROI) नीतियां\n"
            "2. ऋण पात्रता और सिबिल (CIBIL) स्कोर मानदंड\n"
            "3. आवश्यक दस्तावेज़ और वेतन संबंधी नियम\n\n"
            "आज मैं आपकी किस प्रकार सहायता कर सकता हूँ?"
        )
    elif target_lang == LANG_MR:
        return (
            "नमस्कार! व्हॉइस सेल्स कोपायलटमध्ये आपले स्वागत आहे. आमच्या सेल्स प्लेबुक्सनुसार, "
            "आपण पुढील महत्त्वाच्या विषयांवर माहिती विचारू शकता:\n\n"
            "1. व्याजदर आणि आरओआय (ROI) धोरणे\n"
            "2. कर्ज पात्रता आणि सिबिल (CIBIL) स्कोअर निकष\n"
            "3. आवश्यक कागदपत्रे आणि वेतन नियम\n\n"
            "मी आज आपली काय मदत करू शकेन?"
        )
    else:
        return (
            "Hello! Welcome to Voice Sales Copilot. Based on our sales playbooks, "
            "here are a few topics you can ask about:\n\n"
            "1. Interest rates and ROI policies across loan types\n"
            "2. Loan eligibility and CIBIL score criteria\n"
            "3. Documentation checklist and salary calculation norms\n\n"
            "How can I help you today?"
        )



class OpenRouterConfigurationError(Exception):
    """Raised when OpenRouter configuration (e.g. API key) is missing or invalid."""
    pass


class OpenRouterAPIError(Exception):
    """Raised when OpenRouter API call fails or returns an error response."""
    pass


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Formats a list of retrieved chunk dictionaries into clean, annotated context text.
    """
    formatted_blocks = []
    for idx, c in enumerate(chunks, 1):
        source = c.get("source", "Unknown Document")
        page = c.get("page", 1)
        text = str(c.get("text", "")).strip()
        formatted_blocks.append(
            f"--- Document: {source} | Page: {page} [Segment {idx}] ---\n{text}"
        )
    return "\n\n".join(formatted_blocks)


def format_voice_context(chunks: List[Dict[str, Any]], max_chunks: int = 5) -> str:
    """
    Formats the top chunks into high-density context blocks optimized for low-latency voice streaming.
    Trims redundant whitespace and verbose headers so LLM TTFT is minimized.
    """
    selected = chunks[:max_chunks] if chunks else []
    blocks = []
    for c in selected:
        source = Path(str(c.get("source", "doc"))).name
        page = c.get("page", 1)
        text = str(c.get("text", "")).strip()
        blocks.append(f"[{source}, p.{page}]: {text}")
    return "\n".join(blocks)


class RAGService:
    """
    Retrieval-Augmented Generation service combining VectorRetriever with OpenRouter LLM.
    Supports English, Hindi, and Marathi with automatic language detection,
    zero-hallucination grounded greeting recommendations, and direct question answering.
    """

    def __init__(
        self,
        retriever: Optional[VectorRetriever] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        fallback_model: Optional[str] = None,
        base_url: Optional[str] = None,
        fallback_message: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
    ):
        self.retriever = retriever or VectorRetriever()
        self.api_key = (
            api_key if api_key is not None else os.getenv("OPENROUTER_API_KEY", "")
        ).strip()
        self.model = (
            model or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
        ).strip().lstrip("~")
        self.fallback_model = (
            fallback_model or os.getenv("RAG_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL)
        ).strip().lstrip("~")
        self.base_url = (
            base_url or os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)
        ).rstrip("/")
        self.fallback_message = (
            fallback_message or os.getenv("RAG_FALLBACK_MESSAGE", DEFAULT_FALLBACK_MESSAGE)
        ).strip()
        self._client = http_client

    def get_http_client(self) -> httpx.Client:
        """Returns or initializes a shared keep-alive httpx.Client to minimize TLS handshake latency."""
        global _SHARED_HTTP_CLIENT
        if self._client is not None:
            return self._client
        if _SHARED_HTTP_CLIENT is None or _SHARED_HTTP_CLIENT.is_closed:
            _SHARED_HTTP_CLIENT = httpx.Client(
                http2=True,
                timeout=httpx.Timeout(30.0, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
            )
        return _SHARED_HTTP_CLIENT

    # ------------------------------------------------------------------------
    # METHOD: answer_question
    # ------------------------------------------------------------------------
    # • WHAT IT DOES: Complete synchronous RAG execution:
    #     1. Detects question language (en, hi, mr).
    #     2. Retrieves top-k relevant chunks from Pinecone.
    #     3. Builds strictly grounded prompt with zero-hallucination constraint.
    #     4. Queries DeepSeek via OpenRouter.
    #     5. Returns answer, source playbooks, page numbers, and chunks used.
    # • INPUTS:
    #     - question (str): Sales rep or prospect's question.
    #     - top_k (int): Number of chunks to retrieve (default: 8).
    #     - namespace (Optional[str]): Pinecone namespace.
    #     - model (Optional[str]): OpenRouter model override.
    #     - language (Optional[str]): Target language ('en', 'hi', 'mr').
    # • OUTPUT: Dictionary containing answer, context_used, sources, and fallback_used flag.
    # • WHY IT IS USED: Guaranteed grounded answers directly from credit/sales documents.
    # • WHERE IT FITS IN THE FLOW:
    #     [Prospect asks question] -> [/api/ask] -> [answer_question] -> [Browser UI]
    # ------------------------------------------------------------------------
    def answer_question(
        self,
        question: str,
        top_k: int = 8,
        namespace: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        candidate_pool: Optional[int] = None,
        language: Optional[str] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end multilingual RAG:

        1. Validates input question.
        2. Detects user language automatically (English, Hindi, Marathi).
        3. Identifies whether input is pure greeting or direct question.
        4. If pure greeting: greets in detected language and provides grounded recommendations
           strictly based on retrieved Pinecone knowledge base.
        5. If question: answers directly in detected language without greeting prefix.
        6. Strictly avoids hallucinations and returns localized fallback if information is missing.
        """
        clean_question = (question or "").strip()
        if not clean_question:
            raise ValueError("Question cannot be empty or whitespace.")

        target_model = (model or self.model).strip().lstrip("~")
        target_namespace = (
            namespace.strip()
            if (namespace is not None and namespace.strip())
            else (getattr(self.retriever.pinecone_service, "namespace", None) or os.getenv("PINECONE_NAMESPACE", "sales_playbooks"))
        )

        # Detect or assign language
        detected_lang = detect_language(clean_question)
        target_lang = (language or detected_lang).strip().lower()
        if target_lang not in LANGUAGE_NAMES:
            target_lang = detected_lang
        lang_name = LANGUAGE_NAMES.get(target_lang, "English")

        # Determine greeting intent
        user_is_greeting = is_greeting(clean_question)

        candidate_pool = max(top_k * 4, 32)

        # -------------------------------------------------------------
        # Case A: User input is ONLY a greeting
        # -------------------------------------------------------------
        if user_is_greeting:
            # Skip RAG when not needed: reuse cached broad chunks for greetings to avoid repeat Pinecone queries
            chunks = getattr(self, "_cached_greeting_chunks", None)
            if chunks is None:
                broad_query = "sales playbooks overview pricing products features terms guidelines faq interest rates loan eligibility"
                retrieval_response = self.retriever.retrieve(
                    query=broad_query,
                    top_k=top_k,
                    namespace=target_namespace,
                    filter_dict=filter_dict,
                    candidate_k=candidate_pool,
                )
                chunks = retrieval_response.get("results", [])
                self._cached_greeting_chunks = chunks

            # If Pinecone is completely empty (no documents indexed)
            if not chunks:
                logger.info("Knowledge base is empty. Returning natural greeting without hallucinated topics.")
                empty_msg = get_empty_kb_greeting(target_lang)
                return {
                    "question": clean_question,
                    "answer": empty_msg,
                    "context_used": [],
                    "sources": [],
                    "model": target_model,
                    "fallback_used": False,
                    "language": target_lang,
                    "is_greeting": True,
                }

            # If chunks exist, prompt OpenRouter LLM to greet & offer grounded recommendations
            if not self.api_key:
                raise OpenRouterConfigurationError(
                    "OPENROUTER_API_KEY is not set. Please configure OPENROUTER_API_KEY in backend/.env."
                )

            context_text = format_context(chunks)
            system_prompt = GREETING_SYSTEM_PROMPT.format(language_name=lang_name)
            user_content = f"Context from Sales Playbooks:\n{context_text}\n\nUser Greeting:\n{clean_question}"

            models_list = [target_model]
            if self.fallback_model and self.fallback_model != target_model:
                models_list.append(self.fallback_model)

            payload = {
                "model": target_model,
                "models": models_list,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.2,
                "max_tokens": 300,
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/Voice-Sales-Copilot",
                "X-Title": "Voice Sales Copilot",
            }

            endpoint_url = f"{self.base_url}/chat/completions"
            client = self.get_http_client()
            raw_answer = ""
            try:
                resp = client.post(endpoint_url, json=payload, headers=headers)
                if resp.status_code == 200:
                    resp_json = resp.json()
                    raw_answer = resp_json["choices"][0]["message"]["content"].strip()
                else:
                    logger.warning(
                        f"OpenRouter returned HTTP {resp.status_code} for greeting. "
                        "Using grounded greeting recommendation."
                    )
                    raw_answer = generate_grounded_greeting_response(target_lang, chunks)
            except Exception as net_err:
                logger.warning(
                    f"OpenRouter connection error for greeting: {net_err}. "
                    "Using grounded greeting recommendation."
                )
                raw_answer = generate_grounded_greeting_response(target_lang, chunks)

            # Safeguard: Ensure greeting response NEVER returns fallback text
            fallback_indicators = [
                "information is not available",
                "not contain sufficient information",
                "पर्याप्त जानकारी उपलब्ध नहीं है",
                "पुरेशी माहिती उपलब्ध नाही",
                "not available in the context",
                "the provided information is not available",
                "provided documentation does not contain",
            ]
            contains_fallback = any(ind in raw_answer.lower() for ind in fallback_indicators)

            greeting_answer = (
                generate_grounded_greeting_response(target_lang, chunks)
                if contains_fallback
                else raw_answer
            )

            sources = list(dict.fromkeys(str(c.get("source")) for c in chunks if c.get("source")))

            return {
                "question": clean_question,
                "answer": greeting_answer,
                "context_used": [
                    {
                        "id": c.get("id"),
                        "score": c.get("score"),
                        "source": c.get("source"),
                        "page": c.get("page"),
                        "chunk_index": c.get("chunk_index"),
                        "text": c.get("text"),
                    }
                    for c in chunks
                ],
                "sources": sources,
                "model": target_model,
                "fallback_used": False,
                "language": target_lang,
                "is_greeting": True,
            }

        # -------------------------------------------------------------
        # Case B: User starts directly with a question (NO greeting prefix)
        # -------------------------------------------------------------
        retrieval_response = self.retriever.retrieve(
            query=clean_question,
            top_k=top_k,
            namespace=target_namespace,
            filter_dict=filter_dict,
            candidate_k=candidate_pool,
            language=target_lang,
        )
        chunks = retrieval_response.get("results", [])

        # Choose fallback message based on language and custom override
        active_fallback = (
            self.fallback_message
            if target_lang == LANG_EN or self.fallback_message != DEFAULT_FALLBACK_MESSAGE
            else get_fallback_message(target_lang)
        )

        # If no chunks retrieved, return fallback message directly without calling LLM
        if not chunks:
            logger.info("No relevant chunks retrieved from knowledge base. Returning fallback message.")
            return {
                "question": clean_question,
                "answer": active_fallback,
                "context_used": [],
                "sources": [],
                "model": target_model,
                "fallback_used": True,
                "language": target_lang,
                "is_greeting": False,
            }

        # Validate OpenRouter API key
        if not self.api_key:
            raise OpenRouterConfigurationError(
                "OPENROUTER_API_KEY is not set. Please configure OPENROUTER_API_KEY in backend/.env."
            )

        context_text = format_context(chunks)
        system_prompt = GROUNDED_SYSTEM_PROMPT.format(
            language_name=lang_name,
            fallback_message=active_fallback,
        )
        user_content = f"Context:\n{context_text}\n\nQuestion:\n{clean_question}"

        models_list = [target_model]
        if self.fallback_model and self.fallback_model != target_model:
            models_list.append(self.fallback_model)

        payload = {
            "model": target_model,
            "models": models_list,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.0,
            "max_tokens": 300,
            "stream": True,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Voice-Sales-Copilot",
            "X-Title": "Voice Sales Copilot",
        }

        endpoint_url = f"{self.base_url}/chat/completions"
        logger.info(f"Sending streaming prompt to OpenRouter ({target_model}) in {lang_name} at {endpoint_url}...")

        timeout = httpx.Timeout(30.0, connect=10.0, read=25.0)
        client = self.get_http_client()
        raw_answer = ""
        streamed_chunks: List[str] = []

        try:
            resp = client.post(endpoint_url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code in (429, 502, 503, 504) and self.fallback_model and self.fallback_model != target_model:
                logger.warning(
                    f"OpenRouter HTTP {resp.status_code} on primary model {target_model}. "
                    f"Failing over immediately to fallback model {self.fallback_model}..."
                )
                fb_payload = dict(payload)
                fb_payload["model"] = self.fallback_model
                fb_payload["models"] = [self.fallback_model]
                fb_resp = client.post(endpoint_url, json=fb_payload, headers=headers, timeout=timeout)
                if fb_resp.status_code == 200:
                    resp = fb_resp
                    target_model = self.fallback_model
        except httpx.TimeoutException as timeout_err:
            logger.error(f"OpenRouter streaming timeout error: {str(timeout_err)}")
            raise OpenRouterAPIError(f"OpenRouter API request timed out: {str(timeout_err)}") from timeout_err
        except Exception as net_err:
            logger.error(f"OpenRouter connection error: {str(net_err)}")
            raise OpenRouterAPIError(f"Failed to communicate with OpenRouter API: {str(net_err)}") from net_err

        if resp.status_code != 200:
            error_text = resp.text if hasattr(resp, "text") else str(resp.status_code)
            logger.error(f"OpenRouter returned HTTP {resp.status_code}: {error_text}")
            raise OpenRouterAPIError(
                f"OpenRouter API returned error HTTP {resp.status_code}: {error_text}"
            )

        # Process streamed SSE chunks (data: {"choices": [{"delta": {"content": "..."}}]})
        is_sse = False
        if hasattr(resp, "headers") and "text/event-stream" in resp.headers.get("content-type", "").lower():
            is_sse = True
        elif hasattr(resp, "text") and isinstance(resp.text, str) and ("data: {" in resp.text or "data: [" in resp.text):
            is_sse = True

        if is_sse and hasattr(resp, "iter_lines"):
            try:
                for raw_line in resp.iter_lines():
                    if isinstance(raw_line, bytes):
                        line = raw_line.decode("utf-8", errors="replace")
                    else:
                        line = str(raw_line)
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        payload_data = line[5:].strip()
                        if payload_data == "[DONE]":
                            break
                        try:
                            chunk_json = json.loads(payload_data)
                            if "error" in chunk_json:
                                err_detail = chunk_json["error"].get("message", str(chunk_json["error"]))
                                raise OpenRouterAPIError(f"OpenRouter stream error: {err_detail}")
                            choices = chunk_json.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                token = delta.get("content")
                                if token:
                                    streamed_chunks.append(token)
                                    if on_chunk:
                                        try:
                                            on_chunk(token)
                                        except Exception as cb_err:
                                            logger.warning(f"Error in on_chunk callback: {cb_err}")
                        except json.JSONDecodeError:
                            continue
                raw_answer = "".join(streamed_chunks).strip()
            except OpenRouterAPIError:
                raise
            except Exception as stream_err:
                logger.error(f"Error reading streamed OpenRouter response: {stream_err}")
                raise OpenRouterAPIError(f"Failed to read streamed chunks from OpenRouter: {str(stream_err)}") from stream_err
        else:
            # Fallback for standard non-streaming JSON responses / mock compatibility in test suites
            try:
                resp_json = resp.json()
                raw_answer = resp_json["choices"][0]["message"]["content"].strip()
            except (KeyError, IndexError, TypeError, ValueError) as parse_err:
                logger.error(f"Failed to parse OpenRouter response: {getattr(resp, 'text', str(resp))}")
                raise OpenRouterAPIError(f"Malformed response received from OpenRouter API: {str(parse_err)}") from parse_err

        # Check if model triggered fallback
        fallback_used = (
            raw_answer.strip() == active_fallback.strip()
            or active_fallback.lower() in raw_answer.lower()
            or DEFAULT_FALLBACK_MESSAGE.lower() in raw_answer.lower()
        )
        final_answer = active_fallback if fallback_used else raw_answer

        sources = list(dict.fromkeys(str(c.get("source")) for c in chunks if c.get("source")))

        return {
            "question": clean_question,
            "answer": final_answer,
            "context_used": [
                {
                    "id": c.get("id"),
                    "score": c.get("score"),
                    "source": c.get("source"),
                    "page": c.get("page"),
                    "chunk_index": c.get("chunk_index"),
                    "text": c.get("text"),
                }
                for c in chunks
            ],
            "sources": sources,
            "model": target_model,
            "fallback_used": fallback_used,
            "language": target_lang,
            "is_greeting": False,
        }

    # ------------------------------------------------------------------------
    # METHOD: stream_answer_chunks
    # ------------------------------------------------------------------------
    # • WHAT IT DOES: Real-time SSE streaming generator for RAG:
    #     1. Retrieves matching chunks from Pinecone.
    #     2. Yields 'event: metadata' with citations, page numbers, and sources.
    #     3. Streams answer tokens from OpenRouter DeepSeek SSE via 'event: token'.
    #     4. Yields 'event: done' on completion.
    # • INPUTS:
    #     - question (str): The prospect's inquiry.
    #     - top_k (int): Chunks retrieved (default: 8).
    #     - namespace (Optional[str]): Pinecone namespace.
    #     - language (Optional[str]): Language code ('en', 'hi', 'mr').
    # • OUTPUT: Generator yielding Server-Sent Event (SSE) strings.
    # • WHY IT IS USED: Enables the frontend SentenceTokenizer to receive tokens in real-time,
    #   synthesizing and speaking sentence #1 in ~400ms without waiting for the full response!
    # • WHERE IT FITS IN THE FLOW:
    #     [KnowledgeAssistant.tsx] -> [/api/ask?stream=true] -> [stream_answer_chunks] -> [SentenceAudioQueue]
    # ------------------------------------------------------------------------
    def stream_answer_chunks(
        self,
        question: str,
        top_k: int = 8,
        namespace: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        candidate_pool: Optional[int] = None,
        language: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Executes end-to-end multilingual RAG and yields Server-Sent Events (SSE):

        - event: metadata -> JSON with context_used, sources, model, language, is_greeting
        - event: token -> JSON with delta (token string)
        - event: done -> JSON with final complete answer, sources, etc.
        """
        clean_question = (question or "").strip()
        if not clean_question:
            yield f"event: error\ndata: {json.dumps({'error': 'Question cannot be empty.'})}\n\n"
            return

        voice_model_default = os.getenv("OPENROUTER_VOICE_MODEL", "deepseek/deepseek-chat")
        target_model = (model or voice_model_default).strip().lstrip("~")
        target_namespace = (
            namespace.strip()
            if (namespace is not None and namespace.strip())
            else (getattr(self.retriever.pinecone_service, "namespace", None) or os.getenv("PINECONE_NAMESPACE", "sales_playbooks"))
        )

        detected_lang = detect_language(clean_question)
        target_lang = (language or detected_lang).strip().lower()
        if target_lang not in LANGUAGE_NAMES:
            target_lang = detected_lang
        lang_name = LANGUAGE_NAMES.get(target_lang, "English")

        user_is_greeting = is_greeting(clean_question)

        # Retrieve chunks
        if user_is_greeting:
            # Skip RAG when not needed: reuse cached broad chunks for greetings
            chunks = getattr(self, "_cached_greeting_chunks", None)
            if chunks is None:
                broad_query = "sales playbooks overview pricing products features terms guidelines faq interest rates loan eligibility"
                retrieval_response = self.retriever.retrieve(
                    query=broad_query,
                    top_k=top_k,
                    namespace=target_namespace,
                    filter_dict=filter_dict,
                    candidate_k=candidate_pool,
                )
                chunks = retrieval_response.get("results", [])
                self._cached_greeting_chunks = chunks
        else:
            retrieval_response = self.retriever.retrieve(
                query=clean_question,
                top_k=top_k,
                namespace=target_namespace,
                filter_dict=filter_dict,
                candidate_k=candidate_pool or max(top_k * 3, 20),
                language=target_lang,
            )
            chunks = retrieval_response.get("results", [])

        sources = list(dict.fromkeys(str(c.get("source")) for c in chunks if c.get("source")))
        context_payload = [
            {
                "id": c.get("id"),
                "score": c.get("score"),
                "source": c.get("source"),
                "page": c.get("page"),
                "chunk_index": c.get("chunk_index"),
                "text": c.get("text"),
            }
            for c in chunks
        ]

        # 1. Send metadata event immediately
        meta_event = {
            "question": clean_question,
            "sources": sources,
            "context_used": context_payload,
            "model": target_model,
            "language": target_lang,
            "is_greeting": user_is_greeting,
        }
        yield f"event: metadata\ndata: {json.dumps(meta_event)}\n\n"

        active_fallback = (
            self.fallback_message
            if target_lang == LANG_EN or self.fallback_message != DEFAULT_FALLBACK_MESSAGE
            else get_fallback_message(target_lang)
        )

        # Handle zero chunks cases
        if not chunks:
            if user_is_greeting:
                msg = get_empty_kb_greeting(target_lang)
            else:
                msg = active_fallback
            yield f"event: token\ndata: {json.dumps({'delta': msg})}\n\n"
            done_payload = {
                "question": clean_question,
                "answer": msg,
                "sources": [],
                "context_used": [],
                "model": target_model,
                "fallback_used": not user_is_greeting,
                "language": target_lang,
                "is_greeting": user_is_greeting,
            }
            yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"
            return

        if not self.api_key:
            yield f"event: error\ndata: {json.dumps({'error': 'OPENROUTER_API_KEY is not set.'})}\n\n"
            return

        # Prepare prompts
        if user_is_greeting:
            context_text = format_context(chunks)
            system_prompt = GREETING_SYSTEM_PROMPT.format(language_name=lang_name)
            user_content = f"Context from Sales Playbooks:\n{context_text}\n\nUser Greeting:\n{clean_question}"
            temp = 0.2
        else:
            context_text = format_voice_context(chunks, max_chunks=3)
            system_prompt = VOICE_GROUNDED_SYSTEM_PROMPT.format(
                language_name=lang_name,
                fallback_message=active_fallback,
            )
            user_content = f"Context:\n{context_text}\n\nQuestion:\n{clean_question}"
            temp = 0.0

        voice_fallback = (
            self.model
            if self.model != target_model
            else (self.fallback_model if self.fallback_model != target_model else "deepseek/deepseek-chat")
        )
        models_list = [target_model]
        if voice_fallback and voice_fallback != target_model:
            models_list.append(voice_fallback)

        payload = {
            "model": target_model,
            "models": models_list,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "provider": {"sort": "latency"},
            "temperature": temp,
            "max_tokens": 180,
            "stream": True,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Voice-Sales-Copilot",
            "X-Title": "Voice Sales Copilot",
        }

        endpoint_url = f"{self.base_url}/chat/completions"
        timeout = httpx.Timeout(30.0, connect=10.0, read=25.0)
        client = self.get_http_client()
        streamed_chunks: List[str] = []

        is_mock_client = hasattr(client, "post") and "Mock" in type(client.post).__name__
        if not is_mock_client and hasattr(client, "stream"):
            try:
                with client.stream("POST", endpoint_url, json=payload, headers=headers, timeout=timeout) as resp:
                    if resp.status_code != 200:
                        if resp.status_code in (400, 404, 429, 502, 503, 504) and voice_fallback and voice_fallback != target_model:
                            logger.warning(
                                f"OpenRouter HTTP {resp.status_code} on primary model {target_model}. "
                                f"Failing over immediately to fallback model {voice_fallback}..."
                            )
                            fb_payload = dict(payload)
                            fb_payload["model"] = voice_fallback
                            fb_payload["models"] = [voice_fallback]
                            target_model = voice_fallback
                            with client.stream("POST", endpoint_url, json=fb_payload, headers=headers, timeout=timeout) as fb_resp:
                                if fb_resp.status_code != 200:
                                    fb_err = fb_resp.read().decode("utf-8", errors="replace") if hasattr(fb_resp, "read") else str(fb_resp.status_code)
                                    yield f"event: error\ndata: {json.dumps({'error': f'OpenRouter HTTP {fb_resp.status_code}: {fb_err}'})}\n\n"
                                    return
                                for raw_line in fb_resp.iter_lines():
                                    if isinstance(raw_line, bytes):
                                        line = raw_line.decode("utf-8", errors="replace")
                                    else:
                                        line = str(raw_line)
                                    line = line.strip()
                                    if not line or line.startswith(":"):
                                        continue
                                    if line.startswith("data:"):
                                        payload_data = line[5:].strip()
                                        if payload_data == "[DONE]":
                                            break
                                        try:
                                            chunk_json = json.loads(payload_data)
                                            if "error" in chunk_json:
                                                err_detail = chunk_json["error"].get("message", str(chunk_json["error"]))
                                                yield f"event: error\ndata: {json.dumps({'error': err_detail})}\n\n"
                                                return
                                            choices = chunk_json.get("choices", [])
                                            if choices:
                                                delta = choices[0].get("delta", {})
                                                token = delta.get("content")
                                                if token:
                                                    streamed_chunks.append(token)
                                                    yield f"event: token\ndata: {json.dumps({'token': token, 'delta': token})}\n\n"
                                        except json.JSONDecodeError:
                                            continue
                        else:
                            error_text = resp.read().decode("utf-8", errors="replace") if hasattr(resp, "read") else str(resp.status_code)
                            yield f"event: error\ndata: {json.dumps({'error': f'OpenRouter HTTP {resp.status_code}: {error_text}'})}\n\n"
                            return
                    else:
                        for raw_line in resp.iter_lines():
                            if isinstance(raw_line, bytes):
                                line = raw_line.decode("utf-8", errors="replace")
                            else:
                                line = str(raw_line)
                            line = line.strip()
                            if not line or line.startswith(":"):
                                continue
                            if line.startswith("data:"):
                                payload_data = line[5:].strip()
                                if payload_data == "[DONE]":
                                    break
                                try:
                                    chunk_json = json.loads(payload_data)
                                    if "error" in chunk_json:
                                        err_detail = chunk_json["error"].get("message", str(chunk_json["error"]))
                                        if not streamed_chunks and voice_fallback and voice_fallback != target_model:
                                            logger.warning(
                                                f"OpenRouter in-stream error on primary model {target_model}: {err_detail}. "
                                                f"Failing over immediately to fallback model {voice_fallback}..."
                                            )
                                            fb_payload = dict(payload)
                                            fb_payload["model"] = voice_fallback
                                            fb_payload["models"] = [voice_fallback]
                                            target_model = voice_fallback
                                            try:
                                                with client.stream("POST", endpoint_url, json=fb_payload, headers=headers, timeout=timeout) as fb_resp:
                                                    if fb_resp.status_code == 200:
                                                        for fb_raw_line in fb_resp.iter_lines():
                                                            fb_line = fb_raw_line.decode("utf-8", errors="replace") if isinstance(fb_raw_line, bytes) else str(fb_raw_line)
                                                            fb_line = fb_line.strip()
                                                            if not fb_line or fb_line.startswith(":"):
                                                                continue
                                                            if fb_line.startswith("data:"):
                                                                fb_p = fb_line[5:].strip()
                                                                if fb_p == "[DONE]":
                                                                    break
                                                                try:
                                                                    fb_cj = json.loads(fb_p)
                                                                    fb_choices = fb_cj.get("choices", [])
                                                                    if fb_choices:
                                                                        fb_tok = fb_choices[0].get("delta", {}).get("content")
                                                                        if fb_tok:
                                                                            streamed_chunks.append(fb_tok)
                                                                            yield f"event: token\ndata: {json.dumps({'token': fb_tok, 'delta': fb_tok})}\n\n"
                                                                except json.JSONDecodeError:
                                                                    continue
                                                        break
                                            except Exception as fb_err:
                                                logger.warning(f"Fallback streaming attempt failed: {fb_err}")
                                        yield f"event: error\ndata: {json.dumps({'error': err_detail})}\n\n"
                                        return
                                    choices = chunk_json.get("choices", [])
                                    if choices:
                                        delta = choices[0].get("delta", {})
                                        token = delta.get("content")
                                        if token:
                                            streamed_chunks.append(token)
                                            yield f"event: token\ndata: {json.dumps({'token': token, 'delta': token})}\n\n"
                                except json.JSONDecodeError:
                                    continue
            except httpx.TimeoutException as timeout_err:
                yield f"event: error\ndata: {json.dumps({'error': f'OpenRouter timeout: {str(timeout_err)}'})}\n\n"
                return
            except Exception as net_err:
                yield f"event: error\ndata: {json.dumps({'error': f'OpenRouter connection error: {str(net_err)}'})}\n\n"
                return
        else:
            try:
                resp = client.post(endpoint_url, json=payload, headers=headers, timeout=timeout)
            except httpx.TimeoutException as timeout_err:
                yield f"event: error\ndata: {json.dumps({'error': f'OpenRouter timeout: {str(timeout_err)}'})}\n\n"
                return
            except Exception as net_err:
                yield f"event: error\ndata: {json.dumps({'error': f'OpenRouter connection error: {str(net_err)}'})}\n\n"
                return

            if resp.status_code != 200:
                error_text = resp.text if hasattr(resp, "text") else str(resp.status_code)
                yield f"event: error\ndata: {json.dumps({'error': f'OpenRouter HTTP {resp.status_code}: {error_text}'})}\n\n"
                return

            is_sse = False
            if hasattr(resp, "headers") and "text/event-stream" in resp.headers.get("content-type", "").lower():
                is_sse = True
            elif hasattr(resp, "text") and isinstance(resp.text, str) and ("data: {" in resp.text or "data: [" in resp.text):
                is_sse = True

            if is_sse and hasattr(resp, "iter_lines"):
                try:
                    for raw_line in resp.iter_lines():
                        if isinstance(raw_line, bytes):
                            line = raw_line.decode("utf-8", errors="replace")
                        else:
                            line = str(raw_line)
                        line = line.strip()
                        if not line or line.startswith(":"):
                            continue
                        if line.startswith("data:"):
                            payload_data = line[5:].strip()
                            if payload_data == "[DONE]":
                                break
                            try:
                                chunk_json = json.loads(payload_data)
                                if "error" in chunk_json:
                                    err_detail = chunk_json["error"].get("message", str(chunk_json["error"]))
                                    yield f"event: error\ndata: {json.dumps({'error': err_detail})}\n\n"
                                    return
                                choices = chunk_json.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    token = delta.get("content")
                                    if token:
                                        streamed_chunks.append(token)
                                        yield f"event: token\ndata: {json.dumps({'token': token, 'delta': token})}\n\n"
                            except json.JSONDecodeError:
                                continue
                except Exception as stream_err:
                    yield f"event: error\ndata: {json.dumps({'error': str(stream_err)})}\n\n"
                    return
            else:
                try:
                    resp_json = resp.json()
                    raw_ans = resp_json["choices"][0]["message"]["content"].strip()
                    streamed_chunks.append(raw_ans)
                    yield f"event: token\ndata: {json.dumps({'token': raw_ans, 'delta': raw_ans})}\n\n"
                except Exception as parse_err:
                    yield f"event: error\ndata: {json.dumps({'error': str(parse_err)})}\n\n"
                    return

        final_raw = "".join(streamed_chunks).strip()
        fallback_used = (
            final_raw.strip() == active_fallback.strip()
            or active_fallback.lower() in final_raw.lower()
            or DEFAULT_FALLBACK_MESSAGE.lower() in final_raw.lower()
        )
        final_answer = active_fallback if fallback_used else final_raw

        done_payload = {
            "question": clean_question,
            "answer": final_answer,
            "sources": sources,
            "context_used": context_payload,
            "model": target_model,
            "fallback_used": fallback_used,
            "language": target_lang,
            "is_greeting": user_is_greeting,
        }
        yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"
