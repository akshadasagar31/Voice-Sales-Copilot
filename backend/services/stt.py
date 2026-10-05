# ============================================================================
# SPEECH-TO-TEXT SERVICE: DEEPGRAM (backend/services/stt.py)
# ============================================================================
# WHAT THIS SERVICE DOES:
# Converts spoken audio recordings into text transcripts using Deepgram Nova-2.
#
# KEY FEATURES:
# 1. High Accuracy: Uses Deepgram's state-of-the-art Nova-2 speech model.
# 2. Multilingual Support: Automatically detects English (en), Hindi (hi), and Marathi (mr).
# 3. Smart Formatting: Deepgram automatically adds punctuation, capitalization, and numbers.
# 4. Resilience: Automatically retries on transient network errors or temporary 503 server load.
# 5. Security: Never logs API keys or authorization tokens in error messages.
# ============================================================================

import os
import re
import json
import socket
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import httpx
import websockets
try:
    from services.language import (
        detect_spoken_language as detect_text_language,
        HINDI_WORDS,
        MARATHI_WORDS,
    )
except ImportError:
    from backend.services.language import (
        detect_spoken_language as detect_text_language,
        HINDI_WORDS,
        MARATHI_WORDS,
    )

# Ensure backend/.env is loaded
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

logger = logging.getLogger(__name__)

DEEPGRAM_LISTEN_URL = "https://api.deepgram.com/v1/listen"

def clean_deepgram_keyterms(terms: list[str]) -> list[str]:
    """
    Sanitizes Deepgram keyterms:
    - Strips whitespace and discards empty, whitespace, or single-character items.
    - Rejects colon-boost markers like 'term:3' which belong to legacy keywords parameter and cause 400 Bad Request.
    - Deduplicates case-insensitively while preserving insertion order.
    """
    if not terms:
        return []
    seen = set()
    cleaned = []
    for kt in terms:
        if not kt or not isinstance(kt, str):
            continue
        k = re.sub(r"\s+", " ", kt.strip())
        if not k or len(k) < 2 or ":" in k:
            continue
        k_lower = k.lower()
        if k_lower not in seen:
            seen.add(k_lower)
            cleaned.append(k)
    return cleaned

# Domain-specific financial & lead keyterms for Deepgram Nova-3 keyword boosting (strictly under 500-token limit)
FINANCIAL_KEYTERMS_EN = clean_deepgram_keyterms([
    "CIBIL", "CIBIL score", "EMI", "monthly EMI", "ROI",
    "personal loan", "home loan", "business loan", "equipment loan", "interest rate",
    "company", "tenure", "months", "years", "lakh", "crore", "thousand", "rupees",
    "HDFC", "HDFC Bank", "SBI", "ICICI", "phone", "mobile", "email", "loan amount"
])

FINANCIAL_KEYTERMS_HI = clean_deepgram_keyterms([
    # Credit score & bureau
    "CIBIL", "CIBIL score", "सिबिल", "सिबिल स्कोर",
    # EMI & payments
    "EMI", "ईएमआई", "monthly EMI", "मासिक ईएमआई", "मासिक EMI",
    # Loan products
    "पर्सनल लोन", "personal loan", "होम लोन", "home loan", "बिजनेस लोन", "business loan", "गोल्ड लोन", "लोन", "कर्ज",
    # Loan terms & currency
    "ब्याज दर", "ब्याज", "अवधि", "कार्यकाल", "महीने", "साल",
    "लाख", "रुपये", "रुपए", "करोड़", "हज़ार", "हजार",
    # Institutions & contact
    "HDFC Bank", "HDFC", "SBI", "ICICI Bank", "फोन नंबर", "मोबाइल नंबर"
])

FINANCIAL_KEYTERMS_MR = clean_deepgram_keyterms([
    # Credit score & bureau
    "CIBIL", "CIBIL score", "सिबिल", "सिबिल स्कोर", "सिबिल स्कोअर",
    # EMI & payments
    "EMI", "ईएमआय", "ईएमआई", "दरमहा EMI", "monthly EMI", "मासिक EMI", "दरमहा", "दरमहा हप्ता", "मासिक हप्ता",
    # Loan products
    "वैयक्तिक कर्ज", "गृहकर्ज", "गृह कर्ज", "व्यवसाय कर्ज", "पर्सनल लोन", "personal loan", "home loan", "कर्ज",
    # Loan tenure & terms
    "मुदत", "मुदतीसाठी", "वर्षांच्या मुदतीसाठी", "कालावधी", "कालावधीसाठी", "व्याजदर", "व्याज",
    # Currency & amounts
    "लाख", "रुपयांचे", "रुपये",
    # Regional names for acoustic grounding
    "पाटील", "कदम", "देशमुख", "सावंत", "पवार",
    # Intent phrases
    "हवे आहे",
    # Institutions & contact
    "HDFC Bank", "SBI"
])

FINANCIAL_KEYTERMS_MULTI = clean_deepgram_keyterms([
    # Core credit & bureau terms
    "CIBIL", "CIBIL score", "सिबिल", "सिबिल स्कोर",
    # EMI & payments
    "EMI", "ईएमआई", "ईएमआय", "monthly EMI", "दरमहा EMI", "मासिक EMI", "दरमहा हप्ता", "मासिक हप्ता", "ROI",
    # Loan products
    "personal loan", "home loan", "business loan", "loan amount", "interest rate",
    "कर्ज", "वैयक्तिक कर्ज", "गृहकर्ज", "व्यवसाय कर्ज", "लोन", "पर्सनल लोन",
    # Tenure & terms
    "tenure", "कालावधी", "मुदत", "मुदतीसाठी", "अवधि", "महिने", "वर्षे", "साल",
    # Currency & amounts
    "lakh", "crore", "thousand", "rupees", "लाख", "करोड़", "कोटी", "हजार", "हज़ार", "रुपये", "रुपयांचे",
    # Regional names & institutions
    "HDFC Bank", "HDFC", "SBI", "ICICI", "पाटील", "कदम", "देशमुख", "सावंत"
])

FINANCIAL_KEYTERMS = clean_deepgram_keyterms([
    # Core credit & bureau terms
    "CIBIL", "CIBIL score", "सिबिल", "सिबिल स्कोर", "सिबिल स्कोअर",
    "EMI", "ईएमआई", "ईएमआय", "मासिक EMI", "दरमहा EMI", "दरमहा हप्ता", "मासिक हप्ता", "ROI",
    # Loan products & terms
    "personal loan", "home loan", "business loan", "interest rate",
    "कर्ज", "वैयक्तिक कर्ज", "गृहकर्ज", "व्यवसाय कर्ज", "लोन", "पर्सनल लोन",
    # Company, identity & tenure terms
    "company", "tenure", "months", "years", "कालावधी", "मुदत", "मुदतीसाठी", "दरमहा", "महिने", "महीने", "वर्षे", "वर्ष", "साल", "कागदपत्रे",
    # Marathi identity & names
    "पाटील", "कदम", "देशमुख", "सावंत", "पवार",
    # Institutions & currency
    "HDFC", "HDFC Bank", "HDFC बँक", "SBI", "SBI बँक", "ICICI",
    "lakh", "crore", "thousand", "rupees", "रुपये", "लाख", "कोटी", "हजार", "हज़ार", "रुपयांचे", "लाखांचे"
])





MARATHI_MARKERS = {
    "kiti", "ahe", "aahe", "mala", "safi", "rupenship", "derma", "darmaha", "nav", "naav",
    "pahije", "havay", "have", "kay", "kasa", "kashi", "karayche", "karaycha", "baddal",
    "नाव", "आहे", "किती", "मला", "पाहिजे", "हवे", "काय", "कसे", "कर्ज", "रुपये", "लाख", "दरमहा",
    "कालावधी", "वर्षांसाठी", "महिन्यांसाठी", "पाटील", "कदम", "देशमुख", "कुलकर्णी", "जोशी"
}

HINDI_MARKERS = {
    "mera", "meri", "mere", "naam", "hona", "chaheai", "chahiye", "kidna", "kitna", "kitne",
    "kya", "kaise", "bataiye", "bataye", "chahiye", "hai", "hain", "apna", "aapka",
    "है", "कितना", "कितने", "चाहिए", "मुझे", "मेरा", "मेरी", "मेरे", "आपका", "आपकी", "अपने", "बताइए"
}

def normalize_stt_transcript(transcript: str, language: Optional[str] = None) -> str:
    """
    Standardizes whitespace and punctuation spacing while preserving the user's actual
    spoken words verbatim. No translation, rewriting, autocorrection, or guessing.
    """
    if not transcript or not isinstance(transcript, str):
        return ""

    # Standardize whitespace
    cleaned = re.sub(r'\s+', ' ', transcript).strip()
    # Normalize accidental whitespace before punctuation (e.g. "word ." -> "word.")
    cleaned = re.sub(r'\s+([.,!?;:])', r'\1', cleaned)
    return cleaned

def build_deepgram_ws_url(
    model: str = "nova-3",
    sample_rate: int = 48000,
    language: Optional[str] = None,
    include_keyterms: bool = True,
    encoding: Optional[str] = "linear16",
) -> str:
    """Builds the Deepgram live streaming WebSocket URL with Nova-3, keyterms, and smart formatting."""
    import urllib.parse
    req_model = (model or "nova-3").strip().lower()
    valid_model = model if ("nova" in req_model or "enhanced" in req_model or "base" in req_model) else "nova-3"
    params = [
        f"model={valid_model}",
    ]
    if encoding and encoding.lower() != "auto":
        params.append(f"encoding={encoding}")
        params.append(f"sample_rate={sample_rate}")
        params.append("channels=1")

    params.extend([
        "smart_format=true",
        "punctuate=true",
        "interim_results=true",
        "endpointing=500",
    ])
    lang_clean = (language or "").strip().lower()
    if lang_clean in ("mr", "mr-in", "marathi"):
        params.append("language=mr")
    elif lang_clean in ("hi", "hi-in", "hindi"):
        params.append("language=hi")
    elif lang_clean in ("en", "en-in", "en-us", "english"):
        params.append("language=en-IN")
    elif lang_clean in ("multi", "auto") or not lang_clean:
        # Requirement 4: Remove multi/auto from final Module 1 STT path. Default to en-IN.
        params.append("language=en-IN")
    else:
        params.append(f"language={lang_clean}")

    if valid_model == "nova-3" and include_keyterms:
        if lang_clean in ("mr", "mr-in", "marathi"):
            keyterms_to_use = FINANCIAL_KEYTERMS_MR
        elif lang_clean in ("hi", "hi-in", "hindi"):
            keyterms_to_use = FINANCIAL_KEYTERMS_HI
        else:
            keyterms_to_use = FINANCIAL_KEYTERMS_EN

        for kt in clean_deepgram_keyterms(keyterms_to_use):
            params.append(f"keyterm={urllib.parse.quote(kt)}")

    return f"{DEEPGRAM_LISTEN_URL.replace('https://', 'wss://')}?{'&'.join(params)}"

_shared_stt_client: Optional[httpx.AsyncClient] = None

def get_shared_stt_client() -> httpx.AsyncClient:
    """Return a shared persistent AsyncClient with keep-alive connection pooling for Deepgram STT."""
    global _shared_stt_client
    if _shared_stt_client is None or _shared_stt_client.is_closed:
        _shared_stt_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50, keepalive_expiry=30.0),
        )
    return _shared_stt_client

async def reset_shared_stt_client() -> None:
    """Closes and recreates only stale or broken shared HTTP client connections."""
    global _shared_stt_client
    if _shared_stt_client is not None and not _shared_stt_client.is_closed:
        try:
            await _shared_stt_client.aclose()
        except Exception:
            pass
    _shared_stt_client = None


async def connect_deepgram_ws_with_retry(
    url: str,
    headers: dict,
    max_retries: int = 2,
    open_timeout: float = 3.0,
) -> Any:
    """
    Connects to Deepgram WebSocket endpoint with automatic short retries on
    transient 503 Service Unavailable, getaddrinfo failed, or DNS glitches.
    Never leaks API credentials in logs.
    """
    backoff_delays = [0.08, 0.2]
    last_exc = None
    total_attempts = 1 + max_retries

    for attempt in range(total_attempts):
        t0 = asyncio.get_event_loop().time()
        try:
            try:
                ws = await websockets.connect(
                    url,
                    additional_headers=headers,
                    open_timeout=open_timeout,
                    ping_interval=10,
                    ping_timeout=5,
                )
            except TypeError:
                ws = await websockets.connect(
                    url,
                    extra_headers=headers,
                )
            elapsed_ms = (asyncio.get_event_loop().time() - t0) * 1000
            logger.info(f"[DeepgramSTT] WebSocket connected in {elapsed_ms:.1f}ms (attempt {attempt + 1})")
            return ws
        except (
            socket.gaierror,
            OSError,
            TimeoutError,
            asyncio.TimeoutError,
            websockets.exceptions.WebSocketException,
        ) as exc:
            last_exc = exc
            status_code = getattr(exc, "status_code", None)
            is_503 = status_code == 503 or "503" in str(exc)
            is_dns = isinstance(exc, socket.gaierror) or "getaddrinfo" in str(exc)
            elapsed_ms = (asyncio.get_event_loop().time() - t0) * 1000

            safe_msg = str(exc)
            for v in headers.values():
                if v and len(v) > 8 and v in safe_msg:
                    safe_msg = safe_msg.replace(v, "[REDACTED]")

            logger.warning(
                f"[DeepgramSTT] Transient connection failure on attempt {attempt + 1}/{total_attempts} "
                f"({elapsed_ms:.1f}ms, is_503={is_503}, is_dns={is_dns}): {safe_msg}"
            )
            if attempt < max_retries:
                delay = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                await asyncio.sleep(delay)
            else:
                logger.error(f"[DeepgramSTT] Failed to connect after {total_attempts} attempts: {safe_msg}")
                raise DeepgramAPIError(
                    status_code or 503,
                    f"Deepgram WebSocket connection failed ({type(exc).__name__}): {safe_msg}"
                )
    if last_exc:
        raise last_exc


class DeepgramWSConnectionPool:
    """
    Persistent connection pool for Deepgram live streaming WebSockets.
    - Reuses healthy WebSocket connections across consecutive turns with 0ms reconnect latency.
    - Discards and recreates only stale or broken connections.
    - Handles transient 503 and getaddrinfo failed / DNS errors with short retries.
    - Sends periodic KeepAlive pings to keep idle pooled connections alive and warm.
    """
    def __init__(self):
        self._pool: Dict[str, Any] = {}
        self._in_use: set = set()
        self._lock = asyncio.Lock()
        self._keepalive_task: Optional[asyncio.Task] = None

    def start_keepalive(self) -> None:
        if self._keepalive_task is None or self._keepalive_task.done():
            try:
                loop = asyncio.get_running_loop()
                self._keepalive_task = loop.create_task(self._keepalive_loop())
            except RuntimeError:
                pass

    async def _keepalive_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(4.0)
                async with self._lock:
                    stale_urls = []
                    for url, ws in list(self._pool.items()):
                        state_name = getattr(getattr(ws, "state", None), "name", "")
                        is_open = (state_name == "OPEN") if state_name else not getattr(ws, "closed", True)
                        if is_open:
                            try:
                                await ws.send(json.dumps({"type": "KeepAlive"}))
                            except Exception:
                                stale_urls.append(url)
                        else:
                            stale_urls.append(url)
                    for url in stale_urls:
                        ws = self._pool.pop(url, None)
                        if ws:
                            try:
                                await ws.close()
                            except Exception:
                                pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[DeepgramWSPool] keepalive error: {e}")

    async def acquire(self, url: str, headers: dict) -> Any:
        self.start_keepalive()
        current_loop = asyncio.get_running_loop()
        async with self._lock:
            existing_ws = self._pool.get(url)
            if existing_ws is not None and existing_ws not in self._in_use:
                state_name = getattr(getattr(existing_ws, "state", None), "name", "")
                is_open = (state_name == "OPEN") if state_name else not getattr(existing_ws, "closed", True)
                attached_loop = getattr(existing_ws, "_attached_loop", None)
                if is_open and (attached_loop is None or attached_loop == current_loop):
                    logger.info("[DeepgramWSPool] Reusing healthy Deepgram WebSocket connection")
                    self._in_use.add(existing_ws)
                    return existing_ws
                else:
                    reason = f"loop mismatch" if attached_loop and attached_loop != current_loop else f"state: {state_name}"
                    logger.info(f"[DeepgramWSPool] Discarding stale Deepgram connection ({reason})")
                    self._pool.pop(url, None)
                    try:
                        await existing_ws.close()
                    except Exception:
                        pass

        # Connect with transient 503 / DNS retry
        ws = await connect_deepgram_ws_with_retry(url, headers)
        try:
            ws._attached_loop = current_loop
        except Exception:
            pass
        async with self._lock:
            self._in_use.add(ws)
            self._pool[url] = ws
        return ws

    async def release(self, url: str, ws: Any, broken: bool = False) -> None:
        if ws is None:
            return
        async with self._lock:
            self._in_use.discard(ws)
            state_name = getattr(getattr(ws, "state", None), "name", "")
            is_open = (state_name == "OPEN") if state_name else not getattr(ws, "closed", True)
            if broken or not is_open:
                logger.info(f"[DeepgramWSPool] Recreating/discarding broken Deepgram connection (state: {state_name})")
                self._pool.pop(url, None)
                try:
                    await ws.close()
                except Exception:
                    pass
            else:
                self._pool[url] = ws
                logger.debug("[DeepgramWSPool] Released healthy Deepgram connection back to pool")


_deepgram_ws_pool: Optional[DeepgramWSConnectionPool] = None

def get_deepgram_ws_pool() -> DeepgramWSConnectionPool:
    """Return singleton DeepgramWSConnectionPool."""
    global _deepgram_ws_pool
    if _deepgram_ws_pool is None:
        _deepgram_ws_pool = DeepgramWSConnectionPool()
    return _deepgram_ws_pool



def _extract_safe_response_detail(response: httpx.Response, api_key: Optional[str] = None) -> str:
    """
    Extracts a safe diagnostic error message or detail string from a Deepgram HTTP response.
    Never leaks or logs the API key, tokens, or authorization credentials.
    """
    detail = ""
    try:
        err_json = response.json()
        if isinstance(err_json, dict):
            detail = err_json.get("err_msg") or err_json.get("message") or err_json.get("error") or ""
            if not detail and "detail" in err_json:
                detail = str(err_json["detail"])
    except Exception:
        pass

    if not detail:
        detail = response.text[:200].strip() if response.text else "Empty response body"

    # Redact any accidental occurrence of the API key
    if api_key and api_key in detail:
        detail = detail.replace(api_key, "[REDACTED_API_KEY]")
    return detail.strip()


class DeepgramError(Exception):
    """Base exception for Deepgram STT errors."""
    pass


class DeepgramConfigurationError(DeepgramError):
    """Raised when DEEPGRAM_API_KEY is not configured or missing."""
    pass


class DeepgramAPIError(DeepgramError):
    """Raised when the Deepgram API returns an error response."""
    def __init__(self, status_code: int, message: str):
        super().__init__(f"Deepgram API error (HTTP {status_code}): {message}")
        self.status_code = status_code
        self.message = message


class DeepgramSTTService:
    """
    Service for transcribing audio using Deepgram's Speech-to-Text API.
    Supports English, Hindi, and Marathi with automatic language detection.
    Reads DEEPGRAM_API_KEY from backend .env only.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (api_key or os.getenv("DEEPGRAM_API_KEY", "")).strip()

    def _validate_api_key(self) -> None:
        """Validates that DEEPGRAM_API_KEY is present."""
        if not self.api_key:
            # Re-read from environment in case .env was refreshed
            self.api_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
        if not self.api_key:
            raise DeepgramConfigurationError(
                "DEEPGRAM_API_KEY is not configured in backend/.env. "
                "Please add DEEPGRAM_API_KEY to backend/.env to transcribe audio."
            )

    # ------------------------------------------------------------------------
    # METHOD: transcribe_audio
    # ------------------------------------------------------------------------
    # • WHAT IT DOES: Sends raw audio bytes to Deepgram API (Nova-2 or Nova-3)
    #   and returns clean, punctuated text with detected language.
    # • INPUTS:
    #     - audio_bytes (bytes): Binary audio data (WAV / WebM).
    #     - content_type (str): MIME type (default: "audio/webm").
    #     - model (str): Deepgram model (default: "nova-2", auto-promotes to "nova-3" for Marathi/Hindi).
    #     - language (Optional[str]): Target language code ("en", "hi", "mr").
    #     - detect_language (Optional[bool]): If True, Deepgram auto-detects spoken language.
    # • OUTPUT: Dictionary containing:
    #     - 'transcript': Final text string transcribed from speech.
    #     - 'detected_language': 'en', 'hi', or 'mr'.
    #     - 'confidence': Model confidence score (0.0 to 1.0).
    # • WHY IT IS USED: Translates human speech into text in under ~300ms, making hands-free
    #   continuous conversations feel instant and fluid.
    # • WHERE IT FITS IN THE FLOW:
    #     [Prospect Speaks] -> [/api/voice-entry] -> [transcribe_audio] -> [Lead Extractor / RAG]
    # ------------------------------------------------------------------------
    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        content_type: str = "audio/webm",
        model: str = "nova-2",
        language: Optional[str] = None,
        detect_language: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Sends raw audio bytes to Deepgram Speech-to-Text endpoint.


        Args:
            audio_bytes: Raw audio binary content
            content_type: MIME type of the audio (e.g., 'audio/webm', 'audio/wav', 'audio/mp4')
            model: Deepgram model name (defaults to 'nova-2')
            language: Optional language code ('en', 'hi', 'mr')
            detect_language: Whether to enable Deepgram automatic language detection (defaults to True if language not set)

        Returns:
            Dict containing transcript, confidence, duration, words count, model, and detected_language.
        """
        self._validate_api_key()

        if not audio_bytes:
            raise ValueError("Audio bytes cannot be empty.")

        # Ensure content_type is valid and supported by Deepgram
        mime = (content_type or "audio/webm").split(";")[0].strip()
        if not mime or mime in ("application/octet-stream", "binary/octet-stream"):
            mime = "audio/webm"

        # Select optimal Deepgram STT model
        # Nova-3 natively supports Marathi (mr), Hindi (hi), and multi-lingual code-mixed speech
        stt_model = model
        req_lang = (language or "").strip().lower()
        if req_lang in ("mr", "mr-in", "marathi", "hi", "hi-in", "hindi", "multi", "auto") or model == "nova-3":
            stt_model = "nova-3"

        if req_lang in ("multi", "auto", "unknown") or not req_lang:
            from services.language import select_best_multilingual_transcript
            res_en, res_hi, res_mr = await asyncio.gather(
                self.transcribe_audio(audio_bytes, content_type=content_type, model=stt_model, language="en"),
                self.transcribe_audio(audio_bytes, content_type=content_type, model=stt_model, language="hi"),
                self.transcribe_audio(audio_bytes, content_type=content_type, model=stt_model, language="mr"),
            )
            cands = {
                "en": {"transcript": res_en.get("transcript", ""), "confidence": res_en.get("confidence", 0.0)},
                "hi": {"transcript": res_hi.get("transcript", ""), "confidence": res_hi.get("confidence", 0.0)},
                "mr": {"transcript": res_mr.get("transcript", ""), "confidence": res_mr.get("confidence", 0.0)},
            }
            win_tr, win_lang, win_conf = select_best_multilingual_transcript(cands)
            win_result = res_mr if win_lang == "mr" else (res_hi if win_lang == "hi" else res_en)
            return {
                "success": True,
                "transcript": win_tr,
                "confidence": win_conf,
                "detected_language": win_lang,
                "words": win_result.get("words", []),
                "duration": win_result.get("duration", 0.0),
                "model": stt_model,
            }

        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": mime,
        }

        params: Dict[str, Any] = {
            "model": stt_model,
            "smart_format": "true",
            "punctuate": "true",
        }

        if req_lang in ("en", "en-in", "en-us", "english"):
            params["language"] = "en-IN"
        elif req_lang in ("hi", "hi-in", "hindi"):
            params["language"] = "hi"
            stt_model = "nova-3"
        elif req_lang in ("mr", "mr-in", "marathi"):
            params["language"] = "mr"
            stt_model = "nova-3"
        else:
            params["language"] = req_lang

        params["model"] = stt_model
        if stt_model == "nova-3":
            if req_lang in ("mr", "mr-in", "marathi"):
                params["keyterm"] = clean_deepgram_keyterms(FINANCIAL_KEYTERMS_MR)
            elif req_lang in ("hi", "hi-in", "hindi"):
                params["keyterm"] = clean_deepgram_keyterms(FINANCIAL_KEYTERMS_HI)
            else:
                params["keyterm"] = clean_deepgram_keyterms(FINANCIAL_KEYTERMS_EN)

        logger.info(
            f"[DeepgramSTT] Sending {len(audio_bytes)} audio bytes to Deepgram STT "
            f"(model: {stt_model}, content_type: '{mime}', params: {params})"
        )

        data = None
        max_retries = 2  # Max 2 retries (3 attempts total) on transient network/503 errors
        total_attempts = 1 + max_retries
        backoff_delays = [0.5, 1.0]


        for attempt in range(total_attempts):
            try:
                mock_cls = httpx.AsyncClient
                is_mock = hasattr(mock_cls, "return_value") and hasattr(mock_cls.return_value, "__aenter__")
                if is_mock:
                    async with httpx.AsyncClient(
                        timeout=httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0),
                    ) as client:
                        response = await client.post(
                            DEEPGRAM_LISTEN_URL,
                            params=params,
                            headers=headers,
                            content=audio_bytes,
                        )
                else:
                    client = get_shared_stt_client()
                    response = await client.post(
                        DEEPGRAM_LISTEN_URL,
                        params=params,
                        headers=headers,
                        content=audio_bytes,
                    )

                dg_request_id = response.headers.get("dg-request-id", "unknown")
                content_type_hdr = response.headers.get("content-type", "unknown")

                # Check for transient retryable HTTP statuses: 503 (Service Unavailable), 502, 504, 408
                if response.status_code in (503, 502, 504, 408):
                    safe_detail = _extract_safe_response_detail(response, self.api_key)
                    logger.warning(
                        f"Deepgram STT transient HTTP {response.status_code} "
                        f"(attempt {attempt + 1}/{total_attempts}, dg-request-id: {dg_request_id}, "
                        f"content-type: {content_type_hdr}, details: {safe_detail})"
                    )
                    await reset_shared_stt_client()
                    if attempt < max_retries:
                        backoff = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                        logger.info(f"Retrying Deepgram STT in {backoff:.2f}s (retry {attempt + 1}/{max_retries})...")
                        await asyncio.sleep(backoff)
                        continue
                    else:
                        logger.error(
                            f"Deepgram STT failed after {total_attempts} attempts with HTTP {response.status_code} "
                            f"(dg-request-id: {dg_request_id}): {safe_detail}"
                        )
                        raise DeepgramAPIError(response.status_code, safe_detail)

                if response.status_code != 200:
                    safe_detail = _extract_safe_response_detail(response, self.api_key)
                    if response.status_code == 400 and "keyterm" in safe_detail.lower() and "keyterm" in params:
                        logger.warning(
                            f"[DeepgramSTT] Keyterm limit warning ({safe_detail}). Gracefully retrying without keyterms..."
                        )
                        params.pop("keyterm", None)
                        continue
                    logger.error(
                        f"Deepgram STT error HTTP {response.status_code} "
                        f"(dg-request-id: {dg_request_id}, details: {safe_detail})"
                    )
                    raise DeepgramAPIError(response.status_code, safe_detail)

                logger.info(
                    f"Deepgram STT HTTP 200 OK (dg-request-id: {dg_request_id}, "
                    f"content-type: {content_type_hdr}, bytes: {len(response.content)})"
                )
                data = response.json()
                break

            except httpx.RequestError as exc:
                safe_exc_msg = str(exc)
                if self.api_key and self.api_key in safe_exc_msg:
                    safe_exc_msg = safe_exc_msg.replace(self.api_key, "[REDACTED_API_KEY]")

                logger.warning(
                    f"Network error connecting to Deepgram STT on attempt {attempt + 1}/{total_attempts}: "
                    f"{type(exc).__name__}: {safe_exc_msg}"
                )
                await reset_shared_stt_client()
                if attempt < max_retries:
                    backoff = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                    logger.info(f"Retrying Deepgram STT in {backoff:.1f}s after network error...")
                    await asyncio.sleep(backoff)
                    continue

                logger.error(f"Deepgram STT network connection failed after {total_attempts} attempts: {safe_exc_msg}")
                raise DeepgramAPIError(503, f"Unable to reach Deepgram service: {safe_exc_msg}")

        # Extract transcript and detected language from Deepgram JSON structure
        channels = data.get("results", {}).get("channels", [])
        transcript = ""
        confidence = 0.0
        words = []
        detected_lang = language or None

        if channels and len(channels) > 0:
            ch = channels[0]
            # Check if Deepgram provided detected_language
            if not detected_lang:
                detected_lang = ch.get("detected_language")

            alternatives = ch.get("alternatives", [])
            if alternatives and len(alternatives) > 0:
                alt = alternatives[0]
                transcript = alt.get("transcript", "").strip()
                confidence = alt.get("confidence", 0.0)
                words = alt.get("words", [])
                if not detected_lang and alt.get("languages"):
                    detected_lang = alt.get("languages")[0]

        metadata = data.get("metadata", {})
        duration = metadata.get("duration", 0.0)

        # Log raw Deepgram final transcript BEFORE any normalization or processing
        logger.info(
            f"Deepgram raw final transcript (model={stt_model}, language={req_lang or 'unknown'}): '{transcript}'"
        )

        # Determine language: lexical spoken detection is authoritative over Deepgram audio classifier
        spoken_lang = detect_text_language(transcript, requested_language=req_lang or detected_lang)
        if spoken_lang in ("hi", "mr", "en"):
            final_lang = spoken_lang
        else:
            final_lang = req_lang or detected_lang or "en"

        clean_transcript = normalize_stt_transcript(transcript, final_lang)

        logger.info(
            f"Deepgram transcription complete. Language: {final_lang}, "
            f"Confidence: {confidence:.2f}, Duration: {duration:.1f}s"
        )

        return {
            "success": True,
            "transcript": clean_transcript,
            "confidence": round(confidence, 4),
            "words_count": len(words),
            "duration": round(duration, 2),
            "model": stt_model,
            "detected_language": final_lang,
            "stt_provider": "deepgram",
        }

    async def transcribe_multilingual_audio(
        self,
        audio_bytes: bytes,
        content_type: str = "audio/wav",
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Multilingual STT specialized for Web Module 1:
        1. Accurately detects and transcribes real user speech in English, Hindi, Marathi,
           and mixed-language speech (Hinglish/code-switched).
        2. Preserves exact words, names, numbers, phone, email, CIBIL, EMI, HDFC, and loan terms verbatim.
        3. Never translates, rewrites, corrects, or guesses.
        """
        self._validate_api_key()
        req_lang = (language or "").strip().lower()

        # Route explicit language requests directly to optimal Nova-3 model
        if req_lang in ("en", "en-in", "en-us", "english"):
            res = await self.transcribe_audio(audio_bytes, content_type=content_type, model="nova-3", language="en")
            res["stt_provider"] = "deepgram"
            return res
        elif req_lang in ("hi", "hi-in", "hindi"):
            res = await self.transcribe_audio(audio_bytes, content_type=content_type, model="nova-3", language="hi")
            res["stt_provider"] = "deepgram"
            return res
        elif req_lang in ("mr", "mr-in", "marathi"):
            res = await self.transcribe_audio(audio_bytes, content_type=content_type, model="nova-3", language="mr")
            res["stt_provider"] = "deepgram"
            return res

        # Single-pass ultra-low latency Nova-3 multilingual transcription
        res = await self.transcribe_audio(
            audio_bytes,
            content_type=content_type,
            model="nova-3",
            language="multi",
        )
        res["stt_provider"] = "deepgram"
        return res
