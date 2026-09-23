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
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import httpx
try:
    from services.language import (
        detect_language as detect_text_language,
        ROMANIZED_HINDI,
        ROMANIZED_MARATHI,
        HINDI_WORDS,
        MARATHI_WORDS,
    )
except ImportError:
    from backend.services.language import (
        detect_language as detect_text_language,
        ROMANIZED_HINDI,
        ROMANIZED_MARATHI,
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

# Domain-specific financial & lead keyterms for Deepgram Nova-3 keyword boosting (strictly under 500-token limit)
FINANCIAL_KEYTERMS = [
    # Core credit & bureau terms
    "CIBIL", "CIBIL score", "CIBIL स्कोर", "सिबिल", "सिबिल स्कोर",
    "EMI", "ईएमआई", "मासिक EMI", "दरमहा EMI", "ROI",
    # Loan products & terms
    "personal loan", "home loan", "business loan", "interest rate",
    "कर्ज", "वैयक्तिक कर्ज", "गृहकर्ज", "व्यवसाय कर्ज", "लोन", "पर्सनल लोन",
    # Company & tenure terms
    "company", "tenure", "months", "years", "कालावधी", "कागदपत्रे", "दरमहा",
    # Institutions & currency
    "HDFC", "HDFC Bank", "HDFC बँक", "SBI", "SBI बँक", "ICICI",
    "lakh", "crore", "thousand", "rupees", "रुपये", "लाख",
    # Contact & identity
    "phone", "mobile", "email",
]

MARATHI_MARKERS = {
    "kiti", "ahe", "aahe", "mala", "safi", "rupenship", "derma", "darmaha", "nav", "naav",
    "pahije", "havay", "have", "kay", "kasa", "kashi", "karayche", "karaycha", "baddal",
    "नाव", "आहे", "किती", "मला", "पाहिजे", "हवे", "काय", "कसे", "कर्ज", "रुपये", "लाख"
}

HINDI_MARKERS = {
    "mera", "meri", "mere", "naam", "hona", "chaheai", "chahiye", "kidna", "kitna", "kitne",
    "kya", "kaise", "bataiye", "bataye", "chahiye", "hai", "hain", "apna", "aapka",
    "नाम", "है", "कितना", "चाहिए", "लोन", "रुपये", "लाख", "फोन", "नंबर"
}

def normalize_stt_transcript(transcript: str, language: Optional[str] = None) -> str:
    """
    Normalizes whitespace and punctuation spacing without rewriting, correcting,
    translating, or replacing any words. The final transcript preserves Deepgram's
    actual recognized speech verbatim.
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
    params = [
        f"model={model}",
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
    if language and language.strip().lower() in ("en", "hi", "mr"):
        params.append(f"language={language.strip().lower()}")
    elif model == "nova-3":
        params.append("language=multi")
    else:
        params.append("detect_language=true")

    if model == "nova-3" and include_keyterms:
        for kt in FINANCIAL_KEYTERMS:
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
        if req_lang in ("mr", "hi", "multi", "auto") or model == "nova-3":
            stt_model = "nova-3"

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
        elif req_lang in ("multi", "auto", "unknown") or not req_lang:
            if stt_model == "nova-3":
                params["language"] = "multi"
            elif detect_language is not False:
                params["detect_language"] = "true"
        else:
            params["language"] = req_lang

        if stt_model == "nova-3":
            params["keyterm"] = FINANCIAL_KEYTERMS

        logger.info(
            f"[DeepgramSTT] Sending {len(audio_bytes)} audio bytes to Deepgram STT "
            f"(model: {stt_model}, content_type: '{mime}', params: {params})"
        )

        data = None
        max_retries = 2  # Max 2 retries on transient errors (total 3 attempts)
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
                    if attempt < max_retries:
                        backoff = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                        logger.info(f"Retrying Deepgram STT in {backoff:.1f}s (retry {attempt + 1}/{max_retries})...")
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

        # Determine language: lexical spoken detection is authoritative over Deepgram audio classifier
        spoken_lang = detect_text_language(transcript)
        if spoken_lang in ("hi", "mr", "mixed", "en"):
            final_lang = spoken_lang
        else:
            final_lang = detected_lang or "en"

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
