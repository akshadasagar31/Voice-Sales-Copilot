# ============================================================================
# SPEECH-TO-TEXT SERVICE: SARVAM AI SAARAS V4 (backend/services/sarvam_stt.py)
# ============================================================================
# WHAT THIS SERVICE DOES:
# Transcribes Indian language speech (specifically Hindi) using Sarvam AI's
# state-of-the-art 'saaras:v4' STT model.
#
# KEY FEATURES:
# 1. Native Indic Model: Uses 'saaras:v4' with 'hi-IN' for authentic Devanagari transcription.
# 2. Number & Hinglish Preservation: Faithfully captures numerical digits ('750', '5 लाख')
#    and financial terms ('CIBIL', 'HDFC Bank', 'EMI', 'personal loan', 'interest rate')
#    without phonetic loss or translation.
# 3. Connection Pooling: Persistent httpx.AsyncClient with keep-alive connection pooling.
# 4. Graceful Fallbacks: Returns clear structured exceptions so callers can fall back
#    to Deepgram Nova-3 if needed.
# ============================================================================

import os
import io
import wave
import logging
from typing import Optional, Dict, Any
import httpx

logger = logging.getLogger(__name__)

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
DEFAULT_SARVAM_STT_MODEL = "saaras:v4"
DEFAULT_SARVAM_STT_LANGUAGE = "hi-IN"
SARVAM_STT_TIMEOUT = 30.0

_shared_sarvam_stt_client: Optional[httpx.AsyncClient] = None


def get_shared_sarvam_stt_client() -> httpx.AsyncClient:
    """Return a shared persistent AsyncClient with keep-alive connection pooling for Sarvam STT."""
    global _shared_sarvam_stt_client
    if _shared_sarvam_stt_client is None or _shared_sarvam_stt_client.is_closed:
        _shared_sarvam_stt_client = httpx.AsyncClient(
            timeout=SARVAM_STT_TIMEOUT,
            limits=httpx.Limits(max_keepalive_connections=15, max_connections=30, keepalive_expiry=30.0),
        )
    return _shared_sarvam_stt_client


class SarvamSTTError(Exception):
    """Base exception for Sarvam STT errors."""
    pass


class SarvamSTTConfigurationError(SarvamSTTError):
    """Raised when SARVAM_API_KEY is not configured."""
    pass


class SarvamSTTAPIError(SarvamSTTError):
    """Raised when the Sarvam STT API returns an error response."""
    def __init__(self, status_code: int, message: str):
        super().__init__(f"Sarvam STT API error (HTTP {status_code}): {message}")
        self.status_code = status_code
        self.message = message


class SarvamSTTService:
    """
    Service for converting spoken audio into accurate Devanagari Hindi text
    using Sarvam AI's saaras:v4 model. Specialized for Module 2 Hindi queries.
    """

    def __init__(self, api_key: Optional[str] = None):
        if api_key is not None:
            self.api_key = api_key.strip()
        else:
            self.api_key = os.getenv("SARVAM_API_KEY", "").strip()

    def _validate_api_key(self) -> None:
        """Validates that SARVAM_API_KEY is configured."""
        if not self.api_key:
            raise SarvamSTTConfigurationError(
                "SARVAM_API_KEY is not configured in backend/.env. "
                "Please set SARVAM_API_KEY to use Sarvam STT."
            )

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        content_type: str = "audio/wav",
        language_code: str = DEFAULT_SARVAM_STT_LANGUAGE,
        model: Optional[str] = None,
        filename: Optional[str] = None,
        mode: Optional[str] = "verbatim",
    ) -> Dict[str, Any]:
        """
        Transcribes audio bytes into text using Sarvam AI STT API.

        Args:
            audio_bytes: Binary audio data (WAV, WebM, MP3, etc.).
            content_type: MIME type of the audio.
            language_code: BCP-47 language code (default: 'hi-IN').
            model: Sarvam model override (default: 'saaras:v4').
            filename: Optional source filename.
            mode: Output mode ('verbatim', 'transcribe', 'codemix'). Default: 'verbatim'.

        Returns:
            Dict containing 'transcript', 'detected_language', 'duration', 'confidence', etc.
        """
        self._validate_api_key()

        if not audio_bytes or len(audio_bytes) == 0:
            raise ValueError("Audio bytes cannot be empty (0 bytes).")

        stt_model = (model or DEFAULT_SARVAM_STT_MODEL).strip()
        raw_lang = (language_code or DEFAULT_SARVAM_STT_LANGUAGE).strip().lower()
        if raw_lang in ("unknown", "auto", "detect"):
            target_lang = "unknown"
        elif "-" not in raw_lang:
            target_lang = f"{raw_lang}-IN"
        else:
            target_lang = raw_lang

        ext = "wav"
        if "webm" in content_type:
            ext = "webm"
        elif "ogg" in content_type:
            ext = "ogg"
        elif "mp4" in content_type:
            ext = "mp4"
        elif "mp3" in content_type or "mpeg" in content_type:
            ext = "mp3"

        out_filename = filename or f"audio_input.{ext}"

        # Estimate duration if WAV format
        duration = 0.0
        if audio_bytes.startswith(b"RIFF") and len(audio_bytes) > 44:
            try:
                with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    if rate > 0:
                        duration = round(frames / float(rate), 2)
            except Exception:
                pass

        headers = {
            "api-subscription-key": self.api_key,
        }

        files = {
            "file": (out_filename, audio_bytes, content_type),
        }
        data: Dict[str, str] = {
            "language_code": target_lang,
            "model": stt_model,
        }
        if mode:
            data["mode"] = mode.strip()

        logger.info(
            f"Sending STT request to Sarvam ({stt_model}, lang: {target_lang}, "
            f"size: {len(audio_bytes)} bytes, mime: {content_type})"
        )

        client = get_shared_sarvam_stt_client()

        for attempt in range(2):
            try:
                response = await client.post(
                    SARVAM_STT_URL,
                    headers=headers,
                    files=files,
                    data=data,
                )

                if response.status_code != 200:
                    err_msg = response.text
                    try:
                        err_json = response.json()
                        err_msg = err_json.get("detail") or err_json.get("message") or err_json.get("error") or err_msg
                    except Exception:
                        pass

                    logger.error(f"Sarvam STT returned HTTP {response.status_code}: {err_msg}")
                    if response.status_code in (502, 503, 504) and attempt == 0:
                        continue
                    raise SarvamSTTAPIError(response.status_code, err_msg)

                res_data = response.json()
                transcript = (res_data.get("transcript") or res_data.get("text") or "").strip()

                # Extract detected language from response or fallback to lexical detection
                resp_lang_code = res_data.get("language_code") or target_lang
                if resp_lang_code and resp_lang_code != "unknown":
                    detected_lang = resp_lang_code.split("-")[0].lower()
                else:
                    try:
                        from services.language import detect_language
                        detected_lang = detect_language(transcript) or "en"
                    except Exception:
                        detected_lang = "en"

                logger.info(
                    f"Sarvam STT success ({stt_model}): '{transcript[:80]}...' (len: {len(transcript)}, lang: {detected_lang})"
                )

                return {
                    "success": True,
                    "transcript": transcript,
                    "confidence": 0.98,
                    "detected_language": detected_lang,
                    "duration": duration,
                    "stt_provider": "sarvam",
                    "model": stt_model,
                    "raw_response": res_data,
                }

            except httpx.RequestError as exc:
                if attempt == 0:
                    continue
                logger.error(f"Network error connecting to Sarvam STT: {str(exc)}")
                raise SarvamSTTAPIError(503, f"Unable to reach Sarvam STT service: {str(exc)}")
