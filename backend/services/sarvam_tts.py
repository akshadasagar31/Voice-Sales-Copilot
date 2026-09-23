# ============================================================================
# TEXT-TO-SPEECH SERVICE: SARVAM AI BULBUL V3 (backend/services/sarvam_tts.py)
# ============================================================================
# WHAT THIS SERVICE DOES:
# Synthesizes natural, high-fidelity Indian language speech (specifically Marathi)
# using Sarvam AI's Bulbul v3 model.
#
# KEY FEATURES:
# 1. Native Indic Model: Uses 'bulbul:v3' with 'mr-IN' and authentic voices ('priya'/'ritu').
# 2. Code-Mixed Fluency: Preserves English financial terms (CIBIL score, personal loan,
#    ROI, EMI, interest rate, tenure, HDFC Bank) without phonetic degradation.
# 3. Connection Pooling: Uses persistent httpx.AsyncClient for rapid streaming response.
# 4. Graceful Fallbacks: Returns clear errors so callers can fall back to Deepgram if needed.
# ============================================================================

import os
import re
import base64
import logging
from pathlib import Path
from typing import Optional
import httpx
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
else:
    load_dotenv(override=True)

logger = logging.getLogger(__name__)

SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
DEFAULT_SARVAM_MODEL = "bulbul:v3"
DEFAULT_SARVAM_MARATHI_VOICE = os.getenv("SARVAM_MARATHI_VOICE", "simran").strip().lower()
DEFAULT_SARVAM_HINDI_VOICE = os.getenv("SARVAM_HINDI_VOICE", "simran").strip().lower()
DEFAULT_SARVAM_ENGLISH_VOICE = os.getenv("SARVAM_ENGLISH_VOICE", "simran").strip().lower()
DEFAULT_SARVAM_VOICE = DEFAULT_SARVAM_MARATHI_VOICE
SARVAM_TIMEOUT = 30.0

import asyncio

_shared_sarvam_client: Optional[httpx.AsyncClient] = None


def get_shared_sarvam_client() -> httpx.AsyncClient:
    """Return a shared persistent AsyncClient with keep-alive connection pooling for Sarvam TTS."""
    global _shared_sarvam_client
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if _shared_sarvam_client is not None and not _shared_sarvam_client.is_closed:
        transport = getattr(_shared_sarvam_client, "_transport", None)
        pool = getattr(transport, "_pool", None)
        loop = getattr(pool, "_loop", None)
        if loop is not None and (loop.is_closed() or (current_loop is not None and loop is not current_loop)):
            _shared_sarvam_client = None

    if _shared_sarvam_client is None or _shared_sarvam_client.is_closed:
        _shared_sarvam_client = httpx.AsyncClient(
            timeout=SARVAM_TIMEOUT,
            limits=httpx.Limits(max_keepalive_connections=15, max_connections=30, keepalive_expiry=30.0),
        )
    return _shared_sarvam_client


class SarvamTTSError(Exception):
    """Base exception for Sarvam TTS errors."""
    pass


class SarvamTTSConfigurationError(SarvamTTSError):
    """Raised when SARVAM_API_KEY is not configured."""
    pass


class SarvamTTSAPIError(SarvamTTSError):
    """Raised when the Sarvam TTS API returns an error."""
    def __init__(self, status_code: int, message: str):
        super().__init__(f"Sarvam TTS API error (HTTP {status_code}): {message}")
        self.status_code = status_code
        self.message = message


def clean_hindi_financial_text(text: str) -> str:
    """
    Prepares Hindi text for Sarvam Bulbul v3 speech synthesis.
    Preserves English financial terms (CIBIL score, EMI, personal loan, HDFC Bank, ROI, etc.)
    and expands financial symbols into natural spoken Hindi words.
    """
    if not text:
        return ""

    cleaned = text.strip()

    # Remove markdown bold/italics and code blocks
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"`[^`]*`", "", cleaned)
    cleaned = re.sub(r"#{1,6}\s+", "", cleaned)

    # Expand common financial symbols for smooth spoken delivery
    # Rupee symbol ₹ or Rs. -> रुपये
    cleaned = re.sub(r"₹\s*([\d,]+(?:\.\d+)?)", r"\1 रुपये", cleaned)
    cleaned = re.sub(r"\bRs\.?\s*([\d,]+(?:\.\d+)?)", r"\1 रुपये", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("₹", " रुपये ")

    # Percentage symbol % -> प्रतिशत
    cleaned = re.sub(r"([\d,]+(?:\.\d+)?)\s*%", r"\1 प्रतिशत", cleaned)

    # Per annum abbreviation p.a. -> प्रति वर्ष
    cleaned = re.sub(r"(?i)\b(?:p\.a\.|p/a|pa)\b", "प्रति वर्ष", cleaned)
    cleaned = re.sub(r"(?i)\bp\.a\.", "प्रति वर्ष", cleaned)

    # Standardize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def clean_marathi_financial_text(text: str) -> str:
    """
    Prepares Marathi text for Sarvam Bulbul v3 speech synthesis.
    Preserves English financial terms (CIBIL score, EMI, loan, interest rate, etc.)
    and expands financial symbols into natural spoken Marathi words.
    """
    if not text:
        return ""

    cleaned = text.strip()

    # Remove markdown bold/italics and code blocks
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"`[^`]*`", "", cleaned)
    cleaned = re.sub(r"#{1,6}\s+", "", cleaned)

    # Expand common financial symbols for smooth spoken delivery
    # Rupee symbol ₹ or Rs. -> रुपये
    cleaned = re.sub(r"₹\s*([\d,]+(?:\.\d+)?)", r"\1 रुपये", cleaned)
    cleaned = re.sub(r"\bRs\.?\s*([\d,]+(?:\.\d+)?)", r"\1 रुपये", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("₹", " रुपये ")

    # Percentage symbol % -> टक्के
    cleaned = re.sub(r"([\d,]+(?:\.\d+)?)\s*%", r"\1 टक्के", cleaned)

    # Per annum abbreviation p.a. -> प्रति वर्ष
    cleaned = re.sub(r"(?i)\b(?:p\.a\.|p/a|pa)\b", "प्रति वर्ष", cleaned)
    cleaned = re.sub(r"(?i)\bp\.a\.", "प्रति वर्ष", cleaned)

    # Standardize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def clean_english_financial_text(text: str) -> str:
    """
    Prepares English text for Sarvam Bulbul v3 speech synthesis.
    Expands financial symbols and abbreviations into natural spoken English words.
    """
    if not text:
        return ""

    cleaned = text.strip()

    # Remove markdown bold/italics and code blocks
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"`[^`]*`", "", cleaned)
    cleaned = re.sub(r"#{1,6}\s+", "", cleaned)

    # Expand common financial symbols for smooth spoken delivery
    # Rupee symbol ₹ or Rs. -> rupees
    cleaned = re.sub(r"₹\s*([\d,]+(?:\.\d+)?)", r"\1 rupees", cleaned)
    cleaned = re.sub(r"\bRs\.?\s*([\d,]+(?:\.\d+)?)", r"\1 rupees", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("₹", " rupees ")

    # Percentage symbol % -> percent
    cleaned = re.sub(r"([\d,]+(?:\.\d+)?)\s*%", r"\1 percent", cleaned)

    # Per annum abbreviation p.a. -> per annum
    cleaned = re.sub(r"(?i)\b(?:p\.a\.|p/a|pa)\b", "per annum", cleaned)
    cleaned = re.sub(r"(?i)\bp\.a\.", "per annum", cleaned)

    # Standardize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


class SarvamTTSService:
    """
    Service for converting text into natural Indian language speech using Sarvam AI Bulbul v3.
    Specialized for Module 2 Indian English (en-IN), Marathi (mr-IN), and Hindi (hi-IN) voice responses.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._explicit_key = api_key is not None
        if self._explicit_key:
            self.api_key = (api_key or "").strip()
        else:
            env_file = Path(__file__).resolve().parent.parent / ".env"
            if env_file.exists():
                load_dotenv(dotenv_path=env_file, override=True)
            self.api_key = os.getenv("SARVAM_API_KEY", "").strip()

    def _validate_api_key(self) -> None:
        """Validates that SARVAM_API_KEY is configured."""
        if not self._explicit_key:
            env_file = Path(__file__).resolve().parent.parent / ".env"
            if env_file.exists():
                load_dotenv(dotenv_path=env_file, override=True)
            self.api_key = os.getenv("SARVAM_API_KEY", "").strip()

        if not self.api_key:
            raise SarvamTTSConfigurationError(
                "SARVAM_API_KEY is not configured in backend/.env. "
                "Please set SARVAM_API_KEY to use Sarvam Bulbul v3 TTS."
            )

    async def synthesize_speech(
        self,
        text: str,
        language_code: str = "mr-IN",
        speaker: Optional[str] = None,
        model: Optional[str] = None,
    ) -> bytes:
        """
        Sends text to Sarvam AI text-to-speech API and returns raw WAV audio bytes.

        Args:
            text: Text to synthesize (English, Hindi, or Marathi with financial terms).
            language_code: BCP-47 language code ('en-IN', 'mr-IN', or 'hi-IN').
            speaker: Voice speaker name ('simran', 'priya', 'ritu', etc.).
            model: Sarvam model name (default: 'bulbul:v3').

        Returns:
            bytes: Binary audio bytes (RIFF WAVE format).
        """
        self._validate_api_key()

        raw_text = (text or "").strip()
        if not raw_text:
            raise ValueError("Text to synthesize cannot be empty or whitespace.")

        target_lang = language_code if "-" in language_code else f"{language_code.lower()}-IN"
        is_hindi = target_lang.lower().startswith("hi")
        is_english = target_lang.lower().startswith("en")

        if is_hindi:
            spoken_text = clean_hindi_financial_text(raw_text)
            default_voice = os.getenv("SARVAM_HINDI_VOICE", DEFAULT_SARVAM_HINDI_VOICE).strip().lower()
        elif is_english:
            spoken_text = clean_english_financial_text(raw_text)
            default_voice = os.getenv("SARVAM_ENGLISH_VOICE", DEFAULT_SARVAM_ENGLISH_VOICE).strip().lower()
        else:
            spoken_text = clean_marathi_financial_text(raw_text)
            default_voice = os.getenv("SARVAM_MARATHI_VOICE", DEFAULT_SARVAM_MARATHI_VOICE).strip().lower()

        if not spoken_text:
            raise ValueError("Text contains no speakable characters.")

        voice_speaker = (speaker or default_voice).strip().lower()
        # Ensure speaker is one of the supported natural voices (defaults to natural female 'simran')
        valid_speakers = {
            "simran", "priya", "ritu", "ishita", "neha", "pooja", "kavya",
            "shreya", "roopa", "tanya", "shruti", "suhani", "kavitha", "rupali",
            "aditya", "arvind", "amartya", "ashutosh", "rahul", "rohan", "amit",
            "dev", "ratan", "varun", "manan", "sumit", "kabir", "aayan", "shubh",
            "advait", "anand", "tarun", "sunny", "mani", "gokul", "vijay", "mohit",
            "rehan", "soham"
        }
        if voice_speaker not in valid_speakers:
            voice_speaker = "simran"

        tts_model = (model or DEFAULT_SARVAM_MODEL or "bulbul:v3").strip()

        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "text": spoken_text,
            "model": tts_model,
            "language_code": target_lang,
            "speaker": voice_speaker,
        }

        logger.info(
            f"Sending Sarvam Bulbul v3 TTS request "
            f"(speaker: {voice_speaker}, lang: {target_lang}, len: {len(spoken_text)})"
        )

        client = get_shared_sarvam_client()

        for attempt in range(2):
            try:
                response = await client.post(
                    SARVAM_TTS_URL,
                    headers=headers,
                    json=payload,
                )

                # Alternate payload format if standard schema fails
                if response.status_code in (400, 422):
                    fallback_payload = {
                        "inputs": [spoken_text],
                        "target_language_code": target_lang,
                        "speaker": voice_speaker,
                        "model": tts_model,
                    }
                    alt_response = await client.post(
                        SARVAM_TTS_URL,
                        headers=headers,
                        json=fallback_payload,
                    )
                    if alt_response.status_code == 200:
                        response = alt_response

                if response.status_code != 200:
                    err_msg = response.text
                    err_code = None
                    try:
                        err_json = response.json()
                        if isinstance(err_json, dict):
                            err_obj = err_json.get("error")
                            if isinstance(err_obj, dict):
                                err_msg = err_obj.get("message") or str(err_obj)
                                err_code = err_obj.get("code")
                            elif isinstance(err_obj, str):
                                err_msg = err_obj
                            else:
                                err_msg = err_json.get("message") or response.text
                                err_code = err_json.get("code")
                    except Exception:
                        pass

                    logger.error(f"Sarvam TTS returned HTTP {response.status_code} ({err_code}): {err_msg}")

                    if response.status_code == 402 or err_code == "insufficient_quota_error":
                        raise SarvamTTSAPIError(
                            402,
                            "Sarvam AI account has no remaining credits (insufficient_quota_error). "
                            "Please recharge credits at https://dashboard.sarvam.ai or update SARVAM_API_KEY in backend/.env."
                        )

                    if response.status_code in (502, 503, 504) and attempt == 0:
                        continue
                    raise SarvamTTSAPIError(response.status_code, err_msg)

                data = response.json()
                audios = data.get("audios") or []
                if not audios or not isinstance(audios, list):
                    raise SarvamTTSAPIError(500, "Sarvam API did not return audio data in response.")

                audio_bytes = base64.b64decode(audios[0])
                if not audio_bytes:
                    raise SarvamTTSAPIError(500, "Sarvam returned empty decoded audio.")

                logger.info(
                    f"Sarvam TTS synthesis complete ({len(audio_bytes)} bytes WAV, speaker: {voice_speaker})."
                )
                return audio_bytes

            except (httpx.RequestError, RuntimeError) as exc:
                if attempt == 0:
                    continue
                logger.error(f"Network error connecting to Sarvam TTS: {str(exc)}")
                raise SarvamTTSAPIError(503, f"Unable to reach Sarvam TTS service: {str(exc)}")
