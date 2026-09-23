# ============================================================================
# TEXT-TO-SPEECH SERVICE: DEEPGRAM AURA (backend/services/tts.py)
# ============================================================================
# WHAT THIS SERVICE DOES:
# Converts written text into human-sounding spoken audio using Deepgram Aura TTS.
#
# KEY FEATURES:
# 1. Low-Latency Voice Models: Uses Deepgram Aura voices for fast speech synthesis.
# 2. Multilingual Voice Mapping:
#    - English: aura-asteria-en
#    - Hindi:   aura-luna-en
#    - Marathi: aura-stella-en
# 3. Devanagari Phonetic Fallback: Deepgram Aura TTS operates on romanized phonemes.
#    If the input contains Devanagari script (Hindi/Marathi), the service automatically
#    converts the text into clean phonetic representations so it sounds natural.
# 4. Concise Confirmation Generator: Builds short, clear spoken confirmations
#    summarizing captured lead information.
# ============================================================================

import os
import asyncio
import logging
from typing import Dict, Any, Optional, Union
import httpx
from services.language import (
    detect_language,
    devanagari_to_phonetic,
    count_devanagari_chars,
    LANG_EN,
    LANG_HI,
    LANG_MR,
)
from services.sarvam_tts import (
    SarvamTTSService,
    SarvamTTSError,
    SarvamTTSConfigurationError,
    SarvamTTSAPIError,
    clean_marathi_financial_text,
)

logger = logging.getLogger(__name__)

DEEPGRAM_SPEAK_URL = "https://api.deepgram.com/v1/speak"
DEFAULT_TTS_MODEL = "aura-asteria-en"

_shared_tts_clients: Dict[int, httpx.AsyncClient] = {}

def get_shared_tts_client() -> httpx.AsyncClient:
    """Return a shared persistent AsyncClient with keep-alive connection pooling for Deepgram TTS."""
    try:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
    except RuntimeError:
        loop_id = 0

    client = _shared_tts_clients.get(loop_id)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50, keepalive_expiry=30.0),
        )
        if loop_id != 0:
            _shared_tts_clients[loop_id] = client
    return client

LANGUAGE_TTS_MODELS = {
    LANG_EN: os.getenv("DEEPGRAM_TTS_MODEL_EN", "aura-asteria-en"),
    LANG_HI: os.getenv("DEEPGRAM_TTS_MODEL_HI", "aura-luna-en"),
    LANG_MR: os.getenv("DEEPGRAM_TTS_MODEL_MR", "aura-stella-en"),
}



class DeepgramTTSError(Exception):
    """Base exception for Deepgram TTS errors."""
    pass


class DeepgramTTSConfigurationError(DeepgramTTSError):
    """Raised when DEEPGRAM_API_KEY is not configured or missing."""
    pass


class DeepgramTTSAPIError(DeepgramTTSError):
    """Raised when the Deepgram TTS API returns an error response."""
    def __init__(self, status_code: int, message: str):
        super().__init__(f"Deepgram TTS API error (HTTP {status_code}): {message}")
        self.status_code = status_code
        self.message = message


class DeepgramTTSService:
    """
    Service for converting text into speech using Deepgram's Text-to-Speech (TTS) API.
    Supports English, Hindi, and Marathi.
    Reads DEEPGRAM_API_KEY from backend .env only.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (api_key or os.getenv("DEEPGRAM_API_KEY", "")).strip()

    def _validate_api_key(self) -> None:
        """Validates that DEEPGRAM_API_KEY is present."""
        if not self.api_key:
            raise DeepgramTTSConfigurationError(
                "DEEPGRAM_API_KEY is not configured in backend/.env. "
                "Please add DEEPGRAM_API_KEY to backend/.env to synthesize speech."
            )

    # ------------------------------------------------------------------------
    # METHOD: synthesize_speech
    # ------------------------------------------------------------------------
    # • WHAT IT DOES: Sends text to Deepgram Aura TTS API and returns MP3 audio bytes.
    # • INPUTS:
    #     - text (str): The sentence to speak (English, Hindi, or Marathi).
    #     - model (Optional[str]): Voice model override.
    #     - language (Optional[str]): Language code ('en', 'hi', 'mr').
    # • OUTPUT: Binary audio bytes (`audio/mpeg` MP3 stream).
    # • WHY IT IS USED: Automatically transliterates Indian Devanagari text into phonetic
    #   Latin script before calling Deepgram Aura, ensuring natural native pronunciation.
    # • WHERE IT FITS IN THE FLOW:
    #     [SentenceAudioQueue] -> [api/tts] -> [synthesize_speech] -> [Browser Speakers]
    # ------------------------------------------------------------------------
    async def synthesize_speech(
        self,
        text: str,
        model: Optional[str] = None,
        language: Optional[str] = None,
        skip_sarvam: bool = False,
    ) -> bytes:
        """
        Sends text to Deepgram Text-to-Speech endpoint and returns raw audio bytes (audio/mpeg).

        Args:
            text: Text content to convert into speech (English, Hindi, or Marathi).
            model: Deepgram voice model override.
            language: Optional language code ('en', 'hi', 'mr').
            skip_sarvam: If True, skips Sarvam TTS and synthesizes directly with Deepgram Aura.

        Returns:
            bytes: Binary audio bytes (audio/mpeg).
        """
        self._validate_api_key()

        clean_text = (text or "").strip()
        if not clean_text:
            raise ValueError("Text to synthesize cannot be empty or whitespace.")

        # Determine target language and voice model
        target_lang = (language or detect_language(clean_text) or LANG_EN).strip().lower()

        # Primary TTS Engine: Sarvam Bulbul v3 with female voice 'simran' (skipped if skip_sarvam=True)
        if not skip_sarvam:
            if target_lang in ("hi", "hi-in", "hindi"):
                sarvam_lang = "hi-IN"
            elif target_lang in ("mr", "mr-in", "marathi"):
                sarvam_lang = "mr-IN"
            else:
                sarvam_lang = "en-IN"

            try:
                sarvam_tts = SarvamTTSService()
                return await sarvam_tts.synthesize_speech(
                    clean_text,
                    language_code=sarvam_lang,
                    speaker="simran",
                    model="bulbul:v3",
                )
            except Exception as sarvam_err:
                logger.warning(
                    f"[TTSService] Sarvam Bulbul v3 ({sarvam_lang}) synthesis error ({sarvam_err}). "
                    "Falling back to Deepgram Aura TTS."
                )


        if model and model.strip():
            voice_model = model.strip()
        elif target_lang in LANGUAGE_TTS_MODELS:
            voice_model = LANGUAGE_TTS_MODELS[target_lang]
        else:
            voice_model = DEFAULT_TTS_MODEL

        # For Hindi and Marathi Devanagari text in Deepgram fallback, transliterate to clear phonetic Latin script
        # so Deepgram's text-to-speech engine pronounces every word accurately and naturally!
        has_devanagari = count_devanagari_chars(clean_text) >= 2 or target_lang in (LANG_HI, LANG_MR)
        if has_devanagari:
            spoken_text = devanagari_to_phonetic(clean_text)
            logger.info(
                f"Deepgram TTS converted Devanagari ({target_lang}) to phonetic text: "
                f"'{spoken_text[:100]}...' (original length: {len(clean_text)})"
            )
        else:
            spoken_text = clean_text

        url = f"{DEEPGRAM_SPEAK_URL}?model={voice_model}"

        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {"text": spoken_text}

        logger.info(
            f"Sending TTS request to Deepgram (model: {voice_model}, "
            f"language: {target_lang}, spoken text length: {len(spoken_text)})"
        )

        for attempt in range(2):
            try:
                client = get_shared_tts_client()
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload,
                )

                if response.status_code != 200:
                    error_detail = response.text
                    try:
                        err_json = response.json()
                        error_detail = (
                            err_json.get("err_msg")
                            or err_json.get("message")
                            or err_json.get("error")
                            or error_detail
                        )
                    except Exception:
                        pass

                    logger.error(f"Deepgram TTS returned HTTP {response.status_code}: {error_detail}")
                    if response.status_code in (502, 503, 504) and attempt == 0:
                        await asyncio.sleep(0.15)
                        continue
                    raise DeepgramTTSAPIError(response.status_code, error_detail)

                audio_bytes = response.content
                if not audio_bytes:
                    raise DeepgramTTSAPIError(500, "Deepgram returned an empty audio response.")

                logger.info(f"Deepgram TTS synthesis complete. Audio size: {len(audio_bytes)} bytes.")
                return audio_bytes

            except (httpx.RequestError, RuntimeError) as exc:
                if attempt == 0:
                    await asyncio.sleep(0.15)
                    continue
                logger.error(f"Network error connecting to Deepgram TTS: {str(exc)}")
                raise DeepgramTTSAPIError(503, f"Unable to reach Deepgram TTS service: {str(exc)}")


# ----------------------------------------------------------------------------
# FUNCTION: generate_concise_response
# ----------------------------------------------------------------------------
# • WHAT IT DOES: Formulates a short 1-2 sentence conversational voice confirmation
#   summarizing newly captured or updated prospect fields (name, loan amount, company).
# • INPUTS:
#     - lead (Union[Dict, Lead]): The validated lead object or dictionary.
#     - language (Optional[str]): Target language ("en", "hi", "mr").
#     - is_update (bool): True if an existing lead record was updated in-place.
# • OUTPUT: String response text in the prospect's language (e.g. "I've saved Rajesh Kumar's $50k Equipment Loan to the CRM.").
# • WHY IT IS USED: Keeps voice interactions fast, natural, and non-robotic.
# • WHERE IT FITS IN THE FLOW:
#     [Lead Extracted] -> [generate_concise_response] -> [synthesize_speech / SSE Stream]
# ----------------------------------------------------------------------------
def generate_concise_response(
    lead: Union[Dict[str, Any], Any],
    language: Optional[str] = None,
    is_update: bool = False,
) -> str:
    """
    Generates a natural, concise 1-2 sentence sales copilot voice confirmation

    for a captured or saved/updated lead (Module 1).
    Supports English ('en'), Hindi ('hi'), and Marathi ('mr').

    Args:
        lead: A Lead Pydantic model or dictionary representation of lead fields.
        language: Optional language code ('en', 'hi', 'mr').
        is_update: True if an existing lead was updated rather than newly created.

    Returns:
        str: Concise spoken confirmation text.
    """
    lang = (language or LANG_EN).strip().lower()

    if isinstance(lead, dict):
        lead_dict = lead
    elif hasattr(lead, "model_dump"):
        lead_dict = lead.model_dump()
    elif hasattr(lead, "dict"):
        lead_dict = lead.dict()
    else:
        lead_dict = {}

    name = lead_dict.get("name")
    company = lead_dict.get("company")
    loan_type = lead_dict.get("loan_type")
    loan_amount = lead_dict.get("loan_amount")
    tenure_months = lead_dict.get("tenure_months")

    # Format loan amount for speech
    amount_str = ""
    if loan_amount is not None:
        try:
            amt = float(loan_amount)
            if amt >= 1000000:
                amount_str = f"${amt / 1000000:.1f} million".replace(".0 million", " million")
            elif amt >= 1000:
                amount_str = f"${amt:,.0f}"
            else:
                amount_str = f"${amt:.0f}"
        except (ValueError, TypeError):
            amount_str = str(loan_amount)

    # English confirmation
    if lang == LANG_EN or lang not in (LANG_HI, LANG_MR):
        subject = name or "the prospect"
        company_clause = f" from {company}" if company else ""
        loan_clause = f" for a {amount_str} {loan_type}" if (amount_str and loan_type) else (
            f" for a {amount_str} loan" if amount_str else (
                f" for a {loan_type}" if loan_type else ""
            )
        )
        tenure_clause = f" over {tenure_months} months" if tenure_months else ""

        if is_update:
            if name:
                return f"Updated {name}'s lead details in the CRM."
            return "Lead details have been successfully updated in the PostgreSQL CRM database."

        if name or company or loan_type or loan_amount:
            return f"Lead for {subject}{company_clause}{loan_clause}{tenure_clause} has been successfully recorded in the CRM."
        return "Lead details have been successfully verified and saved to the CRM database."

    # Hindi confirmation
    if lang == LANG_HI:
        if is_update:
            if name:
                return f"सीआरएम में {name} के लीड विवरण को अपडेट कर दिया गया है।"
            return "सीआरएम में लीड विवरण सफलतापूर्वक अपडेट कर दिया गया है।"
        if name:
            comp = f" ({company})" if company else ""
            return f"सीआरएम में {name}{comp} के लिए लीड विवरण सफलतापूर्वक सहेज लिया गया है।"
        return "लीड विवरण सफलतापूर्वक सत्यापित करके सीआरएम डेटाबेस में सहेज लिया गया है।"

    # Marathi confirmation
    if lang == LANG_MR:
        if is_update:
            if name:
                return f"सीआरएम मध्ये {name} यांचे लीड तपशील अपडेट केले आहेत."
            return "सीआरएम मध्ये लीड तपशील यशस्वीरित्या अपडेट केले आहेत."
        if name:
            comp = f" ({company})" if company else ""
            return f"सीआरएम मध्ये {name}{comp} यांचे लीड तपशील यशस्वीरित्या नोंदवले गेले आहेत."
        return "लीड तपशील यशस्वीरित्या सत्यापित करून सीआरएम डेटाबेसमध्ये नोंदवले गेले आहेत."

    return "Lead processed successfully in the CRM database."
