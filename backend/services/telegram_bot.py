# ============================================================================
# TELEGRAM BOT INTEGRATION: VOICE COPILOT BOT (backend/services/telegram_bot.py)
# ============================================================================
# WHAT THIS SERVICE DOES:
# Connects Telegram messenger directly to Module 2 Knowledge Assistant RAG & Voice pipeline.
#
# KEY FEATURES:
# 1. Bi-Modal Input: Accepts both text queries and voice notes (.ogg/opus) from Telegram users.
# 2. Multilingual STT: Transcribes Telegram voice notes using Sarvam AI saaras:v4
#    (with automatic fallback to Deepgram Nova-3), accurately preserving terms in English,
#    Hindi, and Marathi.
# 3. Grounded Playbook RAG: Queries 1536-d Pinecone vector knowledge base via RAGService
#    and OpenRouter LLM for zero-hallucination answers.
# 4. Low-Latency Dispatch: Immediately dispatches text answer as soon as LLM completes,
#    then generates and sends voice response in sequence.
# 5. Natural Voice Synthesis: Converts answer text to high-quality Indian speech using
#    Sarvam Bulbul v3 ('simran') with automatic fallback to Deepgram Aura.
# ============================================================================

import os
import io
import re
import asyncio
import subprocess
import logging
from typing import Dict, Any, Optional, Union, Tuple
import httpx

from services.language import (
    detect_language,
    LANG_EN,
    LANG_HI,
    LANG_MR,
)
from services.lead_extractor import (
    LeadExtractorService,
    Lead,
    get_next_missing_parameter,
    get_missing_parameter_prompt,
    REQUIRED_LEAD_FIELDS,
)
from services.lead_repository import (
    LeadRepository,
)
from services.rag import RAGService
from services.retriever import VectorRetriever
from services.vector_store import PineconeService
from services.sarvam_stt import (
    SarvamSTTService,
    SarvamSTTConfigurationError,
    SarvamSTTAPIError,
)
from services.stt import DeepgramSTTService, DeepgramAPIError
from services.sarvam_tts import (
    SarvamTTSService,
    SarvamTTSConfigurationError,
    SarvamTTSAPIError,
)
from services.tts import DeepgramTTSService, DeepgramTTSAPIError

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"


class TelegramBotError(Exception):
    """Base exception for Telegram Bot operations."""
    pass


class TelegramBotConfigurationError(TelegramBotError):
    """Raised when TELEGRAM_BOT_TOKEN is not configured."""
    pass


class TelegramBotAPIError(TelegramBotError):
    """Raised when Telegram Bot API returns an HTTP error."""
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Telegram API error (HTTP {status_code}): {detail}")


class VoiceCopilotBot:
    """
    Module 1 & 2 Telegram Voice & Sales Copilot Bot.
    Enables mobile customers and field agents to interact via text or voice notes.
    Collects structured leads incrementally across multi-turn conversations,
    never re-asking for provided fields, and commits complete leads to PostgreSQL CRM.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        rag_service: Optional[RAGService] = None,
        sarvam_stt: Optional[SarvamSTTService] = None,
        deepgram_stt: Optional[DeepgramSTTService] = None,
        sarvam_tts: Optional[SarvamTTSService] = None,
        deepgram_tts: Optional[DeepgramTTSService] = None,
        lead_extractor: Optional[LeadExtractorService] = None,
        lead_repository: Optional[LeadRepository] = None,
        client: Optional[httpx.AsyncClient] = None,
    ):
        if bot_token is not None:
            self.bot_token = bot_token.strip()
        else:
            self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=5.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50, keepalive_expiry=60.0),
        )

        # Injected or lazily-initialized services
        self._rag_service = rag_service
        self._sarvam_stt = sarvam_stt
        self._deepgram_stt = deepgram_stt
        self._sarvam_tts = sarvam_tts
        self._deepgram_tts = deepgram_tts
        self._lead_extractor = lead_extractor
        self._lead_repository = lead_repository
        self._lead_sessions: Dict[str, Dict[str, Any]] = {}
        self._last_update_id: int = 0
        self._polling_task: Optional[asyncio.Task] = None

    @property
    def is_configured(self) -> bool:
        """Returns True if a non-empty Telegram bot token is configured."""
        return bool(self.bot_token)

    def _require_token(self) -> None:
        if not self.bot_token:
            raise TelegramBotConfigurationError(
                "TELEGRAM_BOT_TOKEN is not configured in backend/.env. "
                "Please set TELEGRAM_BOT_TOKEN to enable the Telegram Bot integration."
            )

    @property
    def api_url(self) -> str:
        self._require_token()
        return f"{TELEGRAM_API_BASE}/bot{self.bot_token}"

    @property
    def file_base_url(self) -> str:
        self._require_token()
        return f"{TELEGRAM_API_BASE}/file/bot{self.bot_token}"

    # ------------------------------------------------------------------------
    # Service Initializers & Session Management
    # ------------------------------------------------------------------------

    def get_lead_extractor(self) -> LeadExtractorService:
        if self._lead_extractor is None:
            self._lead_extractor = LeadExtractorService()
        return self._lead_extractor

    def get_lead_repository(self) -> LeadRepository:
        if self._lead_repository is None:
            self._lead_repository = LeadRepository()
            try:
                self._lead_repository.ensure_leads_table()
            except Exception as e:
                logger.warning(f"[TelegramBot] ensure_leads_table warning: {e}")
        return self._lead_repository

    def get_lead_session(self, chat_id: Union[int, str]) -> Dict[str, Any]:
        """Returns or initializes the multi-turn lead collection session for a given chat ID."""
        key = str(chat_id)
        if key not in self._lead_sessions:
            self._lead_sessions[key] = {
                "lead": {},
                "language": "en",
                "lead_id": None,
                "is_complete": False,
            }
        return self._lead_sessions[key]

    def reset_lead_session(self, chat_id: Union[int, str]) -> Dict[str, Any]:
        """Resets the multi-turn lead collection session for a fresh application."""
        key = str(chat_id)
        self._lead_sessions[key] = {
            "lead": {},
            "language": "en",
            "lead_id": None,
            "is_complete": False,
        }
        return self._lead_sessions[key]

    def get_rag_service(self) -> RAGService:
        if self._rag_service is None:
            namespace = os.getenv("PINECONE_NAMESPACE", "sales_playbooks")
            pinecone_service = PineconeService(namespace=namespace)
            retriever = VectorRetriever(pinecone_service=pinecone_service)
            self._rag_service = RAGService(retriever=retriever)
        return self._rag_service

    def get_sarvam_stt(self) -> SarvamSTTService:
        if self._sarvam_stt is None:
            self._sarvam_stt = SarvamSTTService()
        return self._sarvam_stt

    def get_deepgram_stt(self) -> DeepgramSTTService:
        if self._deepgram_stt is None:
            self._deepgram_stt = DeepgramSTTService()
        return self._deepgram_stt

    def get_sarvam_tts(self) -> SarvamTTSService:
        if self._sarvam_tts is None:
            self._sarvam_tts = SarvamTTSService()
        return self._sarvam_tts

    def get_deepgram_tts(self) -> DeepgramTTSService:
        if self._deepgram_tts is None:
            self._deepgram_tts = DeepgramTTSService()
        return self._deepgram_tts

    # ------------------------------------------------------------------------
    # Telegram API Methods
    # ------------------------------------------------------------------------

    async def send_chat_action(self, chat_id: Union[int, str], action: str = "typing") -> bool:
        """
        Informs the user that the bot is working (typing, record_voice, etc.).
        """
        if not self.is_configured:
            return False
        try:
            url = f"{self.api_url}/sendChatAction"
            resp = await self.client.post(url, json={"chat_id": chat_id, "action": action})
            return resp.status_code == 200
        except Exception as e:
            logger.debug(f"[TelegramBot] send_chat_action failed ({action}): {e}")
            return False

    async def _maintain_chat_action(
        self,
        chat_id: Union[int, str],
        action: str = "typing",
        interval: float = 4.0,
    ) -> None:
        """
        Sends an immediate chat action and periodically refreshes it every `interval` seconds
        until cancelled. Telegram chat actions automatically expire after 5 seconds.
        """
        try:
            while True:
                await self.send_chat_action(chat_id, action)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"[TelegramBot] Chat action heartbeat error: {e}")

    async def send_message(
        self,
        chat_id: Union[int, str],
        text: str,
        parse_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Sends a clean plain text message to the specified Telegram chat.
        Automatically retries without Markdown if Telegram rejects unescaped entities.
        """
        self._require_token()
        url = f"{self.api_url}/sendMessage"
        payload: Dict[str, Any] = {"chat_id": chat_id, "text": text}
        if parse_mode:
            payload["parse_mode"] = parse_mode

        response = await self.client.post(url, json=payload)
        if response.status_code != 200:
            err_text = response.text
            # If Markdown parsing failed, retry plain text
            if parse_mode and ("can't parse entities" in err_text.lower() or response.status_code == 400):
                logger.warning(f"[TelegramBot] Markdown parse failed for chat {chat_id}. Retrying plain text.")
                payload.pop("parse_mode", None)
                retry_resp = await self.client.post(url, json=payload)
                if retry_resp.status_code == 200:
                    return retry_resp.json()
            raise TelegramBotAPIError(response.status_code, err_text)

        return response.json()

    @staticmethod
    def _fit_telegram_caption(text: str, max_len: int = 1020) -> str:
        """
        Safely trims long messages to stay strictly within Telegram's 1024-char caption limit
        without breaking words or cutting sentences awkwardly.
        """
        if len(text) <= max_len:
            return text
        truncated = text[:max_len]
        last_newline = truncated.rfind("\n")
        if last_newline > max_len // 2:
            return truncated[:last_newline].strip()
        last_period = truncated.rfind(". ")
        if last_period > max_len // 2:
            return truncated[:last_period + 1].strip()
        return truncated.rstrip() + "..."

    @classmethod
    def _clean_plain_text(cls, text: str) -> str:
        """
        Formats LLM/RAG responses into clean, beautifully aligned, mobile-friendly text:
        1. Formats answers point-wise with standard bullet points (•) or numbers (1., 2.) where appropriate.
        2. Aligns text cleanly without ragged indentation or awkward spacing on mobile screens.
        3. Removes all Markdown symbols (*, _, `, #, >, ~, etc.).
        4. Removes technical labels (Answer:, Response:, Copilot:, Sources:, etc.) and provider names.
        5. Fits perfectly inside a single Telegram response container.
        """
        if not text:
            return ""

        cleaned = text

        # 1. Remove HTML tags (<br> -> newline, other tags stripped)
        cleaned = re.sub(r"<br\s*/?>", "\n", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<[^>]+>", "", cleaned)

        # 2. Remove Markdown links [text](url) -> text, images ![alt](url) -> ""
        cleaned = re.sub(r"!\[[^\]]*\]\([^\)]+\)", "", cleaned)
        cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)

        # 3. Remove code blocks and inline code
        cleaned = re.sub(r"```[\s\S]*?```", "", cleaned)
        cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
        cleaned = cleaned.replace("`", "")

        # 4. Remove blockquotes and horizontal rules
        cleaned = re.sub(r"^>\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"^[ \t]*[-*_]{3,}[ \t]*$", "", cleaned, flags=re.MULTILINE)

        # 5. Remove header hashes at beginning of lines
        cleaned = re.sub(r"^#+\s*", "", cleaned, flags=re.MULTILINE)

        # 6. Remove standalone source citation lines completely
        cleaned = re.sub(
            r"^[ \t]*\[?(?:source|sources|document|playbook|reference|confidence|latency|model):\s*[^\]\n]+\]?[ \t]*$",
            "",
            cleaned,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        # 7. Remove common technical response prefixes/labels at start of lines
        cleaned = re.sub(
            r"^[ \t]*(?:answer|response|copilot|playbook\s*answer|knowledge\s*assistant|query|question|transcript|user\s*query|disclaimer|note):\s*",
            "",
            cleaned,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        # 8. Remove provider / model names if present
        provider_patterns = [
            r"\bSarvam\s*(?:AI)?\s*(?:Bulbul(?:\s*v\d+)?)?\b",
            r"\bDeepgram\s*(?:Nova-?\d+|Aura)?\b",
            r"\bOpenRouter\b",
            r"\bPinecone\b",
            r"\bsaaras(?::v\d+)?\b",
            r"\bbulbul(?::v\d+)?\b",
        ]
        for pattern in provider_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # 9. Format inline numbering e.g. '1) item, 2) item' or '(1) item (2) item'
        def _format_inline_num(m: re.Match) -> str:
            sep = m.group(1) or ""
            keep_colon = ":" if ":" in sep else ""
            num = re.sub(r"\D", "", m.group(2))
            return f"{keep_colon}\n{num}. "

        cleaned = re.sub(
            r"([\.\:\;\,\n]\s*|^)([1-9]|\([1-9]\))[\)\.]\s+",
            _format_inline_num,
            cleaned,
        )

        # 10. Format semicolon-separated list after colon into bullet points (if 2+ semicolons)
        lines = cleaned.split("\n")
        processed_lines = []
        for line in lines:
            if ":" in line and line.count(";") >= 2:
                intro, rest = line.split(":", 1)
                processed_lines.append(intro.strip() + ":")
                for part in rest.split(";"):
                    part = part.strip()
                    if part:
                        processed_lines.append(f"• {part}")
            else:
                processed_lines.append(line)
        cleaned = "\n".join(processed_lines)

        # 11. Convert bullet points (*, -, +, •) to clean bullet •
        cleaned = re.sub(r"^[ \t]*[\*\-\+•]\s+", "• ", cleaned, flags=re.MULTILINE)

        # 12. Format numbered points cleanly at line starts: '1. ', '2. '
        cleaned = re.sub(r"^[ \t]*(\d+)[\.\)]\s+", r"\1. ", cleaned, flags=re.MULTILINE)

        # 13. Remove bold/italic markdown markers and strikethroughs
        cleaned = cleaned.replace("**", "").replace("__", "")
        cleaned = cleaned.replace("*", "").replace("_", "").replace("~~", "")

        # 14. Clean lines and whitespace alignment for mobile screens
        result_lines = []
        for line in cleaned.split("\n"):
            l = line.strip()
            if l:
                # Strip trailing comma if it was left from an inline numbered list
                if (l.startswith("• ") or re.match(r"^\d+\.\s", l)) and l.endswith(","):
                    l = l[:-1].strip()
                # Normalize multiple horizontal spaces/tabs within line
                l = re.sub(r"[ \t]+", " ", l)
                result_lines.append(l)
            else:
                if result_lines and result_lines[-1] != "":
                    result_lines.append("")

        # 15. Format list grouping and vertical spacing:
        # Keep list items adjacent (no blank lines between items), but separate intro/outro text
        final_lines = []
        for line in result_lines:
            is_bullet = line.startswith("• ") or bool(re.match(r"^\d+\.\s", line))
            if is_bullet:
                if final_lines and final_lines[-1] != "" and not (final_lines[-1].startswith("• ") or bool(re.match(r"^\d+\.\s", final_lines[-1]))):
                    final_lines.append("")
                final_lines.append(line)
            else:
                if final_lines and (final_lines[-1].startswith("• ") or bool(re.match(r"^\d+\.\s", final_lines[-1]))) and line != "":
                    final_lines.append("")
                final_lines.append(line)

        formatted = "\n".join(final_lines)
        formatted = re.sub(r"\n{3,}", "\n\n", formatted)

        # 16. Ensure length fits cleanly into Telegram container without truncation issues
        return cls._fit_telegram_caption(formatted.strip())

    @staticmethod
    def _convert_to_voice_ogg(audio_bytes: bytes) -> Tuple[bytes, str, str]:
        """
        Converts synthesized audio bytes (WAV, MP3, etc.) to Telegram-native Opus-in-OGG.
        Telegram sendVoice displays this format as an interactive voice message bubble with waveform.
        Returns: (converted_bytes, filename, mime_type)
        """
        if not audio_bytes:
            return b"", "voice_answer.ogg", "audio/ogg"

        # 1. Already an OGG container
        if audio_bytes.startswith(b"OggS"):
            return audio_bytes, "voice_answer.ogg", "audio/ogg"

        # 2. Convert using ffmpeg (via imageio_ffmpeg or system ffmpeg)
        try:
            ffmpeg_exe = None
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                import shutil
                ffmpeg_exe = shutil.which("ffmpeg")

            if ffmpeg_exe:
                proc = subprocess.run(
                    [
                        ffmpeg_exe, "-y",
                        "-i", "pipe:0",
                        "-c:a", "libopus",
                        "-b:a", "32k",
                        "-ac", "1",
                        "-ar", "48000",
                        "-f", "ogg",
                        "pipe:1",
                    ],
                    input=audio_bytes,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                )
                if proc.stdout and proc.stdout.startswith(b"OggS"):
                    return proc.stdout, "voice_answer.ogg", "audio/ogg"
        except Exception as conv_err:
            logger.warning(f"[TelegramBot] Conversion to Telegram Opus OGG failed: {conv_err}")

        # 3. If already MP3
        if audio_bytes.startswith(b"ID3") or audio_bytes.startswith(b"\xff\xfb") or audio_bytes.startswith(b"\xff\xf3"):
            return audio_bytes, "voice_answer.mp3", "audio/mpeg"

        # 4. Fallback: default to OGG filename
        return audio_bytes, "voice_answer.ogg", "audio/ogg"

    async def send_voice(
        self,
        chat_id: Union[int, str],
        audio_bytes: bytes,
        caption: Optional[str] = None,
        parse_mode: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Sends a playable voice note to Telegram.
        Automatically formats audio as Opus-encoded OGG for native Telegram voice bubble display.
        If sendVoice fails, falls back gracefully to sendAudio.
        """
        self._require_token()

        # Format audio for Telegram voice message (Opus in OGG)
        converted_bytes, default_filename, mime_type = await asyncio.to_thread(
            self._convert_to_voice_ogg, audio_bytes
        )
        final_filename = filename or default_filename

        voice_url = f"{self.api_url}/sendVoice"
        files = {"voice": (final_filename, converted_bytes, mime_type)}
        data: Dict[str, Any] = {"chat_id": str(chat_id)}
        if caption:
            data["caption"] = caption[:1024]
            if parse_mode:
                data["parse_mode"] = parse_mode

        try:
            resp = await self.client.post(voice_url, data=data, files=files)
            if resp.status_code == 200:
                return resp.json()

            # If caption markdown caused 400 error, retry without parse_mode
            if caption and resp.status_code == 400 and ("parse" in resp.text.lower() or "entity" in resp.text.lower()):
                data.pop("parse_mode", None)
                retry_resp = await self.client.post(voice_url, data=data, files=files)
                if retry_resp.status_code == 200:
                    return retry_resp.json()
                resp = retry_resp

            logger.warning(f"[TelegramBot] sendVoice returned HTTP {resp.status_code}: {resp.text}. Falling back to sendAudio.")
        except Exception as e:
            logger.warning(f"[TelegramBot] sendVoice error: {e}. Falling back to sendAudio.")

        # Fallback: sendAudio
        audio_url = f"{self.api_url}/sendAudio"
        fallback_filename = "voice_answer.mp3" if "mp3" in final_filename else final_filename
        files = {"audio": (fallback_filename, converted_bytes, mime_type)}
        audio_data: Dict[str, Any] = {"chat_id": str(chat_id)}
        if caption:
            audio_data["caption"] = caption[:1024]
            if parse_mode:
                audio_data["parse_mode"] = parse_mode
        resp = await self.client.post(audio_url, data=audio_data, files=files)
        if resp.status_code != 200:
            raise TelegramBotAPIError(resp.status_code, resp.text)
        return resp.json()

    async def download_file(self, file_id: str) -> Tuple[bytes, str]:
        """
        Downloads a binary file from Telegram by file_id.
        Returns: (file_bytes, mime_or_ext)
        """
        self._require_token()
        get_file_url = f"{self.api_url}/getFile"
        resp = await self.client.get(get_file_url, params={"file_id": file_id})
        if resp.status_code != 200:
            raise TelegramBotAPIError(resp.status_code, resp.text)

        file_info = resp.json().get("result", {})
        file_path = file_info.get("file_path")
        if not file_path:
            raise TelegramBotError(f"No file_path returned by Telegram for file_id: {file_id}")

        download_url = f"{self.file_base_url}/{file_path}"
        file_resp = await self.client.get(download_url)
        if file_resp.status_code != 200:
            raise TelegramBotAPIError(file_resp.status_code, file_resp.text)

        file_bytes = file_resp.content
        ext = file_path.split(".")[-1].lower() if "." in file_path else "ogg"
        return file_bytes, ext

    # ------------------------------------------------------------------------
    # Voice & Knowledge Copilot Pipeline
    # ------------------------------------------------------------------------

    async def _synthesize_voice_answer(self, text: str, detected_lang: str) -> Optional[bytes]:
        """
        Synthesizes speech audio using Sarvam Bulbul v3 ('simran') with automatic
        fallback to Deepgram Aura.
        """
        clean_text = (text or "").strip()
        if not clean_text:
            return None

        # Map language and female voice 'simran' for Sarvam Bulbul v3
        if detected_lang == LANG_HI:
            sarvam_lang = "hi-IN"
            speaker = os.getenv("SARVAM_HINDI_VOICE", "simran").strip().lower() or "simran"
        elif detected_lang == LANG_MR:
            sarvam_lang = "mr-IN"
            speaker = os.getenv("SARVAM_MARATHI_VOICE", "simran").strip().lower() or "simran"
        else:
            sarvam_lang = "en-IN"
            speaker = os.getenv("SARVAM_ENGLISH_VOICE", "simran").strip().lower() or "simran"

        # 1. Primary: Sarvam Bulbul v3
        try:
            sarvam_tts = self.get_sarvam_tts()
            return await sarvam_tts.synthesize_speech(
                clean_text,
                language_code=sarvam_lang,
                speaker=speaker,
                model="bulbul:v3",
            )
        except Exception as sarvam_err:
            logger.warning(f"[TelegramBot] Sarvam TTS synthesis failed ({sarvam_err}). Falling back to Deepgram.")

        # 2. Fallback: Deepgram Aura
        try:
            deepgram_tts = self.get_deepgram_tts()
            return await deepgram_tts.synthesize_speech(
                clean_text,
                language=detected_lang,
            )
        except Exception as dg_err:
            logger.error(f"[TelegramBot] Deepgram TTS fallback failed ({dg_err}).")
            return None

    async def _process_lead_turn(
        self,
        chat_id: Union[int, str],
        transcript: str,
        query_type: str = "text",
        forced_lang: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Processes an incoming customer turn (text or transcribed voice note) for Module 1 Lead Collection:
        1. Retrieves accumulated session lead details for this chat_id.
        2. Detects language (English, Hindi, or Marathi).
        3. Calls Module 1 LeadExtractorService to extract all mentioned fields.
        4. Merges with previously provided details (never forgets, never asks for already provided fields).
        5. Identifies missing required fields (name, phone, loan_type, loan_amount).
        6. If all required fields are collected:
           - Saves the complete lead to PostgreSQL database + CRM via LeadRepository.
           - Sets short thank you message.
        7. If any required fields are missing:
           - Asks ONLY for the next missing required field.
        8. Synthesizes voice note with the exact same answer (Sarvam Bulbul v3 / Deepgram Aura).
        9. Sends response inside a single clean container (voice bubble with text caption).
        """
        clean_text = (transcript or "").strip()
        if not clean_text:
            return {"status": "ignored", "reason": "empty_text"}

        # 1. Start continuous typing action heartbeat
        typing_task = asyncio.create_task(self._maintain_chat_action(chat_id, "typing"))

        session = self.get_lead_session(chat_id)
        # If previous application was completed and customer initiates a new conversation
        if session.get("is_complete"):
            session = self.reset_lead_session(chat_id)

        existing_lead = session.get("lead") or {}

        # Resolve language
        detected_lang = forced_lang
        if not detected_lang or detected_lang == "unknown":
            if re.match(r"^[\d\s\+\-\,\.]+$", clean_text) and session.get("language"):
                detected_lang = session.get("language")
            else:
                detected_lang = detect_language(clean_text)
                if session.get("language") in (LANG_HI, LANG_MR) and not re.search(r"[a-zA-Z]{3,}", clean_text):
                    detected_lang = session.get("language")
        if detected_lang not in (LANG_EN, LANG_HI, LANG_MR):
            detected_lang = session.get("language") or LANG_EN
        session["language"] = detected_lang

        # 2. Extract lead info via existing Module 1 LeadExtractorService
        lead_extractor = self.get_lead_extractor()
        try:
            extract_result = await asyncio.to_thread(
                lead_extractor.extract_lead,
                transcript=clean_text,
                existing_lead=existing_lead,
                language=detected_lang,
            )
        except Exception as extract_err:
            logger.error(f"[TelegramBot] Lead extraction error for chat {chat_id}: {extract_err}")
            next_p = get_next_missing_parameter(existing_lead)
            extract_result = {
                "status": "error",
                "lead": existing_lead,
                "next_missing_parameter": next_p,
                "is_complete": next_p is None,
                "message": get_missing_parameter_prompt(next_p, existing_lead, lang=detected_lang),
            }

        final_lead = extract_result.get("lead") or existing_lead
        next_missing = extract_result.get("next_missing_parameter")
        is_complete = bool(extract_result.get("is_complete", False))
        response_message = extract_result.get("message", "")

        # Update session with accumulated lead data (remember previously provided details)
        session["lead"] = final_lead

        # 3. Once all required fields are collected, save complete lead to existing CRM + database
        saved_record = None
        if is_complete and not session.get("is_complete"):
            try:
                repo = self.get_lead_repository()
                lead_obj = Lead.model_validate(final_lead)
                if session.get("lead_id"):
                    saved_record = await asyncio.to_thread(repo.update_lead, session["lead_id"], lead_obj)
                else:
                    saved_record = await asyncio.to_thread(repo.create_lead, lead_obj)
                    if saved_record and "id" in saved_record:
                        session["lead_id"] = saved_record["id"]
                session["is_complete"] = True
                if saved_record and isinstance(saved_record, dict):
                    merged_lead = {**final_lead, **saved_record}
                    session["lead"] = merged_lead
                    final_lead = merged_lead
                logger.info(f"[TelegramBot] Complete lead successfully saved to CRM (id={session.get('lead_id')})")
            except Exception as db_err:
                logger.error(f"[TelegramBot] Error saving complete lead to database: {db_err}")

        # Cancel typing heartbeat
        typing_task.cancel()

        # 4. Clean formatting for mobile response (no Markdown symbols, no technical labels)
        plain_answer = self._clean_plain_text(response_message)

        # 5. Synthesize voice note and deliver single clean response container
        has_voice = False
        voice_task = asyncio.create_task(self._maintain_chat_action(chat_id, "record_voice"))
        try:
            audio_bytes = await self._synthesize_voice_answer(plain_answer, detected_lang)
            if audio_bytes:
                await self.send_voice(
                    chat_id=chat_id,
                    audio_bytes=audio_bytes,
                    caption=plain_answer,
                    parse_mode=None,
                    filename="voice_answer.ogg",
                )
                has_voice = True
            else:
                await self.send_message(chat_id, plain_answer, parse_mode=None)
        except Exception as send_err:
            logger.warning(f"[TelegramBot] Failed to send voice container: {send_err}. Falling back to text message.")
            try:
                await self.send_message(chat_id, plain_answer, parse_mode=None)
            except Exception as fallback_err:
                logger.error(f"[TelegramBot] Failed fallback text message: {fallback_err}")
        finally:
            voice_task.cancel()

        return {
            "status": "success",
            "chat_id": chat_id,
            "query_type": query_type,
            "query": clean_text,
            "answer": plain_answer,
            "module": "module1",
            "lead": final_lead,
            "next_missing_parameter": next_missing,
            "is_complete": is_complete,
            "lead_id": session.get("lead_id"),
            "language": detected_lang,
            "has_voice": has_voice,
        }

    # ------------------------------------------------------------------------
    # Module 2: Knowledge Assistant (Playbook RAG) Turn Processing
    # ------------------------------------------------------------------------

    async def _process_knowledge_turn(
        self,
        chat_id: Union[int, str],
        question: str,
        query_type: str = "text",
        forced_lang: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Processes an incoming customer question for Module 2 Knowledge Assistant (RAG):
        1. Queries Pinecone vector store + OpenRouter LLM via RAGService.
        2. Formats response cleanly for mobile (no markdown symbols, technical labels stripped, point-wise).
        3. Synthesizes voice note with exact same answer (Sarvam Bulbul / Deepgram Aura).
        4. Delivers single response container (voice note with clean caption, or text fallback).
        5. NEVER asks for customer name or lead parameters.
        """
        clean_question = (question or "").strip()
        if not clean_question:
            return {"status": "ignored", "reason": "empty_text"}

        typing_task = asyncio.create_task(self._maintain_chat_action(chat_id, "typing"))

        # Detect or assign language
        detected_lang = forced_lang
        if not detected_lang or detected_lang == "unknown":
            detected_lang = detect_language(clean_question)
        if detected_lang not in (LANG_EN, LANG_HI, LANG_MR):
            detected_lang = LANG_EN

        # 1. Query Grounded Knowledge RAG Service
        rag_service = self.get_rag_service()
        try:
            rag_result = await asyncio.to_thread(
                rag_service.answer_question,
                question=clean_question,
                language=detected_lang,
            )
            raw_answer = rag_result.get("answer", "")
        except Exception as rag_err:
            logger.error(f"[TelegramBot] RAG query error for chat {chat_id}: {rag_err}")
            raw_answer = "I am sorry, but I am unable to access the sales knowledge base right now. Please try asking again in a moment."
            rag_result = {"sources": [], "fallback_used": True}

        typing_task.cancel()

        # 2. Clean formatting for mobile response
        plain_answer = self._clean_plain_text(raw_answer)

        # 3. Synthesize voice note and deliver single clean response container
        has_voice = False
        voice_task = asyncio.create_task(self._maintain_chat_action(chat_id, "record_voice"))
        try:
            audio_bytes = await self._synthesize_voice_answer(plain_answer, detected_lang)
            if audio_bytes:
                await self.send_voice(
                    chat_id=chat_id,
                    audio_bytes=audio_bytes,
                    caption=plain_answer,
                    parse_mode=None,
                    filename="voice_answer.ogg",
                )
                has_voice = True
            else:
                await self.send_message(chat_id, plain_answer, parse_mode=None)
        except Exception as send_err:
            logger.warning(f"[TelegramBot] Failed to send voice container: {send_err}. Falling back to text message.")
            try:
                await self.send_message(chat_id, plain_answer, parse_mode=None)
            except Exception as fallback_err:
                logger.error(f"[TelegramBot] Failed fallback text message: {fallback_err}")
        finally:
            voice_task.cancel()

        return {
            "status": "success",
            "chat_id": chat_id,
            "query_type": query_type,
            "query": clean_question,
            "answer": plain_answer,
            "module": "module2",
            "sources": rag_result.get("sources", []),
            "language": detected_lang,
            "has_voice": has_voice,
        }

    # ------------------------------------------------------------------------
    # Intent Classification (Module 1 Lead Collection vs. Module 2 RAG)
    # ------------------------------------------------------------------------

    @classmethod
    def detect_intent(cls, text: str, session: Optional[Dict[str, Any]] = None) -> str:
        """
        Determines whether the user input is a Knowledge Question (Module 2 RAG)
        or a Loan Application / Lead Collection turn (Module 1).
        """
        if not text or not text.strip():
            return "lead"

        raw = text.strip()
        lower = raw.lower()

        # 1. Explicit Lead Data Indicators (takes precedence if user is providing their details)
        # Check if user is introducing their name
        name_intro_patterns = [
            r"\b(?:my name is|i am|this is|i'm|myself)\s+[a-zA-Z\s]+",
            r"\b(?:मेरा नाम|माझे नाव)\s+[\u0900-\u097F\s]+",
        ]
        for pat in name_intro_patterns:
            if re.search(pat, lower, flags=re.IGNORECASE):
                return "lead"

        # Check for 10-digit Indian phone numbers
        if re.search(r"\b(?:\+?91[\-\s]?)?[6-9]\d{9}\b", raw):
            return "lead"

        # Check for standalone numbers (phone numbers or loan amounts)
        # e.g. "9876543210", "500000", "5,00,000", "10 lakh", "25 lac"
        standalone_amount_or_phone = re.match(r"^\s*(?:rs\.?|inr|₹)?\s*\d[\d,\.]*\s*(?:lakhs?|lac|crore|cr|k|thousand)?\s*$", lower)
        if standalone_amount_or_phone:
            return "lead"

        # Explicit application triggers (without question words)
        apply_triggers = [
            r"\b(?:apply for|i want to apply|want to apply|start application|submit application|book loan)\b",
            r"\b(?:लोन अप्लाई|आवेदन करना|अर्ज करायचा)\b",
        ]
        has_apply_trigger = any(re.search(pat, lower) for pat in apply_triggers)

        # 2. Knowledge / Inquiry Indicators (Module 2 RAG)
        # Check for question mark
        ends_with_qmark = raw.endswith("?")

        # Question prefixes and starters
        knowledge_starters = [
            "what", "how", "why", "when", "where", "which", "who", "whom", "whose",
            "can i", "could you", "tell me", "explain", "describe", "is it", "are there",
            "do you", "does it", "will i", "should i", "would you",
            "क्या", "कैसे", "क्यों", "कब", "कहाँ", "कौन", "बताइए", "बताओ", "समझाओ",
            "काय", "कसे", "का", "केव्हा", "कुठे", "कोण", "सांगा", "समजवा",
        ]
        starts_with_question = any(lower.startswith(q) for q in knowledge_starters)

        # Knowledge domain keywords
        knowledge_keywords = [
            "services do you provide", "services provided", "what services", "what products",
            "what is", "what are", "what does",
            "interest rate", "interest rates", "rate of interest", "roi", "cibil", "credit score",
            "eligibility", "eligible", "criteria", "documentation", "documents required",
            "documents needed", "process", "procedure", "tenure", "prepayment", "foreclosure",
            "charges", "penalty", "fees", "processing fee", "faq", "difference between",
            "ब्याज दर", "पात्रता", "दस्तावेज", "नियम", "शर्तें", "कागदपत्रे", "व्याजदर", "माहिती",
            "कोणत्या सेवा", "काय सेवा", "सेवाएं",
        ]
        contains_knowledge_kw = any(kw in lower for kw in knowledge_keywords)

        # If it has question marks or question starters or knowledge keywords:
        if (ends_with_qmark or starts_with_question or contains_knowledge_kw) and not has_apply_trigger:
            return "knowledge"

        # 3. Active session continuation
        # If user is in an active lead session and already provided some fields (e.g. name or phone),
        # then short answers like "Personal Loan", "Home Loan", "salaried", "business" are lead fields.
        if session and session.get("lead") and not session.get("is_complete"):
            lead_data = session.get("lead", {})
            if any(lead_data.get(k) for k in ["name", "phone", "loan_type"]):
                # If they didn't ask an explicit question, continue lead collection
                return "lead"

        # 4. If they explicitly asked to apply
        if has_apply_trigger:
            return "lead"

        # 5. Check if it matches loan types alone (e.g. "Personal Loan", "Home Loan", "Business Loan")
        loan_types = ["personal loan", "home loan", "business loan", "car loan", "gold loan", "education loan", "पर्सनल लोन", "होम लोन", "व्यवसाय कर्ज", "गृह कर्ज"]
        if lower in loan_types:
            return "lead"

        # If it's phrased like an inquiry, send to knowledge
        if any(w in lower for w in ["about", "info", "details", "help", "guide"]):
            return "knowledge"

        # Otherwise, default to lead
        return "lead"

    async def process_text_message(self, chat_id: Union[int, str], text: str) -> Dict[str, Any]:
        """
        Handles incoming text messages from Telegram:
        - Commands: /start, /help, /info, /reset, /new
        - Intent Routing:
          - Knowledge Questions ("What services do you provide?", "What is a personal loan?") -> Module 2 RAG
          - Loan Applications ("My name is...", "Apply for loan") -> Module 1 Lead Collection
        """
        clean_text = (text or "").strip()
        if not clean_text:
            return {"status": "ignored", "reason": "empty_text"}

        # Handle Reset Commands
        if clean_text in ("/reset", "/new", "/clear"):
            self.reset_lead_session(chat_id)
            reset_msg = "Your application session has been reset. You can start fresh anytime by sending your details or asking questions."
            await self.send_message(chat_id, reset_msg, parse_mode=None)
            return {"status": "command_handled", "command": clean_text}

        # Handle Start / Help Commands
        if clean_text in ("/start", "/help", "/info"):
            self.reset_lead_session(chat_id)
            welcome_msg = (
                "Welcome to Voice Sales Copilot!\n\n"
                "I am your AI Loan & Sales Assistant. You can ask me any question about our loan products, interest rates, and eligibility, or apply for a loan directly.\n\n"
                "Feel free to speak or type in English, Hindi, or Marathi!"
            )
            await self.send_message(chat_id, welcome_msg, parse_mode=None)
            return {"status": "command_handled", "command": clean_text}

        session = self.get_lead_session(chat_id)
        intent = self.detect_intent(clean_text, session=session)

        if intent == "knowledge":
            return await self._process_knowledge_turn(chat_id=chat_id, question=clean_text, query_type="text")
        else:
            return await self._process_lead_turn(chat_id=chat_id, transcript=clean_text, query_type="text")

    async def process_voice_message(
        self,
        chat_id: Union[int, str],
        voice_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handles incoming voice notes (.ogg/opus) from Telegram:
        1. Emits typing action immediately with background heartbeat.
        2. Downloads voice note from Telegram.
        3. Transcribes using Sarvam saaras:v4 (fallback Deepgram Nova-3).
        4. Detects intent (Knowledge Question vs. Loan Application).
        5. Dispatches to Module 2 RAG or Module 1 Lead Collection.
        6. Delivers response with voice note and text caption.
        """
        file_id = voice_data.get("file_id")
        if not file_id:
            logger.error("[TelegramBot] Voice message missing file_id.")
            return {"status": "error", "detail": "missing_file_id"}

        typing_task = asyncio.create_task(self._maintain_chat_action(chat_id, "typing"))

        try:
            audio_bytes, ext = await self.download_file(file_id)
        except Exception as e:
            typing_task.cancel()
            logger.error(f"[TelegramBot] Failed to download voice file: {e}")
            await self.send_message(
                chat_id,
                "⚠️ Sorry, I could not download your voice note. Please try speaking again.",
            )
            return {"status": "error", "detail": str(e)}

        content_type = "audio/ogg" if ext in ("ogg", "oga") else f"audio/{ext}"

        transcript = ""
        detected_stt_lang = "unknown"

        # Primary: Sarvam AI saaras:v4
        try:
            sarvam_stt = self.get_sarvam_stt()
            stt_res = await sarvam_stt.transcribe_audio(
                audio_bytes=audio_bytes,
                content_type=content_type,
                language_code="unknown",
                model="saaras:v4",
                filename="telegram_voice.ogg",
            )
            transcript = (stt_res.get("transcript") or "").strip()
            detected_stt_lang = stt_res.get("detected_language") or "unknown"
            logger.info(f"[TelegramBot] Sarvam STT transcribed: '{transcript}' (lang: {detected_stt_lang})")
        except Exception as sarvam_err:
            logger.warning(f"[TelegramBot] Sarvam STT failed ({sarvam_err}). Falling back to Deepgram Nova-3.")

        # Fallback: Deepgram STT
        if not transcript:
            try:
                deepgram_stt = self.get_deepgram_stt()
                dg_res = await deepgram_stt.transcribe_audio(
                    audio_bytes=audio_bytes,
                    content_type=content_type,
                    model="nova-3",
                    detect_language=True,
                )
                transcript = (dg_res.get("transcript") or "").strip()
                logger.info(f"[TelegramBot] Deepgram STT fallback transcribed: '{transcript}'")
            except Exception as dg_err:
                logger.error(f"[TelegramBot] Deepgram STT fallback failed: {dg_err}")

        typing_task.cancel()

        if not transcript:
            await self.send_message(
                chat_id,
                "🎤 I could not detect any speech in your voice note. Please try speaking closer to the mic.",
            )
            return {"status": "empty_transcript"}

        resolved_lang = detected_stt_lang[:2] if detected_stt_lang not in ("unknown", "") else None
        session = self.get_lead_session(chat_id)
        intent = self.detect_intent(transcript, session=session)

        if intent == "knowledge":
            return await self._process_knowledge_turn(
                chat_id=chat_id,
                question=transcript,
                query_type="voice",
                forced_lang=resolved_lang,
            )
        else:
            return await self._process_lead_turn(
                chat_id=chat_id,
                transcript=transcript,
                query_type="voice",
                forced_lang=resolved_lang,
            )

    async def handle_webhook_update(self, update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for incoming Telegram updates from the webhook endpoint.
        Dispatches to process_voice_message or process_text_message.
        """
        if not update or not isinstance(update, dict):
            return {"status": "ignored", "reason": "invalid_update"}

        message = update.get("message") or update.get("edited_message")
        if not message or not isinstance(message, dict):
            return {"status": "ignored", "reason": "no_message"}

        chat = message.get("chat", {})
        chat_id = chat.get("id")
        if not chat_id:
            return {"status": "ignored", "reason": "no_chat_id"}

        # Voice or Audio note
        voice_data = message.get("voice") or message.get("audio")
        if voice_data and isinstance(voice_data, dict):
            return await self.process_voice_message(chat_id, voice_data)

        # Text message
        text = message.get("text")
        if text:
            return await self.process_text_message(chat_id, text)

        return {"status": "ignored", "reason": "unsupported_message_type"}

    # ------------------------------------------------------------------------
    # Real-Time Background Polling (Zero-Latency Local / Development Mode)
    # ------------------------------------------------------------------------

    async def poll_once(self) -> int:
        """
        Polls Telegram getUpdates for pending messages and dispatches them immediately.
        Advances the offset so processed updates are acknowledged and never re-fetched.
        Returns the number of updates processed.
        """
        if not self.is_configured:
            return 0

        url = f"{self.api_url}/getUpdates"
        params: Dict[str, Any] = {"timeout": 2, "limit": 10}
        if self._last_update_id > 0:
            params["offset"] = self._last_update_id + 1

        try:
            resp = await self.client.get(url, params=params)
            if resp.status_code != 200:
                return 0
            data = resp.json()
            updates = data.get("result", [])
            for u in updates:
                up_id = u.get("update_id", 0)
                if up_id > self._last_update_id:
                    self._last_update_id = up_id
                # Spawn processing task immediately without blocking poll loop
                asyncio.create_task(self.handle_webhook_update(u))
            return len(updates)
        except Exception as e:
            logger.debug(f"[TelegramBot] Polling getUpdates: {e}")
            return 0

    async def start_polling_loop(self) -> None:
        """Runs the continuous polling worker until cancelled."""
        logger.info("[TelegramBot] Background polling worker started for Telegram updates.")
        while True:
            try:
                await self.poll_once()
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[TelegramBot] Polling loop error: {e}")
                await asyncio.sleep(2.0)

    def start_polling_task(self) -> Optional[asyncio.Task]:
        """Starts the background polling task if not already running."""
        if not self.is_configured:
            return None
        if self._polling_task is None or self._polling_task.done():
            self._polling_task = asyncio.create_task(self.start_polling_loop())
            logger.info("[TelegramBot] Polling task created.")
        return self._polling_task

    def stop_polling_task(self) -> None:
        """Stops the background polling task if running."""
        if self._polling_task and not self._polling_task.done():
            self._polling_task.cancel()
            self._polling_task = None
            logger.info("[TelegramBot] Polling task cancelled.")


# Shared singleton instance
_bot_instance: Optional[VoiceCopilotBot] = None


def get_voice_copilot_bot() -> VoiceCopilotBot:
    """Returns the shared VoiceCopilotBot instance."""
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = VoiceCopilotBot()
    return _bot_instance
