# ============================================================================
# MODEL MANAGER SERVICE (backend/services/model_manager.py)
# ============================================================================
# Manages STT, LLM, and TTS models persistently.
# Allows dynamic switching, adding custom models, editing, deleting, and activating.
# Provides dynamic active model resolution for all backend services.
# ============================================================================

import os
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CONFIG_FILE = DATA_DIR / "models_config.json"

DEFAULT_BUILTIN_MODELS = [
    # ------------------------------------------------------------------------
    # STT (Speech-to-Text) Models
    # ------------------------------------------------------------------------
    {
        "id": "deepgram-nova-3",
        "name": "Deepgram Nova-3",
        "category": "stt",
        "provider": "Deepgram",
        "model_id": "nova-3",
        "description": "Multilingual low-latency STT (English, Hindi, Marathi with domain biasing)",
        "is_builtin": True,
        "is_active": True,
        "configuration": {
            "smart_format": True,
            "punctuate": True,
            "encoding": "linear16",
            "sample_rate": 48000
        },
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z"
    },
    {
        "id": "deepgram-nova-2",
        "name": "Deepgram Nova-2",
        "category": "stt",
        "provider": "Deepgram",
        "model_id": "nova-2",
        "description": "High-accuracy general English and multi-dialect STT",
        "is_builtin": True,
        "is_active": False,
        "configuration": {
            "smart_format": True,
            "punctuate": True,
            "encoding": "linear16",
            "sample_rate": 48000
        },
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z"
    },
    {
        "id": "sarvam-saaras-v4",
        "name": "Sarvam Saaras v4",
        "category": "stt",
        "provider": "Sarvam AI",
        "model_id": "saaras:v4",
        "description": "Dedicated Indian regional language speech-to-text",
        "is_builtin": True,
        "is_active": False,
        "configuration": {
            "language_code": "hi-IN",
            "model": "saaras:v4"
        },
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z"
    },
    {
        "id": "sarvam-saaras-v2",
        "name": "Sarvam Saaras v2",
        "category": "stt",
        "provider": "Sarvam AI",
        "model_id": "saaras:v2",
        "description": "Fast Indic speech recognition model",
        "is_builtin": True,
        "is_active": False,
        "configuration": {
            "language_code": "hi-IN",
            "model": "saaras:v2"
        },
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z"
    },

    # ------------------------------------------------------------------------
    # LLM (Language Model) Models
    # ------------------------------------------------------------------------
    {
        "id": "deepseek-chat",
        "name": "DeepSeek Chat (V3)",
        "category": "llm",
        "provider": "OpenRouter",
        "model_id": "deepseek/deepseek-chat",
        "description": "Ultra-fast response generation & grounded RAG intelligence",
        "is_builtin": True,
        "is_active": True,
        "configuration": {
            "temperature": 0.1,
            "max_tokens": 1024,
            "top_p": 0.95
        },
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z"
    },
    {
        "id": "llama-3.3-70b",
        "name": "Llama 3.3 70B Instruct",
        "category": "llm",
        "provider": "OpenRouter",
        "model_id": "meta-llama/llama-3.3-70b-instruct",
        "description": "Meta's flagship high-capability instruction model for sales objection handling",
        "is_builtin": True,
        "is_active": False,
        "configuration": {
            "temperature": 0.15,
            "max_tokens": 1200,
            "top_p": 0.9
        },
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z"
    },
    {
        "id": "llama-3.1-8b",
        "name": "Llama 3.1 8B Instruct",
        "category": "llm",
        "provider": "OpenRouter",
        "model_id": "meta-llama/llama-3.1-8b-instruct",
        "description": "Ultra-fast low-latency assistant model",
        "is_builtin": True,
        "is_active": False,
        "configuration": {
            "temperature": 0.1,
            "max_tokens": 800
        },
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z"
    },
    {
        "id": "deepseek-r1",
        "name": "DeepSeek R1 (Reasoning)",
        "category": "llm",
        "provider": "OpenRouter",
        "model_id": "deepseek/deepseek-r1",
        "description": "Advanced reasoning model for complex financial & credit inquiries",
        "is_builtin": True,
        "is_active": False,
        "configuration": {
            "temperature": 0.6,
            "max_tokens": 2048
        },
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z"
    },
    {
        "id": "gpt-4o-mini",
        "name": "OpenAI GPT-4o Mini",
        "category": "llm",
        "provider": "OpenRouter",
        "model_id": "openai/gpt-4o-mini",
        "description": "Fast and cost-effective multi-modal intelligence",
        "is_builtin": True,
        "is_active": False,
        "configuration": {
            "temperature": 0.2,
            "max_tokens": 1000
        },
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z"
    },

    # ------------------------------------------------------------------------
    # TTS (Text-to-Speech) Models
    # ------------------------------------------------------------------------
    {
        "id": "sarvam-bulbul-v3",
        "name": "Sarvam Bulbul v3",
        "category": "tts",
        "provider": "Sarvam AI",
        "model_id": "bulbul:v3",
        "description": "State-of-the-art natural Indian speech synthesis (Hindi, Marathi, English)",
        "is_builtin": True,
        "is_active": True,
        "configuration": {
            "default_speaker": "ritu",
            "pace": 1.0,
            "sample_rate": 48000
        },
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z"
    },
    {
        "id": "sarvam-bulbul-v2",
        "name": "Sarvam Bulbul v2",
        "category": "tts",
        "provider": "Sarvam AI",
        "model_id": "bulbul:v2",
        "description": "Standard Indian language voice synthesis",
        "is_builtin": True,
        "is_active": False,
        "configuration": {
            "default_speaker": "simran",
            "pace": 1.0,
            "sample_rate": 48000
        },
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z"
    },
    {
        "id": "deepgram-aura-asteria",
        "name": "Deepgram Aura (Asteria)",
        "category": "tts",
        "provider": "Deepgram",
        "model_id": "aura-asteria-en",
        "description": "Natural conversational English female voice",
        "is_builtin": True,
        "is_active": False,
        "configuration": {
            "voice": "aura-asteria-en",
            "sample_rate": 48000
        },
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z"
    },
    {
        "id": "deepgram-aura-luna",
        "name": "Deepgram Aura (Luna)",
        "category": "tts",
        "provider": "Deepgram",
        "model_id": "aura-luna-en",
        "description": "Warm and engaging conversational English voice",
        "is_builtin": True,
        "is_active": False,
        "configuration": {
            "voice": "aura-luna-en",
            "sample_rate": 48000
        },
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z"
    }
]

DEFAULT_ACTIVE_MODELS = {
    "stt": "deepgram-nova-3",
    "llm": "deepseek-chat",
    "tts": "sarvam-bulbul-v3"
}


def mask_api_key(key: Optional[str]) -> Optional[str]:
    """Masks an API key for safe UI and API presentation."""
    if not key or not str(key).strip():
        return None
    k = str(key).strip()
    if len(k) <= 8:
        return "••••••••"
    return f"{k[:4]}••••{k[-4:]}"


def infer_provider(category: str, name: str, api_key: Optional[str] = None) -> str:
    """Infers the most likely AI provider from model name, category, or API key."""
    n = (name or "").strip().lower()
    k = (api_key or "").strip().lower()
    cat = (category or "").strip().lower()

    if cat == "stt":
        if any(term in n for term in ("nova", "deepgram", "flux")):
            return "Deepgram"
        if any(term in n for term in ("saaras", "sarvam")):
            return "Sarvam AI"
        if "whisper" in n or k.startswith("sk-proj-"):
            return "OpenAI"
        return "Deepgram" if k.startswith("660") or len(k) == 40 else "Custom"

    elif cat == "tts":
        if any(term in n for term in ("bulbul", "sarvam")):
            return "Sarvam AI"
        if any(term in n for term in ("aura", "deepgram", "asteria", "luna", "stella")):
            return "Deepgram"
        if "eleven" in n:
            return "ElevenLabs"
        return "Sarvam AI" if k.startswith("sk_") else "Custom"

    elif cat == "llm":
        if "/" in name or k.startswith("sk-or-"):
            return "OpenRouter"
        if any(term in n for term in ("gpt", "o1", "o3", "chatgpt")) or k.startswith("sk-proj-"):
            return "OpenAI"
        if "claude" in n:
            return "Anthropic"
        if "gemini" in n:
            return "Google"
        return "OpenRouter"

    return "Custom"


class ModelManager:
    """
    Thread-safe persistent manager for STT, LLM, and TTS models.
    Saves configuration to backend/data/models_config.json.
    """

    def __init__(self, config_path: Optional[Any] = None):
        self.config_file = Path(config_path) if config_path else CONFIG_FILE
        self._models: List[Dict[str, Any]] = []
        self._active_models: Dict[str, str] = dict(DEFAULT_ACTIVE_MODELS)
        self._load()

    def _ensure_data_dir(self) -> None:
        self.config_file.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        """Loads models configuration from persistent JSON storage, or initializes defaults."""
        self._ensure_data_dir()
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._models = data.get("models", [])
                    self._active_models = data.get("active_models", dict(DEFAULT_ACTIVE_MODELS))
                    logger.info(f"[ModelManager] Loaded {len(self._models)} models from {self.config_file}")
                    return
            except Exception as e:
                logger.error(f"[ModelManager] Error reading {self.config_file}: {e}. Reinitializing defaults.")

        # Initialize defaults if file doesn't exist or failed to load
        self._models = [dict(m) for m in DEFAULT_BUILTIN_MODELS]
        self._active_models = dict(DEFAULT_ACTIVE_MODELS)
        self._save()

    def _save(self) -> None:
        """Atomically persists models configuration to disk."""
        self._ensure_data_dir()
        temp_file = self.config_file.with_suffix(".tmp")
        payload = {
            "active_models": self._active_models,
            "models": self._models,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            temp_file.replace(self.config_file)
            logger.info(f"[ModelManager] Successfully persisted models configuration ({len(self._models)} models)")
        except Exception as e:
            logger.error(f"[ModelManager] Failed to persist models to {self.config_file}: {e}")
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass

    # ------------------------------------------------------------------------
    # Query Methods
    # ------------------------------------------------------------------------
    def get_all_models(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns all models, optionally filtered by category ('stt', 'llm', 'tts'), with API keys masked."""
        cat_clean = (category or "").strip().lower()
        res = []
        for m in self._models:
            m_copy = dict(m)
            # Sync is_active flag with active_models map
            m_cat = m_copy.get("category", "").lower()
            m_copy["is_active"] = (self._active_models.get(m_cat) == m_copy.get("id"))
            # Mask API key for client-facing exposure
            raw_key = m.get("api_key")
            m_copy["has_api_key"] = bool(raw_key and str(raw_key).strip())
            m_copy["api_key_masked"] = mask_api_key(raw_key)
            if "api_key" in m_copy and m_copy["api_key"]:
                m_copy["api_key"] = mask_api_key(raw_key)

            if not cat_clean or m_cat == cat_clean:
                res.append(m_copy)
        return res

    def list_models(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Alias for get_all_models."""
        return self.get_all_models(category=category)

    def get_model(self, model_id: str, unmasked: bool = False) -> Optional[Dict[str, Any]]:
        """Finds a model by its unique ID key. unmasked=True returns raw api_key for internal services."""
        clean_id = (model_id or "").strip()
        for m in self._models:
            if m.get("id") == clean_id:
                m_copy = dict(m)
                m_cat = m_copy.get("category", "").lower()
                m_copy["is_active"] = (self._active_models.get(m_cat) == m_copy.get("id"))
                raw_key = m.get("api_key")
                m_copy["has_api_key"] = bool(raw_key and str(raw_key).strip())
                m_copy["api_key_masked"] = mask_api_key(raw_key)
                if not unmasked and "api_key" in m_copy and m_copy["api_key"]:
                    m_copy["api_key"] = mask_api_key(raw_key)
                return m_copy
        return None

    def get_active_model(self, category: str) -> Optional[Dict[str, Any]]:
        """Returns the full unmasked model dictionary for the active model in the category, or None if unassigned."""
        cat_clean = category.strip().lower()
        active_key = self._active_models.get(cat_clean)
        if active_key:
            return self.get_model(active_key, unmasked=True)
        return None

    def get_active_model_key(self, category: str) -> Optional[str]:
        """Returns the unmasked API key for the active model in the category, from config or env."""
        active = self.get_active_model(category)
        if active and active.get("api_key") and str(active["api_key"]).strip():
            return str(active["api_key"]).strip()
        if active:
            provider = (active.get("provider") or "").lower()
            name = (active.get("name") or "").lower()
            model_id = (active.get("model_id") or "").lower()
            if "deepgram" in provider or "nova" in name:
                return os.getenv("DEEPGRAM_API_KEY")
            elif "sarvam" in provider or "saaras" in name or "bulbul" in name:
                return os.getenv("SARVAM_API_KEY")
            elif "groq" in provider or "groq" in name or "groq" in model_id:
                return os.getenv("GROQ_API_KEY")
            elif "openrouter" in provider:
                return os.getenv("OPENROUTER_API_KEY")
            elif "openai" in provider:
                return os.getenv("OPENAI_API_KEY")
        return None

    def get_active_models(self) -> Dict[str, Optional[Dict[str, Any]]]:
        """Returns the full model objects for currently active STT, LLM, and TTS models (masked for client)."""
        active: Dict[str, Optional[Dict[str, Any]]] = {}
        for cat in ("stt", "llm", "tts"):
            active_key = self._active_models.get(cat)
            model_obj = self.get_model(active_key, unmasked=False) if active_key else None
            active[cat] = model_obj
        return active

    def get_active_model_id(self, category: str) -> str:
        """
        Returns the exact API model_id string (e.g. 'nova-3', 'deepseek/deepseek-chat', 'bulbul:v3')
        for the currently active model in the given category.
        """
        cat_clean = category.strip().lower()
        active = self.get_active_model(cat_clean)
        if active and active.get("model_id"):
            return active["model_id"]

        # Default fallbacks
        if cat_clean == "stt":
            return "nova-3"
        elif cat_clean == "llm":
            return os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")
        elif cat_clean == "tts":
            return os.getenv("SARVAM_MODEL", "bulbul:v3")
        return ""

    def get_active_model_config(self, category: str) -> Dict[str, Any]:
        """Returns the configuration dict for the active model in the category."""
        active = self.get_active_model(category)
        if active and isinstance(active.get("configuration"), dict):
            return dict(active["configuration"])
        return {}

    # ------------------------------------------------------------------------
    # Mutation Methods
    # ------------------------------------------------------------------------
    def set_active_model(self, category: str, model_id: str) -> Dict[str, Any]:
        """Switches the active model for a category ('stt', 'llm', 'tts'). Activating one deactivates previous."""
        cat_clean = category.strip().lower()
        if cat_clean not in ("stt", "llm", "tts"):
            raise ValueError(f"Invalid category '{category}'. Must be 'stt', 'llm', or 'tts'.")

        target = self.get_model(model_id, unmasked=True)
        if not target:
            # Check by model_id attribute as well
            for m in self._models:
                if m.get("model_id") == model_id and m.get("category", "").lower() == cat_clean:
                    target = m
                    break
        if not target:
            raise ValueError(f"Model '{model_id}' not found in category '{category}'.")

        target_cat = target.get("category", "").lower()
        if target_cat != cat_clean:
            raise ValueError(f"Model '{model_id}' is in category '{target_cat}', not '{cat_clean}'.")

        actual_id = target["id"]
        # Exactly one active model per category
        self._active_models[cat_clean] = actual_id
        for m in self._models:
            if m.get("category", "").lower() == cat_clean:
                m["is_active"] = (m.get("id") == actual_id)
                m["updated_at"] = datetime.now(timezone.utc).isoformat()

        self._save()
        logger.info(f"[ModelManager] Activated {cat_clean.upper()} model: {actual_id} ({target.get('name')})")
        return self.get_model(actual_id, unmasked=False)

    def add_model(
        self,
        category: str,
        name: str,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model_id: Optional[str] = None,
        configuration: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        set_active: bool = False
    ) -> Dict[str, Any]:
        """
        Adds a new custom model using Category, Model Name, and API Key.
        Provider, model_id, and configuration default or infer automatically.
        """
        cat_clean = category.strip().lower()
        if cat_clean not in ("stt", "llm", "tts"):
            raise ValueError(f"Invalid category '{category}'. Must be 'stt', 'llm', or 'tts'.")

        name_clean = (name or "").strip()
        if not name_clean:
            raise ValueError("Model name cannot be empty.")

        api_key_clean = (api_key or "").strip()
        model_id_clean = (model_id or name_clean).strip()
        provider_clean = (provider or infer_provider(cat_clean, name_clean, api_key_clean)).strip()

        # Generate a clean slug id
        base_slug = re.sub(r"[^a-z0-9]+", "-", f"{provider_clean}-{model_id_clean}".lower()).strip("-")
        model_key = f"custom-{base_slug}"
        # Ensure unique id
        existing_ids = {m["id"] for m in self._models}
        if model_key in existing_ids:
            model_key = f"custom-{base_slug}-{uuid.uuid4().hex[:6]}"

        now_str = datetime.now(timezone.utc).isoformat()
        new_model = {
            "id": model_key,
            "name": name_clean,
            "category": cat_clean,
            "provider": provider_clean,
            "model_id": model_id_clean,
            "api_key": api_key_clean,
            "description": (description or f"Custom {cat_clean.upper()} model via {provider_clean}").strip(),
            "is_builtin": False,
            "is_active": False,
            "configuration": configuration if isinstance(configuration, dict) else {},
            "created_at": now_str,
            "updated_at": now_str
        }

        self._models.append(new_model)

        if set_active:
            self._active_models[cat_clean] = model_key
            for m in self._models:
                if m.get("category", "").lower() == cat_clean:
                    m["is_active"] = (m.get("id") == model_key)

        self._save()
        logger.info(f"[ModelManager] Added custom model: {model_key} ({name_clean}, provider={provider_clean})")
        return self.get_model(model_key, unmasked=False)

    def update_model(self, model_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Updates an existing model."""
        target_idx = None
        for idx, m in enumerate(self._models):
            if m.get("id") == model_id:
                target_idx = idx
                break

        if target_idx is None:
            raise ValueError(f"Model with ID '{model_id}' does not exist.")

        target = self._models[target_idx]

        # Built-in models can only have configuration, name, description updated
        if "name" in data and str(data["name"]).strip():
            target["name"] = str(data["name"]).strip()
        if "description" in data:
            target["description"] = str(data["description"]).strip()
        if "configuration" in data and isinstance(data["configuration"], dict):
            target["configuration"] = data["configuration"]

        # Custom models can also update provider, model_id, category, api_key
        if not target.get("is_builtin"):
            if "provider" in data and str(data["provider"]).strip():
                target["provider"] = str(data["provider"]).strip()
            if "model_id" in data and str(data["model_id"]).strip():
                target["model_id"] = str(data["model_id"]).strip()
            if "category" in data and str(data["category"]).strip().lower() in ("stt", "llm", "tts"):
                target["category"] = str(data["category"]).strip().lower()
            if "api_key" in data:
                target["api_key"] = str(data["api_key"]).strip()

        target["updated_at"] = datetime.now(timezone.utc).isoformat()

        if data.get("is_active"):
            cat = target["category"].lower()
            self._active_models[cat] = target["id"]
            for m in self._models:
                if m.get("category", "").lower() == cat:
                    m["is_active"] = (m.get("id") == target["id"])

        self._save()
        logger.info(f"[ModelManager] Updated model: {model_id}")
        return self.get_model(model_id)

    def delete_model(self, model_id: str) -> bool:
        """
        Permanently deletes any model (built-in or custom).
        If the model is currently active, clears the active selection for its category.
        """
        target_idx = None
        for idx, m in enumerate(self._models):
            if m.get("id") == model_id:
                target_idx = idx
                break

        if target_idx is None:
            raise ValueError(f"Model with ID '{model_id}' does not exist.")

        target = self._models[target_idx]
        cat = target.get("category", "").lower()

        # If deleted model was the active engine for its category, clear active model
        if self._active_models.get(cat) == model_id:
            self._active_models[cat] = None
            logger.info(f"[ModelManager] Cleared active model for category '{cat}' due to deletion")

        # Remove from list
        del self._models[target_idx]

        # Ensure no remaining models in category are marked active if active was cleared
        if self._active_models.get(cat) is None:
            for m in self._models:
                if m.get("category", "").lower() == cat:
                    m["is_active"] = False

        self._save()
        logger.info(f"[ModelManager] Permanently deleted model: {model_id} (category: {cat})")
        return True

    def restore_defaults(self) -> List[Dict[str, Any]]:
        """
        Restores missing default built-in models and restores default active models if unassigned.
        """
        existing_ids = {m.get("id") for m in self._models}
        restored_count = 0
        for default_m in DEFAULT_BUILTIN_MODELS:
            if default_m["id"] not in existing_ids:
                restored_copy = dict(default_m)
                cat = restored_copy.get("category", "").lower()
                if not self._active_models.get(cat):
                    if DEFAULT_ACTIVE_MODELS.get(cat) == restored_copy["id"]:
                        self._active_models[cat] = restored_copy["id"]
                        restored_copy["is_active"] = True
                    else:
                        restored_copy["is_active"] = False
                else:
                    restored_copy["is_active"] = False
                self._models.append(restored_copy)
                restored_count += 1

        # Also restore default active model if category still has None and default active model exists in models
        for cat, def_id in DEFAULT_ACTIVE_MODELS.items():
            if not self._active_models.get(cat):
                matching = [m for m in self._models if m.get("id") == def_id]
                if matching:
                    self._active_models[cat] = def_id
                    matching[0]["is_active"] = True

        self._save()
        logger.info(f"[ModelManager] Restored {restored_count} default built-in models")
        return self.list_models()


# Global singleton instance
_model_manager: Optional[ModelManager] = None

def get_model_manager() -> ModelManager:
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager

def get_active_model_id(category: str) -> str:
    """Helper function to dynamically get the active model string for 'stt', 'llm', or 'tts'."""
    return get_model_manager().get_active_model_id(category)

def get_active_model_config(category: str) -> Dict[str, Any]:
    """Helper function to dynamically get the active model configuration dict."""
    return get_model_manager().get_active_model_config(category)

def get_active_model(category: str) -> Optional[Dict[str, Any]]:
    """Helper function to dynamically get the full active model dictionary with unmasked credentials."""
    return get_model_manager().get_active_model(category)

def get_active_model_key(category: str) -> Optional[str]:
    """Helper function to dynamically get the active model API key."""
    return get_model_manager().get_active_model_key(category)
