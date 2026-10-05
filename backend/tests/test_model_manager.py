import os
import sys
import json
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from backend.main import app
except ImportError:
    from main import app
from services.model_manager import (
    ModelManager,
    infer_provider,
    mask_api_key,
    get_model_manager,
)


@pytest.fixture
def temp_manager(tmp_path):
    config_file = tmp_path / "models_config.json"
    manager = ModelManager(config_path=str(config_file))
    return manager


def test_infer_provider_logic():
    assert infer_provider("stt", "deepgram-nova-3") == "Deepgram"
    assert infer_provider("stt", "nova-2-conversationalai") == "Deepgram"
    assert infer_provider("stt", "sarvam-saaras-v4") == "Sarvam AI"
    assert infer_provider("tts", "sarvam-bulbul-v3") == "Sarvam AI"
    assert infer_provider("tts", "deepgram-aura-asteria") == "Deepgram"
    assert infer_provider("llm", "deepseek/deepseek-chat") == "OpenRouter"
    assert infer_provider("llm", "meta-llama/llama-3.3-70b-instruct") == "OpenRouter"
    assert infer_provider("llm", "openai/gpt-4o-mini") == "OpenRouter"


def test_mask_api_key():
    assert mask_api_key(None) is None
    assert mask_api_key("") is None
    masked = mask_api_key("sk-1234567890abcdef")
    assert masked.startswith("sk-1")
    assert masked.endswith("cdef")
    assert "•" in masked
    assert mask_api_key("short") == "••••••••"


def test_only_one_active_model_per_category(temp_manager):
    # Built-in defaults: STT is nova-3, LLM is deepseek-chat, TTS is sarvam-bulbul-v3
    active_stt = temp_manager.get_active_model("stt")
    assert active_stt is not None
    assert active_stt["id"] == "deepgram-nova-3"

    # Switch to nova-2
    updated = temp_manager.set_active_model("stt", "deepgram-nova-2")
    assert updated["id"] == "deepgram-nova-2"
    assert updated["is_active"] is True

    # Check that previous model is deactivated
    old_model = temp_manager.get_model("deepgram-nova-3")
    assert old_model["is_active"] is False

    # Verify active models map
    active_map = temp_manager.get_active_models()
    assert active_map["stt"]["id"] == "deepgram-nova-2"
    assert active_map["stt"]["is_active"] is True


def test_add_custom_model_using_only_category_name_key(temp_manager):
    # User adds custom LLM with only category, name, and api_key
    custom_model = temp_manager.add_model(
        category="llm",
        name="meta-llama/llama-3.3-70b-instruct",
        api_key="mock_key_openrouter_test_1234",
        set_active=True,
    )

    assert custom_model["name"] == "meta-llama/llama-3.3-70b-instruct"
    assert custom_model["category"] == "llm"
    assert custom_model["provider"] == "OpenRouter"
    assert custom_model["has_api_key"] is True
    assert custom_model["is_active"] is True
    assert custom_model["api_key_masked"].startswith("mock")
    assert custom_model["api_key_masked"].endswith("1234")

    # Exactly one active model for LLM: previous deepseek-chat should be deactivated
    deepseek = temp_manager.get_model("deepseek-chat")
    assert deepseek["is_active"] is False

    # Internal service unmasked retrieval returns raw key
    active_unmasked = temp_manager.get_active_model("llm")
    assert active_unmasked["api_key"] == "mock_key_openrouter_test_1234"

    # Client-facing list masks keys
    all_models = temp_manager.get_all_models(category="llm")
    custom_in_list = next(m for m in all_models if m["id"] == custom_model["id"])
    assert custom_in_list["api_key_masked"].startswith("mock")
    assert custom_in_list.get("api_key") != "mock_key_openrouter_test_1234"


def test_delete_inactive_custom_model_only(temp_manager):
    custom = temp_manager.add_model(category="tts", name="custom-safe-delete")
    active_before = temp_manager.get_active_model_id("tts")

    assert temp_manager.delete_model(custom["id"]) is True
    assert temp_manager.get_model(custom["id"]) is None
    assert temp_manager.get_active_model_id("tts") == active_before


def test_delete_allows_builtin_and_active_models(temp_manager):
    # Built-in model can be deleted
    assert temp_manager.delete_model("sarvam-bulbul-v3") is True
    assert temp_manager.get_model("sarvam-bulbul-v3") is None

    # Active model can be deleted and clears active selection
    custom = temp_manager.add_model(category="tts", name="custom-active-delete", set_active=True)
    assert temp_manager.get_active_model_id("tts") == custom["model_id"]
    assert temp_manager.delete_model(custom["id"]) is True
    assert temp_manager.get_model(custom["id"]) is None
    assert temp_manager._active_models.get("tts") is None


def test_delete_missing_custom_model_is_rejected_without_state_change(temp_manager):
    active_before = dict(temp_manager._active_models)
    with pytest.raises(ValueError, match="does not exist"):
        temp_manager.delete_model("missing-model")
    assert temp_manager._active_models == active_before


def test_api_models_endpoints():
    client = TestClient(app)

    # 1. GET /api/models
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert len(data["models"]) >= 10

    # 2. GET /api/models/active
    active_res = client.get("/api/models/active")
    assert active_res.status_code == 200
    active_data = active_res.json()
    assert "stt" in active_data["active"]
    assert "llm" in active_data["active"]
    assert "tts" in active_data["active"]

    # 3. POST /api/models - Add custom model with Category, Model Name, API Key ONLY
    create_res = client.post(
        "/api/models",
        json={
            "category": "tts",
            "name": "custom-tts-model",
            "api_key": "test_tts_key_1234567890",
            "is_active": True,
        },
    )
    assert create_res.status_code == 201
    created_model = create_res.json()["model"]
    assert created_model["name"] == "custom-tts-model"
    assert created_model["category"] == "tts"
    assert created_model["has_api_key"] is True
    assert created_model["is_active"] is True

    # Active models can be permanently deleted; active selection for that category is cleared.
    active_delete_res = client.delete(f"/api/models/{created_model['id']}")
    assert active_delete_res.status_code == 200
    assert client.get("/api/models/active").json()["active"]["tts"] is None

    # Missing IDs are reported as 404 distinctly.
    assert client.delete("/api/models/not-a-real-model").status_code == 404

    # Built-in models can also be permanently deleted.
    del_builtin_res = client.delete("/api/models/sarvam-bulbul-v3")
    assert del_builtin_res.status_code == 200
    models_after_del = client.get("/api/models").json()["models"]
    assert not any(m["id"] == "sarvam-bulbul-v3" for m in models_after_del)

    # 4. POST /api/models/restore-defaults - Restore missing built-in models
    restore_res = client.post("/api/models/restore-defaults")
    assert restore_res.status_code == 200
    restored_models = restore_res.json()["models"]
    assert any(m["id"] == "sarvam-bulbul-v3" for m in restored_models)
    # Check that default active model was also restored
    assert client.get("/api/models/active").json()["active"]["tts"]["id"] == "sarvam-bulbul-v3"
