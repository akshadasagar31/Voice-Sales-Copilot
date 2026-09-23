import os
import sys
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app
from services.language import (
    detect_language,
    is_greeting,
    get_fallback_message,
    get_empty_kb_greeting,
    LANG_EN,
    LANG_HI,
    LANG_MR,
)
from services.rag import RAGService, DEFAULT_FALLBACK_MESSAGE
from services.stt import DeepgramSTTService
from services.tts import DeepgramTTSService


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Unit Tests: Language Detection
# ---------------------------------------------------------------------------

def test_detect_language_english():
    """Verify English text is accurately detected as 'en'."""
    assert detect_language("What are our volume discount tiers for enterprise contracts?") == LANG_EN
    assert detect_language("How do we handle competitor claims regarding 99.99% uptime SLAs?") == LANG_EN
    assert detect_language("Hello, can you help me?") == LANG_EN
    assert detect_language("") == LANG_EN
    assert detect_language("    ") == LANG_EN


def test_detect_language_hindi():
    """Verify Hindi Devanagari text is accurately detected as 'hi'."""
    assert detect_language("नमस्ते") == LANG_HI
    assert detect_language("क्लाउड सुरक्षा और अनुपालन नीतियां क्या हैं?") == LANG_HI
    assert detect_language("नमस्ते, क्या आप डिस्काउंट के बारे में बता सकते हैं?") == LANG_HI
    assert detect_language("कंपनी के उत्पाद और सेवाएं क्या हैं?") == LANG_HI
    assert detect_language("ऋण के लिए आवश्यक दस्तावेज क्या हैं?") == LANG_HI


def test_detect_language_marathi():
    """Verify Marathi Devanagari text (with unique characters and lexicon) is detected as 'mr'."""
    assert detect_language("नमस्कार") == LANG_MR
    assert detect_language("कंपनीचे एंटरप्राइज सवलत दर आणि करार कालावधी पर्याय काय आहेत?") == LANG_MR
    assert detect_language("कर्जासाठी आवश्यक कागदपत्रे कोणती आहेत?") == LANG_MR
    # Marathi specific 'ळ'
    assert detect_language("वेळ किती लागेल?") == LANG_MR
    assert detect_language("सगळे नियम सांगा") == LANG_MR


def test_detect_language_romanized():
    """Verify transliterated / Romanized Hindi and Marathi detection."""
    assert detect_language("namaste aap kaise ho") == LANG_HI
    assert detect_language("namaskar kasa ahes") == LANG_MR


# ---------------------------------------------------------------------------
# Unit Tests: Greeting Detection vs Direct Questions
# ---------------------------------------------------------------------------

def test_is_greeting_pure_greetings():
    """Verify pure greetings in English, Hindi, and Marathi return True."""
    assert is_greeting("Hi") is True
    assert is_greeting("Hello") is True
    assert is_greeting("Hey") is True
    assert is_greeting("हाय") is True
    assert is_greeting("नमस्ते") is True
    assert is_greeting("नमस्कार") is True
    assert is_greeting("Hi/Hello/Hey/हाय/नमस्ते/नमस्कार") is True
    assert is_greeting("Hello there!") is True
    assert is_greeting("Good morning") is True
    assert is_greeting("नमस्ते जी") is True
    assert is_greeting("नमस्कार सर") is True
    assert is_greeting("सुप्रभात") is True
    assert is_greeting("शुभ सकाळ") is True


def test_is_greeting_questions_return_false():
    """
    Verify that substantive questions starting with greetings return False,
    so the assistant will answer directly without an unnecessary greeting prefix.
    """
    assert is_greeting("What are our enterprise discount tiers?") is False
    assert is_greeting("Hello, what are the enterprise discount tiers?") is False
    assert is_greeting("Hi, explain the SOC2 Type II compliance policy.") is False
    assert is_greeting("Hey, what is the interest rate?") is False
    assert is_greeting("नमस्ते, एंटरप्राइज डिस्काउंट क्या है?") is False
    assert is_greeting("नमस्ते, ब्याज दर क्या है?") is False
    assert is_greeting("नमस्कार, कंपनीची किंमत आणि सवलत धोरण काय आहे?") is False
    assert is_greeting("नमस्कार, कर्जाचे नियम काय आहेत?") is False
    assert is_greeting("कर्जासाठी पात्रता निकष काय आहेत?") is False


# ---------------------------------------------------------------------------
# Unit Tests: RAG Grounded Greeting Recommendations vs Direct Answers
# ---------------------------------------------------------------------------

def test_rag_greeting_with_chunks_offers_grounded_recommendations():
    """
    Verify that when user inputs only a greeting and chunks exist,
    the assistant greets naturally and offers recommendations grounded in context.
    """
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = {
        "results": [
            {
                "id": "playbook.pdf_p1_c0",
                "score": 0.95,
                "text": "Enterprise cloud pricing: Tier 1 (10-50 seats), Tier 2 (50-200 seats), Tier 3 (200+ seats).",
                "source": "playbook.pdf",
                "page": 1,
                "chunk_index": 0,
            }
        ]
    }

    mock_http_response = MagicMock()
    mock_http_response.status_code = 200
    mock_http_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Hello! Welcome to Voice Sales Copilot. Based on our sales playbooks, I can assist you with:\n1. Enterprise volume discount tiers\n2. Pricing structures for 10-200+ seats\nWhat would you like to know?"
                }
            }
        ]
    }

    mock_http_client = MagicMock()
    mock_http_client.post.return_value = mock_http_response

    rag_service = RAGService(
        retriever=mock_retriever,
        api_key="fake-key",
        http_client=mock_http_client,
    )

    result = rag_service.answer_question("Hello")

    assert result["is_greeting"] is True
    assert result["language"] == "en"
    assert "Hello! Welcome" in result["answer"]
    assert len(result["context_used"]) == 1
    assert result["sources"] == ["playbook.pdf"]
    assert result["fallback_used"] is False

    # Check LLM call prompt
    call_args = mock_http_client.post.call_args
    payload = call_args[1]["json"]
    system_msg = payload["messages"][0]["content"]
    assert "CRITICAL RULES" in system_msg
    assert "English" in system_msg


def test_rag_greeting_empty_kb_returns_truthful_message():
    """
    Verify that when user inputs a greeting but knowledge base is empty,
    assistant greets and advises user without hallucinating topics.
    """
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = {"results": []}

    mock_http_client = MagicMock()
    rag_service = RAGService(
        retriever=mock_retriever,
        api_key="fake-key",
        http_client=mock_http_client,
    )

    # English greeting with empty KB
    res_en = rag_service.answer_question("Hi")
    assert res_en["is_greeting"] is True
    assert res_en["language"] == "en"
    assert "no sales playbooks or documents have been uploaded" in res_en["answer"]

    # Hindi greeting with empty KB
    res_hi = rag_service.answer_question("नमस्ते")
    assert res_hi["is_greeting"] is True
    assert res_hi["language"] == "hi"
    assert "नॉलेज बेस में कोई सेल्स प्लेबुक" in res_hi["answer"]

    # Marathi greeting with empty KB
    res_mr = rag_service.answer_question("नमस्कार")
    assert res_mr["is_greeting"] is True
    assert res_mr["language"] == "mr"
    assert "नॉलेज बेसमध्ये कोणतीही सेल्स प्लेबुक" in res_mr["answer"]

    # LLM was not called to prevent hallucinations
    mock_http_client.post.assert_not_called()


def test_rag_greeting_never_returns_fallback_even_if_llm_emits_fallback_phrase():
    """
    Verify that if the LLM mistakenly returns a fallback phrase for a greeting,
    it is intercepted and replaced with a warm, grounded greeting recommendation,
    strictly ensuring fallback_used is False.
    """
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = {
        "results": [
            {
                "id": "policy.pdf_p1_c0",
                "score": 0.9,
                "text": "Home Loan ROI starts at 8.75% for CIBIL 750+. Processing fee 0.5%.",
                "source": "policy.pdf",
                "page": 1,
                "chunk_index": 0,
            }
        ]
    }

    mock_http_response = MagicMock()
    mock_http_response.status_code = 200
    mock_http_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "The provided information is not available in the context."
                }
            }
        ]
    }

    mock_http_client = MagicMock()
    mock_http_client.post.return_value = mock_http_response

    rag_service = RAGService(
        retriever=mock_retriever,
        api_key="fake-key",
        http_client=mock_http_client,
    )

    for greeting in ["Hi", "Hello", "Hey", "हाय", "नमस्ते", "नमस्कार", "Hi/Hello/Hey/हाय/नमस्ते/नमस्कार"]:
        result = rag_service.answer_question(greeting)
        assert result["is_greeting"] is True
        assert result["fallback_used"] is False
        assert "The provided information is not available" not in result["answer"]
        assert len(result["answer"]) > 10


def test_rag_direct_question_in_hindi_answers_directly():
    """
    Verify that direct Hindi questions are answered directly without greetings.
    """
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = {
        "results": [
            {
                "id": "playbook.pdf_p2_c0",
                "score": 0.93,
                "text": "Enterprise discount: 20% discount for 100+ licenses.",
                "source": "playbook.pdf",
                "page": 2,
                "chunk_index": 0,
            }
        ]
    }

    mock_http_response = MagicMock()
    mock_http_response.status_code = 200
    mock_http_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "100 से अधिक लाइसेंस के लिए 20% छूट दी जाती है।"
                }
            }
        ]
    }

    mock_http_client = MagicMock()
    mock_http_client.post.return_value = mock_http_response

    rag_service = RAGService(
        retriever=mock_retriever,
        api_key="fake-key",
        http_client=mock_http_client,
    )

    result = rag_service.answer_question("100 से अधिक लाइसेंस पर क्या छूट है?")

    assert result["is_greeting"] is False
    assert result["language"] == "hi"
    assert "20% छूट" in result["answer"]
    assert result["sources"] == ["playbook.pdf"]

    call_args = mock_http_client.post.call_args
    payload = call_args[1]["json"]
    system_msg = payload["messages"][0]["content"]
    assert "DO NOT include any greeting" in system_msg
    assert "Hindi" in system_msg


def test_rag_direct_question_in_marathi_answers_directly():
    """
    Verify that direct Marathi questions are answered directly without greetings.
    """
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = {
        "results": [
            {
                "id": "playbook.pdf_p3_c0",
                "score": 0.91,
                "text": "Data residency policy: All customer data is encrypted at rest using AES-256 in Mumbai data center.",
                "source": "playbook.pdf",
                "page": 3,
                "chunk_index": 0,
            }
        ]
    }

    mock_http_response = MagicMock()
    mock_http_response.status_code = 200
    mock_http_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "सर्व डेटा मुंबई डेटा सेंटरमध्ये AES-256 द्वारे एनक्रिप्ट केला जातो."
                }
            }
        ]
    }

    mock_http_client = MagicMock()
    mock_http_client.post.return_value = mock_http_response

    rag_service = RAGService(
        retriever=mock_retriever,
        api_key="fake-key",
        http_client=mock_http_client,
    )

    result = rag_service.answer_question("डेटा सुरक्षिततेचे नियम काय आहेत?")

    assert result["is_greeting"] is False
    assert result["language"] == "mr"
    assert "AES-256" in result["answer"]
    assert result["sources"] == ["playbook.pdf"]

    call_args = mock_http_client.post.call_args
    payload = call_args[1]["json"]
    system_msg = payload["messages"][0]["content"]
    assert "DO NOT include any greeting" in system_msg
    assert "Marathi" in system_msg


# ---------------------------------------------------------------------------
# Integration Tests: /api/ask with Language Support
# ---------------------------------------------------------------------------

def test_api_ask_returns_detected_language_and_greeting_status(client):
    """Verify POST /api/ask returns language and is_greeting flags."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = {"results": []}

    with patch("services.retriever.VectorRetriever", return_value=mock_retriever):
        # English greeting
        res_greet = client.post("/api/ask", json={"question": "Hello"})
        assert res_greet.status_code == 200
        data_greet = res_greet.json()
        assert data_greet["language"] == "en"
        assert data_greet["is_greeting"] is True

        # Hindi question with no chunks
        res_hi = client.post("/api/ask", json={"question": "कंपनी के सुरक्षा नियम क्या हैं?"})
        assert res_hi.status_code == 200
        data_hi = res_hi.json()
        assert data_hi["language"] == "hi"
        assert data_hi["is_greeting"] is False
        assert data_hi["fallback_used"] is True
        assert "पर्याप्त जानकारी उपलब्ध नहीं है" in data_hi["answer"]


# ---------------------------------------------------------------------------
# Unit Tests: STT & TTS Multilingual Support
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stt_transcribe_audio_returns_detected_language():
    """Verify DeepgramSTTService populates detected_language."""
    mock_deepgram_response = {
        "metadata": {"duration": 3.12},
        "results": {
            "channels": [
                {
                    "detected_language": "hi",
                    "alternatives": [
                        {
                            "transcript": "नमस्ते मैं आपकी क्या सहायता कर सकता हूँ",
                            "confidence": 0.985,
                            "words": [{"word": "नमस्ते"}],
                        }
                    ],
                }
            ]
        },
    }

    mock_http_response = MagicMock()
    mock_http_response.status_code = 200
    mock_http_response.json.return_value = mock_deepgram_response

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_http_response

    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__.return_value = mock_client
        service = DeepgramSTTService(api_key="fake-stt-key")
        result = await service.transcribe_audio(b"FAKE_AUDIO", language="hi")

    assert result["success"] is True
    assert result["transcript"] == "नमस्ते मैं आपकी क्या सहायता कर सकता हूँ"
    assert result["detected_language"] == "hi"


@pytest.mark.asyncio
async def test_tts_synthesize_multilingual_speech():
    """Verify DeepgramTTSService synthesizes speech with language parameter."""
    service = DeepgramTTSService(api_key="fake-tts-key")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"FAKE_HINDI_MP3_AUDIO"

    with patch("httpx.AsyncClient.post", return_value=mock_response) as mock_post:
        audio = await service.synthesize_speech(
            "नमस्ते, मैं आपकी क्या सहायता कर सकता हूँ?",
            language="hi",
        )
        assert audio == b"FAKE_HINDI_MP3_AUDIO"
        assert mock_post.called
        call_kwargs = mock_post.call_args[1]
        assert "Namaste" in call_kwargs["json"]["text"] or "नमस्ते" in call_kwargs["json"]["text"]


def test_devanagari_to_phonetic_hindi_and_marathi():
    """Verify Devanagari text transliteration produces natural phonetic Latin text for TTS."""
    from services.language import devanagari_to_phonetic

    hindi_text = "नमस्ते, मैं आपकी सहायता कर सकता हूँ"
    phonetic_hi = devanagari_to_phonetic(hindi_text)
    assert "Namaste" in phonetic_hi
    assert "sahaayataa" in phonetic_hi or "sahayata" in phonetic_hi

    marathi_text = "नमस्कार, आम्ही मदत करू शकतो"
    phonetic_mr = devanagari_to_phonetic(marathi_text)
    assert "Namaskar" in phonetic_mr
    assert "shakato" in phonetic_mr or "shakt" in phonetic_mr


@pytest.mark.asyncio
async def test_tts_model_routing_per_language():
    """Verify language-specific Deepgram Aura model selection for en, hi, mr."""
    service = DeepgramTTSService(api_key="fake-tts-key")
    mock_response = MagicMock(status_code=200, content=b"AUDIO")

    with patch("httpx.AsyncClient.post", return_value=mock_response) as mock_post:
        # English -> aura-asteria-en
        await service.synthesize_speech("Hello", language="en")
        assert "model=aura-asteria-en" in mock_post.call_args[0][0]

        # Hindi -> aura-luna-en
        await service.synthesize_speech("नमस्ते", language="hi")
        assert "model=aura-luna-en" in mock_post.call_args[0][0]

        # Marathi -> aura-stella-en
        await service.synthesize_speech("नमस्कार", language="mr")
        assert "model=aura-stella-en" in mock_post.call_args[0][0]


@pytest.mark.asyncio
async def test_stt_model_routing_and_lexical_override():
    """Verify STT uses nova-3 for Hindi/Marathi and lexical detection overrides false English label."""
    service = DeepgramSTTService(api_key="fake-stt-key")

    mock_dg_res = {
        "metadata": {"duration": 2.0},
        "results": {
            "channels": [
                {
                    "detected_language": "en",  # Deepgram audio classifier falsely guessed English
                    "alternatives": [
                        {
                            "transcript": "कंपनीचे एंटरप्राइज सवलत दर काय आहेत?",  # Authentic Marathi Devanagari
                            "confidence": 0.95,
                            "words": [{"word": "कंपनीचे"}],
                        }
                    ],
                }
            ]
        },
    }

    mock_http_response = MagicMock(status_code=200)
    mock_http_response.json.return_value = mock_dg_res

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_http_response

    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.__aenter__.return_value = mock_client
        res = await service.transcribe_audio(b"AUDIO", language="mr")

        # Verify nova-3 was selected
        assert res["model"] == "nova-3"
        # Verify lexical detection overrode false "en" to "mr"
        assert res["detected_language"] == "mr"
        assert res["transcript"] == "कंपनीचे एंटरप्राइज सवलत दर काय आहेत?"

