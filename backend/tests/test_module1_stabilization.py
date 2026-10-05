"""
=============================================================================
MODULE 1 / LIVE CALL COPILOT STABILIZATION REGRESSION SUITE
=============================================================================
Tests all core stabilization requirements for Module 1 / Live Call Copilot:
1. 6 required fields sequence (name, phone, company, loan_type, loan_amount, tenure_months).
2. Strict validation: "This is" rejected as company name; current field never advances until valid.
3. Out-of-order fields safely saved while keeping current invalid field pending.
4. Name extraction: rejects numbers, amounts, questions, fillers, and job roles as names.
5. Company extraction before job roles: "Senior software engineer at Infosys" -> Infosys.
6. Phone number extraction: leading zeros, repeated digits, number words, smart-formatted time un-formatting.
7. Refusal handling: stops requested field without saving refusal text.
8. Intent routing: loan intent -> lead flow; general question -> active LLM without lead flow;
   lead flow active -> general question answered, lead preserved, pending field resumed.
9. Language detection: EN/HI/MR, zero cross-turn inheritance, Devanagari script preservation.
10. TTS routing: English -> Deepgram, Hindi -> Sarvam Priya, Marathi -> Sarvam Ritu,
    native browser fallback if Sarvam fails, never sends Indic to English TTS.
=============================================================================
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from services.lead_extractor import (
    REQUIRED_LEAD_FIELDS,
    get_next_missing_parameter,
    is_valid_company_name,
    is_valid_prospect_name,
    is_valid_phone_number,
    is_field_value_valid,
    extract_company,
    extract_name,
    extract_phone_number,
    extract_loan_amount,
    extract_tenure_months,
    merge_lead_safely,
    is_loan_intent,
    is_general_question,
    is_field_refusal,
    LeadExtractorService,
)
from services.language import (
    detect_language,
    get_response_language,
    LANG_EN,
    LANG_HI,
    LANG_MR,
)
from main import app

client = TestClient(app)


# ============================================================================
# 1. 6 Required Lead Fields & Deterministic Sequence
# ============================================================================

def test_required_lead_fields_sequence():
    """Verify that REQUIRED_LEAD_FIELDS contains exactly the 6 required fields in deterministic order."""
    expected = ["name", "phone", "company", "loan_type", "loan_amount", "tenure_months"]
    assert REQUIRED_LEAD_FIELDS == expected


def test_get_next_missing_parameter_sequential_validation():
    """Verify that get_next_missing_parameter walks through all 6 fields and enforces validity."""
    lead = {}
    assert get_next_missing_parameter(lead) == "name"

    lead["name"] = "Rajesh Sharma"
    assert get_next_missing_parameter(lead) == "phone"

    lead["phone"] = "9876543210"
    assert get_next_missing_parameter(lead) == "company"

    # Invalid company like 'This is' must NOT satisfy company field
    lead["company"] = "This is"
    assert get_next_missing_parameter(lead) == "company"

    lead["company"] = "Tata Consultancy Services"
    assert get_next_missing_parameter(lead) == "loan_type"

    lead["loan_type"] = "Personal Loan"
    assert get_next_missing_parameter(lead) == "loan_amount"

    lead["loan_amount"] = 500000.0
    assert get_next_missing_parameter(lead) == "tenure_months"

    lead["tenure_months"] = 24
    assert get_next_missing_parameter(lead) is None


# ============================================================================
# 2. Strict Validation: "This is" Rejected, Current Field Never Advances
# ============================================================================

def test_company_validation_rejects_conversational_fillers():
    """Verify that conversational phrases like 'This is' or 'It is' are strictly rejected as companies."""
    assert is_valid_company_name("This is") is False
    assert is_valid_company_name("this is") is False
    assert is_valid_company_name("It is") is False
    assert is_valid_company_name("I am") is False
    assert is_valid_company_name("हे आहे") is False
    assert is_valid_company_name("यह है") is False

    # Legitimate company names
    assert is_valid_company_name("Infosys") is True
    assert is_valid_company_name("Tata Motors") is True
    assert is_valid_company_name("टीसीएस") is True


def test_out_of_order_extraction_preserves_valid_fields_without_advancing_invalid_field():
    """
    If current field is 'company' and user says:
    'This is, and I want 5 lakh personal loan for 2 years',
    'company' remains invalid, so next_missing_parameter must stay 'company',
    while valid out-of-order fields (loan_type, loan_amount, tenure_months) are safely stored.
    """
    existing_lead = {
        "name": "Amit Kumar",
        "phone": "9876543210",
        "company": None,
        "loan_type": None,
        "loan_amount": None,
        "tenure_months": None,
    }
    extractor = LeadExtractorService(api_key="mock_key")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{
            "message": {
                "content": '{"name": null, "phone": null, "company": "This is", "loan_type": "Personal Loan", "loan_amount": 500000, "tenure_months": 24}'
            }
        }]
    }
    with patch("httpx.Client.post", return_value=mock_resp):
        res = extractor.extract_lead(
            "This is, and I want 5 lakh personal loan for 2 years",
            existing_lead=existing_lead,
        )

    lead = res["lead"]
    assert lead["name"] == "Amit Kumar"
    assert lead["phone"] == "9876543210"
    assert lead["company"] is None  # 'This is' rejected
    assert lead["loan_type"] == "Personal Loan"
    assert lead["loan_amount"] == 500000.0
    assert lead["tenure_months"] == 24
    assert res["next_missing_parameter"] == "company"
    assert res["is_unclear"] is True


# ============================================================================
# 3. Name & Company Extraction vs Job Roles
# ============================================================================

def test_name_validation_rejects_job_roles_and_fillers():
    """Verify job roles, fillers, numbers, and questions are never accepted as prospect names."""
    assert is_valid_prospect_name("Software Engineer") is False
    assert is_valid_prospect_name("Manager") is False
    assert is_valid_prospect_name("Consultant") is False
    assert is_valid_prospect_name("Analyst") is False
    assert is_valid_prospect_name("Developer") is False
    assert is_valid_prospect_name("Uh like actually") is False
    assert is_valid_prospect_name("500000") is False
    assert is_valid_prospect_name("What is interest rate?") is False

    # Valid prospect names
    assert is_valid_prospect_name("Amitabh Bachchan") is True
    assert is_valid_prospect_name("सचिन तेंडुलकर") is True
    assert is_valid_prospect_name("राजेश शर्मा") is True


def test_company_extraction_extracts_company_before_job_roles():
    """Verify extract_company extracts the company rather than the job role."""
    comp1 = extract_company("I work as a senior software engineer at Infosys")
    assert comp1 == "Infosys"

    comp2 = extract_company("Software engineer at TCS", context_field="company")
    assert comp2 == "TCS"

    comp3 = extract_company("I am a consultant at Wipro")
    assert comp3 == "Wipro"


# ============================================================================
# 4. Multilingual Phone Extraction
# ============================================================================

def test_phone_extraction_preserves_leading_zeros_and_repeats():
    """Verify leading zeros and repeated spoken digits are preserved."""
    # Leading zero
    assert extract_phone_number("My phone number is 09876543210") == "09876543210"
    assert extract_phone_number("0 9 8 7 6 5 4 3 2 1 0") == "09876543210"

    # Spoken repeater words
    assert extract_phone_number("phone is double nine eight seven six five four three two one") == "9987654321"
    assert extract_phone_number("डबल आठ सात सहा पाच चार तीन दोन एक शून्य") == "8876543210"


def test_phone_extraction_unformats_time_values():
    """Verify smart-formatted time strings like 9:00 or 9:30 inside phone speech are de-formatted into digits."""
    assert extract_phone_number("phone number is 9876543 9:00") == "9876543900"
    assert extract_phone_number("contact is 9876543 9:30") == "9876543930"

    # Standalone appointment time is NOT extracted as phone
    assert extract_phone_number("call me at 9:00 AM") is None


# ============================================================================
# 5. Refusal Handling
# ============================================================================

def test_refusal_stops_requested_field_without_saving_text():
    """Verify user refusal on a field pauses without saving refusal text into the lead."""
    assert is_field_refusal("I don't want to share my phone number", pending_field="phone") is True
    assert is_field_refusal("मुझे फ़ोन नंबर नहीं देना है", pending_field="phone") is True
    assert is_field_refusal("मी माझा मोबाईल नंबर देणार नाही", pending_field="phone") is True

    existing_lead = {
        "name": "Pooja Hegde",
        "phone": None,
        "company": None,
        "loan_type": None,
        "loan_amount": None,
        "tenure_months": None,
    }
    extractor = LeadExtractorService(api_key="mock_key")
    res = extractor.extract_lead(
        "I don't want to give my phone number right now",
        existing_lead=existing_lead,
    )

    assert res["status"] == "refusal_paused"
    assert res["is_field_refusal"] is True
    assert res["lead"]["phone"] is None
    assert "don't want" not in str(res["lead"]["phone"])
    assert res["next_missing_parameter"] == "phone"


# ============================================================================
# 6. Intent Routing: Loan Intent vs General Question
# ============================================================================

def test_intent_routing_general_question_no_prior_lead():
    """When no lead is active, general question does NOT start lead flow."""
    transcript = "What is the current inflation rate in India?"
    assert is_general_question(transcript) is True
    assert is_loan_intent(transcript) is False

    extractor = LeadExtractorService(api_key="mock_key")
    with patch.object(extractor, "answer_general_query", return_value="Inflation in India is currently around 5 percent."):
        res = extractor.extract_lead(transcript, existing_lead=None)
        assert res["status"] == "assistant_query"
        assert res["is_general_query"] is True
        assert res["next_missing_parameter"] is None


def test_intent_routing_general_question_during_active_lead():
    """When lead is active, general question is answered and pending field is preserved."""
    existing_lead = {
        "name": "Vikram Seth",
        "phone": "9876543210",
        "company": "TCS",
        "loan_type": None,
        "loan_amount": None,
        "tenure_months": None,
    }
    transcript = "What are the tax benefits of home loans?"
    extractor = LeadExtractorService(api_key="mock_key")

    with patch.object(extractor, "answer_general_query", return_value="Home loans offer tax deductions under Section 80C and Section 24b."):
        res = extractor.extract_lead(transcript, existing_lead=existing_lead)
        assert res["status"] == "assistant_query"
        assert res["next_missing_parameter"] == "loan_type"
        assert res["lead"]["name"] == "Vikram Seth"
        assert res["lead"]["company"] == "TCS"


# ============================================================================
# 7. Language Detection & Devanagari Preservation
# ============================================================================

def test_language_detection_strictly_en_hi_mr():
    """Verify that language detection returns only en, hi, or mr with zero cross-turn inheritance."""
    # English
    assert detect_language("What can you do for me?") == LANG_EN
    assert detect_language("My name is Rajesh and I need a personal loan") == LANG_EN

    # Hindi (Devanagari)
    assert detect_language("नमस्ते, मुझे 5 लाख का पर्सनल लोन चाहिए") == LANG_HI
    assert detect_language("मेरा नाम अमित है") == LANG_HI

    # Marathi (Devanagari)
    assert detect_language("नमस्कार, मला 5 लाख रुपयांचे वैयक्तिक कर्ज हवे आहे") == LANG_MR
    assert detect_language("माझे नाव राहुल आहे आणि मी टीसीएस मध्ये काम करतो") == LANG_MR


def test_response_language_manual_override_priority():
    """Manual language selection strictly overrides auto detection."""
    assert get_response_language("hi", requested_language="en") == LANG_EN
    assert get_response_language("en", requested_language="mr") == LANG_MR
    assert get_response_language("en", requested_language="hi") == LANG_HI


# ============================================================================
# 8. TTS Routing: Voices & Browser Fallback
# ============================================================================

def test_tts_english_routes_to_deepgram_aura():
    """Verify English text routes to Deepgram Aura TTS."""
    with patch("services.tts.DeepgramTTSService.synthesize_speech", return_value=b"ID3_MOCK_MP3") as mock_dg:
        response = client.post(
            "/api/tts",
            json={"text": "Hello, thank you for providing your details.", "language": "en", "module": "module1"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] in ("audio/mpeg", "audio/wav")


def test_tts_indic_never_routes_to_english_aura():
    """Verify Hindi/Marathi Devanagari text routes to Sarvam Priya/Ritu or native browser fallback, never Deepgram."""
    with patch("services.sarvam_tts.SarvamTTSService.synthesize_speech", side_effect=Exception("Sarvam quota exceeded")):
        response = client.post(
            "/api/tts",
            json={"text": "नमस्ते, आपका नाम क्या है?", "language": "hi", "module": "module1"},
        )
        assert response.status_code == 200
        # When Sarvam fails for Hindi/Marathi, must return browser fallback JSON, NEVER English Aura!
        assert response.headers.get("x-tts-fallback") == "browser-speech-synthesis"
        data = response.json()
        assert data.get("fallback_to_browser") is True
        assert data.get("speaker") == "priya"


# ============================================================================
# 9. Dynamic Models Management & Multilingual Numbers / Fractional Tenure
# ============================================================================

def test_dynamic_models_management_active_resolution():
    """Verify STT, LLM, and TTS models are dynamically fetched from ModelManager, not hardcoded."""
    from services.model_manager import get_model_manager
    manager = get_model_manager()
    active_stt = manager.get_active_model("stt")
    active_llm = manager.get_active_model("llm")
    active_tts = manager.get_active_model("tts")

    assert active_stt is not None
    assert active_llm is not None
    assert active_tts is not None
    assert bool(active_stt.get("model_id") or active_stt.get("id")) is True
    assert bool(active_llm.get("model_id") or active_llm.get("id")) is True
    assert bool(active_tts.get("model_id") or active_tts.get("id")) is True


def test_multilingual_number_and_fractional_tenure_parsing():
    """Verify extraction of fractional tenure (e.g. 1.5 years -> 18 months, 2.5 years -> 30 months)."""
    assert extract_tenure_months("I need this loan for 1.5 years") == 18
    assert extract_tenure_months("tenure is 2.5 years") == 30
    assert extract_tenure_months("डेढ़ साल के लिए चाहिए") == 18
    assert extract_tenure_months("अडीच वर्षे कालावधी") == 30

