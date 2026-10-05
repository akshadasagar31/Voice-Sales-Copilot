# ============================================================================
# MODULE 1 REGRESSION TESTS: LANGUAGE DETECTION & PHONE PRESERVATION
# (backend/tests/test_module1_language_and_phone.py)
# ============================================================================
# Verifies:
# 1. Spoken language detection strictly from current turn audio (EN, HI, MR).
# 2. No unconditional Devanagari boost in multi-stream STT arbitration.
# 3. Short English phrases ("what can you do for me?") stay English.
# 4. Zero language inheritance across turns (HI -> EN -> MR).
# 5. Smart-formatted times (9:00, 9:30) de-formatted only in phone extraction.
# 6. Leading zeros and repeated digits preserved exactly as spoken.
# 7. Phone field strictly validated before advancing lead flow.
# ============================================================================

import json
import pytest

from main import generate_module1_ws_response
from services.language import (
    detect_spoken_language,
    select_best_multilingual_transcript,
    LANG_EN,
    LANG_HI,
    LANG_MR,
)
from services.lead_extractor import (
    extract_phone_number,
    is_valid_phone_number,
)


# ----------------------------------------------------------------------------
# 1. Language Detection & Stream Arbitration Tests
# ----------------------------------------------------------------------------

def test_detect_spoken_language_short_english_phrases():
    """Verify short English phrases cleanly detect as English."""
    assert detect_spoken_language("what can you do for me?") == LANG_EN
    assert detect_spoken_language("who are you?") == LANG_EN
    assert detect_spoken_language("how can you help me?") == LANG_EN
    assert detect_spoken_language("hello") == LANG_EN
    assert detect_spoken_language("what are your interest rates?") == LANG_EN
    assert detect_spoken_language("can you tell me more about personal loans?") == LANG_EN


def test_detect_spoken_language_hindi_phrases():
    """Verify authentic Hindi utterances detect as Hindi."""
    assert detect_spoken_language("मुझे 5 लाख का पर्सनल लोन चाहिए") == LANG_HI
    assert detect_spoken_language("लोन के लिए क्या नियम हैं?") == LANG_HI
    assert detect_spoken_language("आप कौन हैं?") == LANG_HI
    assert detect_spoken_language("नमस्ते, मुझे जानकारी चाहिए") == LANG_HI


def test_detect_spoken_language_marathi_phrases():
    """Verify authentic Marathi utterances detect as Marathi."""
    assert detect_spoken_language("मला पाच लाख कर्ज हवे आहे") == LANG_MR
    assert detect_spoken_language("कर्जाचे व्याजदर किती आहेत?") == LANG_MR
    assert detect_spoken_language("तुम्ही काय करू शकता?") == LANG_MR
    assert detect_spoken_language("नमस्कार, मला कर्ज पाहिजे") == LANG_MR


def test_select_best_multilingual_transcript_no_deva_boost_for_english():
    """
    Verify that when 3 concurrent STT streams run for English speech,
    phonetic Devanagari output from Hindi/Marathi does NOT win over high-confidence English.
    """
    candidates = {
        "en": {"transcript": "what can you do for me", "confidence": 0.95},
        "hi": {"transcript": "वाट कॅन यु डू फॉर मी", "confidence": 0.50},
        "mr": {"transcript": "व्हॉट कॅन यू डू फॉर मी", "confidence": 0.45},
    }
    winning_tr, winning_lang, winning_conf = select_best_multilingual_transcript(candidates)
    assert winning_lang == LANG_EN
    assert winning_tr == "what can you do for me"


def test_select_best_multilingual_transcript_marathi_winner():
    """Verify authentic Marathi speech wins over phonetic English."""
    candidates = {
        "mr": {"transcript": "मला पाच लाख कर्ज हवे आहे", "confidence": 0.92},
        "hi": {"transcript": "मला पाच लाख कर्ज हवे आहे", "confidence": 0.55},
        "en": {"transcript": "mala pach lakh karja have ahe", "confidence": 0.40},
    }
    winning_tr, winning_lang, winning_conf = select_best_multilingual_transcript(candidates)
    assert winning_lang == LANG_MR
    assert "कर्ज" in winning_tr


def test_select_best_multilingual_transcript_hindi_winner():
    """Verify authentic Hindi speech wins over phonetic English."""
    candidates = {
        "hi": {"transcript": "मुझे पर्सनल लोन चाहिए", "confidence": 0.94},
        "mr": {"transcript": "मुझे पर्सनल लोन चाहिए", "confidence": 0.50},
        "en": {"transcript": "mujhe personal loan chahiye", "confidence": 0.45},
    }
    winning_tr, winning_lang, winning_conf = select_best_multilingual_transcript(candidates)
    assert winning_lang == LANG_HI
    assert "लोन" in winning_tr


def test_turn_by_turn_language_isolation_no_inheritance():
    """
    Verify zero language inheritance across turns:
    Turn 1: Hindi query
    Turn 2: English question from same session -> must detect as EN, not HI!
    Turn 3: Marathi question from same session -> must detect as MR, not EN!
    """
    # Turn 1: Hindi
    resp1 = generate_module1_ws_response(
        current="नमस्ते, मुझे पर्सनल लोन चाहिए",
        req_lang="auto",
        last_conf=0.95,
        existing_lead_str="",
    )
    assert resp1["detected_language"] == LANG_HI

    # Turn 2: English with previous turn context and lead
    resp2 = generate_module1_ws_response(
        current="what can you do for me?",
        req_lang="hi",  # Previous turn's language passed as req_lang
        last_conf=0.95,
        existing_lead_str=json.dumps(resp1.get("lead", {})),
    )
    assert resp2["detected_language"] == LANG_EN
    assert resp2["language"] == LANG_EN

    # Turn 3: Marathi
    resp3 = generate_module1_ws_response(
        current="मला पाच लाख कर्ज हवे आहे",
        req_lang="en",  # Previous turn's language passed as req_lang
        last_conf=0.95,
        existing_lead_str=json.dumps(resp2.get("lead", {})),
    )
    assert resp3["detected_language"] == LANG_MR
    assert resp3["language"] == LANG_MR


# ----------------------------------------------------------------------------
# 2. Phone Extraction & Time De-formatting Tests
# ----------------------------------------------------------------------------

def test_extract_phone_number_deformats_time_patterns():
    """
    Verify that smart-formatted times like 9:00, 9:30 inside phone speech
    are de-formatted to digits (900, 930) and never rejected or saved as times.
    """
    # 9:00 at end of phone number
    assert extract_phone_number("my phone number is 9876543 9:00") == "9876543900"
    # 9:30 in phone number
    assert extract_phone_number("call me on 9876543 9:30") == "9876543930"
    # 10:00 in phone number
    assert extract_phone_number("contact is 987654 10:00") == "9876541000"
    # 9:00 with incidental AM tag
    assert extract_phone_number("number is 9876543 9:00 AM") == "9876543900"


def test_extract_phone_number_rejects_standalone_appointment_time():
    """
    Verify standalone appointment / schedule times are NOT extracted as phone numbers.
    """
    assert extract_phone_number("call me at 9:00 AM") is None
    assert extract_phone_number("available at 5:00 PM") is None
    assert extract_phone_number("reach me around 4:00") is None


def test_extract_phone_number_preserves_leading_zeros():
    """
    Verify that leading zeros (e.g. 09876543210) are preserved exactly as spoken.
    """
    # Direct 11-digit with leading zero
    assert extract_phone_number("09876543210") == "09876543210"
    # Spaced digits with leading zero
    assert extract_phone_number("0 9 8 7 6 5 4 3 2 1 0") == "09876543210"
    # Spoken number words with leading zero
    assert extract_phone_number("zero nine eight seven six five four three two one zero") == "09876543210"


def test_extract_phone_number_preserves_repeated_digits():
    """
    Verify repeated digits (double, triple, consecutive identical digits) are preserved.
    """
    assert extract_phone_number("double nine eight seven six five four three two one") == "9987654321"
    assert extract_phone_number("triple nine eight seven six five four three two") == "9998765432"
    assert extract_phone_number("my number is 9998887776") == "9998887776"


def test_is_valid_phone_number():
    """Verify phone validation bounds (7-15 digits, rejects letters/short)."""
    assert is_valid_phone_number("9876543210") is True
    assert is_valid_phone_number("09876543210") is True
    assert is_valid_phone_number("9987654321") is True
    assert is_valid_phone_number("12345") is False  # Too short (< 7)
    assert is_valid_phone_number("call me") is False
    assert is_valid_phone_number("") is False
    assert is_valid_phone_number(None) is False


# ----------------------------------------------------------------------------
# 3. Lead Flow Phone Validation Tests
# ----------------------------------------------------------------------------

def test_lead_flow_phone_validation_does_not_advance_on_invalid_input():
    """
    When pending field is 'phone':
    - Non-phone speech (e.g. 'call me tomorrow') must NOT advance flow to company.
    - Incomplete digits (< 7) must NOT advance flow.
    - Valid phone (e.g. '9876543210' or '9876543 9:00') advances to 'company'.
    """
    active_lead = {"name": "Amit Kumar"}

    # Turn 1: Invalid input for phone
    resp1 = generate_module1_ws_response(
        current="call me tomorrow",
        req_lang="en",
        last_conf=0.95,
        existing_lead_str=json.dumps(active_lead),
    )
    # Phone must NOT be saved, next_missing_parameter remains 'phone'
    assert resp1["lead"].get("phone") is None
    assert resp1["next_missing_parameter"] == "phone"
    assert resp1["is_unclear"] is True

    # Turn 2: Incomplete digits (< 7)
    resp2 = generate_module1_ws_response(
        current="12345",
        req_lang="en",
        last_conf=0.95,
        existing_lead_str=json.dumps(active_lead),
    )
    assert resp2["lead"].get("phone") is None
    assert resp2["next_missing_parameter"] == "phone"
    assert resp2["is_unclear"] is True

    # Turn 3: Valid phone with smart-formatted time
    resp3 = generate_module1_ws_response(
        current="My phone number is 9876543 9:00",
        req_lang="en",
        last_conf=0.95,
        existing_lead_str=json.dumps(active_lead),
    )
    assert resp3["lead"].get("phone") == "9876543900"
    assert resp3["next_missing_parameter"] == "company"
    assert resp3["is_unclear"] is False
