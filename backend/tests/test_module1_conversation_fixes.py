# ============================================================================
# MODULE 1 CONVERSATION & MULTILINGUAL VOICE HANDLING TESTS
# (backend/tests/test_module1_conversation_fixes.py)
# ============================================================================
import json
from unittest.mock import AsyncMock
import pytest
from fastapi.testclient import TestClient

from main import app
from services.lead_extractor import (
    Lead,
    extract_phone_number,
    extract_loan_amount,
    extract_tenure_months,
    extract_name,
    get_clarification_prompt,
    get_missing_parameter_prompt,
    merge_lead_safely,
)
from services.number_normalizer import (
    normalize_spoken_numbers,
    parse_numeric_phrase,
)
from services.language import (
    is_assistant_query,
    get_assistant_query_response,
    devanagari_to_phonetic,
    detect_spoken_language,
)
from services.stt import FINANCIAL_KEYTERMS, DeepgramSTTService

client = TestClient(app)


# ----------------------------------------------------------------------------
# 1. Hindi/Marathi STT Keyterms & Language Routing
# ----------------------------------------------------------------------------
def test_financial_keyterms_include_devanagari():
    """Verify FINANCIAL_KEYTERMS includes Hindi and Marathi tenure, credit, and currency terms."""
    keyterms = FINANCIAL_KEYTERMS
    devanagari_terms = ["महिने", "महीने", "वर्षे", "वर्ष", "साल", "कोटी", "हजार", "हज़ार"]
    for term in devanagari_terms:
        assert term in keyterms, f"Expected {term} to be in FINANCIAL_KEYTERMS"


def test_language_routing_neutral_digits_respects_user_language():
    """When user selects Hindi or Marathi and sends digits/names, language routing respects requested language."""
    # Digits only with user requesting 'hi'
    detected = detect_spoken_language("9876543210", requested_language="hi")
    assert detected == "hi"

    # Digits only with user requesting 'mr'
    detected = detect_spoken_language("9876543210", requested_language="mr")
    assert detected == "mr"


# ----------------------------------------------------------------------------
# 2. Interim STT Ignored & Lead Data Protection
# ----------------------------------------------------------------------------
def test_interim_stt_ignored_in_voice_entry():
    """Verify /api/voice-entry returns early with status=interim_ignored when is_interim=True."""
    payload = {
        "existing_lead": json.dumps({"name": "Rajesh Kumar", "phone": "9876543210"}),
        "is_interim": "true",
        "is_final": "false",
    }
    response = client.post("/api/voice-entry", data=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "interim_ignored"
    # Preserves existing lead intact
    assert data["lead"]["name"] == "Rajesh Kumar"
    assert data["lead"]["phone"] == "9876543210"


def test_protect_existing_lead_data_on_empty_speech():
    """Ensure existing valid lead fields are never overwritten by empty or partial speech."""
    existing = {"name": "Sunil Patil", "phone": "9876543210", "loan_amount": 500000.0}
    incoming = {"name": None, "phone": None, "loan_amount": None, "tenure_months": 24}
    merged = merge_lead_safely(existing, incoming)
    assert merged["name"] == "Sunil Patil"
    assert merged["phone"] == "9876543210"
    assert merged["loan_amount"] == 500000.0
    assert merged["tenure_months"] == 24


# ----------------------------------------------------------------------------
# 3. Spoken Number Extraction: Fractional Tenure, Phone, and Loan Amount
# ----------------------------------------------------------------------------
def test_spoken_fractional_tenure_marathi():
    """Marathi fractional years converted to months:
       - 'दीड वर्ष' -> 18 months
       - 'अडीच वर्षे' -> 30 months
       - 'साडे तीन वर्षे' -> 42 months
    """
    assert extract_tenure_months("मला दीड वर्षांसाठी कर्ज हवे आहे") == 18
    assert extract_tenure_months("मुदत अडीच वर्षे हवी आहे") == 30
    assert extract_tenure_months("साडे तीन वर्षे मुदत हवी") == 42


def test_spoken_fractional_tenure_hindi():
    """Hindi fractional years converted to months:
       - 'डेढ़ साल' -> 18 months
       - 'ढाई साल' -> 30 months
       - 'साढ़े चार साल' -> 54 months
    """
    assert extract_tenure_months("मुझे डेढ़ साल के लिए लोन चाहिए") == 18
    assert extract_tenure_months("लोन ढाई साल के लिए चाहिए") == 30
    assert extract_tenure_months("साढ़े चार साल की अवधि") == 54


def test_spoken_loan_amounts_fractional_lakh():
    """Fractional lakh amounts parsed properly:
       - 'ढाई लाख' / 'अडीच लाख' -> 250000
       - 'दीड लाख' -> 150000
       - 'सवा लाख' -> 125000
       - 'पावणे दोन लाख' -> 175000
    """
    assert extract_loan_amount("मला दीड लाख रुपयांचे कर्ज हवे आहे") == 150000.0
    assert extract_loan_amount("ढाई लाख का पर्सनल लोन चाहिए") == 250000.0
    assert extract_loan_amount("अडीच लाख रुपये पाहिजेत") == 250000.0
    assert extract_loan_amount("सवा लाख रुपये") == 125000.0
    assert extract_loan_amount("पावणे दोन लाख") == 175000.0


def test_spoken_phone_with_double_digits():
    """Spoken phone numbers with 'double nine...' parsed correctly."""
    text = "my contact is double nine eight seven six five four three two one"
    phone = extract_phone_number(text)
    assert phone == "9987654321"


def test_spoken_phone_marathi_digits():
    """Marathi spoken digit words normalize to 10-digit phone number."""
    text = "माझा नंबर नऊ आठ सात सहा पाच चार तीन दोन एक शून्य आहे"
    phone = extract_phone_number(text)
    assert phone == "9876543210"


# ----------------------------------------------------------------------------
# 4. Context-Aware Natural Questions (Replacing Hardcoded Repetitions)
# ----------------------------------------------------------------------------
def test_missing_parameter_prompt_context_awareness():
    """Prompts adapt with prospect name and known context instead of robotic static strings."""
    lead = Lead(name="Rajesh Sharma", loan_type="home")

    # Name is known, so asking for phone addresses Rajesh
    prompt_en_phone = get_missing_parameter_prompt("phone", lead, lang="en")
    assert "Rajesh" in prompt_en_phone

    # Loan type is known, so asking for loan amount contextualizes with home
    prompt_en_amt = get_missing_parameter_prompt("loan_amount", lead, lang="en")
    assert "home" in prompt_en_amt.lower()

    # Marathi context-awareness for phone addresses Rajesh
    prompt_mr_phone = get_missing_parameter_prompt("phone", lead, lang="mr")
    assert "Rajesh" in prompt_mr_phone


def test_missing_parameter_prompt_rotation():
    """Consecutive calls rotate through natural prompt phrasing."""
    lead = Lead()
    prompts_seen = set()
    for i in range(1, 6):
        p = get_missing_parameter_prompt("loan_type", lead, lang="en", attempt=i)
        prompts_seen.add(p)
    # Should have at least 2 distinct natural variations
    assert len(prompts_seen) >= 2


# ----------------------------------------------------------------------------
# 5. Assistant Knowledge Queries & Continuing the Lead Flow
# ----------------------------------------------------------------------------
def test_assistant_knowledge_queries_detection():
    """Queries about knowledge, capabilities, and identity detected correctly in EN/HI/MR."""
    # English
    assert is_assistant_query("What knowledge do you have?")
    assert is_assistant_query("What can you do?")
    assert is_assistant_query("Who are you?")

    # Hindi
    assert is_assistant_query("आप क्या जानते हैं?")
    assert is_assistant_query("आप क्या कर सकते हैं?")
    assert is_assistant_query("आप कौन हैं?")

    # Marathi
    assert is_assistant_query("तुमच्याकडे काय माहिती आहे?")
    assert is_assistant_query("तुम्ही काय करू शकता?")
    assert is_assistant_query("तुम्ही कोण आहात?")


def test_assistant_query_response_with_continuation_flow():
    """Assistant query returns helpful identity/knowledge answer and appends continuation prompt."""
    cont = "May I please have your name?"
    resp_en = get_assistant_query_response("What knowledge do you have?", "en", continuation_prompt=cont)
    assert "financial sales copilot" in resp_en.lower() or "interest rates" in resp_en.lower()
    assert cont in resp_en

    # Marathi
    cont_mr = "कृपया आपले नाव सांगू शकाल का?"
    resp_mr = get_assistant_query_response("तुमच्याकडे काय माहिती आहे?", "mr", continuation_prompt=cont_mr)
    assert "आर्थिक विक्री सहाय्यक" in resp_mr or "माहिती" in resp_mr
    assert cont_mr in resp_mr


def test_voice_entry_assistant_query_continues_lead_flow(monkeypatch):
    """Voice entry endpoint answers assistant knowledge query AND points to next lead parameter."""
    mock_transcribe = AsyncMock(return_value={"transcript": "What can you do?", "duration": 2.0})
    monkeypatch.setattr(DeepgramSTTService, "transcribe_audio", mock_transcribe)

    files = {"file": ("test.wav", b"dummy audio content", "audio/wav")}
    data = {
        "module": "module1",
        "existing_lead": json.dumps({}),
        "language": "en",
        "is_interim": "false",
        "is_final": "true",
    }
    response = client.post("/api/voice-entry", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert res["is_assistant_query"] is True
    assert res["assistant_response"] is not None
    # Must specify next missing parameter to continue lead flow
    assert res["next_missing_parameter"] in ["name", "phone", "loan_amount", "loan_type", "tenure_months", "company"]
    # Immediate sentence contains both the answer and the continuation question
    assert len(res["immediate_sentence1"]) > len(res["assistant_response"])


# ----------------------------------------------------------------------------
# 6. Dynamic Field-Specific Clarification for Unclear Speech
# ----------------------------------------------------------------------------
def test_field_specific_clarification_prompts():
    """Clarification prompts are specific to the missing field and language."""
    # Phone clarification in Hindi
    hi_phone = get_clarification_prompt("phone", lang="hi")
    assert "10" in hi_phone or "नंबर" in hi_phone or "फ़ोन" in hi_phone

    # Loan amount clarification in Marathi
    mr_amount = get_clarification_prompt("loan_amount", lang="mr")
    assert "रक्कम" in mr_amount or "कर्ज" in mr_amount

    # Tenure clarification in English
    en_tenure = get_clarification_prompt("tenure_months", lang="en")
    assert "tenure" in en_tenure.lower() or "month" in en_tenure.lower() or "year" in en_tenure.lower()


# ----------------------------------------------------------------------------
# 7. Natural Hindi/Marathi TTS Pronunciation
# ----------------------------------------------------------------------------
def test_devanagari_phonetic_dictionary():
    """Phonetic substitution replaces specific financial Devanagari words for natural TTS."""
    marathi_text = "बँकेचे व्याजदर आणि मासिक हप्ते किती आहेत?"
    phonetic = devanagari_to_phonetic(marathi_text)
    assert len(phonetic) > 0
    # Special word phonetic transliteration
    assert devanagari_to_phonetic("हप्ते") == "hapte"
    assert devanagari_to_phonetic("व्याजदर") == "vyaajdar"


# ----------------------------------------------------------------------------
# 8. Conversational Handling for Requested Lead Details (Inquiries & Refusals)
# ----------------------------------------------------------------------------
from services.lead_extractor import is_field_inquiry_or_refusal, LeadExtractorService, extract_company


def test_field_inquiry_detection():
    """Verify is_field_inquiry_or_refusal detects questions, concerns, and refusals in EN, HI, MR."""
    # English questions and refusals
    assert is_field_inquiry_or_refusal("Why do you need my phone number?", "phone")
    assert is_field_inquiry_or_refusal("Why do you need my company name?", "company")
    assert is_field_inquiry_or_refusal("I don't want to give my phone number", "phone")
    assert is_field_inquiry_or_refusal("I refuse to disclose my company", "company")
    assert is_field_inquiry_or_refusal("Is my data safe with you?", "phone")
    assert is_field_inquiry_or_refusal("Why should I give my salary or loan amount?", "loan_amount")
    assert is_field_inquiry_or_refusal("Skip this", "name")
    assert is_field_inquiry_or_refusal("No", "phone")

    # Hindi questions and refusals
    assert is_field_inquiry_or_refusal("फोन नंबर क्यों चाहिए?", "phone")
    assert is_field_inquiry_or_refusal("कंपनी का नाम क्यों पूछ रहे हैं?", "company")
    assert is_field_inquiry_or_refusal("मैं अपना नंबर नहीं दूंगा", "phone")
    assert is_field_inquiry_or_refusal("नहीं बताना चाहता", "company")

    # Marathi questions and refusals
    assert is_field_inquiry_or_refusal("कंपनीचे नाव कशासाठी पाहिजे?", "company")
    assert is_field_inquiry_or_refusal("फोन नंबर कशाला हवा?", "phone")
    assert is_field_inquiry_or_refusal("मी नंबर देणार नाही", "phone")
    assert is_field_inquiry_or_refusal("नाही सांगणार", "company")


def test_field_inquiry_never_saved_as_lead_data():
    """Verify extractors never extract questions or refusals as lead field values."""
    assert extract_company("Why do you need my company name?", context_field="company") is None
    assert extract_company("I don't want to tell my company", context_field="company") is None
    assert extract_company("कंपनीचे नाव कशासाठी पाहिजे?", context_field="company") is None
    assert extract_company("नाही सांगणार", context_field="company") is None
    assert extract_phone_number("Why do you need my phone number?") is None
    assert extract_phone_number("I refuse to give my number") is None
    assert extract_phone_number("फोन नंबर क्यों चाहिए?") is None
    assert extract_name("Why do you need my name?", context_field="name") is None
    assert extract_name("I won't share my name", context_field="name") is None
    assert extract_loan_amount("Why loan amount is needed?", context_field="loan_amount") is None


def test_explain_field_requirement_answers_and_continues():
    """Verify explain_field_requirement provides why detail is needed and appends continuation prompt."""
    extractor = LeadExtractorService()

    # English phone refusal
    resp_en = extractor.explain_field_requirement(
        "I don't want to give my phone number",
        "phone",
        existing_lead={"name": "Rajesh Kumar"},
        language="en",
    )
    assert "phone" in resp_en.lower() or "verification" in resp_en.lower() or "contact" in resp_en.lower()
    # Must ask for phone to continue flow
    assert "phone" in resp_en.lower() or "number" in resp_en.lower()

    # English company question
    resp_comp = extractor.explain_field_requirement(
        "Why do you need my company name?",
        "company",
        existing_lead={"name": "Rajesh Kumar", "phone": "9876543210"},
        language="en",
    )
    assert "company" in resp_comp.lower() or "employer" in resp_comp.lower() or "discount" in resp_comp.lower() or "bank" in resp_comp.lower()

    # Marathi phone question
    resp_mr = extractor.explain_field_requirement(
        "फोन नंबर कशासाठी पाहिजे?",
        "phone",
        existing_lead={"name": "सचिन"},
        language="mr",
    )
    assert "फोन" in resp_mr or "नंबर" in resp_mr or "सुरक्षित" in resp_mr


def test_voice_entry_handles_field_inquiry_and_keeps_pending(monkeypatch):
    """Voice entry endpoint handles field questions/refusals without saving junk and keeps field pending."""
    mock_transcribe = AsyncMock(return_value={"transcript": "Why do you need my company name?", "duration": 2.0})
    monkeypatch.setattr(DeepgramSTTService, "transcribe_audio", mock_transcribe)

    files = {"file": ("test.wav", b"dummy audio content", "audio/wav")}
    data = {
        "module": "module1",
        "existing_lead": json.dumps({"name": "Pooja Sharma", "phone": "9876543210"}),
        "language": "en",
        "is_interim": "false",
        "is_final": "true",
    }
    response = client.post("/api/voice-entry", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert res.get("is_field_inquiry") is True
    assert res.get("is_assistant_query") is True
    # Company is NOT filled with the question!
    assert res["lead"].get("company") is None
    # Company remains the pending field
    assert res["next_missing_parameter"] == "company"
    # Immediate sentence explains why company is needed
    assert "company" in res["immediate_sentence1"].lower() or "employer" in res["immediate_sentence1"].lower() or "discount" in res["immediate_sentence1"].lower()


def test_voice_entry_handles_refusal_and_keeps_pending(monkeypatch):
    """Voice entry endpoint handles refusal, does not save refusal string, stops asking, and pauses flow."""
    mock_transcribe = AsyncMock(return_value={"transcript": "I don't want to give my number", "duration": 2.0})
    monkeypatch.setattr(DeepgramSTTService, "transcribe_audio", mock_transcribe)

    files = {"file": ("test.wav", b"dummy audio content", "audio/wav")}
    data = {
        "module": "module1",
        "existing_lead": json.dumps({"name": "Amit Sharma"}),
        "language": "en",
        "is_interim": "false",
        "is_final": "true",
    }
    response = client.post("/api/voice-entry", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert res.get("is_field_refusal") is True
    assert res.get("is_paused") is True
    assert res.get("is_field_inquiry") is True
    # Refusal string is never saved as phone
    assert res["lead"].get("phone") is None
    # Next missing parameter remains the exact same pending field without moving to another
    assert res["next_missing_parameter"] == "phone"
    # Acknowledges and does not badger user for phone
    imm = res["immediate_sentence1"].lower()
    assert "ready" in imm or "continue" in imm or "let me know" in imm or "okay" in imm


def test_refusal_detection_and_stop_asking():
    """Verify is_field_refusal accurately classifies refusals and handle_field_refusal stops asking."""
    from services.lead_extractor import is_field_refusal, is_flow_resume_intent

    # English refusals
    assert is_field_refusal("don't proceed", "phone")
    assert is_field_refusal("do not proceed", "company")
    assert is_field_refusal("I don't want to give my number", "phone")
    assert is_field_refusal("I refuse", "name")
    assert is_field_refusal("stop", "phone")
    assert is_field_refusal("pause", "company")

    # Hindi refusals
    assert is_field_refusal("आगे मत बढ़ो", "phone")
    assert is_field_refusal("रोक दो", "phone")
    assert is_field_refusal("मैं अपना नंबर नहीं दूंगा", "phone")
    assert is_field_refusal("नहीं बताना", "company")

    # Marathi refusals
    assert is_field_refusal("पुढे जाऊ नका", "phone")
    assert is_field_refusal("थांबा", "phone")
    assert is_field_refusal("मी नंबर देणार नाही", "phone")
    assert is_field_refusal("नाही सांगणार", "company")

    # Flow resume phrases should NOT be detected as refusal
    assert not is_field_refusal("let's continue", "phone")
    assert not is_field_refusal("proceed", "phone")
    assert not is_field_refusal("आगे बढ़ो", "phone")
    assert not is_field_refusal("पुढे चला", "phone")

    # Verify handle_field_refusal produces natural pause response
    extractor = LeadExtractorService()
    resp_en = extractor.handle_field_refusal("don't proceed", "phone", existing_lead={"name": "Rajesh"}, language="en")
    assert "ready" in resp_en.lower() or "continue" in resp_en.lower() or "let me know" in resp_en.lower() or "problem" in resp_en.lower()

    resp_hi = extractor.handle_field_refusal("आगे मत बढ़ो", "phone", existing_lead={"name": "राजेश"}, language="hi")
    assert any(k in resp_hi for k in ("तैयार", "बता", "ठीक है", "बात", "आगे", "राजेश"))

    resp_mr = extractor.handle_field_refusal("पुढे जाऊ नका", "phone", existing_lead={"name": "सचिन"}, language="mr")
    assert any(k in resp_mr for k in ("तयार", "सांगा", "ठीक आहे", "पुढे", "सचिन"))


def test_refusal_does_not_advance_field_or_save_data():
    """Verify extract_lead pauses flow on refusal, keeps same field pending, and never saves refusal."""
    extractor = LeadExtractorService()
    lead_state = {"name": "Priya Nair"}

    result = extractor.extract_lead("don't proceed", existing_lead=lead_state, language="en")
    assert result["status"] == "refusal_paused"
    assert result.get("is_field_refusal") is True
    assert result.get("is_paused") is True
    assert result["lead"].get("phone") is None
    assert result["next_missing_parameter"] == "phone"
    assert "ready" in result["message"].lower() or "continue" in result["message"].lower() or "let me know" in result["message"].lower()


def test_flow_resume_after_refusal():
    """Verify lead flow resumes only when user voluntarily continues and asks for pending field."""
    extractor = LeadExtractorService()
    lead_state = {"name": "Priya Nair"}  # pending: phone

    # 1. User says "let's continue"
    res_en = extractor.extract_lead("let's continue", existing_lead=lead_state, language="en")
    assert res_en["status"] == "resumed"
    assert res_en.get("is_resumed") is True
    assert res_en["next_missing_parameter"] == "phone"
    assert "phone" in res_en["message"].lower() or "number" in res_en["message"].lower() or "Priya" in res_en["message"]

    # 2. Hindi resume "आगे बढ़ो"
    lead_hi = {"name": "अनिल"}
    res_hi = extractor.extract_lead("आगे बढ़ो", existing_lead=lead_hi, language="hi")
    assert res_hi["status"] == "resumed"
    assert res_hi.get("is_resumed") is True
    assert res_hi["next_missing_parameter"] == "phone"


def test_flow_resume_with_data_after_refusal():
    """Verify providing requested information directly after refusal populates field and advances flow."""
    from unittest.mock import MagicMock
    mock_client = MagicMock()
    mock_client.post.return_value.status_code = 200
    mock_client.post.return_value.json.return_value = {
        "choices": [{"message": {"content": json.dumps({"name": "Priya Nair", "phone": "9876543210"})}}]
    }
    extractor = LeadExtractorService(http_client=mock_client)
    lead_state = {"name": "Priya Nair"}  # pending: phone

    # User provides phone number directly
    res = extractor.extract_lead("9876543210", existing_lead=lead_state, language="en")
    assert res["lead"]["phone"] == "9876543210"
    # Advances to company
    assert res["next_missing_parameter"] == "company"


