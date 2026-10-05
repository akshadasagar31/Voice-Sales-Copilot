# ============================================================================
# MODULE 1 COMPREHENSIVE TESTS: LEAD EXTRACTION & UNCLEAR-INPUT HANDLING
# (backend/tests/test_module1_lead_extraction_enhancements.py)
# ============================================================================
# Tests verifying:
# 1. Final vs. interim STT (interim speech is bypassed/ignored).
# 2. English/Hindi/Marathi/mixed number-word normalization.
# 3. Tenure extraction with spoken numbers (“thirty five months” / “पैंतीस महीने” / “पस्तीस महिने” -> 35).
# 4. Spoken phone numbers and loan amounts.
# 5. Invalid prospect names (numbers, tenure, loan details, questions strictly rejected).
# 6. Preserving stored fields (never overwrite valid fields with uncertain data).
# 7. Natural, polite, context-aware clarification for unclear inputs.
# 8. Assistant questions (“What is your name?”, “Who are you?”, “What can you do?”)
#    receiving natural assistant answers and never becoming lead data.
# ============================================================================

import json
import asyncio
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from main import app
from services.lead_extractor import (
    Lead,
    LeadExtractorService,
    extract_phone_number,
    extract_loan_amount,
    extract_tenure_months,
    extract_name,
    is_valid_prospect_name,
    merge_lead_safely,
    get_clarification_prompt,
    get_missing_parameter_prompt,
)
from services.number_normalizer import (
    normalize_spoken_numbers,
    parse_numeric_phrase,
)
from services.language import (
    is_assistant_query,
    get_assistant_query_response,
)

client = TestClient(app)


# ----------------------------------------------------------------------------
# 1. Final vs. Interim STT Tests
# ----------------------------------------------------------------------------
def test_interim_stt_bypasses_extraction_in_service():
    """Verify that is_interim=True bypasses LLM and returns interim_ignored without modifying lead."""
    service = LeadExtractorService(api_key="mock_key")
    mock_http = MagicMock()
    service._client = mock_http

    existing = {"name": "Rajesh", "phone": "9876543210"}
    res = service.extract_lead(
        "my name is Rajesh and I want personal loan",
        existing_lead=existing,
        is_interim=True,
    )
    assert res["status"] == "interim_ignored"
    assert res["is_interim"] is True
    # LLM must not have been called
    mock_http.post.assert_not_called()
    assert res["lead"]["name"] == "Rajesh"


def test_non_final_stt_bypasses_extraction_in_service():
    """Verify that is_final=False bypasses LLM and returns interim_ignored."""
    service = LeadExtractorService(api_key="mock_key")
    mock_http = MagicMock()
    service._client = mock_http

    res = service.extract_lead(
        "looking for 5 lakh",
        is_final=False,
    )
    assert res["status"] == "interim_ignored"
    assert res["is_interim"] is True
    mock_http.post.assert_not_called()


def test_api_extract_lead_endpoint_interim_ignored():
    """Verify POST /api/extract-lead with is_interim=True returns interim_ignored."""
    payload = {
        "transcript": "hello I want",
        "is_interim": True,
        "existing_lead": {"name": "Amit"},
    }
    resp = client.post("/api/extract-lead", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "interim_ignored"
    assert data["is_interim"] is True
    assert data["lead"]["name"] == "Amit"


# ----------------------------------------------------------------------------
# 2. Spoken Number-Word Normalization Tests
# ----------------------------------------------------------------------------
def test_number_word_normalization_english():
    """Verify English spoken numbers normalize to digits."""
    assert normalize_spoken_numbers("thirty five months") == "35 months"
    assert normalize_spoken_numbers("twenty four months") == "24 months"
    assert normalize_spoken_numbers("three years") == "3 years"
    assert normalize_spoken_numbers("fifty thousand") == "50 thousand"
    assert normalize_spoken_numbers("twenty five lakh") == "25 lakh"


def test_number_word_normalization_hindi():
    """Verify Hindi Devanagari and Romanized numbers normalize to digits."""
    assert normalize_spoken_numbers("पैंतीस महीने") == "35 महीने"
    assert normalize_spoken_numbers("चौबीस महीने") == "24 महीने"
    assert normalize_spoken_numbers("तीन साल") == "3 साल"
    assert normalize_spoken_numbers("पचास हजार") == "50 हजार"
    assert normalize_spoken_numbers("पच्चीस लाख") == "25 लाख"
    # Romanized
    assert normalize_spoken_numbers("paintees mahine") == "35 mahine"
    assert normalize_spoken_numbers("tees mahine") == "30 mahine"


def test_number_word_normalization_marathi():
    """Verify Marathi Devanagari and Romanized numbers normalize to digits."""
    assert normalize_spoken_numbers("पस्तीस महिने") == "35 महिने"
    assert normalize_spoken_numbers("चोवीस महिने") == "24 महिने"
    assert normalize_spoken_numbers("दोन वर्षे") == "2 वर्षे"
    assert normalize_spoken_numbers("पंचवीस लाख") == "25 लाख"
    # Romanized
    assert normalize_spoken_numbers("pastees mahine") == "35 mahine"
    assert normalize_spoken_numbers("don varsha") == "2 varsha"


# ----------------------------------------------------------------------------
# 3. Tenure Extraction Tests
# ----------------------------------------------------------------------------
def test_tenure_extraction_spoken_numbers():
    """
    Verify spoken numbers in tenure:
    “thirty five months” / “पैंतीस महीने” / “पस्तीस महिने” -> 35 months.
    """
    assert extract_tenure_months("thirty five months") == 35
    assert extract_tenure_months("पैंतीस महीने") == 35
    assert extract_tenure_months("पस्तीस महिने") == 35
    assert extract_tenure_months("tenure of thirty five months") == 35
    assert extract_tenure_months("duration is twenty four months") == 24
    assert extract_tenure_months("तीन साल") == 36
    assert extract_tenure_months("three years") == 36
    assert extract_tenure_months("दोन वर्षे") == 24
    # With context_field="tenure_months"
    assert extract_tenure_months("thirty five", context_field="tenure_months") == 35
    assert extract_tenure_months("पैंतीस", context_field="tenure_months") == 35
    assert extract_tenure_months("पस्तीस", context_field="tenure_months") == 35


# ----------------------------------------------------------------------------
# 4. Spoken Phone & Loan Amount Extraction Tests
# ----------------------------------------------------------------------------
def test_phone_number_extraction_spoken_digits():
    """Verify spoken digit sequences are normalized and extracted as 10-digit phones."""
    en_phone = "my number is nine eight seven six five four three two one zero"
    assert extract_phone_number(en_phone) == "9876543210"

    hi_phone = "मेरा फोन नंबर नौ आठ सात छह पाँच चार तीन दो एक शून्य है"
    assert extract_phone_number(hi_phone) == "9876543210"

    mr_phone = "माझा नंबर नऊ आठ सात सहा पाच चार तीन दोन एक शून्य आहे"
    assert extract_phone_number(mr_phone) == "9876543210"


def test_phone_number_extraction_preserves_spoken_leading_zeroes():
    """Module 1 preserves an explicitly dictated contact sequence verbatim."""
    spoken_contact = "zero zero nine six eight nine one"
    assert normalize_spoken_numbers(spoken_contact) == "0096891"
    assert extract_phone_number(spoken_contact) == "0096891"

    from main import generate_module1_ws_response
    response = generate_module1_ws_response(
        spoken_contact,
        req_lang="en",
        last_conf=0.99,
        existing_lead_str=json.dumps({"name": "Asha"}),
    )
    assert response["lead"]["phone"] == "0096891"


def test_module1_deterministic_response_is_sent_once():
    """A loan-intent turn emits one final response, not a final plus sentence events."""
    from main import _send_pipelined_mod1_response, generate_module1_ws_response

    class RecordingWebSocket:
        def __init__(self):
            self.messages = []

        async def send_json(self, payload):
            self.messages.append(payload)

    payload = generate_module1_ws_response("I want a loan", req_lang="en", last_conf=0.99)
    ws = RecordingWebSocket()
    asyncio.run(_send_pipelined_mod1_response(ws, payload))

    assert len(ws.messages) == 1
    assert ws.messages[0]["type"] == "final"
    assert ws.messages[0]["has_subsequent_sentences"] is False
    assert ws.messages[0]["immediate_sentence1"]


def test_loan_amount_extraction_spoken_words():
    """Verify spoken words in loan amounts are extracted correctly."""
    assert extract_loan_amount("twenty five lakh") == 2500000.0
    assert extract_loan_amount("पच्चीस लाख") == 2500000.0
    assert extract_loan_amount("पंचवीस लाख") == 2500000.0
    assert extract_loan_amount("fifty thousand") == 50000.0
    assert extract_loan_amount("पचास हजार") == 50000.0
    assert extract_loan_amount("two crore") == 20000000.0
    assert extract_loan_amount("दोन कोटी") == 20000000.0
    assert extract_loan_amount("2 crore") == 20000000.0


# ----------------------------------------------------------------------------
# 5. Strict Prospect Name Validation Tests
# ----------------------------------------------------------------------------
def test_never_save_tenure_numbers_loan_or_questions_as_prospect_name():
    """
    Never save tenure, numbers, loan details, or questions as the prospect name.
    """
    # Spoken tenure numbers
    assert is_valid_prospect_name("thirty five months") is False
    assert is_valid_prospect_name("पैंतीस महीने") is False
    assert is_valid_prospect_name("पस्तीस महिने") is False
    assert is_valid_prospect_name("35 months") is False
    assert is_valid_prospect_name("24 months") is False
    assert is_valid_prospect_name("three years") is False
    assert is_valid_prospect_name("दोन वर्षे") is False

    # Standalone numbers
    assert is_valid_prospect_name("thirty five") is False
    assert is_valid_prospect_name("पैंतीस") is False
    assert is_valid_prospect_name("पस्तीस") is False
    assert is_valid_prospect_name("twenty five lakh") is False
    assert is_valid_prospect_name("500000") is False

    # Loan intent phrases
    assert is_valid_prospect_name("I want loan") is False
    assert is_valid_prospect_name("personal loan") is False
    assert is_valid_prospect_name("home loan") is False
    assert is_valid_prospect_name("लोन चाहिए") is False
    assert is_valid_prospect_name("कर्ज पाहिजे") is False

    # Assistant questions
    assert is_valid_prospect_name("What is your name?") is False
    assert is_valid_prospect_name("Who are you?") is False
    assert is_valid_prospect_name("What can you do?") is False
    assert is_valid_prospect_name("आप कौन हैं?") is False
    assert is_valid_prospect_name("तुमचे नाव काय आहे?") is False

    # Valid real prospect names
    assert is_valid_prospect_name("Rajesh Sharma") is True
    assert is_valid_prospect_name("अमित कुमार") is True
    assert is_valid_prospect_name("राहुल पाटील") is True
    assert is_valid_prospect_name("Priya Verma") is True


# ----------------------------------------------------------------------------
# 6. Preserving Stored Fields (Never Overwrite with Uncertain Data)
# ----------------------------------------------------------------------------
def test_never_overwrite_correctly_stored_fields():
    """
    Verify that merge_lead_safely preserves valid existing fields
    even if incoming data has None, empty string, or invalid candidates.
    """
    existing = {
        "name": "Rajesh Sharma",
        "phone": "9876543210",
        "company": "TCS",
        "loan_type": "Personal Loan",
        "loan_amount": 500000.0,
        "tenure_months": 24,
    }

    # Incoming has None for all existing fields and invalid candidate for name
    incoming = {
        "name": "thirty five months",  # invalid name candidate!
        "phone": None,
        "company": None,
        "loan_type": None,
        "loan_amount": None,
        "tenure_months": 36,  # valid update
    }

    merged = merge_lead_safely(existing, incoming)
    assert merged["name"] == "Rajesh Sharma"  # Preserved!
    assert merged["phone"] == "9876543210"    # Preserved!
    assert merged["company"] == "TCS"          # Preserved!
    assert merged["loan_type"] == "Personal Loan"  # Preserved!
    assert merged["loan_amount"] == 500000.0  # Preserved!
    assert merged["tenure_months"] == 36       # Updated!


def test_service_extract_lead_preserves_valid_fields_on_unclear_turn():
    """Verify extract_lead preserves existing fields when user gives unclear input."""
    service = LeadExtractorService(api_key="mock_key")
    mock_http = MagicMock()
    service._client = mock_http

    # Existing lead has name and phone
    existing = {
        "name": "Rajesh Sharma",
        "phone": "9876543210",
        "company": None,
        "loan_type": None,
        "loan_amount": None,
        "tenure_months": None,
    }

    # Mock LLM returns empty/null for everything
    mock_llm_json = {
        "name": None,
        "phone": None,
        "company": None,
        "loan_type": None,
        "loan_amount": None,
        "tenure_months": None,
        "notes": None,
    }
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"choices": [{"message": {"content": json.dumps(mock_llm_json)}}]}
    mock_http.post.return_value = mock_resp

    # User mumbles or gives unclear response
    res = service.extract_lead("ummm well not sure", existing_lead=existing)
    lead_out = res["lead"]
    assert lead_out["name"] == "Rajesh Sharma"  # MUST NOT BE WIPED!
    assert lead_out["phone"] == "9876543210"    # MUST NOT BE WIPED!


# ----------------------------------------------------------------------------
# 7. Unclear Input & Context-Aware Clarification Tests
# ----------------------------------------------------------------------------
def test_unclear_input_generates_context_aware_clarification():
    """
    If any detail is unclear, never guess. Generate a natural, polite,
    context-aware clarification for that specific field in the appropriate language.
    Do NOT hardcode or repeat one fixed clarification message every time.
    """
    # 1. Unclear name
    prompt_name_en1 = get_clarification_prompt("name", raw_transcript="what", lang="en", attempt=1)
    prompt_name_en2 = get_clarification_prompt("name", raw_transcript="xyz", lang="en", attempt=2)
    assert "name" in prompt_name_en1.lower()
    # Ensure variations exist
    prompts_name = {get_clarification_prompt("name", lang="en", attempt=i) for i in range(10)}
    assert len(prompts_name) > 1  # Not hardcoded to one single message!

    # 2. Unclear phone in Hindi
    prompt_phone_hi = get_clarification_prompt("phone", lang="hi", attempt=1)
    assert "फोन" in prompt_phone_hi or "मोबाइल" in prompt_phone_hi or "नंबर" in prompt_phone_hi
    prompts_phone_hi = {get_clarification_prompt("phone", lang="hi", attempt=i) for i in range(10)}
    assert len(prompts_phone_hi) > 1

    # 3. Unclear tenure in Marathi
    prompt_tenure_mr = get_clarification_prompt("tenure_months", lang="mr", attempt=1)
    assert "कालावधी" in prompt_tenure_mr or "मुदत" in prompt_tenure_mr or "महिने" in prompt_tenure_mr
    prompts_tenure_mr = {get_clarification_prompt("tenure_months", lang="mr", attempt=i) for i in range(10)}
    assert len(prompts_tenure_mr) > 1


def test_service_generates_clarification_when_field_is_unclear():
    """Verify service sets is_unclear=True and uses clarification prompt when answer is unclear."""
    service = LeadExtractorService(api_key="mock_key")
    mock_http = MagicMock()
    service._client = mock_http

    # Existing lead has name; missing field is phone
    existing = {"name": "Rajesh", "phone": None}

    # User says something unclear like "hello hello" which extracts no phone
    mock_llm_json = {"name": None, "phone": None, "company": None, "loan_type": None, "loan_amount": None, "tenure_months": None}
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"choices": [{"message": {"content": json.dumps(mock_llm_json)}}]}
    mock_http.post.return_value = mock_resp

    res = service.extract_lead("can you hear me", existing_lead=existing, language="en")
    assert res["is_unclear"] is True
    assert res["clarified_field"] == "phone"
    assert "phone" in res["message"].lower() or "mobile" in res["message"].lower() or "contact" in res["message"].lower()


# ----------------------------------------------------------------------------
# 8. Assistant Questions Tests
# ----------------------------------------------------------------------------
def test_assistant_questions_receive_natural_answers_never_lead_data():
    """
    Assistant questions such as “What is your name?”, “Who are you?”, or “What can you do?”
    must receive natural assistant answers and must never become lead data.
    """
    service = LeadExtractorService(api_key="mock_key")
    mock_http = MagicMock()
    service._client = mock_http

    # Turn with "What is your name?"
    res1 = service.extract_lead("What is your name?", language="en")
    assert res1["status"] == "assistant_query"
    assert res1["is_assistant_query"] is True
    assert "voicecopilot" in res1["message"].lower()
    assert res1["lead"]["name"] is None  # NOT saved as prospect name!
    mock_http.post.assert_not_called()

    # Turn with "Who are you?"
    res2 = service.extract_lead("Who are you?", language="en")
    assert res2["status"] == "assistant_query"
    assert res2["is_assistant_query"] is True
    assert "sales copilot" in res2["message"].lower() or "voicecopilot" in res2["message"].lower()
    mock_http.post.assert_not_called()

    # Turn with "What can you do?"
    res3 = service.extract_lead("What can you do?", language="en")
    assert res3["status"] == "assistant_query"
    assert res3["is_assistant_query"] is True
    assert "lead" in res3["message"].lower() or "crm" in res3["message"].lower()
    mock_http.post.assert_not_called()

    # Hindi: "आप कौन हैं?"
    res_hi = service.extract_lead("आप कौन हैं?", language="hi")
    assert res_hi["status"] == "assistant_query"
    assert "कोपायलट" in res_hi["message"]

    # Marathi: "तुम्ही काय करू शकता?"
    res_mr = service.extract_lead("तुम्ही काय करू शकता?", language="mr")
    assert res_mr["status"] == "assistant_query"
    assert "सीआरएम" in res_mr["message"] or "मदत" in res_mr["message"]


# ----------------------------------------------------------------------------
# 9. Module 1 Regression Tests: Spoken digits, Company rejection & Never Overwrite
# ----------------------------------------------------------------------------
from services.lead_extractor import (
    is_valid_company_name,
    extract_company,
    is_valid_phone_number,
    is_field_value_valid,
)
from main import generate_module1_ws_response


def test_spoken_digits_double_five_and_leading_zeros():
    """
    Test:
    “Double five, four zero nine six six nine one” must be extracted as a phone number
    and saved with all digits/leading zeros preserved.
    """
    transcript1 = "Double five, four zero nine six six nine one"
    normalized1 = normalize_spoken_numbers(transcript1)
    phone1 = extract_phone_number(transcript1)
    assert "554096691" in normalized1
    assert phone1 == "554096691"

    # Leading zero dictation preserved
    transcript2 = "zero zero nine six six nine one four zero"
    normalized2 = normalize_spoken_numbers(transcript2)
    phone2 = extract_phone_number(transcript2)
    assert "009669140" in normalized2
    assert phone2 == "009669140"


def test_company_this_is_alone_rejected_and_flow_does_not_advance():
    """
    Test:
    - “This is” alone is NOT a company name and must NOT advance the flow.
    - Ask again for the company name when the user has not provided a valid company.
    """
    # 1. Validation check
    assert is_valid_company_name("This is") is False
    assert is_valid_company_name("It is") is False
    assert is_valid_company_name("This") is False
    assert extract_company("This is", context_field="company") is None

    # 2. Flow check: user is on company step
    existing_lead = {"name": "Akshada", "phone": "554096691"}
    resp = generate_module1_ws_response(
        current="This is",
        req_lang="en",
        last_conf=0.95,
        existing_lead_str=json.dumps(existing_lead),
    )

    # Must NOT advance to loan_type
    assert resp["next_missing_parameter"] == "company"
    assert resp["is_unclear"] is True
    assert resp["lead"]["name"] == "Akshada"
    assert resp["lead"]["phone"] == "554096691"
    assert resp["lead"].get("company") is None
    # Must ask again for company name
    sentence1 = resp["immediate_sentence1"].lower()
    assert "company" in sentence1 or "employer" in sentence1 or "work" in sentence1 or "organization" in sentence1


def test_valid_company_advances_flow():
    """When a valid company name is provided, flow advances to loan_type."""
    existing_lead = {"name": "Akshada", "phone": "554096691"}
    resp = generate_module1_ws_response(
        current="I work at Infosys",
        req_lang="en",
        last_conf=0.95,
        existing_lead_str=json.dumps(existing_lead),
    )
    assert resp["next_missing_parameter"] == "loan_type"
    assert resp["is_unclear"] is False
    assert resp["lead"]["company"] == "Infosys"
    assert resp["lead"]["name"] == "Akshada"
    assert resp["lead"]["phone"] == "554096691"


def test_never_overwrite_valid_previously_collected_fields_on_subsequent_turns():
    """
    Never overwrite valid previously collected fields (name, phone, company)
    with conversational filler or subsequent statements.
    """
    existing_lead = {
        "name": "Akshada",
        "phone": "554096691",
        "company": "Infosys",
    }
    # Turn 4: user specifies loan type
    resp = generate_module1_ws_response(
        current="This is a personal loan",
        req_lang="en",
        last_conf=0.95,
        existing_lead_str=json.dumps(existing_lead),
    )
    # Existing fields must remain intact
    assert resp["lead"]["name"] == "Akshada"
    assert resp["lead"]["phone"] == "554096691"
    assert resp["lead"]["company"] == "Infosys"
    assert resp["lead"]["loan_type"] == "Personal Loan"
    assert resp["next_missing_parameter"] == "loan_amount"


def test_multilingual_phone_extraction_repeated_digits_and_smart_times():
    """
    Test multilingual phone extraction:
    - Repeater words in English ('double', 'triple') and Devanagari ('डबल', 'ट्रिपल').
    - Smart-formatted times like '9:00' -> 900 when clearly part of phone speech.
    - True appointment scheduling times like 'call me at 9:00 AM' -> not extracted as phone.
    """
    # 1. Smart-formatted times inside phone dictation:
    assert extract_phone_number("phone is 9876543 9:00") == "9876543900"
    assert extract_phone_number("my number is 98 2:00 45678") == "9820045678"

    # 2. English repeater words:
    assert extract_phone_number("double nine eight seven six five four three two one") == "9987654321"
    assert extract_phone_number("triple eight seven six five four three two one") == "8887654321"

    # 3. Hindi repeater words:
    assert extract_phone_number("डबल नौ आठ सात छह पाँच चार तीन दो एक") == "9987654321"
    assert extract_phone_number("ट्रिपल आठ सात छह पाँच चार तीन दो एक") == "8887654321"

    # 4. Marathi repeater words:
    assert extract_phone_number("डबल नऊ आठ सात सहा पाच चार तीन दोन एक") == "9987654321"

    # 5. True scheduling times must NOT be extracted as phone numbers:
    assert extract_phone_number("call me at 9:00 AM") is None
    assert extract_phone_number("call at 2:00 PM tomorrow") is None


def test_multilingual_phone_leading_zeros_and_validation():
    """
    Preserve leading zeros and enforce 7-15 digit phone validity.
    """
    assert extract_phone_number("0096891234") == "0096891234"
    assert extract_phone_number("0987654321") == "0987654321"
    assert is_valid_phone_number("0096891234") is True
    assert is_valid_phone_number("0987654321") is True
    assert is_valid_phone_number("12345") is False
    assert is_valid_phone_number("abcdefghij") is False


def test_lead_model_validation_rejects_invalid_inputs():
    """
    Ensure Lead Pydantic model rejects invalid inputs such as:
    - 'This is' / 'It is' / 'हे आहे' / 'यह है' for company
    - Under-length strings for phone
    """
    l1 = Lead.model_validate({"company": "This is"})
    assert l1.company is None

    l2 = Lead.model_validate({"company": "It is"})
    assert l2.company is None

    l3 = Lead.model_validate({"company": "हे आहे"})
    assert l3.company is None

    l4 = Lead.model_validate({"company": "यह है"})
    assert l4.company is None

    l5 = Lead.model_validate({"company": "Google India"})
    assert l5.company == "Google India"

    l6 = Lead.model_validate({"phone": "123"})
    assert l6.phone is None

    l7 = Lead.model_validate({"phone": "0096891234"})
    assert l7.phone == "0096891234"


def test_save_valid_out_of_order_fields_without_advancing_flow():
    """
    Validate the CURRENT requested field before advancing:
    - Save valid out-of-order fields (e.g. loan_type, loan_amount),
    - but keep the current field (phone) as next_missing_parameter until valid.
    """
    existing_lead = {"name": "Rajesh"}  # next missing is phone
    # User provides loan details out of order:
    resp1 = generate_module1_ws_response(
        current="I want a personal loan of 5 lakh",
        req_lang="en",
        last_conf=0.95,
        existing_lead_str=json.dumps(existing_lead),
    )

    # 1. Out-of-order fields must be saved:
    assert resp1["lead"]["loan_type"] == "Personal Loan"
    assert resp1["lead"]["loan_amount"] == 500000.0
    assert resp1["lead"]["name"] == "Rajesh"
    assert resp1["lead"].get("phone") is None

    # 2. Flow MUST NOT advance past phone:
    assert resp1["next_missing_parameter"] == "phone"
    assert resp1["is_unclear"] is True
    # Immediate sentence must prompt for phone
    phone_prompt = resp1["immediate_sentence1"].lower()
    assert "phone" in phone_prompt or "number" in phone_prompt or "contact" in phone_prompt

    # 3. Next turn: user provides phone number
    resp2 = generate_module1_ws_response(
        current="My phone number is 9876543210",
        req_lang="en",
        last_conf=0.95,
        existing_lead_str=json.dumps(resp1["lead"]),
    )
    # Phone is now captured:
    assert resp2["lead"]["phone"] == "9876543210"
    assert resp2["is_unclear"] is False
    # Since loan_type and loan_amount are already captured, it advances straight to company!
    assert resp2["next_missing_parameter"] == "company"


def test_invalid_company_does_not_overwrite_or_advance_flow():
    """
    Never overwrite valid fields or advance on invalid inputs like 'This is' for company.
    """
    existing_lead = {
        "name": "Rajesh",
        "phone": "9876543210",
        "company": "Tata Consultancy Services",
    }
    # User utters 'This is' on subsequent turn:
    resp = generate_module1_ws_response(
        current="This is",
        req_lang="en",
        last_conf=0.95,
        existing_lead_str=json.dumps(existing_lead),
    )
    # Company must NOT be overwritten by 'This is'
    assert resp["lead"]["company"] == "Tata Consultancy Services"
    assert resp["lead"]["name"] == "Rajesh"
    assert resp["lead"]["phone"] == "9876543210"


def test_model_deletion_guardrails():
    """
    Test:
    - Allow permanently deleting any model (built-in or custom, active or inactive).
    - Clear active selection if active model is deleted.
    - Restore defaults via /api/models/restore-defaults.
    """
    # 1. Add custom model
    add_resp = client.post(
        "/api/models",
        json={
            "category": "stt",
            "name": "Custom STT Test",
            "provider": "Deepgram",
            "model_id": "nova-2-custom-test",
        },
    )
    assert add_resp.status_code == 201
    created_id = add_resp.json()["model"]["id"]

    try:
        # 2. Set as active, then delete -> allowed (200) and clears active model
        act_resp = client.post("/api/models/activate", json={"category": "stt", "model_id": created_id})
        assert act_resp.status_code == 200

        del_active_resp = client.delete(f"/api/models/{created_id}")
        assert del_active_resp.status_code == 200
        assert del_active_resp.json()["status"] == "success"
        assert client.get("/api/models/active").json()["active"]["stt"] is None

        # 3. Built-in model deletion allowed (200)
        resp_builtin = client.delete("/api/models/deepgram-nova-3")
        assert resp_builtin.status_code == 200
        assert resp_builtin.json()["status"] == "success"
    finally:
        # Restore defaults
        client.post("/api/models/restore-defaults")
