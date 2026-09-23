import pytest
import json
from unittest.mock import MagicMock, patch
from pydantic import ValidationError
from fastapi.testclient import TestClient

from main import app
from services.lead_extractor import (
    Lead,
    LeadExtractorService,
    LeadExtractionError,
    LeadValidationError,
    MalformedLLMOutputError,
    OpenRouterConfigurationError,
    OpenRouterAPIError,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Pydantic Model Unit Tests
# ---------------------------------------------------------------------------

def test_pydantic_lead_model_valid_full():
    """Verify Lead model accepts and validates all 9 fields with correct types."""
    lead = Lead(
        name="Rahul Sharma",
        phone="+91-9876543210",
        email="rahul.sharma@tcs.com",
        company="Tata Consultancy Services",
        role="Senior Project Manager",
        loan_type="Personal Loan",
        loan_amount=500000.0,
        tenure_months=36,
        notes="Customer wants quick disbursal for home renovation.",
    )
    assert lead.name == "Rahul Sharma"
    assert lead.phone == "+91-9876543210"
    assert lead.email == "rahul.sharma@tcs.com"
    assert lead.company == "Tata Consultancy Services"
    assert lead.role == "Senior Project Manager"
    assert lead.loan_type == "Personal Loan"
    assert lead.loan_amount == 500000.0
    assert lead.tenure_months == 36
    assert lead.notes == "Customer wants quick disbursal for home renovation."


def test_pydantic_lead_model_missing_fields_default_to_null():
    """Verify missing fields strictly default to None (null) and are never guessed."""
    lead = Lead(name="Priya Patel")
    assert lead.name == "Priya Patel"
    assert lead.phone is None
    assert lead.email is None
    assert lead.company is None
    assert lead.role is None
    assert lead.loan_type is None
    assert lead.loan_amount is None
    assert lead.tenure_months is None
    assert lead.notes is None

    # Empty instantiation
    empty_lead = Lead()
    assert empty_lead.name is None
    assert empty_lead.loan_amount is None


def test_pydantic_lead_model_clean_placeholder_strings():
    """Verify placeholder strings like 'N/A', 'none', 'null', '' are normalized to None."""
    lead = Lead(
        name="Amit Kumar",
        phone="N/A",
        email="none",
        company="null",
        role="unknown",
        loan_type="unspecified",
        loan_amount="N/A",
        tenure_months="none",
        notes="",
    )
    assert lead.name == "Amit Kumar"
    assert lead.phone is None
    assert lead.email is None
    assert lead.company is None
    assert lead.role is None
    assert lead.loan_type is None
    assert lead.loan_amount is None
    assert lead.tenure_months is None
    assert lead.notes is None


def test_pydantic_lead_model_currency_and_tenure_parsing():
    """Verify currency symbols/commas and tenure phrases are sanitized to numeric types."""
    lead = Lead(
        loan_amount="$50,000.50",
        tenure_months="48 months",
    )
    assert lead.loan_amount == 50000.50
    assert lead.tenure_months == 48


def test_pydantic_lead_model_validation_error_invalid_loan_amount():
    """Verify invalid loan_amount format raises Pydantic ValidationError."""
    with pytest.raises(ValidationError):
        Lead(loan_amount="not_a_valid_number")


def test_pydantic_lead_model_validation_error_invalid_tenure():
    """Verify invalid tenure_months format raises Pydantic ValidationError."""
    with pytest.raises(ValidationError):
        Lead(tenure_months="no tenure mentioned")


# ---------------------------------------------------------------------------
# 2. Service Unit Tests with Mocked HTTP / LLM
# ---------------------------------------------------------------------------

def test_service_valid_lead_extraction():
    """Verify LeadExtractorService successfully parses and validates a complete lead."""
    mock_llm_json = {
        "name": "Sarah Jenkins",
        "phone": "555-0199",
        "email": "sarah.jenkins@fintech.co",
        "company": "FinTech Global",
        "role": "Director of Finance",
        "loan_type": "Personal Loan",
        "loan_amount": 75000.0,
        "tenure_months": 24,
        "notes": "Looking for low APR personal loan to consolidate tech equipment purchase.",
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": json.dumps(mock_llm_json)}}]
    }

    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp

    service = LeadExtractorService(api_key="mock_key", http_client=mock_client)
    result = service.extract_lead("Sarah Jenkins from FinTech Global called regarding a 75000 personal loan for 24 months.")

    assert result["status"] == "success"
    lead_data = result["lead"]
    assert lead_data["name"] == "Sarah Jenkins"
    assert lead_data["phone"] == "555-0199"
    assert lead_data["email"] == "sarah.jenkins@fintech.co"
    assert lead_data["company"] == "FinTech Global"
    assert lead_data["role"] == "Director of Finance"
    assert lead_data["loan_type"] == "Personal Loan"
    assert lead_data["loan_amount"] == 75000.0
    assert lead_data["tenure_months"] == 24
    assert "consolidate" in lead_data["notes"].lower()


def test_service_missing_fields_remain_null():
    """Verify fields not mentioned in transcript are strictly null in extracted lead."""
    mock_llm_json = {
        "name": "David Miller",
        "phone": None,
        "email": None,
        "company": None,
        "role": None,
        "loan_type": "Home Loan",
        "loan_amount": 250000.0,
        "tenure_months": None,
        "notes": None,
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": json.dumps(mock_llm_json)}}]
    }

    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp

    service = LeadExtractorService(api_key="mock_key", http_client=mock_client)
    result = service.extract_lead("David Miller called asking about a home loan of 250000.")

    lead_data = result["lead"]
    assert lead_data["name"] == "David Miller"
    assert lead_data["loan_type"] == "Home Loan"
    assert lead_data["loan_amount"] == 250000.0
    assert lead_data["phone"] is None
    assert lead_data["email"] is None
    assert lead_data["company"] is None
    assert lead_data["role"] is None
    assert lead_data["tenure_months"] is None
    assert lead_data["notes"] is None


def test_service_markdown_code_block_wrapped_json():
    """Verify LLM outputs enclosed in ```json ... ``` are cleanly extracted."""
    raw_markdown = """```json
{
  "name": "Alex Mercer",
  "phone": "+1-415-555-2671",
  "email": "alex@mercer.dev",
  "company": "Mercer Labs",
  "role": "Founder & CTO",
  "loan_type": "Business Loan",
  "loan_amount": 100000,
  "tenure_months": 36,
  "notes": "Startup expansion funding"
}
```"""

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": raw_markdown}}]
    }

    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp

    service = LeadExtractorService(api_key="mock_key", http_client=mock_client)
    result = service.extract_lead("Alex Mercer from Mercer Labs needs a 100k business loan.")

    assert result["lead"]["name"] == "Alex Mercer"
    assert result["lead"]["loan_amount"] == 100000.0
    assert result["lead"]["tenure_months"] == 36


def test_service_malformed_llm_output_raises_error():
    """Verify non-JSON response from LLM raises MalformedLLMOutputError."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "I couldn't extract any lead because the transcript is incomplete."}}]
    }

    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp

    service = LeadExtractorService(api_key="mock_key", http_client=mock_client)
    with pytest.raises(MalformedLLMOutputError) as exc_info:
        service.extract_lead("Hello, can you help me?")

    assert "LLM did not return valid JSON" in str(exc_info.value)


def test_service_pydantic_validation_error_raises_lead_validation_error():
    """Verify LLM returning unparseable schema fields raises LeadValidationError."""
    invalid_llm_json = {
        "name": "Test User",
        "loan_amount": "invalid_non_numeric_amount",
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": json.dumps(invalid_llm_json)}}]
    }

    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp

    service = LeadExtractorService(api_key="mock_key", http_client=mock_client)
    with pytest.raises(LeadValidationError) as exc_info:
        service.extract_lead("Test User wants a loan.")

    assert "Lead data failed Pydantic validation" in str(exc_info.value)
    assert exc_info.value.errors is not None


def test_service_empty_transcript_validation():
    """Verify empty or whitespace transcript raises ValueError."""
    service = LeadExtractorService(api_key="mock_key")
    with pytest.raises(ValueError) as exc_info:
        service.extract_lead("   ")
    assert "cannot be empty" in str(exc_info.value)


def test_service_missing_api_key():
    """Verify missing API key raises OpenRouterConfigurationError."""
    service = LeadExtractorService(api_key="")
    with pytest.raises(OpenRouterConfigurationError) as exc_info:
        service.extract_lead("Transcript text")
    assert "OPENROUTER_API_KEY is not set" in str(exc_info.value)


def test_service_openrouter_api_error():
    """Verify OpenRouter HTTP error raises OpenRouterAPIError."""
    mock_resp = MagicMock()
    mock_resp.status_code = 502
    mock_resp.text = "Bad Gateway from OpenRouter"

    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp

    service = LeadExtractorService(api_key="mock_key", http_client=mock_client)
    with pytest.raises(OpenRouterAPIError) as exc_info:
        service.extract_lead("Transcript text")
    assert "HTTP 502" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 3. FastAPI Endpoint Integration Tests (/api/extract-lead)
# ---------------------------------------------------------------------------

def test_api_extract_lead_endpoint_success():
    """Verify POST /api/extract-lead returns HTTP 200 and structured lead JSON."""
    mock_llm_json = {
        "name": "Marcus Vance",
        "phone": "555-9011",
        "email": "marcus@apex.com",
        "company": "Apex Logistics",
        "role": "COO",
        "loan_type": "Commercial Fleet Loan",
        "loan_amount": 120000.0,
        "tenure_months": 60,
        "notes": "Acquisition of 4 delivery vans",
    }

    with patch("services.lead_extractor.httpx.Client") as mock_http_class:
        mock_instance = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps(mock_llm_json)}}]
        }
        mock_instance.post.return_value = mock_resp
        mock_http_class.return_value = mock_instance

        resp = client.post(
            "/api/extract-lead",
            json={
                "transcript": "Marcus Vance, COO of Apex Logistics, phone 555-9011, wants a commercial fleet loan for 120000 over 60 months."
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["lead"]["name"] == "Marcus Vance"
        assert data["lead"]["company"] == "Apex Logistics"
        assert data["lead"]["loan_amount"] == 120000.0
        assert data["lead"]["tenure_months"] == 60


def test_api_extract_lead_endpoint_empty_transcript():
    """Verify POST /api/extract-lead with empty transcript returns HTTP 400."""
    resp = client.post(
        "/api/extract-lead",
        json={"transcript": "   "},
    )
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_api_extract_lead_endpoint_malformed_llm_output():
    """Verify POST /api/extract-lead handles malformed LLM outputs with HTTP 502."""
    with patch("services.lead_extractor.httpx.Client") as mock_http_class:
        mock_instance = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Not a valid JSON string"}}]
        }
        mock_instance.post.return_value = mock_resp
        mock_http_class.return_value = mock_instance

        resp = client.post(
            "/api/extract-lead",
            json={"transcript": "Hello, customer calling here."},
        )
        assert resp.status_code == 502
        assert "malformed" in resp.json()["detail"].lower()


def test_api_extract_lead_endpoint_pydantic_validation_error():
    """Verify POST /api/extract-lead handles Pydantic validation failures with HTTP 422."""
    with patch("services.lead_extractor.httpx.Client") as mock_http_class:
        mock_instance = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps({"loan_amount": "invalid_letters"})}}]
        }
        mock_instance.post.return_value = mock_resp
        mock_http_class.return_value = mock_instance

        resp = client.post(
            "/api/extract-lead",
            json={"transcript": "Customer said loan amount is invalid letters."},
        )
        assert resp.status_code == 422
        error_detail = resp.json()["detail"]
        assert "Pydantic lead validation failed" in error_detail["error"]


def test_extract_lead_with_existing_lead_merging():
    """Verify extract_lead preserves existing non-null lead fields when integrating new turns."""
    service = LeadExtractorService(api_key="mock_key")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    # LLM returns updated loan amount but omits name/company (or outputs null)
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "name": None,
                        "company": None,
                        "loan_type": "Equipment Loan",
                        "loan_amount": 5000000.0,
                        "tenure_months": 36,
                        "notes": "Acme equipment expansion.",
                    })
                }
            }
        ]
    }

    mock_http = MagicMock()
    mock_http.post.return_value = mock_resp
    service._client = mock_http

    existing = {
        "name": "Rajesh Kumar",
        "company": "Acme Corporation",
        "role": "VP Technology",
        "loan_amount": None,
    }

    result = service.extract_lead(
        transcript="Acme confirmed they need a 5 million equipment loan over 36 months.",
        existing_lead=existing,
    )

    lead = result["lead"]
    # Verify prior facts are retained through programmatic defense
    assert lead["name"] == "Rajesh Kumar"
    assert lead["company"] == "Acme Corporation"
    assert lead["role"] == "VP Technology"
    # Verify new facts are integrated
    assert lead["loan_type"] == "Equipment Loan"
    assert lead["loan_amount"] == 5000000.0
    assert lead["tenure_months"] == 36
    assert lead["notes"] == "Acme equipment expansion."


def test_api_extract_lead_with_existing_lead_and_language():
    """Verify POST /api/extract-lead accepts existing_lead and language payloads."""
    with patch("services.lead_extractor.httpx.Client") as mock_http_class:
        mock_instance = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "name": "अमित वर्मा",
                            "company": "Infosys",
                            "loan_type": "Business Loan",
                            "loan_amount": 2500000.0,
                            "tenure_months": 24,
                            "notes": "Hindi speech extraction test",
                        })
                    }
                }
            ]
        }
        mock_instance.post.return_value = mock_resp
        mock_http_class.return_value = mock_instance

        resp = client.post(
            "/api/extract-lead",
            json={
                "transcript": "अमित वर्मा को 25 लाख का बिजनेस लोन चाहिए।",
                "existing_lead": {"company": "Infosys"},
                "language": "hi",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["lead"]["name"] == "अमित वर्मा"
        assert data["lead"]["company"] == "Infosys"
        assert data["lead"]["loan_amount"] == 2500000.0


def test_extract_lead_greeting_only_english():
    """Verify English pure greetings return is_greeting=True without LLM call."""
    service = LeadExtractorService(api_key="test-key")
    for greeting in ["Hello", "Hi", "Hey", "Good morning", "Hello there"]:
        res = service.extract_lead(greeting, language="en")
        assert res["is_greeting"] is True
        assert res["status"] == "greeting"
        assert "Hello! How can I help you today?" in res["message"]
        assert res["lead"]["name"] is None
        assert res["lead"]["company"] is None


def test_extract_lead_greeting_only_hindi():
    """Verify Hindi pure greetings return is_greeting=True and natural Hindi greeting."""
    service = LeadExtractorService(api_key="test-key")
    for greeting in ["नमस्ते", "हेलो", "सुप्रभात", "नमस्कार जी"]:
        res = service.extract_lead(greeting, language="hi")
        assert res["is_greeting"] is True
        assert res["status"] == "greeting"
        assert "नमस्ते! मैं आज आपकी क्या सहायता कर सकता हूँ?" in res["message"]
        assert res["lead"]["name"] is None


def test_extract_lead_greeting_only_marathi():
    """Verify Marathi pure greetings return is_greeting=True and natural Marathi greeting."""
    service = LeadExtractorService(api_key="test-key")
    for greeting in ["नमस्कार", "हॅलो", "शुभ सकाळ", "नमस्कार सर"]:
        res = service.extract_lead(greeting, language="mr")
        assert res["is_greeting"] is True
        assert res["status"] == "greeting"
        assert "नमस्कार! मी आज आपली काय मदत करू शकेन?" in res["message"]
        assert res["lead"]["name"] is None


def test_extract_lead_greeting_with_prospect_info():
    """Verify greeting accompanied by actual customer info is NOT bypassed."""
    service = LeadExtractorService(api_key="test-key")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "name": "Rajesh Kumar",
                        "company": "Acme Corp",
                        "loan_type": "Personal Loan",
                        "loan_amount": 5000000.0,
                        "tenure_months": 36,
                        "notes": "Greeting + lead info provided",
                    })
                }
            }
        ]
    }
    mock_http = MagicMock()
    mock_http.post.return_value = mock_resp
    service._client = mock_http

    res = service.extract_lead("Hello, I am Rajesh from Acme Corp, looking for a 50 lakh personal loan.")
    assert res.get("is_greeting") is not True
    assert res["status"] == "success"
    assert res["lead"]["name"] == "Rajesh Kumar"
    assert res["lead"]["company"] == "Acme Corp"


def test_api_extract_lead_greeting_endpoint():
    """Verify POST /api/extract-lead returns HTTP 200 with is_greeting=True for greetings."""
    resp = client.post(
        "/api/extract-lead",
        json={"transcript": "Hello", "language": "en"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_greeting"] is True
    assert data["status"] == "greeting"
    assert data["lead"]["name"] is None


def test_api_extract_lead_streaming_greeting():
    """Verify POST /api/extract-lead with stream=True returns SSE stream for greeting."""
    resp = client.post(
        "/api/extract-lead",
        json={"transcript": "Hello", "language": "en", "stream": True},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    body = resp.text
    assert "event: metadata" in body
    assert "event: token" in body
    assert "event: done" in body


def test_stream_lead_turn_substantive_flow():
    """Verify stream_lead_turn extracts lead, emits 'lead' event, and streams tokens."""
    service = LeadExtractorService(api_key="mock_key")
    mock_llm_json = {
        "name": "Sunil Gupta",
        "company": "Gupta Enterprises",
        "loan_type": "Business Loan",
        "loan_amount": 2000000.0,
        "tenure_months": 24,
        "notes": "Fast processing needed",
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": json.dumps(mock_llm_json)}}]
    }
    mock_http = MagicMock()
    mock_http.post.return_value = mock_resp
    service._client = mock_http

    chunks = list(service.stream_lead_turn(
        transcript="Sunil Gupta from Gupta Enterprises needs a 20 lakh business loan for 24 months.",
        language="en",
    ))
    joined = "".join(chunks)
    assert "event: lead" in joined
    assert "Sunil Gupta" in joined
    assert "event: token" in joined
    assert "event: done" in joined


def test_stream_lead_turn_single_pass_delimiter():
    """Verify single-pass streaming correctly separates spoken tokens from lead JSON payload."""
    from services.lead_extractor import LEAD_STREAM_DELIMITER
    service = LeadExtractorService(api_key="mock_key")
    mock_lead = {
        "name": "Pooja Sharma",
        "company": "Pooja Designs",
        "loan_type": "Personal Loan",
        "loan_amount": 500000.0,
        "tenure_months": 12,
        "phone": "9876543210",
        "email": "pooja@designs.com",
        "role": "Owner",
        "notes": "Urgent funding",
    }
    mock_content = (
        f"I have recorded the loan inquiry for Pooja Sharma from Pooja Designs.\n"
        f"{LEAD_STREAM_DELIMITER}\n"
        f"```json\n{json.dumps(mock_lead)}\n```"
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": mock_content}}]
    }
    mock_http = MagicMock()
    mock_http.post.return_value = mock_resp
    service._client = mock_http

    chunks = list(service.stream_lead_turn(
        transcript="Pooja Sharma from Pooja Designs needs a 5 lakh personal loan.",
        language="en",
    ))
    joined = "".join(chunks)

    assert "event: token" in joined
    assert "Pooja Sharma from Pooja Designs" in joined
    assert LEAD_STREAM_DELIMITER not in joined  # Delimiter must never leak into client tokens
    assert "event: lead" in joined
    assert "pooja@designs.com" in joined
    assert "event: done" in joined


def test_sequential_stateful_parameter_collection_multi_turn():
    """
    Verify Module 1 sequential stateful parameter collection:
    Turn 1: User gives name only -> saved, asks for phone.
    Turn 2: User gives phone -> state merged, asks for company.
    Turn 3: User gives company -> state merged, asks for loan_type.
    Turn 4: User gives loan_type -> state merged, asks for loan_amount.
    Turn 5: User gives loan_amount -> state merged, asks for tenure_months.
    Turn 6: User gives tenure -> state complete, gives short thank-you confirmation.
    """
    from services.lead_extractor import LEAD_STREAM_DELIMITER

    service = LeadExtractorService(api_key="mock_key")
    mock_http = MagicMock()
    service._client = mock_http

    # --- Turn 1: User gives name only ---
    t1_json = {"name": "Rajesh Sharma", "phone": None, "company": None, "loan_type": None, "loan_amount": None, "tenure_months": None}
    t1_content = f"Thank you, Rajesh Sharma. What is your contact phone number?\n{LEAD_STREAM_DELIMITER}\n{json.dumps(t1_json)}"
    mock_resp1 = MagicMock(status_code=200)
    mock_resp1.json.return_value = {"choices": [{"message": {"content": t1_content}}]}
    mock_http.post.return_value = mock_resp1

    chunks1 = list(service.stream_lead_turn(
        transcript="My name is Rajesh Sharma",
        language="en",
    ))
    joined1 = "".join(chunks1)
    assert "Thank you, Rajesh Sharma. What is your contact phone number?" in joined1
    assert "phone" in joined1  # next_missing_parameter in lead event

    lead_events1 = [json.loads(c.replace("event: lead\ndata: ", "").strip()) for c in chunks1 if c.startswith("event: lead")]
    assert len(lead_events1) == 1
    lead1 = lead_events1[0]["lead"]
    assert lead1["name"] == "Rajesh Sharma"
    assert lead_events1[0]["next_missing_parameter"] == "phone"
    assert lead_events1[0]["is_complete"] is False

    # --- Turn 2: User provides phone number ---
    t2_json = {"name": "Rajesh Sharma", "phone": "9876543210", "company": None, "loan_type": None, "loan_amount": None, "tenure_months": None}
    t2_content = f"Thank you, Rajesh Sharma. What company or organization do you work for?\n{LEAD_STREAM_DELIMITER}\n{json.dumps(t2_json)}"
    mock_resp2 = MagicMock(status_code=200)
    mock_resp2.json.return_value = {"choices": [{"message": {"content": t2_content}}]}
    mock_http.post.return_value = mock_resp2

    chunks2 = list(service.stream_lead_turn(
        transcript="9876543210",
        existing_lead=lead1,
        language="en",
    ))
    joined2 = "".join(chunks2)
    assert "What company or organization do you work for?" in joined2
    lead_events2 = [json.loads(c.replace("event: lead\ndata: ", "").strip()) for c in chunks2 if c.startswith("event: lead")]
    lead2 = lead_events2[0]["lead"]
    assert lead2["name"] == "Rajesh Sharma"
    assert lead2["phone"] == "9876543210"
    assert lead_events2[0]["next_missing_parameter"] == "company"
    assert lead_events2[0]["is_complete"] is False

    # --- Turn 3: User provides company ---
    t3_json = {"name": "Rajesh Sharma", "phone": "9876543210", "company": "Infosys", "loan_type": None, "loan_amount": None, "tenure_months": None}
    t3_content = f"Thank you. What type of loan are you looking for, such as a personal loan, home loan, or business loan?\n{LEAD_STREAM_DELIMITER}\n{json.dumps(t3_json)}"
    mock_resp3 = MagicMock(status_code=200)
    mock_resp3.json.return_value = {"choices": [{"message": {"content": t3_content}}]}
    mock_http.post.return_value = mock_resp3

    chunks3 = list(service.stream_lead_turn(
        transcript="I work at Infosys",
        existing_lead=lead2,
        language="en",
    ))
    joined3 = "".join(chunks3)
    assert "What type of loan are you looking for" in joined3
    lead_events3 = [json.loads(c.replace("event: lead\ndata: ", "").strip()) for c in chunks3 if c.startswith("event: lead")]
    lead3 = lead_events3[0]["lead"]
    assert lead3["company"] == "Infosys"
    assert lead_events3[0]["next_missing_parameter"] == "loan_type"
    assert lead_events3[0]["is_complete"] is False

    # --- Turn 4: User provides loan type ---
    t4_json = {"name": "Rajesh Sharma", "phone": "9876543210", "company": "Infosys", "loan_type": "Personal Loan", "loan_amount": None, "tenure_months": None}
    t4_content = f"Understood. What loan amount do you require for your Personal Loan?\n{LEAD_STREAM_DELIMITER}\n{json.dumps(t4_json)}"
    mock_resp4 = MagicMock(status_code=200)
    mock_resp4.json.return_value = {"choices": [{"message": {"content": t4_content}}]}
    mock_http.post.return_value = mock_resp4

    chunks4 = list(service.stream_lead_turn(
        transcript="I need a personal loan",
        existing_lead=lead3,
        language="en",
    ))
    joined4 = "".join(chunks4)
    assert "What loan amount do you require" in joined4
    lead_events4 = [json.loads(c.replace("event: lead\ndata: ", "").strip()) for c in chunks4 if c.startswith("event: lead")]
    lead4 = lead_events4[0]["lead"]
    assert lead4["loan_type"] == "Personal Loan"
    assert lead_events4[0]["next_missing_parameter"] == "loan_amount"
    assert lead_events4[0]["is_complete"] is False

    # --- Turn 5: User provides loan amount ---
    t5_json = {"name": "Rajesh Sharma", "phone": "9876543210", "company": "Infosys", "loan_type": "Personal Loan", "loan_amount": 500000.0, "tenure_months": None}
    t5_content = f"What loan tenure or repayment duration (in months or years) are you looking for?\n{LEAD_STREAM_DELIMITER}\n{json.dumps(t5_json)}"
    mock_resp5 = MagicMock(status_code=200)
    mock_resp5.json.return_value = {"choices": [{"message": {"content": t5_content}}]}
    mock_http.post.return_value = mock_resp5

    chunks5 = list(service.stream_lead_turn(
        transcript="5 lakh rupees",
        existing_lead=lead4,
        language="en",
    ))
    joined5 = "".join(chunks5)
    assert "What loan tenure or repayment duration" in joined5
    lead_events5 = [json.loads(c.replace("event: lead\ndata: ", "").strip()) for c in chunks5 if c.startswith("event: lead")]
    lead5 = lead_events5[0]["lead"]
    assert lead5["loan_amount"] == 500000.0
    assert lead_events5[0]["next_missing_parameter"] == "tenure_months"
    assert lead_events5[0]["is_complete"] is False

    # --- Turn 6: User provides tenure -> All 6 collected -> Thank-you confirmation ---
    t6_json = {"name": "Rajesh Sharma", "phone": "9876543210", "company": "Infosys", "loan_type": "Personal Loan", "loan_amount": 500000.0, "tenure_months": 24}
    t6_content = f"Thank you, Rajesh Sharma! All your details have been recorded. Our team will contact you shortly.\n{LEAD_STREAM_DELIMITER}\n{json.dumps(t6_json)}"
    mock_resp6 = MagicMock(status_code=200)
    mock_resp6.json.return_value = {"choices": [{"message": {"content": t6_content}}]}
    mock_http.post.return_value = mock_resp6

    chunks6 = list(service.stream_lead_turn(
        transcript="2 years",
        existing_lead=lead5,
        language="en",
    ))
    joined6 = "".join(chunks6)
    assert "Thank you, Rajesh Sharma! All your details have been recorded." in joined6
    lead_events6 = [json.loads(c.replace("event: lead\ndata: ", "").strip()) for c in chunks6 if c.startswith("event: lead")]
    lead6 = lead_events6[0]["lead"]
    assert lead6["name"] == "Rajesh Sharma"
    assert lead6["phone"] == "9876543210"
    assert lead6["company"] == "Infosys"
    assert lead6["loan_type"] == "Personal Loan"
    assert lead6["loan_amount"] == 500000.0
    assert lead6["tenure_months"] == 24
    assert lead_events6[0]["next_missing_parameter"] is None
    assert lead_events6[0]["is_complete"] is True


def test_sequential_stateful_hindi_prompts_and_thank_you():
    """Verify sequential prompts and thank-you confirmation in Hindi for all 6 required fields."""
    from services.lead_extractor import get_missing_parameter_prompt, get_next_missing_parameter

    lead = {"name": "राजेश शर्मा", "phone": None, "company": None, "loan_type": None, "loan_amount": None, "tenure_months": None}
    assert get_next_missing_parameter(lead) == "phone"
    p_phone = get_missing_parameter_prompt("phone", lead, lang="hi")
    assert "फोन नंबर" in p_phone

    lead["phone"] = "9876543210"
    assert get_next_missing_parameter(lead) == "company"
    p_comp = get_missing_parameter_prompt("company", lead, lang="hi")
    assert "कंपनी" in p_comp

    lead["company"] = "टाटा मोटर्स"
    assert get_next_missing_parameter(lead) == "loan_type"
    p_loan = get_missing_parameter_prompt("loan_type", lead, lang="hi")
    assert "लोन" in p_loan

    lead["loan_type"] = "होम लोन"
    assert get_next_missing_parameter(lead) == "loan_amount"
    p_amount = get_missing_parameter_prompt("loan_amount", lead, lang="hi")
    assert "लोन राशि" in p_amount

    lead["loan_amount"] = 2500000.0
    assert get_next_missing_parameter(lead) == "tenure_months"
    p_tenure = get_missing_parameter_prompt("tenure_months", lead, lang="hi")
    assert "अवधि" in p_tenure

    lead["tenure_months"] = 60
    assert get_next_missing_parameter(lead) is None
    p_complete = get_missing_parameter_prompt(None, lead, lang="hi")
    assert "धन्यवाद" in p_complete
    assert "दर्ज कर ली गई है" in p_complete


def test_sequential_stateful_marathi_prompts_and_thank_you():
    """Verify sequential prompts and thank-you confirmation in Marathi for all 6 required fields."""
    from services.lead_extractor import get_missing_parameter_prompt, get_next_missing_parameter

    lead = {"name": "राजेश शर्मा", "phone": None, "company": None, "loan_type": None, "loan_amount": None, "tenure_months": None}
    assert get_next_missing_parameter(lead) == "phone"
    p_phone = get_missing_parameter_prompt("phone", lead, lang="mr")
    assert "फोन नंबर" in p_phone

    lead["phone"] = "9876543210"
    assert get_next_missing_parameter(lead) == "company"
    p_comp = get_missing_parameter_prompt("company", lead, lang="mr")
    assert "कंपनी" in p_comp

    lead["company"] = "इन्फोसिस"
    assert get_next_missing_parameter(lead) == "loan_type"
    p_loan = get_missing_parameter_prompt("loan_type", lead, lang="mr")
    assert "कर्ज" in p_loan

    lead["loan_type"] = "वैयक्तिक कर्ज"
    assert get_next_missing_parameter(lead) == "loan_amount"
    p_amount = get_missing_parameter_prompt("loan_amount", lead, lang="mr")
    assert "रकमेचे कर्ज" in p_amount

    lead["loan_amount"] = 500000.0
    assert get_next_missing_parameter(lead) == "tenure_months"
    p_tenure = get_missing_parameter_prompt("tenure_months", lead, lang="mr")
    assert "कालावधी" in p_tenure

    lead["tenure_months"] = 36
    assert get_next_missing_parameter(lead) is None
    p_complete = get_missing_parameter_prompt(None, lead, lang="mr")
    assert "धन्यवाद" in p_complete
    assert "नोंदवले गेले आहेत" in p_complete


def test_extract_company_and_tenure_multilingual():
    """Verify deterministic safety net extractors for company and tenure in English, Hindi, and Marathi."""
    from services.lead_extractor import extract_company, extract_tenure_months

    # Company
    assert extract_company("My company is Tata Consultancy Services") == "Tata Consultancy Services"
    assert extract_company("I am working at Wipro Technologies") == "Wipro Technologies"
    assert extract_company("मेरी कंपनी इंफोसिस है") == "इंफोसिस"
    assert extract_company("टीसीएस मध्ये काम करतो") == "टीसीएस"

    # Tenure
    assert extract_tenure_months("I need it for 2 years") == 24
    assert extract_tenure_months("duration of 36 months") == 36
    assert extract_tenure_months("5 साल के लिए") == 60
    assert extract_tenure_months("12 महीने") == 12
    assert extract_tenure_months("3 वर्षे कालावधी") == 36
    assert "3 वर्षे कालावधी" in "3 वर्षे कालावधी"
    assert extract_tenure_months("24 महिने") == 24


def test_loan_intent_phrases_never_saved_as_prospect_name():
    """
    Verify bug fix: Phrases like 'I want loan' are NEVER saved as prospect name.
    Name must remain None, and copilot must ask for the name.
    """
    from services.lead_extractor import is_valid_prospect_name, extract_name

    # 1. Direct validator tests
    assert is_valid_prospect_name("I want loan") is False
    assert is_valid_prospect_name("I want a loan") is False
    assert is_valid_prospect_name("Need loan") is False
    assert is_valid_prospect_name("Personal loan") is False
    assert is_valid_prospect_name("Loan chahiye") is False
    assert is_valid_prospect_name("Mala karj pahije") is False
    assert is_valid_prospect_name("Looking for loan") is False
    assert is_valid_prospect_name("Rajesh") is True
    assert is_valid_prospect_name("Rajesh Sharma") is True
    assert is_valid_prospect_name("अमित वर्मा") is True
    assert is_valid_prospect_name("राहुल पाटील") is True

    # 2. extract_name regex guard tests
    assert extract_name("I want loan") is None
    assert extract_name("I want a loan") is None
    assert extract_name("Need loan") is None
    assert extract_name("Loan chahiye") is None
    assert extract_name("Mala karj pahije") is None
    assert extract_name("My name is Rajesh") == "Rajesh"
    assert extract_name("मेरा नाम अमित है") == "अमित"

    # 3. Pydantic model validation tests
    l1 = Lead(name="I want loan")
    assert l1.name is None

    l2 = Lead(name="Personal loan")
    assert l2.name is None

    l3 = Lead(name="Rajesh")
    assert l3.name == "Rajesh"

    # 4. Service extract_lead tests with mock LLM returning invalid name
    service = LeadExtractorService(api_key="mock_key")
    mock_http = MagicMock()
    service._client = mock_http

    # Simulate an LLM erroneously returning name='I want loan'
    bad_llm_json = {
        "name": "I want loan",
        "phone": None,
        "company": None,
        "loan_type": "Personal Loan",
        "loan_amount": 500000.0,
        "tenure_months": None,
        "notes": None,
    }
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"choices": [{"message": {"content": json.dumps(bad_llm_json)}}]}
    mock_http.post.return_value = mock_resp

    res = service.extract_lead("I want loan")
    lead_out = res["lead"]
    assert lead_out["name"] is None, "Name must be sanitized to None!"
    assert res["next_missing_parameter"] == "name", "Next missing parameter must be name!"
    assert "name" in res["message"].lower() or "नाम" in res["message"] or "नाव" in res["message"]


def test_explicit_real_name_extraction_and_multi_turn_merge():
    """
    Verify that explicit real names are preserved and multi-turn state merges
    phone, company, loan type, amount, and tenure without overwriting existing fields.
    """
    service = LeadExtractorService(api_key="mock_key")
    mock_http = MagicMock()
    service._client = mock_http

    # Turn 1: User says 'I want loan' (with loan type and amount)
    t1_json = {"name": None, "phone": None, "company": None, "loan_type": "Personal Loan", "loan_amount": 500000.0, "tenure_months": None}
    mock_resp1 = MagicMock(status_code=200)
    mock_resp1.json.return_value = {"choices": [{"message": {"content": json.dumps(t1_json)}}]}
    mock_http.post.return_value = mock_resp1

    turn1 = service.extract_lead("I want personal loan of 5 lakh")
    t1_lead = turn1["lead"]
    assert t1_lead["name"] is None
    assert t1_lead["loan_type"] == "Personal Loan"
    assert t1_lead["loan_amount"] == 500000.0
    assert turn1["next_missing_parameter"] == "name"

    # Turn 2: User says 'My name is Rajesh'
    t2_json = {"name": "Rajesh", "phone": None, "company": None, "loan_type": None, "loan_amount": None, "tenure_months": None}
    mock_resp2 = MagicMock(status_code=200)
    mock_resp2.json.return_value = {"choices": [{"message": {"content": json.dumps(t2_json)}}]}
    mock_http.post.return_value = mock_resp2

    turn2 = service.extract_lead("My name is Rajesh", existing_lead=t1_lead)
    t2_lead = turn2["lead"]
    assert t2_lead["name"] == "Rajesh"
    assert t2_lead["loan_type"] == "Personal Loan"  # Preserved from Turn 1
    assert t2_lead["loan_amount"] == 500000.0        # Preserved from Turn 1
    assert turn2["next_missing_parameter"] == "phone"

    # Turn 3: User provides phone number
    t3_json = {"name": None, "phone": "9876543210", "company": None, "loan_type": None, "loan_amount": None, "tenure_months": None}
    mock_resp3 = MagicMock(status_code=200)
    mock_resp3.json.return_value = {"choices": [{"message": {"content": json.dumps(t3_json)}}]}
    mock_http.post.return_value = mock_resp3

    turn3 = service.extract_lead("9876543210", existing_lead=t2_lead)
    t3_lead = turn3["lead"]
    assert t3_lead["name"] == "Rajesh"          # Preserved
    assert t3_lead["phone"] == "9876543210"     # Newly added
    assert t3_lead["loan_type"] == "Personal Loan" # Preserved
    assert t3_lead["loan_amount"] == 500000.0   # Preserved
    assert turn3["next_missing_parameter"] == "company"

    # Turn 4: User provides company and tenure
    t4_json = {"name": None, "phone": None, "company": "TCS", "loan_type": None, "loan_amount": None, "tenure_months": 36}
    mock_resp4 = MagicMock(status_code=200)
    mock_resp4.json.return_value = {"choices": [{"message": {"content": json.dumps(t4_json)}}]}
    mock_http.post.return_value = mock_resp4

    turn4 = service.extract_lead("I work at TCS for 36 months", existing_lead=t3_lead)
    t4_lead = turn4["lead"]
    assert t4_lead["name"] == "Rajesh"          # Preserved
    assert t4_lead["phone"] == "9876543210"     # Preserved
    assert t4_lead["company"] == "TCS"          # Newly added
    assert t4_lead["loan_type"] == "Personal Loan" # Preserved
    assert t4_lead["loan_amount"] == 500000.0   # Preserved
    assert t4_lead["tenure_months"] == 36       # Newly added
    assert turn4["is_complete"] is True





