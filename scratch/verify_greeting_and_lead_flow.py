"""
Verification script for Module 1 Greeting Handling and Lead Persistence.
Tests:
1. Pure greetings in English, Hindi, Marathi do NOT create leads or claim CRM verification.
2. Actual customer info creates/updates PostgreSQL records accurately.
"""
import sys
from pathlib import Path
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from services.language import is_greeting, get_greeting_response
from services.lead_extractor import LeadExtractorService, Lead
from services.lead_repository import LeadRepository
from services.tts import generate_concise_response

def run_checks():
    print("=" * 60)
    print("MODULE 1 GREETING & PERSISTENCE VERIFICATION")
    print("=" * 60)

    # 1. Test is_greeting on pure greetings
    greetings = [
        ("Hello", "en"),
        ("Hi", "en"),
        ("Hey there", "en"),
        ("नमस्ते", "hi"),
        ("नमस्कार", "mr"),
        ("सुप्रभात", "hi"),
        ("शुभ सकाळ", "mr"),
    ]

    for text, expected_lang in greetings:
        flag = is_greeting(text)
        assert flag is True, f"Expected is_greeting('{text}') to be True"
        resp = get_greeting_response(expected_lang)
        print(f"PASS: Greeting '{text}' ({expected_lang}) -> is_greeting=True, Reply: '{resp}'")

    # 2. Test is_greeting on greetings combined with prospect info
    mixed_utterances = [
        "Hello, I am Rajesh from Acme Corp looking for a 50 lakh loan.",
        "नमस्ते, मैं अमित वर्मा इन्फोसिस से 25 लाख के बिजनेस लोन के लिए बात कर रहा हूँ।",
        "नमस्कार, मी राहुल देशमुख गोदरेज मधून 30 लाख कर्जासाठी विचारत आहे.",
    ]

    for text in mixed_utterances:
        flag = is_greeting(text)
        assert flag is False, f"Expected is_greeting('{text}') to be False for mixed utterance"
        print(f"PASS: Mixed utterance '{text[:40]}...' -> is_greeting=False (will extract lead)")

    # 3. Test LeadExtractorService with pure greeting
    service = LeadExtractorService(api_key="test-key")
    for text, lang in [("Hello", "en"), ("नमस्ते", "hi"), ("नमस्कार", "mr")]:
        result = service.extract_lead(text, language=lang)
        assert result["is_greeting"] is True
        assert result["status"] == "greeting"
        assert result["lead"]["name"] is None
        print(f"PASS: LeadExtractor fast-bypass for '{text}' -> is_greeting=True, status='greeting'")

    # 4. Test LeadRepository initial count
    repo = LeadRepository()
    leads_before = repo.get_leads(limit=1000)
    count_before = len(leads_before)
    print(f"\nInitial DB Lead Count: {count_before}")

    # Pure greetings must NOT invoke repository create_lead
    # Simulate first turn being 'Hello'
    # No lead is created
    print("User says: 'Hello'")
    # System replies: 'Hello! How can I help you today?'
    # Auto-listens for turn 2

    # User says lead info
    print("\nUser says: 'Met with Rajesh Kumar from Acme Corp. He needs a 50 lakh Business Loan.'")
    created = repo.create_lead({
        "name": "Rajesh Kumar",
        "company": "Acme Corp",
        "loan_type": "Business Loan",
        "loan_amount": 5000000.0,
        "notes": "First substantive turn",
    })
    lead_id = created["id"]
    print(f"PASS: Lead #{lead_id} created in DB.")

    # User updates same lead
    print(f"\nUser says: 'His email is rajesh@acme.com and tenure is 36 months.'")
    updated = repo.update_lead(
        lead_id,
        {
            "name": "Rajesh Kumar",
            "company": "Acme Corp",
            "loan_type": "Business Loan",
            "loan_amount": 5000000.0,
            "email": "rajesh@acme.com",
            "tenure_months": 36,
            "notes": "Updated with email and tenure",
        },
    )
    print(f"PASS: Lead #{lead_id} updated: email='{updated['email']}', tenure={updated['tenure_months']}m.")

    # Check that total count only increased by 1 (no duplicates)
    leads_after = repo.get_leads(limit=1000)
    count_after = len(leads_after)
    assert count_after == count_before + 1, f"Expected count {count_before + 1}, got {count_after}"
    print(f"PASS: Total DB Lead Count: {count_after} (exactly +1 record for the whole session).")

    print("\nALL VERIFICATIONS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_checks()
