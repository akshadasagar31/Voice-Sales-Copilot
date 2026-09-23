import json
import httpx

BASE_URL = "http://127.0.0.1:8001"

def test_live_module1_tts():
    print("\n========================================================")
    print("1. VERIFYING MODULE 1 TTS WITH SARVAM BULBUL V3 'simran'")
    print("========================================================")

    test_cases = [
        ("en", "Thank you, Rajesh! All your details have been recorded.", "en-IN"),
        ("hi", "धन्यवाद राजेश जी! आपकी सभी जानकारी दर्ज कर ली गई है।", "hi-IN"),
        ("mr", "धन्यवाद राजेश जी! आपले सर्व तपशील नोंदवले गेले आहेत.", "mr-IN"),
    ]

    for lang, text, expected_lang_code in test_cases:
        payload = {
            "text": text,
            "language": lang,
            # Notice: No module="module2" passed; this is Module 1
        }
        res = httpx.post(f"{BASE_URL}/api/tts", json=payload, timeout=20.0)
        assert res.status_code == 200, f"Failed for {lang}: HTTP {res.status_code}"
        provider = res.headers.get("x-tts-provider")
        voice = res.headers.get("x-tts-voice")
        tts_lang = res.headers.get("x-tts-language")
        content_type = res.headers.get("content-type")
        audio_len = len(res.content)

        print(f"[{lang.upper()}] Status: {res.status_code} | Provider: {provider} | Voice: {voice} | Lang: {tts_lang} | Content-Type: {content_type} | Audio bytes: {audio_len}")
        assert provider == "sarvam", f"Expected provider 'sarvam', got '{provider}'"
        assert voice == "simran", f"Expected voice 'simran', got '{voice}'"
        assert tts_lang == expected_lang_code, f"Expected language '{expected_lang_code}', got '{tts_lang}'"
        assert audio_len > 1000, "Audio response is suspiciously small"

    print("ALL MODULE 1 TTS CHECKS PASSED (en-IN, hi-IN, mr-IN with simran)!")


def test_live_module1_sequential_stateful_flow():
    print("\n========================================================")
    print("2. VERIFYING MODULE 1 SEQUENTIAL & STATEFUL COLLECTION")
    print("========================================================")

    turns = [
        ("Turn 1 (Name only)", "My name is Rajesh Sharma", "phone"),
        ("Turn 2 (Phone)", "9876543210", "loan_type"),
        ("Turn 3 (Loan Type)", "I am looking for a personal loan", "loan_amount"),
        ("Turn 4 (Loan Amount)", "I need 5 lakh rupees", None),
    ]

    current_lead = None
    lead_id = None

    for turn_name, transcript, expected_next_missing in turns:
        print(f"\n--- {turn_name}: User says '{transcript}' ---")
        payload = {
            "transcript": transcript,
            "existing_lead": current_lead,
            "lead_id": lead_id,
            "language": "en",
            "stream": True,
        }

        spoken_tokens = []
        lead_event_data = None
        done_event_data = None

        with httpx.stream("POST", f"{BASE_URL}/api/extract-lead", json=payload, timeout=30.0) as resp:
            assert resp.status_code == 200, f"Turn failed with status {resp.status_code}"
            current_event = ""
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("event: "):
                    current_event = line[7:].strip()
                elif line.startswith("data: "):
                    data_str = line[6:].strip()
                    try:
                        data = json.loads(data_str)
                        if current_event == "token":
                            spoken_tokens.append(data.get("token", ""))
                        elif current_event == "lead":
                            lead_event_data = data
                        elif current_event == "done":
                            done_event_data = data
                    except Exception:
                        pass

        full_spoken = "".join(spoken_tokens).strip()
        print(f"Assistant Spoken: \"{full_spoken}\"")
        assert lead_event_data is not None, "Missing event: lead"
        current_lead = lead_event_data["lead"]
        lead_id = lead_event_data["lead_id"]
        next_missing = lead_event_data.get("next_missing_parameter")
        is_complete = lead_event_data.get("is_complete")

        print(f"Saved Lead State in DB (ID #{lead_id}):")
        print(f"  Name:        {current_lead.get('name')}")
        print(f"  Phone:       {current_lead.get('phone')}")
        print(f"  Loan Type:   {current_lead.get('loan_type')}")
        print(f"  Loan Amount: {current_lead.get('loan_amount')}")
        print(f"  Next Missing Parameter: {next_missing}")
        print(f"  Is Complete: {is_complete}")

        assert next_missing == expected_next_missing, f"Expected next missing '{expected_next_missing}', got '{next_missing}'"

        if expected_next_missing is None:
            assert is_complete is True
            assert any(thx in full_spoken.lower() for thx in ["thank", "recorded", "noted"]), "Turn 4 missing thank-you confirmation"
        else:
            assert is_complete is False

    print("\nALL SEQUENTIAL STATEFUL FLOW CHECKS PASSED!")


if __name__ == "__main__":
    test_live_module1_tts()
    test_live_module1_sequential_stateful_flow()
    print("\nSUCCESS: All live verification checks for Module 1 passed with 100% compliance!")
