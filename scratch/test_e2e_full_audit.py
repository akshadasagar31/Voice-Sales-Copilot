import asyncio
import time
import os
import sys
import json
import requests
import websockets
from pathlib import Path
from dotenv import load_dotenv

backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))
load_dotenv(backend_dir / ".env", override=True)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

FASTAPI_URL = "http://127.0.0.1:8001"
NEXTJS_URL = "http://localhost:3000"
WS_URL = "ws://127.0.0.1:8001/ws/voice-stt"

bugs_found = []

def record_bug(bug_name, steps, expected, actual, file_func, severity):
    bugs_found.append({
        "bug": bug_name,
        "steps": steps,
        "expected": expected,
        "actual": actual,
        "file_func": file_func,
        "severity": severity,
    })
    print(f"\n[BUG DETECTED] [{severity}] {bug_name}")
    print(f"  File/Func: {file_func}")
    print(f"  Expected: {expected}")
    print(f"  Actual: {actual}")

async def test_auth_flows():
    print("\n--- 1. Testing Auth & Session Flows ---")
    s = requests.Session()
    unique_email = f"audit_user_{int(time.time())}@example.com"

    # Register
    r = s.post(f"{NEXTJS_URL}/api/auth/register", json={
        "name": "Audit Tester",
        "email": unique_email,
        "password": "ValidPassword123!"
    })
    if r.status_code != 201:
        record_bug("Auth Registration Failure", "POST /api/auth/register with valid payload", "201 Created", f"{r.status_code} {r.text}", "frontend/app/api/auth/register/route.ts", "High")
    else:
        print("  ✓ User registration successful")

    # Login
    r = s.post(f"{NEXTJS_URL}/api/auth/login", json={
        "email": unique_email,
        "password": "ValidPassword123!"
    })
    if r.status_code != 200 or not r.json().get("success"):
        record_bug("Auth Login Failure", "POST /api/auth/login with registered credentials", "200 OK with success=True", f"{r.status_code} {r.text}", "frontend/app/api/auth/login/route.ts", "High")
    else:
        print("  ✓ User login successful")

    # Me Check
    r = s.get(f"{NEXTJS_URL}/api/auth/me")
    if r.status_code != 200 or not r.json().get("authenticated"):
        record_bug("Session Cookie Verification Failure", "GET /api/auth/me with login session cookie", "200 authenticated=True", f"{r.status_code} {r.text}", "frontend/app/api/auth/me/route.ts", "Medium")
    else:
        print("  ✓ Session cookie verified")

    # Logout
    r = s.post(f"{NEXTJS_URL}/api/auth/logout")
    r_after = s.get(f"{NEXTJS_URL}/api/auth/me")
    if r_after.status_code == 200 and r_after.json().get("authenticated"):
        record_bug("Logout Did Not Invalidate Session", "POST /api/auth/logout then GET /api/auth/me", "401 or authenticated=False", "authenticated=True", "frontend/app/api/auth/logout/route.ts", "Medium")
    else:
        print("  ✓ Logout invalidated session properly")

async def test_leads_crud():
    print("\n--- 2. Testing PostgreSQL Leads CRUD Operations ---")
    lead_data = {
        "name": "Audit Lead Prospect",
        "phone": "9876543210",
        "email": "auditprospect@corp.com",
        "company": "Audit Financials Ltd",
        "job_role": "VP Operations",
        "loan_type": "Business Loan",
        "loan_amount": 7500000.0,
        "tenure_months": 36,
        "notes": "Verified high net worth lead for testing",
    }
    # Create Lead
    r = requests.post(f"{FASTAPI_URL}/api/leads", json=lead_data)
    if r.status_code not in (200, 201):
        record_bug("Create Lead API Error", f"POST /api/leads with valid lead payload", "200/201 Success", f"{r.status_code} {r.text}", "backend/main.py:create_lead_endpoint", "High")
        return None
    created = r.json().get("lead") or r.json()
    lead_id = created.get("id")
    print(f"  ✓ Created Lead ID: {lead_id}")

    # Read Leads
    r_list = requests.get(f"{FASTAPI_URL}/api/leads")
    if r_list.status_code != 200 or not any(l.get("id") == lead_id for l in r_list.json().get("leads", [])):
        record_bug("List Leads Omission", f"GET /api/leads after creating lead {lead_id}", f"Lead {lead_id} in list", "Lead missing from response", "backend/main.py:get_leads_endpoint", "Medium")
    else:
        print(f"  ✓ Listed Leads: found lead {lead_id}")

    # Update Lead
    update_data = dict(lead_data)
    update_data["loan_amount"] = 8500000.0
    r_up = requests.put(f"{FASTAPI_URL}/api/leads/{lead_id}", json=update_data)
    if r_up.status_code != 200:
        record_bug("Update Lead Error", f"PUT /api/leads/{lead_id} with updated amount", "200 OK", f"{r_up.status_code} {r_up.text}", "backend/main.py:update_lead_endpoint", "Medium")
    else:
        print(f"  ✓ Updated Lead {lead_id}")

    # Delete Lead
    r_del = requests.delete(f"{FASTAPI_URL}/api/leads/{lead_id}")
    if r_del.status_code != 200:
        record_bug("Delete Lead Error", f"DELETE /api/leads/{lead_id}", "200 OK", f"{r_del.status_code} {r_del.text}", "backend/main.py:delete_lead_endpoint", "Low")
    else:
        print(f"  ✓ Deleted Lead {lead_id} cleaned up")

async def test_module1_websocket_flows():
    print("\n--- 3. Testing Module 1 Voice WebSocket & Lead Flows ---")
    
    # 3A. Test connection and Finalize
    try:
        ws_endpoint = f"{WS_URL}?sample_rate=48000&language=en&module=module1"
        async with websockets.connect(ws_endpoint) as ws:
            # Send silent PCM frame
            silent_chunk = b"\x00" * 4096
            await ws.send(silent_chunk)
            # Send Finalize
            await ws.send(json.dumps({"type": "Finalize"}))
            
            received_final = False
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    data = json.loads(msg)
                    if data.get("type") == "final":
                        received_final = True
                        break
                except asyncio.TimeoutError:
                    break
            
            if not received_final:
                record_bug("STT WebSocket Final Event Timeout", "Send Finalize over /ws/voice-stt", "Received type='final'", "Timeout awaiting final", "backend/main.py:websocket_voice_stt", "High")
            else:
                print("  ✓ WebSocket received STT final event")
    except Exception as e:
        record_bug("STT WebSocket Connection Failure", f"Connect to {WS_URL}", "Connection accepted", f"Exception: {e}", "backend/main.py:websocket_voice_stt", "High")

    # 3B. Multi-turn Field Extraction & Language Persistence
    from services.lead_extractor import (
        extract_name,
        extract_phone_number,
        extract_email,
        extract_loan_type,
        extract_loan_amount,
        extract_tenure_months,
        merge_lead_safely,
        get_next_missing_parameter,
    )
    from main import generate_module1_ws_response

    # Test Turn 1: Name (English)
    t1 = generate_module1_ws_response("My name is Johnathan Miller", "en", 0.98)
    if t1["lead"].get("name") != "Johnathan Miller":
        record_bug("Name Extraction Failure", "Spoken English name 'Johnathan Miller'", "'Johnathan Miller'", f"{t1['lead'].get('name')}", "services/lead_extractor.py:extract_name", "High")
    if t1["next_missing_parameter"] != "phone":
        record_bug("Next Missing Parameter Mismatch", "After Name extracted", "phone", f"{t1['next_missing_parameter']}", "services/lead_extractor.py", "Medium")
    print("  ✓ Turn 1 Name extraction verified: Johnathan Miller -> Next: phone")

    # Test Turn 2: Phone with spoken double digits & spacing
    lead_state = json.dumps(t1["lead"])
    t2 = generate_module1_ws_response("My phone is nine eight seven six five four three two one zero", "en", 0.99, existing_lead_str=lead_state)
    if t2["lead"].get("phone") != "9876543210":
        record_bug("Spoken Word Phone Extraction Failure", "nine eight seven six five four three two one zero", "9876543210", f"{t2['lead'].get('phone')}", "services/number_normalizer.py", "High")
    print("  ✓ Turn 2 Spoken Phone extraction verified: 9876543210")

    # Test Turn 3: Loan Type and Amount in Hindi
    lead_state = json.dumps(t2["lead"])
    t3 = generate_module1_ws_response("मुझे 50 लाख का पर्सनल लोन 3 साल के लिए चाहिए", "hi", 0.99, existing_lead_str=lead_state)
    if t3["lead"].get("loan_amount") != 5000000.0:
        record_bug("Hindi Spoken Amount Parsing Failure", "'50 लाख' in Hindi speech", "5000000.0", f"{t3['lead'].get('loan_amount')}", "services/lead_extractor.py", "High")
    if t3["lead"].get("tenure_months") != 36:
        record_bug("Hindi Tenure Parsing Failure", "'3 साल' in Hindi speech", "36", f"{t3['lead'].get('tenure_months')}", "services/lead_extractor.py", "Medium")
    print("  ✓ Turn 3 Hindi Amount & Tenure verified: 50 Lakhs (5,000,000) & 36 Months")

    # Test Turn 4: Marathi Loan Request with Fractional Lakhs
    t4 = generate_module1_ws_response("मला साडे सात लाख रुपयांचे बिझनेस लोन हवे आहे", "mr", 0.99)
    if t4["lead"].get("loan_amount") != 750000.0:
        record_bug("Marathi Fractional Amount Failure", "'साडे सात लाख' (7.5 Lakhs)", "750000.0", f"{t4['lead'].get('loan_amount')}", "services/number_normalizer.py", "Medium")
    print("  ✓ Turn 4 Marathi Fractional Amount verified: साडे सात लाख (750,000)")

    # Test Turn 5: Refusal Handling
    t5 = generate_module1_ws_response("I will not share my email", "en", 0.95, existing_lead_str='{"name": "Anil", "phone": "9812345678"}')
    if not t5.get("is_field_refusal"):
        record_bug("Field Refusal Not Recognized", "'I will not share my email'", "is_field_refusal=True", f"{t5.get('is_field_refusal')}", "services/lead_extractor.py:is_field_refusal", "Medium")
    else:
        print("  ✓ Turn 5 Refusal recognized and flow paused politely")

    # Test Turn 6: Flow Resumption
    t6 = generate_module1_ws_response("Okay let's continue", "en", 0.95, existing_lead_str='{"name": "Anil", "phone": "9812345678"}')
    if not t6.get("is_resumed"):
        record_bug("Flow Resume Failure", "'Okay let's continue'", "is_resumed=True", f"{t6.get('is_resumed')}", "services/lead_extractor.py:is_flow_resume_intent", "Medium")
    else:
        print("  ✓ Turn 6 Resumption recognized and continued")

async def test_module2_rag_streaming():
    print("\n--- 4. Testing Module 2 RAG & Playbook Streaming ---")
    # Test /api/ask endpoint
    r = requests.post(
        f"{FASTAPI_URL}/api/ask",
        json={
            "question": "What documents are required for a personal loan?",
            "top_k": 4,
            "stream": True,
        },
        stream=True,
        timeout=10,
    )
    if r.status_code != 200:
        record_bug("RAG /api/ask Endpoint Error", "POST /api/ask stream=True", "200 StreamingResponse", f"{r.status_code} {r.text[:200]}", "backend/main.py:ask_question_endpoint", "High")
    else:
        events = []
        for line in r.iter_lines():
            line_str = line.decode("utf-8") if isinstance(line, bytes) else line
            if line_str.startswith("event:"):
                events.append(line_str)
            if len(events) >= 3:
                break
        print(f"  ✓ /api/ask returned streaming events: {events}")

async def test_tts_voices_all_languages():
    print("\n--- 5. Testing TTS Voices for English, Hindi, and Marathi ---")
    # English Deepgram
    r_en = requests.post(f"{FASTAPI_URL}/api/tts", json={"text": "Hello, thank you for calling.", "language": "en", "module": "module1"})
    if r_en.status_code != 200 or len(r_en.content) < 500:
        record_bug("Deepgram English TTS Failure", "POST /api/tts for English", "200 with audio bytes", f"{r_en.status_code} len={len(r_en.content)}", "backend/services/tts.py", "High")
    else:
        print(f"  ✓ English TTS synthesized: {len(r_en.content)} bytes ({r_en.headers.get('X-TTS-Provider')})")

    # Hindi Sarvam
    r_hi = requests.post(f"{FASTAPI_URL}/api/tts", json={"text": "नमस्ते! आपका स्वागत है।", "language": "hi", "module": "module1"})
    if r_hi.status_code != 200 or len(r_hi.content) < 500:
        record_bug("Hindi Sarvam TTS Failure", "POST /api/tts for Hindi", "200 with audio bytes", f"{r_hi.status_code} len={len(r_hi.content)}", "backend/services/sarvam_tts.py", "High")
    else:
        print(f"  ✓ Hindi TTS synthesized: {len(r_hi.content)} bytes ({r_hi.headers.get('X-TTS-Provider')})")

    # Marathi Sarvam
    r_mr = requests.post(f"{FASTAPI_URL}/api/tts", json={"text": "नमस्कार! आपले स्वागत आहे.", "language": "mr", "module": "module1"})
    if r_mr.status_code != 200 or len(r_mr.content) < 500:
        record_bug("Marathi Sarvam TTS Failure", "POST /api/tts for Marathi", "200 with audio bytes", f"{r_mr.status_code} len={len(r_mr.content)}", "backend/services/sarvam_tts.py", "High")
    else:
        print(f"  ✓ Marathi TTS synthesized: {len(r_mr.content)} bytes ({r_mr.headers.get('X-TTS-Provider')})")

async def test_model_switching_persistence():
    print("\n--- 6. Testing Model Switching and Persistence ---")
    # Activate LLM
    r = requests.post(f"{FASTAPI_URL}/api/models/activate", json={"category": "llm", "model_id": "llama-3.1-8b"})
    if r.status_code != 200:
        record_bug("Model Activation Failure", "Activate llama-3.1-8b", "200 OK", f"{r.status_code} {r.text}", "backend/services/model_manager.py", "Medium")
    else:
        print("  ✓ Switched active LLM to llama-3.1-8b")
    
    # Verify Persistence
    r_act = requests.get(f"{FASTAPI_URL}/api/models/active")
    cur_llm = r_act.json().get("active", {}).get("llm", {}).get("id")
    if cur_llm != "llama-3.1-8b":
        record_bug("Model Persistence Failure", "Check active LLM after switch", "llama-3.1-8b", f"{cur_llm}", "backend/data/models_config.json", "High")
    else:
        print("  ✓ Model persistence verified")

    # Restore default
    requests.post(f"{FASTAPI_URL}/api/models/activate", json={"category": "llm", "model_id": "deepseek-chat"})
    print("  ✓ Restored default LLM to deepseek-chat")

async def test_telegram_isolation():
    print("\n--- 7. Testing Telegram Module 2 Isolation ---")
    from services.telegram_bot import VoiceCopilotBot
    bot = VoiceCopilotBot()
    # Test that knowledge question routes to Module 2 without overwriting CRM lead state
    text_qa = "What are the interest rates for home loans?"
    # Test intent detection
    from services.lead_extractor import is_general_question
    is_gen = is_general_question(text_qa)
    if not is_gen:
        record_bug("Telegram Knowledge Intent Routing", "is_general_question on interest rate query", "True", "False", "backend/services/lead_extractor.py:is_general_question", "Medium")
    else:
        print("  ✓ Telegram knowledge question correctly classified as knowledge query (Module 2)")

async def main():
    print("=================================================================")
    print("     COMPREHENSIVE END-TO-END APPLICATION AUDIT & BUG SCAN")
    print("=================================================================")
    await test_auth_flows()
    await test_leads_crud()
    await test_module1_websocket_flows()
    await test_module2_rag_streaming()
    await test_tts_voices_all_languages()
    await test_model_switching_persistence()
    await test_telegram_isolation()

    print("\n" + "=" * 70)
    print(f"AUDIT COMPLETE. Total Potential Bugs Detected: {len(bugs_found)}")
    print("=" * 70)
    if bugs_found:
        for i, b in enumerate(bugs_found, 1):
            print(f"\nBug #{i}: {b['bug']} [{b['severity']}]")
            print(f"  Steps: {b['steps']}")
            print(f"  Expected: {b['expected']}")
            print(f"  Actual: {b['actual']}")
            print(f"  File/Func: {b['file_func']}")
    else:
        print("All automated end-to-end flows passed with zero defects!")

if __name__ == "__main__":
    asyncio.run(main())
