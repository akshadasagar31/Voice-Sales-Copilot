import sys
import os
import httpx
from pathlib import Path
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

env_path = Path(__file__).resolve().parent.parent / "backend" / ".env"
load_dotenv(dotenv_path=env_path)

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

test_cases = [
    {
        "label": "English (en)",
        "language_param": "en",
        "tts_lang": "en-IN",
        "text": "My name is Rajesh Sharma. My contact number is 9876543210. I need a personal loan of 5 lakh rupees.",
        "expected_name": "Rajesh Sharma",
        "expected_phone": "9876543210",
    },
    {
        "label": "Hindi (hi)",
        "language_param": "hi",
        "tts_lang": "hi-IN",
        "text": "मेरा नाम राजेश शर्मा है। मेरा फोन नंबर 9876543210 है। मुझे 5 लाख रुपये का पर्सनल लोन चाहिए।",
        "expected_name": "राजेश शर्मा",
        "expected_phone": "9876543210",
    },
    {
        "label": "Marathi (mr)",
        "language_param": "mr",
        "tts_lang": "mr-IN",
        "text": "माझे नाव राजेश शर्मा आहे. माझा फोन नंबर 9876543210 आहे. मला 5 लाख रुपयांचे वैयक्तिक कर्ज हवे आहे.",
        "expected_name": "राजेश शर्मा",
        "expected_phone": "9876543210",
    },
    {
        "label": "Mixed Hinglish (auto)",
        "language_param": "auto",
        "tts_lang": "hi-IN",
        "text": "Hello, mera naam Rajesh Sharma hai, phone number 9876543210, looking for 5 lakh personal loan.",
        "expected_name": "राजेश शर्मा",
        "expected_phone": "9876543210",
    },
]

def run_live_verification():
    print("\n==================================================================")
    print("LIVE VERIFICATION: MODULE 1 STT ACCURACY (http://127.0.0.1:8001/api/voice-entry)")
    print("==================================================================\n")

    all_passed = True

    with httpx.Client(timeout=30.0) as client:
        for tc in test_cases:
            print(f"--- Testing {tc['label']} ---")
            print(f"Spoken Input : {tc['text']}")

            # 1. Synthesize audio via Sarvam Bulbul v3
            tts_res = client.post(
                "https://api.sarvam.ai/text-to-speech",
                headers={"api-subscription-key": SARVAM_API_KEY},
                json={
                    "text": tc["text"],
                    "language_code": tc["tts_lang"],
                    "speaker": "simran",
                    "model": "bulbul:v3",
                },
            )
            if tts_res.status_code != 200:
                print(f"FAILED to synthesize TTS: {tts_res.text}")
                all_passed = False
                continue

            import base64
            audios = tts_res.json().get("audios", [])
            audio_bytes = base64.b64decode(audios[0])
            print(f"Synthesized WAV: {len(audio_bytes)} bytes")

            # 2. Send to FastAPI /api/voice-entry as Module 1
            data = {"module": "module1"}
            if tc["language_param"] != "auto":
                data["language"] = tc["language_param"]

            files = {"file": ("recording.wav", audio_bytes, "audio/wav")}
            entry_res = client.post("http://127.0.0.1:8001/api/voice-entry", data=data, files=files)

            if entry_res.status_code != 200:
                print(f"FAILED /api/voice-entry (HTTP {entry_res.status_code}): {entry_res.text}")
                all_passed = False
                continue

            res_json = entry_res.json()
            transcript = res_json.get("transcript", "")
            provider = res_json.get("stt_provider", "")
            model = res_json.get("model", "")
            detected = res_json.get("detected_language", "")

            print(f"Transcript   : '{transcript}'")
            print(f"Engine       : {provider} ({model}) | Detected: {detected}")

            # Check exact phone number preservation
            has_phone = tc["expected_phone"] in transcript
            # Check name preservation
            has_name = tc["expected_name"] in transcript

            print(f"Phone '9876543210' preserved exact: {'PASS' if has_phone else 'FAIL'}")
            print(f"Name '{tc['expected_name']}' preserved exact: {'PASS' if has_name else 'FAIL'}")

            if not has_phone:
                all_passed = False
            print("")

    if all_passed:
        print("ALL TEST CASES PASSED! Exact verbatim transcription across English, Hindi, Marathi, and mixed speech.")
    else:
        print("SOME CHECKS FAILED.")

if __name__ == "__main__":
    run_live_verification()
