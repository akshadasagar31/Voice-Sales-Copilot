import sys
import os
import httpx
from pathlib import Path
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

env_path = Path(__file__).resolve().parent.parent / "backend" / ".env"
load_dotenv(dotenv_path=env_path)

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

test_cases = [
    {
        "label": "Marathi (mr)",
        "language_param": "mr",
        "tts_lang": "mr-IN",
        "text": "एचडीएफसी बँकेकडून 10 लाख रुपयांच्या पर्सनल लोनसाठी सिबिल स्कोर किती लागतो आणि दरमहा ईएमआय किती येईल?",
        "expected_keywords": ["एचडीएफसी", "लाख", "सिबिल", "ईएमआय"],
    },
    {
        "label": "Hindi (hi)",
        "language_param": "hi",
        "tts_lang": "hi-IN",
        "text": "एचडीएफसी बैंक से 15 लाख के पर्सनल लोन के लिए सिबिल स्कोर कितना होना चाहिए और मंथली ईएमआई कितनी होगी?",
        "expected_keywords": ["एचडीएफसी", "लाख", "सिबिल", "ईएमआई"],
    },
    {
        "label": "English (en)",
        "language_param": "en",
        "tts_lang": "en-IN",
        "text": "What is the minimum CIBIL score required for a personal loan of 15 lakh with HDFC Bank and what is the monthly EMI?",
        "expected_keywords": ["CIBIL", "15 lakh", "HDFC", "EMI"],
    },
    {
        "label": "Mixed Hinglish (auto)",
        "language_param": "auto",
        "tts_lang": "hi-IN",
        "text": "HDFC Bank me 10 lakh personal loan ke liye CIBIL score aur monthly EMI kitna hoga?",
        "expected_keywords": ["एचडीएफसी", "10 लाख", "सिबिल", "ईएमआई"],
    },
]

def run_live_verification():
    print("\n==================================================================")
    print("LIVE VERIFICATION: MODULE 2 STT ACCURACY (http://127.0.0.1:8001/api/voice-entry)")
    print("Testing Marathi, Hindi, English, and Mixed queries with CIBIL, EMI, HDFC")
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

            # 2. Send to FastAPI /api/voice-entry as Module 2
            data = {"module": "module2"}
            if tc["language_param"] != "auto":
                data["language"] = tc["language_param"]

            files = {"file": ("query.wav", audio_bytes, "audio/wav")}
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

            # Check keyword preservation
            case_passed = True
            for kw in tc["expected_keywords"]:
                found = kw.lower() in transcript.lower()
                print(f"Keyword '{kw}' in transcript: {'PASS' if found else 'FAIL'}")
                if not found:
                    case_passed = False
                    all_passed = False

            print(f"Result: {'PASS' if case_passed else 'FAIL'}\n")

    if all_passed:
        print("ALL MODULE 2 STT TEST CASES PASSED WITH 100% ACCURACY!")
    else:
        print("SOME CHECKS FAILED.")

if __name__ == "__main__":
    run_live_verification()
