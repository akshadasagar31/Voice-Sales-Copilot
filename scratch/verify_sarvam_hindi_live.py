import asyncio
import httpx
import os
import base64
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(Path("backend/.env"))

sarvam_key = os.getenv("SARVAM_API_KEY")

async def test_live_module2_hindi():
    spoken_hindi = "नमस्ते! मेरा CIBIL स्कोर 750 है, मुझे HDFC Bank से personal loan पर कितना interest rate और मासिक EMI मिलेगा? लोन की राशि 5 लाख रुपये है।"

    async with httpx.AsyncClient(timeout=30.0) as client:
        print("1. Generating authentic spoken Hindi test audio via Sarvam bulbul:v3...")
        tts_gen = await client.post(
            "https://api.sarvam.ai/text-to-speech",
            headers={"api-subscription-key": sarvam_key, "Content-Type": "application/json"},
            json={"text": spoken_hindi, "language_code": "hi-IN", "speaker": "aditya", "model": "bulbul:v3"}
        )
        assert tts_gen.status_code == 200, f"TTS gen failed: {tts_gen.text}"
        wav_bytes = base64.b64decode(tts_gen.json()["audios"][0])
        print(f"   Generated {len(wav_bytes)} bytes WAV audio.")

        print("\n2. Testing POST http://127.0.0.1:8001/api/voice-entry (Module 2 Hindi)...")
        files = {"file": ("query.wav", wav_bytes, "audio/wav")}
        data = {"language": "hi", "module": "module2"}
        stt_res = await client.post("http://127.0.0.1:8001/api/voice-entry", files=files, data=data)
        print(f"   Status: {stt_res.status_code}")
        stt_json = stt_res.json()
        prov = stt_json.get("stt_provider")
        model = stt_json.get("model")
        transcript = stt_json.get("transcript")
        print(f"   Provider: {prov}, Model: {model}")
        print(f"   Transcript: '{transcript}'")
        assert prov == "sarvam", f"Expected sarvam provider, got {prov}"
        assert model == "saaras:v4", f"Expected saaras:v4, got {model}"
        assert "750" in transcript, "Expected digit 750 in transcript"
        assert "लाख" in transcript, "Expected लाख in transcript"

        print("\n3. Testing POST http://127.0.0.1:8001/api/tts (Module 2 Hindi)...")
        response_text = "एचडीएफसी बैंक में पर्सनल लोन के लिए न्यूनतम सिबिल स्कोर 750 होना चाहिए। 5 लाख रुपये के लोन पर ईएमआई लगभग 10,500 रुपये प्रति माह होगी।"
        tts_payload = {
            "text": response_text,
            "language": "hi",
            "module": "module2"
        }
        tts_res = await client.post("http://127.0.0.1:8001/api/tts", json=tts_payload)
        print(f"   Status: {tts_res.status_code}")
        p = tts_res.headers.get("x-tts-provider")
        v = tts_res.headers.get("x-tts-voice")
        l = tts_res.headers.get("x-tts-language")
        print(f"   Headers: X-TTS-Provider={p}, X-TTS-Voice={v}, X-TTS-Language={l}")
        print(f"   Media-Type: {tts_res.headers.get('content-type')}, Audio Size: {len(tts_res.content)} bytes")
        assert p == "sarvam", f"Expected sarvam, got {p}"
        assert v == "aditya", f"Expected aditya, got {v}"
        assert l == "hi-IN", f"Expected hi-IN, got {l}"
        assert tts_res.content.startswith(b"RIFF"), "Expected RIFF WAV header"

    print("\nALL LIVE MODULE 2 HINDI VERIFICATIONS PASSED PERFECTLY!")

if __name__ == "__main__":
    asyncio.run(test_live_module2_hindi())
