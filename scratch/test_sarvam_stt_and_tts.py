import sys
import os
import base64
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import httpx

backend_dir = Path(r"c:\Users\Akshada\OneDrive\Pictures\Documents\projects\Voice-Sales-Copilot\backend")
sys.path.insert(0, str(backend_dir))
sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(backend_dir / ".env")

api_key = os.getenv("SARVAM_API_KEY")

async def test_sarvam_models():
    models = ["saaras:v3", "saaras:v4", "saarika:v2.5", "saarika:flash", "saarika:v2"]
    hindi_text = "नमस्ते! मेरा CIBIL स्कोर 750 है, मुझे HDFC Bank से personal loan पर कितना interest rate और मासिक EMI मिलेगा? लोन की राशि 5 लाख रुपये है।"

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Generate TTS audio
        tts_res = await client.post(
            "https://api.sarvam.ai/text-to-speech",
            headers={"api-subscription-key": api_key, "Content-Type": "application/json"},
            json={
                "text": hindi_text,
                "language_code": "hi-IN",
                "speaker": "aditya",
                "model": "bulbul:v3"
            }
        )
        audio_bytes = base64.b64decode(tts_res.json()["audios"][0])
        print(f"Spoken: '{hindi_text}'\n")

        for m in models:
            files = {"file": ("speech.wav", audio_bytes, "audio/wav")}
            data = {"language_code": "hi-IN", "model": m}
            try:
                stt_res = await client.post(
                    "https://api.sarvam.ai/speech-to-text",
                    headers={"api-subscription-key": api_key},
                    files=files,
                    data=data
                )
                if stt_res.status_code == 200:
                    res = stt_res.json()
                    transcript = res.get("transcript") or res.get("text") or res
                    print(f"[{m:<16}] -> '{transcript}'")
                else:
                    print(f"[{m:<16}] Error {stt_res.status_code}: {stt_res.text}")
            except Exception as e:
                print(f"[{m:<16}] Exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_sarvam_models())
