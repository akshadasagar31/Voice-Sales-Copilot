import sys
import os
import asyncio
import httpx
from pathlib import Path
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

env_path = Path(__file__).resolve().parent.parent / "backend" / ".env"
load_dotenv(dotenv_path=env_path)

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

test_cases = [
    {
        "lang": "en",
        "sarvam_code": "en-IN",
        "text": "What is the minimum CIBIL score required for a personal loan of 15 lakh with HDFC Bank and what is the monthly EMI?",
    },
    {
        "lang": "hi",
        "sarvam_code": "hi-IN",
        "text": "एचडीएफसी बैंक से 15 लाख के पर्सनल लोन के लिए सिबिल स्कोर कितना होना चाहिए और मंथली ईएमआई कितनी होगी?",
    },
    {
        "lang": "mr",
        "sarvam_code": "mr-IN",
        "text": "एचडीएफसी बँकेकडून 10 लाख रुपयांच्या पर्सनल लोनसाठी सिबिल स्कोर किती लागतो आणि दरमहा ईएमआय किती येईल?",
    },
    {
        "lang": "mixed",
        "sarvam_code": "hi-IN",
        "text": "HDFC Bank me 10 lakh personal loan ke liye CIBIL score aur monthly EMI kitna hoga?",
    },
]

async def run_benchmark():
    print("=== BENCHMARKING MODULE 2 FINANCIAL QUERIES: SARVAM SAARAS:V4 vs DEEPGRAM NOVA-3 ===")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for tc in test_cases:
            print(f"\n------------------------------------------------------------")
            print(f"Language: {tc['lang'].upper()} | Expected: {tc['text']}")
            
            # Synthesize audio with Sarvam Bulbul v3
            tts_res = await client.post(
                "https://api.sarvam.ai/text-to-speech",
                headers={"api-subscription-key": SARVAM_API_KEY},
                json={
                    "text": tc["text"],
                    "language_code": tc["sarvam_code"],
                    "speaker": "simran",
                    "model": "bulbul:v3",
                },
            )
            if tts_res.status_code != 200:
                print(f"TTS Error: {tts_res.text}")
                continue
            
            import base64
            audios = tts_res.json().get("audios", [])
            audio_bytes = base64.b64decode(audios[0])
            print(f"Audio size: {len(audio_bytes)} bytes")
            
            # 1. Test Sarvam saaras:v4 with target lang
            sarvam_res = await client.post(
                "https://api.sarvam.ai/speech-to-text",
                headers={"api-subscription-key": SARVAM_API_KEY},
                files={"file": ("audio.wav", audio_bytes, "audio/wav")},
                data={"language_code": tc["sarvam_code"], "model": "saaras:v4"},
            )
            if sarvam_res.status_code == 200:
                s_tr = sarvam_res.json().get("transcript", "")
                print(f"[Sarvam saaras:v4 ({tc['sarvam_code']})]: '{s_tr}'")
            else:
                print(f"[Sarvam Error]: {sarvam_res.text}")

            # 2. Test Sarvam saaras:v4 with unknown (auto)
            sarvam_auto_res = await client.post(
                "https://api.sarvam.ai/speech-to-text",
                headers={"api-subscription-key": SARVAM_API_KEY},
                files={"file": ("audio.wav", audio_bytes, "audio/wav")},
                data={"language_code": "unknown", "model": "saaras:v4"},
            )
            if sarvam_auto_res.status_code == 200:
                s_auto_tr = sarvam_auto_res.json().get("transcript", "")
                s_auto_lang = sarvam_auto_res.json().get("language_code", "")
                print(f"[Sarvam saaras:v4 (unknown)]         : '{s_auto_tr}' (lang: {s_auto_lang})")
            else:
                print(f"[Sarvam Auto Error]: {sarvam_auto_res.text}")

            # 3. Test Deepgram Nova-3
            dg_url = f"https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true&language={tc['lang'] if tc['lang'] != 'mixed' else 'multi'}"
            dg_res = await client.post(
                dg_url,
                headers={"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "audio/wav"},
                content=audio_bytes,
            )
            if dg_res.status_code == 200:
                alt = dg_res.json().get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0]
                dg_tr = alt.get("transcript", "")
                print(f"[Deepgram Nova-3 ({tc['lang']})]            : '{dg_tr}'")
            else:
                print(f"[Deepgram Error]: {dg_res.text}")

asyncio.run(run_benchmark())
