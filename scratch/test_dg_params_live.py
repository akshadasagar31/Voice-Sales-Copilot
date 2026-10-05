import asyncio
import os
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, "backend")
from dotenv import load_dotenv
load_dotenv("backend/.env")

import httpx

API_KEY = os.getenv("DEEPGRAM_API_KEY")
URL = "https://api.deepgram.com/v1/listen"

async def test_dg(params, audio_bytes, label):
    headers = {"Authorization": f"Token {API_KEY}", "Content-Type": "audio/wav"}
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(URL, params=params, headers=headers, content=audio_bytes)
        print(f"[{label}] status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            ch = data.get("results", {}).get("channels", [{}])[0]
            alt = ch.get("alternatives", [{}])[0]
            detected = ch.get("detected_language")
            languages = alt.get("languages")
            tr = alt.get("transcript")
            print(f"    detected_language: {detected}")
            print(f"    languages in alt: {languages}")
            print(f"    transcript: {tr}")
        else:
            print(f"    error: {resp.text[:300]}")

async def main():
    mr_audio = open("scratch/financial_mr_48k.wav", "rb").read()
    hi_audio = open("scratch/financial_hi_48k.wav", "rb").read()

    print("=== Testing Marathi Audio with different Deepgram params ===")
    await test_dg({"model": "nova-3", "detect_language": "true"}, mr_audio, "Nova-3 + detect_language=true")
    await test_dg({"model": "nova-3", "language": "multi"}, mr_audio, "Nova-3 + language=multi")
    await test_dg({"model": "nova-3"}, mr_audio, "Nova-3 alone (no lang param)")
    await test_dg({"model": "nova-3", "language": "mr"}, mr_audio, "Nova-3 + language=mr")

    print("\n=== Testing Hindi Audio with different Deepgram params ===")
    await test_dg({"model": "nova-3", "detect_language": "true"}, hi_audio, "Nova-3 + detect_language=true")
    await test_dg({"model": "nova-3", "language": "multi"}, hi_audio, "Nova-3 + language=multi")
    await test_dg({"model": "nova-3"}, hi_audio, "Nova-3 alone (no lang param)")
    await test_dg({"model": "nova-3", "language": "hi"}, hi_audio, "Nova-3 + language=hi")

asyncio.run(main())
