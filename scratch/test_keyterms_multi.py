import asyncio
import os
import sys
import httpx
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv("backend/.env")
API_KEY = os.getenv("DEEPGRAM_API_KEY")

async def test_keyterms():
    mr_audio = open("scratch/financial_mr_48k.wav", "rb").read()
    headers = {"Authorization": f"Token {API_KEY}", "Content-Type": "audio/wav"}
    url = "https://api.deepgram.com/v1/listen"
    
    # Try language=multi with Devanagari keyterms
    params = [
        ("model", "nova-3"),
        ("language", "multi"),
        ("keyterm", "कर्ज"),
        ("keyterm", "रुपये"),
        ("keyterm", "लाख"),
        ("keyterm", "पाहिजे"),
    ]
    
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(url, params=params, headers=headers, content=mr_audio)
        print("Status:", resp.status_code)
        if resp.status_code == 200:
            tr = resp.json().get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript")
            print("Transcript with Devanagari keyterms:", repr(tr))
        else:
            print("Error:", resp.text[:200])

asyncio.run(test_keyterms())
