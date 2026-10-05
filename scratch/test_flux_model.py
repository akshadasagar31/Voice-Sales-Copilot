import asyncio
import os
import sys
import httpx
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv("backend/.env")
API_KEY = os.getenv("DEEPGRAM_API_KEY")

async def test_flux():
    mr_audio = open("scratch/financial_mr_48k.wav", "rb").read()
    hi_audio = open("scratch/financial_hi_48k.wav", "rb").read()
    headers = {"Authorization": f"Token {API_KEY}", "Content-Type": "audio/wav"}
    url = "https://api.deepgram.com/v2/listen"
    
    async with httpx.AsyncClient(timeout=20) as client:
        # Test flux-general-multi on MR
        resp_mr = await client.post(url, params={"model": "flux-general-multi"}, headers=headers, content=mr_audio)
        print("Flux MR status:", resp_mr.status_code)
        if resp_mr.status_code == 200:
            print("Flux MR:", resp_mr.json().get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript"))
        else:
            print("Flux MR error:", resp_mr.text[:200])

        # Test flux-general-multi on HI
        resp_hi = await client.post(url, params={"model": "flux-general-multi"}, headers=headers, content=hi_audio)
        print("Flux HI status:", resp_hi.status_code)
        if resp_hi.status_code == 200:
            print("Flux HI:", resp_hi.json().get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript"))
        else:
            print("Flux HI error:", resp_hi.text[:200])

asyncio.run(test_flux())
