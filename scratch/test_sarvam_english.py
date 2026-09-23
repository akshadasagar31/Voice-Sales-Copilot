import os
import asyncio
import httpx
from dotenv import load_dotenv

load_dotenv("backend/.env")

async def test_en():
    api_key = os.getenv("SARVAM_API_KEY")
    url = "https://api.sarvam.ai/text-to-speech"
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json",
    }
    for voice in ["simran", "priya", "ritu", "ishita", "divya"]:
        payload = {
            "text": "Hello! The minimum CIBIL score for an HDFC Bank personal loan is 750.",
            "model": "bulbul:v3",
            "language_code": "en-IN",
            "speaker": voice,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            print(f"Voice: {voice}, Status: {resp.status_code}, Length: {len(resp.content)}")
            if resp.status_code != 200:
                print(resp.text)

if __name__ == "__main__":
    asyncio.run(test_en())
