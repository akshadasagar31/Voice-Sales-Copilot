import os
import asyncio
import httpx
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / "backend" / ".env"
load_dotenv(dotenv_path=env_path)

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

async def test_fake_bytes():
    fake_bytes = b"\x1a\x45\xdf\xa3" + b"sample-voice-note-audio-bytes-content-12345"
    async with httpx.AsyncClient(timeout=10.0) as client:
        res = await client.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": SARVAM_API_KEY},
            files={"file": ("test.webm", fake_bytes, "audio/webm")},
            data={"language_code": "unknown", "model": "saaras:v4"}
        )
        print("Status code:", res.status_code)
        print("Response:", res.text)

asyncio.run(test_fake_bytes())
