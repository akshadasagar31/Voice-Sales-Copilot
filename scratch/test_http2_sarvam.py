import asyncio
import os
import time
import httpx
from dotenv import load_dotenv

load_dotenv("backend/.env")
api_key = os.getenv("SARVAM_API_KEY", "").strip()

async def compare_http():
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "text": "Hello! May I have your name, please?",
        "model": "bulbul:v3",
        "language_code": "en-IN",
        "speaker": "simran",
    }
    url = "https://api.sarvam.ai/text-to-speech"

    # Test HTTP/2 warm
    async with httpx.AsyncClient(http2=True, timeout=15.0, limits=httpx.Limits(max_keepalive_connections=15, keepalive_expiry=120.0)) as c2:
        # Prewarm
        t0 = time.perf_counter()
        r1 = await c2.post(url, headers=headers, json=payload)
        t_c2_1 = (time.perf_counter() - t0) * 1000
        print(f"HTTP/2 Turn 1: {t_c2_1:.1f}ms (version: {r1.http_version})")

        # Turn 2
        t0 = time.perf_counter()
        r2 = await c2.post(url, headers=headers, json=payload)
        t_c2_2 = (time.perf_counter() - t0) * 1000
        print(f"HTTP/2 Turn 2: {t_c2_2:.1f}ms")

        # Turn 3
        t0 = time.perf_counter()
        r3 = await c2.post(url, headers=headers, json=payload)
        t_c2_3 = (time.perf_counter() - t0) * 1000
        print(f"HTTP/2 Turn 3: {t_c2_3:.1f}ms")

asyncio.run(compare_http())
