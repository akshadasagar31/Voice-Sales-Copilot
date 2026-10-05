import asyncio
import os
import time
import httpx
from dotenv import load_dotenv

load_dotenv("backend/.env")
api_key = os.getenv("SARVAM_API_KEY", "").strip()

async def benchmark_sarvam():
    print(f"Testing Sarvam TTS with API key: {api_key[:6]}...{api_key[-4:] if api_key else 'NONE'}")
    url = "https://api.sarvam.ai/text-to-speech"
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json",
    }
    # Short sentence typical of Module 1: "Hello! May I have your name, please?"
    text_en = "Hello! May I have your name, please?"
    text_mr = "नमस्कार! आपले नाव काय आहे?"
    
    # Test 1: Single unpooled client (cold TCP + TLS)
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=15.0) as client:
        payload = {
            "text": text_en,
            "model": "bulbul:v3",
            "language_code": "en-IN",
            "speaker": "simran",
        }
        res = await client.post(url, headers=headers, json=payload)
        t_cold = (time.perf_counter() - t0) * 1000
        print(f"Cold request status: {res.status_code}, time: {t_cold:.1f}ms, audio length: {len(res.content)}")

    # Test 2: Persistent pooled client (warm connection)
    async with httpx.AsyncClient(timeout=15.0, limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=60.0)) as pooled_client:
        # Request 1 (warm up the socket)
        t0 = time.perf_counter()
        res1 = await pooled_client.post(url, headers=headers, json=payload)
        t_req1 = (time.perf_counter() - t0) * 1000
        print(f"Pooled request 1: {res1.status_code}, time: {t_req1:.1f}ms")

        # Request 2 (reused TCP + TLS socket)
        t0 = time.perf_counter()
        res2 = await pooled_client.post(url, headers=headers, json=payload)
        t_req2 = (time.perf_counter() - t0) * 1000
        print(f"Pooled request 2 (warm): {res2.status_code}, time: {t_req2:.1f}ms")

        # Request 3 (Marathi text)
        payload_mr = {
            "text": text_mr,
            "model": "bulbul:v3",
            "language_code": "mr-IN",
            "speaker": "ritu",
        }
        t0 = time.perf_counter()
        res3 = await pooled_client.post(url, headers=headers, json=payload_mr)
        t_req3 = (time.perf_counter() - t0) * 1000
        print(f"Pooled request 3 (Marathi warm): {res3.status_code}, time: {t_req3:.1f}ms")

asyncio.run(benchmark_sarvam())
