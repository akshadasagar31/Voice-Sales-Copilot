import asyncio
import os
import sys
import time
import httpx
from dotenv import load_dotenv

sys.path.insert(0, "backend")
load_dotenv("backend/.env")

from services.sarvam_tts import get_sarvam_tts_service, get_shared_sarvam_client, prewarm_sarvam_client
from services.tts import get_cached_tts_audio, set_cached_tts_audio
from main import app
from starlette.testclient import TestClient

async def run_benchmark():
    print("=" * 65)
    print("MODULE 1 SARVAM BULBUL V3 TTS LATENCY BENCHMARK (5 TURNS)")
    print("=" * 65)

    # 1. Test Pre-warming
    t0 = time.perf_counter()
    pw_ok = await prewarm_sarvam_client()
    t_pw = (time.perf_counter() - t0) * 1000
    print(f"[Pre-warm] Connection to api.sarvam.ai established in {t_pw:.1f}ms (ok={pw_ok})")

    turns = [
        {"turn": 1, "text": "Hello! May I have your name, please?", "lang": "en-IN", "speaker": "simran"},
        {"turn": 2, "text": "Thank you, Rajesh! Could you please share your phone number?", "lang": "en-IN", "speaker": "simran"},
        {"turn": 3, "text": "What is the loan amount you are looking for?", "lang": "en-IN", "speaker": "simran"},
        {"turn": 4, "text": "What type of loan are you interested in — Personal, Home, or Business loan?", "lang": "en-IN", "speaker": "simran"},
        {"turn": 5, "text": "नमस्कार! आपले नाव काय आहे?", "lang": "mr-IN", "speaker": "ritu"},
    ]

    service = get_sarvam_tts_service()

    print("\n--- Phase 1: Direct Service Synthesis (Persistent Connection Pool) ---")
    service_latencies = []
    for item in turns:
        turn_num = item["turn"]
        text = item["text"]
        lang = item["lang"]
        speaker = item["speaker"]
        t_start = time.perf_counter()
        audio = await service.synthesize_speech(text, language_code=lang, speaker=speaker, model="bulbul:v3")
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        service_latencies.append(elapsed_ms)
        safe_text = text[:45].encode("ascii", errors="replace").decode("ascii")
        print(f"Turn {turn_num} ({lang}): {elapsed_ms:.1f}ms | Audio: {len(audio):,} bytes | Text: '{safe_text}...'")

    print("\n--- Phase 2: FastAPI /api/tts Endpoint (Same Async Loop with Pooled Client) ---")
    transport = httpx.ASGITransport(app=app)
    endpoint_latencies = []
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for item in turns:
            turn_num = item["turn"]
            text = item["text"]
            lang = "mr" if "mr" in item["lang"] else "en"
            speaker = item["speaker"]
            t_start = time.perf_counter()
            res = await client.post(
                "/api/tts",
                json={"text": text, "module": "module1", "language": lang, "speaker": speaker, "model": "bulbul:v3"},
            )
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            endpoint_latencies.append(elapsed_ms)
            cache_status = res.headers.get("x-tts-cache", "MISS")
            provider = res.headers.get("x-tts-provider", "unknown")
            print(f"Turn {turn_num} Endpoint ({lang}): {elapsed_ms:.1f}ms | Status: {res.status_code} | Provider: {provider} | Cache: {cache_status}")

        print("\n--- Phase 3: Repeat Calls (Demonstrating Instant In-Memory Cache) ---")
        cached_latencies = []
        for item in turns:
            turn_num = item["turn"]
            text = item["text"]
            lang = "mr" if "mr" in item["lang"] else "en"
            speaker = item["speaker"]
            t_start = time.perf_counter()
            res = await client.post(
                "/api/tts",
                json={"text": text, "module": "module1", "language": lang, "speaker": speaker, "model": "bulbul:v3"},
            )
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            cached_latencies.append(elapsed_ms)
            cache_status = res.headers.get("x-tts-cache", "MISS")
            provider = res.headers.get("x-tts-provider", "unknown")
            print(f"Turn {turn_num} Cached ({lang}): {elapsed_ms:.2f}ms | Provider: {provider} | Cache: {cache_status}")

    print("\n" + "=" * 65)
    print("SUMMARY RESULTS:")
    print(f"Direct Service avg latency: {sum(service_latencies)/len(service_latencies):.1f}ms")
    print(f"Endpoint 1st run avg latency: {sum(endpoint_latencies)/len(endpoint_latencies):.1f}ms")
    print(f"Endpoint cached avg latency: {sum(cached_latencies)/len(cached_latencies):.2f}ms")
    print("=" * 65)

if __name__ == "__main__":
    asyncio.run(run_benchmark())
