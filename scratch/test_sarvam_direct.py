import asyncio
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv("backend/.env")
sys.path.insert(0, os.path.abspath("backend"))

from services.sarvam_tts import SarvamTTSService

async def test():
    s = SarvamTTSService()
    print("API Key present:", bool(s.api_key), "prefix:", s.api_key[:8] if s.api_key else "None")
    t0 = time.perf_counter()
    try:
        b = await s.synthesize_speech("Hello, this is a test.", language_code="en-IN", speaker="simran")
        print(f"Success! Received {len(b)} bytes in {(time.perf_counter() - t0)*1000:.1f}ms")
    except Exception as e:
        print(f"Error ({type(e).__name__}): {e} in {(time.perf_counter() - t0)*1000:.1f}ms")

if __name__ == "__main__":
    asyncio.run(test())
