import asyncio
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv("backend/.env")
sys.path.insert(0, os.path.abspath("backend"))

from services.tts import DeepgramTTSService

async def test():
    d = DeepgramTTSService()
    t0 = time.perf_counter()
    b = await d.synthesize_speech("The minimum CIBIL score is 750.", language="en")
    print(f"Deepgram TTS took {(time.perf_counter() - t0)*1000:.1f}ms, bytes: {len(b)}")

if __name__ == "__main__":
    asyncio.run(test())
