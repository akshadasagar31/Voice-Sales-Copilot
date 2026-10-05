import asyncio
import os
import sys
import time

sys.path.insert(0, "backend")
from services.sarvam_tts import SarvamTTSService

async def profile_sarvam_service():
    names = ['Sachin', 'Pooja', 'Amol', 'Vikram', 'Ananya']
    service = SarvamTTSService()
    for i, name in enumerate(names, 1):
        text = f"Thank you, {name}! Could you please share your phone number?"
        t0 = time.perf_counter()
        audio = await service.synthesize_speech(text, language_code="en-IN", speaker="simran", model="bulbul:v3")
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"Service Call {i} ({name}): took {elapsed:.1f}ms (audio: {len(audio)} bytes)")

asyncio.run(profile_sarvam_service())
