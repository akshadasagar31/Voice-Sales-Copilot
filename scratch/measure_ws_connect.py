import asyncio
import os
import time
import websockets
from dotenv import load_dotenv

load_dotenv("backend/.env")
api_key = os.getenv("DEEPGRAM_API_KEY")

async def measure():
    from services.stt import build_deepgram_ws_url
    url = build_deepgram_ws_url(model="nova-3", sample_rate=48000, language="en")
    headers = {"Authorization": f"Token {api_key}"}

    for i in range(5):
        t0 = time.perf_counter()
        try:
            async with websockets.connect(url, additional_headers=headers) as ws:
                t1 = time.perf_counter()
                print(f"Turn {i+1}: connected in {(t1-t0)*1000:.1f}ms, state: {ws.state.name}", flush=True)
                await ws.send(b"\x00\x00" * 1000)
                await ws.send('{"type": "Finalize"}')
                async for msg in ws:
                    # just read until done
                    pass
        except Exception as e:
            t1 = time.perf_counter()
            print(f"Turn {i+1}: FAILED in {(t1-t0)*1000:.1f}ms: {type(e).__name__}: {e}", flush=True)

asyncio.run(measure())
