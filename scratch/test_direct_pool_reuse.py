import asyncio
import os
import sys
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
sys.path.insert(0, os.path.abspath("backend"))

from services.stt import build_deepgram_ws_url, get_deepgram_ws_pool

async def test_pool():
    api_key = os.getenv("DEEPGRAM_API_KEY")
    url = build_deepgram_ws_url(model="nova-3", sample_rate=48000, language="en")
    headers = {"Authorization": f"Token {api_key}"}
    pool = get_deepgram_ws_pool()

    for turn in range(1, 4):
        print(f"\n>>> Acquiring WS for Turn {turn}...")
        t0 = time.perf_counter()
        ws = await pool.acquire(url, headers)
        t_acq = (time.perf_counter() - t0) * 1000
        print(f"  Acquired in {t_acq:.1f}ms! State: {ws.state.name}")

        # Send silence and Finalize
        t_fin = time.perf_counter()
        await ws.send(b"\x00\x00" * 4800)
        await ws.send(json.dumps({"type": "Finalize"}))

        # Read response
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            mtype = data.get("type")
            if mtype == "Results" and data.get("is_final"):
                break
            if mtype == "Metadata":
                break
        t_resp = (time.perf_counter() - t_fin) * 1000
        print(f"  Turn {turn} Finalize -> Response in {t_resp:.1f}ms! State: {ws.state.name}")

        await pool.release(url, ws, broken=False)
        print(f"  Turn {turn} released back to pool.")

asyncio.run(test_pool())
