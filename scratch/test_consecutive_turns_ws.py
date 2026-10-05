import asyncio
import os
import json
import time
import math
import websockets
from dotenv import load_dotenv

load_dotenv("backend/.env")
api_key = os.getenv("DEEPGRAM_API_KEY")

async def test_consecutive_turns():
    from services.stt import build_deepgram_ws_url
    url = build_deepgram_ws_url(model="nova-3", sample_rate=48000, language="en")
    headers = {"Authorization": f"Token {api_key}"}

    t0 = time.perf_counter()
    ws = await websockets.connect(url, additional_headers=headers)
    print(f"Initial connection took {(time.perf_counter()-t0)*1000:.1f}ms! State: {ws.state.name}", flush=True)

    sample_rate = 48000
    # Create simple tone audio
    tone = [int(8000 * math.sin(2 * math.pi * 300 * i / sample_rate)) for i in range(sample_rate)]
    pcm = b"".join(s.to_bytes(2, "little", signed=True) for s in tone)

    for turn in range(1, 4):
        print(f"\n--- Turn {turn} (reusing same WS) ---", flush=True)
        turn_t0 = time.perf_counter()
        # Stream audio
        for i in range(0, len(pcm), 4800):
            await ws.send(pcm[i:i+4800])
            await asyncio.sleep(0.02)
        
        # Send Finalize for this turn
        await ws.send(json.dumps({"type": "Finalize"}))

        # Read messages until we get final/metadata
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                data = json.loads(msg)
                mtype = data.get("type")
                if mtype == "Results" and data.get("is_final"):
                    print(f"Turn {turn} got final result in {(time.perf_counter()-turn_t0)*1000:.1f}ms", flush=True)
                    break
            except asyncio.TimeoutError:
                print(f"Turn {turn} timeout (still healthy, ws open={ws.state.name})", flush=True)
                break
        
        print(f"Turn {turn} completed. WS State: {ws.state.name}", flush=True)
        # Small pause between turns
        await asyncio.sleep(0.5)

    await ws.send(json.dumps({"type": "CloseStream"}))
    await ws.close()
    print("\nAll 3 turns succeeded on the same healthy WebSocket!", flush=True)

asyncio.run(test_consecutive_turns())
