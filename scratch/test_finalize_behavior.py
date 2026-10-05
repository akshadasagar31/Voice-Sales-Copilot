import asyncio
import os
import json
import time
import websockets
from dotenv import load_dotenv

load_dotenv("backend/.env")
api_key = os.getenv("DEEPGRAM_API_KEY")

async def test_finalize_behavior():
    from services.stt import build_deepgram_ws_url
    url = build_deepgram_ws_url(model="nova-3", sample_rate=48000, language="en")
    headers = {"Authorization": f"Token {api_key}"}
    
    ws = await websockets.connect(url, additional_headers=headers)
    print(f"Connected! State: {ws.state.name}")

    # Send 1 sec audio
    await ws.send(b"\x00\x00" * 48000)
    print("Sent audio. Now sending Finalize...")
    await ws.send(json.dumps({"type": "Finalize"}))
    
    t0 = time.perf_counter()
    while True:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
            data = json.loads(msg)
            print(f"+{(time.perf_counter()-t0)*1000:.1f}ms received: {data.get('type')}, ws state: {ws.state.name}")
        except asyncio.TimeoutError:
            print(f"+{(time.perf_counter()-t0)*1000:.1f}ms timeout waiting for msg, ws state: {ws.state.name}")
            break
        except Exception as e:
            print(f"+{(time.perf_counter()-t0)*1000:.1f}ms exception: {type(e).__name__}: {e}")
            break

    print(f"Final state: {ws.state.name}")
    try:
        await ws.close()
    except Exception:
        pass

asyncio.run(test_finalize_behavior())
