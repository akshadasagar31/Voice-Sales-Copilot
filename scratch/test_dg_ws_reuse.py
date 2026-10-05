import asyncio
import os
import json
import time
import websockets
from dotenv import load_dotenv

load_dotenv("backend/.env")
api_key = os.getenv("DEEPGRAM_API_KEY")

async def test_reuse():
    from services.stt import build_deepgram_ws_url
    url = build_deepgram_ws_url(model="nova-3", sample_rate=48000, language="en")
    headers = {"Authorization": f"Token {api_key}"}
    print(f"Connecting to Deepgram: {url[:60]}...")
    t0 = time.perf_counter()
    async with websockets.connect(url, additional_headers=headers) as ws:
        t_conn = (time.perf_counter() - t0) * 1000
        print(f"Connected in {t_conn:.1f}ms! State: {ws.state.name}")

        # Send some dummy 48kHz audio (1 sec silence)
        dummy_pcm = b"\x00\x00" * 48000
        
        # Turn 1
        print("Starting Turn 1...")
        await ws.send(dummy_pcm)
        await ws.send(json.dumps({"type": "Finalize"}))
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            t = data.get("type")
            print("Turn 1 got:", t)
            if t in ("Metadata",):
                break
        print(f"Turn 1 completed! WS state: {ws.state.name}")

        # Wait a moment
        await asyncio.sleep(0.5)

        # Turn 2 on the SAME websocket
        print("Starting Turn 2 on same WS...")
        await ws.send(dummy_pcm)
        await ws.send(json.dumps({"type": "Finalize"}))
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            t = data.get("type")
            print("Turn 2 got:", t)
            if t in ("Metadata",):
                break
        print(f"Turn 2 completed! WS state: {ws.state.name}")

        await ws.send(json.dumps({"type": "CloseStream"}))
        print("CloseStream sent.")

asyncio.run(test_reuse())
