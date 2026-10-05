import asyncio
import os
import json
import time
import websockets
from dotenv import load_dotenv

load_dotenv("backend/.env")
api_key = os.getenv("DEEPGRAM_API_KEY")

async def test_keepalive_after_finalize():
    from services.stt import build_deepgram_ws_url
    url = build_deepgram_ws_url(model="nova-3", sample_rate=48000, language="en")
    headers = {"Authorization": f"Token {api_key}"}

    ws = await websockets.connect(url, additional_headers=headers)
    print(f"Connected! State: {ws.state.name}")

    with open("scratch/financial_en_48k.wav", "rb") as f:
        pcm = f.read()[44:]
    
    # Send some audio
    await ws.send(pcm[:48000 * 2])
    print("Sent 1s audio, now sending Finalize...")
    await ws.send(json.dumps({"type": "Finalize"}))

    # Read until Metadata
    while True:
        msg = await ws.recv()
        d = json.loads(msg)
        if d.get("type") == "Metadata":
            print("Received Metadata!")
            break

    print(f"State right after Metadata: {ws.state.name}")
    # Now send KeepAlive every 2s for 6s
    for i in range(3):
        await asyncio.sleep(2.0)
        await ws.send(json.dumps({"type": "KeepAlive"}))
        print(f"KeepAlive {i+1} sent! State: {ws.state.name}")
    
    print(f"Finished! Still open? {ws.state.name}")
    await ws.close()

asyncio.run(test_keepalive_after_finalize())
