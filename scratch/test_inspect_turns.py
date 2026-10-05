import asyncio
import os
import json
import time
import math
import websockets
from dotenv import load_dotenv

load_dotenv("backend/.env")
api_key = os.getenv("DEEPGRAM_API_KEY")

async def test_inspect_turns():
    from services.stt import build_deepgram_ws_url
    url = build_deepgram_ws_url(model="nova-3", sample_rate=48000, language="en")
    headers = {"Authorization": f"Token {api_key}"}

    ws = await websockets.connect(url, additional_headers=headers)
    print(f"Connected! State: {ws.state.name}", flush=True)

    sample_rate = 48000
    tone = [int(8000 * math.sin(2 * math.pi * 300 * i / sample_rate)) for i in range(sample_rate)]
    pcm = b"".join(s.to_bytes(2, "little", signed=True) for s in tone)

    for turn in range(1, 4):
        print(f"\n=================== Turn {turn} ===================", flush=True)
        # Stream audio
        for i in range(0, len(pcm), 4800):
            await ws.send(pcm[i:i+4800])
            await asyncio.sleep(0.01)
        
        await ws.send(json.dumps({"type": "Finalize"}))
        print("Finalize sent, awaiting messages...", flush=True)

        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.5)
                data = json.loads(msg)
                mtype = data.get("type")
                is_final = data.get("is_final")
                speech_final = data.get("speech_final")
                alts = (data.get("channel") or {}).get("alternatives") or []
                tr = alts[0].get("transcript") if alts else ""
                print(f"[{mtype}] is_final={is_final}, speech_final={speech_final}, tr='{tr}'", flush=True)
                if mtype == "Metadata":
                    print("Turn reached Metadata boundary!", flush=True)
                    break
            except asyncio.TimeoutError:
                print(f"Turn {turn} timeout waiting for Metadata", flush=True)
                break
        
        print(f"Turn {turn} finished. WS State: {ws.state.name}", flush=True)

    await ws.send(json.dumps({"type": "CloseStream"}))
    await ws.close()

asyncio.run(test_inspect_turns())
