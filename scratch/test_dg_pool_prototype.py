import asyncio
import os
import json
import time
import socket
import ssl
from pathlib import Path
import websockets
from dotenv import load_dotenv

load_dotenv("backend/.env")
api_key = os.getenv("DEEPGRAM_API_KEY")

async def test_3_consecutive_speech_turns():
    from services.stt import build_deepgram_ws_url
    url = build_deepgram_ws_url(model="nova-3", sample_rate=48000, language="en")
    headers = {"Authorization": f"Token {api_key}"}

    with open("scratch/financial_en_48k.wav", "rb") as f:
        wav_bytes = f.read()
    pcm = wav_bytes[44:] if wav_bytes.startswith(b"RIFF") else wav_bytes

    print(f"Connecting to Deepgram...", flush=True)
    t0 = time.perf_counter()
    ws = await websockets.connect(url, additional_headers=headers)
    print(f"Connected in {(time.perf_counter()-t0)*1000:.1f}ms! State: {ws.state.name}", flush=True)

    for turn in range(1, 4):
        print(f"\n--- Turn {turn} (reusing same WS, State: {ws.state.name}) ---", flush=True)
        turn_start = time.perf_counter()
        chunk_size = 4096 * 2
        for i in range(0, len(pcm), chunk_size):
            await ws.send(pcm[i:i+chunk_size])
            await asyncio.sleep(0.02)
        
        await ws.send(json.dumps({"type": "Finalize"}))
        print(f"Turn {turn} audio streamed. Awaiting transcript...", flush=True)

        got_transcript = ""
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                data = json.loads(msg)
                mtype = data.get("type")
                if mtype == "Results":
                    alts = (data.get("channel") or {}).get("alternatives") or []
                    if alts and alts[0].get("transcript"):
                        got_transcript = alts[0]["transcript"]
                        print(f"  [Result] is_final={data.get('is_final')}: '{got_transcript}'", flush=True)
                elif mtype == "Metadata":
                    print(f"  [Metadata] Turn boundary reached!", flush=True)
                    break
            except asyncio.TimeoutError:
                print("  Timeout waiting for Metadata", flush=True)
                break

        turn_duration = (time.perf_counter() - turn_start) * 1000
        print(f"Turn {turn} complete in {turn_duration:.1f}ms! Final transcript: '{got_transcript}'", flush=True)
        print(f"WS State: {ws.state.name}", flush=True)
        await asyncio.sleep(0.5)

    await ws.send(json.dumps({"type": "CloseStream"}))
    await ws.close()
    print("\nAll 3 consecutive turns completed successfully!", flush=True)

asyncio.run(test_3_consecutive_speech_turns())
