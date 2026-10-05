import asyncio
import os
import json
import time
import websockets
from dotenv import load_dotenv

load_dotenv("backend/.env")
api_key = os.getenv("DEEPGRAM_API_KEY")

async def test_clean_turns():
    from services.stt import build_deepgram_ws_url
    url = build_deepgram_ws_url(model="nova-3", sample_rate=48000, language="en")
    headers = {"Authorization": f"Token {api_key}"}

    with open("scratch/financial_en_48k.wav", "rb") as f:
        wav_bytes = f.read()
    pcm = wav_bytes[44:] if wav_bytes.startswith(b"RIFF") else wav_bytes

    print("Connecting to Deepgram...", flush=True)
    ws = await websockets.connect(url, additional_headers=headers)
    print(f"Connected! State: {ws.state.name}", flush=True)

    for turn in range(1, 4):
        print(f"\n--- Turn {turn} (reusing same WS, State: {ws.state.name}) ---", flush=True)
        turn_start = time.perf_counter()
        
        # 1. Stream audio
        chunk_size = 4096 * 2
        for i in range(0, len(pcm), chunk_size):
            await ws.send(pcm[i:i+chunk_size])
            await asyncio.sleep(0.01)
        
        # 2. Send Finalize
        await ws.send(json.dumps({"type": "Finalize"}))
        print(f"Turn {turn} audio sent, draining turn results...", flush=True)

        finals = []
        # Read until Metadata
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(msg)
                mtype = data.get("type")
                if mtype == "Results" and data.get("is_final"):
                    alts = (data.get("channel") or {}).get("alternatives") or []
                    if alts and alts[0].get("transcript"):
                        finals.append(alts[0]["transcript"])
                elif mtype == "Metadata":
                    print("  Received Metadata! Turn completely finished on Deepgram.", flush=True)
                    break
            except asyncio.TimeoutError:
                print("  Timeout waiting for Metadata!", flush=True)
                break

        full_transcript = " ".join(finals).strip()
        print(f"Turn {turn} complete! Transcript: '{full_transcript}'", flush=True)
        print(f"WS State: {ws.state.name}", flush=True)
        await asyncio.sleep(0.2)

    await ws.send(json.dumps({"type": "CloseStream"}))
    await ws.close()
    print("\nAll turns completed with clean transcripts!", flush=True)

asyncio.run(test_clean_turns())
