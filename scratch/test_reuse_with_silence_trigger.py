import asyncio
import os
import json
import time
import websockets
from dotenv import load_dotenv

load_dotenv("backend/.env")
api_key = os.getenv("DEEPGRAM_API_KEY")

async def test_reuse_endpointing():
    from services.stt import build_deepgram_ws_url
    url = build_deepgram_ws_url(model="nova-3", sample_rate=48000, language="en")
    headers = {"Authorization": f"Token {api_key}"}

    with open("scratch/financial_en_48k.wav", "rb") as f:
        wav_bytes = f.read()
    pcm = wav_bytes[44:] if wav_bytes.startswith(b"RIFF") else wav_bytes

    print("Connecting to Deepgram...", flush=True)
    t0 = time.perf_counter()
    ws = await websockets.connect(url, additional_headers=headers)
    print(f"Connected in {(time.perf_counter()-t0)*1000:.1f}ms! State: {ws.state.name}", flush=True)

    silence_pcm = b"\x00\x00" * int(48000 * 0.6) # 600ms silence > 500ms endpointing

    for turn in range(1, 4):
        print(f"\n=================== Turn {turn} (reusing same WS, State: {ws.state.name}) ===================", flush=True)
        turn_start = time.perf_counter()
        
        # 1. Stream speech audio
        chunk_size = 4096 * 2
        for i in range(0, len(pcm), chunk_size):
            await ws.send(pcm[i:i+chunk_size])
            await asyncio.sleep(0.01)
        
        # 2. Client finalized speech -> send silence flush to trigger endpointing (without closing WS)
        print(f"Speech finished, sending silence flush to trigger endpointing...", flush=True)
        for i in range(0, len(silence_pcm), chunk_size):
            await ws.send(silence_pcm[i:i+chunk_size])
            await asyncio.sleep(0.01)

        finals = []
        # Read until speech_final or is_final after silence
        got_final = False
        while not got_final:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2.5)
                data = json.loads(msg)
                mtype = data.get("type")
                if mtype == "Results":
                    is_final = data.get("is_final")
                    speech_final = data.get("speech_final")
                    alts = (data.get("channel") or {}).get("alternatives") or []
                    tr = alts[0].get("transcript") if alts else ""
                    if is_final and tr:
                        finals.append(tr)
                        print(f"  [Result] is_final={is_final}, speech_final={speech_final}: '{tr}'", flush=True)
                    if speech_final:
                        print("  [speech_final=True reached!]", flush=True)
                        got_final = True
            except asyncio.TimeoutError:
                print("  Timeout waiting for speech_final", flush=True)
                break

        full_transcript = " ".join(finals).strip()
        dur = (time.perf_counter() - turn_start) * 1000
        print(f"Turn {turn} finished in {dur:.1f}ms! Transcript: '{full_transcript}'", flush=True)
        print(f"WS State: {ws.state.name}", flush=True)
        assert ws.state.name == "OPEN", f"WS closed on turn {turn}!"
        await asyncio.sleep(0.5)

    await ws.send(json.dumps({"type": "CloseStream"}))
    await ws.close()
    print("\nSUCCESS: All 3 turns reused the healthy WebSocket seamlessly!", flush=True)

asyncio.run(test_reuse_endpointing())
