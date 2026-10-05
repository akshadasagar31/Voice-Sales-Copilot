import asyncio
import os
import json
import time
import websockets
from dotenv import load_dotenv

load_dotenv("backend/.env")
api_key = os.getenv("DEEPGRAM_API_KEY")

async def test_endpointing_reuse():
    from services.stt import build_deepgram_ws_url
    url = build_deepgram_ws_url(model="nova-3", sample_rate=48000, language="en")
    headers = {"Authorization": f"Token {api_key}"}
    
    t0 = time.perf_counter()
    ws = await websockets.connect(url, additional_headers=headers)
    print(f"Connected in {(time.perf_counter()-t0)*1000:.1f}ms! State: {ws.state.name}", flush=True)

    # Generate 1 sec of tone (sine wave) + 1 sec of silence
    import math
    sample_rate = 48000
    tone_samples = [int(10000 * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(sample_rate)]
    tone_pcm = b"".join(s.to_bytes(2, "little", signed=True) for s in tone_samples)
    silence_pcm = b"\x00\x00" * int(sample_rate * 0.8) # 800ms silence (> 500ms endpointing)

    for turn in range(1, 4):
        print(f"\n--- Turn {turn} (reusing WS, state={ws.state.name}) ---", flush=True)
        # Send audio chunks
        chunk_size = 3840
        for i in range(0, len(tone_pcm), chunk_size):
            await ws.send(tone_pcm[i:i+chunk_size])
            await asyncio.sleep(0.04)
        
        # Send silence to trigger endpointing
        for i in range(0, len(silence_pcm), chunk_size):
            await ws.send(silence_pcm[i:i+chunk_size])
            await asyncio.sleep(0.04)
        
        # Read messages until speech_final or is_final
        got_final = False
        while not got_final:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                data = json.loads(msg)
                mtype = data.get("type")
                is_sp_final = data.get("speech_final")
                is_final = data.get("is_final")
                alts = (data.get("channel") or {}).get("alternatives") or []
                tr = alts[0].get("transcript") if alts else ""
                print(f"Turn {turn} msg: type={mtype}, is_final={is_final}, speech_final={is_sp_final}, tr='{tr}'", flush=True)
                if is_sp_final or (is_final and tr):
                    got_final = True
            except asyncio.TimeoutError:
                print(f"Turn {turn} timed out waiting for final", flush=True)
                break
        
        print(f"Turn {turn} finished. WS State: {ws.state.name}", flush=True)
        await asyncio.sleep(0.2)

    await ws.send(json.dumps({"type": "CloseStream"}))
    await ws.close()
    print("Closed cleanly.")

asyncio.run(test_endpointing_reuse())
