import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "backend"))

import json
import time
import asyncio
import websockets
import wave
from dotenv import load_dotenv

load_dotenv("backend/.env")

async def test_live_stream():
    # Convert test_hi_dev.mp3 or synthesize wav
    from backend.services.tts import DeepgramTTSService
    tts = DeepgramTTSService()
    
    text = "What is the CIBIL score required for a 15 lakh personal loan with HDFC Bank?"
    print(f"Synthesizing test audio: '{text}'...")
    audio_mp3 = await tts.synthesize_speech(text, language="en")
    
    # Save temp mp3 and convert to wav 16kHz mono using soundfile or wave
    # Since Deepgram WS expects linear16 PCM if sample_rate is specified
    ws_url = "ws://127.0.0.1:8001/ws/voice-stt?sample_rate=48000&language=en"
    
    print(f"Connecting to {ws_url}...")
    async with websockets.connect(ws_url) as ws:
        print("Connected!")
        
        # We can send raw audio or let's test sending audio
        # Note: browser MediaStream AudioContext provides Float32 chunks which we convert to Int16 PCM (linear16)!
        # Let's test sending Int16 PCM chunks
        
        # Run receiver task
        async def receiver():
            t0 = time.perf_counter()
            async for msg in ws:
                data = json.loads(msg)
                elapsed = (time.perf_counter() - t0) * 1000
                m_type = data.get("type")
                tr = data.get("transcript", "")
                print(f"[{elapsed:6.1f}ms] WS EVENT '{m_type}': {tr!r}")
                if m_type == "final":
                    print("--> Got final transcript!")
                    break
        
        recv_task = asyncio.create_task(receiver())
        
        # Stream 48000Hz linear16 silence + simulated tone or real pcm
        # Let's send 1.5 seconds of simulated audio
        chunk_samples = 4096 # standard ScriptProcessor buffer size
        chunk_bytes = chunk_samples * 2 # 16-bit
        dummy_chunk = b"\x00" * chunk_bytes
        
        for _ in range(15):
            await ws.send(dummy_chunk)
            await asyncio.sleep(4096 / 48000) # ~85ms
            
        print("Sending CloseStream...")
        await ws.send(json.dumps({"type": "CloseStream"}))
        
        try:
            await asyncio.wait_for(recv_task, timeout=3.0)
        except asyncio.TimeoutError:
            print("Receiver timed out.")
            recv_task.cancel()

if __name__ == "__main__":
    asyncio.run(test_live_stream())
