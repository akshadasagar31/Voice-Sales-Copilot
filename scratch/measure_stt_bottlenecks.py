import os
import sys
import time
import json
import asyncio
import httpx
import websockets
from dotenv import load_dotenv

# Load env from backend/.env
load_dotenv("backend/.env")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
FASTAPI_PORT = 8001

if not DEEPGRAM_API_KEY:
    print("ERROR: DEEPGRAM_API_KEY not found in backend/.env")
    sys.exit(1)

# Find test audio files
test_audio_path = "backend/test_hi_dev.mp3"
if not os.path.exists(test_audio_path):
    print(f"ERROR: {test_audio_path} not found")
    sys.exit(1)

print(f"Using test audio: {test_audio_path}")

async def measure_deepgram_batch_http(audio_bytes: bytes, mime_type: str = "audio/mp3"):
    """Measures direct Deepgram HTTP batch API latency."""
    url = "https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true&punctuate=true"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": mime_type,
    }
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=15.0) as client:
        res = await client.post(url, headers=headers, content=audio_bytes)
    t1 = time.perf_counter()
    duration = (t1 - t0) * 1000
    res_data = res.json()
    transcript = ""
    try:
        transcript = res_data["results"]["channels"][0]["alternatives"][0]["transcript"]
    except Exception:
        pass
    return duration, res.status_code, transcript

async def measure_fastapi_voice_entry(audio_bytes: bytes, filename: str = "test.mp3"):
    """Measures FastAPI /api/voice-entry latency (FastAPI -> Deepgram HTTP)."""
    url = f"http://127.0.0.1:{FASTAPI_PORT}/api/voice-entry"
    files = {"file": (filename, audio_bytes, "audio/mpeg")}
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=15.0) as client:
        res = await client.post(url, files=files)
    t1 = time.perf_counter()
    duration = (t1 - t0) * 1000
    transcript = ""
    try:
        transcript = res.json().get("transcript", "")
    except Exception:
        pass
    return duration, res.status_code, transcript

async def measure_nextjs_voice_entry(audio_bytes: bytes, filename: str = "test.mp3"):
    """Measures Next.js /api/voice-entry proxy latency (Next.js -> FastAPI -> Deepgram HTTP)."""
    url = "http://localhost:3000/api/voice-entry"
    files = {"file": (filename, audio_bytes, "audio/mpeg")}
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.post(url, files=files)
        t1 = time.perf_counter()
        duration = (t1 - t0) * 1000
        transcript = ""
        try:
            transcript = res.json().get("transcript", "")
        except Exception:
            pass
        return duration, res.status_code, transcript
    except Exception as e:
        return None, None, str(e)

async def test_nova3_keyterm_support():
    """Verify Deepgram Nova-3 keyterm vs keywords behavior and accuracy."""
    print("\n--- Verifying Deepgram Nova-3 'keyterm' Parameter Support ---")
    
    # 1. Test invalid query parameter 'keywords'
    url_keywords = "https://api.deepgram.com/v1/listen?model=nova-3&keywords=CIBIL:2&keywords=EMI:2"
    async with httpx.AsyncClient(timeout=10.0) as client:
        res_kw = await client.post(
            url_keywords, 
            headers={"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "audio/wav"},
            content=b"\x00" * 3200
        )
        print(f"Deepgram Nova-3 with 'keywords' status: {res_kw.status_code}, response: {res_kw.text[:120]}")
    
    # 2. Test valid query parameter 'keyterm'
    url_keyterm = "https://api.deepgram.com/v1/listen?model=nova-3&keyterm=CIBIL&keyterm=EMI&keyterm=lakh"
    async with httpx.AsyncClient(timeout=10.0) as client:
        res_kt = await client.post(
            url_keyterm, 
            headers={"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "audio/wav"},
            content=b"\x00" * 3200
        )
        print(f"Deepgram Nova-3 with 'keyterm' status: {res_kt.status_code}, success: {res_kt.status_code == 200}")

async def measure_streaming_ws_latency(audio_pcm: bytes, sample_rate: int = 16000):
    """
    Measures WebSocket streaming latency:
    Streams chunks incrementally over /ws/voice-stt (mimicking live mic speech).
    Then measures:
    - Time from first chunk to first interim result.
    - Time from last audio chunk sent to final transcript received (post-speech latency).
    """
    ws_url = f"ws://127.0.0.1:{FASTAPI_PORT}/ws/voice-stt?sample_rate={sample_rate}"
    chunk_size = int(sample_rate * 2 * 0.05) # 50ms chunk (16-bit PCM = 2 bytes/sample)
    
    first_interim_delay = None
    post_speech_latency = None
    interim_count = 0
    final_transcript = ""
    
    async with websockets.connect(ws_url) as ws:
        # Task to receive messages
        async def receiver():
            nonlocal first_interim_delay, post_speech_latency, interim_count, final_transcript
            t_start = time.perf_counter()
            end_speech_time = None
            
            while True:
                try:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    msg_type = data.get("type")
                    
                    if msg_type == "interim":
                        if first_interim_delay is None:
                            first_interim_delay = (time.perf_counter() - t_start) * 1000
                        interim_count += 1
                        
                    elif msg_type == "final":
                        final_transcript = data.get("transcript", "")
                        if end_speech_time is not None:
                            post_speech_latency = (time.perf_counter() - end_speech_time) * 1000
                        break
                except Exception as e:
                    break
        
        recv_task = asyncio.create_task(receiver())
        
        # Stream chunks with simulated real-time pacing (50ms interval)
        t_start = time.perf_counter()
        offset = 0
        while offset < len(audio_pcm):
            chunk = audio_pcm[offset:offset+chunk_size]
            await ws.send(chunk)
            offset += chunk_size
            await asyncio.sleep(0.04) # 40ms sleep for 50ms chunk
            
        end_speech_time = time.perf_counter()
        # Send end of speech
        await ws.send(json.dumps({"type": "finalize"}))
        
        # Now wait for receiver to get final transcript
        try:
            await asyncio.wait_for(recv_task, timeout=5.0)
        except asyncio.TimeoutError:
            recv_task.cancel()
            
    return first_interim_delay, post_speech_latency, interim_count, final_transcript

async def main():
    with open(test_audio_path, "rb") as f:
        audio_bytes = f.read()

    print(f"\n=======================================================")
    print(f"MEASURING ACTUAL MODULE 2 STT BOTTLENECKS")
    print(f"Audio file size: {len(audio_bytes)} bytes")
    print(f"=======================================================")
    
    # 1. Direct Deepgram HTTP batch API
    print("\n1. Measuring Direct Deepgram HTTP batch API...")
    times_dg = []
    for i in range(3):
        dur, status, tr = await measure_deepgram_batch_http(audio_bytes)
        times_dg.append(dur)
        print(f"   Trial {i+1}: {dur:.1f}ms (HTTP {status}) -> '{tr[:60]}...'")
    avg_dg = sum(times_dg) / len(times_dg)
    print(f"   Avg Direct Deepgram Batch HTTP: {avg_dg:.1f}ms")
    
    # 2. FastAPI /api/voice-entry (adds FastAPI multipart handling)
    print("\n2. Measuring FastAPI /api/voice-entry...")
    times_fa = []
    for i in range(3):
        dur, status, tr = await measure_fastapi_voice_entry(audio_bytes)
        times_fa.append(dur)
        print(f"   Trial {i+1}: {dur:.1f}ms (HTTP {status}) -> '{tr[:60]}...'")
    avg_fa = sum(times_fa) / len(times_fa)
    print(f"   Avg FastAPI /api/voice-entry: {avg_fa:.1f}ms (FastAPI overhead: {avg_fa - avg_dg:+.1f}ms)")
    
    # 3. Next.js /api/voice-entry (double-proxy: Next.js -> FastAPI -> Deepgram HTTP)
    print("\n3. Measuring Next.js /api/voice-entry (Double Proxy)...")
    times_next = []
    for i in range(3):
        dur, status, tr = await measure_nextjs_voice_entry(audio_bytes)
        if dur is not None:
            times_next.append(dur)
            print(f"   Trial {i+1}: {dur:.1f}ms (HTTP {status}) -> '{tr[:60]}...'")
        else:
            print(f"   Trial {i+1}: Next.js endpoint not reachable or error: {tr}")
    if times_next:
        avg_next = sum(times_next) / len(times_next)
        print(f"   Avg Next.js /api/voice-entry: {avg_next:.1f}ms (Total proxy overhead: {avg_next - avg_dg:+.1f}ms)")
    else:
        avg_next = None

    # 4. Verify Nova-3 keyterm support
    await test_nova3_keyterm_support()

if __name__ == "__main__":
    asyncio.run(main())
