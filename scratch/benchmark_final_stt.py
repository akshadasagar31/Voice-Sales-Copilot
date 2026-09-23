import os
import sys
import time
import json
import asyncio
import wave

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "backend"))
sys.stdout.reconfigure(encoding='utf-8')

import httpx
import websockets
from dotenv import load_dotenv

load_dotenv("backend/.env")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
FASTAPI_PORT = 8001

test_cases = [
    {
        "id": "EN_FIN",
        "lang": "en",
        "label": "English Financial (CIBIL, 15 lakh, EMI, HDFC)",
        "file": "scratch/financial_en_48k.wav",
        "expected_terms": ["CIBIL", "15 lakh", "EMI", "loan", "HDFC"]
    },
    {
        "id": "HI_FIN",
        "lang": "hi",
        "label": "Hindi Financial (15 लाख, सिबिल, ईएमआई)",
        "file": "scratch/financial_hi_48k.wav",
        "expected_terms": ["15 lakh", "CIBIL score", "personal loan"]
    },
    {
        "id": "MR_FIN",
        "lang": "mr",
        "label": "Marathi Financial (10 लाख, सिबिल, कर्ज, ईएमआई)",
        "file": "scratch/financial_mr_48k.wav",
        "expected_terms": ["10 लाख", "personal loan", "CIBIL"]
    }
]

async def benchmark_batch_http(wav_bytes: bytes, lang: str):
    """Measures the 'Before' architecture: full audio sent after speech stops via HTTP multipart."""
    url = f"http://127.0.0.1:{FASTAPI_PORT}/api/voice-entry"
    files = {"file": ("recording.wav", wav_bytes, "audio/wav")}
    data = {"language": lang}
    
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=15.0) as client:
        res = await client.post(url, files=files, data=data)
    t1 = time.perf_counter()
    
    post_speech_latency = (t1 - t0) * 1000
    res_data = res.json()
    transcript = res_data.get("transcript", "")
    return {
        "post_speech_latency_ms": post_speech_latency,
        "first_interim_ms": None, # Never shows interim results in batch mode
        "transcript": transcript,
    }

async def benchmark_streaming_ws(pcm_bytes: bytes, lang: str):
    """Measures the 'After' architecture: real-time streaming WebSocket with interim transcripts."""
    ws_url = f"ws://127.0.0.1:{FASTAPI_PORT}/ws/voice-stt?sample_rate=48000&language={lang}"
    
    first_interim_time = None
    interim_count = 0
    final_transcript = ""
    post_speech_latency = None
    
    chunk_samples = 4096
    chunk_bytes = chunk_samples * 2 # 16-bit PCM = 8192 bytes
    t_start = time.perf_counter()
    
    async with websockets.connect(ws_url) as ws:
        async def receiver():
            nonlocal first_interim_time, interim_count, final_transcript
            async for msg in ws:
                data = json.loads(msg)
                m_type = data.get("type")
                if m_type == "interim":
                    if first_interim_time is None:
                        first_interim_time = (time.perf_counter() - t_start) * 1000
                    interim_count += 1
                elif m_type == "final":
                    final_transcript = data.get("transcript", "")
                    break
                    
        recv_task = asyncio.create_task(receiver())
        
        # Stream chunks simulating speech duration at natural 48kHz audio playback rate
        offset = 0
        while offset < len(pcm_bytes):
            chunk = pcm_bytes[offset:offset+chunk_bytes]
            await ws.send(chunk)
            offset += chunk_bytes
            await asyncio.sleep(4096 / 48000) # ~85ms
            
        t_speech_end = time.perf_counter()
        
        # Send CloseStream when speech ends (simulating silence detector trigger)
        await ws.send(json.dumps({"type": "CloseStream"}))
        
        try:
            await asyncio.wait_for(recv_task, timeout=4.0)
        except asyncio.TimeoutError:
            recv_task.cancel()
            
        t_done = time.perf_counter()
        post_speech_latency = (t_done - t_speech_end) * 1000
        
    return {
        "post_speech_latency_ms": post_speech_latency,
        "first_interim_ms": first_interim_time or 0.0,
        "interim_count": interim_count,
        "transcript": final_transcript,
    }

async def main():
    print("=" * 85)
    print("MODULE 2 STT BENCHMARK: BEFORE (BATCH HTTP) vs AFTER (STREAMING WEBSOCKET)")
    print("Deepgram Model: Nova-3 | Keyterms Boosted: True | LLM Rewriting: False (Verbatim)")
    print("=" * 85)
    
    summary = []
    
    for tc in test_cases:
        file_path = tc["file"]
        if not os.path.exists(file_path):
            print(f"File {file_path} not found. Skipping.")
            continue
            
        with open(file_path, "rb") as f:
            wav_bytes = f.read()
            
        with wave.open(file_path, "rb") as wf:
            pcm_bytes = wf.readframes(48000 * 30)
            
        duration = len(pcm_bytes) / (48000 * 2)
        print(f"\n--- Testing {tc['label']} (Duration: {duration:.2f}s, {len(pcm_bytes)} PCM bytes) ---")
        
        # 1. Measure Before (Batch HTTP)
        before = await benchmark_batch_http(wav_bytes, tc["lang"])
        print(f"[BEFORE - Batch HTTP]")
        print(f"  Post-speech latency: {before['post_speech_latency_ms']:.1f}ms")
        print(f"  First interim text:  None (0ms user visual feedback during speech)")
        print(f"  Transcript:          {before['transcript']!r}")
        
        # 2. Measure After (Streaming WS)
        after = await benchmark_streaming_ws(pcm_bytes, tc["lang"])
        print(f"[AFTER - Streaming WS]")
        print(f"  Post-speech latency: {after['post_speech_latency_ms']:.1f}ms")
        print(f"  First interim text:  {after['first_interim_ms']:.1f}ms ({after['interim_count']} live updates during speech)")
        print(f"  Transcript:          {after['transcript']!r}")
        
        speedup = before['post_speech_latency_ms'] / max(1, after['post_speech_latency_ms'])
        
        summary.append({
            "label": tc["label"],
            "duration_s": round(duration, 2),
            "before_latency_ms": round(before["post_speech_latency_ms"], 1),
            "after_interim_ms": round(after["first_interim_ms"], 1),
            "after_latency_ms": round(after["post_speech_latency_ms"], 1),
            "speedup": round(speedup, 2),
            "before_transcript": before["transcript"],
            "after_transcript": after["transcript"],
            "expected_terms": tc["expected_terms"]
        })
        
    print("\n" + "=" * 105)
    print("FINAL LATENCY & ACCURACY BENCHMARK TABLE")
    print("=" * 105)
    print(f"{'Language Scenario':<36} | {'Before Post-Speech':<18} | {'After 1st Interim':<18} | {'After Post-Speech':<18} | {'Speedup':<8}")
    print("-" * 105)
    for s in summary:
        print(f"{s['label']:<36} | {s['before_latency_ms']:>15.1f}ms | {s['after_interim_ms']:>15.1f}ms | {s['after_latency_ms']:>15.1f}ms | {s['speedup']:>6.2f}x")
        
    print("\n" + "=" * 105)
    print("TRANSCRIPT FIDELITY & FINANCIAL TERM MATCHING")
    print("=" * 105)
    for s in summary:
        print(f"\nScenario: {s['label']}")
        print(f"  Key Expected Terms:  {', '.join(s['expected_terms'])}")
        print(f"  Batch Transcript:    {s['before_transcript']}")
        print(f"  Verbatim Transcript: {s['after_transcript']}")
        
    with open("scratch/final_stt_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nSaved benchmark results to scratch/final_stt_benchmark.json")

if __name__ == "__main__":
    asyncio.run(main())
