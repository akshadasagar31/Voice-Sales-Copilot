import os
import sys
import time
import json
import asyncio

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "backend"))

import httpx
import websockets
from dotenv import load_dotenv

load_dotenv("backend/.env")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

test_audio_path = "backend/test_hi_dev.mp3"

# Also synthesize test audio in English, Hindi, and Marathi with financial terms using Cartesia TTS
async def generate_test_financial_audio():
    from backend.services.tts import DeepgramTTSService
    tts_service = DeepgramTTSService()
    
    test_cases = [
        {
            "name": "en_financial",
            "lang": "en",
            "text": "What is the minimum CIBIL score required for a personal loan of 15 lakh with HDFC and ICICI, and what is the monthly EMI?",
            "file": "scratch/test_en_financial.mp3"
        },
        {
            "name": "hi_financial",
            "lang": "hi",
            "text": "15 लाख के पर्सनल लोन के लिए सिबिल स्कोर कितना होना चाहिए और ईएमआई कितनी आएगी?",
            "file": "scratch/test_hi_financial.mp3"
        },
        {
            "name": "mr_financial",
            "lang": "mr",
            "text": "10 लाख रुपयांच्या कर्जासाठी सिबिल स्कोर किती लागतो आणि दरमहा ईएमआई किती असेल?",
            "file": "scratch/test_mr_financial.mp3"
        }
    ]
    
    for tc in test_cases:
        if not os.path.exists(tc["file"]):
            print(f"Synthesizing {tc['name']} using Deepgram Aura TTS...")
            audio_bytes = await tts_service.synthesize_speech(tc["text"], language=tc["lang"])
            with open(tc["file"], "wb") as f:
                f.write(audio_bytes)
            print(f"Saved {tc['file']} ({len(audio_bytes)} bytes)")
        else:
            print(f"Found existing {tc['file']}")

async def test_batch_http(audio_bytes: bytes, mime_type: str = "audio/wav", use_keyterms: bool = True):
    """Simulates existing batch STT path."""
    url = "https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true&punctuate=true"
    if use_keyterms:
        from backend.services.stt import FINANCIAL_KEYTERMS
        import urllib.parse
        kt_params = "&".join(f"keyterm={urllib.parse.quote(kt)}" for kt in FINANCIAL_KEYTERMS[:15])
        url += f"&{kt_params}"
    
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": mime_type,
    }
    
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=15.0) as client:
        res = await client.post(url, headers=headers, content=audio_bytes)
    t1 = time.perf_counter()
    
    res_data = res.json()
    transcript = ""
    try:
        transcript = res_data["results"]["channels"][0]["alternatives"][0]["transcript"]
    except Exception as e:
        transcript = f"Error: {e}, text: {res.text[:100]}"
    
    total_time = (t1 - t0) * 1000
    # In batch mode:
    # First interim visible time = None (infinity / until batch finishes)
    # Post-speech latency = total_time (everything happens after speech ends)
    return {
        "mode": "batch_http",
        "time_to_first_interim_ms": total_time, # user sees nothing until done
        "post_speech_latency_ms": total_time,
        "total_turn_around_ms": total_time,
        "transcript": transcript,
        "status": res.status_code
    }

async def test_streaming_ws(audio_bytes: bytes, is_mp3: bool = True, sample_rate: int = 16000, use_keyterms: bool = True):
    """Simulates real-time streaming WebSocket STT path."""
    from backend.services.stt import FINANCIAL_KEYTERMS
    import urllib.parse
    
    if is_mp3:
        ws_url = f"wss://api.deepgram.com/v1/listen?model=nova-3&smart_format=true&punctuate=true&interim_results=true&endpointing=300"
    else:
        ws_url = f"wss://api.deepgram.com/v1/listen?model=nova-3&encoding=linear16&sample_rate={sample_rate}&smart_format=true&punctuate=true&interim_results=true&endpointing=300"
        
    if use_keyterms:
        kt_params = "&".join(f"keyterm={urllib.parse.quote(kt)}" for kt in FINANCIAL_KEYTERMS[:15])
        ws_url += f"&{kt_params}"
        
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
    
    chunk_size = 2048 if is_mp3 else int(sample_rate * 2 * 0.05) # ~50ms chunk
    first_interim_delay = None
    post_speech_latency = None
    final_transcript = ""
    interim_transcripts = []
    
    t_start = time.perf_counter()
    end_speech_time = None
    
    async with websockets.connect(ws_url, additional_headers=headers) as ws:
        async def receiver():
            nonlocal first_interim_delay, post_speech_latency, final_transcript, interim_transcripts
            async for msg in ws:
                try:
                    data = json.loads(msg)
                    if data.get("type") == "Results":
                        channel = data.get("channel", {})
                        alts = channel.get("alternatives", [])
                        if alts and alts[0].get("transcript"):
                            text = alts[0]["transcript"].strip()
                            if text:
                                if first_interim_delay is None:
                                    first_interim_delay = (time.perf_counter() - t_start) * 1000
                                if data.get("is_final"):
                                    final_transcript = (final_transcript + " " + text).strip()
                                else:
                                    interim_transcripts.append(text)
                except Exception:
                    break
                    
        recv_task = asyncio.create_task(receiver())
        
        # Stream chunks with 50ms pacing (simulating live mic speech)
        offset = 0
        while offset < len(audio_bytes):
            chunk = audio_bytes[offset:offset+chunk_size]
            await ws.send(chunk)
            offset += chunk_size
            await asyncio.sleep(0.045) # real-time pace
            
        end_speech_time = time.perf_counter()
        
        # Signal finalize/end of audio stream
        await ws.send(json.dumps({"type": "CloseStream"}))
        
        try:
            await asyncio.wait_for(recv_task, timeout=4.0)
        except asyncio.TimeoutError:
            recv_task.cancel()
            
        post_speech_latency = (time.perf_counter() - end_speech_time) * 1000
        
    return {
        "mode": "streaming_ws",
        "time_to_first_interim_ms": first_interim_delay or 0,
        "post_speech_latency_ms": post_speech_latency,
        "total_turn_around_ms": post_speech_latency, # from speech stop to final transcript
        "transcript": final_transcript,
        "interim_count": len(interim_transcripts),
        "status": 200
    }

async def main():
    print("=" * 60)
    print("MODULE 2 STT BENCHMARK: BATCH HTTP vs STREAMING WEBSOCKET")
    print("=" * 60)
    
    # 1. Synthesize financial test audio in EN, HI, MR
    await generate_test_financial_audio()
    
    test_files = [
        ("English Financial", "scratch/test_en_financial.mp3", "en"),
        ("Hindi Financial", "scratch/test_hi_financial.mp3", "hi"),
        ("Marathi Financial", "scratch/test_mr_financial.mp3", "mr"),
    ]
    
    results = []
    
    for label, file_path, lang in test_files:
        print(f"\n--- Testing {label} ({file_path}) ---")
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
            
        audio_dur = len(audio_bytes) / 4000 # approximate duration for speech mp3
        print(f"Audio File Size: {len(audio_bytes)} bytes")
        
        # A. Batch HTTP (without keyterms)
        b_nokt = await test_batch_http(audio_bytes, "audio/mp3", use_keyterms=False)
        print(f"1. Batch HTTP (No keyterms):")
        print(f"   Post-speech latency: {b_nokt['post_speech_latency_ms']:.1f}ms")
        print(f"   Transcript: '{b_nokt['transcript']}'")
        
        # B. Batch HTTP (with keyterms)
        b_kt = await test_batch_http(audio_bytes, "audio/mp3", use_keyterms=True)
        print(f"2. Batch HTTP (With keyterms):")
        print(f"   Post-speech latency: {b_kt['post_speech_latency_ms']:.1f}ms")
        print(f"   Transcript: '{b_kt['transcript']}'")
        
        # C. Streaming WebSocket (with keyterms)
        s_kt = await test_streaming_ws(audio_bytes, is_mp3=True, use_keyterms=True)
        print(f"3. Streaming WebSocket (With keyterms):")
        print(f"   Time to first interim transcript: {s_kt['time_to_first_interim_ms']:.1f}ms")
        print(f"   Post-speech latency: {s_kt['post_speech_latency_ms']:.1f}ms")
        print(f"   Transcript: '{s_kt['transcript']}'")
        print(f"   Interim updates: {s_kt['interim_count']}")
        
        results.append({
            "language": label,
            "duration": audio_dur,
            "batch_latency_ms": b_kt['post_speech_latency_ms'],
            "streaming_interim_ms": s_kt['time_to_first_interim_ms'],
            "streaming_post_speech_ms": s_kt['post_speech_latency_ms'],
            "batch_transcript": b_kt['transcript'],
            "streaming_transcript": s_kt['transcript'],
        })
        
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Language':<20} | {'Batch Post-Speech':<18} | {'WS 1st Interim':<15} | {'WS Post-Speech':<15} | {'Speedup':<10}")
    print("-" * 85)
    for r in results:
        speedup = r['batch_latency_ms'] / max(1, r['streaming_post_speech_ms'])
        print(f"{r['language']:<20} | {r['batch_latency_ms']:>14.1f}ms | {r['streaming_interim_ms']:>11.1f}ms | {r['streaming_post_speech_ms']:>11.1f}ms | {speedup:>8.2f}x")

    with open("scratch/stt_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
