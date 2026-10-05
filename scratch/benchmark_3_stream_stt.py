import asyncio
import json
import os
import sys
import time
import websockets
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv("backend/.env")
API_KEY = os.getenv("DEEPGRAM_API_KEY")

async def stream_single_dg(lang, pcm_bytes, sample_rate=48000):
    url = f"wss://api.deepgram.com/v1/listen?model=nova-3&sample_rate={sample_rate}&encoding=linear16&channels=1&language={lang}&smart_format=true&punctuate=true&interim_results=true&endpointing=500"
    headers = {"Authorization": f"Token {API_KEY}"}
    
    t_start = time.perf_counter()
    transcript_parts = []
    confidences = []
    t_first_interim = None
    t_final = None

    try:
        async with websockets.connect(url, additional_headers=headers) as ws:
            t_connect = time.perf_counter() - t_start

            async def send_audio():
                chunk_size = 4096
                for i in range(0, len(pcm_bytes), chunk_size):
                    await ws.send(pcm_bytes[i : i + chunk_size])
                    await asyncio.sleep(0.02)
                await ws.send(json.dumps({"type": "Finalize"}))

            async def recv_transcripts():
                nonlocal t_first_interim, t_final
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=6.0)
                        data = json.loads(msg)
                        m_type = data.get("type")
                        if m_type == "Results":
                            alts = (data.get("channel") or {}).get("alternatives") or []
                            if alts:
                                tr = (alts[0].get("transcript") or "").strip()
                                conf = alts[0].get("confidence", 0.0)
                                if tr and t_first_interim is None:
                                    t_first_interim = time.perf_counter() - t_start
                                if data.get("is_final") and tr:
                                    transcript_parts.append(tr)
                                    if conf > 0:
                                        confidences.append(conf)
                        elif m_type == "Metadata":
                            t_final = time.perf_counter() - t_start
                            break
                    except asyncio.TimeoutError:
                        break

            sender = asyncio.create_task(send_audio())
            receiver = asyncio.create_task(recv_transcripts())
            await asyncio.gather(sender, receiver)

    except Exception as e:
        import traceback
        print(f"[{lang}] Exception: {e}")
        traceback.print_exc()
        return {
            "lang": lang,
            "error": str(e),
            "transcript": "",
            "confidence": 0.0,
            "t_connect_ms": 0,
            "t_first_interim_ms": 0,
            "t_final_ms": 0,
        }

    full_tr = " ".join(transcript_parts).strip()
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return {
        "lang": lang,
        "transcript": full_tr,
        "confidence": round(avg_conf, 3),
        "t_connect_ms": round((t_connect or 0) * 1000, 1),
        "t_first_interim_ms": round((t_first_interim or 0) * 1000, 1),
        "t_final_ms": round((t_final or 0) * 1000, 1),
    }

async def benchmark_3_stream(audio_path, label):
    print(f"\n=======================================================")
    print(f"BENCHMARK: 3-Stream Concurrent Auto STT for: {label}")
    print(f"Audio file: {audio_path}")
    print(f"=======================================================")
    
    audio_bytes = open(audio_path, "rb").read()
    pcm_bytes = audio_bytes[44:] if audio_bytes.startswith(b"RIFF") else audio_bytes
    audio_duration_sec = len(pcm_bytes) / (48000 * 2)
    print(f"Audio Duration: {audio_duration_sec:.2f} seconds")

    t_wall_start = time.perf_counter()
    # Run 3 streams concurrently
    results = await asyncio.gather(
        stream_single_dg("en-IN", pcm_bytes),
        stream_single_dg("hi", pcm_bytes),
        stream_single_dg("mr", pcm_bytes),
    )
    t_wall_total = (time.perf_counter() - t_wall_start) * 1000

    print(f"\n--- Individual Stream Results (Concurrent Execution) ---")
    for r in results:
        print(f"Stream [{r['lang']}]:")
        print(f"   Transcript:     {r['transcript']!r}")
        print(f"   Confidence:     {r['confidence']}")
        print(f"   Connect Latency: {r['t_connect_ms']} ms")
        print(f"   1st Interim:    {r['t_first_interim_ms']} ms")
        print(f"   Final Result:   {r['t_final_ms']} ms")

    print(f"\nTotal Wall-Clock Time: {t_wall_total:.1f} ms")
    return results

async def main():
    await benchmark_3_stream("scratch/financial_mr_48k.wav", "Marathi Financial Audio")
    await benchmark_3_stream("scratch/financial_hi_48k.wav", "Hindi Financial Audio")

if __name__ == "__main__":
    asyncio.run(main())
