import asyncio
import os
import sys
import json
import time
from pathlib import Path
from starlette.testclient import TestClient

sys.path.insert(0, os.path.abspath("backend"))

from main import app
from services.stt import get_deepgram_ws_pool

def run_consecutive_turns_test():
    client = TestClient(app)
    
    with open("scratch/financial_en_48k.wav", "rb") as f:
        wav_bytes = f.read()
    pcm_bytes = wav_bytes[44:] if wav_bytes.startswith(b"RIFF") else wav_bytes

    print("\n=======================================================")
    print("TESTING 5 CONSECUTIVE VOICE TURNS OVER /ws/voice-stt")
    print("=======================================================")

    pool = get_deepgram_ws_pool()
    turn_results = []

    for turn in range(1, 6):
        print(f"\n--- Starting Turn {turn}/5 ---", flush=True)
        t_start = time.perf_counter()

        ws_url = "/ws/voice-stt?sample_rate=48000&language=en&module=module1"
        with client.websocket_connect(ws_url) as ws:
            t_connect = (time.perf_counter() - t_start) * 1000
            print(f"Turn {turn}: WebSocket connected in {t_connect:.1f}ms", flush=True)

            chunk_size = 4096 * 2
            # Stream audio
            for i in range(0, len(pcm_bytes), chunk_size):
                ws.send_bytes(pcm_bytes[i:i + chunk_size])
                time.sleep(0.01)

            # Signal speech ended
            t_speech_end = time.perf_counter()
            ws.send_text(json.dumps({"type": "Finalize"}))
            print(f"Turn {turn}: Sent Finalize, awaiting server response...", flush=True)

            final_msg = None
            sentences = []
            while True:
                try:
                    data = ws.receive_json()
                    msg_type = data.get("type")
                    if msg_type == "interim":
                        pass
                    elif msg_type == "final":
                        final_msg = data
                        print(f"Turn {turn}: Final transcript: '{data.get('transcript')}'", flush=True)
                    elif msg_type == "sentence":
                        sentences.append(data.get("sentence"))
                    elif msg_type == "stream_complete":
                        break
                    elif msg_type == "error":
                        print(f"Turn {turn} ERROR: {data.get('message')}", flush=True)
                        break
                except Exception as e:
                    print(f"Turn {turn} recv error: {e}", flush=True)
                    break

            t_turn_total = (time.perf_counter() - t_start) * 1000
            t_final_latency = (time.perf_counter() - t_speech_end) * 1000
            print(f"Turn {turn} COMPLETED in {t_turn_total:.1f}ms (SpeechEnd -> Response: {t_final_latency:.1f}ms)")
            assert final_msg is not None, f"Turn {turn} failed to receive final message"
            assert final_msg.get("type") == "final"
            assert len(final_msg.get("transcript", "")) > 0
            turn_results.append({
                "turn": turn,
                "duration_ms": t_turn_total,
                "latency_ms": t_final_latency,
                "transcript": final_msg.get("transcript"),
            })

        time.sleep(0.3)

    print("\n=======================================================")
    print("5 CONSECUTIVE VOICE TURNS SUMMARY:")
    for r in turn_results:
        print(f"  Turn {r['turn']}: Total={r['duration_ms']:.1f}ms, Latency={r['latency_ms']:.1f}ms | Transcript: '{r['transcript']}'")
    print("=======================================================")
    print("ALL 5 TURNS PASSED WITH NO DNS/503 FAILURES!")

if __name__ == "__main__":
    run_consecutive_turns_test()
