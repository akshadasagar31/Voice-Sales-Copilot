import asyncio
import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath("backend"))

from fastapi.testclient import TestClient
from main import app

def run_benchmark():
    client = TestClient(app)

    print("\n=======================================================")
    print("MODULE 1 VOICE LATENCY BENCHMARK")
    print("=======================================================")

    # 1. Test TTS Endpoint Latency (Sarvam Bulbul v3 & Cache)
    print("\n[Stage 1] Testing TTS Endpoint Latency (/api/tts)...")
    payload = {
        "text": "Hello, may I have your name please?",
        "language": "en",
        "module": "module1",
        "speaker": "simran"
    }

    t0 = time.perf_counter()
    r1 = client.post("/api/tts", json=payload)
    t_tts1 = (time.perf_counter() - t0) * 1000
    print(f"  TTS Run 1: Status={r1.status_code}, Bytes={len(r1.content)}, Latency={t_tts1:.1f}ms, Provider={r1.headers.get('X-TTS-Provider')}")

    t0 = time.perf_counter()
    r2 = client.post("/api/tts", json=payload)
    t_tts2 = (time.perf_counter() - t0) * 1000
    print(f"  TTS Run 2 (Cached): Status={r2.status_code}, Bytes={len(r2.content)}, Latency={t_tts2:.1f}ms, Cache={r2.headers.get('X-TTS-Cache', 'NONE')}")

    assert r1.status_code == 200, f"TTS failed with status {r1.status_code}"
    assert r2.status_code == 200, f"TTS cached run failed with status {r2.status_code}"
    assert t_tts2 < 50, f"TTS cache lookup was too slow: {t_tts2}ms"

    # 2. Test WS Voice STT Finalize Handshake
    print("\n[Stage 2] Testing /ws/voice-stt Finalize Fast-Path Latency (Turn 1)...")
    with client.websocket_connect("/ws/voice-stt?sample_rate=48000&language=en&module=module1") as ws:
        # Send silence chunk
        ws.send_bytes(b"\x00\x00" * 4800)
        t_fin = time.perf_counter()
        ws.send_text(json.dumps({"type": "Finalize"}))
        msg = ws.receive_json()
        t_recv1 = (time.perf_counter() - t_fin) * 1000
        print(f"  Turn 1: Finalize sent -> Received response in {t_recv1:.1f}ms: Type={msg.get('type')}, Transcript='{msg.get('transcript')}'")
        assert msg.get("type") == "final", f"Expected 'final' message, got {msg.get('type')}"

    print("\n[Stage 3] Testing /ws/voice-stt Finalize Fast-Path Latency (Turn 2 - Pooled WS)...")
    with client.websocket_connect("/ws/voice-stt?sample_rate=48000&language=en&module=module1") as ws:
        # Send silence chunk
        ws.send_bytes(b"\x00\x00" * 4800)
        t_fin = time.perf_counter()
        ws.send_text(json.dumps({"type": "Finalize"}))
        msg = ws.receive_json()
        t_recv2 = (time.perf_counter() - t_fin) * 1000
        print(f"  Turn 2: Finalize sent -> Received response in {t_recv2:.1f}ms: Type={msg.get('type')}, Transcript='{msg.get('transcript')}'")
        assert msg.get("type") == "final", f"Expected 'final' message, got {msg.get('type')}"
        assert t_recv2 < 600, f"Turn 2 Finalize was too slow: {t_recv2:.1f}ms (target < 600ms)"

    print("\n=======================================================")
    print("ALL MODULE 1 LATENCY BENCHMARKS PASSED SUCCESSFULLY!")
    print("=======================================================\n")

if __name__ == "__main__":
    run_benchmark()
