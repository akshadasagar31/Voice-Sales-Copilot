import asyncio
import json
import os
import sys
import time
import websockets
import httpx
from pathlib import Path

# Force UTF-8 encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

FASTAPI_URL = "http://127.0.0.1:8001"
WS_URL = "ws://127.0.0.1:8001/ws/voice-stt"

async def test_streaming_stt_and_tts(lang_name: str, language_param: str, audio_file: str, expected_lang: str):
    print(f"\n=======================================================")
    print(f"TESTING REAL STREAMING STT & TTS: {lang_name}")
    print(f"Language Param: '{language_param}' | Expected Resp Lang: '{expected_lang}'")
    print(f"=======================================================")

    audio_path = Path(audio_file)
    if not audio_path.exists():
        print(f"Audio file {audio_file} not found, generating or checking...")
        return False

    with open(audio_path, "rb") as f:
        wav_bytes = f.read()

    # Skip 44-byte WAV header to get raw 16-bit PCM bytes
    pcm_bytes = wav_bytes[44:] if wav_bytes.startswith(b"RIFF") else wav_bytes

    url = f"{WS_URL}?sample_rate=48000&language={language_param}&module=module1"
    print(f"Connecting to: {url}")

    start_stream_time = time.time()
    received_interims = []
    final_payload = None

    try:
        async with websockets.connect(url) as ws:
            chunk_size = 4096 * 2  # ~85ms at 48kHz 16-bit mono
            total_sent = 0

            # Stream audio chunks progressively (mimicking live browser microphone)
            for i in range(0, len(pcm_bytes), chunk_size):
                chunk = pcm_bytes[i:i + chunk_size]
                await ws.send(chunk)
                total_sent += len(chunk)
                await asyncio.sleep(0.04)  # stream at realistic speech rate

                # Check if interim results arrived while speaking
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                    data = json.loads(msg)
                    if data.get("type") == "interim":
                        received_interims.append(data.get("transcript"))
                        print(f"  [Interim while speaking]: {data.get('transcript')[:60]}...")
                    elif data.get("type") == "final":
                        final_payload = data
                        print(f"  [FAST-PATH FINAL RECEIVED]: {data.get('transcript')[:60]}...")
                except (asyncio.TimeoutError, Exception):
                    pass

            speech_end_time = time.time()
            if not final_payload:
                print(f"Speech streaming finished ({len(pcm_bytes)} bytes). Sending CloseStream...")

                # Client VAD detects silence -> sends CloseStream
                await ws.send(json.dumps({"type": "CloseStream"}))

                # Await final response from server
                async for raw_msg in ws:
                    data = json.loads(raw_msg)
                    msg_type = data.get("type")
                    if msg_type == "interim":
                        received_interims.append(data.get("transcript"))
                    elif msg_type == "final":
                        final_payload = data
                        break

            final_latency_ms = (time.time() - speech_end_time) * 1000
            print(f"  >>> Speech End -> Final STT & Deterministic Prompt: {final_latency_ms:.1f}ms <<<")

    except Exception as e:
        print(f"WebSocket STT error: {e}")
        return False

    if not final_payload:
        print("ERROR: Did not receive final payload from WebSocket!")
        return False

    transcript = final_payload.get("transcript", "")
    detected_lang = final_payload.get("detected_language", "")
    resp_lang = final_payload.get("language", "")
    sentence1 = final_payload.get("immediate_sentence1", "")
    lead = final_payload.get("lead", {})

    print(f"Verbatim Transcript : '{transcript}'")
    print(f"Detected Language   : {detected_lang}")
    print(f"Response Language   : {resp_lang} (Expected: {expected_lang})")
    print(f"Immediate Sentence 1: '{sentence1}'")
    print(f"Extracted Lead      : {lead}")

    assert resp_lang == expected_lang, f"Expected {expected_lang} but got {resp_lang}"

    # Now test immediate TTS for Sentence 1
    if sentence1:
        tts_start = time.time()
        async with httpx.AsyncClient(timeout=10.0) as client:
            tts_res = await client.post(
                f"{FASTAPI_URL}/api/tts",
                json={
                    "text": sentence1,
                    "language": resp_lang,
                    "module": "module1",
                    "speaker": "simran",
                },
            )
            tts_latency_ms = (time.time() - tts_start) * 1000
            provider = tts_res.headers.get("X-TTS-Provider", "unknown")
            voice = tts_res.headers.get("X-TTS-Voice", "unknown")
            audio_len = len(tts_res.content)
            total_turn_to_audio_ms = final_latency_ms + tts_latency_ms

            print(f"  >>> TTS Generation Latency: {tts_latency_ms:.1f}ms | Provider: {provider} | Voice: {voice} | Bytes: {audio_len} <<<")
            print(f"  >>> TOTAL SPEECH END TO FIRST AUDIO: {total_turn_to_audio_ms:.1f}ms (Target: ~1000ms) <<<")
            assert tts_res.status_code == 200, f"TTS status {tts_res.status_code}"
            assert audio_len > 1000, "TTS audio bytes empty"

    return True

async def main():
    results = {}
    fixtures = [
        ("English (en-IN)", "en", "scratch/financial_en_48k.wav", "en"),
        ("Hindi (hi)", "hi", "scratch/financial_hi_48k.wav", "hi"),
        ("Marathi (mr)", "mr", "scratch/financial_mr_48k.wav", "mr"),
    ]

    for name, lang_param, fixture, exp_lang in fixtures:
        ok = await test_streaming_stt_and_tts(name, lang_param, fixture, exp_lang)
        results[name] = ok

    print("\n=======================================================")
    print("ALL REAL SPEECH STREAMING TESTS SUMMARY:")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print("=======================================================")

if __name__ == "__main__":
    asyncio.run(main())
