import asyncio
import json
import os
import sys
import websockets

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


async def simulate_browser_turn(audio_path, module, language_param=None):
    ws_url = f"ws://127.0.0.1:8001/ws/voice-stt?sample_rate=48000&module={module}"
    if language_param:
        ws_url += f"&language={language_param}"

    print(f"\n--- Testing {module.upper()} with audio '{os.path.basename(audio_path)}', param language='{language_param}' ---")
    print(f"Connecting to: {ws_url}")

    audio_bytes = open(audio_path, "rb").read()
    # Strip WAV header (44 bytes) to send raw PCM linear16
    pcm_bytes = audio_bytes[44:] if audio_bytes.startswith(b"RIFF") else audio_bytes

    events_received = []

    try:
        async with websockets.connect(ws_url) as ws:
            # Stream in chunks of 4096 bytes (simulating microphone frames)
            chunk_size = 4096
            for i in range(0, len(pcm_bytes), chunk_size):
                chunk = pcm_bytes[i : i + chunk_size]
                await ws.send(chunk)
                await asyncio.sleep(0.02)  # 20ms real-time pacing

            # Send Finalize
            print("Finished streaming audio frames. Sending {'type': 'Finalize'}...")
            await ws.send(json.dumps({"type": "Finalize"}))

            # Await response events from backend
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=12.0)
                    if isinstance(msg, str):
                        data = json.loads(msg)
                        m_type = data.get("type")
                        events_received.append(data)
                        if m_type == "interim":
                            pass  # interim transcript
                        elif m_type == "final":
                            print(">> RECEIVED TYPE=FINAL:")
                            print("   Transcript:", repr(data.get("transcript")))
                            print("   Detected Language:", data.get("detected_language"))
                            print("   Response Language:", data.get("language"))
                            print("   Immediate Sentence 1:", repr(data.get("immediate_sentence1")))
                            print("   Next Missing Param:", data.get("next_missing_parameter"))
                            print("   Lead:", data.get("lead"))
                            if not data.get("has_subsequent_sentences"):
                                break
                        elif m_type == "sentence":
                            print(">> RECEIVED TYPE=SENTENCE:", repr(data.get("sentence")))
                        elif m_type == "stream_complete":
                            print(">> RECEIVED TYPE=STREAM_COMPLETE")
                            break
                        elif m_type == "error":
                            print(">> RECEIVED TYPE=ERROR:", data.get("message"))
                            break
                except asyncio.TimeoutError:
                    print(">> TIMEOUT waiting for WS events")
                    break
    except Exception as e:
        print(f"WS Exception: {e}")

    return events_received


async def main():
    mr_wav = "scratch/financial_mr_48k.wav"
    hi_wav = "scratch/financial_hi_48k.wav"

    print("=================================================================")
    print("MODULE 1 TESTS (Live Call Voice Copilot)")
    print("=================================================================")
    # 1. Module 1 with default (auto / None) - what user experiences by default
    await simulate_browser_turn(mr_wav, module="module1", language_param=None)
    await simulate_browser_turn(hi_wav, module="module1", language_param=None)

    # 2. Module 1 with explicit language (if user selects dropdown)
    await simulate_browser_turn(mr_wav, module="module1", language_param="mr")
    await simulate_browser_turn(hi_wav, module="module1", language_param="hi")

    print("\n=================================================================")
    print("MODULE 2 TESTS (Knowledge Assistant)")
    print("=================================================================")
    # 3. Module 2 with default (auto / None)
    await simulate_browser_turn(mr_wav, module="module2", language_param=None)
    await simulate_browser_turn(hi_wav, module="module2", language_param=None)

    # 4. Module 2 with explicit language
    await simulate_browser_turn(mr_wav, module="module2", language_param="mr")
    await simulate_browser_turn(hi_wav, module="module2", language_param="hi")


if __name__ == "__main__":
    asyncio.run(main())
