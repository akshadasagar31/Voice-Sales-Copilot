import asyncio
import sys
import json
import httpx
import websockets
from pathlib import Path
from dotenv import load_dotenv

backend_dir = Path(r"c:\Users\Akshada\OneDrive\Pictures\Documents\projects\Voice-Sales-Copilot\backend")
sys.path.insert(0, str(backend_dir))
sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(backend_dir / ".env")

from services.sarvam_tts import SarvamTTSService

TEST_CASES = [
    {
        "category": "Marathi",
        "spoken": "एचडीएफसी बँकेच्या वैयक्तिक कर्जासाठी किमान CIBIL score किती आवश्यक आहे?",
        "lang_code": "mr-IN",
        "speaker": "priya",
        "target_lang": "auto"
    },
    {
        "category": "Mixed Language (Marathi + English)",
        "spoken": "माझा CIBIL score 750 आहे, मला HDFC Bank कडून personal loan वर किती interest rate मिळेल?",
        "lang_code": "mr-IN",
        "speaker": "priya",
        "target_lang": "auto"
    },
    {
        "category": "Hindi",
        "spoken": "पर्सनल लोन के लिए न्यूनतम सिबिल स्कोर कितना होना चाहिए?",
        "lang_code": "hi-IN",
        "speaker": "aditya",
        "target_lang": "auto"
    },
    {
        "category": "English",
        "spoken": "What is the minimum CIBIL score for personal loans?",
        "lang_code": "en-IN",
        "speaker": "rahul",
        "target_lang": "auto"
    },
]

async def test_endpoint():
    svc = SarvamTTSService()
    ws_base_url = "ws://127.0.0.1:8001/ws/voice-stt"

    print("=" * 100)
    print("LIVE END-TO-END VERIFICATION: FASTAPI /ws/voice-stt (DUAL-STREAM AUTO-ROUTING)")
    print("=" * 100)

    results = []

    for item in TEST_CASES:
        cat = item["category"]
        spoken = item["spoken"]
        lang_code = item["lang_code"]
        speaker = item["speaker"]
        target_lang = item["target_lang"]

        # Generate audio using Sarvam TTS (synthesizes realistic human mic audio)
        audio_bytes = await svc.synthesize_speech(spoken, language_code=lang_code, speaker=speaker)

        import io, wave
        wav = wave.open(io.BytesIO(audio_bytes))
        sr = wav.getframerate()
        pcm_bytes = wav.readframes(wav.getnframes())

        url = f"{ws_base_url}?language={target_lang}&sample_rate={sr}&encoding=linear16"

        final_transcript = ""
        detected_lang = ""
        confidence = 0.0

        async with websockets.connect(url) as ws:
            async def receive():
                nonlocal final_transcript, detected_lang, confidence
                async for raw_msg in ws:
                    msg = json.loads(raw_msg)
                    if msg.get("type") == "final":
                        final_transcript = msg.get("transcript", "")
                        detected_lang = msg.get("detected_language", "")
                        confidence = msg.get("confidence", 0.0)
                    elif msg.get("type") == "metadata":
                        break

            recv_task = asyncio.create_task(receive())

            # Stream audio chunks simulating microphone streaming
            chunk_size = 4096
            for offset in range(0, len(pcm_bytes), chunk_size):
                chunk = pcm_bytes[offset:offset + chunk_size]
                await ws.send(chunk)
                await asyncio.sleep(0.04)

            await asyncio.sleep(0.2)
            await ws.send(json.dumps({"type": "CloseStream"}))

            try:
                await asyncio.wait_for(recv_task, timeout=5.0)
            except asyncio.TimeoutError:
                recv_task.cancel()

        print(f"\n[{cat}]")
        print(f"Spoken:           {spoken}")
        print(f"Final Transcript: {final_transcript}")
        print(f"Language:         {detected_lang} (Confidence: {confidence:.2f})")

        results.append({
            "category": cat,
            "spoken": spoken,
            "final": final_transcript,
            "lang": detected_lang,
            "conf": confidence
        })

    print("\n" + "=" * 100)
    print("VERIFICATION COMPLETE SUMMARY")
    print("=" * 100)
    for r in results:
        print(f"{r['category']:<35} | {r['lang']:<5} | '{r['final']}'")

if __name__ == "__main__":
    asyncio.run(test_endpoint())
