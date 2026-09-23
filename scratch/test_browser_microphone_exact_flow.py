import asyncio
import sys
import io
import wave
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import websockets
import httpx

backend_dir = Path(r"c:\Users\Akshada\OneDrive\Pictures\Documents\projects\Voice-Sales-Copilot\backend")
sys.path.insert(0, str(backend_dir))
sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(backend_dir / ".env")

from services.sarvam_tts import SarvamTTSService

TEST_CASES = [
    {
        "category": "Marathi",
        "spoken": "एचडीएफसी बँकेच्या वैयक्तिक कर्जासाठी किमान CIBIL score आणि दरमहा EMI किती आवश्यक आहे?",
        "lang_code": "mr-IN",
        "speaker": "priya",
    },
    {
        "category": "Mixed Language (Marathi + English)",
        "spoken": "माझा CIBIL score 750 आहे, मला HDFC Bank कडून personal loan वर किती interest rate आणि मासिक EMI मिळेल?",
        "lang_code": "mr-IN",
        "speaker": "priya",
    },
    {
        "category": "Hindi",
        "spoken": "मेरा CIBIL score 750 है, मुझे HDFC Bank से personal loan पर कितना interest rate और मासिक EMI मिलेगा?",
        "lang_code": "hi-IN",
        "speaker": "aditya",
    },
    {
        "category": "English",
        "spoken": "My CIBIL score is 750. What interest rate and monthly EMI can I get for a personal loan at HDFC Bank?",
        "lang_code": "en-IN",
        "speaker": "rahul",
    },
]

# Exact replica of browser's convertFloat32ToInt16 in KnowledgeAssistant.tsx
def convert_float32_to_int16(chunk_f32: np.ndarray) -> bytes:
    clipped = np.clip(chunk_f32, -1.0, 1.0)
    int16_arr = (clipped * 32767).astype(np.int16)
    return int16_arr.tobytes()

async def run_browser_microphone_test(svc: SarvamTTSService, test_case: dict):
    cat = test_case["category"]
    spoken = test_case["spoken"]
    lang_code = test_case["lang_code"]
    speaker = test_case["speaker"]

    # 1. Synthesize natural human speech
    wav_bytes = await svc.synthesize_speech(spoken, language_code=lang_code, speaker=speaker)

    # 2. Extract PCM frames and sample rate
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_int16 = wf.readframes(n_frames)

    # Convert to float32 (-1.0 to 1.0) exactly matching browser AudioBuffer channel data
    audio_int16 = np.frombuffer(raw_int16, dtype=np.int16)
    audio_f32 = (audio_int16 / 32768.0).astype(np.float32)

    # 3. Connect to /ws/voice-stt over WebSocket (mode: auto / dual-stream)
    ws_url = f"ws://127.0.0.1:8001/ws/voice-stt?sample_rate={sample_rate}&encoding=linear16"

    interims = []
    final_result = None

    async with websockets.connect(ws_url) as ws:
        async def listener():
            nonlocal final_result
            async for raw in ws:
                msg = json.loads(raw)
                mtype = msg.get("type")
                if mtype == "interim":
                    tr = (msg.get("transcript") or "").strip()
                    if tr and (not interims or interims[-1] != tr):
                        interims.append(tr)
                elif mtype == "final":
                    final_result = msg
                elif mtype == "metadata":
                    break

        listen_task = asyncio.create_task(listener())

        # 4. Stream 4096-sample Float32Array chunks at real-time cadence
        chunk_size = 4096
        chunk_duration = chunk_size / sample_rate

        for i in range(0, len(audio_f32), chunk_size):
            f32_chunk = audio_f32[i : i + chunk_size]
            # Exact browser conversion
            pcm16_bytes = convert_float32_to_int16(f32_chunk)
            await ws.send(pcm16_bytes)
            await asyncio.sleep(chunk_duration)

        # 5. Natural VAD pause (650ms)
        await asyncio.sleep(0.65)

        # 6. Finalize speech turn via CloseStream
        await ws.send(json.dumps({"type": "CloseStream"}))

        try:
            await asyncio.wait_for(listen_task, timeout=6.0)
        except asyncio.TimeoutError:
            listen_task.cancel()

    final_text = (final_result.get("transcript") if final_result else (interims[-1] if interims else "")).strip()
    detected_lang = final_result.get("detected_language") if final_result else "unknown"
    confidence = final_result.get("confidence") if final_result else 0.0

    return {
        "category": cat,
        "spoken": spoken,
        "final_transcript": final_text,
        "detected_language": detected_lang,
        "confidence": confidence,
        "interims_emitted": len(interims),
        "first_interim": interims[0] if interims else "",
    }

async def main():
    print("=" * 105)
    print("EXACT BROWSER MICROPHONE STT FLOW TEST (Float32Array -> convertFloat32ToInt16 -> /ws/voice-stt)")
    print("Checking CIBIL, 750, HDFC Bank, personal loan, EMI, and interest rate accuracy across all languages")
    print("=" * 105)

    svc = SarvamTTSService()
    results = []

    for tc in TEST_CASES:
        print(f"\n[Testing {tc['category']}]...")
        res = await run_browser_microphone_test(svc, tc)
        results.append(res)
        print(f"  Spoken:           '{res['spoken']}'")
        print(f"  Interims Emitted: {res['interims_emitted']} (First: '{res['first_interim']}')")
        print(f"  Final Transcript: '{res['final_transcript']}'")
        print(f"  Detected Lang:    {res['detected_language']} (Confidence: {res['confidence']:.2f})")
        await asyncio.sleep(1.0)

    print("\n" + "=" * 105)
    print("VERIFICATION SUMMARY: SPOKEN -> FINAL TRANSCRIPT")
    print("=" * 105)
    for r in results:
        print(f"\nCategory:          {r['category']}")
        print(f"Spoken Text:       {r['spoken']}")
        print(f"Final Transcript:  {r['final_transcript']}")
        print(f"Detected Language: {r['detected_language']} (Confidence: {r['confidence']:.2f})")

        # Check domain term accuracy
        print("Required Terms Checked:")
        for term in ["CIBIL", "750", "HDFC", "loan", "EMI", "interest rate"]:
            found = term.lower() in r["final_transcript"].lower() or term in r["final_transcript"]
            status = "PASS" if found else "NOTE"
            print(f"  - {term:<14}: [{status}]")

    print("\n" + "=" * 105)

if __name__ == "__main__":
    asyncio.run(main())
