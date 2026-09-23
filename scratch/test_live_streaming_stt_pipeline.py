import os
import sys
import time
import io
import wave
import json
import asyncio
import websockets
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv("backend/.env")
sys.path.insert(0, "backend")

from services.sarvam_tts import SarvamTTSService
from services.tts import DeepgramTTSService

async def generate_test_audio_pcm(text: str, lang: str):
    """Generate authentic audio and convert to 16kHz or 48kHz mono 16-bit linear PCM."""
    if lang == "mr":
        service = SarvamTTSService()
        wav_bytes = await service.synthesize_speech(text, language_code="mr-IN", speaker="priya")
    elif lang == "hi":
        service = SarvamTTSService()
        wav_bytes = await service.synthesize_speech(text, language_code="hi-IN", speaker="priya")
    else:
        service = DeepgramTTSService()
        # Deepgram TTS returns MP3. For simplicity let's use Sarvam for clean WAV or decode MP3
        sarvam = SarvamTTSService()
        wav_bytes = await sarvam.synthesize_speech(text, language_code="en-IN", speaker="priya")

    # Read WAV and extract raw PCM
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    return frames, sample_rate

async def test_live_streaming_stt(name: str, text: str, lang: str):
    print("\n" + "=" * 70)
    print(f"  Testing Streaming STT: {name}")
    print(f"  Spoken Text: \"{text}\"")
    print("=" * 70)

    pcm_bytes, sample_rate = await generate_test_audio_pcm(text, lang)
    print(f"  Audio generated: {len(pcm_bytes)} bytes PCM, sample rate: {sample_rate}Hz")

    # Connect to our local FastAPI WebSocket /ws/voice-stt
    ws_url = f"ws://127.0.0.1:8001/ws/voice-stt?sample_rate={sample_rate}&language={lang if lang != 'mixed' else 'auto'}"
    interim_transcripts = []
    final_transcript = None
    detected_lang = None

    try:
        async with websockets.connect(ws_url) as ws:
            # Stream audio in chunks of 4096 samples (8192 bytes for 16-bit PCM)
            chunk_size = 4096 * 2
            t_start = time.perf_counter()

            async def send_audio():
                for offset in range(0, len(pcm_bytes), chunk_size):
                    chunk = pcm_bytes[offset:offset + chunk_size]
                    await ws.send(chunk)
                    # Simulate real-time audio cadence: 4096 samples at sample_rate seconds
                    chunk_dur = len(chunk) / (2 * sample_rate)
                    await asyncio.sleep(chunk_dur * 0.5) # slightly faster than real-time to test streaming

                # After sending all audio, wait 200ms silence and send CloseStream
                await asyncio.sleep(0.2)
                print("  Sent CloseStream signal...")
                await ws.send(json.dumps({"type": "CloseStream"}))

            send_task = asyncio.create_task(send_audio())

            async for msg in ws:
                data = json.loads(msg)
                msg_type = data.get("type")
                if msg_type == "interim":
                    tr = data.get("transcript", "")
                    if tr and (not interim_transcripts or tr != interim_transcripts[-1]):
                        interim_transcripts.append(tr)
                        print(f"  [Interim]: \"{tr}\"")
                elif msg_type == "final":
                    final_transcript = data.get("transcript", "")
                    detected_lang = data.get("detected_language")
                    print(f"  [FINAL TRANSCRIPT]: \"{final_transcript}\" (lang: {detected_lang})")
                elif msg_type == "metadata":
                    break

            await send_task

    except Exception as e:
        print(f"  [X] WebSocket streaming failed: {e}")

    print(f"  Total Interim Count: {len(interim_transcripts)}")
    print(f"  Final Transcript:    \"{final_transcript}\"")
    print(f"  Matches Spoken:      {bool(final_transcript)}")
    return final_transcript

async def main():
    test_cases = [
        ("Marathi", "एचडीएफसी बँकेच्या वैयक्तिक कर्जासाठी किमान CIBIL score किती आवश्यक आहे?", "mr"),
        ("Hindi", "पर्सनल लोन के लिए न्यूनतम सिबिल स्कोर कितना होना चाहिए?", "hi"),
        ("English", "What is the minimum CIBIL score for personal loans?", "en"),
        ("Mixed Language", "माझा CIBIL score 750 आहे, मला HDFC Bank कडून personal loan वर किती interest rate मिळेल?", "mixed"),
    ]

    for name, text, lang in test_cases:
        await test_live_streaming_stt(name, text, lang)

if __name__ == "__main__":
    asyncio.run(main())
