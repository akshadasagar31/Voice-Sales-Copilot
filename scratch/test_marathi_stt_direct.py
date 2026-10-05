import os
import sys
import asyncio
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath("backend"))
load_dotenv("backend/.env")

from services.stt import DeepgramSTTService
from services.sarvam_stt import SarvamSTTService

async def main():
    wav_path = "scratch/financial_mr_48k.wav"
    if not os.path.exists(wav_path):
        print(f"File not found: {wav_path}")
        return
    with open(wav_path, "rb") as f:
        audio_bytes = f.read()

    print(f"Audio size: {len(audio_bytes)} bytes")

    # 1. Deepgram Nova-3 with mr and rich keyterms
    dg = DeepgramSTTService()
    try:
        dg_res = await dg.transcribe_audio(
            audio_bytes,
            content_type="audio/wav",
            model="nova-3",
            language="mr",
        )
        print("\n--- Deepgram Nova-3 (language='mr', with current keyterms) ---")
        print("Transcript:", dg_res.get("transcript"))
        print("Detected Lang:", dg_res.get("detected_language"))
        print("Confidence:", dg_res.get("confidence"))
    except Exception as e:
        print("Deepgram mr failed:", e)

    # 2. Deepgram Nova-3 with multi
    try:
        dg_multi = await dg.transcribe_audio(audio_bytes, content_type="audio/wav", model="nova-3", language="multi")
        print("\n--- Deepgram Nova-3 (language='multi') ---")
        print("Transcript:", dg_multi.get("transcript"))
        print("Detected Lang:", dg_multi.get("detected_language"))
        print("Confidence:", dg_multi.get("confidence"))
    except Exception as e:
        print("Deepgram multi failed:", e)

    # 3. Sarvam saaras:v4 with mr-IN
    sarvam = SarvamSTTService()
    try:
        sarvam_res = await sarvam.transcribe_audio(audio_bytes, content_type="audio/wav", language_code="mr-IN", model="saaras:v4")
        print("\n--- Sarvam saaras:v4 (language_code='mr-IN') ---")
        print("Transcript:", sarvam_res.get("transcript"))
        print("Detected Lang:", sarvam_res.get("detected_language"))
    except Exception as e:
        print("Sarvam mr-IN failed:", e)

if __name__ == "__main__":
    asyncio.run(main())
