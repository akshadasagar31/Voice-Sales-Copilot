import asyncio
import os
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, "backend")
from dotenv import load_dotenv

load_dotenv("backend/.env")

from services.stt import DeepgramSTTService
from services.language import detect_spoken_language, get_response_language


async def main():
    stt = DeepgramSTTService()
    hi_audio = open("scratch/financial_hi_48k.wav", "rb").read()
    mr_audio = open("scratch/financial_mr_48k.wav", "rb").read()

    print("=== TEST 1: HINDI AUDIO ===")
    res_hi_none = await stt.transcribe_audio(
        hi_audio, content_type="audio/wav", model="nova-3", language=None
    )
    print("1a. language=None (auto/default):")
    print("   Transcript:", repr(res_hi_none.get("transcript")))
    print("   Detected Lang:", res_hi_none.get("detected_language"))
    print("   Spoken Lang:", detect_spoken_language(res_hi_none.get("transcript", "")))
    print("   Resp Lang:", get_response_language(detect_spoken_language(res_hi_none.get("transcript", ""))))

    res_hi_hi = await stt.transcribe_audio(
        hi_audio, content_type="audio/wav", model="nova-3", language="hi"
    )
    print("1b. language='hi':")
    print("   Transcript:", repr(res_hi_hi.get("transcript")))
    print("   Detected Lang:", res_hi_hi.get("detected_language"))
    print("   Spoken Lang:", detect_spoken_language(res_hi_hi.get("transcript", "")))
    print("   Resp Lang:", get_response_language(detect_spoken_language(res_hi_hi.get("transcript", ""))))

    print("\n=== TEST 2: MARATHI AUDIO ===")
    res_mr_none = await stt.transcribe_audio(
        mr_audio, content_type="audio/wav", model="nova-3", language=None
    )
    print("2a. language=None (auto/default):")
    print("   Transcript:", repr(res_mr_none.get("transcript")))
    print("   Detected Lang:", res_mr_none.get("detected_language"))
    print("   Spoken Lang:", detect_spoken_language(res_mr_none.get("transcript", "")))
    print("   Resp Lang:", get_response_language(detect_spoken_language(res_mr_none.get("transcript", ""))))

    res_mr_mr = await stt.transcribe_audio(
        mr_audio, content_type="audio/wav", model="nova-3", language="mr"
    )
    print("2b. language='mr':")
    print("   Transcript:", repr(res_mr_mr.get("transcript")))
    print("   Detected Lang:", res_mr_mr.get("detected_language"))
    print("   Spoken Lang:", detect_spoken_language(res_mr_mr.get("transcript", "")))
    print("   Resp Lang:", get_response_language(detect_spoken_language(res_mr_mr.get("transcript", ""))))


if __name__ == "__main__":
    asyncio.run(main())
