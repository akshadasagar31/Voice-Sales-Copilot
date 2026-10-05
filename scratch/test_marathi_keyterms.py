import os
import sys
import asyncio
import httpx
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv("backend/.env")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

async def test_deepgram(audio_bytes, params_dict, label):
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav",
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        res = await client.post(
            "https://api.deepgram.com/v1/listen",
            params=params_dict,
            headers=headers,
            content=audio_bytes,
        )
        data = res.json()
        transcript = ""
        conf = 0.0
        try:
            alt = data["results"]["channels"][0]["alternatives"][0]
            transcript = alt.get("transcript", "")
            conf = alt.get("confidence", 0.0)
        except Exception:
            pass
        print(f"\n=== {label} ===")
        print("Transcript:", transcript)
        print("Confidence:", conf)

async def main():
    wav_path = "scratch/financial_mr_48k.wav"
    with open(wav_path, "rb") as f:
        audio_bytes = f.read()

    # Test 1: language=mr with standard params
    p1 = {
        "model": "nova-3",
        "language": "mr",
        "smart_format": "true",
        "punctuate": "true",
    }
    await test_deepgram(audio_bytes, p1, "Nova-3 language=mr baseline")

    # Test 2: language=mr with specialized Marathi keyterms
    mr_keyterms = [
        "CIBIL", "CIBIL score", "सिबिल", "सिबिल स्कोर",
        "EMI", "दरमहा EMI", "ईएमआय", "दरमहा",
        "कर्ज", "पर्सनल लोन", "personal loan", "business loan",
        "10 लाख", "लाख", "रुपये", "रुपयांचे", "साठी", "किती", "आहे"
    ]
    p2 = {
        "model": "nova-3",
        "language": "mr",
        "smart_format": "true",
        "punctuate": "true",
        "keyterm": mr_keyterms,
    }
    await test_deepgram(audio_bytes, p2, "Nova-3 language=mr + specialized keyterms")

    # Test 3: language=multi with specialized keyterms
    p3 = {
        "model": "nova-3",
        "language": "multi",
        "smart_format": "true",
        "punctuate": "true",
        "keyterm": mr_keyterms,
    }
    await test_deepgram(audio_bytes, p3, "Nova-3 language=multi + specialized keyterms")

if __name__ == "__main__":
    asyncio.run(main())
