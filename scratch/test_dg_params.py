import sys
import asyncio
import os
import httpx
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv("backend/.env")
api_key = os.getenv("DEEPGRAM_API_KEY")
headers = {"Authorization": f"Token {api_key}", "Content-Type": "audio/wav"}

async def test_call(name, params):
    with open("scratch/financial_mr_48k.wav", "rb") as f:
        audio = f.read()
    async with httpx.AsyncClient(timeout=25.0) as client:
        res = await client.post("https://api.deepgram.com/v1/listen", params=params, headers=headers, content=audio)
        data = res.json()
        tr = data.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")
        conf = data.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("confidence", 0.0)
        print(f"{name} -> \"{tr}\" (conf: {conf:.3f})")

async def test_call_hi(name, params):
    with open("scratch/financial_hi_48k.wav", "rb") as f:
        audio = f.read()
    async with httpx.AsyncClient(timeout=25.0) as client:
        res = await client.post("https://api.deepgram.com/v1/listen", params=params, headers=headers, content=audio)
        data = res.json()
        tr = data.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")
        conf = data.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("confidence", 0.0)
        print(f"{name} -> \"{tr}\" (conf: {conf:.3f})")

async def main():
    print("Testing Deepgram Nova-3 Marathi (mr) on financial_mr_48k.wav:")
    await test_call("1. Baseline", {"model": "nova-3", "language": "mr", "smart_format": "true", "punctuate": "true"})
    await test_call("2. keyterm (CIBIL, CIBIL score, EMI, दरमहा EMI)", {"model": "nova-3", "language": "mr", "smart_format": "true", "punctuate": "true", "keyterm": ["CIBIL", "CIBIL score", "EMI", "दरमहा EMI", "पर्सनल लोन"]})
    await test_call("3. keywords (CIBIL score:3, EMI:3)", {"model": "nova-3", "language": "mr", "smart_format": "true", "punctuate": "true", "keywords": ["CIBIL:3", "CIBIL score:3", "EMI:3", "दरमहा EMI:3"]})
    await test_call("4. keyterm Devanagari (सिबिल स्कोर, ईएमआई, दरमहा)", {"model": "nova-3", "language": "mr", "smart_format": "true", "punctuate": "true", "keyterm": ["सिबिल स्कोर", "सिबिल", "ईएमआय", "दरमहा", "कर्ज"]})

    print("\nTesting Deepgram Nova-3 Hindi (hi) on financial_hi_48k.wav:")
    await test_call_hi("1. Baseline", {"model": "nova-3", "language": "hi", "smart_format": "true", "punctuate": "true"})
    await test_call_hi("2. keyterm (CIBIL, CIBIL score, EMI, monthly EMI)", {"model": "nova-3", "language": "hi", "smart_format": "true", "punctuate": "true", "keyterm": ["CIBIL", "CIBIL score", "EMI", "monthly EMI", "पर्सनल लोन"]})
    await test_call_hi("3. keywords (CIBIL:3, monthly EMI:3)", {"model": "nova-3", "language": "hi", "smart_format": "true", "punctuate": "true", "keywords": ["CIBIL:3", "CIBIL score:3", "EMI:3", "monthly EMI:3"]})

if __name__ == "__main__":
    asyncio.run(main())
