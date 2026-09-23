import os
import sys
import asyncio
import json
import websockets
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv("backend/.env")
api_key = os.getenv("DEEPGRAM_API_KEY")

async def test_lang(param):
    url = f"wss://api.deepgram.com/v1/listen?model=nova-3&encoding=linear16&sample_rate=16000&channels=1&{param}"
    headers = {"Authorization": f"Token {api_key}"}
    try:
        async with websockets.connect(url, additional_headers=headers) as ws:
            await ws.send(json.dumps({"type": "CloseStream"}))
            metadata = None
            async for msg in ws:
                data = json.loads(msg)
                if data.get("type") == "Metadata":
                    metadata = data
                    break
            model_info = metadata.get("models") if metadata else None
            print(f"[✓] {param:30} -> SUCCESS (models: {model_info})")
    except Exception as e:
        print(f"[X] {param:30} -> FAILED: {e}")

async def main():
    params = [
        "language=multi",
        "language=en",
        "language=hi",
        "language=mr",
        "detect_language=true",
        "model=nova-2&detect_language=true",
    ]
    for p in params:
        await test_lang(p)

if __name__ == "__main__":
    asyncio.run(main())
