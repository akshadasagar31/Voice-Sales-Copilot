import sys
from pathlib import Path
from dotenv import load_dotenv
import asyncio
import os
import json
import websockets

backend_dir = Path(r"c:\Users\Akshada\OneDrive\Pictures\Documents\projects\Voice-Sales-Copilot\backend")
sys.path.insert(0, str(backend_dir))
load_dotenv(backend_dir / ".env")

async def test_ws():
    url = "wss://api.deepgram.com/v1/listen?model=nova-3&language=mr"
    headers = {"Authorization": f"Token {os.getenv('DEEPGRAM_API_KEY')}"}
    async with websockets.connect(url, additional_headers=headers) as ws:
        await ws.send(b"\x00" * 4096)
        await asyncio.sleep(0.5)
        await ws.send(json.dumps({"type": "CloseStream"}))
        async for m in ws:
            d = json.loads(m)
            print("Received WS msg type:", d.get("type"), "keys:", list(d.keys()))

if __name__ == "__main__":
    asyncio.run(test_ws())
