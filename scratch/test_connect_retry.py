import asyncio
import os
import time
import socket
import websockets
from dotenv import load_dotenv

load_dotenv("backend/.env")
api_key = os.getenv("DEEPGRAM_API_KEY")

_dns_cache = {"ip": None, "timestamp": 0}

def get_deepgram_ip() -> str:
    now = time.time()
    if _dns_cache["ip"] and (now - _dns_cache["timestamp"] < 300):
        return _dns_cache["ip"]
    try:
        # Resolve IPv4 directly with fast lookup
        info = socket.getaddrinfo("api.deepgram.com", 443, socket.AF_INET, socket.SOCK_STREAM)
        if info:
            ip = info[0][4][0]
            _dns_cache["ip"] = ip
            _dns_cache["timestamp"] = now
            return ip
    except Exception as e:
        print(f"DNS lookup failed: {e}, using cached: {_dns_cache['ip']}")
        if _dns_cache["ip"]:
            return _dns_cache["ip"]
    return "api.deepgram.com"

async def connect_with_retry(url: str, headers: dict, max_retries: int = 2):
    last_err = None
    delays = [0.1, 0.25]
    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        try:
            # Short open_timeout so we never hang 10s
            ws = await websockets.connect(
                url,
                additional_headers=headers,
                open_timeout=3.0,
                ping_interval=10,
                ping_timeout=5,
            )
            dur = (time.perf_counter() - t0) * 1000
            print(f"Attempt {attempt + 1}: CONNECTED in {dur:.1f}ms! State: {ws.state.name}", flush=True)
            return ws
        except (
            socket.gaierror,
            OSError,
            TimeoutError,
            asyncio.TimeoutError,
            websockets.exceptions.InvalidStatusCode,
            websockets.exceptions.WebSocketException,
        ) as e:
            dur = (time.perf_counter() - t0) * 1000
            last_err = e
            is_503 = getattr(e, "status_code", None) == 503 or "503" in str(e)
            is_dns = isinstance(e, socket.gaierror) or "getaddrinfo" in str(e)
            print(f"Attempt {attempt + 1}: FAILED in {dur:.1f}ms ({type(e).__name__}, 503={is_503}, dns={is_dns}): {e}", flush=True)
            if attempt < max_retries:
                delay = delays[min(attempt, len(delays) - 1)]
                print(f"  Retrying in {delay*1000:.0f}ms...", flush=True)
                await asyncio.sleep(delay)
            else:
                raise last_err

async def run_test():
    from services.stt import build_deepgram_ws_url
    url = build_deepgram_ws_url(model="nova-3", sample_rate=48000, language="en")
    headers = {"Authorization": f"Token {api_key}"}

    print("Testing 5 consecutive connections with connect_with_retry:")
    for i in range(5):
        print(f"\n--- Connect {i+1} ---")
        ws = await connect_with_retry(url, headers)
        await ws.close()

asyncio.run(run_test())
