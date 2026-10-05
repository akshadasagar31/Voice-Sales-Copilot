import asyncio
import os
import sys
import json
import time
import socket
import websockets
from dotenv import load_dotenv

sys.path.insert(0, "backend")
load_dotenv("backend/.env")
api_key = os.getenv("DEEPGRAM_API_KEY")

class DeepgramWSConnectionPool:
    def __init__(self):
        self._pool = {}
        self._in_use = set()
        self._lock = asyncio.Lock()
        self._keepalive_task = None

    async def start(self):
        if self._keepalive_task is None or self._keepalive_task.done():
            self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def _keepalive_loop(self):
        while True:
            try:
                await asyncio.sleep(5.0)
                async with self._lock:
                    stale_keys = []
                    for url, ws in list(self._pool.items()):
                        if ws in self._in_use:
                            continue
                        if ws.state.name == "OPEN":
                            try:
                                await ws.send(json.dumps({"type": "KeepAlive"}))
                            except Exception:
                                stale_keys.append(url)
                        else:
                            stale_keys.append(url)
                    for k in stale_keys:
                        ws = self._pool.pop(k, None)
                        if ws:
                            try:
                                await ws.close()
                            except Exception:
                                pass
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def acquire(self, url: str, headers: dict) -> websockets.ClientConnection:
        await self.start()
        async with self._lock:
            existing_ws = self._pool.get(url)
            if existing_ws is not None and existing_ws not in self._in_use:
                if existing_ws.state.name == "OPEN":
                    print(f"[Pool] REUSING healthy Deepgram connection (state={existing_ws.state.name})", flush=True)
                    self._in_use.add(existing_ws)
                    return existing_ws
                else:
                    print(f"[Pool] Discarding stale connection (state={existing_ws.state.name})", flush=True)
                    self._pool.pop(url, None)
                    try:
                        await existing_ws.close()
                    except Exception:
                        pass

        # Connect with short retry for transient 503 / getaddrinfo failed
        ws = await connect_deepgram_ws_with_retry(url, headers)
        async with self._lock:
            self._in_use.add(ws)
            self._pool[url] = ws
        return ws

    async def release(self, url: str, ws: websockets.ClientConnection, broken: bool = False):
        async with self._lock:
            self._in_use.discard(ws)
            if broken or ws.state.name != "OPEN":
                print(f"[Pool] Releasing broken/stale connection (state={ws.state.name}), discarding", flush=True)
                self._pool.pop(url, None)
                try:
                    await ws.close()
                except Exception:
                    pass
            else:
                self._pool[url] = ws
                print(f"[Pool] Released healthy connection back to pool (state={ws.state.name})", flush=True)

async def connect_deepgram_ws_with_retry(url: str, headers: dict, max_retries: int = 2) -> websockets.ClientConnection:
    delays = [0.08, 0.2]
    last_exc = None
    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        try:
            ws = await websockets.connect(
                url,
                additional_headers=headers,
                open_timeout=3.0,
                ping_interval=10,
                ping_timeout=5,
            )
            print(f"[Connect] Connected in {(time.perf_counter()-t0)*1000:.1f}ms (attempt {attempt+1})", flush=True)
            return ws
        except (
            socket.gaierror,
            OSError,
            TimeoutError,
            asyncio.TimeoutError,
            websockets.exceptions.InvalidStatusCode,
            websockets.exceptions.WebSocketException,
        ) as exc:
            last_exc = exc
            dur = (time.perf_counter() - t0) * 1000
            print(f"[Connect] Transient failure on attempt {attempt+1}/{max_retries+1} ({dur:.1f}ms): {type(exc).__name__}: {exc}", flush=True)
            if attempt < max_retries:
                delay = delays[min(attempt, len(delays) - 1)]
                await asyncio.sleep(delay)
            else:
                raise last_exc

async def simulate_turn(pool: DeepgramWSConnectionPool, turn_num: int, url: str, headers: dict, pcm: bytes):
    print(f"\n=================== SIMULATING VOICE TURN {turn_num} ===================", flush=True)
    t_start = time.perf_counter()
    
    # 1. Acquire connection from pool
    t_acq = time.perf_counter()
    dg_ws = await pool.acquire(url, headers)
    acq_ms = (time.perf_counter() - t_acq) * 1000
    print(f"Turn {turn_num}: acquire took {acq_ms:.1f}ms! State: {dg_ws.state.name}", flush=True)

    dg_broken = False
    accumulated_finals = []
    latest_interim = ""
    sent_final = False
    finalize_requested = False

    try:
        # Stream audio chunks
        chunk_size = 4096 * 2
        for i in range(0, len(pcm), chunk_size):
            await dg_ws.send(pcm[i:i+chunk_size])
            await asyncio.sleep(0.015)
            # check for incoming interim results
            try:
                msg = await asyncio.wait_for(dg_ws.recv(), timeout=0.005)
                data = json.loads(msg)
                if data.get("type") == "Results":
                    alts = (data.get("channel") or {}).get("alternatives") or []
                    if alts and alts[0].get("transcript"):
                        tr = alts[0]["transcript"]
                        if data.get("is_final"):
                            accumulated_finals.append(tr)
            except (asyncio.TimeoutError, Exception):
                pass

        # User finished speaking -> client sends Finalize
        finalize_requested = True
        # Send 500ms silence flush to trigger endpointing without closing WS
        silence_pcm = b"\x00\x00" * int(48000 * 0.5)
        for i in range(0, len(silence_pcm), chunk_size):
            await dg_ws.send(silence_pcm[i:i+chunk_size])
            await asyncio.sleep(0.01)

        # Wait for speech_final
        while not sent_final:
            try:
                msg = await asyncio.wait_for(dg_ws.recv(), timeout=2.0)
                data = json.loads(msg)
                if data.get("type") == "Results":
                    alts = (data.get("channel") or {}).get("alternatives") or []
                    if alts and alts[0].get("transcript"):
                        tr = alts[0]["transcript"]
                        if data.get("is_final"):
                            accumulated_finals.append(tr)
                    if data.get("speech_final") or data.get("is_final"):
                        sent_final = True
            except asyncio.TimeoutError:
                print("Wait timeout, finalizing with whatever accumulated")
                sent_final = True
                break

        full_transcript = " ".join(accumulated_finals).strip()
        total_turn_ms = (time.perf_counter() - t_start) * 1000
        print(f"Turn {turn_num} DONE in {total_turn_ms:.1f}ms! Transcript: '{full_transcript}'", flush=True)
        assert dg_ws.state.name == "OPEN", f"WS closed after turn {turn_num}!"

    except Exception as e:
        dg_broken = True
        print(f"Turn {turn_num} error: {e}", flush=True)
        raise
    finally:
        await pool.release(url, dg_ws, broken=dg_broken)

async def main():
    from services.stt import build_deepgram_ws_url
    url = build_deepgram_ws_url(model="nova-3", sample_rate=48000, language="en")
    headers = {"Authorization": f"Token {api_key}"}

    with open("scratch/financial_en_48k.wav", "rb") as f:
        wav_bytes = f.read()
    pcm = wav_bytes[44:] if wav_bytes.startswith(b"RIFF") else wav_bytes

    pool = DeepgramWSConnectionPool()

    print("STARTING 5 CONSECUTIVE VOICE TURNS TEST WITH WEBSOCKET REUSE...")
    for turn in range(1, 6):
        await simulate_turn(pool, turn, url, headers, pcm)
        await asyncio.sleep(0.3)

    print("\n=======================================================")
    print("ALL 5 CONSECUTIVE VOICE TURNS COMPLETED SUCCESSFULLY!")
    print("=======================================================")

if __name__ == "__main__":
    asyncio.run(main())
