import asyncio
import json
import time
import websockets

async def test_live_voice_turns():
    url = "ws://127.0.0.1:8001/ws/voice-stt?sample_rate=48000&language=en&module=module1"

    print("\n=======================================================")
    print("TESTING LIVE /ws/voice-stt CONSECUTIVE TURNS (PORT 8001)")
    print("=======================================================")

    for turn in range(1, 4):
        print(f"\n--- Turn {turn}/3 ---")
        t0 = time.perf_counter()
        async with websockets.connect(url) as ws:
            t_connect = (time.perf_counter() - t0) * 1000
            print(f"  Connected in {t_connect:.1f}ms")

            # Stream brief audio
            await ws.send(b"\x00\x00" * 4800)

            # Send Finalize
            t_fin = time.perf_counter()
            await ws.send(json.dumps({"type": "Finalize"}))

            # Receive final response
            msg = None
            while True:
                msg_str = await asyncio.wait_for(ws.recv(), timeout=5.0)
                m = json.loads(msg_str)
                if m.get("type") == "final":
                    msg = m
                    break
            t_recv = (time.perf_counter() - t_fin) * 1000
            print(f"  Turn {turn} Finalize -> Response: {t_recv:.1f}ms | Type: {msg.get('type')}")
            assert msg.get("type") == "final"
            if turn > 1:
                print(f"  Turn {turn} reused pooled connection! Latency: {t_recv:.1f}ms")
                assert t_recv < 750, f"Turn {turn} took too long: {t_recv:.1f}ms"

    print("\n=======================================================")
    print("ALL LIVE CONSECUTIVE TURNS PASSED!")
    print("=======================================================\n")

if __name__ == "__main__":
    asyncio.run(test_live_voice_turns())
