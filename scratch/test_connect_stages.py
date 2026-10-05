import asyncio
import time
import socket
import ssl
import websockets
from dotenv import load_dotenv

load_dotenv("backend/.env")

async def test_stages():
    loop = asyncio.get_running_loop()
    
    # Stage 1: default loop.getaddrinfo
    t0 = time.perf_counter()
    addrs = await loop.getaddrinfo("api.deepgram.com", 443)
    t1 = time.perf_counter()
    print(f"1. loop.getaddrinfo('api.deepgram.com'): {(t1-t0)*1000:.1f}ms -> {len(addrs)} addrs: {[a[4] for a in addrs]}")

    # Stage 2: IPv4 only getaddrinfo
    t0 = time.perf_counter()
    addrs_v4 = await loop.getaddrinfo("api.deepgram.com", 443, family=socket.AF_INET)
    t1 = time.perf_counter()
    print(f"2. IPv4 getaddrinfo: {(t1-t0)*1000:.1f}ms -> {[a[4] for a in addrs_v4]}")

    # Stage 3: TCP connect to IP vs Host
    ip = addrs_v4[0][4][0]
    ssl_ctx = ssl.create_default_context()
    
    # TCP + SSL via hostname
    t0 = time.perf_counter()
    r, w = await asyncio.open_connection("api.deepgram.com", 443, ssl=ssl_ctx)
    t1 = time.perf_counter()
    print(f"3. open_connection('api.deepgram.com') with SSL: {(t1-t0)*1000:.1f}ms")
    w.close()
    await w.wait_closed()

    # TCP + SSL via IP with server_hostname
    t0 = time.perf_counter()
    r, w = await asyncio.open_connection(ip, 443, ssl=ssl_ctx, server_hostname="api.deepgram.com")
    t1 = time.perf_counter()
    print(f"4. open_connection(ip='{ip}', server_hostname='api.deepgram.com'): {(t1-t0)*1000:.1f}ms")
    w.close()
    await w.wait_closed()

asyncio.run(test_stages())
