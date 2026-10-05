import asyncio
import socket
import ssl
import time

def measure_dns():
    t0 = time.perf_counter()
    try:
        # Standard getaddrinfo as used by asyncio / websockets
        res = socket.getaddrinfo("api.deepgram.com", 443, socket.AF_UNSPEC, socket.SOCK_STREAM)
        t_dns = (time.perf_counter() - t0) * 1000
        print(f"socket.getaddrinfo took {t_dns:.1f}ms: {[r[4] for r in res]}")
    except Exception as e:
        t_dns = (time.perf_counter() - t0) * 1000
        print(f"socket.getaddrinfo FAILED after {t_dns:.1f}ms: {e}")

def measure_dns_ipv4():
    t0 = time.perf_counter()
    try:
        # IPv4 only
        res = socket.getaddrinfo("api.deepgram.com", 443, socket.AF_INET, socket.SOCK_STREAM)
        t_dns = (time.perf_counter() - t0) * 1000
        print(f"IPv4 getaddrinfo took {t_dns:.1f}ms: {[r[4] for r in res]}")
    except Exception as e:
        t_dns = (time.perf_counter() - t0) * 1000
        print(f"IPv4 getaddrinfo FAILED after {t_dns:.1f}ms: {e}")

def measure_dns_gethostbyname():
    t0 = time.perf_counter()
    try:
        ip = socket.gethostbyname("api.deepgram.com")
        t_dns = (time.perf_counter() - t0) * 1000
        print(f"socket.gethostbyname took {t_dns:.1f}ms: {ip}")
    except Exception as e:
        t_dns = (time.perf_counter() - t0) * 1000
        print(f"socket.gethostbyname FAILED after {t_dns:.1f}ms: {e}")

for i in range(3):
    print(f"\n--- Test {i+1} ---")
    measure_dns()
    measure_dns_ipv4()
    measure_dns_gethostbyname()
