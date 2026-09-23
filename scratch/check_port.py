import httpx
import sys

for port in [8000, 8001]:
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/docs", timeout=2.0)
        print(f"Port {port}: status {r.status_code}")
    except Exception as e:
        print(f"Port {port}: {e}")
