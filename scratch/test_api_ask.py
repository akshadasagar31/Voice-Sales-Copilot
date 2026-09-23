import httpx, json, time, sys

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

questions = [
    "What is the maximum loan amount?",
    "What are the CIBIL score requirements for personal loan?",
    "What documents are required for personal loan?",
]

with httpx.Client(timeout=30.0) as client:
    for q in questions:
        print("\n" + "=" * 60)
        print("Question:", q)
        t0 = time.perf_counter()
        first_token_ms = None
        accumulated = ""
        with client.stream(
            "POST",
            "http://127.0.0.1:8001/api/ask",
            json={
                "question": q,
                "top_k": 5,
                "namespace": "sales_playbooks",
                "stream": True,
            },
        ) as resp:
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    d_str = line[6:].strip()
                    if not d_str:
                        continue
                    try:
                        d = json.loads(d_str)
                        tok = d.get("token") or d.get("delta") or ""
                        if tok:
                            if first_token_ms is None:
                                first_token_ms = (time.perf_counter() - t0) * 1000
                            accumulated += tok
                    except:
                        pass
        total_ms = (time.perf_counter() - t0) * 1000
        print(f"TTFT: {first_token_ms:.1f}ms | Total: {total_ms:.1f}ms")
        print("Answer:", accumulated.strip())
