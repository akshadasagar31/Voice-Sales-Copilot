import requests
import json
import time

query = "What is the minimum CIBIL score required for HDFC home loans?"
print(f"Sending live streaming query: {query}")
t0 = time.perf_counter()
res = requests.post(
    "http://127.0.0.1:8001/api/ask",
    json={"question": query, "top_k": 4, "stream": True},
    stream=True,
    timeout=25
)
print("Response Status:", res.status_code)
tokens = []
metadata = None
done = None
evt = ""

for line in res.iter_lines():
    if not line:
        continue
    line_str = line.decode("utf-8", errors="replace")
    if line_str.startswith("event:"):
        evt = line_str[6:].strip()
    elif line_str.startswith("data:"):
        data_str = line_str[5:].strip()
        try:
            parsed = json.loads(data_str)
            if evt == "metadata":
                metadata = parsed
                print(f"Metadata event: model={parsed.get('model')}, sources={parsed.get('sources')}")
            elif evt == "token":
                t = parsed.get("token") or parsed.get("delta") or ""
                tokens.append(t)
            elif evt == "done":
                done = parsed
                print(f"Done event: answer=\"{parsed.get('answer')}\"")
        except Exception as e:
            pass

total_ms = (time.perf_counter() - t0) * 1000
print(f"Stream completed in {total_ms:.1f}ms with {len(tokens)} tokens.")
print("Assembled Answer:", "".join(tokens))
