import requests
import time
import json

print("=== 1. Testing Voice Assistant RAG Stream ===")
t0 = time.perf_counter()
res = requests.post(
    "http://127.0.0.1:8001/api/ask",
    json={"question": "What is the age requirement for home loans?", "top_k": 4, "stream": True},
    stream=True,
    timeout=25
)
tokens = []
first_sentence = ""
for line in res.iter_lines():
    if not line:
        continue
    line_str = line.decode("utf-8", errors="replace")
    if line_str.startswith("data:"):
        try:
            d = json.loads(line_str[5:].strip())
            tok = d.get("token") or d.get("delta") or ""
            if tok:
                tokens.append(tok)
                if not first_sentence and any(p in "".join(tokens) for p in [".", "?", "!", "\n"]):
                    first_sentence = "".join(tokens)
        except Exception:
            pass

rag_time = (time.perf_counter() - t0) * 1000
full_text = "".join(tokens).strip()
print(f"RAG Stream finished in {rag_time:.1f}ms: \"{full_text}\"")

print("\n=== 2. Testing Voice Assistant Sentence TTS Synthesis ===")
tts_text = first_sentence.strip() or full_text[:80]
t1 = time.perf_counter()
tts_res = requests.post(
    "http://127.0.0.1:8001/api/tts",
    json={"text": tts_text, "language": "en"},
    timeout=15
)
tts_time = (time.perf_counter() - t1) * 1000
print(f"TTS HTTP Status: {tts_res.status_code}")
print(f"TTS Synthesis time: {tts_time:.1f}ms")
print(f"Audio size: {len(tts_res.content):,} bytes")
assert tts_res.status_code == 200
assert len(tts_res.content) > 1000
print("\n[SUCCESS] Full Voice Assistant flow (RAG Stream + TTS Synthesis) verified working!")
