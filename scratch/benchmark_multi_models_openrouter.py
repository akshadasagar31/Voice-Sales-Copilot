import os
import sys
import time
import json
import httpx
from dotenv import load_dotenv

load_dotenv('backend/.env')
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

api_key = os.getenv('OPENROUTER_API_KEY')
client = httpx.Client(timeout=30.0)
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json',
    'HTTP-Referer': 'https://github.com/Voice-Sales-Copilot',
}

models = [
    "deepseek/deepseek-chat",
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemini-2.0-flash-001",
    "qwen/qwen-2.5-72b-instruct",
]

context = "[HDFC_Credit_Policy.pdf, p.4]: For Personal Loans, the minimum applicant CIBIL score required is 750. Applicants with CIBIL between 650 and 749 require credit committee approval."
prompt = "You are a sales assistant. Answer directly in 1 sentence using the context."
question = "What is the minimum CIBIL score for personal loans?"

print("=========================================================")
print("BENCHMARKING CANDIDATE MODELS ON OPENROUTER FOR TTFT")
print("=========================================================")

for m in models:
    payload = {
        "model": m,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
        ],
        "provider": {"sort": "latency"},
        "temperature": 0.0,
        "max_tokens": 80,
        "stream": True,
    }
    t0 = time.perf_counter()
    first_tok = None
    ans = ""
    err = None
    try:
        with client.stream("POST", "https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                err = f"HTTP {resp.status_code}: {resp.read().decode('utf-8')[:80]}"
            else:
                for line in resp.iter_lines():
                    line = line.strip()
                    if line.startswith("data:") and line[5:].strip() != "[DONE]":
                        try:
                            d = json.loads(line[5:].strip()).get("choices", [{}])[0].get("delta", {}).get("content")
                            if d:
                                if first_tok is None: first_tok = time.perf_counter() - t0
                                ans += d
                        except: pass
    except Exception as e:
        err = str(e)
    
    if err:
        print(f"Model: {m:<35} | Error: {err}")
    else:
        ttft_ms = first_tok * 1000 if first_tok else 0
        print(f"Model: {m:<35} | TTFT: {ttft_ms:6.1f} ms | Ans: {ans.strip()[:60]}...")
