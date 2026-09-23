import os
import sys
import time
import json
import httpx
from dotenv import load_dotenv

load_dotenv('backend/.env')
sys.path.insert(0, 'backend')
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

api_key = os.getenv('OPENROUTER_API_KEY')
client = httpx.Client(timeout=30.0)
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json',
    'HTTP-Referer': 'https://github.com/Voice-Sales-Copilot',
}

payload = {
    "model": "deepseek/deepseek-chat",
    "models": ["deepseek/deepseek-chat", "meta-llama/llama-3.3-70b-instruct"],
    "messages": [
        {"role": "system", "content": "You are a sales assistant. Answer in 1 short sentence."},
        {"role": "user", "content": "What is the minimum CIBIL score for personal loans?"}
    ],
    "provider": {"sort": "latency"},
    "temperature": 0.0,
    "max_tokens": 100,
    "stream": True,
}

t0 = time.perf_counter()
first_token = None
tokens = []
status_code = None

with client.stream("POST", "https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers) as resp:
    status_code = resp.status_code
    print("Response status:", status_code)
    for line in resp.iter_lines():
        line = line.strip()
        if not line or line.startswith(":"): continue
        if line.startswith("data:"):
            p = line[5:].strip()
            if p == "[DONE]": break
            try:
                j = json.loads(p)
                d = j.get("choices", [{}])[0].get("delta", {}).get("content")
                if d:
                    if first_token is None:
                        first_token = time.perf_counter() - t0
                    tokens.append(d)
            except: pass

print(f"Status: {status_code} | TTFT: {first_token*1000 if first_token else 'N/A':.1f}ms | Ans: {''.join(tokens)}")
