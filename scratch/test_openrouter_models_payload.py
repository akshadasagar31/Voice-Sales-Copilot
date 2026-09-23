import os
import requests
import json
import time
from dotenv import load_dotenv

load_dotenv("backend/.env")
api_key = os.getenv("OPENROUTER_API_KEY")

primary_model = "deepseek/deepseek-chat"
fallback_model = "meta-llama/llama-3.3-70b-instruct"

print(f"Testing OpenRouter fallback payload with:")
print(f"  Primary:  {primary_model}")
print(f"  Fallback: {fallback_model}")

payload = {
    "model": primary_model,
    "models": [primary_model, fallback_model],
    "messages": [
        {"role": "user", "content": "Briefly state what 1 + 1 is."}
    ],
    "stream": True,
}

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

t0 = time.perf_counter()
res = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    json=payload,
    headers=headers,
    stream=True,
    timeout=15
)
print("Status Code:", res.status_code)
tokens = []
for line in res.iter_lines():
    if not line:
        continue
    line_str = line.decode("utf-8", errors="replace")
    if line_str.startswith("data:"):
        data_str = line_str[5:].strip()
        if data_str == "[DONE]":
            break
        try:
            d = json.loads(data_str)
            choices = d.get("choices", [])
            if choices:
                content = choices[0].get("delta", {}).get("content", "")
                if content:
                    tokens.append(content)
        except Exception:
            pass

print(f"Streamed in {(time.perf_counter() - t0)*1000:.1f}ms: {''.join(tokens).strip()}")
