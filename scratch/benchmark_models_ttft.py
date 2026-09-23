import os
import time
import httpx
import json
from dotenv import load_dotenv

load_dotenv('backend/.env')
api_key = os.getenv('OPENROUTER_API_KEY')
client = httpx.Client(timeout=30.0)
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json',
    'HTTP-Referer': 'https://github.com/Voice-Sales-Copilot',
}

for test_model in ["meta-llama/llama-3.3-70b-instruct", "deepseek/deepseek-chat"]:
    payload = {
        'model': test_model,
        'messages': [
            {'role': 'system', 'content': 'You are a sales assistant. Answer directly in 1-2 sentences.'},
            {'role': 'user', 'content': 'What is the minimum CIBIL score for personal loans?'}
        ],
        'provider': {'sort': 'latency'},
        'temperature': 0.0,
        'max_tokens': 100,
        'stream': True
    }
    t0 = time.perf_counter()
    first_token_time = None
    accumulated = ""
    with client.stream('POST', 'https://openrouter.ai/api/v1/chat/completions', json=payload, headers=headers) as resp:
        for line in resp.iter_lines():
            line = line.strip()
            if not line or line.startswith(':'): continue
            if line.startswith('data:'):
                p = line[5:].strip()
                if p == '[DONE]': break
                try:
                    j = json.loads(p)
                    delta = j.get('choices', [{}])[0].get('delta', {}).get('content')
                    if delta:
                        if first_token_time is None:
                            first_token_time = time.perf_counter() - t0
                        accumulated += delta
                except: pass
    print(f"Model: {test_model} | TTFT: {first_token_time*1000:.1f}ms | Ans: {accumulated[:60]}...")
