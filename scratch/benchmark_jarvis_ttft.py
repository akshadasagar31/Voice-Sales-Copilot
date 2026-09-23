import os
import time
import httpx
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv('backend/.env')
api_key = os.getenv('OPENROUTER_API_KEY')
model = os.getenv('OPENROUTER_MODEL', 'deepseek/deepseek-chat')
fb_model = os.getenv('RAG_FALLBACK_MODEL', 'meta-llama/llama-3.3-70b-instruct')

client = httpx.Client(timeout=30.0)
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json',
    'HTTP-Referer': 'https://github.com/Voice-Sales-Copilot',
}
payload = {
    'model': model,
    'models': [model, fb_model],
    'messages': [
        {'role': 'system', 'content': 'You are a sales assistant. Answer directly in 1-2 sentences.'},
        {'role': 'user', 'content': 'What is the minimum CIBIL score for personal loans?'}
    ],
    'provider': {'sort': 'latency'},
    'temperature': 0.0,
    'max_tokens': 150,
    'stream': True
}

t0 = time.perf_counter()
first_token_time = None
first_clause_time = None
tokens = []
accumulated = ""

with client.stream('POST', 'https://openrouter.ai/api/v1/chat/completions', json=payload, headers=headers) as resp:
    print('HTTP status:', resp.status_code)
    for line in resp.iter_lines():
        line = line.strip()
        if not line or line.startswith(':'):
            continue
        if line.startswith('data:'):
            p = line[5:].strip()
            if p == '[DONE]':
                break
            try:
                j = json.loads(p)
                delta = j.get('choices', [{}])[0].get('delta', {}).get('content')
                if delta:
                    if first_token_time is None:
                        first_token_time = time.perf_counter() - t0
                    tokens.append(delta)
                    accumulated += delta
                    if first_clause_time is None and (',' in accumulated or len(accumulated) >= 30):
                        first_clause_time = time.perf_counter() - t0
            except:
                pass

print(f'TTFT (Time to first token): {first_token_time*1000:.1f}ms')
if first_clause_time:
    print(f'Time to first clause/phrase: {first_clause_time*1000:.1f}ms')
print(f'Total tokens: {len(tokens)}')
print(f'Full answer: {accumulated}')
