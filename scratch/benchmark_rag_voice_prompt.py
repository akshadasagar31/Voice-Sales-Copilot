import os
import time
import httpx
import json
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

prompt = """You are a fast, concise sales assistant for Voice Sales Copilot.
Answer the user's question directly, accurately, and truthfully in 1 to 2 clear spoken sentences in English, strictly grounded in the Context below.
CRITICAL RULES:
1. Base your answer EXCLUSIVELY on the provided Context. Do NOT hallucinate.
2. DO NOT include any greeting or preamble. Start IMMEDIATELY with the factual answer.
3. If the answer is missing, respond with: "I am sorry, but the provided documentation does not contain sufficient information to answer this question."
4. Keep the answer concise (1 to 2 sentences) for immediate voice speech synthesis."""

context = """--- Document: HDFC_Credit_Policy.pdf | Page: 4 [Segment 1] ---
For Personal Loans, the minimum applicant CIBIL score required is 750. Applicants with CIBIL between 650 and 749 require credit committee approval and minimum monthly salary of Rs 40,000."""

payload = {
    'model': model,
    'models': [model, fb_model],
    'messages': [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': f"Context:\n{context}\n\nQuestion:\nWhat is the minimum CIBIL score for personal loans?"}
    ],
    'provider': {'sort': 'latency'},
    'temperature': 0.0,
    'max_tokens': 150,
    'stream': True
}

t0 = time.perf_counter()
first_token_t = None
first_clause_t = None
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
                    if first_token_t is None:
                        first_token_t = time.perf_counter() - t0
                    accumulated += delta
                    if first_clause_t is None and (',' in accumulated or len(accumulated) >= 28):
                        first_clause_t = time.perf_counter() - t0
            except: pass

print(f"TTFT: {first_token_t*1000:.1f}ms")
if first_clause_t:
    print(f"First Clause Time: {first_clause_t*1000:.1f}ms")
print("Answer:", accumulated)
