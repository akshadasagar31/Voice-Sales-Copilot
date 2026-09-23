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

standard_prompt = """You are a helpful, accurate, and truthful sales assistant for Voice Sales Copilot.
Your task is to answer the user's question STRICTLY and ONLY using the provided Context below in English.

CRITICAL RULES:
1. Base your answer EXCLUSIVELY on the provided Context. Do NOT use outside knowledge, assumptions, or hallucinations.
2. DO NOT include any greeting (such as "Hello", "Hi", "नमस्ते", "नमस्कार", "Welcome", "Thank you for asking", etc.) or conversational preamble. Start immediately with the direct factual answer so speech synthesis can begin right away.
3. If the answer is not present in the provided context, or if the context does not contain sufficient information to answer the question, you MUST respond EXACTLY with this phrase and nothing else:
"I am sorry, but the provided documentation does not contain sufficient information to answer this question."
4. Do NOT provide apologies, caveats, or speculation beyond the exact fallback phrase if the answer is missing.
5. If the context answers the question, provide a concise, direct, and professional answer in English in 1 to 3 clear sentences."""

compact_voice_prompt = """You are a sales copilot answering by voice in English.
Answer STRICTLY from the Context below in 1-2 direct spoken sentences.
RULES:
1. Base your answer ONLY on the provided Context without outside knowledge.
2. Start IMMEDIATELY with the answer. No greetings or preamble.
3. If the answer is not in the context, output EXACTLY:
"I am sorry, but the provided documentation does not contain sufficient information to answer this question."
4. Keep the answer concise for immediate voice playback."""

context_verbose = """--- Document: HDFC_Credit_Policy.pdf | Page: 4 [Segment 1] ---
For Personal Loans, the minimum applicant CIBIL score required is 750. Applicants with CIBIL between 650 and 749 require credit committee approval and minimum monthly salary of Rs 40,000.

--- Document: HDFC_Credit_Policy.pdf | Page: 5 [Segment 2] ---
Interest rates for Personal Loans range from 10.50% to 14.00% depending on CIBIL score and employer category."""

context_compact = """[HDFC_Credit_Policy.pdf, p.4]: For Personal Loans, the minimum applicant CIBIL score required is 750. Applicants with CIBIL between 650 and 749 require credit committee approval and minimum monthly salary of Rs 40,000.
[HDFC_Credit_Policy.pdf, p.5]: Interest rates for Personal Loans range from 10.50% to 14.00% depending on CIBIL score and employer category."""

question = "What is the minimum CIBIL score for personal loans?"

model = "meta-llama/llama-3.3-70b-instruct"

# Test 1: Standard
p1 = {
    "model": model,
    "messages": [
        {"role": "system", "content": standard_prompt},
        {"role": "user", "content": f"Context:\n{context_verbose}\n\nQuestion:\n{question}"}
    ],
    "provider": {"sort": "latency"},
    "temperature": 0.0,
    "max_tokens": 120,
    "stream": True,
}

t0 = time.perf_counter()
tok1 = None
ans1 = ""
with client.stream("POST", "https://openrouter.ai/api/v1/chat/completions", json=p1, headers=headers) as resp:
    for line in resp.iter_lines():
        line = line.strip()
        if line.startswith("data:") and line[5:].strip() != "[DONE]":
            try:
                d = json.loads(line[5:].strip()).get("choices", [{}])[0].get("delta", {}).get("content")
                if d:
                    if tok1 is None: tok1 = time.perf_counter() - t0
                    ans1 += d
            except: pass

# Test 2: Compact Voice
p2 = {
    "model": model,
    "messages": [
        {"role": "system", "content": compact_voice_prompt},
        {"role": "user", "content": f"Context:\n{context_compact}\n\nQuestion:\n{question}"}
    ],
    "provider": {"sort": "latency"},
    "temperature": 0.0,
    "max_tokens": 120,
    "stream": True,
}

t0 = time.perf_counter()
tok2 = None
ans2 = ""
with client.stream("POST", "https://openrouter.ai/api/v1/chat/completions", json=p2, headers=headers) as resp:
    for line in resp.iter_lines():
        line = line.strip()
        if line.startswith("data:") and line[5:].strip() != "[DONE]":
            try:
                d = json.loads(line[5:].strip()).get("choices", [{}])[0].get("delta", {}).get("content")
                if d:
                    if tok2 is None: tok2 = time.perf_counter() - t0
                    ans2 += d
            except: pass

print(f"Standard Prompt TTFT: {tok1*1000:.1f}ms | Ans: '{ans1[:60]}...'")
print(f"Compact Voice TTFT:   {tok2*1000:.1f}ms | Ans: '{ans2[:60]}...'")
print(f"Latency Difference:   {(tok1 - tok2)*1000:.1f}ms")
