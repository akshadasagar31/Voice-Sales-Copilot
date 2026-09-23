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

# Fetch provider info for deepseek and llama
models_to_test = [
    "deepseek/deepseek-chat",
    "meta-llama/llama-3.3-70b-instruct",
]

context_4chunks = """--- Document: HDFC_Credit_Policy.pdf | Page: 4 [Segment 1] ---
For Personal Loans, the minimum applicant CIBIL score required is 750. Applicants with CIBIL between 650 and 749 require credit committee approval and minimum monthly salary of Rs 40,000.

--- Document: HDFC_Credit_Policy.pdf | Page: 5 [Segment 2] ---
Interest rates for Personal Loans range from 10.50% to 14.00% depending on CIBIL score and employer category (Tier 1 vs Tier 2).

--- Document: HDFC_Credit_Policy.pdf | Page: 8 [Segment 3] ---
Maximum loan amount is Rs 40 Lakhs with repayment tenure between 12 months and 60 months. Foreclosure is allowed after 12 EMIs.

--- Document: HDFC_Credit_Policy.pdf | Page: 12 [Segment 4] ---
Self-employed applicants require 3 years of audited financials and minimum annual turnover of Rs 50 Lakhs."""

context_2chunks = """--- Document: HDFC_Credit_Policy.pdf | Page: 4 [Segment 1] ---
For Personal Loans, the minimum applicant CIBIL score required is 750. Applicants with CIBIL between 650 and 749 require credit committee approval and minimum monthly salary of Rs 40,000.

--- Document: HDFC_Credit_Policy.pdf | Page: 5 [Segment 2] ---
Interest rates for Personal Loans range from 10.50% to 14.00% depending on CIBIL score and employer category."""

system_prompt = """You are a helpful, accurate sales assistant for Voice Sales Copilot.
Answer the user's question STRICTLY and ONLY using the provided Context in English.
CRITICAL RULES:
1. Base your answer EXCLUSIVELY on the provided Context.
2. DO NOT include any greeting or conversational preamble. Start immediately with the direct factual answer.
3. If the answer is not present, respond with: "I am sorry, but the provided documentation does not contain sufficient information to answer this question."
4. Provide a concise, direct, and professional answer in 1 to 2 clear sentences."""

question = "What is the minimum CIBIL score for personal loans?"

print("=================================================================")
print("BENCHMARKING OPENROUTER MODELS & CONTEXT SIZES FOR TTFT")
print("=================================================================")

for model in models_to_test:
    for ctx_name, ctx in [("4 Chunks (Heavy Context)", context_4chunks), ("2 Chunks (Trimmed Context)", context_2chunks)]:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion:\n{question}"}
            ],
            "provider": {"sort": "latency"},
            "temperature": 0.0,
            "max_tokens": 120,
            "stream": True,
        }
        
        # Run 2 iterations to get warm timing
        ttft_runs = []
        full_ans = ""
        for run in range(2):
            t0 = time.perf_counter()
            first_tok = None
            ans = ""
            try:
                with client.stream("POST", "https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers) as resp:
                    if resp.status_code != 200:
                        print(f"Error {resp.status_code} for {model}: {resp.read().decode('utf-8')[:100]}")
                        break
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
                                    if first_tok is None:
                                        first_tok = time.perf_counter() - t0
                                    ans += d
                            except: pass
                if first_tok is not None:
                    ttft_runs.append(first_tok * 1000)
                full_ans = ans
            except Exception as e:
                print(f"Exception on {model}: {e}")
        
        avg_ttft = sum(ttft_runs) / len(ttft_runs) if ttft_runs else 0
        print(f"\nModel: {model} | {ctx_name}")
        print(f"  TTFT Runs: {[round(r, 1) for r in ttft_runs]} ms | Avg TTFT: {avg_ttft:.1f} ms")
        print(f"  Answer: '{full_ans.strip()[:90]}...'")
