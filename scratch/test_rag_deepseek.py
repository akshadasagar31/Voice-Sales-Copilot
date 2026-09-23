import sys, os, time, json
from dotenv import load_dotenv
load_dotenv("backend/.env")
sys.path.insert(0, "backend")

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from services.rag import RAGService

rag = RAGService()
q = "What is the maximum loan amount?"

t0 = time.perf_counter()

# Step 1: Retrieval
t_r0 = time.perf_counter()
chunks_res = rag.retriever.retrieve(q, top_k=5)
t_r1 = time.perf_counter()
print(f"1. Retrieval took: {(t_r1 - t_r0)*1000:.1f}ms (chunks: {chunks_res.get('total_results')})")

# Step 2: OpenRouter Stream
t_llm0 = time.perf_counter()
first_token_ms = None
tokens = []

for event_str in rag.stream_answer_chunks(q):
    if "event: token" in event_str:
        if first_token_ms is None:
            first_token_ms = (time.perf_counter() - t0) * 1000
        data_line = [l for l in event_str.split("\n") if l.startswith("data: ")][0]
        data = json.loads(data_line[6:])
        delta = data.get("delta") or data.get("token") or ""
        tokens.append(delta)

t_end = time.perf_counter()
print(f"2. Time to First Token from start: {first_token_ms:.1f}ms")
print(f"3. Total pipeline time: {(t_end - t0)*1000:.1f}ms")
print("Answer:", "".join(tokens))
