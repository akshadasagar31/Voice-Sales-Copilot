import os
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "./backend")
import httpx
from dotenv import load_dotenv
from services.rag import RAGService
from services.retriever import VectorRetriever

load_dotenv("backend/.env")

# Simulate a situation where primary model is unavailable or rate limited (HTTP 429)
rag = RAGService(
    model="deepseek/deepseek-unavailable-test",
    fallback_model="meta-llama/llama-3.3-70b-instruct",
)

print(f"Testing RAGService fallback streaming...")
print(f"  Configured primary:  {rag.model}")
print(f"  Configured fallback: {rag.fallback_model}")

stream_gen = rag.stream_answer_chunks(
    question="What is the minimum CIBIL score for HDFC loans?",
    top_k=4
)

received_tokens = []
received_events = []

for sse_chunk in stream_gen:
    lines = sse_chunk.strip().split("\n")
    event = ""
    data = ""
    for line in lines:
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data = line[5:].strip()
    received_events.append((event, data))
    if event == "token":
        import json
        try:
            d = json.loads(data)
            received_tokens.append(d.get("delta") or d.get("token") or "")
        except:
            pass

print(f"\nTotal SSE events received: {len(received_events)}")
print(f"Streamed tokens count: {len(received_tokens)}")
print(f"Assembled answer: {''.join(received_tokens)}")
assert len(received_tokens) > 0, "Fallback failed to produce tokens!"
print("\n[SUCCESS] RAGService successfully failed over to meta-llama/llama-3.3-70b-instruct and streamed answer!")
