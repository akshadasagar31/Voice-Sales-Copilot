import sys, os
from dotenv import load_dotenv
load_dotenv("backend/.env")
sys.path.insert(0, "backend")

from services.vector_store import PineconeService

p = PineconeService()
queries = [
    "What documents are required for a personal loan?",
    "What is the maximum loan amount?",
    "What is the interest rate for Cat A companies?",
    "What is the processing fee?",
]

for q in queries:
    matches = p.search_records(q, top_k=20, namespace="sales_playbooks")
    seen_texts = set()
    unique = []
    for m in matches:
        t = m.get("text", "").strip()[:60]
        if t not in seen_texts:
            seen_texts.add(t)
            unique.append(m)
    print(f"\nQuery: '{q}' -> Total raw: {len(matches)}, Unique: {len(unique)}")
    for u in unique[:3]:
        print(f"  [p.{u.get('page')}, c.{u.get('chunk_index')}]: {u.get('text', '')[:100]}...")
