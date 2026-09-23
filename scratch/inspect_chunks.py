import sys, os
from dotenv import load_dotenv
load_dotenv("backend/.env")
sys.path.insert(0, "backend")

from services.vector_store import PineconeService

p = PineconeService()
matches = p.search_records("What is the minimum CIBIL score required for personal loan?", top_k=25, namespace="sales_playbooks")
print(f"Total raw matches: {len(matches)}")
seen_texts = set()
unique_matches = []
for m in matches:
    t = m.get("text", "").strip()[:80]
    if t not in seen_texts:
        seen_texts.add(t)
        unique_matches.append(m)

print(f"Unique chunks: {len(unique_matches)}")
for i, m in enumerate(unique_matches):
    print(f"\nUnique #{i+1}: page={m.get('page')} chunk={m.get('chunk_index')} id={m.get('id')}")
    print(" ", m.get("text", "").strip()[:180])
