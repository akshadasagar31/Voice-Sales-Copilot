"""
Benchmark script comparing retrieval accuracy and latency between:
1. Reference index 'voice-sales-copilot' (local_fast embedder)
2. New integrated index 'voice-sales-copilot-llama' (server-side llama-text-embed-v2)
across English, Hindi, and Marathi queries.
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

env_path = backend_dir / ".env"
load_dotenv(dotenv_path=env_path)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pinecone import Pinecone
from services.embedder import TextEmbedder

def run_comparison():
    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    pc = Pinecone(api_key=api_key)

    old_idx = pc.Index("voice-sales-copilot")
    new_idx = pc.Index("voice-sales-copilot-llama")
    embedder = TextEmbedder(provider="local_fast", dimension=1536)

    test_queries = [
        ("EN", "What are our volume discount tiers for enterprise contracts?"),
        ("EN", "What is the minimum CIBIL score for loan eligibility?"),
        ("HI", "ऋण पात्रता और न्यूनतम सिबिल (CIBIL) स्कोर मानदंड क्या हैं?"),
        ("HI", "ब्याज दरें और आरओआई (ROI) नीतियां क्या हैं?"),
        ("MR", "व्यवसाय कर्जासाठी व्याजदर आणि अटी काय आहेत?"),
        ("MR", "कंपनीचे एंटरप्राइज सवलत दर आणि करार पर्याय काय आहेत?"),
    ]

    print("=" * 80)
    print("PINECONE RAG BENCHMARK: OLD (voice-sales-copilot) vs NEW (voice-sales-copilot-llama)")
    print("=" * 80)

    # Warmup both indexes
    _ = old_idx.query(vector=[0.01]*1536, top_k=1, namespace="sales_playbooks")
    _ = new_idx.search(namespace="sales_playbooks", top_k=1, inputs={"text": "warmup"})

    results_table = []

    for lang, q in test_queries:
        # 1. OLD PATH: Client-side local embed + vector search
        t0 = time.perf_counter()
        q_vec = embedder.embed_query(q)
        t_embed = time.perf_counter()
        old_res = old_idx.query(vector=q_vec, top_k=3, namespace="sales_playbooks", include_metadata=True)
        t_old_end = time.perf_counter()

        old_latency = (t_old_end - t0) * 1000.0
        old_embed_time = (t_embed - t0) * 1000.0
        old_top = old_res.matches[0] if old_res.matches else None
        old_top_text = (old_top.metadata.get("text", "")[:80] + "...") if old_top else "None"
        old_source = old_top.metadata.get("source", "N/A") if old_top else "N/A"

        # 2. NEW PATH: Pinecone Integrated server-side search (llama-text-embed-v2)
        t0 = time.perf_counter()
        new_res = new_idx.search(namespace="sales_playbooks", top_k=3, inputs={"text": q})
        t_new_end = time.perf_counter()

        new_latency = (t_new_end - t0) * 1000.0
        new_hits = new_res.result.hits if hasattr(new_res, "result") else []
        new_top = new_hits[0] if new_hits else None
        new_fields = new_top.fields if new_top else {}
        new_top_text = (new_fields.get("text", "")[:80] + "...") if new_fields else "None"
        new_source = new_fields.get("source", "N/A") if new_fields else "N/A"

        results_table.append({
            "lang": lang,
            "query": q,
            "old_latency": old_latency,
            "old_embed_time": old_embed_time,
            "old_score": old_top.score if old_top else 0.0,
            "old_source": old_source,
            "old_text": old_top_text,
            "new_latency": new_latency,
            "new_score": new_top.score if new_top else 0.0,
            "new_source": new_source,
            "new_text": new_top_text,
        })

        print(f"\n[{lang}] Query: '{q}'")
        print(f"  OLD Path: {old_latency:.1f} ms (embed: {old_embed_time:.1f}ms, score: {old_top.score if old_top else 0.0:.4f}, doc: {old_source})")
        print(f"  NEW Path: {new_latency:.1f} ms (server-side llama embed, score: {new_top.score if new_top else 0.0:.4f}, doc: {new_source})")
        print(f"  Old Top Hit: {old_top_text.strip()}")
        print(f"  New Top Hit: {new_top_text.strip()}")

    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    avg_old = sum(r["old_latency"] for r in results_table) / len(results_table)
    avg_new = sum(r["new_latency"] for r in results_table) / len(results_table)
    print(f"Average Old Retrieval Latency: {avg_old:.1f} ms")
    print(f"Average New Integrated Retrieval Latency: {avg_new:.1f} ms (Difference: {avg_new - avg_old:+.1f} ms)")
    print("=" * 80)

if __name__ == "__main__":
    run_comparison()
