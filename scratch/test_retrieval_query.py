import sys, os, time
from dotenv import load_dotenv
load_dotenv("backend/.env")
sys.path.insert(0, "backend")

from services.retriever import VectorRetriever

r = VectorRetriever()

t0 = time.perf_counter()
res1 = r.retrieve("What is the minimum CIBIL score required for personal loan?")
t1 = time.perf_counter()
print(f"First call took: {(t1 - t0)*1000:.1f}ms")

t2 = time.perf_counter()
res2 = r.retrieve("What are the interest rates for personal loan?")
t3 = time.perf_counter()
print(f"Second call took: {(t3 - t2)*1000:.1f}ms")
