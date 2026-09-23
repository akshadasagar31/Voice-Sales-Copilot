import sys
sys.path.insert(0, 'backend')
import time
from dotenv import load_dotenv
load_dotenv('backend/.env')

from services.retriever import VectorRetriever

vr = VectorRetriever()
# First call (cold)
t0 = time.perf_counter()
res = vr.retrieve("What is the minimum CIBIL score for personal loans?", top_k=4)
t1 = time.perf_counter()
print(f"Cold retrieval time: {(t1-t0)*1000:.1f}ms")

# Second call (warm)
t0 = time.perf_counter()
res = vr.retrieve("What are the interest rates for HDFC loans?", top_k=4)
t1 = time.perf_counter()
print(f"Warm retrieval time: {(t1-t0)*1000:.1f}ms")
