import sys, os, time
from dotenv import load_dotenv
load_dotenv("backend/.env")
sys.path.insert(0, "backend")

from services.vector_store import PineconeService, _INDEX_CACHE, _VERIFIED_INDEXES

print("Verified indexes initially:", _VERIFIED_INDEXES)
print("Index cache initially:", list(_INDEX_CACHE.keys()))

p1 = PineconeService()
t0 = time.perf_counter()
idx1 = p1.get_index()
t1 = time.perf_counter()
print(f"p1.get_index() took: {(t1 - t0)*1000:.2f}ms")
print("Verified indexes after p1:", _VERIFIED_INDEXES)
print("Index cache after p1:", list(_INDEX_CACHE.keys()))

p2 = PineconeService()
t2 = time.perf_counter()
idx2 = p2.get_index()
t3 = time.perf_counter()
print(f"p2.get_index() took: {(t3 - t2)*1000:.2f}ms")
