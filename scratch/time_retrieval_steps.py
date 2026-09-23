import sys, os, time
from dotenv import load_dotenv
load_dotenv("backend/.env")
sys.path.insert(0, "backend")

from services.retriever import VectorRetriever

r = VectorRetriever()

q = "What is the maximum loan amount?"

t0 = time.perf_counter()
t_devanagari = time.perf_counter()
# Check devanagari
from services.language import count_devanagari_chars, translate_indic_query_to_english
has_dev = count_devanagari_chars(q) >= 2
t_dev = time.perf_counter()
print(f"Devanagari check: {(t_dev - t_devanagari)*1000:.2f}ms")

# Pinecone client / index connection
t_idx0 = time.perf_counter()
idx = r.pinecone_service.get_index()
t_idx1 = time.perf_counter()
print(f"get_index(): {(t_idx1 - t_idx0)*1000:.2f}ms")

# Pinecone search
t_s0 = time.perf_counter()
matches = r.pinecone_service.search_records(query_text=q, top_k=8, namespace="sales_playbooks")
t_s1 = time.perf_counter()
print(f"search_records(): {(t_s1 - t_s0)*1000:.2f}ms")

# Pinecone search 2nd time
t_s2 = time.perf_counter()
matches2 = r.pinecone_service.search_records(query_text=q, top_k=8, namespace="sales_playbooks")
t_s3 = time.perf_counter()
print(f"search_records() 2nd time: {(t_s3 - t_s2)*1000:.2f}ms")
