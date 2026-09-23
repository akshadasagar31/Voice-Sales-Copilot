import sys
import os
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from services.retriever import VectorRetriever

retriever = VectorRetriever()

tests = [
    ("Hindi Docs (original)", "पर्सनल लोन के लिए कौन से दस्तावेज आवश्यक हैं?"),
    ("Hindi Docs (English translation)", "What documents are required for a personal loan?"),
    ("Marathi CIBIL (original)", "एचडीएफसी बँकेच्या वैयक्तिक कर्जासाठी किमान CIBIL score किती आवश्यक आहे?"),
    ("Marathi CIBIL (English translation)", "What is the minimum CIBIL score required for HDFC Bank personal loan?"),
    ("Marathi Docs (original)", "वैयक्तिक कर्जासाठी कोणती कागदपत्रे आवश्यक आहेत?"),
    ("Marathi Docs (English translation)", "What documents are required for a personal loan?"),
    ("Marathi Interest (original)", "वैयक्तिक कर्जाचा व्याजदर किती आहे?"),
    ("Marathi Interest (English translation)", "What is the interest rate for a personal loan?"),
]

for label, q in tests:
    print("=" * 70)
    print(f"{label}: '{q}'")
    res = retriever.retrieve(q, top_k=3)
    results = res.get("results", [])
    print(f"Found {len(results)} chunks:")
    for i, r in enumerate(results):
        score = r.get("score", 0)
        page = r.get("page", "?")
        source = r.get("source", "doc")
        text = r.get("text", "").replace("\n", " ")[:120]
        print(f"  #{i+1} [score={score:.4f}, p.{page}, {source}]: {text}...")
