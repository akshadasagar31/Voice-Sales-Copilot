import sys
import os
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from services.retriever import VectorRetriever
from services.rag import RAGService

retriever = VectorRetriever()
rag = RAGService(retriever=retriever)

queries = [
    ('English CIBIL', 'en', 'What is the minimum CIBIL score for personal loans?'),
    ('Hindi CIBIL', 'hi', 'पर्सनल लोन के लिए न्यूनतम सिबिल स्कोर कितना होना चाहिए?'),
    ('Marathi CIBIL', 'mr', 'एचडीएफसी बँकेच्या वैयक्तिक कर्जासाठी किमान CIBIL score किती आवश्यक आहे?'),
    ('Hindi Documents', 'hi', 'पर्सनल लोन के लिए कौन से दस्तावेज आवश्यक हैं?'),
    ('Marathi Documents', 'mr', 'वैयक्तिक कर्जासाठी कोणती कागदपत्रे आवश्यक आहेत?'),
    ('Hindi Interest', 'hi', 'पर्सनल लोन पर ब्याज दर क्या है?'),
    ('Marathi Interest', 'mr', 'वैयक्तिक कर्जाचा व्याजदर किती आहे?')
]

print("=" * 80)
print("TESTING RETRIEVAL DIRECTLY")
print("=" * 80)

for label, lang, q in queries:
    print(f"\n--- {label} ({lang}): '{q}' ---")
    ret_res = retriever.retrieve(q, top_k=3)
    results = ret_res.get("results", [])
    print(f"Retrieved {len(results)} chunks:")
    for i, r in enumerate(results):
        score = r.get("score", 0)
        source = r.get("source", "unknown")
        page = r.get("page", "?")
        text = r.get("text", "").replace("\n", " ")[:130]
        print(f"  #{i+1} [score={score:.4f}, p.{page}, {source}]: {text}...")

    # Now test full RAG answer
    ans_res = rag.answer_question(q, top_k=3, language=lang)
    print(f"  >> RAG Answer ({ans_res.get('language')}): {ans_res.get('answer')}")
    print(f"  >> Fallback used: {ans_res.get('fallback_used')}")
