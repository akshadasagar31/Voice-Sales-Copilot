import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from services.retriever import VectorRetriever
from services.rag import RAGService

# Test translation function
import re
INDIC_TERMS = [
    (r'(?i)\b(एचडीएफसी\s+बँक[ा-ीa-z]*|एचडीएफसी\s+बैंक)\b', 'HDFC Bank'),
    (r'(?i)\b(वैयक्तिक\s+कर्ज|पर्सनल\s+लोन|व्यक्तिगत\s+ऋण)\b', 'personal loan'),
    (r'(?i)\b(गृह\s+कर्ज|होम\s+लोन|घर\s+कर्ज)\b', 'home loan'),
    (r'(?i)\b(कार\s+लोन|वाहन\s+कर्ज)\b', 'car loan'),
    (r'(?i)\b(व्यवसाय\s+कर्ज|बिजनेस\s+लोन)\b', 'business loan'),
    (r'(?i)\b(बॅलन्स\s+ट्रान्सफर|बैलेंस\s+ट्रांसफर)\b', 'balance transfer'),
    (r'(?i)\b(क्रेडिट\s+कार्ड)\b', 'credit card'),
    (r'(?i)\b(कर्ज|ऋण|लोन)\b', 'loan'),
    (r'(?i)\b(सिबिल|cibil)\s*(स्कोर|score)?\b', 'CIBIL score'),
    (r'(?i)\b(कागदपत्रे|कागदपत्र|दस्तावेज|दस्तावेज़|कागजात)\b', 'documents documentation checklist'),
    (r'(?i)\b(व्याजदर|ब्याज\s*दर|व्याज|ब्याज|दर)\b', 'interest rate ROI pricing'),
    (r'(?i)\b(रॅक\s+दर|रैक\s+रेट)\b', 'rack rate'),
    (r'(?i)\b(किमान|न्यूनतम|कम\s+से\s+कम)\b', 'minimum'),
    (r'(?i)\b(कमाल|अधिकतम|जास्तीत\s+जास्त)\b', 'maximum'),
    (r'(?i)\b(पात्रता|पात्र|मानदंड|निकष|नियम|अटी|शर्तें)\b', 'eligibility criteria norms'),
    (r'(?i)\b(पगार|वेतन|उत्पन्न|आय|कमाई|मासिक\s+आय|मासिक\s+उत्पन्न)\b', 'salary income'),
    (r'(?i)\b(कालावधी|अवधि|मुदत|वर्ष|वर्षे|साल|महिने|माह)\b', 'tenure duration months'),
    (r'(?i)\b(प्रोसेसिंग\s+फी|प्रोसेसिंग\s+शुल्क|शुल्क|फीस|फी|चार्जेस)\b', 'processing fee charges'),
    (r'(?i)\b(फोरक्लोजर|प्रीपेमेंट|दंड)\b', 'foreclosure prepayment penalty charges'),
    (r'(?i)\b(वय|उम्र|आयु)\b', 'age limit'),
    (r'(?i)\b(आवश्यक|लागणारे|लागतील|चाहिए|गरज)\b', 'required'),
    (r'(?i)\b(काय|कोणते|कोणती|कोणत्या|किती|कसा|कशी|कसे|मिळेल|असेल)\b', ''),
    (r'(?i)\b(क्या|कौनसा|कौनसी|कौनसे|कितना|कितनी|कितने|होगा|होगी|होना|है|हैं|आहे|आहेत)\b', ''),
    (r'(?i)\b(साठी|बद्दल|मध्ये|कडून|च्या|चे|ची|ला)\b', ''),
    (r'(?i)\b(के\s+लिए|के\s+बारे\s+में|में|से|का|की|के)\b', ''),
]

def translate_indic_query(q):
    res = q
    for pat, rep in INDIC_TERMS:
        res = re.sub(pat, rep, res)
    res = re.sub(r'[\u0900-\u097F]+', ' ', res)
    res = re.sub(r'\s+', ' ', res).strip()
    return res

retriever = VectorRetriever()
rag = RAGService(retriever=retriever)
client = rag.get_http_client()

test_cases = [
    ("Marathi Docs", "mr", "वैयक्तिक कर्जासाठी कोणती कागदपत्रे आवश्यक आहेत?"),
    ("Hindi Docs", "hi", "पर्सनल लोन के लिए कौन से दस्तावेज आवश्यक हैं?"),
    ("Marathi CIBIL", "mr", "एचडीएफसी बँकेच्या वैयक्तिक कर्जासाठी CIBIL score चा नियम काय आहे?"),
    ("Hindi CIBIL", "hi", "एचडीएफसी बैंक में पर्सनल लोन के लिए सिबिल स्कोर का क्या नियम है?"),
    ("Marathi Interest", "mr", "वैयक्तिक कर्जाचा व्याजदर किती आहे?"),
    ("Hindi Interest", "hi", "पर्सनल लोन पर ब्याज दर क्या है?"),
    ("Marathi Minimum CIBIL", "mr", "एचडीएफसी बँकेच्या वैयक्तिक कर्जासाठी किमान CIBIL score किती आवश्यक आहे?"),
]

for label, lang, q in test_cases:
    print("=" * 80)
    print(f"CASE: {label} ({lang})")
    print(f"Question: '{q}'")
    t_q = translate_indic_query(q)
    print(f"Translated query for retrieval: '{t_q}'")
    
    # Retrieve using translated query
    ret_res = retriever.retrieve(t_q, top_k=3)
    chunks = ret_res.get("results", [])
    print(f"Retrieved {len(chunks)} chunks. Top chunk score={chunks[0].get('score'):.4f} from {chunks[0].get('source')} p.{chunks[0].get('page')}")
    print(f"Top chunk text snippet: {chunks[0].get('text', '')[:120]}...")
    
    # Format voice context
    from services.rag import format_voice_context
    ctx = format_voice_context(chunks, max_chunks=3)
    
    lang_name = "Marathi" if lang == "mr" else "Hindi"
    sys_prompt = f"""You are a sales copilot answering by voice in {lang_name}.
The Context below is in English from uploaded sales playbooks.
Answer the user's question directly, accurately, and truthfully in 1 to 2 clear spoken sentences in {lang_name}, strictly grounded in the Context below.

CRITICAL RULES:
1. Base your answer EXCLUSIVELY on the provided Context. Translate and synthesize the facts from the English Context into natural {lang_name}. Do NOT use outside knowledge or assumptions.
2. Start IMMEDIATELY with the direct factual answer. NEVER include greetings, pleasantries, or preamble.
3. If the topic or answer is completely absent from the context, you MUST respond EXACTLY with this phrase and nothing else:
"FALLBACK"
4. Keep the answer concise (1 to 2 sentences) for instant spoken response."""

    resp = client.post(
        f"{rag.base_url}/chat/completions",
        json={
            "model": "meta-llama/llama-3.3-70b-instruct",
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion:\n{q}"}
            ],
            "temperature": 0.0,
        },
        headers={"Authorization": f"Bearer {rag.api_key}", "Content-Type": "application/json"}
    )
    ans = resp.json()["choices"][0]["message"]["content"]
    print(f"Grounded Answer in {lang_name}: {ans}\n")
