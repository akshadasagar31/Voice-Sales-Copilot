import sys
import re
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from services.retriever import VectorRetriever

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

queries = [
    'पर्सनल लोन के लिए कौन से दस्तावेज आवश्यक हैं?',
    'एचडीएफसी बँकेच्या वैयक्तिक कर्जासाठी किमान CIBIL score किती आवश्यक आहे?',
    'वैयक्तिक कर्जासाठी कोणती कागदपत्रे आवश्यक आहेत?',
    'वैयक्तिक कर्जाचा व्याजदर किती आहे?',
    'पर्सनल लोन पर ब्याज दर क्या है?',
    'वैयक्तिक कर्जासाठी किमान मासिक पगार किती पाहिजे?'
]

for q in queries:
    t_q = translate_indic_query(q)
    print("=" * 70)
    print(f"Original:   '{q}'")
    print(f"Translated: '{t_q}'")
    res = retriever.retrieve(t_q, top_k=2)
    for r in res.get("results", []):
        score = r.get("score", 0)
        page = r.get("page", "?")
        source = r.get("source", "doc")
        text = r.get("text", "").replace("\n", " ")[:130]
        print(f"  [score={score:.4f}, p.{page}, {source}]: {text}...")
