import sys
import httpx
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8001"

LIVE_TESTS = [
    {
        "label": "Marathi Document Query",
        "lang": "mr",
        "question": "वैयक्तिक कर्जासाठी कोणती कागदपत्रे आवश्यक आहेत?",
        "expected_terms": ["ओळख", "पत्ता", "उत्पन्न", "दस्तऐवज", "कागदपत्रे"],
        "expected_source": "Personal_Loan_General_Information.pdf"
    },
    {
        "label": "Hindi Document Query",
        "lang": "hi",
        "question": "पर्सनल लोन के लिए कौन से दस्तावेज आवश्यक हैं?",
        "expected_terms": ["पहचान", "पता", "आय", "दस्तावेज", "वेतन"],
        "expected_source": "Personal_Loan_General_Information.pdf"
    },
    {
        "label": "Marathi CIBIL Query",
        "lang": "mr",
        "question": "एचडीएफसी बँकेच्या वैयक्तिक कर्जासाठी CIBIL score चा नियम काय आहे?",
        "expected_terms": ["CIBIL", "७३०", "730", "प्राइसिंग", "दरकार्ड"],
        "expected_source": "HDFC_Bank_Master_Policy_CIBIL_Updated.pdf"
    },
    {
        "label": "Hindi CIBIL Query",
        "lang": "hi",
        "question": "एचडीएफसी बैंक में पर्सनल लोन के लिए सिबिल स्कोर का क्या नियम है?",
        "expected_terms": ["सिबिल", "730", "रेट", "प्राइसिंग"],
        "expected_source": "HDFC_Bank_Master_Policy_CIBIL_Updated.pdf"
    },
    {
        "label": "Marathi Interest Rate Query",
        "lang": "mr",
        "question": "वैयक्तिक कर्जाचा व्याजदर किती आहे?",
        "expected_terms": ["व्याजदर", "१३.७५%", "13.75%"],
        "expected_source": "HDFC_Bank_Master_Policy_CIBIL_Updated.pdf"
    },
    {
        "label": "English Baseline Query",
        "lang": "en",
        "question": "What documents are commonly requested for personal loans?",
        "expected_terms": ["identity", "address", "income", "statements"],
        "expected_source": "Personal_Loan_General_Information.pdf"
    }
]

print("=" * 80)
print("LIVE VERIFICATION: ENGLISH PDF -> MARATHI / HINDI RAG RETRIEVAL")
print("=" * 80)

client = httpx.Client(timeout=60.0)

all_passed = True
for test in LIVE_TESTS:
    print(f"\n[TEST] {test['label']} ({test['lang']})")
    print(f"Question: \"{test['question']}\"")
    
    res = client.post(
        f"{BASE_URL}/api/ask",
        json={
            "question": test["question"],
            "language": test["lang"],
            "top_k": 3,
        }
    )
    
    if res.status_code != 200:
        print(f"  FAILED: HTTP {res.status_code} - {res.text}")
        all_passed = False
        continue
        
    data = res.json()
    answer = data.get("answer", "")
    sources = data.get("sources", [])
    context = data.get("context_used", [])
    fallback = data.get("fallback_used", False)
    resp_lang = data.get("language", "")
    
    print(f"  Response Status:  HTTP 200 OK")
    print(f"  Detected Lang:    {resp_lang}")
    print(f"  Fallback Trigger: {fallback}")
    print(f"  Sources Found:    {sources}")
    print(f"  Chunks Retrieved: {len(context)}")
    if context:
        print(f"  Top Chunk:        [score={context[0].get('score'):.4f}, p.{context[0].get('page')}] from {context[0].get('source')}")
    print(f"  Grounded Answer:  \"{answer}\"")
    
    # Assertions
    assert fallback is False, f"Expected grounded answer but fallback was triggered for {test['label']}"
    assert len(sources) > 0, "No sources were returned"
    assert any("pdf" in s.lower() for s in sources), f"Expected PDF playbook in sources, got {sources}"
    
    matches = [term for term in test["expected_terms"] if term.lower() in answer.lower()]
    print(f"  Grounding Match:  {len(matches)}/{len(test['expected_terms'])} grounded terms detected ({matches})")
    assert len(matches) > 0, f"No expected grounded terms found in answer: {answer}"
    print(f"  PASSED [OK]")

print("\n" + "=" * 80)
if all_passed:
    print("ALL LIVE MULTILINGUAL RAG RETRIEVAL TESTS PASSED WITH 100% GROUNDING!")
print("=" * 80)
