import sys
import re
from pathlib import Path

# Add backend to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from services.language import (
    detect_spoken_language,
    get_response_language,
    MARATHI_DISTINCTIVE_WORDS,
    MARATHI_SPECIFIC_CHARS,
    LANG_EN,
    LANG_HI,
    LANG_MR,
)

words = set(MARATHI_DISTINCTIVE_WORDS) | {'रुपयेंचे', 'लाखेंचे', 'आणि', 'आम्ही', 'हवे', 'हवं'}

def test_select(mr_text, multi_text):
    mr_text = (mr_text or '').strip()
    multi_text = (multi_text or '').strip()
    if not mr_text and not multi_text:
        return '', LANG_EN, 'none'
    if not mr_text:
        lang = get_response_language(detect_spoken_language(multi_text), text=multi_text)
        return multi_text, lang, 'multi_only'
    if not multi_text:
        lang = get_response_language(detect_spoken_language(mr_text), text=mr_text)
        return mr_text, lang, 'mr_only'
    
    multi_spoken = detect_spoken_language(multi_text)
    multi_resp = get_response_language(multi_spoken, text=multi_text)
    
    mr_dev_words = set(re.findall(r'[\u0900-\u097F]+', mr_text))
    mr_distinct_hits = sum(1 for w in mr_dev_words if w in words)
    has_mr_char = any(ch in MARATHI_SPECIFIC_CHARS for ch in mr_text)
    
    # 1. If multi stream resolved to Hindi
    if multi_resp == LANG_HI:
        if mr_distinct_hits >= 3:
            return mr_text, LANG_MR, 'Stream MR (Marathi)'
        return multi_text, LANG_HI, 'Stream Multi (Hindi)'
        
    # 2. Marathi markers or characters present (when multi did not resolve to Hindi)
    if has_mr_char or mr_distinct_hits >= 1 or multi_resp == LANG_MR:
        return mr_text if (mr_distinct_hits > 0 or has_mr_char) else multi_text, LANG_MR, 'Stream MR (Marathi)'
        
    # 3. English
    if multi_resp == LANG_EN and multi_spoken == LANG_EN:
        return multi_text, LANG_EN, 'Stream Multi (English)'
        
    return multi_text, multi_resp, f'Stream Multi ({multi_resp})'

cases = [
    ('Hindi Speech (Test 5)', '150 लाख ही पर्सनल लावून की लय सिव्हिल स्कोर खेळणे होण्याच्या आहे मगली EMIKidney IT', '15 lakh t personal loan key lei CIBIL score kidna', LANG_HI),
    ('Marathi Speech (Test 4)', '10 लाख रुपयेंचे पर्सनल लोन सॅफ आयसिपोल स्कोर किती ऐकतो आणि ड्रामा आम्ही किती असो', 'Ten lakh rupee personal loan', LANG_MR),
    ('Synthetic Priority Test', 'माझे नाव अमोल देशमुख आहे आणि 5 लाख रुपयांचे कर्ज पाहिजे', 'Mera naam Amol Deshmukh hai aur 5 lakh rupaye chahiye', LANG_MR),
    ('Marathi English Mixed', 'मला 10 लाख रुपयांचे personal loan हवे आहे, माझा CIBIL score 750 आहे', '10 lakh personal loan CIBIL score 750', LANG_MR),
    ('English Speech (Test 1)', 'हॉटेल 15 लाख पर्सनल लोन विथ एचडीएफसी बँक', 'What is the minimum CIBIL score required for a personal loan of 15 lakh with HDFC Bank', LANG_EN),
    ('Short Marathi', 'कर्ज पाहिजे', 'loan pahije', LANG_MR),
]

if __name__ == '__main__':
    all_pass = True
    for name, mr, multi, expected in cases:
        res = test_select(mr, multi)
        passed = (res[1] == expected)
        if not passed:
            all_pass = False
        print(f"{name}: {'PASS' if passed else 'FAIL'} -> got {res[1]} (expected {expected})")
    print('ALL PASSED:', all_pass)
