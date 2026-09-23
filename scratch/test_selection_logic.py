import sys
from pathlib import Path

backend_dir = Path(r"c:\Users\Akshada\OneDrive\Pictures\Documents\projects\Voice-Sales-Copilot\backend")
sys.path.insert(0, str(backend_dir))
sys.stdout.reconfigure(encoding="utf-8")
from services.language import detect_language, MARATHI_WORDS, HINDI_WORDS, MARATHI_SPECIFIC_CHARS

cases = [
    {
        "category": "Marathi",
        "mr": "HDFC बँकेच्या वैयक्तिक कर्जासाठी किमान CIBIL score किती आवश्यक आहे",
        "multi": "HDFC bank केचा वैयक्तिक कर्जासाठी किमान CIBIL score कितनी आवश्यका है?"
    },
    {
        "category": "Mixed Marathi-English",
        "mr": "माझा CIBIL score ७० एकशे पन्नास आहे. मला HDFC बँक कडून personal loanवर किती interest rate मिळेल",
        "multi": "माज़ा CIBILScore796है. मला HDFC bank करुण personal loan work किति interest rate मिले."
    },
    {
        "category": "Hindi",
        "mr": "Personal loan केलीये Newenthum Sibill score कितना होणार आहे?",
        "multi": "Personal loan के लिए न्यूनतम CIBIL score कितना होना चाहिए?"
    },
    {
        "category": "English",
        "mr": "What is the minimum SIBIL score for personal loans?",
        "multi": "What is the minimum CIBIL score for personal loans?"
    }
]

# Add financial/conversational Hindi words for accurate scoring
EXPANDED_HINDI_WORDS = set(HINDI_WORDS) | {
    "लिए", "न्यूनतम", "कितना", "कितने", "कितनी", "होना", "होता", "होती", "होते",
    "होगा", "होगी", "होंगे", "ऋण", "ब्याज", "व्याज", "दर", "महीने", "प्रतिमाह"
}

def select_best_transcript(mr_text: str, multi_text: str):
    mr_text = (mr_text or "").strip()
    multi_text = (multi_text or "").strip()
    
    if not mr_text and not multi_text:
        return "", "en", "none"
    if not mr_text:
        return multi_text, detect_language(multi_text), "multi_only"
    if not multi_text:
        return mr_text, detect_language(mr_text), "mr_only"
        
    import re
    mr_devanagari_words = set(re.findall(r"[\u0900-\u097F]+", mr_text))
    multi_devanagari_words = set(re.findall(r"[\u0900-\u097F]+", multi_text))
    
    mr_hits = sum(1 for w in mr_devanagari_words if w in MARATHI_WORDS)
    mr_has_char = any(ch in MARATHI_SPECIFIC_CHARS for ch in mr_text)
    
    multi_hi_hits = sum(1 for w in multi_devanagari_words if w in EXPANDED_HINDI_WORDS)
    multi_mr_hits = sum(1 for w in multi_devanagari_words if w in MARATHI_WORDS)
    
    # 1. High confidence Marathi: specific Marathi characters (ळ, ऱ) OR strong Marathi lexical hits
    if mr_has_char or (mr_hits >= 2 and mr_hits > multi_hi_hits):
        return mr_text, "mr", "Stream MR"
        
    # 2. High confidence Hindi: multi stream has clear Hindi markers and MR lacks distinctive Marathi
    if multi_hi_hits > 0 and multi_hi_hits >= mr_hits:
        return multi_text, "hi", "Stream Multi (Hindi)"
        
    # 3. If mr_hits >= 1 and multi has 0 Hindi markers:
    if mr_hits >= 1 and multi_hi_hits == 0:
        return mr_text, "mr", "Stream MR"
        
    # 4. English / Latin / Default:
    lang = detect_language(multi_text)
    return multi_text, lang, f"Stream Multi ({lang})"

for c in cases:
    cat = c["category"]
    chosen, lang, src = select_best_transcript(c["mr"], c["multi"])
    print(f"[{cat}] -> Winner: ({src}, lang={lang})\n   '{chosen}'\n")

