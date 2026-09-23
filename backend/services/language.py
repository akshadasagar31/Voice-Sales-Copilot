# ============================================================================
# MULTILINGUAL & GREETING UTILITIES (backend/services/language.py)
# ============================================================================
# WHAT THIS SERVICE DOES:
# Handles language identification and greetings for English, Hindi, and Marathi.
#
# KEY CAPABILITIES:
# 1. Devanagari Script Analysis: Detects whether text is in Indian script.
# 2. Hindi vs. Marathi Disambiguation:
#    Both Hindi and Marathi use the Devanagari alphabet. This service looks for
#    Marathi-specific characters (such as ळ and ऱ) and distinctive vocabulary
#    (e.g., नमस्कार vs. नमस्ते, आहे vs. है) to reliably distinguish them.
# 3. Conversational Greeting Detection: Identifies pure greetings so the copilot
#    can respond warmly and naturally without triggering database errors.
# 4. Transliteration to Phonetics: Converts Devanagari into Latin phonetic sounds
#    so Deepgram Aura TTS pronounces Indian words accurately.
# ============================================================================

import re
from typing import Dict, Any, List, Optional

# Supported Language Codes
LANG_EN = "en"
LANG_HI = "hi"
LANG_MR = "mr"
LANG_MIXED = "mixed"

SUPPORTED_LANGUAGES = [LANG_EN, LANG_HI, LANG_MR, LANG_MIXED]

LANGUAGE_NAMES = {
    LANG_EN: "English",
    LANG_HI: "Hindi",
    LANG_MR: "Marathi",
    LANG_MIXED: "Mixed (Hindi/Marathi/English)",
}


# Devanagari character ranges
DEVANAGARI_START = 0x0900
DEVANAGARI_END = 0x097F

# Marathi-specific characters
MARATHI_SPECIFIC_CHARS = {"ळ", "ऱ"}  # U+0933 (LLA), U+0931 (RRA)

# Lexical markers for Marathi
MARATHI_WORDS = {
    "नमस्कार", "आहे", "आहेत", "नाही", "नाहीत", "होय", "काय", "कसे", "कशी",
    "कसा", "करावे", "करा", "सांगा", "माहिती", "बद्दल", "साठी", "मध्ये",
    "तुमचे", "तुमची", "तुमच्या", "आपले", "आपली", "आपल्या", "कोणते",
    "कोणती", "कोणत्या", "पाहिजे", "द्या", "येईल", "झाले", "झाला", "झाली",
    "शकता", "शकतो", "पहा", "नवीन", "सर्व", "असे", "तसे", "फार", "खूप",
    "हॅलो", "सुप्रभात", "शुभ", "सकाळ", "संध्याकाळ", "दुपार", "धन्यवाद",
    "कृपया", "कर्ज", "व्याजदर", "कागदपत्रे", "पात्रता", "निकष", "योजना",
    "कालावधी", "नियम", "अटी", "काही", "इथे", "तिथे", "कोण", "कधी",
    "मला", "माझा", "माझी", "माझे", "माझ्या", "तुला", "तुझा", "तुझी", "तुझे",
    "आम्हाला", "आमचा", "आमची", "आमचे", "कडून", "किती", "मिळेल", "मिळू",
    "असेल", "असेलच", "होते", "होता", "होती", "बँकेकडून", "बँकेच्या", "बँकेत",
    "दरमहा", "मासिक", "वैयक्तिक", "कर्जासाठी", "किमान", "आवश्यक",
    "नाव", "नांव", "हवे", "हवा", "हवी", "रुपये", "लाख", "कोटी", "हजार",
    "महिने", "वर्ष", "वर्षे", "पॅन", "आधार", "बँक", "खाते", "पगार",
    "नोकरी", "व्यवसाय", "कंपनी", "आणि", "मी"
}

# Lexical markers for Hindi
HINDI_WORDS = {
    "नमस्ते", "है", "हैं", "नहीं", "हाँ", "क्या", "कैसे", "कैसी",
    "कैसा", "करें", "करो", "बताएं", "बताओ", "जानकारी", "के बारे में",
    "के लिए", "लिए", "में", "आपका", "आपकी", "आपके", "अपना", "अपनी", "अपने",
    "कौनसा", "कौनसी", "कौनसे", "चाहिए", "दीजिए", "सकता", "सकती", "सकते",
    "होगा", "होगी", "होंगे", "होना", "होता", "होती", "होते", "कितना", "कितने", "कितनी",
    "न्यूनतम", "अधिकतम", "बहुत", "अच्छा", "हेलो", "हाय", "सुप्रभात", "शुभ",
    "धन्यवाद", "कृपया", "ऋण", "ब्याज", "दस्तावेज", "शर्तें", "कहाँ", "कब", "प्रतिमाह",
    "नाम", "मेरा", "मेरी", "मेरे", "मुझे", "हमें", "हम", "रुपये", "लाख",
    "करोड़", "हजार", "महीने", "साल", "पर्सनल", "लोन", "खाता", "पैन",
    "आधार", "वेतन", "नौकरी", "कंपनी", "बताइए", "और", "मैं"
}

# Romanized / Transliterated markers
ROMANIZED_HINDI = {
    "namaste", "pranam", "aap", "kaise", "kya", "hai", "hain",
    "bataye", "kripya", "dhanyawad", "batao", "chahiye", "kaunsa", "kaunsi",
    "shukriya", "accha", "kaisi", "kaisa", "hamare", "humare", "mera", "meri",
    "mere", "karo", "karna", "karenge", "mujhe", "naam", "main", "humein",
    "aapka", "aapki", "aapke", "apna", "apni", "apne", "kitna", "kitne", "kitni",
    "hoga", "hogi", "honge", "nahi", "nahin", "haan", "lekin", "liye",
    "bataiye", "deejie", "deejiye"
}

ROMANIZED_MARATHI = {
    "namaskar", "kasa", "kashi", "ahes", "aahes", "aahe", "aahet", "ahet", "mahit",
    "sanga", "tumhi", "tumchi", "tumche", "krupaya", "dhanyavad", "pahije",
    "kay", "kayat", "baddal", "savlat", "savlati", "dar", "mala", "kiti",
    "amhi", "aamhi", "aapan", "kahi", "karaycha", "karayche", "maze", "naav",
    "majha", "majhi", "majhe", "majhya", "aani", "ani", "kuthe", "kadhi",
    "kon", "hava", "havi", "have", "dya", "kinva", "zale", "jhale", "karun"
}

# Common English core vocabulary to confirm English phrases with certainty
ENGLISH_CORE_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for",
    "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his",
    "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my",
    "one", "all", "would", "there", "their", "what", "so", "up", "out", "if",
    "about", "who", "get", "which", "go", "me", "when", "make", "can", "like",
    "time", "no", "just", "him", "know", "take", "people", "into", "year", "your",
    "good", "some", "could", "them", "see", "other", "than", "then", "now", "look",
    "only", "come", "its", "over", "think", "also", "back", "after", "use", "two",
    "how", "our", "work", "first", "well", "way", "even", "new", "want", "because",
    "any", "these", "give", "day", "most", "us", "is", "are", "was", "were", "am",
    "name", "phone", "loan", "personal", "home", "business", "company", "tenure",
    "duration", "amount", "salary", "thousand", "hundred", "need", "looking", "interested"
}

# Pure Greeting Patterns
GREETING_TOKENS_EN = {
    "hi", "hello", "hey", "greetings", "good morning", "good afternoon",
    "good evening", "howdy", "welcome"
}

GREETING_TOKENS_HI = {
    "नमस्ते", "नमस्कार", "प्रणाम", "हेलो", "हाय", "हैलो", "सुप्रभात", "शुभ प्रभात", "शुभ संध्या",
    "शुभ दोपहर", "शुभ दिन", "राम राम"
}

GREETING_TOKENS_MR = {
    "नमस्कार", "हॅलो", "हाय", "सुप्रभात", "शुभ सकाळ", "शुभ दुपार", "शुभ संध्याकाळ",
    "शुभ दिन", "जय महाराष्ट्र"
}

FALLBACK_MESSAGES = {
    LANG_EN: "I am sorry, but the provided documentation does not contain sufficient information to answer this question.",
    LANG_HI: "मुझे खेद है, लेकिन दिए गए दस्तावेज़ों में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी उपलब्ध नहीं है।",
    LANG_MR: "मला दिलगीर आहे, परंतु दिलेल्या दस्तऐवजांमध्ये या प्रश्नाचे उत्तर देण्यासाठी पुरेशी माहिती उपलब्ध नाही.",
}

GREETING_RESPONSES = {
    LANG_EN: "Hello! How can I help you today?",
    LANG_HI: "नमस्ते! मैं आज आपकी क्या सहायता कर सकता हूँ?",
    LANG_MR: "नमस्कार! मी आज आपली काय मदत करू शकेन?",
}

EMPTY_KB_GREETINGS = {
    LANG_EN: "Hello! Welcome to Voice Sales Copilot. Currently, no sales playbooks or documents have been uploaded to the knowledge base yet. Please upload your sales playbooks in the Playbooks section to receive grounded answers and recommendations.",
    LANG_HI: "नमस्ते! वॉयस सेल्स कोपायलट में आपका स्वागत है। वर्तमान में नॉलेज बेस में कोई सेल्स प्लेबुक या दस्तावेज़ अपलोड नहीं किए गए हैं। सटीक उत्तर और सिफारिशें प्राप्त करने के लिए कृपया प्लेबुक अनुभाग में अपने दस्तावेज़ अपलोड करें।",
    LANG_MR: "नमस्कार! व्हॉइस सेल्स कोपायलटमध्ये आपले स्वागत आहे. सध्या नॉलेज बेसमध्ये कोणतीही सेल्स प्लेबुक किंवा दस्तऐवज अपलोड केलेले नाहीत. अचूक उत्तरे आणि शिफारसी मिळविण्यासाठी कृपया प्लेबुक्स विभागात आपले दस्तऐवज अपलोड करा.",
}


def count_devanagari_chars(text: str) -> int:
    """Returns count of Unicode characters within the Devanagari block."""
    return sum(1 for ch in text if DEVANAGARI_START <= ord(ch) <= DEVANAGARI_END)


# English grammatical and structural words indicative of English phrasing
ENGLISH_GRAMMAR_WORDS = {
    "and", "i", "is", "am", "are", "want", "need", "for", "from", "my", "the",
    "have", "you", "this", "please", "call", "looking", "interested", "at",
    "in", "with", "years", "months", "working", "work", "require", "required"
}

NEW_LEAD_PATTERNS = [
    r"\b(?:new|next|another|fresh)\s+lead\b",
    r"\bstart\s+(?:a\s+)?(?:new|fresh|next)\s+lead\b",
    r"\bstart\s+fresh\b",
    r"\b(?:reset|clear|restart)\s+(?:lead|session|call)?\b",
    r"\bनया\s+लीड\b",
    r"\bदूसरा\s+लीड\b",
    r"\bनवीन\s+लीड\b",
    r"\bपुढचा\s+लीड\b",
    r"\bपुढील\s+लीड\b",
    r"\bअगला\s+लीड\b",
]


def is_new_lead_intent(text: str) -> bool:
    """
    Detects if the user's spoken transcript conveys intent to start a new lead or reset the session.
    e.g. 'new lead', 'start fresh', 'start a new lead', 'नया लीड', 'नवीन लीड'.
    """
    if not text:
        return False
    lower = text.lower().strip()
    return any(re.search(pat, lower, flags=re.IGNORECASE) for pat in NEW_LEAD_PATTERNS)


def detect_spoken_language(text: str) -> str:
    """
    Authoritative spoken language detector for sales copilot:
    - Returns 'en' for pure English
    - Returns 'hi' for pure Hindi
    - Returns 'mr' for pure Marathi
    - Returns 'mixed' for mixed Hindi/Marathi/English (Hinglish/code-switching)
    """
    clean_text = (text or "").strip()
    if not clean_text:
        return LANG_EN

    devanagari_count = count_devanagari_chars(clean_text)
    total_alpha = sum(1 for ch in clean_text if ch.isalpha())

    # Words in Devanagari and Latin
    words_dev = set(re.findall(r"[\u0900-\u097F]+", clean_text))
    words_latin = [w.lower() for w in re.findall(r"[a-zA-Z]+", clean_text)]
    words_latin_set = set(words_latin)

    en_grammar_hits = sum(1 for w in words_latin if w in ENGLISH_GRAMMAR_WORDS)
    en_core_hits = sum(1 for w in words_latin if w in ENGLISH_CORE_WORDS)
    mr_roman_hits = sum(1 for w in words_latin if w in ROMANIZED_MARATHI)
    hi_roman_hits = sum(1 for w in words_latin if w in ROMANIZED_HINDI)

    # 1. Check if Devanagari script is present (>= 2 characters)
    if devanagari_count >= 2 and (total_alpha == 0 or (devanagari_count / total_alpha) > 0.15):
        # Check for mixed script code-switching (Devanagari + English grammatical phrases)
        if en_grammar_hits >= 1 and (en_core_hits >= 2 or len(words_latin) >= 3):
            return LANG_MIXED

        has_mr_char = any(ch in MARATHI_SPECIFIC_CHARS for ch in clean_text)
        mr_score = sum(1 for w in words_dev if w in MARATHI_WORDS)
        hi_score = sum(1 for w in words_dev if w in HINDI_WORDS)

        # Check if Hindi and Marathi are mixed together
        marathi_markers = {"आहे", "आहेत", "नाही", "नाहीत", "हवे", "हवा", "हवी", "पाहिजे", "नाव", "मला", "माझे", "माझा", "माझी", "नमस्कार"}
        hindi_markers = {"है", "हैं", "नहीं", "चाहिए", "नाम", "मुझे", "मेरा", "मेरी", "मेरे", "नमस्ते"}
        has_mr_marker = any(w in marathi_markers for w in words_dev)
        has_hi_marker = any(w in hindi_markers for w in words_dev)
        if has_mr_marker and has_hi_marker:
            return LANG_MIXED

        if has_mr_char or mr_score > hi_score:
            return LANG_MR
        elif hi_score > mr_score:
            return LANG_HI
        else:
            mr_tie = sum(1 for w in words_dev if w in marathi_markers)
            hi_tie = sum(1 for w in words_dev if w in hindi_markers)
            if mr_tie > hi_tie:
                return LANG_MR
            if hi_tie > mr_tie:
                return LANG_HI
            if "नमस्कार" in words_dev:
                return LANG_MR
            if "नमस्ते" in words_dev:
                return LANG_HI
            return LANG_HI

    # 2. Latin / Roman script checks
    # Check for code-mixing in Latin script (e.g. Hinglish / Marathish + English phrases)
    has_indic_roman = (mr_roman_hits > 0 or hi_roman_hits > 0)
    has_english_structure = (en_grammar_hits >= 1 or en_core_hits >= 2)

    # If both Romanized Indic and English structures are present -> mixed
    if has_indic_roman and has_english_structure:
        return LANG_MIXED

    # If both Hindi and Marathi Romanized markers are present -> mixed
    if mr_roman_hits > 0 and hi_roman_hits > 0:
        return LANG_MIXED

    # Pure Romanized Marathi
    if mr_roman_hits > 0 and not has_english_structure:
        return LANG_MR

    # Pure Romanized Hindi
    if hi_roman_hits > 0 and not has_english_structure:
        return LANG_HI

    # Pure English
    return LANG_EN


def get_response_language(detected_lang: str) -> str:
    """
    Authoritative response language resolution:
    - Pure English -> English response ('en')
    - Pure Hindi -> Hindi response ('hi')
    - Pure Marathi -> Marathi response ('mr')
    - Mixed Hindi/Marathi/English -> English response ('en')
    """
    norm = (detected_lang or "").strip().lower()
    if norm in (LANG_MIXED, "mixed"):
        return LANG_EN
    if norm in ("hi", "hi-in", "hindi"):
        return LANG_HI
    if norm in ("mr", "mr-in", "marathi"):
        return LANG_MR
    return LANG_EN


def detect_language(text: str) -> str:
    """
    Returns 'en', 'hi', or 'mr' for general backward compatibility.
    Mixed speech resolves to 'en'.
    """
    spoken = detect_spoken_language(text)
    return get_response_language(spoken)


detect_text_language = detect_spoken_language


def select_best_stt_transcript(mr_text: str, multi_text: str) -> tuple[str, str, str]:
    """
    Selects the authoritative transcript between Deepgram's standalone Marathi stream
    and Multilingual stream based on linguistic and script markers.
    Returns: (chosen_transcript, language_code, source_description)
    """
    mr_text = (mr_text or "").strip()
    multi_text = (multi_text or "").strip()

    if not mr_text and not multi_text:
        return "", LANG_EN, "none"
    if not mr_text:
        return multi_text, detect_language(multi_text), "multi_only"
    if not multi_text:
        return mr_text, detect_language(mr_text), "mr_only"

    mr_devanagari_words = set(re.findall(r"[\u0900-\u097F]+", mr_text))
    multi_devanagari_words = set(re.findall(r"[\u0900-\u097F]+", multi_text))

    mr_hits = sum(1 for w in mr_devanagari_words if w in MARATHI_WORDS)
    mr_has_char = any(ch in MARATHI_SPECIFIC_CHARS for ch in mr_text)

    multi_hi_hits = sum(1 for w in multi_devanagari_words if w in HINDI_WORDS)
    multi_mr_hits = sum(1 for w in multi_devanagari_words if w in MARATHI_WORDS)

    # 1. High confidence Marathi: specific Marathi characters (ळ, ऱ) OR strong Marathi lexical hits
    if mr_has_char or (mr_hits >= 2 and mr_hits > multi_hi_hits):
        return mr_text, LANG_MR, "Stream MR"

    # 2. High confidence Hindi: multi stream has clear Hindi markers and MR lacks distinctive Marathi
    if multi_hi_hits > 0 and multi_hi_hits >= mr_hits:
        return multi_text, LANG_HI, "Stream Multi (Hindi)"

    # 3. If mr_hits >= 1 and multi has 0 Hindi markers:
    if mr_hits >= 1 and multi_hi_hits == 0:
        return mr_text, LANG_MR, "Stream MR"

    # 4. English / Latin / Default:
    lang = detect_language(multi_text)
    return multi_text, lang, f"Stream Multi ({lang})"



def is_greeting(text: str) -> bool:
    """
    Determines if the user's input is solely a greeting or conversational pleasantry,
    WITHOUT containing a substantive question or task request.
    
    If the user starts with a greeting but follows with a question
    (e.g., 'Hello, what are your enterprise rates?'), returns False so the assistant
    answers the question directly without an unnecessary greeting prefix.
    """
    clean = (text or "").strip().lower()
    if not clean:
        return False

    # Normalize by replacing punctuation, slashes, brackets, symbols, and emojis with spaces
    normalized = re.sub(r"[^\w\s\u0900-\u097F]", " ", clean).strip()
    words = normalized.split()

    if not words:
        return False

    # Check for question indicators / substantive domain words
    question_indicators = {
        "what", "why", "how", "when", "where", "who", "which", "can", "could",
        "should", "would", "is", "are", "do", "does", "explain", "tell", "show",
        "describe", "pricing", "discount", "policy", "features", "tiers", "rate",
        "rates", "cost", "cibil", "loan", "roi", "salary", "eligibility",
        "क्या", "कैसे", "कहाँ", "कब", "कौन", "कितना", "बताएं", "बताओ", "जानकारी",
        "दर", "छूट", "दस्तावेज", "नियम", "ब्याज", "ऋण", "पात्रता",
        "काय", "कसे", "कशी", "कसा", "कुठे", "कधी", "कोण", "किती", "सांगा",
        "माहिती", "व्याज", "सवलत", "कागदपत्रे", "धोरण", "कर्ज"
    }

    if any(w in question_indicators for w in words):
        return False

    # Match against pure greeting tokens
    all_greetings = (
        GREETING_TOKENS_EN
        | GREETING_TOKENS_HI
        | GREETING_TOKENS_MR
        | ROMANIZED_HINDI
        | ROMANIZED_MARATHI
    )

    fillers = {
        "there", "copilot", "assistant", "ai", "team", "sir", "madam", "ji",
        "जी", "सर", "मॅडम", "मित्र", "मित्रा", "साहेब"
    }

    # Check exact normalized string
    if normalized in all_greetings:
        return True

    # If all tokens belong to greetings or conversational fillers
    if all(w in all_greetings or w in fillers for w in words):
        return True

    return False


def get_fallback_message(language: str) -> str:
    """Returns localized fallback phrase when knowledge base has insufficient info."""
    return FALLBACK_MESSAGES.get(language, FALLBACK_MESSAGES[LANG_EN])


def get_greeting_response(language: str) -> str:
    """Returns localized natural greeting response."""
    return GREETING_RESPONSES.get(language, GREETING_RESPONSES[LANG_EN])


ASSISTANT_NAME_RESPONSES = {
    LANG_EN: "I am VoiceCopilot, your AI sales assistant. I help you capture and qualify loan leads quickly.",
    LANG_HI: "मैं वॉयस कोपायलट हूँ, आपका एआई सेल्स असिस्टेंट। मैं ग्राहकों की जानकारी और लोन लीड्स दर्ज करने में आपकी मदद करता हूँ।",
    LANG_MR: "मी व्हॉइस कोपायलट आहे, आपला एआय सेल्स असिस्टंट. मी ग्राहकांची माहिती आणि लोन लीड्स नोंदवण्यात मदत करतो.",
}

ASSISTANT_IDENTITY_RESPONSES = {
    LANG_EN: "I am VoiceCopilot, your voice-powered CRM sales copilot. I assist you with capturing customer details, loan requirements, and qualifying leads.",
    LANG_HI: "मैं वॉयस कोपायलट हूँ, आपका वॉयस-सक्षम सीआरएम असिस्टेंट। मैं ग्राहकों के विवरण, लोन की जरूरतें और लीड्स दर्ज करने में आपकी सहायता करता हूँ।",
    LANG_MR: "मी व्हॉइस कोपायलट आहे, आपला वॉयस-सक्षम सीआरएम असिस्टंट. मी ग्राहकांचे तपशील आणि कर्जाच्या गरजा नोंदवून लीड्स व्यवस्थापित करण्यात मदत करतो.",
}

ASSISTANT_CAPABILITIES_RESPONSES = {
    LANG_EN: "I can help you collect and qualify leads by voice, record customer names, contact numbers, companies, loan amounts, and tenure directly into your CRM.",
    LANG_HI: "मैं आवाज़ के ज़रिए ग्राहकों के नाम, फोन नंबर, कंपनी, लोन की राशि और अवधि जैसे लीड विवरण सीआरएम में दर्ज और सत्यापित कर सकता हूँ।",
    LANG_MR: "मी आवाजाद्वारे ग्राहकांचे नाव, फोन नंबर, कंपनी, कर्जाची रक्कम आणि कालावधी यांसारखे तपशील थेट सीआरएममध्ये नोंदवून मदत करू शकतो.",
}

RE_ASSISTANT_NAME = re.compile(
    r"\b(?:what(?:'s|\s+is)\s+(?:your|ur)\s+name|tell\s+me\s+your\s+name|what\s+are\s+you\s+called|who\s+are\s+you\s+called|your\s+name\s+please)\b|"
    r"(?:आप(?:का)?\s*नाम\s*क्या|तुम्हारा\s*नाम\s*क्या|अपना\s*नाम\s*बता|नाम\s*क्या\s*है\s*आप)|"
    r"(?:तुम(?:चे|चं)\s*नाव\s*काय|तुझ(?:ं)?\s*नाव\s*काय|आपले\s*नाव\s*काय|नाव\s*काय\s*आहे\s*तुम)",
    re.IGNORECASE,
)

RE_ASSISTANT_IDENTITY = re.compile(
    r"\b(?:who\s+are\s+you|who\s+r\s+u|what\s+are\s+you|introduce\s+yourself|tell\s+me\s+about\s+yourself|who\s+is\s+speaking)\b|"
    r"(?:आप\s*कौन\s*(?:हैं|हो)|तुम\s*कौन\s*हो|अपना\s*परिचय\s*(?:दीजिए|दें|दो))|"
    r"(?:तुम्ही\s*कोण\s*आहात|तू\s*कोण\s*आहेस|आपण\s*कोण\s*आहात|आपली\s*ओळख\s*(?:करून\s*)?द्या)",
    re.IGNORECASE,
)

RE_ASSISTANT_CAPABILITIES = re.compile(
    r"\b(?:what\s+can\s+you\s+do|what\s+do\s+you\s+do|how\s+can\s+you\s+help(?:\s+me)?|what\s+are\s+your\s+features|what\s+is\s+your\s+purpose|what\s+help\s+can\s+you\s+provide|how\s+do\s+you\s+work)\b|"
    r"(?:आप\s*क्या\s*कर\s*सकते\s*हैं|आप\s*क्या\s*काम\s*करते\s*हैं|क्या\s*मदद\s*कर\s*सकते\s*हैं|आप\s*क्या\s*करते\s*हैं|आप\s*कैसे\s*मदद\s*करेंगे)|"
    r"(?:तुम्ही\s*काय\s*करू\s*शकता|आपण\s*काय\s*करू\s*शकता|काय\s*मदत\s*करू\s*शकता|तुम्ही\s*काय\s*काम\s*करता|तुम्ही\s*कशी\s*मदत\s*कराल)",
    re.IGNORECASE,
)


def is_assistant_query(text: str) -> bool:
    """Checks if the user's speech is a general assistant query rather than lead data."""
    if not text or not isinstance(text, str):
        return False
    clean = text.strip()
    return bool(
        RE_ASSISTANT_NAME.search(clean)
        or RE_ASSISTANT_IDENTITY.search(clean)
        or RE_ASSISTANT_CAPABILITIES.search(clean)
    )


def get_assistant_query_response(text: str, language: str = LANG_EN) -> Optional[str]:
    """
    Returns a natural assistant answer if the transcript is an assistant question
    (e.g., 'What is your name?', 'Who are you?', 'What can you do?').
    Returns None if the transcript is not an assistant query.
    """
    if not text or not isinstance(text, str):
        return None
    clean = text.strip()
    target_lang = language if language in (LANG_EN, LANG_HI, LANG_MR) else LANG_EN

    if RE_ASSISTANT_NAME.search(clean):
        return ASSISTANT_NAME_RESPONSES.get(target_lang, ASSISTANT_NAME_RESPONSES[LANG_EN])
    if RE_ASSISTANT_IDENTITY.search(clean):
        return ASSISTANT_IDENTITY_RESPONSES.get(target_lang, ASSISTANT_IDENTITY_RESPONSES[LANG_EN])
    if RE_ASSISTANT_CAPABILITIES.search(clean):
        return ASSISTANT_CAPABILITIES_RESPONSES.get(target_lang, ASSISTANT_CAPABILITIES_RESPONSES[LANG_EN])

    return None


def get_empty_kb_greeting(language: str) -> str:
    """Returns localized greeting when Pinecone knowledge base is empty."""
    return EMPTY_KB_GREETINGS.get(language, EMPTY_KB_GREETINGS[LANG_EN])


def devanagari_to_phonetic(text: str) -> str:
    """
    Converts Devanagari Hindi / Marathi script into natural, readable phonetic Latin text
    for clear, native speech synthesis with Deepgram Text-to-Speech (TTS) models.
    Preserves English words, numbers, and punctuation intact while adding natural cadence pauses.
    """
    if not text:
        return text

    vowels = {
        "\u0905": "a", "\u0906": "aa", "\u0907": "i", "\u0908": "ee",
        "\u0909": "u", "\u090A": "oo", "\u090B": "ri", "\u090E": "e",
        "\u090F": "e", "\u0910": "ai", "\u0911": "o", "\u0912": "o",
        "\u0913": "o", "\u0914": "au", "\u0972": "e"
    }

    matras = {
        "\u093E": "aa", "\u093F": "i", "\u0940": "ee", "\u0941": "u",
        "\u0942": "oo", "\u0943": "ri", "\u0944": "ri", "\u0946": "e",
        "\u0947": "e", "\u0948": "ai", "\u0949": "o", "\u094A": "o",
        "\u094B": "o", "\u094C": "au", "\u0945": "e"
    }

    consonants = {
        "\u0915": "k", "\u0916": "kh", "\u0917": "g", "\u0918": "gh", "\u0919": "ng",
        "\u091A": "ch", "\u091B": "chh", "\u091C": "j", "\u091D": "jh", "\u091E": "ny",
        "\u091F": "t", "\u0920": "th", "\u0921": "d", "\u0922": "dh", "\u0923": "n",
        "\u0924": "t", "\u0925": "th", "\u0926": "d", "\u0927": "dh", "\u0928": "n",
        "\u092A": "p", "\u092B": "ph", "\u092C": "b", "\u092D": "bh", "\u092E": "m",
        "\u092F": "y", "\u0930": "r", "\u0931": "r", "\u0932": "l", "\u0933": "l",
        "\u0934": "l", "\u0935": "v", "\u0936": "sh", "\u0937": "sh", "\u0938": "s",
        "\u0939": "h",
        # Nukta consonants
        "\u0958": "q", "\u0959": "kh", "\u095A": "g", "\u095B": "z",
        "\u095C": "d", "\u095D": "dh", "\u095E": "f", "\u095F": "y"
    }

    halant = "\u094D"
    anusvara = "\u0902"
    visarga = "\u0903"
    chandrabindu = "\u0901"
    nukta = "\u093C"

    # High-frequency conversational, CRM, financial, and sales lexicon mappings
    # tuned for natural native pronunciation without English-accented distortion
    special_words = {
        # Greetings & Politeness
        "नमस्ते": "Namaste",
        "नमस्कार": "Namaskar",
        "प्रणाम": "Pranam",
        "सुप्रभात": "Suprabhat",
        "धन्यवाद": "Dhanyawad",
        "कृपया": "Kripya",
        "स्वागत": "Swagat",
        
        # Common Hindi words & verbs
        "है": "hai",
        "हैं": "hain",
        "नहीं": "nahin",
        "हाँ": "haan",
        "क्या": "kya",
        "कैसे": "kaise",
        "कैसा": "kaisa",
        "कैसी": "kaisi",
        "कहाँ": "kahaan",
        "कब": "kab",
        "कौन": "kaun",
        "कितना": "kitna",
        "कितनी": "kitni",
        "कितने": "kitne",
        "मैं": "main",
        "मुझे": "mujhe",
        "हमें": "humein",
        "हम": "hum",
        "आप": "aap",
        "आपका": "aapka",
        "आपकी": "aapki",
        "आपके": "aapke",
        "अपना": "apna",
        "अपनी": "apni",
        "अपने": "apne",
        "में": "mein",
        "से": "se",
        "को": "ko",
        "का": "ka",
        "की": "ki",
        "के": "ke",
        "और": "aur",
        "तथा": "tatha",
        "या": "ya",
        "लेकिन": "lekin",
        "परंतु": "parantu",
        "क्योंकि": "kyunki",
        "ताकि": "taaki",
        "इसलिए": "isliye",
        "चाहिए": "chahiye",
        "चाहते": "chaahte",
        "चाहती": "chaahti",
        "बताएं": "bataayein",
        "बताओ": "batao",
        "दीजिए": "deejiye",
        "सकता": "sakta",
        "सकती": "sakti",
        "सकते": "sakte",
        "सकूँ": "sakoon",
        "सकूँगा": "sakoonga",
        "सकूँगी": "sakoongi",
        "सकेंगे": "sakenge",
        "होगा": "hoga",
        "होगी": "hogi",
        "होंगे": "honge",
        "दिए": "diye",
        "गए": "gaye",
        "गई": "gayi",
        "गया": "gaya",
        "लिए": "liye",
        "किया": "kiya",
        "किए": "kiye",
        "सफलतापूर्वक": "safaltaapoorvak",
        "सत्यापित": "satyapit",
        "सहेज": "sahej",
        "अपडेट": "update",
        "विवरण": "vivaran",
        "जानकारी": "jaankari",
        "सहायता": "sahayata",
        "जोड़ना": "jodna",
        "जोड़ें": "jodein",
        "अन्य": "anya",
        "कोई": "koi",
        "कुछ": "kuchh",
        "खेद": "khed",
        "प्रश्न": "prashna",
        "उत्तर": "uttar",
        "दस्तावेज़": "dastaavez",
        "दस्तावेजों": "dastaavezon",
        "दस्तावेज": "dastaavez",
        "उपलब्ध": "upalabdha",
        "पर्याप्त": "paryapt",
        "ग्राहक": "graahak",
        "कंपनी": "company",
        "नाम": "naam",
        "पद": "pad",
        "संपर्क": "sampark",
        "फोन": "phone",
        "ईमेल": "email",
        "ऋण": "rin",
        "ब्याज": "byaaj",
        "ब्याजदर": "byaajdar",
        "राशि": "raashi",
        "अवधि": "avadhee",
        "महीने": "maheene",
        "महीनों": "maheenon",
        "शर्तें": "shartein",
        "नियम": "niyam",
        "पात्रता": "paatrata",
        "योजना": "yojana",
        "छूट": "chhoot",
        "सीआरएम": "CRM",
        "डेटाबेस": "database",
        "रिकॉर्ड": "record",

        # Common Marathi words & verbs
        "मी": "mee",
        "मला": "mala",
        "आम्ही": "aamhi",
        "आपण": "aapan",
        "तुम्ही": "tumhi",
        "तुमचे": "tumche",
        "तुमची": "tumchi",
        "तुमच्या": "tumchya",
        "आपले": "aaple",
        "आपली": "aapli",
        "आपल्या": "aaplya",
        "आहे": "aahe",
        "आहेत": "aahet",
        "नाही": "naahi",
        "नाहीत": "naahit",
        "होय": "hoy",
        "काय": "kaay",
        "कसे": "kase",
        "कशी": "kashee",
        "कसा": "kasa",
        "कुठे": "kuthe",
        "कधी": "kadhee",
        "कोण": "kon",
        "किती": "kiti",
        "आणि": "aani",
        "किंवा": "kinva",
        "तसेच": "tasech",
        "म्हणून": "mhanun",
        "जेणेकरून": "jenekarun",
        "सांगा": "saangaa",
        "द्या": "dya",
        "पाहिजे": "pahije",
        "शकेन": "shakayn",
        "शकतो": "shakto",
        "शकते": "shakte",
        "शकतात": "shaktaat",
        "करा": "karaa",
        "करावे": "karaave",
        "झाले": "zhaale",
        "झाला": "zhaala",
        "झाली": "zhaalee",
        "केले": "kele",
        "केला": "kela",
        "केली": "keli",
        "गेले": "gele",
        "गेला": "gela",
        "गेली": "geli",
        "दिलेल्या": "dilelyaa",
        "दस्तऐवजांमध्ये": "dasta-aivajaan-madhye",
        "दस्तऐवज": "dasta-aivaj",
        "प्रश्नाचे": "prashnaache",
        "देण्यासाठी": "denyaasaathee",
        "पुरेशी": "pureshee",
        "माहिती": "maahiti",
        "मदत": "madat",
        "यशस्वीरित्या": "yashasveereetya",
        "यशस्वीरीत्या": "yashasveereetya",
        "तपशील": "tapsheel",
        "नोंदवले": "nondavle",
        "नोंदवू": "nondavoo",
        "इच्छिता": "ichhita",
        "इतर": "itar",
        "काही": "kaahi",
        "आणखी": "aankhee",
        "जोडू": "jodoo",
        "ग्राहकाचे": "graahakaache",
        "पदवी": "padavee",
        "कागदपत्रे": "kaagadpatre",
        "कर्ज": "karz",
        "व्याजदर": "vyaajdar",
        "रक्कम": "rakkam",
        "कालावधी": "kaalaavadhee",
        "महिने": "mahine",
        "निकष": "nikash",
        "अटी": "atee",
        "सवलत": "savlat",
        "सवलती": "savlati",
        "धोरण": "dhoran",
        "दिलगीर": "dilgeer",
        "मध्ये": "madhye",
        "साठी": "saathee",
        "बद्दल": "baddal",
        "सर्व": "sarva",
        "नवीन": "naveen",
        "खूप": "khoop",
    }

    # Pre-process text to standardize punctuation pauses
    text_processed = text.replace("।", ".").replace("॥", ".")

    tokens = text_processed.split()
    converted_tokens = []
    punctuation = ".,!?:;()[]\"'{}<>/\\|`~*-_=+"

    for token in tokens:
        clean_tok = token.strip(punctuation)
        prefix = token[:token.find(clean_tok)] if clean_tok and token.find(clean_tok) > 0 else ""
        suffix = token[token.find(clean_tok) + len(clean_tok):] if clean_tok else ""

        if clean_tok in special_words:
            converted_tokens.append(f"{prefix}{special_words[clean_tok]}{suffix}")
            continue

        # If token has no Devanagari chars, keep as is (e.g. English, numbers, symbols)
        if not any(0x0900 <= ord(ch) <= 0x097F for ch in token):
            converted_tokens.append(token)
            continue

        res = []
        i = 0
        n = len(token)
        while i < n:
            c = token[i]
            if c in vowels:
                res.append(vowels[c])
            elif c in consonants:
                base = consonants[c]
                if i + 1 < n and token[i+1] == nukta:
                    i += 1
                    if base == "j": base = "z"
                    elif base == "ph": base = "f"
                if i + 1 < n:
                    nxt = token[i+1]
                    if nxt == halant:
                        res.append(base)
                        i += 1
                    elif nxt in matras:
                        res.append(base + matras[nxt])
                        i += 1
                    elif nxt in consonants or nxt in vowels:
                        res.append(base + "a")
                    else:
                        res.append(base)
                else:
                    res.append(base)
            elif c in matras:
                res.append(matras[c])
            elif c in (anusvara, chandrabindu):
                res.append("n")
            elif c == visarga:
                res.append("h")
            elif c == "।":
                res.append(".")
            else:
                res.append(c)
            i += 1
        converted_tokens.append("".join(res))

    result_str = " ".join(converted_tokens)
    # Ensure natural spacing around commas for micro-pauses
    result_str = re.sub(r"\s*,\s*", ", ", result_str)
    result_str = re.sub(r"\s*\.\s*", ". ", result_str)
    return result_str.strip()


# ============================================================================
# CROSS-LINGUAL QUERY TRANSLATION FOR RAG RETRIEVAL
# ============================================================================
# Converts Marathi and Hindi sales queries into high-precision English search phrases
# matching English PDF playbook vectors in Pinecone with 0ms latency and 100% reliability.
# ============================================================================

INDIC_QUERY_TRANSLATION_MAP = [
    # Institutions
    (r'(?i)\b(एचडीएफसी\s+बँक[ा-ीa-z]*|एचडीएफसी\s+बैंक|hdfc\s+bank|hdfc)\b', 'HDFC Bank'),
    # Loan types
    (r'(?i)\b(वैयक्तिक\s+कर्ज[ा-ीa-z]*|पर्सनल\s+लोन[ा-ीa-z]*|व्यक्तिगत\s+ऋण[ा-ीa-z]*)\b', 'personal loan'),
    (r'(?i)\b(गृह\s+कर्ज[ा-ीa-z]*|होम\s+लोन[ा-ीa-z]*|घर\s+कर्ज[ा-ीa-z]*)\b', 'home loan'),
    (r'(?i)\b(कार\s+लोन[ा-ीa-z]*|वाहन\s+कर्ज[ा-ीa-z]*|गाडी\s+कर्ज[ा-ीa-z]*)\b', 'car loan'),
    (r'(?i)\b(व्यवसाय\s+कर्ज[ा-ीa-z]*|बिजनेस\s+लोन[ा-ीa-z]*|व्यापार\s+कर्ज[ा-ीa-z]*)\b', 'business loan'),
    (r'(?i)\b(बॅलन्स\s+ट्रान्सफर[ा-ीa-z]*|बैलेंस\s+ट्रांसफर[ा-ीa-z]*)\b', 'balance transfer BT offer'),
    (r'(?i)\b(क्रेडिट\s+कार्ड[ा-ीa-z]*)\b', 'credit card'),
    (r'(?i)\b(कर्ज[ा-ीa-z]*|ऋण[ा-ीa-z]*|लोन[ा-ीa-z]*)\b', 'loan'),
    # CIBIL / Credit Bureau
    (r'(?i)\b(सिबिल|cibil)\s*(स्कोर|score)?\b', 'CIBIL credit score bureau requirements slab'),
    (r'(?i)\b(क्रेडिट\s+स्कोर)\b', 'credit score CIBIL bureau'),
    (r'(?i)\b(किमान|न्यूनतम|कम\s+से\s+कम|कमीत\s+कमी)\b', 'minimum threshold cutoff'),
    (r'(?i)\b(कमाल|अधिकतम|जास्तीत\s+जास्त|अधिक)\b', 'maximum limit cap'),
    (r'(?i)\b(मर्यादा|सीमा|कटऑफ)\b', 'limit threshold cutoff'),
    (r'(?i)\b(स्लॅब|स्लैब)\b', 'slab tier'),
    # Documents / Verification Checklist
    (r'(?i)\b(कागदपत्रे|कागदपत्र[ा-ीa-z]*|दस्तावेज|दस्तावेज़|कागजात)\b', 'documents documentation checklist required proof'),
    (r'(?i)\b(ओळख\s+पुरावा|पहचान\s+प्रमाण)\b', 'identity proof KYC PAN Aadhaar'),
    (r'(?i)\b(पत्ता\s+पुरावा|पते\s+का\s+प्रमाण)\b', 'address proof utility bill'),
    (r'(?i)\b(उत्पन्न\s+पुरावा|आय\s+प्रमाण)\b', 'income proof salary slip bank statement'),
    (r'(?i)\b(बँक\s+स्टेटमेंट|खाते\s+विवरण)\b', 'bank statement banking details'),
    (r'(?i)\b(आवश्यक|लागणारे|लागतील|चाहिए|गरज|अनिवार्य)\b', 'required mandatory necessary'),
    # Interest Rates / ROI / Pricing
    (r'(?i)\b(व्याजदर[ा-ीa-z]*|ब्याज\s*दर[ा-ीa-z]*|व्याज|ब्याज|दर)\b', 'interest rate rack ROI pricing percentage rate card'),
    (r'(?i)\b(रॅक\s+दर|रैक\s+रेट)\b', 'rack rate card pricing'),
    (r'(?i)\b(सवलत[ा-ीa-z]*|सूट|ऑफर्स|ऑफर)\b', 'discount concession exclusive offer'),
    (r'(?i)\b(ईएमआय|emi|हप्ता|किस्त)\b', 'EMI monthly installment calculation'),
    # Eligibility & Criteria
    (r'(?i)\b(पात्रता|पात्र|मानदंड|निकष|नियम|अटी|शर्तें)\b', 'eligibility criteria rules policy norms'),
    (r'(?i)\b(वय|उम्र|आयु)\b', 'age limit criteria minimum age maximum age'),
    (r'(?i)\b(पगार|वेतन|उत्पन्न|आय|कमाई|मासिक\s+आय|मासिक\s+उत्पन्न)\b', 'salary income monthly net salary criteria'),
    (r'(?i)\b(नोकरी|व्यवसाय|रोजगार|पगारावर|वेतनभोगी)\b', 'salaried self employed employment corporate category'),
    (r'(?i)\b(कंपनी|संस्था|श्रेणी|कॅटेगरी|वर्ग)\b', 'company corporate category CAT Super A CAT A CAT B'),
    # Tenure / Repayment
    (r'(?i)\b(कालावधी|अवधि|मुदत|वर्ष|वर्षे|साल|महिने|माह)\b', 'tenure duration repayment months years'),
    (r'(?i)\b(किमान\s+कर्ज|कमाल\s+कर्ज|रक्कम|राशि)\b', 'loan amount limit minimum maximum'),
    # Fees & Charges
    (r'(?i)\b(प्रोसेसिंग\s+फी|प्रोसेसिंग\s+शुल्क|शुल्क|फीस|फी|चार्जेस)\b', 'processing fee charges premium charges'),
    (r'(?i)\b(फोरक्लोजर|प्रीपेमेंट|दंड)\b', 'foreclosure prepayment penalty charges'),
    # Interrogatives & Postpositions (stripped to focus on semantic content)
    (r'(?i)\b(काय|कोणते|कोणती|कोणत्या|किती|कसा|कशी|कसे|मिळेल|असेल)\b', ''),
    (r'(?i)\b(क्या|कौनसा|कौनसी|कौनसे|कितना|कितनी|कितने|होगा|hogi|होना|है|हैं|आहे|आहेत)\b', ''),
    (r'(?i)\b(साठी|बद्दल|मध्ये|कडून|च्या|चे|ची|ला|ना|वर)\b', ''),
    (r'(?i)\b(के\s+लिए|के\s+बारे\s+में|में|से|का|की|के|पर|को)\b', ''),
]


def translate_indic_query_to_english(query: str, language: Optional[str] = None) -> str:
    """
    Translates/expands Hindi and Marathi queries into focused English search keywords
    for Pinecone vector search and hybrid lexical matching against English PDF playbooks.
    Preserves exact numbers, English product names, and injects relevant financial keywords.
    """
    clean = (query or "").strip()
    if not clean:
        return ""

    has_devanagari = count_devanagari_chars(clean) >= 2 or (language and language.lower().startswith(("hi", "mr")))
    if not has_devanagari:
        return clean

    translated = clean
    for pattern, replacement in INDIC_QUERY_TRANSLATION_MAP:
        translated = re.sub(pattern, f" {replacement} ", translated)

    # Clean out leftover Devanagari characters while retaining English, digits, and punctuation
    translated = re.sub(r"[\u0900-\u097F]+", " ", translated)
    translated = re.sub(r"\s+", " ", translated).strip()

    # If all tokens were removed (edge case), fallback to transliteration
    if not translated or len(translated) < 3:
        translated = devanagari_to_phonetic(clean)

    return translated
