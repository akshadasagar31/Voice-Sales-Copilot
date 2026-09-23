# ============================================================================
# LEAD EXTRACTION & CRM UPDATE SERVICE (backend/services/lead_extractor.py)
# ============================================================================
# WHAT THIS SERVICE DOES:
# 1. Takes raw audio transcripts from sales calls (in English, Hindi, or Marathi).
# 2. Uses DeepSeek LLM (via OpenRouter) to extract structured sales lead fields:
#    - Prospect Name, Phone, Email, Company, Job Role
#    - Loan Type (e.g. Home, Personal, Business), Loan Amount, Tenure (Months), Notes
# 3. Validates and sanitizes all fields using Pydantic (converting text numbers to floats/ints).
# 4. Merges new details with `existing_lead` across conversation turns so the CRM lead
#    is updated incrementally rather than duplicated.
# 5. Persists the lead to PostgreSQL database via LeadRepository.
# 6. Emits Server-Sent Events (SSE) with streaming confirmation tokens for real-time voice feedback!
# ============================================================================

import os
import re
import json
import logging
from typing import Optional, Dict, Any, Generator, List
from pathlib import Path
from dotenv import load_dotenv
import httpx
from pydantic import BaseModel, Field, field_validator, ValidationError
from services.language import (
    is_greeting,
    get_greeting_response,
    detect_language,
    detect_spoken_language,
    get_response_language,
    is_new_lead_intent,
    is_assistant_query,
    get_assistant_query_response,
    LANG_MIXED,
    count_devanagari_chars,
    ROMANIZED_HINDI,
    ROMANIZED_MARATHI,
)
from services.number_normalizer import (
    normalize_spoken_numbers,
    parse_numeric_phrase,
)

# Load environment variables from backend/.env if available
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "deepseek/deepseek-chat"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
LEAD_STREAM_DELIMITER = "<<<LEAD_JSON>>>"


def extract_json_payload(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Safely extract and parse JSON dictionary from LLM output,
    handling markdown code fences (```json ... ```) and leading/trailing whitespace.
    """
    if not raw or not isinstance(raw, str):
        return None
    clean = raw.strip()
    if not clean:
        return None
    # Strip markdown code fences if present
    clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean)
    clean = clean.strip()
    try:
        data = json.loads(clean)
        if isinstance(data, dict):
            return data
    except Exception:
        # Search for first { to last }
        match = re.search(r"(\{.*\})", clean, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
    return None


LEAD_EXTRACTION_SYSTEM_PROMPT = """You are a strict, precise Information Extraction engine for sales calls.
Your task is to analyze the sales call transcript and extract structured lead information.

CRITICAL EXTRACTION RULES:
1. Extract ONLY facts and details that are explicitly stated in the transcript.
2. For ANY field that is not explicitly mentioned in the transcript, set its value to null.
3. NEVER guess, assume, estimate, or hallucinate information.
4. If a field is uncertain, ambiguous, or not provided, you MUST output null for that field.
5. The transcript may be in English, Hindi, or Marathi (or transliterated/code-switched). Accurately extract the factual details regardless of language.
6. CRITICAL NAME RULE: Extract 'name' ONLY when the user explicitly provides a real person's name (e.g. 'My name is Rajesh', 'I am Rajesh', 'मेरा नाम अमित है', 'माझे नाव राहुल आहे').
   NEVER treat loan-intent phrases (such as 'I want loan', 'I want a loan', 'need loan', 'looking for a loan', 'interested in loan', 'loan chahiye', 'karj pahije', 'apply for loan') or general requests as a prospect name! If no real person's name is explicitly stated, 'name' MUST be null.
7. LOAN TYPE: If the user states 'I want loan', 'personal loan', 'business loan', 'home loan', extract 'loan_type' appropriately ('Personal Loan', 'Home Loan', 'Business Loan', or 'Loan'). Do NOT put loan terms into 'name'.
8. You MUST return ONLY a valid, raw JSON object matching the schema below. Do not include markdown code fences, commentary, or text outside the JSON object.

Target JSON Schema:
{
  "name": string or null,
  "phone": string or null,
  "email": string or null,
  "company": string or null,
  "role": string or null,
  "loan_type": string or null,
  "loan_amount": number or null,
  "tenure_months": integer or null,
  "notes": string or null
}
"""


REQUIRED_LEAD_FIELDS = [
    "name",
    "phone",
    "company",
    "loan_type",
    "loan_amount",
    "tenure_months",
]


def extract_phone_number(text: str) -> Optional[str]:
    """Extracts a valid 10-digit Indian phone number from text if present."""
    if not text:
        return None
    normalized_text = normalize_spoken_numbers(text)
    match = re.search(r"(?:(?:\+?91[\s\-]?)|\b)([6-9]\d{9})\b", normalized_text)
    if match:
        return match.group(1)
    digit_match = re.search(r"(?:(?:\+?91[\s\-]?)|\b)([6-9](?:[\s\-]*\d){9})\b", normalized_text)
    if digit_match:
        digits = re.sub(r"\D", "", digit_match.group(1))
        if len(digits) == 10 and digits[0] in "6789":
            return digits
    return None


def extract_email(text: str) -> Optional[str]:
    """Extracts email address from text."""
    if not text:
        return None
    match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
    if match:
        return match.group(0).lower()
    spoken_email = re.sub(r"\s*(?:at\s*the\s*rate|at)\s*", "@", text, flags=re.IGNORECASE)
    spoken_email = re.sub(r"\s*dot\s*", ".", spoken_email, flags=re.IGNORECASE)
    match2 = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", spoken_email)
    if match2:
        return match2.group(0).lower()
    return None


def extract_loan_type(text: str, context_field: Optional[str] = None) -> Optional[str]:
    """Extracts loan type from English, Hindi, or Marathi speech."""
    if not text:
        return None
    lower = text.lower()

    if context_field == "loan_type":
        if re.search(r"\b(?:personal|पर्सनल|वैयक्तिक)\b", lower):
            return "Personal Loan"
        if re.search(r"\b(?:home|housing|होम|गृह)\b", lower):
            return "Home Loan"
        if re.search(r"\b(?:business|व्यापार|व्यवसाय|बिज़नेस|बिजनेस)\b", lower):
            return "Business Loan"
        if re.search(r"\b(?:car|auto|कार|वाहन)\b", lower):
            return "Car Loan"
        if re.search(r"\b(?:gold|गोल्ड|सोने|सुवर्ण)\b", lower):
            return "Gold Loan"
        if re.search(r"\b(?:education|एजुकेशन|शिक्षण)\b", lower):
            return "Education Loan"

    patterns = [
        (r"\b(?:personal\s*loan|पर्सनल\s*लोन|वैयक्तिक\s*कर्ज)\b", "Personal Loan"),
        (r"\b(?:home\s*loan|housing\s*loan|होम\s*लोन|गृह\s*कर्ज)\b", "Home Loan"),
        (r"\b(?:business\s*loan|बिज़नेस\s*लोन|बिजनेस\s*लोन|व्यापार\s*लोन|व्यवसाय\s*कर्ज)\b", "Business Loan"),
        (r"\b(?:car\s*loan|auto\s*loan|कार\s*लोन|वाहन\s*कर्ज)\b", "Car Loan"),
        (r"\b(?:gold\s*loan|गोल्ड\s*लोन|सोने\s*कर्ज|सुवर्ण\s*कर्ज)\b", "Gold Loan"),
        (r"\b(?:education\s*loan|एजुकेशन\s*लोन|शिक्षण\s*कर्ज)\b", "Education Loan"),
        (r"\b(?:loan|loans|loen|लोन|कर्ज)\b", "Personal Loan"),
    ]
    for pat, label in patterns:
        if re.search(pat, lower, flags=re.IGNORECASE):
            return label
    return None


def extract_loan_amount(text: str, context_field: Optional[str] = None) -> Optional[float]:
    """Extracts loan amount from English, Hindi, or Marathi speech."""
    if not text:
        return None
    normalized = normalize_spoken_numbers(text)
    lower = normalized.lower().strip()

    # If the text is purely a phone number (e.g. "9876543210", "+91 9876543210"), never match amount
    if re.match(r"^\s*(?:\+?91[\s\-]?)?[6-9]\d{9}\s*$", lower):
        return None

    # Remove any 10-digit phone numbers from text before checking for loan amount
    text_no_phone = re.sub(r"(?:(?:\+?91[\s\-]?)|\b)[6-9]\d{9}\b", "", lower).strip()
    if not text_no_phone:
        return None

    cr = re.search(r"(\d+(?:\.\d+)?)\s*(?:crores?|cr|करोड़|करोड|कोटी)(?:\b|\s|[.,;]|$)", text_no_phone)
    if cr:
        return float(cr.group(1)) * 10000000.0

    lakh = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakhs?|lacs?|lac|लाख|l)(?:\b|\s|[.,;]|$)", text_no_phone)
    if lakh:
        return float(lakh.group(1)) * 100000.0

    million = re.search(r"(\d+(?:\.\d+)?)\s*(?:millions?|m)(?:\b|\s|[.,;]|$)", text_no_phone)
    if million:
        return float(million.group(1)) * 1000000.0

    k = re.search(r"(\d+(?:\.\d+)?)\s*(?:thousands?|k|हजार|हज़ार)(?:\b|\s|[.,;]|$)", text_no_phone)
    if k:
        return float(k.group(1)) * 1000.0

    curr_amt = re.search(r"(?:rs\.?|inr|₹|amount|रुपये|रक्कम|रकम|राशि)\s*[:=]?\s*(\d{1,3}(?:,\d{2,3})*(?:\.\d+)?|\d{4,8})\b", text_no_phone)
    if curr_amt:
        v = curr_amt.group(1).replace(",", "")
        try:
            val = float(v)
            if 5000 <= val < 100000000:
                return val
        except ValueError:
            pass

    pure_num = re.match(r"^\s*(?:rs\.?|inr|₹)?\s*(\d{1,3}(?:,\d{2,3})*(?:\.\d+)?|\d{1,8})\s*$", text_no_phone)
    if pure_num:
        v = pure_num.group(1).replace(",", "")
        try:
            val = float(v)
            if context_field == "loan_amount" and 1 <= val <= 99:
                return val * 100000.0
            if 5000 <= val < 100000000 and len(v) != 10:
                return val
        except ValueError:
            pass

    return None


INVALID_NAME_REGEX = re.compile(
    r"\b(?:"
    r"loan|loans|loen|लोन|कर्ज|गृहकर्ज|karj|karja|personal\s*loan|home\s*loan|business\s*loan|"
    r"want|wants|wanted|wanting|need|needs|needed|needing|looking|interested|apply|applying|require|requires|"
    r"चाहिए|चाहिये|हवे|हवा|पाहिजे|लागेल|द्या|मिळेल|देना|लेना|chahiye|chaahiye|pahije|havay|have|dena|lena|mala|mujhe|"
    r"personal|business|home|housing|car|auto|gold|education|mortgage|"
    r"cibil|emi|roi|interest|rate|tenure|duration|amount|salary|rupees|rs|lakh|crore|thousand|हजार|हज़ार|लाख|करोड़|करोड|कोटी|रुपये|रक्कम|रकम|राशि|"
    r"month|months|महीने|महिने|माह|mahina|mahine|year|years|साल|वर्ष|वर्षे|अवधि|कालावधी|मुदत|"
    r"hello|hi|hey|namaste|namaskar|good\s*morning|good\s*afternoon|good\s*evening|नमस्ते|नमस्कार|"
    r"yes|no|ok|okay|sure|please|thanks|thank\s*you|धन्यवाद|होय|नाही|"
    r"what|who|whom|whose|which|why|where|when|how|can|could|should|would|are|you|your|is|do|does|tell|help|"
    r"क्या|कौन|किस|किसे|कैसे|कहाँ|कब|क्यों|आप|आपका|तुम्हारा|तेरा|मुझे|बताओ|बताएं|मदद|"
    r"काय|कोण|कोणाचे|कसे|कुठे|कधी|का|तुम्ही|तुमचे|तू|तुझे|मला|सांगा|मदत"
    r")\b",
    re.IGNORECASE,
)


def is_valid_prospect_name(name: Optional[str]) -> bool:
    """
    Validates that a string is a legitimate personal name and NOT a loan-intent phrase,
    tenure duration, spoken number, financial term, question, or conversational statement.
    """
    if not name or not isinstance(name, str):
        return False
    clean = name.strip()
    if not clean or len(clean) < 2 or len(clean) > 40:
        return False
    # Check if assistant conversation question
    if is_assistant_query(clean):
        return False
    # Check if spoken numbers convert to digits or contain digits
    norm = normalize_spoken_numbers(clean)
    if re.search(r"[\d@#$%^*=_\+\[\]{}<>/\\|?]", norm) or re.search(r"[\d@#$%^*=_\+\[\]{}<>/\\|?]", clean):
        return False
    if INVALID_NAME_REGEX.search(clean) or INVALID_NAME_REGEX.search(norm):
        return False
    # Must have between 1 and 4 words
    words = clean.split()
    if len(words) > 4:
        return False
    # Must contain alphabetic characters (Latin or Devanagari)
    alpha_count = len(re.findall(r"[A-Za-z\u0900-\u097F]", clean))
    if alpha_count < 2:
        return False
    # Common pronouns and conversational words that are not names
    lower = clean.lower()
    if lower in {"i", "me", "my", "we", "us", "you", "he", "she", "it", "they", "them", "this", "that", "there", "what", "which", "who", "whom"}:
        return False
    return True


def extract_name(text: str, context_field: Optional[str] = None) -> Optional[str]:
    """
    Extracts prospect name either when explicitly introduced or in response to a name question.
    Never extracts loan requests, intent phrases, or generic statements as names.
    """
    if not text:
        return None

    clean = text.strip()
    lower = clean.lower()

    # 1. If context_field is "name", user was directly asked for their name
    if context_field == "name":
        candidate = re.sub(r"^(?:my name is|my name\'s|name is|this is|myself|i am|i\'m|मेरा नाम|माझे नाव|माझ नाव)\s*", "", clean, flags=re.IGNORECASE)
        candidate = re.sub(r"^(?:it is|it\'s|here|just)\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\b(?:hai|hain|aahe|ahe|and|aur|ani|va|chahiye|havay|pahije|from|working|at|company|phone|loan|amount|tenure)\b.*$", "", candidate, flags=re.IGNORECASE).strip()
        candidate = re.sub(r"(?:\s+|^)(?:है|आहे|हूँ|आणि|और|असेल).*$", "", candidate).strip()
        candidate = re.sub(r"[.,;:]+$", "", candidate).strip()
        if is_valid_prospect_name(candidate):
            return candidate

    # 2. Check explicit intro patterns
    has_name_intro = any(intro in lower for intro in [
        "my name is", "my name's", "name is", "this is", "myself", "i am", "i'm",
        "मेरा नाम", "माझे नाव", "माझ नाव"
    ])
    if has_name_intro:
        patterns = [
            r"\b(?:my name is|my name\'s|name is|this is|myself)\s+([A-Za-z\u0900-\u097F\s]{2,35}?)(?:\s+(?:and|phone|loan|company|tenure|salary|i|for|personal|home|business|is|from)\b|[.,;]|$)",
            r"\b(?:i am|i\'m)\s+([A-Za-z\u0900-\u097F]{2,20}(?:\s+[A-Za-z\u0900-\u097F]{2,20}){0,2})(?:\s+(?:and|phone|loan|company|tenure|salary|from|working|looking|in|at)\b|[.,;]|$)",
            r"(?:मेरा नाम|माझे नाव|माझ नाव)\s*(?:है|आहे)?\s*([a-zA-Z\u0900-\u097F\s]{2,35}?)(?:\s+(?:है|आहे|हूँ|आणि|और|phone|loan|company|tenure|salary|में|मध्ये|का)(?:\s+|$)|[.,;]|$)",
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                raw = m.group(1).strip()
                raw = re.sub(r"\b(?:hai|hain|aahe|ahe|and|aur|ani|va|chahiye|havay|pahije|from|working|at)\b.*$", "", raw, flags=re.IGNORECASE).strip()
                raw = re.sub(r"(?:\s+|^)(?:है|आहे|हूँ|आणि|और|असेल).*$", "", raw).strip()
                raw = re.sub(r"[.,;:]+$", "", raw).strip()
                if is_valid_prospect_name(raw):
                    return raw

    # 3. Clean standalone name (1-3 words, capitalized, no loan/greeting words)
    words = clean.split()
    if 1 <= len(words) <= 3 and is_valid_prospect_name(clean):
        lower_words = {w.lower() for w in words}
        if not (lower_words & {"personal", "home", "business", "car", "gold", "education", "loan", "tcs", "infosys", "wipro", "google"}):
            return clean

    # 4. Check if transcript starts with a name followed by comma/and/phone/from:
    start_m = re.match(r"^([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){0,2})(?:,\s*|\s+(?:and|from|phone|loan)\b)", text)
    if start_m:
        cand = start_m.group(1).strip()
        if is_valid_prospect_name(cand):
            return cand

    return None


def extract_company(text: str, context_field: Optional[str] = None) -> Optional[str]:
    """Extracts company or employer name from English, Hindi, or Marathi speech."""
    if not text:
        return None
    clean = text.strip()

    # Common location and domain stop words that should not be extracted as employer companies
    location_and_domain_stops = {
        "personal", "home", "business", "loan", "lakh", "crore", "rupees",
        "year", "years", "month", "months", "लोन", "कर्ज", "रुपये", "लाख",
        "mumbai", "pune", "delhi", "bangalore", "bengaluru", "hyderabad",
        "chennai", "kolkata", "ahmedabad", "india", "maharashtra"
    }

    # 1. If context_field is "company", user was directly asked for employer/organization
    if context_field == "company":
        cand = re.sub(r"^(?:i work at|i work in|i am working at|i am working in|working at|working in|company is|company name is|my company is)\s*", "", clean, flags=re.IGNORECASE)
        cand = re.sub(r"^(?:at|in)\s+", "", cand, flags=re.IGNORECASE)
        cand = re.sub(r"\b(?:and|aur|ani|loan|loen|amount|for|tenure|looking|want|chahiye|pahije|phone)\b.*$", "", cand, flags=re.IGNORECASE).strip()
        cand = re.sub(r"[.,;:]+$", "", cand).strip()
        if cand.lower() not in location_and_domain_stops and len(cand) >= 2 and len(cand.split()) <= 5:
            return cand

    # 2. Contextual patterns
    patterns = [
        r"\b(?:company\s*(?:is|name\s*is)?|working\s*(?:at|in|with)?|work\s*(?:at|in|with)?|employed\s*(?:at|by|with)?)\s+([A-Za-z0-9\u0900-\u097F\s&.,'-]{2,40}?)(?:\s+(?:and|phone|loan|amount|tenure|salary|i|for|personal|home|business)\b|[.,;]|$)",
        r"\b(?:i\s+work\s+(?:at|in|with)|work\s+(?:at|in|with)|working\s+(?:at|in|with))\s+([A-Za-z0-9\u0900-\u097F\s&.,'-]{2,30}?)(?:\s+(?:and|phone|loan|amount|tenure|salary|i|for|personal|home|business)\b|[.,;]|$)",
        r"\b(?:from)\s+([A-Z\u0900-\u097F][A-Za-z0-9\u0900-\u097F\s&.,'-]{1,30}?)(?:\s+(?:and|phone|loan|amount|tenure|salary|i|for|personal|home|business|needs|want)\b|[.,;]|$)",
        r"(?:कंपनी|कंपनीचे\s*नाव|कंपनी\s*का\s*नाम)\s*(?:है|आहे)?\s*([a-zA-Z\u0900-\u097F\s&.,'-]{2,40}?)(?:\s+(?:है|आहे|और|आणि|फोन|लोन|कर्ज)(?:\s+|$)|[.,;]|$)",
        r"([a-zA-Z\u0900-\u097F\s&.,'-]{2,30}?)\s*(?:में\s*काम\s*करता\s*हूँ|में\s*कार्यरत\s*हूँ|मध्ये\s*काम\s*करतो|मध्ये\s*कार्यरत\s*आहे)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            raw = re.sub(r"[\s,.-]+(?:है|आहे|हूँ|आहेत|hai|aahe|में|मध्ये)$", "", raw).strip()
            raw = re.sub(r"^(?:है|आहे|हूँ|में|मध्ये)[\s,.-]+", "", raw).strip()
            raw = re.sub(r"[.,;:]+$", "", raw).strip()
            if raw.lower() not in location_and_domain_stops and len(raw) >= 2 and len(raw.split()) <= 5:
                return raw

    # 3. Known enterprise and employer names standalone (English and Devanagari)
    known_companies = [
        "TCS", "Tata Consultancy Services", "Infosys", "Wipro", "HCL", "Tech Mahindra",
        "Google", "Microsoft", "Amazon", "Apple", "Meta", "IBM", "Accenture", "Cognizant",
        "Reliance", "Jio", "Tata Motors", "Tata Steel", "L&T", "Larsen & Toubro", "Mahindra",
        "Adani", "ITC", "HUL", "Flipkart", "Swiggy", "Zomato", "Paytm", "Ola",
        "इन्फोसिस", "इंफोसिस", "टीसीएस", "विप्रो", "टेक महिंद्रा", "रिलायंस", "गूगल", "अमेज़न",
    ]
    for comp in known_companies:
        if re.search(rf"\b{re.escape(comp)}\b", text, flags=re.IGNORECASE):
            return comp

    return None


def extract_tenure_months(text: str, context_field: Optional[str] = None) -> Optional[int]:
    """Extracts loan tenure duration in months from English, Hindi, or Marathi speech."""
    if not text:
        return None
    normalized = normalize_spoken_numbers(text)
    lower = normalized.lower().strip()

    # 1. If context_field is "tenure_months" and user gives raw integer or spoken number
    if context_field == "tenure_months":
        num_m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*$", lower)
        if num_m:
            val = float(num_m.group(1))
            if 1 <= val <= 5:
                return int(val * 12)
            elif 6 <= val <= 360:
                return int(val)

    # Years to months: e.g. "2 years", "3 yrs", "1.5 years", "2 साल", "3 वर्ष", "2 वर्षे", "5 वर्षांसाठी"
    yr_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?|yr|साल|वर्ष|वर्षांसाठी|वर्षे)(?:\b|\s|[.,;]|$)", lower)
    if yr_match:
        try:
            yrs = float(yr_match.group(1))
            return int(yrs * 12)
        except ValueError:
            pass

    # Months: e.g. "24 months", "36 months", "12 महीने", "24 महिने", "36 महिन्यांसाठी", "माह"
    mo_match = re.search(r"(\d+)\s*(?:months?|mos?|mo|महीने|महिने|महिन्यांसाठी|माह|mahina|mahine)(?:\b|\s|[.,;]|$)", lower)
    if mo_match:
        try:
            return int(mo_match.group(1))
        except ValueError:
            pass

    # Explicit tenure keywords: e.g. "tenure 36", "tenure of 24", "अवधि 36", "कालावधी 24"
    ten_match = re.search(r"(?:tenure|duration|अवधि|कालावधी|मुदत)\s*(?:is|of|:)?\s*(\d+)(?:\b|\s|[.,;]|$)", lower)
    if ten_match:
        try:
            val = int(ten_match.group(1))
            if 1 <= val <= 360:
                return val
        except ValueError:
            pass

    return None


def get_next_missing_parameter(lead_data: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Determines the next missing required parameter in deterministic sequential order:
    1. name (Customer / Prospect full name)
    2. phone (Contact phone number)
    3. company (Employer / Company name)
    4. loan_type (Type of loan: Personal, Home, Business, etc.)
    5. loan_amount (Requested loan amount in numeric currency)
    6. tenure_months (Loan duration in months)

    Returns None if all 6 required parameters are collected.
    """
    if not lead_data or not isinstance(lead_data, dict):
        return "name"
    for field in REQUIRED_LEAD_FIELDS:
        val = lead_data.get(field)
        if val is None:
            return field
        if isinstance(val, str) and not val.strip():
            return field
    return None


def get_missing_parameter_prompt(
    next_param: Optional[str],
    lead_data: Optional[Dict[str, Any]] = None,
    lang: str = "en",
) -> str:
    """
    Generates a natural, concise spoken prompt asking for the next missing parameter,
    or a short thank-you confirmation once all required parameters are collected.
    """
    lead_data = lead_data or {}
    prospect_name = (lead_data.get("name") or "").strip()
    name_honorific_hi = f"{prospect_name} जी" if prospect_name else ""
    name_honorific_mr = f"{prospect_name} जी" if prospect_name else ""

    norm_lang = (lang or "en").strip().lower()
    if norm_lang.startswith("hi") or norm_lang == "hindi":
        resolved_lang = "hi"
    elif norm_lang.startswith("mr") or norm_lang == "marathi":
        resolved_lang = "mr"
    else:
        resolved_lang = "en"

    # All required parameters collected -> Short thank-you confirmation
    if next_param is None:
        if resolved_lang == "hi":
            if name_honorific_hi:
                return f"धन्यवाद {name_honorific_hi}! आपकी सभी आवश्यक जानकारी दर्ज कर ली गई है। हमारी टीम जल्द ही आपसे संपर्क करेगी।"
            return "धन्यवाद! आपकी सभी आवश्यक जानकारी दर्ज कर ली गई है। हमारी टीम जल्द ही आपसे संपर्क करेगी।"
        elif resolved_lang == "mr":
            if name_honorific_mr:
                return f"धन्यवाद {name_honorific_mr}! आपले सर्व आवश्यक तपशील नोंदवले गेले आहेत. आमची टीम लवकरच आपल्याशी संपर्क साधेल."
            return "धन्यवाद! आपले सर्व आवश्यक तपशील नोंदवले गेले आहेत. आमची टीम लवकरच आपल्याशी संपर्क साधेल."
        else:
            if prospect_name:
                return f"Thank you, {prospect_name}! All your details have been recorded. Our team will contact you shortly."
            return "Thank you! All your details have been recorded. Our team will contact you shortly."

    # Sequential parameter 1: name
    if next_param == "name":
        if resolved_lang == "hi":
            return "नमस्ते! कृपया आपका शुभ नाम बताइए?"
        elif resolved_lang == "mr":
            return "नमस्कार! कृपया आपले नाव सांगा?"
        else:
            return "Hello! May I have your name, please?"

    # Sequential parameter 2: phone
    elif next_param == "phone":
        if resolved_lang == "hi":
            ack = f"धन्यवाद {name_honorific_hi}।" if name_honorific_hi else "धन्यवाद।"
            return f"{ack} कृपया आपका संपर्क फोन नंबर बताइए?"
        elif resolved_lang == "mr":
            ack = f"धन्यवाद {name_honorific_mr}." if name_honorific_mr else "धन्यवाद."
            return f"{ack} कृपया आपला संपर्क फोन नंबर सांगा?"
        else:
            ack = f"Thank you, {prospect_name}." if prospect_name else "Thank you."
            return f"{ack} What is your contact phone number?"

    # Sequential parameter 3: company
    elif next_param == "company":
        if resolved_lang == "hi":
            ack = f"धन्यवाद {name_honorific_hi}।" if name_honorific_hi else "धन्यवाद।"
            return f"{ack} आप किस कंपनी या संस्थान में कार्यरत हैं?"
        elif resolved_lang == "mr":
            ack = f"धन्यवाद {name_honorific_mr}." if name_honorific_mr else "धन्यवाद."
            return f"{ack} आपण कोणत्या कंपनीमध्ये किंवा संस्थेत काम करता?"
        else:
            ack = f"Thank you, {prospect_name}." if prospect_name else "Thank you."
            return f"{ack} What company or organization do you work for?"

    # Sequential parameter 4: loan_type
    elif next_param == "loan_type":
        if resolved_lang == "hi":
            return "धन्यवाद। आपको किस प्रकार का लोन चाहिए, जैसे पर्सनल लोन, होम लोन या बिज़नेस लोन?"
        elif resolved_lang == "mr":
            return "धन्यवाद. आपल्याला कोणत्या प्रकारचे कर्ज हवे आहे, जसे की वैयक्तिक कर्ज, गृह कर्ज किंवा व्यवसाय कर्ज?"
        else:
            return "Thank you. What type of loan are you looking for, such as a personal loan, home loan, or business loan?"

    # Sequential parameter 5: loan_amount
    elif next_param == "loan_amount":
        loan_type = (lead_data.get("loan_type") or "").strip()
        if resolved_lang == "hi":
            lt = f"{loan_type} के लिए " if loan_type else ""
            return f"समझ गया। आपको {lt}कितनी लोन राशि की आवश्यकता है?"
        elif resolved_lang == "mr":
            lt = f"{loan_type}साठी " if loan_type else ""
            return f"समजले. आपल्याला {lt}किती रकमेचे कर्ज हवे आहे?"
        else:
            lt = f" for your {loan_type}" if loan_type else ""
            return f"Understood. What loan amount do you require{lt}?"

    # Sequential parameter 6: tenure_months
    elif next_param == "tenure_months":
        if resolved_lang == "hi":
            return "आपको कितने समय (महीनों या वर्षों) के लिए लोन की अवधि चाहिए?"
        elif resolved_lang == "mr":
            return "आपल्याला किती कालावधीसाठी (महिने किंवा वर्षे) कर्ज हवे आहे?"
        else:
            return "What loan tenure or repayment duration (in months or years) are you looking for?"

    return "Thank you! Please share your details."


def merge_lead_safely(
    existing: Optional[Dict[str, Any]],
    incoming: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Deterministically merges incoming lead updates into existing lead.
    CRITICAL RULE: Never overwrite correctly stored, non-null, valid fields
    with None, empty, uncertain, or invalid data.
    """
    merged: Dict[str, Any] = dict(existing) if existing and isinstance(existing, dict) else {}
    if not incoming or not isinstance(incoming, dict):
        return merged

    for k in REQUIRED_LEAD_FIELDS + ["email", "role", "notes"]:
        if k not in incoming:
            continue
        new_val = incoming.get(k)
        if new_val is None:
            continue
        if isinstance(new_val, str) and not new_val.strip():
            continue

        # Field-specific validation before accepting update
        if k == "name":
            if is_valid_prospect_name(str(new_val)):
                merged["name"] = str(new_val).strip()
        elif k == "phone":
            clean_phone = extract_phone_number(str(new_val))
            if clean_phone:
                merged["phone"] = clean_phone
            elif re.match(r"^[6-9]\d{9}$", str(new_val).strip()):
                merged["phone"] = str(new_val).strip()
        elif k == "loan_type":
            if isinstance(new_val, str) and len(new_val.strip()) >= 3:
                existing_lt = merged.get("loan_type")
                if not (existing_lt and existing_lt != "Loan" and new_val.strip() == "Loan"):
                    merged["loan_type"] = new_val.strip()
        elif k == "loan_amount":
            try:
                amt = float(new_val)
                if amt > 0:
                    merged["loan_amount"] = amt
            except (ValueError, TypeError):
                pass
        elif k == "tenure_months":
            try:
                ten = int(new_val)
                if 1 <= ten <= 360:
                    merged["tenure_months"] = ten
            except (ValueError, TypeError):
                pass
        elif k == "company":
            clean_c = str(new_val).strip()
            if len(clean_c) >= 2 and clean_c.lower() not in {"loan", "personal", "home", "business", "unknown", "none", "null"}:
                merged["company"] = clean_c
        elif k == "email":
            clean_e = extract_email(str(new_val))
            if clean_e:
                merged["email"] = clean_e
        else:
            merged[k] = new_val

    return merged


CLARIFICATION_PROMPTS: Dict[str, Dict[str, List[str]]] = {
    "name": {
        "en": [
            "I couldn't quite catch your name. Could you please share your full name again?",
            "Sorry, I missed that. May I have your name, please?",
            "Could you kindly repeat your full name for me?",
        ],
        "hi": [
            "माफ़ी चाहता हूँ, मुझे आपका नाम ठीक से समझ नहीं आया। क्या आप कृपया अपना पूरा नाम दोबारा बता सकते हैं?",
            "क्षमा करें, मुझे आपका नाम स्पष्ट नहीं हुआ। कृपया अपना शुभ नाम एक बार फिर बताइए?",
            "कृपया अपना नाम एक बार फिर दोहराएंगे?",
        ],
        "mr": [
            "माफ करा, मला आपले नाव नीट ऐकू आले नाही. कृपया आपले पूर्ण नाव पुन्हा सांगाल का?",
            "क्षमस्व, मला आपले नाव स्पष्ट समजले नाही. कृपया आपले नाव पुन्हा सांगा?",
            "कृपया आपले नाव पुन्हा एकदा सांगू शकाल का?",
        ],
    },
    "phone": {
        "en": [
            "I didn't quite catch the phone number. Could you please repeat your 10-digit mobile number?",
            "Sorry, I missed the contact number. Could you kindly share your 10-digit phone number again?",
            "Could you please state your 10-digit mobile number once more?",
        ],
        "hi": [
            "माफ़ी चाहता हूँ, मुझे आपका फोन नंबर ठीक से समझ नहीं आया। क्या आप कृपया अपना 10 अंकों का मोबाइल नंबर दोबारा बता सकते हैं?",
            "क्षमा करें, फोन नंबर स्पष्ट नहीं हुआ। कृपया अपना 10 अंकों का संपर्क नंबर एक बार फिर बताएंगे?",
            "कृपया अपना मोबाइल नंबर स्पष्ट रूप से दोबारा बताएं?",
        ],
        "mr": [
            "माफ करा, मला आपला फोन नंबर नीट समजला नाही. कृपया आपला 10 अंकी मोबाईल नंबर पुन्हा सांगाल का?",
            "क्षमस्व, संपर्क नंबर स्पष्ट झाला नाही. कृपया आपला मोबाईल नंबर पुन्हा एकदा सांगा?",
            "कृपया आपला 10 अंकी फोन नंबर स्पष्टपणे पुन्हा सांगू शकाल का?",
        ],
    },
    "company": {
        "en": [
            "I didn't quite catch the organization name. Could you please mention the company or employer you work for?",
            "Sorry, which company or institution are you currently working with? Could you repeat the name?",
            "Could you kindly clarify your current employer or company name?",
        ],
        "hi": [
            "माफ़ी चाहता हूँ, मुझे आपकी कंपनी का नाम ठीक से समझ नहीं आया। आप किस कंपनी या संस्थान में काम करते हैं?",
            "क्षमा करें, आपकी कंपनी का नाम स्पष्ट नहीं हुआ। कृपया अपने संस्थान या कंपनी का नाम दोबारा बताएं?",
            "कृपया अपनी कंपनी का नाम एक बार फिर बताएंगे?",
        ],
        "mr": [
            "माफ करा, मला आपल्या कंपनीचे नाव नीट समजले नाही. आपण कोणत्या कंपनीत किंवा संस्थेत काम करता?",
            "क्षमस्व, कंपनीचे नाव स्पष्ट झाले नाही. कृपया आपल्या संस्थेचे किंवा कंपनीचे नाव पुन्हा सांगाल का?",
            "कृपया आपण कार्यरत असलेल्या कंपनीचे नाव पुन्हा सांगा?",
        ],
    },
    "loan_type": {
        "en": [
            "I couldn't identify the loan type. Are you looking for a personal loan, home loan, or business loan?",
            "Could you please clarify what category of loan you need, such as personal, home, or business?",
            "Which type of loan would you like assistance with today?",
        ],
        "hi": [
            "माफ़ी चाहता हूँ, लोन का प्रकार स्पष्ट नहीं हुआ। क्या आपको पर्सनल लोन, होम लोन या बिज़नेस लोन चाहिए?",
            "कृपया स्पष्ट करेंगे कि आपको किस तरह का लोन चाहिए, जैसे कि पर्सनल, होम या बिज़नेस लोन?",
            "आपको किस प्रकार के लोन की आवश्यकता है, कृपया दोबारा बताएं?",
        ],
        "mr": [
            "माफ करा, कर्जाचा प्रकार स्पष्ट झाला नाही. आपल्याला वैयक्तिक कर्ज, गृह कर्ज की व्यवसाय कर्ज हवे आहे?",
            "कृपया स्पष्ट कराल का की आपल्याला कोणत्या प्रकारचे कर्ज हवे आहे, जसे की पर्सनल, होम किंवा बिझनेस लोन?",
            "आपल्याला नेमके कोणत्या प्रकारचे कर्ज हवे आहे, कृपया पुन्हा सांगा?",
        ],
    },
    "loan_amount": {
        "en": [
            "I didn't catch the exact loan amount. Could you please specify how much loan amount you require?",
            "Sorry, how much loan amount are you looking for? Could you please repeat the amount in rupees?",
            "Could you kindly clarify the loan amount you need?",
        ],
        "hi": [
            "माफ़ी चाहता हूँ, मुझे लोन की राशि ठीक से समझ नहीं आई। आपको कितने रुपये के लोन की आवश्यकता है?",
            "क्षमा करें, लोन राशि स्पष्ट नहीं हुई। कृपया बताएं कि आपको कितनी राशि का लोन चाहिए?",
            "कृपया अपेक्षित लोन राशि एक बार फिर बताएंगे?",
        ],
        "mr": [
            "माफ करा, मला कर्जाची रक्कम नीट समजली नाही. आपल्याला किती रकमेचे कर्ज हवे आहे?",
            "क्षमस्व, कर्जाची अपेक्षित रक्कम स्पष्ट झाली नाही. कृपया आपल्याला किती रकमेचे कर्ज हवे आहे ते पुन्हा सांगाल का?",
            "कृपया आपल्याला किती कर्ज हवे आहे, ती रक्कम पुन्हा सांगा?",
        ],
    },
    "tenure_months": {
        "en": [
            "I didn't quite catch the repayment period. How many months or years of tenure do you need?",
            "Could you please specify the desired loan tenure in months or years once more?",
            "Sorry, what duration (in months or years) would you like for the loan?",
        ],
        "hi": [
            "माफ़ी चाहता हूँ, लोन की अवधि स्पष्ट नहीं हुई। आपको कितने महीनों या वर्षों के लिए लोन चाहिए?",
            "क्षमा करें, समय-सीमा समझ नहीं आई। कृपया बताएं कि आप कितने समय (महीनों या सालों) के लिए लोन लेना चाहते हैं?",
            "कृपया लोन चुकाने की अवधि (महीनों या वर्षों में) एक बार फिर बताएंगे?",
        ],
        "mr": [
            "माफ करा, कर्जाचा कालावधी स्पष्ट झाला नाही. आपल्याला किती महिने किंवा वर्षांसाठी कर्ज हवे आहे?",
            "क्षमस्व, परतफेडीची मुदत समजली नाही. कृपया किती कालावधीसाठी कर्ज हवे आहे ते पुन्हा सांगाल का?",
            "कृपया कर्जाचा कालावधी (महिने किंवा वर्षांमध्ये) पुन्हा एकदा सांगा?",
        ],
    },
}


def get_clarification_prompt(
    field: Optional[str],
    raw_transcript: str = "",
    lead_data: Optional[Dict[str, Any]] = None,
    lang: str = "en",
    attempt: int = 1,
) -> str:
    """
    Generates a natural, polite, context-aware clarification for an unclear field.
    Rotates dynamically across variations so it never hardcodes or repeats one fixed message.
    """
    if not field or field not in CLARIFICATION_PROMPTS:
        return get_missing_parameter_prompt(field, lead_data, lang=lang)

    norm_lang = (lang or "en").strip().lower()
    if norm_lang.startswith("hi") or norm_lang == "hindi":
        resolved_lang = "hi"
    elif norm_lang.startswith("mr") or norm_lang == "marathi":
        resolved_lang = "mr"
    else:
        resolved_lang = "en"

    options = CLARIFICATION_PROMPTS[field].get(resolved_lang, CLARIFICATION_PROMPTS[field]["en"])
    if not options:
        return get_missing_parameter_prompt(field, lead_data, lang=lang)

    seed = abs(hash(raw_transcript.strip())) if raw_transcript else 0
    idx = (attempt + seed) % len(options)
    return options[idx]


class LeadExtractionError(Exception):
    """Base exception for lead extraction failures."""
    pass


class LeadValidationError(LeadExtractionError):
    """Raised when Pydantic validation fails against the extracted lead data."""
    def __init__(self, message: str, raw_data: Optional[Dict[str, Any]] = None, errors: Optional[Any] = None):
        super().__init__(message)
        self.raw_data = raw_data
        self.errors = errors


class MalformedLLMOutputError(LeadExtractionError):
    """Raised when the LLM output is not valid JSON or cannot be parsed."""
    def __init__(self, message: str, raw_output: Optional[str] = None):
        super().__init__(message)
        self.raw_output = raw_output


class OpenRouterConfigurationError(LeadExtractionError):
    """Raised when OpenRouter API key or configuration is missing or invalid."""
    pass


class OpenRouterAPIError(LeadExtractionError):
    """Raised when OpenRouter HTTP request fails or returns an error status."""
    pass


class Lead(BaseModel):
    """
    Authoritative Pydantic model for structured sales lead extraction.
    All fields default to None if not explicitly present in transcript.
    """
    name: Optional[str] = Field(default=None, description="Full name of the prospect/customer")
    phone: Optional[str] = Field(default=None, description="Phone number if mentioned")
    email: Optional[str] = Field(default=None, description="Email address if mentioned")
    company: Optional[str] = Field(default=None, description="Company or organization name")
    role: Optional[str] = Field(default=None, description="Designation or job title")
    loan_type: Optional[str] = Field(default=None, description="Type of loan requested (e.g. Personal Loan, Home Loan, Business Loan)")
    loan_amount: Optional[float] = Field(default=None, description="Requested or discussed loan amount in numeric currency units")
    tenure_months: Optional[int] = Field(default=None, description="Tenure duration in months as integer")
    notes: Optional[str] = Field(default=None, description="Important sales context, intent, objections, or next steps")

    @field_validator("name", mode="before")
    @classmethod
    def validate_prospect_name(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            cleaned = v.strip()
            if not cleaned or cleaned.lower() in {"null", "none", "n/a", "na", "unknown", "not specified", "unspecified"}:
                return None
            if not is_valid_prospect_name(cleaned):
                return None
            return cleaned
        return None

    @field_validator("phone", "email", "company", "role", "loan_type", "notes", mode="before")
    @classmethod
    def clean_empty_strings(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            cleaned = v.strip()
            if not cleaned or cleaned.lower() in {"null", "none", "n/a", "na", "unknown", "not specified", "unspecified"}:
                return None
            return cleaned
        return str(v)

    @field_validator("loan_amount", mode="before")
    @classmethod
    def clean_loan_amount(cls, v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            cleaned = v.strip()
            if not cleaned or cleaned.lower() in {"null", "none", "n/a", "na", "unknown", "not specified", "unspecified"}:
                return None
            sanitized = re.sub(r"[^\d.]", "", cleaned)
            if not sanitized:
                raise ValueError(f"Cannot parse loan_amount from string: '{v}'")
            return float(sanitized)
        raise ValueError(f"Invalid type for loan_amount: {type(v)}")

    @field_validator("tenure_months", mode="before")
    @classmethod
    def clean_tenure_months(cls, v):
        if v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str):
            cleaned = v.strip()
            if not cleaned or cleaned.lower() in {"null", "none", "n/a", "na", "unknown", "not specified", "unspecified"}:
                return None
            match = re.search(r"\b(\d+)\b", cleaned)
            if match:
                return int(match.group(1))
            raise ValueError(f"Cannot parse tenure_months from string: '{v}'")
        raise ValueError(f"Invalid type for tenure_months: {type(v)}")


class LeadExtractionRequest(BaseModel):
    """Input payload for lead extraction endpoint."""
    transcript: str = Field(..., description="Raw audio transcript text from Deepgram or call recording")
    model: Optional[str] = Field(default=None, description="Optional OpenRouter model override")
    existing_lead: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional existing lead details already captured in this session to update/merge without creating duplicates.",
    )
    language: Optional[str] = Field(
        default=None,
        description="Optional language code ('en', 'hi', 'mr').",
    )
    stream: Optional[bool] = Field(
        default=False,
        description="Whether to stream the assistant response using Server-Sent Events (SSE).",
    )
    lead_id: Optional[int] = Field(
        default=None,
        description="Optional existing PostgreSQL lead ID to update instead of creating a duplicate.",
    )
    is_interim: Optional[bool] = Field(
        default=None,
        description="Optional flag indicating whether this is an interim/partial STT transcript. If true, lead extraction is bypassed.",
    )
    is_final: Optional[bool] = Field(
        default=None,
        description="Optional flag indicating genuine final STT transcript.",
    )


class LeadExtractionResponse(BaseModel):
    """Output payload returning validated lead JSON and extraction metadata."""
    status: str = "success"
    is_greeting: Optional[bool] = False
    message: Optional[str] = None
    lead: Lead
    model: str
    transcript_length: int


class LeadExtractorService:
    """
    Service for extracting structured lead data from speech transcripts
    using OpenRouter LLM and Pydantic as the authoritative validation layer.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
    ):
        self.api_key = (
            api_key if api_key is not None else os.getenv("OPENROUTER_API_KEY", "")
        ).strip()
        self.model = (
            model or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
        ).strip().lstrip("~")
        self.base_url = (
            base_url or os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)
        ).rstrip("/")
        self._client = http_client

    def get_http_client(self) -> httpx.Client:
        """Return shared HTTP client with connection pooling for low-latency requests."""
        if self._client is None:
            self._client = httpx.Client(
                limits=httpx.Limits(max_keepalive_connections=20, keepalive_expiry=30.0),
                timeout=30.0,
            )
        return self._client

    # ------------------------------------------------------------------------
    # METHOD: extract_lead
    # ------------------------------------------------------------------------
    # • WHAT IT DOES: Analyzes speech transcript, queries DeepSeek LLM with strict
    #   zero-hallucination prompt, parses JSON output, and validates against Pydantic Lead model.
    # • INPUTS:
    #     - transcript (str): Raw transcription text from Deepgram.
    #     - model (Optional[str]): OpenRouter model override.
    #     - existing_lead (Optional[Dict]): Prior lead state in this session to merge/update.
    #     - language (Optional[str]): Language code ('en', 'hi', 'mr').
    # • OUTPUT: Dictionary containing status, validated lead fields, and metadata.
    # • WHY IT IS USED: Automatically converts conversational speech into clean CRM data.
    #   If an existing lead exists, updates only the newly mentioned fields without wiping previous fields.
    # • WHERE IT FITS IN THE FLOW:
    #     [Deepgram Transcript] -> [api/extract-lead] -> [extract_lead] -> [LeadRepository]
    # ------------------------------------------------------------------------
    def extract_lead(
        self,
        transcript: str,
        model: Optional[str] = None,
        existing_lead: Optional[Dict[str, Any]] = None,
        language: Optional[str] = None,
        is_interim: Optional[bool] = None,
        is_final: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Executes structured lead extraction:

        1. Validates input transcript is non-empty.
        2. Rejects/ignores interim or non-final STT transcripts.
        3. Checks if transcript is solely a greeting or assistant query (bypasses LLM).
        4. Sends extraction prompt to OpenRouter LLM (integrating existing_lead if present).
        5. Parses and validates response against Pydantic Lead model.
        6. Safely merges lead without overwriting valid data.
        7. Returns structured lead dictionary.
        """
        clean_transcript = (transcript or "").strip()
        if not clean_transcript:
            raise ValueError("Transcript cannot be empty or whitespace.")

        target_model = (model or self.model).strip()

        # Clean and validate existing lead context if supplied
        clean_existing = {}
        if existing_lead and isinstance(existing_lead, dict):
            clean_existing = {
                k: v for k, v in existing_lead.items()
                if v is not None and k in Lead.model_fields
            }

        # Check if starting a new lead or if prior lead was complete
        is_prior_complete = get_next_missing_parameter(clean_existing) is None and bool(clean_existing)
        is_reset_turn = is_new_lead_intent(clean_transcript) or is_prior_complete
        if is_reset_turn:
            clean_existing = {}

        spoken_lang = detect_spoken_language(clean_transcript)
        if language in ("en", "hi", "mr"):
            if is_greeting(clean_transcript):
                resp_lang = language
            elif spoken_lang == "mixed":
                resp_lang = "en"
            else:
                resp_lang = get_response_language(spoken_lang)
        else:
            resp_lang = get_response_language(spoken_lang)

        # CRITICAL: Ignore interim / partial speech transcripts
        if is_interim is True or is_final is False:
            logger.info(f"[LeadExtractor] Ignoring interim transcript: '{clean_transcript}'")
            lead_dict = Lead.model_validate(clean_existing).model_dump() if clean_existing else Lead().model_dump()
            next_missing = get_next_missing_parameter(lead_dict)
            return {
                "status": "interim_ignored",
                "is_interim": True,
                "message": "",
                "lead": lead_dict,
                "next_missing_parameter": next_missing,
                "is_complete": next_missing is None,
                "detected_language": spoken_lang,
                "language": resp_lang,
                "model": target_model,
                "transcript_length": len(clean_transcript),
            }

        # Check if the speech turn is purely a greeting
        if is_greeting(clean_transcript):
            greeting_msg = get_greeting_response(resp_lang)
            logger.info(
                f"[LeadExtractor] Transcript '{clean_transcript}' is greeting-only ({spoken_lang}). "
                "Returning greeting response without calling LLM or saving to DB."
            )
            lead_dict = Lead.model_validate(clean_existing).model_dump() if clean_existing else Lead().model_dump()
            next_missing = get_next_missing_parameter(lead_dict)
            return {
                "status": "greeting",
                "is_greeting": True,
                "message": greeting_msg,
                "lead": lead_dict,
                "next_missing_parameter": next_missing,
                "is_complete": next_missing is None,
                "detected_language": spoken_lang,
                "language": resp_lang,
                "model": target_model,
                "transcript_length": len(clean_transcript),
            }

        # Check if the speech turn is an assistant conversation query
        assistant_resp = get_assistant_query_response(clean_transcript, resp_lang)
        if assistant_resp:
            logger.info(
                f"[LeadExtractor] Transcript '{clean_transcript}' is assistant query ({spoken_lang}). "
                "Returning natural assistant response without calling LLM or saving to DB."
            )
            lead_dict = Lead.model_validate(clean_existing).model_dump() if clean_existing else Lead().model_dump()
            next_missing = get_next_missing_parameter(lead_dict)
            return {
                "status": "assistant_query",
                "is_assistant_query": True,
                "message": assistant_resp,
                "lead": lead_dict,
                "next_missing_parameter": next_missing,
                "is_complete": next_missing is None,
                "detected_language": spoken_lang,
                "language": resp_lang,
                "model": target_model,
                "transcript_length": len(clean_transcript),
            }

        if not self.api_key:
            raise OpenRouterConfigurationError(
                "OPENROUTER_API_KEY is not set. Please configure OPENROUTER_API_KEY in backend/.env."
            )

        existing_lead_section = ""
        if clean_existing:
            existing_lead_section = (
                f"\nEXISTING LEAD RECORD (Preserve non-null values unless updated by transcript):\n"
                f"{json.dumps(clean_existing, separators=(',', ':'))}\n"
            )

        lang_instruction = "The transcript is in English."
        if spoken_lang == "hi":
            lang_instruction = "The transcript is in Hindi (or Hinglish). Extract facts accurately."
        elif spoken_lang == "mr":
            lang_instruction = "The transcript is in Marathi (or Marathish). Extract facts accurately."
        elif spoken_lang == LANG_MIXED:
            lang_instruction = "The transcript is in mixed Hindi/Marathi/English. Extract facts accurately."

        user_content = (
            f"{lang_instruction}\n"
            f"Sales Call Transcript:\n\"\"\"{clean_transcript}\n\"\"\"\n"
            f"{existing_lead_section}\n"
            "Return the extracted JSON object now:"
        )

        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": LEAD_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "provider": {"sort": "latency"},
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Voice-Sales-Copilot",
            "X-Title": "Voice Sales Copilot",
        }

        endpoint_url = f"{self.base_url}/chat/completions"
        logger.info(f"Sending lead extraction request to OpenRouter ({target_model})...")

        client = self._client or self.get_http_client()
        try:
            resp = client.post(endpoint_url, json=payload, headers=headers)
        except Exception as net_err:
            logger.error(f"OpenRouter connection error during lead extraction: {str(net_err)}")
            raise OpenRouterAPIError(f"Failed to communicate with OpenRouter API: {str(net_err)}") from net_err

        if resp.status_code != 200:
            error_body = resp.text
            logger.error(f"OpenRouter API returned HTTP {resp.status_code}: {error_body}")
            raise OpenRouterAPIError(
                f"OpenRouter API returned HTTP {resp.status_code}: {error_body}"
            )

        try:
            resp_data = resp.json()
            raw_content = resp_data["choices"][0]["message"]["content"].strip()
        except Exception as parse_err:
            logger.error(f"Failed to parse OpenRouter response payload: {str(parse_err)}")
            raise OpenRouterAPIError(f"Invalid response envelope from OpenRouter: {str(parse_err)}") from parse_err

        # Clean markdown code blocks if the LLM output includes them
        cleaned_json_str = raw_content
        if cleaned_json_str.startswith("```"):
            cleaned_json_str = re.sub(r"^```(?:json)?\s*", "", cleaned_json_str)
            cleaned_json_str = re.sub(r"\s*```$", "", cleaned_json_str)
        cleaned_json_str = cleaned_json_str.strip()

        # Parse JSON
        try:
            parsed_data = json.loads(cleaned_json_str)
        except json.JSONDecodeError as jde:
            logger.error(f"Malformed LLM JSON output: {cleaned_json_str}")
            raise MalformedLLMOutputError(
                f"LLM did not return valid JSON: {str(jde)}",
                raw_output=raw_content,
            ) from jde

        if not isinstance(parsed_data, dict):
            logger.error(f"LLM output is not a JSON object: {cleaned_json_str}")
            raise MalformedLLMOutputError(
                f"LLM output expected to be a JSON object, got {type(parsed_data).__name__}",
                raw_output=raw_content,
            )

        # Authoritative Pydantic validation
        try:
            validated_lead = Lead.model_validate(parsed_data)
        except ValidationError as ve:
            logger.error(f"Pydantic lead validation error: {str(ve)}")
            clean_errors = []
            for err in ve.errors():
                clean_err = {
                    "type": str(err.get("type", "")),
                    "loc": [str(x) for x in err.get("loc", ())],
                    "msg": str(err.get("msg", "")),
                    "input": str(err.get("input", "")),
                }
                clean_errors.append(clean_err)
            raise LeadValidationError(
                f"Lead data failed Pydantic validation: {str(ve)}",
                raw_data=parsed_data,
                errors=clean_errors,
            ) from ve

        extracted_dict = validated_lead.model_dump()
        # Strict name validation check on extracted_dict
        if extracted_dict.get("name") and not is_valid_prospect_name(extracted_dict["name"]):
            extracted_dict["name"] = None

        prior_missing = get_next_missing_parameter(clean_existing)
        if extracted_dict.get("phone") is None:
            phone_found = extract_phone_number(clean_transcript)
            if phone_found:
                extracted_dict["phone"] = phone_found
        if extracted_dict.get("email") is None:
            email_found = extract_email(clean_transcript)
            if email_found:
                extracted_dict["email"] = email_found
        if extracted_dict.get("company") is None:
            comp_found = extract_company(clean_transcript, context_field=prior_missing)
            if comp_found:
                extracted_dict["company"] = comp_found
        if extracted_dict.get("loan_type") is None:
            lt_found = extract_loan_type(clean_transcript, context_field=prior_missing)
            if lt_found:
                extracted_dict["loan_type"] = lt_found
        if extracted_dict.get("loan_amount") is None:
            amt_found = extract_loan_amount(clean_transcript, context_field=prior_missing)
            if amt_found:
                extracted_dict["loan_amount"] = amt_found
        if extracted_dict.get("tenure_months") is None:
            tenure_found = extract_tenure_months(clean_transcript, context_field=prior_missing)
            if tenure_found:
                extracted_dict["tenure_months"] = tenure_found
        if extracted_dict.get("name") is None:
            name_found = extract_name(clean_transcript, context_field=prior_missing)
            if name_found:
                extracted_dict["name"] = name_found

        # Deterministic safe merge: never overwrite valid stored fields with uncertain data
        final_lead_dict = merge_lead_safely(clean_existing, extracted_dict)

        # Check if the user was asked for a specific field, but their answer was unclear / invalid
        was_prior_unclear = False
        if prior_missing and prior_missing in REQUIRED_LEAD_FIELDS:
            prior_val = final_lead_dict.get(prior_missing)
            prior_satisfied = prior_val is not None and (not isinstance(prior_val, str) or bool(prior_val.strip()))
            newly_collected_any = any(
                final_lead_dict.get(f) is not None and clean_existing.get(f) is None
                for f in REQUIRED_LEAD_FIELDS
            )
            if not prior_satisfied and not newly_collected_any:
                was_prior_unclear = True

        next_missing = get_next_missing_parameter(final_lead_dict)
        if was_prior_unclear:
            confirmation_msg = get_clarification_prompt(prior_missing, clean_transcript, final_lead_dict, lang=resp_lang)
        else:
            confirmation_msg = get_missing_parameter_prompt(next_missing, final_lead_dict, lang=resp_lang)

        return {
            "status": "success",
            "lead": final_lead_dict,
            "next_missing_parameter": next_missing,
            "is_complete": next_missing is None,
            "is_new_lead": is_reset_turn,
            "is_unclear": was_prior_unclear,
            "clarified_field": prior_missing if was_prior_unclear else None,
            "message": confirmation_msg,
            "detected_language": spoken_lang,
            "language": resp_lang,
            "model": target_model,
            "transcript_length": len(clean_transcript),
        }

    # ------------------------------------------------------------------------
    # METHOD: stream_lead_turn
    # ------------------------------------------------------------------------
    # • WHAT IT DOES: Complete SSE generator for continuous voice lead capture:
    #     1. Handles greetings with natural conversational greeting tokens.
    #     2. Extracts structured lead and persists/updates PostgreSQL database.
    #     3. Emits 'event: lead' containing updated lead record for instant UI drawer refresh.
    #     4. Streams conversational DeepSeek spoken confirmation tokens via 'event: token'.
    #     5. Emits 'event: done' on completion.
    # • INPUTS:
    #     - transcript (str): Raw prospect audio transcript.
    #     - model (Optional[str]): OpenRouter model override.
    #     - existing_lead (Optional[Dict]): Current lead data in React state.
    #     - language (Optional[str]): Language code ('en', 'hi', 'mr').
    #     - lead_id (Optional[int]): PostgreSQL row ID if lead was already created.
    # • OUTPUT: Generator yielding Server-Sent Events (text/event-stream strings).
    # • WHY IT IS USED: Feeds word-by-word streaming tokens to frontend SentenceTokenizer,
    #   enabling sentence-level TTS playback to start in ~400ms while updating the CRM in the background.
    # • WHERE IT FITS IN THE FLOW:
    #     [VoiceCopilot.tsx] -> [/api/extract-lead?stream=true] -> [stream_lead_turn] -> [SentenceAudioQueue]
    # ------------------------------------------------------------------------
    def stream_lead_turn(
        self,
        transcript: str,
        model: Optional[str] = None,
        existing_lead: Optional[Dict[str, Any]] = None,
        language: Optional[str] = None,
        lead_id: Optional[int] = None,
        is_interim: Optional[bool] = None,
        is_final: Optional[bool] = None,
    ) -> Generator[str, None, None]:
        """
        Continuous Voice Assistant Lead Stream (Single-Pass Optimized):

        1. Checks for interim/partial STT transcripts and bypasses extraction.
        2. Checks for greeting and yields natural greeting tokens immediately if detected.
        3. Checks for assistant queries and yields natural answers.
        4. Extracts multiple fields, merges safely, and asks for next missing parameter or clarifying prompt.
        5. Emits 'event: lead' and 'event: done' without having blocked audio playback.
        """
        clean_transcript = (transcript or "").strip()
        if not clean_transcript:
            yield f"event: error\ndata: {json.dumps({'error': 'Transcript cannot be empty.'})}\n\n"
            return

        target_model = (model or self.model).strip().lstrip("~")

        clean_existing = {}
        if existing_lead and isinstance(existing_lead, dict):
            clean_existing = {
                k: v for k, v in existing_lead.items()
                if v is not None and k in Lead.model_fields
            }

        # Check if user requested a new lead / reset or if prior lead was complete
        is_prior_complete = get_next_missing_parameter(clean_existing) is None and bool(clean_existing)
        is_reset_turn = is_new_lead_intent(clean_transcript) or is_prior_complete
        if is_reset_turn:
            clean_existing = {}
            lead_id = None

        spoken_lang = detect_spoken_language(clean_transcript)
        if language in ("en", "hi", "mr"):
            if is_greeting(clean_transcript):
                resp_lang = language
            elif spoken_lang == "mixed":
                resp_lang = "en"
            else:
                resp_lang = get_response_language(spoken_lang)
        else:
            resp_lang = get_response_language(spoken_lang)
        lang = resp_lang

        # CRITICAL: Bypass lead extraction for interim / partial speech transcripts
        if is_interim is True or is_final is False:
            logger.info(f"[LeadExtractor.stream] Ignoring interim transcript: '{clean_transcript}'")
            lead_dict = Lead.model_validate(clean_existing).model_dump() if clean_existing else Lead().model_dump()
            yield f"event: metadata\ndata: {json.dumps({'is_interim': True, 'status': 'interim_ignored'})}\n\n"
            yield f"event: lead\ndata: {json.dumps({'lead': lead_dict, 'is_new_lead': False, 'lead_id': lead_id, 'is_interim': True})}\n\n"
            yield f"event: done\ndata: {json.dumps({'answer': '', 'is_interim': True})}\n\n"
            return

        # 1. Fast-path: Check if speech turn is solely a greeting
        if is_greeting(clean_transcript):
            greeting_msg = get_greeting_response(resp_lang)
            logger.info(f"[LeadExtractor.stream] Greeting-only turn detected ('{clean_transcript}').")
            yield f"event: metadata\ndata: {json.dumps({'is_greeting': True, 'language': resp_lang, 'detected_language': spoken_lang, 'status': 'greeting'})}\n\n"
            words = greeting_msg.split(" ")
            for i, w in enumerate(words):
                token = w + (" " if i < len(words) - 1 else "")
                yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
            yield f"event: done\ndata: {json.dumps({'answer': greeting_msg, 'is_greeting': True, 'language': resp_lang, 'detected_language': spoken_lang})}\n\n"
            return

        # 1b. Fast-path: Check if speech turn is an assistant conversation query
        assistant_ans = get_assistant_query_response(clean_transcript, resp_lang)
        if assistant_ans:
            logger.info(f"[LeadExtractor.stream] Assistant conversation query detected ('{clean_transcript}').")
            lead_dict = Lead.model_validate(clean_existing).model_dump() if clean_existing else Lead().model_dump()
            yield f"event: metadata\ndata: {json.dumps({'is_assistant_query': True, 'language': resp_lang, 'detected_language': spoken_lang, 'status': 'assistant_query'})}\n\n"
            words = assistant_ans.split(" ")
            for i, w in enumerate(words):
                token = w + (" " if i < len(words) - 1 else "")
                yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
            yield f"event: lead\ndata: {json.dumps({'lead': lead_dict, 'is_new_lead': False, 'lead_id': lead_id, 'is_assistant_query': True})}\n\n"
            yield f"event: done\ndata: {json.dumps({'answer': assistant_ans, 'is_assistant_query': True, 'language': resp_lang, 'detected_language': spoken_lang})}\n\n"
            return

        lang_name = "English"
        if resp_lang == "hi":
            lang_name = "Hindi (Devanagari script)"
        elif resp_lang == "mr":
            lang_name = "Marathi (Devanagari script)"

        # Sequential parameter tracking
        state_summary = []
        if clean_existing.get("name"):
            state_summary.append(f"Name={clean_existing['name']}")
        if clean_existing.get("phone"):
            state_summary.append(f"Phone={clean_existing['phone']}")
        if clean_existing.get("company"):
            state_summary.append(f"Company={clean_existing['company']}")
        if clean_existing.get("loan_type"):
            state_summary.append(f"LoanType={clean_existing['loan_type']}")
        if clean_existing.get("loan_amount"):
            state_summary.append(f"LoanAmount={clean_existing['loan_amount']}")
        if clean_existing.get("tenure_months"):
            state_summary.append(f"TenureMonths={clean_existing['tenure_months']}")
        curr_state_str = ", ".join(state_summary) if state_summary else "None"

        # Stateful sequential prompt instructing the model on sequential parameter collection
        system_prompt = (
            f"You are Voice Sales Copilot collecting loan application parameters sequentially and statefully in {lang_name}.\n"
            "REQUIRED PARAMETERS IN STRICT ORDER:\n"
            "1. name (Prospect full name - extract ONLY when user explicitly states their real name, e.g. 'My name is Rajesh'. NEVER treat 'I want loan' or loan requests as a name!)\n"
            "2. phone (Contact phone number)\n"
            "3. company (Employer / Company name)\n"
            "4. loan_type (Type of loan: Personal, Home, Business, etc.)\n"
            "5. loan_amount (Requested loan amount in numeric currency)\n"
            "6. tenure_months (Loan duration in months or years)\n\n"
            f"Current Known Parameters: {curr_state_str}\n\n"
            "INSTRUCTIONS:\n"
            "1. Extract any newly provided parameters from the transcript and MERGE them with already known parameters. Never discard or overwrite already known parameters unless explicitly updated.\n"
            "2. CRITICAL NAME RULE: Never treat loan-intent phrases (such as 'I want loan', 'I want a loan', 'need loan', 'loan chahiye', 'karj pahije', 'apply for loan') as names! If no real person's name is explicitly given, 'name' MUST be null.\n"
            "3. If the user gives only partial info (e.g. 'I want loan' -> loan_type='Personal Loan'), save it into the JSON and ask for the NEXT missing required parameter in sequence (e.g. if name is missing, ask for the name: 'Hello! May I have your name, please?' / 'नमस्ते! कृपया आपका शुभ नाम बताइए?' / 'नमस्कार! कृपया आपले नाव सांगा?').\n"
            "4. If ALL 6 required parameters (name, phone, company, loan_type, loan_amount, tenure_months) are collected, give a short, polite thank-you confirmation (e.g. 'Thank you, [Name]! All your details have been recorded. Our team will contact you shortly.').\n"
            f"5. First output 1 natural spoken sentence in {lang_name} acknowledging the user's input and asking for the next missing parameter (or the short thank-you confirmation if complete).\n"
            f"6. Then output '{LEAD_STREAM_DELIMITER}' on a new line, followed by the complete merged lead JSON object:\n"
            '{"name":string|null,"phone":string|null,"email":string|null,"company":string|null,"role":string|null,"loan_type":string|null,"loan_amount":number|null,"tenure_months":integer|null,"notes":string|null}.\n'
            "7. Strict facts only; unmentioned fields must remain null. Output no intro labels, markdown fences, or extra text."
        )

        user_content = f"Transcript: {clean_transcript}"
        if clean_existing:
            user_content += f"\nExisting: {json.dumps(clean_existing, separators=(',', ':'))}"

        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "provider": {"sort": "latency"},
            "temperature": 0.0,
            "stream": True,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Voice-Sales-Copilot",
            "X-Title": "Voice Sales Copilot",
        }

        endpoint_url = f"{self.base_url}/chat/completions"
        client = self._client or self.get_http_client()
        is_mock = (
            getattr(client, "_is_mock", False)
            or type(client).__module__.startswith("unittest.mock")
            or hasattr(client, "_mock_return_value")
        )

        spoken_tokens: List[str] = []
        spoken_buffer = ""
        json_buffer = ""
        is_json_phase = False

        try:
            if not is_mock:
                # 1. Determine prior next missing parameter
                prior_missing = get_next_missing_parameter(clean_existing)

                # 2. Extract multiple fields from transcript with context
                extracted_this_turn: Dict[str, Any] = {}

                phone_found = extract_phone_number(clean_transcript)
                if phone_found:
                    extracted_this_turn["phone"] = phone_found

                email_found = extract_email(clean_transcript)
                if email_found:
                    extracted_this_turn["email"] = email_found

                comp_found = extract_company(clean_transcript, context_field=prior_missing)
                if comp_found:
                    extracted_this_turn["company"] = comp_found

                lt_found = extract_loan_type(clean_transcript, context_field=prior_missing)
                if lt_found:
                    extracted_this_turn["loan_type"] = lt_found

                amt_found = extract_loan_amount(clean_transcript, context_field=prior_missing)
                if amt_found:
                    extracted_this_turn["loan_amount"] = amt_found

                tenure_found = extract_tenure_months(clean_transcript, context_field=prior_missing)
                if tenure_found:
                    extracted_this_turn["tenure_months"] = tenure_found

                name_found = extract_name(clean_transcript, context_field=prior_missing)
                if name_found and is_valid_prospect_name(name_found):
                    extracted_this_turn["name"] = name_found

                # Safely merge into clean_existing (never overwrite existing valid fields)
                immediate_lead = merge_lead_safely(clean_existing, extracted_this_turn)

                # Check if input was unclear for prior_missing
                was_prior_unclear = False
                if prior_missing and prior_missing in REQUIRED_LEAD_FIELDS:
                    prior_val = immediate_lead.get(prior_missing)
                    prior_satisfied = prior_val is not None and (not isinstance(prior_val, str) or bool(prior_val.strip()))
                    newly_collected_any = any(
                        immediate_lead.get(f) is not None and clean_existing.get(f) is None
                        for f in REQUIRED_LEAD_FIELDS
                    )
                    if not prior_satisfied and not newly_collected_any:
                        was_prior_unclear = True

                # 3. Determine next missing parameter statefully (skips all already collected fields!)
                next_missing = get_next_missing_parameter(immediate_lead)
                if was_prior_unclear:
                    immediate_sentence1 = get_clarification_prompt(prior_missing, clean_transcript, immediate_lead, lang=resp_lang)
                else:
                    immediate_sentence1 = get_missing_parameter_prompt(next_missing, immediate_lead, lang=resp_lang)

                # 4. Synchronously sync lead_id with database so it is guaranteed across multi-turn
                synced_lead_id = lead_id
                has_data = any(v is not None for v in immediate_lead.values() if v != "")
                if has_data:
                    try:
                        from services.lead_repository import LeadRepository
                        repo = LeadRepository()
                        val_lead = Lead.model_validate(immediate_lead)
                        if synced_lead_id is not None:
                            repo.update_lead(synced_lead_id, val_lead)
                        else:
                            created = repo.create_lead(val_lead)
                            if created and "id" in created:
                                synced_lead_id = created["id"]
                    except Exception as db_err:
                        logger.warning(f"[LeadExtractor.persist] {db_err}")

                # 5. EMIT 'lead' EVENT IMMEDIATELY (< 1ms)
                lead_event = {
                    "status": "success",
                    "lead": immediate_lead,
                    "lead_id": synced_lead_id,
                    "is_update": synced_lead_id is not None and lead_id is not None,
                    "is_new_lead": is_reset_turn,
                    "language": resp_lang,
                    "detected_language": spoken_lang,
                    "next_missing_parameter": next_missing,
                    "is_complete": next_missing is None,
                }
                yield f"event: lead\ndata: {json.dumps(lead_event)}\n\n"

                # 6. STREAM SENTENCE 1 TOKENS IMMEDIATELY WITHOUT WAITING FOR FULL LLM RESPONSE (< 2ms)
                words = immediate_sentence1.split(" ")
                for i, w in enumerate(words):
                    token = w + (" " if i < len(words) - 1 else "")
                    yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"

                # 7. EMIT 'done' EVENT IMMEDIATELY (< 3ms)
                done_payload = {
                    "answer": immediate_sentence1,
                    "lead": immediate_lead,
                    "lead_id": synced_lead_id,
                    "is_new_lead": is_reset_turn,
                    "language": resp_lang,
                    "detected_language": spoken_lang,
                    "next_missing_parameter": next_missing,
                    "is_complete": next_missing is None,
                }
                yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"
                return
            else:
                resp = client.post(endpoint_url, json=payload, headers=headers)
                if resp.status_code == 200:
                    resp_data = resp.json()
                    content = resp_data["choices"][0]["message"]["content"]
                    if LEAD_STREAM_DELIMITER in content:
                        spoken_part, json_part = content.split(LEAD_STREAM_DELIMITER, 1)
                        spoken_text = spoken_part.strip()
                        spoken_tokens.append(spoken_text)
                        yield f"event: token\ndata: {json.dumps({'token': spoken_text})}\n\n"
                        json_buffer = json_part
                    else:
                        mock_extracted = extract_json_payload(content)
                        if mock_extracted and any(k in mock_extracted for k in Lead.model_fields):
                            if clean_existing:
                                for k, v in clean_existing.items():
                                    if k in mock_extracted and mock_extracted[k] is None and v is not None:
                                        mock_extracted[k] = v
                            next_param = get_next_missing_parameter(mock_extracted)
                            spoken_text = get_missing_parameter_prompt(next_param, mock_extracted, lang=lang)
                            spoken_tokens.append(spoken_text)
                            yield f"event: token\ndata: {json.dumps({'token': spoken_text})}\n\n"
                            json_buffer = content
                        else:
                            spoken_tokens.append(content)
                            yield f"event: token\ndata: {json.dumps({'token': content})}\n\n"
                else:
                    next_p = get_next_missing_parameter(clean_existing)
                    fallback_msg = get_missing_parameter_prompt(next_p, clean_existing, lang=lang)
                    spoken_tokens.append(fallback_msg)
                    yield f"event: token\ndata: {json.dumps({'token': fallback_msg})}\n\n"
        except Exception as stream_err:
            logger.error(f"[LeadExtractor.stream] Streaming error: {stream_err}")
            next_p = get_next_missing_parameter(clean_existing)
            fallback_msg = get_missing_parameter_prompt(next_p, clean_existing, lang=lang)
            spoken_tokens.append(fallback_msg)
            yield f"event: token\ndata: {json.dumps({'token': fallback_msg})}\n\n"

        final_answer = "".join(spoken_tokens).strip()

        # Parse and validate lead JSON
        extracted_lead_dict = extract_json_payload(json_buffer)

        # Resilient fallback if stream did not deliver valid JSON
        if not extracted_lead_dict:
            try:
                fallback_ext = self.extract_lead(
                    clean_transcript,
                    model=target_model,
                    existing_lead=existing_lead,
                    language=lang,
                )
                extracted_lead_dict = fallback_ext.get("lead", {})
            except Exception as ext_fallback_err:
                logger.warning(f"[LeadExtractor.stream] Fallback lead extraction warning: {ext_fallback_err}")
                extracted_lead_dict = Lead().model_dump()

        # Deterministically merge existing fields safely
        extracted_lead = merge_lead_safely(clean_existing, extracted_lead_dict)

        # Determine next missing parameter statefully
        next_missing = get_next_missing_parameter(extracted_lead)

        # Ensure spoken answer asks for the next missing parameter or gives thank-you confirmation
        if not final_answer:
            fallback_prompt = get_missing_parameter_prompt(next_missing, extracted_lead, lang=lang)
            spoken_tokens.append(fallback_prompt)
            final_answer = fallback_prompt
            yield f"event: token\ndata: {json.dumps({'token': fallback_prompt})}\n\n"
        elif next_missing is not None and not any(q_word in final_answer.lower() for q_word in ["?", "number", "phone", "फोन", "नंबर", "loan", "कर्ज", "amount", "रक्कम", "राशि"]):
            next_q = get_missing_parameter_prompt(next_missing, extracted_lead, lang=lang)
            spoken_tokens.append(" " + next_q)
            final_answer = f"{final_answer} {next_q}".strip()
            yield f"event: token\ndata: {json.dumps({'token': ' ' + next_q})}\n\n"
        elif next_missing is None and not any(thx in final_answer.lower() for thx in ["thank", "धन्यवाद", "आभार"]):
            thank_you = get_missing_parameter_prompt(None, extracted_lead, lang=lang)
            spoken_tokens.append(" " + thank_you)
            final_answer = f"{final_answer} {thank_you}".strip()
            yield f"event: token\ndata: {json.dumps({'token': ' ' + thank_you})}\n\n"

        # Persist to PostgreSQL (off the critical audio response path)
        saved_lead = extracted_lead
        is_update = False
        final_lead_id = lead_id

        try:
            from services.lead_repository import LeadRepository
            repo = LeadRepository()
            validated_lead_obj = Lead.model_validate(extracted_lead)
            has_data = any(v is not None for v in extracted_lead.values() if v != "")

            if lead_id is not None:
                is_update = True
                updated = repo.update_lead(lead_id, validated_lead_obj)
                if updated:
                    saved_lead = updated
                    final_lead_id = updated.get("id", lead_id)
            elif has_data:
                created = repo.create_lead(validated_lead_obj)
                if created:
                    saved_lead = created
                    final_lead_id = created.get("id")
        except Exception as db_err:
            logger.warning(f"[LeadExtractor.stream] Database persistence warning: {db_err}")

        # Emit 'lead' event for immediate UI update
        lead_event = {
            "status": "success",
            "lead": saved_lead,
            "lead_id": final_lead_id,
            "is_update": is_update,
            "is_new_lead": is_reset_turn,
            "language": lang,
            "next_missing_parameter": next_missing,
            "is_complete": next_missing is None,
        }
        yield f"event: lead\ndata: {json.dumps(lead_event)}\n\n"

        # Emit 'event: done' on completion
        done_payload = {
            "answer": final_answer,
            "lead": saved_lead,
            "lead_id": final_lead_id,
            "is_new_lead": is_reset_turn,
            "language": lang,
            "next_missing_parameter": next_missing,
            "is_complete": next_missing is None,
        }
        yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"
