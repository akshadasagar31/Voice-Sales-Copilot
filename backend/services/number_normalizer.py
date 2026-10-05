# ============================================================================
# NUMBER NORMALIZER FOR ENGLISH, HINDI, MARATHI & MIXED SPEECH
# (backend/services/number_normalizer.py)
# ============================================================================
# Converts spoken number words in English, Hindi (Devanagari & Romanized),
# and Marathi (Devanagari & Romanized) into numeric digits.
#
# Examples:
#   "thirty five months" -> "35 months"
#   "पैंतीस महीने" -> "35 महीने"
#   "पस्तीस महिने" -> "35 महिने"
#   "nine eight seven six five four three two one zero" -> "9876543210"
#   "twenty five lakh" -> "25 lakh"
#   "पच्चीस लाख" -> "25 लाख"
#   "पंचवीस लाख" -> "25 लाख"
# ============================================================================

import re
from typing import Optional, Dict, Tuple

# Unit and small numbers: 0 to 99 across English, Hindi, and Marathi
CARDINAL_NUMBERS: Dict[str, int] = {
    # English 0-19
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    # English tens
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,

    # Hindi Devanagari 0 to 99
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4,
    "पाँच": 5, "पांच": 5, "छह": 6, "छः": 6, "सात": 7, "आठ": 8, "नौ": 9,
    "दस": 10, "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14,
    "पंद्रह": 15, "सोलह": 16, "सत्रह": 17, "अठारह": 18, "उन्नीस": 19,
    "बीस": 20, "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24,
    "पच्चीस": 25, "छब्बीस": 26, "सत्ताईस": 27, "अट्ठाईस": 28, "उनतीस": 29,
    "तीस": 30, "इकत्तीस": 31, "बत्तीस": 32, "तैंतीस": 33, "चौंतीस": 34,
    "पैंतीस": 35, "छत्तीस": 36, "सैंतीस": 37, "अड़तीस": 38, "उनतालीस": 39,
    "चालीस": 40, "इकतालीस": 41, "बयालीस": 42, "तैंतालीस": 43, "चवालीस": 44,
    "पैंतालीस": 45, "छियालीस": 46, "सैंतालीस": 47, "अड़तालीस": 48, "उनचास": 49,
    "पचास": 50, "इक्यावन": 51, "बावन": 52, "तिरेपन": 53, "चौवन": 54,
    "पचपन": 55, "छप्पन": 56, "सत्तावन": 57, "अट्ठावन": 58, "उनसठ": 59,
    "साठ": 60, "इकसठ": 61, "बासठ": 62, "तिरसठ": 63, "चौंसठ": 64,
    "पैंसठ": 65, "छियासठ": 66, "सरसठ": 67, "अड़सठ": 68, "उनहत्तर": 69,
    "सत्तर": 70, "इकहत्तर": 71, "बहत्तर": 72, "तिहत्तर": 73, "चौहत्तर": 74,
    "पचहत्तर": 75, "छिहत्तर": 76, "सतहत्तर": 77, "अठहत्तर": 78, "उन्नासी": 79,
    "अस्सी": 80, "इक्यासी": 81, "बयासी": 82, "तिरासी": 83, "चौरासी": 84,
    "पचासी": 85, "छियासी": 86, "सत्तासी": 87, "अट्ठासी": 88, "नवासी": 89,
    "नब्बे": 90, "इक्यानवे": 91, "बानवे": 92, "तिरानवे": 93, "चौरानवे": 94,
    "पंचानवे": 95, "छियानवे": 96, "सत्तानवे": 97, "अट्ठानवे": 98, "निन्यानवे": 99,

    # Marathi Devanagari distinct numbers
    "दोन": 2, "पाच": 5, "सहा": 6, "नऊ": 9, "दहा": 10,
    "अकरा": 11, "बारा": 12, "तेरा": 13, "चौदा": 14, "पंधरा": 15,
    "सोळा": 16, "सतरा": 17, "अठरा": 18, "एकोणीस": 19, "वीस": 20,
    "एकवीस": 21, "बावीस": 22, "तेवीस": 23, "चोवीस": 24, "पंचवीस": 25,
    "सव्वीस": 26, "सत्तावीस": 27, "अठ्ठावीस": 28, "एकोणतीस": 29, "तीस": 30,
    "एकतीस": 31, "बत्तीस": 32, "तेहतीस": 33, "चौतीस": 34, "पस्तीस": 35,
    "छत्तीस": 36, "सदतीस": 37, "अडतीस": 38, "एकोणचाळीस": 39, "चाळीस": 40,
    "एक्केचाळीस": 41, "बेचाळीस": 42, "त्रेचाळीस": 43, "चव्वेचाळीस": 44, "पंचेचाळीस": 45,
    "शेहेचाळीस": 46, "सत्तेचाळीस": 47, "अठ्ठेचाळीस": 48, "एकोणपन्नास": 49, "पन्नास": 50,
    "एक्कावन्न": 51, "बावन्न": 52, "त्रेपन्न": 53, "चोपन्न": 54, "पंचावन्न": 55,
    "छप्पन्न": 56, "सत्तावन्न": 57, "अठ्ठावन्न": 58, "एकोणसाठ": 59, "साठ": 60,
    "एकसष्ठ": 61, "बासष्ठ": 62, "त्रेसष्ठ": 63, "चौसष्ठ": 64, "पासष्ठ": 65,
    "सहासष्ठ": 66, "सदुसष्ठ": 67, "अडुसष्ठ": 68, "एकोणसत्तर": 69, "सत्तर": 70,
    "एकाहत्तर": 71, "बाहत्तर": 72, "त्र्याहत्तर": 73, "चौर्‍याहत्तर": 74, "पंच्याहत्तर": 75,
    "शहात्तर": 76, "सत्त्याहत्तर": 77, "अठ्ठ्याहत्तर": 78, "एकोणऐंशी": 79, "ऐंशी": 80,
    "एक्याऐंशी": 81, "ब्याऐंशी": 82, "त्र्याऐंशी": 83, "चौऱ्याऐंशी": 84, "पंच्याऐंशी": 85,
    "शहाऐंशी": 86, "सत्त्याऐंशी": 87, "अठ्ठ्याऐंशी": 88, "एकोणनव्वद": 89, "नव्वद": 90,
    "एक्याण्णव": 91, "ब्याण्णव": 92, "त्र्याण्णव": 93, "चौऱ्याण्णव": 94, "पंच्याण्णव": 95,
    "शहाण्णव": 96, "सत्त्याण्णव": 97, "अठ्ठ्याण्णव": 98, "नव्याण्णव": 99,

    # Romanized Hindi & Marathi key numerals
    "shunya": 0, "ek": 1, "do": 2, "don": 2, "teen": 3, "chaar": 4, "char": 4,
    "paanch": 5, "panch": 5, "paach": 5, "pach": 5, "chhah": 6, "chhe": 6, "saha": 6,
    "saat": 7, "sat": 7, "aath": 8, "ath": 8, "nau": 9, "das": 10, "daha": 10,
    "gyarah": 11, "barah": 12, "bara": 12, "terah": 13, "tera": 13, "chaudah": 14, "chauda": 14,
    "pandrah": 15, "pandhara": 15, "solah": 16, "sola": 16, "satrah": 17, "satara": 17,
    "atharah": 18, "athara": 18, "unnees": 19, "ekonees": 19, "bees": 20, "vees": 20,
    "chaubees": 24, "chovees": 24, "pachchees": 25, "panchvees": 25, "tees": 30,
    "paintees": 35, "pastees": 35, "chalees": 40, "pachaas": 50, "pannaas": 50,
    "saath": 60, "sattar": 70, "assee": 80, "aishi": 80, "nabbe": 90, "navvad": 90,
}

# Scale multipliers
SCALE_MULTIPLIERS: Dict[str, int] = {
    # English
    "hundred": 100,
    "thousand": 1000,
    "lakh": 100000,
    "lakhs": 100000,
    "lac": 100000,
    "lacs": 100000,
    "crore": 10000000,
    "crores": 10000000,
    "cr": 10000000,
    "million": 1000000,
    "millions": 1000000,

    # Hindi Devanagari
    "सौ": 100,
    "हज़ार": 1000,
    "हजार": 1000,
    "लाख": 100000,
    "करोड़": 10000000,
    "करोड": 10000000,

    # Marathi Devanagari
    "शंभर": 100,
    "कोटी": 10000000,

    # Romanized
    "sau": 100,
    "shambhar": 100,
    "hazaar": 1000,
    "hajaar": 1000,
    "laakh": 100000,
    "koti": 10000000,
}

# Indian fractional number words
FRACTIONAL_WORDS: Dict[str, float] = {
    # 1.5
    "डेढ़": 1.5,
    "डेढ": 1.5,
    "dedh": 1.5,
    "दीड": 1.5,
    "deed": 1.5,
    # 2.5
    "ढाई": 2.5,
    "dhai": 2.5,
    "अडीच": 2.5,
    "adeech": 2.5,
    # 1.25
    "सव्वा": 1.25,
    "savva": 1.25,
    "सवा": 1.25,
    "sawa": 1.25,
    # 0.75
    "पावणे": 0.75,
    "पौने": 0.75,
    "paune": 0.75,
}

# Prefix modifiers
HALF_PREFIXES = {"साढ़े", "साडे", "saadhe", "saade"}
QUARTER_LESS_PREFIXES = {"पावणे", "paune", "पौने"}
QUARTER_MORE_PREFIXES = {"सव्वा", "savva", "सवा", "sawa"}

# English tens words that combine with units
ENGLISH_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
ENGLISH_UNITS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}


def parse_compound_english_number(words: list[str], start_idx: int) -> Tuple[Optional[int], int]:
    """
    Parses compound English numbers like 'thirty five' or 'thirty-five'.
    Returns (parsed_value, words_consumed).
    """
    if start_idx >= len(words):
        return None, 0

    first = words[start_idx].lower().replace("-", " ")
    parts = first.split()

    if len(parts) == 2:
        ten, unit = parts[0], parts[1]
        if ten in ENGLISH_TENS and unit in ENGLISH_UNITS:
            return ENGLISH_TENS[ten] + ENGLISH_UNITS[unit], 1

    if first in ENGLISH_TENS:
        if start_idx + 1 < len(words):
            second = words[start_idx + 1].lower()
            if second in ENGLISH_UNITS:
                return ENGLISH_TENS[first] + ENGLISH_UNITS[second], 2
        return ENGLISH_TENS[first], 1

    if first in CARDINAL_NUMBERS:
        return CARDINAL_NUMBERS[first], 1

    return None, 0


REPEATER_WORDS: Dict[str, int] = {
    "double": 2,
    "triple": 3,
    "treble": 3,
    "डबल": 2,
    "ट्रिपल": 3,
}

DEVANAGARI_DIGITS_MAP = str.maketrans("०१२३४५६७८९", "0123456789")


def _get_single_digit(word: str) -> Optional[str]:
    """Returns single digit string '0'-'9' if word represents a single digit in EN, HI, MR."""
    w = word.strip().lower()
    if w.isdigit() and len(w) == 1:
        return w
    if w in CARDINAL_NUMBERS and 0 <= CARDINAL_NUMBERS[w] <= 9:
        return str(CARDINAL_NUMBERS[w])
    return None


def normalize_spoken_numbers(text: str) -> str:
    """
    Normalizes spoken number words into digits in text across English, Hindi, Marathi, and mixed speech.

    Examples:
        "thirty five months" -> "35 months"
        "पैंतीस महीने" -> "35 महीने"
        "पस्तीस महिने" -> "35 महिने"
        "two years" -> "2 years"
        "तीन साल" -> "3 साल"
        "पाच वर्षे" -> "5 वर्षे"
        "twenty five lakh" -> "25 lakh"
        "पच्चीस लाख" -> "25 लाख"
        "पंचवीस लाख" -> "25 लाख"
        "ढाई लाख" -> "2.5 लाख"
        "डेढ़ लाख" -> "1.5 लाख"
        "दीड वर्ष" -> "1.5 वर्ष"
        "अडीच वर्षे" -> "2.5 वर्षे"
        "nine eight seven six five four three two one zero" -> "9876543210"
        "double nine eight seven six five four three two one" -> "9987654321"
    """
    if not text or not isinstance(text, str):
        return ""

    # Pre-pass 1: Translate Devanagari numerals (०-९ -> 0-9)
    preprocessed = text.translate(DEVANAGARI_DIGITS_MAP)

    # Pre-pass 2: Un-format smart-formatted times inside numeric speech (e.g. 9:00 -> 900, 9:30 -> 930, 2:00 -> 200, 0:00 -> 000)
    preprocessed = re.sub(r"(\b\d{1,2})\s*:\s*(\d{2})\b", r"\1\2", preprocessed)

    # Pre-pass 3: Handle compound Hindi/Marathi forms like 'साडेतीन' -> 'साडे तीन', 'साडेचार' -> 'साडे चार'
    preprocessed = re.sub(r"\bसाडे([एक|दोन|तीन|चार|पाच|सहा|सात|आठ|नऊ|दहा])", r"साडे \1", preprocessed)
    preprocessed = re.sub(r"\bसाढ़े([एक|दो|तीन|चार|पांच|पाँच|छह|सात|आठ|नौ|दस])", r"साढ़े \1", preprocessed)

    tokens = re.split(r"(\s+|[.,;?!])", preprocessed)
    result_tokens = []
    i = 0

    while i < len(tokens):
        token = tokens[i]
        clean_token = token.strip().lower()

        # If whitespace or punctuation, preserve
        if not clean_token or not re.search(r"[\w\u0900-\u097F]", clean_token):
            result_tokens.append(token)
            i += 1
            continue

        # Check for 'double' or 'triple' digit dictation: e.g. "double nine", "triple five", "डबल नौ"
        if clean_token in REPEATER_WORDS and i + 1 < len(tokens):
            multiplier = REPEATER_WORDS[clean_token]
            # Look ahead for digit
            j = i + 1
            while j < len(tokens) and (not tokens[j].strip() or tokens[j].strip() in (",", ".", "-", ";", ":")):
                j += 1
            if j < len(tokens):
                nxt_digit = _get_single_digit(tokens[j].strip().lower())
                if nxt_digit:
                    expanded_digits = nxt_digit * multiplier
                    # Look ahead further for phone sequence
                    seq = [expanded_digits]
                    k = j + 1
                    last_idx = j
                    while k < len(tokens):
                        tk_str = tokens[k].strip()
                        if not tk_str or tk_str in (",", ".", "-", ";", ":"):
                            k += 1
                            continue
                        d = _get_single_digit(tk_str.lower())
                        if d:
                            seq.append(d)
                            last_idx = k
                            k += 1
                        elif tk_str.lower() in REPEATER_WORDS and k + 1 < len(tokens):
                            m_mult = REPEATER_WORDS[tk_str.lower()]
                            next_k = k + 1
                            while next_k < len(tokens) and (not tokens[next_k].strip() or tokens[next_k].strip() in (",", ".", "-", ";", ":")):
                                next_k += 1
                            if next_k < len(tokens):
                                kd = _get_single_digit(tokens[next_k].strip().lower())
                                if kd:
                                    seq.append(kd * m_mult)
                                    last_idx = next_k
                                    k = next_k + 1
                                    continue
                            break
                        else:
                            break
                    joined_seq = "".join(seq)
                    if len(joined_seq) >= 7:
                        result_tokens.append(joined_seq)
                        i = last_idx + 1
                        continue
                    else:
                        result_tokens.append(expanded_digits)
                        i = j + 1
                        continue

        # Check for fractional prefixes: "साढ़े तीन" -> "3.5", "साडे तीन" -> "3.5", "saadhe teen" -> "3.5"
        if clean_token in HALF_PREFIXES:
            j = i + 1
            while j < len(tokens) and tokens[j].isspace():
                j += 1
            if j < len(tokens):
                nxt_word = tokens[j].strip().lower()
                if nxt_word in CARDINAL_NUMBERS:
                    base_val = CARDINAL_NUMBERS[nxt_word]
                    result_tokens.append(f"{base_val + 0.5:g}")
                    i = j + 1
                    continue

        # Check for quarter-less prefix: "पावणे पाच" -> "4.75"
        if clean_token in QUARTER_LESS_PREFIXES:
            j = i + 1
            while j < len(tokens) and tokens[j].isspace():
                j += 1
            if j < len(tokens):
                nxt_word = tokens[j].strip().lower()
                if nxt_word in CARDINAL_NUMBERS:
                    base_val = CARDINAL_NUMBERS[nxt_word]
                    result_tokens.append(f"{base_val - 0.25:g}")
                    i = j + 1
                    continue

        # Check for quarter-more prefix: "सव्वा दोन" -> "2.25", or standalone "सव्वा" before scale (e.g. "सव्वा लाख" -> "1.25 लाख")
        if clean_token in QUARTER_MORE_PREFIXES:
            j = i + 1
            while j < len(tokens) and tokens[j].isspace():
                j += 1
            if j < len(tokens):
                nxt_word = tokens[j].strip().lower()
                if nxt_word in CARDINAL_NUMBERS:
                    base_val = CARDINAL_NUMBERS[nxt_word]
                    result_tokens.append(f"{base_val + 0.25:g}")
                    i = j + 1
                    continue
                elif nxt_word in SCALE_MULTIPLIERS:
                    result_tokens.append("1.25")
                    i += 1
                    continue

        # Check standalone fractional words: "ढाई" -> "2.5", "डेढ़" -> "1.5", "दीड" -> "1.5", "अडीच" -> "2.5"
        if clean_token in FRACTIONAL_WORDS:
            frac_val = FRACTIONAL_WORDS[clean_token]
            result_tokens.append(f"{frac_val:g}")
            i += 1
            continue

        # Check for hyphenated compound like 'thirty-five'
        if "-" in clean_token:
            parts = clean_token.split("-")
            if len(parts) == 2 and parts[0] in ENGLISH_TENS and parts[1] in ENGLISH_UNITS:
                result_tokens.append(str(ENGLISH_TENS[parts[0]] + ENGLISH_UNITS[parts[1]]))
                i += 1
                continue

        # Look ahead for English two-word numbers: e.g. "thirty", " ", "five"
        if clean_token in ENGLISH_TENS:
            j = i + 1
            while j < len(tokens) and tokens[j].isspace():
                j += 1
            if j < len(tokens) and tokens[j].strip().lower() in ENGLISH_UNITS:
                val = ENGLISH_TENS[clean_token] + ENGLISH_UNITS[tokens[j].strip().lower()]
                result_tokens.append(str(val))
                i = j + 1
                continue
            else:
                result_tokens.append(str(ENGLISH_TENS[clean_token]))
                i += 1
                continue

        # Check single-digit word sequence for phone numbers (e.g. 7-10 consecutive spoken/written digits)
        first_d = _get_single_digit(clean_token)
        if first_d is not None:
            digit_seq = [first_d]
            last_digit_idx = i
            temp_j = i + 1
            while temp_j < len(tokens):
                nxt_str = tokens[temp_j].strip()
                if not nxt_str or nxt_str in (",", ".", "-", ";", ":"):
                    temp_j += 1
                    continue
                d = _get_single_digit(nxt_str.lower())
                if d is not None:
                    digit_seq.append(d)
                    last_digit_idx = temp_j
                    temp_j += 1
                elif nxt_str.lower() in REPEATER_WORDS and temp_j + 1 < len(tokens):
                    mult = REPEATER_WORDS[nxt_str.lower()]
                    # find next digit
                    k = temp_j + 1
                    while k < len(tokens) and (not tokens[k].strip() or tokens[k].strip() in (",", ".", "-", ";", ":")):
                        k += 1
                    if k < len(tokens):
                        kd = _get_single_digit(tokens[k].strip().lower())
                        if kd is not None:
                            digit_seq.append(kd * mult)
                            last_digit_idx = k
                            temp_j = k + 1
                            continue
                    break
                else:
                    break

            if len("".join(digit_seq)) >= 7:  # Long sequence of digits, clearly a phone number!
                result_tokens.append("".join(digit_seq))
                i = last_digit_idx + 1
                continue

        # Check cardinal single-word numbers (English, Hindi, Marathi)
        if clean_token in CARDINAL_NUMBERS:
            val = CARDINAL_NUMBERS[clean_token]
            result_tokens.append(str(val))
            i += 1
            continue

        result_tokens.append(token)
        i += 1

    return "".join(result_tokens)


def parse_numeric_phrase(text: str) -> Optional[float]:
    """
    Parses a spoken or numeric phrase into a number.
    e.g., 'thirty five' -> 35.0, '25 lakh' -> 2500000.0, 'पैंतीस' -> 35.0,
          'ढाई लाख' -> 250000.0, 'दीड लाख' -> 150000.0
    """
    if not text:
        return None
    normalized = normalize_spoken_numbers(text.strip())

    # Check for direct number with multiplier (e.g. "25 lakh", "50 thousand", "2 crore")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(crores?|cr|कोटी|करोड़|करोड|lakhs?|lacs?|lac|लाख|thousands?|k|हजार|हज़ार|millions?|m)?", normalized, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        scale = (m.group(2) or "").lower()
        if scale in ("crore", "crores", "cr", "कोटी", "करोड़", "करोड"):
            return val * 10000000.0
        elif scale in ("lakh", "lakhs", "lac", "lacs", "लाख"):
            return val * 100000.0
        elif scale in ("thousand", "thousands", "k", "हजार", "हज़ार"):
            return val * 1000.0
        elif scale in ("million", "millions", "m"):
            return val * 1000000.0
        return val

    return None
