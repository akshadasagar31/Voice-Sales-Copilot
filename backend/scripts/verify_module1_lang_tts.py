import asyncio
import os
import sys
import io
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import httpx
from services.language import detect_language
from services.lead_extractor import get_missing_parameter_prompt, get_next_missing_parameter

async def verify_pipeline():
    print("=== STARTING END-TO-END MODULE 1 VERIFICATION ===")
    
    # 1. English speech test
    en_transcript = "My name is Rajesh, phone number 9876543210, looking for personal loan 5 lakh"
    en_lang = detect_language(en_transcript)
    assert en_lang == "en", f"Expected 'en', got '{en_lang}'"
    print(f"[OK] English text detection: '{en_transcript}' -> {en_lang}")

    # Verify English prompt generation
    next_param = "company"
    lead_data = {"name": "Rajesh", "phone": "9876543210", "loan_type": "Personal Loan", "loan_amount": 500000}
    en_prompt = get_missing_parameter_prompt(next_param, lead_data, lang=en_lang)
    assert "company" in en_prompt.lower() or "organization" in en_prompt.lower()
    assert not any('\u0900' <= c <= '\u097f' for c in en_prompt), "English prompt contained Devanagari!"
    print(f"[OK] English question generated: '{en_prompt}'")

    # 2. Hindi speech test
    hi_transcript = "मेरा नाम राजेश है, मुझे 5 लाख का पर्सनल लोन चाहिए"
    hi_lang = detect_language(hi_transcript)
    assert hi_lang == "hi", f"Expected 'hi', got '{hi_lang}'"
    print(f"[OK] Hindi text detection: '{hi_transcript}' -> {hi_lang}")

    hi_prompt = get_missing_parameter_prompt(next_param, lead_data, lang=hi_lang)
    assert any('\u0900' <= c <= '\u097f' for c in hi_prompt), "Hindi prompt must contain Devanagari"
    assert "कंपनी" in hi_prompt
    print(f"[OK] Hindi question generated: '{hi_prompt}'")

    # 3. Marathi speech test
    mr_transcript = "नमस्कार, माझे नाव राजेश आहे आणि मला 5 लाख रुपये वैयक्तिक कर्ज पाहिजे"
    mr_lang = detect_language(mr_transcript)
    assert mr_lang == "mr", f"Expected 'mr', got '{mr_lang}'"
    print(f"[OK] Marathi text detection: '{mr_transcript}' -> {mr_lang}")

    mr_prompt = get_missing_parameter_prompt(next_param, lead_data, lang=mr_lang)
    assert any('\u0900' <= c <= '\u097f' for c in mr_prompt)
    assert "कोणत्या कंपनीमध्ये" in mr_prompt or "कंपनी" in mr_prompt
    print(f"[OK] Marathi question generated: '{mr_prompt}'")

    # 4. Mixed Hinglish speech test
    mixed_hi_transcript = "Mera naam Rajesh hai, mujhe personal loan chahiye"
    mixed_hi_lang = detect_language(mixed_hi_transcript)
    assert mixed_hi_lang == "hi", f"Expected 'hi', got '{mixed_hi_lang}'"
    print(f"[OK] Mixed Hinglish detection: '{mixed_hi_transcript}' -> {mixed_hi_lang}")

    # 5. Mixed Marathi speech test
    mixed_mr_transcript = "Majha naav Rajesh aahe, mala personal loan pahije"
    mixed_mr_lang = detect_language(mixed_mr_transcript)
    assert mixed_mr_lang == "mr", f"Expected 'mr', got '{mixed_mr_lang}'"
    print(f"[OK] Mixed Marathi detection: '{mixed_mr_transcript}' -> {mixed_mr_lang}")

    # 6. Test FastAPI /api/tts endpoint with Module 1 payload for each language
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8001", timeout=30.0) as client:
        for lang_code, prompt_text in [("en", en_prompt), ("hi", hi_prompt), ("mr", mr_prompt)]:
            resp = await client.post(
                "/api/tts",
                json={
                    "text": prompt_text,
                    "language": lang_code,
                    "module": "module1",
                    "speaker": "simran"
                }
            )
            assert resp.status_code == 200, f"TTS failed with status {resp.status_code}: {resp.text}"
            audio_bytes = resp.content
            assert len(audio_bytes) > 500, f"Audio bytes too small: {len(audio_bytes)}"
            provider = resp.headers.get("x-tts-provider")
            print(f"[OK] Module 1 TTS response for {lang_code}: {len(audio_bytes)} bytes audio, provider: {provider}")

    print("=== ALL VERIFICATION CHECKS PASSED SUCCESSFULLY ===")

if __name__ == "__main__":
    asyncio.run(verify_pipeline())
