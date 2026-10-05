import httpx
import json

BASE_URL = "http://127.0.0.1:8001"

def test_tts_endpoints():
    client = httpx.Client(base_url=BASE_URL, timeout=30.0)

    print("\n--- 1. Testing Hindi TTS in Module 1 ---")
    resp_hi_m1 = client.post("/api/tts", json={
        "text": "धन्यवाद राजेश शर्मा जी! आपकी सभी जानकारी दर्ज कर ली गई है।",
        "language": "hi",
        "module": "module1"
    })
    print(f"Status: {resp_hi_m1.status_code}")
    print(f"Content-Type: {resp_hi_m1.headers.get('content-type')}")
    print(f"X-TTS-Provider: {resp_hi_m1.headers.get('x-tts-provider')}")
    print(f"X-TTS-Voice: {resp_hi_m1.headers.get('x-tts-voice')}")
    print(f"X-TTS-Language: {resp_hi_m1.headers.get('x-tts-language')}")
    print(f"X-TTS-Fallback: {resp_hi_m1.headers.get('x-tts-fallback')}")
    if resp_hi_m1.headers.get('content-type') == 'application/json':
        print(f"Response JSON: {resp_hi_m1.json()}")
        assert resp_hi_m1.json().get("speaker") == "priya"
        assert resp_hi_m1.json().get("language") == "hi-IN"
    else:
        assert resp_hi_m1.headers.get("x-tts-voice") == "priya"
        assert resp_hi_m1.headers.get("x-tts-language") == "hi-IN"

    print("\n--- 2. Testing Marathi TTS in Module 1 ---")
    resp_mr_m1 = client.post("/api/tts", json={
        "text": "धन्यवाद राजेश शर्मा जी! आपले सर्व आवश्यक तपशील नोंदवले गेले आहेत.",
        "language": "mr",
        "module": "module1"
    })
    print(f"Status: {resp_mr_m1.status_code}")
    print(f"Content-Type: {resp_mr_m1.headers.get('content-type')}")
    print(f"X-TTS-Provider: {resp_mr_m1.headers.get('x-tts-provider')}")
    print(f"X-TTS-Voice: {resp_mr_m1.headers.get('x-tts-voice')}")
    print(f"X-TTS-Language: {resp_mr_m1.headers.get('x-tts-language')}")
    print(f"X-TTS-Fallback: {resp_mr_m1.headers.get('x-tts-fallback')}")
    if resp_mr_m1.headers.get('content-type') == 'application/json':
        print(f"Response JSON: {resp_mr_m1.json()}")
        assert resp_mr_m1.json().get("speaker") == "ritu"
        assert resp_mr_m1.json().get("language") == "mr-IN"
    else:
        assert resp_mr_m1.headers.get("x-tts-voice") == "ritu"
        assert resp_mr_m1.headers.get("x-tts-language") == "mr-IN"

    print("\n--- 3. Testing Hindi TTS in Module 2 ---")
    resp_hi_m2 = client.post("/api/tts", json={
        "text": "एचडीएफसी बैंक में पर्सनल लोन के लिए न्यूनतम सिबिल स्कोर 750 होना चाहिए।",
        "language": "hi",
        "module": "module2"
    })
    print(f"Status: {resp_hi_m2.status_code}")
    print(f"Content-Type: {resp_hi_m2.headers.get('content-type')}")
    print(f"X-TTS-Provider: {resp_hi_m2.headers.get('x-tts-provider')}")
    print(f"X-TTS-Voice: {resp_hi_m2.headers.get('x-tts-voice')}")
    print(f"X-TTS-Language: {resp_hi_m2.headers.get('x-tts-language')}")
    print(f"X-TTS-Fallback: {resp_hi_m2.headers.get('x-tts-fallback')}")
    if resp_hi_m2.headers.get('content-type') == 'application/json':
        print(f"Response JSON: {resp_hi_m2.json()}")
        assert resp_hi_m2.json().get("speaker") == "priya"
        assert resp_hi_m2.json().get("language") == "hi-IN"
    else:
        assert resp_hi_m2.headers.get("x-tts-voice") == "priya"
        assert resp_hi_m2.headers.get("x-tts-language") == "hi-IN"

    print("\n--- 4. Testing Marathi TTS in Module 2 ---")
    resp_mr_m2 = client.post("/api/tts", json={
        "text": "एचडीएफसी बँकेच्या वैयक्तिक कर्जासाठी किमान CIBIL score ७५० आवश्यक आहे.",
        "language": "mr",
        "module": "module2"
    })
    print(f"Status: {resp_mr_m2.status_code}")
    print(f"Content-Type: {resp_mr_m2.headers.get('content-type')}")
    print(f"X-TTS-Provider: {resp_mr_m2.headers.get('x-tts-provider')}")
    print(f"X-TTS-Voice: {resp_mr_m2.headers.get('x-tts-voice')}")
    print(f"X-TTS-Language: {resp_mr_m2.headers.get('x-tts-language')}")
    print(f"X-TTS-Fallback: {resp_mr_m2.headers.get('x-tts-fallback')}")
    if resp_mr_m2.headers.get('content-type') == 'application/json':
        print(f"Response JSON: {resp_mr_m2.json()}")
        assert resp_mr_m2.json().get("speaker") == "ritu"
        assert resp_mr_m2.json().get("language") == "mr-IN"
    else:
        assert resp_mr_m2.headers.get("x-tts-voice") == "ritu"
        assert resp_mr_m2.headers.get("x-tts-language") == "mr-IN"

    print("\n--- 5. Testing English TTS in Module 1 & 2 (Deepgram Aura) ---")
    resp_en = client.post("/api/tts", json={
        "text": "Hello Rajesh, what is your required loan amount?",
        "language": "en",
        "module": "module1"
    })
    print(f"Status: {resp_en.status_code}")
    print(f"Content-Type: {resp_en.headers.get('content-type')}")
    print(f"X-TTS-Provider: {resp_en.headers.get('x-tts-provider')}")
    assert resp_en.status_code == 200
    assert resp_en.headers.get("x-tts-provider") == "deepgram"

    print("\n--- 6. Testing Devanagari text explicitly requesting Deepgram model (Guarded Fallback) ---")
    resp_guarded = client.post("/api/tts", json={
        "text": "नमस्कार मला कर्ज हवे आहे.",
        "language": "mr",
        "model": "aura-asteria-en"
    })
    print(f"Status: {resp_guarded.status_code}")
    print(f"X-TTS-Fallback: {resp_guarded.headers.get('x-tts-fallback')}")
    data = resp_guarded.json() if resp_guarded.headers.get('content-type') == 'application/json' else {}
    print(f"Fallback to browser: {data.get('fallback_to_browser')}")
    print(f"Speaker: {data.get('speaker')}")
    print(f"Language: {data.get('language')}")
    # Must NOT have used deepgram aura!
    assert resp_guarded.headers.get("x-tts-fallback") == "browser-speech-synthesis" or resp_guarded.headers.get("x-tts-provider") == "sarvam"
    if data.get("fallback_to_browser"):
        assert data.get("speaker") == "ritu"
        assert data.get("language") == "mr-IN"

    print("\nALL VERIFICATIONS PASSED!")

if __name__ == "__main__":
    test_tts_endpoints()
