import os
import sys
import time
import requests
import base64
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv('backend/.env')
api_key = os.getenv('SARVAM_API_KEY')
url = 'https://api.sarvam.ai/text-to-speech'
headers = {'api-subscription-key': api_key, 'Content-Type': 'application/json'}

terms_test = [
    "एचडीएफसी बँकेच्या personal loan साठी किमान CIBIL score ७५० आवश्यक आहे.",
    "गृहकर्जासाठी interest rate ८.५% पासून सुरू होतो आणि processing fee ०.५% आहे.",
    "तुमचा monthly EMI आणि tenure कालावधी किती आहे?",
    "Pre-closure charges आणि foreclosure norms काय आहेत?"
]

os.makedirs("scratch/sarvam_test_audio", exist_ok=True)

for i, text in enumerate(terms_test):
    payload = {
        'text': text,
        'model': 'bulbul:v3',
        'language_code': 'mr-IN',
        'speaker': 'priya'
    }
    t0 = time.perf_counter()
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    dur = (time.perf_counter() - t0) * 1000
    if resp.ok:
        audios = resp.json().get('audios', [])
        b = base64.b64decode(audios[0])
        path = f"scratch/sarvam_test_audio/test_{i}.wav"
        with open(path, "wb") as f:
            f.write(b)
        print(f"[{i+1}/4] OK ({dur:.0f}ms, {len(b)} bytes): {text[:40]}...")
    else:
        print(f"[{i+1}/4] ERROR ({resp.status_code}): {resp.text}")
