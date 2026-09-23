import os
import sys
import time
import requests
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv('backend/.env')

api_key = os.getenv('SARVAM_API_KEY')
url = 'https://api.sarvam.ai/text-to-speech'
headers = {'api-subscription-key': api_key, 'Content-Type': 'application/json'}

short_clauses = [
    'एचडीएफसी बँकेच्या नियमांनुसार,',
    'किमान CIBIL score ७५० आवश्यक आहे.',
    'आणि व्याजदर १०.५ टक्के पासून सुरू होतो.'
]

for speaker in ['ritu', 'priya']:
    print(f'=== Testing Speaker: {speaker} ===')
    for clause in short_clauses:
        t0 = time.perf_counter()
        payload = {
            'text': clause,
            'model': 'bulbul:v3',
            'language_code': 'mr-IN',
            'speaker': speaker
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        dur = (time.perf_counter() - t0) * 1000
        print(f'Clause: "{clause}" -> {dur:.1f}ms (Status: {resp.status_code})')
