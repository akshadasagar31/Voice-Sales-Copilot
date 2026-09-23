import os
import sys
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

test_phrases = [
    "व्याजदर १०.५% p.a. पासून सुरू होतो.",
    "व्याजदर १०.५ टक्के प्रति वर्ष पासून सुरू होतो.",
    "किमान CIBIL score ७५० आणि कमाल loan amount ₹५० लाख आहे.",
    "किमान सिबिल स्कोअर ७५० आणि कमाल लोन अमाऊंट ५० लाख रुपये आहे."
]

for i, phrase in enumerate(test_phrases):
    payload = {
        'text': phrase,
        'model': 'bulbul:v3',
        'language_code': 'mr-IN',
        'speaker': 'priya'
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    print(f"Phrase {i+1}: status {resp.status_code}, audios: {len(resp.json().get('audios', [])) if resp.ok else resp.text}")
