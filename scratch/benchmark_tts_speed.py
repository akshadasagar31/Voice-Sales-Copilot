import os
import time
import httpx
from dotenv import load_dotenv

load_dotenv('backend/.env')
api_key = os.getenv('DEEPGRAM_API_KEY')

client = httpx.Client(timeout=30.0)
url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"
headers = {
    "Authorization": f"Token {api_key}",
    "Content-Type": "application/json"
}

short_phrase = "The minimum CIBIL score is 750."
long_sentence = "The minimum CIBIL score required for a personal loan is typically 750 or above, though some lenders may approve loans with scores as low as 650 depending on other factors like income and repayment history."

# Warmup / connection pool
client.post(url, json={"text": "Hello"}, headers=headers)

# Short phrase timing
t0 = time.perf_counter()
r1 = client.post(url, json={"text": short_phrase}, headers=headers)
t_short = (time.perf_counter() - t0) * 1000

# Long sentence timing
t0 = time.perf_counter()
r2 = client.post(url, json={"text": long_sentence}, headers=headers)
t_long = (time.perf_counter() - t0) * 1000

print(f"Short phrase (31 chars): {t_short:.1f}ms, bytes: {len(r1.content)}")
print(f"Long sentence (186 chars): {t_long:.1f}ms, bytes: {len(r2.content)}")
