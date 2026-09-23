"""Quick scratch test: verify the Sarvam API key works against TTS/STT endpoints."""
import os
import httpx
import base64
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("SARVAM_API_KEY")
print(f"API_KEY present: {bool(API_KEY)}, value: {API_KEY}")

if not API_KEY:
    print("ERROR: SARVAM_API_KEY not set in .env")
    exit(1)

# Test TTS endpoint (matching sarvam_tts.py SARVAM_TTS_URL)
url = "https://api.sarvam.ai/text-to-speech"
headers = {
    "api-subscription-key": API_KEY,
    "Content-Type": "application/json",
}
payload = {
    "text": "नमस्ते, यह एक परीक्षण है।",
    "target_language_code": "hi-IN",
    "speaker": "simran",
    "model": "bulbul:v3",
    "pipeline": "default",
}

print("\n--- Testing TTS endpoint (https://api.sarvam.ai/text-to-speech) ---")
try:
    with httpx.Client(timeout=30) as client:
        resp = client.post(url, json=payload, headers=headers)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            audios = data.get("audios", [])
            if audios:
                audio_bytes = base64.b64decode(audios[0])
                print(f"SUCCESS: Received {len(audio_bytes)} bytes of audio")
            else:
                print(f"WARNING: No audios in response: {data}")
        else:
            print(f"ERROR body: {resp.text[:500]}")
except Exception as e:
    print(f"EXCEPTION: {e}")

# Test STT endpoint (matching sarvam_stt.py SARVAM_STT_URL)
stt_url = "https://api.sarvam.ai/speech-to-text"
stt_headers = {
    "api-subscription-key": API_KEY,
}
# Create a minimal valid WAV file (silence)
wav_header = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"

print("\n--- Testing STT endpoint (https://api.sarvam.ai/speech-to-text) ---")
try:
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            stt_url,
            headers=stt_headers,
            files={"file": ("test.wav", wav_header, "audio/wav")},
            data={
                "language_code": "hi-IN",
                "model": "saaras:v4",
            },
        )
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            print(f"SUCCESS: {resp.json()}")
        else:
            print(f"ERROR body: {resp.text[:500]}")
except Exception as e:
    print(f"EXCEPTION: {e}")