import os
import sys
import asyncio
import httpx
from pathlib import Path
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

env_path = Path(__file__).resolve().parent.parent / "backend" / ".env"
load_dotenv(dotenv_path=env_path)

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

async def test_sarvam_and_deepgram_stt():
    test_cases = [
        ("en", "en-IN", "My name is Rajesh Sharma. My contact number is 9876543210. I need a personal loan of 5 lakh rupees."),
        ("hi", "hi-IN", "मेरा नाम राजेश शर्मा है। मेरा फोन नंबर 9876543210 है। मुझे 5 लाख रुपये का पर्सनल लोन चाहिए।"),
        ("mr", "mr-IN", "माझे नाव राजेश शर्मा आहे. माझा फोन नंबर 9876543210 आहे. मला 5 लाख रुपयांचे वैयक्तिक कर्ज हवे आहे."),
        ("mixed", "hi-IN", "Hello, mera naam Rajesh Sharma hai, phone number 9876543210, looking for 5 lakh personal loan."),
    ]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for label, lang_code, text in test_cases:
            print(f"\n========================================================")
            print(f"TEST CASE: {label.upper()} ({lang_code})")
            print(f"Original Text: {text}")
            print(f"========================================================")
            
            tts_res = await client.post(
                "https://api.sarvam.ai/text-to-speech",
                headers={"api-subscription-key": SARVAM_API_KEY},
                json={
                    "text": text,
                    "language_code": lang_code,
                    "speaker": "simran",
                    "model": "bulbul:v3"
                }
            )
            if tts_res.status_code != 200:
                print(f"TTS synthesis failed: {tts_res.text}")
                continue
            
            import base64
            audios = tts_res.json().get("audios", [])
            if not audios:
                print("No audio returned by TTS")
                continue
            audio_bytes = base64.b64decode(audios[0])
            print(f"Generated WAV audio: {len(audio_bytes)} bytes")
            
            # 1. Test Sarvam saaras:v4 with target lang and unknown
            for stt_lang in [lang_code, "unknown"]:
                try:
                    sarvam_stt_res = await client.post(
                        "https://api.sarvam.ai/speech-to-text",
                        headers={"api-subscription-key": SARVAM_API_KEY},
                        files={"file": ("test.wav", audio_bytes, "audio/wav")},
                        data={"language_code": stt_lang, "model": "saaras:v4"}
                    )
                    if sarvam_stt_res.status_code == 200:
                        sj = sarvam_stt_res.json()
                        tr = sj.get("transcript")
                        lc = sj.get("language_code")
                        lp = sj.get("language_probability")
                        print(f"[Sarvam saaras:v4 (lang={stt_lang})] Transcript: '{tr}' (detected: {lc}, prob: {lp})")
                    else:
                        print(f"[Sarvam saaras:v4 (lang={stt_lang})] Status: {sarvam_stt_res.status_code}, Res: {sarvam_stt_res.text}")
                except Exception as e:
                    print(f"[Sarvam saaras:v4 (lang={stt_lang})] Error: {e}")

            # 2. Test Deepgram Nova-3
            for dg_mode in ["language=multi", "detect_language=true", f"language={lang_code[:2]}"]:
                try:
                    dg_url = f"https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true&punctuate=true&{dg_mode}"
                    dg_res = await client.post(
                        dg_url,
                        headers={"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "audio/wav"},
                        content=audio_bytes
                    )
                    if dg_res.status_code == 200:
                        alt = dg_res.json().get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0]
                        transcript = alt.get("transcript", "")
                        conf = alt.get("confidence", 0)
                        det_lang = dg_res.json().get("results", {}).get("channels", [{}])[0].get("detected_language")
                        print(f"[Deepgram Nova-3 ({dg_mode})] Transcript: '{transcript}' (conf: {conf:.2f}, det_lang: {det_lang})")
                    else:
                        print(f"[Deepgram Nova-3 ({dg_mode})] Status: {dg_res.status_code}, Res: {dg_res.text[:150]}")
                except Exception as e:
                    print(f"[Deepgram Nova-3 ({dg_mode})] Error: {e}")

asyncio.run(test_sarvam_and_deepgram_stt())
