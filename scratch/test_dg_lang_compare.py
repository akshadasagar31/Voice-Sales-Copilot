import os
import sys
import time
import io
import wave
import json
import asyncio
import websockets
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv("backend/.env")
api_key = os.getenv("DEEPGRAM_API_KEY")

sys.path.insert(0, ".")
sys.path.insert(0, "backend")

from scratch.test_live_streaming_stt_pipeline import generate_test_audio_pcm

async def test_dg_direct(name, text, lang, dg_lang_param):
    pcm_bytes, sample_rate = await generate_test_audio_pcm(text, lang)
    url = (
        f"wss://api.deepgram.com/v1/listen"
        f"?model=nova-3"
        f"&encoding=linear16"
        f"&sample_rate={sample_rate}"
        f"&channels=1"
        f"&smart_format=true"
        f"&punctuate=true"
        f"&interim_results=true"
        f"&endpointing=300"
        f"&{dg_lang_param}"
        f"&keyterm=CIBIL"
        f"&keyterm=CIBIL+score"
        f"&keyterm=EMI"
        f"&keyterm=HDFC"
        f"&keyterm=HDFC+Bank"
        f"&keyterm=personal+loan"
        f"&keyterm=कर्ज"
        f"&keyterm=कर्जासाठी"
        f"&keyterm=वैयक्तिक+कर्ज"
        f"&keyterm=व्याजदर"
    )
    headers = {"Authorization": f"Token {api_key}"}

    all_finals = []
    last_interim = ""
    detected_langs = []

    try:
        async with websockets.connect(url, additional_headers=headers) as ws:
            chunk_size = 4096 * 2
            for offset in range(0, len(pcm_bytes), chunk_size):
                await ws.send(pcm_bytes[offset:offset + chunk_size])
                await asyncio.sleep(0.05)

            await asyncio.sleep(0.3)
            await ws.send(json.dumps({"type": "CloseStream"}))

            async for msg in ws:
                data = json.loads(msg)
                if data.get("type") == "Results":
                    channel = data.get("channel", {})
                    if channel.get("detected_language"):
                        detected_langs.append(channel.get("detected_language"))
                    alts = channel.get("alternatives", [])
                    if alts:
                        t = alts[0].get("transcript", "").strip()
                        if data.get("is_final"):
                            if t:
                                all_finals.append(t)
                        else:
                            last_interim = t
                elif data.get("type") == "Metadata":
                    break

    except Exception as e:
        print(f"Error: {e}")

    full_final = " ".join(all_finals).strip() or last_interim
    print(f"[{dg_lang_param:18}] -> \"{full_final}\" (detected: {detected_langs[:2]})")

async def main():
    test_cases = [
        ("Marathi", "एचडीएफसी बँकेच्या वैयक्तिक कर्जासाठी किमान CIBIL score किती आवश्यक आहे?", "mr"),
        ("Hindi", "पर्सनल लोन के लिए न्यूनतम सिबिल स्कोर कितना होना चाहिए?", "hi"),
        ("English", "What is the minimum CIBIL score for personal loans?", "en"),
        ("Mixed", "माझा CIBIL score 750 आहे, मला HDFC Bank कडून personal loan वर किती interest rate मिळेल?", "mr"),
    ]

    for name, text, lang in test_cases:
        print(f"\n--- {name}: \"{text}\" ---")
        for p in ["language=multi", f"language={lang}"]:
            await test_dg_direct(name, text, lang, p)

if __name__ == "__main__":
    asyncio.run(main())
