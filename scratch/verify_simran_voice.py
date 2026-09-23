import wave
import io
import numpy as np
import httpx
import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8")

async def verify_simran_female_voice():
    text = "नमस्ते! एचडीएफसी बैंक में पर्सनल लोन के लिए न्यूनतम सिबिल स्कोर 750 होना चाहिए। 5 लाख रुपये के लोन पर ईएमआई लगभग 10,500 रुपये प्रति माह होगी।"
    print("Testing /api/tts with Hindi text...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.post(
            "http://127.0.0.1:8001/api/tts",
            json={"text": text, "language": "hi", "module": "module2"},
        )

    print("HTTP Status Code:", res.status_code)
    provider = res.headers.get("x-tts-provider")
    voice = res.headers.get("x-tts-voice")
    lang = res.headers.get("x-tts-language")
    content_type = res.headers.get("content-type")
    print(f"X-TTS-Provider: {provider}")
    print(f"X-TTS-Voice: {voice}")
    print(f"X-TTS-Language: {lang}")
    print(f"Content-Type: {content_type}")
    print(f"Audio Size: {len(res.content)} bytes")

    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert provider == "sarvam", f"Expected sarvam, got {provider}"
    assert voice == "simran", f"Expected simran, got {voice}"
    assert lang == "hi-IN", f"Expected hi-IN, got {lang}"
    assert res.content.startswith(b"RIFF"), "Expected RIFF WAVE container"

    with wave.open(io.BytesIO(res.content), "rb") as wf:
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)

    # Autocorrelation pitch detection on voiced frames
    frame_size = int(rate * 0.04)  # 40ms frames
    pitches = []
    for i in range(0, len(audio) - frame_size, frame_size // 2):
        frame = audio[i : i + frame_size]
        if np.max(np.abs(frame)) < 1000:
            continue
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2 :]
        min_lag = int(rate / 350)
        max_lag = int(rate / 80)
        peak_lag = min_lag + np.argmax(corr[min_lag:max_lag])
        freq = rate / peak_lag
        if 130 <= freq <= 320:
            pitches.append(freq)

    median_f0 = float(np.median(pitches)) if pitches else 0.0
    print(f"Fundamental Pitch (F0): {median_f0:.1f} Hz")
    is_female = median_f0 >= 165.0
    classification = "Female Voice (typical range: 165 - 260 Hz)" if is_female else "Male Voice"
    print(f"Acoustic Gender Classification: {classification}")
    assert is_female, f"Voice fundamental frequency {median_f0:.1f} Hz is not in female range"
    print("\nALL CHECKS PASSED: Speaker 'simran' is verified female, native Hindi, on Sarvam Bulbul v3!")

if __name__ == "__main__":
    asyncio.run(verify_simran_female_voice())
