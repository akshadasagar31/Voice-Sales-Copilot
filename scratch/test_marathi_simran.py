import asyncio
import httpx
import os
import io
import wave
import base64
import numpy as np
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(Path("backend/.env"))
key = os.getenv("SARVAM_API_KEY")

async def test_marathi_simran_audio():
    text = "नमस्कार! एचडीएफसी बँकेमध्ये पर्सनल लोनसाठी किमान सिबिल स्कोर ७५० असावा लागतो."
    print("Calling Sarvam Bulbul v3 API with language_code='mr-IN' and speaker='simran'...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.post(
            "https://api.sarvam.ai/text-to-speech",
            headers={"api-subscription-key": key, "Content-Type": "application/json"},
            json={
                "text": text,
                "language_code": "mr-IN",
                "speaker": "simran",
                "model": "bulbul:v3",
            },
        )
        print("Sarvam API HTTP Status:", res.status_code)
        assert res.status_code == 200, f"Error: {res.text}"
        data = res.json()
        wav_bytes = base64.b64decode(data["audios"][0])

    print(f"Generated Marathi WAV Audio Size: {len(wav_bytes)} bytes")

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        rate = wf.getframerate()
        nframes = wf.getnframes()
        duration = nframes / float(rate)
        raw_pcm = wf.readframes(nframes)
        audio = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32)

    print(f"Audio Properties -> Sample Rate: {rate} Hz, Channels: {wf.getnchannels()}, Duration: {duration:.2f} seconds")

    # Pitch calculation (F0)
    frame_size = int(rate * 0.04)
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
    gender_label = "Female Voice (standard range 165-260 Hz)" if is_female else "Male Voice"
    print(f"Acoustic Gender: {gender_label}")
    assert is_female, f"Expected female voice, got F0 {median_f0:.1f} Hz"
    print("VERIFICATION SUCCESS: Sarvam Bulbul v3 supports speaker 'simran' for 'mr-IN' with high-fidelity female speech!")

if __name__ == "__main__":
    asyncio.run(test_marathi_simran_audio())
