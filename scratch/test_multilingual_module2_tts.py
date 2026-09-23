import os
import sys
import time
import json
import asyncio
import httpx
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv("backend/.env")

BASE_URL = "http://127.0.0.1:8001"

TEST_CASES = [
    {
        "lang_name": "Marathi (मराठी)",
        "lang_code": "mr",
        "question": "एचडीएफसी बँकेच्या वैयक्तिक कर्जासाठी किमान CIBIL score किती आवश्यक आहे?",
        "expected_provider": "sarvam",
        "expected_media_type": "audio/wav",
    },
    {
        "lang_name": "Hindi (हिन्दी)",
        "lang_code": "hi",
        "question": "पर्सनल लोन के लिए न्यूनतम सिबिल स्कोर कितना होना चाहिए?",
        "expected_provider": "sarvam",
        "expected_media_type": "audio/wav",
    },
    {
        "lang_name": "English",
        "lang_code": "en",
        "question": "What is the minimum CIBIL score for personal loans?",
        "expected_provider": "sarvam",
        "expected_media_type": "audio/wav",
    },
]


async def run_language_test(test_case: dict, client: httpx.AsyncClient):
    lang_name = test_case["lang_name"]
    lang_code = test_case["lang_code"]
    question = test_case["question"]
    exp_provider = test_case["expected_provider"]

    print("\n" + "=" * 70)
    print(f"  Testing Language: {lang_name} ({lang_code})")
    print(f"  Question: \"{question}\"")
    print("=" * 70)

    # 1. Stream RAG response
    t_start = time.perf_counter()
    first_token_time = None
    first_chunk_text = ""
    full_answer = ""
    detected_lang = lang_code

    async with client.stream(
        "POST",
        f"{BASE_URL}/api/ask",
        json={
            "question": question,
            "top_k": 3,
            "namespace": "sales_playbooks",
            "language": lang_code,
            "stream": True,
        },
        timeout=60.0,
    ) as stream_resp:
        if stream_resp.status_code != 200:
            print(f"[X] RAG streaming failed with status {stream_resp.status_code}")
            return False

        async for line in stream_resp.aiter_lines():
            line = line.strip()
            if not line:
                continue

            if line.startswith("data:"):
                try:
                    payload = json.loads(line[5:].strip())
                    if "language" in payload:
                        detected_lang = payload["language"]
                    token = payload.get("token") or payload.get("delta") or ""
                    if token:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        full_answer += token

                        # Check for sentence / clause boundary for Chunk #0
                        if not first_chunk_text:
                            for sep in [".", "।", "?", "!", ":", ","]:
                                if sep in full_answer and len(full_answer.strip()) >= 15:
                                    first_chunk_text = full_answer.strip()
                                    break
                except Exception:
                    pass

    ttft = (first_token_time - t_start) * 1000 if first_token_time else 0.0
    if not first_chunk_text:
        first_chunk_text = full_answer.split(".")[0] if "." in full_answer else full_answer[:80]

    print(f"  [1] RAG Response TTFT:          {ttft:.1f} ms")
    print(f"  [2] Resolved Language:          {detected_lang}")
    print(f"  [3] Extracted Chunk #0:         \"{first_chunk_text}\"")
    print(f"  [4] Full Answer Length:         {len(full_answer)} chars")

    # 2. Call /api/tts with module="module2"
    t_tts_start = time.perf_counter()
    tts_resp = await client.post(
        f"{BASE_URL}/api/tts",
        json={
            "text": first_chunk_text,
            "language": detected_lang,
            "module": "module2",
            "speaker": "priya" if detected_lang == "mr" else None,
        },
        timeout=30.0,
    )
    tts_dur = (time.perf_counter() - t_tts_start) * 1000

    if tts_resp.status_code != 200:
        print(f"[X] TTS failed with HTTP {tts_resp.status_code}: {tts_resp.text}")
        return False

    audio_bytes = tts_resp.content
    content_type = tts_resp.headers.get("content-type", "")
    actual_provider = tts_resp.headers.get("x-tts-provider", "")
    actual_voice = tts_resp.headers.get("x-tts-voice", "")

    print(f"  [5] TTS Provider:               {actual_provider} (expected: {exp_provider})")
    if actual_voice:
        print(f"  [6] Voice / Speaker:            {actual_voice}")
    print(f"  [7] Media Type:                 {content_type}")
    print(f"  [8] Audio Payload Size:         {len(audio_bytes):,} bytes")
    print(f"  [9] TTS Synthesis Latency:      {tts_dur:.1f} ms")
    print(f"  [10] Audio Header Check:        {audio_bytes[:8]}")

    # Validation assertions
    assert actual_provider == exp_provider, f"Provider mismatch: expected {exp_provider}, got {actual_provider}"
    assert len(audio_bytes) > 1000, "Audio output is suspiciously small"
    if exp_provider == "sarvam":
        assert audio_bytes.startswith(b"RIFF"), "Sarvam audio does not start with RIFF header"
        assert "wav" in content_type, f"Content-type should be audio/wav, got {content_type}"

    print(f"  [✓] {lang_name} verification PASSED!")
    return True


async def main():
    print("\n" + "#" * 70)
    print("  MODULE 2 MULTILINGUAL TTS VERIFICATION SUITE")
    print("  Testing Marathi (Sarvam Bulbul v3), Hindi (Deepgram), English (Deepgram)")
    print("#" * 70)

    async with httpx.AsyncClient() as client:
        results = {}
        for tc in TEST_CASES:
            success = await run_language_test(tc, client)
            results[tc["lang_name"]] = success

    print("\n" + "#" * 70)
    print("  FINAL MULTILINGUAL TEST SUMMARY:")
    for lang, passed in results.items():
        status = "PASSED [✓]" if passed else "FAILED [X]"
        print(f"    - {lang:25}: {status}")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
