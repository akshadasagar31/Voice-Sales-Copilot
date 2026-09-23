#!/usr/bin/env python3
"""
Simple Sarvam Bulbul v3 TTS Test using voice 'priya'
----------------------------------------------------
- Reads SARVAM_API_KEY from environment variables (or .env files)
- Accepts language and text dynamically at runtime (no hardcoded sentences)
- Generates speech audio using Sarvam Bulbul v3 ('priya')
- Saves WAV audio files into 'sarvam_voice_samples/'
- Does NOT modify existing project functionality.
"""

import os
import sys
import time
import base64
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows terminal for Devanagari / Indic scripts
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    # Check current directory and backend directory
    load_dotenv()
    load_dotenv(Path(__file__).parent / "backend" / ".env")
except ImportError:
    pass

try:
    import requests
except ImportError:
    print("Error: 'requests' library is required. Please install it with 'pip install requests'.", file=sys.stderr)
    sys.exit(1)

# Language code normalization map for common inputs
LANGUAGE_MAP = {
    "en": "en-IN",
    "en-in": "en-IN",
    "english": "en-IN",
    "hi": "hi-IN",
    "hi-in": "hi-IN",
    "hindi": "hi-IN",
    "mr": "mr-IN",
    "mr-in": "mr-IN",
    "marathi": "mr-IN",
    "bn": "bn-IN",
    "bn-in": "bn-IN",
    "bengali": "bn-IN",
    "gu": "gu-IN",
    "gu-in": "gu-IN",
    "gujarati": "gu-IN",
    "kn": "kn-IN",
    "kn-in": "kn-IN",
    "kannada": "kn-IN",
    "ml": "ml-IN",
    "ml-in": "ml-IN",
    "malayalam": "ml-IN",
    "pa": "pa-IN",
    "pa-in": "pa-IN",
    "punjabi": "pa-IN",
    "ta": "ta-IN",
    "ta-in": "ta-IN",
    "tamil": "ta-IN",
    "te": "te-IN",
    "te-in": "te-IN",
    "telugu": "te-IN",
    "od": "od-IN",
    "or": "od-IN",
    "od-in": "od-IN",
    "odia": "od-IN",
}


def normalize_language_code(lang_input: str) -> str:
    """Normalize user input to BCP-47 Sarvam language code (e.g. 'mr' -> 'mr-IN')."""
    cleaned = (lang_input or "").strip().lower()
    if cleaned in LANGUAGE_MAP:
        return LANGUAGE_MAP[cleaned]
    # If user provided custom BCP-47 like 'te-IN' or 'hi-IN'
    if "-" in cleaned:
        parts = cleaned.split("-")
        return f"{parts[0].lower()}-{parts[1].upper()}"
    return f"{cleaned}-IN"


def synthesize_sarvam_bulbul_v3(text: str, language_code: str, api_key: str, speaker: str = "priya") -> bytes:
    """
    Calls Sarvam AI text-to-speech REST API using model 'bulbul:v3' and speaker 'priya'.
    Returns decoded audio bytes.
    """
    url = "https://api.sarvam.ai/text-to-speech"
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json",
    }

    # Primary payload format for bulbul:v3
    payload = {
        "text": text,
        "model": "bulbul:v3",
        "language_code": language_code,
        "speaker": speaker,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=30)

    # Fallback to alternate schema if API expects 'inputs' and 'target_language_code'
    if not response.ok and response.status_code in (400, 422):
        fallback_payload = {
            "inputs": [text],
            "target_language_code": language_code,
            "speaker": speaker,
            "model": "bulbul:v3",
        }
        alt_response = requests.post(url, headers=headers, json=fallback_payload, timeout=30)
        if alt_response.ok:
            response = alt_response

    if not response.ok:
        try:
            err_details = response.json()
        except Exception:
            err_details = response.text
        raise RuntimeError(f"Sarvam API Error (HTTP {response.status_code}): {err_details}")

    data = response.json()
    audios = data.get("audios") or []
    if not audios or not isinstance(audios, list):
        raise ValueError(f"No audio data returned by Sarvam API. Response: {data}")

    # Decode base64 audio
    audio_base64 = audios[0]
    return base64.b64decode(audio_base64)


def main():
    parser = argparse.ArgumentParser(
        description="Simple Sarvam Bulbul v3 TTS test using 'priya' (interactive or CLI)"
    )
    parser.add_argument(
        "-l", "--lang", "--language",
        dest="language",
        help="Language code or name (e.g. 'en-IN', 'hi-IN', 'mr-IN', 'hi', 'mr', 'en')",
        default=None,
    )
    parser.add_argument(
        "-t", "--text",
        dest="text",
        help="Text string to synthesize (no hardcoded sentences)",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default="sarvam_voice_samples",
        help="Directory to save output WAV file (default: sarvam_voice_samples)",
    )
    args = parser.parse_args()

    # 1. Read SARVAM_API_KEY from environment variables
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        print("\n[!] Warning: SARVAM_API_KEY not found in environment variables or .env file.", file=sys.stderr)
        # Allow interactive entry if running in an interactive terminal
        if sys.stdin.isatty():
            entered_key = input("Enter your SARVAM_API_KEY: ").strip()
            if entered_key:
                api_key = entered_key
        if not api_key:
            print("\nError: SARVAM_API_KEY is required.", file=sys.stderr)
            print("Please set it in your environment or backend/.env file:", file=sys.stderr)
            print("  Windows PowerShell: $env:SARVAM_API_KEY='your_api_key_here'", file=sys.stderr)
            print("  Linux/macOS:        export SARVAM_API_KEY='your_api_key_here'\n", file=sys.stderr)
            sys.exit(1)

    print("\n" + "=" * 60)
    print("  Sarvam AI Bulbul v3 TTS Test (Speaker: 'priya')")
    print("=" * 60)

    # 2. Enter language at runtime (if not provided via CLI)
    raw_lang = args.language
    if not raw_lang:
        print("\nSupported Languages:")
        print("  - English (en / en-IN)")
        print("  - Hindi   (hi / hi-IN)")
        print("  - Marathi (mr / mr-IN)")
        print("  - Bengali (bn), Gujarati (gu), Kannada (kn)")
        print("  - Malayalam (ml), Punjabi (pa), Tamil (ta), Telugu (te), Odia (od)")
        raw_lang = input("\nEnter language [e.g. en, hi, mr, en-IN]: ").strip()
        while not raw_lang:
            print("Language cannot be empty.")
            raw_lang = input("Enter language: ").strip()

    language_code = normalize_language_code(raw_lang)
    print(f"Selected language: {language_code}")

    # 3. Enter text at runtime (NO hardcoded sentences)
    text = args.text
    if not text:
        print("\nEnter any text to synthesize (press Enter when done):")
        text = input("> ").strip()
        while not text:
            print("Text cannot be empty.")
            text = input("> ").strip()

    print("\nSynthesizing with Sarvam Bulbul v3...")
    print(f"  Model:   bulbul:v3")
    print(f"  Speaker: priya")
    print(f"  Lang:    {language_code}")
    print(f"  Text:    \"{text}\"")

    # 4. Generate audio
    start_time = time.perf_counter()
    try:
        audio_bytes = synthesize_sarvam_bulbul_v3(
            text=text,
            language_code=language_code,
            api_key=api_key,
            speaker="priya",
        )
    except Exception as e:
        print(f"\n[X] Synthesis failed: {e}", file=sys.stderr)
        sys.exit(1)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # 5. Save audio file into sarvam_voice_samples/
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    clean_lang = language_code.replace("-", "_").lower()
    filename = f"sarvam_bulbul_v3_priya_{clean_lang}_{timestamp}.wav"
    output_path = output_dir / filename

    with open(output_path, "wb") as f:
        f.write(audio_bytes)

    print("\n" + "=" * 60)
    print("  [✓] Audio successfully generated!")
    print("=" * 60)
    print(f"  Saved File:  {output_path.resolve()}")
    print(f"  File Size:   {len(audio_bytes):,} bytes ({len(audio_bytes) / 1024:.1f} KB)")
    print(f"  Latency:     {elapsed_ms:.1f} ms")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
