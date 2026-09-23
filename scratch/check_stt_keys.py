import os
import httpx
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / "backend" / ".env"
load_dotenv(dotenv_path=env_path)

sarvam_key = os.getenv("SARVAM_API_KEY")
deepgram_key = os.getenv("DEEPGRAM_API_KEY")

print("SARVAM_API_KEY:", bool(sarvam_key))
print("DEEPGRAM_API_KEY:", bool(deepgram_key))
