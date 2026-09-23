"""
End-to-End Real Flow Verification for Telegram Voice Note Processing.
Trace: Telegram voice -> webhook -> download voice file -> Sarvam STT -> RAG -> text reply -> Sarvam TTS -> voice reply.
Uses real Telegram Opus audio (scratch/telegram_voice_real.oga), real Sarvam STT, real RAG, real Sarvam TTS, and Opus conversion.
"""

import os
import sys
import asyncio
import dotenv

# Force UTF-8 on Windows console
sys.stdout.reconfigure(encoding="utf-8")
dotenv.load_dotenv("backend/.env")
sys.path.insert(0, "backend")

from unittest.mock import AsyncMock, patch, MagicMock
from services.telegram_bot import VoiceCopilotBot

async def run_real_pipeline():
    print("=== STARTING REAL FLOW TRACE ===")

    # 1. Prepare actual Telegram voice note file
    voice_path = "scratch/telegram_voice_real.oga"
    assert os.path.exists(voice_path), f"File {voice_path} not found!"
    real_voice_bytes = open(voice_path, "rb").read()
    print(f"Step 1: Loaded real Telegram voice note ({len(real_voice_bytes)} bytes, header: {real_voice_bytes[:4]})")
    assert real_voice_bytes.startswith(b"OggS"), "Audio is not an authentic OGG container!"

    # 2. Instantiate real bot with live services
    bot = VoiceCopilotBot()
    assert bot.is_configured, "Bot token not configured!"
    print(f"Step 2: VoiceCopilotBot initialized for @VoiceCopilotBot")

    # Mock only Telegram network delivery (since fake chat_id has no active user session),
    # but run REAL Sarvam STT, REAL RAG, REAL Sarvam TTS, and REAL Opus OGG conversion.
    sent_messages = []
    sent_voices = []

    async def fake_send_message(chat_id, text, parse_mode="Markdown"):
        sent_messages.append({"chat_id": chat_id, "text": text, "parse_mode": parse_mode})
        print(f"  -> [Telegram API: sendMessage] chat_id={chat_id}, text_preview={text[:80]}...")
        return {"ok": True, "result": {"message_id": 101, "chat": {"id": chat_id}, "text": text}}

    async def fake_post(url, *args, **kwargs):
        data = kwargs.get("data", {})
        files = kwargs.get("files", {})
        if "/sendVoice" in url:
            voice_file = files.get("voice")
            sent_voices.append({
                "url": url,
                "data": data,
                "filename": voice_file[0] if voice_file else None,
                "bytes_len": len(voice_file[1]) if voice_file else 0,
                "mime": voice_file[2] if voice_file else None,
                "header": voice_file[1][:4] if voice_file else None,
            })
            print(f"  -> [Telegram API: sendVoice] file={voice_file[0]}, mime={voice_file[2]}, size={len(voice_file[1])} bytes, header={voice_file[1][:4]}")
            # Verify that sendVoice receives valid OGG Opus bytes
            assert voice_file[1].startswith(b"OggS"), "Voice file sent to sendVoice is NOT valid OGG Opus!"
            return MagicMock(status_code=200, json=lambda: {"ok": True, "result": {"message_id": 102, "voice": {"mime_type": "audio/ogg"}}})
        return MagicMock(status_code=200, json=lambda: {"ok": True})

    # Hook download_file to return our real Telegram voice file bytes
    bot.download_file = AsyncMock(return_value=(real_voice_bytes, "oga"))
    bot.send_message = AsyncMock(side_effect=fake_send_message)
    bot.client.post = AsyncMock(side_effect=fake_post)

    # 3. Simulate webhook update with voice payload
    webhook_update = {
        "update_id": 88888888,
        "message": {
            "message_id": 500,
            "from": {"id": 987654321, "first_name": "TestUser"},
            "chat": {"id": 987654321, "type": "private"},
            "date": 1726915000,
            "voice": {
                "file_id": "real_telegram_voice_oga_001",
                "mime_type": "audio/ogg",
                "duration": 5,
                "file_size": len(real_voice_bytes),
            },
        },
    }

    print("\nStep 3: Processing incoming Telegram webhook voice update...")
    result = await bot.handle_webhook_update(webhook_update)

    print("\n=== PIPELINE EXECUTION RESULT ===")
    print("Status:", result.get("status"))
    print("Query Type:", result.get("query_type"))
    print("Transcript (Sarvam STT):", result.get("transcript"))
    print("Answer (RAG):", result.get("answer"))
    print("Language:", result.get("language"))
    print("Sources:", result.get("sources"))
    print("Has Voice Reply:", result.get("has_voice"))

    # Assertions on pipeline outcome
    assert result.get("status") == "success", f"Pipeline failed: {result}"
    assert result.get("query_type") == "voice"
    assert len(result.get("transcript", "")) > 0, "Transcript is empty!"
    assert len(result.get("answer", "")) > 0, "Answer is empty!"
    assert result.get("has_voice") is True, "Voice reply was not generated/sent!"
    assert len(sent_messages) >= 1, "Text message was not dispatched!"
    assert len(sent_voices) >= 1, "Voice note was not dispatched!"

    sent_text = sent_messages[0]["text"]
    print(f"\nStep 4: Outbound Clean Plain-Text Verification:")
    print(f"  - Message text: {sent_text}")
    assert "You asked" not in sent_text, "Found 'You asked' in sent text!"
    assert "Answer" not in sent_text, "Found 'Answer' in sent text!"
    assert "*" not in sent_text, "Found '*' in sent text!"
    assert "_" not in sent_text, "Found '_' in sent text!"

    voice_sent = sent_voices[0]
    print(f"\nStep 5: Outbound Voice Note Verification:")
    print(f"  - Target URL: {voice_sent['url']}")
    print(f"  - Filename: {voice_sent['filename']}")
    print(f"  - MIME type: {voice_sent['mime']}")
    print(f"  - Bytes count: {voice_sent['bytes_len']}")
    print(f"  - Magic header: {voice_sent['header']} (b'OggS' = Telegram Opus OGG)")
    print(f"  - Caption: {voice_sent['data'].get('caption')}")
    assert voice_sent["data"].get("caption") is None, f"Voice caption is not None: {voice_sent['data'].get('caption')}"

    print("\n✅ REAL FLOW TRACE COMPLETED SUCCESSFULLY WITH ZERO ERRORS & CLEAN PLAIN TEXT!")

if __name__ == "__main__":
    asyncio.run(run_real_pipeline())
