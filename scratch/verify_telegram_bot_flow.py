import asyncio
import os
import sys
import httpx
from unittest.mock import AsyncMock, patch

# Ensure backend root on sys.path
sys.path.insert(0, os.path.abspath("backend"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from services.telegram_bot import VoiceCopilotBot, get_voice_copilot_bot


async def verify_telegram_flow():
    print("=== 1. Checking VoiceCopilotBot status ===")
    bot = get_voice_copilot_bot()
    print(f"Bot configured: {bot.is_configured}")

    print("\n=== 2. Testing Text Message Flow (Live RAG & TTS) ===")
    # Patch outgoing Telegram API calls so we don't attempt to contact Telegram servers without a live bot token
    with patch.object(bot, "send_chat_action", new_callable=AsyncMock) as mock_action, \
         patch.object(bot, "send_message", new_callable=AsyncMock) as mock_msg, \
         patch.object(bot, "send_voice", new_callable=AsyncMock) as mock_voice:

        mock_msg.return_value = {"ok": True}
        mock_voice.return_value = {"ok": True}

        update_text = {
            "update_id": 90001,
            "message": {
                "message_id": 101,
                "chat": {"id": 999888, "type": "private"},
                "text": "What is the minimum CIBIL score for an HDFC personal loan?",
            },
        }

        result = await bot.handle_webhook_update(update_text)
        print("Text Query Result:", result)
        assert result["status"] == "success"
        assert result["query_type"] == "text"
        assert len(result["answer"]) > 0
        assert result["has_voice"] is True
        print(f"✓ Answer generated: {result['answer'][:100]}...")
        print(f"✓ Voice synthesized: {result['has_voice']}")
        print(f"✓ Outgoing Telegram send_message called: {mock_msg.called}")
        print(f"✓ Outgoing Telegram send_voice called: {mock_voice.called}")

    print("\n=== 3. Testing Voice Note Flow (Live STT, RAG, TTS) ===")
    with patch.object(bot, "download_file", new_callable=AsyncMock) as mock_download, \
         patch.object(bot, "send_chat_action", new_callable=AsyncMock) as mock_action, \
         patch.object(bot, "send_message", new_callable=AsyncMock) as mock_msg, \
         patch.object(bot, "send_voice", new_callable=AsyncMock) as mock_voice:

        # Synthesize a real Marathi voice query using Sarvam TTS to feed into STT
        from services.sarvam_tts import SarvamTTSService
        sarvam_tts = SarvamTTSService()
        marathi_query = "एचडीएफसी बँकेकडून 10 लाख रुपयांच्या पर्सनल लोनसाठी सिबिल स्कोर किती लागतो?"
        real_wav = await sarvam_tts.synthesize_speech(marathi_query, language_code="mr-IN", speaker="simran")
        print(f"Generated sample Marathi speech query ({len(real_wav)} bytes)")

        mock_download.return_value = (real_wav, "wav")
        mock_msg.return_value = {"ok": True}
        mock_voice.return_value = {"ok": True}

        update_voice = {
            "update_id": 90002,
            "message": {
                "message_id": 102,
                "chat": {"id": 999888, "type": "private"},
                "voice": {
                    "file_id": "test_voice_file_mr",
                    "mime_type": "audio/wav",
                    "duration": 4,
                },
            },
        }

        voice_res = await bot.handle_webhook_update(update_voice)
        print("Voice Query Result:", voice_res)
        assert voice_res["status"] == "success"
        assert voice_res["query_type"] == "voice"
        assert len(voice_res["transcript"]) > 0
        assert len(voice_res["answer"]) > 0
        assert voice_res["has_voice"] is True
        print(f"✓ Transcribed: '{voice_res['transcript']}'")
        print(f"✓ RAG Answer: '{voice_res['answer'][:100]}...'")
        print(f"✓ Voice reply delivered: {voice_res['has_voice']}")

    print("\n=== All Telegram Bot Integration Flows Verified Successfully! ===")


if __name__ == "__main__":
    asyncio.run(verify_telegram_flow())
