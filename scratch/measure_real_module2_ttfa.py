import os
import sys
import time
import json
import asyncio
import httpx
from pathlib import Path
from dotenv import load_dotenv

load_dotenv('backend/.env')
sys.path.insert(0, 'backend')
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from services.stt import build_deepgram_ws_url, normalize_stt_transcript
from services.rag import RAGService
from services.tts import DeepgramTTSService

def simulate_sentence_tokenizer_chunk(tokens):
    """
    Simulates the exact logic of SentenceTokenizer.findSentenceEnd:
    1. Early break on comma after >= 12 chars
    2. Early break on colon after >= 8 chars
    3. Early break on space after >= 26 chars
    4. Full sentence terminators (. ! ? । ॥)
    Returns (chunk0, time_at_chunk0)
    """
    buf = ""
    is_first = True
    for t_time, tok in tokens:
        buf += tok
        if is_first:
            for i, c in enumerate(buf):
                if c == "," and i >= 12:
                    return buf[:i+1].strip(), t_time
                if c == ":" and i >= 8:
                    return buf[:i+1].strip(), t_time
                if c == " " and i >= 26:
                    return buf[:i+1].strip(), t_time
                if c in (".", "?", "!", "\u0964"):
                    return buf[:i+1].strip(), t_time
    return buf.strip(), tokens[-1][0] if tokens else 0.0

def simulate_baseline_tokenizer_chunk(tokens):
    """
    Simulates the previous baseline SentenceTokenizer:
    Requires >= 28 chars before comma, otherwise waits for full period (.)
    """
    buf = ""
    for t_time, tok in tokens:
        buf += tok
        for i, c in enumerate(buf):
            if c == "," and i >= 28:
                return buf[:i+1].strip(), t_time
            if c in (".", "?", "!", "\u0964"):
                return buf[:i+1].strip(), t_time
    return buf.strip(), tokens[-1][0] if tokens else 0.0

def benchmark_real_pipeline(query: str, language: str = "en"):
    print(f"\n=======================================================")
    print(f"BENCHMARKING MODULE 2 VOICE QUERY: '{query}' ({language})")
    print(f"=======================================================")

    rag = RAGService()
    tts = DeepgramTTSService()

    # 1. Warm-up Pinecone & OpenRouter connections
    rag.retriever.retrieve("warmup query", top_k=1)
    tts_client = httpx.Client(timeout=10.0)

    # 2. MEASURE LIVE RAG STREAMING
    t_start_rag = time.perf_counter()
    tokens = []
    metadata = None

    generator = rag.stream_answer_chunks(question=query, top_k=4, language=language)
    first_token_time = None

    for raw_event in generator:
        if raw_event.startswith("event: token"):
            data_line = [l for l in raw_event.split("\n") if l.startswith("data:")][0]
            token_val = json.loads(data_line[5:].strip()).get("token", "")
            now = time.perf_counter() - t_start_rag
            if first_token_time is None and token_val:
                first_token_time = now
            if token_val:
                tokens.append((now, token_val))
        elif raw_event.startswith("event: metadata"):
            data_line = [l for l in raw_event.split("\n") if l.startswith("data:")][0]
            metadata = json.loads(data_line[5:].strip())

    full_answer = "".join(t[1] for t in tokens).strip()
    print(f"Model used: {metadata.get('model')}")
    print(f"Full Answer ({len(full_answer)} chars): {full_answer[:120]}...")
    print(f"TTFT (Time to First Token): {first_token_time*1000:.1f}ms")

    # 3. COMPARE TOKENIZER DISPATCH: BASELINE vs OPTIMIZED
    base_chunk0, base_chunk_time = simulate_baseline_tokenizer_chunk(tokens)
    opt_chunk0, opt_chunk_time = simulate_sentence_tokenizer_chunk(tokens)

    print(f"\n--- Chunk #0 Comparison ---")
    print(f"BASELINE Chunk #0 ({len(base_chunk0)} chars): '{base_chunk0}' (emitted at {base_chunk_time*1000:.1f}ms)")
    print(f"OPTIMIZED Chunk #0 ({len(opt_chunk0)} chars): '{opt_chunk0}' (emitted at {opt_chunk_time*1000:.1f}ms)")

    # 4. MEASURE LIVE DEEPGRAM AURA TTS SYNTHESIS FOR BOTH
    t0 = time.perf_counter()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    base_audio = loop.run_until_complete(tts.synthesize_speech(base_chunk0, language=language))
    base_tts_time = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    opt_audio = loop.run_until_complete(tts.synthesize_speech(opt_chunk0, language=language))
    opt_tts_time = (time.perf_counter() - t0) * 1000

    print(f"BASELINE Chunk #0 TTS Synthesis: {base_tts_time:.1f}ms ({len(base_audio)} bytes)")
    print(f"OPTIMIZED Chunk #0 TTS Synthesis: {opt_tts_time:.1f}ms ({len(opt_audio)} bytes)")

    # 5. END-TO-END TTFA BREAKDOWN (From User Speech Cease)
    # Baseline Silence Detection: 320ms, Optimized Silence Detection: 200ms
    base_silence = 320.0
    opt_silence = 200.0

    # Web Audio dispatch / decode delay (typical ~10ms for short buffer)
    web_audio_decode = 8.0

    base_ttfa = base_silence + (base_chunk_time * 1000) + base_tts_time + web_audio_decode
    opt_ttfa = opt_silence + (opt_chunk_time * 1000) + opt_tts_time + web_audio_decode

    print(f"\n--- End-to-End TTFA (Time-to-First-Audio) Breakdown ---")
    print(f"| Component                     | Baseline          | Optimized (Jarvis) |")
    print(f"| ----------------------------- | ----------------- | ------------------ |")
    print(f"| Silence Detection Delay       | {base_silence:.0f} ms            | {opt_silence:.0f} ms             |")
    print(f"| LLM Stream -> Chunk #0 Ready  | {base_chunk_time*1000:.1f} ms         | {opt_chunk_time*1000:.1f} ms          |")
    print(f"| Deepgram Aura TTS Synthesis   | {base_tts_time:.1f} ms         | {opt_tts_time:.1f} ms          |")
    print(f"| Web Audio Decode & Playback   | {web_audio_decode:.1f} ms           | {web_audio_decode:.1f} ms            |")
    print(f"| ----------------------------- | ----------------- | ------------------ |")
    print(f"| TOTAL FIRST AUDIBLE SOUND     | {base_ttfa:.1f} ms        | {opt_ttfa:.1f} ms         |")
    print(f"| Latency Reduction             |                   | -{(base_ttfa - opt_ttfa):.1f} ms (-{((base_ttfa-opt_ttfa)/base_ttfa)*100:.1f}%) |")

    return {
        "query": query,
        "base_ttfa": base_ttfa,
        "opt_ttfa": opt_ttfa,
        "opt_chunk0": opt_chunk0,
    }

if __name__ == "__main__":
    queries = [
        ("What is the minimum CIBIL score for personal loans?", "en"),
        ("एचडीएफसी बँकेच्या वैयक्तिक कर्जासाठी व्याजदर काय आहे?", "mr"),
    ]
    results = []
    for q, lang in queries:
        try:
            res = benchmark_real_pipeline(q, lang)
            results.append(res)
        except Exception as e:
            print(f"Error benchmarking query '{q}':", e)
