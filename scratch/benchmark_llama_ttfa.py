import os
import sys
import time
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv('backend/.env')
sys.path.insert(0, 'backend')
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from services.rag import RAGService
from services.tts import DeepgramTTSService

def simulate_sentence_tokenizer_chunk(tokens):
    buf = ""
    for t_time, tok in tokens:
        buf += tok
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

def benchmark_model(model_name: str, query: str):
    print(f"\n=======================================================")
    print(f"TESTING MODEL: {model_name} | Query: '{query}'")
    print(f"=======================================================")

    rag = RAGService(model=model_name)
    tts = DeepgramTTSService()

    t_start = time.perf_counter()
    tokens = []
    first_token_time = None

    generator = rag.stream_answer_chunks(question=query, top_k=4, model=model_name)
    for raw_event in generator:
        if raw_event.startswith("event: token"):
            data_line = [l for l in raw_event.split("\n") if l.startswith("data:")][0]
            token_val = json.loads(data_line[5:].strip()).get("token", "")
            now = time.perf_counter() - t_start
            if first_token_time is None and token_val:
                first_token_time = now
            if token_val:
                tokens.append((now, token_val))

    full_answer = "".join(t[1] for t in tokens).strip()
    print(f"TTFT: {first_token_time*1000:.1f}ms")
    print(f"Full Answer: {full_answer[:100]}...")

    chunk0, chunk_time = simulate_sentence_tokenizer_chunk(tokens)
    print(f"Chunk #0 ({len(chunk0)} chars): '{chunk0}' (emitted at {chunk_time*1000:.1f}ms)")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    t0 = time.perf_counter()
    audio = loop.run_until_complete(tts.synthesize_speech(chunk0))
    tts_time = (time.perf_counter() - t0) * 1000
    print(f"Chunk #0 TTS Synthesis: {tts_time:.1f}ms ({len(audio)} bytes)")

    silence_delay = 200.0
    web_audio_decode = 8.0
    ttfa = silence_delay + (chunk_time * 1000) + tts_time + web_audio_decode

    print(f"\n--- TTFA Summary for {model_name} ---")
    print(f"Silence Delay: {silence_delay:.0f}ms")
    print(f"LLM to Chunk #0: {chunk_time*1000:.1f}ms")
    print(f"TTS Synthesis: {tts_time:.1f}ms")
    print(f"Web Audio: {web_audio_decode:.0f}ms")
    print(f"TOTAL TTFA: {ttfa:.1f}ms")

if __name__ == "__main__":
    benchmark_model("meta-llama/llama-3.3-70b-instruct", "What is the minimum CIBIL score for personal loans?")
