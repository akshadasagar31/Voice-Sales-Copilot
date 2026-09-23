import os
import sys
import time
import json
import asyncio
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

def run_benchmark(model_name: str, query: str):
    rag = RAGService(model=model_name)
    tts = DeepgramTTSService()

    t_start = time.perf_counter()
    tokens = []
    first_token_time = None

    generator = rag.stream_answer_chunks(question=query, top_k=3, model=model_name)
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
    chunk0, chunk_time = simulate_sentence_tokenizer_chunk(tokens)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    t0 = time.perf_counter()
    audio = loop.run_until_complete(tts.synthesize_speech(chunk0))
    tts_time = (time.perf_counter() - t0) * 1000

    silence_delay = 200.0
    web_audio = 8.0
    ttfa = silence_delay + (chunk_time * 1000) + tts_time + web_audio

    return {
        "model": model_name,
        "ttft_ms": first_token_time * 1000 if first_token_time else 0.0,
        "chunk0_ms": chunk_time * 1000,
        "tts_ms": tts_time,
        "ttfa_ms": ttfa,
        "chunk0": chunk0,
        "answer": full_answer[:80],
    }

if __name__ == "__main__":
    q = "What is the minimum CIBIL score for personal loans?"
    models = ["meta-llama/llama-3.3-70b-instruct", "deepseek/deepseek-chat"]
    print("=================================================================")
    print("HEAD-TO-HEAD BENCHMARK: LLAMA 3.3 70B vs DEEPSEEK CHAT")
    print("=================================================================")
    for m in models:
        try:
            res = run_benchmark(m, q)
            print(f"\nModel: {res['model']}")
            print(f"  TTFT:            {res['ttft_ms']:6.1f} ms")
            print(f"  Chunk #0 Ready:  {res['chunk0_ms']:6.1f} ms ('{res['chunk0']}')")
            print(f"  TTS Synthesis:   {res['tts_ms']:6.1f} ms")
            print(f"  TOTAL TTFA:      {res['ttfa_ms']:6.1f} ms")
            print(f"  Grounded Answer: '{res['answer']}...'")
        except Exception as e:
            print(f"Error on {m}: {e}")
