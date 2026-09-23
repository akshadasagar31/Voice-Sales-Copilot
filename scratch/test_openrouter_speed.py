import httpx, os, time, json
from dotenv import load_dotenv
load_dotenv("backend/.env")

api_key = os.getenv("OPENROUTER_API_KEY")
url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

async def benchmark_config(name, payload):
    t0 = time.perf_counter()
    first_token = None
    first_sentence = None
    buffer = ""
    async with httpx.AsyncClient(timeout=15.0) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data:"):
                    d = line[5:].strip()
                    if d == "[DONE]":
                        break
                    try:
                        c = json.loads(d)
                        tok = c["choices"][0]["delta"].get("content", "")
                        if tok:
                            if first_token is None:
                                first_token = (time.perf_counter() - t0) * 1000
                            buffer += tok
                            if ("," in buffer and len(buffer) > 12) or ("." in buffer and len(buffer) > 8) or (len(buffer) > 25):
                                if first_sentence is None:
                                    first_sentence = (time.perf_counter() - t0) * 1000
                    except:
                        pass
    total = (time.perf_counter() - t0) * 1000
    print(f"{name:35} | TTFT: {first_token or 0:6.1f}ms | First Clause/Sent: {first_sentence or 0:6.1f}ms | Total: {total:6.1f}ms")
    print(f"  Sample: '{buffer[:80]}...'")

import asyncio
async def main():
    print("Testing OpenRouter DeepSeek configurations:")
    
    # 1. Base deepseek-chat
    p1 = {
        "model": "deepseek/deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a sales assistant. Answer directly and concisely in 1 sentence."},
            {"role": "user", "content": "What is the maximum loan amount?"}
        ],
        "stream": True,
        "max_tokens": 80,
    }
    await benchmark_config("deepseek/deepseek-chat (base)", p1)

    # 2. deepseek-chat with latency sort
    p2 = dict(p1)
    p2["provider"] = {"sort": "latency"}
    await benchmark_config("deepseek-chat (sort: latency)", p2)

    # 3. deepseek-chat with low temperature and prompt caching
    p3 = dict(p1)
    p3["temperature"] = 0.0
    await benchmark_config("deepseek-chat (temp: 0.0)", p3)

asyncio.run(main())
