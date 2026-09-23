import requests
import json
import time

def test_query(question):
    print(f"=== Query: {question} ===")
    start = time.perf_counter()
    res = requests.post(
        "http://127.0.0.1:8001/api/ask",
        json={"question": question, "top_k": 4, "stream": True},
        stream=True,
        timeout=25
    )
    headers_time = (time.perf_counter() - start) * 1000
    
    first_token_time = None
    first_sentence_time = None
    first_sentence_text = ""
    buffer = ""
    all_sentences = []

    def check_sentence(b, is_first):
        if is_first and len(b) >= 28 and "," in b:
            idx = b.find(",")
            if idx >= 28:
                return idx
        for i, ch in enumerate(b):
            if ch in [".", "!", "?", "\n"]:
                if i >= 4 and (i + 1 == len(b) or b[i+1].isspace()):
                    return i
        return -1

    for line in res.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8")
        if line_str.startswith("data:"):
            data_str = line_str[5:].strip()
            try:
                data = json.loads(data_str)
                token = data.get("token") or data.get("delta") or ""
                if token:
                    now = (time.perf_counter() - start) * 1000
                    if first_token_time is None:
                        first_token_time = now
                    buffer += token
                    
                    idx = check_sentence(buffer, len(all_sentences) == 0)
                    if idx != -1:
                        s = buffer[:idx+1].strip()
                        buffer = buffer[idx+1:]
                        if first_sentence_time is None:
                            first_sentence_time = now
                            first_sentence_text = s
                        all_sentences.append((now, s))
            except:
                pass
                
    total_time = (time.perf_counter() - start) * 1000
    
    # If no delimiter found during stream, remaining buffer is flushed
    if buffer.strip() and not first_sentence_text:
        first_sentence_text = buffer.strip()
        first_sentence_time = total_time
        all_sentences.append((total_time, first_sentence_text))

    tts_time = 0
    if first_sentence_text:
        t0 = time.perf_counter()
        tts_res = requests.post("http://127.0.0.1:8001/api/tts", json={"text": first_sentence_text, "language": "en"})
        tts_time = (time.perf_counter() - t0) * 1000
        
    print(f"1. HTTP Headers Received: {headers_time:.1f} ms")
    print(f"2. Time to First Token (TTFT): {first_token_time:.1f} ms" if first_token_time else "2. TTFT: N/A")
    print(f"3. First Sentence / Chunk: {first_sentence_time:.1f} ms -> \"{first_sentence_text}\"")
    print(f"4. First Sentence TTS Synthesis: {tts_time:.1f} ms")
    print(f"5. First Spoken Audio Ready (Optimized TTFA): {(first_sentence_time + tts_time):.1f} ms")
    print(f"6. Total Stream Completion: {total_time:.1f} ms")
    print(f"Total Sentences Formed: {len(all_sentences)}")
    print()
    return {
        "headers": headers_time,
        "ttft": first_token_time,
        "first_sentence": first_sentence_time,
        "tts": tts_time,
        "first_audio": first_sentence_time + tts_time if first_sentence_time else 0,
        "total": total_time
    }

if __name__ == "__main__":
    r1 = test_query("What is the minimum CIBIL score required for HDFC home loans?")
    r2 = test_query("What are the age requirements for loan applicants?")
