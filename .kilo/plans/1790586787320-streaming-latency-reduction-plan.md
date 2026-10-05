# Streaming Architecture Plan: End-to-End Latency Reduction

**Goal**: Minimize speech-to-audio latency from sequential (~2–4s) to real-time streaming with a target of **300–600ms** (treated as an optimization target, with explicit measurement of actual *final-STT → first-audio* latency).

---

## Architecture Overview

```
User Speech 
   → AudioWorklet (16/48kHz PCM capture on audio thread, replacing ScriptProcessorNode)
   → Existing Backend WebSocket `/ws/voice-stt` (Deepgram Nova-3 / Active STT from ModelManager)
   → Speech Final Signal (<200ms target)
   → Fast-Path Lead Prompt OR Streaming LLM Tokens (Active LLM from ModelManager)
   → Existing SentenceTokenizer (Sentence 1 boundary detected immediately)
   → Existing SentenceAudioQueue (Active TTS from ModelManager: Sarvam / Deepgram)
   → Immediate Audio Playback (<400ms target from STT final)
   → Background pre-fetch of Sentence 2, 3 in parallel
```

---

## Core Principles & Constraints

1. **Reuse Existing `/ws/voice-stt`**:
   - **No duplicate STT WebSocket**. Use the existing `/ws/voice-stt` route in `backend/main.py`.
   - `/ws/voice-stt` already dynamically resolves the active STT model and credentials from `ModelManager`.
   - Sends real-time `interim` results for live UI captions and instant `final` results upon Deepgram `speech_final`.

2. **AudioWorklet Replaces ScriptProcessorNode**:
   - Deploy `frontend/public/worklets/pcm-capture-processor.js` to run on the browser's dedicated audio rendering thread.
   - Replaces deprecated `ScriptProcessorNode` in `LiveCallVoiceCopilot.tsx` to eliminate main-thread UI jank, stutter, and garbage-collection frame drops.
   - Pre-roll buffer (~850ms) and acoustic VAD remain intact to ensure first syllables are never lost.

3. **Reuse Existing `SentenceTokenizer` + `SentenceAudioQueue`**:
   - Do NOT create a duplicate queue or tokenizer.
   - In `frontend/lib/sentenceStreamingTTS.ts`, reuse the working `SentenceTokenizer` and `SentenceAudioQueue`.
   - Boundary detection: Triggers immediately on first sentence punctuation (comma, colon, or sentence end for early tokens) to feed TTS with minimal token accumulation delay.

4. **Immediate Streaming Pipeline: LLM → Sentence Detection → TTS → Playback**:
   - Do not wait for complete LLM response.
   - Fast-path turns (Module 1 sequential parameter prompts, greetings, refusals) yield sentence 1 in <5ms.
   - Dynamic LLM turns (Module 1 general questions, Module 2 knowledge queries) stream tokens via SSE/WS directly into `SentenceTokenizer`.
   - As soon as sentence 1 completes, it is enqueued into `SentenceAudioQueue`, which starts playback immediately on first audio arrival while sentence 2 synthesizes in parallel.

5. **Single Pipeline per Voice Turn with Instant Barge-In / Cancellation**:
   - Strict `turnIdRef` incrementing on every new turn.
   - Instant acoustic barge-in halts assistant audio within 0ms, clears audio queues, cancels in-flight fetch requests, and captures interrupting speech.
   - Stale turns automatically discard late tokens, responses, and audio chunks.

6. **Dynamic Model Resolution from `ModelManager`**:
   - STT: Uses active model (built-in Deepgram Nova-3 or custom STT).
   - LLM: Uses active model (built-in DeepSeek or custom LLM) in `LeadExtractorService` and `RAGService`.
   - TTS: Uses active model (built-in Sarvam or Deepgram, or custom TTS) for `/api/tts/synthesize` and sentence-streaming audio.

7. **Preserve Module 1 Deterministic Lead Flow & Module 2 RAG/Pinecone**:
   - Module 1: Deterministic 6-field sequential prompt flow (`name -> phone -> company -> loan_type -> loan_amount -> tenure_months`), PostgreSQL CRM saving, field inquiries, refusals, resumption, and interleaved general question handling are 100% preserved.
   - Module 2: RAG retrieval, Pinecone vector search, document playbooks, and knowledge queries remain untouched and functional.

8. **Target Metric & Empirical Measurement**:
   - 300–600ms is treated as a **target range, not a guarantee** across all network conditions and third-party APIs.
   - Measure and log exact wall-clock timestamps on every turn:
     - $T_0$: Speech ended (VAD silence cutoff)
     - $T_1$: Final STT received (`speech_final`)
     - $T_2$: First LLM token / fast-path sentence 1 ready
     - $T_3$: First TTS audio buffer decoded
     - $T_4$: First audio playback starts (`onSentenceStart`)
   - Primary measured metric: **Actual Final-STT → First-Audio Latency** ($T_4 - T_1$) and **Total Speech-End → First-Audio Latency** ($T_4 - T_0$).

---

## Exact Files to Change

| Component | File Path | Action | Description |
|-----------|-----------|--------|-------------|
| **Frontend Audio** | `frontend/public/worklets/pcm-capture-processor.js` | **NEW** | `AudioWorkletProcessor` capturing 16kHz/48kHz PCM frames and computing RMS on the audio thread. |
| **Frontend Voice Copilot** | `frontend/app/components/LiveCallVoiceCopilot.tsx` | **MODIFY** | Replace `createScriptProcessor` with `AudioWorkletNode`. Connect directly to existing `/ws/voice-stt`. Measure and log `final-STT -> first-audio` latency. |
| **Frontend TTS Queue** | `frontend/lib/sentenceStreamingTTS.ts` | **MODIFY** | Optimize `SentenceTokenizer` for fast sentence-1 boundary detection. Maintain `SentenceAudioQueue` parallel pre-fetching and gapless Web Audio scheduling. |
| **Backend STT / Voice** | `backend/main.py` | **MODIFY** | Verify `/ws/voice-stt` handles binary PCM frames smoothly from AudioWorklet, preserves dynamic STT resolution via `ModelManager`, and returns fast-path sentence 1 on `speech_final`. |
| **Backend LLM Stream** | `backend/services/lead_extractor.py` | **MODIFY** | Optimize `stream_lead_turn` / `answer_general_query` to stream tokens immediately with pre-warmed keep-alive client session. |
| **Testing & Benchmark** | `testing/latency_benchmark.py` | **NEW** | Automated benchmark script measuring STT latency, LLM first-token latency, TTS latency, and actual final-STT → first-audio playback latency over 50 iterations. |

---

## Implementation Details

### 1. `frontend/public/worklets/pcm-capture-processor.js` (NEW)
Replaces `ScriptProcessorNode`. Runs on the Web Audio rendering thread:
```javascript
class PCMCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.bufferSize = options.processorOptions?.bufferSize || 2048;
    this.buffer = new Float32Array(this.bufferSize);
    this.bufferIndex = 0;
  }

  process(inputs, outputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channel = input[0];

    // Compute frame RMS for VAD speech detection
    let sum = 0;
    for (let i = 0; i < channel.length; i++) {
      sum += channel[i] * channel[i];
    }
    const rms = Math.sqrt(sum / channel.length);

    // Buffer samples into chunks
    for (let i = 0; i < channel.length; i++) {
      this.buffer[this.bufferIndex++] = channel[i];
      if (this.bufferIndex >= this.bufferSize) {
        this.port.postMessage({
          type: "audio_data",
          buffer: this.buffer.slice(0, this.bufferSize),
          rms: rms,
        });
        this.bufferIndex = 0;
      }
    }
    return true;
  }
}

registerProcessor("pcm-capture-processor", PCMCaptureProcessor);
```

### 2. `frontend/app/components/LiveCallVoiceCopilot.tsx` (MODIFY)
- Load worklet via `await audioCtx.audioWorklet.addModule("/worklets/pcm-capture-processor.js")`.
- Connect: `source -> highpass -> lowpass -> voicePeaking -> audioWorkletNode -> silentGain -> destination`.
- `audioWorkletNode.port.onmessage`:
  - Receives `Float32Array` buffer and `rms`.
  - Maintains ~850ms pre-roll circular buffer.
  - Detects speech onset and silence cutoff (400ms).
  - Streams binary PCM to the **existing** `/ws/voice-stt` WebSocket with `sendStreamingChunk`.
- On `speech_final` from `/ws/voice-stt`:
  - Record $T_1$ (`performance.now()`).
  - Trigger fast-path prompt or streaming LLM tokens.
  - Log actual latency breakdown when audio begins: `latency = T4 - T1`.

### 3. `frontend/lib/sentenceStreamingTTS.ts` (MODIFY)
- Reuse existing `SentenceTokenizer` and `SentenceAudioQueue`.
- Ensure first sentence boundary triggers on comma or colon after at least 3 words if LLM stream is active, or standard sentence punctuation (`.`, `!`, `?`).
- `SentenceAudioQueue`:
  - Enqueues sentence 1 into TTS fetch immediately.
  - Starts audio playback on sentence 1 arrival via Web Audio API.
  - Pre-fetches sentence 2 and 3 concurrently while sentence 1 plays.
  - Emits `onSentenceStart` with precise timing to measure final-STT → first-audio latency.

### 4. `backend/main.py` (MODIFY)
- Ensure existing `/ws/voice-stt` endpoint:
  - Dynamically uses the active STT model and key from `ModelManager`.
  - Forwards binary PCM frames to Deepgram Nova-3 (`endpointing=350`, `interim_results=true`).
  - On `speech_final`, calls `generate_module1_ws_response` to yield the deterministic sentence 1 prompt and next missing lead parameter in <5ms.
  - Cleanly handles socket close, barge-in `CloseStream`, and error recovery without hanging connections.

---

## Verification Plan

### Automated Tests
1. **Existing Backend Tests**:
   ```bash
   .\backend\.venv\Scripts\python.exe -m pytest backend/tests/
   ```
   *Must maintain 297/297 passing tests (zero regressions).*

2. **Latency Benchmark Suite**:
   ```bash
   .\backend\.venv\Scripts\python.exe testing/latency_benchmark.py
   ```
   *Measures STT turnaround, LLM first token, TTS synthesis, and end-to-end latency.*

3. **Frontend Build & Typecheck**:
   ```bash
   npm run build
   ```

### Manual Verification
1. **Microphone Capture via AudioWorklet**:
   - Start voice mode in `LiveCallVoiceCopilot.tsx`.
   - Confirm AudioWorklet loads and processes audio without console warnings or ScriptProcessor deprecation notices.
2. **WebSocket STT Streaming**:
   - Verify `/ws/voice-stt` receives audio frames and produces live interim transcripts.
3. **Turn-by-Turn Latency Logging**:
   - Speak: `"My name is Rahul Sharma"`
   - Verify console outputs wall-clock latency: `[VoiceCopilot:Latency] Final-STT to First-Audio Playback: XXX ms`.
4. **Barge-In**:
   - Interrupt while assistant is speaking. Verify playback stops in 0ms, audio queue clears, and new turn begins immediately.
5. **ModelManager Switching**:
   - Switch models in `/models` tab; verify Module 1 reflects new models on next voice turn.

---

## Module 1 Voice Issues Resolution (10 Mandatory Requirements)

1. **Immediate Streaming STT Onset**: Start streaming/interim STT immediately when the prospect starts speaking (on 2 consecutive frames ~85ms of sustained speech VAD).
2. **Continuous Live Interim Display**: Continuously display live partial transcripts in the UI while the prospect is speaking via WebSocket `interim` events.
3. **Final-Only Lead Extraction**: Use ONLY final transcripts (`data.type === "final"`) for lead extraction, CRM updates, and LLM replies. Interims are strictly visual.
4. **Explicit Language Mapping (No `multi/auto` in Final STT)**:
   - Marathi → Deepgram `mr`
   - Hindi → Deepgram `hi`
   - English → Deepgram `en-IN`
   - Completely remove `multi` and `auto` from Module 1 final STT path.
5. **Romanized Marathi Detection & Exact Name Preservation**: Expand `ROMANIZED_MARATHI` and `ROMANIZED_MARATHI_EXCLUSIVE` in `language.py`. Preserve prospect names and transcripts verbatim with zero translation or autocorrection.
6. **Keyterm Sanitization**: Clean and filter malformed, empty, or colon-tagged (`term:3`) items in `stt.py` to prevent Deepgram 400 Bad Request errors.
7. **Single Unified Architecture**: Preserve existing `AudioWorkletNode` + `/ws/voice-stt` architecture with no parallel or duplicate STT pipelines.
8. **No Changes to Unrelated Systems**: Keep Module 2, RAG, Pinecone, Telegram, UI, TTS, Models Management, and working lead logic untouched.
9. **Exact Minimal Files to Change**:
   - `backend/services/stt.py`
   - `backend/services/language.py`
   - `backend/main.py`
   - `frontend/app/components/LiveCallVoiceCopilot.tsx`
   - `backend/tests/test_module1_stt.py`
10. **Full Verification**: Run all backend tests (`pytest backend/tests/`), TypeScript checks (`npx tsc --noEmit`), and frontend production build (`npm run build`).