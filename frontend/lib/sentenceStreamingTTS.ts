// ============================================================================
// REAL-TIME SENTENCE STREAMING TTS ENGINE (frontend/lib/sentenceStreamingTTS.ts)
// ============================================================================
// WHY THIS EXISTS:
// In traditional voice AI, the system waits for the entire LLM response to complete
// (which takes 3 to 6 seconds) before sending the full text to Text-to-Speech (TTS).
// This introduces frustrating delays for callers.
//
// HOW THIS WORKS (PIPELINE ARCHITECTURE):
// 1. SentenceTokenizer: As the LLM (DeepSeek) streams text word-by-word via SSE,
//    the tokenizer buffers characters until it finds a punctuation boundary (. ? ! । ॥).
// 2. Immediate Synthesis: As soon as the FIRST complete sentence forms (~300-500ms),
//    we immediately fire a parallel request to Deepgram TTS.
// 3. SentenceAudioQueue: While the browser plays sentence #1 out loud, sentences #2
//    and #3 are pre-fetched and synthesized in the background.
// 4. Instant Acoustic Barge-In: If the user interrupts by speaking, `bargeIn()` stops
//    playback, aborts in-flight fetch requests via AbortController, and resets the queue in 0ms.
// ============================================================================

// Common abbreviations in English and romanized text that end with a period
// but should NOT trigger a sentence break (e.g. "Mr. Sharma" is one sentence, not two).
const ABBREVIATIONS = new Set([
  "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "inc", "ltd", "corp",
  "co", "e.g", "i.e", "vs", "etc", "approx", "dept", "vol", "no",
  "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
]);

/**
 * Tokenizer that buffers incoming streaming tokens and yields complete sentences
 * as soon as punctuation boundaries are encountered.
 */
export class SentenceTokenizer {
  // Accumulates text tokens until a valid punctuation mark is reached
  private buffer: string = "";
  // Tracks if the current chunk is the first sentence of the turn for low-latency clause breaking
  private isFirstSentence: boolean = true;

  /**
   * Feeds a streaming text chunk and returns an array of any newly completed sentences.
   * If no sentence boundary has been reached yet, returns an empty array.
   */
  feed(chunk: string): string[] {
    this.buffer += chunk;
    const sentences: string[] = [];

    while (this.buffer.length > 0) {
      const match = this.findSentenceEnd(this.buffer);
      if (match === -1) {
        break;
      }

      const sentence = this.buffer.slice(0, match + 1).trim();
      this.buffer = this.buffer.slice(match + 1);

      if (sentence.length > 0) {
        sentences.push(sentence);
        this.isFirstSentence = false;
      }
    }

    return sentences;
  }

  /**
   * Flushes any remaining text in the buffer when the stream completes.
   */
  flush(): string | null {
    const remaining = this.buffer.trim();
    this.buffer = "";
    this.isFirstSentence = true;
    return remaining.length > 0 ? remaining : null;
  }

  /**
   * Clears the internal buffer (used during barge-in / abort).
   */
  reset(): void {
    this.buffer = "";
    this.isFirstSentence = true;
  }

  /**
   * Scans text for the first valid sentence terminator (. ! ? । ॥ \n\n).
   * Guards against decimals, currencies, ellipses, and abbreviations.
   */
  private findSentenceEnd(text: string): number {
    for (let i = 0; i < text.length; i++) {
      const char = text[i];

      // Double newline or newline before bullet points / list items
      if (char === "\n") {
        if (i + 1 < text.length && (text[i + 1] === "\n" || text[i + 1] === "-" || text[i + 1] === "*")) {
          return i;
        }
        if (i >= 12) {
          return i;
        }
      }

      // Colon followed by newline or bullet list or space (e.g. "Here are the key loan policies:\n")
      if (char === ":" && i >= 8) {
        if (i + 1 === text.length || /\s/.test(text[i + 1])) {
          return i;
        }
      }

      // Semicolon followed by whitespace (clause boundary in spoken responses)
      if (char === ";" && i >= 12) {
        if (i + 1 === text.length || /\s/.test(text[i + 1])) {
          return i;
        }
      }

      // Devanagari Danda (।) or Double Danda (॥)
      if (char === "\u0964" || char === "\u0965") {
        return i;
      }

      // Question mark or exclamation mark
      if (char === "?" || char === "!") {
        return i;
      }

      // Em-dash or en-dash clause break
      if ((char === "—" || char === "–") && i >= 10) {
        return i;
      }

      // Fast-path Jarvis-like first clause break:
      // If this is the very first sentence chunk:
      // 1. Allow comma clause break after >= 12 chars (e.g. "For personal loans,", "According to policy,")
      // 2. Allow colon clause break after >= 8 chars (e.g. "Under policy:")
      // 3. Allow natural word boundary (space) after >= 26 chars if no punctuation mark has appeared yet
      // This enables immediate ~180-220ms Deepgram Aura synthesis of the first 5-7 words for sub-second TTFA!
      if (this.isFirstSentence) {
        if (char === "," && i >= 12) {
          // Guard against numbers like 10,000 or 1,00,000
          const isNumberComma = i > 0 && /\d/.test(text[i - 1]) && i + 1 < text.length && /\d/.test(text[i + 1]);
          if (!isNumberComma && (i + 1 === text.length || /\s/.test(text[i + 1]))) {
            return i;
          }
        }
        if (char === ":" && i >= 8) {
          if (i + 1 === text.length || /\s/.test(text[i + 1])) {
            return i;
          }
        }
        if (char === " " && i >= 26) {
          return i;
        }
      }

      // Period (.)
      if (char === ".") {
        // 1. Guard against ellipsis (...)
        if (i + 1 < text.length && text[i + 1] === ".") {
          continue;
        }
        if (i > 0 && text[i - 1] === ".") {
          continue;
        }

        // 2. Guard against numbers/decimals (e.g. 99.99%, $1.5, 3.14)
        if (i > 0 && /\d/.test(text[i - 1])) {
          if (i + 1 < text.length && /\d/.test(text[i + 1])) {
            continue;
          }
          if (i + 1 === text.length) {
            continue;
          }
        }

        // 3. Guard against abbreviations (e.g. "Mr.", "Dr.", "e.g.", "i.e.")
        const precedingWordMatch = text.slice(0, i).match(/([a-zA-Z]+)$/);
        if (precedingWordMatch) {
          const word = precedingWordMatch[1].toLowerCase();
          if (ABBREVIATIONS.has(word) || word.length === 1) {
            continue;
          }
        }

        // Ensure there is trailing whitespace or end of string
        if (i + 1 === text.length || /\s/.test(text[i + 1])) {
          // Minimum sentence length guard: must have at least 4 characters
          if (i >= 4) {
            return i;
          }
        }
      }
    }

    return -1;
  }
}

interface QueuedSentence {
  id: number;
  text: string;
  lang: string;
  dataPromise: Promise<{ blob: Blob; audioBuffer?: AudioBuffer | null } | null>;
}

export interface SentenceAudioQueueCallbacks {
  onSentenceStart?: (text: string, index: number) => void;
  onSentenceEnd?: (text: string, index: number) => void;
  onQueueComplete?: () => void;
  onError?: (error: any) => void;
}

export interface SentenceAudioQueueOptions {
  module?: string;
  speaker?: string;
}

/**
 * Manages parallel TTS synthesis and in-order audio playback for streaming sentences.
 */
export class SentenceAudioQueue {
  private queue: QueuedSentence[] = [];
  private currentPlayingIndex: number = 0;
  private isPlaying: boolean = false;
  private isDrained: boolean = false;
  private isStreamCompleted: boolean = false;
  private audio: HTMLAudioElement;
  private currentObjectUrl: string | null = null;
  private abortController: AbortController = new AbortController();
  private callbacks: SentenceAudioQueueCallbacks = {};
  private options: SentenceAudioQueueOptions = {};
  private currentSentenceCounter: number = 0;
  private activeTurnId: number = 0;

  private activeSourceNode: AudioBufferSourceNode | null = null;
  private audioContext: AudioContext | null = null;

  constructor(
    audioElement: HTMLAudioElement,
    callbacks: SentenceAudioQueueCallbacks = {},
    audioCtx?: AudioContext | null,
    options: SentenceAudioQueueOptions = {}
  ) {
    this.audio = audioElement;
    this.callbacks = callbacks;
    if (audioCtx) {
      this.audioContext = audioCtx;
    }
    this.options = options;
  }

  /**
   * Updates configuration options (e.g. module or speaker override).
   */
  setOptions(options: SentenceAudioQueueOptions): void {
    this.options = { ...this.options, ...options };
  }

  /**
   * Supplies an AudioContext (e.g. from user gesture unlock) for bulletproof audio playback.
   */
  setAudioContext(ctx: AudioContext | null): void {
    this.audioContext = ctx;
  }

  private getAudioContext(): AudioContext | null {
    if (this.audioContext && this.audioContext.state !== "closed") {
      return this.audioContext;
    }
    if (typeof window !== "undefined") {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      if (AudioCtx) {
        this.audioContext = new AudioCtx();
        return this.audioContext;
      }
    }
    return null;
  }

  /**
   * Sets the active turn ID and resets queue state for a new turn.
   */
  /**
   * Sets the active turn ID and resets queue state for a new turn.
   */
  startNewTurn(turnId: number): void {
    this.bargeIn();
    this.activeTurnId = turnId;
    this.currentSentenceCounter = 0;
    this.currentPlayingIndex = 0;
    this.isPlaying = false;
    this.isDrained = false;
    this.isStreamCompleted = false;
    this.abortController = new AbortController();

    // Pre-warm / resume AudioContext on turn start to eliminate audio startup latency
    const ctx = this.getAudioContext();
    if (ctx && ctx.state === "suspended") {
      ctx.resume().catch(() => {});
    }
  }

  /**
   * Enqueues a completed sentence: immediately triggers Deepgram TTS fetch in parallel.
   * If the audio player is currently idle, playback starts immediately with this first sentence.
   */
  enqueueSentence(sentence: string, lang: string, extraOptions?: SentenceAudioQueueOptions): void {
    const cleanSentence = sentence.trim();
    if (!cleanSentence) return;

    const sentenceId = this.currentSentenceCounter++;
    const turnId = this.activeTurnId;
    const signal = this.abortController.signal;
    const targetModule = extraOptions?.module || this.options.module;
    const targetSpeaker = extraOptions?.speaker || this.options.speaker;

    console.log(`[StreamingTTS:Queue] Enqueuing Sentence #${sentenceId} (${lang}, module: ${targetModule || "default"}): "${cleanSentence}"`);

    // Pre-resume AudioContext immediately so audio subsystem is ready when synthesis finishes
    const ctx = this.getAudioContext();
    if (ctx && ctx.state === "suspended") {
      ctx.resume().catch(() => {});
    }

    // STEP 1: Fetch TTS audio in parallel immediately.
    // We decode the audio in the background as soon as the blob arrives, so playNext can play it with 0ms latency.
    const dataPromise = (async (): Promise<{ blob: Blob; audioBuffer?: AudioBuffer | null } | null> => {
      try {
        console.log(`[StreamingTTS:Fetch] Fetching TTS audio for sentence #${sentenceId}...`);
        const res = await fetch("/api/tts", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal, // Allows aborting this fetch immediately if the user barges in
          body: JSON.stringify({
            text: cleanSentence,
            language: lang,
            module: targetModule,
            speaker: targetSpeaker,
          }),
        });

        if (!res.ok) {
          console.error(`[StreamingTTS:Error] TTS fetch returned HTTP ${res.status} for sentence #${sentenceId}`);
          return null;
        }

        // Verify that this speech turn is still active and hasn't been cancelled
        if (this.activeTurnId !== turnId || signal.aborted) {
          console.warn(`[StreamingTTS:Cancel] Sentence #${sentenceId} discarded: turn changed (${this.activeTurnId} !== ${turnId}) or signal aborted (${signal.aborted})`);
          return null;
        }

        const blob = await res.blob();
        console.log(`[StreamingTTS:Synthesis] Sentence #${sentenceId} synthesized (${blob.size} bytes). Decoding in background...`);

        // Pre-decode into Web Audio AudioBuffer in background to eliminate playback initiation latency
        let audioBuffer: AudioBuffer | null = null;
        const currentCtx = this.getAudioContext();
        if (currentCtx && currentCtx.state !== "closed") {
          try {
            if (currentCtx.state === "suspended") {
              await currentCtx.resume().catch(() => {});
            }
            const arrayBuffer = await blob.arrayBuffer();
            audioBuffer = await currentCtx.decodeAudioData(arrayBuffer);
            console.log(`[StreamingTTS:WebAudio] Sentence #${sentenceId} pre-decoded in background (${audioBuffer.duration.toFixed(2)}s).`);
          } catch (decodeErr) {
            console.warn(`[StreamingTTS:WebAudio] Background decode fallback for sentence #${sentenceId}:`, decodeErr);
          }
        }

        return { blob, audioBuffer };
      } catch (err: any) {
        if (err?.name === "AbortError" || signal.aborted) {
          console.warn(`[StreamingTTS:Cancel] Sentence #${sentenceId} fetch aborted.`);
          return null;
        }
        console.error(`[StreamingTTS:Error] TTS synthesis failed for sentence #${sentenceId}:`, err);
        return null;
      }
    })();

    this.queue.push({
      id: sentenceId,
      text: cleanSentence,
      lang,
      dataPromise,
    });

    // STEP 2: If the audio player is not currently speaking, start playing immediately!
    if (!this.isPlaying) {
      this.playNext();
    }
  }

  /**
   * Signals that the LLM has finished streaming all tokens.
   */
  markStreamComplete(): void {
    this.isStreamCompleted = true;
    console.log(`[StreamingTTS:Stream] LLM token stream marked complete. Queue length: ${this.queue.length}, current index: ${this.currentPlayingIndex}`);
    if (!this.isPlaying && this.currentPlayingIndex >= this.queue.length) {
      this.finishQueue();
    }
  }

  /**
   * Plays the next queued sentence in strict in-order sequence.
   */
  private async playNext(): Promise<void> {
    if (this.currentPlayingIndex >= this.queue.length) {
      if (this.isStreamCompleted) {
        this.finishQueue();
      } else {
        // Wait for more sentences to stream in
        this.isPlaying = false;
      }
      return;
    }

    this.isPlaying = true;
    const currentItem = this.queue[this.currentPlayingIndex];
    const turnId = this.activeTurnId;

    try {
      console.log(`[StreamingTTS:Playback] Awaiting audio data for sentence #${currentItem.id}: "${currentItem.text.slice(0, 45)}..."`);
      // Wait for current sentence's audio data (meanwhile subsequent sentences are already being fetched & decoded!)
      const data = await currentItem.dataPromise;

      if (this.activeTurnId !== turnId || this.abortController.signal.aborted) {
        console.warn(`[StreamingTTS:Cancel] Playback of sentence #${currentItem.id} canceled: activeTurnId (${this.activeTurnId}) !== turnId (${turnId}) or aborted (${this.abortController.signal.aborted})`);
        return;
      }

      if (!data || !data.blob) {
        console.warn(`[StreamingTTS:Skip] Sentence #${currentItem.id} had no audio blob, advancing to next sentence.`);
        this.currentPlayingIndex++;
        this.playNext();
        return;
      }

      const { blob, audioBuffer } = data;

      // Check if Web Audio API (AudioContext) is active and running
      const ctx = this.getAudioContext();
      if (ctx && ctx.state === "suspended") {
        await ctx.resume().catch((e) => console.warn("[StreamingTTS:AudioContext] Failed to resume on playNext:", e));
      }

      if (ctx && ctx.state === "running") {
        console.log(`[StreamingTTS:Playback] Starting Web Audio API playback for sentence #${currentItem.id}...`);
        await this.playViaWebAudio(blob, audioBuffer || null, currentItem, turnId);
        return;
      }

      // Secondary path: HTMLAudioElement playback
      console.log(`[StreamingTTS:Playback] AudioContext state is '${ctx?.state || "unavailable"}'. Using HTMLAudioElement for sentence #${currentItem.id}...`);
      if (this.currentObjectUrl) {
        URL.revokeObjectURL(this.currentObjectUrl);
        this.currentObjectUrl = null;
      }

      const url = URL.createObjectURL(blob);
      this.currentObjectUrl = url;

      this.audio.src = url;
      this.audio.muted = false;
      this.audio.volume = 1.0;
      this.audio.playbackRate = (currentItem.lang === "hi" || currentItem.lang === "mr") ? 0.90 : 1.0;

      this.callbacks.onSentenceStart?.(currentItem.text, currentItem.id);

      this.audio.onended = () => {
        console.log(`[StreamingTTS:Playback] HTMLAudioElement sentence #${currentItem.id} ended.`);
        if (this.activeTurnId !== turnId) return;

        if (this.currentObjectUrl) {
          URL.revokeObjectURL(this.currentObjectUrl);
          this.currentObjectUrl = null;
        }

        this.callbacks.onSentenceEnd?.(currentItem.text, currentItem.id);
        this.currentPlayingIndex++;
        this.playNext();
      };

      this.audio.onerror = (e) => {
        console.error(`[StreamingTTS:Error] HTMLAudioElement error on sentence #${currentItem.id}:`, e);
        if (this.activeTurnId !== turnId) return;
        // Fallback to Web Audio API on element error
        this.playViaWebAudio(blob, audioBuffer || null, currentItem, turnId);
      };

      try {
        await this.audio.play();
        console.log(`[StreamingTTS:Playback] HTMLAudioElement sentence #${currentItem.id} is now playing.`);
      } catch (playErr: any) {
        console.warn(`[StreamingTTS:Autoplay] HTMLAudioElement.play() blocked/rejected (${playErr.name}: ${playErr.message}). Engaging Web Audio API fallback...`);
        if (this.activeTurnId !== turnId) return;
        await this.playViaWebAudio(blob, audioBuffer || null, currentItem, turnId);
      }
    } catch (err: any) {
      if (err?.name === "AbortError" || this.activeTurnId !== turnId) {
        console.warn(`[StreamingTTS:Cancel] Sentence #${currentItem.id} aborted in playNext.`);
        return;
      }
      console.error(`[StreamingTTS:Error] Exception in playNext for sentence #${currentItem.id}:`, err);
      this.callbacks.onError?.(err);
      this.currentPlayingIndex++;
      this.playNext();
    }
  }

  /**
   * Resilient Web Audio API fallback engine: plays decoded PCM buffer directly,
   * completely immune to DOM element visibility or HTMLAudioElement autoplay restrictions.
   */
  private async playViaWebAudio(
    blob: Blob,
    preDecodedBuffer: AudioBuffer | null,
    currentItem: QueuedSentence,
    turnId: number
  ): Promise<void> {
    try {
      const ctx = this.getAudioContext();
      if (!ctx) {
        console.error(`[StreamingTTS:Error] No AudioContext available for sentence #${currentItem.id}.`);
        this.callbacks.onSentenceEnd?.(currentItem.text, currentItem.id);
        this.currentPlayingIndex++;
        this.playNext();
        return;
      }

      if (ctx.state === "suspended") {
        await ctx.resume().catch((e) => console.warn("[StreamingTTS:AudioContext] Resume failed in playViaWebAudio:", e));
      }

      let audioBuffer = preDecodedBuffer;
      if (!audioBuffer) {
        console.log(`[StreamingTTS:WebAudio] Decoding audio data (${blob.size} bytes) for sentence #${currentItem.id}...`);
        const arrayBuffer = await blob.arrayBuffer();
        audioBuffer = await ctx.decodeAudioData(arrayBuffer);
      }

      if (this.activeTurnId !== turnId || this.abortController.signal.aborted) {
        console.warn(`[StreamingTTS:Cancel] Sentence #${currentItem.id} canceled during/after audio decoding.`);
        return;
      }

      const sourceNode = ctx.createBufferSource();
      sourceNode.buffer = audioBuffer;
      sourceNode.playbackRate.value = (currentItem.lang === "hi" || currentItem.lang === "mr") ? 0.90 : 1.0;
      sourceNode.connect(ctx.destination);
      this.activeSourceNode = sourceNode;

      this.callbacks.onSentenceStart?.(currentItem.text, currentItem.id);

      sourceNode.onended = () => {
        console.log(`[StreamingTTS:WebAudio] Sentence #${currentItem.id} audio playback completed (${audioBuffer!.duration.toFixed(2)}s).`);
        if (this.activeTurnId !== turnId) return;
        this.activeSourceNode = null;
        if (this.currentObjectUrl) {
          URL.revokeObjectURL(this.currentObjectUrl);
          this.currentObjectUrl = null;
        }
        this.callbacks.onSentenceEnd?.(currentItem.text, currentItem.id);
        this.currentPlayingIndex++;
        this.playNext();
      };

      sourceNode.start(0);
      console.log(`[StreamingTTS:WebAudio] Audio playing out loud! Sentence #${currentItem.id} (${audioBuffer.duration.toFixed(2)}s, sampleRate: ${audioBuffer.sampleRate}Hz)`);
    } catch (webAudioErr: any) {
      console.error(`[StreamingTTS:Error] Web Audio playback failure on sentence #${currentItem.id}:`, webAudioErr);
      if (this.activeTurnId !== turnId) return;
      this.callbacks.onSentenceEnd?.(currentItem.text, currentItem.id);
      this.currentPlayingIndex++;
      this.playNext();
    }
  }

  /**
   * Called when all sentences have finished playing.
   */
  private finishQueue(): void {
    if (this.isDrained) return;
    this.isDrained = true;
    this.isPlaying = false;
    console.log(`[StreamingTTS] All ${this.queue.length} sentences finished playback.`);
    this.callbacks.onQueueComplete?.();
  }

  /**
   * Updates callbacks with latest handler references.
   */
  updateCallbacks(callbacks: SentenceAudioQueueCallbacks): void {
    this.callbacks = callbacks;
  }

  /**
   * INSTANT BARGE-IN: Stops current playback, clears all pending items, and aborts in-flight TTS fetches.
   */
  bargeIn(): void {
    console.log("[StreamingTTS] Barge-in triggered. Halting audio and clearing queue.");

    // 1. Abort in-flight network requests
    this.abortController.abort();

    // 2. Stop Web Audio API active source immediately
    if (this.activeSourceNode) {
      try {
        this.activeSourceNode.onended = null;
        this.activeSourceNode.stop();
        this.activeSourceNode.disconnect();
      } catch (_) { }
      this.activeSourceNode = null;
    }

    // 3. Pause and reset audio player immediately WITHOUT destructive load() that destroys gesture priming
    try {
      this.audio.pause();
      this.audio.currentTime = 0;
      this.audio.onended = null;
      this.audio.onerror = null;
    } catch (e) {
      // ignore
    }

    // 4. Revoke existing blob URL
    if (this.currentObjectUrl) {
      URL.revokeObjectURL(this.currentObjectUrl);
      this.currentObjectUrl = null;
    }

    // 5. Reset queue state
    this.queue = [];
    this.isPlaying = false;
    this.isDrained = true;
  }

  /**
   * Returns true if audio is currently playing.
   */
  isCurrentlyPlaying(): boolean {
    return this.isPlaying;
  }
}
