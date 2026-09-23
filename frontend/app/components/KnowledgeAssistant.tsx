// ============================================================================
// MODULE 2: REAL-TIME RAG KNOWLEDGE ASSISTANT
// (frontend/app/components/KnowledgeAssistant.tsx)
// ============================================================================
// WHAT THIS COMPONENT DOES:
// An intelligent sales copilot that answers prospect questions and objections in
// real-time using Retrieval-Augmented Generation (RAG) strictly grounded in
// uploaded PDF sales playbooks and credit policies.
//
// PIPELINE ARCHITECTURE:
// 1. Hands-Free Voice Mode: Web Audio API listens with ~850ms pre-roll PCM buffer.
// 2. 400ms Silence Detection: Automatically detects end of user question.
// 3. Deepgram STT: Transcribes question and detects language (EN, HI, MR).
// 4. Pinecone Vector Search: Fast semantic similarity retrieval of top-k playbook chunks.
// 5. DeepSeek RAG LLM: Streams grounded answer tokens via Server-Sent Events (SSE).
//    Zero hallucination guardrail: Returns strict fallback if answer not in playbooks.
// 6. Sentence Streaming TTS: Synthesizes and plays sentence #1 in ~400ms while sentences
//    #2 and #3 synthesize in parallel.
// 7. Instant Acoustic Barge-In: 0ms audio interruption if prospect interrupts.
// 8. Auto-Listening: Hands-free loop continuously listens for the next question.
// ============================================================================

"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  Sparkles,
  Send,
  BookOpen,
  Copy,
  Check,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  FileText,
  SlidersHorizontal,
  Bot,
  User,
  Zap,
  RotateCcw,
  ShieldAlert,
  Mic,
  Square,
  Volume2,
  Loader2,
  Radio,
  X,
  PhoneOff,
} from "lucide-react";
import { SentenceTokenizer, SentenceAudioQueue } from "@/lib/sentenceStreamingTTS";


interface ContextChunk {
  id?: string;
  score?: number;
  source?: string;
  page?: number;
  chunk_index?: number;
  text?: string;
}

interface QAMessage {
  id: string;
  question: string;
  answer: string;
  sources: string[];
  context_used: ContextChunk[];
  model: string;
  fallback_used?: boolean;
  timestamp: string;
  language?: string;
  is_greeting?: boolean;
}

type VoiceState = "idle" | "listening" | "processing" | "speaking";
type ProcessingStage = "transcribing" | "retrieving" | "synthesizing" | null;

interface PromptItem {
  text: string;
  lang: string;
  label: string;
}

/**
 * Converts accumulated Float32Array PCM audio chunks into a standard 16-bit linear PCM WAV Blob.
 * This guarantees lossless audio encoding with zero container delay or header corruption,
 * and preserves the leading audio from the pre-roll buffer so no first words are missed.
 */
function encodeWav(buffers: Float32Array[], sampleRate: number): Blob {
  let totalLength = 0;
  for (let i = 0; i < buffers.length; i++) {
    totalLength += buffers[i].length;
  }
  const arrayBuffer = new ArrayBuffer(44 + totalLength * 2);
  const view = new DataView(arrayBuffer);

  const writeString = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  // RIFF chunk descriptor
  writeString(0, "RIFF");
  view.setUint32(4, 36 + totalLength * 2, true);
  writeString(8, "WAVE");

  // "fmt " sub-chunk
  writeString(12, "fmt ");
  view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
  view.setUint16(20, 1, true); // AudioFormat (1 for uncompressed PCM)
  view.setUint16(22, 1, true); // NumChannels (1 mono)
  view.setUint32(24, sampleRate, true); // SampleRate
  view.setUint32(28, sampleRate * 2, true); // ByteRate (SampleRate * 1 channel * 2 bytes/sample)
  view.setUint16(32, 2, true); // BlockAlign (1 channel * 2 bytes/sample)
  view.setUint16(34, 16, true); // BitsPerSample (16 bits)

  // "data" sub-chunk
  writeString(36, "data");
  view.setUint32(40, totalLength * 2, true);

  // Write 16-bit linear PCM samples
  let offset = 44;
  for (let b = 0; b < buffers.length; b++) {
    const chunk = buffers[b];
    for (let i = 0; i < chunk.length; i++) {
      const s = Math.max(-1, Math.min(1, chunk[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }
  }

  return new Blob([view], { type: "audio/wav" });
}

/**
 * Converts a Float32Array PCM chunk (-1.0 to 1.0) into a standard 16-bit linear PCM ArrayBuffer.
 * Sent in real-time over WebSocket to Deepgram Nova-3 for zero-latency streaming STT.
 */
function convertFloat32ToInt16(chunk: Float32Array): ArrayBuffer {
  const pcm16 = new Int16Array(chunk.length);
  for (let i = 0; i < chunk.length; i++) {
    const s = Math.max(-1, Math.min(1, chunk[i]));
    pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return pcm16.buffer;
}

const SUGGESTED_PROMPTS: PromptItem[] = [
  { text: "What are our volume discount tiers for enterprise contracts?", lang: "EN", label: "English" },
  { text: "How do we handle competitor claims regarding 99.99% uptime SLAs?", lang: "EN", label: "English" },
  { text: "नमस्ते", lang: "HI", label: "Greeting (हिंदी)" },
  { text: "क्लाउड सुरक्षा, डेटा एन्क्रिप्शन और अनुपालन नीतियां क्या हैं?", lang: "HI", label: "हिंदी" },
  { text: "नमस्कार", lang: "MR", label: "Greeting (मराठी)" },
  { text: "कंपनीचे एंटरप्राइज सवलत दर आणि करार कालावधी पर्याय काय आहेत?", lang: "MR", label: "मराठी" },
];

export default function KnowledgeAssistant() {
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(4);
  const [namespace, setNamespace] = useState("sales_playbooks");
  const [showSettings, setShowSettings] = useState(false);
  const [expandedContexts, setExpandedContexts] = useState<Record<string, boolean>>({});
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [messages, setMessages] = useState<QAMessage[]>([]);
  const [selectedLanguage, setSelectedLanguage] = useState<string>("auto");
  const [detectedLanguage, setDetectedLanguage] = useState<string>("en");

  // Continuous Hands-Free ChatGPT-Style Voice Interaction States
  const [voiceState, setVoiceState] = useState<VoiceState>("idle");
  const [isContinuousMode, setIsContinuousMode] = useState(false);
  const [isSilenceCountdown, setIsSilenceCountdown] = useState(false);
  const [processingStage, setProcessingStage] = useState<ProcessingStage>(null);
  const [listeningSeconds, setListeningSeconds] = useState(0);
  const [audioLevels, setAudioLevels] = useState<number[]>(new Array(24).fill(14));
  const [liveTranscript, setLiveTranscript] = useState<string | null>(null);
  const [activePlayingMsgId, setActivePlayingMsgId] = useState<string | null>(null);
  const [ttsAudioUrl, setTtsAudioUrl] = useState<string | null>(null);

  // Audio capture & Web Audio API refs
  const audioStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioSourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const silentGainRef = useRef<GainNode | null>(null);
  const timerIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const silenceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const ttsAudioRef = useRef<HTMLAudioElement | null>(null);
  const activeAudioUrlRef = useRef<string | null>(null);
  const sentenceAudioQueueRef = useRef<SentenceAudioQueue | null>(null);

  // Live Deepgram Nova-3 WebSocket STT refs
  const sttSocketRef = useRef<WebSocket | null>(null);
  const hasWsFinalizedRef = useRef<boolean>(false);
  const wsFallbackTimerRef = useRef<NodeJS.Timeout | null>(null);
  const isWsStreamingRef = useRef<boolean>(false);
  const wsPendingQueueRef = useRef<Float32Array[]>([]);

  // Continuous pre-roll & recorded PCM buffers (guarantees NO missed first words on barge-in)
  const preRollBuffersRef = useRef<Float32Array[]>([]);
  const recordedBuffersRef = useRef<Float32Array[]>([]);
  const isSpeechTurnActiveRef = useRef<boolean>(false);
  const speechDetectedRef = useRef<boolean>(false);
  const consecutiveBargeInFramesRef = useRef<number>(0);
  const consecutiveSpeechFramesRef = useRef<number>(0);
  const speakerBleedBaselineRef = useRef<number>(18);
  const bleedRmsRef = useRef<number>(0.012);
  const sentenceStartTimestampRef = useRef<number>(0);
  const speechEndTimestampRef = useRef<number>(0);

  // Turn ID & Concurrency control
  const activeTurnIdRef = useRef<number>(0);
  const activeTurnAbortControllerRef = useRef<AbortController | null>(null);
  const isVoicePipelineActiveRef = useRef<boolean>(false);
  const isSTTProcessingRef = useRef<boolean>(false);
  const activeTTSAbortControllerRef = useRef<AbortController | null>(null);
  const activeTTSRequestIdRef = useRef<number>(0);

  // Mode and state tracking
  const isContinuousModeRef = useRef<boolean>(false);
  const voiceStateRef = useRef<VoiceState>("idle");

  // Sync refs with state
  useEffect(() => {
    isContinuousModeRef.current = isContinuousMode;
  }, [isContinuousMode]);

  useEffect(() => {
    voiceStateRef.current = voiceState;
  }, [voiceState]);

  // Invalidate any active voice pipeline turn and abort ongoing network operations
  const cancelActiveTurn = useCallback((abortInFlight = true) => {
    activeTurnIdRef.current += 1;
    if (abortInFlight && activeTurnAbortControllerRef.current) {
      try {
        activeTurnAbortControllerRef.current.abort();
      } catch (_) { }
      activeTurnAbortControllerRef.current = null;
    }
    if (wsFallbackTimerRef.current) {
      clearTimeout(wsFallbackTimerRef.current);
      wsFallbackTimerRef.current = null;
    }
    if (sttSocketRef.current) {
      try {
        sttSocketRef.current.close();
      } catch (_) { }
      sttSocketRef.current = null;
    }
    wsPendingQueueRef.current = [];
    isWsStreamingRef.current = false;
    isVoicePipelineActiveRef.current = false;
    isSTTProcessingRef.current = false;
  }, []);

  // Authoritative Audio Controller: immediately stops playback, aborts TTS fetches, and resets audio element
  const stopAllAudioPlayback = useCallback((cancelPendingRequests = true) => {
    sentenceStartTimestampRef.current = 0;
    if (sentenceAudioQueueRef.current) {
      try {
        sentenceAudioQueueRef.current.bargeIn();
      } catch (_) { }
    }

    if (cancelPendingRequests) {
      if (activeTTSAbortControllerRef.current) {
        try {
          activeTTSAbortControllerRef.current.abort();
        } catch (_) { }
        activeTTSAbortControllerRef.current = null;
      }
      activeTTSRequestIdRef.current += 1;
      cancelActiveTurn(true);
    }

    if (ttsAudioRef.current) {
      try {
        ttsAudioRef.current.pause();
        ttsAudioRef.current.currentTime = 0;
        ttsAudioRef.current.onended = null;
        ttsAudioRef.current.onerror = null;
      } catch (_) { }
    }

    if (activeAudioUrlRef.current) {
      try {
        URL.revokeObjectURL(activeAudioUrlRef.current);
      } catch (_) { }
      activeAudioUrlRef.current = null;
    }
    setTtsAudioUrl(null);
    setActivePlayingMsgId(null);
  }, [cancelActiveTurn]);

  // Prime/unlock audio playback within direct user gesture context to bypass browser autoplay restrictions
  const unlockAudioPlayback = useCallback(() => {
    if (typeof window === "undefined") return;
    try {
      if (!audioContextRef.current || audioContextRef.current.state === "closed") {
        const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
        if (AudioCtx) {
          audioContextRef.current = new AudioCtx();
        }
      }
      if (audioContextRef.current && audioContextRef.current.state === "suspended") {
        audioContextRef.current.resume().then(() => {
          console.log("[VoiceAssistant:Audio] AudioContext resumed, state:", audioContextRef.current?.state);
        }).catch((e) => {
          console.warn("[VoiceAssistant:Audio] AudioContext resume notice:", e);
        });
      }
      if (sentenceAudioQueueRef.current && audioContextRef.current) {
        sentenceAudioQueueRef.current.setAudioContext(audioContextRef.current);
      }
      if (!ttsAudioRef.current) {
        ttsAudioRef.current = new Audio();
      }
      if (ttsAudioRef.current) {
        ttsAudioRef.current.muted = false;
        ttsAudioRef.current.volume = 1.0;
        const primePromise = ttsAudioRef.current.play();
        if (primePromise !== undefined) {
          primePromise.catch(() => { });
        }
      }
    } catch (e) {
      console.warn("[VoiceAssistant:Audio] Audio autoplay prime warning:", e);
    }
  }, []);

  // Global user gesture unlock listener: automatically primes audio on first user click or key anywhere
  useEffect(() => {
    if (typeof window === "undefined") return;
    const globalUnlock = () => {
      unlockAudioPlayback();
    };
    window.addEventListener("click", globalUnlock, { passive: true });
    window.addEventListener("keydown", globalUnlock, { passive: true });
    return () => {
      window.removeEventListener("click", globalUnlock);
      window.removeEventListener("keydown", globalUnlock);
    };
  }, [unlockAudioPlayback]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cancelActiveTurn(true);
      stopAllAudioPlayback(true);
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
      if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
      if (analyserRef.current) {
        try {
          analyserRef.current.disconnect();
        } catch (_) { }
        analyserRef.current = null;
      }
      if (scriptProcessorRef.current) {
        try {
          scriptProcessorRef.current.disconnect();
        } catch (_) { }
        scriptProcessorRef.current = null;
      }
      if (silentGainRef.current) {
        try {
          silentGainRef.current.disconnect();
        } catch (_) { }
        silentGainRef.current = null;
      }
      if (audioSourceNodeRef.current) {
        try {
          audioSourceNodeRef.current.disconnect();
        } catch (_) { }
        audioSourceNodeRef.current = null;
      }
      if (audioStreamRef.current) {
        audioStreamRef.current.getTracks().forEach((track) => track.stop());
        audioStreamRef.current = null;
      }
      if (audioContextRef.current && audioContextRef.current.state !== "closed") {
        audioContextRef.current.close().catch(() => { });
        audioContextRef.current = null;
      }
    };
  }, [cancelActiveTurn, stopAllAudioPlayback]);

  // Clean raw LLM text for natural speech synthesis
  const cleanTextForSpeech = (raw: string): string => {
    return raw
      .replace(/\[\^?\d+\]/g, "")
      .replace(/#{1,6}\s+/g, "")
      .replace(/(\*\*|\*|__|_)/g, "")
      .replace(/`{1,3}[^`]*`{1,3}/g, "")
      .replace(/^\s*[-*+]\s+/gm, "")
      .replace(/\n{2,}/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  };

  const formatTime = (totalSeconds: number) => {
    const m = Math.floor(totalSeconds / 60);
    const s = totalSeconds % 60;
    return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  };

  // -------------------------------------------------------------
  // Voice Lifecycle: Audio Completion & Error Handling (Turn ID Aware)
  // -------------------------------------------------------------
  const handleTTSFinished = useCallback(
    (finishedTurnId: number) => {
      // Discard stale or superseded turn callbacks
      if (finishedTurnId !== activeTurnIdRef.current) {
        console.log(`[VoiceAssistant] Ignoring TTS onended for stale turn ${finishedTurnId} (active: ${activeTurnIdRef.current})`);
        return;
      }

      isVoicePipelineActiveRef.current = false;
      isSTTProcessingRef.current = false;
      stopAllAudioPlayback(false);

      // Continuous ChatGPT-style mode: automatically listen for the next question!
      if (isContinuousModeRef.current) {
        setVoiceState("listening");
        voiceStateRef.current = "listening";
        setProcessingStage(null);

        // Reset turn speech buffer state for immediate next question
        isSpeechTurnActiveRef.current = false;
        speechDetectedRef.current = false;
        recordedBuffersRef.current = [];
        consecutiveSpeechFramesRef.current = 0;
        consecutiveBargeInFramesRef.current = 0;

        if (silenceTimerRef.current) {
          clearTimeout(silenceTimerRef.current);
          silenceTimerRef.current = null;
        }
        setIsSilenceCountdown(false);
      } else {
        setVoiceState("idle");
        voiceStateRef.current = "idle";
        setProcessingStage(null);
      }
    },
    [stopAllAudioPlayback]
  );

  const handleTTSError = useCallback(
    (errorTurnId: number) => {
      if (errorTurnId !== activeTurnIdRef.current) {
        return;
      }
      console.warn(`[VoiceAssistant] TTS audio encountered an error during playback for turn ${errorTurnId}.`);
      isVoicePipelineActiveRef.current = false;
      isSTTProcessingRef.current = false;
      stopAllAudioPlayback(false);

      if (isContinuousModeRef.current) {
        setVoiceState("listening");
        voiceStateRef.current = "listening";
        setProcessingStage(null);

        isSpeechTurnActiveRef.current = false;
        speechDetectedRef.current = false;
        recordedBuffersRef.current = [];
        consecutiveSpeechFramesRef.current = 0;
        consecutiveBargeInFramesRef.current = 0;

        if (silenceTimerRef.current) {
          clearTimeout(silenceTimerRef.current);
          silenceTimerRef.current = null;
        }
        setIsSilenceCountdown(false);
      } else {
        setVoiceState("idle");
        voiceStateRef.current = "idle";
        setProcessingStage(null);
      }
    },
    [stopAllAudioPlayback]
  );



  // --------------------------------------------------------------------------
  // HANDLER: executeVoicePipeline
  // --------------------------------------------------------------------------
  // • WHAT IT DOES: Executes the complete end-to-end voice query pipeline:
  //     1. Calls Deepgram STT (/api/voice-entry) to transcribe user speech.
  //     2. Streams RAG answer tokens from /api/ask using Server-Sent Events (SSE).
  //     3. Feeds incoming tokens into SentenceTokenizer to detect punctuation boundaries.
  //     4. Enqueues completed sentences into SentenceAudioQueue for parallel Deepgram TTS synthesis.
  //     5. Plays sentence #1 out loud within ~400ms while remaining sentences synthesize in background.
  //     6. Automatically re-arms listening mode when speaking finishes.
  // • INPUTS:
  //     - blob (Blob): 16-bit linear PCM WAV recording of the user's question.
  //     - turnId (number): Unique incremental turn counter guarding against stale responses.
  //     - abortController (AbortController): Cancellation controller for instant barge-in.
  // • OUTPUT: None (updates UI state, streams audio playback, and auto-listens).
  // • WHY IT IS USED: Eliminates conversational latency so that talking to sales playbooks
  //   feels as fast as talking to a live human colleague.
  // • WHERE IT FITS IN THE FLOW:
  //     [User Question finishes] -> [submitCurrentSpeechTurn] -> [executeVoicePipeline] -> [SentenceAudioQueue]
  // --------------------------------------------------------------------------
  // --------------------------------------------------------------------------
  // HANDLER: executeRagTurn
  // --------------------------------------------------------------------------
  // • WHAT IT DOES: Feeds the verbatim transcribed user question (from either
  //   live Deepgram WebSocket or HTTP fallback) directly into Pinecone RAG + DeepSeek.
  // • Zero transcript rewriting by LLM: the transcribed words are passed verbatim.
  // --------------------------------------------------------------------------
  const executeRagTurn = async (
    transcribedQuery: string,
    langHint: string | undefined,
    turnId: number,
    abortController: AbortController
  ) => {
    const isTurnStale = () => {
      return turnId !== activeTurnIdRef.current || abortController.signal.aborted;
    };

    if (isTurnStale()) return;

    if (langHint && ["en", "hi", "mr"].includes(langHint)) {
      setDetectedLanguage(langHint);
    }
    setVoiceState("processing");
    voiceStateRef.current = "processing";
    setLiveTranscript(transcribedQuery);
    setProcessingStage("retrieving");
    setIsLoading(true);

    try {
      // Step 3b: Pre-warm audio element and sentence audio queue BEFORE firing RAG network request
      let audio = ttsAudioRef.current;
      if (!audio && typeof window !== "undefined") {
        audio = new Audio();
        audio.preload = "auto";
        (audio as any).playsInline = true;
        audio.setAttribute("playsinline", "true");
        ttsAudioRef.current = audio;
      }

      if (audio) {
        if (!sentenceAudioQueueRef.current) {
          sentenceAudioQueueRef.current = new SentenceAudioQueue(audio, {}, audioContextRef.current, { module: "module2", speaker: "simran" });
        } else {
          sentenceAudioQueueRef.current.setOptions({ module: "module2", speaker: "simran" });
          if (audioContextRef.current) {
            sentenceAudioQueueRef.current.setAudioContext(audioContextRef.current);
          }
        }
        sentenceAudioQueueRef.current.updateCallbacks({
          onSentenceStart: () => {
            if (isTurnStale()) return;
            setVoiceState("speaking");
            voiceStateRef.current = "speaking";
            setProcessingStage(null);
            setIsLoading(false);
            sentenceStartTimestampRef.current = Date.now();
            const latency = speechEndTimestampRef.current > 0 ? Date.now() - speechEndTimestampRef.current : 0;
            if (latency > 0) {
              console.log(`[VoiceAssistant:Latency] Speech end -> First audio response: ${latency}ms (Target: <1000ms)`);
            }
            consecutiveBargeInFramesRef.current = 0;
          },
          onQueueComplete: () => {
            if (turnId === activeTurnIdRef.current) {
              handleTTSFinished(turnId);
            }
          },
          onError: (err) => {
            console.warn("[VoiceAssistant] Streaming TTS error:", err);
            if (turnId === activeTurnIdRef.current) {
              handleTTSError(turnId);
            }
          },
        });
        sentenceAudioQueueRef.current.startNewTurn(turnId);
      }

      if (isTurnStale()) return;

      const requestedLang = selectedLanguage !== "auto" ? selectedLanguage : (langHint || undefined);
      const ragRes = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: abortController.signal,
        body: JSON.stringify({
          question: transcribedQuery,
          top_k: topK,
          namespace: namespace.trim() || "sales_playbooks",
          language: requestedLang,
          stream: true,
        }),
      });

      if (isTurnStale()) return;

      if (!ragRes.ok) {
        const errData = await ragRes.json().catch(() => ({}));
        throw new Error(errData.error || errData.detail || "RAG pipeline failed to retrieve answer.");
      }

      const reader = ragRes.body?.getReader();
      if (!reader) {
        throw new Error("No readable stream received from RAG service.");
      }

      const decoder = new TextDecoder("utf-8");
      let sseBuffer = "";
      let accumulatedAnswer = "";
      let resolvedLang = langHint || (selectedLanguage !== "auto" ? selectedLanguage : "en");
      let messageCreated = false;
      let hasRevealedAnswer = false;
      const newMsgId = `msg-voice-${Date.now()}`;
      setActivePlayingMsgId(newMsgId);

      const tokenizer = new SentenceTokenizer();

      const handleSSEEvent = (event: string, dataStr: string) => {
        if (!dataStr) return;
        try {
          const parsed = JSON.parse(dataStr);
          if (event === "metadata") {
            if (parsed.language) {
              resolvedLang = parsed.language;
              setDetectedLanguage(parsed.language);
            }
            if (!messageCreated) {
              messageCreated = true;
              const newMessage: QAMessage = {
                id: newMsgId,
                question: transcribedQuery,
                answer: "",
                sources: parsed.sources || [],
                context_used: parsed.context_used || [],
                model: parsed.model || "deepseek/deepseek-chat",
                fallback_used: parsed.fallback_used ?? false,
                language: parsed.language || resolvedLang,
                is_greeting: parsed.is_greeting ?? false,
                timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
              };
              setMessages((prev) => [newMessage, ...prev]);
            }
          } else if (event === "token") {
            const token = parsed.token ?? parsed.delta ?? "";
            if (token) {
              accumulatedAnswer += token;

              // If metadata event was omitted, ensure message is created immediately
              if (!messageCreated) {
                messageCreated = true;
                const newMessage: QAMessage = {
                  id: newMsgId,
                  question: transcribedQuery,
                  answer: accumulatedAnswer,
                  sources: [],
                  context_used: [],
                  model: "deepseek/deepseek-chat",
                  fallback_used: false,
                  language: resolvedLang,
                  is_greeting: false,
                  timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
                };
                setMessages((prev) => [newMessage, ...prev]);
              }

              // Feed incoming token into SentenceTokenizer
              const completedSentences = tokenizer.feed(token);

              // REVEAL ANSWER AS SOON AS FIRST DEEPSEEK SENTENCE ARRIVES:
              // Immediately remove the loading shimmer card so the user sees the answer card
              if (!hasRevealedAnswer && (completedSentences.length > 0 || accumulatedAnswer.length > 0)) {
                hasRevealedAnswer = true;
                setIsLoading(false);
                setProcessingStage("synthesizing");
              }

              // STREAM TTS IMMEDIATELY SENTENCE-BY-SENTENCE:
              // As each sentence completes, dispatch to Deepgram TTS without delay
              for (const s of completedSentences) {
                const speechText = cleanTextForSpeech(s);
                if (speechText) {
                  sentenceAudioQueueRef.current?.enqueueSentence(speechText, resolvedLang);
                }
              }

              // Update answer text in message card in real-time
              setMessages((prev) =>
                prev.map((m) => (m.id === newMsgId ? { ...m, answer: accumulatedAnswer } : m))
              );
            }
          } else if (event === "done") {
            setIsLoading(false);
            if (parsed.answer && !accumulatedAnswer) {
              accumulatedAnswer = parsed.answer;
            }
            if (parsed.language) {
              resolvedLang = parsed.language;
            }
            setMessages((prev) =>
              prev.map((m) =>
                m.id === newMsgId
                  ? {
                    ...m,
                    answer: accumulatedAnswer || parsed.answer || "No answer generated.",
                    sources: parsed.sources || m.sources,
                    language: resolvedLang,
                  }
                  : m
              )
            );
          } else if (event === "error") {
            setIsLoading(false);
            console.error("[VoiceAssistant] SSE stream error:", parsed.error);
            setError(parsed.error || "Streaming error from model.");
          }
        } catch (parseErr) {
          console.warn("[VoiceAssistant] Failed to parse SSE data:", dataStr, parseErr);
        }
      };

      let currentEvent = "";
      let currentData = "";

      try {
        while (true) {
          if (isTurnStale()) {
            reader.cancel().catch(() => { });
            return;
          }

          const { done, value } = await reader.read();
          if (done) break;

          sseBuffer += decoder.decode(value, { stream: true });
          const lines = sseBuffer.split(/\r?\n/);
          sseBuffer = lines.pop() ?? "";

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) {
              if (currentEvent && currentData) {
                handleSSEEvent(currentEvent, currentData);
              }
              currentEvent = "";
              currentData = "";
              continue;
            }

            if (trimmed.startsWith("event:")) {
              currentEvent = trimmed.slice(6).trim();
            } else if (trimmed.startsWith("data:")) {
              currentData = trimmed.slice(5).trim();
            }
          }
        }

        if (currentEvent && currentData) {
          handleSSEEvent(currentEvent, currentData);
        }
      } catch (streamErr: any) {
        if (streamErr?.name === "AbortError" || isTurnStale()) {
          console.log(`[VoiceAssistant] Turn ${turnId} stream aborted.`);
          return;
        }
        throw streamErr;
      }

      if (isTurnStale()) return;

      if (!messageCreated) {
        const finalMessage: QAMessage = {
          id: newMsgId,
          question: transcribedQuery,
          answer: accumulatedAnswer || "No answer generated.",
          sources: [],
          context_used: [],
          model: "deepseek/deepseek-chat",
          fallback_used: false,
          language: resolvedLang,
          timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        };
        setMessages((prev) => [finalMessage, ...prev]);
      }

      const trailingSentence = tokenizer.flush();
      if (trailingSentence) {
        const speechText = cleanTextForSpeech(trailingSentence);
        if (speechText) {
          sentenceAudioQueueRef.current?.enqueueSentence(speechText, resolvedLang);
        }
      }

      sentenceAudioQueueRef.current?.markStreamComplete();
    } catch (err: any) {
      if (isTurnStale()) return;
      console.error("Voice pipeline error:", err);
      setError(err.message || "An error occurred during voice processing.");

      isVoicePipelineActiveRef.current = false;
      isSTTProcessingRef.current = false;

      if (isContinuousModeRef.current) {
        setVoiceState("listening");
        voiceStateRef.current = "listening";
        setProcessingStage(null);
        isSpeechTurnActiveRef.current = false;
        speechDetectedRef.current = false;
        recordedBuffersRef.current = [];
      } else {
        setVoiceState("idle");
        voiceStateRef.current = "idle";
        setProcessingStage(null);
      }
    } finally {
      setIsLoading(false);
    }
  };

  // --------------------------------------------------------------------------
  // HANDLER: executeVoicePipeline (HTTP Fallback Path)
  // --------------------------------------------------------------------------
  const executeVoicePipeline = async (blob: Blob, turnId: number, abortController: AbortController) => {
    const isTurnStale = () => {
      return turnId !== activeTurnIdRef.current || abortController.signal.aborted;
    };

    if (isTurnStale()) return;

    setVoiceState("processing");
    setProcessingStage("transcribing");
    setError(null);

    console.log(
      `[VoiceAssistant] HTTP STT fallback started for turn ${turnId}: ` +
      `blobSize=${blob.size} bytes, blobType='${blob.type}'`
    );

    try {
      const ext = blob.type.includes("mp4")
        ? "mp4"
        : blob.type.includes("ogg")
          ? "ogg"
          : blob.type.includes("wav")
            ? "wav"
            : "webm";
      const filename = `voice_query_${Date.now()}.${ext}`;

      const formData = new FormData();
      if (typeof File !== "undefined") {
        const audioFile = new File([blob], filename, { type: blob.type || "audio/webm" });
        formData.append("file", audioFile);
      } else {
        formData.append("file", blob, filename);
      }
      if (selectedLanguage !== "auto") {
        formData.append("language", selectedLanguage);
      }
      formData.append("module", "module2");

      let sttRes: Response | null = null;
      let lastSTTErrMsg: string | null = null;

      for (let attempt = 1; attempt <= 2; attempt++) {
        if (isTurnStale()) return;
        try {
          sttRes = await fetch("/api/voice-entry", {
            method: "POST",
            body: formData,
            signal: abortController.signal,
          });

          if (
            (sttRes.status === 503 ||
              sttRes.status === 408 ||
              sttRes.status === 502 ||
              sttRes.status === 504) &&
            attempt < 2
          ) {
            console.warn(`Deepgram STT transient HTTP ${sttRes.status} on attempt ${attempt}. Retrying in 600ms...`);
            await new Promise((resolve) => setTimeout(resolve, 600));
            continue;
          }
          break;
        } catch (fetchErr: any) {
          if (fetchErr?.name === "AbortError" || isTurnStale()) {
            console.log(`Deepgram STT fetch aborted cleanly for turn ${turnId}.`);
            return;
          }
          lastSTTErrMsg = fetchErr?.message || "Network error";
          if (attempt < 2) {
            await new Promise((resolve) => setTimeout(resolve, 600));
          }
        }
      }

      if (isTurnStale()) return;

      if (!sttRes) {
        throw new Error(lastSTTErrMsg || "Unable to reach Deepgram STT service.");
      }

      const sttData = await sttRes.json();
      if (!sttRes.ok) {
        throw new Error(sttData.error || sttData.detail || "Deepgram STT transcription failed.");
      }

      if (isTurnStale()) return;

      const transcribedQuery = (sttData.transcript || "").trim();
      if (!transcribedQuery) {
        console.log(`[VoiceAssistant] Empty transcript for turn ${turnId}. Re-arming listening.`);
        isVoicePipelineActiveRef.current = false;
        isSTTProcessingRef.current = false;

        if (isContinuousModeRef.current) {
          setError("No clear speech detected. Listening for your question...");
          setVoiceState("listening");
          voiceStateRef.current = "listening";
          setProcessingStage(null);
          isSpeechTurnActiveRef.current = false;
          speechDetectedRef.current = false;
          recordedBuffersRef.current = [];
        } else {
          setError("No speech detected in audio. Please speak clearly into your microphone.");
          setVoiceState("idle");
          voiceStateRef.current = "idle";
          setProcessingStage(null);
        }
        return;
      }

      await executeRagTurn(transcribedQuery, sttData.detected_language, turnId, abortController);
    } catch (err: any) {
      if (isTurnStale()) return;
      console.error("HTTP Voice pipeline error:", err);
      setError(err.message || "An error occurred during voice processing.");

      isVoicePipelineActiveRef.current = false;
      isSTTProcessingRef.current = false;

      if (isContinuousModeRef.current) {
        setVoiceState("listening");
        voiceStateRef.current = "listening";
        setProcessingStage(null);
        isSpeechTurnActiveRef.current = false;
        speechDetectedRef.current = false;
        recordedBuffersRef.current = [];
      } else {
        setVoiceState("idle");
        voiceStateRef.current = "idle";
        setProcessingStage(null);
      }
    }
  };

  // --------------------------------------------------------------------------
  // LIVE STREAMING STT: Stream PCM to Deepgram Nova-3 over WebSocket
  // --------------------------------------------------------------------------
  const sendStreamingChunk = useCallback((chunk: Float32Array) => {
    const ws = sttSocketRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(convertFloat32ToInt16(chunk));
      } catch (_) { }
    } else if (ws && ws.readyState === WebSocket.CONNECTING) {
      wsPendingQueueRef.current.push(chunk);
    }
  }, []);

  const startStreamingSTT = useCallback((initialBuffers: Float32Array[]) => {
    if (sttSocketRef.current) {
      try { sttSocketRef.current.close(); } catch (_) { }
      sttSocketRef.current = null;
    }
    if (wsFallbackTimerRef.current) {
      clearTimeout(wsFallbackTimerRef.current);
      wsFallbackTimerRef.current = null;
    }

    hasWsFinalizedRef.current = false;
    isWsStreamingRef.current = true;
    wsPendingQueueRef.current = [...initialBuffers];

    const baseWs = (
      process.env.NEXT_PUBLIC_FASTAPI_WS_URL ||
      process.env.NEXT_PUBLIC_FASTAPI_URL ||
      "http://127.0.0.1:8001"
    ).replace(/^http/, "ws");

    const sampleRate = audioContextRef.current?.sampleRate || 48000;
    const langParam = selectedLanguage !== "auto" ? `&language=${selectedLanguage}` : "";
    const wsUrl = `${baseWs}/ws/voice-stt?sample_rate=${sampleRate}${langParam}&module=module2`;

    try {
      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";
      sttSocketRef.current = ws;

      ws.onopen = () => {
        // Stream pre-roll and initial speech frames immediately
        while (wsPendingQueueRef.current.length > 0) {
          const buf = wsPendingQueueRef.current.shift();
          if (buf && ws.readyState === WebSocket.OPEN) {
            try {
              ws.send(convertFloat32ToInt16(buf));
            } catch (_) { }
          }
        }
      };

      ws.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          if (data.type === "interim") {
            const interimText = (data.transcript || "").trim();
            if (interimText) {
              setLiveTranscript(interimText);
            }
          } else if (data.type === "final") {
            const verbatimText = (data.transcript || "").trim();
            if (verbatimText && !hasWsFinalizedRef.current) {
              hasWsFinalizedRef.current = true;
              if (wsFallbackTimerRef.current) {
                clearTimeout(wsFallbackTimerRef.current);
                wsFallbackTimerRef.current = null;
              }
              if (silenceTimerRef.current) {
                clearTimeout(silenceTimerRef.current);
                silenceTimerRef.current = null;
              }
              setIsSilenceCountdown(false);
              try { ws.close(); } catch (_) { }
              sttSocketRef.current = null;
              isWsStreamingRef.current = false;

              // Immediately transition into turn execution with verbatim transcript
              cancelActiveTurn(true);
              const turnId = activeTurnIdRef.current;
              const turnAbortController = new AbortController();
              activeTurnAbortControllerRef.current = turnAbortController;
              isVoicePipelineActiveRef.current = true;
              isSTTProcessingRef.current = true;
              setVoiceState("processing");
              voiceStateRef.current = "processing";

              executeRagTurn(verbatimText, data.detected_language, turnId, turnAbortController);
            } else if (!verbatimText && !hasWsFinalizedRef.current) {
              hasWsFinalizedRef.current = true;
              if (wsFallbackTimerRef.current) {
                clearTimeout(wsFallbackTimerRef.current);
                wsFallbackTimerRef.current = null;
              }
              try { ws.close(); } catch (_) { }
              sttSocketRef.current = null;
              isWsStreamingRef.current = false;
              if (isContinuousModeRef.current) {
                setVoiceState("listening");
                voiceStateRef.current = "listening";
                setProcessingStage(null);
                isSpeechTurnActiveRef.current = false;
                speechDetectedRef.current = false;
                recordedBuffersRef.current = [];
              } else {
                setVoiceState("idle");
                voiceStateRef.current = "idle";
                setProcessingStage(null);
              }
            }
          }
        } catch (parseErr) {
          console.warn("[VoiceAssistant] STT WS message parse error:", parseErr);
        }
      };

      ws.onerror = (err) => {
        console.warn("[VoiceAssistant] STT WebSocket error:", err);
      };

      ws.onclose = () => {
        if (sttSocketRef.current === ws) {
          sttSocketRef.current = null;
          isWsStreamingRef.current = false;
        }
      };
    } catch (wsErr) {
      console.warn("[VoiceAssistant] Unable to open STT WebSocket:", wsErr);
      sttSocketRef.current = null;
      isWsStreamingRef.current = false;
    }
  }, [cancelActiveTurn, selectedLanguage]);

  // -------------------------------------------------------------
  // Instant Automatic Barge-In: Halts TTS in 0ms and activates STT immediately
  // -------------------------------------------------------------
  const executeBargeIn = useCallback(() => {
    console.log("[VoiceAssistant] Automatic barge-in triggered! Halting TTS immediately and activating STT for new instruction.");
    consecutiveBargeInFramesRef.current = 0;
    consecutiveSpeechFramesRef.current = 0;

    // 1. Cut off active turn and stop any playing audio immediately (0ms delay)
    stopAllAudioPlayback(true);
    cancelActiveTurn(true);

    // 2. Keep continuous voice mode active and switch immediately to listening
    isContinuousModeRef.current = true;
    setIsContinuousMode(true);
    setVoiceState("listening");
    voiceStateRef.current = "listening";
    setProcessingStage(null);
    setLiveTranscript("");

    // 3. Clear any silence timer from previous turn
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    setIsSilenceCountdown(false);

    // 4. Immediately activate STT pipeline WITH PRE-ROLL BUFFER so NO FIRST WORDS ARE MISSED!
    isSpeechTurnActiveRef.current = true;
    speechDetectedRef.current = true;
    const initialBuffers = [...preRollBuffersRef.current];
    recordedBuffersRef.current = [...initialBuffers];
    startStreamingSTT(initialBuffers);
  }, [cancelActiveTurn, startStreamingSTT, stopAllAudioPlayback]);

  // -------------------------------------------------------------
  // Voice Lifecycle: Submit Current Speech Turn (PCM -> WAV -> STT Pipeline)
  // -------------------------------------------------------------
  const submitCurrentSpeechTurn = useCallback(() => {
    if (isVoicePipelineActiveRef.current) return;

    speechEndTimestampRef.current = Date.now();
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    setIsSilenceCountdown(false);
    isSpeechTurnActiveRef.current = false;
    speechDetectedRef.current = false;
    wsPendingQueueRef.current = [];

    const buffersToEncode = [...recordedBuffersRef.current];
    recordedBuffersRef.current = [];

    if (buffersToEncode.length === 0) return;

    const sampleRate = audioContextRef.current?.sampleRate || 48000;
    const totalSamples = buffersToEncode.reduce((acc, b) => acc + b.length, 0);
    const duration = totalSamples / sampleRate;

    // Discard tiny bursts / accidental noise (< 180ms)
    if (duration < 0.18) {
      if (sttSocketRef.current) {
        try { sttSocketRef.current.close(); } catch (_) { }
        sttSocketRef.current = null;
      }
      isWsStreamingRef.current = false;
      console.log(`[VoiceAssistant] Audio too brief (${duration.toFixed(2)}s). Resuming listening.`);
      return;
    }

    // Check if WebSocket live STT stream is active: request Deepgram finalization
    const ws = sttSocketRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify({ type: "CloseStream" }));
      } catch (_) { }

      // Safety fallback: if WS doesn't emit final within 1200ms, seamlessly fallback to batch HTTP STT
      if (wsFallbackTimerRef.current) clearTimeout(wsFallbackTimerRef.current);
      wsFallbackTimerRef.current = setTimeout(() => {
        if (!hasWsFinalizedRef.current) {
          console.log("[VoiceAssistant] STT WS timed out (1200ms). Falling back to HTTP STT pipeline...");
          if (sttSocketRef.current) {
            try { sttSocketRef.current.close(); } catch (_) { }
            sttSocketRef.current = null;
          }
          isWsStreamingRef.current = false;
          const wavBlob = encodeWav(buffersToEncode, sampleRate);
          cancelActiveTurn(true);
          const turnId = activeTurnIdRef.current;
          const turnAbortController = new AbortController();
          activeTurnAbortControllerRef.current = turnAbortController;
          isVoicePipelineActiveRef.current = true;
          isSTTProcessingRef.current = true;
          executeVoicePipeline(wavBlob, turnId, turnAbortController);
        }
      }, 1200);
      return;
    }

    // Standard HTTP STT fallback if WebSocket was not connected
    const wavBlob = encodeWav(buffersToEncode, sampleRate);
    console.log(`[VoiceAssistant] Speech turn captured: ${duration.toFixed(2)}s, ${wavBlob.size} bytes. Starting HTTP STT pipeline...`);

    cancelActiveTurn(true);
    const turnId = activeTurnIdRef.current;
    const turnAbortController = new AbortController();
    activeTurnAbortControllerRef.current = turnAbortController;
    isVoicePipelineActiveRef.current = true;
    isSTTProcessingRef.current = true;

    executeVoicePipeline(wavBlob, turnId, turnAbortController);
  }, [cancelActiveTurn]);

  // -------------------------------------------------------------
  // Voice Lifecycle: Start Listening & Continuous Audio Monitoring
  // -------------------------------------------------------------
  const startListening = useCallback(async () => {
    try {
      setError(null);
      stopAllAudioPlayback(true);
      isVoicePipelineActiveRef.current = false;
      isSTTProcessingRef.current = false;

      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
        silenceTimerRef.current = null;
      }
      setIsSilenceCountdown(false);
      speechDetectedRef.current = false;
      isSpeechTurnActiveRef.current = false;
      recordedBuffersRef.current = [];
      preRollBuffersRef.current = [];
      consecutiveSpeechFramesRef.current = 0;
      consecutiveBargeInFramesRef.current = 0;
      wsPendingQueueRef.current = [];

      if (typeof window === "undefined" || !navigator?.mediaDevices?.getUserMedia) {
        setError("Microphone access is not supported in this browser environment.");
        setIsContinuousMode(false);
        setVoiceState("idle");
        return;
      }

      // Re-use or request microphone stream with hardware/browser AEC enabled
      let stream = audioStreamRef.current;
      const needsNewStream =
        !stream ||
        !stream.active ||
        stream.getAudioTracks().length === 0 ||
        stream.getAudioTracks().some((t) => t.readyState === "ended");

      if (needsNewStream) {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: { ideal: true },
            noiseSuppression: { ideal: true },
            autoGainControl: { ideal: true },
          },
        });
        audioStreamRef.current = stream;
      }

      if (!stream) {
        throw new Error("Unable to obtain microphone stream.");
      }

      stream.getAudioTracks().forEach((t) => (t.enabled = true));

      // Disconnect and reset any previous audio processing nodes to prevent stale node reuse
      if (audioSourceNodeRef.current) {
        try {
          audioSourceNodeRef.current.disconnect();
        } catch (_) { }
        audioSourceNodeRef.current = null;
      }
      if (analyserRef.current) {
        try {
          analyserRef.current.disconnect();
        } catch (_) { }
        analyserRef.current = null;
      }
      if (scriptProcessorRef.current) {
        try {
          scriptProcessorRef.current.disconnect();
        } catch (_) { }
        scriptProcessorRef.current = null;
      }
      if (silentGainRef.current) {
        try {
          silentGainRef.current.disconnect();
        } catch (_) { }
        silentGainRef.current = null;
      }

      // Close and reset previous AudioContext before creating a new one
      if (audioContextRef.current && audioContextRef.current.state !== "closed") {
        try {
          await audioContextRef.current.close();
        } catch (_) { }
        audioContextRef.current = null;
      }

      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      const audioCtx = new AudioContextClass();
      audioContextRef.current = audioCtx;
      if (audioCtx.state === "suspended") {
        await audioCtx.resume();
      }

      // Create all nodes from the SAME fresh AudioContext instance
      const source = audioCtx.createMediaStreamSource(stream);
      audioSourceNodeRef.current = source;

      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 128;
      analyserRef.current = analyser;

      source.connect(analyser);

      // ScriptProcessorNode for real-time PCM audio capture, pre-roll buffering, and VAD
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      scriptProcessorRef.current = processor;

      // Silent gain node keeps processor clock running without looping mic audio back to speakers
      const silentGain = audioCtx.createGain();
      silentGain.gain.value = 0;
      silentGainRef.current = silentGain;

      source.connect(processor);
      processor.connect(silentGain);
      silentGain.connect(audioCtx.destination);

      const maxPreRollCount = Math.max(6, Math.ceil((audioCtx.sampleRate * 0.85) / 4096));

      processor.onaudioprocess = (e: AudioProcessingEvent) => {
        if (!isContinuousModeRef.current && voiceStateRef.current === "idle") {
          return;
        }

        const inputChannel = e.inputBuffer.getChannelData(0);
        const chunk = new Float32Array(inputChannel.length);
        chunk.set(inputChannel);

        // 1. Calculate RMS energy
        let sumSq = 0;
        for (let i = 0; i < chunk.length; i++) {
          sumSq += chunk[i] * chunk[i];
        }
        const rms = Math.sqrt(sumSq / chunk.length);

        // 2. Frequency analysis via AnalyserNode
        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(dataArray);

        let sumFreq = 0;
        for (let i = 0; i < dataArray.length; i++) {
          sumFreq += dataArray[i];
        }
        const avgFreq = sumFreq / dataArray.length;

        // Vocal formant frequency bins (bins 1 to 6 in a 64-bin FFT cover ~375Hz to ~2600Hz)
        const vocalBins = [1, 2, 3, 4, 5, 6];
        let vocalSum = 0;
        for (const idx of vocalBins) {
          vocalSum += dataArray[idx] || 0;
        }
        const vocalFreq = vocalSum / vocalBins.length;

        // Visualizer waveform (24 bars)
        const sampleCount = 24;
        const step = Math.floor(dataArray.length / sampleCount) || 1;
        const levels = [];
        for (let i = 0; i < sampleCount; i++) {
          levels.push(Math.min(100, Math.max(14, Math.round(((dataArray[i * step] || 0) / 255) * 100))));
        }
        setAudioLevels(levels);

        // 3. Update rolling pre-roll buffer (keeps last ~850ms at all times)
        preRollBuffersRef.current.push(chunk);
        if (preRollBuffersRef.current.length > maxPreRollCount) {
          preRollBuffersRef.current.shift();
        }

        // -------------------------------------------------------------
        // AUTOMATIC BARGE-IN: Continuously detect user speech while assistant is talking or generating
        // -------------------------------------------------------------
        const isAudioPlaying =
          Boolean(sentenceAudioQueueRef.current?.isCurrentlyPlaying()) ||
          Boolean(ttsAudioRef.current && !ttsAudioRef.current.paused && ttsAudioRef.current.currentTime > 0);

        const isAssistantTalking = voiceStateRef.current === "speaking" || isAudioPlaying;
        const isAssistantProcessing = voiceStateRef.current === "processing";

        if (isAssistantTalking || isAssistantProcessing) {
          if (isAssistantTalking) {
            // Adapt dynamic speaker bleed baseline while audio is actively playing from speakers
            speakerBleedBaselineRef.current = speakerBleedBaselineRef.current * 0.90 + avgFreq * 0.10;
            bleedRmsRef.current = bleedRmsRef.current * 0.90 + rms * 0.10;

            const now = Date.now();
            // Fast 200ms grace period to let initial audio start transient settle
            const hasGracePeriodElapsed =
              sentenceStartTimestampRef.current > 0
                ? now - sentenceStartTimestampRef.current > 200
                : true;

            // Detect genuine near-field human user speech distinctly above speaker bleed
            const isUserBargeIn =
              hasGracePeriodElapsed &&
              (
                (rms >= 0.028 && vocalFreq >= 20 && (rms >= bleedRmsRef.current * 1.5 || avgFreq >= speakerBleedBaselineRef.current + 6)) ||
                (rms >= 0.040 && vocalFreq >= 18) ||
                (avgFreq >= 26 && vocalFreq >= 24 && avgFreq >= speakerBleedBaselineRef.current + 8)
              );

            if (isUserBargeIn) {
              consecutiveBargeInFramesRef.current++;
              // Require 2 consecutive frames (~170ms) of sustained speech to confirm instant barge-in
              if (consecutiveBargeInFramesRef.current >= 2) {
                console.log(`[VoiceAssistant] User barge-in confirmed (rms=${rms.toFixed(4)}, bleed=${bleedRmsRef.current.toFixed(4)}). Halting TTS in 0ms and activating STT...`);
                executeBargeIn();
                return;
              }
            } else {
              consecutiveBargeInFramesRef.current = Math.max(0, consecutiveBargeInFramesRef.current - 1);
            }
          } else if (isAssistantProcessing) {
            // Assistant is retrieving/generating; no speaker audio is playing yet
            const isUserSpeech = rms >= 0.020 || (avgFreq >= 20 && vocalFreq >= 16);
            if (isUserSpeech) {
              consecutiveBargeInFramesRef.current++;
              if (consecutiveBargeInFramesRef.current >= 2) {
                console.log(`[VoiceAssistant] User barge-in during response generation (rms=${rms.toFixed(4)}). Halting in 0ms and activating STT...`);
                executeBargeIn();
                return;
              }
            } else {
              consecutiveBargeInFramesRef.current = Math.max(0, consecutiveBargeInFramesRef.current - 1);
            }
          }
          return;
        }

        // -------------------------------------------------------------
        // LISTENING STATE: Detect speech onset, buffer audio, detect natural pause
        // -------------------------------------------------------------
        if (voiceStateRef.current === "listening") {
          const isSpeech = rms >= 0.016 || (avgFreq >= 18 && vocalFreq >= 15);

          if (isSpeech) {
            consecutiveSpeechFramesRef.current++;
            if (!isSpeechTurnActiveRef.current && consecutiveSpeechFramesRef.current >= 1) {
              isSpeechTurnActiveRef.current = true;
              speechDetectedRef.current = true;
              // Prepend pre-roll buffer so the first words/syllables are NEVER missed!
              recordedBuffersRef.current = [...preRollBuffersRef.current, chunk];
              startStreamingSTT([...preRollBuffersRef.current, chunk]);
            } else if (isSpeechTurnActiveRef.current) {
              recordedBuffersRef.current.push(chunk);
              sendStreamingChunk(chunk);
            }

            if (silenceTimerRef.current) {
              clearTimeout(silenceTimerRef.current);
              silenceTimerRef.current = null;
              setIsSilenceCountdown(false);
            }
          } else {
            consecutiveSpeechFramesRef.current = 0;
            if (isSpeechTurnActiveRef.current) {
              // Append trailing chunk to avoid abrupt cut-off
              recordedBuffersRef.current.push(chunk);
              sendStreamingChunk(chunk);
              if (!silenceTimerRef.current) {
                setIsSilenceCountdown(true);
                silenceTimerRef.current = setTimeout(() => {
                  submitCurrentSpeechTurn();
                }, 250); // Fast conversational silence detection (250ms) for instant reply
              }
            }
          }
        }
      };

      setVoiceState("listening");
      voiceStateRef.current = "listening";
      setListeningSeconds(0);

      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = setInterval(() => {
        setListeningSeconds((prev) => prev + 1);
      }, 1000);
    } catch (err: any) {
      console.error("Microphone access error:", err);
      if (err?.name === "NotAllowedError") {
        setError("Microphone permission was denied. Please allow microphone access in your browser.");
      } else if (err?.name === "NotFoundError") {
        setError("No microphone input device was found on this computer.");
      } else {
        setError(err?.message || "Failed to access microphone.");
      }
      setIsContinuousMode(false);
      setVoiceState("idle");
    }
  }, [cancelActiveTurn, executeBargeIn, sendStreamingChunk, startStreamingSTT, stopAllAudioPlayback, submitCurrentSpeechTurn]);

  const cancelListening = useCallback(() => {
    cancelActiveTurn(true);

    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    setIsSilenceCountdown(false);

    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }
    if (analyserRef.current) {
      try {
        analyserRef.current.disconnect();
      } catch (_) { }
      analyserRef.current = null;
    }
    if (scriptProcessorRef.current) {
      try {
        scriptProcessorRef.current.disconnect();
      } catch (_) { }
      scriptProcessorRef.current = null;
    }
    if (silentGainRef.current) {
      try {
        silentGainRef.current.disconnect();
      } catch (_) { }
      silentGainRef.current = null;
    }
    if (audioSourceNodeRef.current) {
      try {
        audioSourceNodeRef.current.disconnect();
      } catch (_) { }
      audioSourceNodeRef.current = null;
    }
    if (audioStreamRef.current) {
      audioStreamRef.current.getTracks().forEach((track) => track.stop());
      audioStreamRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      audioContextRef.current.close().catch(() => { });
      audioContextRef.current = null;
    }
    recordedBuffersRef.current = [];
    preRollBuffersRef.current = [];
    isSpeechTurnActiveRef.current = false;
    speechDetectedRef.current = false;
    setVoiceState("idle");
    voiceStateRef.current = "idle";
    setProcessingStage(null);
    setAudioLevels(new Array(24).fill(14));
  }, [cancelActiveTurn]);

  // -------------------------------------------------------------
  // Stop / Exit Continuous Voice Mode
  // -------------------------------------------------------------
  const exitContinuousVoiceMode = useCallback(() => {
    isContinuousModeRef.current = false;
    setIsContinuousMode(false);
    setIsSilenceCountdown(false);

    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }

    cancelActiveTurn(true);
    stopAllAudioPlayback(true);
    cancelListening();
    setVoiceState("idle");
    voiceStateRef.current = "idle";
    setProcessingStage(null);
  }, [cancelActiveTurn, cancelListening, stopAllAudioPlayback]);

  // -------------------------------------------------------------
  // Blue Circular Mic Button Click Handler
  // -------------------------------------------------------------
  const handleVoiceButtonClick = () => {
    unlockAudioPlayback();

    if (voiceState === "idle") {
      isContinuousModeRef.current = true;
      setIsContinuousMode(true);
      startListening();
    } else {
      exitContinuousVoiceMode();
    }
  };

  // -------------------------------------------------------------
  // Re-play / Listen to any message answer via Deepgram TTS (Sentence-Level Streaming)
  // -------------------------------------------------------------
  const playTTSForMessage = async (msgId: string, text: string, lang?: string) => {
    unlockAudioPlayback();

    if (activePlayingMsgId === msgId && voiceState === "speaking") {
      // Interrupt playback
      cancelActiveTurn(true);
      stopAllAudioPlayback(true);
      setActivePlayingMsgId(null);
      setVoiceState("idle");
      return;
    }

    // Cancel any active voice pipeline turn and assign sequential request ID
    cancelActiveTurn(true);
    activeTTSRequestIdRef.current += 1;
    const currentRequestId = activeTTSRequestIdRef.current;

    // Immediately stop all playing audio and cancel pending requests
    stopAllAudioPlayback(true);

    const abortController = new AbortController();
    activeTTSAbortControllerRef.current = abortController;

    setVoiceState("speaking");
    voiceStateRef.current = "speaking";
    setProcessingStage("synthesizing");
    setActivePlayingMsgId(msgId);

    try {
      let audio = ttsAudioRef.current;
      if (!audio && typeof window !== "undefined") {
        audio = new Audio();
        audio.preload = "auto";
        (audio as any).playsInline = true;
        audio.setAttribute("playsinline", "true");
        ttsAudioRef.current = audio;
      }

      if (audio) {
        if (!sentenceAudioQueueRef.current) {
          sentenceAudioQueueRef.current = new SentenceAudioQueue(audio, {}, audioContextRef.current, { module: "module2", speaker: "simran" });
        } else {
          sentenceAudioQueueRef.current.setOptions({ module: "module2", speaker: "simran" });
          if (audioContextRef.current) {
            sentenceAudioQueueRef.current.setAudioContext(audioContextRef.current);
          }
        }
        sentenceAudioQueueRef.current.updateCallbacks({
          onSentenceStart: () => {
            if (currentRequestId !== activeTTSRequestIdRef.current) return;
            setVoiceState("speaking");
            voiceStateRef.current = "speaking";
            setProcessingStage("synthesizing");
            sentenceStartTimestampRef.current = Date.now();
            consecutiveBargeInFramesRef.current = 0;
          },
          onQueueComplete: () => {
            if (currentRequestId === activeTTSRequestIdRef.current) {
              stopAllAudioPlayback(false);
              setActivePlayingMsgId(null);
              setVoiceState("idle");
              voiceStateRef.current = "idle";
              setProcessingStage(null);
            }
          },
          onError: (err) => {
            console.warn("Card TTS playback error:", err);
            if (currentRequestId === activeTTSRequestIdRef.current) {
              stopAllAudioPlayback(false);
              setActivePlayingMsgId(null);
              setVoiceState("idle");
              voiceStateRef.current = "idle";
              setProcessingStage(null);
            }
          },
        });

        sentenceAudioQueueRef.current.startNewTurn(currentRequestId);

        const targetLang = lang || (selectedLanguage !== "auto" ? selectedLanguage : "en");
        const cleanText = cleanTextForSpeech(text);
        const tokenizer = new SentenceTokenizer();
        const sentences = tokenizer.feed(cleanText);
        const trailing = tokenizer.flush();
        if (trailing) sentences.push(trailing);

        if (sentences.length === 0 && cleanText) {
          sentences.push(cleanText);
        }

        for (const s of sentences) {
          sentenceAudioQueueRef.current.enqueueSentence(s, targetLang);
        }

        sentenceAudioQueueRef.current.markStreamComplete();
      }
    } catch (err: any) {
      if (err?.name === "AbortError") {
        console.log("Card TTS fetch aborted cleanly.");
        return;
      }
      console.error("Message TTS error:", err);
      if (currentRequestId === activeTTSRequestIdRef.current) {
        setError(err.message || "Failed to synthesize speech.");
        setVoiceState("idle");
        voiceStateRef.current = "idle";
        setActivePlayingMsgId(null);
      }
    } finally {
      if (currentRequestId === activeTTSRequestIdRef.current) {
        setProcessingStage(null);
      }
    }
  };

  // -------------------------------------------------------------
  // Text-based RAG Submission (Streams answer & plays TTS automatically)
  // -------------------------------------------------------------
  const handleAsk = async (queryText?: string) => {
    const queryToSubmit = (queryText || question).trim();
    if (!queryToSubmit || isLoading) return;

    unlockAudioPlayback();
    cancelActiveTurn(true);
    stopAllAudioPlayback(true);
    setIsLoading(true);
    setError(null);

    const turnId = activeTurnIdRef.current;
    const turnAbortController = new AbortController();
    activeTurnAbortControllerRef.current = turnAbortController;
    isVoicePipelineActiveRef.current = true;

    if (!queryText) {
      setQuestion("");
    }

    try {
      await executeRagTurn(queryToSubmit, undefined, turnId, turnAbortController);
    } catch (err: any) {
      if (err?.name !== "AbortError") {
        console.error("Ask error:", err);
        setError(err.message || "Failed to retrieve grounded answer.");
      }
    } finally {
      setIsLoading(false);
      isVoicePipelineActiveRef.current = false;
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  };

  const toggleContext = (msgId: string) => {
    setExpandedContexts((prev) => ({
      ...prev,
      [msgId]: !prev[msgId],
    }));
  };

  const handleCopy = (text: string, msgId: string) => {
    navigator.clipboard?.writeText(text);
    setCopiedId(msgId);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className="space-y-6 w-full">
      {/* Single persistent audio element for Deepgram TTS playback */}
      <audio
        ref={ttsAudioRef}
        preload="auto"
        playsInline
        style={{ display: "none" }}
      />

      {/* Header Banner — CallNow Hero Style */}
      <div className="hero">
        <div className="flex flex-col md:flex-row md:items-start justify-between gap-4">
          <div className="flex-1">
            <div className="flex items-center gap-2.5 mb-2">
              <span className="hero-badge">
                <Sparkles className="h-3 w-3" />
                RAG Grounded Intelligence
              </span>
            </div>
            <h1 className="hero-title">
              Ask Assistant
            </h1>
            <p className="hero-subtitle mt-1 text-xs sm:text-sm text-[var(--ink-soft)] leading-relaxed max-w-3xl">
              Ask questions about your uploaded sales playbooks, loan policies, and knowledge documents. Get fast, source-grounded answers in English, Hindi, or Marathi.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2 self-start shrink-0">
            {/* Multilingual Selector & Auto-Detection Status */}
            <div className="flex items-center gap-1 p-1 rounded-[var(--radius)] bg-[var(--surface-2)] border border-[var(--border)] text-xs">
              <span className="text-[10px] text-[var(--ink-muted)] px-1 font-semibold">
                🌐
              </span>
              <button
                type="button"
                onClick={() => setSelectedLanguage("auto")}
                className={`px-2 py-0.5 rounded text-[10px] font-semibold transition-all cursor-pointer ${selectedLanguage === "auto"
                    ? "bg-[var(--accent)] text-white shadow-sm"
                    : "text-[var(--ink-muted)] hover:text-white"
                  }`}
                title={`Auto Language Detection (Active: ${detectedLanguage.toUpperCase()})`}
              >
                Auto ({detectedLanguage.toUpperCase()})
              </button>
              <button
                type="button"
                onClick={() => setSelectedLanguage("en")}
                className={`px-2 py-0.5 rounded text-[10px] font-medium transition-all cursor-pointer ${selectedLanguage === "en"
                    ? "bg-[var(--accent)] text-white shadow-sm"
                    : "text-[var(--ink-muted)] hover:text-white"
                  }`}
              >
                EN
              </button>
              <button
                type="button"
                onClick={() => setSelectedLanguage("hi")}
                className={`px-2 py-0.5 rounded text-[10px] font-medium transition-all cursor-pointer ${selectedLanguage === "hi"
                    ? "bg-[var(--accent)] text-white shadow-sm"
                    : "text-[var(--ink-muted)] hover:text-white"
                  }`}
              >
                हिंदी
              </button>
              <button
                type="button"
                onClick={() => setSelectedLanguage("mr")}
                className={`px-2 py-0.5 rounded text-[10px] font-medium transition-all cursor-pointer ${selectedLanguage === "mr"
                    ? "bg-[var(--accent)] text-white shadow-sm"
                    : "text-[var(--ink-muted)] hover:text-white"
                  }`}
              >
                मराठी
              </button>
            </div>
          </div>
        </div>
      </div>



      {/* CONTINUOUS GEMINI-STYLE VOICE HUD (Active while in continuous mode or when voiceState !== idle) */}
      {(isContinuousMode || voiceState !== "idle") && (
        <div className="card overflow-hidden border-2 border-blue-500/40 bg-gradient-to-r from-blue-950/50 via-indigo-950/30 to-[var(--surface)] shadow-2xl shadow-blue-500/15">
          <div className="p-4 sm:p-5">
            {/* HUD Top Bar with Session Status and Exit Button */}
            <div className="flex items-center justify-between gap-3 mb-3 pb-3 border-b border-blue-500/20">
              <div className="flex items-center gap-2">
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500" />
                </span>
                <span className="text-xs font-bold text-white tracking-wide flex items-center gap-1.5">
                  Gemini Live Voice Mode
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                    Ongoing Conversation
                  </span>
                </span>
              </div>

              <div className="flex items-center gap-2">
                {isSilenceCountdown && (
                  <span className="status-pill status-pill-warning text-[10px] animate-pulse">
                    Silence detected · Submitting in 0.9s...
                  </span>
                )}

                {/* The ONLY button that terminates Voice Mode */}
                <button
                  onClick={exitContinuousVoiceMode}
                  className="btn btn-danger btn-sm text-xs gap-1.5 cursor-pointer shadow-md shadow-red-500/20"
                  title="Exit Continuous Voice Mode"
                >
                  <PhoneOff className="h-3.5 w-3.5" />
                  <span>Exit Voice Mode</span>
                </button>
              </div>
            </div>

            {/* LARGE ANIMATED BLUE/PURPLE GLOWING ORB (NO MIC ICON) */}
            <div className="py-6 sm:py-8 flex flex-col items-center justify-center relative select-none">
              {/* Outer Radiant Glow Rings & Ambient Halos */}
              <div className="relative flex items-center justify-center my-2">
                {/* Expanding Pulsing Glow Ring 1 */}
                <div className="absolute w-48 h-48 sm:w-60 sm:h-60 md:w-72 md:h-72 rounded-full bg-gradient-to-tr from-blue-600/30 via-indigo-600/25 to-purple-600/30 blur-2xl animate-orb-ring-1 pointer-events-none" />

                {/* Expanding Pulsing Glow Ring 2 */}
                <div className="absolute w-56 h-56 sm:w-68 sm:h-68 md:w-80 md:h-80 rounded-full bg-gradient-to-tr from-purple-600/20 via-blue-500/20 to-cyan-400/20 blur-3xl animate-orb-ring-2 pointer-events-none" />

                {/* THE LARGE ANIMATED GLOWING ORB */}
                <div
                  onClick={handleVoiceButtonClick}
                  className={`relative w-40 h-40 sm:w-48 sm:h-48 md:w-56 md:h-56 rounded-full flex items-center justify-center cursor-pointer transition-all duration-300 animate-orb-breathe ${voiceState === "speaking"
                      ? "ring-4 ring-cyan-400/50 shadow-[0_0_90px_rgba(56,189,248,0.7)]"
                      : voiceState === "processing"
                        ? "ring-4 ring-indigo-500/50 shadow-[0_0_80px_rgba(99,102,241,0.7)]"
                        : "ring-2 ring-blue-400/40 shadow-[0_0_70px_rgba(99,102,241,0.6)]"
                    }`}
                  style={{
                    transform:
                      voiceState === "listening"
                        ? `scale(${1 + Math.max(0, (Math.max(...audioLevels) - 14) / 240)})`
                        : undefined,
                  }}
                  title={
                    voiceState === "speaking"
                      ? "Assistant is speaking — Speak anytime to interrupt"
                      : voiceState === "listening"
                        ? "Listening to your voice..."
                        : "Processing question..."
                  }
                >
                  {/* Rich Gradient Base Sphere with Inner Shadows */}
                  <div className="absolute inset-0 rounded-full bg-gradient-to-tr from-blue-700 via-indigo-600 to-purple-600 shadow-[0_0_70px_rgba(99,102,241,0.8),inset_0_0_45px_rgba(168,85,247,0.6)] overflow-hidden">
                    {/* Swirling Dynamic Mesh Layer */}
                    <div className="absolute -inset-2 rounded-full bg-gradient-to-br from-cyan-400 via-blue-500 to-purple-700 opacity-80 blur-[2px] animate-orb-spin-slow mix-blend-screen" />

                    {/* Fluid Radial Depth Wave */}
                    <div className="absolute inset-0 rounded-full bg-radial from-transparent via-purple-500/35 to-blue-950/70 mix-blend-overlay" />

                    {/* Specular Curved Highlights */}
                    <div className="absolute top-3 left-7 w-24 h-12 rounded-full bg-gradient-to-b from-white/40 to-transparent blur-[3px] -rotate-35" />
                    <div className="absolute bottom-4 right-7 w-20 h-10 rounded-full bg-gradient-to-t from-cyan-300/35 to-transparent blur-[2px] rotate-20" />
                  </div>

                  {/* Glowing Core (NO MIC ICON) */}
                  <div className="relative z-10 w-20 h-20 sm:w-24 sm:h-24 rounded-full bg-gradient-to-tr from-blue-100/75 via-indigo-100/60 to-purple-100/75 blur-sm flex items-center justify-center">
                    <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-full bg-white/85 blur-[2px]" />
                  </div>
                </div>
              </div>

              {/* Status Header below the Orb */}
              <div className="mt-3 text-center space-y-1">
                <div className="flex items-center justify-center gap-2">
                  <span className="font-semibold text-sm sm:text-base text-white">
                    {voiceState === "listening" && "Listening to your voice..."}
                    {voiceState === "processing" && "Thinking & generating answer..."}
                    {voiceState === "speaking" && "Assistant is speaking..."}
                  </span>
                  {voiceState === "listening" && (
                    <span className="font-mono text-xs text-[var(--ink-muted)] bg-[var(--surface-2)] px-2 py-0.5 rounded border border-[var(--border)]">
                      {formatTime(listeningSeconds)}
                    </span>
                  )}
                </div>
                <p className="text-xs text-[var(--ink-soft)] max-w-md mx-auto">
                  {voiceState === "listening" && "Speak naturally · Pausing briefly submits your question automatically"}
                  {voiceState === "processing" && "Transcribing with Deepgram STT and retrieving grounded knowledge"}
                  {voiceState === "speaking" && "Continuous mode active · Speak at any time to automatically interrupt"}
                </p>
              </div>
            </div>

            {/* 1. LISTENING STATE CONTROLS & BARS */}
            {voiceState === "listening" && (
              <div className="space-y-3 pt-2 border-t border-blue-500/20">
                {/* Real-time Frequency Waveform Bars */}
                <div className="h-10 flex items-end justify-center gap-1.5 sm:gap-2 px-2 py-1.5 bg-[var(--surface-2)]/80 rounded-[var(--radius)] border border-blue-500/20">
                  {audioLevels.map((lvl, idx) => (
                    <div
                      key={idx}
                      className="w-1.5 sm:w-2 bg-gradient-to-t from-blue-500 via-indigo-400 to-cyan-300 rounded-full transition-all duration-75"
                      style={{
                        height: `${Math.max(6, Math.round((lvl / 100) * 34))}px`,
                      }}
                    />
                  ))}
                </div>

                {liveTranscript && (
                  <div className="p-3 rounded-[var(--radius)] bg-[var(--surface-2)] border border-blue-500/30 text-xs text-[var(--ink)]">
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-blue-400 block mb-0.5">
                      Hearing you live (Deepgram Nova-3):
                    </span>
                    &ldquo;{liveTranscript}&rdquo;
                  </div>
                )}
              </div>
            )}

            {/* 2. PROCESSING STATE */}
            {voiceState === "processing" && (
              <div className="space-y-3 pt-2 border-t border-blue-500/20">
                {/* Multi-step progress pipeline badge */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs">
                  <div
                    className={`p-2.5 rounded-[var(--radius)] border flex items-center gap-2 ${processingStage === "transcribing"
                        ? "bg-blue-600/20 border-blue-500 text-blue-300 font-semibold"
                        : "bg-[var(--surface-2)] border-[var(--border)] text-[var(--ink-muted)]"
                      }`}
                  >
                    {processingStage === "transcribing" ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-400" />
                    ) : (
                      <Check className="h-3.5 w-3.5 text-[var(--success)]" />
                    )}
                    <span>1. Deepgram STT</span>
                  </div>

                  <div
                    className={`p-2.5 rounded-[var(--radius)] border flex items-center gap-2 ${processingStage === "retrieving"
                        ? "bg-indigo-600/20 border-indigo-500 text-indigo-300 font-semibold"
                        : "bg-[var(--surface-2)] border-[var(--border)] text-[var(--ink-muted)]"
                      }`}
                  >
                    {processingStage === "retrieving" ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin text-indigo-400" />
                    ) : processingStage === "synthesizing" ? (
                      <Check className="h-3.5 w-3.5 text-[var(--success)]" />
                    ) : (
                      <Sparkles className="h-3.5 w-3.5 text-[var(--ink-muted)]" />
                    )}
                    <span>2. Pinecone + RAG</span>
                  </div>

                  <div
                    className={`p-2.5 rounded-[var(--radius)] border flex items-center gap-2 ${processingStage === "synthesizing"
                        ? "bg-cyan-600/20 border-cyan-500 text-cyan-300 font-semibold"
                        : "bg-[var(--surface-2)] border-[var(--border)] text-[var(--ink-muted)]"
                      }`}
                  >
                    {processingStage === "synthesizing" ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin text-cyan-400" />
                    ) : (
                      <Volume2 className="h-3.5 w-3.5 text-[var(--ink-muted)]" />
                    )}
                    <span>3. Sarvam Bulbul v3 TTS</span>
                  </div>
                </div>

                {liveTranscript && (
                  <div className="p-3 rounded-[var(--radius)] bg-[var(--surface-2)] border border-blue-500/20 text-xs text-[var(--ink)]">
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-blue-400 block mb-0.5">
                      Transcribed Question:
                    </span>
                    &ldquo;{liveTranscript}&rdquo;
                  </div>
                )}
              </div>
            )}

            {/* 3. SPEAKING STATE & AUTOMATIC BARGE-IN */}
            {voiceState === "speaking" && (
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pt-2 border-t border-blue-500/20">
                <div className="flex items-center gap-3">
                  <div className="h-8 w-8 rounded-full bg-blue-600/20 border border-blue-500/40 text-blue-400 flex items-center justify-center shrink-0">
                    <Volume2 className="h-4 w-4 animate-pulse text-cyan-400" />
                  </div>
                  <div>
                    <span className="text-xs font-bold text-white flex items-center gap-1.5">
                      Sarvam Bulbul v3 Voice Response
                      <span className="status-pill status-pill-info font-mono text-[10px]">
                        simran ({detectedLanguage === "mr" ? "mr-IN" : detectedLanguage === "hi" ? "hi-IN" : "en-IN"})
                      </span>
                    </span>
                    <p className="text-[11px] text-[var(--ink-soft)] mt-0.5">
                      Automatically listening again after answer.
                    </p>
                  </div>
                </div>

                {/* Soundwave Bars & Auto-Barge-in Indicator (NO manual stop/interrupt button) */}
                <div className="flex items-center gap-2.5 self-end sm:self-auto">
                  <div className="flex items-center gap-1 h-6 px-3 bg-[var(--surface-2)] rounded-full border border-blue-500/30">
                    <div className="w-1 bg-cyan-400 rounded-full animate-wave-1" />
                    <div className="w-1 bg-blue-400 rounded-full animate-wave-2" />
                    <div className="w-1 bg-indigo-400 rounded-full animate-wave-3" />
                    <div className="w-1 bg-cyan-400 rounded-full animate-wave-4" />
                    <div className="w-1 bg-blue-400 rounded-full animate-wave-5" />
                    <div className="w-1 bg-indigo-400 rounded-full animate-wave-6" />
                  </div>
                  <span className="text-[11px] text-cyan-300 font-medium hidden sm:inline">
                    Speak anytime to interrupt
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Input Form Box — CallNow Card Style with ChatGPT-Style Blue Voice Button */}
      <div className="card">
        <div className="card-body space-y-3">
          <div className="relative">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={3}
              placeholder="Ask a sales question... (e.g. 'What is our refund policy on annual plans?' or 'How do we answer competitor uptime claims?')"
              className="form-control w-full pr-40 resize-none"
            />

            {/* Action Bar: Blue Circular Voice Button + Ask Assistant Button */}
            <div className="absolute right-2.5 bottom-2.5 flex items-center gap-2">
              {/* THE BLUE CIRCULAR VOICE / WAVEFORM BUTTON */}
              <button
                type="button"
                onClick={handleVoiceButtonClick}
                className={`w-10 h-10 rounded-full flex items-center justify-center transition-all cursor-pointer select-none focus:outline-none ${voiceState === "idle"
                    ? "bg-gradient-to-tr from-blue-600 via-indigo-600 to-blue-500 hover:from-blue-500 hover:to-indigo-500 text-white shadow-lg shadow-blue-500/30 hover:scale-105 border border-blue-400/40"
                    : voiceState === "listening"
                      ? "bg-gradient-to-tr from-rose-600 to-red-500 text-white shadow-lg shadow-red-500/40 animate-pulse ring-4 ring-red-500/30 scale-105"
                      : voiceState === "processing"
                        ? "bg-gradient-to-tr from-blue-700 to-indigo-700 text-white ring-4 ring-blue-500/30"
                        : "bg-gradient-to-tr from-cyan-600 to-blue-600 text-white shadow-lg shadow-cyan-500/40 ring-4 ring-cyan-500/30"
                  }`}
                title={
                  voiceState === "idle"
                    ? "Start ChatGPT-style Continuous Voice Mode"
                    : "Continuous Voice Mode Active — Click to Exit"
                }
              >
                {voiceState === "idle" && (
                  <Mic className="h-4 w-4 text-white" />
                )}
                {voiceState === "listening" && (
                  <Square className="h-4 w-4 fill-white text-white" />
                )}
                {voiceState === "processing" && (
                  <Loader2 className="h-4 w-4 text-white animate-spin" />
                )}
                {voiceState === "speaking" && (
                  <Mic className="h-4 w-4 text-white animate-pulse" />
                )}
              </button>

              {/* Standard Ask Assistant Text Button */}
              <button
                onClick={() => handleAsk()}
                disabled={!question.trim() || isLoading}
                className="btn btn-primary btn-sm cursor-pointer gap-1.5 disabled:opacity-50 disabled:pointer-events-none"
              >
                {isLoading ? (
                  <>
                    <div className="h-3 w-3 border-2 border-white/40 border-t-white rounded-full animate-spin" />
                    <span>Searching...</span>
                  </>
                ) : (
                  <>
                    <span>Ask</span>
                    <Send className="h-3.5 w-3.5" />
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Suggested Multilingual Quick Prompts */}
          <div className="flex flex-wrap items-center gap-1.5 pt-1">
            <span className="text-[11px] text-[var(--ink-muted)] font-medium mr-1">Quick Prompts:</span>
            {SUGGESTED_PROMPTS.map((item, idx) => (
              <button
                key={idx}
                onClick={() => {
                  setQuestion(item.text);
                  handleAsk(item.text);
                }}
                disabled={isLoading || voiceState !== "idle"}
                className="btn btn-secondary btn-sm text-[11px] text-left truncate max-w-xs cursor-pointer flex items-center gap-1.5"
              >
                <span className="text-[9px] px-1 py-0.2 rounded bg-[var(--surface-3)] font-mono text-[var(--accent)] font-semibold">
                  {item.lang}
                </span>
                <span className="truncate">&ldquo;{item.text}&rdquo;</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="p-4 rounded-[var(--radius)] bg-[var(--danger-soft)] border border-[var(--danger)]/30 text-[var(--danger)] text-xs flex items-start gap-3">
          <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="font-semibold text-white">Notice</p>
            <p className="mt-0.5">{error}</p>
          </div>
          <button
            onClick={() => handleAsk()}
            className="btn btn-danger btn-sm gap-1"
          >
            <RotateCcw className="h-3 w-3" />
            <span>Retry</span>
          </button>
        </div>
      )}

      {/* Loading Shimmer Card */}
      {isLoading && (
        <div className="card p-6 space-y-4 animate-pulse">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-[var(--radius)] bg-[var(--accent-soft)] flex items-center justify-center text-[var(--accent)]">
              <Bot className="h-4 w-4" />
            </div>
            <div>
              <div className="h-4 w-40 bg-[var(--border)] rounded mb-1" />
              <p className="text-[11px] text-[var(--accent)] font-medium">
                Searching Pinecone index &amp; synthesizing grounded answer...
              </p>
            </div>
          </div>
          <div className="space-y-2">
            <div className="h-3.5 bg-[var(--border)] rounded w-full" />
            <div className="h-3.5 bg-[var(--border)] rounded w-5/6" />
            <div className="h-3.5 bg-[var(--border)] rounded w-3/4" />
          </div>
        </div>
      )}

      {/* Conversation / Answers Feed */}
      <div className="space-y-4">
        {messages.length === 0 && !isLoading && voiceState === "idle" && (
          <div className="card p-12 text-center border-dashed space-y-3">
            <div className="h-12 w-12 rounded-[var(--radius-lg)] bg-[var(--accent-soft)] text-[var(--accent)] flex items-center justify-center mx-auto">
              <BookOpen className="h-6 w-6" />
            </div>
            <h3 className="text-sm font-semibold text-[var(--ink)]">No questions asked yet</h3>
            <p className="text-xs text-[var(--ink-soft)] max-w-md mx-auto">
              Click the <strong className="text-blue-400">blue circular mic button</strong> to start a continuous conversation with Gemini-style automatic silence detection and auto-relisten, or select a quick prompt.
            </p>
          </div>
        )}

        {messages.map((msg) => {
          const isThisMsgPlaying = activePlayingMsgId === msg.id && voiceState === "speaking";

          return (
            <div
              key={msg.id}
              className={`card overflow-hidden divide-y divide-[var(--border)] transition-all ${isThisMsgPlaying ? "ring-2 ring-blue-500/50 shadow-lg shadow-blue-500/10" : ""
                }`}
            >
              {/* User Question Header */}
              <div className="p-4 sm:p-5 bg-[var(--surface-2)] flex items-start gap-3">
                <div className="h-7 w-7 rounded-[var(--radius)] bg-[var(--surface)] border border-[var(--border)] text-[var(--ink-soft)] flex items-center justify-center shrink-0 mt-0.5">
                  <User className="h-3.5 w-3.5" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2 mb-1">
                    <span className="text-xs font-semibold text-[var(--ink-soft)]">Sales Question</span>
                    <span className="text-[10px] text-[var(--ink-muted)]">{msg.timestamp}</span>
                  </div>
                  <p className="text-sm font-medium text-[var(--ink)]">{msg.question}</p>
                </div>
              </div>

              {/* AI Assistant Answer */}
              <div className="p-4 sm:p-6 space-y-4 bg-[var(--surface)]">
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <div className="h-7 w-7 rounded-[var(--radius)] bg-[var(--accent-soft)] text-[var(--accent)] flex items-center justify-center shrink-0">
                      <Bot className="h-3.5 w-3.5" />
                    </div>
                    <div>
                      <span className="text-xs font-bold text-[var(--ink)] flex items-center gap-1.5 flex-wrap">
                        Copilot Answer
                        <span className="badge badge-secondary font-mono text-[10px]">
                          {msg.model}
                        </span>
                        {msg.language && (
                          <span className="status-pill status-pill-info font-mono text-[10px] py-0 px-1.5 uppercase">
                            {msg.language === "hi" ? "हिंदी (HI)" : msg.language === "mr" ? "मराठी (MR)" : "English (EN)"}
                          </span>
                        )}
                        {msg.is_greeting && (
                          <span className="status-pill status-pill-success text-[10px] py-0 px-1.5 flex items-center gap-1">
                            ✨ Grounded Recommendations
                          </span>
                        )}
                        {isThisMsgPlaying && (
                          <span className="status-pill status-pill-info font-mono text-[10px] animate-pulse">
                            Speaking...
                          </span>
                        )}
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center gap-1.5">
                    {/* Speaker / Read Aloud Button */}
                    <button
                      onClick={() => playTTSForMessage(msg.id, msg.answer, msg.language)}
                      className={`btn btn-sm gap-1 text-xs cursor-pointer ${isThisMsgPlaying
                          ? "btn-danger"
                          : "btn-secondary"
                        }`}
                      title={isThisMsgPlaying ? "Interrupt / Stop Speech" : "Listen to answer via Sarvam Bulbul v3 TTS"}
                    >
                      {isThisMsgPlaying ? (
                        <>
                          <Square className="h-3 w-3 fill-white" />
                          <span className="text-[11px]">Stop</span>
                        </>
                      ) : (
                        <>
                          <Volume2 className="h-3.5 w-3.5 text-blue-400" />
                          <span className="text-[11px]">Listen</span>
                        </>
                      )}
                    </button>

                    {/* Copy Button */}
                    <button
                      onClick={() => handleCopy(msg.answer, msg.id)}
                      className="btn btn-secondary btn-sm gap-1 text-xs cursor-pointer"
                      title="Copy Answer"
                    >
                      {copiedId === msg.id ? (
                        <>
                          <Check className="h-3 w-3 text-[var(--success)]" />
                          <span className="text-[var(--success)] text-[11px]">Copied</span>
                        </>
                      ) : (
                        <>
                          <Copy className="h-3 w-3" />
                          <span className="text-[11px]">Copy</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>

                {/* Fallback Notice Banner */}
                {msg.fallback_used && (
                  <div className="p-3 rounded-[var(--radius)] bg-[var(--warning-soft)] border border-[var(--warning)]/30 text-[var(--warning)] text-xs flex items-center gap-2.5">
                    <ShieldAlert className="h-4 w-4 shrink-0" />
                    <span>
                      <strong className="text-white">Information Unavailable:</strong> The indexed playbooks do not contain
                      verified information for this question. A grounded fallback was returned.
                    </span>
                  </div>
                )}

                {/* Formatted Answer Text */}
                <div className="text-sm text-[var(--ink)] leading-relaxed whitespace-pre-line font-normal bg-[var(--surface-2)] p-4 rounded-[var(--radius)] border border-[var(--border)]">
                  {msg.answer}
                </div>

                {/* Source Tags */}
                {msg.sources.length > 0 && (
                  <div className="flex flex-wrap items-center gap-2 text-xs pt-1">
                    <span className="text-[11px] text-[var(--ink-muted)] font-medium">Sources Cited:</span>
                    {msg.sources.map((src, sIdx) => (
                      <span
                        key={sIdx}
                        className="badge badge-primary gap-1 text-[11px]"
                      >
                        <FileText className="h-3 w-3" />
                        {src}
                      </span>
                    ))}
                  </div>
                )}

                {/* Collapsible Context Chunks Details */}
                {msg.context_used.length > 0 && (
                  <div className="pt-2 border-t border-[var(--border)]">
                    <button
                      onClick={() => toggleContext(msg.id)}
                      className="flex items-center gap-1.5 text-xs text-[var(--ink-soft)] hover:text-[var(--ink)] transition cursor-pointer font-medium"
                    >
                      {expandedContexts[msg.id] ? (
                        <>
                          <ChevronUp className="h-3.5 w-3.5" />
                          <span>Hide Retrieved Chunks ({msg.context_used.length})</span>
                        </>
                      ) : (
                        <>
                          <ChevronDown className="h-3.5 w-3.5" />
                          <span>View Retrieved Chunks ({msg.context_used.length})</span>
                        </>
                      )}
                    </button>

                    {expandedContexts[msg.id] && (
                      <div className="mt-3 space-y-2.5">
                        {msg.context_used.map((chunk, cIdx) => (
                          <div
                            key={cIdx}
                            className="p-3 rounded-[var(--radius)] bg-[var(--surface-2)] border border-[var(--border)] text-xs space-y-1.5"
                          >
                            <div className="flex items-center justify-between text-[11px] text-[var(--ink-soft)]">
                              <span className="font-semibold text-[var(--ink)] flex items-center gap-1.5">
                                <FileText className="h-3 w-3 text-[var(--accent)]" />
                                {chunk.source || "Document"} (Page {chunk.page ?? 1})
                              </span>
                              {chunk.score !== undefined && (
                                <span className="badge badge-primary font-mono text-[10px]">
                                  Match: {(chunk.score * 100).toFixed(1)}%
                                </span>
                              )}
                            </div>
                            <p className="text-[var(--ink-soft)] text-xs italic leading-relaxed pl-2 border-l-2 border-[var(--accent)]">
                              &ldquo;{chunk.text}&rdquo;
                            </p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
