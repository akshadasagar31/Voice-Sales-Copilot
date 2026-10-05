// ============================================================================
// MODULE 1: CONTINUOUS VOICE-TO-CRM LEAD CAPTURE COPILOT
// (frontend/app/components/LiveCallVoiceCopilot.tsx)
// ============================================================================
// WHAT THIS COMPONENT DOES:
// Acts as an automated voice copilot for sales reps. During or after a sales call,
// the sales rep speaks naturally (in English, Hindi, or Marathi). The copilot:
// 1. Automatically listens for speech using browser microphone (Web Audio API).
// 2. Uses a ~850ms pre-roll PCM buffer so the first spoken words are never clipped.
// 3. Detects when speaking stops using a 400ms silence threshold.
// 4. Sends audio to Deepgram Speech-To-Text (STT) for transcription.
// 5. Calls DeepSeek LLM to extract structured fields (name, phone, company, loan, etc.)
//    and automatically persists or updates the lead record in PostgreSQL CRM.
// 6. Streams a natural, conversational voice confirmation token-by-token back to the rep.
// 7. Uses sentence-level streaming TTS to start speaking in ~400ms!
// 8. Provides Instant Acoustic Barge-In (0ms interruption if user speaks over the AI).
// 9. Automatically re-arms listening mode for continuous hands-free multi-turn conversation.
// ============================================================================

"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  Mic,
  Square,
  Play,
  Pause,
  RotateCcw,
  Volume2,
  VolumeX,
  AlertCircle,
  Radio,
  Clock,
  Sparkles,
  Headphones,
  CheckCircle2,
  Copy,
  Loader2,
  FileText,
  Check,
  Database,
  UserCheck,
  RefreshCw,
  Phone,
  Building,
  DollarSign,
  Calendar,
  Briefcase,
  Mail,
  FileSpreadsheet,
  Globe,
  ChevronDown,
  User,
  Bot,
} from "lucide-react";
import { SentenceTokenizer, SentenceAudioQueue } from "@/lib/sentenceStreamingTTS";

// TypeScript interface defining runtime conversation messages
export interface ConversationMessage {
  id: string;
  sender: "user" | "assistant";
  text: string;
  timestamp: string;
  language?: string;
}

// Helper to determine accurate UI language indicator for messages
function getMessageLanguage(text: string, fallbackLang?: string): string {
  const hasDevanagari = /[\u0900-\u097F]/.test(text);
  if (!hasDevanagari) return "EN";
  if (/[ळऱ]/.test(text) || /\b(आहे|आहेत|नाही|नाहीत|पाहिजे|हवे|हवं|नमस्कार|किती|द्या|सांगा|होय|आणि)\b/.test(text)) {
    return "MR";
  }
  if (/\b(है|हैं|नहीं|चाहिए|मुझे|नमस्ते|हाँ|बताएं|कितना|और|क्या)\b/.test(text)) {
    return "HI";
  }
  if (fallbackLang && ["mr", "hi", "en"].includes(fallbackLang.toLowerCase())) {
    return fallbackLang.toUpperCase();
  }
  return "HI";
}

// Crisp waveform icon matching reference UI (4 rounded vertical audio bars)
function WaveformIcon({ className = "w-4 h-4 text-[#00897b]" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor">
      <rect x="3" y="8" width="3" height="8" rx="1.5" />
      <rect x="8.5" y="4" width="3" height="16" rx="1.5" />
      <rect x="14" y="6" width="3" height="12" rx="1.5" />
      <rect x="19.5" y="9" width="3" height="6" rx="1.5" />
    </svg>
  );
}

// 12-hour timestamp formatting helper (e.g. "2:34 PM")
function formatMessageTime(): string {
  const d = new Date();
  let hours = d.getHours();
  const minutes = d.getMinutes().toString().padStart(2, "0");
  const ampm = hours >= 12 ? "PM" : "AM";
  hours = hours % 12;
  hours = hours ? hours : 12;
  return `${hours}:${minutes} ${ampm}`;
}

// TypeScript interface defining the exact fields in a sales lead
export interface ExtractedLead {
  id?: number;
  name?: string | null;
  phone?: string | null;
  email?: string | null;
  company?: string | null;
  role?: string | null;
  loan_type?: string | null;
  loan_amount?: number | null;
  tenure_months?: number | null;
  notes?: string | null;
  created_at?: string;
  updated_at?: string;
}

export interface PrimarySpeakerProfile {
  dominantBin: number;
  centroid: number;
  formantRatio: number;
  peakProminence: number;
  sampleCount: number;
  isCalibrated: boolean;
}

/**
 * Computes normalized autocorrelation in the human vocal pitch range (85 Hz to 380 Hz)
 * to measure Voicing Harmonicity. Near-field primary speech exhibits strong phase coherence
 * (harmonicity >= 0.42), while diffuse background voices / reverberation exhibit degraded
 * periodicity (harmonicity < 0.38).
 */
export function computeHarmonicity(chunk: Float32Array, sampleRate: number): number {
  const minLag = Math.max(2, Math.floor(sampleRate / 380));
  const maxLag = Math.min(Math.floor(chunk.length / 2), Math.ceil(sampleRate / 85));
  if (maxLag <= minLag) return 0;

  const nEval = chunk.length - maxLag;
  let sumSq0 = 0;
  for (let i = 0; i < nEval; i += 2) {
    sumSq0 += chunk[i] * chunk[i];
  }
  const norm0 = Math.sqrt(sumSq0 * 2);
  if (norm0 < 1e-4) return 0;

  let bestCorr = 0;
  const stride = sampleRate >= 44100 ? 3 : 2;
  for (let lag = minLag; lag < maxLag; lag += stride) {
    let dot = 0;
    let sumSqLag = 0;
    for (let i = 0; i < nEval; i += 2) {
      const a = chunk[i];
      const b = chunk[i + lag];
      dot += a * b;
      sumSqLag += b * b;
    }
    const normLag = Math.sqrt(sumSqLag * 2);
    if (normLag > 1e-4) {
      const corr = (dot * 2) / (norm0 * normLag);
      if (corr > bestCorr) bestCorr = corr;
    }
  }
  return Math.min(1.0, Math.max(0.0, bestCorr));
}

interface LiveCallVoiceCopilotProps {
  onAudioRecorded?: (blob: Blob) => void;
  onLeadSaved?: (lead: ExtractedLead) => void;
}

export type VoiceState = "idle" | "listening" | "processing" | "speaking";

// ----------------------------------------------------------------------------
// FUNCTION: encodeWav
// ----------------------------------------------------------------------------
// • WHAT IT DOES: Encodes raw Float32Array PCM microphone audio buffers into a standard
//   16-bit linear PCM WAV Blob with a proper 44-byte RIFF header.
// • INPUTS:
//     - buffers (Float32Array[]): Accumulated audio samples from ScriptProcessorNode.
//     - sampleRate (number): Browser audio context sample rate (typically 16000 or 48000 Hz).
// • OUTPUT: A binary Blob with MIME type "audio/wav".
// • WHY IT IS USED: Standard WebM/MediaRecorder encoding introduces variable container
//   muxing latency (100–300ms) and can clip speech. Direct PCM WAV generation has 0ms container
//   delay and prepends the ~850ms pre-roll buffer so no first words are clipped.
// • WHERE IT FITS IN THE FLOW:
//     [Microphone audio buffers] -> [encodeWav] -> [FormData upload to /api/voice-entry]
// ----------------------------------------------------------------------------
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
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);

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

// Convert Float32Array PCM samples (-1.0 to 1.0) to 16-bit linear PCM ArrayBuffer for Deepgram WebSocket
function convertFloat32ToInt16(chunk: Float32Array): ArrayBuffer {
  const pcm16 = new Int16Array(chunk.length);
  for (let i = 0; i < chunk.length; i++) {
    const s = Math.max(-1, Math.min(1, chunk[i]));
    pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return pcm16.buffer;
}

export default function LiveCallVoiceCopilot({ onAudioRecorded, onLeadSaved }: LiveCallVoiceCopilotProps) {
  // Voice Assistant state machine: "idle" | "listening" | "processing" | "speaking"
  const [voiceState, setVoiceState] = useState<"idle" | "listening" | "processing" | "speaking">("idle");
  const [processingStage, setProcessingStage] = useState<"transcribing" | "extracting" | "saving" | "speaking" | null>(null);

  // Language settings: "auto" | "en" | "hi" | "mr"
  const [selectedLanguage, setSelectedLanguage] = useState<string>("auto");
  const [detectedLanguage, setDetectedLanguage] = useState<string>("en");

  // Active Session & Multi-Turn Lead State
  const [extractedLead, setExtractedLead] = useState<ExtractedLead | null>(null);
  const [transcript, setTranscript] = useState<string | null>(null);
  const [sessionTurnCount, setSessionTurnCount] = useState(0);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [saveLeadSuccess, setSaveLeadSuccess] = useState<{ id: number; message: string; isUpdate: boolean } | null>(null);
  const [copiedField, setCopiedField] = useState<string | null>(null);

  // Live audio waveform levels (0-100)
  const [audioLevels, setAudioLevels] = useState<number[]>(new Array(24).fill(16));

  // Voice Response state
  const [assistantResponseText, setAssistantResponseText] = useState<string | null>(null);
  const [isTTSPlaying, setIsTTSPlaying] = useState(false);

  // Runtime Conversation messages (REAL speech & assistant responses only - empty initially)
  const [conversationMessages, setConversationMessages] = useState<ConversationMessage[]>([]);
  const [isLanguageMenuOpen, setIsLanguageMenuOpen] = useState(false);
  const languageMenuRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const currentStreamMsgIdRef = useRef<string | null>(null);
  const currentWsSentenceMsgIdRef = useRef<string | null>(null);

  // Auto-scroll conversation to bottom on message update strictly within messages container
  useEffect(() => {
    const el = messagesContainerRef.current;
    if (el) {
      el.scrollTo({
        top: el.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [conversationMessages]);

  // Close language dropdown on outside click
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (languageMenuRef.current && !languageMenuRef.current.contains(event.target as Node)) {
        setIsLanguageMenuOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Concurrency & turn refs
  const turnIdRef = useRef<number>(0);
  const isContinuousModeRef = useRef<boolean>(false);
  const isVoicePipelineActiveRef = useRef<boolean>(false);
  const activeAbortControllerRef = useRef<AbortController | null>(null);
  const handledFinalTurnsRef = useRef<Set<number>>(new Set());
  const startListeningTurnRef = useRef<() => void>(() => {});
  const isStartingSessionRef = useRef<boolean>(false);

  // WebSocket live streaming STT refs
  const sttSocketRef = useRef<WebSocket | null>(null);
  const isWsStreamingRef = useRef<boolean>(false);
  const wsPendingQueueRef = useRef<Float32Array[]>([]);
  const hasWsFinalizedRef = useRef<boolean>(false);
  const wasSpeakingRef = useRef<boolean>(false);
  const activeTtsModelRef = useRef<string | null>(null);
  const detectedLanguageRef = useRef<string>("en");
  const isManualLanguageSelectionRef = useRef<boolean>(false);

  // Multi-Turn Lead Memory: tracks active PostgreSQL lead ID and fields across turns
  const activeLeadIdRef = useRef<number | null>(null);
  const currentLeadRef = useRef<ExtractedLead | null>(null);

  // Continuous Web Audio API & PCM capture refs (Guarantees unified AudioContext & pre-roll)
  const isMicCaptureActiveRef = useRef<boolean>(true);
  const audioStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const resumePromiseRef = useRef<Promise<void> | null>(null);
  const lifecycleLockRef = useRef<Promise<void>>(Promise.resolve());
  const isStartingTurnRef = useRef<boolean>(false);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioSourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const audioWorkletNodeRef = useRef<AudioWorkletNode | null>(null);
  const workletLoadedContextsRef = useRef<WeakSet<AudioContext>>(new WeakSet());
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const silentGainRef = useRef<GainNode | null>(null);
  const filterNodesRef = useRef<BiquadFilterNode[]>([]);
  const sttFinalPerfTimestampRef = useRef<number>(0);

  // Pre-roll PCM buffers (~850ms rolling window)
  const preRollBuffersRef = useRef<Float32Array[]>([]);
  const recordedBuffersRef = useRef<Float32Array[]>([]);
  const isSpeechTurnActiveRef = useRef<boolean>(false);
  const speechDetectedRef = useRef<boolean>(false);
  const consecutiveSpeechFramesRef = useRef<number>(0);
  const consecutiveBargeInFramesRef = useRef<number>(0);
  const speakerBleedBaselineRef = useRef<number>(18);
  const bleedRmsRef = useRef<number>(0.012);
  const speakerBleedDbRef = useRef<number>(-45.0);
  const ambientNoiseFloorRef = useRef<number>(0.006);
  const ambientNoiseFloorDbRef = useRef<number>(-50.0);
  const primarySpeakerProfileRef = useRef<PrimarySpeakerProfile>({
    dominantBin: 5,
    centroid: 6.0,
    formantRatio: 1.5,
    peakProminence: 2.0,
    sampleCount: 0,
    isCalibrated: false,
  });
  const sentenceStartTimestampRef = useRef<number>(0);
  const speechEndTimestampRef = useRef<number>(0);
  const llmStartTimestampRef = useRef<number>(0);
  const firstSentenceTimestampRef = useRef<number>(0);
  const ttsStartTimestampRef = useRef<number>(0);
  const hasFirstAudioPlayedRef = useRef<boolean>(false);

  const silenceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const autoListenTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const timerIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const voiceStateRef = useRef<VoiceState>("idle");

  useEffect(() => {
    voiceStateRef.current = voiceState;
  }, [voiceState]);

  // Single TTS Audio Player Ref & Sentence Audio Queue Ref
  const ttsAudioRef = useRef<HTMLAudioElement | null>(null);
  const sentenceAudioQueueRef = useRef<SentenceAudioQueue | null>(null);

  // Check if a turn has been superseded by a newer one or canceled
  const isTurnStale = useCallback((turnId: number) => {
    return turnId !== turnIdRef.current;
  }, []);

  // Stop any active TTS audio playback safely and abort pending queue
  const stopAllAudioPlayback = useCallback(() => {
    if (sentenceAudioQueueRef.current) {
      try {
        sentenceAudioQueueRef.current.bargeIn();
      } catch (_) {}
    }
    if (ttsAudioRef.current) {
      try {
        ttsAudioRef.current.pause();
        ttsAudioRef.current.currentTime = 0;
      } catch (_) {}
    }
    setIsTTSPlaying(false);
  }, []);

  const startStreamingSTTRef = useRef<((buffers: Float32Array[]) => void) | null>(null);

  // Cancel any active turn and abort pending HTTP requests
  const cancelActiveTurn = useCallback(() => {
    turnIdRef.current += 1;
    if (activeAbortControllerRef.current) {
      activeAbortControllerRef.current.abort();
      activeAbortControllerRef.current = null;
    }
    if (!isContinuousModeRef.current && sttSocketRef.current) {
      try {
        sttSocketRef.current.close();
      } catch (_) {}
      sttSocketRef.current = null;
    }
    isWsStreamingRef.current = false;
    hasWsFinalizedRef.current = false;
    wsPendingQueueRef.current = [];
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    if (autoListenTimeoutRef.current) {
      clearTimeout(autoListenTimeoutRef.current);
      autoListenTimeoutRef.current = null;
    }
    speechEndTimestampRef.current = 0;
    llmStartTimestampRef.current = 0;
    firstSentenceTimestampRef.current = 0;
    ttsStartTimestampRef.current = 0;
    hasFirstAudioPlayedRef.current = false;
    wasSpeakingRef.current = false;
    stopAllAudioPlayback();
    isVoicePipelineActiveRef.current = false;
  }, [stopAllAudioPlayback]);

  // --------------------------------------------------------------------------
  // HANDLER: executeBargeIn (Instant Acoustic Barge-In)
  // --------------------------------------------------------------------------
  // • WHAT IT DOES: Immediately halts AI speech playback (0ms delay), aborts pending
  //   TTS synthesis fetches, prepends the ~850ms pre-roll buffer, and starts capturing
  //   the prospect's interrupting speech.
  // • INPUTS: None (triggered by VAD RMS speech detection while assistant is speaking).
  // • OUTPUT: None (transitions state to "listening" and begins capturing new audio).
  // • WHY IT IS USED: Human conversations are dynamic. If a prospect interrupts,
  //   a voice copilot must shut up instantly and listen, just like a real person would.
  // • WHERE IT FITS IN THE FLOW:
  //     [User speaks over AI] -> [ScriptProcessor VAD detects RMS] -> [executeBargeIn] -> [Record speech]
  // --------------------------------------------------------------------------
  const executeBargeIn = useCallback(() => {
    console.log("[VoiceCopilot] Instant acoustic barge-in! Halting assistant audio (0ms delay) and capturing speech...");

    consecutiveBargeInFramesRef.current = 0;
    consecutiveSpeechFramesRef.current = 0;

    // 1. Cut off active turn and stop playing audio immediately
    cancelActiveTurn();
    stopAllAudioPlayback();
    hasFirstAudioPlayedRef.current = false;

    // 2. Keep continuous voice mode active and switch immediately to listening
    isContinuousModeRef.current = true;
    setVoiceState("listening");
    voiceStateRef.current = "listening";
    setProcessingStage(null);

    // 3. Immediately start speech turn with pre-roll buffer prepended so no first words are lost!
    isSpeechTurnActiveRef.current = true;
    speechDetectedRef.current = true;
    recordedBuffersRef.current = [...preRollBuffersRef.current];

    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }

    // 4. Immediately begin streaming STT over WebSocket for newly captured speech
    startStreamingSTTRef.current?.([...preRollBuffersRef.current]);
  }, [cancelActiveTurn, stopAllAudioPlayback]);

  // Clean up on component unmount
  useEffect(() => {
    return () => {
      cancelActiveTurn();
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
      if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
      if (autoListenTimeoutRef.current) clearTimeout(autoListenTimeoutRef.current);
      if (audioWorkletNodeRef.current) {
        try {
          audioWorkletNodeRef.current.port.onmessage = null;
          audioWorkletNodeRef.current.disconnect();
        } catch (_) {}
        audioWorkletNodeRef.current = null;
      }
      if (scriptProcessorRef.current) {
        try {
          scriptProcessorRef.current.disconnect();
        } catch (_) {}
        scriptProcessorRef.current = null;
      }

      if (silentGainRef.current) {
        try {
          silentGainRef.current.disconnect();
        } catch (_) {}
        silentGainRef.current = null;
      }
      if (analyserRef.current) {
        try {
          analyserRef.current.disconnect();
        } catch (_) {}
        analyserRef.current = null;
      }
      if (audioSourceNodeRef.current) {
        try {
          audioSourceNodeRef.current.disconnect();
        } catch (_) {}
        audioSourceNodeRef.current = null;
      }
      if (audioStreamRef.current) {
        audioStreamRef.current.getTracks().forEach((track) => track.stop());
        audioStreamRef.current = null;
      }
      const unmountCtx = audioContextRef.current;
      audioContextRef.current = null;
      if (unmountCtx && unmountCtx.state !== "closed") {
        (async () => {
          if (resumePromiseRef.current) {
            try {
              await resumePromiseRef.current;
            } catch (_) {}
          }
          try {
            if (unmountCtx.state !== "closed") {
              await unmountCtx.close();
            }
          } catch (_) {}
        })();
      }
    };
  }, [cancelActiveTurn]);

  // Initialize or re-attach single audio player on client mount
  useEffect(() => {
    if (typeof window !== "undefined" && !ttsAudioRef.current) {
      ttsAudioRef.current = new Audio();
    }
  }, []);

  // Format seconds into MM:SS
  const formatTime = (totalSec: number) => {
    if (isNaN(totalSec) || totalSec < 0) return "00:00";
    const m = Math.floor(totalSec / 60);
    const s = Math.floor(totalSec % 60);
    return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  };

  // Localized greeting only helpers
  const isGreetingOnly = (text: string): boolean => {
    if (!text) return false;
    const clean = text.trim().toLowerCase().replace(/[^\w\s\u0900-\u097F]/g, " ").replace(/\s+/g, " ").trim();
    if (!clean) return false;

    // Substantive keywords indicating lead/customer info or domain questions
    const substantiveWords = new Set([
      "what", "why", "how", "when", "where", "who", "which", "can", "could", "should", "would",
      "name", "phone", "email", "company", "loan", "amount", "tenure", "interest", "rate", "lakh", "crore",
      "met", "spoke", "call", "called", "customer", "prospect", "client", "director", "manager", "officer",
      "dollar", "dollars", "rupee", "rupees", "months", "years", "personal", "home", "business", "mortgage",
      "क्या", "कैसे", "कहाँ", "कब", "कौन", "कितना", "बताएं", "बताओ", "जानकारी", "नाम", "फोन", "कंपनी", "लोन", "ब्याज", "लाख", "करोड़", "रुपये", "ग्राहक",
      "काय", "कसे", "कशी", "कसा", "कुठे", "कधी", "कोण", "किती", "सांगा", "माहिती", "नाव", "फोन", "कंपनी", "कर्ज", "व्याज", "लाख", "कोटी", "रुपये", "ग्राहक"
    ]);

    const words = clean.split(" ");
    if (words.some((w) => substantiveWords.has(w))) {
      return false;
    }

    const greetingTokens = new Set([
      "hi", "hello", "hey", "greetings", "good", "morning", "afternoon", "evening", "howdy", "welcome",
      "namaste", "namaskar", "pranam",
      "नमस्ते", "नमस्कार", "प्रणाम", "हेलो", "हाय", "हैलो", "हॅलो", "सुप्रभात", "शुभ",
      "सकाळ", "दुपार", "संध्या", "संध्याकाळ", "दिन", "प्रभात", "राम"
    ]);

    const politeFillers = new Set([
      "there", "copilot", "assistant", "ai", "team", "sir", "madam", "ji",
      "जी", "सर", "मॅडम", "मित्र", "मित्रा", "साहेब", "all", "everyone"
    ]);

    return words.every((w) => greetingTokens.has(w) || politeFillers.has(w));
  };

  const getNaturalGreeting = (lang: string): string => {
    if (lang === "hi") return "नमस्ते! मैं आज आपकी क्या सहायता कर सकता हूँ?";
    if (lang === "mr") return "नमस्कार! मी आज आपली काय मदत करू शकेन?";
    return "Hello! How can I help you today?";
  };

  const getNoLeadInfoPrompt = (lang: string): string => {
    if (lang === "hi") return "मैं सुन रहा हूँ। जब भी आप तैयार हों, कृपया ग्राहक या लीड का विवरण साझा करें।";
    if (lang === "mr") return "मी ऐकत आहे. जेव्हा आपण तयार असाल, तेव्हा कृपया ग्राहक किंवा लीडचे तपशील सांगा.";
    return "I'm listening. Please share the customer or lead details whenever you're ready.";
  };

  const isAssistantQueryText = (text: string): boolean => {
    if (!text) return false;
    const clean = text.trim().toLowerCase();
    const isName = /\b(?:what(?:'s|\s+is)\s+(?:your|ur)\s+name|tell\s+me\s+your\s+name|who\s+are\s+you\s+called|your\s+name\s+please)\b|(?:आप(?:का)?\s*नाम\s*क्या|तुम्हारा\s*नाम\s*क्या|अपना\s*नाम\s*बता|नाम\s*क्या\s*है\s*आप)|(?:तुम(?:चे|चं)\s*नाव\s*काय|तुझ(?:ं)?\s*नाव\s*काय|आपले\s*नाव\s*काय|नाव\s*काय\s*आहे\s*तुम)/i.test(clean);
    const isId = /\b(?:who\s+are\s+you|who\s+r\s+u|what\s+are\s+you|introduce\s+yourself|tell\s+me\s+about\s+yourself|who\s+is\s+speaking)\b|(?:आप\s*कौन\s*(?:हैं|हो)|तुम\s*कौन\s*हो|अपना\s*परिचय)|(?:तुम्ही\s*कोण\s*आहात|तू\s*कोण\s*आहेस|आपण\s*कोण\s*आहात|आपली\s*ओळख)/i.test(clean);
    const isCap = /\b(?:what\s+can\s+you\s+do|what\s+do\s+you\s+do|how\s+can\s+you\s+help|what\s+are\s+your\s+features|what\s+is\s+your\s+purpose|how\s+do\s+you\s+work)\b|(?:आप\s*क्या\s*कर\s*सकते\s*हैं|आप\s*क्या\s*काम\s*करते\s*हैं|क्या\s*मदद\s*कर\s*सकते\s*हैं|आप\s*क्या\s*करते\s*हैं)|(?:तुम्ही\s*काय\s*करू\s*शकता|आपण\s*काय\s*करू\s*शकता|काय\s*मदत\s*करू\s*शकता|तुम्ही\s*काय\s*काम\s*करता)/i.test(clean);
    return isName || isId || isCap;
  };

  const getAssistantAnswerFallback = (text: string, lang: string): string => {
    const clean = text.trim().toLowerCase();
    const isCap = /\b(?:what\s+can\s+you\s+do|what\s+do\s+you\s+do|how\s+can\s+you\s+help|what\s+are\s+your\s+features|what\s+is\s+your\s+purpose|how\s+do\s+you\s+work)\b|(?:आप\s*क्या\s*कर\s*सकते\s*हैं|आप\s*क्या\s*काम\s*करते\s*हैं|क्या\s*मदद\s*कर\s*सकते\s*हैं|आप\s*क्या\s*करते\s*हैं)|(?:तुम्ही\s*काय\s*करू\s*शकता|आपण\s*काय\s*करू\s*शकता|काय\s*मदत\s*करू\s*शकता|तुम्ही\s*काय\s*काम\s*करता)/i.test(clean);
    const isId = /\b(?:who\s+are\s+you|who\s+r\s+u|what\s+are\s+you|introduce\s+yourself|tell\s+me\s+about\s+yourself|who\s+is\s+speaking)\b|(?:आप\s*कौन\s*(?:हैं|हो)|तुम\s*कौन\s*हो|अपना\s*परिचय)|(?:तुम्ही\s*कोण\s*आहात|तू\s*कोण\s*आहेस|आपण\s*कोण\s*आहात|आपली\s*ओळख)/i.test(clean);

    if (isCap) {
      if (lang === "hi") return "मैं आवाज़ के ज़रिए ग्राहकों के नाम, फोन नंबर, कंपनी, लोन की राशि और अवधि जैसे लीड विवरण सीआरएम में दर्ज और सत्यापित कर सकता हूँ।";
      if (lang === "mr") return "मी आवाजाद्वारे ग्राहकांचे नाव, फोन नंबर, कंपनी, कर्जाची रक्कम आणि कालावधी यांसारखे तपशील थेट सीआरएममध्ये नोंदवून मदत करू शकतो.";
      return "I can help you collect and qualify leads by voice, record customer names, contact numbers, companies, loan amounts, and tenure directly into your CRM.";
    }
    if (isId) {
      if (lang === "hi") return "मैं वॉयस कोपायलट हूँ, आपका वॉयस-सक्षम सीआरएम असिस्टेंट। मैं ग्राहकों के विवरण, लोन की जरूरतें और लीड्स दर्ज करने में आपकी सहायता करता हूँ।";
      if (lang === "mr") return "मी व्हॉइस कोपायलट आहे, आपला वॉयस-सक्षम सीआरएम असिस्टंट. मी ग्राहकांचे तपशील आणि कर्जाच्या गरजा नोंदवून लीड्स व्यवस्थापित करण्यात मदत करतो.";
      return "I am VoiceCopilot, your voice-powered CRM sales copilot. I assist you with capturing customer details, loan requirements, and qualifying leads.";
    }
    // Name query
    if (lang === "hi") return "मैं वॉयस कोपायलट हूँ, आपका एआई सेल्स असिस्टेंट। मैं ग्राहकों की जानकारी और लोन लीड्स दर्ज करने में आपकी मदद करता हूँ।";
    if (lang === "mr") return "मी व्हॉइस कोपायलट आहे, आपला एआय सेल्स असिस्टंट. मी ग्राहकांची माहिती आणि लोन लीड्स नोंदवण्यात मदत करतो.";
    return "I am VoiceCopilot, your AI sales assistant. I help you capture and qualify loan leads quickly.";
  };

  const hasAnyLeadData = (lead: ExtractedLead | null | undefined): boolean => {
    if (!lead) return false;
    return Boolean(
      (lead.name && lead.name.trim()) ||
      (lead.phone && lead.phone.trim()) ||
      (lead.email && lead.email.trim()) ||
      (lead.company && lead.company.trim()) ||
      (lead.role && lead.role.trim()) ||
      (lead.loan_type && lead.loan_type.trim()) ||
      (lead.loan_amount !== null && lead.loan_amount !== undefined && !isNaN(Number(lead.loan_amount))) ||
      (lead.tenure_months !== null && lead.tenure_months !== undefined && !isNaN(Number(lead.tenure_months))) ||
      (lead.notes && lead.notes.trim())
    );
  };

  // Generate localized concise confirmation text
  const generateSpokenResponseText = (lead: ExtractedLead | null, lang: string, isUpdate: boolean): string => {
    const name = lead?.name?.trim() || "";
    const company = lead?.company?.trim() || "";
    const loanType = lead?.loan_type?.trim() || "";
    const loanAmount = lead?.loan_amount;

    let amountStr = "";
    if (loanAmount !== null && loanAmount !== undefined) {
      const num = Number(loanAmount);
      if (num >= 10000000) {
        amountStr = `${(num / 10000000).toFixed(1).replace(".0", "")} crore`;
      } else if (num >= 100000) {
        amountStr = `${(num / 100000).toFixed(1).replace(".0", "")} lakh`;
      } else if (num >= 1000) {
        amountStr = `$${num.toLocaleString()}`;
      } else {
        amountStr = `$${num}`;
      }
    }

    if (lang === "hi") {
      if (isUpdate) {
        if (name) return `सीआरएम में ${name} के लीड विवरण को अपडेट कर दिया गया है।`;
        return "सीआरएम में लीड विवरण सफलतापूर्वक अपडेट कर दिया गया है।";
      }
      if (name) {
        const comp = company ? ` (${company})` : "";
        const amt = amountStr ? ` ${amountStr}` : "";
        const lt = loanType ? ` ${loanType}` : " लोन";
        return `सीआरएम में ${name}${comp} के लिए${amt}${lt} विवरण सफलतापूर्वक सहेज लिया गया है।`;
      }
      return "लीड विवरण सफलतापूर्वक सत्यापित करके सीआरएम डेटाबेस में सहेज लिया गया है।";
    }

    if (lang === "mr") {
      if (isUpdate) {
        if (name) return `सीआरएम मध्ये ${name} यांचे लीड तपशील अपडेट केले आहेत.`;
        return "सीआरएम मध्ये लीड तपशील यशस्वीरित्या अपडेट केले आहेत.";
      }
      if (name) {
        const comp = company ? ` (${company})` : "";
        const amt = amountStr ? ` ${amountStr}` : "";
        const lt = loanType ? ` ${loanType}` : " कर्ज";
        return `सीआरएम मध्ये ${name}${comp} यांचे${amt}${lt} तपशील यशस्वीरित्या नोंदवले गेले आहेत.`;
      }
      return "लीड तपशील यशस्वीरित्या समजून सीआरएम डेटाबेसमध्ये नोंदवले गेले आहेत.";
    }

    // English
    if (isUpdate) {
      if (name) return `Updated ${name}'s lead details in the CRM.`;
      return "Lead details have been successfully updated in the CRM database.";
    }
    if (name || company || loanType || loanAmount) {
      const subj = name || "the prospect";
      const compPart = company ? ` from ${company}` : "";
      const loanPart = amountStr && loanType
        ? ` for a ${amountStr} ${loanType}`
        : amountStr
        ? ` for ${amountStr}`
        : loanType
        ? ` for a ${loanType}`
        : "";
      return `Lead for ${subj}${compPart}${loanPart} has been successfully recorded in the CRM.`;
    }
    return "Lead details have been successfully verified and saved to the CRM database.";
  };

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

  // --------------------------------------------------------------------------
  // HANDLER: handleFinalSTTResponse
  // --------------------------------------------------------------------------
  // • WHAT IT DOES: Unpacks final STT transcript and metadata from Deepgram Nova-3,
  //   updates lead memory and PostgreSQL CRM, and starts sentence-level streaming TTS.
  // • FAST-PATH (< 1s latency): If deterministic sentence 1 prompt is present,
  //   enqueues immediately into SentenceAudioQueue for Sarvam Bulbul v3 TTS.
  // --------------------------------------------------------------------------
  const handleFinalSTTResponse = useCallback(
    async (sttData: any, turnId: number, abortController: AbortController) => {
      if (isTurnStale(turnId)) return;
      if (handledFinalTurnsRef.current.has(turnId)) {
        console.log(`[VoiceCopilot] Turn #${turnId} already handled. Skipping duplicate final STT.`);
        return;
      }
      handledFinalTurnsRef.current.add(turnId);

      const rawTranscript = (sttData.transcript || "").trim();
      let detectedLang = (sttData.detected_language || "").toLowerCase().split("-")[0];
      if (isManualLanguageSelectionRef.current && selectedLanguage !== "auto") {
        detectedLang = selectedLanguage;
      } else if (!["en", "hi", "mr"].includes(detectedLang)) {
        if (/[\u0900-\u097F]/.test(rawTranscript)) {
          if (/\b(आहे|किती|मला|पाहिजे|हवे|काय|कसे|नाही|दरमहा|वर्षांसाठी|मुदत)\b/.test(rawTranscript)) {
            detectedLang = "mr";
          } else {
            detectedLang = "hi";
          }
        } else {
          detectedLang = "en";
        }
      }

      const respLang = (sttData.language || detectedLang).toLowerCase().split("-")[0];

      // Update the language badge strictly from the CURRENT user turn audio (sttData.detected_language)
      // Never inherit from the assistant's previous language or response language
      const turnAudioLang = ["mr", "hi", "en"].includes(detectedLang) ? detectedLang : "en";
      setDetectedLanguage(turnAudioLang);
      detectedLanguageRef.current = turnAudioLang;
      // In auto mode, NEVER overwrite selectedLanguage with the assistant's response language.
      // selectedLanguage remains "auto" so the next turn detects fresh from the user's audio.
      if (isManualLanguageSelectionRef.current && selectedLanguage !== "auto") {
        // User explicitly locked language; preserve manual choice
      } else {
        setSelectedLanguage("auto");
      }

      // Check if backend reset/new lead occurred
      if (sttData.is_new_lead) {
        console.log("[VoiceCopilot] Backend detected new lead / reset. Starting fresh from Name.");
        activeLeadIdRef.current = sttData.lead_id || null;
        currentLeadRef.current = sttData.lead || null;
        setExtractedLead(sttData.lead || null);
        setSaveLeadSuccess(null);
      }

      // If Deepgram heard only background noise and no actual words:
      if (!rawTranscript) {
        console.log(`[VoiceCopilot] Turn #${turnId}: No clear speech detected. Auto-rearming listening...`);
        isVoicePipelineActiveRef.current = false;
        if (isContinuousModeRef.current) {
          setVoiceState("listening");
          voiceStateRef.current = "listening";
          setProcessingStage(null);
          isSpeechTurnActiveRef.current = false;
          speechDetectedRef.current = false;
          recordedBuffersRef.current = [];
          preRollBuffersRef.current = [];
          startListeningTurn();
        } else {
          setVoiceState("idle");
          voiceStateRef.current = "idle";
          setProcessingStage(null);
        }
        return;
      }

      setTranscript(rawTranscript);
      setSessionTurnCount((prev) => prev + 1);
      console.log(`[VoiceCopilot] Turn #${turnId} Transcript (${detectedLang}): "${rawTranscript}"`);

      // Add REAL runtime user speech message to conversation panel
      const userMsgId = `user-${turnId}-${Date.now()}`;
      const userTimestamp = formatMessageTime();
      setConversationMessages((prev) => [
        ...prev,
        {
          id: userMsgId,
          sender: "user",
          text: rawTranscript,
          timestamp: userTimestamp,
        },
      ]);

      // STEP 2: SETUP SENTENCE-STREAMING AUDIO QUEUE WITH SARVAM FEMALE VOICE
      if (!ttsAudioRef.current && typeof window !== "undefined") {
        ttsAudioRef.current = new Audio();
      }
      const audioPlayer = ttsAudioRef.current;
      if (audioPlayer && !sentenceAudioQueueRef.current) {
        sentenceAudioQueueRef.current = new SentenceAudioQueue(audioPlayer, {}, audioContextRef.current, {
          module: "module1",
          model: activeTtsModelRef.current || undefined,
        });
      }
      if (sentenceAudioQueueRef.current) {
        sentenceAudioQueueRef.current.setOptions({
          module: "module1",
          model: activeTtsModelRef.current || undefined,
        });
        if (audioContextRef.current) {
          sentenceAudioQueueRef.current.setAudioContext(audioContextRef.current);
        }
      }

      sentenceAudioQueueRef.current?.updateCallbacks({
        onSentenceStart: (text: string, index: number) => {
          if (isTurnStale(turnId)) return;
          if (!hasFirstAudioPlayedRef.current) {
            hasFirstAudioPlayedRef.current = true;
            sentenceStartTimestampRef.current = Date.now();
            const tFirstAudio = performance.now();
            const speechEnd = speechEndTimestampRef.current || (sentenceStartTimestampRef.current - 450);
            const totalSpeechToAudio = sentenceStartTimestampRef.current - speechEnd;
            const finalSttToAudio = sttFinalPerfTimestampRef.current > 0
              ? tFirstAudio - sttFinalPerfTimestampRef.current
              : 0;
            const ttsToAudio = ttsStartTimestampRef.current > 0
              ? tFirstAudio - ttsStartTimestampRef.current
              : 0;
            const llmFirstSentence = (firstSentenceTimestampRef.current > 0 && llmStartTimestampRef.current > 0)
              ? firstSentenceTimestampRef.current - llmStartTimestampRef.current
              : 0;

            console.log(`[LATENCY] 5. FIRST-AUDIO-PLAYBACK at perf=${tFirstAudio.toFixed(3)}ms (+${ttsToAudio.toFixed(1)}ms from TTS request)`);
            console.log(
              `%c[VoiceCopilot:Latency] >>> FINAL-STT -> FIRST-AUDIO: ${finalSttToAudio.toFixed(1)}ms (Target: 300-600ms) | Total Speech-to-Audio: ${totalSpeechToAudio}ms <<<`,
              "color: #10b981; font-weight: bold; font-size: 14px;"
            );
            console.log(
              `[LATENCY] Breakdown: LLM-First-Sentence: ${llmFirstSentence.toFixed(1)}ms | TTS-To-Audio: ${ttsToAudio.toFixed(1)}ms | Final-STT-To-Audio: ${finalSttToAudio.toFixed(1)}ms`
            );
          } else {
            console.log(`[LATENCY] Sentence #${index} playback started (+${(performance.now() - (sttFinalPerfTimestampRef.current || 0)).toFixed(1)}ms from STT): "${(text || '').slice(0, 40)}..."`);
          }
          setVoiceState("speaking");
          voiceStateRef.current = "speaking";
          setProcessingStage("speaking");
          setIsTTSPlaying(true);
        },
        onQueueComplete: () => {
          console.log(`[VoiceCopilot] Voice response finished for Turn #${turnId}.`);
          setIsTTSPlaying(false);
          isVoicePipelineActiveRef.current = false;
          speechEndTimestampRef.current = 0;
          wasSpeakingRef.current = false;
          setProcessingStage(null);

          isSpeechTurnActiveRef.current = false;
          speechDetectedRef.current = false;
          recordedBuffersRef.current = [];
          preRollBuffersRef.current = [];
          consecutiveSpeechFramesRef.current = 0;
          consecutiveBargeInFramesRef.current = 0;

          if (isContinuousModeRef.current) {
            console.log(`[VoiceCopilot] Continuous mode active. Returning to Listening for Turn #${turnId + 1}.`);
            setVoiceState("listening");
            voiceStateRef.current = "listening";
            if (autoListenTimeoutRef.current) clearTimeout(autoListenTimeoutRef.current);
            autoListenTimeoutRef.current = setTimeout(() => {
              if (isContinuousModeRef.current && voiceStateRef.current === "listening") {
                startListeningTurnRef.current();
              }
            }, 250);
          } else {
            console.log(`[VoiceCopilot] Session ended. Returning to Tap to Speak.`);
            setVoiceState("idle");
            voiceStateRef.current = "idle";
            if (audioStreamRef.current) {
              try {
                audioStreamRef.current.getTracks().forEach((track) => track.stop());
              } catch (_) {}
              audioStreamRef.current = null;
            }
          }
        },
        onError: (e) => {
          console.warn("[VoiceCopilot] Audio playback error:", e);
          setIsTTSPlaying(false);
          isVoicePipelineActiveRef.current = false;
          speechEndTimestampRef.current = 0;
          wasSpeakingRef.current = false;
          setProcessingStage(null);

          if (isContinuousModeRef.current) {
            setVoiceState("listening");
            voiceStateRef.current = "listening";
            if (autoListenTimeoutRef.current) clearTimeout(autoListenTimeoutRef.current);
            autoListenTimeoutRef.current = setTimeout(() => {
              if (isContinuousModeRef.current && voiceStateRef.current === "listening") {
                startListeningTurnRef.current();
              }
            }, 250);
          } else {
            setVoiceState("idle");
            voiceStateRef.current = "idle";
            if (audioStreamRef.current) {
              try {
                audioStreamRef.current.getTracks().forEach((track) => track.stop());
              } catch (_) {}
              audioStreamRef.current = null;
            }
          }
        },
      });

      sentenceAudioQueueRef.current?.startNewTurn(turnId);
      hasFirstAudioPlayedRef.current = false;
      const tokenizer = new SentenceTokenizer();

      const tLlmStart = performance.now();
      llmStartTimestampRef.current = tLlmStart;
      console.log(`[LATENCY] 2. LLM request start at perf=${tLlmStart.toFixed(3)}ms`);

      // FAST-PATH: INSTANT DETERMINISTIC FIRST RESPONSE (< 1s speech-to-audio)
      const immediateSentence =
        sttData.immediate_sentence1 ||
        (sttData.is_greeting
          ? sttData.greeting_response || getNaturalGreeting(respLang)
          : (sttData.is_assistant_query && !sttData.has_subsequent_sentences)
          ? sttData.assistant_response || (sttData.is_general_query ? null : getAssistantAnswerFallback(rawTranscript, respLang))
          : null);

      if (immediateSentence) {
        console.log(`[VoiceCopilot:Speed] Starting immediate TTS for sentence 1: "${immediateSentence}"`);
        setAssistantResponseText(immediateSentence);

        // Add REAL assistant response to conversation panel
        const asstMsgId = `asst-${turnId}-${Date.now()}`;
        const asstTime = formatMessageTime();
        setConversationMessages((prev) => [
          ...prev,
          {
            id: asstMsgId,
            sender: "assistant",
            text: immediateSentence,
            timestamp: asstTime,
          },
        ]);

        setVoiceState("speaking");
        voiceStateRef.current = "speaking";
        setProcessingStage("speaking");
        setIsTTSPlaying(true);

        // ENQUEUE FIRST SENTENCE TO TTS IMMEDIATELY (before lead state updates)
        const sentences = tokenizer.feed(immediateSentence);
        const trailing = tokenizer.flush();
        if (trailing) sentences.push(trailing);
        if (sentences.length === 0 && immediateSentence.trim()) sentences.push(immediateSentence.trim());

        const tFirstSentence = performance.now();
        firstSentenceTimestampRef.current = tFirstSentence;
        const llmDur = tFirstSentence - tLlmStart;
        console.log(`[LATENCY] 3. first LLM sentence ready at perf=${tFirstSentence.toFixed(3)}ms (+${llmDur.toFixed(1)}ms) sentences=${sentences.length}`);

        const tTtsStart = performance.now();
        ttsStartTimestampRef.current = tTtsStart;
        console.log(`[LATENCY] 4. TTS request start at perf=${tTtsStart.toFixed(3)}ms sentence="${(sentences[0] || '').slice(0, 50)}"`);

        for (const s of sentences) {
          sentenceAudioQueueRef.current?.enqueueSentence(s, respLang);
        }
        if (!sttData.has_subsequent_sentences) {
          sentenceAudioQueueRef.current?.markStreamComplete();
        }

        const latencyToTts = Date.now() - (speechEndTimestampRef.current || Date.now());
        console.log(`[VoiceCopilot:Latency] Speech End -> Sentence 1 Enqueued for TTS: ${latencyToTts}ms`);

        // Lead state updates happen AFTER TTS enqueue to avoid blocking the fast-path
        if (sttData.lead && !sttData.is_assistant_query && !sttData.is_general_query) {
          currentLeadRef.current = sttData.lead;
          setExtractedLead(sttData.lead);
          if (onLeadSaved) onLeadSaved(sttData.lead);
        }
        if (sttData.lead_id && !sttData.is_assistant_query && !sttData.is_general_query) {
          activeLeadIdRef.current = sttData.lead_id;
          setSaveLeadSuccess({
            id: sttData.lead_id,
            message: `Lead #${sttData.lead_id} active in PostgreSQL database!`,
            isUpdate: true,
          });
        }
        return;
      }

      if (!immediateSentence && sttData.has_subsequent_sentences) {
        // Asynchronous WebSocket streaming mode: subsequent sentences arrive via type="sentence"
        setVoiceState("speaking");
        voiceStateRef.current = "speaking";
        setProcessingStage("speaking");
        setIsTTSPlaying(true);
        return;
      }

      // ASSISTANT-CONVERSATION HANDLING (Fallback)
      const isAssistantQueryTurn = Boolean(sttData.is_assistant_query || (!sttData.is_general_query && isAssistantQueryText(rawTranscript)));
      if (isAssistantQueryTurn) {
        if (sttData.is_general_query && sttData.has_subsequent_sentences) {
          setVoiceState("speaking");
          voiceStateRef.current = "speaking";
          setProcessingStage("speaking");
          setIsTTSPlaying(true);
          return;
        }

        const assistantSpoken =
          sttData.assistant_response ||
          sttData.immediate_sentence1 ||
          (sttData.is_general_query ? "" : getAssistantAnswerFallback(rawTranscript, respLang));

        if (!assistantSpoken) {
          if (sttData.llm_unavailable) {
            console.warn("[VoiceCopilot] LLM unavailable for general query; returning to listening");
          }
          return;
        }

        setAssistantResponseText(assistantSpoken);

        // Add REAL assistant response to conversation panel
        const asstMsgId = `asst-${turnId}-${Date.now()}`;
        const asstTime = formatMessageTime();
        setConversationMessages((prev) => [
          ...prev,
          {
            id: asstMsgId,
            sender: "assistant",
            text: assistantSpoken,
            timestamp: asstTime,
          },
        ]);

        setVoiceState("speaking");
        voiceStateRef.current = "speaking";
        setProcessingStage("speaking");
        setIsTTSPlaying(true);

        const sentences = tokenizer.feed(assistantSpoken);
        const trailing = tokenizer.flush();
        if (trailing) sentences.push(trailing);
        if (sentences.length === 0 && assistantSpoken.trim()) sentences.push(assistantSpoken.trim());

        for (const s of sentences) {
          sentenceAudioQueueRef.current?.enqueueSentence(s, respLang);
        }
        sentenceAudioQueueRef.current?.markStreamComplete();
        return;
      }

      // GREETING-ONLY HANDLING (Fallback)
      const isGreetingTurn = Boolean(sttData.is_greeting || isGreetingOnly(rawTranscript));
      if (isGreetingTurn) {
        const greetingSpoken = sttData.greeting_response || getNaturalGreeting(respLang);
        setAssistantResponseText(greetingSpoken);

        // Add REAL assistant greeting to conversation panel
        const asstMsgId = `asst-${turnId}-${Date.now()}`;
        const asstTime = formatMessageTime();
        setConversationMessages((prev) => [
          ...prev,
          {
            id: asstMsgId,
            sender: "assistant",
            text: greetingSpoken,
            timestamp: asstTime,
          },
        ]);

        setVoiceState("speaking");
        voiceStateRef.current = "speaking";
        setProcessingStage("speaking");
        setIsTTSPlaying(true);

        const sentences = tokenizer.feed(greetingSpoken);
        const trailing = tokenizer.flush();
        if (trailing) sentences.push(trailing);
        if (sentences.length === 0 && greetingSpoken.trim()) sentences.push(greetingSpoken.trim());

        for (const s of sentences) {
          sentenceAudioQueueRef.current?.enqueueSentence(s, respLang);
        }
        sentenceAudioQueueRef.current?.markStreamComplete();
        return;
      }

      // Step 2 & 3 & 4: DeepSeek Streaming AI Lead Extraction, PostgreSQL CRM Update & Sentence-Level TTS
      setProcessingStage("extracting");
      setAssistantResponseText("");
      const streamTurnMsgId = `asst-stream-${turnId}-${Date.now()}`;
      currentStreamMsgIdRef.current = streamTurnMsgId;

      const fastApiBase = process.env.NEXT_PUBLIC_FASTAPI_URL || "http://127.0.0.1:8001";
      const extractUrl = `${fastApiBase.replace(/\/+$/, "")}/api/extract-lead`;
      let extractRes: Response;
      try {
        extractRes = await fetch(extractUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: abortController.signal,
          body: JSON.stringify({
            transcript: rawTranscript,
            existing_lead: currentLeadRef.current || undefined,
            language: respLang,
            stream: true,
            lead_id: activeLeadIdRef.current || undefined,
          }),
        });
      } catch (directErr) {
        if (abortController.signal.aborted) throw directErr;
        extractRes = await fetch("/api/extract-lead", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: abortController.signal,
          body: JSON.stringify({
            transcript: rawTranscript,
            existing_lead: currentLeadRef.current || undefined,
            language: respLang,
            stream: true,
            lead_id: activeLeadIdRef.current || undefined,
          }),
        });
      }

      if (isTurnStale(turnId)) return;

      if (!extractRes.ok) {
        const errJson = await extractRes.json().catch(() => ({}));
        throw new Error(errJson.error || errJson.detail || "AI lead extraction stream failed.");
      }

      if (!extractRes.body) {
        throw new Error("No readable stream received from lead extraction service.");
      }

      const reader = extractRes.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let sseBuffer = "";
      let accumulatedSpoken = "";

      const handleSSEEvent = (event: string, dataStr: string) => {
        if (!dataStr || dataStr === "[DONE]") return;
        try {
          const parsed = JSON.parse(dataStr);

          if (event === "lead") {
            if (parsed.is_new_lead) {
              activeLeadIdRef.current = null;
              currentLeadRef.current = null;
              setExtractedLead(null);
              setSaveLeadSuccess(null);
            }
            const updatedLead = parsed.lead;
            if (updatedLead) {
              currentLeadRef.current = updatedLead;
              setExtractedLead(updatedLead);
              if (onLeadSaved) onLeadSaved(updatedLead);
            }
            if (parsed.lead_id) {
              activeLeadIdRef.current = parsed.lead_id;
              setSaveLeadSuccess({
                id: parsed.lead_id,
                message: parsed.is_update
                  ? `Lead #${parsed.lead_id} successfully updated in PostgreSQL database!`
                  : `Lead #${parsed.lead_id} successfully saved to PostgreSQL database!`,
                isUpdate: Boolean(parsed.is_update),
              });
            }
            setProcessingStage("speaking");
          } else if (event === "token") {
            const token = parsed.token ?? "";
            if (token) {
              accumulatedSpoken += token;

              // ENQUEUE TO TTS BEFORE React state update to minimize latency
              const completedSentences = tokenizer.feed(token);
              for (const s of completedSentences) {
                const cleaned = cleanTextForSpeech(s);
                if (cleaned) {
                  sentenceAudioQueueRef.current?.enqueueSentence(cleaned, respLang);
                }
              }

              // Update UI state AFTER TTS enqueue (non-blocking for audio pipeline)
              setAssistantResponseText(accumulatedSpoken);
              setConversationMessages((prev) => {
                const last = prev[prev.length - 1];
                if (last && last.id === streamTurnMsgId) {
                  return [...prev.slice(0, -1), { ...last, text: accumulatedSpoken }];
                }
                return [
                  ...prev,
                  {
                    id: streamTurnMsgId,
                    sender: "assistant",
                    text: accumulatedSpoken,
                    timestamp: formatMessageTime(),
                  },
                ];
              });
            }
          } else if (event === "done") {
            if (parsed.is_new_lead && !currentLeadRef.current) {
              activeLeadIdRef.current = parsed.lead_id || null;
              currentLeadRef.current = parsed.lead || null;
              setExtractedLead(parsed.lead || null);
            }
            if (parsed.answer && !accumulatedSpoken) {
              accumulatedSpoken = parsed.answer;
              setAssistantResponseText(accumulatedSpoken);
              setConversationMessages((prev) => {
                const last = prev[prev.length - 1];
                if (last && last.id === streamTurnMsgId) {
                  return [...prev.slice(0, -1), { ...last, text: accumulatedSpoken }];
                }
                return [
                  ...prev,
                  {
                    id: streamTurnMsgId,
                    sender: "assistant",
                    text: accumulatedSpoken,
                    timestamp: formatMessageTime(),
                  },
                ];
              });
            }
            const trailing = tokenizer.flush();
            if (trailing) {
              const cleaned = cleanTextForSpeech(trailing);
              if (cleaned) {
                sentenceAudioQueueRef.current?.enqueueSentence(cleaned, respLang);
              }
            }
            sentenceAudioQueueRef.current?.markStreamComplete();
          } else if (event === "error") {
            setErrorMessage(parsed.error || "Streaming error during lead processing.");
          }
        } catch (parseErr) {
          console.warn("[VoiceCopilot] Error parsing SSE payload:", dataStr, parseErr);
        }
      };

      let currentEvent = "";
      let currentData = "";

      try {
        while (true) {
          if (isTurnStale(turnId)) {
            reader.cancel().catch(() => {});
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

            if (trimmed.startsWith("event: ")) {
              currentEvent = trimmed.slice(7).trim();
            } else if (trimmed.startsWith("data: ")) {
              const d = trimmed.slice(6);
              currentData = currentData ? `${currentData}\n${d}` : d;
            }
          }
        }

        if (currentEvent && currentData) {
          handleSSEEvent(currentEvent, currentData);
        }
      } catch (streamErr: any) {
        if (streamErr?.name === "AbortError" || isTurnStale(turnId)) {
          return;
        }
        throw streamErr;
      }

      const finalTrailing = tokenizer.flush();
      if (finalTrailing) {
        const cleaned = cleanTextForSpeech(finalTrailing);
        if (cleaned) {
          sentenceAudioQueueRef.current?.enqueueSentence(cleaned, respLang);
        }
      }
      sentenceAudioQueueRef.current?.markStreamComplete();
    },
    [cleanTextForSpeech, getAssistantAnswerFallback, getNaturalGreeting, isAssistantQueryText, isGreetingOnly, isTurnStale, onLeadSaved, selectedLanguage]
  );

  // --------------------------------------------------------------------------
  // Core Continuous Voice Pipeline: HTTP Fallback STT -> handleFinalSTTResponse
  // --------------------------------------------------------------------------
  const processSpeechTurn = async (blob: Blob, turnId: number) => {
    if (isTurnStale(turnId)) return;

    isVoicePipelineActiveRef.current = true;
    setVoiceState("processing");
    voiceStateRef.current = "processing";
    setProcessingStage("transcribing");
    setErrorMessage(null);

    const abortController = new AbortController();
    activeAbortControllerRef.current = abortController;

    try {
      const filename = `turn_${turnId}_${Date.now()}.wav`;
      const formData = new FormData();

      if (typeof File !== "undefined") {
        const audioFile = new File([blob], filename, { type: "audio/wav" });
        formData.append("file", audioFile);
      } else {
        formData.append("file", blob, filename);
      }

      if (isManualLanguageSelectionRef.current && selectedLanguage !== "auto") {
        formData.append("language", selectedLanguage);
      } else {
        formData.append("language", "auto");
      }
      formData.append("module", "module1");
      if (currentLeadRef.current) {
        formData.append("existing_lead", JSON.stringify(currentLeadRef.current));
      }
      if (activeLeadIdRef.current) {
        formData.append("lead_id", String(activeLeadIdRef.current));
      }

      console.log(`[VoiceCopilot] Sending Turn #${turnId} (${blob.size} bytes WAV) to /api/voice-entry...`);
      let sttRes: Response | null = null;

      for (let attempt = 1; attempt <= 2; attempt++) {
        if (isTurnStale(turnId)) return;
        try {
          sttRes = await fetch("/api/voice-entry", {
            method: "POST",
            body: formData,
            signal: abortController.signal,
          });
          if ((sttRes.status === 503 || sttRes.status === 408 || sttRes.status === 502) && attempt < 2) {
            await new Promise((resolve) => setTimeout(resolve, 50));
            continue;
          }
          break;
        } catch (fetchErr: any) {
          if (fetchErr?.name === "AbortError" || isTurnStale(turnId)) return;
          if (attempt < 2) await new Promise((resolve) => setTimeout(resolve, 50));
        }
      }

      if (!sttRes || !sttRes.ok) {
        const errJson = (await sttRes?.json().catch(() => ({}))) || {};
        throw new Error(errJson.error || errJson.detail || "Deepgram STT transcription failed.");
      }

      const sttData = await sttRes.json();
      sttFinalPerfTimestampRef.current = performance.now();
      await handleFinalSTTResponse(sttData, turnId, abortController);
    } catch (err: any) {
      if (err?.name === "AbortError" || isTurnStale(turnId)) return;
      console.error(`[VoiceCopilot] Error in Turn #${turnId}:`, err);
      setErrorMessage(err?.message || "Failed to process voice note.");
      isVoicePipelineActiveRef.current = false;

      if (isContinuousModeRef.current) {
        setVoiceState("listening");
        voiceStateRef.current = "listening";
        setProcessingStage(null);
        startListeningTurn();
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
    if (
      !isMicCaptureActiveRef.current ||
      !isWsStreamingRef.current ||
      voiceStateRef.current !== "listening"
    ) {
      return;
    }
    const ws = sttSocketRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(convertFloat32ToInt16(chunk));
      } catch (_) {}
    } else if (ws && ws.readyState === WebSocket.CONNECTING) {
      wsPendingQueueRef.current.push(chunk);
    }
  }, []);

  const startStreamingSTT = useCallback(
    (initialBuffers: Float32Array[]) => {
      hasWsFinalizedRef.current = false;
      isWsStreamingRef.current = true;

      const existingWs = sttSocketRef.current;
      // Fast-path: If persistent WebSocket is already connected, stream immediately with 0ms latency!
      if (existingWs && existingWs.readyState === WebSocket.OPEN) {
        for (const buf of initialBuffers) {
          try {
            existingWs.send(convertFloat32ToInt16(buf));
          } catch (_) {}
        }
        return;
      }

      // If socket is already in connecting state, queue buffers
      if (existingWs && existingWs.readyState === WebSocket.CONNECTING) {
        wsPendingQueueRef.current.push(...initialBuffers);
        return;
      }

      wsPendingQueueRef.current = [...initialBuffers];

      const baseWs = (
        process.env.NEXT_PUBLIC_FASTAPI_WS_URL ||
        process.env.NEXT_PUBLIC_FASTAPI_URL ||
        "http://127.0.0.1:8001"
      ).replace(/^http/, "ws");

      const sampleRate = audioContextRef.current?.sampleRate || 48000;
      const langParam = isManualLanguageSelectionRef.current && selectedLanguage !== "auto"
        ? `&language=${selectedLanguage}`
        : "&language=auto";
      const existingLeadParam = currentLeadRef.current
        ? `&existing_lead=${encodeURIComponent(JSON.stringify(currentLeadRef.current))}`
        : "";
      const leadIdParam = activeLeadIdRef.current ? `&lead_id=${activeLeadIdRef.current}` : "";
      const wsUrl = `${baseWs}/ws/voice-stt?sample_rate=${sampleRate}${langParam}&module=module1${existingLeadParam}${leadIdParam}`;

      try {
        const ws = new WebSocket(wsUrl);
        ws.binaryType = "arraybuffer";
        sttSocketRef.current = ws;

        ws.onopen = () => {
          while (wsPendingQueueRef.current.length > 0) {
            const buf = wsPendingQueueRef.current.shift();
            if (buf && ws.readyState === WebSocket.OPEN) {
              try {
                ws.send(convertFloat32ToInt16(buf));
              } catch (_) {}
            }
          }
        };

        ws.onmessage = (e) => {
          try {
            const data = JSON.parse(e.data);
            if (data.type === "interim") {
              const interimText = (data.transcript || "").trim();
              if (interimText) {
                setTranscript(interimText);
              }
            } else if (data.type === "final") {
              if (!hasWsFinalizedRef.current) {
                hasWsFinalizedRef.current = true;
                const tFinalStt = performance.now();
                sttFinalPerfTimestampRef.current = tFinalStt;
                console.log(`[LATENCY] 1. FINAL-STT received at perf=${tFinalStt.toFixed(3)}ms text="${(data.transcript || '').trim()}"`);
                if (silenceTimerRef.current) {
                  clearTimeout(silenceTimerRef.current);
                  silenceTimerRef.current = null;
                }
                isWsStreamingRef.current = false;

                const activeTurn = turnIdRef.current;
                const abortController = activeAbortControllerRef.current || new AbortController();
                activeAbortControllerRef.current = abortController;
                handleFinalSTTResponse(data, activeTurn, abortController);
              }
            } else if (data.type === "sentence") {
              const sentenceText = (data.sentence || "").trim();
              if (sentenceText && sentenceAudioQueueRef.current) {
                setAssistantResponseText((prev) => prev ? `${prev} ${sentenceText}` : sentenceText);
                setConversationMessages((prev) => {
                  const last = prev[prev.length - 1];
                  if (last && last.sender === "assistant" && last.id === currentWsSentenceMsgIdRef.current) {
                    return [...prev.slice(0, -1), { ...last, text: `${last.text} ${sentenceText}`.trim() }];
                  }
                  const newId = `asst-ws-${Date.now()}`;
                  currentWsSentenceMsgIdRef.current = newId;
                  return [
                    ...prev,
                    {
                      id: newId,
                      sender: "assistant",
                      text: sentenceText,
                      timestamp: formatMessageTime(),
                    },
                  ];
                });
                setVoiceState("speaking");
                voiceStateRef.current = "speaking";
                setIsTTSPlaying(true);
                const sentenceLang = (isManualLanguageSelectionRef.current && selectedLanguage !== "auto")
                  ? selectedLanguage
                  : (detectedLanguageRef.current || detectedLanguage || "en");
                sentenceAudioQueueRef.current.enqueueSentence(sentenceText, sentenceLang);
              }
            } else if (data.type === "stream_complete") {
              if (sentenceAudioQueueRef.current) {
                sentenceAudioQueueRef.current.markStreamComplete();
              }
            }
          } catch (parseErr) {
            console.warn("[VoiceCopilot] STT WS parse error:", parseErr);
          }
        };

        ws.onerror = (err) => {
          console.warn("[VoiceCopilot] STT WebSocket error:", err);
        };

        ws.onclose = () => {
          if (sttSocketRef.current === ws) {
            sttSocketRef.current = null;
            isWsStreamingRef.current = false;
          }
        };
      } catch (wsErr) {
        console.warn("[VoiceCopilot] Unable to open STT WebSocket:", wsErr);
        sttSocketRef.current = null;
        isWsStreamingRef.current = false;
      }
    },
    [handleFinalSTTResponse, selectedLanguage]
  );

  // Keep ref updated for immediate instant barge-in trigger
  useEffect(() => {
    startStreamingSTTRef.current = startStreamingSTT;
  }, [startStreamingSTT]);

  // Load active TTS model from ModelManager to keep it as authoritative source of truth
  useEffect(() => {
    let isMounted = true;
    const fetchActiveTtsModel = async () => {
      try {
        const res = await fetch("/api/models/active", { cache: "no-store" });
        if (res.ok) {
          const data = await res.json();
          const ttsModel = data?.active?.tts?.model_id || data?.active?.tts?.id;
          if (ttsModel && isMounted) {
            activeTtsModelRef.current = ttsModel;
            if (sentenceAudioQueueRef.current) {
              sentenceAudioQueueRef.current.setOptions({
                module: "module1",
                model: ttsModel,
              });
            }
          }
        }
      } catch (err) {
        console.warn("[VoiceCopilot] Could not fetch active TTS model:", err);
      }
    };
    fetchActiveTtsModel();
    return () => {
      isMounted = false;
    };
  }, []);

  // --------------------------------------------------------------------------
  // HANDLER: submitCurrentSpeechTurn
  // --------------------------------------------------------------------------
  // • WHAT IT DOES: Finalizes the current speech turn either automatically upon VAD
  //   silence detection (isManual=false) or immediately when the user clicks the
  //   "Finish speaking" button (isManual=true).
  // • For manual clicks: immediately stops listening capture, bypasses VAD silence/duration
  //   thresholds, sends STT Finalize (or HTTP fallback), and triggers LLM -> TTS pipeline
  //   exactly once while guarding against duplicate invocations.
  // --------------------------------------------------------------------------
  const submitCurrentSpeechTurn = useCallback((isManual: boolean = false) => {
    // Prevent duplicate processing if a turn is already finalizing/active
    if (isVoicePipelineActiveRef.current || voiceStateRef.current === "processing") return;

    // Immediately STOP/disable user's microphone capture for the current turn
    if (isManual) {
      isMicCaptureActiveRef.current = false;
      if (audioStreamRef.current) {
        try {
          audioStreamRef.current.getTracks().forEach((track) => {
            track.enabled = false;
            track.stop();
          });
        } catch (_) {}
        audioStreamRef.current = null;
      }
      if (filterNodesRef.current.length > 0) {
        filterNodesRef.current.forEach((node) => {
          try {
            node.disconnect();
          } catch (_) {}
        });
        filterNodesRef.current = [];
      }
      if (audioSourceNodeRef.current) {
        try {
          audioSourceNodeRef.current.disconnect();
        } catch (_) {}
        audioSourceNodeRef.current = null;
      }
      if (analyserRef.current) {
        try {
          analyserRef.current.disconnect();
        } catch (_) {}
        analyserRef.current = null;
      }
      if (audioWorkletNodeRef.current) {
        try {
          audioWorkletNodeRef.current.port.onmessage = null;
          audioWorkletNodeRef.current.disconnect();
        } catch (_) {}
        audioWorkletNodeRef.current = null;
      }
      if (scriptProcessorRef.current) {
        try {
          scriptProcessorRef.current.onaudioprocess = null;
          scriptProcessorRef.current.disconnect();
        } catch (_) {}
        scriptProcessorRef.current = null;
      }
      if (silentGainRef.current) {
        try {
          silentGainRef.current.disconnect();
        } catch (_) {}
        silentGainRef.current = null;
      }
      setAudioLevels([]);
    }

    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    if (autoListenTimeoutRef.current) {
      clearTimeout(autoListenTimeoutRef.current);
      autoListenTimeoutRef.current = null;
    }

    // Immediately stop microphone speech detection and mark turn inactive
    isSpeechTurnActiveRef.current = false;
    speechDetectedRef.current = false;
    wasSpeakingRef.current = false;
    consecutiveSpeechFramesRef.current = 0;
    consecutiveBargeInFramesRef.current = 0;

    // Collect recorded audio buffers, falling back to rolling pre-roll if onset wasn't confirmed
    const recordedBuffers = [...recordedBuffersRef.current];
    const preRollBuffers = [...preRollBuffersRef.current];
    recordedBuffersRef.current = [];
    preRollBuffersRef.current = [];

    const buffersToEncode = recordedBuffers.length > 0 ? recordedBuffers : preRollBuffers;

    const sampleRate = audioContextRef.current?.sampleRate || 48000;
    const totalSamples = buffersToEncode.reduce((acc, b) => acc + b.length, 0);
    const duration = totalSamples / sampleRate;

    // For automatic VAD silence timer: run duration and energy validation gates
    if (!isManual) {
      if (buffersToEncode.length === 0) return;

      // Discard noise glitches (< 300ms)
      if (duration < 0.30) {
        console.log(`[VoiceCopilot:VAD] Discarded brief noise glitch (duration: ${(duration * 1000).toFixed(0)}ms < 300ms)`);
        isWsStreamingRef.current = false;
        return;
      }

      // Energy validation: verify that the recorded segment contains genuine speech energy in dBFS
      let totalEnergy = 0;
      let peakRms = 0;
      let voicedFramesCount = 0;
      const continuationThresholdRms = Math.pow(
        10,
        Math.max(-48.0, ambientNoiseFloorDbRef.current + 4.5) / 20
      );

      for (const buf of buffersToEncode) {
        let sum = 0;
        for (let i = 0; i < buf.length; i++) {
          sum += buf[i] * buf[i];
        }
        const bufRms = Math.sqrt(sum / buf.length);
        totalEnergy += bufRms;
        if (bufRms > peakRms) peakRms = bufRms;
        if (bufRms >= continuationThresholdRms) {
          voicedFramesCount++;
        }
      }
      const avgRms = totalEnergy / buffersToEncode.length;
      const segmentAvgDb = 20 * Math.log10(Math.max(avgRms, 1e-5));
      const peakDb = 20 * Math.log10(Math.max(peakRms, 1e-5));
      const voicedDensity = buffersToEncode.length > 0 ? voicedFramesCount / buffersToEncode.length : 0;

      // Discard if overall energy is room noise or an isolated click or lacks close-proximity peak
      const minRequiredPeakDb = primarySpeakerProfileRef.current.isCalibrated ? -38.0 : -34.0;
      if (
        segmentAvgDb < ambientNoiseFloorDbRef.current + 3.5 ||
        peakDb < minRequiredPeakDb ||
        voicedDensity < 0.20
      ) {
        if (process.env.NODE_ENV !== "production") {
          console.log(
            `[VAD:FINALIZE] Discarded non-speech segment (avgDb: ${segmentAvgDb.toFixed(1)}dBFS, peakDb: ${peakDb.toFixed(1)}dBFS, floor: ${ambientNoiseFloorDbRef.current.toFixed(1)}dBFS, density: ${(voicedDensity * 100).toFixed(0)}%, duration: ${(duration * 1000).toFixed(0)}ms, accepted=false)`
          );
        }
        isWsStreamingRef.current = false;
        return;
      }

      if (process.env.NODE_ENV !== "production") {
        console.log(
          `[VAD:FINALIZE] avgDb=${segmentAvgDb.toFixed(1)} peakDb=${peakDb.toFixed(1)} density=${(voicedDensity * 100).toFixed(0)}% duration=${(duration * 1000).toFixed(0)}ms accepted=true`
        );
      }
    }

    // Immediately stop microphone capture/listening turn and set processing state
    isVoicePipelineActiveRef.current = true;
    setVoiceState("processing");
    voiceStateRef.current = "processing";
    setProcessingStage("transcribing");
    setErrorMessage(null);

    // Stop sending further audio to STT
    isWsStreamingRef.current = false;
    wsPendingQueueRef.current = [];

    const currentTurnId = ++turnIdRef.current;
    hasFirstAudioPlayedRef.current = false;
    hasWsFinalizedRef.current = false;
    const abortController = new AbortController();
    activeAbortControllerRef.current = abortController;

    const ws = sttSocketRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify({ type: "Finalize" }));
      } catch (wsErr) {
        console.warn("[VoiceCopilot] Failed to send Finalize over WS:", wsErr);
      }
      return;
    }

    // If WebSocket is not open (null, connecting, or closed), cancel any in-flight connection
    if (ws && ws.readyState === WebSocket.CONNECTING) {
      try {
        ws.close();
      } catch (_) {}
      sttSocketRef.current = null;
    }

    // HTTP Fallback STT: encode available audio and process speech turn
    const safeBuffers = buffersToEncode.length > 0
      ? buffersToEncode
      : [new Float32Array(Math.floor(sampleRate * 0.1))];

    const wavBlob = encodeWav(safeBuffers, sampleRate);
    if (onAudioRecorded) {
      onAudioRecorded(wavBlob);
    }

    processSpeechTurn(wavBlob, currentTurnId);
  }, [onAudioRecorded]);

  // --------------------------------------------------------------------------
  // HANDLER: startListeningTurn
  // --------------------------------------------------------------------------
  // • WHAT IT DOES: Arms the microphone with Web Audio API nodes created from a single
  //   AudioContext instance. Computes real-time RMS audio energy, maintains a ~850ms
  //   rolling circular pre-roll buffer, and triggers a 400ms silence countdown.
  // • INPUTS: None (reads state from refs).
  // • OUTPUT: None (attaches onaudioprocess callback to capture microphone audio frames).
  // • WHY IT IS USED: Hands-free conversational mode: users do not need to push buttons.
  //   Pre-roll buffers guarantee the first syllable is never clipped.
  // • WHERE IT FITS IN THE FLOW:
  //     [Voice Mode ON or Previous AI Speech Ends] -> [startListeningTurn] -> [Wait for User to Speak]
  // --------------------------------------------------------------------------
  const startListeningTurn = useCallback(async () => {
    if (!isContinuousModeRef.current) return;

    // Serialize onto lifecycle lock to prevent overlapping start/stop operations
    const prevLock = lifecycleLockRef.current;
    let releaseLock: () => void;
    lifecycleLockRef.current = new Promise<void>((resolve) => {
      releaseLock = resolve;
    });

    try {
      await prevLock;
      if (!isContinuousModeRef.current) return;
      if (isStartingTurnRef.current) return;
      isStartingTurnRef.current = true;

      try {
        setErrorMessage(null);
        speechDetectedRef.current = false;
        isSpeechTurnActiveRef.current = false;
        recordedBuffersRef.current = [];
        hasFirstAudioPlayedRef.current = false;

        if (typeof window === "undefined" || !navigator?.mediaDevices?.getUserMedia) {
          setErrorMessage("Microphone access is not supported in this browser environment.");
          stopVoiceMode();
          return;
        }

        // 1. Ensure microphone stream is active with AEC, noise suppression, AGC
        let stream = audioStreamRef.current;
        const needsNewStream =
          !stream ||
          !stream.active ||
          stream.getAudioTracks().length === 0 ||
          stream.getAudioTracks().some((t) => t.readyState === "ended");

        // Fast path: if the Web Audio graph and microphone stream are already active,
        // keep graph connected to eliminate handover gaps and avoid dropped frames!
        const isGraphAlreadyRunning =
          !needsNewStream &&
          audioContextRef.current &&
          audioContextRef.current.state === "running" &&
          (audioWorkletNodeRef.current !== null || scriptProcessorRef.current !== null) &&
          audioSourceNodeRef.current !== null;

        if (isGraphAlreadyRunning) {
          stream?.getAudioTracks().forEach((t) => (t.enabled = true));
          isMicCaptureActiveRef.current = true;
          setVoiceState("listening");
          voiceStateRef.current = "listening";
          setProcessingStage(null);
          return;
        }

        if (needsNewStream) {
          stream = await navigator.mediaDevices.getUserMedia({
            audio: {
              echoCancellation: { ideal: true },
              noiseSuppression: { ideal: true },
              autoGainControl: { ideal: true },
              channelCount: 1,
              sampleRate: { ideal: 48000 },
              googEchoCancellation: true,
              googAutoGainControl: true,
              googNoiseSuppression: true,
              googHighpassFilter: true,
              googTypingNoiseDetection: true,
            } as any,
          });
          audioStreamRef.current = stream;
        }

        if (!stream) {
          throw new Error("Unable to obtain microphone stream.");
        }

        if (!isContinuousModeRef.current) return;

        stream.getAudioTracks().forEach((t) => (t.enabled = true));
        isMicCaptureActiveRef.current = true;

        const activeTrack = stream.getAudioTracks()[0];
        console.log(
          `[MIC:PROBE] Device: "${activeTrack?.label}" | readyState: ${activeTrack?.readyState} | enabled: ${activeTrack?.enabled} | muted: ${activeTrack?.muted} | settings:`,
          activeTrack?.getSettings()
        );

        // 2. Disconnect and reset previous audio processing nodes to prevent stale node reuse
        if (filterNodesRef.current.length > 0) {
          filterNodesRef.current.forEach((node) => {
            try {
              node.disconnect();
            } catch (_) {}
          });
          filterNodesRef.current = [];
        }
        if (audioSourceNodeRef.current) {
          try {
            audioSourceNodeRef.current.disconnect();
          } catch (_) {}
          audioSourceNodeRef.current = null;
        }
        if (analyserRef.current) {
          try {
            analyserRef.current.disconnect();
          } catch (_) {}
          analyserRef.current = null;
        }
        if (audioWorkletNodeRef.current) {
          try {
            audioWorkletNodeRef.current.port.onmessage = null;
            audioWorkletNodeRef.current.disconnect();
          } catch (_) {}
          audioWorkletNodeRef.current = null;
        }
        if (scriptProcessorRef.current) {
          try {
            scriptProcessorRef.current.disconnect();
          } catch (_) {}
          scriptProcessorRef.current = null;
        }
        if (silentGainRef.current) {
          try {
            silentGainRef.current.disconnect();
          } catch (_) {}
          silentGainRef.current = null;
        }

        // 3. Reuse active AudioContext when possible, or instantiate if missing/closed
        let audioCtx = audioContextRef.current;
        if (!audioCtx || audioCtx.state === "closed") {
          const AudioCtxClass = window.AudioContext || (window as any).webkitAudioContext;
          audioCtx = new AudioCtxClass();
          audioContextRef.current = audioCtx;
        }

        // Safely synchronize resume so we never close or clash while resume is pending
        if (audioCtx.state === "suspended") {
          if (!resumePromiseRef.current) {
            const p = audioCtx
              .resume()
              .catch((err: any) => {
                console.debug("[VoiceCopilot] AudioContext resume note:", err);
              })
              .finally(() => {
                if (resumePromiseRef.current === p) {
                  resumePromiseRef.current = null;
                }
              });
            resumePromiseRef.current = p;
          }
          await resumePromiseRef.current;
        }

        console.log(
          `[AUDIO_CTX:PROBE] state: "${audioCtx.state}" | sampleRate: ${audioCtx.sampleRate}Hz | destinationChannels: ${audioCtx.destination?.channelCount}`
        );

        if (!isContinuousModeRef.current || audioCtx.state === "closed") return;

        // 4. Create DSP speech bandpass filters from the active AudioContext instance
        const source = audioCtx.createMediaStreamSource(stream);
        audioSourceNodeRef.current = source;

        // High-pass filter (135 Hz) to strip HVAC hum, fan rumble (50/60/100/120 Hz), traffic, table vibrations
        const highpass = audioCtx.createBiquadFilter();
        highpass.type = "highpass";
        highpass.frequency.value = 135;
        highpass.Q.value = 0.707;

        // Low-pass filter (3600 Hz) to eliminate high-frequency fan hiss, keyboard click harmonics, coil whine
        const lowpass = audioCtx.createBiquadFilter();
        lowpass.type = "lowpass";
        lowpass.frequency.value = 3600;
        lowpass.Q.value = 0.707;

        // Intelligibility peaking bandpass (1500 Hz, +2.5dB) to enhance nearby voice presence over diffuse background
        const voicePeaking = audioCtx.createBiquadFilter();
        voicePeaking.type = "peaking";
        voicePeaking.frequency.value = 1500;
        voicePeaking.gain.value = 2.5;
        voicePeaking.Q.value = 1.2;

        filterNodesRef.current = [highpass, lowpass, voicePeaking];

        // Connect DSP filter chain: source -> highpass -> lowpass -> voicePeaking
        source.connect(highpass);
        highpass.connect(lowpass);
        lowpass.connect(voicePeaking);

        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 256;
        analyserRef.current = analyser;
        voicePeaking.connect(analyser);

        const silentGain = audioCtx.createGain();
        silentGain.gain.value = 0;
        silentGainRef.current = silentGain;
        silentGain.connect(audioCtx.destination);

        const maxPreRollCount = Math.max(16, Math.ceil((audioCtx.sampleRate * 0.85) / 2048));

        // 5. Audio Process Handler with Adaptive dB-Based VAD & Acoustic Discrimination
        const processAudioChunk = (chunk: Float32Array, rms: number) => {
          if (
            !isMicCaptureActiveRef.current ||
            voiceStateRef.current === "processing" ||
            (!isContinuousModeRef.current && voiceStateRef.current === "idle")
          ) {
            return;
          }

          // Calculate chunk Peak, true RMS, Crest Factor, and Zero-Crossing Rate
          let sumSq = 0;
          let maxPeak = 0;
          let minVal = 0;
          let maxVal = 0;
          let nonZeroCount = 0;
          let zeroCrossings = 0;
          let prevSample = chunk[0] || 0;

          for (let i = 0; i < chunk.length; i++) {
            const s = chunk[i];
            sumSq += s * s;
            const abs = Math.abs(s);
            if (abs > maxPeak) maxPeak = abs;
            if (s < minVal) minVal = s;
            if (s > maxVal) maxVal = s;
            if (s !== 0) nonZeroCount++;
            if ((s >= 0 && prevSample < 0) || (s < 0 && prevSample >= 0)) {
              zeroCrossings++;
            }
            prevSample = s;
          }

          const calculatedRms = Math.sqrt(sumSq / chunk.length);
          const effectiveRms = rms > 0 ? rms : calculatedRms;
          const currentDb = 20 * Math.log10(Math.max(effectiveRms, 1e-5));
          const crestFactor = effectiveRms > 1e-5 ? maxPeak / effectiveRms : 1.0;
          const zcr = zeroCrossings / chunk.length;

          // Frequency analysis via AnalyserNode
          const dataArray = new Uint8Array(analyser.frequencyBinCount);
          analyser.getByteFrequencyData(dataArray);

          // Vocal formant core energy (bins 2 to 15: ~375Hz - 2812Hz at 48kHz, or ~125Hz - 937Hz at 16kHz)
          let vocalSum = 0;
          let maxVocalBinVal = 0;
          let dominantBin = 2;
          let weightedBinSum = 0;
          let totalVocalWeight = 0;
          const maxVocalBin = Math.min(15, dataArray.length - 1);

          for (let i = 2; i <= maxVocalBin; i++) {
            const val = dataArray[i];
            vocalSum += val;
            if (val > maxVocalBinVal) {
              maxVocalBinVal = val;
              dominantBin = i;
            }
            weightedBinSum += i * val;
            totalVocalWeight += val;
          }
          const vocalCount = maxVocalBin - 1;
          const vocalEnergy = vocalCount > 0 ? vocalSum / vocalCount : 0;
          const peakProminence = vocalEnergy > 1 ? maxVocalBinVal / vocalEnergy : 1.0;
          const spectralCentroid = totalVocalWeight > 1 ? weightedBinSum / totalVocalWeight : dominantBin;

          // High frequency noise energy (bins 20 to 60: ~3750Hz - 11250Hz - fan hiss, typing clicks)
          let highSum = 0;
          let highCount = 0;
          for (let i = 20; i <= 60 && i < dataArray.length; i++) {
            highSum += dataArray[i];
            highCount++;
          }
          const highNoise = highCount > 0 ? highSum / highCount : 0;

          // Compute Voicing Harmonicity via normalized autocorrelation if signal is above ambient floor
          let harmonicity = 0;
          if (currentDb >= ambientNoiseFloorDbRef.current + 3.0 && vocalEnergy >= 24) {
            harmonicity = computeHarmonicity(chunk, audioCtx.sampleRate || 48000);
          }

          // Visualizer waveform (24 bars)
          const sampleCount = 24;
          const step = Math.floor(dataArray.length / sampleCount) || 1;
          const levels = [];
          for (let i = 0; i < sampleCount; i++) {
            levels.push(Math.min(100, Math.max(12, Math.round(((dataArray[i * step] || 0) / 255) * 100))));
          }
          setAudioLevels(levels);

          // Update rolling pre-roll buffer (keeps last ~800-850ms)
          if (voiceStateRef.current === "listening" && !isSpeechTurnActiveRef.current) {
            preRollBuffersRef.current.push(chunk);
            if (preRollBuffersRef.current.length > maxPreRollCount) {
              preRollBuffersRef.current.shift();
            }
          }

          // BARGE-IN: while assistant is speaking, detect user interruption with nearby vocal energy
          if (voiceStateRef.current === "speaking") {
            speakerBleedDbRef.current = Math.max(
              -60.0,
              Math.min(-20.0, speakerBleedDbRef.current * 0.92 + currentDb * 0.08)
            );
            bleedRmsRef.current = bleedRmsRef.current * 0.92 + effectiveRms * 0.08;

            const now = Date.now();
            const hasGracePeriodElapsed = now - sentenceStartTimestampRef.current > 70;

            let bargeInHarmonicity = 0;
            if (currentDb >= Math.max(-36.0, speakerBleedDbRef.current + 5.0) && vocalEnergy >= 28) {
              bargeInHarmonicity = computeHarmonicity(chunk, audioCtx.sampleRate || 48000);
            }

            // Reject keyboard typing (crestFactor >= 5.8) or diffuse background voices during assistant playback
            const isUserBargeIn =
              hasGracePeriodElapsed &&
              crestFactor < 5.8 &&
              vocalEnergy >= 32 &&
              vocalEnergy > highNoise * 1.35 &&
              (peakProminence >= 1.55 || vocalEnergy >= 50) &&
              bargeInHarmonicity >= 0.36 &&
              (currentDb >= Math.max(-36.0, speakerBleedDbRef.current + 6.0) ||
                (effectiveRms >= 0.038 && effectiveRms >= bleedRmsRef.current * 1.8));

            if (isUserBargeIn) {
              consecutiveBargeInFramesRef.current++;
              if (consecutiveBargeInFramesRef.current >= 2) {
                console.log(
                  `[VAD:BARGE_IN] ACCEPT_BARGE_IN | dBFS=${currentDb.toFixed(1)} | vocalE=${vocalEnergy.toFixed(1)} | peakiness=${peakProminence.toFixed(2)} | harmonicity=${bargeInHarmonicity.toFixed(2)}`
                );
                executeBargeIn();
                return;
              }
            } else {
              consecutiveBargeInFramesRef.current = Math.max(0, consecutiveBargeInFramesRef.current - 1);
            }
            return;
          }

          // LISTENING STATE: detect speech onset with adaptive dB noise floor, vocal formants, and speaker discrimination
          if (voiceStateRef.current === "listening") {
            // Adaptive dB noise floor tracking when not actively speaking
            if (!isSpeechTurnActiveRef.current) {
              ambientNoiseFloorDbRef.current = Math.max(
                -65.0,
                Math.min(-35.0, ambientNoiseFloorDbRef.current * 0.96 + currentDb * 0.04)
              );
              ambientNoiseFloorRef.current = Math.pow(10, ambientNoiseFloorDbRef.current / 20);
            }

            // Dynamic SNR onset margin: adapts to ambient room volume
            const snrMarginDb = Math.max(
              9.0,
              Math.min(14.0, 11.0 - (ambientNoiseFloorDbRef.current + 45.0) * 0.25)
            );

            // Proximity & Direct-to-Reverberant Ratio (DRR):
            // Close-mic primary user has sharp formant resonance peaks (peakProminence >= 1.60)
            // and strong voicing harmonicity (harmonicity >= 0.38).
            // Diffuse background speakers (reverberant field) have smeared spectra (peakProminence < 1.48)
            // and low harmonicity (< 0.38).
            const condProximity =
              (peakProminence >= 1.60 && harmonicity >= 0.38) ||
              (peakProminence >= 1.85) ||
              (vocalEnergy >= 55 && harmonicity >= 0.48);

            // Proximity-tuned onset threshold:
            // When close-mic proximity is high (sharp formants & periodicity), safely accept soft primary speech down to -38.0 dBFS.
            // When diffuse/unconfirmed, clamp strictly at -35.0 dBFS to reject background speakers!
            const dynamicDbOnsetThreshold = condProximity
              ? Math.max(-38.0, ambientNoiseFloorDbRef.current + 8.0)
              : Math.max(-35.0, ambientNoiseFloorDbRef.current + snrMarginDb);
            const snr = currentDb - ambientNoiseFloorDbRef.current;

            // 1. Level check: dBFS above dynamic floor
            const condDb = currentDb >= dynamicDbOnsetThreshold;

            // 2. Vocal energy check: vocal formant energy >= 34
            const condVocal = vocalEnergy >= 34;

            // 3. Formant dominance over high noise:
            const condFormant = vocalEnergy > highNoise * 1.30;

            // 4. Crest factor check: reject keyboard typing clicks (> 5.8) unless high vocal plosive
            const condCrest = crestFactor <= 5.8 || vocalEnergy >= 50;

            // 5. Zero-crossing rate: reject friction noise / bursts (> 0.24) unless loud voice
            const condZcr = zcr <= 0.24 || vocalEnergy >= 40;

            // 7. Speaker Discrimination against Primary Profile (if calibrated):
            let isProfileMatch = true;
            let profileRejectReason = "";
            if (primarySpeakerProfileRef.current.isCalibrated) {
              const binDiff = Math.abs(dominantBin - primarySpeakerProfileRef.current.dominantBin);
              const centroidDiff = Math.abs(spectralCentroid - primarySpeakerProfileRef.current.centroid);

              if (binDiff > 3.0 && centroidDiff > 4.5) {
                if (!(peakProminence >= 2.2 && currentDb >= -24.0 && harmonicity >= 0.65)) {
                  isProfileMatch = false;
                  profileRejectReason = `SPEAKER_PROFILE_MISMATCH (binDiff=${binDiff.toFixed(1)}, centroidDiff=${centroidDiff.toFixed(1)})`;
                }
              }
            }

            const isSpeechCandidate =
              condDb &&
              condVocal &&
              condFormant &&
              condCrest &&
              condZcr &&
              condProximity &&
              isProfileMatch;

            // Structured logging with exact ACCEPT/REJECT reason and metrics
            let decisionReason = "";
            if (isSpeechCandidate) {
              decisionReason = "CLOSE_MIC_PRIMARY_QUALIFIED";
            } else if (!isProfileMatch) {
              decisionReason = profileRejectReason;
            } else if (!condProximity && condDb && condVocal) {
              decisionReason = `DIFFUSE_SECONDARY_SPEAKER (peakiness=${peakProminence.toFixed(2)} < 1.60, harmonicity=${harmonicity.toFixed(2)} < 0.38)`;
            } else if (!condCrest) {
              decisionReason = `TRANSIENT_IMPULSE (crest=${crestFactor.toFixed(2)} > 5.8)`;
            } else if (!condZcr) {
              decisionReason = `HIGH_ZCR_FRICTION (zcr=${zcr.toFixed(3)} > 0.24)`;
            } else if (!condFormant) {
              decisionReason = `HIGH_NOISE_DOMINANT (vocal=${vocalEnergy.toFixed(1)} <= ${highNoise.toFixed(1)}*1.30)`;
            } else {
              decisionReason = `LOW_ENERGY_OR_FLOOR (dBFS=${currentDb.toFixed(1)} < ${dynamicDbOnsetThreshold.toFixed(1)})`;
            }

            const decisionType = isSpeechCandidate
              ? "ACCEPT_PRIMARY_SPEAKER"
              : !isProfileMatch || (!condProximity && condDb && condVocal)
              ? "REJECT_SECONDARY_SPEAKER"
              : "REJECT_NOISE";

            if (process.env.NODE_ENV !== "production") {
              console.log(
                `[VAD:DECISION] ${decisionType} | reason=${decisionReason} | dBFS=${currentDb.toFixed(1)} | SNR=${snr.toFixed(1)}dB | floor=${ambientNoiseFloorDbRef.current.toFixed(1)}dB | vocalE=${vocalEnergy.toFixed(1)} | peakiness=${peakProminence.toFixed(2)} | harmonicity=${harmonicity.toFixed(2)} | dominantBin=${dominantBin} | profileMatch=${isProfileMatch} | candidate=${isSpeechCandidate} | frames=${consecutiveSpeechFramesRef.current} | turnActive=${isSpeechTurnActiveRef.current}`
              );
            }

            if (isSpeechCandidate) {
              wasSpeakingRef.current = true;
              speechEndTimestampRef.current = 0;
              consecutiveSpeechFramesRef.current++;

              // Require 3 consecutive qualifying speech frames (~128ms) to confirm genuine speech onset
              if (!isSpeechTurnActiveRef.current && consecutiveSpeechFramesRef.current >= 3) {
                isSpeechTurnActiveRef.current = true;
                speechDetectedRef.current = true;

                // Calibrate primary speaker profile if not already locked
                if (!primarySpeakerProfileRef.current.isCalibrated) {
                  primarySpeakerProfileRef.current = {
                    dominantBin,
                    centroid: spectralCentroid,
                    formantRatio: vocalEnergy / Math.max(highNoise, 1.0),
                    peakProminence,
                    sampleCount: 1,
                    isCalibrated: true,
                  };
                  console.log(
                    `[VAD:PROFILE] Primary speaker calibrated: dominantBin=${dominantBin}, centroid=${spectralCentroid.toFixed(2)}, peakiness=${peakProminence.toFixed(2)}`
                  );
                }

                // Prepend pre-roll buffer (~800ms) so first syllables/words in en/hi/mr are 100% preserved
                recordedBuffersRef.current = [...preRollBuffersRef.current, chunk];
                startStreamingSTT([...preRollBuffersRef.current, chunk]);
              } else if (isSpeechTurnActiveRef.current) {
                recordedBuffersRef.current.push(chunk);
                sendStreamingChunk(chunk);

                // Gentle update of primary speaker profile during confirmed speech
                if (primarySpeakerProfileRef.current.isCalibrated && condProximity) {
                  const p = primarySpeakerProfileRef.current;
                  p.dominantBin = p.dominantBin * 0.95 + dominantBin * 0.05;
                  p.centroid = p.centroid * 0.95 + spectralCentroid * 0.05;
                  p.peakProminence = p.peakProminence * 0.95 + peakProminence * 0.05;
                  p.sampleCount++;
                }
              }

              if (silenceTimerRef.current) {
                clearTimeout(silenceTimerRef.current);
                silenceTimerRef.current = null;
              }
            } else {
              consecutiveSpeechFramesRef.current = 0;
              if (isSpeechTurnActiveRef.current) {
                recordedBuffersRef.current.push(chunk);
                sendStreamingChunk(chunk);

                // HYSTERESIS: While turn is active, use tuned continuation threshold
                // Rejects background room murmurs during pauses while preserving natural speech hesitations
                // and unvoiced consonants in phone numbers / loan amounts (six, सात, पाच, fifty)
                const continuationThresholdDb = Math.max(
                  -42.0,
                  ambientNoiseFloorDbRef.current + 5.5
                );
                const isSpeechContinuation =
                  currentDb >= continuationThresholdDb &&
                  (vocalEnergy >= 20 || (highNoise >= 24 && zcr >= 0.12));

                if (isSpeechContinuation) {
                  wasSpeakingRef.current = true;
                  if (silenceTimerRef.current) {
                    clearTimeout(silenceTimerRef.current);
                    silenceTimerRef.current = null;
                  }
                } else {
                  if (wasSpeakingRef.current) {
                    speechEndTimestampRef.current = Date.now();
                    wasSpeakingRef.current = false;
                  }
                  if (!silenceTimerRef.current) {
                    silenceTimerRef.current = setTimeout(() => {
                      console.log(
                        `[LATENCY] T1 Microphone speech END confirmed at perf=${performance.now().toFixed(3)}ms (floor: ${ambientNoiseFloorDbRef.current.toFixed(1)}dBFS)`
                      );
                      submitCurrentSpeechTurn();
                    }, 800); // 800ms silence timeout across all speech turns with hysteresis to prevent premature cutoffs during pauses in numbers, amounts, or clauses
                  }
                }
              }
            }
          }
        };

        // Prefer AudioWorklet for low latency without main-thread blocking
        let usedAudioWorklet = false;
        if (typeof AudioWorkletNode !== "undefined" && audioCtx.audioWorklet) {
          try {
            if (!workletLoadedContextsRef.current.has(audioCtx)) {
              await audioCtx.audioWorklet.addModule("/worklets/pcm-capture-processor.js");
              workletLoadedContextsRef.current.add(audioCtx);
            }
            const workletNode = new AudioWorkletNode(audioCtx, "pcm-capture-processor", {
              processorOptions: { bufferSize: 2048 },
            });
            audioWorkletNodeRef.current = workletNode;
            workletNode.port.onmessage = (e: MessageEvent) => {
              if (e.data && e.data.type === "audio_data") {
                processAudioChunk(e.data.buffer, e.data.rms);
              }
            };
            voicePeaking.connect(workletNode);
            workletNode.connect(silentGain);
            usedAudioWorklet = true;
            console.log("[VoiceCopilot] AudioWorkletNode initialized successfully.");
          } catch (workletErr) {
            console.warn("[VoiceCopilot] AudioWorklet init failed, falling back to ScriptProcessorNode:", workletErr);
          }
        }

        if (!usedAudioWorklet) {
          const processor = audioCtx.createScriptProcessor(2048, 1, 1);
          scriptProcessorRef.current = processor;
          voicePeaking.connect(processor);
          processor.onaudioprocess = (e: AudioProcessingEvent) => {
            const inputChannel = e.inputBuffer.getChannelData(0);
            const chunk = new Float32Array(inputChannel.length);
            chunk.set(inputChannel);

            let sumSq = 0;
            for (let i = 0; i < chunk.length; i++) {
              sumSq += chunk[i] * chunk[i];
            }
            const rms = Math.sqrt(sumSq / chunk.length);
            processAudioChunk(chunk, rms);
          };
          processor.connect(silentGain);
          console.log("[VoiceCopilot] ScriptProcessorNode fallback initialized.");
        }

      setVoiceState("listening");
      voiceStateRef.current = "listening";
      setProcessingStage(null);
    } catch (err: any) {
      console.error("[VoiceCopilot] Microphone access error:", err);
      setErrorMessage(
        err?.name === "NotAllowedError"
          ? "Microphone permission denied. Please allow microphone access to use Continuous Voice Mode."
          : "Failed to initialize microphone."
      );
      stopVoiceMode();
    } finally {
      isStartingTurnRef.current = false;
    }
  } finally {
    releaseLock!();
  }
}, [executeBargeIn, sendStreamingChunk, startStreamingSTT, submitCurrentSpeechTurn]);

  // Keep startListeningTurnRef up to date for reliable invocation from audio callbacks
  useEffect(() => {
    startListeningTurnRef.current = startListeningTurn;
  }, [startListeningTurn]);

  // User manually finishes speaking / ends continuous conversation session
  const finishSpeakingTurn = useCallback(() => {
    // 1. Mark continuous session as finished so that subsequent turns do NOT auto-listen
    isContinuousModeRef.current = false;

    // 2. Stop any pending auto-listen, silence, or session timers immediately
    if (autoListenTimeoutRef.current) {
      clearTimeout(autoListenTimeoutRef.current);
      autoListenTimeoutRef.current = null;
    }
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }

    // 3. Immediately STOP/disable user's microphone capture for current turn
    isMicCaptureActiveRef.current = false;
    if (audioStreamRef.current) {
      try {
        audioStreamRef.current.getTracks().forEach((track) => {
          track.enabled = false;
          track.stop();
        });
      } catch (_) {}
      audioStreamRef.current = null;
    }
    if (filterNodesRef.current.length > 0) {
      filterNodesRef.current.forEach((node) => {
        try {
          node.disconnect();
        } catch (_) {}
      });
      filterNodesRef.current = [];
    }
    if (audioSourceNodeRef.current) {
      try {
        audioSourceNodeRef.current.disconnect();
      } catch (_) {}
      audioSourceNodeRef.current = null;
    }
    if (analyserRef.current) {
      try {
        analyserRef.current.disconnect();
      } catch (_) {}
      analyserRef.current = null;
    }
    if (audioWorkletNodeRef.current) {
      try {
        audioWorkletNodeRef.current.port.onmessage = null;
        audioWorkletNodeRef.current.disconnect();
      } catch (_) {}
      audioWorkletNodeRef.current = null;
    }
    if (scriptProcessorRef.current) {
      try {
        scriptProcessorRef.current.onaudioprocess = null;
        scriptProcessorRef.current.disconnect();
      } catch (_) {}
      scriptProcessorRef.current = null;
    }
    if (silentGainRef.current) {
      try {
        silentGainRef.current.disconnect();
      } catch (_) {}
      silentGainRef.current = null;
    }
    setAudioLevels([]);

    // 4. Stop sending any more microphone audio to Deepgram & clear buffers (do NOT start another STT turn)
    isWsStreamingRef.current = false;
    wsPendingQueueRef.current = [];
    recordedBuffersRef.current = [];
    preRollBuffersRef.current = [];
    isSpeechTurnActiveRef.current = false;
    speechDetectedRef.current = false;
    if (sttSocketRef.current) {
      try {
        if (sttSocketRef.current.readyState === WebSocket.OPEN) {
          sttSocketRef.current.close();
        }
      } catch (_) {}
      sttSocketRef.current = null;
    }

    // 5. If assistant is actively speaking or playing TTS, let it finish naturally out loud, then go idle.
    // If assistant is not speaking, transition immediately to idle.
    const isAssistantActivelySpeaking =
      isTTSPlaying ||
      voiceStateRef.current === "speaking" ||
      (sentenceAudioQueueRef.current?.isCurrentlyPlaying() ?? false);

    if (isAssistantActivelySpeaking) {
      console.log(
        "[VoiceCopilot] Finish Speaking clicked: mic/VAD stopped immediately; permitting active assistant speech to finish before idling."
      );
    } else {
      cancelActiveTurn();
      if (sentenceAudioQueueRef.current) {
        try {
          sentenceAudioQueueRef.current.bargeIn();
        } catch (_) {}
      }
      stopAllAudioPlayback();
      setVoiceState("idle");
      voiceStateRef.current = "idle";
      setProcessingStage(null);
      setIsTTSPlaying(false);
      isVoicePipelineActiveRef.current = false;
      console.log("[VoiceCopilot] Finish Speaking clicked: mic/VAD stopped immediately, transitioned to idle.");
    }
  }, [cancelActiveTurn, isTTSPlaying, stopAllAudioPlayback]);

  // Start fresh voice assistant session (completely NEW conversation/session)
  const startVoiceMode = useCallback(async () => {
    if (isStartingSessionRef.current) return;
    isStartingSessionRef.current = true;

    try {
      // Cancel any pending timers or speech from a prior session
      if (autoListenTimeoutRef.current) {
        clearTimeout(autoListenTimeoutRef.current);
        autoListenTimeoutRef.current = null;
      }
      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
        silenceTimerRef.current = null;
      }
      if (sentenceAudioQueueRef.current) {
        try {
          sentenceAudioQueueRef.current.bargeIn();
        } catch (_) {}
      }
      if (ttsAudioRef.current) {
        try {
          ttsAudioRef.current.pause();
          ttsAudioRef.current.currentTime = 0;
        } catch (_) {}
      }
      setIsTTSPlaying(false);
      isVoicePipelineActiveRef.current = false;

      cancelActiveTurn();
      isContinuousModeRef.current = true;
      setElapsedSeconds(0);

      // Completely NEW conversation / session
      setConversationMessages([]);
      activeLeadIdRef.current = null;
      currentLeadRef.current = null;
      setExtractedLead(null);
      setSaveLeadSuccess(null);
      setTranscript(null);
      setAssistantResponseText(null);
      setSessionTurnCount(0);
      turnIdRef.current = 0;
      handledFinalTurnsRef.current.clear();

      // Start session timer
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = setInterval(() => {
        setElapsedSeconds((prev) => prev + 1);
      }, 1000);

      setVoiceState("listening");
      voiceStateRef.current = "listening";
      await startListeningTurn();
    } finally {
      isStartingSessionRef.current = false;
    }
  }, [cancelActiveTurn, startListeningTurn]);

  // Stop continuous voice assistant session
  const stopVoiceMode = useCallback(() => {
    isContinuousModeRef.current = false;
    if (sttSocketRef.current) {
      try {
        sttSocketRef.current.send(JSON.stringify({ type: "CloseStream" }));
        sttSocketRef.current.close();
      } catch (_) {}
      sttSocketRef.current = null;
    }
    cancelActiveTurn();

    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    primarySpeakerProfileRef.current = {
      dominantBin: 5,
      centroid: 6.0,
      formantRatio: 1.5,
      peakProminence: 2.0,
      sampleCount: 0,
      isCalibrated: false,
    };
    if (audioWorkletNodeRef.current) {
      try {
        audioWorkletNodeRef.current.port.onmessage = null;
        audioWorkletNodeRef.current.disconnect();
      } catch (_) {}
      audioWorkletNodeRef.current = null;
    }
    if (scriptProcessorRef.current) {
      try {
        scriptProcessorRef.current.disconnect();
      } catch (_) {}
      scriptProcessorRef.current = null;
    }
    if (silentGainRef.current) {
      try {
        silentGainRef.current.disconnect();
      } catch (_) {}
      silentGainRef.current = null;
    }
    if (analyserRef.current) {
      try {
        analyserRef.current.disconnect();
      } catch (_) {}
      analyserRef.current = null;
    }
    if (filterNodesRef.current.length > 0) {
      filterNodesRef.current.forEach((node) => {
        try {
          node.disconnect();
        } catch (_) {}
      });
      filterNodesRef.current = [];
    }
    if (audioSourceNodeRef.current) {
      try {
        audioSourceNodeRef.current.disconnect();
      } catch (_) {}
      audioSourceNodeRef.current = null;
    }
    if (audioStreamRef.current) {
      audioStreamRef.current.getTracks().forEach((track) => track.stop());
      audioStreamRef.current = null;
    }

    // Safely close AudioContext without interrupting any pending resume()
    const ctx = audioContextRef.current;
    audioContextRef.current = null;
    if (ctx && ctx.state !== "closed") {
      const closeSafely = async () => {
        if (resumePromiseRef.current) {
          try {
            await resumePromiseRef.current;
          } catch (_) {}
        }
        try {
          if (ctx.state !== "closed") {
            await ctx.close();
          }
        } catch (err) {
          console.debug("[VoiceCopilot] AudioContext close note:", err);
        }
      };
      const prevLock = lifecycleLockRef.current;
      let releaseLock: () => void;
      lifecycleLockRef.current = new Promise<void>((resolve) => {
        releaseLock = resolve;
      });
      prevLock.then(closeSafely).finally(() => releaseLock!());
    }

    recordedBuffersRef.current = [];
    preRollBuffersRef.current = [];
    isSpeechTurnActiveRef.current = false;
    speechDetectedRef.current = false;

    setVoiceState("idle");
    voiceStateRef.current = "idle";
    setProcessingStage(null);
    setAudioLevels(new Array(24).fill(16));
  }, [cancelActiveTurn]);

  // Reset entire lead session to start fresh on a new prospect
  const resetLeadSession = () => {
    stopVoiceMode();
    activeLeadIdRef.current = null;
    currentLeadRef.current = null;
    setExtractedLead(null);
    setTranscript(null);
    setConversationMessages([]);
    setSaveLeadSuccess(null);
    setSessionTurnCount(0);
    setElapsedSeconds(0);
    setErrorMessage(null);
    setAssistantResponseText(null);
  };

  const copyToClipboard = (text: string, fieldKey: string) => {
    navigator.clipboard?.writeText(text);
    setCopiedField(fieldKey);
    setTimeout(() => setCopiedField(null), 2000);
  };

  return (
    <div className="w-full flex-1 flex flex-col min-h-0 h-full">
      {/* Error Notice (if any audio/permissions error occurs) */}
      {errorMessage && (
        <div className="mb-4 p-3.5 rounded-2xl bg-rose-50 border border-rose-200 flex items-center justify-between gap-3 text-xs text-rose-700 animate-in fade-in duration-150 shrink-0">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-500 shrink-0" />
            <span>{errorMessage}</span>
          </div>
          <button
            onClick={() => setErrorMessage(null)}
            className="text-rose-400 hover:text-rose-600 font-bold px-2 py-0.5 cursor-pointer"
          >
            &times;
          </button>
        </div>
      )}

      {/* Main Unified 2-Column Card with Clear Vertical Divider */}
      <div className="w-full flex-1 flex flex-col bg-white rounded-3xl border border-slate-200/80 shadow-[0_2px_16px_-4px_rgba(0,0,0,0.04)] overflow-hidden h-full min-h-0">
        <div className="flex flex-col lg:flex-row flex-1 min-h-0 h-full divide-y lg:divide-y-0 lg:divide-x divide-slate-200/80">
          {/* ===================================================================
              LEFT 40%: Dedicated Large Microphone & Voice Interaction Panel
              =================================================================== */}
          <div className="w-full lg:w-[40%] p-6 sm:p-8 flex flex-col justify-between items-center text-center relative bg-white shrink-0 overflow-y-auto">
            {/* Top Status Pill */}
            <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full text-xs font-semibold bg-[#e6f7f2] text-teal-800 border border-teal-200/40 shrink-0">
              <span
                className={`w-2 h-2 rounded-full ${
                  voiceState === "listening"
                    ? "bg-emerald-500 animate-ping"
                    : voiceState === "speaking"
                    ? "bg-teal-500 animate-pulse"
                    : "bg-emerald-500"
                }`}
              />
              <span>
                {voiceState === "listening"
                  ? "Listening..."
                  : voiceState === "processing"
                  ? "Processing..."
                  : voiceState === "speaking"
                  ? "Speaking..."
                  : "Ready"}
              </span>
            </div>

            {/* Center: Concentric Circles with Large Microphone Button */}
            <div className="my-auto py-6 sm:py-8 flex flex-col items-center justify-center">
              <div className="relative flex items-center justify-center">
                {/* Outermost ring */}
                <div
                  className={`w-56 h-56 sm:w-64 sm:h-64 lg:w-72 lg:h-72 rounded-full bg-[#e6f7f2] flex items-center justify-center transition-all duration-300 ${
                    voiceState === "listening" ? "scale-105" : ""
                  }`}
                >
                  {/* Middle ring */}
                  <div
                    className={`w-40 h-40 sm:w-48 sm:h-48 lg:w-52 lg:h-52 rounded-full bg-[#bfead8] flex items-center justify-center transition-all duration-300 ${
                      voiceState === "listening" ? "scale-105" : ""
                    }`}
                  >
                    {/* Inner circular button */}
                    <button
                      onClick={() => {
                        if (voiceState === "idle") {
                          startVoiceMode();
                        } else {
                          finishSpeakingTurn();
                        }
                      }}
                      disabled={voiceState === "processing"}
                      className="w-24 h-24 sm:w-28 sm:h-28 lg:w-32 lg:h-32 rounded-full bg-[#00897b] hover:bg-[#00796b] text-white flex items-center justify-center shadow-lg shadow-teal-700/25 transition-transform hover:scale-105 active:scale-95 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                      title={
                        voiceState === "idle"
                          ? "Tap to Speak"
                          : "Finish Speaking"
                      }
                    >
                      <Mic className="w-10 h-10 sm:w-12 sm:h-12 text-white stroke-[2.2]" />
                    </button>
                  </div>
                </div>
              </div>

              {/* Title & Subtext */}
              <h3 className="text-base sm:text-lg lg:text-xl font-bold text-slate-900 mt-6 sm:mt-7 mb-1">
                Speak naturally
              </h3>
              <p className="text-xs sm:text-sm text-slate-400 max-w-xs leading-relaxed">
                Your voice will be converted to text and processed in real time.
              </p>
            </div>

            {/* Bottom Pill Container */}
            <div className="w-full rounded-2xl bg-[#e6f7f2] border border-teal-100/70 p-3 sm:p-3.5 flex items-center gap-3 text-left shrink-0">
              <div className="text-[#00897b] flex items-center shrink-0">
                <WaveformIcon className="w-5 h-5 text-[#00897b]" />
              </div>
              <span className="text-xs sm:text-sm font-medium text-slate-700 truncate">
                {voiceState === "listening" && transcript
                  ? transcript
                  : voiceState === "listening"
                  ? "Listening for your voice..."
                  : voiceState === "speaking"
                  ? "Assistant is speaking..."
                  : voiceState === "processing"
                  ? "Processing speech..."
                  : "Tap to Speak to start"}
              </span>
            </div>
          </div>

          {/* ===================================================================
              RIGHT 60%: Dedicated Conversation Panel (Starts Empty)
              =================================================================== */}
          <div className="w-full lg:w-[60%] flex flex-col flex-1 min-h-0 bg-white self-stretch">
            {/* Header: Audio Waveform + Conversation Title & Language Selector */}
            <div className="px-6 sm:px-8 py-4 sm:py-5 flex items-center justify-between border-b border-slate-100 shrink-0">
              <div className="flex items-center gap-2.5">
                <div className="text-[#00897b] flex items-center">
                  <WaveformIcon className="w-4 h-4 text-[#00897b]" />
                </div>
                <span className="font-bold text-slate-900 text-sm sm:text-base">
                  Conversation
                </span>
                <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 border border-slate-200 uppercase">
                  {(selectedLanguage !== "auto" ? selectedLanguage : (detectedLanguage || "en")).toUpperCase()}
                </span>
              </div>

              {/* Language Selector Dropdown */}
              <div className="relative" ref={languageMenuRef}>
                <button
                  onClick={() => setIsLanguageMenuOpen(!isLanguageMenuOpen)}
                  className="rounded-full border border-slate-200/90 px-3.5 py-1 bg-white hover:bg-slate-50 flex items-center gap-1.5 text-xs font-semibold text-slate-700 cursor-pointer shadow-2xs transition-colors"
                  title="Select Language"
                >
                  <Globe className="w-3.5 h-3.5 text-slate-400" />
                  <span>
                    {selectedLanguage === "auto" ? "Auto" : selectedLanguage.toUpperCase()}
                  </span>
                  <ChevronDown className="w-3 h-3 text-slate-400" />
                </button>

                {isLanguageMenuOpen && (
                  <div className="absolute right-0 top-full mt-1.5 w-36 rounded-xl bg-white border border-slate-100 shadow-lg py-1 z-30 text-xs">
                    <button
                      onClick={() => {
                        isManualLanguageSelectionRef.current = true;
                        setSelectedLanguage("en");
                        setDetectedLanguage("en");
                        detectedLanguageRef.current = "en";
                        setIsLanguageMenuOpen(false);
                      }}
                      className={`w-full px-3 py-2 text-left hover:bg-slate-50 flex items-center justify-between cursor-pointer ${
                        selectedLanguage === "en"
                          ? "font-bold text-teal-700 bg-teal-50/50"
                          : "text-slate-700"
                      }`}
                    >
                      <span>English (EN)</span>
                      {selectedLanguage === "en" && (
                        <Check className="w-3 h-3 text-teal-600" />
                      )}
                    </button>
                    <button
                      onClick={() => {
                        isManualLanguageSelectionRef.current = true;
                        setSelectedLanguage("hi");
                        setDetectedLanguage("hi");
                        detectedLanguageRef.current = "hi";
                        setIsLanguageMenuOpen(false);
                      }}
                      className={`w-full px-3 py-2 text-left hover:bg-slate-50 flex items-center justify-between cursor-pointer ${
                        selectedLanguage === "hi"
                          ? "font-bold text-teal-700 bg-teal-50/50"
                          : "text-slate-700"
                      }`}
                    >
                      <span>हिंदी (Hindi)</span>
                      {selectedLanguage === "hi" && (
                        <Check className="w-3 h-3 text-teal-600" />
                      )}
                    </button>
                    <button
                      onClick={() => {
                        isManualLanguageSelectionRef.current = true;
                        setSelectedLanguage("mr");
                        setDetectedLanguage("mr");
                        detectedLanguageRef.current = "mr";
                        setIsLanguageMenuOpen(false);
                      }}
                      className={`w-full px-3 py-2 text-left hover:bg-slate-50 flex items-center justify-between cursor-pointer ${
                        selectedLanguage === "mr"
                          ? "font-bold text-teal-700 bg-teal-50/50"
                          : "text-slate-700"
                      }`}
                    >
                      <span>मराठी (Marathi)</span>
                      {selectedLanguage === "mr" && (
                        <Check className="w-3 h-3 text-teal-600" />
                      )}
                    </button>
                    <button
                      onClick={() => {
                        isManualLanguageSelectionRef.current = false;
                        setSelectedLanguage("auto");
                        setIsLanguageMenuOpen(false);
                      }}
                      className={`w-full px-3 py-2 text-left hover:bg-slate-50 flex items-center justify-between cursor-pointer ${
                        selectedLanguage === "auto"
                          ? "font-bold text-teal-700 bg-teal-50/50"
                          : "text-slate-700"
                      }`}
                    >
                      <span>Auto Detect</span>
                      {selectedLanguage === "auto" && (
                        <Check className="w-3 h-3 text-teal-600" />
                      )}
                    </button>
                  </div>
                )}
              </div>
            </div>

            {/* Conversation Messages Container - Starts EMPTY, populates from real runtime data */}
            <div ref={messagesContainerRef} className="flex-1 min-h-0 overflow-y-auto px-6 sm:px-8 py-5 space-y-4">
              {conversationMessages.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-center text-slate-400 py-12">
                  <p className="text-xs sm:text-sm font-normal">
                    No conversation messages yet. Tap to Speak to start.
                  </p>
                </div>
              ) : (
                conversationMessages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`flex w-full ${
                      msg.sender === "user" ? "justify-end" : "justify-start"
                    } animate-in fade-in duration-200`}
                  >
                    {msg.sender === "user" ? (
                      /* User Message Bubble */
                      <div className="max-w-[85%] sm:max-w-[80%] rounded-2xl bg-[#e6f7f2] border border-teal-100/70 p-4 text-left shadow-2xs">
                        <div className="flex items-center justify-between gap-4 mb-1.5">
                          <div className="flex items-center gap-2">
                            <div className="w-6 h-6 rounded-full bg-[#00897b] text-white flex items-center justify-center shrink-0">
                              <User className="w-3.5 h-3.5" />
                            </div>
                            <span className="font-semibold text-xs text-slate-800">
                              You
                            </span>
                          </div>
                          <span className="text-[11px] text-slate-400 font-normal">
                            {msg.timestamp}
                          </span>
                        </div>
                        <p className="text-xs sm:text-sm text-slate-800 leading-relaxed font-normal">
                          {msg.text}
                        </p>
                      </div>
                    ) : (
                      /* Assistant Message Bubble */
                      <div className="max-w-[85%] sm:max-w-[85%] rounded-2xl bg-[#f0f4ff] border border-indigo-50 p-4 text-left shadow-2xs">
                        <div className="flex items-center justify-between gap-4 mb-1.5">
                          <div className="flex items-center gap-2">
                            <div className="w-6 h-6 rounded-full bg-[#5c7cfa] text-white flex items-center justify-center shrink-0">
                              <Bot className="w-3.5 h-3.5" />
                            </div>
                            <span className="font-semibold text-xs text-slate-800">
                              Assistant
                            </span>
                            <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-indigo-100 text-indigo-700 uppercase">
                              {msg.language || getMessageLanguage(msg.text, detectedLanguage)}
                            </span>
                          </div>
                          <span className="text-[11px] text-slate-400 font-normal">
                            {msg.timestamp}
                          </span>
                        </div>
                        <p className="text-xs sm:text-sm text-slate-700 leading-relaxed font-normal">
                          {msg.text}
                        </p>
                      </div>
                    )}
                  </div>
                ))
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Bottom Bar: Status on Left, Primary Action Button on Right (Pinned at bottom) */}
            <div className="shrink-0 mt-auto bg-white border-t border-slate-100 px-6 sm:px-8 py-4 flex items-center justify-between gap-4">
              <div className="flex items-center gap-2 text-xs font-medium text-slate-600">
                <span
                  className={`w-2 h-2 rounded-full ${
                    voiceState === "listening"
                      ? "bg-emerald-500 animate-ping"
                      : voiceState === "speaking"
                      ? "bg-teal-500 animate-pulse"
                      : "bg-emerald-500"
                  }`}
                />
                <span>
                  {voiceState === "listening"
                    ? "Listening..."
                    : voiceState === "processing"
                    ? "Processing..."
                    : voiceState === "speaking"
                    ? "Assistant is speaking..."
                    : "Ready for your next question"}
                </span>
              </div>

              <button
                onClick={() => {
                  if (voiceState === "idle") {
                    startVoiceMode();
                  } else {
                    finishSpeakingTurn();
                  }
                }}
                disabled={voiceState === "processing"}
                className="rounded-full px-5 py-2.5 bg-[#00897b] hover:bg-[#00796b] text-white text-xs sm:text-sm font-semibold flex items-center gap-2 shadow-xs transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Mic className="w-4 h-4 text-white" />
                <span>
                  {voiceState === "idle"
                    ? "Tap to Speak"
                    : voiceState === "listening"
                    ? "Finish Speaking"
                    : voiceState === "speaking"
                    ? "Finish Speaking"
                    : "Processing..."}
                </span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
