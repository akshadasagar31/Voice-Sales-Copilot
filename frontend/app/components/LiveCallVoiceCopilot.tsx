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
} from "lucide-react";
import { SentenceTokenizer, SentenceAudioQueue } from "@/lib/sentenceStreamingTTS";

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


interface LiveCallVoiceCopilotProps {
  onAudioRecorded?: (blob: Blob) => void;
  onLeadSaved?: (lead: ExtractedLead) => void;
}

// Preset simulation scenarios for multi-turn testing in English, Hindi, and Marathi
const SIMULATION_PRESETS = [
  {
    id: "turn1_en",
    title: "Turn 1: Prospect Intro (English)",
    lang: "EN",
    text: "Had a great introductory call with Rajesh Kumar, Vice President of Technology at Acme Corporation. He is actively exploring enterprise financing options.",
  },
  {
    id: "turn2_en",
    title: "Turn 2: Loan Details (English - Updates Same Lead)",
    lang: "EN",
    text: "Rajesh confirmed Acme needs an Equipment Loan of 5000000 dollars over a 36-month term. His direct contact number is +91-9876543210.",
  },
  {
    id: "hi_full",
    title: "Complete Lead (हिंदी - Hindi)",
    lang: "HI",
    text: "नमस्ते, मैंने इंफोसिस के डायरेक्टर अमित वर्मा से बात की। उन्हें 25 लाख रुपये का बिजनेस लोन 24 महीने के लिए चाहिए। उनका फोन नंबर 9811223344 है।",
  },
  {
    id: "mr_full",
    title: "Complete Lead (मराठी - Marathi)",
    lang: "MR",
    text: "नमस्कार, मी टेक महिंद्राचे व्यवस्थापक राहुल पाटील यांच्याशी बोललो. त्यांना 30 लाख रुपयांचे व्यवसाय कर्ज 36 महिन्यांच्या कालावधीसाठी हवे आहे. त्यांचा फोन नंबर 9822334455 आहे.",
  },
];

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

export default function LiveCallVoiceCopilot({ onAudioRecorded, onLeadSaved }: LiveCallVoiceCopilotProps) {
  // Active sub-tab: "assistant" (continuous voice mode) or "simulation"
  const [activeSubTab, setActiveSubTab] = useState<"assistant" | "simulation">("assistant");

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

  // Concurrency & turn refs
  const turnIdRef = useRef<number>(0);
  const isContinuousModeRef = useRef<boolean>(false);
  const isVoicePipelineActiveRef = useRef<boolean>(false);
  const activeAbortControllerRef = useRef<AbortController | null>(null);

  // Multi-Turn Lead Memory: tracks active PostgreSQL lead ID and fields across turns
  const activeLeadIdRef = useRef<number | null>(null);
  const currentLeadRef = useRef<ExtractedLead | null>(null);

  // Continuous Web Audio API & PCM capture refs (Guarantees unified AudioContext & pre-roll)
  const audioStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const resumePromiseRef = useRef<Promise<void> | null>(null);
  const lifecycleLockRef = useRef<Promise<void>>(Promise.resolve());
  const isStartingTurnRef = useRef<boolean>(false);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioSourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const silentGainRef = useRef<GainNode | null>(null);
  const filterNodesRef = useRef<BiquadFilterNode[]>([]);

  // Pre-roll PCM buffers (~850ms rolling window)
  const preRollBuffersRef = useRef<Float32Array[]>([]);
  const recordedBuffersRef = useRef<Float32Array[]>([]);
  const isSpeechTurnActiveRef = useRef<boolean>(false);
  const speechDetectedRef = useRef<boolean>(false);
  const consecutiveSpeechFramesRef = useRef<number>(0);
  const consecutiveBargeInFramesRef = useRef<number>(0);
  const speakerBleedBaselineRef = useRef<number>(18);
  const bleedRmsRef = useRef<number>(0.012);
  const ambientNoiseFloorRef = useRef<number>(0.006);
  const sentenceStartTimestampRef = useRef<number>(0);
  const speechEndTimestampRef = useRef<number>(0);

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
    return turnId !== turnIdRef.current || !isContinuousModeRef.current;
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

  // Cancel any active turn and abort pending HTTP requests
  const cancelActiveTurn = useCallback(() => {
    turnIdRef.current += 1;
    if (activeAbortControllerRef.current) {
      activeAbortControllerRef.current.abort();
      activeAbortControllerRef.current = null;
    }
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    if (autoListenTimeoutRef.current) {
      clearTimeout(autoListenTimeoutRef.current);
      autoListenTimeoutRef.current = null;
    }
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
  }, [cancelActiveTurn, stopAllAudioPlayback]);

  // Clean up on component unmount
  useEffect(() => {
    return () => {
      cancelActiveTurn();
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
      if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
      if (autoListenTimeoutRef.current) clearTimeout(autoListenTimeoutRef.current);
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
  // Core Continuous Voice Pipeline: STT -> DeepSeek Stream -> DB CRM Update -> Sentence TTS -> Auto-Listen
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
      // Step 1: Deepgram STT transcription with retry
      const filename = `turn_${turnId}_${Date.now()}.wav`;
      const formData = new FormData();

      if (typeof File !== "undefined") {
        const audioFile = new File([blob], filename, { type: "audio/wav" });
        formData.append("file", audioFile);
      } else {
        formData.append("file", blob, filename);
      }

      formData.append("language", selectedLanguage === "auto" ? "multi" : selectedLanguage);
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
            await new Promise((resolve) => setTimeout(resolve, 600));
            continue;
          }
          break;
        } catch (fetchErr: any) {
          if (fetchErr?.name === "AbortError" || isTurnStale(turnId)) return;
          if (attempt < 2) await new Promise((resolve) => setTimeout(resolve, 600));
        }
      }

      if (!sttRes || !sttRes.ok) {
        const errJson = (await sttRes?.json().catch(() => ({}))) || {};
        throw new Error(errJson.error || errJson.detail || "Deepgram STT transcription failed.");
      }

      // STEP 1: UNPACK DEEPGRAM TRANSCRIPT
      // Deepgram returns the transcribed text and detected language (en, hi, or mr).
      const sttData = await sttRes.json();
      if (isTurnStale(turnId)) return;

      const rawTranscript = (sttData.transcript || "").trim();
      const rawDetected = (
        sttData.detected_language || (selectedLanguage !== "auto" ? selectedLanguage : "en")
      ).toLowerCase();
      let detectedLang = rawDetected.split("-")[0];
      if (!["en", "hi", "mr", "mixed"].includes(detectedLang)) {
        detectedLang = "en";
      }

      setDetectedLanguage(detectedLang);
      const respLang = (sttData.language || (detectedLang === "mixed" ? "en" : detectedLang)).toLowerCase();

      // Check if backend reset/new lead occurred
      if (sttData.is_new_lead) {
        console.log("[VoiceCopilot] Backend detected new lead / reset. Starting fresh from Name.");
        activeLeadIdRef.current = sttData.lead_id || null;
        currentLeadRef.current = sttData.lead || null;
        setExtractedLead(sttData.lead || null);
        setSaveLeadSuccess(null);
      }

      // If Deepgram / Sarvam heard only background noise and no actual words:
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

      // STEP 2: SETUP SENTENCE-STREAMING AUDIO QUEUE
      // We initialize an Audio player and SentenceAudioQueue that will fetch
      // Deepgram TTS for sentences in parallel and play them sequentially.
      if (!ttsAudioRef.current && typeof window !== "undefined") {
        ttsAudioRef.current = new Audio();
      }
      const audioPlayer = ttsAudioRef.current;
      if (audioPlayer && !sentenceAudioQueueRef.current) {
        sentenceAudioQueueRef.current = new SentenceAudioQueue(audioPlayer, {}, audioContextRef.current, {
          module: "module1",
          speaker: "simran",
        });
      }
      if (sentenceAudioQueueRef.current) {
        sentenceAudioQueueRef.current.setOptions({ module: "module1", speaker: "simran" });
        if (audioContextRef.current) {
          sentenceAudioQueueRef.current.setAudioContext(audioContextRef.current);
        }
      }

      sentenceAudioQueueRef.current?.updateCallbacks({
        onSentenceStart: () => {
          if (isTurnStale(turnId)) return;
          sentenceStartTimestampRef.current = Date.now();
          const speechEnd = speechEndTimestampRef.current || (sentenceStartTimestampRef.current - 450);
          const totalLatency = sentenceStartTimestampRef.current - speechEnd;
          console.log(`%c[VoiceCopilot:Latency] >>> FIRST AUDIO RESPONSE PLAYING: ${totalLatency}ms after speech ended! <<<`, "color: #10b981; font-weight: bold; font-size: 14px;");
          setVoiceState("speaking");
          voiceStateRef.current = "speaking";
          setProcessingStage("speaking");
          setIsTTSPlaying(true);
        },
        onQueueComplete: () => {
          // STEP 3: AUTO-LISTEN AFTER SPEAKING COMPLETES
          // As soon as the copilot finishes speaking the response, it automatically
          // re-arms the microphone so the user can speak again without clicking any buttons!
          console.log(`[VoiceCopilot] Voice response finished for Turn #${turnId}. Auto-listening...`);
          setIsTTSPlaying(false);
          isVoicePipelineActiveRef.current = false;

          // Automatically resume continuous listening for next prospect speech turn
          if (isContinuousModeRef.current) {
            setVoiceState("listening");
            voiceStateRef.current = "listening";
            setProcessingStage(null);

            isSpeechTurnActiveRef.current = false;
            speechDetectedRef.current = false;
            recordedBuffersRef.current = [];
            preRollBuffersRef.current = [];
            consecutiveSpeechFramesRef.current = 0;
            consecutiveBargeInFramesRef.current = 0;

            if (autoListenTimeoutRef.current) clearTimeout(autoListenTimeoutRef.current);
            autoListenTimeoutRef.current = setTimeout(() => {
              if (isContinuousModeRef.current && !isVoicePipelineActiveRef.current) {
                startListeningTurn();
              }
            }, 100);
          } else {
            setVoiceState("idle");
            voiceStateRef.current = "idle";
            setProcessingStage(null);
          }
        },
        onError: (e) => {
          console.warn("[VoiceCopilot] Audio playback error:", e);
          setIsTTSPlaying(false);
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
        },
      });

      sentenceAudioQueueRef.current?.startNewTurn(turnId);
      const tokenizer = new SentenceTokenizer();

      // FAST-PATH: INSTANT DETERMINISTIC FIRST RESPONSE (< 500ms speech-to-audio)
      // Uses deterministic lead state from /api/voice-entry to start Sarvam Bulbul v3 TTS immediately
      // for Sentence 1 without waiting for full LLM / DB processing.
      const immediateSentence = sttData.immediate_sentence1 || (sttData.is_greeting ? (sttData.greeting_response || getNaturalGreeting(respLang)) : (sttData.is_assistant_query ? (sttData.assistant_response || getAssistantAnswerFallback(rawTranscript, respLang)) : null));
      if (immediateSentence) {
        console.log(`[VoiceCopilot:Speed] Starting immediate Sarvam TTS for sentence 1: "${immediateSentence}"`);
        setAssistantResponseText(immediateSentence);
        if (sttData.lead && !sttData.is_assistant_query) {
          currentLeadRef.current = sttData.lead;
          setExtractedLead(sttData.lead);
          if (onLeadSaved) onLeadSaved(sttData.lead);
        }
        if (sttData.lead_id && !sttData.is_assistant_query) {
          activeLeadIdRef.current = sttData.lead_id;
          setSaveLeadSuccess({
            id: sttData.lead_id,
            message: `Lead #${sttData.lead_id} active in PostgreSQL database!`,
            isUpdate: true,
          });
        }

        setVoiceState("speaking");
        voiceStateRef.current = "speaking";
        setProcessingStage("speaking");
        setIsTTSPlaying(true);

        const sentences = tokenizer.feed(immediateSentence);
        const trailing = tokenizer.flush();
        if (trailing) sentences.push(trailing);
        if (sentences.length === 0 && immediateSentence.trim()) sentences.push(immediateSentence.trim());

        for (const s of sentences) {
          sentenceAudioQueueRef.current?.enqueueSentence(s, respLang);
        }
        sentenceAudioQueueRef.current?.markStreamComplete();

        const latencyToTts = Date.now() - (speechEndTimestampRef.current || Date.now());
        console.log(`[VoiceCopilot:Latency] Speech End -> Sentence 1 Enqueued for TTS: ${latencyToTts}ms`);
        return;
      }

      // ASSISTANT-CONVERSATION HANDLING (Fallback):
      // Questions like "What is your name?", "Who are you?", and "What can you do?"
      // get natural assistant answers without modifying lead data or asking lead fields.
      const isAssistantQueryTurn = Boolean(sttData.is_assistant_query || isAssistantQueryText(rawTranscript));
      if (isAssistantQueryTurn) {
        console.log(`[VoiceCopilot] Turn #${turnId} identified as assistant-conversation query ("${rawTranscript}").`);
        const assistantSpoken = sttData.assistant_response || sttData.immediate_sentence1 || getAssistantAnswerFallback(rawTranscript, respLang);
        setAssistantResponseText(assistantSpoken);
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

      // GREETING-ONLY HANDLING (Fallback):
      // When the user says only a greeting ("Hello", "Hi", "नमस्ते", "नमस्कार"),
      // respond naturally without modifying PostgreSQL CRM data and automatically resume listening.
      const isGreetingTurn = Boolean(sttData.is_greeting || isGreetingOnly(rawTranscript));
      if (isGreetingTurn) {
        console.log(`[VoiceCopilot] Turn #${turnId} identified as greeting-only ("${rawTranscript}").`);
        const greetingSpoken = sttData.greeting_response || getNaturalGreeting(respLang);
        setAssistantResponseText(greetingSpoken);
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

      const extractRes = await fetch("/api/extract-lead", {
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
              setAssistantResponseText(accumulatedSpoken);

              // Feed token to sentence tokenizer to immediately yield completed sentences
              const completedSentences = tokenizer.feed(token);
              for (const s of completedSentences) {
                const cleaned = cleanTextForSpeech(s);
                if (cleaned) {
                  sentenceAudioQueueRef.current?.enqueueSentence(cleaned, respLang);
                }
              }
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

      // Flush any remaining tokens into sentence audio queue
      const finalTrailing = tokenizer.flush();
      if (finalTrailing) {
        const cleaned = cleanTextForSpeech(finalTrailing);
        if (cleaned) {
          sentenceAudioQueueRef.current?.enqueueSentence(cleaned, respLang);
        }
      }
      sentenceAudioQueueRef.current?.markStreamComplete();
    } catch (err: any) {
      if (err?.name === "AbortError" || isTurnStale(turnId)) {
        return;
      }
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
  // HANDLER: submitCurrentSpeechTurn
  // --------------------------------------------------------------------------
  // • WHAT IT DOES: Finalizes the recorded speech segment after 400ms of silence,
  //   discards noise glitches (< 300ms), encodes PCM to 16-bit WAV, and invokes processSpeechTurn.
  // • INPUTS: None (reads from recordedBuffersRef).
  // • OUTPUT: None (dispatches async processing pipeline).
  // • WHY IT IS USED: Converts continuously captured microphone buffers into a single
  //   clean turn audio file ready for speech-to-text.
  // • WHERE IT FITS IN THE FLOW:
  //     [User finishes speaking + 400ms silence] -> [submitCurrentSpeechTurn] -> [processSpeechTurn]
  // --------------------------------------------------------------------------
  const submitCurrentSpeechTurn = useCallback(() => {
    if (isVoicePipelineActiveRef.current) return;

    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    isSpeechTurnActiveRef.current = false;
    speechDetectedRef.current = false;

    const buffersToEncode = [...recordedBuffersRef.current];
    recordedBuffersRef.current = [];
    preRollBuffersRef.current = []; // Clear pre-roll buffer so previous turn's audio does not bleed into the next turn

    if (buffersToEncode.length === 0) return;

    const sampleRate = audioContextRef.current?.sampleRate || 48000;
    const totalSamples = buffersToEncode.reduce((acc, b) => acc + b.length, 0);
    const duration = totalSamples / sampleRate;

    // Discard noise glitches (< 400ms)
    if (duration < 0.40) {
      return;
    }

    // Energy validation: verify that the recorded segment contains genuine speech energy
    let totalEnergy = 0;
    let peakRms = 0;
    for (const buf of buffersToEncode) {
      let sum = 0;
      for (let i = 0; i < buf.length; i++) {
        sum += buf[i] * buf[i];
      }
      const bufRms = Math.sqrt(sum / buf.length);
      totalEnergy += bufRms;
      if (bufRms > peakRms) peakRms = bufRms;
    }
    const avgRms = totalEnergy / buffersToEncode.length;

    // If overall energy is only room noise (e.g. distant murmur or quiet fan hum)
    if (avgRms < 0.013 && peakRms < 0.022) {
      console.log(`[VoiceCopilot:VAD] Discarded low-energy ambient sound (avgRms: ${avgRms.toFixed(4)}, peak: ${peakRms.toFixed(4)})`);
      return;
    }

    const wavBlob = encodeWav(buffersToEncode, sampleRate);
    if (onAudioRecorded) {
      onAudioRecorded(wavBlob);
    }

    const currentTurnId = ++turnIdRef.current;
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
          scriptProcessorRef.current !== null &&
          audioSourceNodeRef.current !== null;

        if (isGraphAlreadyRunning) {
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

        if (!isContinuousModeRef.current || audioCtx.state === "closed") return;

        // 4. Create DSP speech bandpass filters from the active AudioContext instance
        const source = audioCtx.createMediaStreamSource(stream);
        audioSourceNodeRef.current = source;

        // High-pass filter (85 Hz) to strip low-frequency mechanical rumble, AC hum, desk thumps, breathing plosives
        const highpass = audioCtx.createBiquadFilter();
        highpass.type = "highpass";
        highpass.frequency.value = 85;
        highpass.Q.value = 0.7;

        // Low-pass filter (3800 Hz) to eliminate high-frequency fan hiss, keyboard clicks, coil whine
        const lowpass = audioCtx.createBiquadFilter();
        lowpass.type = "lowpass";
        lowpass.frequency.value = 3800;
        lowpass.Q.value = 0.7;

        // Intelligibility peaking bandpass (1200 Hz, +3dB) to enhance nearby voice clarity
        const voicePeaking = audioCtx.createBiquadFilter();
        voicePeaking.type = "peaking";
        voicePeaking.frequency.value = 1200;
        voicePeaking.gain.value = 3;
        voicePeaking.Q.value = 1.0;

        filterNodesRef.current = [highpass, lowpass, voicePeaking];

        // Connect DSP filter chain: source -> highpass -> lowpass -> voicePeaking
        source.connect(highpass);
        highpass.connect(lowpass);
        lowpass.connect(voicePeaking);

        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 256;
        analyserRef.current = analyser;
        voicePeaking.connect(analyser);

        const processor = audioCtx.createScriptProcessor(4096, 1, 1);
        scriptProcessorRef.current = processor;
        voicePeaking.connect(processor);

        const silentGain = audioCtx.createGain();
        silentGain.gain.value = 0;
        silentGainRef.current = silentGain;

        processor.connect(silentGain);
        silentGain.connect(audioCtx.destination);

        const maxPreRollCount = Math.max(10, Math.ceil((audioCtx.sampleRate * 0.9) / 4096));

      // 5. Audio Process Handler with Advanced Speech Discrimination
      processor.onaudioprocess = (e: AudioProcessingEvent) => {
        if (!isContinuousModeRef.current && voiceStateRef.current === "idle") {
          return;
        }

        const inputChannel = e.inputBuffer.getChannelData(0);
        const chunk = new Float32Array(inputChannel.length);
        chunk.set(inputChannel);

        // Calculate RMS of filtered speech signal
        let sumSq = 0;
        for (let i = 0; i < chunk.length; i++) {
          sumSq += chunk[i] * chunk[i];
        }
        const rms = Math.sqrt(sumSq / chunk.length);

        // Frequency analysis via AnalyserNode
        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(dataArray);

        // Vocal formant energy (bins 1 to 18: ~180Hz - 3400Hz at 48kHz)
        let vocalSum = 0;
        const maxVocalBin = Math.min(18, dataArray.length - 1);
        for (let i = 1; i <= maxVocalBin; i++) {
          vocalSum += dataArray[i];
        }
        const vocalEnergy = maxVocalBin > 0 ? vocalSum / maxVocalBin : 0;

        // High frequency noise energy (bins 24 to 64: ~4500Hz - 12000Hz - fan hiss & keyboard clicks)
        let highSum = 0;
        let highCount = 0;
        for (let i = 24; i <= 64 && i < dataArray.length; i++) {
          highSum += dataArray[i];
          highCount++;
        }
        const highNoise = highCount > 0 ? highSum / highCount : 0;

        // Visualizer waveform (24 bars)
        const sampleCount = 24;
        const step = Math.floor(dataArray.length / sampleCount) || 1;
        const levels = [];
        for (let i = 0; i < sampleCount; i++) {
          levels.push(Math.min(100, Math.max(12, Math.round(((dataArray[i * step] || 0) / 255) * 100))));
        }
        setAudioLevels(levels);

        // Update rolling pre-roll buffer (keeps last ~800-900ms)
        if (voiceStateRef.current === "listening" && !isSpeechTurnActiveRef.current) {
          preRollBuffersRef.current.push(chunk);
          if (preRollBuffersRef.current.length > maxPreRollCount) {
            preRollBuffersRef.current.shift();
          }
        }

        // BARGE-IN: while speaking, detect user interruption with nearby vocal energy
        if (voiceStateRef.current === "speaking") {
          bleedRmsRef.current = bleedRmsRef.current * 0.92 + rms * 0.08;

          const now = Date.now();
          const hasGracePeriodElapsed = now - sentenceStartTimestampRef.current > 220;

          const isUserBargeIn =
            hasGracePeriodElapsed &&
            ((rms >= 0.035 && rms >= bleedRmsRef.current * 1.8 && vocalEnergy >= 24) ||
              (vocalEnergy >= 30 && vocalEnergy > highNoise * 1.2));

          if (isUserBargeIn) {
            consecutiveBargeInFramesRef.current++;
            if (consecutiveBargeInFramesRef.current >= 2) {
              executeBargeIn();
              return;
            }
          } else {
            consecutiveBargeInFramesRef.current = Math.max(0, consecutiveBargeInFramesRef.current - 1);
          }
          return;
        }

        // LISTENING STATE: detect speech onset with adaptive noise floor and vocal formants
        if (voiceStateRef.current === "listening") {
          // Dynamic adaptive noise floor tracking when not actively speaking
          if (!isSpeechTurnActiveRef.current) {
            ambientNoiseFloorRef.current = ambientNoiseFloorRef.current * 0.96 + rms * 0.04;
          }
          // Dynamic threshold: nearby speech must rise clearly above the ambient noise floor
          const dynamicRmsThreshold = Math.max(0.016, Math.min(0.065, ambientNoiseFloorRef.current * 2.6 + 0.007));
          
          // Require both RMS above ambient floor AND strong vocal formant energy dominating high noise
          const isSpeech =
            rms >= dynamicRmsThreshold &&
            vocalEnergy >= 16 &&
            vocalEnergy > highNoise * 0.85;

          if (isSpeech) {
            speechEndTimestampRef.current = 0;
            consecutiveSpeechFramesRef.current++;
            // Require 3 consecutive speech frames (~250ms of sustained speech) before triggering
            // This prevents keyboard clicks, fan fluctuations, and momentary glitches from starting a turn!
            if (!isSpeechTurnActiveRef.current && consecutiveSpeechFramesRef.current >= 3) {
              isSpeechTurnActiveRef.current = true;
              speechDetectedRef.current = true;
              // Prepend pre-roll buffer so leading words (first syllable) are fully preserved
              recordedBuffersRef.current = [...preRollBuffersRef.current, chunk];
            } else if (isSpeechTurnActiveRef.current) {
              recordedBuffersRef.current.push(chunk);
            }

            if (silenceTimerRef.current) {
              clearTimeout(silenceTimerRef.current);
              silenceTimerRef.current = null;
            }
          } else {
            consecutiveSpeechFramesRef.current = 0;
            if (isSpeechTurnActiveRef.current) {
              recordedBuffersRef.current.push(chunk);
              if (!speechEndTimestampRef.current) {
                speechEndTimestampRef.current = Date.now();
              }
              if (!silenceTimerRef.current) {
                silenceTimerRef.current = setTimeout(() => {
                  submitCurrentSpeechTurn();
                }, 850); // 850ms natural cadence pause - ensures continuous speech, multi-clause inputs, and phone numbers are never cut off mid-utterance
              }
            }
          }
        }
      };

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
}, [executeBargeIn, submitCurrentSpeechTurn]);

  // User manually finishes speaking or VAD silence timer fires
  const finishSpeakingTurn = useCallback(() => {
    submitCurrentSpeechTurn();
  }, [submitCurrentSpeechTurn]);

  // Start continuous voice assistant session
  const startVoiceMode = useCallback(() => {
    cancelActiveTurn();
    isContinuousModeRef.current = true;
    setElapsedSeconds(0);

    // If previous lead was completed, start fresh from Name
    const isPriorComplete = currentLeadRef.current && (
      Boolean(currentLeadRef.current.name) &&
      Boolean(currentLeadRef.current.phone) &&
      Boolean(currentLeadRef.current.company) &&
      Boolean(currentLeadRef.current.loan_type) &&
      Boolean(currentLeadRef.current.loan_amount) &&
      Boolean(currentLeadRef.current.tenure_months)
    );
    if (isPriorComplete) {
      activeLeadIdRef.current = null;
      currentLeadRef.current = null;
      setExtractedLead(null);
      setSaveLeadSuccess(null);
      setSessionTurnCount(0);
      const initialPrompt = selectedLanguage === "hi"
        ? "नमस्ते! कृपया आपका शुभ नाम बताइए?"
        : selectedLanguage === "mr"
        ? "नमस्कार! कृपया आपले नाव सांगा?"
        : "Hello! May I have your name, please?";
      setAssistantResponseText(initialPrompt);
    }

    // Start session timer
    if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
    timerIntervalRef.current = setInterval(() => {
      setElapsedSeconds((prev) => prev + 1);
    }, 1000);

    startListeningTurn();
  }, [cancelActiveTurn, selectedLanguage, startListeningTurn]);

  // Stop continuous voice assistant session
  const stopVoiceMode = useCallback(() => {
    isContinuousModeRef.current = false;
    cancelActiveTurn();

    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
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
    setSaveLeadSuccess(null);
    setSessionTurnCount(0);
    setElapsedSeconds(0);
    setErrorMessage(null);

    const initialPrompt = selectedLanguage === "hi"
      ? "नमस्ते! कृपया आपका शुभ नाम बताइए?"
      : selectedLanguage === "mr"
      ? "नमस्कार! कृपया आपले नाव सांगा?"
      : "Hello! May I have your name, please?";
    setAssistantResponseText(initialPrompt);
  };

  // Run simulation preset directly
  const runSimulationTurn = async (presetText: string, presetLang: string) => {
    cancelActiveTurn();
    const currentTurnId = ++turnIdRef.current;
    isVoicePipelineActiveRef.current = true;
    setVoiceState("processing");
    setProcessingStage("extracting");
    setTranscript(presetText);
    setDetectedLanguage(presetLang.toLowerCase());

    const isUpdate = activeLeadIdRef.current !== null;
    const targetLang = presetLang.toLowerCase();

    try {
      const playSimulationAudio = (textToPlay: string, lang: string) => {
        setVoiceState("speaking");
        setProcessingStage("speaking");
        setIsTTSPlaying(true);

        if (!ttsAudioRef.current && typeof window !== "undefined") {
          ttsAudioRef.current = new Audio();
        }

        const audioPlayer = ttsAudioRef.current;
        if (!audioPlayer) {
          setIsTTSPlaying(false);
          setVoiceState("idle");
          setProcessingStage(null);
          return;
        }

        if (!sentenceAudioQueueRef.current) {
          sentenceAudioQueueRef.current = new SentenceAudioQueue(audioPlayer, {}, audioContextRef.current, {
            module: "module1",
            speaker: "simran",
          });
        }
        if (sentenceAudioQueueRef.current) {
          sentenceAudioQueueRef.current.setOptions({ module: "module1", speaker: "simran" });
          if (audioContextRef.current) {
            sentenceAudioQueueRef.current.setAudioContext(audioContextRef.current);
          }
        }

        sentenceAudioQueueRef.current.updateCallbacks({
          onSentenceStart: () => {
            if (turnIdRef.current !== currentTurnId) return;
            setVoiceState("speaking");
            setIsTTSPlaying(true);
          },
          onQueueComplete: () => {
            if (turnIdRef.current === currentTurnId) {
              setIsTTSPlaying(false);
              setVoiceState("idle");
              setProcessingStage(null);
            }
          },
          onError: () => {
            if (turnIdRef.current === currentTurnId) {
              setIsTTSPlaying(false);
              setVoiceState("idle");
              setProcessingStage(null);
            }
          },
        });

        sentenceAudioQueueRef.current.startNewTurn(currentTurnId);

        const tokenizer = new SentenceTokenizer();
        const sentences = tokenizer.feed(textToPlay);
        const trailing = tokenizer.flush();
        if (trailing) sentences.push(trailing);
        if (sentences.length === 0 && textToPlay.trim()) sentences.push(textToPlay.trim());

        for (const s of sentences) {
          sentenceAudioQueueRef.current.enqueueSentence(s, lang);
        }
        sentenceAudioQueueRef.current.markStreamComplete();
      };

      // Check greeting preset
      if (isGreetingOnly(presetText)) {
        const greetingSpoken = getNaturalGreeting(targetLang);
        setAssistantResponseText(greetingSpoken);
        playSimulationAudio(greetingSpoken, targetLang);
        return;
      }

      const extractRes = await fetch("/api/extract-lead", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          transcript: presetText,
          existing_lead: currentLeadRef.current || undefined,
          language: targetLang,
        }),
      });

      const extractData = await extractRes.json();
      const updatedLeadData: ExtractedLead = extractData.lead;

      // Persistence
      let savedRecord: ExtractedLead = updatedLeadData;
      if (activeLeadIdRef.current !== null) {
        const targetId = activeLeadIdRef.current;
        const updateRes = await fetch("/api/leads", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id: targetId, ...updatedLeadData }),
        });
        const updateData = await updateRes.json();
        savedRecord = updateData.lead || { id: targetId, ...updatedLeadData };
        setSaveLeadSuccess({
          id: targetId,
          message: `Lead #${targetId} successfully updated in PostgreSQL database!`,
          isUpdate: true,
        });
      } else {
        const createRes = await fetch("/api/leads", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(updatedLeadData),
        });
        const createData = await createRes.json();
        savedRecord = createData.lead;
        activeLeadIdRef.current = savedRecord.id || 1;
        setSaveLeadSuccess({
          id: savedRecord.id || 1,
          message: `Lead #${savedRecord.id || 1} successfully saved to PostgreSQL database!`,
          isUpdate: false,
        });
      }

      currentLeadRef.current = savedRecord;
      setExtractedLead(savedRecord);
      setSessionTurnCount((prev) => prev + 1);

      // Voice response
      const spokenText = generateSpokenResponseText(savedRecord, presetLang.toLowerCase(), isUpdate);
      setAssistantResponseText(spokenText);
      playSimulationAudio(spokenText, targetLang);
    } catch (err: any) {
      console.error("Simulation error:", err);
      setErrorMessage(err?.message || "Simulation failed.");
      setVoiceState("idle");
    } finally {
      isVoicePipelineActiveRef.current = false;
    }
  };

  const copyToClipboard = (text: string, fieldKey: string) => {
    navigator.clipboard?.writeText(text);
    setCopiedField(fieldKey);
    setTimeout(() => setCopiedField(null), 2000);
  };

  return (
    <div className="card overflow-hidden w-full">
      {/* Top Header */}
      <div className="card-header flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-[var(--radius)] bg-[var(--accent-soft)] text-[var(--accent)] flex items-center justify-center shrink-0">
            {voiceState === "listening" ? (
              <Radio className="h-4 w-4 text-[var(--danger)] animate-pulse" />
            ) : voiceState === "speaking" ? (
              <Volume2 className="h-4 w-4 text-[var(--accent)] animate-bounce" />
            ) : voiceState === "processing" ? (
              <Loader2 className="h-4 w-4 text-[var(--warning)] animate-spin" />
            ) : (
              <Mic className="h-4 w-4" />
            )}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="card-title">Live Call Voice Copilot</span>

              {/* Dynamic Status Badge */}
              {voiceState === "idle" && (
                <span className="status-pill status-pill-neutral">Standby · Voice Mode Ready</span>
              )}
              {voiceState === "listening" && (
                <span className="status-pill status-pill-danger flex items-center gap-1.5">
                  <span className="h-1.5 w-1.5 rounded-full bg-red-500 animate-ping" />
                  Listening Live · {formatTime(elapsedSeconds)}
                </span>
              )}
              {voiceState === "processing" && (
                <span className="status-pill status-pill-warning flex items-center gap-1.5">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  {processingStage === "transcribing"
                    ? "Deepgram STT..."
                    : processingStage === "extracting"
                    ? "AI Lead Extraction..."
                    : "Saving to PostgreSQL..."}
                </span>
              )}
              {voiceState === "speaking" && (
                <span className="status-pill status-pill-success flex items-center gap-1.5">
                  <Volume2 className="h-3 w-3 animate-pulse" />
                  Speaking Confirmation
                </span>
              )}

              {/* Active Lead ID Indicator */}
              {activeLeadIdRef.current && (
                <span className="badge badge-primary font-mono text-[10px]">
                  Lead #{activeLeadIdRef.current} Active
                </span>
              )}
            </div>
            <p className="card-subtitle">
              Continuous Voice Assistant: Speaks naturally → extracts facts → updates PostgreSQL lead without duplicates
            </p>
          </div>
        </div>

        {/* Right Header Navigation: Language Selector & Sub-Tabs */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Multilingual Selector */}
          <div className="flex items-center gap-1 p-0.5 rounded-[var(--radius)] bg-[var(--surface-2)] border border-[var(--border)] text-xs">
            <button
              onClick={() => setSelectedLanguage("auto")}
              className={`px-2 py-1 rounded-[var(--radius-sm)] transition-colors cursor-pointer ${
                selectedLanguage === "auto"
                  ? "bg-[var(--accent)] text-white font-medium"
                  : "text-[var(--ink-muted)] hover:text-[var(--ink)]"
              }`}
              title={`Auto Language Detection (Active: ${detectedLanguage.toUpperCase()})`}
            >
              Auto ({detectedLanguage.toUpperCase()})
            </button>
            <button
              onClick={() => setSelectedLanguage("en")}
              className={`px-2 py-1 rounded-[var(--radius-sm)] transition-colors cursor-pointer ${
                selectedLanguage === "en"
                  ? "bg-[var(--accent)] text-white font-medium"
                  : "text-[var(--ink-muted)] hover:text-[var(--ink)]"
              }`}
            >
              EN
            </button>
            <button
              onClick={() => setSelectedLanguage("hi")}
              className={`px-2 py-1 rounded-[var(--radius-sm)] transition-colors cursor-pointer ${
                selectedLanguage === "hi"
                  ? "bg-[var(--accent)] text-white font-medium"
                  : "text-[var(--ink-muted)] hover:text-[var(--ink)]"
              }`}
            >
              हिंदी
            </button>
            <button
              onClick={() => setSelectedLanguage("mr")}
              className={`px-2 py-1 rounded-[var(--radius-sm)] transition-colors cursor-pointer ${
                selectedLanguage === "mr"
                  ? "bg-[var(--accent)] text-white font-medium"
                  : "text-[var(--ink-muted)] hover:text-[var(--ink)]"
              }`}
            >
              मराठी
            </button>
          </div>

          {/* Sub-Tab Switcher */}
          <div className="flex items-center gap-1 p-1 rounded-[var(--radius)] bg-[var(--surface-2)] border border-[var(--border)]">
            <button
              onClick={() => setActiveSubTab("assistant")}
              className={`btn btn-sm cursor-pointer gap-1.5 ${
                activeSubTab === "assistant"
                  ? "btn-primary"
                  : "btn-secondary border-transparent bg-transparent"
              }`}
            >
              <Mic className="h-3.5 w-3.5" />
              <span>Voice Assistant</span>
            </button>
            <button
              onClick={() => setActiveSubTab("simulation")}
              className={`btn btn-sm cursor-pointer gap-1.5 ${
                activeSubTab === "simulation"
                  ? "btn-primary"
                  : "btn-secondary border-transparent bg-transparent"
              }`}
            >
              <Headphones className="h-3.5 w-3.5" />
              <span>Call Presets</span>
            </button>
          </div>
        </div>
      </div>

      {/* Error Notice */}
      {errorMessage && (
        <div className="m-4 p-3.5 rounded-[var(--radius)] bg-[var(--danger-soft)] border border-[var(--danger)]/30 flex items-start gap-2.5 text-xs text-[var(--danger)]">
          <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
          <div className="flex-1">
            <span className="font-semibold text-white">Voice Assistant Notice: </span>
            <span>{errorMessage}</span>
          </div>
          <button
            onClick={() => setErrorMessage(null)}
            className="text-[var(--ink-muted)] hover:text-white text-xs font-mono px-1.5 cursor-pointer"
          >
            &times;
          </button>
        </div>
      )}

      {/* SUB-TAB 1: CONTINUOUS VOICE ASSISTANT */}
      {activeSubTab === "assistant" && (
        <div className="card-body space-y-6">
          {/* Central Interactive Voice Orb Card */}
          <div className="p-8 rounded-[var(--radius-lg)] bg-[var(--surface-2)] border border-[var(--border)] flex flex-col items-center justify-center text-center relative overflow-hidden">
            {/* Background Glow effects */}
            {voiceState === "listening" && (
              <div className="absolute inset-0 bg-red-500/10 pointer-events-none animate-pulse" />
            )}
            {voiceState === "speaking" && (
              <div className="absolute inset-0 bg-[var(--accent-soft)] pointer-events-none animate-pulse" />
            )}

            {/* Interactive Voice Orb */}
            <div className="relative mb-5 z-10">
              <button
                onClick={() => {
                  if (voiceState === "idle") {
                    startVoiceMode();
                  } else if (voiceState === "listening") {
                    finishSpeakingTurn();
                  } else if (voiceState === "speaking") {
                    // Instant Barge-In: interrupt assistant speech and resume listening immediately
                    stopAllAudioPlayback();
                    if (isContinuousModeRef.current) {
                      setVoiceState("listening");
                      startListeningTurn();
                    } else {
                      setVoiceState("idle");
                    }
                  }
                }}
                disabled={voiceState === "processing"}
                className={`relative w-24 h-24 sm:w-28 sm:h-28 rounded-full flex items-center justify-center transition-all duration-300 shadow-xl cursor-pointer ${
                  voiceState === "idle"
                    ? "bg-gradient-to-tr from-[var(--accent)] to-[var(--purple)] hover:scale-105"
                    : voiceState === "listening"
                    ? "bg-gradient-to-tr from-red-600 to-pink-600 animate-pulse scale-105 ring-4 ring-red-500/30"
                    : voiceState === "processing"
                    ? "bg-gradient-to-tr from-amber-600 to-yellow-500 ring-4 ring-amber-500/30"
                    : "bg-gradient-to-tr from-[var(--accent)] to-teal-500 scale-105 ring-4 ring-[var(--accent)]/30"
                }`}
                title={
                  voiceState === "idle"
                    ? "Click to start Continuous Voice Mode"
                    : voiceState === "listening"
                    ? "Click to finish speaking turn immediately"
                    : voiceState === "speaking"
                    ? "Click to interrupt assistant (Barge-In)"
                    : undefined
                }
              >
                {voiceState === "idle" && <Mic className="h-10 w-10 text-white" />}
                {voiceState === "listening" && <Radio className="h-10 w-10 text-white animate-pulse" />}
                {voiceState === "processing" && <Loader2 className="h-10 w-10 text-white animate-spin" />}
                {voiceState === "speaking" && <Volume2 className="h-10 w-10 text-white animate-bounce" />}
              </button>

              {/* Pulsing halo rings when listening */}
              {voiceState === "listening" && (
                <div className="absolute inset-0 rounded-full border-2 border-red-500/40 animate-ping pointer-events-none" />
              )}
            </div>

            {/* Status Headings & Instructions */}
            <h3 className="text-base font-bold text-[var(--ink)] mb-1 z-10">
              {voiceState === "idle" && "Continuous Voice Assistant Standby"}
              {voiceState === "listening" && "Listening for your voice..."}
              {voiceState === "processing" && (
                processingStage === "transcribing"
                  ? "Transcribing Speech (Deepgram)..."
                  : processingStage === "extracting"
                  ? "Extracting Lead Facts (OpenRouter)..."
                  : "Saving / Updating PostgreSQL Lead..."
              )}
              {voiceState === "speaking" && "Assistant Speaking Confirmation..."}
            </h3>

            <p className="text-xs text-[var(--ink-soft)] max-w-md z-10">
              {voiceState === "idle" &&
                "Click the orb once to start. Speak naturally about prospect details, loan types, amounts, or updates. Automatically listens again after each answer."}
              {voiceState === "listening" &&
                "Speak clearly into your microphone. When you pause or finish, the assistant automatically captures your speech and updates the lead."}
              {voiceState === "processing" &&
                "Understanding new speech details, updating existing lead facts, and saving directly to PostgreSQL database."}
              {voiceState === "speaking" &&
                "Assistant is speaking confirmation. Microphone is muted during playback to prevent feedback. Will auto-listen when done."}
            </p>

            {/* Waveform visualizer */}
            <div className="w-full max-w-md h-12 bg-[var(--surface)] rounded-[var(--radius)] border border-[var(--border)] p-2 flex items-center justify-center gap-1.5 mt-5 z-10">
              {voiceState === "listening" ? (
                audioLevels.map((level, idx) => (
                  <div
                    key={idx}
                    className="w-1.5 sm:w-2 bg-gradient-to-t from-[var(--accent)] to-[var(--purple)] rounded-full transition-all duration-75"
                    style={{ height: `${level}%` }}
                  />
                ))
              ) : voiceState === "speaking" ? (
                audioLevels.map((level, idx) => (
                  <div
                    key={idx}
                    className="w-1.5 sm:w-2 bg-[var(--success)] rounded-full transition-all duration-75"
                    style={{ height: `${Math.min(100, Math.max(20, (level * 1.5)))}%` }}
                  />
                ))
              ) : (
                audioLevels.map((_, idx) => (
                  <div
                    key={idx}
                    className="w-1.5 sm:w-2 bg-[var(--border)] rounded-full"
                    style={{ height: `${16 + (idx % 3) * 6}%` }}
                  />
                ))
              )}
            </div>

            {/* Action Buttons Row */}
            <div className="flex flex-wrap items-center justify-center gap-3 mt-5 z-10">
              {voiceState === "idle" ? (
                <button
                  onClick={startVoiceMode}
                  className="btn btn-gradient btn-lg cursor-pointer gap-2"
                >
                  <Mic className="h-4 w-4" />
                  <span>Start Continuous Voice Mode</span>
                </button>
              ) : (
                <>
                  {voiceState === "listening" && (
                    <button
                      onClick={finishSpeakingTurn}
                      className="btn btn-secondary btn-sm cursor-pointer gap-1.5"
                    >
                      <Square className="h-3.5 w-3.5 fill-current" />
                      <span>Finish Speaking Now</span>
                    </button>
                  )}
                  <button
                    onClick={stopVoiceMode}
                    className="btn btn-danger btn-sm cursor-pointer gap-1.5"
                  >
                    <Square className="h-3.5 w-3.5" />
                    <span>Stop Voice Mode</span>
                  </button>
                  <button
                    onClick={resetLeadSession}
                    className="btn btn-secondary btn-sm cursor-pointer gap-1.5"
                    title="Start a new prospect session and clear active lead ID"
                  >
                    <RotateCcw className="h-3.5 w-3.5" />
                    <span>New Lead (Reset)</span>
                  </button>
                </>
              )}
            </div>
          </div>

          {/* Assistant Voice Response Bubble (Shows confirmation speech) */}
          {assistantResponseText && (
            <div className="rounded-[var(--radius)] bg-[var(--surface-2)] border border-[var(--border)] p-4 space-y-2 animate-in fade-in duration-200">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Volume2 className="h-4 w-4 text-[var(--accent)]" />
                  <span className="text-xs font-bold text-[var(--ink)]">Assistant Confirmation Response</span>
                  {isTTSPlaying && (
                    <span className="status-pill status-pill-success text-[10px] animate-pulse">
                      Speaking Live
                    </span>
                  )}
                </div>
                <span className="badge badge-primary font-mono text-[10px]">
                  {detectedLanguage.toUpperCase()}
                </span>
              </div>
              <p className="text-xs sm:text-sm text-[var(--ink)] leading-relaxed bg-[var(--surface)] p-3 rounded-[var(--radius)] border border-[var(--border)]">
                "{assistantResponseText}"
              </p>
            </div>
          )}

          {/* Live Speech Transcript Bubble */}
          {transcript && (
            <div className="rounded-[var(--radius)] bg-[var(--surface)] border border-[var(--border)] p-4 space-y-2">
              <div className="flex items-center justify-between border-b border-[var(--border)] pb-2">
                <div className="flex items-center gap-2">
                  <FileText className="h-4 w-4 text-[var(--success)]" />
                  <span className="text-xs font-semibold text-[var(--ink)]">Latest User Speech (Turn #{sessionTurnCount})</span>
                  <span className="badge badge-secondary font-mono text-[10px]">
                    {detectedLanguage.toUpperCase()}
                  </span>
                </div>
                <button
                  onClick={() => copyToClipboard(transcript, "transcript")}
                  className="btn btn-secondary btn-sm text-[10px] gap-1 cursor-pointer"
                >
                  {copiedField === "transcript" ? (
                    <>
                      <Check className="h-3 w-3 text-[var(--success)]" />
                      <span>Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy className="h-3 w-3" />
                      <span>Copy</span>
                    </>
                  )}
                </button>
              </div>
              <p className="text-xs text-[var(--ink-soft)] italic select-text">
                "{transcript}"
              </p>
            </div>
          )}

          {/* STRUCTURED LEAD CARD (Live in PostgreSQL) */}
          {extractedLead && (
            <div className="rounded-[var(--radius-lg)] bg-[var(--surface)] border border-[var(--border)] p-5 space-y-4 shadow-sm animate-in fade-in duration-200">
              {/* Header with DB Status */}
              <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-[var(--border)]">
                <div className="flex items-center gap-2.5">
                  <div className="h-8 w-8 rounded-[var(--radius)] bg-[var(--accent-soft)] flex items-center justify-center text-[var(--accent)] shrink-0">
                    <Database className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-bold text-[var(--ink)]">
                        {extractedLead.name ? `${extractedLead.name}'s Lead Details` : "Active Lead Record"}
                      </span>
                      {saveLeadSuccess && (
                        <span className="status-pill status-pill-success text-[10px]">
                          {saveLeadSuccess.isUpdate ? "Updated in PostgreSQL" : "Saved in PostgreSQL"} (Lead #{saveLeadSuccess.id})
                        </span>
                      )}
                    </div>
                    <span className="text-[11px] text-[var(--ink-muted)]">
                      {activeLeadIdRef.current
                        ? `PostgreSQL Lead #${activeLeadIdRef.current} · Multi-turn updates accumulate to this record`
                        : "Validated with Pydantic and ready for CRM integration"}
                    </span>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => copyToClipboard(JSON.stringify(extractedLead, null, 2), "lead_json")}
                    className="btn btn-secondary btn-sm gap-1.5 cursor-pointer text-xs"
                    title="Copy Lead as JSON"
                  >
                    {copiedField === "lead_json" ? (
                      <>
                        <Check className="h-3.5 w-3.5 text-[var(--success)]" />
                        <span className="text-[var(--success)]">JSON Copied</span>
                      </>
                    ) : (
                      <>
                        <Copy className="h-3.5 w-3.5" />
                        <span>Copy JSON</span>
                      </>
                    )}
                  </button>
                  <button
                    onClick={resetLeadSession}
                    className="btn btn-secondary btn-sm gap-1.5 cursor-pointer text-xs"
                    title="Finish this lead and start a new prospect"
                  >
                    <UserCheck className="h-3.5 w-3.5 text-[var(--accent)]" />
                    <span>New Lead</span>
                  </button>
                </div>
              </div>

              {/* 9 Standard Lead Fields Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                {/* 1. Name */}
                <div className="p-3 rounded-[var(--radius)] bg-[var(--surface-2)] border border-[var(--border)]">
                  <span className="text-[10px] text-[var(--ink-muted)] uppercase font-semibold flex items-center gap-1.5 mb-1">
                    <UserCheck className="h-3 w-3 text-[var(--accent)]" />
                    Prospect Name
                  </span>
                  <span className="text-xs font-bold text-[var(--ink)] block truncate">
                    {extractedLead.name || <span className="text-[var(--ink-muted)] italic font-normal">Pending mention...</span>}
                  </span>
                </div>

                {/* 2. Company & Role */}
                <div className="p-3 rounded-[var(--radius)] bg-[var(--surface-2)] border border-[var(--border)]">
                  <span className="text-[10px] text-[var(--ink-muted)] uppercase font-semibold flex items-center gap-1.5 mb-1">
                    <Building className="h-3 w-3 text-[var(--purple)]" />
                    Company &amp; Role
                  </span>
                  <span className="text-xs font-semibold text-[var(--ink)] block truncate">
                    {[extractedLead.role, extractedLead.company].filter(Boolean).join(" at ") || (
                      <span className="text-[var(--ink-muted)] italic font-normal">Pending mention...</span>
                    )}
                  </span>
                </div>

                {/* 3. Phone & Email */}
                <div className="p-3 rounded-[var(--radius)] bg-[var(--surface-2)] border border-[var(--border)]">
                  <span className="text-[10px] text-[var(--ink-muted)] uppercase font-semibold flex items-center gap-1.5 mb-1">
                    <Phone className="h-3 w-3 text-[var(--success)]" />
                    Contact Info
                  </span>
                  <span className="text-xs font-mono text-[var(--ink)] block truncate">
                    {[extractedLead.phone, extractedLead.email].filter(Boolean).join(" · ") || (
                      <span className="text-[var(--ink-muted)] italic font-normal font-sans">Pending mention...</span>
                    )}
                  </span>
                </div>

                {/* 4. Loan Type */}
                <div className="p-3 rounded-[var(--radius)] bg-[var(--surface-2)] border border-[var(--border)]">
                  <span className="text-[10px] text-[var(--ink-muted)] uppercase font-semibold flex items-center gap-1.5 mb-1">
                    <Briefcase className="h-3 w-3 text-[var(--accent)]" />
                    Loan Type
                  </span>
                  <span className="text-xs font-semibold text-[var(--accent)] block truncate">
                    {extractedLead.loan_type || <span className="text-[var(--ink-muted)] italic font-normal">Pending mention...</span>}
                  </span>
                </div>

                {/* 5. Loan Amount */}
                <div className="p-3 rounded-[var(--radius)] bg-[var(--surface-2)] border border-[var(--border)]">
                  <span className="text-[10px] text-[var(--ink-muted)] uppercase font-semibold flex items-center gap-1.5 mb-1">
                    <DollarSign className="h-3 w-3 text-[var(--success)]" />
                    Loan Amount
                  </span>
                  <span className="text-xs font-bold text-[var(--success)] block">
                    {extractedLead.loan_amount !== null && extractedLead.loan_amount !== undefined
                      ? `$${extractedLead.loan_amount.toLocaleString()}`
                      : <span className="text-[var(--ink-muted)] italic font-normal">Pending mention...</span>}
                  </span>
                </div>

                {/* 6. Tenure */}
                <div className="p-3 rounded-[var(--radius)] bg-[var(--surface-2)] border border-[var(--border)]">
                  <span className="text-[10px] text-[var(--ink-muted)] uppercase font-semibold flex items-center gap-1.5 mb-1">
                    <Calendar className="h-3 w-3 text-[var(--warning)]" />
                    Tenure
                  </span>
                  <span className="text-xs font-semibold text-[var(--ink)] block">
                    {extractedLead.tenure_months
                      ? `${extractedLead.tenure_months} months`
                      : <span className="text-[var(--ink-muted)] italic font-normal">Pending mention...</span>}
                  </span>
                </div>
              </div>

              {/* 7. Notes */}
              {extractedLead.notes && (
                <div className="p-3 rounded-[var(--radius)] bg-[var(--surface-2)] border border-[var(--border)]">
                  <span className="text-[10px] text-[var(--ink-muted)] uppercase font-semibold block mb-1">
                    Sales Notes &amp; Intent
                  </span>
                  <p className="text-xs text-[var(--ink-soft)] leading-relaxed">
                    {extractedLead.notes}
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* SUB-TAB 2: CALL PRESETS (Instant Multi-Turn Simulation) */}
      {activeSubTab === "simulation" && (
        <div className="card-body space-y-5">
          <div className="p-4 rounded-[var(--radius)] bg-[var(--surface-2)] border border-[var(--border)]">
            <h4 className="text-xs font-bold text-[var(--ink)] mb-1 flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-[var(--accent)]" />
              Multilingual Multi-Turn Test Presets
            </h4>
            <p className="text-xs text-[var(--ink-muted)]">
              Click a preset below to simulate a live voice turn. Notice how Turn 2 automatically updates the same lead record without creating duplicate database rows!
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {SIMULATION_PRESETS.map((preset) => (
              <div
                key={preset.id}
                className="p-4 rounded-[var(--radius)] bg-[var(--surface)] border border-[var(--border)] hover:border-[var(--accent)]/40 transition-all flex flex-col justify-between space-y-3"
              >
                <div>
                  <div className="flex items-center justify-between gap-2 mb-2">
                    <span className="text-xs font-bold text-[var(--ink)]">{preset.title}</span>
                    <span className="badge badge-primary font-mono text-[10px]">{preset.lang}</span>
                  </div>
                  <p className="text-xs text-[var(--ink-soft)] leading-relaxed italic bg-[var(--surface-2)] p-2.5 rounded-[var(--radius-sm)] border border-[var(--border)]">
                    "{preset.text}"
                  </p>
                </div>

                <button
                  onClick={() => runSimulationTurn(preset.text, preset.lang)}
                  disabled={voiceState === "processing"}
                  className="btn btn-secondary btn-sm w-full cursor-pointer gap-2"
                >
                  <Play className="h-3 w-3 fill-current text-[var(--accent)]" />
                  <span>Run Turn Simulation</span>
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
