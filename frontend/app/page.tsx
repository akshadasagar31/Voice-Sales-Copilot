import Link from "next/link";
import {
  Mic,
  Sparkles,
  ArrowRight,
  ShieldCheck,
  Zap,
  CheckCircle2,
  Headphones,
  BookOpen,
  Lock,
} from "lucide-react";

export default function Home() {
  return (
    <div className="min-h-screen bg-[var(--bg)] text-[var(--ink)] flex flex-col justify-between selection:bg-[var(--accent-soft-strong)] selection:text-white" data-bs-theme="dark">
      {/* Top Navigation — CallNow Topbar Style */}
      <header className="w-full px-6 sm:px-10 lg:px-12 xl:px-14 py-4 flex items-center justify-between border-b border-[var(--border)]">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-[var(--accent)] text-black flex items-center justify-center font-bold shadow-md shadow-indigo-500/20">
            <Mic className="h-4 w-4 text-black" />
          </div>
          <div className="flex items-center gap-2">
            <span className="font-bold text-base tracking-tight text-[var(--ink)]">VoiceCopilot</span>
            <span className="text-[10px] uppercase font-semibold px-2 py-0.5 rounded-full bg-[var(--accent-soft)] text-[var(--accent)] border border-[var(--accent-soft-strong)]">
              AI Sales Copilot
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2.5">
          <Link
            href="/login"
            className="btn btn-sm btn-ghost text-[var(--ink-soft)] hover:text-white"
          >
            Sign In
          </Link>
          <Link
            href="/register"
            className="btn btn-sm btn-gradient shadow-md"
          >
            Get Started Free
          </Link>
        </div>
      </header>

      {/* Hero Section — CallNow Aesthetics */}
      <main className="flex-1 w-full flex flex-col items-center justify-center px-6 sm:px-10 lg:px-12 xl:px-14 py-12 sm:py-16 text-center">
        {/* Status Pill Badge */}
        <div className="status-pill primary mb-6">
          <Sparkles className="h-3.5 w-3.5 mr-1" />
          <span>Real-time voice intelligence, Deepgram STT &amp; RAG playbooks</span>
        </div>

        {/* Heading */}
        <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tight text-[var(--ink)] max-w-3xl leading-[1.15]">
          Empower Every Sales Call with{" "}
          <span className="bg-gradient-to-r from-[var(--accent)] via-sky-400 to-emerald-400 bg-clip-text text-transparent">
            Live AI Guidance
          </span>
        </h1>

        <p className="mt-5 text-base sm:text-lg text-[var(--ink-soft)] max-w-2xl leading-relaxed">
          Record calls, transcribe speech in real time with Deepgram, detect customer objections in milliseconds, and instantly query 1536-dimensional Pinecone vector playbooks.
        </p>

        {/* Action Buttons */}
        <div className="mt-8 flex flex-col sm:flex-row items-center gap-3 sm:gap-4 w-full sm:w-auto">
          <Link
            href="/dashboard"
            className="btn btn-lg btn-gradient w-full sm:w-auto shadow-lg shadow-indigo-500/20"
          >
            <span>Launch Copilot Dashboard</span>
            <ArrowRight className="h-4 w-4" />
          </Link>

          <Link
            href="/login"
            className="btn btn-lg btn-secondary w-full sm:w-auto"
          >
            <Lock className="h-4 w-4 text-[var(--accent)]" />
            <span>Sign In to Account</span>
          </Link>

          <Link
            href="/register"
            className="btn btn-lg btn-outline-primary w-full sm:w-auto"
          >
            <span>Create New Account</span>
          </Link>
        </div>

        {/* Feature Pillars Grid — CallNow Metric Cards */}
        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-5 text-left w-full">
          <div className="card p-5">
            <div className="metric-card-icon mb-3">
              <Headphones className="h-4 w-4 text-[var(--accent)]" />
            </div>
            <h3 className="text-sm font-semibold text-[var(--ink)]">Voice-to-CRM &amp; Deepgram STT</h3>
            <p className="text-xs text-[var(--ink-soft)] mt-1.5 leading-relaxed">
              In-browser MediaRecorder captures audio notes and automatically streams to Deepgram Nova-2 with sub-second turnaround.
            </p>
          </div>

          <div className="card p-5">
            <div className="metric-card-icon info mb-3">
              <Zap className="h-4 w-4 text-sky-400" />
            </div>
            <h3 className="text-sm font-semibold text-[var(--ink)]">Sub-Second Battlecards</h3>
            <p className="text-xs text-[var(--ink-soft)] mt-1.5 leading-relaxed">
              Detect customer objections and pricing pushbacks; surface compliance guidelines and counter-narratives in &lt;300ms.
            </p>
          </div>

          <div className="card p-5">
            <div className="metric-card-icon success mb-3">
              <BookOpen className="h-4 w-4 text-emerald-400" />
            </div>
            <h3 className="text-sm font-semibold text-[var(--ink)]">1536-Dim RAG Playbooks</h3>
            <p className="text-xs text-[var(--ink-soft)] mt-1.5 leading-relaxed">
              Upload compliance manuals and sales battlecards into Pinecone vector storage with high-recall hybrid lexical reranking.
            </p>
          </div>
        </div>
      </main>

      {/* Footer — CallNow Clean Style */}
      <footer className="w-full px-6 sm:px-10 lg:px-12 xl:px-14 py-6 border-t border-[var(--border)] text-center text-xs text-[var(--ink-muted)]">
        <div className="flex flex-wrap items-center justify-center gap-6 mb-2">
          <span className="status-pill success">
            <ShieldCheck className="h-3.5 w-3.5 mr-1 text-emerald-400" /> SOC2 Type II Certified
          </span>
          <span className="status-pill info">
            <CheckCircle2 className="h-3.5 w-3.5 mr-1 text-sky-400" /> HIPAA Compliant Architecture
          </span>
        </div>
        <p className="mt-2">© {new Date().getFullYear()} Voice Sales &amp; Knowledge Copilot. Designed with CallNow CRM principles.</p>
      </footer>
    </div>
  );
}
