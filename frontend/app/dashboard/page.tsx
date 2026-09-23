"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Mic,
  MessageSquare,
  ArrowRight,
  ArrowLeft,
  ChevronDown,
  User,
  Settings,
  LogOut,
  RotateCcw,
  Database,
  Filter,
  BookOpen,
  X,
  CheckCircle2,
  Volume2,
  ShieldCheck,
  Globe,
} from "lucide-react";
import KnowledgeAssistant from "../components/KnowledgeAssistant";
import KnowledgePlaybooks from "../components/KnowledgePlaybooks";
import LiveCallVoiceCopilot, { ExtractedLead } from "../components/LiveCallVoiceCopilot";

export default function DashboardPage() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<string>("dashboard");
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const [profileModalOpen, setProfileModalOpen] = useState(false);
  const [settingsModalOpen, setSettingsModalOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  const userMenuRef = useRef<HTMLDivElement>(null);

  const [currentUser, setCurrentUser] = useState<{
    id?: number;
    name: string;
    email: string;
    company?: string;
    role?: string;
  }>({
    name: "",
    email: "",
    company: "",
    role: "",
  });

  const [crmLeads, setCrmLeads] = useState<ExtractedLead[]>([]);
  const [isLoadingLeads, setIsLoadingLeads] = useState<boolean>(true);

  // Close user menu on outside click
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (userMenuRef.current && !userMenuRef.current.contains(event.target as Node)) {
        setUserMenuOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const fetchLeads = useCallback(async () => {
    try {
      setIsLoadingLeads(true);
      const res = await fetch("/api/leads?limit=20");
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data?.leads)) {
          setCrmLeads(data.leads);
        }
      }
    } catch (err) {
      console.error("Failed to load leads from CRM:", err);
    } finally {
      setIsLoadingLeads(false);
    }
  }, []);

  useEffect(() => {
    fetchLeads();
  }, [fetchLeads]);

  useEffect(() => {
    fetch("/api/auth/me")
      .then((res) => {
        if (!res.ok) {
          router.push("/login");
          return null;
        }
        return res.json();
      })
      .then((data) => {
        if (data?.authenticated && data.user) {
          setCurrentUser({
            id: data.user.id,
            name: data.user.name || "Sales User",
            email: data.user.email || "",
            company: data.user.company || "",
            role: data.user.role || "Senior AE",
          });
        }
      })
      .catch(() => {
        // Fallback to defaults gracefully
      });
  }, [router]);

  const handleLogout = async () => {
    try {
      await fetch("/api/auth/logout", {
        method: "POST",
        credentials: "include",
      });
    } catch {
      // ignore
    }
    window.location.href = "/";
  };

  const getInitials = (name: string) => {
    if (!name || !name.trim()) return "U";
    const parts = name.trim().split(" ");
    if (parts.length >= 2) {
      return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
    }
    return name.slice(0, 2).toUpperCase();
  };

  useEffect(() => {
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      const tab = params.get("tab");
      if (tab && ["dashboard", "live-call", "assistant", "knowledge"].includes(tab)) {
        setActiveTab(tab);
      }
    }
  }, []);

  const filteredLeads = crmLeads.filter((lead) => {
    if (!searchQuery.trim()) return true;
    const q = searchQuery.toLowerCase();
    return Boolean(
      lead.name?.toLowerCase().includes(q) ||
      lead.company?.toLowerCase().includes(q) ||
      lead.role?.toLowerCase().includes(q) ||
      lead.loan_type?.toLowerCase().includes(q) ||
      lead.notes?.toLowerCase().includes(q) ||
      lead.email?.toLowerCase().includes(q) ||
      lead.phone?.toLowerCase().includes(q)
    );
  });

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#fbfcfa] via-[#f8faf9] to-[#f3f7f5] text-slate-800 flex flex-col relative selection:bg-teal-100 selection:text-teal-900 font-sans">
      {/* =====================================================================
          TOP NAVIGATION BAR (Clean, Minimal, SaaS design matching reference)
          ===================================================================== */}
      <header className="sticky top-0 z-40 bg-white/95 backdrop-blur-md border-b border-slate-100/90 shadow-[0_1px_3px_rgba(0,0,0,0.02)] w-full">
        <div className="w-full px-6 sm:px-10 lg:px-12 xl:px-14 h-16 flex items-center justify-between">
          {/* Left: Brand Logo & Title */}
          <button
            onClick={() => setActiveTab("dashboard")}
            className="flex items-center gap-2.5 text-inherit no-underline cursor-pointer group text-left"
          >
            <div className="text-teal-600 flex items-center justify-center group-hover:scale-105 transition-transform">
              <svg
                className="w-7 h-7"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
                <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                <line x1="12" x2="12" y1="19" y2="22" />
              </svg>
            </div>
            <span className="text-lg font-bold tracking-tight text-slate-900">
              Voice Sales Copilot
            </span>
          </button>

          {/* Right: Navigation Items & User Menu */}
          <div className="flex items-center gap-6 sm:gap-8">
            {/* Dashboard Nav Item */}
            <button
              onClick={() => setActiveTab("dashboard")}
              className={`flex items-center gap-2 text-sm font-semibold py-5 relative transition-colors cursor-pointer ${
                activeTab === "dashboard"
                  ? "text-teal-600"
                  : "text-slate-500 hover:text-slate-800"
              }`}
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
                <path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z" />
              </svg>
              <span>Dashboard</span>
              {activeTab === "dashboard" && (
                <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-teal-600 rounded-full" />
              )}
            </button>

            {/* Live Copilot Nav Item */}
            <button
              onClick={() => setActiveTab("live-call")}
              className={`flex items-center gap-2 text-sm font-semibold py-5 relative transition-colors cursor-pointer ${
                activeTab === "live-call"
                  ? "text-teal-600"
                  : "text-slate-500 hover:text-slate-800"
              }`}
            >
              <Mic className="w-4 h-4" />
              <span>Live Copilot</span>
              {activeTab === "live-call" && (
                <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-teal-600 rounded-full" />
              )}
            </button>

            {/* Ask Assistant Nav Item */}
            <button
              onClick={() => setActiveTab("assistant")}
              className={`flex items-center gap-2 text-sm font-semibold py-5 relative transition-colors cursor-pointer ${
                activeTab === "assistant"
                  ? "text-teal-600"
                  : "text-slate-500 hover:text-slate-800"
              }`}
            >
              <MessageSquare className="w-4 h-4" />
              <span>Ask Assistant</span>
              {activeTab === "assistant" && (
                <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-teal-600 rounded-full" />
              )}
            </button>

            {/* Vertical Divider */}
            <div className="h-5 w-px bg-slate-200" />

            {/* User Avatar & Dropdown */}
            <div className="relative" ref={userMenuRef}>
              <button
                onClick={() => setUserMenuOpen((prev) => !prev)}
                className="flex items-center gap-1.5 p-1 rounded-full hover:bg-slate-100/80 transition cursor-pointer"
                aria-expanded={userMenuOpen}
              >
                <div className="w-8 h-8 rounded-full bg-teal-600 text-white font-bold text-xs flex items-center justify-center shadow-xs">
                  {getInitials(currentUser.name)}
                </div>
                <ChevronDown
                  className={`w-3.5 h-3.5 text-slate-400 transition-transform duration-150 ${
                    userMenuOpen ? "rotate-180" : ""
                  }`}
                />
              </button>

              {/* User Dropdown Menu */}
              {userMenuOpen && (
                <div className="absolute right-0 top-full mt-2 w-56 rounded-2xl bg-white shadow-xl border border-slate-100 py-2 z-50 text-slate-700 animate-in fade-in zoom-in-95 duration-100">
                  {/* User Info Header */}
                  <div className="px-4 py-2.5 border-b border-slate-100">
                    <p className="text-xs font-bold text-slate-900 truncate">{currentUser.name}</p>
                    <p className="text-[11px] text-slate-500 truncate">
                      {currentUser.email || "sales.rep@enterprise.ai"}
                    </p>
                    <span className="inline-block mt-1 text-[10px] font-medium px-2 py-0.5 rounded-full bg-teal-50 text-teal-700 border border-teal-100">
                      {currentUser.role || "Sales AE"}
                    </span>
                  </div>

                  {/* Menu Links */}
                  <div className="py-1">
                    <button
                      onClick={() => {
                        setUserMenuOpen(false);
                        setProfileModalOpen(true);
                      }}
                      className="w-full px-4 py-2 text-xs flex items-center gap-2.5 text-slate-600 hover:text-slate-900 hover:bg-slate-50 transition cursor-pointer"
                    >
                      <User className="w-3.5 h-3.5 text-slate-400" />
                      <span>Profile</span>
                    </button>
                    <button
                      onClick={() => {
                        setUserMenuOpen(false);
                        setActiveTab("knowledge");
                      }}
                      className="w-full px-4 py-2 text-xs flex items-center gap-2.5 text-slate-600 hover:text-slate-900 hover:bg-slate-50 transition cursor-pointer"
                    >
                      <BookOpen className="w-3.5 h-3.5 text-slate-400" />
                      <span>Knowledge Playbooks</span>
                    </button>
                    <button
                      onClick={() => {
                        setUserMenuOpen(false);
                        setSettingsModalOpen(true);
                      }}
                      className="w-full px-4 py-2 text-xs flex items-center gap-2.5 text-slate-600 hover:text-slate-900 hover:bg-slate-50 transition cursor-pointer"
                    >
                      <Settings className="w-3.5 h-3.5 text-slate-400" />
                      <span>Settings</span>
                    </button>
                  </div>

                  {/* Sign Out */}
                  <div className="border-t border-slate-100 pt-1 mt-1">
                    <button
                      onClick={handleLogout}
                      className="w-full px-4 py-2 text-xs flex items-center gap-2.5 text-rose-600 hover:bg-rose-50 transition cursor-pointer"
                    >
                      <LogOut className="w-3.5 h-3.5 text-rose-500" />
                      <span>Sign Out</span>
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* =====================================================================
          MAIN DASHBOARD VIEWPORT
          ===================================================================== */}
      <main className="flex-1 w-full px-6 sm:px-10 lg:px-12 xl:px-14 py-6 sm:py-8 relative z-10 flex flex-col">
        {activeTab === "dashboard" ? (
          /* ===================================================================
             MINIMAL & SPACIOUS HOME VIEW (Polished Spacing & Dynamic Greeting)
             =================================================================== */
          <div className="w-full flex flex-col space-y-6 sm:space-y-8 animate-in fade-in duration-200">
            {/* Hero Section */}
            <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-white/95 via-white/90 to-teal-50/30 border border-slate-200/80 shadow-[0_4px_24px_-6px_rgba(0,0,0,0.04),0_1px_3px_rgba(0,0,0,0.02)] p-6 sm:p-8 lg:p-10">
              {/* Subtle ambient background glow */}
              <div className="absolute top-0 right-0 w-96 h-96 bg-gradient-to-bl from-teal-100/35 via-emerald-100/20 to-transparent rounded-full blur-3xl pointer-events-none -mr-20 -mt-20" />
              <div className="absolute bottom-0 left-0 w-80 h-80 bg-gradient-to-tr from-cyan-100/25 via-teal-50/20 to-transparent rounded-full blur-3xl pointer-events-none -ml-20 -mb-20" />

              <div className="grid grid-cols-1 lg:grid-cols-12 items-center gap-6 lg:gap-8 relative z-10">
                {/* Left Column Text */}
                <div className="lg:col-span-7">
                  <div className="inline-flex items-center gap-1.5 text-teal-800 bg-teal-50/90 border border-teal-200/70 px-3.5 py-1 rounded-full text-xs sm:text-sm font-semibold mb-3 shadow-xs">
                    <span>Welcome back{currentUser.name ? `, ${currentUser.name}` : ""}</span>
                    <span className="text-sm">👋</span>
                  </div>
                  <h1 className="text-3xl sm:text-4xl lg:text-5xl font-black text-slate-900 tracking-tight leading-tight mb-3">
                    Let’s make more sales!
                  </h1>
                  <p className="text-slate-500 text-sm sm:text-base leading-relaxed max-w-xl">
                    Your AI-powered voice sales copilot is ready to help you connect, engage, and close more deals.
                  </p>
                </div>

                {/* Right Column Illustration Graphic (Soft mint organic blob + Mic + Soundwaves) */}
                <div className="lg:col-span-5 flex justify-center lg:justify-end">
                  <div className="relative w-56 sm:w-64 h-36 sm:h-40 flex items-center justify-center">
                    {/* Organic cloud/blob backdrop */}
                    <div className="absolute inset-0 bg-gradient-to-tr from-teal-100/70 via-cyan-100/60 to-emerald-50/70 rounded-[50%_60%_70%_40%/60%_50%_60%_50%] filter blur-sm" />
                    <div className="absolute inset-2 bg-gradient-to-br from-teal-50 via-cyan-50/80 to-transparent rounded-[60%_40%_50%_70%/50%_60%_40%_60%]" />

                    {/* Left Sound Wave Bars */}
                    <div className="relative z-10 flex items-center gap-1.5 mr-3.5">
                      <span className="w-1.5 h-3 bg-teal-300 rounded-full" />
                      <span className="w-1.5 h-6 bg-teal-400 rounded-full" />
                      <span className="w-1.5 h-9 bg-teal-500 rounded-full" />
                    </div>

                    {/* Center Microphone */}
                    <div className="relative z-10 text-teal-600 flex items-center justify-center drop-shadow-sm">
                      <svg
                        className="w-14 h-14 sm:w-16 sm:h-16"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                        <line x1="12" x2="12" y1="19" y2="22" />
                      </svg>
                    </div>

                    {/* Right Sound Wave Bars */}
                    <div className="relative z-10 flex items-center gap-1.5 ml-3.5">
                      <span className="w-1.5 h-9 bg-teal-500 rounded-full" />
                      <span className="w-1.5 h-6 bg-teal-400 rounded-full" />
                      <span className="w-1.5 h-3 bg-teal-300 rounded-full" />
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Action Cards (Live Copilot & Ask Assistant) */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5 sm:gap-6 lg:gap-8 w-full">
              {/* Card 1: Live Copilot */}
              <div
                onClick={() => setActiveTab("live-call")}
                className="group relative overflow-hidden bg-gradient-to-br from-white via-white to-teal-50/35 border border-slate-200/85 rounded-3xl p-6 sm:p-8 shadow-[0_4px_20px_-4px_rgba(0,0,0,0.04),0_2px_6px_-2px_rgba(0,0,0,0.02)] hover:shadow-[0_16px_36px_-8px_rgba(13,148,136,0.15),0_4px_12px_-2px_rgba(0,0,0,0.04)] hover:border-teal-400/80 hover:-translate-y-1 transition-all duration-300 cursor-pointer flex flex-col justify-between"
              >
                {/* Ambient corner glow */}
                <div className="absolute top-0 right-0 w-44 h-44 bg-gradient-to-bl from-teal-100/40 to-transparent rounded-full blur-2xl pointer-events-none -mr-10 -mt-10 group-hover:scale-125 transition-transform duration-500" />

                <div className="relative z-10">
                  <div className="flex items-center justify-between mb-5">
                    <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-teal-50 to-emerald-50 text-teal-600 border border-teal-100/80 flex items-center justify-center shadow-xs group-hover:scale-105 group-hover:bg-teal-600 group-hover:text-white group-hover:border-teal-600 transition-all duration-300">
                      <Mic className="w-7 h-7" />
                    </div>
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-teal-700 bg-teal-50 border border-teal-100/80 px-2.5 py-1 rounded-full">
                      Module 1 · Live Voice
                    </span>
                  </div>
                  <h3 className="text-xl sm:text-2xl font-bold text-slate-900 mb-2 group-hover:text-teal-700 transition-colors">
                    Live Copilot
                  </h3>
                  <p className="text-slate-500 text-sm leading-relaxed mb-6">
                    Talk to your AI assistant during calls and get real-time guidance.
                  </p>
                </div>

                <div className="relative z-10 flex items-center justify-between pt-4 border-t border-slate-100">
                  <span className="text-xs font-semibold text-slate-400 group-hover:text-teal-600 transition-colors">
                    Launch voice session
                  </span>
                  <div className="w-10 h-10 rounded-full bg-slate-50 border border-slate-200/60 text-slate-600 flex items-center justify-center group-hover:bg-teal-600 group-hover:text-white group-hover:border-teal-600 group-hover:shadow-md group-hover:shadow-teal-500/20 transition-all duration-300">
                    <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
                  </div>
                </div>
              </div>

              {/* Card 2: Ask Assistant */}
              <div
                onClick={() => setActiveTab("assistant")}
                className="group relative overflow-hidden bg-gradient-to-br from-white via-white to-indigo-50/35 border border-slate-200/85 rounded-3xl p-6 sm:p-8 shadow-[0_4px_20px_-4px_rgba(0,0,0,0.04),0_2px_6px_-2px_rgba(0,0,0,0.02)] hover:shadow-[0_16px_36px_-8px_rgba(99,102,241,0.15),0_4px_12px_-2px_rgba(0,0,0,0.04)] hover:border-indigo-400/80 hover:-translate-y-1 transition-all duration-300 cursor-pointer flex flex-col justify-between"
              >
                {/* Ambient corner glow */}
                <div className="absolute top-0 right-0 w-44 h-44 bg-gradient-to-bl from-indigo-100/40 to-transparent rounded-full blur-2xl pointer-events-none -mr-10 -mt-10 group-hover:scale-125 transition-transform duration-500" />

                <div className="relative z-10">
                  <div className="flex items-center justify-between mb-5">
                    <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-indigo-50 to-violet-50 text-indigo-600 border border-indigo-100/80 flex items-center justify-center shadow-xs group-hover:scale-105 group-hover:bg-indigo-600 group-hover:text-white group-hover:border-indigo-600 transition-all duration-300">
                      <MessageSquare className="w-7 h-7" />
                    </div>
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-indigo-700 bg-indigo-50 border border-indigo-100/80 px-2.5 py-1 rounded-full">
                      Module 2 · Knowledge RAG
                    </span>
                  </div>
                  <h3 className="text-xl sm:text-2xl font-bold text-slate-900 mb-2 group-hover:text-indigo-700 transition-colors">
                    Ask Assistant
                  </h3>
                  <p className="text-slate-500 text-sm leading-relaxed mb-6">
                    Get instant answers, tips and objection handling help.
                  </p>
                </div>

                <div className="relative z-10 flex items-center justify-between pt-4 border-t border-slate-100">
                  <span className="text-xs font-semibold text-slate-400 group-hover:text-indigo-600 transition-colors">
                    Query sales playbooks
                  </span>
                  <div className="w-10 h-10 rounded-full bg-slate-50 border border-slate-200/60 text-slate-600 flex items-center justify-center group-hover:bg-indigo-600 group-hover:text-white group-hover:border-indigo-600 group-hover:shadow-md group-hover:shadow-indigo-500/20 transition-all duration-300">
                    <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : activeTab === "live-call" ? (
          /* ===================================================================
             LIVE CALL COPILOT (MODULE 1) & SAVED CRM LEADS
             =================================================================== */
          <div className="space-y-8 animate-in fade-in duration-200 w-full">
            {/* Header with Back to Dashboard */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setActiveTab("dashboard")}
                  className="p-2.5 rounded-xl border border-slate-200 bg-white text-slate-500 hover:text-slate-900 hover:border-slate-300 shadow-xs transition cursor-pointer"
                  title="Back to Dashboard"
                >
                  <ArrowLeft className="w-4 h-4" />
                </button>
                <div>
                  <h2 className="text-2xl font-bold text-slate-900">Live Call Copilot</h2>
                  <p className="text-xs text-slate-500">
                    Voice-to-CRM Recording, Multilingual STT &amp; Sequential Lead Extraction
                  </p>
                </div>
              </div>
            </div>

            {/* Module 1 Component */}
            <div className="bg-white rounded-3xl p-6 shadow-sm border border-slate-100 w-full">
              <LiveCallVoiceCopilot
                onLeadSaved={(savedLead) => {
                  setCrmLeads((prev) => [savedLead, ...prev.filter((l) => l.id !== savedLead.id)]);
                }}
              />
            </div>

            {/* Saved CRM Pipeline Table */}
            <div className="bg-white rounded-3xl border border-slate-100 shadow-sm p-6 w-full">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6 pb-4 border-b border-slate-100">
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="text-base font-bold text-slate-900">Saved Leads &amp; CRM Pipeline</h3>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-teal-50 text-teal-700 border border-teal-100">
                      PostgreSQL CRM
                    </span>
                    {crmLeads.length > 0 && (
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-100">
                        {crmLeads.length} saved
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-slate-500 mt-1">
                    Structured loan leads extracted with Pydantic and persisted to PostgreSQL
                  </p>
                </div>

                <div className="flex items-center gap-2">
                  <div className="relative">
                    <input
                      type="text"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      placeholder="Filter leads..."
                      className="text-xs h-9 px-3 py-1.5 rounded-xl border border-slate-200 bg-slate-50 focus:bg-white focus:outline-none focus:border-teal-500 w-44 sm:w-56"
                    />
                  </div>
                  <button
                    onClick={() => fetchLeads()}
                    className="p-2 rounded-xl border border-slate-200 text-slate-600 hover:bg-slate-50 transition cursor-pointer"
                    title="Refresh leads"
                  >
                    <RotateCcw className={`w-3.5 h-3.5 ${isLoadingLeads ? "animate-spin" : ""}`} />
                  </button>
                </div>
              </div>

              {/* Table */}
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs text-slate-600">
                  <thead className="text-[11px] uppercase tracking-wider text-slate-400 bg-slate-50/60 border-b border-slate-100">
                    <tr>
                      <th className="py-3 px-4 font-semibold">Prospect &amp; Company</th>
                      <th className="py-3 px-4 font-semibold">Loan Type</th>
                      <th className="py-3 px-4 font-semibold">Loan Amount</th>
                      <th className="py-3 px-4 font-semibold">Database / Source</th>
                      <th className="py-3 px-4 font-semibold text-right">Captured</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {isLoadingLeads ? (
                      <tr>
                        <td colSpan={5} className="py-12 text-center text-slate-400">
                          <RotateCcw className="w-5 h-5 animate-spin mx-auto mb-2 text-teal-600" />
                          <span>Loading leads from PostgreSQL...</span>
                        </td>
                      </tr>
                    ) : filteredLeads.length > 0 ? (
                      filteredLeads.map((lead) => (
                        <tr
                          key={lead.id ? `lead-${lead.id}` : `lead-${lead.name}-${lead.company}`}
                          className="hover:bg-slate-50/50 transition-colors"
                        >
                          <td className="py-3.5 px-4">
                            <div className="font-semibold text-slate-900 flex items-center gap-1.5">
                              <span>{lead.name || "Unnamed Prospect"}</span>
                              {lead.id && (
                                <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-slate-100 text-slate-600">
                                  #{lead.id}
                                </span>
                              )}
                            </div>
                            <div className="text-[11px] text-slate-400 truncate max-w-xs mt-0.5">
                              {[lead.role, lead.company].filter(Boolean).join(" · ") || "Voice Lead"}
                              {lead.phone || lead.email
                                ? ` · ${[lead.phone, lead.email].filter(Boolean).join(" / ")}`
                                : ""}
                            </div>
                          </td>
                          <td className="py-3.5 px-4">
                            <span className="px-2.5 py-1 rounded-full text-[11px] font-medium bg-teal-50 text-teal-700 border border-teal-100">
                              {lead.loan_type || "General Inquiry"}
                            </span>
                          </td>
                          <td className="py-3.5 px-4 font-mono font-medium text-slate-700">
                            {lead.loan_amount !== null && lead.loan_amount !== undefined
                              ? `$${Number(lead.loan_amount).toLocaleString()}`
                              : "—"}
                            {lead.tenure_months ? ` (${lead.tenure_months} mo)` : ""}
                          </td>
                          <td className="py-3.5 px-4">
                            <span className="text-emerald-600 font-mono text-[11px] flex items-center gap-1 font-medium">
                              <Database className="w-3.5 h-3.5" />
                              PostgreSQL
                            </span>
                          </td>
                          <td className="py-3.5 px-4 text-right text-[11px] text-slate-400 font-mono">
                            {lead.created_at
                              ? new Date(lead.created_at).toLocaleDateString()
                              : "Just now"}
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={5} className="py-12 text-center text-slate-400">
                          <Database className="w-8 h-8 text-slate-300 mx-auto mb-2" />
                          <p className="font-semibold text-slate-700 text-xs">No saved leads in PostgreSQL</p>
                          <p className="text-[11px] text-slate-400 mt-1 max-w-sm mx-auto">
                            Leads extracted from live microphone voice calls will automatically appear here.
                          </p>
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        ) : activeTab === "assistant" ? (
          /* ===================================================================
             ASK ASSISTANT (MODULE 2)
             =================================================================== */
          <div className="space-y-6 animate-in fade-in duration-200 w-full">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setActiveTab("dashboard")}
                  className="p-2.5 rounded-xl border border-slate-200 bg-white text-slate-500 hover:text-slate-900 hover:border-slate-300 shadow-xs transition cursor-pointer"
                  title="Back to Dashboard"
                >
                  <ArrowLeft className="w-4 h-4" />
                </button>
                <div>
                  <h2 className="text-2xl font-bold text-slate-900">Ask Assistant</h2>
                  <p className="text-xs text-slate-500">
                    Real-Time Sales Objections &amp; Playbook RAG Intelligence
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-3xl p-6 shadow-sm border border-slate-100 w-full">
              <KnowledgeAssistant />
            </div>
          </div>
        ) : (
          /* ===================================================================
             KNOWLEDGE PLAYBOOKS (PDF Vectors & Uploads)
             =================================================================== */
          <div className="space-y-6 animate-in fade-in duration-200 w-full">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setActiveTab("dashboard")}
                  className="p-2.5 rounded-xl border border-slate-200 bg-white text-slate-500 hover:text-slate-900 hover:border-slate-300 shadow-xs transition cursor-pointer"
                  title="Back to Dashboard"
                >
                  <ArrowLeft className="w-4 h-4" />
                </button>
                <div>
                  <h2 className="text-2xl font-bold text-slate-900">Knowledge Playbooks</h2>
                  <p className="text-xs text-slate-500">
                    Upload and manage PDF battlecards indexed into 1536-dim vector space
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-3xl p-6 shadow-sm border border-slate-100 w-full">
              <KnowledgePlaybooks onOpenAssistant={() => setActiveTab("assistant")} />
            </div>
          </div>
        )}
      </main>

      {/* =====================================================================
          FLOWING BOTTOM WAVES (Matching Reference Image)
          ===================================================================== */}
      {activeTab === "dashboard" && (
        <div className="pointer-events-none fixed bottom-0 left-0 right-0 w-full overflow-hidden leading-none z-0">
          <svg
            className="relative block w-full h-36 sm:h-52"
            viewBox="0 0 1440 240"
            fill="none"
            preserveAspectRatio="none"
          >
            <path
              d="M0,150 C320,220 440,110 720,160 C1000,210 1180,90 1440,140 L1440,240 L0,240 Z"
              fill="#f0fdfa"
              opacity="0.85"
            />
            <path
              d="M0,185 C360,120 500,210 800,165 C1100,120 1280,200 1440,175 L1440,240 L0,240 Z"
              fill="#e0f2fe"
              opacity="0.5"
            />
          </svg>
        </div>
      )}

      {/* =====================================================================
          PROFILE MODAL
          ===================================================================== */}
      {profileModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-xs p-4">
          <div className="bg-white rounded-3xl p-6 sm:p-8 max-w-md w-full shadow-2xl border border-slate-100 animate-in fade-in zoom-in-95 duration-150">
            <div className="flex items-center justify-between pb-4 border-b border-slate-100 mb-6">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-2xl bg-teal-50 text-teal-600 flex items-center justify-center">
                  <User className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-slate-900">User Profile</h3>
                  <p className="text-xs text-slate-500">Sales Representative Information</p>
                </div>
              </div>
              <button
                onClick={() => setProfileModalOpen(false)}
                className="p-1 rounded-lg text-slate-400 hover:text-slate-600 cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-4 text-xs text-slate-600">
              <div>
                <label className="text-[11px] font-semibold text-slate-400 block mb-1">Full Name</label>
                <div className="p-3 bg-slate-50 rounded-xl font-medium text-slate-800">{currentUser.name}</div>
              </div>
              <div>
                <label className="text-[11px] font-semibold text-slate-400 block mb-1">Email Address</label>
                <div className="p-3 bg-slate-50 rounded-xl font-medium text-slate-800">{currentUser.email}</div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-[11px] font-semibold text-slate-400 block mb-1">Company</label>
                  <div className="p-3 bg-slate-50 rounded-xl font-medium text-slate-800">{currentUser.company || "Enterprise AI"}</div>
                </div>
                <div>
                  <label className="text-[11px] font-semibold text-slate-400 block mb-1">Role</label>
                  <div className="p-3 bg-slate-50 rounded-xl font-medium text-slate-800">{currentUser.role || "Senior AE"}</div>
                </div>
              </div>
            </div>

            <div className="mt-8 flex justify-end">
              <button
                onClick={() => setProfileModalOpen(false)}
                className="px-5 py-2.5 rounded-xl bg-teal-600 hover:bg-teal-700 text-white font-semibold text-xs shadow-sm transition cursor-pointer"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* =====================================================================
          SETTINGS MODAL
          ===================================================================== */}
      {settingsModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-xs p-4">
          <div className="bg-white rounded-3xl p-6 sm:p-8 max-w-md w-full shadow-2xl border border-slate-100 animate-in fade-in zoom-in-95 duration-150">
            <div className="flex items-center justify-between pb-4 border-b border-slate-100 mb-6">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-2xl bg-indigo-50 text-indigo-600 flex items-center justify-center">
                  <Settings className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-slate-900">Voice Copilot Settings</h3>
                  <p className="text-xs text-slate-500">Audio, Speech &amp; Model Configurations</p>
                </div>
              </div>
              <button
                onClick={() => setSettingsModalOpen(false)}
                className="p-1 rounded-lg text-slate-400 hover:text-slate-600 cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-4 text-xs text-slate-600">
              <div className="p-3.5 bg-slate-50 rounded-2xl border border-slate-100 flex items-start gap-3">
                <Globe className="w-4 h-4 text-teal-600 mt-0.5 shrink-0" />
                <div>
                  <p className="font-semibold text-slate-900">Multilingual Speech Detection</p>
                  <p className="text-[11px] text-slate-500 mt-0.5">
                    Automatic detection for English (en-IN), Hindi (hi-IN), and Marathi (mr-IN).
                  </p>
                </div>
              </div>

              <div className="p-3.5 bg-slate-50 rounded-2xl border border-slate-100 flex items-start gap-3">
                <Volume2 className="w-4 h-4 text-indigo-600 mt-0.5 shrink-0" />
                <div>
                  <p className="font-semibold text-slate-900">Text-to-Speech Engine</p>
                  <p className="text-[11px] text-slate-500 mt-0.5">
                    Sarvam Bulbul v3 with female voice <span className="font-mono font-medium text-slate-700">simran</span> + Deepgram Aura automatic fallback.
                  </p>
                </div>
              </div>

              <div className="p-3.5 bg-slate-50 rounded-2xl border border-slate-100 flex items-start gap-3">
                <ShieldCheck className="w-4 h-4 text-emerald-600 mt-0.5 shrink-0" />
                <div>
                  <p className="font-semibold text-slate-900">Acoustic Barge-In &amp; Continuous Mode</p>
                  <p className="text-[11px] text-slate-500 mt-0.5">
                    Hands-free voice turn management with automatic re-arm and 0ms interruption.
                  </p>
                </div>
              </div>
            </div>

            <div className="mt-8 flex justify-end">
              <button
                onClick={() => setSettingsModalOpen(false)}
                className="px-5 py-2.5 rounded-xl bg-teal-600 hover:bg-teal-700 text-white font-semibold text-xs shadow-sm transition cursor-pointer"
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
