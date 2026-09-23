"use client";

import React, { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Mic,
  Sparkles,
  Lock,
  Mail,
  Eye,
  EyeOff,
  ArrowRight,
  AlertCircle,
} from "lucide-react";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setErrorMessage("");

    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });

      const data = await res.json();

      if (!res.ok) {
        setErrorMessage(data.error || "Authentication failed. Please check your credentials.");
        setIsLoading(false);
        return;
      }

      router.refresh();
      router.push("/dashboard");
    } catch (err: any) {
      setErrorMessage("Unable to connect to server. Please try again.");
      setIsLoading(false);
    }
  };

  const handleDemoLogin = async () => {
    setIsLoading(true);
    setErrorMessage("");

    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ isDemo: true }),
      });

      const data = await res.json();

      if (!res.ok) {
        setErrorMessage(data.error || "Failed to initialize demo session.");
        setIsLoading(false);
        return;
      }

      router.refresh();
      router.push("/dashboard");
    } catch (err: any) {
      setErrorMessage("Unable to start demo session.");
      setIsLoading(false);
    }
  };

  return (
    <div className="login-shell" data-bs-theme="dark">
      <div className="login-card">
        {/* Brand Header */}
        <div className="login-brand">
          <div className="brand-mark">
            <Mic className="h-4 w-4" />
          </div>
          <span className="brand-text">VoiceCopilot CRM</span>
        </div>

        {/* Titles */}
        <h1 className="login-title">Sign in to your workspace</h1>
        <p className="login-subtitle">
          Manage your active sales calls, playbooks &amp; voice copilot
        </p>

        {/* Quick Demo Access Button */}
        <button
          type="button"
          onClick={handleDemoLogin}
          disabled={isLoading}
          className="btn btn-secondary w-full h-10 text-xs font-semibold flex items-center justify-center gap-2 cursor-pointer transition-all hover:border-[var(--accent)]/50 hover:bg-[var(--surface-2)]"
        >
          <Sparkles className="h-3.5 w-3.5 text-[var(--accent)]" />
          <span>One-Click Instant Demo Access</span>
          <ArrowRight className="h-3.5 w-3.5 text-[var(--ink-muted)]" />
        </button>

        {/* Clean Flex-Based OR Divider */}
        <div className="flex items-center gap-3 my-4">
          <div className="h-px bg-[var(--border)] flex-1" />
          <span className="text-[10px] font-semibold text-[var(--ink-muted)] uppercase tracking-wider">
            or credentials
          </span>
          <div className="h-px bg-[var(--border)] flex-1" />
        </div>

        {/* Error Alert */}
        {errorMessage && (
          <div className="mb-4 p-3 rounded-lg bg-[var(--danger-soft)] border border-[var(--danger)] text-rose-300 text-xs flex items-center gap-2.5">
            <AlertCircle className="h-4 w-4 shrink-0 text-rose-400" />
            <span>{errorMessage}</span>
          </div>
        )}

        {/* Login Form */}
        <form onSubmit={handleSubmit} className="space-y-3.5">
          {/* Email */}
          <div>
            <label className="block text-xs font-medium text-[var(--ink-soft)] mb-1.5">
              Work Email Address
            </label>
            <div className="relative flex items-center">
              <div className="absolute left-3 flex items-center pointer-events-none text-[var(--ink-muted)] z-10">
                <Mail className="h-4 w-4" />
              </div>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                placeholder="name@company.com"
                className="form-control has-icon-left h-10 w-full text-sm"
              />
            </div>
          </div>

          {/* Password */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="block text-xs font-medium text-[var(--ink-soft)] mb-0">
                Password
              </label>
              <a
                href="#forgot"
                onClick={(e) => {
                  e.preventDefault();
                  alert("Demo: Password reset instructions will be sent in production.");
                }}
                className="text-[11px] text-[var(--accent)] hover:underline font-medium"
              >
                Forgot?
              </a>
            </div>
            <div className="relative flex items-center">
              <div className="absolute left-3 flex items-center pointer-events-none text-[var(--ink-muted)] z-10">
                <Lock className="h-4 w-4" />
              </div>
              <input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                placeholder="Enter your password"
                className="form-control has-icon-both h-10 w-full text-sm"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 flex items-center text-[var(--ink-muted)] hover:text-white cursor-pointer transition-colors z-10"
                tabIndex={-1}
                aria-label={showPassword ? "Hide password" : "Show password"}
              >
                {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
          </div>

          {/* Remember Me */}
          <div className="flex items-center gap-2 pt-0.5">
            <input
              id="remember-me"
              type="checkbox"
              checked={rememberMe}
              onChange={(e) => setRememberMe(e.target.checked)}
              className="h-4 w-4 rounded border-[var(--border-strong)] bg-[var(--surface-2)] text-[var(--accent)] accent-[var(--accent)] cursor-pointer"
            />
            <label
              htmlFor="remember-me"
              className="text-xs text-[var(--ink-soft)] cursor-pointer select-none"
            >
              Keep me signed in for 30 days
            </label>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={isLoading}
            className="btn btn-primary w-full h-10 text-xs font-semibold shadow-md flex items-center justify-center gap-2 cursor-pointer transition-all disabled:opacity-60 mt-1"
          >
            {isLoading ? (
              <span className="inline-block animate-spin h-4 w-4 border-2 border-black border-t-transparent rounded-full"></span>
            ) : (
              <>
                <span>Sign In to Copilot</span>
                <ArrowRight className="h-3.5 w-3.5" />
              </>
            )}
          </button>
        </form>

        {/* Footer Navigation */}
        <div className="mt-5 text-center border-t border-[var(--border)] pt-4 text-xs text-[var(--ink-muted)]">
          <p>
            Don&apos;t have an account?{" "}
            <Link
              href="/register"
              className="text-[var(--accent)] hover:underline font-semibold"
            >
              Create account &rarr;
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
