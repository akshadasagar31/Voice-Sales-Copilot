"use client";

import React, { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Mic,
  Lock,
  Mail,
  User,
  Building2,
  Briefcase,
  ArrowRight,
  AlertCircle,
  Eye,
  EyeOff,
} from "lucide-react";

export default function RegisterPage() {
  const router = useRouter();
  const [formData, setFormData] = useState({
    fullName: "",
    email: "",
    company: "",
    role: "Account Executive",
    password: "",
  });
  const [showPassword, setShowPassword] = useState(false);
  const [agreed, setAgreed] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setErrorMessage("");

    try {
      const res = await fetch("/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      const data = await res.json();

      if (!res.ok) {
        setErrorMessage(data.error || "Registration failed. Please review your details.");
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

        {/* Title */}
        <h1 className="login-title">Create your AE account</h1>
        <p className="login-subtitle">
          Start your pilot with live voice intelligence &amp; RAG playbooks
        </p>

        {/* Error Alert */}
        {errorMessage && (
          <div className="mb-4 p-3 rounded-lg bg-[var(--danger-soft)] border border-[var(--danger)] text-rose-300 text-xs flex items-center gap-2.5">
            <AlertCircle className="h-4 w-4 shrink-0 text-rose-400" />
            <span>{errorMessage}</span>
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-3.5">
          {/* Full Name */}
          <div>
            <label className="block text-xs font-medium text-[var(--ink-soft)] mb-1.5">
              Full Name
            </label>
            <div className="relative flex items-center">
              <div className="absolute left-3 flex items-center pointer-events-none text-[var(--ink-muted)] z-10">
                <User className="h-4 w-4" />
              </div>
              <input
                type="text"
                name="fullName"
                value={formData.fullName}
                onChange={handleChange}
                required
                placeholder="Enter your full name"
                className="form-control has-icon-left h-10 w-full text-sm"
              />
            </div>
          </div>

          {/* Work Email */}
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
                name="email"
                value={formData.email}
                onChange={handleChange}
                required
                placeholder="name@company.com"
                className="form-control has-icon-left h-10 w-full text-sm"
              />
            </div>
          </div>

          {/* Company & Role Row */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-medium text-[var(--ink-soft)] mb-1.5">
                Company Name
              </label>
              <div className="relative flex items-center">
                <div className="absolute left-3 flex items-center pointer-events-none text-[var(--ink-muted)] z-10">
                  <Building2 className="h-4 w-4" />
                </div>
                <input
                  type="text"
                  name="company"
                  value={formData.company}
                  onChange={handleChange}
                  required
                  placeholder="Acme Corp"
                  className="form-control has-icon-left h-10 w-full text-sm"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-medium text-[var(--ink-soft)] mb-1.5">
                Sales Role
              </label>
              <div className="relative flex items-center">
                <div className="absolute left-3 flex items-center pointer-events-none text-[var(--ink-muted)] z-10">
                  <Briefcase className="h-4 w-4" />
                </div>
                <select
                  name="role"
                  value={formData.role}
                  onChange={handleChange}
                  className="form-select has-icon-left h-10 w-full text-sm cursor-pointer"
                >
                  <option value="Account Executive">Account Executive</option>
                  <option value="Sales Leader">VP / Head of Sales</option>
                  <option value="Solutions Engineer">Solutions Engineer</option>
                  <option value="SDR / BDR">SDR / BDR</option>
                  <option value="Customer Success">Customer Success</option>
                </select>
              </div>
            </div>
          </div>

          {/* Password */}
          <div>
            <label className="block text-xs font-medium text-[var(--ink-soft)] mb-1.5">
              Password
            </label>
            <div className="relative flex items-center">
              <div className="absolute left-3 flex items-center pointer-events-none text-[var(--ink-muted)] z-10">
                <Lock className="h-4 w-4" />
              </div>
              <input
                type={showPassword ? "text" : "password"}
                name="password"
                value={formData.password}
                onChange={handleChange}
                required
                placeholder="Create a password (min 6 characters)"
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
            {/* Strength Meter */}
            <div className="mt-1.5 flex items-center gap-1.5">
              <div className={`h-1 flex-1 rounded-full transition-colors ${formData.password.length > 0 ? "bg-emerald-500" : "bg-[var(--border)]"}`} />
              <div className={`h-1 flex-1 rounded-full transition-colors ${formData.password.length >= 6 ? "bg-emerald-500" : "bg-[var(--border)]"}`} />
              <div className={`h-1 flex-1 rounded-full transition-colors ${formData.password.length >= 8 ? "bg-emerald-500" : "bg-[var(--border)]"}`} />
              <span className="text-[10px] text-emerald-400 font-medium ml-1">
                {formData.password.length >= 8 ? "Strong" : formData.password.length >= 6 ? "Good" : "Enter password"}
              </span>
            </div>
          </div>

          {/* Terms checkbox */}
          <div className="flex items-center gap-2 pt-0.5">
            <input
              id="terms"
              type="checkbox"
              checked={agreed}
              onChange={(e) => setAgreed(e.target.checked)}
              className="h-4 w-4 rounded border-[var(--border-strong)] bg-[var(--surface-2)] text-[var(--accent)] accent-[var(--accent)] cursor-pointer"
            />
            <label htmlFor="terms" className="text-xs text-[var(--ink-soft)] cursor-pointer select-none">
              I agree to the Terms of Service and Privacy Policy
            </label>
          </div>

          {/* Submit */}
          <button
            type="submit"
            disabled={isLoading || !agreed}
            className="btn btn-primary w-full h-10 text-xs font-semibold shadow-md flex items-center justify-center gap-2 cursor-pointer transition-all disabled:opacity-50 mt-1"
          >
            {isLoading ? (
              <span className="inline-block animate-spin h-4 w-4 border-2 border-black border-t-transparent rounded-full" />
            ) : (
              <>
                <span>Create Account &amp; Open Dashboard</span>
                <ArrowRight className="h-3.5 w-3.5" />
              </>
            )}
          </button>
        </form>

        {/* Footer Navigation */}
        <div className="mt-5 text-center border-t border-[var(--border)] pt-4 text-xs text-[var(--ink-muted)]">
          <p>
            Already have an account?{" "}
            <Link
              href="/login"
              className="text-[var(--accent)] hover:underline font-semibold"
            >
              Sign in &rarr;
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
