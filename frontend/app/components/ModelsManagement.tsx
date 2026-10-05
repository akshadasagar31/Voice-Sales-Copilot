"use client";

import React, { useState, useEffect, useMemo } from "react";
import {
  Cpu,
  Mic,
  Volume2,
  Sparkles,
  Plus,
  Search,
  CheckCircle2,
  AlertCircle,
  Loader2,
  RotateCcw,
  Edit2,
  Trash2,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  ShieldCheck,
  Copy,
  Check,
  Sliders,
  Filter,
  ArrowLeft,
  X,
  Code,
  Info,
  Key,
  Eye,
  EyeOff,
} from "lucide-react";

export interface AIModel {
  id: string;
  name: string;
  category: "stt" | "llm" | "tts";
  provider: string;
  model_id: string;
  description?: string;
  is_builtin: boolean;
  is_active: boolean;
  configuration: Record<string, any>;
  has_api_key?: boolean;
  api_key_masked?: string;
  created_at?: string;
  updated_at?: string;
}

export interface ActiveModelsMap {
  stt?: AIModel;
  llm?: AIModel;
  tts?: AIModel;
}

interface ModelsManagementProps {
  onBack?: () => void;
}

const PRESET_CONFIGS = {
  stt: {
    smart_format: true,
    punctuate: true,
    encoding: "linear16",
    sample_rate: 48000,
  },
  llm: {
    temperature: 0.15,
    max_tokens: 1024,
    top_p: 0.95,
  },
  tts: {
    default_speaker: "ritu",
    pace: 1.0,
    sample_rate: 48000,
  },
};

export default function ModelsManagement({ onBack }: ModelsManagementProps) {
  const [models, setModels] = useState<AIModel[]>([]);
  const [activeModels, setActiveModels] = useState<ActiveModelsMap>({});
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [toastMessage, setToastMessage] = useState<{
    text: string;
    type: "success" | "error" | "info";
  } | null>(null);

  // Filters & Search
  const [selectedCategory, setSelectedCategory] = useState<"all" | "stt" | "llm" | "tts">("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedProvider, setSelectedProvider] = useState<string>("all");

  // Accordion state for configuration viewer per model
  const [expandedConfig, setExpandedConfig] = useState<Record<string, boolean>>({});
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Modal states
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [isRestoreModalOpen, setIsRestoreModalOpen] = useState(false);
  const [editingModel, setEditingModel] = useState<AIModel | null>(null);
  const [deletingModel, setDeletingModel] = useState<AIModel | null>(null);

  // Form State for Add / Edit
  const [formCategory, setFormCategory] = useState<"stt" | "llm" | "tts">("llm");
  const [formName, setFormName] = useState("");
  const [formApiKey, setFormApiKey] = useState("");
  const [showApiKey, setShowApiKey] = useState(false);
  const [formSetActive, setFormSetActive] = useState(false);
  const [formProvider, setFormProvider] = useState("OpenRouter");
  const [formModelId, setFormModelId] = useState("");
  const [formDescription, setFormDescription] = useState("");
  const [formConfigJson, setFormConfigJson] = useState("{}");
  const [formJsonError, setFormJsonError] = useState<string | null>(null);

  // Show toast notification helper
  const showToast = (text: string, type: "success" | "error" | "info" = "success") => {
    setToastMessage({ text, type });
    setTimeout(() => {
      setToastMessage((prev) => (prev?.text === text ? null : prev));
    }, 4000);
  };

  // Fetch models and active models
  const fetchModels = async () => {
    setLoading(true);
    try {
      const [modelsRes, activeRes] = await Promise.all([
        fetch("/api/models", { cache: "no-store" }),
        fetch("/api/models/active", { cache: "no-store" }),
      ]);

      if (modelsRes.ok) {
        const data = await modelsRes.json();
        setModels(data.models || []);
      } else {
        showToast("Failed to fetch models list from server.", "error");
      }

      if (activeRes.ok) {
        const activeData = await activeRes.json();
        setActiveModels(activeData.active || {});
      }
    } catch (err: any) {
      console.error("Error loading models:", err);
      showToast("Error connecting to backend model service.", "error");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  // Copy to clipboard helper
  const handleCopy = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 1800);
  };

  // Toggle Config Viewer
  const toggleConfig = (id: string) => {
    setExpandedConfig((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  // Switch Active Model
  const handleActivate = async (category: "stt" | "llm" | "tts", modelId: string) => {
    const targetModel = models.find((m) => m.id === modelId);
    setActionLoading(`activate-${modelId}`);
    try {
      const res = await fetch("/api/models/activate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ category, model_id: modelId }),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || errData.detail || "Failed to switch active model");
      }

      showToast(`Activated ${category.toUpperCase()} model: ${targetModel?.name || modelId}`, "success");
      await fetchModels();
    } catch (err: any) {
      console.error("Activation error:", err);
      showToast(err.message || "Failed to activate model.", "error");
    } finally {
      setActionLoading(null);
    }
  };

  // Open Add Modal - Category, Model Name, and API Key ONLY
  const openAddModal = (categoryOverride?: "stt" | "llm" | "tts") => {
    const cat = categoryOverride || (selectedCategory !== "all" ? selectedCategory : "llm");
    setFormCategory(cat);
    setFormName("");
    setFormApiKey("");
    setShowApiKey(false);
    setFormSetActive(false);
    setIsAddModalOpen(true);
  };

  // Open Edit Modal
  const openEditModal = (model: AIModel) => {
    setEditingModel(model);
    setFormCategory(model.category);
    setFormProvider(model.provider);
    setFormName(model.name);
    setFormModelId(model.model_id);
    setFormApiKey("");
    setShowApiKey(false);
    setFormDescription(model.description || "");
    setFormConfigJson(JSON.stringify(model.configuration || {}, null, 2));
    setFormSetActive(model.is_active);
    setFormJsonError(null);
    setIsEditModalOpen(true);
  };

  // Handle Preset Load
  const loadPreset = (cat: "stt" | "llm" | "tts") => {
    setFormConfigJson(JSON.stringify(PRESET_CONFIGS[cat], null, 2));
    setFormJsonError(null);
  };

  // Handle Create Model Submit - only Category, Model Name, API Key
  const handleCreateSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!formName.trim()) {
      showToast("Please enter a model name.", "error");
      return;
    }

    setActionLoading("submitting-create");
    try {
      const res = await fetch("/api/models", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          category: formCategory,
          name: formName.trim(),
          api_key: formApiKey.trim() || undefined,
          is_active: formSetActive,
        }),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || errData.detail || "Failed to add model.");
      }

      showToast(`Added custom model: ${formName.trim()}`, "success");
      setIsAddModalOpen(false);
      setFormName("");
      setFormApiKey("");
      await fetchModels();
    } catch (err: any) {
      console.error("Create error:", err);
      showToast(err.message || "Failed to add model.", "error");
    } finally {
      setActionLoading(null);
    }
  };

  // Handle Update Model Submit
  const handleUpdateSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!editingModel) return;
    setFormJsonError(null);

    let parsedConfig: Record<string, any> = {};
    if (formConfigJson.trim()) {
      try {
        parsedConfig = JSON.parse(formConfigJson);
      } catch (err: any) {
        setFormJsonError("Invalid JSON configuration. Please check syntax.");
        return;
      }
    }

    if (!formName.trim()) {
      showToast("Model name cannot be empty.", "error");
      return;
    }

    setActionLoading("submitting-update");
    try {
      const updatePayload: Record<string, any> = {
        name: formName.trim(),
        provider: formProvider.trim(),
        model_id: formModelId.trim(),
        description: formDescription.trim(),
        configuration: parsedConfig,
        is_active: formSetActive,
      };
      if (formApiKey.trim()) {
        updatePayload.api_key = formApiKey.trim();
      }

      const res = await fetch(`/api/models/${encodeURIComponent(editingModel.id)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updatePayload),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || errData.detail || "Failed to update model.");
      }

      showToast(`Updated model: ${formName.trim()}`, "success");
      setIsEditModalOpen(false);
      setEditingModel(null);
      await fetchModels();
    } catch (err: any) {
      console.error("Update error:", err);
      showToast(err.message || "Failed to update model.", "error");
    } finally {
      setActionLoading(null);
    }
  };

  // Handle Delete Model (Permanent deletion for any model)
  const handleDeleteConfirm = async () => {
    if (!deletingModel) return;
    setActionLoading(`delete-${deletingModel.id}`);
    try {
      const res = await fetch(`/api/models/${encodeURIComponent(deletingModel.id)}`, {
        method: "DELETE",
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || errData.detail || "Failed to delete model.");
      }

      showToast(`Permanently deleted model: ${deletingModel.name}`, "info");
      setDeletingModel(null);
      await fetchModels();
    } catch (err: any) {
      console.error("Delete error:", err);
      showToast(err.message || "Failed to delete model.", "error");
    } finally {
      setActionLoading(null);
    }
  };

  // Handle Restore Default Models
  const handleRestoreDefaults = async () => {
    setActionLoading("restore-defaults");
    try {
      const res = await fetch("/api/models/restore-defaults", {
        method: "POST",
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || errData.detail || "Failed to restore defaults.");
      }

      showToast("Restored default built-in models successfully.", "success");
      setIsRestoreModalOpen(false);
      await fetchModels();
    } catch (err: any) {
      console.error("Restore defaults error:", err);
      showToast(err.message || "Failed to restore defaults.", "error");
    } finally {
      setActionLoading(null);
    }
  };

  // Unique list of providers for filter dropdown
  const availableProviders = useMemo(() => {
    const set = new Set<string>();
    models.forEach((m) => {
      if (m.provider) set.add(m.provider);
    });
    return Array.from(set).sort();
  }, [models]);

  // Filtered models
  const filteredModels = useMemo(() => {
    return models.filter((m) => {
      // Category filter
      if (selectedCategory !== "all" && m.category !== selectedCategory) {
        return false;
      }
      // Provider filter
      if (selectedProvider !== "all" && m.provider !== selectedProvider) {
        return false;
      }
      // Search query
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const matchesName = m.name.toLowerCase().includes(q);
        const matchesModelId = m.model_id.toLowerCase().includes(q);
        const matchesProvider = m.provider.toLowerCase().includes(q);
        const matchesDesc = (m.description || "").toLowerCase().includes(q);
        return matchesName || matchesModelId || matchesProvider || matchesDesc;
      }
      return true;
    });
  }, [models, selectedCategory, selectedProvider, searchQuery]);

  // Counts per category
  const counts = useMemo(() => {
    return {
      all: models.length,
      stt: models.filter((m) => m.category === "stt").length,
      llm: models.filter((m) => m.category === "llm").length,
      tts: models.filter((m) => m.category === "tts").length,
    };
  }, [models]);

  return (
    <div className="space-y-6 animate-in fade-in duration-200 w-full max-w-7xl mx-auto pb-12">
      {/* Toast Notification Banner */}
      {toastMessage && (
        <div
          className={`fixed top-20 right-6 z-50 flex items-center gap-2.5 px-4 py-3 rounded-2xl shadow-xl border text-sm font-medium transition-all duration-200 animate-in slide-in-from-top-4 ${
            toastMessage.type === "success"
              ? "bg-emerald-50 text-emerald-800 border-emerald-200"
              : toastMessage.type === "error"
              ? "bg-rose-50 text-rose-800 border-rose-200"
              : "bg-teal-50 text-teal-800 border-teal-200"
          }`}
        >
          {toastMessage.type === "success" ? (
            <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
          ) : toastMessage.type === "error" ? (
            <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
          ) : (
            <Info className="w-4 h-4 text-teal-600 shrink-0" />
          )}
          <span>{toastMessage.text}</span>
          <button
            onClick={() => setToastMessage(null)}
            className="ml-2 text-slate-400 hover:text-slate-600 cursor-pointer"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Top Header & Breadcrumb */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          {onBack && (
            <button
              onClick={onBack}
              className="p-2 rounded-full border border-slate-200 bg-white text-slate-500 hover:text-slate-900 hover:border-slate-300 shadow-xs transition cursor-pointer"
              title="Back to Dashboard"
            >
              <ArrowLeft className="w-4 h-4" />
            </button>
          )}
          <div>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-xl bg-teal-50 border border-teal-200 text-teal-600 flex items-center justify-center shadow-xs">
                <Cpu className="w-4 h-4" />
              </div>
              <h1 className="text-2xl font-bold text-slate-900 tracking-tight">AI Models Management</h1>
            </div>
            <p className="text-xs text-slate-500 mt-1">
              Configure, deploy, and dynamically route Speech-to-Text, LLM Reasoning, and Voice Synthesis engines
            </p>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2.5">
          <button
            onClick={fetchModels}
            disabled={loading}
            className="px-3.5 py-2 rounded-xl border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold shadow-xs transition flex items-center gap-2 cursor-pointer disabled:opacity-50"
            title="Refresh Models"
          >
            <RotateCcw className={`w-3.5 h-3.5 ${loading ? "animate-spin text-teal-600" : "text-slate-500"}`} />
            <span>Refresh</span>
          </button>
          <button
            onClick={() => setIsRestoreModalOpen(true)}
            disabled={loading || actionLoading !== null}
            className="px-3.5 py-2 rounded-xl border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold shadow-xs transition flex items-center gap-2 cursor-pointer disabled:opacity-50"
            title="Restore Built-in Default Models"
          >
            <RotateCcw className="w-3.5 h-3.5 text-teal-600" />
            <span>Restore Defaults</span>
          </button>
          <button
            onClick={() => openAddModal()}
            className="px-4 py-2 rounded-xl bg-teal-600 hover:bg-teal-700 text-white text-xs font-semibold shadow-xs transition flex items-center gap-1.5 cursor-pointer"
          >
            <Plus className="w-4 h-4" />
            <span>Add New Model</span>
          </button>
        </div>
      </div>

      {/* =====================================================================
          ACTIVE MODELS CONTROL HUB (3 Interactive Cards)
          ===================================================================== */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Active STT Card */}
        <div className="bg-white rounded-2xl p-5 border border-sky-100 shadow-xs relative overflow-hidden group hover:shadow-md transition">
          <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-sky-400 to-sky-600" />
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-xl bg-sky-50 border border-sky-200 text-sky-600 flex items-center justify-center shadow-xs">
                <Mic className="w-4 h-4" />
              </div>
              <div>
                <span className="text-[10px] font-bold uppercase tracking-wider text-sky-600">STT Engine</span>
                <p className="text-sm font-bold text-slate-900 leading-tight">
                  {activeModels.stt?.name || "No active model"}
                </p>
              </div>
            </div>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              Active
            </span>
          </div>

          <div className="text-xs text-slate-500 flex items-center justify-between pt-2 border-t border-slate-100 mt-2">
            <span className="text-[11px] text-slate-400 font-mono">
              ID: {activeModels.stt?.model_id || "—"}
            </span>
            <span className="text-[11px] font-medium text-slate-600">
              Provider: {activeModels.stt?.provider || "—"}
            </span>
          </div>

          {/* Quick Switch Dropdown */}
          <div className="mt-3.5">
            <label className="block text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1">
              Switch STT Model
            </label>
            <div className="relative">
              <select
                value={activeModels.stt?.id || ""}
                onChange={(e) => handleActivate("stt", e.target.value)}
                disabled={actionLoading !== null}
                className="w-full text-xs font-semibold bg-slate-50 border border-slate-200 text-slate-800 rounded-xl px-3 py-2 pr-8 appearance-none focus:outline-none focus:ring-2 focus:ring-sky-500/20 focus:border-sky-500 transition cursor-pointer"
              >
                {models
                  .filter((m) => m.category === "stt")
                  .map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name} ({m.provider}) {m.id === activeModels.stt?.id ? "● Current" : ""}
                    </option>
                  ))}
              </select>
              <ChevronDown className="w-3.5 h-3.5 text-slate-400 absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none" />
            </div>
          </div>

          {/* Bottom Action Row matching reference image */}
          <div className="flex items-center gap-2 mt-3.5 pt-3 border-t border-slate-100">
            <button
              disabled
              className="px-3 py-1.5 rounded-xl border border-slate-100 bg-slate-50/50 text-slate-300 text-xs font-semibold flex items-center gap-1.5 cursor-not-allowed"
              title="STT Engine is currently active"
            >
              <Check className="w-3.5 h-3.5 text-slate-300" />
              <span>Activate</span>
            </button>
            <button
              onClick={() => activeModels.stt && openEditModal(activeModels.stt)}
              disabled={!activeModels.stt}
              className="px-3.5 py-1.5 rounded-xl border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold shadow-2xs transition flex items-center gap-1.5 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
              title="Edit Active Model"
            >
              <Edit2 className="w-3.5 h-3.5 text-slate-500" />
              <span>Edit</span>
            </button>
            <button
              onClick={() => activeModels.stt && setDeletingModel(activeModels.stt)}
              disabled={!activeModels.stt}
              className="px-3 py-1.5 rounded-xl border border-red-200 bg-white hover:bg-red-50 text-red-500 text-xs font-semibold shadow-2xs transition flex items-center gap-1.5 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
              title="Delete Active Model Permanently"
            >
              <Trash2 className="w-3.5 h-3.5 text-red-500" />
              <span>Delete</span>
            </button>
          </div>
        </div>

        {/* Active LLM Card */}
        <div className="bg-white rounded-2xl p-5 border border-indigo-100 shadow-xs relative overflow-hidden group hover:shadow-md transition">
          <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-indigo-500 to-indigo-700" />
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-xl bg-indigo-50 border border-indigo-200 text-indigo-600 flex items-center justify-center shadow-xs">
                <Sparkles className="w-4 h-4" />
              </div>
              <div>
                <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-600">LLM Reasoning</span>
                <p className="text-sm font-bold text-slate-900 leading-tight">
                  {activeModels.llm?.name || "No active model"}
                </p>
              </div>
            </div>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              Active
            </span>
          </div>

          <div className="text-xs text-slate-500 flex items-center justify-between pt-2 border-t border-slate-100 mt-2">
            <span className="text-[11px] text-slate-400 font-mono truncate max-w-[150px]" title={activeModels.llm?.model_id}>
              ID: {activeModels.llm?.model_id || "—"}
            </span>
            <span className="text-[11px] font-medium text-slate-600">
              Provider: {activeModels.llm?.provider || "—"}
            </span>
          </div>

          {/* Quick Switch Dropdown */}
          <div className="mt-3.5">
            <label className="block text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1">
              Switch LLM Model
            </label>
            <div className="relative">
              <select
                value={activeModels.llm?.id || ""}
                onChange={(e) => handleActivate("llm", e.target.value)}
                disabled={actionLoading !== null}
                className="w-full text-xs font-semibold bg-slate-50 border border-slate-200 text-slate-800 rounded-xl px-3 py-2 pr-8 appearance-none focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition cursor-pointer"
              >
                {models
                  .filter((m) => m.category === "llm")
                  .map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name} ({m.provider}) {m.id === activeModels.llm?.id ? "● Current" : ""}
                    </option>
                  ))}
              </select>
              <ChevronDown className="w-3.5 h-3.5 text-slate-400 absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none" />
            </div>
          </div>

          {/* Bottom Action Row matching reference image */}
          <div className="flex items-center gap-2 mt-3.5 pt-3 border-t border-slate-100">
            <button
              disabled
              className="px-3 py-1.5 rounded-xl border border-slate-100 bg-slate-50/50 text-slate-300 text-xs font-semibold flex items-center gap-1.5 cursor-not-allowed"
              title="LLM Reasoning is currently active"
            >
              <Check className="w-3.5 h-3.5 text-slate-300" />
              <span>Activate</span>
            </button>
            <button
              onClick={() => activeModels.llm && openEditModal(activeModels.llm)}
              disabled={!activeModels.llm}
              className="px-3.5 py-1.5 rounded-xl border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold shadow-2xs transition flex items-center gap-1.5 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
              title="Edit Active Model"
            >
              <Edit2 className="w-3.5 h-3.5 text-slate-500" />
              <span>Edit</span>
            </button>
            <button
              onClick={() => activeModels.llm && setDeletingModel(activeModels.llm)}
              disabled={!activeModels.llm}
              className="px-3 py-1.5 rounded-xl border border-red-200 bg-white hover:bg-red-50 text-red-500 text-xs font-semibold shadow-2xs transition flex items-center gap-1.5 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
              title="Delete Active Model Permanently"
            >
              <Trash2 className="w-3.5 h-3.5 text-red-500" />
              <span>Delete</span>
            </button>
          </div>
        </div>

        {/* Active TTS Card */}
        <div className="bg-white rounded-2xl p-5 border border-amber-100 shadow-xs relative overflow-hidden group hover:shadow-md transition">
          <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-amber-400 to-amber-600" />
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-xl bg-amber-50 border border-amber-200 text-amber-600 flex items-center justify-center shadow-xs">
                <Volume2 className="w-4 h-4" />
              </div>
              <div>
                <span className="text-[10px] font-bold uppercase tracking-wider text-amber-600">TTS Voice Engine</span>
                <p className="text-sm font-bold text-slate-900 leading-tight">
                  {activeModels.tts?.name || "No active model"}
                </p>
              </div>
            </div>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              Active
            </span>
          </div>

          <div className="text-xs text-slate-500 flex items-center justify-between pt-2 border-t border-slate-100 mt-2">
            <span className="text-[11px] text-slate-400 font-mono">
              ID: {activeModels.tts?.model_id || "—"}
            </span>
            <span className="text-[11px] font-medium text-slate-600">
              Provider: {activeModels.tts?.provider || "—"}
            </span>
          </div>

          {/* Quick Switch Dropdown */}
          <div className="mt-3.5">
            <label className="block text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1">
              Switch TTS Model
            </label>
            <div className="relative">
              <select
                value={activeModels.tts?.id || ""}
                onChange={(e) => handleActivate("tts", e.target.value)}
                disabled={actionLoading !== null}
                className="w-full text-xs font-semibold bg-slate-50 border border-slate-200 text-slate-800 rounded-xl px-3 py-2 pr-8 appearance-none focus:outline-none focus:ring-2 focus:ring-amber-500/20 focus:border-amber-500 transition cursor-pointer"
              >
                {models
                  .filter((m) => m.category === "tts")
                  .map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name} ({m.provider}) {m.id === activeModels.tts?.id ? "● Current" : ""}
                    </option>
                  ))}
              </select>
              <ChevronDown className="w-3.5 h-3.5 text-slate-400 absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none" />
            </div>
          </div>

          {/* Bottom Action Row matching reference image */}
          <div className="flex items-center gap-2 mt-3.5 pt-3 border-t border-slate-100">
            <button
              disabled
              className="px-3 py-1.5 rounded-xl border border-slate-100 bg-slate-50/50 text-slate-300 text-xs font-semibold flex items-center gap-1.5 cursor-not-allowed"
              title="TTS Voice Engine is currently active"
            >
              <Check className="w-3.5 h-3.5 text-slate-300" />
              <span>Activate</span>
            </button>
            <button
              onClick={() => activeModels.tts && openEditModal(activeModels.tts)}
              disabled={!activeModels.tts}
              className="px-3.5 py-1.5 rounded-xl border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold shadow-2xs transition flex items-center gap-1.5 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
              title="Edit Active Model"
            >
              <Edit2 className="w-3.5 h-3.5 text-slate-500" />
              <span>Edit</span>
            </button>
            <button
              onClick={() => activeModels.tts && setDeletingModel(activeModels.tts)}
              disabled={!activeModels.tts}
              className="px-3 py-1.5 rounded-xl border border-red-200 bg-white hover:bg-red-50 text-red-500 text-xs font-semibold shadow-2xs transition flex items-center gap-1.5 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
              title="Delete Active Model Permanently"
            >
              <Trash2 className="w-3.5 h-3.5 text-red-500" />
              <span>Delete</span>
            </button>
          </div>
        </div>
      </div>

      {/* =====================================================================
          FILTER & SEARCH TOOLBAR
          ===================================================================== */}
      <div className="bg-white rounded-2xl p-4 border border-slate-200/80 shadow-xs flex flex-col md:flex-row md:items-center justify-between gap-4">
        {/* Category Pills */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 md:pb-0">
          <button
            onClick={() => setSelectedCategory("all")}
            className={`px-3.5 py-1.5 rounded-full text-xs font-semibold transition cursor-pointer shrink-0 ${
              selectedCategory === "all"
                ? "bg-slate-900 text-white shadow-xs"
                : "bg-slate-100 text-slate-600 hover:bg-slate-200 hover:text-slate-900"
            }`}
          >
            All Models ({counts.all})
          </button>
          <button
            onClick={() => setSelectedCategory("stt")}
            className={`px-3.5 py-1.5 rounded-full text-xs font-semibold transition flex items-center gap-1.5 cursor-pointer shrink-0 ${
              selectedCategory === "stt"
                ? "bg-sky-600 text-white shadow-xs"
                : "bg-sky-50 text-sky-700 hover:bg-sky-100 border border-sky-100"
            }`}
          >
            <Mic className="w-3.5 h-3.5" />
            <span>STT ({counts.stt})</span>
          </button>
          <button
            onClick={() => setSelectedCategory("llm")}
            className={`px-3.5 py-1.5 rounded-full text-xs font-semibold transition flex items-center gap-1.5 cursor-pointer shrink-0 ${
              selectedCategory === "llm"
                ? "bg-indigo-600 text-white shadow-xs"
                : "bg-indigo-50 text-indigo-700 hover:bg-indigo-100 border border-indigo-100"
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>LLM ({counts.llm})</span>
          </button>
          <button
            onClick={() => setSelectedCategory("tts")}
            className={`px-3.5 py-1.5 rounded-full text-xs font-semibold transition flex items-center gap-1.5 cursor-pointer shrink-0 ${
              selectedCategory === "tts"
                ? "bg-amber-600 text-white shadow-xs"
                : "bg-amber-50 text-amber-700 hover:bg-amber-100 border border-amber-100"
            }`}
          >
            <Volume2 className="w-3.5 h-3.5" />
            <span>TTS ({counts.tts})</span>
          </button>
        </div>

        {/* Search & Provider Filter */}
        <div className="flex items-center gap-2.5">
          {/* Provider Filter */}
          <div className="relative shrink-0">
            <select
              value={selectedProvider}
              onChange={(e) => setSelectedProvider(e.target.value)}
              className="text-xs font-medium bg-slate-50 border border-slate-200 text-slate-700 rounded-xl px-3 py-1.5 pr-7 appearance-none focus:outline-none focus:ring-1 focus:ring-teal-500 cursor-pointer"
            >
              <option value="all">All Providers</option>
              {availableProviders.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
            <ChevronDown className="w-3 h-3 text-slate-400 absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
          </div>

          {/* Search Input */}
          <div className="relative flex-1 sm:w-60">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search models..."
              className="w-full text-xs bg-slate-50 border border-slate-200 rounded-xl pl-8 pr-3 py-1.5 text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500 transition"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery("")}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 cursor-pointer"
              >
                <X className="w-3 h-3" />
              </button>
            )}
          </div>
        </div>
      </div>

      {/* =====================================================================
          MODELS LIST / GRID
          ===================================================================== */}
      {loading ? (
        <div className="bg-white rounded-3xl p-12 border border-slate-100 flex flex-col items-center justify-center gap-3 text-slate-500">
          <Loader2 className="w-6 h-6 animate-spin text-teal-600" />
          <p className="text-xs font-semibold">Loading persistent models configuration...</p>
        </div>
      ) : filteredModels.length === 0 ? (
        <div className="bg-white rounded-3xl p-12 border border-slate-100 text-center">
          <div className="w-12 h-12 rounded-2xl bg-slate-100 text-slate-400 flex items-center justify-center mx-auto mb-3">
            <Cpu className="w-6 h-6" />
          </div>
          <h3 className="text-base font-bold text-slate-800 mb-1">No models found</h3>
          <p className="text-xs text-slate-500 max-w-sm mx-auto mb-4">
            No models matched your current filter or search criteria.
          </p>
          <button
            onClick={() => openAddModal()}
            className="px-4 py-2 rounded-xl bg-teal-600 hover:bg-teal-700 text-white text-xs font-semibold shadow-xs transition inline-flex items-center gap-1.5 cursor-pointer"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>Add New Custom Model</span>
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredModels.map((model) => {
            const isCatActive =
              (model.category === "stt" && activeModels.stt?.id === model.id) ||
              (model.category === "llm" && activeModels.llm?.id === model.id) ||
              (model.category === "tts" && activeModels.tts?.id === model.id);

            const isConfigOpen = expandedConfig[model.id] || false;

            return (
              <div
                key={model.id}
                className={`bg-white rounded-2xl transition-all duration-200 flex flex-col justify-between overflow-hidden shadow-xs hover:shadow-md ${
                  isCatActive
                    ? "border-2 border-emerald-400 ring-2 ring-emerald-400/10"
                    : "border border-slate-200/90 hover:border-slate-300"
                }`}
              >
                {/* Card Top Section */}
                <div className="p-5">
                  {/* Badges Bar */}
                  <div className="flex items-center justify-between gap-2 mb-3">
                    <div className="flex items-center gap-1.5">
                      {/* Category Badge */}
                      <span
                        className={`text-[10px] font-bold px-2 py-0.5 rounded-md uppercase tracking-wider ${
                          model.category === "stt"
                            ? "bg-sky-50 text-sky-700 border border-sky-200"
                            : model.category === "llm"
                            ? "bg-indigo-50 text-indigo-700 border border-indigo-200"
                            : "bg-amber-50 text-amber-700 border border-amber-200"
                        }`}
                      >
                        {model.category.toUpperCase()}
                      </span>

                      {/* Built-in vs Custom Tag */}
                      <span
                        className={`text-[10px] font-semibold px-2 py-0.5 rounded-md ${
                          model.is_builtin
                            ? "bg-slate-100 text-slate-600 border border-slate-200"
                            : "bg-purple-50 text-purple-700 border border-purple-200"
                        }`}
                      >
                        {model.is_builtin ? "Built-in" : "Custom"}
                      </span>

                      {/* Custom Key Configured Badge */}
                      {model.has_api_key && (
                        <span
                          className="inline-flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-md bg-emerald-50 text-emerald-700 border border-emerald-200"
                          title={model.api_key_masked || "Custom API key configured"}
                        >
                          <Key className="w-2.5 h-2.5 text-emerald-600" />
                          <span>Custom Key</span>
                        </span>
                      )}
                    </div>

                    {/* Active / Inactive Status Badge */}
                    {isCatActive ? (
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                        Active
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-slate-100 text-slate-500 border border-slate-200">
                        <span className="w-1.5 h-1.5 rounded-full bg-slate-400" />
                        Inactive
                      </span>
                    )}
                  </div>

                  {/* Model Title & Provider */}
                  <div className="mb-2">
                    <h3 className="text-base font-bold text-slate-900 tracking-tight leading-snug">
                      {model.name}
                    </h3>
                    <p className="text-xs font-normal text-slate-500 mt-0.5">
                      Provider: <span className="text-slate-700 font-semibold">{model.provider}</span>
                    </p>
                  </div>

                  {/* Model ID Pill with Copy */}
                  <div className="flex items-center justify-between gap-2 bg-slate-50 border border-slate-200/80 rounded-xl px-2.5 py-1.5 my-2.5">
                    <span className="text-[11px] font-mono text-slate-700 truncate select-all">
                      {model.model_id}
                    </span>
                    <button
                      onClick={() => handleCopy(model.id, model.model_id)}
                      className="text-slate-400 hover:text-slate-700 transition cursor-pointer shrink-0"
                      title="Copy Model ID"
                    >
                      {copiedId === model.id ? (
                        <Check className="w-3.5 h-3.5 text-emerald-600" />
                      ) : (
                        <Copy className="w-3.5 h-3.5" />
                      )}
                    </button>
                  </div>

                  {/* Description */}
                  <p className="text-xs text-slate-600 line-clamp-2 leading-relaxed min-h-[36px]">
                    {model.description || "No description provided."}
                  </p>

                  {/* Collapsible Configuration Parameters */}
                  {model.configuration && Object.keys(model.configuration).length > 0 && (
                    <div className="mt-3 pt-3 border-t border-slate-100">
                      <button
                        onClick={() => toggleConfig(model.id)}
                        className="w-full flex items-center justify-between text-[11px] font-semibold text-slate-500 hover:text-slate-800 transition cursor-pointer"
                      >
                        <span className="flex items-center gap-1.5">
                          <Sliders className="w-3 h-3 text-slate-400" />
                          <span>Configuration Parameters</span>
                        </span>
                        {isConfigOpen ? (
                          <ChevronUp className="w-3 h-3 text-slate-400" />
                        ) : (
                          <ChevronDown className="w-3 h-3 text-slate-400" />
                        )}
                      </button>

                      {isConfigOpen && (
                        <div className="mt-2 p-2.5 rounded-xl bg-slate-900 text-slate-200 font-mono text-[11px] overflow-x-auto">
                          <pre className="whitespace-pre-wrap">
                            {JSON.stringify(model.configuration, null, 2)}
                          </pre>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Card Footer Actions */}
                <div className="px-5 py-3.5 bg-slate-50/70 border-t border-slate-100 flex items-center justify-between gap-2">
                  {/* Left: Activate / Active Status */}
                  {isCatActive ? (
                    <div className="flex items-center gap-1.5 text-xs font-bold text-emerald-700 px-3.5 py-1.5 bg-emerald-50 rounded-xl border border-emerald-200 shadow-2xs">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                      <span>Active Engine</span>
                    </div>
                  ) : (
                    <button
                      onClick={() => handleActivate(model.category, model.id)}
                      disabled={actionLoading !== null}
                      className="px-3.5 py-1.5 rounded-xl bg-white hover:bg-teal-50 border border-slate-200 hover:border-teal-300 text-slate-700 hover:text-teal-700 text-xs font-semibold shadow-2xs transition flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
                    >
                      {actionLoading === `activate-${model.id}` ? (
                        <Loader2 className="w-3.5 h-3.5 animate-spin text-teal-600" />
                      ) : (
                        <Check className="w-3.5 h-3.5 text-teal-600" />
                      )}
                      <span>Activate</span>
                    </button>
                  )}

                  {/* Right: Edit & Delete buttons */}
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => openEditModal(model)}
                      className="px-3 py-1.5 rounded-xl border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold shadow-2xs transition flex items-center gap-1.5 cursor-pointer"
                      title={model.is_builtin ? "Edit Configuration" : "Edit Model"}
                    >
                      <Edit2 className="w-3.5 h-3.5 text-slate-500" />
                      <span>Edit</span>
                    </button>

                    <button
                      onClick={() => setDeletingModel(model)}
                      className="px-3 py-1.5 rounded-xl border border-red-200 bg-white hover:bg-red-50 text-red-500 text-xs font-semibold shadow-2xs transition flex items-center gap-1.5 cursor-pointer"
                      title="Delete Model Permanently"
                    >
                      <Trash2 className="w-3.5 h-3.5 text-red-500" />
                      <span>Delete</span>
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* =====================================================================
          ADD NEW MODEL MODAL (Category, Model Name, API Key ONLY)
          ===================================================================== */}
      {isAddModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-xs p-4 animate-in fade-in">
          <div className="bg-white rounded-3xl p-6 max-w-lg w-full shadow-2xl border border-slate-100 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between pb-4 border-b border-slate-100">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-xl bg-teal-50 border border-teal-200 text-teal-600 flex items-center justify-center">
                  <Plus className="w-4 h-4" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-slate-900">Add Custom Model</h3>
                  <p className="text-[11px] text-slate-500">Provide Category, Model Name, and API Key</p>
                </div>
              </div>
              <button
                onClick={() => setIsAddModalOpen(false)}
                className="text-slate-400 hover:text-slate-600 p-1 rounded-lg cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <form onSubmit={handleCreateSubmit} className="space-y-4 pt-4">
              {/* Category Segmented Control */}
              <div>
                <label className="block text-xs font-bold text-slate-700 mb-1.5">
                  Category <span className="text-rose-500">*</span>
                </label>
                <div className="grid grid-cols-3 gap-2">
                  <button
                    type="button"
                    onClick={() => setFormCategory("stt")}
                    className={`py-2 px-3 rounded-xl text-xs font-semibold border transition cursor-pointer flex items-center justify-center gap-1.5 ${
                      formCategory === "stt"
                        ? "bg-sky-50 text-sky-700 border-sky-300 ring-2 ring-sky-500/20"
                        : "bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100"
                    }`}
                  >
                    <Mic className="w-3.5 h-3.5" />
                    <span>STT</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setFormCategory("llm")}
                    className={`py-2 px-3 rounded-xl text-xs font-semibold border transition cursor-pointer flex items-center justify-center gap-1.5 ${
                      formCategory === "llm"
                        ? "bg-indigo-50 text-indigo-700 border-indigo-300 ring-2 ring-indigo-500/20"
                        : "bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100"
                    }`}
                  >
                    <Sparkles className="w-3.5 h-3.5" />
                    <span>LLM</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setFormCategory("tts")}
                    className={`py-2 px-3 rounded-xl text-xs font-semibold border transition cursor-pointer flex items-center justify-center gap-1.5 ${
                      formCategory === "tts"
                        ? "bg-amber-50 text-amber-700 border-amber-300 ring-2 ring-amber-500/20"
                        : "bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100"
                    }`}
                  >
                    <Volume2 className="w-3.5 h-3.5" />
                    <span>TTS</span>
                  </button>
                </div>
              </div>

              {/* Model Name */}
              <div>
                <label className="block text-xs font-bold text-slate-700 mb-1">
                  Model Name <span className="text-rose-500">*</span>
                </label>
                <input
                  type="text"
                  required
                  value={formName}
                  onChange={(e) => setFormName(e.target.value)}
                  placeholder={
                    formCategory === "stt"
                      ? "e.g. deepgram-nova-3 or sarvam-saaras-v4"
                      : formCategory === "llm"
                      ? "e.g. meta-llama/llama-3.3-70b-instruct or deepseek/deepseek-chat"
                      : "e.g. sarvam-bulbul-v3 or deepgram-aura-asteria"
                  }
                  className="w-full text-xs bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-slate-800 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
                />
                <p className="text-[10px] text-slate-400 mt-1">
                  Provider and configuration are automatically resolved from the model identifier.
                </p>
              </div>

              {/* API Key */}
              <div>
                <label className="block text-xs font-bold text-slate-700 mb-1 flex items-center justify-between">
                  <span className="flex items-center gap-1.5">
                    <Key className="w-3.5 h-3.5 text-slate-400" />
                    <span>API Key</span>
                  </span>
                  <span className="text-[10px] text-slate-400 font-normal">Optional</span>
                </label>
                <div className="relative">
                  <input
                    type={showApiKey ? "text" : "password"}
                    value={formApiKey}
                    onChange={(e) => setFormApiKey(e.target.value)}
                    placeholder="e.g. sk-or-v1-... or your provider key"
                    className="w-full text-xs font-mono bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 pr-9 text-slate-800 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
                  />
                  <button
                    type="button"
                    onClick={() => setShowApiKey(!showApiKey)}
                    className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 cursor-pointer p-1"
                  >
                    {showApiKey ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                  </button>
                </div>
                <p className="text-[10px] text-slate-400 mt-1">
                  Leave blank to inherit your system default environment API key.
                </p>
              </div>

              {/* Set Active Checkbox */}
              <div className="flex items-center gap-2 pt-1">
                <input
                  type="checkbox"
                  id="setActiveCheckbox"
                  checked={formSetActive}
                  onChange={(e) => setFormSetActive(e.target.checked)}
                  className="rounded border-slate-300 text-teal-600 focus:ring-teal-500 cursor-pointer"
                />
                <label htmlFor="setActiveCheckbox" className="text-xs font-medium text-slate-700 cursor-pointer">
                  Activate this model immediately upon creation
                </label>
              </div>

              {/* Buttons */}
              <div className="flex items-center justify-end gap-2.5 pt-3 border-t border-slate-100">
                <button
                  type="button"
                  onClick={() => setIsAddModalOpen(false)}
                  className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-600 hover:bg-slate-100 transition cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={actionLoading !== null}
                  className="px-5 py-2 rounded-xl bg-teal-600 hover:bg-teal-700 text-white text-xs font-semibold shadow-xs transition flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
                >
                  {actionLoading === "submitting-create" && (
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  )}
                  <span>Add Model</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* =====================================================================
          EDIT MODEL MODAL
          ===================================================================== */}
      {isEditModalOpen && editingModel && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-xs p-4 animate-in fade-in">
          <div className="bg-white rounded-3xl p-6 max-w-xl w-full shadow-2xl border border-slate-100 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between pb-4 border-b border-slate-100">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-xl bg-slate-100 border border-slate-200 text-slate-700 flex items-center justify-center">
                  <Edit2 className="w-4 h-4" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-slate-900">
                    Edit {editingModel.is_builtin ? "Configuration" : "Model"}: {editingModel.name}
                  </h3>
                  <p className="text-[11px] text-slate-500">
                    {editingModel.is_builtin
                      ? "Customize configuration parameters for core built-in model"
                      : "Modify custom model parameters"}
                  </p>
                </div>
              </div>
              <button
                onClick={() => {
                  setIsEditModalOpen(false);
                  setEditingModel(null);
                }}
                className="text-slate-400 hover:text-slate-600 p-1 rounded-lg cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <form onSubmit={handleUpdateSubmit} className="space-y-4 pt-4">
              {/* Name & Provider */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-bold text-slate-700 mb-1">Model Name</label>
                  <input
                    type="text"
                    required
                    value={formName}
                    onChange={(e) => setFormName(e.target.value)}
                    className="w-full text-xs bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-slate-800 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
                  />
                </div>

                <div>
                  <label className="block text-xs font-bold text-slate-700 mb-1">Provider</label>
                  <input
                    type="text"
                    disabled={editingModel.is_builtin}
                    value={formProvider}
                    onChange={(e) => setFormProvider(e.target.value)}
                    className="w-full text-xs bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-slate-800 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500 disabled:opacity-60 disabled:cursor-not-allowed"
                  />
                </div>
              </div>

              {/* Model ID */}
              <div>
                <label className="block text-xs font-bold text-slate-700 mb-1">API Model ID</label>
                <input
                  type="text"
                  disabled={editingModel.is_builtin}
                  value={formModelId}
                  onChange={(e) => setFormModelId(e.target.value)}
                  className="w-full text-xs font-mono bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-slate-800 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500 disabled:opacity-60 disabled:cursor-not-allowed"
                />
                {editingModel.is_builtin && (
                  <p className="text-[10px] text-slate-400 mt-1">
                    Built-in model identifiers are protected from modification.
                  </p>
                )}
              </div>

              {/* Description */}
              <div>
                <label className="block text-xs font-bold text-slate-700 mb-1">Description</label>
                <input
                  type="text"
                  value={formDescription}
                  onChange={(e) => setFormDescription(e.target.value)}
                  className="w-full text-xs bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-slate-800 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
                />
              </div>

              {/* API Key */}
              <div>
                <label className="block text-xs font-bold text-slate-700 mb-1 flex items-center justify-between">
                  <span className="flex items-center gap-1.5">
                    <Key className="w-3.5 h-3.5 text-slate-400" />
                    <span>API Key</span>
                  </span>
                  <span className="text-[10px] text-slate-400 font-normal">Optional</span>
                </label>
                <div className="relative">
                  <input
                    type={showApiKey ? "text" : "password"}
                    value={formApiKey}
                    onChange={(e) => setFormApiKey(e.target.value)}
                    placeholder={editingModel.has_api_key ? "•••••••••••••••• (Leave blank to keep existing key)" : "Enter custom API key"}
                    className="w-full text-xs font-mono bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 pr-9 text-slate-800 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
                  />
                  <button
                    type="button"
                    onClick={() => setShowApiKey(!showApiKey)}
                    className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 cursor-pointer p-1"
                  >
                    {showApiKey ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                  </button>
                </div>
                <p className="text-[10px] text-slate-400 mt-1">
                  Enter a new API key to update, or leave blank to keep the current setting.
                </p>
              </div>

              {/* Configuration JSON */}
              <div>
                <label className="block text-xs font-bold text-slate-700 mb-1 flex items-center gap-1.5">
                  <Code className="w-3.5 h-3.5 text-slate-400" />
                  <span>Configuration JSON</span>
                </label>
                <textarea
                  rows={5}
                  value={formConfigJson}
                  onChange={(e) => {
                    setFormConfigJson(e.target.value);
                    setFormJsonError(null);
                  }}
                  className={`w-full text-xs font-mono bg-slate-900 text-slate-100 rounded-xl p-3 focus:outline-none focus:ring-2 ${
                    formJsonError ? "border-2 border-rose-500 focus:ring-rose-500/20" : "focus:ring-teal-500/20"
                  }`}
                />
                {formJsonError && (
                  <p className="text-[11px] text-rose-600 font-semibold mt-1">{formJsonError}</p>
                )}
              </div>

              {/* Set Active Checkbox */}
              <div className="flex items-center gap-2 pt-1">
                <input
                  type="checkbox"
                  id="editSetActiveCheckbox"
                  checked={formSetActive}
                  onChange={(e) => setFormSetActive(e.target.checked)}
                  className="rounded border-slate-300 text-teal-600 focus:ring-teal-500 cursor-pointer"
                />
                <label htmlFor="editSetActiveCheckbox" className="text-xs font-medium text-slate-700 cursor-pointer">
                  Set as currently active {editingModel.category.toUpperCase()} model
                </label>
              </div>

              {/* Buttons */}
              <div className="flex items-center justify-end gap-2.5 pt-3 border-t border-slate-100">
                <button
                  type="button"
                  onClick={() => {
                    setIsEditModalOpen(false);
                    setEditingModel(null);
                  }}
                  className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-600 hover:bg-slate-100 transition cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={actionLoading !== null}
                  className="px-5 py-2 rounded-xl bg-teal-600 hover:bg-teal-700 text-white text-xs font-semibold shadow-xs transition flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
                >
                  {actionLoading === "submitting-update" && (
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  )}
                  <span>Save Changes</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* =====================================================================
          DELETE CONFIRMATION DIALOG
          ===================================================================== */}
      {deletingModel && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-xs p-4 animate-in fade-in">
          <div className="bg-white rounded-3xl p-6 max-w-md w-full shadow-2xl border border-slate-100">
            <div className="w-10 h-10 rounded-2xl bg-rose-50 border border-rose-200 text-rose-600 flex items-center justify-center mb-3">
              <Trash2 className="w-5 h-5" />
            </div>
            <h3 className="text-base font-bold text-slate-900 mb-1">
              Delete Model &apos;{deletingModel.name}&apos; Permanently?
            </h3>
            <p className="text-xs text-slate-500 mb-3 leading-relaxed">
              Are you sure you want to permanently delete this model? It will be completely removed from persistent configuration and this action cannot be undone.
            </p>

            {/* Active Model Warning Alert */}
            {activeModels[deletingModel.category]?.id === deletingModel.id && (
              <div className="p-3 mb-3 bg-amber-50 border border-amber-200 rounded-xl text-amber-800 text-xs flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold">Active Engine Notice:</span> This model is currently the active engine for <span className="font-semibold uppercase">{deletingModel.category}</span>. Deleting it will leave this category without an active engine until you select or add a new one.
                </div>
              </div>
            )}

            {/* Model Summary Details */}
            <div className="p-3 bg-slate-50 rounded-xl border border-slate-200/80 mb-4 space-y-1.5 text-xs">
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Category:</span>
                <span className="font-semibold text-slate-700 uppercase">{deletingModel.category}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Type:</span>
                <span className="font-semibold text-slate-700">{deletingModel.is_builtin ? "Default Built-in" : "Custom Added"}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Provider:</span>
                <span className="font-semibold text-slate-700">{deletingModel.provider}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Model ID:</span>
                <span className="font-mono text-slate-700 truncate max-w-[200px]" title={deletingModel.model_id}>
                  {deletingModel.model_id}
                </span>
              </div>
            </div>

            <div className="flex items-center justify-end gap-2.5">
              <button
                type="button"
                onClick={() => setDeletingModel(null)}
                className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-600 hover:bg-slate-100 transition cursor-pointer"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleDeleteConfirm}
                disabled={actionLoading !== null}
                className="px-4 py-2 rounded-xl bg-rose-600 hover:bg-rose-700 text-white text-xs font-semibold shadow-xs transition flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
              >
                {actionLoading === `delete-${deletingModel.id}` && (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                )}
                <span>Delete Permanently</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* =====================================================================
          RESTORE DEFAULTS CONFIRMATION DIALOG
          ===================================================================== */}
      {isRestoreModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-xs p-4 animate-in fade-in">
          <div className="bg-white rounded-3xl p-6 max-w-md w-full shadow-2xl border border-slate-100">
            <div className="w-10 h-10 rounded-2xl bg-teal-50 border border-teal-200 text-teal-600 flex items-center justify-center mb-3">
              <RotateCcw className="w-5 h-5" />
            </div>
            <h3 className="text-base font-bold text-slate-900 mb-1">
              Restore Default Built-in Models?
            </h3>
            <p className="text-xs text-slate-500 mb-4 leading-relaxed">
              This will re-add any missing default built-in models (such as Deepgram Nova-3, Sarvam Bulbul v3, DeepSeek Chat, etc.) to your catalog. Any existing custom models will remain untouched.
            </p>

            <div className="flex items-center justify-end gap-2.5">
              <button
                type="button"
                onClick={() => setIsRestoreModalOpen(false)}
                className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-600 hover:bg-slate-100 transition cursor-pointer"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleRestoreDefaults}
                disabled={actionLoading !== null}
                className="px-4 py-2 rounded-xl bg-teal-600 hover:bg-teal-700 text-white text-xs font-semibold shadow-xs transition flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
              >
                {actionLoading === "restore-defaults" && (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                )}
                <span>Restore Defaults</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
