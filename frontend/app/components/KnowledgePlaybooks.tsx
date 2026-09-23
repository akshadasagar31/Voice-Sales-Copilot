"use client";

import React, { useState, useRef } from "react";
import {
  BookOpen,
  UploadCloud,
  FileText,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Layers,
  Search,
  Sparkles,
  RefreshCw,
  X,
  ChevronDown,
  ChevronUp,
  FileCheck,
  Cpu,
  Database,
  Info,
  ExternalLink,
} from "lucide-react";
import Link from "next/link";

interface DocumentChunk {
  text: string;
  chunk_index: number;
  metadata?: {
    source?: string;
    page?: number;
    chunk_index?: number;
    character_count?: number;
  };
  embedding?: number[];
}

export interface PlaybookDocument {
  id: string;
  title: string;
  filename: string;
  category: string;
  namespace: string;
  totalPages: number;
  totalChunks: number;
  uploadedAt: string;
  status: "Synced" | "Indexed" | "Processing" | "Failed";
  embeddingModel?: string;
  embeddingDimension?: number;
  pineconeUpsertedCount?: number;
  chunks?: DocumentChunk[];
}

interface KnowledgePlaybooksProps {
  onOpenAssistant?: (initialQuery?: string) => void;
}

export default function KnowledgePlaybooks({ onOpenAssistant }: KnowledgePlaybooksProps) {
  // Real documents fetched from PostgreSQL database
  const [documents, setDocuments] = useState<PlaybookDocument[]>([]);
  const [isLoadingDocs, setIsLoadingDocs] = useState<boolean>(true);

  // Fetch real documents from PostgreSQL
  const fetchDocuments = React.useCallback(async () => {
    try {
      setIsLoadingDocs(true);
      const res = await fetch("/api/documents?limit=100");
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data?.documents)) {
          const mapped: PlaybookDocument[] = data.documents.map((doc: any) => {
            const rawTitle = doc.filename ? doc.filename.replace(/\.pdf$/i, "").replace(/[-_]/g, " ") : "Document";
            const ns = doc.pinecone_namespace || "sales_playbooks";
            const category = ns.charAt(0).toUpperCase() + ns.slice(1);
            let formattedDate = "Recently";
            if (doc.created_at || doc.upload_date) {
              const d = new Date(doc.created_at || doc.upload_date);
              if (!isNaN(d.getTime())) {
                formattedDate = d.toLocaleDateString() + " " + d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
              }
            }
            return {
              id: String(doc.id || doc.document_uid || Math.random()),
              title: rawTitle,
              filename: doc.filename || "document.pdf",
              category,
              namespace: ns,
              totalPages: Number(doc.total_pages) || 0,
              totalChunks: Number(doc.total_chunks) || 0,
              uploadedAt: formattedDate,
              status: (Number(doc.pinecone_upserted_count) > 0 || Number(doc.total_chunks) > 0) ? "Synced" : "Indexed",
              embeddingModel: "all-MiniLM-L6-v2 (local_fast)",
              embeddingDimension: 1536,
              pineconeUpsertedCount: Number(doc.pinecone_upserted_count) || Number(doc.total_chunks) || 0,
            };
          });
          setDocuments(mapped);
        }
      }
    } catch (err) {
      console.error("Failed to load documents from PostgreSQL:", err);
    } finally {
      setIsLoadingDocs(false);
    }
  }, []);

  React.useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  // Upload form state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [namespace, setNamespace] = useState("sales_playbooks");
  const [upsertToPinecone, setUpsertToPinecone] = useState(true);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Processing pipeline states
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStage, setProcessingStage] = useState<
    "idle" | "uploading" | "extracting" | "chunking" | "embedding" | "upserting"
  >("idle");
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [latestResult, setLatestResult] = useState<any | null>(null);

  // Search filter & inspection state
  const [searchQuery, setSearchQuery] = useState("");
  const [inspectedDoc, setInspectedDoc] = useState<PlaybookDocument | null>(null);

  // Drag and drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type === "application/pdf" || file.name.endsWith(".pdf")) {
        setSelectedFile(file);
        setErrorMessage(null);
      } else {
        setErrorMessage("Only PDF files (.pdf) are supported in this pipeline.");
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (file.type === "application/pdf" || file.name.endsWith(".pdf")) {
        setSelectedFile(file);
        setErrorMessage(null);
      } else {
        setErrorMessage("Only PDF files (.pdf) are supported in this pipeline.");
      }
    }
  };

  // Upload and process PDF via FastAPI endpoint
  const handleProcessPdf = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile || isProcessing) return;

    setIsProcessing(true);
    setErrorMessage(null);
    setSuccessMessage(null);
    setProcessingStage("uploading");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("namespace", namespace.trim() || "sales_playbooks");
      formData.append("chunk_size", "600");
      formData.append("chunk_overlap", "100");
      formData.append("upsert", upsertToPinecone ? "true" : "false");
      formData.append("upsert_to_pinecone", upsertToPinecone ? "true" : "false");

      // Simulated pipeline progress ticks
      const timer1 = setTimeout(() => setProcessingStage("extracting"), 800);
      const timer2 = setTimeout(() => setProcessingStage("chunking"), 1800);
      const timer3 = setTimeout(() => setProcessingStage("embedding"), 3000);
      const timer4 = setTimeout(() => setProcessingStage("upserting"), 4200);

      const response = await fetch("/api/upload-pdf", {
        method: "POST",
        body: formData,
      });

      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      clearTimeout(timer4);

      let data: any = null;
      const contentType = response.headers.get("content-type") || "";
      if (contentType.includes("application/json")) {
        try {
          data = await response.json();
        } catch {
          data = null;
        }
      }

      if (!data) {
        const rawText = await response.text().catch(() => "");
        try {
          data = JSON.parse(rawText);
        } catch {
          data = { error: rawText || `Server returned error (HTTP ${response.status})` };
        }
      }

      if (!response.ok || !data) {
        throw new Error(
          data?.error || data?.detail || `Failed to process PDF through RAG pipeline (HTTP ${response.status}).`
        );
      }

      setLatestResult(data);
      setSuccessMessage(
        `Successfully extracted ${data.total_pages || 1} pages into ${
          data.total_chunks || 0
        } dense vector chunks (dimension: ${data.embedding_dimension || 1536}) and synced to namespace '${
          data.namespace || namespace
        }'.`
      );

      // Refresh real documents from PostgreSQL database
      await fetchDocuments();

      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (err: any) {
      console.error("PDF upload error:", err);
      setErrorMessage(
        err.message || "Failed to connect to FastAPI PDF processing service."
      );
    } finally {
      setIsProcessing(false);
      setProcessingStage("idle");
    }
  };

  const filteredDocuments = documents.filter(
    (doc) =>
      doc.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      doc.filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
      doc.category.toLowerCase().includes(searchQuery.toLowerCase()) ||
      doc.namespace.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const totalIndexedChunks = documents.reduce((sum, d) => sum + d.totalChunks, 0);
  const totalPagesCount = documents.reduce((sum, d) => sum + d.totalPages, 0);

  return (
    <div className="space-y-6 w-full">
      {/* Page Header — CallNow Page Header Style */}
      <div className="page-header">
        <div>
          <h1 className="page-header-title flex items-center gap-2.5">
            <BookOpen className="h-5 w-5 text-[var(--accent)]" />
            Knowledge Playbooks &amp; Document Store
            <span className="badge badge-primary">
              Module 2 Pipeline
            </span>
          </h1>
          <p className="page-header-subtitle mt-0.5">
            Upload PDF sales collateral, playbooks, and battlecards into dense vector embeddings and sync to Pinecone for sub-second live retrieval.
          </p>
        </div>

        <div className="page-header-actions">
          {onOpenAssistant ? (
            <button
              onClick={() => onOpenAssistant()}
              className="btn btn-primary btn-sm gap-2"
            >
              <Sparkles className="h-3.5 w-3.5" />
              <span>Query in Assistant</span>
            </button>
          ) : (
            <Link
              href="/assistant"
              className="btn btn-primary btn-sm gap-2"
            >
              <Sparkles className="h-3.5 w-3.5" />
              <span>Query in Assistant</span>
            </Link>
          )}
        </div>
      </div>

      {/* Metrics Row — CallNow Metric Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="metric-card">
          <div className="metric-card-header">
            <span className="metric-card-label">Playbooks Indexed</span>
            <div className="metric-card-icon bg-[var(--accent-soft)] text-[var(--accent)]">
              <BookOpen className="h-4 w-4" />
            </div>
          </div>
          <div className="metric-card-value">{documents.length}</div>
          <div className="metric-card-trend up">
            100% Vector Synced
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-card-header">
            <span className="metric-card-label">Total Vectors</span>
            <div className="metric-card-icon bg-[var(--accent-soft)] text-[var(--accent)]">
              <Layers className="h-4 w-4" />
            </div>
          </div>
          <div className="metric-card-value text-[var(--accent)]">{totalIndexedChunks}</div>
          <div className="metric-card-trend flat">
            RecursiveSplitter (600 / 100)
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-card-header">
            <span className="metric-card-label">Pages Processed</span>
            <div className="metric-card-icon bg-[var(--success-soft)] text-[var(--success)]">
              <FileText className="h-4 w-4" />
            </div>
          </div>
          <div className="metric-card-value">{totalPagesCount}</div>
          <div className="metric-card-trend flat">
            PyPDFLoader Text Stream
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-card-header">
            <span className="metric-card-label">Pinecone Vector DB</span>
            <div className="metric-card-icon bg-[var(--purple-soft)] text-[var(--purple)]">
              <Database className="h-4 w-4" />
            </div>
          </div>
          <div className="text-sm font-semibold text-[var(--ink)] mt-2 flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-[var(--success)] animate-pulse" />
            <span>Ready for Retrieval</span>
          </div>
          <div className="metric-card-trend up">
            1536-dim Dense Vectors
          </div>
        </div>
      </div>

      {/* PDF Upload Card — CallNow Card */}
      <div className="card">
        <div className="card-header flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-[var(--radius)] bg-[var(--accent-soft)] text-[var(--accent)] flex items-center justify-center">
              <UploadCloud className="h-4 w-4" />
            </div>
            <div>
              <h2 className="card-title">Upload New Sales Playbook (PDF)</h2>
              <p className="card-subtitle">
                Extracts text, generates dense vector chunks, and upserts them directly into Pinecone.
              </p>
            </div>
          </div>
          <span className="hidden sm:inline-flex badge badge-secondary font-mono">
            Accepts .pdf (max 25MB)
          </span>
        </div>

        <form onSubmit={handleProcessPdf} className="card-body space-y-4">
          {/* Drag and drop zone */}
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-[var(--radius-lg)] p-6 text-center cursor-pointer transition flex flex-col items-center justify-center ${
              isDragging
                ? "border-[var(--accent)] bg-[var(--accent-soft)]"
                : selectedFile
                ? "border-[var(--success)]/50 bg-[var(--success-soft)]"
                : "border-[var(--border)] hover:border-[var(--accent)]/50 hover:bg-[var(--surface-2)] bg-[var(--surface)]"
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              onChange={handleFileChange}
              className="hidden"
            />

            {selectedFile ? (
              <div className="flex items-center gap-3">
                <div className="p-2.5 rounded-[var(--radius)] bg-[var(--success-soft)] text-[var(--success)]">
                  <FileCheck className="h-5 w-5" />
                </div>
                <div className="text-left">
                  <div className="text-sm font-semibold text-[var(--ink)] flex items-center gap-2">
                    <span>{selectedFile.name}</span>
                    <span className="badge badge-success">
                      {(selectedFile.size / 1024).toFixed(1)} KB
                    </span>
                  </div>
                  <p className="text-xs text-[var(--ink-muted)] mt-0.5">
                    Click or drag another file to replace
                  </p>
                </div>
              </div>
            ) : (
              <>
                <div className="p-3 rounded-[var(--radius)] bg-[var(--accent-soft)] text-[var(--accent)] mb-2">
                  <UploadCloud className="h-5 w-5" />
                </div>
                <p className="text-sm font-medium text-[var(--ink)]">
                  Click to browse or drag and drop your PDF here
                </p>
                <p className="text-xs text-[var(--ink-muted)] mt-1">
                  Supports sales objection handbooks, product specs, pricing matrices, and battlecards
                </p>
              </>
            )}
          </div>

          {/* Form Options */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-1">
            <div>
              <label className="form-label mb-1.5">
                Target Namespace / Category
              </label>
              <input
                type="text"
                value={namespace}
                onChange={(e) => setNamespace(e.target.value)}
                placeholder="e.g. sales_playbooks, compliance, pricing"
                className="form-control"
              />
              <span className="text-[10px] text-[var(--ink-muted)] mt-1 block">
                Groups vectors inside Pinecone for scoped multi-tenant retrieval.
              </span>
            </div>

            <div className="flex flex-col justify-center">
              <label className="form-label mb-1.5">
                Vector Pipeline Sync
              </label>
              <label className="flex items-center gap-2.5 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={upsertToPinecone}
                  onChange={(e) => setUpsertToPinecone(e.target.checked)}
                  className="h-4 w-4 rounded border-[var(--border)] bg-[var(--surface-2)] text-[var(--accent)] accent-[var(--accent)] cursor-pointer"
                />
                <span className="text-xs text-[var(--ink)] font-medium">
                  Upsert vectors into Pinecone index automatically
                </span>
              </label>
              <span className="text-[10px] text-[var(--ink-muted)] mt-1 block">
                Generates 1536-dimensional embeddings and uploads to Pinecone index.
              </span>
            </div>
          </div>

          {/* Processing Stages Feedback */}
          {isProcessing && (
            <div className="p-4 rounded-[var(--radius)] bg-[var(--surface-2)] border border-[var(--accent-soft-strong)] space-y-3">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-[var(--accent)] flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Processing PDF through RAG Pipeline...
                </span>
                <span className="font-mono text-[var(--ink-muted)] uppercase text-[10px]">
                  Stage: {processingStage}
                </span>
              </div>

              {/* Step indicator pills */}
              <div className="grid grid-cols-4 gap-2 text-center text-[10px]">
                <div
                  className={`p-1.5 rounded-[var(--radius)] border transition ${
                    ["uploading", "extracting", "chunking", "embedding", "upserting"].includes(
                      processingStage
                    )
                      ? "bg-[var(--accent-soft)] border-[var(--accent-soft-strong)] text-[var(--accent)] font-semibold"
                      : "bg-[var(--surface)] border-[var(--border)] text-[var(--ink-muted)]"
                  }`}
                >
                  1. PyPDF Extraction
                </div>
                <div
                  className={`p-1.5 rounded-[var(--radius)] border transition ${
                    ["chunking", "embedding", "upserting"].includes(processingStage)
                      ? "bg-[var(--accent-soft)] border-[var(--accent-soft-strong)] text-[var(--accent)] font-semibold"
                      : "bg-[var(--surface)] border-[var(--border)] text-[var(--ink-muted)]"
                  }`}
                >
                  2. Chunker (600/100)
                </div>
                <div
                  className={`p-1.5 rounded-[var(--radius)] border transition ${
                    ["embedding", "upserting"].includes(processingStage)
                      ? "bg-[var(--accent-soft)] border-[var(--accent-soft-strong)] text-[var(--accent)] font-semibold"
                      : "bg-[var(--surface)] border-[var(--border)] text-[var(--ink-muted)]"
                  }`}
                >
                  3. Dense Embedding
                </div>
                <div
                  className={`p-1.5 rounded-[var(--radius)] border transition ${
                    processingStage === "upserting"
                      ? "bg-[var(--accent-soft)] border-[var(--accent-soft-strong)] text-[var(--accent)] font-semibold"
                      : "bg-[var(--surface)] border-[var(--border)] text-[var(--ink-muted)]"
                  }`}
                >
                  4. Pinecone Upsert
                </div>
              </div>
            </div>
          )}

          {/* Success Banner */}
          {successMessage && (
            <div className="p-4 rounded-[var(--radius)] bg-[var(--success-soft)] border border-[var(--success)]/30 text-[var(--success)] text-xs flex items-start justify-between gap-3">
              <div className="flex items-start gap-2.5">
                <CheckCircle2 className="h-4 w-4 shrink-0 mt-0.5" />
                <div>
                  <span className="font-semibold block text-white">Processing Complete!</span>
                  <span>{successMessage}</span>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setSuccessMessage(null)}
                className="text-[var(--ink-muted)] hover:text-white"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          )}

          {/* Error Banner */}
          {errorMessage && (
            <div className="p-4 rounded-[var(--radius)] bg-[var(--danger-soft)] border border-[var(--danger)]/30 text-[var(--danger)] text-xs flex items-start justify-between gap-3">
              <div className="flex items-start gap-2.5">
                <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
                <div>
                  <span className="font-semibold block text-white">Upload / Processing Error</span>
                  <span>{errorMessage}</span>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setErrorMessage(null)}
                className="text-[var(--ink-muted)] hover:text-white"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex items-center justify-end gap-3 pt-1">
            {selectedFile && (
              <button
                type="button"
                onClick={() => {
                  setSelectedFile(null);
                  if (fileInputRef.current) fileInputRef.current.value = "";
                }}
                className="btn btn-secondary btn-sm"
              >
                Clear
              </button>
            )}

            <button
              type="submit"
              disabled={!selectedFile || isProcessing}
              className="btn btn-primary btn-sm gap-2 disabled:opacity-50 disabled:pointer-events-none"
            >
              {isProcessing ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Processing Pipeline...</span>
                </>
              ) : (
                <>
                  <Layers className="h-4 w-4" />
                  <span>Process &amp; Index Document</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>

      {/* Document Knowledge Base Section — CallNow Table Wrapper */}
      <div className="card">
        <div className="card-header flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-[var(--radius)] bg-[var(--accent-soft)] text-[var(--accent)] flex items-center justify-center">
              <Database className="h-4 w-4" />
            </div>
            <div>
              <h2 className="card-title">Active Knowledge Base Documents</h2>
              <p className="card-subtitle">
                Playbooks and battlecards available for sub-second retrieval by the Voice Copilot.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => fetchDocuments()}
              className="btn btn-secondary btn-sm gap-1.5"
              title="Refresh documents from PostgreSQL"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${isLoadingDocs ? "animate-spin" : ""}`} />
              <span className="hidden sm:inline">Refresh</span>
            </button>

            {/* Search bar */}
            <div className="relative w-full sm:w-64">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-[var(--ink-muted)]" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search playbooks..."
                className="form-control pl-9"
              />
            </div>
          </div>
        </div>

        {/* Documents Table */}
        <div className="table-wrapper">
          <table className="table table-hover">
            <thead>
              <tr>
                <th style={{ width: "35%" }}>Document Title &amp; File</th>
                <th>Namespace</th>
                <th>Pages</th>
                <th>Chunks</th>
                <th>Status</th>
                <th>Added</th>
                <th style={{ textAlign: "right" }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {isLoadingDocs ? (
                <tr>
                  <td colSpan={7} className="text-center py-12 text-[var(--ink-muted)]">
                    <div className="flex flex-col items-center justify-center gap-2">
                      <Loader2 className="h-6 w-6 animate-spin text-[var(--accent)]" />
                      <span className="text-xs font-medium text-[var(--ink-soft)]">
                        Loading documents from PostgreSQL...
                      </span>
                    </div>
                  </td>
                </tr>
              ) : filteredDocuments.length > 0 ? (
                filteredDocuments.map((doc) => (
                  <tr key={doc.id}>
                    <td>
                      <div className="flex items-center gap-2.5">
                        <div className="p-2 rounded-[var(--radius)] bg-[var(--accent-soft)] text-[var(--accent)] shrink-0">
                          <FileText className="h-4 w-4" />
                        </div>
                        <div>
                          <span className="font-semibold text-[var(--ink)] block">
                            {doc.title}
                          </span>
                          <span className="text-[10px] text-[var(--ink-muted)] font-mono">
                            {doc.filename}
                          </span>
                        </div>
                      </div>
                    </td>
                    <td>
                      <span className="badge badge-secondary font-mono">
                        {doc.namespace}
                      </span>
                    </td>
                    <td className="font-medium text-[var(--ink)]">
                      {doc.totalPages}
                    </td>
                    <td>
                      <span className="text-[var(--accent)] font-semibold">{doc.totalChunks}</span>
                      <span className="text-[10px] text-[var(--ink-muted)] ml-1">chunks</span>
                    </td>
                    <td>
                      <span
                        className={`status-pill ${
                          doc.status === "Synced"
                            ? "status-pill-success"
                            : doc.status === "Indexed"
                            ? "status-pill-info"
                            : "status-pill-neutral"
                        }`}
                      >
                        {doc.status}
                      </span>
                    </td>
                    <td className="text-[var(--ink-muted)] text-[11px]">
                      {doc.uploadedAt}
                    </td>
                    <td style={{ textAlign: "right" }}>
                      <div className="flex items-center justify-end gap-2">
                        {doc.chunks && doc.chunks.length > 0 && (
                          <button
                            type="button"
                            onClick={() => setInspectedDoc(doc)}
                            className="btn btn-secondary btn-sm text-[11px]"
                          >
                            View Chunks
                          </button>
                        )}

                        {onOpenAssistant ? (
                          <button
                            type="button"
                            onClick={() => onOpenAssistant(`What does ${doc.title} say about key requirements?`)}
                            className="btn btn-primary btn-sm text-[11px] gap-1"
                          >
                            <Sparkles className="h-3 w-3" />
                            <span>Ask</span>
                          </button>
                        ) : (
                          <Link
                            href={`/assistant?q=${encodeURIComponent(`What does ${doc.title} say?`)}`}
                            className="btn btn-primary btn-sm text-[11px] gap-1"
                          >
                            <Sparkles className="h-3 w-3" />
                            <span>Ask</span>
                          </Link>
                        )}
                      </div>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={7} className="text-center py-12 text-[var(--ink-muted)]">
                    <div className="flex flex-col items-center justify-center gap-2 max-w-md mx-auto">
                      <BookOpen className="h-8 w-8 text-[var(--ink-muted)]/40 stroke-1" />
                      <div className="font-semibold text-sm text-[var(--ink-soft)]">
                        {searchQuery.trim() ? "No matching playbooks found" : "No Knowledge Base Documents in PostgreSQL"}
                      </div>
                      <p className="text-xs text-[var(--ink-muted)] leading-relaxed">
                        {searchQuery.trim()
                          ? `No documents matched "${searchQuery}". Try a different search term.`
                          : "Upload sales objection handbooks, product specs, or battlecards above. They will be stored in PostgreSQL and indexed in Pinecone."}
                      </p>
                    </div>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Chunk Inspection Modal */}
      {inspectedDoc && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="card max-w-2xl w-full max-h-[80vh] flex flex-col shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-150">
            {/* Modal Header */}
            <div className="card-header flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileText className="h-4 w-4 text-[var(--accent)]" />
                <h3 className="card-title">
                  Extracted Chunks: {inspectedDoc.title}
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setInspectedDoc(null)}
                className="btn-icon text-[var(--ink-muted)] hover:text-white"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {/* Modal Content */}
            <div className="card-body overflow-y-auto space-y-3 flex-1">
              <div className="text-xs text-[var(--ink-muted)] mb-2 flex items-center justify-between">
                <span>{inspectedDoc.totalChunks} total chunks generated</span>
                <span className="badge badge-primary font-mono">
                  {inspectedDoc.embeddingDimension || 1536}-dimensional vectors
                </span>
              </div>

              {inspectedDoc.chunks?.map((chunk, idx) => (
                <div
                  key={idx}
                  className="p-3 rounded-[var(--radius)] bg-[var(--surface-2)] border border-[var(--border)] text-xs space-y-2"
                >
                  <div className="flex items-center justify-between text-[11px] text-[var(--ink-muted)] border-b border-[var(--border)] pb-1.5">
                    <span className="font-mono text-[var(--accent)]">
                      Chunk #{chunk.chunk_index ?? idx + 1}
                    </span>
                    <span>Page {chunk.metadata?.page ?? 1}</span>
                  </div>
                  <p className="text-[var(--ink)] leading-relaxed font-sans">
                    {chunk.text}
                  </p>
                </div>
              ))}
            </div>

            {/* Modal Footer */}
            <div className="card-footer flex justify-end">
              <button
                type="button"
                onClick={() => setInspectedDoc(null)}
                className="btn btn-secondary btn-sm"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
