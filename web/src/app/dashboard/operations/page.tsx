"use client";

import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  RefreshCw,
  Upload,
  CheckCircle,
  ChevronRight,
  ClipboardList,
  Phone,
  Brain,
  X,
  AlertTriangle,
  Mail,
  Bell,
  BellOff,
  Send,
  Trash2,
  Loader2,
} from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ApiErrorBanner } from "@/components/dashboard/ApiErrorBanner";
import { presignUpload, uploadToS3 } from "@/lib/upload";
import {
  fetchFindings,
  fetchTasks,
  acknowledgeFinding,
  subscribeDigest,
  listSubscriptions,
  unsubscribeDigest,
  sendDigestNow,
  fetchSchedulerStatus,
  type Finding,
  type TaskListResponse,
  type Subscription,
  type SchedulerStatus,
} from "@/lib/sentinel-api";

// ─── Triage State Types ─────────────────────────────────────

type TriageStatus =
  | "untriaged"
  | "task_created"
  | "vendor_call"
  | "investigating"
  | "dismissed";

interface TriageRecord {
  findingId: string;
  status: TriageStatus;
  updatedAt: string;
  notes?: string;
}

interface TriageStore {
  records: Record<string, TriageRecord>;
  version: 1;
}

const TRIAGE_KEY = "profit-sentinel-triage-v1";

function loadTriageStore(): TriageStore {
  if (typeof window === "undefined") return { records: {}, version: 1 };
  try {
    const raw = localStorage.getItem(TRIAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch {
    // ignore parse errors
  }
  return { records: {}, version: 1 };
}

function saveTriageStore(store: TriageStore): void {
  localStorage.setItem(TRIAGE_KEY, JSON.stringify(store));
}

// ─── Helpers ────────────────────────────────────────────────

function formatDollar(amount: number): string {
  if (amount >= 1_000_000) return `$${(amount / 1_000_000).toFixed(1)}M`;
  if (amount >= 1_000) return `$${(amount / 1_000).toFixed(1)}K`;
  return `$${amount.toFixed(0)}`;
}

function formatFindingType(t: string): string {
  return t.replace(/([A-Z])/g, " $1").trim();
}

function severityColor(severity: string): { bg: string; text: string; border: string } {
  switch (severity.toLowerCase()) {
    case "critical":
      return { bg: "bg-red-500/10", text: "text-red-400", border: "border-red-500/20" };
    case "high":
      return { bg: "bg-orange-500/10", text: "text-orange-400", border: "border-orange-500/20" };
    case "medium":
      return { bg: "bg-yellow-500/10", text: "text-yellow-400", border: "border-yellow-500/20" };
    case "low":
      return { bg: "bg-blue-500/10", text: "text-blue-400", border: "border-blue-500/20" };
    case "info":
      return { bg: "bg-emerald-500/10", text: "text-emerald-400", border: "border-emerald-500/20" };
    default:
      return { bg: "bg-slate-500/10", text: "text-slate-400", border: "border-slate-500/20" };
  }
}

function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function triageStatusLabel(status: TriageStatus): string {
  switch (status) {
    case "task_created":
      return "Task Created";
    case "vendor_call":
      return "Vendor Call";
    case "investigating":
      return "Investigating";
    case "dismissed":
      return "Dismissed";
    default:
      return "Untriaged";
  }
}

// ─── File Upload Constants ──────────────────────────────────

const ALLOWED_TYPES = [
  "text/csv",
  "application/vnd.ms-excel",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
];
const ALLOWED_EXT = [".csv", ".xls", ".xlsx"];

// ─── Main Page Component ────────────────────────────────────

export default function OperationsHubPage() {
  const router = useRouter();

  // Store context from URL params (e.g. from Stores page)
  const [storeId, setStoreId] = useState<string>("");
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const sid = params.get("store_id");
    if (sid) setStoreId(sid);
  }, []);

  // Data state
  const [findings, setFindings] = useState<Finding[]>([]);
  const [tasks, setTasks] = useState<TaskListResponse["tasks"]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Triage state
  const [triageStore, setTriageStore] = useState<TriageStore>(() =>
    loadTriageStore()
  );

  // Upload state
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Sync triage to localStorage when state changes
  useEffect(() => {
    saveTriageStore(triageStore);
  }, [triageStore]);

  const updateTriage = useCallback(
    (findingId: string, status: TriageStatus) => {
      setTriageStore((prev) => ({
        ...prev,
        records: {
          ...prev.records,
          [findingId]: {
            findingId,
            status,
            updatedAt: new Date().toISOString(),
          },
        },
      }));
    },
    []
  );

  // ─── Data Fetching ──────────────────────────────────────

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [findingsRes, tasksRes] = await Promise.all([
        fetchFindings({
          status: "active",
          sort_by: "dollar_impact",
          page_size: 100,
        }),
        fetchTasks().catch(() => null),
      ]);
      setFindings(findingsRes?.findings ?? []);
      setTasks(tasksRes?.tasks ?? []);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // ─── Upload Handlers ───────────────────────────────────

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const processFile = useCallback(
    async (file: File) => {
      const ext = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();
      if (!ALLOWED_TYPES.includes(file.type) && !ALLOWED_EXT.includes(ext)) {
        setUploadError("Please upload a CSV, XLS, or XLSX file");
        return;
      }

      setUploading(true);
      setUploadError(null);
      try {
        const presign = await presignUpload(file.name, "", storeId);
        await uploadToS3(presign, file);
        const storeParam = storeId ? `&store_id=${encodeURIComponent(storeId)}` : "";
        router.push(
          `/analyze?s3Key=${encodeURIComponent(presign.key)}&filename=${encodeURIComponent(file.name)}&from=dashboard${storeParam}`
        );
      } catch (err) {
        setUploadError(err instanceof Error ? err.message : "Upload failed");
        setUploading(false);
      }
    },
    [router, storeId]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(false);
      const file = e.dataTransfer.files?.[0];
      if (file) processFile(file);
    },
    [processFile]
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) processFile(file);
    },
    [processFile]
  );

  // ─── Triage Filtering ──────────────────────────────────

  // Active findings: everything NOT dismissed. Sorted by impact.
  const activeFindings = findings
    .filter((f) => {
      const record = triageStore.records[f.id];
      return (
        (!record || record.status !== "dismissed") &&
        f.acknowledged !== true
      );
    })
    .sort((a, b) => b.dollar_impact - a.dollar_impact);

  // Count untriaged (no action taken yet)
  const untriagedCount = activeFindings.filter((f) => {
    const record = triageStore.records[f.id];
    return !record || record.status === "untriaged";
  }).length;

  // In-progress summary counts
  const taskCreatedCount = activeFindings.filter(
    (f) => triageStore.records[f.id]?.status === "task_created"
  ).length;
  const vendorCallCount = activeFindings.filter(
    (f) => triageStore.records[f.id]?.status === "vendor_call"
  ).length;
  const investigatingCount = activeFindings.filter(
    (f) => triageStore.records[f.id]?.status === "investigating"
  ).length;
  const activeTaskCount = tasks.filter((t) => t.status !== "completed").length;

  // Recently resolved: dismissed within the last 7 days
  const sevenDaysAgo = Date.now() - 7 * 24 * 60 * 60 * 1000;
  const recentlyResolved = findings.filter((f) => {
    const record = triageStore.records[f.id];
    if (!record || record.status !== "dismissed") return false;
    return new Date(record.updatedAt).getTime() >= sevenDaysAgo;
  });

  // ─── Render ─────────────────────────────────────────────

  return (
    <div className="p-6 lg:p-8 max-w-6xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-white">Operations Hub</h1>
          <p className="text-sm text-slate-400 mt-1">
            Triage findings, track progress, and manage your workflow
          </p>
        </div>
        <button
          onClick={load}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-sm font-medium rounded-lg transition-colors"
        >
          <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
          Refresh
        </button>
      </div>

      {/* Error */}
      <ApiErrorBanner error={error} onRetry={load} />

      {/* Upload Zone (compact) */}
      <div className="mb-6">
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.xls,.xlsx"
          onChange={handleFileInput}
          className="hidden"
        />
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => !uploading && fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-4 cursor-pointer transition-all ${
            uploading
              ? "border-emerald-500/50 bg-emerald-500/5"
              : dragActive
                ? "border-emerald-500 bg-emerald-500/5"
                : "border-slate-700 hover:border-emerald-500/50 bg-slate-800/30"
          }`}
        >
          {uploading ? (
            <div className="flex items-center gap-3">
              <Loader2 size={20} className="text-emerald-400 animate-spin" />
              <span className="text-sm text-white">
                Uploading &amp; preparing...
              </span>
            </div>
          ) : (
            <div className="flex items-center gap-3">
              <Upload size={20} className="text-slate-500" />
              <div>
                <p className="text-sm font-medium text-white">
                  Run New Analysis
                </p>
                <p className="text-xs text-slate-500">
                  Drop a CSV, XLS, or XLSX file here, or click to browse
                </p>
              </div>
            </div>
          )}
        </div>
        {uploadError && (
          <p className="text-red-400 text-sm mt-2">{uploadError}</p>
        )}
      </div>

      {/* Loading */}
      {loading && findings.length === 0 && (
        <div className="flex items-center justify-center py-24">
          <div className="text-center">
            <div className="w-10 h-10 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-slate-400 text-sm">Loading findings...</p>
          </div>
        </div>
      )}

      {/* Content (shown when not loading or when we have data) */}
      {(!loading || findings.length > 0) && !error && (
        <>
          {/* Workflow Summary Bar */}
          {(activeTaskCount > 0 || taskCreatedCount > 0 || vendorCallCount > 0 || investigatingCount > 0) && (
            <div className="mb-6 grid grid-cols-2 lg:grid-cols-4 gap-3">
              {activeTaskCount > 0 && (
                <Link
                  href="/dashboard/tasks"
                  className="flex items-center gap-2.5 bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2.5 hover:border-slate-600/50 transition-colors"
                >
                  <ClipboardList size={14} className="text-emerald-400" />
                  <span className="text-xs text-slate-300">
                    <strong className="text-white">{activeTaskCount}</strong> active task{activeTaskCount !== 1 ? "s" : ""}
                  </span>
                </Link>
              )}
              {taskCreatedCount > 0 && (
                <Link
                  href="/dashboard/tasks"
                  className="flex items-center gap-2.5 bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2.5 hover:border-slate-600/50 transition-colors"
                >
                  <ClipboardList size={14} className="text-emerald-400" />
                  <span className="text-xs text-slate-300">
                    <strong className="text-white">{taskCreatedCount}</strong> task{taskCreatedCount !== 1 ? "s" : ""} created
                  </span>
                </Link>
              )}
              {vendorCallCount > 0 && (
                <Link
                  href="/dashboard/vendor"
                  className="flex items-center gap-2.5 bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2.5 hover:border-slate-600/50 transition-colors"
                >
                  <Phone size={14} className="text-blue-400" />
                  <span className="text-xs text-slate-300">
                    <strong className="text-white">{vendorCallCount}</strong> vendor call{vendorCallCount !== 1 ? "s" : ""}
                  </span>
                </Link>
              )}
              {investigatingCount > 0 && (
                <Link
                  href="/dashboard/explain"
                  className="flex items-center gap-2.5 bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2.5 hover:border-slate-600/50 transition-colors"
                >
                  <Brain size={14} className="text-violet-400" />
                  <span className="text-xs text-slate-300">
                    <strong className="text-white">{investigatingCount}</strong> investigating
                  </span>
                </Link>
              )}
            </div>
          )}

          {/* Findings List */}
          <div className="mb-8">
            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              Findings
              <span className="px-2 py-0.5 bg-slate-500/10 text-slate-400 text-xs rounded-full border border-slate-500/20">
                {activeFindings.length}
              </span>
              {untriagedCount > 0 && (
                <span className="px-2 py-0.5 bg-red-500/10 text-red-400 text-xs rounded-full border border-red-500/20">
                  {untriagedCount} needs triage
                </span>
              )}
            </h2>

            {activeFindings.length === 0 ? (
              <div className="text-center py-12 bg-slate-800/30 border border-slate-700/50 rounded-xl">
                <CheckCircle
                  size={32}
                  className="mx-auto mb-3 text-emerald-400 opacity-60"
                />
                <p className="text-sm font-medium text-white">
                  All caught up! No active findings.
                </p>
                <p className="text-xs text-slate-500 mt-1">
                  Upload a new file to run another analysis
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {activeFindings.map((finding) => (
                  <TriageFindingCard
                    key={finding.id}
                    finding={finding}
                    triageStatus={triageStore.records[finding.id]?.status ?? null}
                    onTriage={updateTriage}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Recently Resolved */}
          <div className="mb-8">
            <details className="group">
              <summary className="cursor-pointer flex items-center gap-2 text-slate-400 hover:text-white transition-colors">
                <ChevronRight className="w-4 h-4 group-open:rotate-90 transition-transform" />
                <span className="text-lg font-semibold">
                  Recently Resolved ({recentlyResolved.length})
                </span>
              </summary>
              <div className="mt-4 space-y-2">
                {recentlyResolved.length === 0 ? (
                  <p className="text-sm text-slate-500 px-4 py-3">
                    No recent activity. Your workflow history will appear here.
                  </p>
                ) : (
                  recentlyResolved.map((finding) => {
                    const record = triageStore.records[finding.id];
                    return (
                      <div
                        key={finding.id}
                        className="flex items-center justify-between px-4 py-3 bg-slate-800/30 rounded-lg"
                      >
                        <div className="flex items-center gap-3">
                          <span className="text-sm text-slate-300">
                            {finding.title || formatFindingType(finding.type)}
                          </span>
                          {record && (
                            <span className="text-xs text-slate-500">
                              {relativeTime(record.updatedAt)}
                            </span>
                          )}
                        </div>
                        <button
                          onClick={() => updateTriage(finding.id, "untriaged")}
                          className="text-xs text-slate-500 hover:text-emerald-400 transition-colors"
                        >
                          Undo
                        </button>
                      </div>
                    );
                  })
                )}
              </div>
            </details>
          </div>

          {/* Email Digest Section */}
          <EmailDigestSection />
        </>
      )}
    </div>
  );
}

// ─── Triage Finding Card ────────────────────────────────────

const TRIAGE_STATUS_BADGES: Record<
  string,
  { label: string; className: string }
> = {
  task_created: {
    label: "Task Created",
    className: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
  },
  vendor_call: {
    label: "Vendor Call",
    className: "bg-blue-500/10 text-blue-400 border-blue-500/20",
  },
  investigating: {
    label: "Investigating",
    className: "bg-violet-500/10 text-violet-400 border-violet-500/20",
  },
};

function TriageFindingCard({
  finding,
  triageStatus,
  onTriage,
}: {
  finding: Finding;
  triageStatus: TriageStatus | null;
  onTriage: (findingId: string, status: TriageStatus) => void;
}) {
  const sev = severityColor(finding.severity);
  const hasAction = triageStatus && triageStatus !== "untriaged";
  const statusBadge = hasAction ? TRIAGE_STATUS_BADGES[triageStatus] : null;

  return (
    <div
      className={`bg-slate-800/50 border rounded-xl p-4 ${
        hasAction ? "border-slate-700/30" : "border-slate-700/50"
      }`}
    >
      <div className="flex items-start gap-3 mb-3">
        {/* Severity badge */}
        <span
          className={`shrink-0 px-2 py-0.5 text-[10px] font-bold uppercase rounded ${sev.bg} ${sev.text} border ${sev.border}`}
        >
          {finding.severity}
        </span>

        {/* Status badge (if triaged) */}
        {statusBadge && (
          <span
            className={`shrink-0 px-2 py-0.5 text-[10px] font-bold rounded border ${statusBadge.className}`}
          >
            {statusBadge.label}
          </span>
        )}

        {/* Title + Description */}
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-white">
            {finding.title || formatFindingType(finding.type)}
          </p>
          {finding.description && (
            <p className="text-sm text-slate-400 mt-1 line-clamp-2">
              {finding.description}
            </p>
          )}
        </div>

        {/* Dollar impact */}
        <span className="text-sm font-bold text-white shrink-0">
          {formatDollar(finding.dollar_impact)}
        </span>
      </div>

      {/* Action buttons */}
      <div className="flex items-center gap-2 flex-wrap">
        <Link
          href={`/dashboard/tasks?delegate=${finding.id}`}
          onClick={() => onTriage(finding.id, "task_created")}
          className={`flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg transition-colors ${
            triageStatus === "task_created"
              ? "bg-emerald-500/20 border border-emerald-500/30 text-emerald-300"
              : "bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 hover:bg-emerald-500/20"
          }`}
        >
          <ClipboardList size={12} />
          {triageStatus === "task_created" ? "View Task" : "Create Task"}
        </Link>
        <Link
          href={`/dashboard/vendor?issue=${finding.id}`}
          onClick={() => onTriage(finding.id, "vendor_call")}
          className={`flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg transition-colors ${
            triageStatus === "vendor_call"
              ? "bg-blue-500/20 border border-blue-500/30 text-blue-300"
              : "bg-blue-500/10 border border-blue-500/20 text-blue-400 hover:bg-blue-500/20"
          }`}
        >
          <Phone size={12} />
          Vendor Call
        </Link>
        <Link
          href={`/dashboard/explain?issue=${finding.id}`}
          onClick={() => onTriage(finding.id, "investigating")}
          className={`flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg transition-colors ${
            triageStatus === "investigating"
              ? "bg-violet-500/20 border border-violet-500/30 text-violet-300"
              : "bg-violet-500/10 border border-violet-500/20 text-violet-400 hover:bg-violet-500/20"
          }`}
        >
          <Brain size={12} />
          Investigate
        </Link>
        <button
          onClick={() => {
            onTriage(finding.id, "dismissed");
            acknowledgeFinding(finding.id).catch(() => {});
          }}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-500/10 border border-slate-500/20 text-slate-400 text-xs rounded-lg hover:bg-slate-500/20 transition-colors"
        >
          <X size={12} />
          Dismiss
        </button>
        {hasAction && (
          <button
            onClick={() => onTriage(finding.id, "untriaged")}
            className="text-[10px] text-slate-600 hover:text-slate-400 transition-colors ml-1"
          >
            Reset
          </button>
        )}
      </div>
    </div>
  );
}


// ─── Email Digest Section ───────────────────────────────────

function EmailDigestSection() {
  const [expanded, setExpanded] = useState(false);
  const [subs, setSubs] = useState<Subscription[]>([]);
  const [status, setStatus] = useState<SchedulerStatus | null>(null);
  const [email, setEmail] = useState("");
  const [sendHour, setSendHour] = useState(6);
  const [tz, setTz] = useState("America/New_York");
  const [sending, setSending] = useState(false);
  const [subscribing, setSubscribing] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    try {
      const [subsRes, statusRes] = await Promise.all([
        listSubscriptions(),
        fetchSchedulerStatus(),
      ]);
      setSubs(subsRes.subscriptions);
      setStatus(statusRes);
    } catch {
      // Non-critical -- fail silently
    }
  }, []);

  useEffect(() => {
    if (expanded) loadData();
  }, [expanded, loadData]);

  const handleSubscribe = async () => {
    if (!email) return;
    setSubscribing(true);
    setMsg(null);
    try {
      await subscribeDigest({ email, send_hour: sendHour, timezone: tz });
      setMsg(`Subscribed ${email}`);
      setEmail("");
      await loadData();
    } catch (err) {
      setMsg(`Error: ${(err as Error).message}`);
    } finally {
      setSubscribing(false);
    }
  };

  const handleUnsubscribe = async (e: string) => {
    try {
      await unsubscribeDigest(e);
      setMsg(`Unsubscribed ${e}`);
      await loadData();
    } catch (err) {
      setMsg(`Error: ${(err as Error).message}`);
    }
  };

  const handleSendNow = async (e: string) => {
    setSending(true);
    setMsg(null);
    try {
      const res = await sendDigestNow(e);
      setMsg(res.message);
    } catch (err) {
      setMsg(`Error: ${(err as Error).message}`);
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="mb-8">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 text-sm text-slate-400 hover:text-white transition-colors"
      >
        <Mail size={14} className="text-emerald-400" />
        <span className="font-medium">Email Digest</span>
        {status && status.subscribers > 0 && (
          <span className="px-1.5 py-0.5 bg-emerald-500/10 text-emerald-400 text-[10px] rounded-full border border-emerald-500/20">
            {status.subscribers} subscriber
            {status.subscribers !== 1 ? "s" : ""}
          </span>
        )}
        <ChevronRight
          size={12}
          className={`transition-transform ${expanded ? "rotate-90" : ""}`}
        />
      </button>

      {expanded && (
        <div className="mt-3 bg-slate-800/50 border border-slate-700/50 rounded-xl p-5 space-y-4">
          {/* Status */}
          {status && (
            <div className="flex items-center gap-3 text-xs text-slate-500">
              {status.enabled ? (
                <span className="flex items-center gap-1 text-emerald-400">
                  <Bell size={12} /> Scheduler active
                </span>
              ) : (
                <span className="flex items-center gap-1 text-slate-500">
                  <BellOff size={12} /> Scheduler disabled
                </span>
              )}
              <span>&middot;</span>
              <span>
                {status.subscribers} subscriber
                {status.subscribers !== 1 ? "s" : ""}
              </span>
              <span>&middot;</span>
              <span>Default hour: {status.send_hour}:00</span>
            </div>
          )}

          {/* Subscribe form */}
          <div className="flex items-end gap-2">
            <div className="flex-1">
              <label className="block text-[10px] text-slate-500 mb-1">
                Email
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="manager@store.com"
                className="w-full bg-slate-900/50 border border-slate-700 text-slate-300 text-sm rounded-lg px-3 py-2 focus:outline-none focus:border-emerald-500"
              />
            </div>
            <div className="w-20">
              <label className="block text-[10px] text-slate-500 mb-1">
                Hour
              </label>
              <select
                value={sendHour}
                onChange={(e) => setSendHour(Number(e.target.value))}
                className="w-full bg-slate-900/50 border border-slate-700 text-slate-300 text-sm rounded-lg px-2 py-2"
              >
                {Array.from({ length: 24 }, (_, i) => (
                  <option key={i} value={i}>
                    {i === 0
                      ? "12 AM"
                      : i < 12
                        ? `${i} AM`
                        : i === 12
                          ? "12 PM"
                          : `${i - 12} PM`}
                  </option>
                ))}
              </select>
            </div>
            <div className="w-40">
              <label className="block text-[10px] text-slate-500 mb-1">
                Timezone
              </label>
              <select
                value={tz}
                onChange={(e) => setTz(e.target.value)}
                className="w-full bg-slate-900/50 border border-slate-700 text-slate-300 text-sm rounded-lg px-2 py-2"
              >
                <option value="America/New_York">Eastern</option>
                <option value="America/Chicago">Central</option>
                <option value="America/Denver">Mountain</option>
                <option value="America/Los_Angeles">Pacific</option>
                <option value="UTC">UTC</option>
              </select>
            </div>
            <button
              onClick={handleSubscribe}
              disabled={subscribing || !email}
              className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-sm font-medium rounded-lg transition-colors"
            >
              {subscribing ? "..." : "Subscribe"}
            </button>
          </div>

          {/* Message */}
          {msg && (
            <p
              className={`text-xs ${msg.startsWith("Error") ? "text-red-400" : "text-emerald-400"}`}
            >
              {msg}
            </p>
          )}

          {/* Subscriber list */}
          {subs.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs text-slate-500 uppercase font-medium">
                Active Subscribers
              </p>
              {subs.map((sub) => (
                <div
                  key={sub.email}
                  className="flex items-center justify-between px-3 py-2 bg-slate-900/30 rounded-lg"
                >
                  <div className="min-w-0">
                    <p className="text-sm text-slate-300 truncate">
                      {sub.email}
                    </p>
                    <p className="text-[10px] text-slate-600">
                      {sub.send_hour}:00{" "}
                      {sub.timezone.split("/")[1]?.replace("_", " ")}
                      {sub.stores.length > 0 &&
                        ` \u00B7 ${sub.stores.join(", ")}`}
                    </p>
                  </div>
                  <div className="flex items-center gap-1.5 shrink-0">
                    <button
                      onClick={() => handleSendNow(sub.email)}
                      disabled={sending}
                      title="Send now"
                      className="p-1.5 hover:bg-emerald-500/10 rounded-lg text-emerald-400 transition-colors"
                    >
                      <Send size={12} />
                    </button>
                    <button
                      onClick={() => handleUnsubscribe(sub.email)}
                      title="Unsubscribe"
                      className="p-1.5 hover:bg-red-500/10 rounded-lg text-red-400 transition-colors"
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
