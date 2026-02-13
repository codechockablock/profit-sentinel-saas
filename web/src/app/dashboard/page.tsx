"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  AlertCircle,
  AlertTriangle,
  TrendingDown,
  DollarSign,
  Store,
  Clock,
  Zap,
  ChevronRight,
  RefreshCw,
  ArrowUpRight,
  ArrowDownRight,
  Minus,
  UserPlus,
  Mail,
  Send,
  Trash2,
  Bell,
  BellOff,
  Truck,
  Upload,
} from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useRef } from "react";
import { ApiErrorBanner } from "@/components/dashboard/ApiErrorBanner";
import { presignUpload, uploadToS3 } from "@/lib/upload";
import {
  fetchDigest,
  fetchDashboardSummary,
  subscribeDigest,
  listSubscriptions,
  unsubscribeDigest,
  sendDigestNow,
  fetchSchedulerStatus,
  type DigestResponse,
  type DashboardSummaryResponse,
  type Issue,
  type TaskPriority,
  type Subscription,
  type SchedulerStatus,
} from "@/lib/sentinel-api";

// ─── Helpers ─────────────────────────────────────────────────

function formatDollar(amount: number): string {
  if (amount >= 1_000_000) return `$${(amount / 1_000_000).toFixed(1)}M`;
  if (amount >= 1_000) return `$${(amount / 1_000).toFixed(1)}K`;
  return `$${amount.toFixed(0)}`;
}

function issueTypeLabel(t: string): string {
  return t.replace(/([A-Z])/g, " $1").trim();
}

function priorityColor(score: number): { bg: string; text: string; border: string; label: TaskPriority } {
  if (score >= 10) return { bg: "bg-red-500/10", text: "text-red-400", border: "border-red-500/30", label: "critical" };
  if (score >= 8) return { bg: "bg-orange-500/10", text: "text-orange-400", border: "border-orange-500/30", label: "high" };
  if (score >= 5) return { bg: "bg-yellow-500/10", text: "text-yellow-400", border: "border-yellow-500/30", label: "medium" };
  return { bg: "bg-blue-500/10", text: "text-blue-400", border: "border-blue-500/30", label: "low" };
}

function trendIcon(dir: string) {
  if (dir === "Worsening") return <ArrowUpRight size={14} className="text-red-400" />;
  if (dir === "Improving") return <ArrowDownRight size={14} className="text-emerald-400" />;
  return <Minus size={14} className="text-slate-500" />;
}

// ─── Component ───────────────────────────────────────────────

export default function DigestPage() {
  const router = useRouter();
  const [data, setData] = useState<DigestResponse | null>(null);
  const [dashData, setDashData] = useState<DashboardSummaryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [topK, setTopK] = useState(10);
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const ALLOWED_TYPES = [
    "text/csv",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  ];
  const ALLOWED_EXT = [".csv", ".xls", ".xlsx"];

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
        const presign = await presignUpload(file.name);
        await uploadToS3(presign, file);
        router.push(
          `/analyze?s3Key=${encodeURIComponent(presign.key)}&filename=${encodeURIComponent(file.name)}&from=dashboard`
        );
      } catch (err) {
        setUploadError(err instanceof Error ? err.message : "Upload failed");
        setUploading(false);
      }
    },
    [router]
  );

  const handleOnboardDrop = useCallback(
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

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [res, dash] = await Promise.all([
        fetchDigest(undefined, topK),
        fetchDashboardSummary().catch(() => null),
      ]);
      setData(res);
      setDashData(dash);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, [topK]);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div className="p-6 lg:p-8 max-w-6xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-white">Morning Digest</h1>
          <p className="text-sm text-slate-400 mt-1">
            Top priority issues across all stores
          </p>
        </div>
        <div className="flex items-center gap-3">
          {data && (
            <select
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="bg-slate-800 border border-slate-700 text-slate-300 text-sm rounded-lg px-3 py-2 focus:outline-none focus:border-emerald-500"
            >
              <option value={5}>Top 5</option>
              <option value={10}>Top 10</option>
              <option value={20}>Top 20</option>
            </select>
          )}
          <button
            onClick={load}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-sm font-medium rounded-lg transition-colors"
          >
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
            Refresh
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <ApiErrorBanner error={error} onRetry={load} />
      )}

      {/* Onboarding — only when load succeeded but returned no data */}
      {!loading && !error && !data && (
        <div className="mb-6">
          {/* Welcome hero */}
          <div className="p-8 bg-slate-800/50 border border-slate-700/50 rounded-xl text-center mb-6">
            <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-emerald-500/10 flex items-center justify-center">
              <Zap size={28} className="text-emerald-400" />
            </div>
            <h2 className="text-xl font-bold text-white mb-2">
              Welcome to Your Morning Digest
            </h2>
            <p className="text-slate-400 mb-8 max-w-lg mx-auto">
              The Morning Digest surfaces your highest-priority inventory issues
              each day. Upload an inventory file to get started.
            </p>

            {/* Upload drop zone */}
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
              onDrop={handleOnboardDrop}
              onClick={() => !uploading && fileInputRef.current?.click()}
              className={`max-w-md mx-auto border-2 border-dashed rounded-2xl p-8 cursor-pointer transition-all ${
                uploading
                  ? "border-emerald-500/50 bg-emerald-500/5"
                  : dragActive
                    ? "border-emerald-500 bg-emerald-500/5"
                    : "border-slate-700 hover:border-emerald-500/50 bg-slate-800/30"
              }`}
            >
              {uploading ? (
                <>
                  <div className="w-10 h-10 mx-auto mb-3 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
                  <p className="text-sm font-medium text-white mb-1">
                    Uploading &amp; preparing...
                  </p>
                  <p className="text-xs text-slate-500">
                    This will take a few seconds
                  </p>
                </>
              ) : (
                <>
                  <Upload className="w-10 h-10 mx-auto mb-3 text-slate-500" />
                  <p className="text-sm font-medium text-white mb-1">
                    Drop your CSV, XLS, or XLSX file here
                  </p>
                  <p className="text-xs text-slate-500">
                    or click to browse files
                  </p>
                </>
              )}
            </div>
            {uploadError && (
              <p className="text-red-400 text-sm mt-3">{uploadError}</p>
            )}
          </div>

          {/* How it works */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <OnboardStepCard
              step={1}
              icon={<Upload size={20} />}
              title="Upload"
              description="Drop an inventory file (CSV or XLSX) for analysis"
            />
            <OnboardStepCard
              step={2}
              icon={<Zap size={20} />}
              title="Analyze"
              description="Our engine scans for 11 types of profit leaks in under 60 seconds"
            />
            <OnboardStepCard
              step={3}
              icon={<AlertCircle size={20} />}
              title="Act"
              description="Get prioritized issues with dollar impact and action items"
            />
          </div>
        </div>
      )}

      {/* Loading */}
      {loading && !data && (
        <div className="flex items-center justify-center py-24">
          <div className="text-center">
            <div className="w-10 h-10 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-slate-400 text-sm">Running pipeline analysis...</p>
          </div>
        </div>
      )}

      {/* Data */}
      {data && !error && (
        <>
          {/* Summary cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <SummaryCard
              label="Total Issues"
              value={String(data.issue_count)}
              icon={<AlertCircle size={20} />}
              color="text-red-400"
            />
            <SummaryCard
              label="Dollar Impact"
              value={formatDollar(data.total_dollar_impact)}
              icon={<DollarSign size={20} />}
              color="text-emerald-400"
            />
            <SummaryCard
              label="Stores"
              value={String(data.store_filter.length)}
              icon={<Store size={20} />}
              color="text-blue-400"
            />
            <SummaryCard
              label="Pipeline"
              value={`${data.digest.pipeline_ms}ms`}
              icon={
                data.digest.pipeline_ms < 3000 ? (
                  <Zap size={20} />
                ) : (
                  <Clock size={20} />
                )
              }
              color={
                data.digest.pipeline_ms < 3000
                  ? "text-emerald-400"
                  : "text-slate-400"
              }
            />
          </div>

          {/* Engine 2 Insights (additive — only shows when data is available) */}
          {dashData && (dashData.prediction_count > 0 || dashData.transfer_stats.stores_registered >= 2) && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
              {/* Engine 2 status */}
              <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
                <div className="flex items-center gap-2 mb-2">
                  <span
                    className={`w-2.5 h-2.5 rounded-full ${
                      dashData.engine2_status === "active"
                        ? "bg-emerald-400"
                        : dashData.engine2_status === "warming_up"
                        ? "bg-yellow-400"
                        : "bg-slate-600"
                    }`}
                  />
                  <span className="text-xs text-slate-400">Smart Analysis</span>
                </div>
                <p className="text-lg font-bold text-white">
                  {dashData.prediction_count} Prediction{dashData.prediction_count !== 1 ? "s" : ""}
                </p>
                {dashData.top_predictions.length > 0 && (
                  <Link
                    href="/dashboard/predictions"
                    className="flex items-center gap-1 text-xs text-emerald-400 mt-1.5 hover:underline"
                  >
                    <AlertTriangle size={10} />
                    View predictions
                    <ChevronRight size={10} />
                  </Link>
                )}
              </div>

              {/* Top predictions preview */}
              {dashData.top_predictions.length > 0 && (
                <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
                  <p className="text-xs text-slate-400 mb-2">Top Predictions</p>
                  <div className="space-y-1.5">
                    {dashData.top_predictions.slice(0, 3).map((pred, i) => (
                      <div key={i} className="flex items-center justify-between text-xs">
                        <span className="text-slate-300 truncate mr-2">
                          {(pred as Record<string, unknown>).entity_id as string || `Prediction ${i + 1}`}
                        </span>
                        <span className="text-amber-400 font-medium shrink-0">
                          {(pred as Record<string, unknown>).confidence
                            ? `${((pred as Record<string, unknown>).confidence as number * 100).toFixed(0)}%`
                            : ""}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Transfer opportunity */}
              {dashData.transfer_stats.stores_registered >= 2 && (
                <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Truck size={14} className="text-emerald-400" />
                    <span className="text-xs text-slate-400">Transfer Opportunities</span>
                  </div>
                  <p className="text-lg font-bold text-white">
                    {dashData.transfer_stats.stores_registered} Stores
                  </p>
                  <Link
                    href="/dashboard/transfers"
                    className="flex items-center gap-1 text-xs text-emerald-400 mt-1.5 hover:underline"
                  >
                    View recommendations
                    <ChevronRight size={10} />
                  </Link>
                </div>
              )}
            </div>
          )}

          {/* Email Digest Section */}
          <EmailDigestSection />

          {/* Issue list */}
          <div className="space-y-3">
            {data.digest.issues.map((issue) => (
              <IssueCard key={issue.id} issue={issue} />
            ))}

            {data.digest.issues.length === 0 && (
              <div className="text-center py-16 text-slate-500">
                <AlertCircle size={32} className="mx-auto mb-3 opacity-50" />
                <p className="font-medium">All Clear</p>
                <p className="text-sm mt-1">No priority issues detected</p>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

// ─── Email Digest ───────────────────────────────────────────

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
      // Non-critical — fail silently
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
            {status.subscribers} subscriber{status.subscribers !== 1 ? "s" : ""}
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
              <span>{status.subscribers} subscriber{status.subscribers !== 1 ? "s" : ""}</span>
              <span>&middot;</span>
              <span>Default hour: {status.send_hour}:00</span>
            </div>
          )}

          {/* Subscribe form */}
          <div className="flex items-end gap-2">
            <div className="flex-1">
              <label className="block text-[10px] text-slate-500 mb-1">Email</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="manager@store.com"
                className="w-full bg-slate-900/50 border border-slate-700 text-slate-300 text-sm rounded-lg px-3 py-2 focus:outline-none focus:border-emerald-500"
              />
            </div>
            <div className="w-20">
              <label className="block text-[10px] text-slate-500 mb-1">Hour</label>
              <select
                value={sendHour}
                onChange={(e) => setSendHour(Number(e.target.value))}
                className="w-full bg-slate-900/50 border border-slate-700 text-slate-300 text-sm rounded-lg px-2 py-2"
              >
                {Array.from({ length: 24 }, (_, i) => (
                  <option key={i} value={i}>
                    {i === 0 ? "12 AM" : i < 12 ? `${i} AM` : i === 12 ? "12 PM" : `${i - 12} PM`}
                  </option>
                ))}
              </select>
            </div>
            <div className="w-40">
              <label className="block text-[10px] text-slate-500 mb-1">Timezone</label>
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
            <p className={`text-xs ${msg.startsWith("Error") ? "text-red-400" : "text-emerald-400"}`}>
              {msg}
            </p>
          )}

          {/* Subscriber list */}
          {subs.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs text-slate-500 uppercase font-medium">Active Subscribers</p>
              {subs.map((sub) => (
                <div
                  key={sub.email}
                  className="flex items-center justify-between px-3 py-2 bg-slate-900/30 rounded-lg"
                >
                  <div className="min-w-0">
                    <p className="text-sm text-slate-300 truncate">{sub.email}</p>
                    <p className="text-[10px] text-slate-600">
                      {sub.send_hour}:00 {sub.timezone.split("/")[1]?.replace("_", " ")}
                      {sub.stores.length > 0 && ` · ${sub.stores.join(", ")}`}
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

// ─── Sub-components ──────────────────────────────────────────

function OnboardStepCard({
  step,
  icon,
  title,
  description,
}: {
  step: number;
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
      <div className="flex items-center gap-3 mb-2">
        <span className="w-6 h-6 rounded-full bg-emerald-500/20 text-emerald-400 text-xs font-bold flex items-center justify-center">
          {step}
        </span>
        <span className="text-emerald-400">{icon}</span>
      </div>
      <p className="text-sm font-medium text-white">{title}</p>
      <p className="text-xs text-slate-500 mt-1">{description}</p>
    </div>
  );
}

function SummaryCard({
  label,
  value,
  icon,
  color,
}: {
  label: string;
  value: string;
  icon: React.ReactNode;
  color: string;
}) {
  return (
    <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
      <div className={`${color} mb-2`}>{icon}</div>
      <p className="text-2xl font-bold text-white">{value}</p>
      <p className="text-xs text-slate-500 mt-1">{label}</p>
    </div>
  );
}

function IssueCard({ issue }: { issue: Issue }) {
  const p = priorityColor(issue.priority_score);
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className={`${p.bg} border ${p.border} rounded-xl overflow-hidden transition-colors`}
    >
      {/* Header row */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-4 px-5 py-4 text-left"
      >
        {/* Priority badge */}
        <span
          className={`shrink-0 px-2 py-0.5 text-[10px] font-bold uppercase rounded ${p.bg} ${p.text} border ${p.border}`}
        >
          {p.label}
        </span>

        {/* Type + store */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-white truncate">
              {issueTypeLabel(issue.issue_type)}
            </span>
            {trendIcon(issue.trend_direction)}
          </div>
          <p className="text-xs text-slate-500 truncate mt-0.5">
            {issue.store_id} &middot; {issue.skus.length} SKU{issue.skus.length !== 1 ? "s" : ""}
            {issue.root_cause && (
              <> &middot; Root cause: {issue.root_cause}</>
            )}
          </p>
        </div>

        {/* Impact */}
        <span className="text-sm font-bold text-white shrink-0">
          {formatDollar(issue.dollar_impact)}
        </span>

        {/* Chevron */}
        <ChevronRight
          size={16}
          className={`text-slate-500 transition-transform ${
            expanded ? "rotate-90" : ""
          }`}
        />
      </button>

      {/* Expanded detail */}
      {expanded && (
        <div className="px-5 pb-4 border-t border-slate-700/30">
          <p className="text-sm text-slate-300 mt-3 mb-4">{issue.context}</p>

          {/* SKU table */}
          {issue.skus.length > 0 && (
            <div className="overflow-x-auto mb-4">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-slate-500 border-b border-slate-700/50">
                    <th className="text-left py-2 pr-4">SKU</th>
                    <th className="text-right py-2 pr-4">Qty</th>
                    <th className="text-right py-2 pr-4">Cost</th>
                    <th className="text-right py-2 pr-4">Retail</th>
                    <th className="text-right py-2">Margin</th>
                  </tr>
                </thead>
                <tbody>
                  {issue.skus.slice(0, 5).map((sku) => (
                    <tr
                      key={sku.sku_id}
                      className="text-slate-300 border-b border-slate-800/50"
                    >
                      <td className="py-1.5 pr-4 font-mono">{sku.sku_id}</td>
                      <td className="text-right py-1.5 pr-4">
                        <span className={sku.qty_on_hand < 0 ? "text-red-400" : ""}>
                          {sku.qty_on_hand.toFixed(0)}
                        </span>
                      </td>
                      <td className="text-right py-1.5 pr-4">${sku.unit_cost.toFixed(2)}</td>
                      <td className="text-right py-1.5 pr-4">${sku.retail_price.toFixed(2)}</td>
                      <td className="text-right py-1.5">
                        {(sku.margin_pct * 100).toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {issue.skus.length > 5 && (
                <p className="text-[10px] text-slate-600 mt-1">
                  +{issue.skus.length - 5} more SKUs
                </p>
              )}
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center gap-2">
            <Link
              href={`/dashboard/explain?issue=${issue.id}`}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-violet-500/10 border border-violet-500/20 text-violet-400 text-xs rounded-lg hover:bg-violet-500/20 transition-colors"
            >
              <TrendingDown size={12} />
              Explain
            </Link>
            <Link
              href={`/dashboard/tasks?delegate=${issue.id}`}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs rounded-lg hover:bg-emerald-500/20 transition-colors"
            >
              <UserPlus size={12} />
              Delegate
            </Link>
            <Link
              href={`/dashboard/vendor?issue=${issue.id}`}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs rounded-lg hover:bg-blue-500/20 transition-colors"
            >
              <ChevronRight size={12} />
              Vendor Prep
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}
