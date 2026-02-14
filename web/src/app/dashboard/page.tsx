"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Minus,
  Zap,
  RefreshCw,
  CheckCircle,
  BarChart3,
  FileSpreadsheet,
  DollarSign,
  Calendar,
  ArrowRight,
} from "lucide-react";
import Link from "next/link";
import {
  BarChart,
  Bar,
  XAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { ApiErrorBanner } from "@/components/dashboard/ApiErrorBanner";
import {
  fetchDashboardSummary,
  fetchFindings,
  fetchStores,
  listAnalyses,
  type DashboardSummaryResponse,
  type Finding,
  type AnalysisListItem,
  type Store,
} from "@/lib/sentinel-api";

// ─── Helpers ──────────────────────────────────────────────────

function formatDollar(amount: number): string {
  if (amount >= 1_000_000) return `$${(amount / 1_000_000).toFixed(1)}M`;
  if (amount >= 1_000) return `$${(amount / 1_000).toFixed(1)}K`;
  return `$${amount.toFixed(0)}`;
}

function severityColor(severity: string): {
  bg: string;
  text: string;
  border: string;
} {
  switch (severity) {
    case "critical":
      return {
        bg: "bg-red-500/10",
        text: "text-red-400",
        border: "border-red-500/30",
      };
    case "high":
      return {
        bg: "bg-orange-500/10",
        text: "text-orange-400",
        border: "border-orange-500/30",
      };
    case "medium":
      return {
        bg: "bg-yellow-500/10",
        text: "text-yellow-400",
        border: "border-yellow-500/30",
      };
    case "low":
      return {
        bg: "bg-blue-500/10",
        text: "text-blue-400",
        border: "border-blue-500/30",
      };
    case "info":
    default:
      return {
        bg: "bg-emerald-500/10",
        text: "text-emerald-400",
        border: "border-emerald-500/30",
      };
  }
}

function computeHealthScore(
  findings: Finding[],
  summary: DashboardSummaryResponse | null,
  analyses: AnalysisListItem[]
): number {
  let score = 100;

  // Deduct for active findings by severity
  const severityDeductions: Record<string, number> = {
    critical: 10,
    high: 6,
    medium: 3,
    low: 1,
    info: 0,
  };
  let findingDeduction = 0;
  for (const f of findings) {
    findingDeduction += severityDeductions[f.severity] ?? 0;
  }
  score -= Math.min(50, findingDeduction);

  // Deduct for department health
  if (summary?.department_status) {
    for (const dept of Object.values(summary.department_status)) {
      if (dept.status === "red") score -= 7;
      else if (dept.status === "yellow") score -= 3;
    }
  }

  // Deduct for stale data
  if (analyses.length === 0) {
    score -= 15;
  } else if (analyses[0]?.created_at) {
    const latestDate = new Date(analyses[0].created_at);
    const daysSince =
      (Date.now() - latestDate.getTime()) / (1000 * 60 * 60 * 24);
    if (daysSince > 7) score -= 15;
    else if (daysSince > 3) score -= 8;
    else if (daysSince > 1) score -= 3;
  }

  // Bonus for recovery progress
  if (summary?.recovery_total) {
    score += Math.min(10, Math.round(summary.recovery_total / 1000));
  }

  // Clamp to 0-100
  return Math.max(0, Math.min(100, score));
}

type TrendDir = "up" | "down" | "flat";

function computeTrend(analyses: AnalysisListItem[]): TrendDir {
  if (analyses.length < 2) return "flat";
  const current = analyses[0]?.total_impact_estimate_high ?? 0;
  const previous = analyses[1]?.total_impact_estimate_high ?? 0;
  if (previous === 0) return "flat";
  // Lower impact = improving (up), higher = worsening (down)
  if (current < previous * 0.9) return "up";
  if (current > previous * 1.1) return "down";
  return "flat";
}

// ─── Main Page ────────────────────────────────────────────────

export default function MorningDigestPage() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [summary, setSummary] = useState<DashboardSummaryResponse | null>(
    null
  );
  const [findings, setFindings] = useState<Finding[]>([]);
  const [analyses, setAnalyses] = useState<AnalysisListItem[]>([]);

  // Store filtering
  const [stores, setStores] = useState<Store[]>([]);
  const [selectedStoreId, setSelectedStoreId] = useState<string>("");

  // Read store_id from URL params on mount
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const sid = params.get("store_id");
    if (sid) setSelectedStoreId(sid);
  }, []);

  // Load stores list
  useEffect(() => {
    fetchStores()
      .then((res) => setStores(res.stores))
      .catch(() => {});
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const storeFilter = selectedStoreId || undefined;
      const [summaryRes, findingsRes, analysesRes] = await Promise.all([
        fetchDashboardSummary(storeFilter).catch(() => null),
        fetchFindings({
          status: "active",
          sort_by: "dollar_impact",
          page_size: 5,
          store_id: storeFilter,
        }).catch(() => null),
        listAnalyses(10).catch(() => null),
      ]);
      setSummary(summaryRes);
      setFindings(findingsRes?.findings ?? []);
      setAnalyses(analysesRes?.analyses ?? []);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, [selectedStoreId]);

  useEffect(() => {
    load();
  }, [load]);

  const hasAnyData =
    summary !== null || analyses.length > 0 || findings.length > 0;

  return (
    <div className="p-6 lg:p-8 max-w-6xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-white">Morning Digest</h1>
          <p className="text-sm text-slate-400 mt-1">
            Your inventory health at a glance
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* Store selector — only shown for multi-store users */}
          {stores.length > 1 && (
            <select
              value={selectedStoreId}
              onChange={(e) => setSelectedStoreId(e.target.value)}
              className="px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-sm text-white focus:outline-none focus:border-emerald-500"
            >
              <option value="">All Stores</option>
              {stores.map((s) => (
                <option key={s.id} value={s.id}>{s.name}</option>
              ))}
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
      {error && <ApiErrorBanner error={error} onRetry={load} />}

      {/* Loading */}
      {loading && !summary && analyses.length === 0 && (
        <div className="flex items-center justify-center py-24">
          <div className="text-center">
            <div className="w-10 h-10 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-slate-400 text-sm">
              Loading your morning digest...
            </p>
          </div>
        </div>
      )}

      {/* Full empty state */}
      {!loading && !error && !hasAnyData && (
        <div className="p-8 bg-slate-800/50 border border-slate-700/50 rounded-xl text-center">
          <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-emerald-500/10 flex items-center justify-center">
            <Zap size={28} className="text-emerald-400" />
          </div>
          <h2 className="text-xl font-bold text-white mb-2">
            Welcome to Morning Digest
          </h2>
          <p className="text-slate-400 mb-6 max-w-lg mx-auto">
            Upload an inventory file to see your health score, top priorities,
            and trends.
          </p>
          <Link
            href="/dashboard/operations"
            className="inline-flex items-center gap-2 px-6 py-3 bg-emerald-600 hover:bg-emerald-500 text-white font-medium rounded-lg transition-colors"
          >
            Get Started — Upload Your Inventory File
            <ArrowRight size={16} />
          </Link>
        </div>
      )}

      {/* Data sections */}
      {hasAnyData && (
        <>
          {/* Row 1: Health Score + Top Priority */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
            <HealthScoreGauge
              findings={findings}
              summary={summary}
              analyses={analyses}
            />
            <div className="lg:col-span-2">
              <TopPriorityCard finding={findings[0] ?? null} />
            </div>
          </div>

          {/* Row 2: Trend Snapshot */}
          <div className="mb-6">
            <TrendSnapshot analyses={analyses} />
          </div>

          {/* Row 3: Quick Stats */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <QuickStatTile
              label="SKUs Analyzed"
              value={
                analyses[0]?.file_row_count
                  ? analyses[0].file_row_count.toLocaleString()
                  : "\u2014"
              }
              icon={<FileSpreadsheet size={18} />}
              href="/dashboard/history"
            />
            <QuickStatTile
              label="Active Findings"
              value={
                summary?.finding_count != null
                  ? String(summary.finding_count)
                  : "\u2014"
              }
              icon={<AlertTriangle size={18} />}
              href="/dashboard/operations"
            />
            <QuickStatTile
              label="Recoverable Profit"
              value={
                summary?.recovery_total != null
                  ? formatDollar(summary.recovery_total)
                  : "\u2014"
              }
              icon={<DollarSign size={18} />}
              href="/dashboard/findings"
            />
            <QuickStatTile
              label="Days Since Analysis"
              value={(() => {
                if (!analyses[0]?.created_at) return "\u2014";
                const days = Math.floor(
                  (Date.now() - new Date(analyses[0].created_at).getTime()) /
                    (1000 * 60 * 60 * 24)
                );
                return String(days);
              })()}
              icon={<Calendar size={18} />}
              href="/dashboard/operations"
              valueColor={(() => {
                if (!analyses[0]?.created_at) return undefined;
                const days = Math.floor(
                  (Date.now() - new Date(analyses[0].created_at).getTime()) /
                    (1000 * 60 * 60 * 24)
                );
                if (days <= 1) return "text-emerald-400";
                if (days <= 3) return "text-yellow-400";
                return "text-red-400";
              })()}
            />
          </div>
        </>
      )}
    </div>
  );
}

// ─── Health Score Gauge ──────────────────────────────────────

function HealthScoreGauge({
  findings,
  summary,
  analyses,
}: {
  findings: Finding[];
  summary: DashboardSummaryResponse | null;
  analyses: AnalysisListItem[];
}) {
  const hasData = summary !== null || analyses.length > 0;
  const score = hasData
    ? computeHealthScore(findings, summary, analyses)
    : null;
  const trend = computeTrend(analyses);

  // SVG gauge config
  const radius = 70;
  const circumference = 2 * Math.PI * radius;
  const progress = score != null ? score / 100 : 0;
  const dashOffset = circumference - progress * circumference;

  // Color based on score
  let strokeColor = "stroke-slate-600";
  let textColor = "text-slate-500";
  let bgRingColor = "stroke-slate-800";

  if (score != null) {
    if (score >= 80) {
      strokeColor = "stroke-emerald-500";
      textColor = "text-emerald-400";
      bgRingColor = "stroke-emerald-500/10";
    } else if (score >= 60) {
      strokeColor = "stroke-yellow-500";
      textColor = "text-yellow-400";
      bgRingColor = "stroke-yellow-500/10";
    } else if (score >= 40) {
      strokeColor = "stroke-orange-500";
      textColor = "text-orange-400";
      bgRingColor = "stroke-orange-500/10";
    } else {
      strokeColor = "stroke-red-500";
      textColor = "text-red-400";
      bgRingColor = "stroke-red-500/10";
    }
  }

  return (
    <Link
      href="/dashboard/operations"
      className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6 flex flex-col items-center justify-center hover:border-slate-600/50 transition-colors group"
    >
      <p className="text-xs text-slate-500 uppercase tracking-wider font-medium mb-4">
        Health Score
      </p>

      <div className="relative w-[180px] h-[180px]">
        <svg
          className="w-full h-full -rotate-90"
          viewBox="0 0 180 180"
        >
          {/* Background ring */}
          <circle
            cx="90"
            cy="90"
            r={radius}
            fill="none"
            strokeWidth="10"
            className={bgRingColor}
          />
          {/* Progress ring */}
          {score != null && (
            <circle
              cx="90"
              cy="90"
              r={radius}
              fill="none"
              strokeWidth="10"
              strokeLinecap="round"
              className={strokeColor}
              strokeDasharray={circumference}
              strokeDashoffset={dashOffset}
              style={{
                transition: "stroke-dashoffset 0.8s ease-out",
              }}
            />
          )}
        </svg>

        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`text-4xl font-bold ${textColor}`}>
            {score != null ? score : "\u2014"}
          </span>
          {score != null && (
            <span className="text-xs text-slate-500 mt-1">/ 100</span>
          )}
        </div>
      </div>

      {/* Trend indicator */}
      {score != null ? (
        <div className="flex items-center gap-1.5 mt-4">
          {trend === "up" && (
            <>
              <TrendingUp size={14} className="text-emerald-400" />
              <span className="text-xs text-emerald-400">Improving</span>
            </>
          )}
          {trend === "down" && (
            <>
              <TrendingDown size={14} className="text-red-400" />
              <span className="text-xs text-red-400">Worsening</span>
            </>
          )}
          {trend === "flat" && (
            <>
              <Minus size={14} className="text-slate-500" />
              <span className="text-xs text-slate-500">Stable</span>
            </>
          )}
        </div>
      ) : (
        <p className="text-xs text-slate-500 mt-4 text-center">
          Run your first analysis to see your health score
        </p>
      )}
    </Link>
  );
}

// ─── Top Priority Card ──────────────────────────────────────

function TopPriorityCard({ finding }: { finding: Finding | null }) {
  if (!finding) {
    return (
      <div className="bg-emerald-500/5 border border-emerald-500/20 rounded-xl p-6 flex items-center gap-4 h-full">
        <div className="w-12 h-12 rounded-xl bg-emerald-500/10 flex items-center justify-center shrink-0">
          <CheckCircle size={24} className="text-emerald-400" />
        </div>
        <div>
          <p className="text-lg font-bold text-white">No Active Findings</p>
          <p className="text-sm text-slate-400 mt-1">
            Your inventory looks healthy
          </p>
        </div>
      </div>
    );
  }

  const sc = severityColor(finding.severity);

  return (
    <div
      className={`${sc.bg} border ${sc.border} rounded-xl p-6 h-full flex flex-col justify-between`}
    >
      <div>
        {/* Severity badge + title */}
        <div className="flex items-center gap-3 mb-3">
          <span
            className={`px-2 py-0.5 text-[10px] font-bold uppercase rounded ${sc.bg} ${sc.text} border ${sc.border}`}
          >
            {finding.severity}
          </span>
          <span className="text-xs text-slate-500">Top Priority</span>
        </div>

        <h3 className="text-lg font-bold text-white mb-1">
          {finding.title ||
            finding.type.replace(/([A-Z])/g, " $1").trim()}
        </h3>

        <p className="text-sm text-slate-400 line-clamp-2 mb-3">
          {finding.description}
        </p>

        {/* Recommended action tip */}
        {finding.recommended_action && (
          <div className="bg-slate-900/40 rounded-lg px-3 py-2 mb-4">
            <p className="text-[10px] text-slate-500 uppercase font-medium mb-0.5">
              Recommended Action
            </p>
            <p className="text-xs text-slate-300 line-clamp-2">
              {finding.recommended_action}
            </p>
          </div>
        )}
      </div>

      <div className="flex items-center justify-between mt-2">
        <span className="text-2xl font-bold text-white">
          {formatDollar(finding.dollar_impact)}
        </span>
        <Link
          href="/dashboard/operations"
          className="flex items-center gap-1.5 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium rounded-lg transition-colors"
        >
          Take Action
          <ArrowRight size={14} />
        </Link>
      </div>
    </div>
  );
}

// ─── Trend Snapshot ─────────────────────────────────────────

function TrendSnapshot({ analyses }: { analyses: AnalysisListItem[] }) {
  const chartData = analyses
    .slice()
    .reverse()
    .map((a) => ({
      date: a.created_at
        ? new Date(a.created_at).toLocaleDateString("en-US", {
            month: "short",
            day: "numeric",
          })
        : "?",
      impact: a.total_impact_estimate_high ?? 0,
    }));

  if (analyses.length < 2) {
    return (
      <Link
        href="/dashboard/history"
        className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6 flex flex-col items-center justify-center text-center hover:border-slate-600/50 transition-colors"
      >
        <BarChart3 size={32} className="text-slate-600 mb-3" />
        <p className="text-sm font-medium text-slate-400">
          Run 2+ analyses to see trends
        </p>
        <p className="text-xs text-slate-600 mt-1">
          Each analysis adds a data point to your trend chart
        </p>
      </Link>
    );
  }

  return (
    <Link
      href="/dashboard/history"
      className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6 hover:border-slate-600/50 transition-colors block"
    >
      <div className="flex items-center justify-between mb-4">
        <p className="text-xs text-slate-500 uppercase tracking-wider font-medium">
          Impact Trend
        </p>
        <span className="text-xs text-slate-600">
          {analyses.length} analyses
        </span>
      </div>
      <ResponsiveContainer width="100%" height={160}>
        <BarChart data={chartData}>
          <XAxis
            dataKey="date"
            tick={{ fill: "#64748b", fontSize: 10 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1e293b",
              border: "1px solid #334155",
              borderRadius: "8px",
              color: "#e2e8f0",
              fontSize: "12px",
            }}
            labelStyle={{ color: "#94a3b8" }}
            formatter={(value: unknown) => [formatDollar(Number(value)), "Impact"]}
          />
          <Bar
            dataKey="impact"
            fill="#10b981"
            radius={[4, 4, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </Link>
  );
}

// ─── Quick Stat Tile ────────────────────────────────────────

function QuickStatTile({
  label,
  value,
  icon,
  href,
  valueColor,
}: {
  label: string;
  value: string;
  icon: React.ReactNode;
  href: string;
  valueColor?: string;
}) {
  return (
    <Link
      href={href}
      className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4 hover:border-slate-600/50 transition-colors group"
    >
      <div className="text-slate-500 mb-2 group-hover:text-slate-400 transition-colors">
        {icon}
      </div>
      <p
        className={`text-2xl font-bold ${valueColor ?? "text-white"}`}
      >
        {value}
      </p>
      <p className="text-xs text-slate-500 mt-1">{label}</p>
    </Link>
  );
}
