"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  History,
  Clock,
  FileSpreadsheet,
  TrendingUp,
  TrendingDown,
  ArrowRight,
  AlertTriangle,
  CheckCircle,
  Minus,
  Trash2,
  Pencil,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  GitCompare,
  ArrowUpRight,
  ArrowDownRight,
  X,
} from "lucide-react";
import {
  listAnalyses,
  compareAnalyses,
  deleteAnalysis,
  renameAnalysis,
  type AnalysisListItem,
  type AnalysisListResponse,
  type CompareResponse,
  type LeakTrend,
} from "@/lib/sentinel-api";

// ─── Helpers ─────────────────────────────────────────────────

function formatDollar(n: number): string {
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `$${(n / 1_000).toFixed(1)}K`;
  return `$${Math.round(n)}`;
}

function formatDate(dateStr: string | null): string {
  if (!dateStr) return "—";
  try {
    return new Date(dateStr).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  } catch {
    return dateStr;
  }
}

function timeAgo(dateStr: string | null): string {
  if (!dateStr) return "";
  try {
    const d = new Date(dateStr);
    const diff = Date.now() - d.getTime();
    const minutes = Math.floor(diff / 60000);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    if (days < 7) return `${days}d ago`;
    return formatDate(dateStr);
  } catch {
    return "";
  }
}

function leakKeyToTitle(key: string): string {
  return key
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

const TREND_COLORS: Record<string, string> = {
  new: "text-red-400 bg-red-500/10 border-red-500/30",
  resolved: "text-emerald-400 bg-emerald-500/10 border-emerald-500/30",
  worsening: "text-orange-400 bg-orange-500/10 border-orange-500/30",
  improving: "text-blue-400 bg-blue-500/10 border-blue-500/30",
  stable: "text-slate-400 bg-slate-500/10 border-slate-500/30",
};

const TREND_ICONS: Record<string, React.ReactNode> = {
  new: <AlertTriangle size={14} />,
  resolved: <CheckCircle size={14} />,
  worsening: <ArrowUpRight size={14} />,
  improving: <ArrowDownRight size={14} />,
  stable: <Minus size={14} />,
};

// ─── Component ───────────────────────────────────────────────

export default function HistoryPage() {
  const [analyses, setAnalyses] = useState<AnalysisListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Comparison state
  const [compareMode, setCompareMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [comparison, setComparison] = useState<CompareResponse | null>(null);
  const [comparing, setComparing] = useState(false);

  // Rename state
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");

  const loadAnalyses = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await listAnalyses(50);
      setAnalyses(res.analyses);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadAnalyses();
  }, [loadAnalyses]);

  const handleDelete = async (id: string) => {
    try {
      await deleteAnalysis(id);
      setAnalyses((prev) => prev.filter((a) => a.id !== id));
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const handleRename = async (id: string) => {
    if (!renameValue.trim()) return;
    try {
      await renameAnalysis(id, renameValue.trim());
      setAnalyses((prev) =>
        prev.map((a) => (a.id === id ? { ...a, analysis_label: renameValue.trim() } : a))
      );
      setRenamingId(null);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const toggleSelection = (id: string) => {
    setSelectedIds((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id);
      if (prev.length >= 2) return [prev[1], id]; // Replace oldest selection
      return [...prev, id];
    });
  };

  const handleCompare = async () => {
    if (selectedIds.length !== 2) return;
    setComparing(true);
    setComparison(null);
    try {
      const res = await compareAnalyses(selectedIds[1], selectedIds[0]);
      setComparison(res);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setComparing(false);
    }
  };

  const totalIssues = (a: AnalysisListItem) =>
    Object.values(a.detection_counts || {}).reduce((s, c) => s + c, 0);

  return (
    <div className="min-h-screen bg-slate-950 text-white p-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-violet-500/10 rounded-lg">
            <History className="text-violet-400" size={24} />
          </div>
          <div>
            <h1 className="text-2xl font-bold">Analysis History</h1>
            <p className="text-slate-400 text-sm">
              Track your analyses over time &middot; Compare reports for trend detection
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => {
              setCompareMode(!compareMode);
              setSelectedIds([]);
              setComparison(null);
            }}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 ${
              compareMode
                ? "bg-violet-500/20 text-violet-300 border border-violet-500/30"
                : "bg-slate-800 text-slate-300 border border-slate-700 hover:border-slate-600"
            }`}
          >
            <GitCompare size={16} />
            {compareMode ? "Exit Compare" : "Compare Reports"}
          </button>
          <button
            onClick={loadAnalyses}
            disabled={loading}
            className="p-2 rounded-lg bg-slate-800 border border-slate-700 hover:border-slate-600 transition-colors"
          >
            <RefreshCw size={16} className={loading ? "animate-spin text-violet-400" : "text-slate-400"} />
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 mb-6 flex items-center gap-3">
          <AlertTriangle className="text-red-400" size={18} />
          <span className="text-red-300 text-sm">{error}</span>
          <button onClick={() => setError(null)} className="ml-auto text-red-400 hover:text-red-300">
            <X size={16} />
          </button>
        </div>
      )}

      {/* Compare bar */}
      {compareMode && (
        <div className="bg-violet-500/5 border border-violet-500/20 rounded-xl p-4 mb-6 flex items-center justify-between">
          <div className="text-sm text-violet-300">
            {selectedIds.length === 0 && "Select 2 analyses to compare"}
            {selectedIds.length === 1 && "Select 1 more analysis to compare"}
            {selectedIds.length === 2 && "Ready to compare — click Compare"}
          </div>
          <button
            onClick={handleCompare}
            disabled={selectedIds.length !== 2 || comparing}
            className="px-4 py-2 bg-violet-600 hover:bg-violet-500 disabled:bg-slate-700 disabled:text-slate-500 rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
          >
            {comparing ? (
              <RefreshCw size={14} className="animate-spin" />
            ) : (
              <GitCompare size={14} />
            )}
            Compare
          </button>
        </div>
      )}

      {/* Comparison Results */}
      {comparison && <ComparisonPanel comparison={comparison} onClose={() => setComparison(null)} />}

      {/* Loading */}
      {loading && (
        <div className="flex items-center justify-center py-20">
          <RefreshCw size={24} className="animate-spin text-violet-400" />
        </div>
      )}

      {/* Empty state */}
      {!loading && analyses.length === 0 && (
        <div className="text-center py-20">
          <FileSpreadsheet size={48} className="text-slate-600 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-slate-400 mb-2">No Analyses Yet</h2>
          <p className="text-slate-500 text-sm">
            Upload a CSV and run an analysis to start tracking your inventory health over time.
          </p>
        </div>
      )}

      {/* Analysis list */}
      {!loading && analyses.length > 0 && (
        <div className="space-y-3">
          {analyses.map((analysis) => {
            const issues = totalIssues(analysis);
            const selected = selectedIds.includes(analysis.id);
            const selectionIndex = selectedIds.indexOf(analysis.id);

            return (
              <div
                key={analysis.id}
                className={`bg-slate-900/80 border rounded-xl p-5 transition-all ${
                  compareMode && selected
                    ? "border-violet-500/50 bg-violet-500/5"
                    : "border-slate-800 hover:border-slate-700"
                } ${compareMode ? "cursor-pointer" : ""}`}
                onClick={compareMode ? () => toggleSelection(analysis.id) : undefined}
              >
                <div className="flex items-start justify-between gap-4">
                  {/* Left: label + metadata */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      {compareMode && (
                        <div
                          className={`w-6 h-6 rounded-full border-2 flex items-center justify-center text-xs font-bold ${
                            selected
                              ? "border-violet-400 bg-violet-500/20 text-violet-300"
                              : "border-slate-600 text-slate-600"
                          }`}
                        >
                          {selected ? selectionIndex + 1 : ""}
                        </div>
                      )}

                      {renamingId === analysis.id ? (
                        <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
                          <input
                            type="text"
                            value={renameValue}
                            onChange={(e) => setRenameValue(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === "Enter") handleRename(analysis.id);
                              if (e.key === "Escape") setRenamingId(null);
                            }}
                            className="bg-slate-800 border border-slate-600 rounded px-2 py-1 text-sm text-white w-64"
                            autoFocus
                          />
                          <button
                            onClick={() => handleRename(analysis.id)}
                            className="text-emerald-400 hover:text-emerald-300"
                          >
                            <CheckCircle size={16} />
                          </button>
                          <button
                            onClick={() => setRenamingId(null)}
                            className="text-slate-400 hover:text-slate-300"
                          >
                            <X size={16} />
                          </button>
                        </div>
                      ) : (
                        <h3 className="text-white font-medium truncate">
                          {analysis.analysis_label || analysis.original_filename || "Untitled Analysis"}
                        </h3>
                      )}
                    </div>

                    <div className="flex flex-wrap items-center gap-3 text-xs text-slate-500">
                      <span className="flex items-center gap-1">
                        <Clock size={12} />
                        {timeAgo(analysis.created_at)}
                      </span>
                      <span>{analysis.file_row_count.toLocaleString()} rows</span>
                      {analysis.processing_time_seconds && (
                        <span>{analysis.processing_time_seconds.toFixed(1)}s</span>
                      )}
                    </div>
                  </div>

                  {/* Right: stats + actions */}
                  <div className="flex items-center gap-4">
                    {/* Issue count badge */}
                    <div
                      className={`px-3 py-1 rounded-full text-xs font-medium border ${
                        issues === 0
                          ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                          : issues < 10
                          ? "bg-yellow-500/10 text-yellow-400 border-yellow-500/30"
                          : "bg-red-500/10 text-red-400 border-red-500/30"
                      }`}
                    >
                      {issues === 0 ? "All Clear" : `${issues} issues`}
                    </div>

                    {/* Impact */}
                    {analysis.total_impact_estimate_high > 0 && (
                      <div className="text-right">
                        <div className="text-sm font-semibold text-white">
                          {formatDollar(analysis.total_impact_estimate_high)}
                        </div>
                        <div className="text-[10px] text-slate-500 uppercase tracking-wider">
                          Impact
                        </div>
                      </div>
                    )}

                    {/* Actions (not in compare mode) */}
                    {!compareMode && (
                      <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
                        <button
                          onClick={() => {
                            setRenamingId(analysis.id);
                            setRenameValue(analysis.analysis_label || "");
                          }}
                          className="p-1.5 rounded hover:bg-slate-800 text-slate-500 hover:text-slate-300 transition-colors"
                          title="Rename"
                        >
                          <Pencil size={14} />
                        </button>
                        <button
                          onClick={() => handleDelete(analysis.id)}
                          className="p-1.5 rounded hover:bg-red-500/10 text-slate-500 hover:text-red-400 transition-colors"
                          title="Delete"
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    )}
                  </div>
                </div>

                {/* Leak type badges */}
                {Object.keys(analysis.detection_counts || {}).length > 0 && (
                  <div className="flex flex-wrap gap-2 mt-3">
                    {Object.entries(analysis.detection_counts)
                      .filter(([, count]) => count > 0)
                      .sort(([, a], [, b]) => b - a)
                      .slice(0, 6)
                      .map(([key, count]) => (
                        <span
                          key={key}
                          className="px-2 py-0.5 bg-slate-800 border border-slate-700 rounded text-[11px] text-slate-400"
                        >
                          {leakKeyToTitle(key)}: {count}
                        </span>
                      ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ─── Comparison Panel ─────────────────────────────────────────

function ComparisonPanel({
  comparison,
  onClose,
}: {
  comparison: CompareResponse;
  onClose: () => void;
}) {
  const { summary, leak_trends, metadata } = comparison;

  const trendColor =
    summary.overall_trend === "improving"
      ? "text-emerald-400"
      : summary.overall_trend === "worsening"
      ? "text-red-400"
      : "text-slate-400";

  const trendIcon =
    summary.overall_trend === "improving" ? (
      <TrendingDown className="text-emerald-400" size={20} />
    ) : summary.overall_trend === "worsening" ? (
      <TrendingUp className="text-red-400" size={20} />
    ) : (
      <Minus className="text-slate-400" size={20} />
    );

  return (
    <div className="bg-slate-900/80 border border-violet-500/20 rounded-xl p-6 mb-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-violet-500/10 rounded-lg">
            <GitCompare className="text-violet-400" size={20} />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white">Cross-Report Comparison</h2>
            <p className="text-xs text-slate-500">
              {metadata.previous_label || "Previous"} → {metadata.current_label || "Current"}
            </p>
          </div>
        </div>
        <button onClick={onClose} className="text-slate-500 hover:text-slate-300">
          <X size={18} />
        </button>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-5">
        {/* Overall Trend */}
        <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-center">
          <div className="flex items-center justify-center gap-2 mb-1">
            {trendIcon}
            <span className={`text-sm font-semibold capitalize ${trendColor}`}>
              {summary.overall_trend}
            </span>
          </div>
          <div className="text-[10px] text-slate-500 uppercase tracking-wider">Overall Trend</div>
        </div>

        {/* Issues Delta */}
        <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-center">
          <div
            className={`text-lg font-bold ${
              summary.issues_delta > 0
                ? "text-red-400"
                : summary.issues_delta < 0
                ? "text-emerald-400"
                : "text-slate-400"
            }`}
          >
            {summary.issues_delta > 0 ? "+" : ""}
            {summary.issues_delta}
          </div>
          <div className="text-[10px] text-slate-500 uppercase tracking-wider">
            Issues ({summary.previous_total_issues} → {summary.current_total_issues})
          </div>
        </div>

        {/* Impact Delta */}
        <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-center">
          <div
            className={`text-lg font-bold ${
              summary.impact_delta_high > 0
                ? "text-red-400"
                : summary.impact_delta_high < 0
                ? "text-emerald-400"
                : "text-slate-400"
            }`}
          >
            {summary.impact_delta_high > 0 ? "+" : ""}
            {formatDollar(summary.impact_delta_high)}
          </div>
          <div className="text-[10px] text-slate-500 uppercase tracking-wider">Impact Change</div>
        </div>

        {/* Dataset */}
        <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-center">
          <div className="text-lg font-bold text-slate-300">
            {summary.current_rows.toLocaleString()}
          </div>
          <div className="text-[10px] text-slate-500 uppercase tracking-wider">
            Rows ({summary.previous_rows.toLocaleString()} prev)
          </div>
        </div>
      </div>

      {/* Quick counts */}
      <div className="flex flex-wrap gap-2 mb-5">
        {summary.new_leak_count > 0 && (
          <span className="px-3 py-1 bg-red-500/10 text-red-400 border border-red-500/30 rounded-full text-xs font-medium">
            {summary.new_leak_count} New
          </span>
        )}
        {summary.resolved_leak_count > 0 && (
          <span className="px-3 py-1 bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 rounded-full text-xs font-medium">
            {summary.resolved_leak_count} Resolved
          </span>
        )}
        {summary.worsening_leak_count > 0 && (
          <span className="px-3 py-1 bg-orange-500/10 text-orange-400 border border-orange-500/30 rounded-full text-xs font-medium">
            {summary.worsening_leak_count} Worsening
          </span>
        )}
        {summary.improving_leak_count > 0 && (
          <span className="px-3 py-1 bg-blue-500/10 text-blue-400 border border-blue-500/30 rounded-full text-xs font-medium">
            {summary.improving_leak_count} Improving
          </span>
        )}
      </div>

      {/* Leak trends table */}
      {leak_trends.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-slate-500 uppercase tracking-wider border-b border-slate-800">
                <th className="pb-2 pr-4">Leak Type</th>
                <th className="pb-2 pr-4">Status</th>
                <th className="pb-2 pr-4 text-right">Previous</th>
                <th className="pb-2 pr-4 text-right">Current</th>
                <th className="pb-2 text-right">Delta</th>
              </tr>
            </thead>
            <tbody>
              {leak_trends.map((t) => (
                <tr key={t.leak_key} className="border-b border-slate-800/50">
                  <td className="py-2.5 pr-4 text-white font-medium">
                    {leakKeyToTitle(t.leak_key)}
                  </td>
                  <td className="py-2.5 pr-4">
                    <span
                      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border ${
                        TREND_COLORS[t.status] || TREND_COLORS.stable
                      }`}
                    >
                      {TREND_ICONS[t.status]}
                      <span className="capitalize">{t.status}</span>
                    </span>
                  </td>
                  <td className="py-2.5 pr-4 text-right text-slate-400 tabular-nums">
                    {t.previous_count}
                  </td>
                  <td className="py-2.5 pr-4 text-right text-white tabular-nums">
                    {t.current_count}
                  </td>
                  <td
                    className={`py-2.5 text-right tabular-nums font-medium ${
                      t.count_delta > 0
                        ? "text-red-400"
                        : t.count_delta < 0
                        ? "text-emerald-400"
                        : "text-slate-500"
                    }`}
                  >
                    {t.count_delta > 0 ? "+" : ""}
                    {t.count_delta}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
