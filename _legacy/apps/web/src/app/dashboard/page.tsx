"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  AlertCircle,
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
} from "lucide-react";
import Link from "next/link";
import {
  fetchDigest,
  type DigestResponse,
  type Issue,
  type TaskPriority,
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
  const [data, setData] = useState<DigestResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [topK, setTopK] = useState(10);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchDigest(undefined, topK);
      setData(res);
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
          <select
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="bg-slate-800 border border-slate-700 text-slate-300 text-sm rounded-lg px-3 py-2 focus:outline-none focus:border-emerald-500"
          >
            <option value={5}>Top 5</option>
            <option value={10}>Top 10</option>
            <option value={20}>Top 20</option>
          </select>
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
        <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 text-sm">
          <AlertCircle size={16} className="inline mr-2" />
          {error}
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
      {data && (
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
              value={String(data.store_ids.length)}
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

// ─── Sub-components ──────────────────────────────────────────

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
