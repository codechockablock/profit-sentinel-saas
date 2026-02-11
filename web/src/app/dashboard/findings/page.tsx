"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  Search,
  DollarSign,
  Loader2,
  ChevronDown,
  ChevronRight,
  Zap,
  Eye,
  EyeOff,
  AlertCircle,
  RefreshCw,
  CheckCircle2,
  BarChart3,
} from "lucide-react";
import {
  fetchFindings,
  acknowledgeFinding,
  restoreFinding,
  type FindingsResponse,
  type Finding,
} from "@/lib/sentinel-api";

// ─── Tier severity colors ────────────────────────────────────
// green → yellow → orange → red → darkred (escalating severity)

const TIER_COLORS: Record<string, { bg: string; text: string; border: string; dot: string }> = {
  critical: { bg: "bg-red-900/20", text: "text-red-300", border: "border-red-800/40", dot: "bg-red-400" },
  high: { bg: "bg-red-500/10", text: "text-red-400", border: "border-red-500/30", dot: "bg-red-400" },
  medium: { bg: "bg-orange-500/10", text: "text-orange-400", border: "border-orange-500/30", dot: "bg-orange-400" },
  low: { bg: "bg-yellow-500/10", text: "text-yellow-400", border: "border-yellow-500/30", dot: "bg-yellow-400" },
  info: { bg: "bg-emerald-500/10", text: "text-emerald-400", border: "border-emerald-500/30", dot: "bg-emerald-400" },
};

function tierColor(severity: string) {
  return TIER_COLORS[severity] || TIER_COLORS.low;
}

function formatDollar(amount: number): string {
  if (amount >= 1_000_000) return `$${(amount / 1_000_000).toFixed(1)}M`;
  if (amount >= 1_000) return `$${(amount / 1_000).toFixed(1)}K`;
  return `$${amount.toFixed(0)}`;
}

function issueTypeLabel(t: string): string {
  return t.replace(/([A-Z])/g, " $1").trim();
}

// ─── Finding Card ────────────────────────────────────────────

function FindingCard({
  finding,
  onAcknowledge,
  onRestore,
}: {
  finding: Finding;
  onAcknowledge: (id: string) => void;
  onRestore: (id: string) => void;
}) {
  const tier = tierColor(finding.severity);
  const hasEnrichment = (finding.engine2_observations ?? 0) > 0;

  return (
    <div className={`${tier.bg} border ${tier.border} rounded-xl p-5 hover:brightness-110 transition`}>
      <div className="flex items-start gap-4">
        {/* Dollar impact — always most prominent */}
        <div className="shrink-0 text-right min-w-[80px]">
          <div className="text-xl font-bold text-white">
            {formatDollar(finding.dollar_impact)}
          </div>
          <div className={`text-[10px] font-medium uppercase mt-0.5 ${tier.text}`}>
            {finding.severity}
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-sm font-semibold text-white truncate">
              {finding.title || issueTypeLabel(finding.type)}
            </span>
            {hasEnrichment && (
              <span className="flex items-center gap-1 text-[10px] text-slate-500">
                <BarChart3 size={10} />
                {finding.engine2_observations} obs
              </span>
            )}
          </div>

          {finding.description && (
            <p className="text-xs text-slate-400 mb-2 line-clamp-2">{finding.description}</p>
          )}

          <div className="flex items-center gap-3 flex-wrap">
            {finding.department && (
              <span className="text-[10px] text-slate-500 bg-slate-800/50 px-2 py-0.5 rounded-full">
                {finding.department}
              </span>
            )}
            {finding.sku && (
              <span className="text-[10px] text-slate-500 font-mono">
                {finding.sku}
              </span>
            )}
            {finding.prediction && (
              <span className="flex items-center gap-1 text-[10px] text-amber-400">
                <Zap size={10} />
                Prediction available
              </span>
            )}
          </div>

          {finding.recommended_action && (
            <div className="mt-2 text-xs text-slate-400 flex items-start gap-1.5">
              <Zap size={10} className="text-emerald-400 shrink-0 mt-0.5" />
              <span>{finding.recommended_action}</span>
            </div>
          )}
        </div>

        {/* Acknowledge / Restore button */}
        <button
          onClick={() =>
            finding.acknowledged ? onRestore(finding.id) : onAcknowledge(finding.id)
          }
          className={`shrink-0 p-2 rounded-lg transition-colors ${
            finding.acknowledged
              ? "text-slate-500 hover:text-white hover:bg-slate-700/50"
              : "text-slate-500 hover:text-emerald-400 hover:bg-emerald-500/10"
          }`}
          title={finding.acknowledged ? "Restore to active" : "Mark as seen"}
        >
          {finding.acknowledged ? <EyeOff size={16} /> : <Eye size={16} />}
        </button>
      </div>
    </div>
  );
}

// ─── Page ────────────────────────────────────────────────────

export default function FindingsPage() {
  const [data, setData] = useState<FindingsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showAcknowledged, setShowAcknowledged] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [active, acked] = await Promise.all([
        fetchFindings({ status: "active", sort_by: "dollar_impact", page_size: 100 }),
        fetchFindings({ status: "acknowledged", sort_by: "dollar_impact", page_size: 100 }),
      ]);
      // Merge into a single response
      setData({
        findings: [...active.findings, ...acked.findings],
        pagination: active.pagination,
        engine2_status: active.engine2_status,
      });
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const handleAcknowledge = async (id: string) => {
    try {
      await acknowledgeFinding(id);
      // Optimistically update local state
      setData((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          findings: prev.findings.map((f) =>
            f.id === id ? { ...f, acknowledged: true } : f
          ),
        };
      });
    } catch {
      // Silently fail — next refresh will fix state
    }
  };

  const handleRestore = async (id: string) => {
    try {
      await restoreFinding(id);
      setData((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          findings: prev.findings.map((f) =>
            f.id === id ? { ...f, acknowledged: false } : f
          ),
        };
      });
    } catch {
      // Silently fail
    }
  };

  if (loading && !data) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="w-8 h-8 text-emerald-400 animate-spin" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8">
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-6 text-red-400">
          <p className="font-medium">Failed to load findings</p>
          <p className="text-sm mt-1">{error}</p>
        </div>
      </div>
    );
  }

  if (!data) return null;

  const activeFindings = data.findings.filter((f) => !f.acknowledged);
  const acknowledgedFindings = data.findings.filter((f) => f.acknowledged);
  const totalImpact = activeFindings.reduce((s, f) => s + f.dollar_impact, 0);

  return (
    <div className="p-6 md:p-8 max-w-5xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <Search className="w-7 h-7 text-amber-400" />
            <h1 className="text-2xl font-bold text-white">Findings</h1>
          </div>
          <p className="text-sm text-slate-400">
            Ranked by dollar impact. Acknowledge findings to move them out of the active view.
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

      {/* Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-white/5 rounded-xl border border-slate-700 p-4">
          <div className="text-xs text-slate-400 mb-1">Active Findings</div>
          <div className="text-2xl font-bold text-white">{activeFindings.length}</div>
        </div>
        <div className="bg-white/5 rounded-xl border border-slate-700 p-4">
          <div className="text-xs text-slate-400 mb-1">Total Impact</div>
          <div className="text-2xl font-bold text-amber-400">{formatDollar(totalImpact)}</div>
        </div>
        <div className="bg-white/5 rounded-xl border border-slate-700 p-4">
          <div className="text-xs text-slate-400 mb-1">Acknowledged</div>
          <div className="text-2xl font-bold text-slate-400">{acknowledgedFindings.length}</div>
        </div>
        <div className="bg-white/5 rounded-xl border border-slate-700 p-4">
          <div className="text-xs text-slate-400 mb-1">Smart Analysis</div>
          <div className="flex items-center gap-2">
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                data.engine2_status === "active"
                  ? "bg-emerald-400"
                  : data.engine2_status === "warming_up"
                  ? "bg-yellow-400"
                  : "bg-slate-600"
              }`}
            />
            <span className="text-sm text-slate-300 capitalize">
              {data.engine2_status === "not_initialized" ? "Not ready" : data.engine2_status.replace("_", " ")}
            </span>
          </div>
        </div>
      </div>

      {/* Active findings */}
      {activeFindings.length > 0 ? (
        <div className="space-y-3 mb-8">
          {activeFindings.map((finding) => (
            <FindingCard
              key={finding.id}
              finding={finding}
              onAcknowledge={handleAcknowledge}
              onRestore={handleRestore}
            />
          ))}
        </div>
      ) : (
        <div className="text-center py-16 mb-8">
          <CheckCircle2 className="w-12 h-12 text-emerald-400 mx-auto mb-4" />
          <p className="text-white font-medium">No active findings</p>
          <p className="text-sm text-slate-400 mt-1">
            All findings have been reviewed or no issues detected.
          </p>
        </div>
      )}

      {/* Acknowledged findings (collapsed section) */}
      {acknowledgedFindings.length > 0 && (
        <div className="border-t border-slate-700/50 pt-6">
          <button
            onClick={() => setShowAcknowledged(!showAcknowledged)}
            className="flex items-center gap-2 text-sm text-slate-400 hover:text-white transition-colors mb-4"
          >
            {showAcknowledged ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            <span className="font-medium">
              Acknowledged ({acknowledgedFindings.length})
            </span>
          </button>

          {showAcknowledged && (
            <div className="space-y-2">
              {acknowledgedFindings.map((finding) => (
                <div key={finding.id} className="opacity-60 hover:opacity-100 transition-opacity">
                  <FindingCard
                    finding={finding}
                    onAcknowledge={handleAcknowledge}
                    onRestore={handleRestore}
                  />
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
