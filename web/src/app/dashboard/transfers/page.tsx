"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  ArrowRight,
  DollarSign,
  Loader2,
  RefreshCw,
  Package,
  Truck,
  Clock,
  Store,
  ShieldCheck,
} from "lucide-react";
import {
  fetchTransfers,
  fetchStores,
  type TransfersResponse,
  type TransferRecommendation,
} from "@/lib/sentinel-api";
import { ApiErrorBanner } from "@/components/dashboard/ApiErrorBanner";

// ─── Helpers ─────────────────────────────────────────────────

function formatDollar(amount: number): string {
  if (amount >= 1_000_000) return `$${(amount / 1_000_000).toFixed(1)}M`;
  if (amount >= 1_000) return `$${(amount / 1_000).toFixed(1)}K`;
  return `$${amount.toFixed(0)}`;
}

const MATCH_CONFIG: Record<string, { label: string; color: string }> = {
  exact_sku: { label: "Exact Match", color: "text-emerald-400 bg-emerald-500/10 border-emerald-500/30" },
  subcategory: { label: "Subcategory", color: "text-blue-400 bg-blue-500/10 border-blue-500/30" },
  category: { label: "Category", color: "text-violet-400 bg-violet-500/10 border-violet-500/30" },
};

// ─── Transfer Card ───────────────────────────────────────────

function TransferCard({
  rec,
  storeNames,
}: {
  rec: TransferRecommendation;
  storeNames: Record<string, string>;
}) {
  const match = MATCH_CONFIG[rec.match_level] || MATCH_CONFIG.category;
  const sourceName = storeNames[rec.source_store] || rec.source_store;
  const destName = storeNames[rec.dest_store] || rec.dest_store;

  return (
    <div className="bg-white/5 rounded-xl border border-slate-700 p-5 hover:bg-white/[0.07] transition">
      {/* Net benefit — most prominent */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="text-xs text-slate-500 mb-0.5">Net Benefit</div>
          <div className="text-2xl font-bold text-emerald-400">
            {formatDollar(rec.net_benefit)}
          </div>
        </div>
        <span className={`text-[10px] px-2 py-0.5 rounded-full border ${match.color}`}>
          {match.label}
        </span>
      </div>

      {/* Source → Destination */}
      <div className="flex items-center gap-3 mb-4">
        <div className="flex-1 min-w-0 bg-slate-800/50 rounded-lg p-3">
          <div className="text-[10px] text-slate-500 mb-1">From</div>
          <div className="text-sm font-medium text-white truncate">{sourceName}</div>
          <div className="text-xs text-slate-400 truncate mt-0.5">
            {rec.source_description}
          </div>
          <div className="text-[10px] text-slate-500 font-mono mt-1">{rec.source_sku}</div>
        </div>

        <ArrowRight size={18} className="text-slate-500 shrink-0" />

        <div className="flex-1 min-w-0 bg-slate-800/50 rounded-lg p-3">
          <div className="text-[10px] text-slate-500 mb-1">To</div>
          <div className="text-sm font-medium text-white truncate">{destName}</div>
          <div className="text-xs text-slate-400 truncate mt-0.5">
            {rec.dest_description}
          </div>
          <div className="text-[10px] text-slate-500 font-mono mt-1">{rec.dest_sku}</div>
        </div>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="bg-slate-800/50 rounded-lg p-2">
          <div className="text-[10px] text-slate-500">Units</div>
          <div className="text-sm font-bold text-white flex items-center gap-1">
            <Package size={12} className="text-slate-400" />
            {rec.units_to_transfer}
          </div>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-2">
          <div className="text-[10px] text-slate-500">Clearance Recovery</div>
          <div className="text-sm font-bold text-orange-400">
            {formatDollar(rec.clearance_recovery)}
          </div>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-2">
          <div className="text-[10px] text-slate-500">Transfer Recovery</div>
          <div className="text-sm font-bold text-emerald-400">
            {formatDollar(rec.transfer_recovery)}
          </div>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-2">
          <div className="text-[10px] text-slate-500">Sell-Through</div>
          <div className="text-sm font-bold text-white flex items-center gap-1">
            <Clock size={12} className="text-slate-400" />
            {rec.estimated_weeks_to_sell} wk
          </div>
        </div>
      </div>

      {/* Confidence & demand pattern */}
      <div className="mt-3 flex items-center gap-3 text-xs text-slate-400">
        <span className="flex items-center gap-1">
          <ShieldCheck size={10} className="text-emerald-400" />
          {(rec.match_confidence * 100).toFixed(0)}% confidence
        </span>
        <span className="text-slate-600">&middot;</span>
        <span>{rec.demand_pattern}</span>
      </div>
    </div>
  );
}

// ─── Page ────────────────────────────────────────────────────

export default function TransfersPage() {
  const [data, setData] = useState<TransfersResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [storeNames, setStoreNames] = useState<Record<string, string>>({});

  // Load store names for display
  useEffect(() => {
    fetchStores()
      .then((res) => {
        const names: Record<string, string> = {};
        for (const s of res.stores) {
          names[s.id] = s.name;
        }
        setStoreNames(names);
      })
      .catch(() => {});
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchTransfers({ max_results: 50 });
      setData(res);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  if (loading && !data) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="w-8 h-8 text-emerald-400 animate-spin" />
      </div>
    );
  }

  const recs = data?.recommendations ?? [];
  const totalBenefit = recs.reduce((s, r) => s + r.net_benefit, 0);
  const totalClearance = recs.reduce((s, r) => s + r.clearance_recovery, 0);
  const totalTransfer = recs.reduce((s, r) => s + r.transfer_recovery, 0);

  return (
    <div className="p-6 md:p-8 max-w-5xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <Truck className="w-7 h-7 text-emerald-400" />
            <h1 className="text-2xl font-bold text-white">Transfer Recommendations</h1>
          </div>
          <p className="text-sm text-slate-400">
            Move dead stock to stores where it sells. Sorted by net benefit.
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

      {/* Content */}
      {data && !error && (
        <>
          {/* Empty state — single store or not ready */}
          {recs.length === 0 ? (
            <div className="text-center py-16">
              <Store className="w-12 h-12 text-slate-500 mx-auto mb-4" />
              <p className="text-white font-medium">
                {data.engine2_status === "not_initialized" || data.engine2_status === "error"
                  ? "Transfer matching is warming up"
                  : "No transfer opportunities right now"}
              </p>
              <p className="text-sm text-slate-400 mt-2 max-w-md mx-auto">
                {data.message ||
                  "Transfer recommendations appear when you have multiple stores in your network and dead stock that could sell better elsewhere."}
              </p>
            </div>
          ) : (
            <>
              {/* Summary */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                <div className="bg-white/5 rounded-xl border border-slate-700 p-4">
                  <div className="text-xs text-slate-400 mb-1">Recommendations</div>
                  <div className="text-2xl font-bold text-white">{recs.length}</div>
                </div>
                <div className="bg-white/5 rounded-xl border border-slate-700 p-4">
                  <div className="text-xs text-slate-400 mb-1">Total Net Benefit</div>
                  <div className="text-2xl font-bold text-emerald-400">{formatDollar(totalBenefit)}</div>
                </div>
                <div className="bg-white/5 rounded-xl border border-slate-700 p-4">
                  <div className="text-xs text-slate-400 mb-1">Clearance Recovery</div>
                  <div className="text-2xl font-bold text-orange-400">{formatDollar(totalClearance)}</div>
                </div>
                <div className="bg-white/5 rounded-xl border border-slate-700 p-4">
                  <div className="text-xs text-slate-400 mb-1">Transfer Recovery</div>
                  <div className="text-2xl font-bold text-emerald-400">{formatDollar(totalTransfer)}</div>
                </div>
              </div>

              {/* Benefit comparison banner */}
              <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-4 mb-8 flex items-start gap-3">
                <DollarSign className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
                <div className="text-sm text-slate-300">
                  Transferring instead of marking down saves{" "}
                  <span className="text-emerald-400 font-semibold">{formatDollar(totalBenefit)}</span>{" "}
                  compared to clearance pricing across {recs.length} items.
                </div>
              </div>

              {/* Recommendation list */}
              <div className="space-y-3">
                {recs.map((rec, i) => (
                  <TransferCard key={`transfer-${i}`} rec={rec} storeNames={storeNames} />
                ))}
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
