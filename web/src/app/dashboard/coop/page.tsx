"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  TrendingUp,
  AlertCircle,
  DollarSign,
  BarChart3,
  ShieldCheck,
  Target,
  ArrowUpRight,
  ArrowDownRight,
  RefreshCw,
} from "lucide-react";
import {
  fetchCoopReport,
  fetchDigest,
  type CoopReportResponse,
  type CoopAlert,
  type VendorRebateStatus,
} from "@/lib/sentinel-api";
import { ApiErrorBanner } from "@/components/dashboard/ApiErrorBanner";

function formatDollar(n: number): string {
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `$${(n / 1_000).toFixed(1)}K`;
  return `$${n.toFixed(0)}`;
}

const ALERT_TYPE_COLORS: Record<string, { icon: React.ReactNode; color: string }> = {
  PatronageLeakage: { icon: <DollarSign size={16} />, color: "text-red-400 bg-red-500/10 border-red-500/20" },
  DeadStockAlert: { icon: <AlertCircle size={16} />, color: "text-amber-400 bg-amber-500/10 border-amber-500/20" },
  RebateThresholdRisk: { icon: <Target size={16} />, color: "text-orange-400 bg-orange-500/10 border-orange-500/20" },
  MixImbalance: { icon: <BarChart3 size={16} />, color: "text-violet-400 bg-violet-500/10 border-violet-500/20" },
  ConsolidationOpportunity: { icon: <TrendingUp size={16} />, color: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20" },
  GMROIWarning: { icon: <AlertCircle size={16} />, color: "text-pink-400 bg-pink-500/10 border-pink-500/20" },
};

const DEFAULT_ALERT = { icon: <AlertCircle size={16} />, color: "text-slate-400 bg-slate-500/10 border-slate-500/20" };

export default function CoopPage() {
  const [storeIds, setStoreIds] = useState<string[]>([]);
  const [selectedStore, setSelectedStore] = useState<string>("");
  const [report, setReport] = useState<CoopReportResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingStores, setLoadingStores] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load store list from digest
  useEffect(() => {
    (async () => {
      try {
        const digest = await fetchDigest(undefined, 20);
        const stores = [...new Set(digest.digest.issues.map((i) => i.store_id))];
        setStoreIds(stores);
        if (stores.length > 0 && !selectedStore) {
          setSelectedStore(stores[0]);
        }
      } catch {
        // Silently fail
      } finally {
        setLoadingStores(false);
      }
    })();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Load co-op report
  const loadReport = useCallback(async () => {
    if (!selectedStore) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetchCoopReport(selectedStore);
      setReport(res);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, [selectedStore]);

  useEffect(() => {
    loadReport();
  }, [loadReport]);

  return (
    <div className="p-6 lg:p-8 max-w-6xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-3">
            <TrendingUp size={24} className="text-emerald-400" />
            Co-op Intelligence
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Patronage, rebate tracking, and category mix analysis
          </p>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={selectedStore}
            onChange={(e) => setSelectedStore(e.target.value)}
            disabled={loadingStores}
            className="bg-slate-800 border border-slate-700 text-slate-300 text-sm rounded-lg px-3 py-2"
          >
            {storeIds.length === 0 && <option value="">No stores</option>}
            {storeIds.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
          <button
            onClick={loadReport}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-sm font-medium rounded-lg transition-colors"
          >
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
          </button>
        </div>
      </div>

      {/* Error */}
      <ApiErrorBanner error={error} onRetry={loadReport} />

      {/* Loading */}
      {loading && (
        <div className="flex items-center justify-center py-24">
          <div className="w-10 h-10 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
        </div>
      )}

      {/* Report */}
      {report && !loading && (
        <div className="space-y-6">
          {/* Summary cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-xl p-4">
              <p className="text-xs text-emerald-400/80">Total Opportunity</p>
              <p className="text-2xl font-bold text-emerald-400 mt-1">
                {formatDollar(report.total_opportunity)}
              </p>
            </div>
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
              <p className="text-xs text-slate-500">Alerts</p>
              <p className="text-2xl font-bold text-white mt-1">{report.report.alerts.length}</p>
            </div>
            {report.report.health_report && (
              <>
                <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
                  <p className="text-xs text-slate-500">GMROI</p>
                  <p className="text-2xl font-bold text-white mt-1">
                    {report.report.health_report.overall_gmroi.toFixed(2)}
                  </p>
                </div>
                <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
                  <p className="text-xs text-slate-500">Dead Stock</p>
                  <p className="text-2xl font-bold text-white mt-1">
                    {(report.report.health_report.dead_stock_pct * 100).toFixed(1)}%
                  </p>
                </div>
              </>
            )}
          </div>

          {/* Alerts */}
          {report.report.alerts.length > 0 && (
            <div>
              <h2 className="text-sm font-medium text-slate-300 uppercase tracking-wider mb-3">
                Opportunities & Alerts
              </h2>
              <div className="space-y-3">
                {report.report.alerts.map((alert, i) => (
                  <AlertCard key={i} alert={alert} />
                ))}
              </div>
            </div>
          )}

          {/* Rebate tracking */}
          {report.rebate_statuses.length > 0 && (
            <div>
              <h2 className="text-sm font-medium text-slate-300 uppercase tracking-wider mb-3">
                Vendor Rebate Tracking
              </h2>
              <div className="space-y-3">
                {report.rebate_statuses.map((rs, i) => (
                  <RebateCard key={i} rebate={rs} />
                ))}
              </div>
            </div>
          )}

          {/* Category analysis */}
          {report.report.category_analysis && (
            <div>
              <h2 className="text-sm font-medium text-slate-300 uppercase tracking-wider mb-3">
                Category Mix Analysis
              </h2>
              <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <p className="text-xs text-slate-500">Total Revenue</p>
                    <p className="text-lg font-bold text-white">
                      {formatDollar(report.report.category_analysis.total_revenue)}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-slate-500">Avg Margin</p>
                    <p className="text-lg font-bold text-white">
                      {(report.report.category_analysis.total_margin_pct * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>

                {report.report.category_analysis.top_expansion_categories.length > 0 && (
                  <div className="mb-3">
                    <p className="text-xs text-emerald-400 font-medium mb-1">Expand</p>
                    <div className="flex flex-wrap gap-2">
                      {report.report.category_analysis.top_expansion_categories.map((c) => (
                        <span key={c} className="flex items-center gap-1 px-2.5 py-1 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs rounded-lg">
                          <ArrowUpRight size={10} />
                          {c}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {report.report.category_analysis.top_contraction_categories.length > 0 && (
                  <div>
                    <p className="text-xs text-red-400 font-medium mb-1">Contract</p>
                    <div className="flex flex-wrap gap-2">
                      {report.report.category_analysis.top_contraction_categories.map((c) => (
                        <span key={c} className="flex items-center gap-1 px-2.5 py-1 bg-red-500/10 border border-red-500/20 text-red-400 text-xs rounded-lg">
                          <ArrowDownRight size={10} />
                          {c}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Sub-components ──────────────────────────────────────────

function AlertCard({ alert }: { alert: CoopAlert }) {
  const meta = ALERT_TYPE_COLORS[alert.alert_type] || DEFAULT_ALERT;
  const [expanded, setExpanded] = useState(false);

  return (
    <div className={`border rounded-xl overflow-hidden ${meta.color}`}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-3 px-5 py-3.5 text-left"
      >
        {meta.icon}
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-white truncate">{alert.title}</p>
          <p className="text-xs text-slate-500 truncate">{alert.alert_type.replace(/([A-Z])/g, " $1").trim()}</p>
        </div>
        <span className="text-sm font-bold text-white shrink-0">
          {formatDollar(alert.dollar_impact)}
        </span>
      </button>
      {expanded && (
        <div className="px-5 pb-4 border-t border-slate-700/30 space-y-2 mt-2">
          <p className="text-sm text-slate-300">{alert.detail}</p>
          <div className="bg-slate-900/50 rounded-lg p-3">
            <p className="text-xs text-emerald-400 font-medium mb-1">Recommendation</p>
            <p className="text-sm text-slate-300">{alert.recommendation}</p>
          </div>
          <p className="text-[10px] text-slate-600">
            Confidence: {(alert.confidence * 100).toFixed(0)}%
          </p>
        </div>
      )}
    </div>
  );
}

function RebateCard({ rebate }: { rebate: VendorRebateStatus }) {
  const progressPct = rebate.next_tier
    ? Math.min(100, ((rebate.ytd_purchases - (rebate.current_tier?.threshold || 0)) / (rebate.next_tier.threshold - (rebate.current_tier?.threshold || 0))) * 100)
    : 100;

  return (
    <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
      <div className="flex items-center justify-between mb-3">
        <div>
          <p className="text-sm font-medium text-white">{rebate.program.vendor}</p>
          <p className="text-xs text-slate-500">
            Current: {rebate.current_tier?.name || "None"} &middot; YTD: {formatDollar(rebate.ytd_purchases)}
          </p>
        </div>
        <div className={`flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-lg border ${
          rebate.on_track
            ? "text-emerald-400 bg-emerald-500/10 border-emerald-500/20"
            : "text-amber-400 bg-amber-500/10 border-amber-500/20"
        }`}>
          <ShieldCheck size={12} />
          {rebate.on_track ? "On Track" : "At Risk"}
        </div>
      </div>

      {/* Progress bar */}
      {rebate.next_tier && (
        <div className="mb-3">
          <div className="flex justify-between text-[10px] text-slate-500 mb-1">
            <span>{rebate.current_tier?.name || "Base"}</span>
            <span>{rebate.next_tier.name} ({formatDollar(rebate.next_tier.threshold)})</span>
          </div>
          <div className="w-full bg-slate-700/50 rounded-full h-2">
            <div
              className={`h-2 rounded-full transition-all ${rebate.on_track ? "bg-emerald-500" : "bg-amber-500"}`}
              style={{ width: `${Math.max(2, progressPct)}%` }}
            />
          </div>
          <div className="flex justify-between text-[10px] text-slate-600 mt-1">
            <span>Shortfall: {formatDollar(rebate.shortfall)}</span>
            <span>{rebate.days_remaining} days left</span>
          </div>
        </div>
      )}

      {/* Incremental value */}
      {rebate.incremental_value > 0 && (
        <p className="text-xs text-emerald-400">
          +{formatDollar(rebate.incremental_value)} if next tier reached
        </p>
      )}

      <p className="text-xs text-slate-500 mt-2">{rebate.recommendation}</p>
    </div>
  );
}
