"use client";

import React, { useState, useEffect, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import {
  ArrowLeft,
  Upload,
  DollarSign,
  Package,
  AlertTriangle,
  TrendingDown,
  TrendingUp,
  Clock,
  Check,
  X,
  Loader2,
  MapPin,
} from "lucide-react";
import { ApiErrorBanner } from "@/components/dashboard/ApiErrorBanner";
import {
  fetchOrgStore,
  fetchActions,
  approveAction,
  deferAction,
  type OrgStoreDetail,
  type ActionItem,
} from "@/lib/eagle-eye-api";

// ─── Helpers ──────────────────────────────────────────────────

function formatDollar(amount: number): string {
  if (amount >= 1_000_000) return `$${(amount / 1_000_000).toFixed(1)}M`;
  if (amount >= 1_000) return `$${(amount / 1_000).toFixed(1)}K`;
  return `$${amount.toFixed(0)}`;
}

function formatDate(iso: string | null): string {
  if (!iso) return "Never";
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function formatTrend(trend: number): string {
  if (trend === 0) return "flat";
  return `${trend > 0 ? "+" : ""}${(trend * 100).toFixed(0)}%`;
}

function statusLabel(status: string): { text: string; color: string } {
  switch (status) {
    case "healthy":
      return { text: "Healthy", color: "text-emerald-400" };
    case "attention":
      return { text: "Needs Attention", color: "text-yellow-400" };
    case "critical":
      return { text: "Critical", color: "text-red-400" };
    case "inactive":
      return { text: "Inactive", color: "text-slate-500" };
    default:
      return { text: status, color: "text-slate-400" };
  }
}

function statusDot(status: string): string {
  switch (status) {
    case "healthy":
      return "bg-emerald-400";
    case "attention":
      return "bg-yellow-400";
    case "critical":
      return "bg-red-400";
    default:
      return "bg-slate-500";
  }
}

function actionTypeIcon(type: string): string {
  switch (type) {
    case "transfer": return "\u{1F504}";
    case "clearance": return "\u{1F3F7}\uFE0F";
    case "reorder": return "\u{1F4E6}";
    case "price_adjustment": return "\u{1F4B0}";
    case "vendor_contact": return "\u{1F4DE}";
    case "threshold_change": return "\u2699\uFE0F";
    default: return "\u{1F4CB}";
  }
}

// ─── Main Page ────────────────────────────────────────────────

export default function StoreDetailPage() {
  const params = useParams();
  const storeId = params.id as string;

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [store, setStore] = useState<OrgStoreDetail | null>(null);
  const [actions, setActions] = useState<ActionItem[]>([]);
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [storeRes, actionsRes] = await Promise.allSettled([
        fetchOrgStore(storeId),
        fetchActions({ store_id: storeId, limit: 20 }),
      ]);

      if (storeRes.status === "fulfilled") {
        setStore(storeRes.value);
      } else {
        setError((storeRes.reason as Error).message);
      }

      if (actionsRes.status === "fulfilled") {
        setActions(actionsRes.value.actions);
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, [storeId]);

  useEffect(() => {
    load();
  }, [load]);

  const handleApprove = async (actionId: string) => {
    setActionLoading(actionId);
    try {
      await approveAction(actionId);
      setActions((prev) =>
        prev.map((a) =>
          a.id === actionId ? { ...a, status: "approved" } : a
        )
      );
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setActionLoading(null);
    }
  };

  const handleDefer = async (actionId: string) => {
    setActionLoading(actionId);
    try {
      await deferAction(actionId);
      setActions((prev) =>
        prev.map((a) =>
          a.id === actionId ? { ...a, status: "deferred" } : a
        )
      );
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setActionLoading(null);
    }
  };

  // Loading
  if (loading && !store) {
    return (
      <div className="p-6 lg:p-8 flex items-center justify-center py-24">
        <div className="text-center">
          <div className="w-10 h-10 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-slate-400 text-sm">Loading store details...</p>
        </div>
      </div>
    );
  }

  const sl = store ? statusLabel(store.status) : null;
  const pendingActions = actions.filter((a) => a.status === "pending");
  const recentActions = actions.filter((a) => a.status !== "pending").slice(0, 10);

  return (
    <div className="p-6 lg:p-8 max-w-4xl">
      {/* Back nav */}
      <Link
        href="/dashboard"
        className="inline-flex items-center gap-1.5 text-sm text-slate-400 hover:text-white transition-colors mb-6"
      >
        <ArrowLeft size={14} />
        Back to Eagle&apos;s Eye
      </Link>

      {error && <ApiErrorBanner error={error} onRetry={load} />}

      {store && (
        <>
          {/* Header */}
          <div className="flex items-start justify-between mb-6">
            <div>
              <div className="flex items-center gap-3 mb-1">
                <div
                  className={`w-3 h-3 rounded-full ${statusDot(store.status)}`}
                />
                <h1 className="text-2xl font-bold text-white">{store.name}</h1>
              </div>
              {store.address && (
                <div className="flex items-center gap-1.5 text-sm text-slate-400 ml-6">
                  <MapPin size={12} />
                  {store.address}
                </div>
              )}
              {sl && (
                <p className={`text-sm font-medium ml-6 mt-1 ${sl.color}`}>
                  {sl.text}
                </p>
              )}
            </div>
            <Link
              href="/analyze"
              className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium rounded-lg transition-colors"
            >
              <Upload size={14} />
              Upload Data
            </Link>
          </div>

          {/* Metric cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <MetricCard
              label="Total Exposure"
              value={formatDollar(store.total_impact)}
              icon={<DollarSign size={16} />}
              trend={store.exposure_trend}
            />
            <MetricCard
              label="Total Items"
              value={store.item_count.toLocaleString()}
              icon={<Package size={16} />}
            />
            <MetricCard
              label="Last Upload"
              value={formatDate(store.last_upload_at)}
              icon={<Clock size={16} />}
            />
            <MetricCard
              label="Pending Actions"
              value={String(pendingActions.length)}
              icon={<AlertTriangle size={16} />}
            />
          </div>

          {/* Pending Actions */}
          {pendingActions.length > 0 && (
            <div className="mb-6">
              <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">
                Pending Actions
              </h2>
              <div className="space-y-2">
                {pendingActions.map((action) => (
                  <div
                    key={action.id}
                    className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4 flex items-start gap-3"
                  >
                    <span className="text-lg mt-0.5">
                      {actionTypeIcon(action.action_type)}
                    </span>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-white">
                        {action.description}
                      </p>
                      {action.reasoning && (
                        <p className="text-xs text-slate-400 mt-1">
                          {action.reasoning}
                        </p>
                      )}
                      <div className="flex items-center gap-3 mt-2">
                        {action.financial_impact > 0 && (
                          <span className="text-xs font-medium text-emerald-400">
                            {formatDollar(action.financial_impact)}
                          </span>
                        )}
                        {action.confidence > 0 && (
                          <span className="text-xs text-slate-500">
                            {Math.round(action.confidence * 100)}% confidence
                          </span>
                        )}
                        <span className="text-xs text-slate-600">
                          {action.source === "agent" ? "AI recommended" : "Manual"}
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      <button
                        onClick={() => handleApprove(action.id)}
                        disabled={actionLoading === action.id}
                        className="px-3 py-1.5 text-xs font-medium bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white rounded-lg transition-colors"
                      >
                        {actionLoading === action.id ? (
                          <Loader2 size={12} className="animate-spin" />
                        ) : (
                          "Approve"
                        )}
                      </button>
                      <button
                        onClick={() => handleDefer(action.id)}
                        disabled={actionLoading === action.id}
                        className="px-3 py-1.5 text-xs font-medium text-slate-400 hover:text-white border border-slate-700 hover:border-slate-600 rounded-lg transition-colors disabled:opacity-50"
                      >
                        Not now
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Recent Activity */}
          {recentActions.length > 0 && (
            <div className="mb-6">
              <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">
                Recent Activity
              </h2>
              <div className="space-y-1.5">
                {recentActions.map((action) => (
                  <div
                    key={action.id}
                    className="flex items-center gap-3 px-4 py-2.5 bg-slate-800/30 border border-slate-700/30 rounded-lg"
                  >
                    <ActionStatusIcon status={action.status} />
                    <span className="text-sm text-white flex-1 truncate">
                      {action.description}
                    </span>
                    <span className="text-xs text-slate-500 shrink-0">
                      {action.status}
                    </span>
                    {action.financial_impact > 0 && (
                      <span className="text-xs font-medium text-slate-400 shrink-0">
                        {formatDollar(action.financial_impact)}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Empty state */}
          {actions.length === 0 && !store.last_upload_at && (
            <div className="p-8 bg-slate-800/50 border border-slate-700/50 rounded-xl text-center">
              <Upload size={32} className="text-emerald-400 mx-auto mb-3" />
              <h3 className="text-lg font-bold text-white mb-2">
                No Data Yet
              </h3>
              <p className="text-slate-400 mb-4">
                Upload inventory data for this store to see analysis, findings, and agent recommendations.
              </p>
              <Link
                href="/analyze"
                className="inline-flex items-center gap-2 px-6 py-3 bg-emerald-600 hover:bg-emerald-500 text-white font-medium rounded-lg transition-colors"
              >
                <Upload size={16} />
                Upload Data
              </Link>
            </div>
          )}
        </>
      )}

      {!store && !loading && (
        <div className="p-8 bg-slate-800/50 border border-slate-700/50 rounded-xl text-center">
          <AlertTriangle size={32} className="text-yellow-400 mx-auto mb-3" />
          <h3 className="text-lg font-bold text-white mb-2">Store Not Found</h3>
          <p className="text-slate-400 mb-4">
            This store may have been removed or you don&apos;t have access to it.
          </p>
          <Link
            href="/dashboard"
            className="inline-flex items-center gap-2 px-6 py-3 bg-slate-700 hover:bg-slate-600 text-white font-medium rounded-lg transition-colors"
          >
            <ArrowLeft size={16} />
            Back to Dashboard
          </Link>
        </div>
      )}
    </div>
  );
}

// ─── Metric Card ──────────────────────────────────────────────

function MetricCard({
  label,
  value,
  icon,
  trend,
}: {
  label: string;
  value: string;
  icon: React.ReactNode;
  trend?: number;
}) {
  return (
    <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
      <div className="text-slate-500 mb-2">{icon}</div>
      <p className="text-2xl font-bold text-white">{value}</p>
      <div className="flex items-center gap-2 mt-1">
        <p className="text-xs text-slate-500">{label}</p>
        {trend !== undefined && trend !== 0 && (
          <span
            className={`flex items-center gap-0.5 text-xs ${
              trend < 0 ? "text-emerald-400" : "text-red-400"
            }`}
          >
            {trend < 0 ? (
              <TrendingDown size={10} />
            ) : (
              <TrendingUp size={10} />
            )}
            {formatTrend(trend)}
          </span>
        )}
      </div>
    </div>
  );
}

// ─── Action Status Icon ──────────────────────────────────────

function ActionStatusIcon({ status }: { status: string }) {
  switch (status) {
    case "approved":
      return <Check size={14} className="text-emerald-400" />;
    case "completed":
      return <Check size={14} className="text-blue-400" />;
    case "rejected":
      return <X size={14} className="text-red-400" />;
    case "deferred":
      return <Clock size={14} className="text-yellow-400" />;
    default:
      return <Clock size={14} className="text-slate-500" />;
  }
}
