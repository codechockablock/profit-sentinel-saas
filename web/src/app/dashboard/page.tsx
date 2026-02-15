"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  Eye,
  RefreshCw,
  Check,
  Clock,
  ChevronDown,
  ChevronRight,
  AlertTriangle,
  TrendingDown,
  TrendingUp,
  Minus,
  ArrowRight,
  Building2,
  DollarSign,
  Package,
  Zap,
  Bot,
} from "lucide-react";
import Link from "next/link";
import { ApiErrorBanner } from "@/components/dashboard/ApiErrorBanner";
import {
  fetchEagleEye,
  fetchBriefing,
  refreshBriefing,
  approveAction,
  deferAction,
  fetchActions,
  createOrg,
  type EagleEyeResponse,
  type AgentBriefing,
  type StoreSummary,
  type RegionSummary,
  type ActionItem,
} from "@/lib/eagle-eye-api";

// ─── Helpers ──────────────────────────────────────────────────

function formatDollar(amount: number): string {
  if (amount >= 1_000_000) return `$${(amount / 1_000_000).toFixed(1)}M`;
  if (amount >= 1_000) return `$${(amount / 1_000).toFixed(1)}K`;
  return `$${amount.toFixed(0)}`;
}

function formatTrend(trend: number): string {
  if (trend === 0) return "flat";
  return `${trend > 0 ? "+" : ""}${(trend * 100).toFixed(0)}%`;
}

function statusColor(status: string): { dot: string; border: string; bg: string } {
  switch (status) {
    case "healthy":
      return { dot: "bg-emerald-400", border: "border-emerald-500/30", bg: "bg-emerald-500/5" };
    case "attention":
      return { dot: "bg-yellow-400", border: "border-yellow-500/30", bg: "bg-yellow-500/5" };
    case "critical":
      return { dot: "bg-red-400", border: "border-red-500/30", bg: "bg-red-500/5" };
    default:
      return { dot: "bg-slate-400", border: "border-slate-700/50", bg: "bg-slate-800/50" };
  }
}

function actionTypeIcon(type: string): string {
  switch (type) {
    case "transfer": return "🔄";
    case "clearance": return "🏷️";
    case "reorder": return "📦";
    case "price_adjustment": return "💰";
    case "vendor_contact": return "📞";
    case "threshold_change": return "⚙️";
    default: return "📋";
  }
}

// ─── Main Page ────────────────────────────────────────────────

export default function EagleEyePage() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<EagleEyeResponse | null>(null);
  const [briefing, setBriefing] = useState<AgentBriefing | null>(null);
  const [briefingLoading, setBriefingLoading] = useState(false);
  const [pendingActions, setPendingActions] = useState<ActionItem[]>([]);
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  // Setup state
  const [showSetup, setShowSetup] = useState(false);
  const [orgName, setOrgName] = useState("");
  const [creating, setCreating] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [eagleRes, briefingRes, actionsRes] = await Promise.allSettled([
        fetchEagleEye(),
        fetchBriefing(),
        fetchActions({ status: "pending", limit: 10 }),
      ]);

      if (eagleRes.status === "fulfilled") {
        setData(eagleRes.value);
        if (!eagleRes.value.org) {
          setShowSetup(true);
        }
      } else {
        setError((eagleRes.reason as Error).message);
      }

      if (briefingRes.status === "fulfilled") {
        setBriefing(briefingRes.value);
      }

      if (actionsRes.status === "fulfilled") {
        setPendingActions(actionsRes.value.actions);
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const handleRefreshBriefing = async () => {
    setBriefingLoading(true);
    try {
      const result = await refreshBriefing();
      setBriefing(result);
      // Refresh actions since briefing may have created new ones
      const actionsRes = await fetchActions({ status: "pending", limit: 10 });
      setPendingActions(actionsRes.actions);
    } catch (err) {
      // Silently fail — briefing is non-critical
    } finally {
      setBriefingLoading(false);
    }
  };

  const handleApprove = async (actionId: string) => {
    setActionLoading(actionId);
    try {
      await approveAction(actionId);
      setPendingActions((prev) => prev.filter((a) => a.id !== actionId));
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
      setPendingActions((prev) => prev.filter((a) => a.id !== actionId));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setActionLoading(null);
    }
  };

  const handleCreateOrg = async () => {
    if (!orgName.trim()) return;
    setCreating(true);
    try {
      await createOrg(orgName.trim());
      setShowSetup(false);
      load();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setCreating(false);
    }
  };

  // Setup screen
  if (showSetup && !loading) {
    return (
      <div className="p-6 lg:p-8 max-w-2xl mx-auto">
        <div className="p-8 bg-slate-800/50 border border-slate-700/50 rounded-xl text-center">
          <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-emerald-500/10 flex items-center justify-center">
            <Building2 size={28} className="text-emerald-400" />
          </div>
          <h2 className="text-xl font-bold text-white mb-2">
            Welcome to Profit Sentinel
          </h2>
          <p className="text-slate-400 mb-6">
            Set up your organization to get started with the executive dashboard.
          </p>
          {error && (
            <div className="mb-4 px-4 py-3 bg-red-500/10 border border-red-500/30 rounded-lg text-sm text-red-400 text-left">
              {error}
            </div>
          )}
          <div className="max-w-sm mx-auto space-y-4">
            <input
              type="text"
              value={orgName}
              onChange={(e) => setOrgName(e.target.value)}
              placeholder="Your business name"
              className="w-full px-4 py-3 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500"
              onKeyDown={(e) => e.key === "Enter" && handleCreateOrg()}
            />
            <button
              onClick={handleCreateOrg}
              disabled={creating || !orgName.trim()}
              className="w-full px-6 py-3 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white font-semibold rounded-lg transition-colors"
            >
              {creating ? "Creating..." : "Create Organization"}
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 lg:p-8 max-w-6xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-emerald-500/10 flex items-center justify-center">
            <Eye size={20} className="text-emerald-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">
              {data?.org?.name || "Eagle\u2019s Eye"}
            </h1>
            <p className="text-sm text-slate-400">
              {data?.org
                ? `${data.org.total_stores} stores \u2022 ${formatDollar(data.org.total_exposure)} total exposure`
                : "Executive network view"}
            </p>
          </div>
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

      {error && <ApiErrorBanner error={error} onRetry={load} />}

      {/* Loading */}
      {loading && !data && (
        <div className="flex items-center justify-center py-24">
          <div className="text-center">
            <div className="w-10 h-10 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-slate-400 text-sm">Loading executive view...</p>
          </div>
        </div>
      )}

      {data?.org && (
        <>
          {/* Agent Briefing */}
          <BriefingSection
            briefing={briefing}
            loading={briefingLoading}
            onRefresh={handleRefreshBriefing}
          />

          {/* Action Queue */}
          {pendingActions.length > 0 && (
            <ActionQueueSection
              actions={pendingActions}
              loadingId={actionLoading}
              onApprove={handleApprove}
              onDefer={handleDefer}
            />
          )}

          {/* Network Summary Cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <SummaryCard
              label="Total Exposure"
              value={formatDollar(data.org.total_exposure)}
              trend={data.org.exposure_trend}
              icon={<DollarSign size={16} />}
            />
            <SummaryCard
              label="Total Stores"
              value={String(data.org.total_stores)}
              icon={<Package size={16} />}
            />
            <SummaryCard
              label="Pending Actions"
              value={String(data.org.total_pending_actions)}
              icon={<Clock size={16} />}
            />
            <SummaryCard
              label="Completed (30d)"
              value={String(data.org.total_completed_actions_30d)}
              icon={<Check size={16} />}
            />
          </div>

          {/* Regions & Stores */}
          <div className="space-y-4 mb-6">
            {data.regions.map((region) => (
              <RegionSection key={region.id} region={region} />
            ))}

            {data.unassigned_stores.length > 0 && (
              <RegionSection
                region={{
                  id: "unassigned",
                  name: "Unassigned Stores",
                  store_count: data.unassigned_stores.length,
                  total_exposure: data.unassigned_stores.reduce(
                    (sum, s) => sum + s.total_impact,
                    0
                  ),
                  exposure_trend: 0,
                  pending_actions: data.unassigned_stores.reduce(
                    (sum, s) => sum + s.pending_actions,
                    0
                  ),
                  stores: data.unassigned_stores,
                }}
              />
            )}
          </div>

          {/* Network Alerts */}
          {data.network_alerts.length > 0 && (
            <div className="mb-6">
              <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">
                Network Alerts
              </h2>
              <div className="space-y-2">
                {data.network_alerts.map((alert, i) => (
                  <div
                    key={i}
                    className="flex items-center gap-3 px-4 py-3 bg-red-500/5 border border-red-500/20 rounded-lg"
                  >
                    <AlertTriangle size={16} className="text-red-400 shrink-0" />
                    <span className="text-sm text-white flex-1">
                      {alert.description}
                    </span>
                    <span className="text-sm font-medium text-red-400">
                      {formatDollar(alert.total_impact)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Empty state — no stores with data */}
          {data.org.total_stores === 0 && (
            <div className="p-8 bg-slate-800/50 border border-slate-700/50 rounded-xl text-center">
              <Zap size={32} className="text-emerald-400 mx-auto mb-3" />
              <h3 className="text-lg font-bold text-white mb-2">
                Add Your First Store
              </h3>
              <p className="text-slate-400 mb-4">
                Create stores and upload inventory data to see your executive view.
              </p>
              <Link
                href="/dashboard/stores"
                className="inline-flex items-center gap-2 px-6 py-3 bg-emerald-600 hover:bg-emerald-500 text-white font-medium rounded-lg transition-colors"
              >
                Manage Stores
                <ArrowRight size={16} />
              </Link>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ─── Briefing Section ──────────────────────────────────────────

function BriefingSection({
  briefing,
  loading,
  onRefresh,
}: {
  briefing: AgentBriefing | null;
  loading: boolean;
  onRefresh: () => void;
}) {
  return (
    <div className="mb-6 bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Bot size={18} className="text-emerald-400" />
          <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
            Agent Briefing
          </h2>
        </div>
        <button
          onClick={onRefresh}
          disabled={loading}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-slate-400 hover:text-white border border-slate-700 hover:border-slate-600 rounded-lg transition-colors disabled:opacity-50"
        >
          <RefreshCw size={12} className={loading ? "animate-spin" : ""} />
          Refresh
        </button>
      </div>

      {loading && !briefing ? (
        <div className="flex items-center gap-3 py-4">
          <div className="w-5 h-5 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm text-slate-400">Generating briefing...</span>
        </div>
      ) : briefing?.briefing ? (
        <div className="prose prose-sm prose-invert max-w-none">
          <p className="text-sm text-slate-300 leading-relaxed whitespace-pre-line">
            {briefing.briefing}
          </p>
          {briefing.generated_at && (
            <p className="text-xs text-slate-600 mt-3">
              Generated{" "}
              {new Date(briefing.generated_at).toLocaleString("en-US", {
                hour: "numeric",
                minute: "2-digit",
                hour12: true,
              })}
            </p>
          )}
        </div>
      ) : (
        <p className="text-sm text-slate-500 py-2">
          No briefing available. Click refresh to generate one.
        </p>
      )}
    </div>
  );
}

// ─── Action Queue Section ────────────────────────────────────

function ActionQueueSection({
  actions,
  loadingId,
  onApprove,
  onDefer,
}: {
  actions: ActionItem[];
  loadingId: string | null;
  onApprove: (id: string) => void;
  onDefer: (id: string) => void;
}) {
  return (
    <div className="mb-6">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">
          Action Queue ({actions.length} pending)
        </h2>
        <Link
          href="/dashboard/tasks"
          className="text-xs text-emerald-400 hover:text-emerald-300 transition-colors"
        >
          View All
        </Link>
      </div>
      <div className="space-y-2">
        {actions.slice(0, 5).map((action) => (
          <div
            key={action.id}
            className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4 flex items-start gap-3"
          >
            <span className="text-lg mt-0.5">
              {actionTypeIcon(action.action_type)}
            </span>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-white truncate">
                {action.description}
              </p>
              {action.reasoning && (
                <p className="text-xs text-slate-400 mt-1 line-clamp-1">
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
              </div>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <button
                onClick={() => onApprove(action.id)}
                disabled={loadingId === action.id}
                className="px-3 py-1.5 text-xs font-medium bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white rounded-lg transition-colors"
              >
                Approve
              </button>
              <button
                onClick={() => onDefer(action.id)}
                disabled={loadingId === action.id}
                className="px-3 py-1.5 text-xs font-medium text-slate-400 hover:text-white border border-slate-700 hover:border-slate-600 rounded-lg transition-colors disabled:opacity-50"
              >
                Not now
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Summary Card ────────────────────────────────────────────

function SummaryCard({
  label,
  value,
  trend,
  icon,
}: {
  label: string;
  value: string;
  trend?: number;
  icon: React.ReactNode;
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

// ─── Region Section (collapsible) ──────────────────────────

function RegionSection({ region }: { region: RegionSummary }) {
  const [expanded, setExpanded] = useState(true);

  return (
    <div className="bg-slate-800/30 border border-slate-700/30 rounded-xl overflow-hidden">
      {/* Region header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-5 py-3 hover:bg-slate-700/20 transition-colors"
      >
        <div className="flex items-center gap-3">
          {expanded ? (
            <ChevronDown size={16} className="text-slate-500" />
          ) : (
            <ChevronRight size={16} className="text-slate-500" />
          )}
          <h3 className="text-sm font-semibold text-white">
            {region.name}
          </h3>
          <span className="text-xs text-slate-500">
            {region.store_count} store{region.store_count !== 1 ? "s" : ""}
          </span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-sm font-medium text-white">
            {formatDollar(region.total_exposure)}
          </span>
          {region.exposure_trend !== 0 && (
            <span
              className={`flex items-center gap-0.5 text-xs ${
                region.exposure_trend < 0 ? "text-emerald-400" : "text-red-400"
              }`}
            >
              {region.exposure_trend < 0 ? (
                <TrendingDown size={10} />
              ) : (
                <TrendingUp size={10} />
              )}
              {formatTrend(region.exposure_trend)}
            </span>
          )}
          {region.pending_actions > 0 && (
            <span className="text-xs text-yellow-400">
              {region.pending_actions} pending
            </span>
          )}
        </div>
      </button>

      {/* Store cards grid */}
      {expanded && (
        <div className="px-5 pb-4 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          {region.stores.map((store) => (
            <StoreCard key={store.id} store={store} />
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Store Card ──────────────────────────────────────────────

function StoreCard({ store }: { store: StoreSummary }) {
  const sc = statusColor(store.status);

  return (
    <Link
      href={`/dashboard/stores/${store.id}`}
      className={`${sc.bg} border ${sc.border} rounded-lg p-3 hover:brightness-110 transition-all block`}
    >
      <div className="flex items-center gap-2 mb-2">
        <div className={`w-2 h-2 rounded-full ${sc.dot}`} />
        <p className="text-xs font-medium text-white truncate">{store.name}</p>
      </div>
      <p className="text-lg font-bold text-white">
        {formatDollar(store.total_impact)}
      </p>
      {store.top_issue && (
        <p className="text-[10px] text-slate-400 mt-1 truncate">
          {store.top_issue}
        </p>
      )}
      {store.pending_actions > 0 && (
        <p className="text-[10px] text-yellow-400 mt-0.5">
          {store.pending_actions} pending
        </p>
      )}
    </Link>
  );
}
