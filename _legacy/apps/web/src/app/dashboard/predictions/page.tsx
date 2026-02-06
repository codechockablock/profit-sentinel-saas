"use client";

import React, { useState, useEffect } from "react";
import {
  AlertTriangle,
  TrendingDown,
  TrendingUp,
  Package,
  Clock,
  DollarSign,
  Loader2,
  Zap,
  ShieldAlert,
  Eye,
} from "lucide-react";
import { fetchPredictions, type PredictiveReportResponse, type InventoryPrediction } from "@/lib/sentinel-api";

const SEVERITY_CONFIG = {
  critical: { color: "text-red-400 bg-red-500/10 border-red-500/30", icon: ShieldAlert },
  warning: { color: "text-amber-400 bg-amber-500/10 border-amber-500/30", icon: AlertTriangle },
  watch: { color: "text-blue-400 bg-blue-500/10 border-blue-500/30", icon: Eye },
};

const TYPE_CONFIG = {
  stockout: { label: "Stockout Risk", icon: Package, accent: "text-red-400" },
  overstock: { label: "Overstock", icon: TrendingDown, accent: "text-amber-400" },
  demand_surge: { label: "Demand Surge", icon: TrendingUp, accent: "text-emerald-400" },
  velocity_drop: { label: "Velocity Drop", icon: TrendingDown, accent: "text-orange-400" },
};

function PredictionCard({ prediction }: { prediction: InventoryPrediction }) {
  const severity = SEVERITY_CONFIG[prediction.severity] || SEVERITY_CONFIG.watch;
  const ptype = TYPE_CONFIG[prediction.prediction_type] || TYPE_CONFIG.stockout;
  const SevIcon = severity.icon;
  const TypeIcon = ptype.icon;

  return (
    <div className={`bg-white/5 rounded-xl border border-slate-700 p-5 hover:bg-white/[0.07] transition`}>
      <div className="flex items-start gap-4">
        <div className={`w-10 h-10 rounded-lg border flex items-center justify-center ${severity.color}`}>
          <SevIcon size={18} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-white font-semibold">{prediction.sku_id}</span>
            <span className={`text-xs px-2 py-0.5 rounded-full ${severity.color} border`}>
              {prediction.severity}
            </span>
          </div>
          <div className="flex items-center gap-2 text-xs text-slate-400 mb-2">
            <TypeIcon size={12} className={ptype.accent} />
            <span>{ptype.label}</span>
            <span className="text-slate-600">&middot;</span>
            <span>{prediction.store_id}</span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
            <div className="bg-slate-800/50 rounded-lg p-2">
              <div className="text-[10px] text-slate-500">Days Until</div>
              <div className="text-sm font-bold text-white flex items-center gap-1">
                <Clock size={12} className="text-slate-400" />
                {prediction.days_until_event.toFixed(1)}
              </div>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-2">
              <div className="text-[10px] text-slate-500">Current Stock</div>
              <div className="text-sm font-bold text-white">{prediction.current_stock}</div>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-2">
              <div className="text-[10px] text-slate-500">Velocity</div>
              <div className="text-sm font-bold text-white">{prediction.current_velocity.toFixed(1)}/day</div>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-2">
              <div className="text-[10px] text-slate-500">Revenue at Risk</div>
              <div className="text-sm font-bold text-red-400">
                ${prediction.estimated_lost_revenue.toLocaleString()}
              </div>
            </div>
          </div>

          <div className="mt-3 text-xs text-slate-400 flex items-center gap-1">
            <Zap size={10} className="text-emerald-400" />
            {prediction.recommended_action}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function PredictionsPage() {
  const [data, setData] = useState<PredictiveReportResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchPredictions()
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
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
          <p className="font-medium">Failed to load predictions</p>
          <p className="text-sm mt-1">{error}</p>
        </div>
      </div>
    );
  }

  if (!data) return null;

  return (
    <div className="p-6 md:p-8 max-w-5xl">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <AlertTriangle className="w-7 h-7 text-amber-400" />
          <h1 className="text-2xl font-bold text-white">Predictive Inventory Alerts</h1>
        </div>
        <p className="text-sm text-slate-400">
          Velocity-based stockout and overstock forecasting with demand change detection.
        </p>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-white/5 rounded-xl border border-slate-700 p-4">
          <div className="text-xs text-slate-400 mb-1">Total Predictions</div>
          <div className="text-2xl font-bold text-white">{data.total_predictions}</div>
        </div>
        <div className="bg-white/5 rounded-xl border border-slate-700 p-4">
          <div className="text-xs text-slate-400 mb-1">Critical Alerts</div>
          <div className="text-2xl font-bold text-red-400">{data.critical_alerts}</div>
        </div>
        <div className="bg-white/5 rounded-xl border border-slate-700 p-4">
          <div className="text-xs text-slate-400 mb-1">Revenue at Risk</div>
          <div className="text-2xl font-bold text-amber-400">${data.total_revenue_at_risk.toLocaleString()}</div>
        </div>
        <div className="bg-white/5 rounded-xl border border-slate-700 p-4">
          <div className="text-xs text-slate-400 mb-1">Carrying Cost</div>
          <div className="text-2xl font-bold text-orange-400">${data.total_carrying_cost_at_risk.toLocaleString()}</div>
        </div>
      </div>

      {/* Top recommendation */}
      <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-4 mb-8 flex items-start gap-3">
        <Zap className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
        <div className="text-sm text-slate-300">{data.top_recommendation}</div>
      </div>

      {/* Stockout predictions */}
      {data.stockout_predictions.length > 0 && (
        <section className="mb-8">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Package size={18} className="text-red-400" />
            Stockout Predictions ({data.stockout_predictions.length})
          </h2>
          <div className="space-y-3">
            {data.stockout_predictions.map((pred, i) => (
              <PredictionCard key={`stockout-${i}`} prediction={pred} />
            ))}
          </div>
        </section>
      )}

      {/* Overstock predictions */}
      {data.overstock_predictions.length > 0 && (
        <section className="mb-8">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <TrendingDown size={18} className="text-amber-400" />
            Overstock Predictions ({data.overstock_predictions.length})
          </h2>
          <div className="space-y-3">
            {data.overstock_predictions.map((pred, i) => (
              <PredictionCard key={`overstock-${i}`} prediction={pred} />
            ))}
          </div>
        </section>
      )}

      {/* Velocity alerts */}
      {data.velocity_alerts.length > 0 && (
        <section className="mb-8">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Zap size={18} className="text-emerald-400" />
            Velocity Change Alerts ({data.velocity_alerts.length})
          </h2>
          <div className="space-y-3">
            {data.velocity_alerts.map((pred, i) => (
              <PredictionCard key={`velocity-${i}`} prediction={pred} />
            ))}
          </div>
        </section>
      )}

      {/* Empty state */}
      {data.total_predictions === 0 && (
        <div className="text-center py-16">
          <DollarSign className="w-12 h-12 text-emerald-400 mx-auto mb-4" />
          <p className="text-white font-medium">Inventory levels are healthy</p>
          <p className="text-sm text-slate-400 mt-1">No stockout or overstock predictions at this time.</p>
        </div>
      )}
    </div>
  );
}
