"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  BarChart3,
  AlertTriangle,
  CheckCircle,
  ChevronDown,
  ChevronRight,
  Shield,
  TrendingDown,
  Truck,
  DollarSign,
  FileCheck,
  Loader2,
} from "lucide-react";
import { fetchVendorScores, type VendorScoresResponse, type VendorScorecard } from "@/lib/sentinel-api";
import { ApiErrorBanner } from "@/components/dashboard/ApiErrorBanner";

const DIMENSION_ICONS: Record<string, React.ElementType> = {
  quality: Shield,
  delivery: Truck,
  pricing: DollarSign,
  compliance: FileCheck,
};

const GRADE_COLORS: Record<string, string> = {
  A: "text-emerald-400 bg-emerald-500/10 border-emerald-500/30",
  B: "text-blue-400 bg-blue-500/10 border-blue-500/30",
  C: "text-amber-400 bg-amber-500/10 border-amber-500/30",
  D: "text-orange-400 bg-orange-500/10 border-orange-500/30",
  F: "text-red-400 bg-red-500/10 border-red-500/30",
};

const RISK_COLORS: Record<string, string> = {
  low: "text-emerald-400 bg-emerald-500/10",
  medium: "text-amber-400 bg-amber-500/10",
  high: "text-orange-400 bg-orange-500/10",
  critical: "text-red-400 bg-red-500/10",
};

function ScoreBar({ score }: { score: number }) {
  const color =
    score >= 80
      ? "bg-emerald-500"
      : score >= 60
        ? "bg-amber-500"
        : score >= 40
          ? "bg-orange-500"
          : "bg-red-500";
  return (
    <div className="flex-1 bg-slate-700/50 rounded-full h-2">
      <div className={`${color} h-2 rounded-full transition-all`} style={{ width: `${Math.min(100, score)}%` }} />
    </div>
  );
}

function VendorCard({ card }: { card: VendorScorecard }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="bg-white/5 rounded-xl border border-slate-700 overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-5 py-4 flex items-center gap-4 hover:bg-white/[0.03] transition"
      >
        <div className={`w-12 h-12 rounded-lg border flex items-center justify-center text-lg font-bold ${GRADE_COLORS[card.overall_grade] || GRADE_COLORS.C}`}>
          {card.overall_grade}
        </div>
        <div className="flex-1 text-left">
          <div className="text-white font-semibold">{card.vendor_name}</div>
          <div className="text-xs text-slate-400">
            Score: {card.overall_score.toFixed(0)} &middot; {card.issues_analyzed} issues &middot; ${card.total_dollar_impact.toLocaleString()} impact
          </div>
        </div>
        <span className={`text-xs px-2 py-1 rounded-full ${RISK_COLORS[card.risk_level] || RISK_COLORS.medium}`}>
          {card.risk_level}
        </span>
        {expanded ? <ChevronDown size={16} className="text-slate-500" /> : <ChevronRight size={16} className="text-slate-500" />}
      </button>

      {expanded && (
        <div className="px-5 pb-5 space-y-4 border-t border-slate-700/50">
          {/* Dimensions */}
          <div className="grid grid-cols-2 gap-3 pt-4">
            {card.dimensions.map((dim) => {
              const Icon = DIMENSION_ICONS[dim.dimension] || Shield;
              return (
                <div key={dim.dimension} className="bg-slate-800/50 rounded-lg p-3">
                  <div className="flex items-center gap-2 mb-2">
                    <Icon size={14} className="text-slate-400" />
                    <span className="text-xs text-slate-400 capitalize">{dim.dimension}</span>
                    <span className={`ml-auto text-xs font-bold px-1.5 py-0.5 rounded border ${GRADE_COLORS[dim.grade] || GRADE_COLORS.C}`}>
                      {dim.grade}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <ScoreBar score={dim.score} />
                    <span className="text-xs text-slate-300 w-8 text-right">{dim.score.toFixed(0)}</span>
                  </div>
                  <div className="text-[10px] text-slate-500 mt-1">{dim.details}</div>
                </div>
              );
            })}
          </div>

          {/* Recommendations */}
          {card.recommendations.length > 0 && (
            <div className="space-y-2">
              <div className="text-xs text-slate-400 font-medium">Recommendations</div>
              {card.recommendations.map((rec, i) => (
                <div key={i} className="flex items-start gap-2 text-xs text-slate-300">
                  <TrendingDown size={12} className="text-amber-400 mt-0.5 shrink-0" />
                  {rec}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function VendorScoresPage() {
  const [data, setData] = useState<VendorScoresResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadScores = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchVendorScores();
      setData(res);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadScores();
  }, [loadScores]);

  if (loading && !data) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="w-8 h-8 text-emerald-400 animate-spin" />
      </div>
    );
  }

  return (
    <div className="p-6 md:p-8 max-w-5xl">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <BarChart3 className="w-7 h-7 text-emerald-400" />
          <h1 className="text-2xl font-bold text-white">Vendor Performance Scores</h1>
        </div>
        <p className="text-sm text-slate-400">
          Weighted scoring across quality, delivery, pricing, and compliance dimensions.
        </p>
      </div>

      {/* Error */}
      <ApiErrorBanner error={error} onRetry={loadScores} />

      {/* Content */}
      {data && !error && (
        <>
          {/* Summary cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            <div className="bg-white/5 rounded-xl border border-slate-700 p-4">
              <div className="text-xs text-slate-400 mb-1">Vendors Scored</div>
              <div className="text-2xl font-bold text-white">{data.total_vendors_scored}</div>
            </div>
            <div className="bg-white/5 rounded-xl border border-slate-700 p-4">
              <div className="text-xs text-slate-400 mb-1">Average Score</div>
              <div className="text-2xl font-bold text-white">{data.average_score.toFixed(0)}</div>
            </div>
            <div className="bg-white/5 rounded-xl border border-slate-700 p-4">
              <div className="text-xs text-slate-400 mb-1">High Risk</div>
              <div className="text-2xl font-bold text-red-400">{data.high_risk_vendors}</div>
            </div>
            <div className="bg-white/5 rounded-xl border border-slate-700 p-4">
              <div className="text-xs text-slate-400 mb-1">Quality Cost</div>
              <div className="text-2xl font-bold text-amber-400">${data.total_quality_cost.toLocaleString()}</div>
            </div>
          </div>

          {/* Top recommendation */}
          {data.top_recommendation && (
            <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-4 mb-8 flex items-start gap-3">
              {data.high_risk_vendors > 0 ? (
                <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
              ) : (
                <CheckCircle className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
              )}
              <div className="text-sm text-slate-300">{data.top_recommendation}</div>
            </div>
          )}

          {/* Vendor cards */}
          <div className="space-y-3">
            {data.scorecards.map((card) => (
              <VendorCard key={card.vendor_id} card={card} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
