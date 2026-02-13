"use client";

import React, { useState, useEffect, useCallback } from "react";
import { useSearchParams } from "next/navigation";
import {
  Phone,
  AlertCircle,
  MessageSquare,
  HelpCircle,
  BookOpen,
  DollarSign,
  Package,
  ChevronRight,
} from "lucide-react";
import {
  fetchVendorCallPrep,
  fetchDigest,
  type VendorCallResponse,
  type Issue,
} from "@/lib/sentinel-api";
import { ApiErrorBanner } from "@/components/dashboard/ApiErrorBanner";

function formatDollar(n: number): string {
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `$${(n / 1_000).toFixed(1)}K`;
  return `$${n.toFixed(0)}`;
}

export default function VendorPrepPage() {
  const searchParams = useSearchParams();
  const issueParam = searchParams.get("issue");

  const [issues, setIssues] = useState<Issue[]>([]);
  const [selectedIssueId, setSelectedIssueId] = useState<string>(issueParam || "");
  const [callPrep, setCallPrep] = useState<VendorCallResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingIssues, setLoadingIssues] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load issues for selector
  useEffect(() => {
    (async () => {
      try {
        const digest = await fetchDigest(undefined, 20);
        setIssues(digest.digest.issues);
      } catch {
        // Silently fail
      } finally {
        setLoadingIssues(false);
      }
    })();
  }, []);

  // Load call prep
  const loadCallPrep = useCallback(async () => {
    if (!selectedIssueId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetchVendorCallPrep(selectedIssueId);
      setCallPrep(res);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, [selectedIssueId]);

  useEffect(() => {
    loadCallPrep();
  }, [loadCallPrep]);

  // Auto-select from URL param
  useEffect(() => {
    if (issueParam) setSelectedIssueId(issueParam);
  }, [issueParam]);

  return (
    <div className="p-6 lg:p-8 max-w-4xl">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white flex items-center gap-3">
          <Phone size={24} className="text-blue-400" />
          Vendor Call Prep
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          AI-generated talking points and questions for vendor calls
        </p>
      </div>

      {/* Issue selector */}
      <div className="mb-6">
        <label className="block text-sm text-slate-400 mb-2">Select an issue</label>
        <select
          value={selectedIssueId}
          onChange={(e) => setSelectedIssueId(e.target.value)}
          disabled={loadingIssues}
          className="w-full bg-slate-800 border border-slate-700 text-slate-300 rounded-lg px-4 py-3 text-sm focus:outline-none focus:border-emerald-500"
        >
          <option value="">Choose an issue...</option>
          {issues.map((issue) => (
            <option key={issue.id} value={issue.id}>
              {issue.issue_type.replace(/([A-Z])/g, " $1").trim()} — {issue.store_id?.trim()} — {formatDollar(issue.dollar_impact)}
            </option>
          ))}
        </select>
      </div>

      {/* Error */}
      <ApiErrorBanner error={error} onRetry={loadCallPrep} />

      {/* Loading */}
      {loading && (
        <div className="flex items-center justify-center py-16">
          <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
        </div>
      )}

      {/* Call prep content */}
      {callPrep && !loading && (
        <div className="space-y-6">
          {/* Vendor header */}
          <div className="bg-blue-500/10 border border-blue-500/20 rounded-xl p-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-blue-400 uppercase tracking-wider font-medium">Vendor</p>
                <p className="text-xl font-bold text-white mt-1">{callPrep.call_prep.vendor_name}</p>
                <p className="text-sm text-slate-400 mt-1">{callPrep.call_prep.store_id}</p>
              </div>
              <div className="text-right">
                <p className="text-xs text-slate-500">Total Impact</p>
                <p className="text-2xl font-bold text-white">
                  {formatDollar(callPrep.call_prep.total_dollar_impact)}
                </p>
              </div>
            </div>
          </div>

          {/* Issue summary */}
          <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
            <div className="flex items-center gap-2 mb-3">
              <BookOpen size={16} className="text-slate-400" />
              <h3 className="text-sm font-medium text-slate-300 uppercase tracking-wider">Issue Summary</h3>
            </div>
            <p className="text-sm text-slate-300 leading-relaxed">{callPrep.call_prep.issue_summary}</p>
          </div>

          {/* Talking points */}
          <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
            <div className="flex items-center gap-2 mb-3">
              <MessageSquare size={16} className="text-emerald-400" />
              <h3 className="text-sm font-medium text-slate-300 uppercase tracking-wider">Talking Points</h3>
            </div>
            <ul className="space-y-3">
              {callPrep.call_prep.talking_points.map((point, i) => (
                <li key={i} className="flex items-start gap-3">
                  <span className="shrink-0 w-5 h-5 bg-emerald-500/10 border border-emerald-500/20 rounded-full flex items-center justify-center text-[10px] text-emerald-400 font-bold mt-0.5">
                    {i + 1}
                  </span>
                  <p className="text-sm text-slate-300">{point}</p>
                </li>
              ))}
            </ul>
          </div>

          {/* Questions to ask */}
          <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
            <div className="flex items-center gap-2 mb-3">
              <HelpCircle size={16} className="text-violet-400" />
              <h3 className="text-sm font-medium text-slate-300 uppercase tracking-wider">Questions to Ask</h3>
            </div>
            <ul className="space-y-3">
              {callPrep.call_prep.questions_to_ask.map((q, i) => (
                <li key={i} className="flex items-start gap-3">
                  <ChevronRight size={14} className="text-violet-400 shrink-0 mt-1" />
                  <p className="text-sm text-slate-300">{q}</p>
                </li>
              ))}
            </ul>
          </div>

          {/* Affected SKUs */}
          {callPrep.call_prep.affected_skus.length > 0 && (
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
              <div className="flex items-center gap-2 mb-3">
                <Package size={16} className="text-slate-400" />
                <h3 className="text-sm font-medium text-slate-300 uppercase tracking-wider">
                  Affected SKUs ({callPrep.call_prep.affected_skus.length})
                </h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-slate-500 border-b border-slate-700/50">
                      <th className="text-left py-2 pr-4">SKU</th>
                      <th className="text-right py-2 pr-4">Qty</th>
                      <th className="text-right py-2 pr-4">Cost</th>
                      <th className="text-right py-2">Retail</th>
                    </tr>
                  </thead>
                  <tbody>
                    {callPrep.call_prep.affected_skus.slice(0, 10).map((sku) => (
                      <tr key={sku.sku_id} className="text-slate-300 border-b border-slate-800/50">
                        <td className="py-1.5 pr-4 font-mono">{sku.sku_id}</td>
                        <td className="text-right py-1.5 pr-4">{sku.qty_on_hand.toFixed(0)}</td>
                        <td className="text-right py-1.5 pr-4">${sku.unit_cost.toFixed(2)}</td>
                        <td className="text-right py-1.5">${sku.retail_price.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Historical context */}
          {callPrep.call_prep.historical_context && (
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
              <div className="flex items-center gap-2 mb-3">
                <DollarSign size={16} className="text-amber-400" />
                <h3 className="text-sm font-medium text-slate-300 uppercase tracking-wider">Vendor History</h3>
              </div>
              <p className="text-sm text-slate-400">{callPrep.call_prep.historical_context}</p>
            </div>
          )}
        </div>
      )}

      {/* Empty state */}
      {!selectedIssueId && !loading && (
        <div className="text-center py-16 text-slate-500">
          <Phone size={32} className="mx-auto mb-3 opacity-50" />
          <p className="font-medium">Select an Issue</p>
          <p className="text-sm mt-1">Choose an issue above to generate vendor call prep materials.</p>
        </div>
      )}
    </div>
  );
}
