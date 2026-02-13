"use client";

import React, { useState, useRef } from "react";
import Papa from "papaparse";
import {
  Stethoscope,
  Upload,
  AlertCircle,
  CheckCircle,
  ChevronRight,
  BarChart3,
  Search,
  FileText,
} from "lucide-react";
import {
  startDiagnostic,
  fetchDiagnosticQuestion,
  answerDiagnosticQuestion,
  fetchDiagnosticReport,
  type DiagnosticStartResponse,
  type DiagnosticQuestion,
  type DiagnosticReport,
} from "@/lib/sentinel-api";
import { ApiErrorBanner } from "@/components/dashboard/ApiErrorBanner";

function formatDollar(n: number): string {
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `$${(n / 1_000).toFixed(1)}K`;
  return `$${n.toFixed(0)}`;
}

type Stage = "upload" | "questioning" | "complete";

export default function DiagnosticPage() {
  const fileRef = useRef<HTMLInputElement>(null);
  const [stage, setStage] = useState<Stage>("upload");
  const [storeName, setStoreName] = useState("My Store");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Session state
  const [sessionId, setSessionId] = useState<string>("");
  const [_sessionInfo, setSessionInfo] = useState<DiagnosticStartResponse | null>(null);
  const [currentQuestion, setCurrentQuestion] = useState<DiagnosticQuestion | null>(null);
  const [report, setReport] = useState<DiagnosticReport | null>(null);

  // Parse CSV file using Papa Parse for robust handling of quotes, delimiters, etc.
  const parseCSV = (text: string): { sku: string; description: string; stock: number; cost: number }[] => {
    const result = Papa.parse<Record<string, string>>(text, {
      header: true,
      skipEmptyLines: true,
      transformHeader: (h) => h.trim().toLowerCase(),
    });

    if (result.errors.length > 0 && result.data.length === 0) return [];

    const cols = result.meta.fields || [];
    const skuCol = cols.find((c) => c.includes("sku") || c.includes("upc") || c.includes("item"));
    const descCol = cols.find((c) => c.includes("desc") || c.includes("name") || c.includes("product"));
    const stockCol = cols.find((c) => c.includes("qty") || c.includes("stock") || c.includes("on_hand") || c.includes("quantity"));
    const costCol = cols.find((c) => c.includes("cost") || c.includes("price"));

    if (!skuCol || !stockCol) return [];

    return result.data
      .map((row) => ({
        sku: (row[skuCol] || "").trim(),
        description: descCol ? (row[descCol] || "").trim() : "",
        stock: parseFloat(row[stockCol]) || 0,
        cost: costCol ? parseFloat(row[costCol]) || 0 : 0,
      }))
      .filter((item) => item.sku);
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      const text = await file.text();
      const items = parseCSV(text);

      if (items.length === 0) {
        setError("Could not parse CSV. Ensure it has SKU and quantity columns.");
        setLoading(false);
        return;
      }

      const res = await startDiagnostic({ items, store_name: storeName });
      setSessionId(res.session_id);
      setSessionInfo(res);

      if (res.patterns_detected > 0) {
        const q = await fetchDiagnosticQuestion(res.session_id);
        setCurrentQuestion(q);
        setStage("questioning");
      } else {
        // No patterns — go straight to report
        const rep = await fetchDiagnosticReport(res.session_id);
        setReport(rep);
        setStage("complete");
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const handleAnswer = async (classification: string) => {
    setLoading(true);
    try {
      const res = await answerDiagnosticQuestion(sessionId, classification);
      if (res.is_complete || !res.next_question) {
        const rep = await fetchDiagnosticReport(sessionId);
        setReport(rep);
        setStage("complete");
      } else {
        setCurrentQuestion(res.next_question);
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const resetSession = () => {
    setStage("upload");
    setSessionId("");
    setSessionInfo(null);
    setCurrentQuestion(null);
    setReport(null);
    setError(null);
    if (fileRef.current) fileRef.current.value = "";
  };

  return (
    <div className="p-6 lg:p-8 max-w-4xl">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white flex items-center gap-3">
          <Stethoscope size={24} className="text-pink-400" />
          Shrinkage Diagnostic
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Guided walkthrough to separate process issues from real losses
        </p>
      </div>

      {/* Error */}
      <ApiErrorBanner error={error} onRetry={() => setError(null)} />

      {/* ─── Upload Stage ─── */}
      {stage === "upload" && (
        <div className="space-y-6">
          <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6">
            <div className="mb-4">
              <label className="block text-sm text-slate-400 mb-2">Store Name</label>
              <input
                type="text"
                value={storeName}
                onChange={(e) => setStoreName(e.target.value)}
                className="w-full px-4 py-2.5 bg-slate-800 border border-slate-600 rounded-lg text-white text-sm focus:outline-none focus:border-emerald-500"
              />
            </div>

            <label className="block text-sm text-slate-400 mb-2">Upload Inventory CSV</label>
            <div
              onClick={() => fileRef.current?.click()}
              className="border-2 border-dashed border-slate-600 rounded-xl p-8 text-center cursor-pointer hover:border-emerald-500/50 transition-colors"
            >
              <Upload size={32} className="mx-auto mb-3 text-slate-500" />
              <p className="text-sm text-slate-400">
                Click to upload CSV with SKU, quantity, and cost columns
              </p>
              <p className="text-xs text-slate-600 mt-1">Negative quantities will be analyzed</p>
            </div>
            <input
              ref={fileRef}
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              className="hidden"
            />
          </div>

          {loading && (
            <div className="flex items-center justify-center py-8">
              <div className="w-8 h-8 border-2 border-pink-500 border-t-transparent rounded-full animate-spin" />
            </div>
          )}
        </div>
      )}

      {/* ─── Questioning Stage ─── */}
      {stage === "questioning" && currentQuestion && (
        <div className="space-y-6">
          {/* Progress */}
          <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
            <div className="flex items-center justify-between text-xs text-slate-500 mb-2">
              <span>
                Pattern {currentQuestion.progress.current} of {currentQuestion.progress.total}
              </span>
              <span>
                {((currentQuestion.progress.current / currentQuestion.progress.total) * 100).toFixed(0)}%
              </span>
            </div>
            <div className="w-full bg-slate-700/50 rounded-full h-2">
              <div
                className="bg-pink-500 h-2 rounded-full transition-all"
                style={{
                  width: `${(currentQuestion.progress.current / currentQuestion.progress.total) * 100}%`,
                }}
              />
            </div>

            {/* Running totals */}
            <div className="grid grid-cols-4 gap-3 mt-4">
              <MiniStat label="Total Shrinkage" value={formatDollar(currentQuestion.running_totals.total_shrinkage)} />
              <MiniStat label="Explained" value={formatDollar(currentQuestion.running_totals.explained)} />
              <MiniStat label="Remaining" value={formatDollar(currentQuestion.running_totals.unexplained)} />
              <MiniStat
                label="Reduction"
                value={`${currentQuestion.running_totals.reduction_pct.toFixed(0)}%`}
                highlight
              />
            </div>
          </div>

          {/* Question card */}
          <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-white">{currentQuestion.pattern_name}</h3>
              <span className="text-sm text-slate-400">
                {currentQuestion.item_count} items &middot; {formatDollar(currentQuestion.total_value)}
              </span>
            </div>

            <p className="text-sm text-slate-300 mb-6">{currentQuestion.question}</p>

            {/* Sample items */}
            {currentQuestion.sample_items.length > 0 && (
              <div className="mb-6">
                <p className="text-xs text-slate-500 mb-2">Sample items:</p>
                <div className="space-y-1">
                  {currentQuestion.sample_items.slice(0, 3).map((item, i) => (
                    <div key={i} className="flex items-center justify-between text-xs bg-slate-900/50 rounded px-3 py-1.5">
                      <span className="text-slate-300 font-mono">{item.sku}</span>
                      <span className="text-slate-500">{item.description}</span>
                      <span className={item.stock < 0 ? "text-red-400" : "text-slate-400"}>
                        {item.stock} units
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Answer buttons */}
            <div className="space-y-2">
              {currentQuestion.suggested_answers.map(([label, classification], i) => (
                <button
                  key={i}
                  onClick={() => handleAnswer(classification)}
                  disabled={loading}
                  className="w-full flex items-center gap-3 px-4 py-3 bg-slate-700/30 hover:bg-slate-700/60 border border-slate-600/50 rounded-xl text-left text-sm text-slate-300 transition-colors disabled:opacity-50"
                >
                  <ChevronRight size={14} className="text-pink-400 shrink-0" />
                  <span>{label}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ─── Complete Stage ─── */}
      {stage === "complete" && report && (
        <div className="space-y-6">
          {/* Success header */}
          <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-xl p-6 text-center">
            <CheckCircle size={40} className="text-emerald-400 mx-auto mb-3" />
            <h2 className="text-2xl font-bold text-white mb-2">
              {report.summary.reduction_percent.toFixed(0)}% Shrinkage Explained
            </h2>
            <p className="text-slate-400 text-sm">
              {formatDollar(report.summary.explained_value)} of {formatDollar(report.summary.total_shrinkage)} identified as process issues
            </p>
          </div>

          {/* Summary */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <MiniCard label="Total Items" value={String(report.summary.total_items)} />
            <MiniCard label="Negative Items" value={String(report.summary.negative_items)} />
            <MiniCard label="Explained" value={formatDollar(report.summary.explained_value)} />
            <MiniCard label="To Investigate" value={formatDollar(report.summary.unexplained_value)} />
          </div>

          {/* By classification */}
          {Object.keys(report.by_classification).length > 0 && (
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
              <div className="flex items-center gap-2 mb-4">
                <BarChart3 size={16} className="text-pink-400" />
                <h3 className="text-sm font-medium text-slate-300 uppercase tracking-wider">
                  By Classification
                </h3>
              </div>
              <div className="space-y-2">
                {Object.entries(report.by_classification)
                  .sort(([, a], [, b]) => b.value - a.value)
                  .map(([cls, data]) => (
                    <div key={cls} className="flex items-center gap-3">
                      <span className="text-xs text-slate-400 w-28 truncate capitalize">
                        {cls.replace(/_/g, " ")}
                      </span>
                      <div className="flex-1 bg-slate-700/50 rounded-full h-2">
                        <div
                          className="bg-pink-500/60 h-2 rounded-full"
                          style={{
                            width: `${Math.max(2, (data.value / report.summary.total_shrinkage) * 100)}%`,
                          }}
                        />
                      </div>
                      <span className="text-xs text-slate-400 w-16 text-right">
                        {formatDollar(data.value)}
                      </span>
                      <span className="text-[10px] text-slate-600 w-12 text-right">
                        {data.items} items
                      </span>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {/* Journey */}
          {report.journey.length > 0 && (
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
              <div className="flex items-center gap-2 mb-4">
                <FileText size={16} className="text-slate-400" />
                <h3 className="text-sm font-medium text-slate-300 uppercase tracking-wider">
                  Classification Journey
                </h3>
              </div>
              <div className="space-y-2">
                {report.journey.map((step, i) => (
                  <div key={i} className="flex items-center gap-3 text-xs">
                    <span className="text-slate-500 w-5">{i + 1}.</span>
                    <span className="text-slate-300 flex-1">{step.pattern}</span>
                    <span className="px-2 py-0.5 bg-slate-700/50 rounded text-slate-400 capitalize">
                      {step.classification.replace(/_/g, " ")}
                    </span>
                    <span className="text-slate-500 w-16 text-right">{formatDollar(step.value)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Items to investigate */}
          {report.items_to_investigate.length > 0 && (
            <div className="bg-red-500/5 border border-red-500/20 rounded-xl p-5">
              <div className="flex items-center gap-2 mb-4">
                <Search size={16} className="text-red-400" />
                <h3 className="text-sm font-medium text-red-400 uppercase tracking-wider">
                  Items to Investigate ({report.items_to_investigate.length})
                </h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-slate-500 border-b border-slate-700/50">
                      <th className="text-left py-2 pr-4">SKU</th>
                      <th className="text-left py-2 pr-4">Description</th>
                      <th className="text-right py-2 pr-4">Stock</th>
                      <th className="text-right py-2">Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.items_to_investigate.slice(0, 10).map((item, i) => (
                      <tr key={i} className="text-slate-300 border-b border-slate-800/50">
                        <td className="py-1.5 pr-4 font-mono">{item.sku}</td>
                        <td className="py-1.5 pr-4 truncate max-w-[200px]">{item.description}</td>
                        <td className="text-right py-1.5 pr-4 text-red-400">{item.stock}</td>
                        <td className="text-right py-1.5">{formatDollar(item.value)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Start new */}
          <button
            onClick={resetSession}
            className="w-full py-3 bg-slate-700/50 hover:bg-slate-700 text-slate-300 font-medium rounded-xl transition-colors text-sm"
          >
            Start New Diagnostic
          </button>
        </div>
      )}
    </div>
  );
}

// ─── Shared components ───────────────────────────────────────

function MiniStat({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="text-center">
      <p className="text-[10px] text-slate-500 uppercase">{label}</p>
      <p className={`text-sm font-bold mt-0.5 ${highlight ? "text-emerald-400" : "text-white"}`}>
        {value}
      </p>
    </div>
  );
}

function MiniCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-4">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="text-xl font-bold text-white mt-1">{value}</p>
    </div>
  );
}
