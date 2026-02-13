"use client";

import React, { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import {
  Brain,
  AlertCircle,
  ChevronRight,
  GitBranch,
  Lightbulb,
  Shield,
  Zap,
  Target,
} from "lucide-react";
import {
  fetchExplanation,
  fetchDigest,
  type ExplainResponse,
  type Issue,
  type SignalContribution,
  type ProofNode,
} from "@/lib/sentinel-api";

function formatDollar(n: number): string {
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `$${(n / 1_000).toFixed(1)}K`;
  return `$${n.toFixed(0)}`;
}

export default function ExplainPage() {
  const searchParams = useSearchParams();
  const issueParam = searchParams.get("issue");

  const [issues, setIssues] = useState<Issue[]>([]);
  const [selectedIssueId, setSelectedIssueId] = useState<string>(issueParam || "");
  const [explanation, setExplanation] = useState<ExplainResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingIssues, setLoadingIssues] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load issues
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

  // Auto-select from URL
  useEffect(() => {
    if (issueParam) setSelectedIssueId(issueParam);
  }, [issueParam]);

  // Load explanation
  useEffect(() => {
    if (!selectedIssueId) return;
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetchExplanation(selectedIssueId);
        setExplanation(res);
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setLoading(false);
      }
    })();
  }, [selectedIssueId]);

  const tree = explanation?.proof_tree;

  return (
    <div className="p-6 lg:p-8 max-w-5xl">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white flex items-center gap-3">
          <Brain size={24} className="text-violet-400" />
          Symbolic Reasoning
        </h1>
        <p className="text-sm text-slate-400 mt-1">
          Transparent proof trees showing how conclusions are reached
        </p>
      </div>

      {/* Issue selector */}
      <div className="mb-6">
        <label className="block text-sm text-slate-400 mb-2">Select an issue to explain</label>
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
      {error && (
        <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 text-sm">
          <AlertCircle size={16} className="inline mr-2" />
          {error}
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="flex items-center justify-center py-16">
          <div className="text-center">
            <div className="w-10 h-10 border-2 border-violet-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-sm text-slate-400">Building proof tree...</p>
          </div>
        </div>
      )}

      {/* Explanation */}
      {tree && !loading && (
        <div className="space-y-6">
          {/* Conclusion card */}
          <div className="bg-violet-500/10 border border-violet-500/20 rounded-xl p-5">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 bg-violet-500/20 rounded-lg flex items-center justify-center">
                <Target size={20} className="text-violet-400" />
              </div>
              <div>
                <p className="text-xs text-violet-400 uppercase tracking-wider font-medium">Conclusion</p>
                <p className="text-lg font-bold text-white">{tree.issue_type}</p>
              </div>
            </div>

            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mt-4">
              <MiniStat label="Root Cause" value={tree.root_cause_display} />
              <MiniStat label="Confidence" value={`${(tree.root_cause_confidence * 100).toFixed(0)}%`} />
              <MiniStat label="Ambiguity" value={`${(tree.root_cause_ambiguity * 100).toFixed(0)}%`} />
              <MiniStat label="Impact" value={formatDollar(tree.dollar_impact)} />
            </div>
          </div>

          {/* Signal contributions */}
          {tree.signal_contributions.length > 0 && (
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
              <div className="flex items-center gap-2 mb-4">
                <Zap size={16} className="text-amber-400" />
                <h3 className="text-sm font-medium text-slate-300 uppercase tracking-wider">
                  Signal Evidence ({tree.active_signals.length} signals)
                </h3>
              </div>
              <div className="space-y-2">
                {tree.signal_contributions.map((sc, i) => (
                  <SignalRow key={i} signal={sc} />
                ))}
              </div>
            </div>
          )}

          {/* Cause scores */}
          {tree.cause_scores.length > 0 && (
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
              <div className="flex items-center gap-2 mb-4">
                <GitBranch size={16} className="text-emerald-400" />
                <h3 className="text-sm font-medium text-slate-300 uppercase tracking-wider">
                  Cause Ranking
                </h3>
              </div>
              <div className="space-y-2">
                {(tree.cause_scores as { cause: string; score: number }[])
                  .sort((a, b) => b.score - a.score)
                  .map((cs, i) => (
                    <div key={i} className="flex items-center gap-3">
                      <span className="text-xs text-slate-500 w-20 truncate">
                        {cs.cause.replace(/_/g, " ")}
                      </span>
                      <div className="flex-1 bg-slate-700/50 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full ${
                            i === 0 ? "bg-emerald-500" : "bg-slate-500"
                          }`}
                          style={{ width: `${Math.max(2, cs.score * 100)}%` }}
                        />
                      </div>
                      <span className="text-xs text-slate-400 w-10 text-right">
                        {(cs.score * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {/* Competing hypotheses */}
          {tree.competing_hypotheses.length > 0 && (
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
              <div className="flex items-center gap-2 mb-4">
                <Shield size={16} className="text-amber-400" />
                <h3 className="text-sm font-medium text-slate-300 uppercase tracking-wider">
                  Competing Hypotheses
                </h3>
              </div>
              <div className="space-y-3">
                {tree.competing_hypotheses.map((h, i) => (
                  <div key={i} className="flex items-center justify-between px-3 py-2 bg-slate-900/50 rounded-lg">
                    <span className="text-sm text-slate-300">{h.cause_display}</span>
                    <div className="flex items-center gap-3 text-xs text-slate-500">
                      <span>#{h.rank}</span>
                      <span>Score: {(h.score * 100).toFixed(0)}%</span>
                      <span className="text-slate-600 truncate max-w-[200px]">{h.why_lower}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Inferred facts */}
          {tree.inferred_facts.length > 0 && (
            <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-5">
              <div className="flex items-center gap-2 mb-4">
                <Lightbulb size={16} className="text-yellow-400" />
                <h3 className="text-sm font-medium text-slate-300 uppercase tracking-wider">
                  Inferred Facts
                </h3>
              </div>
              <div className="space-y-1.5">
                {(tree.inferred_facts as { fact: string; confidence: number; rule: string }[]).map((f, i) => (
                  <div key={i} className="flex items-start gap-2 text-sm">
                    <ChevronRight size={12} className="text-yellow-400 mt-1 shrink-0" />
                    <span className="text-slate-300">{f.fact}</span>
                    <span className="text-[10px] text-slate-600 shrink-0">
                      ({(f.confidence * 100).toFixed(0)}% via {f.rule})
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Recommendations */}
          {tree.recommendations.length > 0 && (
            <div className="bg-emerald-500/5 border border-emerald-500/20 rounded-xl p-5">
              <div className="flex items-center gap-2 mb-4">
                <Lightbulb size={16} className="text-emerald-400" />
                <h3 className="text-sm font-medium text-emerald-400 uppercase tracking-wider">
                  Recommendations
                </h3>
              </div>
              <ul className="space-y-2">
                {tree.recommendations.map((r, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm text-slate-300">
                    <span className="text-emerald-400 mt-0.5">&#8226;</span>
                    {r}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Proof tree visualization */}
          {tree.proof_tree && (
            <details className="bg-slate-800/50 border border-slate-700/50 rounded-xl overflow-hidden">
              <summary className="px-5 py-4 cursor-pointer text-sm font-medium text-slate-300 hover:text-white transition-colors">
                <GitBranch size={14} className="inline mr-2 text-violet-400" />
                View Proof Tree
              </summary>
              <div className="px-5 pb-4 border-t border-slate-700/30 mt-2">
                <ProofNodeTree node={tree.proof_tree} depth={0} />
              </div>
            </details>
          )}
        </div>
      )}

      {/* Empty state */}
      {!selectedIssueId && !loading && (
        <div className="text-center py-16 text-slate-500">
          <Brain size={32} className="mx-auto mb-3 opacity-50" />
          <p className="font-medium">Select an Issue</p>
          <p className="text-sm mt-1">Choose an issue to see the symbolic reasoning proof tree.</p>
        </div>
      )}
    </div>
  );
}

// ─── Sub-components ──────────────────────────────────────────

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-slate-900/50 rounded-lg px-3 py-2">
      <p className="text-[10px] text-slate-500 uppercase">{label}</p>
      <p className="text-sm font-bold text-white mt-0.5">{value}</p>
    </div>
  );
}

function SignalRow({ signal }: { signal: SignalContribution }) {
  return (
    <div className="flex items-center gap-3 px-3 py-2 bg-slate-900/30 rounded-lg">
      <span className="text-xs font-mono px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400">
        {signal.signal.replace(/_/g, " ")}
      </span>
      <div className="flex-1 min-w-0">
        <p className="text-xs text-slate-300 truncate">{signal.description}</p>
        {signal.rules_fired.length > 0 && (
          <p className="text-[10px] text-slate-600 truncate">
            {signal.rules_fired.length} rule{signal.rules_fired.length !== 1 ? "s" : ""} fired
          </p>
        )}
      </div>
    </div>
  );
}

function ProofNodeTree({ node, depth }: { node: ProofNode; depth: number }) {
  const [expanded, setExpanded] = useState(depth < 2);
  const hasChildren = node.children && node.children.length > 0;

  return (
    <div className={`${depth > 0 ? "ml-4 border-l border-violet-500/20 pl-3" : ""}`}>
      <button
        onClick={() => hasChildren && setExpanded(!expanded)}
        className={`flex items-center gap-2 py-1 text-sm ${
          hasChildren ? "cursor-pointer hover:text-white" : "cursor-default"
        } text-slate-300`}
      >
        {hasChildren && (
          <ChevronRight
            size={12}
            className={`text-violet-400 transition-transform ${expanded ? "rotate-90" : ""}`}
          />
        )}
        {!hasChildren && <span className="w-3" />}
        <span className="text-[10px] text-slate-600 font-mono">{node.source}</span>
        <span>{node.statement}</span>
        <span className="text-[10px] text-slate-500">
          ({(node.confidence * 100).toFixed(0)}%)
        </span>
      </button>
      {node.explanation && (
        <p className="text-[10px] text-slate-600 ml-5 mt-0.5">{node.explanation}</p>
      )}
      {expanded && hasChildren && (
        <div className="mt-1">
          {node.children.map((child, i) => (
            <ProofNodeTree key={i} node={child} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
}
