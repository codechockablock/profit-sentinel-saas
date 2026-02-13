"use client";

import React, { useState, useEffect, useCallback } from "react";
import { useSearchParams } from "next/navigation";
import {
  ClipboardList,
  AlertCircle,
  Check,
  Clock,
  ArrowUpCircle,
  UserPlus,
  ChevronDown,
  RefreshCw,
  X,
} from "lucide-react";
import {
  fetchTasks,
  fetchDigest,
  delegateIssue,
  updateTaskStatus,
  type TaskListResponse,
  type TaskResponse,
  type TaskStatus,
  type Issue,
} from "@/lib/sentinel-api";
import { ApiErrorBanner } from "@/components/dashboard/ApiErrorBanner";

// ─── Helpers ─────────────────────────────────────────────────

function formatDollar(n: number): string {
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `$${(n / 1_000).toFixed(1)}K`;
  return `$${n.toFixed(0)}`;
}

const STATUS_META: Record<TaskStatus, { icon: React.ReactNode; color: string; label: string }> = {
  open: { icon: <Clock size={14} />, color: "text-blue-400 bg-blue-500/10 border-blue-500/20", label: "Open" },
  in_progress: { icon: <ArrowUpCircle size={14} />, color: "text-amber-400 bg-amber-500/10 border-amber-500/20", label: "In Progress" },
  completed: { icon: <Check size={14} />, color: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20", label: "Completed" },
  escalated: { icon: <AlertCircle size={14} />, color: "text-red-400 bg-red-500/10 border-red-500/20", label: "Escalated" },
};

const PRIORITY_COLORS: Record<string, string> = {
  critical: "text-red-400 bg-red-500/10 border-red-500/30",
  high: "text-orange-400 bg-orange-500/10 border-orange-500/30",
  medium: "text-yellow-400 bg-yellow-500/10 border-yellow-500/30",
  low: "text-blue-400 bg-blue-500/10 border-blue-500/30",
};

// ─── Component ───────────────────────────────────────────────

export default function TasksPage() {
  const searchParams = useSearchParams();
  const delegateIssueId = searchParams.get("delegate");

  const [taskList, setTaskList] = useState<TaskListResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>("");
  const [priorityFilter, setPriorityFilter] = useState<string>("");

  // Delegation modal state
  const [showDelegateModal, setShowDelegateModal] = useState(false);
  const [delegateIssueData, setDelegateIssueData] = useState<Issue | null>(null);
  const [assignee, setAssignee] = useState("");
  const [delegateNotes, setDelegateNotes] = useState("");
  const [delegating, setDelegating] = useState(false);

  const loadTasks = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchTasks({
        status: statusFilter || undefined,
        priority: priorityFilter || undefined,
      });
      setTaskList(res);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, [statusFilter, priorityFilter]);

  useEffect(() => {
    loadTasks();
  }, [loadTasks]);

  // Handle deep-link delegation
  useEffect(() => {
    if (delegateIssueId) {
      (async () => {
        try {
          const digest = await fetchDigest(undefined, 20);
          const issue = digest.digest.issues.find((i) => i.id === delegateIssueId);
          if (issue) {
            setDelegateIssueData(issue);
            setShowDelegateModal(true);
          }
        } catch {
          // Silently fail — issue might not exist
        }
      })();
    }
  }, [delegateIssueId]);

  const handleDelegate = async () => {
    if (!delegateIssueData || !assignee.trim()) return;
    setDelegating(true);
    try {
      await delegateIssue({
        issue_id: delegateIssueData.id,
        assignee: assignee.trim(),
        notes: delegateNotes.trim() || undefined,
      });
      setShowDelegateModal(false);
      setAssignee("");
      setDelegateNotes("");
      setDelegateIssueData(null);
      await loadTasks();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setDelegating(false);
    }
  };

  const handleStatusUpdate = async (taskId: string, newStatus: TaskStatus) => {
    try {
      await updateTaskStatus(taskId, newStatus);
      await loadTasks();
    } catch (err) {
      setError((err as Error).message);
    }
  };

  return (
    <div className="p-6 lg:p-8 max-w-6xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-3">
            <ClipboardList size={24} className="text-emerald-400" />
            Task Board
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            {taskList ? `${taskList.total} task${taskList.total !== 1 ? "s" : ""}` : "Loading..."}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="bg-slate-800 border border-slate-700 text-slate-300 text-sm rounded-lg px-3 py-2"
          >
            <option value="">All Status</option>
            <option value="open">Open</option>
            <option value="in_progress">In Progress</option>
            <option value="completed">Completed</option>
            <option value="escalated">Escalated</option>
          </select>
          <select
            value={priorityFilter}
            onChange={(e) => setPriorityFilter(e.target.value)}
            className="bg-slate-800 border border-slate-700 text-slate-300 text-sm rounded-lg px-3 py-2"
          >
            <option value="">All Priority</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>
          <button
            onClick={loadTasks}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-sm font-medium rounded-lg transition-colors"
          >
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
            Refresh
          </button>
        </div>
      </div>

      {/* Error */}
      <ApiErrorBanner error={error} onRetry={loadTasks} />

      {/* Loading */}
      {loading && !taskList && (
        <div className="flex items-center justify-center py-24">
          <div className="w-10 h-10 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
        </div>
      )}

      {/* Task list */}
      {taskList && (
        <div className="space-y-3">
          {taskList.tasks.map((tr) => (
            <TaskCard
              key={tr.task.task_id}
              tr={tr}
              onStatusUpdate={handleStatusUpdate}
            />
          ))}

          {taskList.tasks.length === 0 && (
            <div className="text-center py-16 text-slate-500">
              <ClipboardList size={32} className="mx-auto mb-3 opacity-50" />
              <p className="font-medium">No Tasks</p>
              <p className="text-sm mt-1">
                Delegate issues from the Morning Digest to create tasks.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Delegate modal */}
      {showDelegateModal && delegateIssueData && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
          <div className="bg-slate-900 border border-slate-700 rounded-xl p-6 w-full max-w-md">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <UserPlus size={18} className="text-emerald-400" />
                Delegate Issue
              </h2>
              <button
                onClick={() => setShowDelegateModal(false)}
                className="text-slate-400 hover:text-white"
                aria-label="Close delegate modal"
              >
                <X size={18} />
              </button>
            </div>

            <p className="text-sm text-slate-400 mb-4">
              {delegateIssueData.issue_type.replace(/([A-Z])/g, " $1").trim()} &mdash;{" "}
              {formatDollar(delegateIssueData.dollar_impact)} impact
            </p>

            <div className="space-y-4">
              <div>
                <label className="block text-sm text-slate-400 mb-1">
                  Assign to
                </label>
                <input
                  type="text"
                  value={assignee}
                  onChange={(e) => setAssignee(e.target.value)}
                  placeholder="e.g. John Smith"
                  className="w-full px-4 py-2.5 bg-slate-800 border border-slate-600 rounded-lg text-white text-sm focus:outline-none focus:border-emerald-500"
                />
              </div>
              <div>
                <label className="block text-sm text-slate-400 mb-1">
                  Notes (optional)
                </label>
                <textarea
                  value={delegateNotes}
                  onChange={(e) => setDelegateNotes(e.target.value)}
                  placeholder="Additional context for the assignee..."
                  rows={3}
                  className="w-full px-4 py-2.5 bg-slate-800 border border-slate-600 rounded-lg text-white text-sm focus:outline-none focus:border-emerald-500 resize-none"
                />
              </div>
              <button
                onClick={handleDelegate}
                disabled={!assignee.trim() || delegating}
                className="w-full py-2.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white font-medium rounded-lg transition-colors text-sm"
              >
                {delegating ? "Creating task..." : "Create Task"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Task Card ───────────────────────────────────────────────

function TaskCard({
  tr,
  onStatusUpdate,
}: {
  tr: TaskResponse;
  onStatusUpdate: (id: string, status: TaskStatus) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [showStatusMenu, setShowStatusMenu] = useState(false);
  const status = STATUS_META[tr.status];
  const priorityCls = PRIORITY_COLORS[tr.task.priority] || PRIORITY_COLORS.low;

  return (
    <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-4 px-5 py-4 text-left"
      >
        {/* Priority */}
        <span className={`shrink-0 px-2 py-0.5 text-[10px] font-bold uppercase rounded border ${priorityCls}`}>
          {tr.task.priority}
        </span>

        {/* Title */}
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-white truncate">{tr.task.title}</p>
          <p className="text-xs text-slate-500 mt-0.5">
            {tr.task.assignee} &middot; Due{" "}
            {new Date(tr.task.deadline).toLocaleDateString()}
          </p>
        </div>

        {/* Status badge */}
        <span className={`shrink-0 flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-lg border ${status.color}`}>
          {status.icon}
          {status.label}
        </span>

        {/* Impact */}
        <span className="text-sm font-bold text-white shrink-0">
          {formatDollar(tr.task.dollar_impact)}
        </span>

        <ChevronDown
          size={16}
          className={`text-slate-500 transition-transform ${expanded ? "rotate-180" : ""}`}
        />
      </button>

      {expanded && (
        <div className="px-5 pb-4 border-t border-slate-700/30 space-y-4">
          <p className="text-sm text-slate-300 mt-3">{tr.task.description}</p>

          {/* Action items */}
          {tr.task.action_items.length > 0 && (
            <div>
              <p className="text-xs text-slate-500 font-medium mb-2 uppercase tracking-wider">
                Action Items
              </p>
              <ul className="space-y-1.5">
                {tr.task.action_items.map((item, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm text-slate-300">
                    <span className="text-emerald-400 mt-0.5">&#8226;</span>
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Notes */}
          {tr.notes.length > 0 && (
            <div>
              <p className="text-xs text-slate-500 font-medium mb-2 uppercase tracking-wider">
                Notes
              </p>
              {tr.notes.map((note, i) => (
                <p key={i} className="text-sm text-slate-400 italic">{note}</p>
              ))}
            </div>
          )}

          {/* Status update */}
          <div className="relative">
            <button
              onClick={() => setShowStatusMenu(!showStatusMenu)}
              className="flex items-center gap-2 px-3 py-1.5 bg-slate-700/50 text-slate-300 text-xs rounded-lg hover:bg-slate-700 transition-colors"
            >
              Update Status
              <ChevronDown size={12} />
            </button>
            {showStatusMenu && (
              <div className="absolute bottom-full mb-1 left-0 bg-slate-800 border border-slate-700 rounded-lg overflow-hidden shadow-xl z-10">
                {(["open", "in_progress", "completed", "escalated"] as TaskStatus[]).map((s) => (
                  <button
                    key={s}
                    onClick={() => {
                      onStatusUpdate(tr.task.task_id, s);
                      setShowStatusMenu(false);
                    }}
                    className={`block w-full text-left px-4 py-2 text-xs hover:bg-slate-700 transition-colors ${
                      tr.status === s ? "text-emerald-400" : "text-slate-300"
                    }`}
                  >
                    {STATUS_META[s].label}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
