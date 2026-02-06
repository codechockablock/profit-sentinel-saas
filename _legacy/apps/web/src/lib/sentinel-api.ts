/**
 * Sentinel API Client — Typed wrapper for all M8 backend endpoints.
 *
 * All endpoints require authentication (Bearer token from Supabase).
 * Base URL: https://api.profitsentinel.com/api/v1/
 */

import { getApiUrl } from './api-config';
import { getAuthHeaders } from './supabase';

// ─── Base helpers ────────────────────────────────────────────

const BASE = () => `${getApiUrl()}/api/v1`;

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const auth = await getAuthHeaders();
  const res = await fetch(`${BASE()}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...auth,
      ...init?.headers,
    },
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`API ${res.status}: ${body || res.statusText}`);
  }
  return res.json();
}

// ─── Enums ───────────────────────────────────────────────────

export type IssueType =
  | 'ReceivingGap'
  | 'DeadStock'
  | 'MarginErosion'
  | 'NegativeInventory'
  | 'VendorShortShip'
  | 'PurchasingLeakage'
  | 'PatronageMiss'
  | 'ShrinkagePattern'
  | 'ZeroCostAnomaly'
  | 'PriceDiscrepancy'
  | 'Overstock';

export type RootCause =
  | 'Theft'
  | 'VendorIncrease'
  | 'RebateTiming'
  | 'MarginLeak'
  | 'DemandShift'
  | 'QualityIssue'
  | 'PricingError'
  | 'InventoryDrift';

export type TrendDirection = 'Worsening' | 'Stable' | 'Improving';
export type TaskStatus = 'open' | 'in_progress' | 'completed' | 'escalated';
export type TaskPriority = 'critical' | 'high' | 'medium' | 'low';

// ─── Shared Models ───────────────────────────────────────────

export interface Sku {
  sku_id: string;
  qty_on_hand: number;
  unit_cost: number;
  retail_price: number;
  margin_pct: number;
  sales_last_30d: number;
  days_since_receipt: number;
  is_damaged: boolean;
  on_order_qty: number;
  is_seasonal: boolean;
}

export interface CauseScoreDetail {
  cause: string;
  score: number;
  confidence: number;
}

export interface Issue {
  id: string;
  issue_type: IssueType;
  store_id: string;
  dollar_impact: number;
  confidence: number;
  trend_direction: TrendDirection;
  priority_score: number;
  urgency_score: number;
  detection_timestamp: string;
  skus: Sku[];
  context: string;
  root_cause: RootCause | null;
  root_cause_confidence: number | null;
  cause_scores: CauseScoreDetail[];
  root_cause_ambiguity: number | null;
  active_signals: string[];
}

// ─── 1. Morning Digest ──────────────────────────────────────

export interface DigestSummary {
  total_issues: number;
  critical_issues: number;
  total_dollar_impact: number;
  stores_affected: number;
  top_issue_type: string | null;
}

export interface Digest {
  generated_at: string;
  store_filter: string[];
  pipeline_ms: number;
  issues: Issue[];
  summary: DigestSummary;
}

export interface DigestResponse {
  digest: Digest;
  rendered_text: string;
  generated_at: string;
  store_ids: string[];
  issue_count: number;
  total_dollar_impact: number;
}

export async function fetchDigest(
  stores?: string[],
  topK: number = 5
): Promise<DigestResponse> {
  const params = new URLSearchParams({ top_k: String(topK) });
  if (stores?.length) params.set('stores', stores.join(','));
  return apiFetch(`/digest?${params}`);
}

export async function fetchStoreDigest(
  storeId: string,
  topK: number = 5
): Promise<DigestResponse> {
  const params = new URLSearchParams({ top_k: String(topK) });
  return apiFetch(`/digest/${storeId}?${params}`);
}

// ─── 2. Task Delegation ─────────────────────────────────────

export interface Task {
  task_id: string;
  issue_id: string;
  issue_type: IssueType;
  store_id: string;
  assignee: string;
  deadline: string;
  priority: TaskPriority;
  title: string;
  description: string;
  action_items: string[];
  dollar_impact: number;
  skus: Sku[];
  created_at: string;
}

export interface TaskResponse {
  task: Task;
  status: TaskStatus;
  rendered_text: string;
  notes: string[];
}

export interface TaskListResponse {
  tasks: TaskResponse[];
  total: number;
}

export interface DelegateRequest {
  issue_id: string;
  assignee: string;
  deadline?: string;
  notes?: string;
}

export interface DelegateResponse {
  task: Task;
  rendered_text: string;
  task_id: string;
}

export async function delegateIssue(req: DelegateRequest): Promise<DelegateResponse> {
  return apiFetch('/delegate', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

export async function fetchTasks(filters?: {
  store_id?: string;
  priority?: string;
  status?: string;
}): Promise<TaskListResponse> {
  const params = new URLSearchParams();
  if (filters?.store_id) params.set('store_id', filters.store_id);
  if (filters?.priority) params.set('priority', filters.priority);
  if (filters?.status) params.set('status', filters.status);
  const qs = params.toString();
  return apiFetch(`/tasks${qs ? `?${qs}` : ''}`);
}

export async function fetchTask(taskId: string): Promise<TaskResponse> {
  return apiFetch(`/tasks/${taskId}`);
}

export async function updateTaskStatus(
  taskId: string,
  status: TaskStatus,
  notes?: string
): Promise<TaskResponse> {
  return apiFetch(`/tasks/${taskId}`, {
    method: 'PATCH',
    body: JSON.stringify({ status, notes }),
  });
}

// ─── 3. Vendor Call Prep ─────────────────────────────────────

export interface CallPrep {
  issue_id: string;
  store_id: string;
  vendor_name: string;
  issue_summary: string;
  affected_skus: Sku[];
  total_dollar_impact: number;
  talking_points: string[];
  questions_to_ask: string[];
  historical_context: string;
}

export interface VendorCallResponse {
  call_prep: CallPrep;
  rendered_text: string;
}

export async function fetchVendorCallPrep(issueId: string): Promise<VendorCallResponse> {
  return apiFetch(`/vendor-call/${issueId}`);
}

// ─── 4. Co-op Intelligence ──────────────────────────────────

export interface CoopAlert {
  alert_type: string;
  store_id: string;
  title: string;
  dollar_impact: number;
  detail: string;
  recommendation: string;
  confidence: number;
}

export interface VendorRebateStatus {
  program: { vendor: string; tiers: unknown[] };
  store_id: string;
  ytd_purchases: number;
  current_tier: { name: string; threshold: number; rebate_pct: number } | null;
  next_tier: { name: string; threshold: number; rebate_pct: number } | null;
  shortfall: number;
  days_remaining: number;
  daily_run_rate: number;
  projected_total: number;
  on_track: boolean;
  current_rebate_value: number;
  next_tier_rebate_value: number;
  incremental_value: number;
  recommendation: string;
}

export interface CoopReportResponse {
  report: {
    store_id: string;
    generated_at: string;
    alerts: CoopAlert[];
    total_opportunity: number;
    rebate_statuses: VendorRebateStatus[];
    health_report: {
      total_inventory_value: number;
      total_dead_stock_value: number;
      dead_stock_pct: number;
      annual_carrying_cost: number;
      overall_turn_rate: number;
      overall_gmroi: number;
    } | null;
    category_analysis: {
      total_revenue: number;
      total_margin_pct: number;
      total_opportunity: number;
      top_expansion_categories: string[];
      top_contraction_categories: string[];
    } | null;
  };
  rendered_text: string;
  health_summary: string | null;
  rebate_statuses: VendorRebateStatus[];
  total_opportunity: number;
}

export async function fetchCoopReport(storeId: string): Promise<CoopReportResponse> {
  return apiFetch(`/coop/${storeId}`);
}

// ─── 5. Symbolic Reasoning (Explain) ────────────────────────

export interface SignalContribution {
  signal: string;
  cause: string;
  weight: number;
  rationale: string;
}

export interface ProofNode {
  type: string;
  label: string;
  confidence: number;
  children: ProofNode[];
}

export interface CompetingHypothesis {
  cause: string;
  score: number;
  gap: number;
}

export interface ExplainResponse {
  issue_id: string;
  proof_tree: {
    issue_id: string;
    issue_type: string;
    store_id: string;
    dollar_impact: number;
    root_cause: string | null;
    root_cause_display: string;
    root_cause_confidence: number;
    root_cause_ambiguity: number;
    active_signals: string[];
    signal_contributions: SignalContribution[];
    cause_scores: { cause: string; score: number }[];
    proof_root: ProofNode;
    inferred_facts: { fact: string; confidence: number; rule: string }[];
    competing_hypotheses: CompetingHypothesis[];
    recommendations: string[];
    suggested_actions: { action: string; priority: string; confidence: number }[];
  };
  rendered_text: string;
}

export interface BackwardChainResponse {
  issue_id: string;
  goal: string;
  reasoning_steps: { step: number; rule: string; conclusion: string; confidence: number }[];
}

export async function fetchExplanation(issueId: string): Promise<ExplainResponse> {
  return apiFetch(`/explain/${issueId}`);
}

export async function fetchBackwardChain(
  issueId: string,
  goal: string
): Promise<BackwardChainResponse> {
  return apiFetch(`/explain/${issueId}/why`, {
    method: 'POST',
    body: JSON.stringify({ goal }),
  });
}

// ─── 6. Conversational Diagnostic ───────────────────────────

export interface DiagnosticStartRequest {
  items: { sku: string; description: string; stock: number; cost: number }[];
  store_name?: string;
}

export interface DiagnosticStartResponse {
  session_id: string;
  store_name: string;
  total_items: number;
  negative_items: number;
  total_shrinkage: number;
  patterns_detected: number;
}

export interface DiagnosticQuestion {
  pattern_id: string;
  pattern_name: string;
  question: string;
  suggested_answers: [string, string][];
  item_count: number;
  total_value: number;
  sample_items: { sku: string; description: string; stock: number; cost: number; value: number }[];
  progress: { current: number; total: number };
  running_totals: {
    total_shrinkage: number;
    explained: number;
    unexplained: number;
    reduction_pct: number;
  };
}

export interface DiagnosticAnswerResponse {
  answered: { pattern: string; classification: string; value: number };
  progress: { current: number; total: number };
  running_totals: {
    total_shrinkage: number;
    explained: number;
    unexplained: number;
    reduction_pct: number;
  };
  is_complete: boolean;
  next_question: DiagnosticQuestion | null;
}

export interface DiagnosticSummary {
  session_id: string;
  store_name: string;
  status: string;
  total_items: number;
  negative_items: number;
  total_shrinkage: number;
  explained_value: number;
  unexplained_value: number;
  reduction_percent: number;
  patterns_total: number;
  patterns_answered: number;
}

export interface DiagnosticReport {
  session_id: string;
  summary: DiagnosticSummary;
  by_classification: Record<string, { items: number; value: number }>;
  items_to_investigate: { sku: string; description: string; stock: number; value: number }[];
  journey: { pattern: string; classification: string; value: number; note: string }[];
  rendered_text: string;
}

export async function startDiagnostic(req: DiagnosticStartRequest): Promise<DiagnosticStartResponse> {
  return apiFetch('/diagnostic/start', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

export async function fetchDiagnosticQuestion(sessionId: string): Promise<DiagnosticQuestion | null> {
  return apiFetch(`/diagnostic/${sessionId}/question`);
}

export async function answerDiagnosticQuestion(
  sessionId: string,
  classification: string,
  note?: string
): Promise<DiagnosticAnswerResponse> {
  return apiFetch(`/diagnostic/${sessionId}/answer`, {
    method: 'POST',
    body: JSON.stringify({ classification, note: note || '' }),
  });
}

export async function fetchDiagnosticSummary(sessionId: string): Promise<DiagnosticSummary> {
  return apiFetch(`/diagnostic/${sessionId}/summary`);
}

export async function fetchDiagnosticReport(sessionId: string): Promise<DiagnosticReport> {
  return apiFetch(`/diagnostic/${sessionId}/report`);
}

// ---------------------------------------------------------------------------
// Digest Email Subscriptions
// ---------------------------------------------------------------------------

export interface Subscription {
  email: string;
  stores: string[];
  enabled: boolean;
  send_hour: number;
  timezone: string;
  created_at: string;
}

export interface SubscribeRequest {
  email: string;
  stores?: string[];
  send_hour?: number;
  timezone?: string;
}

export interface SubscribeResponse {
  subscription: Subscription;
  message: string;
}

export interface SubscriptionListResponse {
  subscriptions: Subscription[];
  total: number;
}

export interface DigestSendResponse {
  email_id: string | null;
  message: string;
}

export interface SchedulerStatus {
  enabled: boolean;
  running: boolean;
  subscribers: number;
  send_hour: number;
}

export async function subscribeDigest(req: SubscribeRequest): Promise<SubscribeResponse> {
  return apiFetch('/digest/subscribe', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

export async function listSubscriptions(): Promise<SubscriptionListResponse> {
  return apiFetch('/digest/subscriptions');
}

export async function unsubscribeDigest(email: string): Promise<{ message: string }> {
  return apiFetch(`/digest/subscribe/${encodeURIComponent(email)}`, {
    method: 'DELETE',
  });
}

export async function sendDigestNow(email: string): Promise<DigestSendResponse> {
  return apiFetch('/digest/send', {
    method: 'POST',
    body: JSON.stringify({ email }),
  });
}

export async function fetchSchedulerStatus(): Promise<SchedulerStatus> {
  return apiFetch('/digest/scheduler-status');
}

// ---------------------------------------------------------------------------
// Analysis History
// ---------------------------------------------------------------------------

export interface AnalysisListItem {
  id: string;
  analysis_label: string | null;
  original_filename: string | null;
  file_row_count: number;
  file_column_count: number | null;
  detection_counts: Record<string, number>;
  total_impact_estimate_low: number;
  total_impact_estimate_high: number;
  processing_time_seconds: number | null;
  has_full_result: boolean;
  created_at: string | null;
}

export interface AnalysisListResponse {
  analyses: AnalysisListItem[];
  total: number;
}

export interface AnalysisDetail extends AnalysisListItem {
  full_result: Record<string, unknown> | null;
}

export interface LeakTrend {
  leak_key: string;
  current_count: number;
  previous_count: number;
  count_delta: number;
  current_impact: number;
  previous_impact: number;
  impact_delta: number;
  status: 'new' | 'resolved' | 'worsening' | 'improving' | 'stable';
  severity: 'critical' | 'warning' | 'success' | 'info' | 'neutral';
}

export interface ComparisonSummary {
  overall_trend: 'improving' | 'worsening' | 'stable';
  issues_delta: number;
  current_total_issues: number;
  previous_total_issues: number;
  impact_delta_low: number;
  impact_delta_high: number;
  new_leak_count: number;
  resolved_leak_count: number;
  worsening_leak_count: number;
  improving_leak_count: number;
  current_rows: number;
  previous_rows: number;
}

export interface CompareResponse {
  summary: ComparisonSummary;
  leak_trends: LeakTrend[];
  new_leaks: string[];
  resolved_leaks: string[];
  worsening_leaks: string[];
  improving_leaks: string[];
  metadata: {
    current_analysis_id: string | null;
    previous_analysis_id: string | null;
    current_label: string | null;
    previous_label: string | null;
    current_created_at: string | null;
    previous_created_at: string | null;
  };
}

export async function listAnalyses(limit = 20, offset = 0): Promise<AnalysisListResponse> {
  return apiFetch(`/analyses?limit=${limit}&offset=${offset}`);
}

export async function fetchAnalysis(id: string): Promise<AnalysisDetail> {
  return apiFetch(`/analyses/${id}`);
}

export async function renameAnalysis(id: string, label: string): Promise<{ message: string }> {
  return apiFetch(`/analyses/${id}`, {
    method: 'PATCH',
    body: JSON.stringify({ label }),
  });
}

export async function deleteAnalysis(id: string): Promise<{ message: string }> {
  return apiFetch(`/analyses/${id}`, {
    method: 'DELETE',
  });
}

export async function compareAnalyses(
  currentId: string,
  previousId: string
): Promise<CompareResponse> {
  return apiFetch('/analyses/compare', {
    method: 'POST',
    body: JSON.stringify({ current_id: currentId, previous_id: previousId }),
  });
}
