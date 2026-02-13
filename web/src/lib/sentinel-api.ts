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
  evidence_count: number;
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
  total_dollar_impact: number;
  stores_affected: number;
  records_processed: number;
  issues_detected: number;
  issues_filtered_out: number;
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
  store_filter: string[];
  issue_count: number;
  total_dollar_impact: number;
}

export async function fetchDigest(
  stores?: string[],
  topK: number = 5
): Promise<DigestResponse | null> {
  const params = new URLSearchParams({ top_k: String(topK) });
  if (stores?.length) params.set('stores', stores.join(','));
  try {
    return await apiFetch<DigestResponse>(`/digest?${params}`);
  } catch (err) {
    const msg = (err as Error).message || '';
    if (msg.startsWith('API 404:') && msg.includes('NO_DATA')) {
      return null;
    }
    throw err;
  }
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

export async function deleteTask(taskId: string): Promise<void> {
  await apiFetch(`/tasks/${taskId}`, { method: 'DELETE' });
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
  description: string;
  rules_fired: Record<string, unknown>[];
}

export interface ProofNode {
  statement: string;
  confidence: number;
  explanation: string;
  source: string;
  children: ProofNode[];
}

export interface CompetingHypothesis {
  cause: string;
  cause_display: string;
  score: number;
  rank: number;
  why_lower: string;
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
    cause_scores: Record<string, unknown>[];
    proof_tree: ProofNode;
    inferred_facts: Record<string, unknown>[];
    competing_hypotheses: CompetingHypothesis[];
    recommendations: string[];
    suggested_actions: Record<string, unknown>[];
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

// ---------------------------------------------------------------------------
// Vendor Performance Scoring
// ---------------------------------------------------------------------------

export interface DimensionScore {
  dimension: string;
  score: number;
  weight: number;
  weighted_score: number;
  grade: string;
  details: string;
}

export interface VendorScorecard {
  vendor_id: string;
  vendor_name: string;
  overall_score: number;
  overall_grade: string;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  dimensions: DimensionScore[];
  recommendations: string[];
  issues_analyzed: number;
  total_dollar_impact: number;
}

export interface VendorScoresResponse {
  store_id: string;
  scorecards: VendorScorecard[];
  total_vendors_scored: number;
  average_score: number;
  high_risk_vendors: number;
  total_quality_cost: number;
  top_recommendation: string;
}

export async function fetchVendorScores(storeId?: string): Promise<VendorScoresResponse> {
  const params = new URLSearchParams();
  if (storeId) params.set('store_id', storeId);
  const qs = params.toString();
  return apiFetch(`/vendor-scores${qs ? `?${qs}` : ''}`);
}

// ---------------------------------------------------------------------------
// Predictive Inventory Alerts
// ---------------------------------------------------------------------------

export type AlertSeverity = 'critical' | 'warning' | 'watch';
export type PredictionType = 'stockout' | 'overstock' | 'demand_surge' | 'velocity_drop';

export interface InventoryPrediction {
  sku_id: string;
  store_id: string;
  prediction_type: PredictionType;
  severity: AlertSeverity;
  days_until_event: number;
  confidence: number;
  estimated_lost_revenue: number;
  estimated_carrying_cost: number;
  recommendation: string;
  daily_velocity: number;
  current_qty: number;
}

export interface PredictiveReportResponse {
  store_id: string;
  total_predictions: number;
  critical_alerts: number;
  warning_alerts: number;
  total_revenue_at_risk: number;
  total_carrying_cost_at_risk: number;
  stockout_predictions: InventoryPrediction[];
  overstock_predictions: InventoryPrediction[];
  velocity_alerts: InventoryPrediction[];
  top_recommendation: string;
}

export async function fetchPredictions(
  storeId?: string,
  horizonDays?: number
): Promise<PredictiveReportResponse> {
  const params = new URLSearchParams();
  if (storeId) params.set('store_id', storeId);
  if (horizonDays) params.set('horizon_days', String(horizonDays));
  const qs = params.toString();
  return apiFetch(`/predictions${qs ? `?${qs}` : ''}`);
}

// ---------------------------------------------------------------------------
// Enterprise API Keys
// ---------------------------------------------------------------------------

export type ApiTier = 'free' | 'pro' | 'enterprise';

export interface TierLimits {
  requests_per_hour: number;
  requests_per_day: number;
}

export interface ApiKeyRecord {
  key_id: string;
  user_id: string;
  name: string;
  tier: ApiTier;
  is_active: boolean;
  is_test: boolean;
  usage_count: number;
  last_used_at: string | null;
  created_at: string;
  limits: TierLimits;
}

export interface CreateApiKeyRequest {
  name?: string;
  tier?: ApiTier;
  test?: boolean;
}

export interface CreateApiKeyResponse {
  key: string;
  record: ApiKeyRecord;
}

export interface ApiKeyListResponse {
  keys: ApiKeyRecord[];
  total: number;
}

export interface ApiKeyUsageStats {
  key_id: string;
  usage_count_total: number;
  usage_last_hour: number;
  remaining_hourly: number;
  tier: ApiTier;
  limits: TierLimits;
}

export async function createApiKey(req?: CreateApiKeyRequest): Promise<CreateApiKeyResponse> {
  return apiFetch('/api-keys', {
    method: 'POST',
    body: JSON.stringify(req || {}),
  });
}

export async function listApiKeys(): Promise<ApiKeyListResponse> {
  return apiFetch('/api-keys');
}

export async function revokeApiKey(keyId: string): Promise<{ message: string }> {
  return apiFetch(`/api-keys/${keyId}`, {
    method: 'DELETE',
  });
}

export async function fetchApiKeyUsage(keyId: string): Promise<ApiKeyUsageStats> {
  return apiFetch(`/api-keys/${keyId}/usage`);
}

// ---------------------------------------------------------------------------
// POS System Integrations
// ---------------------------------------------------------------------------

export type PosSystemType = 'square' | 'lightspeed' | 'clover' | 'shopify';
export type SyncFrequency = 'manual' | 'daily' | 'weekly' | 'monthly';
export type ConnectionStatus = 'connected' | 'disconnected' | 'syncing' | 'error';

export interface PosSystemInfo {
  system: PosSystemType;
  display_name: string;
  auth_type: string;
  inventory_fields: string[];
  setup_steps: string[];
  docs_url: string;
}

export interface PosConnection {
  connection_id: string;
  user_id: string;
  pos_system: PosSystemType;
  pos_system_display: string;
  store_name: string;
  status: ConnectionStatus;
  sync_frequency: SyncFrequency;
  location_id: string | null;
  last_sync_at: string | null;
  last_sync_status: string | null;
  last_sync_rows: number;
  created_at: string;
}

export interface SyncResult {
  connection_id: string;
  success: boolean;
  rows_synced: number;
  errors: string[];
  duration_seconds: number;
  analysis_triggered: boolean;
  analysis_id: string | null;
}

export interface PosConnectionRequest {
  pos_system: PosSystemType;
  store_name: string;
  sync_frequency?: SyncFrequency;
  location_id?: string;
}

export async function fetchSupportedPosSystems(): Promise<{ systems: PosSystemInfo[] }> {
  return apiFetch('/pos/systems');
}

export async function createPosConnection(req: PosConnectionRequest): Promise<PosConnection> {
  return apiFetch('/pos/connections', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

export async function listPosConnections(): Promise<{ connections: PosConnection[]; total: number }> {
  return apiFetch('/pos/connections');
}

export async function triggerPosSync(connectionId: string): Promise<SyncResult> {
  return apiFetch(`/pos/connections/${connectionId}/sync`, {
    method: 'POST',
  });
}

export async function disconnectPos(connectionId: string): Promise<{ message: string }> {
  return apiFetch(`/pos/connections/${connectionId}/disconnect`, {
    method: 'POST',
  });
}

export async function deletePosConnection(connectionId: string): Promise<{ message: string }> {
  return apiFetch(`/pos/connections/${connectionId}`, {
    method: 'DELETE',
  });
}

// ---------------------------------------------------------------------------
// Findings (Engine 2 enriched)
// ---------------------------------------------------------------------------

export type Engine2Status = 'active' | 'warming_up' | 'not_initialized' | 'error';

export interface Finding {
  id: string;
  type: string;
  title: string;
  description: string;
  severity: string;
  dollar_impact: number;
  department: string | null;
  recommended_action: string | null;
  acknowledged: boolean;
  sku: string | null;
  engine2_observations?: number;
  prediction?: Record<string, unknown>;
}

export interface FindingsResponse {
  findings: Finding[];
  pagination: {
    page: number;
    page_size: number;
    total: number;
    total_pages: number;
  };
  engine2_status: Engine2Status;
}

export async function fetchFindings(params?: {
  page?: number;
  page_size?: number;
  status?: 'active' | 'acknowledged' | 'all';
  sort_by?: 'dollar_impact' | 'priority' | 'date';
  department?: string;
}): Promise<FindingsResponse> {
  const qs = new URLSearchParams();
  if (params?.page) qs.set('page', String(params.page));
  if (params?.page_size) qs.set('page_size', String(params.page_size));
  if (params?.status) qs.set('status', params.status);
  if (params?.sort_by) qs.set('sort_by', params.sort_by);
  if (params?.department) qs.set('department', params.department);
  const q = qs.toString();
  return apiFetch(`/findings${q ? `?${q}` : ''}`);
}

export async function acknowledgeFinding(findingId: string): Promise<{ id: string; acknowledged: boolean }> {
  return apiFetch(`/findings/${findingId}/acknowledge`, { method: 'POST' });
}

export async function restoreFinding(findingId: string): Promise<{ id: string; acknowledged: boolean }> {
  return apiFetch(`/findings/${findingId}/restore`, { method: 'POST' });
}

// ---------------------------------------------------------------------------
// Transfer Recommendations
// ---------------------------------------------------------------------------

export interface TransferRecommendation {
  source_store: string;
  source_sku: string;
  source_description: string;
  units_to_transfer: number;
  dest_store: string;
  dest_sku: string;
  dest_description: string;
  match_level: 'exact_sku' | 'subcategory' | 'category';
  match_confidence: number;
  clearance_recovery: number;
  transfer_recovery: number;
  net_benefit: number;
  demand_pattern: string;
  estimated_weeks_to_sell: number;
}

export interface TransfersResponse {
  recommendations: TransferRecommendation[];
  total: number;
  stores_registered?: number;
  engine2_status: Engine2Status;
  message?: string;
}

export async function fetchTransfers(params?: {
  source_store?: string;
  min_benefit?: number;
  max_results?: number;
}): Promise<TransfersResponse> {
  const qs = new URLSearchParams();
  if (params?.source_store) qs.set('source_store', params.source_store);
  if (params?.min_benefit) qs.set('min_benefit', String(params.min_benefit));
  if (params?.max_results) qs.set('max_results', String(params.max_results));
  const q = qs.toString();
  return apiFetch(`/transfers${q ? `?${q}` : ''}`);
}

// ---------------------------------------------------------------------------
// Dashboard Summary (Engine 2 enhanced)
// ---------------------------------------------------------------------------

export interface DashboardTopFinding {
  id: string;
  type: string;
  title: string;
  severity: string;
  dollar_impact: number;
  department: string | null;
}

export interface DashboardSummaryResponse {
  recovery_total: number;
  finding_count: number;
  department_count: number;
  department_status: Record<string, {
    status: 'green' | 'yellow' | 'red';
    finding_count: number;
    total_impact: number;
  }>;
  top_findings: DashboardTopFinding[];
  prediction_count: number;
  top_predictions: Record<string, unknown>[];
  engine2_status: Engine2Status;
  engine2_summary: Record<string, unknown>;
  transfer_stats: {
    stores_registered: number;
    total_recommendations: number;
  };
}

export async function fetchDashboardSummary(): Promise<DashboardSummaryResponse> {
  return apiFetch('/dashboard');
}
