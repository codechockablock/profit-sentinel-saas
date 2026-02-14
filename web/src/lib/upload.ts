/**
 * S3 Upload Orchestration Module
 *
 * Handles the full upload pipeline:
 * 1. Get presigned URL from backend
 * 2. Upload file directly to S3
 * 3. Get column mapping suggestions
 * 4. Run analysis
 *
 * Supports dual auth:
 *   - Anonymous users: no auth header, 10MB limit, 5 analyses/hour
 *   - Authenticated users: Bearer token, 50MB limit, 100 analyses/hour
 */

import { getApiUrl } from './api-config'
import { getAuthHeaders } from './supabase'

export interface PresignResult {
  key: string;
  url: string;
  filename: string;
  fields: Record<string, string>;
  upload_method: string;
}

export interface MappingResult {
  original_columns: string[];
  sample_data: Record<string, string>[];
  suggestions: Record<string, string>;
  confidences: Record<string, number>;
  importance: Record<string, number>;
  critical_missing: string[];
  notes: string;
}

export interface ItemDetail {
  sku: string;
  score: number;
  description: string;
  quantity: number;
  cost: number;
  revenue: number;
  sold: number;
  margin: number;
  sub_total: number;
  context: string;
}

export interface LeakData {
  top_items: string[];
  scores: number[];
  item_details: ItemDetail[];
  count: number;
  severity: string;
  category: string;
  recommendations: string[];
  title: string;
  icon: string;
  color: string;
  priority: number;
}

export interface EstimatedImpact {
  currency: string;
  low_estimate: number;
  high_estimate: number;
  breakdown: Record<string, number>;
  negative_inventory_alert?: {
    items_found: number;
    potential_untracked_cogs: number;
    raw_data_anomaly_value: number | null;
    is_anomalous: boolean;
    threshold_exceeded: boolean;
    requires_audit: boolean;
    excluded_from_annual_estimate: boolean;
    note: string | null;
  };
}

export interface CauseDiagnosis {
  top_cause: string;
  confidence: number;
  hypotheses?: Array<{
    cause: string;
    probability: number;
    evidence: string[];
  }>;
}

export interface UpgradePrompt {
  message: string;
  cta: string;
  url: string;
}

export interface AnalysisResult {
  leaks: Record<string, LeakData>;
  summary: {
    total_rows_analyzed: number;
    total_items_flagged: number;
    critical_issues: number;
    high_issues: number;
    estimated_impact: EstimatedImpact;
    analysis_time_seconds: number;
  };
  cause_diagnosis?: CauseDiagnosis;
  warnings?: string[];
  status: string;
  is_authenticated?: boolean;
  upgrade_prompt?: UpgradePrompt;
}

export interface PresignResponse {
  presigned_urls: Array<{
    filename: string;
    safe_filename: string;
    key: string;
    url: string;
    fields: Record<string, string>;
    max_size_mb: number;
  }>;
  upload_method: string;
  limits: {
    max_file_size_mb: number;
    allowed_extensions: string[];
  };
}

/** Get API base URL - use direct API for long-running requests */
function getApiBaseUrl(): string {
  return getApiUrl();
}

/** Get headers for API requests (includes auth if logged in) */
async function getHeaders(): Promise<HeadersInit> {
  return await getAuthHeaders();
}

/** Step 1: Get presigned S3 URL */
export async function presignUpload(
  filename: string,
  turnstileToken: string = "",
  storeId: string = "",
): Promise<PresignResult> {
  const formData = new FormData();
  formData.append("filenames", filename);
  if (storeId) {
    formData.append("store_id", storeId);
  }
  if (turnstileToken) {
    formData.append("cf_turnstile_response", turnstileToken);
  }

  const authHeaders = await getHeaders();
  const apiBase = getApiBaseUrl();
  const endpoint = `${apiBase}/uploads/presign`;

  const res = await fetch(endpoint, {
    method: "POST",
    headers: authHeaders,
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || "Failed to get upload URL");
  }

  const data: PresignResponse = await res.json();
  const first = data.presigned_urls[0];
  if (!first) {
    throw new Error("Upload service returned no presigned upload target");
  }

  if (!first.fields || Object.keys(first.fields).length === 0) {
    throw new Error("Upload service returned an invalid presigned POST payload");
  }

  return {
    ...first,
    upload_method: data.upload_method || "POST",
  };
}

/** Step 1b: Get file size limit for current user.
 *
 * Determines the limit client-side based on auth state to avoid
 * hitting POST /uploads/presign which gets rate-limited (429).
 */
export async function getFileSizeLimit(): Promise<number> {
  try {
    const { getSupabase } = await import('./supabase');
    const supabase = getSupabase();
    if (supabase) {
      const { data: { session } } = await supabase.auth.getSession();
      if (session) {
        return 50; // Authenticated: 50MB
      }
    }
  } catch {
    // Fall through to default
  }
  return 10; // Guest: 10MB
}

/** Step 2: Upload file to S3 via presigned URL */
export async function uploadToS3(presign: PresignResult, file: File): Promise<void> {
  const method = (presign.upload_method || "POST").toUpperCase();
  if (method !== "POST") {
    throw new Error(`Unsupported upload method '${method}'. Expected POST.`);
  }

  const formData = new FormData();

  // S3 presigned POST requires all policy fields first.
  for (const [key, value] of Object.entries(presign.fields || {})) {
    formData.append(key, value);
  }

  // File must be appended last for S3 POST form processing.
  formData.append("file", file);

  const res = await fetch(presign.url, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error("Failed to upload file to storage");
  }
}

/** Step 3: Get column mapping suggestions */
export async function suggestMapping(
  key: string,
  filename: string
): Promise<MappingResult> {
  const formData = new FormData();
  formData.append("key", key);
  formData.append("filename", filename);

  const authHeaders = await getHeaders();
  const apiBase = getApiBaseUrl();
  const endpoint = `${apiBase}/uploads/suggest-mapping`;

  const res = await fetch(endpoint, {
    method: "POST",
    headers: authHeaders,
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || "Failed to analyze file structure");
  }

  return res.json();
}

/** Step 4: Run analysis - uses direct API to avoid proxy timeouts */
export async function runAnalysis(
  key: string,
  mapping: Record<string, string>,
  storeId: string = "",
): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append("key", key);
  formData.append("mapping", JSON.stringify(mapping));
  if (storeId) {
    formData.append("store_id", storeId);
  }

  const authHeaders = await getHeaders();
  const apiBase = getApiBaseUrl();
  const endpoint = `${apiBase}/analysis/analyze`;
  const res = await fetch(endpoint, {
    method: "POST",
    headers: authHeaders,
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    if (res.status === 429) {
      throw new Error(errorData.detail || "Rate limit exceeded. Sign up for higher limits.");
    }
    throw new Error(errorData.detail || "Analysis failed");
  }

  return res.json();
}

/** Helper: Validate file type */
export function isValidFileType(file: File): boolean {
  const validExtensions = [".csv", ".xls", ".xlsx"];
  const fileName = file.name.toLowerCase();
  return validExtensions.some((ext) => fileName.endsWith(ext));
}

/** Helper: Validate file size (dynamic based on auth state) */
export function isValidFileSize(file: File, maxSizeMb: number = 10): boolean {
  const maxSize = maxSizeMb * 1024 * 1024;
  return file.size <= maxSize;
}

/** Step 5: Send report to email (guest flow) */
export async function sendReport(
  email: string,
  analysisResult: AnalysisResult
): Promise<{ success: boolean; message: string }> {
  const authHeaders = await getHeaders();
  const apiBase = getApiBaseUrl();
  const endpoint = `${apiBase}/analysis/send-report`;

  const res = await fetch(endpoint, {
    method: "POST",
    headers: {
      ...authHeaders,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      email,
      analysis_result: analysisResult,
    }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    if (res.status === 429) {
      throw new Error("Rate limit exceeded. Please try again later.");
    }
    throw new Error(errorData.detail || "Failed to send report");
  }

  return res.json();
}

/** Helper: Format file size for display */
export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
