/**
 * S3 Upload Orchestration Module
 *
 * Handles the full upload pipeline:
 * 1. Get presigned URL from backend
 * 2. Upload file directly to S3
 * 3. Get column mapping suggestions
 * 4. Run analysis
 */

export interface PresignResult {
  key: string;
  url: string;
  filename: string;
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
}

/** Step 1: Get presigned S3 URL */
export async function presignUpload(filename: string): Promise<PresignResult> {
  const formData = new FormData();
  formData.append("filenames", filename);

  const res = await fetch("/api/uploads/presign", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || "Failed to get upload URL");
  }

  const data = await res.json();
  return data.presigned_urls[0];
}

/** Step 2: Upload file to S3 via presigned URL */
export async function uploadToS3(url: string, file: File): Promise<void> {
  const res = await fetch(url, {
    method: "PUT",
    body: file,
    headers: { "Content-Type": "application/octet-stream" },
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

  const res = await fetch("/api/uploads/suggest-mapping", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || "Failed to analyze file structure");
  }

  return res.json();
}

/** Step 4: Run analysis */
export async function runAnalysis(
  key: string,
  mapping: Record<string, string>
): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append("key", key);
  formData.append("mapping", JSON.stringify(mapping));

  const res = await fetch("/api/analysis/analyze", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
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

/** Helper: Validate file size (max 50MB) */
export function isValidFileSize(file: File): boolean {
  const maxSize = 50 * 1024 * 1024; // 50MB
  return file.size <= maxSize;
}

/** Helper: Format file size for display */
export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
