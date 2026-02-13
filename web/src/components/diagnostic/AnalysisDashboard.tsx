"use client";

import React, { useState, useCallback, useRef, useMemo, useEffect, ChangeEvent, DragEvent } from "react";
import {
  Upload,
  AlertTriangle,
  AlertCircle,
  CheckCircle,
  ChevronDown,
  ChevronRight,
  Zap,
  FileSpreadsheet,
  Loader2,
  RefreshCw,
  Mail,
  TrendingDown,
  Package,
  Tag,
  ShieldAlert,
  Boxes,
  LogIn,
  LogOut,
  UserPlus,
} from "lucide-react";
import { Turnstile } from '@marsidev/react-turnstile';
import {
  presignUpload,
  uploadToS3,
  suggestMapping,
  runAnalysis,
  sendReport,
  isValidFileType,
  isValidFileSize,
  getFileSizeLimit,
  type MappingResult,
  type AnalysisResult,
  type LeakData,
} from "@/lib/upload";
import { LEAK_METADATA, getSeverityBadge, scoreToRiskLabel, formatDollarImpact } from "@/lib/leak-metadata";
import { buildAttributions, type ColumnAttribution } from "@/lib/column-attribution";
import { AttributionTooltip } from "@/components/ui/AttributionTooltip";
import Link from "next/link";
import { getSupabase } from "@/lib/supabase";
import { robustSignOut } from "@/lib/auth-helpers";
import { saveAnalysisSynopsis } from "@/lib/api";
import { AuthModal } from "@/components/auth/AuthModal";
import { UpgradePrompt } from "@/components/UpgradePrompt";

// Types
type Stage = "upload" | "mapping" | "processing" | "results";

// Icon mapping for leak types
const LEAK_ICONS: Record<string, React.ReactNode> = {
  "alert-triangle": <AlertTriangle size={20} />,
  "alert-circle": <AlertCircle size={20} />,
  "trending-down": <TrendingDown size={20} />,
  "package-x": <Package size={20} />,
  "boxes": <Boxes size={20} />,
  "tag": <Tag size={20} />,
  "shield-alert": <ShieldAlert size={20} />,
};

// Standard field options for mapping
const STANDARD_FIELDS = [
  { value: "", label: "-- Skip this column --" },
  { value: "sku", label: "SKU / Item ID" },
  { value: "description", label: "Description" },
  { value: "quantity", label: "Quantity on Hand" },
  { value: "cost", label: "Cost" },
  { value: "revenue", label: "Retail Price" },
  { value: "sold", label: "Sold (Units)" },
  { value: "margin", label: "Profit Margin %" },
  { value: "sub_total", label: "Inventory Value" },
  { value: "vendor", label: "Vendor" },
  { value: "category", label: "Category" },
];

// Critical fields that must be mapped
const CRITICAL_FIELDS = ["sku", "quantity", "cost", "revenue"];

export default function AnalysisDashboard() {
  const [stage, setStage] = useState<Stage>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [s3Key, setS3Key] = useState<string | null>(null);
  const [mappingData, setMappingData] = useState<MappingResult | null>(null);
  const [confirmedMapping, setConfirmedMapping] = useState<Record<string, string>>({});
  const [results, setResults] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [processingMessage, setProcessingMessage] = useState("");
  const [expandedLeaks, setExpandedLeaks] = useState<Set<string>>(new Set());

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);

  // Turnstile captcha state
  const [turnstileToken, setTurnstileToken] = useState<string>("");
  const turnstileRef = useRef<any>(null);

  // Auth state
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [userEmail, setUserEmail] = useState<string | null>(null);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [authModalMode, setAuthModalMode] = useState<"login" | "signup">("signup");
  const [maxFileSizeMb, setMaxFileSizeMb] = useState(10); // Default to anonymous limit

  // Email report state
  const [reportEmail, setReportEmail] = useState("");
  const [reportSending, setReportSending] = useState(false);
  const [reportSent, setReportSent] = useState(false);
  const [reportError, setReportError] = useState<string | null>(null);

  // Save to dashboard state
  const [savingToDashboard, setSavingToDashboard] = useState(false);
  const [savedToDashboard, setSavedToDashboard] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  // Initialize auth state and listen for changes
  useEffect(() => {
    const supabase = getSupabase();
    if (!supabase) return;

    // Check initial auth state
    supabase.auth.getSession().then(({ data: { session } }: { data: { session: { user?: { email?: string } } | null } }) => {
      setIsAuthenticated(!!session);
      setUserEmail(session?.user?.email ?? null);
    });

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event: string, session: { user?: { email?: string } } | null) => {
        setIsAuthenticated(!!session);
        setUserEmail(session?.user?.email ?? null);
      }
    );

    return () => subscription.unsubscribe();
  }, []);

  // Fetch file size limit based on auth state
  useEffect(() => {
    getFileSizeLimit().then(setMaxFileSizeMb);
  }, [isAuthenticated]);

  // Pre-loaded file from dashboard: skip upload and go straight to mapping
  const [fromDashboard, setFromDashboard] = useState(false);
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const paramKey = params.get("s3Key");
    const paramFilename = params.get("filename");
    const paramFrom = params.get("from");

    if (!paramKey || !paramFilename || stage !== "upload") return;

    if (paramFrom === "dashboard") setFromDashboard(true);

    // Clean up URL params so refresh doesn't re-trigger
    window.history.replaceState({}, "", window.location.pathname);

    // Auto-load: call suggestMapping and jump to mapping stage
    (async () => {
      setIsLoading(true);
      setProcessingMessage("Analyzing file structure...");
      try {
        const mapping = await suggestMapping(paramKey, paramFilename);
        setS3Key(paramKey);
        setFile(new File([], paramFilename));
        setMappingData(mapping);
        setConfirmedMapping(mapping.suggestions);
        setStage("mapping");
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to analyze file");
      } finally {
        setIsLoading(false);
        setProcessingMessage("");
      }
    })();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSignOut = async () => {
    await robustSignOut();
    setIsAuthenticated(false);
    setUserEmail(null);
  };

  const openSignUpModal = () => {
    setAuthModalMode("signup");
    setShowAuthModal(true);
  };

  const openSignInModal = () => {
    setAuthModalMode("login");
    setShowAuthModal(true);
  };

  // Column attribution — computed once from mapping, used in result tooltips
  const attributions = useMemo(
    () => buildAttributions(confirmedMapping, mappingData?.original_columns ?? []),
    [confirmedMapping, mappingData]
  );

  // File handling
  const handleDrag = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileSelect = async (selectedFile: File) => {
    setError(null);

    if (!isValidFileType(selectedFile)) {
      setError("Please upload a CSV, XLS, or XLSX file");
      return;
    }

    if (!isValidFileSize(selectedFile, maxFileSizeMb)) {
      setError(`File size must be under ${maxFileSizeMb}MB${!isAuthenticated ? ". Sign up for 50MB uploads." : ""}`);
      return;
    }

    setFile(selectedFile);
    setIsLoading(true);

    // Check if captcha is needed but not ready
    if (!isAuthenticated && process.env.NEXT_PUBLIC_TURNSTILE_SITE_KEY && !turnstileToken) {
      setError("Verifying you're human — please wait a moment and try again.");
      setIsLoading(false);
      return;
    }

    try {
      // Step 1: Get presigned URL
      setProcessingMessage("Preparing upload...");
      const presignResult = await presignUpload(selectedFile.name, turnstileToken);

      // Step 2: Upload to S3
      setProcessingMessage("Uploading file...");
      await uploadToS3(presignResult, selectedFile);
      setS3Key(presignResult.key);

      // Step 3: Get mapping suggestions
      setProcessingMessage("Analyzing file structure...");
      const mapping = await suggestMapping(presignResult.key, selectedFile.name);
      setMappingData(mapping);

      // Initialize confirmed mapping with suggestions
      setConfirmedMapping(mapping.suggestions);

      setStage("mapping");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setIsLoading(false);
      setProcessingMessage("");
      // Reset turnstile for next upload attempt
      if (turnstileRef.current) {
        turnstileRef.current.reset();
        setTurnstileToken("");
      }
    }
  };

  // Mapping handling
  const handleMappingChange = (sourceCol: string, targetField: string) => {
    setConfirmedMapping((prev) => ({
      ...prev,
      [sourceCol]: targetField,
    }));
  };

  const getMissingCriticalFields = () => {
    const mappedTargets = new Set(Object.values(confirmedMapping).filter(Boolean));
    return CRITICAL_FIELDS.filter((field) => !mappedTargets.has(field));
  };

  const canAutoConfirm = () => {
    if (!mappingData) return false;
    const missing = getMissingCriticalFields();
    if (missing.length > 0) return false;

    // Check if all critical fields have high confidence
    for (const [source, target] of Object.entries(mappingData.suggestions)) {
      if (CRITICAL_FIELDS.includes(target)) {
        const confidence = mappingData.confidences[source] || 0;
        if (confidence < 0.8) return false;
      }
    }
    return true;
  };

  const handleRunAnalysis = async () => {
    if (!s3Key) return;

    setStage("processing");
    setIsLoading(true);

    try {
      setProcessingMessage("Scanning for threats...");
      await new Promise((r) => setTimeout(r, 500));

      setProcessingMessage("Analyzing 11 leak types...");
      await new Promise((r) => setTimeout(r, 500));

      setProcessingMessage("Calculating impact...");
      const analysisResults = await runAnalysis(s3Key, confirmedMapping);
      setResults(analysisResults);

      // Auto-expand leaks with items
      const expandSet = new Set<string>();
      Object.entries(analysisResults.leaks).forEach(([key, data]) => {
        if (data.count > 0) expandSet.add(key);
      });
      setExpandedLeaks(expandSet);

      setStage("results");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
      setStage("mapping");
    } finally {
      setIsLoading(false);
      setProcessingMessage("");
    }
  };

  const toggleLeakExpanded = (leakKey: string) => {
    setExpandedLeaks((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(leakKey)) {
        newSet.delete(leakKey);
      } else {
        newSet.add(leakKey);
      }
      return newSet;
    });
  };

  const handleSaveToDashboard = async () => {
    if (!results || !file) return;
    setSavingToDashboard(true);
    setSaveError(null);

    try {
      // Build detection counts from leaks
      const detectionCounts: Record<string, number> = {};
      for (const [key, data] of Object.entries(results.leaks)) {
        detectionCounts[key] = data.count;
      }

      // Compute a simple hash from the file name + size
      const fileHash = `${file.name}-${file.size}-${Date.now()}`;

      const res = await saveAnalysisSynopsis({
        file_hash: fileHash,
        file_row_count: results.summary.total_rows_analyzed,
        detection_counts: detectionCounts,
        total_impact_estimate_low: results.summary.estimated_impact.low_estimate,
        total_impact_estimate_high: results.summary.estimated_impact.high_estimate,
        processing_time_seconds: results.summary.analysis_time_seconds,
        engine_version: "sidecar",
      });

      if (res.success) {
        setSavedToDashboard(true);
      } else {
        setSaveError(res.error || "Failed to save analysis");
      }
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : "Failed to save analysis");
    } finally {
      setSavingToDashboard(false);
    }
  };

  const resetDashboard = () => {
    setStage("upload");
    setFile(null);
    setS3Key(null);
    setMappingData(null);
    setConfirmedMapping({});
    setResults(null);
    setError(null);
    setExpandedLeaks(new Set());
    setReportEmail("");
    setReportSending(false);
    setReportSent(false);
    setReportError(null);
    setSavingToDashboard(false);
    setSavedToDashboard(false);
    setSaveError(null);
    // Reset turnstile
    if (turnstileRef.current) {
      turnstileRef.current.reset();
      setTurnstileToken("");
    }
  };

  const handleSendReport = async () => {
    if (!results || !reportEmail.trim()) return;

    const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailPattern.test(reportEmail.trim())) {
      setReportError("Please enter a valid email address");
      return;
    }

    setReportSending(true);
    setReportError(null);

    try {
      await sendReport(reportEmail.trim(), results);
      setReportSent(true);
    } catch (err) {
      setReportError(err instanceof Error ? err.message : "Failed to send report");
    } finally {
      setReportSending(false);
    }
  };

  // Render upload stage
  if (stage === "upload") {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-900 to-slate-800 text-white">
        {/* Auth header bar */}
        <div className="max-w-xl mx-auto px-4 pt-4 flex justify-end items-center gap-3">
          {isAuthenticated ? (
            <>
              <span className="text-sm text-slate-400">{userEmail}</span>
              <button
                onClick={handleSignOut}
                className="flex items-center gap-1.5 text-sm text-slate-400 hover:text-white transition-colors"
              >
                <LogOut size={14} /> Sign out
              </button>
            </>
          ) : (
            <>
              <button
                onClick={openSignInModal}
                className="flex items-center gap-1.5 text-sm text-slate-400 hover:text-white transition-colors"
              >
                <LogIn size={14} /> Sign in
              </button>
              <button
                onClick={openSignUpModal}
                className="flex items-center gap-1.5 text-sm px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg transition-colors"
              >
                <UserPlus size={14} /> Sign up
              </button>
            </>
          )}
        </div>

        <div className="max-w-xl mx-auto px-4 py-16">
          <div className="text-center mb-10">
            <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-emerald-500 to-emerald-600 flex items-center justify-center shadow-lg shadow-emerald-500/30">
              <Zap size={32} className="text-white" />
            </div>
            <h1 className="text-3xl font-bold mb-2">Profit Sentinel</h1>
            <p className="text-slate-400">Full 11-Type Profit Leak Analysis</p>
          </div>

          {error && (
            <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 text-sm">
              {error}
            </div>
          )}

          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all ${
              dragActive
                ? "border-emerald-500 bg-emerald-500/5"
                : "border-slate-700 hover:border-slate-600 bg-slate-800/50"
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,.xls,.xlsx"
              onChange={(e: ChangeEvent<HTMLInputElement>) =>
                e.target.files?.[0] && handleFileSelect(e.target.files[0])
              }
              hidden
            />

            {isLoading ? (
              <>
                <Loader2 className="w-12 h-12 mx-auto mb-4 text-emerald-500 animate-spin" />
                <p className="text-lg font-medium mb-2">{processingMessage || "Processing..."}</p>
                <p className="text-slate-500 text-sm">Please wait</p>
              </>
            ) : (
              <>
                <Upload className="w-12 h-12 mx-auto mb-4 text-slate-500" />
                <p className="text-lg font-medium mb-2">Drop your inventory file here</p>
                <p className="text-slate-500 text-sm mb-4">or click to browse</p>
                <div className="flex items-center justify-center gap-2 text-xs text-slate-600">
                  <FileSpreadsheet size={14} />
                  <span>CSV, XLS, XLSX supported (max {maxFileSizeMb}MB{!isAuthenticated ? " \u2014 sign up for 50MB" : ""})</span>
                </div>
              </>
            )}
          </div>

          {/* Turnstile captcha — only for guest users, below upload area */}
          {!isAuthenticated && process.env.NEXT_PUBLIC_TURNSTILE_SITE_KEY && (
            <div className="mt-3 flex justify-center" onClick={(e) => e.stopPropagation()}>
              <Turnstile
                ref={turnstileRef}
                siteKey={process.env.NEXT_PUBLIC_TURNSTILE_SITE_KEY}
                onSuccess={(token: string) => setTurnstileToken(token)}
                onError={() => setTurnstileToken("")}
                onExpire={() => {
                  setTurnstileToken("");
                  turnstileRef.current?.reset();
                }}
                options={{
                  theme: "dark",
                  size: "compact",
                }}
              />
            </div>
          )}

          {isAuthenticated && (
            <div className="mt-6 text-center text-sm text-emerald-400/80">
              Results will be saved to your dashboard
            </div>
          )}

          <div className="mt-8 grid grid-cols-2 gap-4 text-sm text-slate-500">
            <div className="flex items-center gap-2">
              <CheckCircle size={16} className="text-emerald-500" />
              <span>Detects 11 leak types</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle size={16} className="text-emerald-500" />
              <span>Encrypted & S3 secured</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle size={16} className="text-emerald-500" />
              <span>Auto-deletes after analysis</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle size={16} className="text-emerald-500" />
              <span>Any POS system supported</span>
            </div>
          </div>
        </div>

        {/* Auth Modal */}
        <AuthModal
          isOpen={showAuthModal}
          onClose={() => setShowAuthModal(false)}
          onSuccess={() => getFileSizeLimit().then(setMaxFileSizeMb)}
          defaultMode={authModalMode}
        />
      </div>
    );
  }

  // Render mapping stage
  if (stage === "mapping" && mappingData) {
    const missingCritical = getMissingCriticalFields();

    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-900 to-slate-800 text-white">
        <header className="border-b border-slate-800 px-6 py-4">
          <div className="max-w-4xl mx-auto flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-emerald-500 to-emerald-600 flex items-center justify-center">
                <Zap size={18} className="text-white" />
              </div>
              <span className="font-semibold">Profit Sentinel</span>
            </div>
            <button
              onClick={resetDashboard}
              className="text-sm text-slate-400 hover:text-white flex items-center gap-2"
            >
              <RefreshCw size={16} />
              Start Over
            </button>
          </div>
        </header>

        <main className="max-w-4xl mx-auto px-6 py-8">
          <div className="mb-8">
            <h1 className="text-2xl font-bold mb-2">Review Column Mapping</h1>
            <p className="text-slate-400">
              We detected {mappingData.original_columns.length} columns. Verify the mappings below.
            </p>
          </div>

          {error && (
            <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 text-sm">
              {error}
            </div>
          )}

          {missingCritical.length > 0 && (
            <div className="mb-6 p-4 bg-amber-500/10 border border-amber-500/30 rounded-xl">
              <div className="flex items-start gap-3">
                <AlertTriangle className="text-amber-500 flex-shrink-0 mt-0.5" size={20} />
                <div>
                  <p className="text-amber-400 font-medium">Missing critical fields</p>
                  <p className="text-sm text-amber-400/80 mt-1">
                    Please map these required columns: {missingCritical.join(", ")}
                  </p>
                </div>
              </div>
            </div>
          )}

          {mappingData.notes && (
            <div className="mb-6 p-4 bg-slate-800 border border-slate-700 rounded-xl text-sm text-slate-400">
              {mappingData.notes}
            </div>
          )}

          <div className="bg-slate-800/50 border border-slate-700 rounded-xl overflow-hidden">
            <div className="grid grid-cols-12 gap-4 px-4 py-3 border-b border-slate-700 text-sm font-medium text-slate-400">
              <div className="col-span-4">Source Column</div>
              <div className="col-span-3">Sample Data</div>
              <div className="col-span-4">Maps To</div>
              <div className="col-span-1 text-right">Conf.</div>
            </div>

            <div className="divide-y divide-slate-700/50">
              {mappingData.original_columns.map((col) => {
                const suggestion = confirmedMapping[col] || "";
                const confidence = mappingData.confidences[col] || 0;
                const importance = mappingData.importance[col] || 0;
                const sampleValue = mappingData.sample_data[0]?.[col] || "";
                const isCritical = CRITICAL_FIELDS.includes(suggestion);

                return (
                  <div
                    key={col}
                    className={`grid grid-cols-12 gap-4 px-4 py-3 items-center ${
                      isCritical ? "bg-emerald-500/5" : ""
                    }`}
                  >
                    <div className="col-span-4">
                      <span className="font-mono text-sm">{col}</span>
                      {importance >= 0.8 && (
                        <span className="ml-2 px-1.5 py-0.5 text-xs bg-red-500/20 text-red-400 rounded">
                          Required
                        </span>
                      )}
                    </div>
                    <div className="col-span-3">
                      <span className="text-slate-500 text-sm truncate block">
                        {String(sampleValue).slice(0, 30) || "-"}
                      </span>
                    </div>
                    <div className="col-span-4">
                      <select
                        value={suggestion}
                        onChange={(e) => handleMappingChange(col, e.target.value)}
                        className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500"
                      >
                        {STANDARD_FIELDS.map((field) => (
                          <option key={field.value} value={field.value}>
                            {field.label}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div className="col-span-1 text-right">
                      <span
                        className={`text-sm ${
                          confidence >= 0.8
                            ? "text-emerald-400"
                            : confidence >= 0.5
                            ? "text-amber-400"
                            : "text-slate-500"
                        }`}
                      >
                        {confidence > 0 ? `${Math.round(confidence * 100)}%` : "-"}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="mt-8 flex items-center justify-between">
            <button
              onClick={resetDashboard}
              className="px-6 py-3 text-slate-400 hover:text-white transition"
            >
              Cancel
            </button>
            <div className="flex gap-4">
              {canAutoConfirm() && (
                <button
                  onClick={handleRunAnalysis}
                  className="px-6 py-3 bg-slate-700 text-white rounded-xl hover:bg-slate-600 transition font-medium"
                >
                  Auto-confirm & Analyze
                </button>
              )}
              <button
                onClick={handleRunAnalysis}
                disabled={missingCritical.length > 0}
                className={`px-8 py-3 rounded-xl font-bold transition ${
                  missingCritical.length > 0
                    ? "bg-slate-700 text-slate-500 cursor-not-allowed"
                    : "bg-gradient-to-r from-emerald-500 to-emerald-600 text-white hover:from-emerald-600 hover:to-emerald-700"
                }`}
              >
                Run Analysis
              </button>
            </div>
          </div>
        </main>
      </div>
    );
  }

  // Render processing stage
  if (stage === "processing") {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-900 to-slate-800 text-white flex items-center justify-center">
        <div className="text-center">
          <div className="w-20 h-20 mx-auto mb-8 rounded-2xl bg-gradient-to-br from-emerald-500 to-emerald-600 flex items-center justify-center animate-pulse">
            <Zap size={40} className="text-white" />
          </div>
          <Loader2 className="w-8 h-8 mx-auto mb-4 text-emerald-500 animate-spin" />
          <p className="text-xl font-medium mb-2">{processingMessage || "Processing..."}</p>
          <p className="text-slate-500">Analyzing {file?.name}</p>
        </div>
      </div>
    );
  }

  // Render results stage
  if (stage === "results" && results) {
    const { leaks, summary, cause_diagnosis, warnings } = results;

    // Sort leaks by priority (lower is more important)
    const sortedLeaks = Object.entries(leaks).sort(
      ([, a], [, b]) => (a.priority || 99) - (b.priority || 99)
    );

    const leaksWithItems = sortedLeaks.filter(([, data]) => data.count > 0);
    const leaksEmpty = sortedLeaks.filter(([, data]) => data.count === 0);

    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-900 to-slate-800 text-white">
        <header className="border-b border-slate-800 px-6 py-4">
          <div className="max-w-6xl mx-auto flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-emerald-500 to-emerald-600 flex items-center justify-center">
                <Zap size={18} className="text-white" />
              </div>
              <span className="font-semibold">Profit Sentinel</span>
            </div>
            <div className="flex items-center gap-3">
              {fromDashboard && (
                <Link
                  href="/dashboard"
                  className="px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm hover:bg-slate-700 transition"
                >
                  Back to Dashboard
                </Link>
              )}
              <button
                onClick={resetDashboard}
                className="px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm hover:bg-slate-700 transition flex items-center gap-2"
              >
                <RefreshCw size={16} />
                Analyze Another File
              </button>
            </div>
          </div>
        </header>

        <main className="max-w-6xl mx-auto px-6 py-8">
          {/* Summary Bar */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
              <p className="text-slate-400 text-sm mb-1">SKUs Analyzed</p>
              <p className="text-2xl font-bold">{summary.total_rows_analyzed.toLocaleString()}</p>
            </div>
            <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
              <p className="text-slate-400 text-sm mb-1">Issues Found</p>
              <p className="text-2xl font-bold text-amber-400">
                {summary.total_items_flagged.toLocaleString()}
              </p>
            </div>
            <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
              <p className="text-slate-400 text-sm mb-1">Leak Types Detected</p>
              <p className="text-2xl font-bold">{leaksWithItems.length}</p>
            </div>
            <div className={`border rounded-xl p-4 ${
              summary.analysis_time_seconds < 3
                ? "bg-emerald-500/10 border-emerald-500/30"
                : "bg-slate-800/50 border-slate-700"
            }`}>
              <p className={`text-sm mb-1 ${
                summary.analysis_time_seconds < 3 ? "text-emerald-400/80" : "text-slate-400"
              }`}>
                {summary.analysis_time_seconds < 1
                  ? "Sub-second Analysis"
                  : summary.analysis_time_seconds < 3
                    ? "Rust-powered Analysis"
                    : "Analysis Time"}
              </p>
              <p className={`text-2xl font-bold flex items-center gap-2 ${
                summary.analysis_time_seconds < 3 ? "text-emerald-400" : ""
              }`}>
                <Zap size={20} className={
                  summary.analysis_time_seconds < 3 ? "text-emerald-400" : "text-slate-500"
                } />
                {summary.analysis_time_seconds.toFixed(1)}s
              </p>
            </div>
          </div>

          {/* Estimated Impact */}
          <div className="bg-gradient-to-r from-emerald-500/10 to-emerald-600/5 border border-emerald-500/30 rounded-xl p-6 mb-8">
            <h2 className="text-lg font-semibold mb-2">Estimated Annual Impact</h2>
            <p className="text-3xl font-bold text-emerald-400">
              {formatDollarImpact(summary.estimated_impact.low_estimate)} &mdash;{" "}
              {formatDollarImpact(summary.estimated_impact.high_estimate)}
            </p>
            <p className="text-slate-400 text-sm mt-2">
              Addressing these leaks could recover this amount annually
            </p>
            {/* Impact Breakdown by Leak Type */}
            {summary.estimated_impact.breakdown && Object.values(summary.estimated_impact.breakdown).some(v => v > 0) && (
              <details className="mt-4 group">
                <summary className="cursor-pointer text-sm text-emerald-400/80 hover:text-emerald-400 transition flex items-center gap-2 list-none [&::-webkit-details-marker]:hidden">
                  <ChevronRight className="w-4 h-4 group-open:rotate-90 transition-transform" />
                  View breakdown by leak type
                </summary>
                <div className="mt-3 space-y-2">
                  {Object.entries(summary.estimated_impact.breakdown)
                    .filter(([, v]) => v > 0)
                    .sort(([, a], [, b]) => b - a)
                    .map(([key, amount]) => {
                      const maxVal = Math.max(...Object.values(summary.estimated_impact.breakdown).filter(v => v > 0));
                      return (
                        <div key={key} className="flex items-center gap-3">
                          <span className="text-sm text-slate-400 min-w-[140px] truncate">
                            {LEAK_METADATA[key]?.title || key.replace(/_/g, " ")}
                          </span>
                          <div className="flex-1 bg-slate-700/50 rounded-full h-2">
                            <div
                              className="bg-emerald-500/50 h-2 rounded-full transition-all"
                              style={{ width: `${(amount / maxVal) * 100}%` }}
                            />
                          </div>
                          <span className="text-sm text-emerald-300 min-w-[70px] text-right font-medium">
                            {formatDollarImpact(amount)}
                          </span>
                        </div>
                      );
                    })}
                </div>
              </details>
            )}
          </div>

          {/* Save to Dashboard / Guest CTA */}
          {isAuthenticated ? (
            <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-5 mb-8">
              {savedToDashboard ? (
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <CheckCircle className="text-emerald-400" size={20} />
                    <span className="text-emerald-300 font-medium">Analysis saved to your dashboard</span>
                  </div>
                  <div className="flex items-center gap-3">
                    {fromDashboard && (
                      <Link
                        href="/dashboard"
                        className="px-4 py-2 text-sm border border-slate-600 hover:bg-slate-700 text-slate-300 rounded-lg transition-colors"
                      >
                        Back to Dashboard
                      </Link>
                    )}
                    <Link
                      href="/dashboard/history"
                      className="px-4 py-2 text-sm bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg transition-colors"
                    >
                      View History
                    </Link>
                  </div>
                </div>
              ) : (
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-white font-medium">Save this analysis to your dashboard</p>
                    <p className="text-sm text-slate-400 mt-1">Track your inventory health over time and compare reports</p>
                  </div>
                  <div className="flex items-center gap-3">
                    {saveError && <span className="text-sm text-red-400">{saveError}</span>}
                    <button
                      onClick={handleSaveToDashboard}
                      disabled={savingToDashboard}
                      className="px-5 py-2.5 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 disabled:text-slate-500 text-white font-medium rounded-lg transition-colors flex items-center gap-2"
                    >
                      {savingToDashboard ? (
                        <><Loader2 size={16} className="animate-spin" /> Saving...</>
                      ) : (
                        "Save to Dashboard"
                      )}
                    </button>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="bg-gradient-to-r from-violet-500/10 to-emerald-500/10 border border-violet-500/30 rounded-xl p-5 mb-8">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-white font-medium">Want to keep this analysis?</p>
                  <p className="text-sm text-slate-400 mt-1">Create a free account to save analyses, track trends, and unlock full features</p>
                </div>
                <button
                  onClick={openSignUpModal}
                  className="px-5 py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white font-medium rounded-lg transition-colors whitespace-nowrap"
                >
                  Sign Up to Save
                </button>
              </div>
            </div>
          )}

          {/* Negative Inventory Audit Alert */}
          {summary.estimated_impact.negative_inventory_alert?.requires_audit && (() => {
            const alert = summary.estimated_impact.negative_inventory_alert!;
            return (
              <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-6 mb-8">
                <div className="flex items-start gap-3">
                  <ShieldAlert className="text-red-400 flex-shrink-0 mt-0.5" size={24} />
                  <div>
                    <h3 className="text-lg font-semibold text-red-400">
                      Negative Inventory Audit Required
                    </h3>
                    <p className="text-sm text-slate-300 mt-1">
                      {alert.items_found} item{alert.items_found !== 1 ? "s" : ""} with negative
                      on-hand quantities detected
                    </p>
                    <p className="text-sm text-slate-400 mt-2">
                      Potential untracked COGS:{" "}
                      <span className="text-red-300 font-semibold">
                        {formatDollarImpact(alert.potential_untracked_cogs)}
                      </span>
                    </p>
                    {alert.note && (
                      <p className="text-xs text-slate-500 mt-2 italic">{alert.note}</p>
                    )}
                    <p className="text-xs text-slate-500 mt-3">
                      This amount is excluded from the annual estimate above and requires a
                      separate physical audit.
                    </p>
                  </div>
                </div>
              </div>
            );
          })()}

          {/* Warnings */}
          {warnings && warnings.length > 0 && (
            <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 mb-8">
              <div className="flex items-start gap-3">
                <AlertTriangle className="text-amber-500 flex-shrink-0" size={20} />
                <div>
                  <p className="text-amber-400 font-medium">Analysis Warnings</p>
                  <ul className="text-sm text-amber-400/80 mt-1 list-disc list-inside">
                    {warnings.map((w, i) => (
                      <li key={i}>{w}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}

          {/* Leak Type Cards */}
          <div className="space-y-4 mb-8">
            <h2 className="text-xl font-bold">Detected Leaks ({leaksWithItems.length})</h2>

            {leaksWithItems.length === 0 ? (
              <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-8 text-center">
                <CheckCircle className="w-16 h-16 mx-auto mb-4 text-emerald-500" />
                <h3 className="text-xl font-bold mb-2">No Issues Detected</h3>
                <p className="text-slate-400">
                  Your inventory data looks healthy. No significant profit leaks found.
                </p>
              </div>
            ) : (
              leaksWithItems.map(([leakKey, data]) => (
                <LeakCard
                  key={leakKey}
                  leakKey={leakKey}
                  data={data}
                  expanded={expandedLeaks.has(leakKey)}
                  onToggle={() => toggleLeakExpanded(leakKey)}
                  attributions={attributions}
                  impactAmount={summary.estimated_impact.breakdown[leakKey] || 0}
                />
              ))
            )}
          </div>

          {/* Empty leak types (collapsed) */}
          {leaksEmpty.length > 0 && (
            <div className="mb-8">
              <details className="group">
                <summary className="cursor-pointer text-slate-500 hover:text-slate-400 transition flex items-center gap-2">
                  <ChevronRight className="w-4 h-4 group-open:rotate-90 transition-transform" />
                  <span>{leaksEmpty.length} leak types with no issues found</span>
                </summary>
                <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-2">
                  {leaksEmpty.map(([leakKey, data]) => (
                    <div
                      key={leakKey}
                      className="px-3 py-2 bg-slate-800/30 border border-slate-700/50 rounded-lg text-sm text-slate-500"
                    >
                      {data.title}
                    </div>
                  ))}
                </div>
              </details>
            </div>
          )}

          {/* Cause Diagnosis */}
          {cause_diagnosis && cause_diagnosis.top_cause && (
            <div className="bg-violet-500/10 border border-violet-500/30 rounded-xl p-6 mb-8">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-violet-400">Root Cause Analysis</h2>
                <span className="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-1 rounded">
                  Hallucination rate: 0%
                </span>
              </div>
              <div className="flex items-center gap-4">
                <div>
                  <p className="text-2xl font-bold">
                    {cause_diagnosis.top_cause.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())}
                  </p>
                  <p className="text-slate-400 text-sm mt-1">
                    Confidence: {Math.round((cause_diagnosis.confidence || 0) * 100)}%
                  </p>
                </div>
              </div>
              {cause_diagnosis.hypotheses && cause_diagnosis.hypotheses.length > 0 && (
                <div className="mt-4 space-y-2">
                  <p className="text-sm text-slate-400 font-medium">Other possibilities:</p>
                  {cause_diagnosis.hypotheses.slice(0, 3).map((h, i) => (
                    <details key={i} className="group/hyp">
                      <summary className="cursor-pointer flex items-center gap-3 list-none [&::-webkit-details-marker]:hidden">
                        <div className="flex-1 bg-slate-700/50 rounded-full h-2">
                          <div
                            className="bg-violet-500/50 h-2 rounded-full"
                            style={{ width: `${Math.round(h.probability * 100)}%` }}
                          />
                        </div>
                        <span className="text-sm text-slate-400 min-w-[80px]">
                          {h.cause.replace(/_/g, " ")} ({Math.round(h.probability * 100)}%)
                        </span>
                        <ChevronRight className="w-3 h-3 text-slate-600 group-open/hyp:rotate-90 transition-transform" />
                      </summary>
                      {h.evidence && h.evidence.length > 0 && (
                        <div className="mt-2 ml-2 border-l-2 border-violet-500/30 pl-3 space-y-1">
                          {h.evidence.map((e, j) => (
                            <p key={j} className="text-xs text-slate-500">{e}</p>
                          ))}
                        </div>
                      )}
                    </details>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Upgrade prompt for anonymous users */}
          {results.upgrade_prompt && (
            <UpgradePrompt
              message={results.upgrade_prompt.message}
              cta={results.upgrade_prompt.cta}
              onSignUp={openSignUpModal}
            />
          )}

          {/* Email Report Section */}
          <div className="bg-gradient-to-r from-emerald-500/10 to-emerald-600/5 border border-emerald-500/30 rounded-xl p-6 mb-8">
            <div className="flex items-start gap-3 mb-4">
              <Mail className="text-emerald-400 flex-shrink-0 mt-0.5" size={22} />
              <div>
                <h2 className="text-lg font-semibold text-white">
                  Get Your Full Report
                </h2>
                <p className="text-sm text-slate-400 mt-1">
                  Receive the complete analysis as a PDF with all SKUs, product names,
                  financial impact, and actionable recommendations.
                </p>
              </div>
            </div>

            {reportSent ? (
              <div className="flex items-center gap-3 bg-emerald-500/20 border border-emerald-500/40 rounded-lg p-4">
                <CheckCircle className="text-emerald-400 flex-shrink-0" size={20} />
                <div>
                  <p className="text-emerald-300 font-medium">Report sent!</p>
                  <p className="text-sm text-slate-400 mt-0.5">
                    Check {reportEmail} for your PDF report. Save it — your uploaded
                    files are automatically deleted within 24 hours.
                  </p>
                </div>
              </div>
            ) : (
              <div>
                <div className="flex gap-3">
                  <input
                    type="email"
                    value={reportEmail}
                    onChange={(e) => {
                      setReportEmail(e.target.value);
                      setReportError(null);
                    }}
                    placeholder="Enter your email address"
                    className="flex-1 bg-slate-800 border border-slate-600 rounded-lg px-4 py-3 text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                    disabled={reportSending}
                  />
                  <button
                    onClick={handleSendReport}
                    disabled={reportSending || !reportEmail.trim()}
                    className={`px-6 py-3 rounded-lg font-semibold text-sm transition flex items-center gap-2 ${
                      reportSending || !reportEmail.trim()
                        ? "bg-slate-700 text-slate-500 cursor-not-allowed"
                        : "bg-emerald-600 hover:bg-emerald-500 text-white"
                    }`}
                  >
                    {reportSending ? (
                      <>
                        <Loader2 size={16} className="animate-spin" />
                        Sending...
                      </>
                    ) : (
                      <>
                        <Mail size={16} />
                        Send Report
                      </>
                    )}
                  </button>
                </div>

                {reportError && (
                  <p className="text-sm text-red-400 mt-2">{reportError}</p>
                )}

                <p className="text-xs text-slate-500 mt-3">
                  By providing your email, you consent to receiving your analysis report.
                  Your email is used solely for report delivery.{" "}
                  <a href="/privacy" className="text-emerald-400/70 hover:text-emerald-400 underline">
                    Privacy Policy
                  </a>
                </p>
              </div>
            )}
          </div>
        </main>

        {/* Auth Modal */}
        <AuthModal
          isOpen={showAuthModal}
          onClose={() => setShowAuthModal(false)}
          onSuccess={() => getFileSizeLimit().then(setMaxFileSizeMb)}
          defaultMode={authModalMode}
        />
      </div>
    );
  }

  return null;
}

// Leak Card Component
function LeakCard({
  leakKey,
  data,
  expanded,
  onToggle,
  attributions,
  impactAmount = 0,
}: {
  leakKey: string;
  data: LeakData;
  expanded: boolean;
  onToggle: () => void;
  attributions: Record<string, ColumnAttribution>;
  impactAmount?: number;
}) {
  const metadata = LEAK_METADATA[leakKey];
  const severityBadge = getSeverityBadge(data.severity || metadata?.severity || "low");

  // Use API color or fallback to metadata
  const borderColor = data.color || metadata?.color || "#6b7280";

  return (
    <div
      className="bg-slate-800/50 border rounded-xl overflow-hidden transition-all"
      style={{ borderColor: `${borderColor}40`, borderLeftWidth: "4px", borderLeftColor: borderColor }}
    >
      {/* Header */}
      <button
        onClick={onToggle}
        className="w-full px-4 py-4 flex items-center justify-between text-left hover:bg-slate-800/30 transition"
      >
        <div className="flex items-center gap-3">
          <span style={{ color: borderColor }}>
            {LEAK_ICONS[data.icon] || <AlertCircle size={20} />}
          </span>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-semibold">{data.title}</span>
              <span className={`px-2 py-0.5 text-xs font-bold rounded border ${severityBadge.className}`}>
                {severityBadge.label}
              </span>
            </div>
            <p className="text-sm text-slate-400 mt-0.5">
              {metadata?.plainEnglish?.slice(0, 100) || data.category}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-right">
            <p className="font-bold" style={{ color: borderColor }}>
              {data.count.toLocaleString()}
            </p>
            <p className="text-xs text-slate-500">items</p>
            {impactAmount > 0 && (
              <p className="text-xs text-emerald-400/70 mt-0.5">
                ~{formatDollarImpact(impactAmount)}/yr
              </p>
            )}
          </div>
          {expanded ? <ChevronDown size={20} /> : <ChevronRight size={20} />}
        </div>
      </button>

      {/* Expanded Content */}
      {expanded && (
        <div className="px-4 pb-4 border-t border-slate-700/50">
          {/* Item List */}
          {data.item_details && data.item_details.length > 0 && (
            <div className="mt-4 space-y-2">
              <p className="text-sm text-slate-400 font-medium mb-2">
                Top {Math.min(10, data.item_details.length)} items:
              </p>
              {data.item_details.slice(0, 10).map((item, i) => {
                const risk = scoreToRiskLabel(item.score);
                return (
                  <div
                    key={i}
                    className="bg-slate-900/50 rounded-lg p-3 text-sm"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-mono font-medium">{item.sku}</span>
                      <span className={`px-2 py-0.5 text-xs rounded border ${risk.className}`}>
                        {risk.percentage} risk
                      </span>
                    </div>
                    {item.description && (
                      <p className="text-slate-500 text-xs mb-2">{item.description}</p>
                    )}
                    <div className="flex gap-4 text-xs text-slate-500 mb-2 group/metrics">
                      <span className="inline-flex items-center gap-0.5">
                        QOH: <AttributionTooltip attribution={attributions.quantity} />
                        <strong className="text-slate-300">{item.quantity}</strong>
                      </span>
                      <span className="inline-flex items-center gap-0.5">
                        Cost: <AttributionTooltip attribution={attributions.cost} />
                        <strong className="text-slate-300">${item.cost.toFixed(2)}</strong>
                      </span>
                      <span className="inline-flex items-center gap-0.5">
                        Retail: <AttributionTooltip attribution={attributions.revenue} />
                        <strong className="text-slate-300">${item.revenue.toFixed(2)}</strong>
                      </span>
                      <span className="inline-flex items-center gap-0.5">
                        Sold: <AttributionTooltip attribution={attributions.sold} />
                        <strong className="text-slate-300">{item.sold}</strong>
                      </span>
                    </div>
                    {item.context && (
                      <p className="text-amber-400/80 text-xs bg-amber-500/10 rounded px-2 py-1">
                        {item.context}
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {/* Recommendations */}
          {data.recommendations && data.recommendations.length > 0 && (
            <div className="mt-4 bg-slate-900/30 rounded-lg p-3">
              <p className="text-sm font-medium text-emerald-400 mb-2">Recommendations:</p>
              <ul className="text-sm text-slate-400 space-y-1">
                {(data.recommendations.length > 0 ? data.recommendations : metadata?.recommendations || []).map(
                  (rec, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className="text-emerald-500 mt-1">-</span>
                      {rec}
                    </li>
                  )
                )}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
