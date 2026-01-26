"use client";

import React, { useState, useCallback, useRef, DragEvent } from 'react';
import Link from 'next/link';
import {
  Upload, AlertTriangle, CheckCircle, TrendingDown,
  ChevronRight, Zap, FileText, Truck, DollarSign, Star
} from 'lucide-react';

// Types
interface PremiumStartResponse {
  session_id: string;
  store_name: string;
  files_processed: { inventory: number; invoices: number; total: number };
  inventory_items: number;
  vendor_invoice_lines: number;
  patterns_discovered: number;
  total_potential_impact: number;
  vendors_analyzed: number;
  is_premium_preview: boolean;
}

interface CorrelationQuestionResponse {
  pattern_index: number;
  pattern_type: string;
  question: string;
  suggested_answers: [string, string][];
  affected_skus_count: number;
  affected_vendors: string[];
  total_impact: number;
  confidence: number;
  description: string;
  evidence_summary: {
    type: string;
    sample_skus: string[];
    sample_items?: Array<Record<string, unknown>>;
  };
  progress: { current: number; total: number };
}

// Classification labels for correlation types
const CORRELATION_TYPE_LABELS: Record<string, { label: string; color: string; icon: React.ReactNode }> = {
  short_ship_negative_stock: { label: 'Short Ship → Negative Stock', color: '#ef4444', icon: <Truck size={18} /> },
  price_increase_margin_erosion: { label: 'Price Increase → Margin Erosion', color: '#f97316', icon: <DollarSign size={18} /> },
  chronic_short_ship: { label: 'Chronic Short Ships', color: '#f59e0b', icon: <AlertTriangle size={18} /> },
  vendor_fill_rate_issue: { label: 'Vendor Fill Rate Issue', color: '#8b5cf6', icon: <Truck size={18} /> },
  cost_variance_anomaly: { label: 'Cost Variance Anomaly', color: '#ec4899', icon: <TrendingDown size={18} /> },
  receiving_gap: { label: 'Receiving Gap', color: '#3b82f6', icon: <FileText size={18} /> },
};

const formatCurrency = (n: number): string => `$${n.toLocaleString()}`;

// Premium Badge Component
function PremiumBadge() {
  return (
    <div
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '0.35rem',
        padding: '0.35rem 0.75rem',
        background: 'linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%)',
        borderRadius: 20,
        fontSize: '0.7rem',
        fontWeight: 700,
        color: '#1a1a1a',
        textTransform: 'uppercase',
        letterSpacing: '0.05em',
      }}
    >
      <Star size={12} fill="#1a1a1a" />
      Premium Preview
    </div>
  );
}

// Main Component
export default function PremiumDiagnosticDashboard() {
  const [stage, setStage] = useState<'upload' | 'diagnostic' | 'complete'>('upload');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionData, setSessionData] = useState<PremiumStartResponse | null>(null);
  const [currentQuestion, setCurrentQuestion] = useState<CorrelationQuestionResponse | null>(null);
  const [answers, setAnswers] = useState<Record<number, string>>({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [storeName, setStoreName] = useState<string>('');

  // File states
  const [inventoryFile, setInventoryFile] = useState<File | null>(null);
  const [invoiceFiles, setInvoiceFiles] = useState<File[]>([]);
  const inventoryInputRef = useRef<HTMLInputElement>(null);
  const invoicesInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState<'inventory' | 'invoices' | null>(null);

  // Drag handlers
  const handleDrag = useCallback((e: DragEvent<HTMLDivElement>, zone: 'inventory' | 'invoices') => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(zone);
    } else if (e.type === 'dragleave') {
      setDragActive(null);
    }
  }, []);

  const handleInventoryDrop = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(null);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setInventoryFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleInvoicesDrop = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(null);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const newFiles = Array.from(e.dataTransfer.files);
      setInvoiceFiles(prev => [...prev, ...newFiles].slice(0, 199)); // Max 199 invoices
    }
  }, []);

  // File upload handler
  const handleStartSession = async () => {
    if (!inventoryFile) {
      setError('Please upload an inventory file');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('inventory_file', inventoryFile);
      formData.append('store_name', storeName || 'My Store');

      for (const file of invoiceFiles) {
        formData.append('invoice_files', file);
      }

      const response = await fetch('/api/premium/diagnostic/start', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to start diagnostic');
      }

      const data: PremiumStartResponse = await response.json();
      setSessionId(data.session_id);
      setSessionData(data);

      if (data.patterns_discovered > 0) {
        await fetchNextQuestion(data.session_id);
        setStage('diagnostic');
      } else {
        setStage('complete');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start session');
    } finally {
      setIsLoading(false);
    }
  };

  const fetchNextQuestion = async (sid: string) => {
    const response = await fetch(`/api/premium/${sid}/question`);
    if (response.ok) {
      const data = await response.json();
      if (data) {
        setCurrentQuestion(data);
      } else {
        setStage('complete');
      }
    }
  };

  const handleAnswer = async (classification: string) => {
    if (!sessionId || !currentQuestion) return;

    setIsLoading(true);
    try {
      const response = await fetch(`/api/premium/${sessionId}/answer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ classification }),
      });

      if (!response.ok) {
        throw new Error('Failed to submit answer');
      }

      const result = await response.json();

      setAnswers(prev => ({
        ...prev,
        [currentQuestion.pattern_index]: classification
      }));

      if (result.is_complete) {
        setStage('complete');
      } else {
        await fetchNextQuestion(sessionId);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit answer');
    } finally {
      setIsLoading(false);
    }
  };

  // Upload Stage
  if (stage === 'upload') {
    return (
      <div
        style={{
          minHeight: '100vh',
          fontFamily: "'Space Grotesk', system-ui",
          background: 'linear-gradient(135deg, #0a0a0f 0%, #0f1419 100%)',
          color: '#e4e4e7',
          padding: '2rem',
        }}
      >
        <div style={{ maxWidth: 800, margin: '0 auto' }}>
          {/* Header */}
          <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
            <div style={{ marginBottom: '1rem' }}>
              <PremiumBadge />
            </div>
            <div
              style={{
                width: 64,
                height: 64,
                margin: '0 auto 1rem',
                background: 'linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%)',
                borderRadius: 16,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Zap size={32} color="#1a1a1a" />
            </div>
            <h1 style={{ fontSize: '1.75rem', fontWeight: 700, marginBottom: '0.5rem' }}>
              Multi-File Vendor Correlation
            </h1>
            <p style={{ color: '#71717a', maxWidth: 500, margin: '0 auto' }}>
              Upload your inventory and vendor invoices. We'll cross-reference them to find
              correlations like short ships causing negative stock.
            </p>
          </div>

          {/* Store Name */}
          <div style={{ marginBottom: '1.5rem' }}>
            <input
              type="text"
              value={storeName}
              onChange={(e) => setStoreName(e.target.value)}
              placeholder="Store name (optional)"
              style={{
                width: '100%',
                padding: '0.75rem 1rem',
                background: '#18181b',
                border: '1px solid #27272a',
                borderRadius: 8,
                color: '#e4e4e7',
                fontSize: '1rem',
                outline: 'none',
              }}
            />
          </div>

          {/* Inventory Upload Zone */}
          <div style={{ marginBottom: '1.5rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 600, fontSize: '0.9rem' }}>
              Inventory File (Required)
            </label>
            <div
              onDragEnter={(e) => handleDrag(e, 'inventory')}
              onDragLeave={(e) => handleDrag(e, 'inventory')}
              onDragOver={(e) => handleDrag(e, 'inventory')}
              onDrop={handleInventoryDrop}
              onClick={() => inventoryInputRef.current?.click()}
              style={{
                border: `2px dashed ${dragActive === 'inventory' ? '#10b981' : inventoryFile ? '#10b981' : '#3f3f46'}`,
                borderRadius: 12,
                padding: '2rem',
                textAlign: 'center',
                cursor: 'pointer',
                background: dragActive === 'inventory' ? 'rgba(16,185,129,0.1)' : inventoryFile ? 'rgba(16,185,129,0.05)' : 'transparent',
                transition: 'all 0.2s',
              }}
            >
              <input
                ref={inventoryInputRef}
                type="file"
                accept=".csv,.xlsx,.xls"
                onChange={(e) => e.target.files?.[0] && setInventoryFile(e.target.files[0])}
                style={{ display: 'none' }}
              />
              {inventoryFile ? (
                <div>
                  <CheckCircle size={32} color="#10b981" style={{ margin: '0 auto 0.5rem' }} />
                  <p style={{ fontWeight: 600, color: '#10b981' }}>{inventoryFile.name}</p>
                  <p style={{ fontSize: '0.8rem', color: '#71717a' }}>Click to change</p>
                </div>
              ) : (
                <div>
                  <Upload size={32} color="#71717a" style={{ margin: '0 auto 0.5rem' }} />
                  <p style={{ color: '#a1a1aa' }}>Drop inventory CSV here or click to browse</p>
                </div>
              )}
            </div>
          </div>

          {/* Invoices Upload Zone */}
          <div style={{ marginBottom: '1.5rem' }}>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 600, fontSize: '0.9rem' }}>
              Vendor Invoices (Optional, up to 199 files)
            </label>
            <div
              onDragEnter={(e) => handleDrag(e, 'invoices')}
              onDragLeave={(e) => handleDrag(e, 'invoices')}
              onDragOver={(e) => handleDrag(e, 'invoices')}
              onDrop={handleInvoicesDrop}
              onClick={() => invoicesInputRef.current?.click()}
              style={{
                border: `2px dashed ${dragActive === 'invoices' ? '#8b5cf6' : invoiceFiles.length > 0 ? '#8b5cf6' : '#3f3f46'}`,
                borderRadius: 12,
                padding: '2rem',
                textAlign: 'center',
                cursor: 'pointer',
                background: dragActive === 'invoices' ? 'rgba(139,92,246,0.1)' : invoiceFiles.length > 0 ? 'rgba(139,92,246,0.05)' : 'transparent',
                transition: 'all 0.2s',
              }}
            >
              <input
                ref={invoicesInputRef}
                type="file"
                accept=".csv,.xlsx,.xls"
                multiple
                onChange={(e) => {
                  if (e.target.files && e.target.files.length > 0) {
                    const newFiles = Array.from(e.target.files);
                    setInvoiceFiles(prev => [...prev, ...newFiles].slice(0, 199));
                  }
                }}
                style={{ display: 'none' }}
              />
              {invoiceFiles.length > 0 ? (
                <div>
                  <Truck size={32} color="#8b5cf6" style={{ margin: '0 auto 0.5rem' }} />
                  <p style={{ fontWeight: 600, color: '#8b5cf6' }}>{invoiceFiles.length} invoice file{invoiceFiles.length !== 1 ? 's' : ''} selected</p>
                  <p style={{ fontSize: '0.8rem', color: '#71717a' }}>Click to add more or drop additional files</p>
                  {invoiceFiles.length > 5 && (
                    <button
                      onClick={(e) => { e.stopPropagation(); setInvoiceFiles([]); }}
                      style={{
                        marginTop: '0.5rem',
                        padding: '0.25rem 0.75rem',
                        background: 'transparent',
                        border: '1px solid #3f3f46',
                        borderRadius: 4,
                        color: '#71717a',
                        fontSize: '0.75rem',
                        cursor: 'pointer',
                      }}
                    >
                      Clear all
                    </button>
                  )}
                </div>
              ) : (
                <div>
                  <Truck size={32} color="#71717a" style={{ margin: '0 auto 0.5rem' }} />
                  <p style={{ color: '#a1a1aa' }}>Drop vendor invoice CSVs here or click to browse</p>
                  <p style={{ fontSize: '0.8rem', color: '#52525b' }}>POs, invoices, receiving reports, etc.</p>
                </div>
              )}
            </div>
          </div>

          {/* Error */}
          {error && (
            <div style={{
              padding: '0.75rem 1rem',
              background: 'rgba(239,68,68,0.1)',
              border: '1px solid rgba(239,68,68,0.3)',
              borderRadius: 8,
              color: '#ef4444',
              marginBottom: '1rem',
              fontSize: '0.9rem',
            }}>
              {error}
            </div>
          )}

          {/* Start Button */}
          <button
            onClick={handleStartSession}
            disabled={!inventoryFile || isLoading}
            style={{
              width: '100%',
              padding: '1rem',
              background: inventoryFile ? 'linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%)' : '#27272a',
              border: 'none',
              borderRadius: 12,
              color: inventoryFile ? '#1a1a1a' : '#71717a',
              fontSize: '1rem',
              fontWeight: 700,
              cursor: inventoryFile ? 'pointer' : 'not-allowed',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.5rem',
            }}
          >
            {isLoading ? (
              'Analyzing correlations...'
            ) : (
              <>
                <Zap size={20} />
                Start Vendor Correlation Analysis
              </>
            )}
          </button>

          {/* Features */}
          <div style={{ marginTop: '2rem', display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem' }}>
            {[
              { icon: <Truck size={18} />, text: 'Short ship detection' },
              { icon: <DollarSign size={18} />, text: 'Cost variance analysis' },
              { icon: <TrendingDown size={18} />, text: 'Margin erosion tracking' },
              { icon: <AlertTriangle size={18} />, text: 'Vendor fill rate scoring' },
            ].map((feature, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#71717a', fontSize: '0.85rem' }}>
                <span style={{ color: '#f59e0b' }}>{feature.icon}</span>
                {feature.text}
              </div>
            ))}
          </div>

          {/* Back link */}
          <div style={{ textAlign: 'center', marginTop: '2rem' }}>
            <Link href="/diagnostic" style={{ color: '#71717a', fontSize: '0.85rem', textDecoration: 'none' }}>
              ← Back to Standard Diagnostic
            </Link>
          </div>
        </div>
      </div>
    );
  }

  // Diagnostic Stage
  if (stage === 'diagnostic' && currentQuestion) {
    const typeInfo = CORRELATION_TYPE_LABELS[currentQuestion.pattern_type] || {
      label: currentQuestion.pattern_type,
      color: '#6b7280',
      icon: <AlertTriangle size={18} />,
    };

    return (
      <div
        style={{
          minHeight: '100vh',
          fontFamily: "'Space Grotesk', system-ui",
          background: 'linear-gradient(135deg, #0a0a0f 0%, #0f1419 100%)',
          color: '#e4e4e7',
        }}
      >
        {/* Header */}
        <header style={{
          padding: '1rem 1.5rem',
          borderBottom: '1px solid #27272a',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{
              width: 36,
              height: 36,
              background: 'linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%)',
              borderRadius: 8,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <Zap size={20} color="#1a1a1a" />
            </div>
            <span style={{ fontWeight: 600 }}>Profit Sentinel</span>
            <PremiumBadge />
          </div>
          <div style={{ fontSize: '0.85rem', color: '#71717a' }}>
            Pattern {currentQuestion.progress.current} of {currentQuestion.progress.total}
          </div>
        </header>

        <main style={{ maxWidth: 700, margin: '0 auto', padding: '2rem 1.5rem' }}>
          {/* Pattern Type Badge */}
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.5rem 1rem',
            background: `${typeInfo.color}20`,
            border: `1px solid ${typeInfo.color}40`,
            borderRadius: 8,
            marginBottom: '1rem',
          }}>
            <span style={{ color: typeInfo.color }}>{typeInfo.icon}</span>
            <span style={{ color: typeInfo.color, fontWeight: 600, fontSize: '0.85rem' }}>{typeInfo.label}</span>
          </div>

          {/* Impact */}
          <div style={{
            display: 'flex',
            gap: '1.5rem',
            marginBottom: '1.5rem',
          }}>
            <div>
              <div style={{ fontSize: '0.75rem', color: '#71717a', textTransform: 'uppercase' }}>Impact</div>
              <div style={{ fontSize: '1.5rem', fontWeight: 700, color: typeInfo.color }}>
                {formatCurrency(currentQuestion.total_impact)}
              </div>
            </div>
            <div>
              <div style={{ fontSize: '0.75rem', color: '#71717a', textTransform: 'uppercase' }}>Items</div>
              <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>{currentQuestion.affected_skus_count}</div>
            </div>
            <div>
              <div style={{ fontSize: '0.75rem', color: '#71717a', textTransform: 'uppercase' }}>Confidence</div>
              <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#10b981' }}>
                {(currentQuestion.confidence * 100).toFixed(0)}%
              </div>
            </div>
          </div>

          {/* Question */}
          <div style={{
            background: 'rgba(24,24,27,0.8)',
            border: '1px solid #27272a',
            borderRadius: 12,
            padding: '1.5rem',
            marginBottom: '1.5rem',
          }}>
            <p style={{ fontSize: '1.1rem', lineHeight: 1.6, marginBottom: '1rem' }}>
              {currentQuestion.question}
            </p>

            {/* Affected Vendors */}
            {currentQuestion.affected_vendors.length > 0 && (
              <div style={{ marginTop: '1rem' }}>
                <div style={{ fontSize: '0.75rem', color: '#71717a', marginBottom: '0.5rem' }}>Affected Vendors</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                  {currentQuestion.affected_vendors.slice(0, 5).map(vendor => (
                    <span
                      key={vendor}
                      style={{
                        padding: '0.25rem 0.75rem',
                        background: '#27272a',
                        borderRadius: 4,
                        fontSize: '0.8rem',
                        fontFamily: 'monospace',
                      }}
                    >
                      {vendor}
                    </span>
                  ))}
                  {currentQuestion.affected_vendors.length > 5 && (
                    <span style={{ color: '#71717a', fontSize: '0.8rem' }}>
                      +{currentQuestion.affected_vendors.length - 5} more
                    </span>
                  )}
                </div>
              </div>
            )}

            {/* Sample SKUs */}
            {currentQuestion.evidence_summary.sample_skus?.length > 0 && (
              <div style={{ marginTop: '1rem' }}>
                <div style={{ fontSize: '0.75rem', color: '#71717a', marginBottom: '0.5rem' }}>Sample SKUs</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                  {currentQuestion.evidence_summary.sample_skus.slice(0, 5).map(sku => (
                    <span
                      key={sku}
                      style={{
                        padding: '0.25rem 0.75rem',
                        background: '#18181b',
                        borderRadius: 4,
                        fontSize: '0.8rem',
                        fontFamily: 'monospace',
                      }}
                    >
                      {sku}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Answer Buttons */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {currentQuestion.suggested_answers.map(([label, classification], i) => (
              <button
                key={i}
                onClick={() => handleAnswer(classification)}
                disabled={isLoading}
                style={{
                  width: '100%',
                  padding: '1rem 1.25rem',
                  background: '#18181b',
                  border: '1px solid #27272a',
                  borderRadius: 8,
                  color: '#e4e4e7',
                  fontSize: '0.95rem',
                  fontWeight: 500,
                  cursor: 'pointer',
                  textAlign: 'left',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  opacity: isLoading ? 0.5 : 1,
                }}
              >
                {label}
                <ChevronRight size={18} color="#71717a" />
              </button>
            ))}
          </div>
        </main>
      </div>
    );
  }

  // Complete Stage
  return (
    <div
      style={{
        minHeight: '100vh',
        fontFamily: "'Space Grotesk', system-ui",
        background: 'linear-gradient(135deg, #0a0a0f 0%, #0f1419 100%)',
        color: '#e4e4e7',
      }}
    >
      {/* Header */}
      <header style={{
        padding: '1rem 1.5rem',
        borderBottom: '1px solid #27272a',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{
            width: 36,
            height: 36,
            background: 'linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%)',
            borderRadius: 8,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}>
            <Zap size={20} color="#1a1a1a" />
          </div>
          <span style={{ fontWeight: 600 }}>Profit Sentinel</span>
          <PremiumBadge />
        </div>
        <button
          onClick={() => {
            setStage('upload');
            setSessionId(null);
            setSessionData(null);
            setCurrentQuestion(null);
            setAnswers({});
            setInventoryFile(null);
            setInvoiceFiles([]);
          }}
          style={{
            padding: '0.4rem 0.8rem',
            background: '#27272a',
            border: '1px solid #3f3f46',
            borderRadius: 6,
            color: '#a1a1aa',
            fontSize: '0.85rem',
            cursor: 'pointer',
          }}
        >
          New Analysis
        </button>
      </header>

      <main style={{ maxWidth: 700, margin: '0 auto', padding: '2rem 1.5rem' }}>
        {/* Success Header */}
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <div style={{
            width: 80,
            height: 80,
            margin: '0 auto 1rem',
            background: 'linear-gradient(135deg, rgba(16,185,129,0.2), rgba(16,185,129,0.1))',
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}>
            <CheckCircle size={40} color="#10b981" />
          </div>
          <h1 style={{ fontSize: '1.75rem', fontWeight: 700, marginBottom: '0.5rem' }}>
            Vendor Correlation Complete
          </h1>
          <p style={{ color: '#71717a' }}>
            Your multi-file analysis is complete. Download the report for full details.
          </p>
        </div>

        {/* Summary Stats */}
        {sessionData && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: '1rem',
            marginBottom: '2rem',
          }}>
            <div style={{
              background: 'rgba(24,24,27,0.8)',
              border: '1px solid #27272a',
              borderRadius: 12,
              padding: '1.25rem',
              textAlign: 'center',
            }}>
              <div style={{ fontSize: '1.75rem', fontWeight: 700, color: '#10b981' }}>
                {sessionData.patterns_discovered}
              </div>
              <div style={{ fontSize: '0.8rem', color: '#71717a' }}>Patterns Found</div>
            </div>
            <div style={{
              background: 'rgba(24,24,27,0.8)',
              border: '1px solid #27272a',
              borderRadius: 12,
              padding: '1.25rem',
              textAlign: 'center',
            }}>
              <div style={{ fontSize: '1.75rem', fontWeight: 700, color: '#f59e0b' }}>
                {formatCurrency(sessionData.total_potential_impact)}
              </div>
              <div style={{ fontSize: '0.8rem', color: '#71717a' }}>Potential Impact</div>
            </div>
            <div style={{
              background: 'rgba(24,24,27,0.8)',
              border: '1px solid #27272a',
              borderRadius: 12,
              padding: '1.25rem',
              textAlign: 'center',
            }}>
              <div style={{ fontSize: '1.75rem', fontWeight: 700, color: '#8b5cf6' }}>
                {sessionData.vendors_analyzed}
              </div>
              <div style={{ fontSize: '0.8rem', color: '#71717a' }}>Vendors Analyzed</div>
            </div>
          </div>
        )}

        {/* Files Processed */}
        {sessionData && (
          <div style={{
            background: 'rgba(24,24,27,0.8)',
            border: '1px solid #27272a',
            borderRadius: 12,
            padding: '1.25rem',
            marginBottom: '1.5rem',
          }}>
            <h3 style={{ fontSize: '0.9rem', color: '#a1a1aa', marginBottom: '0.75rem' }}>Files Processed</h3>
            <div style={{ display: 'flex', gap: '2rem' }}>
              <div>
                <span style={{ color: '#71717a', fontSize: '0.85rem' }}>Inventory: </span>
                <span style={{ fontWeight: 600 }}>{sessionData.files_processed.inventory}</span>
              </div>
              <div>
                <span style={{ color: '#71717a', fontSize: '0.85rem' }}>Invoices: </span>
                <span style={{ fontWeight: 600 }}>{sessionData.files_processed.invoices}</span>
              </div>
              <div>
                <span style={{ color: '#71717a', fontSize: '0.85rem' }}>Items: </span>
                <span style={{ fontWeight: 600 }}>{sessionData.inventory_items.toLocaleString()}</span>
              </div>
              <div>
                <span style={{ color: '#71717a', fontSize: '0.85rem' }}>Invoice Lines: </span>
                <span style={{ fontWeight: 600 }}>{sessionData.vendor_invoice_lines.toLocaleString()}</span>
              </div>
            </div>
          </div>
        )}

        {/* Download Button */}
        <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
          <button
            onClick={() => {
              if (sessionId) {
                window.open(`/api/premium/${sessionId}/report`, '_blank');
              }
            }}
            style={{
              padding: '1rem 2rem',
              background: 'linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%)',
              border: 'none',
              borderRadius: 8,
              color: '#1a1a1a',
              fontSize: '1rem',
              fontWeight: 700,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
            }}
          >
            <FileText size={20} />
            Download Premium Report
          </button>
        </div>

        {/* Back to standard */}
        <div style={{ textAlign: 'center', marginTop: '2rem' }}>
          <Link href="/diagnostic" style={{ color: '#71717a', fontSize: '0.85rem', textDecoration: 'none' }}>
            ← Back to Standard Diagnostic
          </Link>
        </div>
      </main>
    </div>
  );
}
