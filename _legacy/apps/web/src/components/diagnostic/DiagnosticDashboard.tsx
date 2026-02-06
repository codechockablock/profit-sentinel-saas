"use client";

import React, { useState, useCallback, useRef, ChangeEvent, DragEvent, KeyboardEvent } from 'react';
import Link from 'next/link';
import {
  Upload, AlertTriangle, CheckCircle, Package, TrendingDown,
  MessageSquare, ChevronRight, Zap, Brain, ArrowRight, HelpCircle,
  Star, Truck
} from 'lucide-react';

// Types
interface Pattern {
  id: string;
  name: string;
  items: number;
  value: number;
  question: string;
  answers: [string, string][];
}

interface BreakdownItem {
  items: number;
  value: number;
  label: string;
  desc: string;
}

interface ClassificationInfo {
  label: string;
  color: string;
  desc: string;
}

interface ClassificationData {
  items: number;
  value: number;
  patterns: string[];
}

// API response types
interface StartSessionResponse {
  session_id: string;
  store_name: string;
  total_items: number;
  negative_items: number;
  total_shrinkage: number;
  patterns_detected: number;
}

interface QuestionResponse {
  pattern_id: string;
  pattern_name: string;
  question: string;
  suggested_answers: [string, string][];
  item_count: number;
  total_value: number;
  sample_items: Record<string, unknown>[];
  progress: { current: number; total: number };
  running_totals: {
    total_shrinkage: number;
    explained_value: number;
    unexplained_value: number;
    reduction_percent: number;
  };
}

// Classification labels and colors
const CLASSIFICATION_LABELS: Record<string, ClassificationInfo> = {
  receiving_gap: { label: 'Receiving Gap', color: '#3b82f6', desc: 'Sold at POS, not received' },
  non_tracked: { label: 'Non-Tracked', color: '#10b981', desc: 'By design (bins, cut-to-length)' },
  vendor_managed: { label: 'Vendor Managed', color: '#8b5cf6', desc: 'Direct ship items' },
  expiration: { label: 'Expiration', color: '#f59e0b', desc: 'Expires without write-off' },
  theft: { label: 'Theft', color: '#ef4444', desc: 'Likely theft' },
  investigate: { label: 'Investigate', color: '#ef4444', desc: 'Needs investigation' },
  partial: { label: 'Partial', color: '#6b7280', desc: 'Mixed tracking' },
};

const formatCurrency = (n: number): string => `$${n.toLocaleString()}`;
const formatNumber = (n: number): string => n.toLocaleString();

// Premium Preview Banner Component
function PremiumPreviewBanner() {
  return (
    <Link
      href="/diagnostic/premium"
      style={{
        display: 'block',
        textDecoration: 'none',
        background: 'linear-gradient(135deg, rgba(251,191,36,0.1) 0%, rgba(245,158,11,0.05) 100%)',
        border: '1px solid rgba(251,191,36,0.3)',
        borderRadius: 12,
        padding: '1rem 1.25rem',
        marginBottom: '1.5rem',
        transition: 'all 0.2s',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        <div
          style={{
            width: 40,
            height: 40,
            background: 'linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%)',
            borderRadius: 10,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
          }}
        >
          <Star size={20} color="#1a1a1a" fill="#1a1a1a" />
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
            <span style={{ fontWeight: 600, color: '#fbbf24', fontSize: '0.9rem' }}>
              Premium Preview
            </span>
            <span
              style={{
                padding: '0.15rem 0.5rem',
                background: 'rgba(251,191,36,0.2)',
                borderRadius: 4,
                fontSize: '0.65rem',
                color: '#fbbf24',
                fontWeight: 700,
                textTransform: 'uppercase',
              }}
            >
              New
            </span>
          </div>
          <p style={{ color: '#a1a1aa', fontSize: '0.8rem', margin: 0 }}>
            Multi-file vendor correlation — cross-reference invoices to find short ships & cost variances
          </p>
        </div>
        <ChevronRight size={20} color="#fbbf24" />
      </div>
    </Link>
  );
}

// Parse **text** and render as bold
const formatBoldText = (text: string): React.ReactNode => {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    return part;
  });
};

// Progress Ring Component
interface ProgressRingProps {
  percent: number;
  size?: number;
}

const ProgressRing: React.FC<ProgressRingProps> = ({ percent, size = 120 }) => {
  const strokeWidth = 8;
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (percent / 100) * circumference;

  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
        <circle
          stroke="#27272a"
          strokeWidth={strokeWidth}
          fill="none"
          r={radius}
          cx={size / 2}
          cy={size / 2}
        />
        <circle
          stroke="#10b981"
          strokeWidth={strokeWidth}
          fill="none"
          r={radius}
          cx={size / 2}
          cy={size / 2}
          strokeLinecap="round"
          style={{
            strokeDasharray: circumference,
            strokeDashoffset: offset,
            transition: 'stroke-dashoffset 0.5s ease-out',
          }}
        />
      </svg>
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          textAlign: 'center',
        }}
      >
        <span
          style={{
            display: 'block',
            fontSize: '1.5rem',
            fontWeight: 700,
            fontFamily: 'monospace',
            color: '#10b981',
          }}
        >
          {percent.toFixed(1)}%
        </span>
        <span style={{ fontSize: '0.65rem', color: '#71717a' }}>Explained</span>
      </div>
    </div>
  );
};

// Main Component
export default function DiagnosticDashboard() {
  const [stage, setStage] = useState<'upload' | 'diagnostic' | 'complete'>('upload');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [currentQuestion, setCurrentQuestion] = useState<QuestionResponse | null>(null);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [patternNames, setPatternNames] = useState<Record<string, string>>({});
  const [patternValues, setPatternValues] = useState<Record<string, { items: number; value: number }>>({});
  const [totalShrinkage, setTotalShrinkage] = useState<number>(0);
  const [explainedValue, setExplainedValue] = useState<number>(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [storeName, setStoreName] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);

  // Email modal state
  const [showEmailModal, setShowEmailModal] = useState(false);
  const [emailInput, setEmailInput] = useState('');
  const [emailSending, setEmailSending] = useState(false);
  const [emailSent, setEmailSent] = useState(false);
  const [emailError, setEmailError] = useState<string | null>(null);

  const handleDrag = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileUpload = async (file: File) => {
    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('store_name', storeName || 'My Store');

    try {
      const response = await fetch('/api/diagnostic/start', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to start diagnostic');
      }

      const data: StartSessionResponse = await response.json();
      setSessionId(data.session_id);
      setTotalShrinkage(data.total_shrinkage);

      // Fetch first question
      await fetchNextQuestion(data.session_id);
      setStage('diagnostic');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const fetchNextQuestion = async (sid: string) => {
    try {
      const response = await fetch(`/api/diagnostic/${sid}/question`);
      if (!response.ok) {
        throw new Error('Failed to fetch question');
      }

      const data = await response.json();
      if (data) {
        setCurrentQuestion(data);
        if (data.running_totals) {
          setExplainedValue(data.running_totals.explained_value);
        }
      } else {
        // No more questions - complete
        setStage('complete');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch question');
    }
  };

  const handleAnswer = async (classification: string) => {
    if (!sessionId || !currentQuestion) return;

    setIsLoading(true);
    try {
      const response = await fetch(`/api/diagnostic/${sessionId}/answer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ classification, note: '' }),
      });

      if (!response.ok) {
        throw new Error('Failed to submit answer');
      }

      const result = await response.json();

      // Update local state
      setAnswers(prev => ({
        ...prev,
        [currentQuestion.pattern_id]: classification
      }));
      setPatternNames(prev => ({
        ...prev,
        [currentQuestion.pattern_id]: currentQuestion.pattern_name
      }));
      setPatternValues(prev => ({
        ...prev,
        [currentQuestion.pattern_id]: {
          items: currentQuestion.item_count,
          value: currentQuestion.total_value
        }
      }));

      if (result.running_totals) {
        setExplainedValue(result.running_totals.explained_value);
      }

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

  const handleSkip = () => {
    handleAnswer('investigate');
  };

  const handleSendEmail = async () => {
    if (!sessionId || !emailInput.trim()) return;

    // Basic email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(emailInput)) {
      setEmailError('Please enter a valid email address');
      return;
    }

    setEmailSending(true);
    setEmailError(null);

    try {
      const response = await fetch(`/api/diagnostic/${sessionId}/email`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: emailInput,
          include_summary: true,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to send email');
      }

      setEmailSent(true);
    } catch (err) {
      setEmailError(err instanceof Error ? err.message : 'Failed to send email');
    } finally {
      setEmailSending(false);
    }
  };

  const reductionPercent = totalShrinkage > 0 ? (explainedValue / totalShrinkage) * 100 : 0;

  // Upload Stage
  if (stage === 'upload') {
    return (
      <div
        style={{
          minHeight: '100vh',
          fontFamily: "'Space Grotesk', system-ui",
          background: 'linear-gradient(135deg, #0a0a0f 0%, #0f1419 100%)',
          color: '#e4e4e7',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <div style={{ maxWidth: 480, width: '100%', padding: '2rem', textAlign: 'center' }}>
          <div
            style={{
              width: 64,
              height: 64,
              background: 'linear-gradient(135deg, #10b981, #059669)',
              borderRadius: 16,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 1.5rem',
              boxShadow: '0 8px 32px rgba(16,185,129,0.3)',
            }}
          >
            <Zap size={28} color="white" />
          </div>
          <h1 style={{ fontSize: '2rem', fontWeight: 700, marginBottom: '0.5rem' }}>
            Profit Sentinel
          </h1>
          <p style={{ color: '#71717a', marginBottom: '1.5rem' }}>
            Conversational Shrinkage Diagnostic
          </p>

          {/* Premium Preview Banner */}
          <PremiumPreviewBanner />

          <input
            type="text"
            placeholder="Store name (optional)"
            value={storeName}
            onChange={(e: ChangeEvent<HTMLInputElement>) => setStoreName(e.target.value)}
            style={{
              width: '100%',
              padding: '0.75rem 1rem',
              marginBottom: '1.5rem',
              background: 'rgba(39,39,42,0.5)',
              border: '1px solid #3f3f46',
              borderRadius: 8,
              color: '#e4e4e7',
              fontSize: '0.95rem',
            }}
          />

          {error && (
            <div
              style={{
                padding: '0.75rem 1rem',
                marginBottom: '1rem',
                background: 'rgba(239,68,68,0.1)',
                border: '1px solid rgba(239,68,68,0.3)',
                borderRadius: 8,
                color: '#fca5a5',
                fontSize: '0.9rem',
              }}
            >
              {error}
            </div>
          )}

          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            style={{
              border: `2px dashed ${dragActive ? '#10b981' : '#27272a'}`,
              borderRadius: 12,
              padding: '3rem 2rem',
              cursor: 'pointer',
              transition: 'all 0.2s',
              background: dragActive ? 'rgba(16,185,129,0.05)' : 'rgba(24,24,27,0.5)',
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,.xlsx"
              onChange={(e: ChangeEvent<HTMLInputElement>) =>
                e.target.files?.[0] && handleFileUpload(e.target.files[0])
              }
              hidden
            />
            {isLoading ? (
              <>
                <div
                  style={{
                    width: 40,
                    height: 40,
                    border: '3px solid #27272a',
                    borderTopColor: '#10b981',
                    borderRadius: '50%',
                    margin: '0 auto 1rem',
                    animation: 'spin 1s linear infinite',
                  }}
                />
                <h3 style={{ marginBottom: '0.5rem' }}>Analyzing inventory...</h3>
                <p style={{ color: '#71717a', fontSize: '0.9rem' }}>
                  Detecting patterns in your data
                </p>
              </>
            ) : (
              <>
                <Upload size={40} style={{ color: '#52525b', marginBottom: '1rem' }} />
                <h3 style={{ marginBottom: '0.5rem' }}>Drop inventory file here</h3>
                <p style={{ color: '#71717a', fontSize: '0.9rem' }}>or click to browse</p>
              </>
            )}
          </div>

          <div
            style={{
              display: 'flex',
              justifyContent: 'center',
              gap: '2rem',
              marginTop: '2rem',
              color: '#71717a',
              fontSize: '0.85rem',
            }}
          >
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Brain size={16} style={{ color: '#10b981' }} /> 27+ patterns
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <MessageSquare size={16} style={{ color: '#10b981' }} /> Interactive
            </span>
          </div>
        </div>

        <style>{`
          @keyframes spin {
            to { transform: rotate(360deg); }
          }
        `}</style>
      </div>
    );
  }

  // Diagnostic Stage
  if (stage === 'diagnostic' && currentQuestion) {
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
        <header
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '0.75rem 1.5rem',
            background: 'rgba(24,24,27,0.9)',
            borderBottom: '1px solid #27272a',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <div
              style={{
                width: 32,
                height: 32,
                background: 'linear-gradient(135deg, #10b981, #059669)',
                borderRadius: 8,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Zap size={18} color="white" />
            </div>
            <span style={{ fontWeight: 600 }}>Profit Sentinel</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '0.7rem', color: '#71717a' }}>Explained</div>
              <div style={{ fontFamily: 'monospace', fontWeight: 600, color: '#10b981' }}>
                {formatCurrency(explainedValue)}
              </div>
            </div>
            <ProgressRing percent={reductionPercent} size={48} />
          </div>
        </header>

        <main style={{ maxWidth: 700, margin: '0 auto', padding: '2rem 1.5rem' }}>
          {/* Progress bar */}
          <div style={{ marginBottom: '2rem' }}>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                marginBottom: '0.5rem',
                fontSize: '0.8rem',
                color: '#71717a',
              }}
            >
              <span>
                Pattern {currentQuestion.progress.current} of {currentQuestion.progress.total}
              </span>
              <span>{formatCurrency(currentQuestion.total_value)}</span>
            </div>
            <div style={{ height: 4, background: '#27272a', borderRadius: 2 }}>
              <div
                style={{
                  height: '100%',
                  width: `${(currentQuestion.progress.current / currentQuestion.progress.total) * 100}%`,
                  background: 'linear-gradient(90deg, #10b981, #3b82f6)',
                  borderRadius: 2,
                  transition: 'width 0.3s',
                }}
              />
            </div>
          </div>

          {/* Question card */}
          <div
            style={{
              background: 'rgba(24,24,27,0.8)',
              border: '1px solid #27272a',
              borderRadius: 16,
              padding: '2rem',
              marginBottom: '1.5rem',
            }}
          >
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                marginBottom: '1rem',
              }}
            >
              <HelpCircle size={20} style={{ color: '#10b981' }} />
              <span style={{ fontFamily: 'monospace', fontWeight: 600, color: '#10b981' }}>
                {currentQuestion.pattern_name}
              </span>
              <span style={{ marginLeft: 'auto', fontSize: '0.85rem', color: '#71717a' }}>
                {currentQuestion.item_count} items
              </span>
            </div>

            <p
              style={{
                fontSize: '1.1rem',
                lineHeight: 1.6,
                marginBottom: '1.5rem',
                whiteSpace: 'pre-line',
              }}
            >
              {formatBoldText(currentQuestion.question)}
            </p>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {currentQuestion.suggested_answers.map(([label, classification], i) => {
                const classInfo =
                  CLASSIFICATION_LABELS[classification] || CLASSIFICATION_LABELS.investigate;
                return (
                  <button
                    key={i}
                    onClick={() => handleAnswer(classification)}
                    disabled={isLoading}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.75rem',
                      padding: '1rem 1.25rem',
                      background: 'rgba(39,39,42,0.5)',
                      border: '1px solid #3f3f46',
                      borderRadius: 10,
                      color: '#e4e4e7',
                      fontSize: '0.95rem',
                      cursor: isLoading ? 'not-allowed' : 'pointer',
                      transition: 'all 0.15s',
                      textAlign: 'left',
                      opacity: isLoading ? 0.6 : 1,
                    }}
                  >
                    <span
                      style={{
                        width: 8,
                        height: 8,
                        borderRadius: '50%',
                        background: classInfo.color,
                        flexShrink: 0,
                      }}
                    />
                    <span style={{ flex: 1 }}>{label}</span>
                    <ChevronRight size={18} style={{ color: '#52525b' }} />
                  </button>
                );
              })}

              {/* Skip / Investigate option */}
              <button
                onClick={handleSkip}
                disabled={isLoading}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem',
                  padding: '1rem 1.25rem',
                  background: 'rgba(239,68,68,0.1)',
                  border: '1px solid rgba(239,68,68,0.3)',
                  borderRadius: 10,
                  color: '#fca5a5',
                  fontSize: '0.95rem',
                  cursor: isLoading ? 'not-allowed' : 'pointer',
                  transition: 'all 0.15s',
                  textAlign: 'left',
                  opacity: isLoading ? 0.6 : 1,
                }}
              >
                <span
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    background: '#ef4444',
                    flexShrink: 0,
                  }}
                />
                <span style={{ flex: 1 }}>I'm not sure - investigate these</span>
                <ChevronRight size={18} style={{ color: '#ef4444' }} />
              </button>
            </div>
          </div>

          {/* Running total */}
          <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center' }}>
            <div style={{ fontSize: '0.85rem', color: '#71717a' }}>
              Running total:{' '}
              <span style={{ color: '#10b981', fontFamily: 'monospace' }}>
                {reductionPercent.toFixed(1)}%
              </span>{' '}
              explained
            </div>
          </div>

          {/* Answered patterns */}
          {Object.keys(answers).length > 0 && (
            <div
              style={{
                marginTop: '2rem',
                paddingTop: '1.5rem',
                borderTop: '1px solid #27272a',
              }}
            >
              <h4 style={{ fontSize: '0.85rem', color: '#71717a', marginBottom: '0.75rem' }}>
                Answered ({Object.keys(answers).length})
              </h4>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                {Object.entries(answers).map(([id, classification]) => {
                  const classInfo = CLASSIFICATION_LABELS[classification];
                  const patternName = patternNames[id] || id;
                  return (
                    <span
                      key={id}
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '0.3rem',
                        padding: '0.3rem 0.6rem',
                        background: 'rgba(39,39,42,0.5)',
                        borderRadius: 6,
                        fontSize: '0.75rem',
                      }}
                    >
                      <span
                        style={{
                          width: 6,
                          height: 6,
                          borderRadius: '50%',
                          background: classInfo?.color || '#6b7280',
                        }}
                      />
                      {patternName}
                    </span>
                  );
                })}
              </div>
            </div>
          )}
        </main>
      </div>
    );
  }

  // Complete Stage
  const unexplainedValue = totalShrinkage - explainedValue;

  return (
    <div
      style={{
        minHeight: '100vh',
        fontFamily: "'Space Grotesk', system-ui",
        background: 'linear-gradient(135deg, #0a0a0f 0%, #0f1419 100%)',
        color: '#e4e4e7',
      }}
    >
      <header
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '0.75rem 1.5rem',
          background: 'rgba(24,24,27,0.9)',
          borderBottom: '1px solid #27272a',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <div
            style={{
              width: 32,
              height: 32,
              background: 'linear-gradient(135deg, #10b981, #059669)',
              borderRadius: 8,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Zap size={18} color="white" />
          </div>
          <span style={{ fontWeight: 600 }}>Profit Sentinel</span>
        </div>
        <button
          onClick={() => {
            setStage('upload');
            setSessionId(null);
            setCurrentQuestion(null);
            setAnswers({});
            setPatternNames({});
            setPatternValues({});
            setTotalShrinkage(0);
            setExplainedValue(0);
          }}
          style={{
            padding: '0.4rem 0.8rem',
            background: '#27272a',
            border: '1px solid #3f3f46',
            borderRadius: 6,
            color: '#e4e4e7',
            fontSize: '0.8rem',
            cursor: 'pointer',
          }}
        >
          New Analysis
        </button>
      </header>

      <main style={{ maxWidth: 800, margin: '0 auto', padding: '2rem 1.5rem' }}>
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <CheckCircle size={48} style={{ color: '#10b981', marginBottom: '1rem' }} />
          <h1 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>Diagnostic Complete</h1>
          <p style={{ color: '#71717a' }}>{Object.keys(answers).length} patterns reviewed</p>
        </div>

        {/* Big numbers */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            gap: '2rem',
            marginBottom: '2rem',
            flexWrap: 'wrap',
          }}
        >
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: '#71717a' }}>Apparent Shrinkage</div>
            <div
              style={{
                fontSize: '2rem',
                fontWeight: 700,
                fontFamily: 'monospace',
                color: '#ef4444',
              }}
            >
              {formatCurrency(totalShrinkage)}
            </div>
          </div>
          <ArrowRight size={24} style={{ color: '#52525b' }} />
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: '#71717a' }}>To Investigate</div>
            <div
              style={{
                fontSize: '2rem',
                fontWeight: 700,
                fontFamily: 'monospace',
                color: '#10b981',
              }}
            >
              {formatCurrency(unexplainedValue)}
            </div>
          </div>
          <ProgressRing percent={reductionPercent} size={100} />
        </div>

        {/* Summary */}
        <div
          style={{
            background: 'linear-gradient(135deg, rgba(16,185,129,0.1), rgba(59,130,246,0.1))',
            border: '1px solid rgba(16,185,129,0.2)',
            borderRadius: 12,
            padding: '1.5rem',
            textAlign: 'center',
            marginBottom: '1.5rem',
          }}
        >
          <p style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>
            You reduced apparent shrinkage by{' '}
            <strong style={{ color: '#10b981' }}>{formatCurrency(explainedValue)}</strong>
          </p>
          <p style={{ color: '#71717a' }}>
            That's <strong>{reductionPercent.toFixed(1)}%</strong> identified as process issues, not
            theft.
          </p>
        </div>

        {/* Your Answers Breakdown */}
        {Object.keys(answers).length > 0 && (
          <div
            style={{
              background: 'rgba(24,24,27,0.8)',
              border: '1px solid #27272a',
              borderRadius: 12,
              padding: '1.5rem',
              marginBottom: '1.5rem',
            }}
          >
            <h3 style={{ fontSize: '1rem', marginBottom: '1rem', color: '#a1a1aa' }}>
              Your Answers Breakdown
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {Object.entries(
                Object.entries(answers).reduce<Record<string, { items: number; value: number; patterns: string[] }>>(
                  (acc, [patternId, classification]) => {
                    if (!acc[classification]) {
                      acc[classification] = { items: 0, value: 0, patterns: [] };
                    }
                    const pv = patternValues[patternId];
                    if (pv) {
                      acc[classification].items += pv.items;
                      acc[classification].value += pv.value;
                    }
                    acc[classification].patterns.push(patternNames[patternId] || patternId);
                    return acc;
                  },
                  {}
                )
              ).map(([classification, data]) => {
                const classInfo = CLASSIFICATION_LABELS[classification] || { label: classification, color: '#6b7280', desc: '' };
                return (
                  <div
                    key={classification}
                    style={{
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: '0.75rem',
                      padding: '0.75rem',
                      background: 'rgba(39,39,42,0.5)',
                      borderRadius: 8,
                    }}
                  >
                    <span
                      style={{
                        width: 10,
                        height: 10,
                        borderRadius: '50%',
                        background: classInfo.color,
                        marginTop: 4,
                        flexShrink: 0,
                      }}
                    />
                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                        <span style={{ fontWeight: 600 }}>{classInfo.label}</span>
                        <span style={{ fontFamily: 'monospace', color: classInfo.color }}>
                          {formatCurrency(data.value)}
                        </span>
                      </div>
                      <div style={{ fontSize: '0.8rem', color: '#71717a' }}>
                        {data.items.toLocaleString()} items · {data.patterns.length} pattern{data.patterns.length !== 1 ? 's' : ''}
                      </div>
                      <div style={{ fontSize: '0.75rem', color: '#52525b', marginTop: '0.25rem' }}>
                        {data.patterns.join(', ')}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Download/Email buttons */}
        <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
          <button
            onClick={async () => {
              if (sessionId) {
                window.open(`/api/diagnostic/${sessionId}/report`, '_blank');
              }
            }}
            style={{
              padding: '0.75rem 1.5rem',
              background: 'linear-gradient(135deg, #10b981, #059669)',
              border: 'none',
              borderRadius: 8,
              color: 'white',
              fontSize: '0.9rem',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
            }}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
            Download PDF
          </button>
          <button
            onClick={() => {
              setShowEmailModal(true);
              setEmailSent(false);
              setEmailError(null);
            }}
            style={{
              padding: '0.75rem 1.5rem',
              background: 'transparent',
              border: '1px solid #3f3f46',
              borderRadius: 8,
              color: '#e4e4e7',
              fontSize: '0.9rem',
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
            }}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
              <polyline points="22,6 12,13 2,6" />
            </svg>
            Email Report
          </button>
        </div>

        {/* Premium Upsell */}
        <div
          style={{
            marginTop: '2rem',
            padding: '1.5rem',
            background: 'linear-gradient(135deg, rgba(251,191,36,0.1) 0%, rgba(245,158,11,0.05) 100%)',
            border: '1px solid rgba(251,191,36,0.3)',
            borderRadius: 12,
            textAlign: 'center',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
            <Star size={18} color="#fbbf24" fill="#fbbf24" />
            <span style={{ fontWeight: 600, color: '#fbbf24' }}>Want deeper insights?</span>
          </div>
          <p style={{ color: '#a1a1aa', fontSize: '0.85rem', marginBottom: '1rem' }}>
            Try our premium vendor correlation analysis — upload invoices to find short ships, cost variances, and vendor issues.
          </p>
          <Link
            href="/diagnostic/premium"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.6rem 1.25rem',
              background: 'linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%)',
              borderRadius: 8,
              color: '#1a1a1a',
              fontWeight: 600,
              fontSize: '0.9rem',
              textDecoration: 'none',
            }}
          >
            <Truck size={18} />
            Try Vendor Correlation
          </Link>
        </div>

        {/* Email Modal */}
        {showEmailModal && (
          <div
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: 'rgba(0,0,0,0.8)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 1000,
            }}
            onClick={() => setShowEmailModal(false)}
          >
            <div
              style={{
                background: '#18181b',
                border: '1px solid #27272a',
                borderRadius: 16,
                padding: '2rem',
                maxWidth: 400,
                width: '90%',
              }}
              onClick={(e) => e.stopPropagation()}
            >
              {emailSent ? (
                <div style={{ textAlign: 'center' }}>
                  <div
                    style={{
                      width: 56,
                      height: 56,
                      borderRadius: '50%',
                      background: 'rgba(16, 185, 129, 0.2)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      margin: '0 auto 1rem',
                    }}
                  >
                    <CheckCircle size={28} color="#10b981" />
                  </div>
                  <h3 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '0.5rem' }}>
                    Report Sent!
                  </h3>
                  <p style={{ color: '#71717a', marginBottom: '1.5rem' }}>
                    Check your inbox at <strong style={{ color: '#e4e4e7' }}>{emailInput}</strong>
                  </p>
                  <button
                    onClick={() => setShowEmailModal(false)}
                    style={{
                      padding: '0.75rem 1.5rem',
                      background: '#27272a',
                      border: 'none',
                      borderRadius: 8,
                      color: 'white',
                      fontSize: '0.9rem',
                      fontWeight: 600,
                      cursor: 'pointer',
                    }}
                  >
                    Close
                  </button>
                </div>
              ) : (
                <>
                  <h3 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '0.5rem' }}>
                    Email Your Report
                  </h3>
                  <p style={{ color: '#71717a', marginBottom: '1.5rem', fontSize: '0.9rem' }}>
                    Get the full diagnostic report with all SKU details sent to your inbox.
                  </p>
                  <input
                    type="email"
                    value={emailInput}
                    onChange={(e) => setEmailInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleSendEmail();
                    }}
                    placeholder="you@example.com"
                    style={{
                      width: '100%',
                      padding: '0.75rem 1rem',
                      background: '#09090b',
                      border: '1px solid #27272a',
                      borderRadius: 8,
                      color: '#e4e4e7',
                      fontSize: '1rem',
                      marginBottom: '0.75rem',
                      outline: 'none',
                    }}
                    autoFocus
                  />
                  {emailError && (
                    <p style={{ color: '#ef4444', fontSize: '0.85rem', marginBottom: '0.75rem' }}>
                      {emailError}
                    </p>
                  )}
                  <div style={{ display: 'flex', gap: '0.75rem' }}>
                    <button
                      onClick={() => setShowEmailModal(false)}
                      style={{
                        flex: 1,
                        padding: '0.75rem',
                        background: 'transparent',
                        border: '1px solid #3f3f46',
                        borderRadius: 8,
                        color: '#a1a1aa',
                        fontSize: '0.9rem',
                        fontWeight: 600,
                        cursor: 'pointer',
                      }}
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleSendEmail}
                      disabled={emailSending || !emailInput.trim()}
                      style={{
                        flex: 1,
                        padding: '0.75rem',
                        background: emailSending ? '#27272a' : 'linear-gradient(135deg, #10b981, #059669)',
                        border: 'none',
                        borderRadius: 8,
                        color: 'white',
                        fontSize: '0.9rem',
                        fontWeight: 600,
                        cursor: emailSending ? 'not-allowed' : 'pointer',
                        opacity: !emailInput.trim() ? 0.5 : 1,
                      }}
                    >
                      {emailSending ? 'Sending...' : 'Send Report'}
                    </button>
                  </div>
                  <p style={{ color: '#52525b', fontSize: '0.75rem', marginTop: '1rem', textAlign: 'center' }}>
                    Your data is deleted after sending. No spam, ever.
                  </p>
                </>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
