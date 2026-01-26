import React, { useState } from 'react';
import { Upload, AlertTriangle, CheckCircle, Package, TrendingDown, MessageSquare, ChevronRight, Zap, Brain, ArrowRight, HelpCircle, Check, X } from 'lucide-react';

// Demo data - 27 patterns detected
const PATTERNS = [
  { id: 'lumber_2x', name: '2X Lumber', items: 64, value: 129344, question: "I found 64 items matching **2X lumber** (2x4, 2x6, etc.) with $129,344 in negative stock.\n\nAt many stores, lumber is sold at the register but not received into inventory. Is that how it works here?", answers: [["Yes - sold at POS but not received", "receiving_gap"], ["No - these are fully tracked", "investigate"], ["It varies", "partial"]] },
  { id: 'plywood', name: 'Plywood', items: 22, value: 69843, question: "Found 22 **plywood/sheet goods** with $69,843 negative.\n\nSheet goods are often sold at the register. How does your store handle plywood receiving?", answers: [["Sold at POS, not received", "receiving_gap"], ["Fully tracked", "investigate"]] },
  { id: 'drywall', name: 'Drywall', items: 17, value: 54977, question: "Found 17 **drywall** items with $54,977 negative.\n\nDrywall is heavy and often sold at the register. Is it received into inventory?", answers: [["No - sold at POS only", "receiving_gap"], ["Yes - fully tracked", "investigate"]] },
  { id: 'moulding', name: 'Moulding/Trim', items: 68, value: 51202, question: "Found 68 **moulding/trim** items with $51,202 negative.\n\nMoulding is often sold at the register in lengths. How is it tracked here?", answers: [["Sold at POS, not received", "receiving_gap"], ["Fully tracked", "investigate"], ["Some tracked, some not", "partial"]] },
  { id: 'staples', name: 'Landscape Staples', items: 2, value: 44025, question: "Found 2 **landscape staples** with $44,025 negative.\n\nThese are sometimes vendor-managed with lawn products. How are they handled?", answers: [["Vendor managed", "vendor_managed"], ["We track them", "investigate"]] },
  { id: 'boards_1x', name: '1X Boards', items: 44, value: 43814, question: "Found 44 **1X boards** (1x4, 1x6, pine boards, etc.) with $43,814 negative.\n\nAre these sold at the register without being received into inventory?", answers: [["Yes - POS only", "receiving_gap"], ["No - fully tracked", "investigate"]] },
  { id: 'lumber_4x', name: '4X4/6X6 Posts', items: 22, value: 24310, question: "Found 22 **post lumber items** (4x4, 6x6) with $24,310 negative.\n\nAre these handled the same as dimensional lumber?", answers: [["Yes - same as lumber", "receiving_gap"], ["No - tracked differently", "investigate"]] },
  { id: 'lawn_chem', name: 'Lawn Chemicals', items: 52, value: 21236, question: "Found 52 **lawn chemical** items (Scotts, Miracle-Gro, etc.) with $21,236 negative.\n\nLawn chemicals are often vendor-managed. Is that the case here?", answers: [["Yes - vendor managed", "vendor_managed"], ["No - we receive these", "investigate"]] },
  { id: 'beverages', name: 'Beverages', items: 81, value: 16130, question: "Found 81 **beverage** items with $16,130 negative.\n\nBeverages can expire or get damaged. Are expired drinks always written off properly?", answers: [["No - often tossed without write-off", "expiration"], ["Yes - always written off", "investigate"], ["High theft category", "theft"]] },
  { id: 'concrete', name: 'Concrete/Mortar', items: 14, value: 13934, question: "Found 14 **concrete/mortar** items with $13,934 negative.\n\nBagged concrete is heavy. Is it received into inventory?", answers: [["No - sold at POS only", "receiving_gap"], ["Yes - fully tracked", "investigate"]] },
  { id: 'pellets', name: 'Pellets', items: 20, value: 12328, question: "Found 20 **pellet** items with $12,328 negative.\n\nPellets are heavy and seasonal. How are they tracked?", answers: [["Sold at POS only", "receiving_gap"], ["Fully tracked", "investigate"]] },
  { id: 'soil', name: 'Soil/Mulch', items: 52, value: 10542, question: "Found 52 **soil/mulch** items with $10,542 negative.\n\nBagged soil and mulch - sold at register or fully tracked?", answers: [["Sold at POS", "receiving_gap"], ["Fully tracked", "investigate"]] },
  { id: 'filters', name: 'Filters', items: 45, value: 8608, question: "Found 45 **filter** items with $8,608 negative.\n\nFilters sometimes have damage or returns issues. What's happening here?", answers: [["Damage not written off", "expiration"], ["Returns issue", "investigate"], ["Fully tracked", "investigate"]] },
  { id: 'stone', name: 'Stone/Sand', items: 34, value: 7762, question: "Found 34 **stone/sand** items with $7,762 negative.\n\nBagged stone and sand - tracked or POS only?", answers: [["POS only", "receiving_gap"], ["Fully tracked", "investigate"]] },
  { id: 'deck', name: 'Deck Boards', items: 2, value: 6355, question: "Found 2 **deck boards** with $6,355 negative.\n\nDeck boards are typically sold at the register. Are these received?", answers: [["POS only", "receiving_gap"], ["Fully tracked", "investigate"]] },
  { id: 'power', name: 'Power Equipment', items: 4, value: 4605, question: "Found 4 **power equipment** items with $4,605 negative.\n\nThese are high-value items. What's most likely happening?", answers: [["Count error - verify", "investigate"], ["Returns not processed", "investigate"], ["Possible theft", "theft"]] },
  { id: 'batteries', name: 'Batteries', items: 22, value: 4199, question: "Found 22 **battery** items with $4,199 negative.\n\nBatteries are small and high-value - a common theft target. What do you think?", answers: [["Likely theft", "theft"], ["Receiving issue", "investigate"], ["Fully tracked", "investigate"]] },
  { id: 'keys', name: 'Keys', items: 86, value: 3623, question: "Found 86 **key** items with $3,623 negative.\n\nKeys are typically cut at the counter. Are they tracked?", answers: [["No - not tracked", "non_tracked"], ["Yes - tracked", "investigate"]] },
  { id: 'feed', name: 'Animal Feed', items: 1, value: 2882, question: "Found 1 **animal feed** item with $2,882 negative.\n\nBagged feed is heavy. Is it received into inventory?", answers: [["No - POS only", "receiving_gap"], ["Yes - tracked", "investigate"]] },
  { id: 'salt', name: 'Salt/Ice Melt', items: 19, value: 2484, question: "Found 19 **salt/ice melt** items with $2,484 negative.\n\nIce melt is seasonal and heavy. Is it received?", answers: [["No - POS only", "receiving_gap"], ["Yes - tracked", "investigate"]] },
  { id: 'snacks', name: 'Snacks', items: 32, value: 1862, question: "Found 32 **snack** items with $1,862 negative.\n\nSnacks can expire and are theft-prone. What's most likely?", answers: [["Expiration without write-off", "expiration"], ["Likely theft", "theft"], ["Investigate", "investigate"]] },
  { id: 'osb', name: 'OSB', items: 1, value: 1722, question: "Found 1 **OSB** item with $1,722 negative.\n\nIs OSB handled like plywood - sold but not received?", answers: [["Yes - same as plywood", "receiving_gap"], ["No - tracked", "investigate"]] },
  { id: 'rope', name: 'Rope/Chain', items: 35, value: 1721, question: "Found 35 **rope/chain** items with $1,721 negative.\n\nRope and chain are usually sold by the foot. Are these tracked?", answers: [["No - sold by foot", "non_tracked"], ["Yes - tracked by roll", "investigate"]] },
  { id: 'fasteners', name: 'Bin Fasteners', items: 36, value: 1180, question: "Found 36 **loose fasteners** (nuts, bolts) with $1,180 negative.\n\nAre these sold from bins without tracking?", answers: [["Yes - bin items", "non_tracked"], ["No - tracked", "investigate"]] },
  { id: 'tubing', name: 'Tubing', items: 15, value: 603, question: "Found 15 **tubing** items with $603 negative.\n\nIs tubing sold by the foot / cut to length?", answers: [["Yes - cut to order", "non_tracked"], ["No - fixed lengths", "investigate"]] },
];

const TOTAL_SHRINKAGE = 726749;

const formatCurrency = (n) => `$${n.toLocaleString()}`;
const formatNumber = (n) => n.toLocaleString();

const CLASSIFICATION_LABELS = {
  receiving_gap: { label: 'Receiving Gap', color: '#3b82f6', desc: 'Sold at POS, not received' },
  non_tracked: { label: 'Non-Tracked', color: '#10b981', desc: 'By design (bins, cut-to-length)' },
  vendor_managed: { label: 'Vendor Managed', color: '#8b5cf6', desc: 'Direct ship items' },
  expiration: { label: 'Expiration', color: '#f59e0b', desc: 'Expires without write-off' },
  theft: { label: 'Theft', color: '#ef4444', desc: 'Likely theft' },
  investigate: { label: 'Investigate', color: '#ef4444', desc: 'Needs investigation' },
  partial: { label: 'Partial', color: '#6b7280', desc: 'Mixed tracking' },
};

// Progress ring
const ProgressRing = ({ percent, size = 120 }) => {
  const strokeWidth = 8;
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (percent / 100) * circumference;
  
  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
        <circle stroke="#27272a" strokeWidth={strokeWidth} fill="none" r={radius} cx={size/2} cy={size/2} />
        <circle stroke="#10b981" strokeWidth={strokeWidth} fill="none" r={radius} cx={size/2} cy={size/2}
          strokeLinecap="round" style={{ strokeDasharray: circumference, strokeDashoffset: offset, transition: 'stroke-dashoffset 0.5s ease-out' }} />
      </svg>
      <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', textAlign: 'center' }}>
        <span style={{ display: 'block', fontSize: '1.5rem', fontWeight: 700, fontFamily: 'monospace', color: '#10b981' }}>{percent.toFixed(1)}%</span>
        <span style={{ fontSize: '0.65rem', color: '#71717a' }}>Explained</span>
      </div>
    </div>
  );
};

export default function ProfitSentinelConversational() {
  const [stage, setStage] = useState('upload'); // upload, diagnostic, complete
  const [currentIndex, setCurrentIndex] = useState(0);
  const [answers, setAnswers] = useState({});
  
  const handleFileUpload = () => {
    setStage('diagnostic');
  };
  
  const handleAnswer = (patternId, classification) => {
    setAnswers({ ...answers, [patternId]: classification });
    
    if (currentIndex < PATTERNS.length - 1) {
      setCurrentIndex(currentIndex + 1);
    } else {
      setStage('complete');
    }
  };
  
  const handleSkip = () => {
    const pattern = PATTERNS[currentIndex];
    handleAnswer(pattern.id, 'investigate');
  };
  
  // Calculate running totals
  const explainedValue = Object.entries(answers).reduce((sum, [id, classification]) => {
    if (['receiving_gap', 'non_tracked', 'vendor_managed', 'expiration'].includes(classification)) {
      const pattern = PATTERNS.find(p => p.id === id);
      return sum + (pattern?.value || 0);
    }
    return sum;
  }, 0);
  
  const reductionPercent = (explainedValue / TOTAL_SHRINKAGE) * 100;
  
  // Upload stage
  if (stage === 'upload') {
    return (
      <div style={{ minHeight: '100vh', fontFamily: "'Space Grotesk', system-ui", background: 'linear-gradient(135deg, #0a0a0f 0%, #0f1419 100%)', color: '#e4e4e7', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ maxWidth: 480, width: '100%', padding: '2rem', textAlign: 'center' }}>
          <div style={{ width: 64, height: 64, background: 'linear-gradient(135deg, #10b981, #059669)', borderRadius: 16, display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 1.5rem', boxShadow: '0 8px 32px rgba(16,185,129,0.3)' }}>
            <Zap size={28} color="white" />
          </div>
          <h1 style={{ fontSize: '2rem', fontWeight: 700, marginBottom: '0.5rem' }}>Profit Sentinel</h1>
          <p style={{ color: '#71717a', marginBottom: '2rem' }}>Conversational Shrinkage Diagnostic</p>
          
          <div onClick={handleFileUpload} style={{ border: '2px dashed #27272a', borderRadius: 12, padding: '3rem 2rem', cursor: 'pointer', transition: 'all 0.2s', background: 'rgba(24,24,27,0.5)' }}>
            <Upload size={40} style={{ color: '#52525b', marginBottom: '1rem' }} />
            <h3 style={{ marginBottom: '0.5rem' }}>Drop inventory file here</h3>
            <p style={{ color: '#71717a', fontSize: '0.9rem' }}>or click to browse</p>
          </div>
          
          <div style={{ display: 'flex', justifyContent: 'center', gap: '2rem', marginTop: '2rem', color: '#71717a', fontSize: '0.85rem' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}><Brain size={16} style={{ color: '#10b981' }} /> 27 patterns</span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}><MessageSquare size={16} style={{ color: '#10b981' }} /> Interactive</span>
          </div>
        </div>
      </div>
    );
  }
  
  // Diagnostic stage
  if (stage === 'diagnostic') {
    const pattern = PATTERNS[currentIndex];
    
    return (
      <div style={{ minHeight: '100vh', fontFamily: "'Space Grotesk', system-ui", background: 'linear-gradient(135deg, #0a0a0f 0%, #0f1419 100%)', color: '#e4e4e7' }}>
        {/* Header */}
        <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.75rem 1.5rem', background: 'rgba(24,24,27,0.9)', borderBottom: '1px solid #27272a' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <div style={{ width: 32, height: 32, background: 'linear-gradient(135deg, #10b981, #059669)', borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Zap size={18} color="white" />
            </div>
            <span style={{ fontWeight: 600 }}>Profit Sentinel</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '0.7rem', color: '#71717a' }}>Explained</div>
              <div style={{ fontFamily: 'monospace', fontWeight: 600, color: '#10b981' }}>{formatCurrency(explainedValue)}</div>
            </div>
            <ProgressRing percent={reductionPercent} size={48} />
          </div>
        </header>
        
        <main style={{ maxWidth: 700, margin: '0 auto', padding: '2rem 1.5rem' }}>
          {/* Progress bar */}
          <div style={{ marginBottom: '2rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', fontSize: '0.8rem', color: '#71717a' }}>
              <span>Pattern {currentIndex + 1} of {PATTERNS.length}</span>
              <span>{formatCurrency(pattern.value)}</span>
            </div>
            <div style={{ height: 4, background: '#27272a', borderRadius: 2 }}>
              <div style={{ height: '100%', width: `${((currentIndex + 1) / PATTERNS.length) * 100}%`, background: 'linear-gradient(90deg, #10b981, #3b82f6)', borderRadius: 2, transition: 'width 0.3s' }} />
            </div>
          </div>
          
          {/* Question card */}
          <div style={{ background: 'rgba(24,24,27,0.8)', border: '1px solid #27272a', borderRadius: 16, padding: '2rem', marginBottom: '1.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
              <HelpCircle size={20} style={{ color: '#10b981' }} />
              <span style={{ fontFamily: 'monospace', fontWeight: 600, color: '#10b981' }}>{pattern.name}</span>
              <span style={{ marginLeft: 'auto', fontSize: '0.85rem', color: '#71717a' }}>{pattern.items} items</span>
            </div>
            
            <p style={{ fontSize: '1.1rem', lineHeight: 1.6, marginBottom: '1.5rem', whiteSpace: 'pre-line' }}>
              {pattern.question.split('**').map((part, i) => 
                i % 2 === 1 ? <strong key={i} style={{ color: '#f59e0b' }}>{part}</strong> : part
              )}
            </p>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {pattern.answers.map(([label, classification], i) => {
                const classInfo = CLASSIFICATION_LABELS[classification] || CLASSIFICATION_LABELS.investigate;
                return (
                  <button key={i} onClick={() => handleAnswer(pattern.id, classification)} style={{
                    display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '1rem 1.25rem',
                    background: 'rgba(39,39,42,0.5)', border: '1px solid #3f3f46', borderRadius: 10,
                    color: '#e4e4e7', fontSize: '0.95rem', cursor: 'pointer', transition: 'all 0.15s', textAlign: 'left'
                  }}>
                    <span style={{ width: 8, height: 8, borderRadius: '50%', background: classInfo.color, flexShrink: 0 }} />
                    <span style={{ flex: 1 }}>{label}</span>
                    <ChevronRight size={18} style={{ color: '#52525b' }} />
                  </button>
                );
              })}
              
              {/* Skip / Investigate option - same prominence as other answers */}
              <button onClick={handleSkip} style={{
                display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '1rem 1.25rem',
                background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: 10,
                color: '#fca5a5', fontSize: '0.95rem', cursor: 'pointer', transition: 'all 0.15s', textAlign: 'left'
              }}>
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#ef4444', flexShrink: 0 }} />
                <span style={{ flex: 1 }}>I'm not sure - investigate these</span>
                <ChevronRight size={18} style={{ color: '#ef4444' }} />
              </button>
            </div>
          </div>
          
          {/* Running total */}
          <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center' }}>
            <div style={{ fontSize: '0.85rem', color: '#71717a' }}>
              Running total: <span style={{ color: '#10b981', fontFamily: 'monospace' }}>{reductionPercent.toFixed(1)}%</span> explained
            </div>
          </div>
          
          {/* Answered patterns */}
          {Object.keys(answers).length > 0 && (
            <div style={{ marginTop: '2rem', paddingTop: '1.5rem', borderTop: '1px solid #27272a' }}>
              <h4 style={{ fontSize: '0.85rem', color: '#71717a', marginBottom: '0.75rem' }}>Answered ({Object.keys(answers).length})</h4>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                {Object.entries(answers).map(([id, classification]) => {
                  const p = PATTERNS.find(x => x.id === id);
                  const classInfo = CLASSIFICATION_LABELS[classification];
                  return (
                    <span key={id} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', padding: '0.3rem 0.6rem', background: 'rgba(39,39,42,0.5)', borderRadius: 6, fontSize: '0.75rem' }}>
                      <span style={{ width: 6, height: 6, borderRadius: '50%', background: classInfo?.color || '#6b7280' }} />
                      {p?.name}
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
  
  // Complete stage
  const byClass = {};
  Object.entries(answers).forEach(([id, classification]) => {
    if (!byClass[classification]) byClass[classification] = { items: 0, value: 0, patterns: [] };
    const p = PATTERNS.find(x => x.id === id);
    byClass[classification].items += p?.items || 0;
    byClass[classification].value += p?.value || 0;
    byClass[classification].patterns.push(p?.name);
  });
  
  const unexplainedValue = TOTAL_SHRINKAGE - explainedValue;
  
  return (
    <div style={{ minHeight: '100vh', fontFamily: "'Space Grotesk', system-ui", background: 'linear-gradient(135deg, #0a0a0f 0%, #0f1419 100%)', color: '#e4e4e7' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.75rem 1.5rem', background: 'rgba(24,24,27,0.9)', borderBottom: '1px solid #27272a' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <div style={{ width: 32, height: 32, background: 'linear-gradient(135deg, #10b981, #059669)', borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Zap size={18} color="white" />
          </div>
          <span style={{ fontWeight: 600 }}>Profit Sentinel</span>
        </div>
        <button onClick={() => { setStage('upload'); setCurrentIndex(0); setAnswers({}); }} style={{ padding: '0.4rem 0.8rem', background: '#27272a', border: '1px solid #3f3f46', borderRadius: 6, color: '#e4e4e7', fontSize: '0.8rem', cursor: 'pointer' }}>
          New Analysis
        </button>
      </header>
      
      <main style={{ maxWidth: 800, margin: '0 auto', padding: '2rem 1.5rem' }}>
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <CheckCircle size={48} style={{ color: '#10b981', marginBottom: '1rem' }} />
          <h1 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>Diagnostic Complete</h1>
          <p style={{ color: '#71717a' }}>{PATTERNS.length} patterns reviewed</p>
        </div>
        
        {/* Big numbers */}
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '2rem', marginBottom: '2rem', flexWrap: 'wrap' }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: '#71717a' }}>Apparent Shrinkage</div>
            <div style={{ fontSize: '2rem', fontWeight: 700, fontFamily: 'monospace', color: '#ef4444' }}>{formatCurrency(TOTAL_SHRINKAGE)}</div>
          </div>
          <ArrowRight size={24} style={{ color: '#52525b' }} />
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: '#71717a' }}>To Investigate</div>
            <div style={{ fontSize: '2rem', fontWeight: 700, fontFamily: 'monospace', color: '#10b981' }}>{formatCurrency(unexplainedValue)}</div>
          </div>
          <ProgressRing percent={reductionPercent} size={100} />
        </div>
        
        {/* Breakdown */}
        <div style={{ background: 'rgba(24,24,27,0.6)', border: '1px solid #27272a', borderRadius: 12, padding: '1.5rem', marginBottom: '1.5rem' }}>
          <h3 style={{ fontSize: '0.9rem', marginBottom: '1rem' }}>Your Answers</h3>
          
          {Object.entries(byClass).map(([classification, data]) => {
            const classInfo = CLASSIFICATION_LABELS[classification];
            const isExplained = ['receiving_gap', 'non_tracked', 'vendor_managed', 'expiration'].includes(classification);
            
            return (
              <div key={classification} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '0.75rem 0', borderBottom: '1px solid #27272a' }}>
                <span style={{ width: 10, height: 10, borderRadius: '50%', background: classInfo?.color || '#6b7280' }} />
                <span style={{ flex: 1 }}>{classInfo?.label || classification}</span>
                <span style={{ fontSize: '0.85rem', color: '#71717a' }}>{data.patterns.length} patterns</span>
                <span style={{ fontFamily: 'monospace', fontWeight: 500, color: isExplained ? '#10b981' : '#ef4444' }}>{formatCurrency(data.value)}</span>
              </div>
            );
          })}
        </div>
        
        {/* Summary */}
        <div style={{ background: 'linear-gradient(135deg, rgba(16,185,129,0.1), rgba(59,130,246,0.1))', border: '1px solid rgba(16,185,129,0.2)', borderRadius: 12, padding: '1.5rem', textAlign: 'center' }}>
          <p style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>
            You reduced apparent shrinkage by <strong style={{ color: '#10b981' }}>{formatCurrency(explainedValue)}</strong>
          </p>
          <p style={{ color: '#71717a' }}>
            That's <strong>{reductionPercent.toFixed(1)}%</strong> identified as process issues, not theft.
          </p>
        </div>
      </main>
    </div>
  );
}
