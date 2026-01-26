import React, { useState, useCallback, useRef } from 'react';
import { Upload, AlertTriangle, CheckCircle, DollarSign, Package, TrendingDown, MessageSquare, ChevronRight, X, FileText, Zap, Brain, BarChart3 } from 'lucide-react';

// Mock data for demo (would come from backend in production)
const DEMO_ANALYSIS = {
  totalItems: 156139,
  negativeItems: 3996,
  totalShrinkage: 726749,
  explained: 548453,
  unexplained: 178295,
  reductionPercent: 75.5,
  breakdown: {
    nonTracked: { items: 315, value: 46306, label: 'Non-Tracked', desc: 'Bins, cut-to-length' },
    receivingGap: { items: 322, value: 423418, label: 'Receiving Gap', desc: 'POS only, not received' },
    expiration: { items: 74, value: 13853, label: 'Expiration', desc: 'Beverages, perishables' },
    vendorManaged: { items: 53, value: 64877, label: 'Vendor Managed', desc: 'Direct ship items' },
    unexplained: { items: 3232, value: 178295, label: 'Unexplained', desc: 'Needs investigation' }
  },
  topUnexplained: [
    { sku: '1X3X8SPRUCE', desc: '1X3X8 SPRUCE', stock: -950, value: 1900 },
    { sku: 'U1L10', desc: 'U1L-10 BATTERY', stock: -53, value: 1849 },
    { sku: 'G8153900', desc: 'FILTER AC ULTRA 16X25X1IN', stock: -106, value: 1495 },
    { sku: 'HUSPREMIX50:1G', desc: 'HUS PRE MIX 50:1 1GAL', stock: -82, value: 1476 },
    { sku: 'G0853499', desc: 'GENERATOR PORTABLE 6500W', stock: -2, value: 1465 },
    { sku: '117803', desc: 'DR PEPPER 20-OZ', stock: -1239, value: 1425 },
    { sku: 'G3019346', desc: 'GENERATOR 5500W ELEC START', stock: -2, value: 1409 },
    { sku: 'G3019379', desc: 'GENERATOR 5000W RECOIL', stock: -2, value: 1257 },
    { sku: 'G7377088', desc: 'RAISED BED GARDEN SOIL', stock: -135, value: 1250 },
    { sku: '125B', desc: 'HUSQVARNA 28CC BLOWER', stock: -7, value: 1117 },
  ],
  questions: [
    { id: 'q1', pattern: 'SPRUCE', question: 'How does your store handle SPRUCE boards?', value: 12500, items: 45 },
    { id: 'q2', pattern: 'BATTERY', question: 'Are batteries tracked through receiving?', value: 8200, items: 120 },
    { id: 'q3', pattern: 'FILTER', question: 'How are HVAC filters managed?', value: 6800, items: 89 },
  ],
  rulesApplied: 72
};

const SUGGESTED_ANSWERS = {
  'q1': ['Sold at POS but not received into inventory', 'Fully tracked', 'Varies by size'],
  'q2': ['Fully tracked', 'High theft item - needs investigation', 'Vendor managed'],
  'q3': ['Fully tracked', 'Seasonal damage not written off', 'Returns not processed'],
};

const formatCurrency = (num) => `$${num.toLocaleString()}`;
const formatNumber = (num) => num.toLocaleString();

// Stat Card Component
const StatCard = ({ icon: Icon, label, value, subValue, color, large }) => (
  <div className={`stat-card ${large ? 'large' : ''} ${color}`}>
    <div className="stat-icon">
      <Icon size={large ? 28 : 22} />
    </div>
    <div className="stat-content">
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
      {subValue && <div className="stat-sub">{subValue}</div>}
    </div>
  </div>
);

// Progress Ring Component
const ProgressRing = ({ percent, size = 160, strokeWidth = 12 }) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (percent / 100) * circumference;
  
  return (
    <div className="progress-ring-container">
      <svg width={size} height={size} className="progress-ring">
        <circle
          className="progress-ring-bg"
          strokeWidth={strokeWidth}
          fill="none"
          r={radius}
          cx={size / 2}
          cy={size / 2}
        />
        <circle
          className="progress-ring-fill"
          strokeWidth={strokeWidth}
          fill="none"
          r={radius}
          cx={size / 2}
          cy={size / 2}
          style={{
            strokeDasharray: circumference,
            strokeDashoffset: offset
          }}
        />
      </svg>
      <div className="progress-ring-text">
        <span className="progress-percent">{percent.toFixed(1)}%</span>
        <span className="progress-label">Reduction</span>
      </div>
    </div>
  );
};

// Breakdown Bar Component
const BreakdownBar = ({ data }) => {
  const total = Object.values(data).reduce((sum, d) => sum + d.value, 0);
  const colors = ['#10b981', '#3b82f6', '#f59e0b', '#8b5cf6', '#ef4444'];
  
  return (
    <div className="breakdown-section">
      <div className="breakdown-bar">
        {Object.entries(data).map(([key, d], i) => (
          <div
            key={key}
            className="breakdown-segment"
            style={{
              width: `${(d.value / total) * 100}%`,
              backgroundColor: colors[i]
            }}
            title={`${d.label}: ${formatCurrency(d.value)}`}
          />
        ))}
      </div>
      <div className="breakdown-legend">
        {Object.entries(data).map(([key, d], i) => (
          <div key={key} className="legend-item">
            <span className="legend-dot" style={{ backgroundColor: colors[i] }} />
            <span className="legend-label">{d.label}</span>
            <span className="legend-value">{formatCurrency(d.value)}</span>
            <span className="legend-items">({d.items} items)</span>
          </div>
        ))}
      </div>
    </div>
  );
};

// Question Card Component
const QuestionCard = ({ question, onAnswer, isActive }) => {
  const [answer, setAnswer] = useState('');
  const suggestions = SUGGESTED_ANSWERS[question.id] || [];
  
  return (
    <div className={`question-card ${isActive ? 'active' : ''}`}>
      <div className="question-header">
        <MessageSquare size={18} />
        <span className="question-pattern">{question.pattern}</span>
        <span className="question-value">{formatCurrency(question.value)}</span>
      </div>
      <p className="question-text">{question.question}</p>
      <div className="question-suggestions">
        {suggestions.map((s, i) => (
          <button
            key={i}
            className="suggestion-btn"
            onClick={() => onAnswer(question.id, s)}
          >
            {s}
          </button>
        ))}
      </div>
      <div className="question-custom">
        <input
          type="text"
          placeholder="Or type your answer..."
          value={answer}
          onChange={(e) => setAnswer(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && answer && onAnswer(question.id, answer)}
        />
        <button 
          className="submit-btn"
          onClick={() => answer && onAnswer(question.id, answer)}
          disabled={!answer}
        >
          <ChevronRight size={18} />
        </button>
      </div>
    </div>
  );
};

// Top Items Table Component
const TopItemsTable = ({ items }) => (
  <div className="items-table-container">
    <table className="items-table">
      <thead>
        <tr>
          <th>SKU</th>
          <th>Description</th>
          <th className="right">Stock</th>
          <th className="right">Value</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item, i) => (
          <tr key={i} className={i < 3 ? 'highlight' : ''}>
            <td className="sku">{item.sku}</td>
            <td className="desc">{item.desc}</td>
            <td className="right stock">{formatNumber(item.stock)}</td>
            <td className="right value">{formatCurrency(item.value)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

// Main App Component
export default function ProfitSentinelDashboard() {
  const [stage, setStage] = useState('upload'); // upload, analyzing, results
  const [analysis, setAnalysis] = useState(null);
  const [activeQuestion, setActiveQuestion] = useState(0);
  const [learnedRules, setLearnedRules] = useState([]);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);
  
  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);
  
  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, []);
  
  const handleFile = (file) => {
    setStage('analyzing');
    // Simulate analysis time
    setTimeout(() => {
      setAnalysis(DEMO_ANALYSIS);
      setStage('results');
    }, 2500);
  };
  
  const handleAnswer = (questionId, answer) => {
    const question = analysis.questions.find(q => q.id === questionId);
    setLearnedRules([...learnedRules, { pattern: question.pattern, answer }]);
    
    if (activeQuestion < analysis.questions.length - 1) {
      setActiveQuestion(activeQuestion + 1);
    }
  };
  
  // Upload Stage
  if (stage === 'upload') {
    return (
      <div className="app upload-stage">
        <div className="upload-container">
          <div className="brand">
            <div className="brand-icon">
              <Zap size={32} />
            </div>
            <h1>Profit Sentinel</h1>
            <p className="tagline">AI-Powered Shrinkage Analysis</p>
          </div>
          
          <div 
            className={`dropzone ${dragActive ? 'active' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,.xlsx"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
              hidden
            />
            <Upload size={48} className="upload-icon" />
            <h3>Drop your inventory file here</h3>
            <p>or click to browse</p>
            <span className="file-types">CSV or Excel files supported</span>
          </div>
          
          <div className="features">
            <div className="feature">
              <Brain size={24} />
              <span>950+ knowledge facts</span>
            </div>
            <div className="feature">
              <BarChart3 size={24} />
              <span>72 business rules</span>
            </div>
            <div className="feature">
              <TrendingDown size={24} />
              <span>75%+ shrinkage reduction</span>
            </div>
          </div>
        </div>
        
        <style>{`
          @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
          
          * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
          }
          
          .app {
            min-height: 100vh;
            font-family: 'Space Grotesk', sans-serif;
            background: #0a0a0f;
            color: #e4e4e7;
          }
          
          .upload-stage {
            display: flex;
            align-items: center;
            justify-content: center;
            background: 
              radial-gradient(ellipse at 20% 20%, rgba(16, 185, 129, 0.08) 0%, transparent 50%),
              radial-gradient(ellipse at 80% 80%, rgba(59, 130, 246, 0.08) 0%, transparent 50%),
              #0a0a0f;
          }
          
          .upload-container {
            max-width: 560px;
            width: 100%;
            padding: 2rem;
          }
          
          .brand {
            text-align: center;
            margin-bottom: 3rem;
          }
          
          .brand-icon {
            width: 72px;
            height: 72px;
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 1.5rem;
            box-shadow: 0 8px 32px rgba(16, 185, 129, 0.3);
          }
          
          .brand-icon svg {
            color: white;
          }
          
          .brand h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #fff 0%, #a1a1aa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
          }
          
          .tagline {
            color: #71717a;
            font-size: 1.1rem;
          }
          
          .dropzone {
            border: 2px dashed #27272a;
            border-radius: 16px;
            padding: 4rem 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
            background: rgba(24, 24, 27, 0.5);
          }
          
          .dropzone:hover, .dropzone.active {
            border-color: #10b981;
            background: rgba(16, 185, 129, 0.05);
          }
          
          .upload-icon {
            color: #52525b;
            margin-bottom: 1.5rem;
            transition: color 0.2s;
          }
          
          .dropzone:hover .upload-icon, .dropzone.active .upload-icon {
            color: #10b981;
          }
          
          .dropzone h3 {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
          }
          
          .dropzone p {
            color: #71717a;
            margin-bottom: 1rem;
          }
          
          .file-types {
            font-size: 0.85rem;
            color: #52525b;
          }
          
          .features {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-top: 3rem;
          }
          
          .feature {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: #71717a;
            font-size: 0.9rem;
          }
          
          .feature svg {
            color: #10b981;
          }
        `}</style>
      </div>
    );
  }
  
  // Analyzing Stage
  if (stage === 'analyzing') {
    return (
      <div className="app analyzing-stage">
        <div className="analyzing-container">
          <div className="spinner" />
          <h2>Analyzing Inventory...</h2>
          <p>Applying 72 business rules to 156K items</p>
          <div className="progress-bar">
            <div className="progress-fill" />
          </div>
        </div>
        
        <style>{`
          @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
          
          * { margin: 0; padding: 0; box-sizing: border-box; }
          
          .app {
            min-height: 100vh;
            font-family: 'Space Grotesk', sans-serif;
            background: #0a0a0f;
            color: #e4e4e7;
          }
          
          .analyzing-stage {
            display: flex;
            align-items: center;
            justify-content: center;
          }
          
          .analyzing-container {
            text-align: center;
          }
          
          .spinner {
            width: 64px;
            height: 64px;
            border: 4px solid #27272a;
            border-top-color: #10b981;
            border-radius: 50%;
            margin: 0 auto 2rem;
            animation: spin 1s linear infinite;
          }
          
          @keyframes spin {
            to { transform: rotate(360deg); }
          }
          
          .analyzing-container h2 {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
          }
          
          .analyzing-container p {
            color: #71717a;
            margin-bottom: 2rem;
          }
          
          .progress-bar {
            width: 300px;
            height: 4px;
            background: #27272a;
            border-radius: 2px;
            overflow: hidden;
          }
          
          .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #10b981, #3b82f6);
            animation: progress 2.5s ease-out forwards;
          }
          
          @keyframes progress {
            from { width: 0%; }
            to { width: 100%; }
          }
        `}</style>
      </div>
    );
  }
  
  // Results Stage
  return (
    <div className="app results-stage">
      <header className="header">
        <div className="header-left">
          <div className="logo">
            <Zap size={24} />
          </div>
          <h1>Profit Sentinel</h1>
        </div>
        <div className="header-right">
          <span className="rules-badge">
            <Brain size={16} />
            {analysis.rulesApplied + learnedRules.length} Rules Applied
          </span>
          <button className="new-analysis-btn" onClick={() => setStage('upload')}>
            <Upload size={16} />
            New Analysis
          </button>
        </div>
      </header>
      
      <main className="main">
        <section className="summary-section">
          <div className="summary-grid">
            <div className="summary-left">
              <h2>Shrinkage Analysis</h2>
              <div className="big-numbers">
                <div className="big-number">
                  <span className="label">Apparent Shrinkage</span>
                  <span className="value red">{formatCurrency(analysis.totalShrinkage)}</span>
                </div>
                <div className="arrow">→</div>
                <div className="big-number">
                  <span className="label">Actual to Investigate</span>
                  <span className="value green">{formatCurrency(analysis.unexplained)}</span>
                </div>
              </div>
              <div className="mini-stats">
                <StatCard icon={Package} label="Items Analyzed" value={formatNumber(analysis.totalItems)} color="blue" />
                <StatCard icon={AlertTriangle} label="Negative Stock" value={formatNumber(analysis.negativeItems)} color="yellow" />
                <StatCard icon={CheckCircle} label="Issues Explained" value={formatNumber(analysis.negativeItems - analysis.breakdown.unexplained.items)} color="green" />
              </div>
            </div>
            <div className="summary-right">
              <ProgressRing percent={analysis.reductionPercent} />
              <p className="reduction-text">
                Saved <strong>{formatCurrency(analysis.explained)}</strong> in investigation time
              </p>
            </div>
          </div>
        </section>
        
        <section className="breakdown-section-wrapper">
          <h3>Breakdown by Category</h3>
          <BreakdownBar data={analysis.breakdown} />
        </section>
        
        <div className="two-columns">
          <section className="questions-section">
            <h3>
              <MessageSquare size={20} />
              Diagnostic Questions
            </h3>
            <p className="section-desc">Answer questions to teach the system about your store</p>
            
            {analysis.questions.map((q, i) => (
              <QuestionCard
                key={q.id}
                question={q}
                onAnswer={handleAnswer}
                isActive={i === activeQuestion}
              />
            ))}
            
            {learnedRules.length > 0 && (
              <div className="learned-rules">
                <h4>Learned Rules</h4>
                {learnedRules.map((rule, i) => (
                  <div key={i} className="learned-rule">
                    <CheckCircle size={16} />
                    <strong>{rule.pattern}</strong>: {rule.answer}
                  </div>
                ))}
              </div>
            )}
          </section>
          
          <section className="items-section">
            <h3>
              <AlertTriangle size={20} />
              Top Unexplained Items
            </h3>
            <p className="section-desc">These items need investigation</p>
            <TopItemsTable items={analysis.topUnexplained} />
          </section>
        </div>
      </main>
      
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        .app {
          min-height: 100vh;
          font-family: 'Space Grotesk', sans-serif;
          background: #0a0a0f;
          color: #e4e4e7;
        }
        
        .results-stage {
          background: 
            radial-gradient(ellipse at 0% 0%, rgba(16, 185, 129, 0.05) 0%, transparent 50%),
            #0a0a0f;
        }
        
        .header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 1rem 2rem;
          background: rgba(24, 24, 27, 0.8);
          border-bottom: 1px solid #27272a;
          backdrop-filter: blur(8px);
          position: sticky;
          top: 0;
          z-index: 100;
        }
        
        .header-left {
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }
        
        .logo {
          width: 40px;
          height: 40px;
          background: linear-gradient(135deg, #10b981 0%, #059669 100%);
          border-radius: 10px;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        
        .logo svg { color: white; }
        
        .header h1 {
          font-size: 1.25rem;
          font-weight: 600;
        }
        
        .header-right {
          display: flex;
          align-items: center;
          gap: 1rem;
        }
        
        .rules-badge {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.5rem 1rem;
          background: rgba(16, 185, 129, 0.1);
          border: 1px solid rgba(16, 185, 129, 0.2);
          border-radius: 8px;
          font-size: 0.85rem;
          color: #10b981;
        }
        
        .new-analysis-btn {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.5rem 1rem;
          background: #27272a;
          border: 1px solid #3f3f46;
          border-radius: 8px;
          color: #e4e4e7;
          font-family: inherit;
          font-size: 0.85rem;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .new-analysis-btn:hover {
          background: #3f3f46;
        }
        
        .main {
          max-width: 1400px;
          margin: 0 auto;
          padding: 2rem;
        }
        
        .summary-section {
          background: linear-gradient(135deg, rgba(24, 24, 27, 0.8) 0%, rgba(24, 24, 27, 0.4) 100%);
          border: 1px solid #27272a;
          border-radius: 16px;
          padding: 2rem;
          margin-bottom: 2rem;
        }
        
        .summary-grid {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 3rem;
        }
        
        .summary-left h2 {
          font-size: 1.1rem;
          color: #71717a;
          font-weight: 500;
          margin-bottom: 1.5rem;
        }
        
        .big-numbers {
          display: flex;
          align-items: center;
          gap: 2rem;
          margin-bottom: 2rem;
        }
        
        .big-number .label {
          display: block;
          font-size: 0.85rem;
          color: #71717a;
          margin-bottom: 0.25rem;
        }
        
        .big-number .value {
          font-size: 2.5rem;
          font-weight: 700;
          font-family: 'JetBrains Mono', monospace;
        }
        
        .big-number .value.red { color: #ef4444; }
        .big-number .value.green { color: #10b981; }
        
        .arrow {
          font-size: 2rem;
          color: #52525b;
        }
        
        .mini-stats {
          display: flex;
          gap: 1rem;
        }
        
        .stat-card {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 0.75rem 1rem;
          background: rgba(39, 39, 42, 0.5);
          border-radius: 10px;
          border: 1px solid #3f3f46;
        }
        
        .stat-icon {
          width: 40px;
          height: 40px;
          border-radius: 8px;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        
        .stat-card.blue .stat-icon { background: rgba(59, 130, 246, 0.15); color: #3b82f6; }
        .stat-card.yellow .stat-icon { background: rgba(245, 158, 11, 0.15); color: #f59e0b; }
        .stat-card.green .stat-icon { background: rgba(16, 185, 129, 0.15); color: #10b981; }
        
        .stat-value {
          font-size: 1.25rem;
          font-weight: 600;
          font-family: 'JetBrains Mono', monospace;
        }
        
        .stat-label {
          font-size: 0.8rem;
          color: #71717a;
        }
        
        .summary-right {
          text-align: center;
        }
        
        .progress-ring-container {
          position: relative;
          display: inline-block;
        }
        
        .progress-ring {
          transform: rotate(-90deg);
        }
        
        .progress-ring-bg {
          stroke: #27272a;
        }
        
        .progress-ring-fill {
          stroke: #10b981;
          stroke-linecap: round;
          transition: stroke-dashoffset 1s ease-out;
        }
        
        .progress-ring-text {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          text-align: center;
        }
        
        .progress-percent {
          display: block;
          font-size: 2rem;
          font-weight: 700;
          font-family: 'JetBrains Mono', monospace;
          color: #10b981;
        }
        
        .progress-label {
          font-size: 0.85rem;
          color: #71717a;
        }
        
        .reduction-text {
          margin-top: 1rem;
          color: #71717a;
        }
        
        .reduction-text strong {
          color: #10b981;
        }
        
        .breakdown-section-wrapper {
          background: rgba(24, 24, 27, 0.5);
          border: 1px solid #27272a;
          border-radius: 16px;
          padding: 1.5rem 2rem;
          margin-bottom: 2rem;
        }
        
        .breakdown-section-wrapper h3 {
          font-size: 1rem;
          font-weight: 500;
          margin-bottom: 1.5rem;
        }
        
        .breakdown-bar {
          display: flex;
          height: 24px;
          border-radius: 12px;
          overflow: hidden;
          margin-bottom: 1.5rem;
        }
        
        .breakdown-segment {
          transition: opacity 0.2s;
        }
        
        .breakdown-segment:hover {
          opacity: 0.8;
        }
        
        .breakdown-legend {
          display: flex;
          flex-wrap: wrap;
          gap: 1.5rem;
        }
        
        .legend-item {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.9rem;
        }
        
        .legend-dot {
          width: 12px;
          height: 12px;
          border-radius: 3px;
        }
        
        .legend-label {
          color: #a1a1aa;
        }
        
        .legend-value {
          font-family: 'JetBrains Mono', monospace;
          font-weight: 500;
        }
        
        .legend-items {
          color: #52525b;
          font-size: 0.8rem;
        }
        
        .two-columns {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 2rem;
        }
        
        .questions-section, .items-section {
          background: rgba(24, 24, 27, 0.5);
          border: 1px solid #27272a;
          border-radius: 16px;
          padding: 1.5rem;
        }
        
        .questions-section h3, .items-section h3 {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 1rem;
          font-weight: 500;
          margin-bottom: 0.5rem;
        }
        
        .section-desc {
          color: #71717a;
          font-size: 0.85rem;
          margin-bottom: 1.5rem;
        }
        
        .question-card {
          background: #18181b;
          border: 1px solid #27272a;
          border-radius: 12px;
          padding: 1.25rem;
          margin-bottom: 1rem;
          opacity: 0.6;
          transition: all 0.2s;
        }
        
        .question-card.active {
          opacity: 1;
          border-color: #10b981;
          box-shadow: 0 0 0 1px rgba(16, 185, 129, 0.2);
        }
        
        .question-header {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-bottom: 0.75rem;
          color: #71717a;
        }
        
        .question-pattern {
          font-family: 'JetBrains Mono', monospace;
          font-weight: 500;
          color: #10b981;
        }
        
        .question-value {
          margin-left: auto;
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.85rem;
        }
        
        .question-text {
          font-weight: 500;
          margin-bottom: 1rem;
        }
        
        .question-suggestions {
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
          margin-bottom: 1rem;
        }
        
        .suggestion-btn {
          padding: 0.5rem 0.75rem;
          background: rgba(39, 39, 42, 0.5);
          border: 1px solid #3f3f46;
          border-radius: 6px;
          color: #a1a1aa;
          font-family: inherit;
          font-size: 0.8rem;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .suggestion-btn:hover {
          background: rgba(16, 185, 129, 0.1);
          border-color: #10b981;
          color: #10b981;
        }
        
        .question-custom {
          display: flex;
          gap: 0.5rem;
        }
        
        .question-custom input {
          flex: 1;
          padding: 0.5rem 0.75rem;
          background: #27272a;
          border: 1px solid #3f3f46;
          border-radius: 6px;
          color: #e4e4e7;
          font-family: inherit;
          font-size: 0.85rem;
        }
        
        .question-custom input:focus {
          outline: none;
          border-color: #10b981;
        }
        
        .submit-btn {
          padding: 0.5rem;
          background: #10b981;
          border: none;
          border-radius: 6px;
          color: white;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .submit-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        
        .submit-btn:not(:disabled):hover {
          background: #059669;
        }
        
        .learned-rules {
          margin-top: 1.5rem;
          padding-top: 1.5rem;
          border-top: 1px solid #27272a;
        }
        
        .learned-rules h4 {
          font-size: 0.9rem;
          font-weight: 500;
          margin-bottom: 0.75rem;
          color: #71717a;
        }
        
        .learned-rule {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.5rem 0;
          font-size: 0.85rem;
          color: #10b981;
        }
        
        .items-table-container {
          overflow-x: auto;
        }
        
        .items-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 0.85rem;
        }
        
        .items-table th {
          text-align: left;
          padding: 0.75rem;
          color: #71717a;
          font-weight: 500;
          border-bottom: 1px solid #27272a;
        }
        
        .items-table th.right {
          text-align: right;
        }
        
        .items-table td {
          padding: 0.75rem;
          border-bottom: 1px solid #27272a;
        }
        
        .items-table td.right {
          text-align: right;
        }
        
        .items-table tr.highlight td {
          background: rgba(239, 68, 68, 0.05);
        }
        
        .items-table .sku {
          font-family: 'JetBrains Mono', monospace;
          font-weight: 500;
          color: #f59e0b;
        }
        
        .items-table .desc {
          color: #a1a1aa;
        }
        
        .items-table .stock {
          font-family: 'JetBrains Mono', monospace;
          color: #ef4444;
        }
        
        .items-table .value {
          font-family: 'JetBrains Mono', monospace;
          font-weight: 500;
        }
        
        @media (max-width: 1024px) {
          .two-columns {
            grid-template-columns: 1fr;
          }
          
          .summary-grid {
            flex-direction: column;
            text-align: center;
          }
          
          .big-numbers {
            flex-direction: column;
            gap: 1rem;
          }
          
          .arrow {
            transform: rotate(90deg);
          }
          
          .mini-stats {
            flex-wrap: wrap;
            justify-content: center;
          }
        }
      `}</style>
    </div>
  );
}
