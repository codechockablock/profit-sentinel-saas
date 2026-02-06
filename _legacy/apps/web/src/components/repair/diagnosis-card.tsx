'use client'

interface Hypothesis {
  category_slug: string
  category_name: string
  probability: number
  explanation?: string
  icon?: string
}

interface DiagnoseResponse {
  problem_id: string
  status: string
  hypotheses: Hypothesis[]
  top_hypothesis: Hypothesis
  confidence: number
  entropy: number
  needs_more_info: boolean
  follow_up_questions: string[]
  diy_feasible?: boolean
  professional_recommended?: boolean
}

interface DiagnosisCardProps {
  diagnosis: DiagnoseResponse
  onGetSolution: () => void
}

function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.8) return 'text-emerald-400'
  if (confidence >= 0.6) return 'text-amber-400'
  return 'text-red-400'
}

function getConfidenceLabel(confidence: number): string {
  if (confidence >= 0.8) return 'High Confidence'
  if (confidence >= 0.6) return 'Medium Confidence'
  return 'Low Confidence'
}

function getProbabilityWidth(prob: number): string {
  return `${Math.round(prob * 100)}%`
}

function getCategoryIcon(slug: string): string {
  const icons: Record<string, string> = {
    plumbing: '\u{1F6BF}', // shower
    electrical: '\u{26A1}', // lightning
    hvac: '\u{1F321}', // thermometer
    carpentry: '\u{1FAB5}', // wood
    painting: '\u{1F3A8}', // palette
    flooring: '\u{1F3E0}', // house
    roofing: '\u{1F3D7}', // building
    appliances: '\u{1F50C}', // plug
    outdoor: '\u{1F33F}', // herb
    automotive: '\u{1F697}', // car
  }

  // Try to match parent category
  const parent = slug.split('-')[0]
  return icons[parent] || icons[slug] || '\u{1F527}' // wrench default
}

export function DiagnosisCard({ diagnosis, onGetSolution }: DiagnosisCardProps) {
  const { top_hypothesis, hypotheses, confidence, needs_more_info, follow_up_questions, diy_feasible } = diagnosis

  return (
    <div className="bg-slate-800/50 rounded-2xl border border-slate-700/50 overflow-hidden">
      {/* Top Diagnosis */}
      <div className="p-6 bg-gradient-to-r from-emerald-500/10 to-blue-500/10 border-b border-slate-700/50">
        <div className="flex items-start gap-4">
          <div className="text-4xl">
            {getCategoryIcon(top_hypothesis.category_slug)}
          </div>
          <div className="flex-1">
            <h2 className="text-2xl font-bold text-white mb-1">
              {top_hypothesis.category_name}
            </h2>
            {top_hypothesis.explanation && (
              <p className="text-slate-400 text-sm">
                {top_hypothesis.explanation}
              </p>
            )}
          </div>
        </div>

        {/* Confidence Badge */}
        <div className="mt-4 flex items-center gap-3">
          <div className={`text-lg font-bold ${getConfidenceColor(confidence)}`}>
            {Math.round(confidence * 100)}%
          </div>
          <div className={`px-3 py-1 rounded-full text-sm font-medium ${getConfidenceColor(confidence)} bg-current/10`}>
            {getConfidenceLabel(confidence)}
          </div>
          {diy_feasible !== undefined && (
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${diy_feasible ? 'text-emerald-400 bg-emerald-500/10' : 'text-amber-400 bg-amber-500/10'}`}>
              {diy_feasible ? 'DIY Friendly' : 'May Need Pro'}
            </div>
          )}
        </div>
      </div>

      {/* Other Hypotheses */}
      {hypotheses.length > 1 && (
        <div className="p-6 border-b border-slate-700/50">
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-4">
            Other Possibilities
          </h3>
          <div className="space-y-3">
            {hypotheses.slice(1, 4).map((h, idx) => (
              <div key={idx} className="flex items-center gap-3">
                <span className="text-xl">{getCategoryIcon(h.category_slug)}</span>
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-slate-300 text-sm">{h.category_name}</span>
                    <span className="text-slate-500 text-xs">
                      {Math.round(h.probability * 100)}%
                    </span>
                  </div>
                  <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-slate-500 rounded-full transition-all duration-500"
                      style={{ width: getProbabilityWidth(h.probability) }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Follow-up Questions */}
      {needs_more_info && follow_up_questions.length > 0 && (
        <div className="p-6 bg-amber-500/5 border-b border-slate-700/50">
          <h3 className="text-sm font-semibold text-amber-400 uppercase tracking-wide mb-3 flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-5a.75.75 0 01.75.75v4.5a.75.75 0 01-1.5 0v-4.5A.75.75 0 0110 5zm0 10a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
            </svg>
            Help Us Narrow It Down
          </h3>
          <ul className="space-y-2">
            {follow_up_questions.map((q, idx) => (
              <li key={idx} className="flex items-start gap-2 text-slate-300 text-sm">
                <span className="text-amber-400 font-bold">{idx + 1}.</span>
                {q}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Action */}
      <div className="p-6">
        <button
          onClick={onGetSolution}
          className="w-full bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-bold text-lg py-4 rounded-xl hover:from-emerald-600 hover:to-emerald-700 transition transform hover:scale-[1.02] shadow-lg shadow-emerald-500/25 flex items-center justify-center gap-2"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
            <path fillRule="evenodd" d="M12.577 4.878a.75.75 0 01.919-.53l4.78 1.281a.75.75 0 01.531.919l-1.281 4.78a.75.75 0 01-1.449-.387l.81-3.022a19.407 19.407 0 00-5.594 5.203.75.75 0 01-1.139.093L7 10.06l-4.72 4.72a.75.75 0 01-1.06-1.06l5.25-5.25a.75.75 0 011.06 0l3.074 3.073a20.923 20.923 0 015.545-4.931l-3.042-.815a.75.75 0 01-.53-.919z" clipRule="evenodd" />
          </svg>
          Get Step-by-Step Solution
        </button>
      </div>
    </div>
  )
}
