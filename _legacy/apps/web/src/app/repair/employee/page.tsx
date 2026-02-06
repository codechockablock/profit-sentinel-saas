'use client'

import { useState } from 'react'
import { API_URL } from '@/lib/api-config'
import { PhotoUpload } from '@/components/repair/photo-upload'
import { VoiceInput } from '@/components/repair/voice-input'
import { DiagnosisCard } from '@/components/repair/diagnosis-card'
import { SolutionView } from '@/components/repair/solution-view'
import { useToast } from '@/components/toast'

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
  likely_parts_needed?: string[]
  tools_needed?: string[]
  safety_concerns?: string[]
  diy_feasible?: boolean
  professional_recommended?: boolean
}

// Mock employee - would come from auth in production
const MOCK_EMPLOYEE = {
  employee_id: 'emp-001',
  name: 'Alex',
  store_id: 'demo-store-001',
}

type ViewState = 'input' | 'diagnosing' | 'results' | 'solution' | 'feedback'

export default function EmployeeRepairPage() {
  const [view, setView] = useState<ViewState>('input')
  const [textDescription, setTextDescription] = useState('')
  const [voiceTranscript, setVoiceTranscript] = useState('')
  const [imageBase64, setImageBase64] = useState<string | null>(null)
  const [diagnosis, setDiagnosis] = useState<DiagnoseResponse | null>(null)
  const [selectedCorrection, setSelectedCorrection] = useState<string | null>(null)
  const [correctionNotes, setCorrectionNotes] = useState('')
  const { addToast } = useToast()

  const handleDiagnose = async () => {
    if (!textDescription && !voiceTranscript && !imageBase64) {
      addToast({ type: 'error', message: 'Please provide a description, voice input, or photo' })
      return
    }

    setView('diagnosing')

    try {
      const response = await fetch(`${API_URL}/repair/diagnose`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text_description: textDescription || undefined,
          voice_transcript: voiceTranscript || undefined,
          image_base64: imageBase64 || undefined,
          store_id: MOCK_EMPLOYEE.store_id,
          employee_id: MOCK_EMPLOYEE.employee_id,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Diagnosis failed')
      }

      const data: DiagnoseResponse = await response.json()
      setDiagnosis(data)
      setView('results')
    } catch (err) {
      console.error('Diagnosis error:', err)
      setView('input')
      addToast({ type: 'error', message: err instanceof Error ? err.message : 'Diagnosis failed' })
    }
  }

  const handleFeedback = async () => {
    if (!diagnosis || !selectedCorrection) return

    try {
      const response = await fetch(`${API_URL}/repair/correction`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          problem_id: diagnosis.problem_id,
          employee_id: MOCK_EMPLOYEE.employee_id,
          correct_category_slug: selectedCorrection,
          correction_notes: correctionNotes || undefined,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Feedback failed')
      }

      addToast({
        type: 'success',
        message: 'Thanks for the feedback! This helps everyone.',
        duration: 3000,
      })

      handleReset()
    } catch (err) {
      console.error('Feedback error:', err)
      addToast({ type: 'error', message: err instanceof Error ? err.message : 'Feedback failed' })
    }
  }

  const handleReset = () => {
    setView('input')
    setTextDescription('')
    setVoiceTranscript('')
    setImageBase64(null)
    setDiagnosis(null)
    setSelectedCorrection(null)
    setCorrectionNotes('')
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 text-slate-200">
      {/* Header */}
      <header className="py-6 px-6 border-b border-slate-700/50">
        <div className="max-w-2xl mx-auto flex items-center gap-4">
          <div className="w-10 h-10 rounded-lg bg-emerald-500/20 flex items-center justify-center">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6 text-emerald-400">
              <path fillRule="evenodd" d="M12 6.75a5.25 5.25 0 016.775-5.025.75.75 0 01.313 1.248l-3.32 3.319c.063.475.276.934.641 1.299.365.365.824.578 1.3.64l3.318-3.319a.75.75 0 011.248.313 5.25 5.25 0 01-5.472 6.756c-1.018-.086-1.87.1-2.309.634L7.344 21.3A3.298 3.298 0 112.7 16.657l8.684-7.151c.533-.44.72-1.291.634-2.309A5.342 5.342 0 0112 6.75zM4.117 19.125a.75.75 0 01.75-.75h.008a.75.75 0 01.75.75v.008a.75.75 0 01-.75.75h-.008a.75.75 0 01-.75-.75v-.008z" clipRule="evenodd" />
            </svg>
          </div>
          <div>
            <h1 className="text-xl font-bold">Repair Assistant</h1>
            <p className="text-sm text-slate-400">Learn as you help customers</p>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-2xl mx-auto px-6 py-8">
        {/* Input View */}
        {view === 'input' && (
          <div className="space-y-6">
            {/* Tip Card */}
            <div className="bg-blue-500/10 border border-blue-500/30 rounded-2xl p-4">
              <p className="text-blue-300 text-sm">
                Use this tool to help customers identify repair problems.
                Each diagnosis teaches you something new about common issues.
              </p>
            </div>

            {/* Photo Upload */}
            <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700/50">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5 text-blue-400">
                  <path fillRule="evenodd" d="M1 5.25A2.25 2.25 0 013.25 3h13.5A2.25 2.25 0 0119 5.25v9.5A2.25 2.25 0 0116.75 17H3.25A2.25 2.25 0 011 14.75v-9.5zm1.5 5.81v3.69c0 .414.336.75.75.75h13.5a.75.75 0 00.75-.75v-2.69l-2.22-2.219a.75.75 0 00-1.06 0l-1.91 1.909.47.47a.75.75 0 11-1.06 1.06L6.53 8.091a.75.75 0 00-1.06 0l-2.97 2.97zM12 7a1 1 0 11-2 0 1 1 0 012 0z" clipRule="evenodd" />
                </svg>
                Customer's Photo
              </h2>
              <PhotoUpload
                onCapture={(base64) => setImageBase64(base64)}
                imageBase64={imageBase64}
                onClear={() => setImageBase64(null)}
              />
            </div>

            {/* Voice Input */}
            <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700/50">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5 text-purple-400">
                  <path d="M7 4a3 3 0 016 0v6a3 3 0 11-6 0V4z" />
                  <path d="M5.5 9.643a.75.75 0 00-1.5 0V10c0 3.06 2.29 5.585 5.25 5.954V17.5h-1.5a.75.75 0 000 1.5h4.5a.75.75 0 000-1.5h-1.5v-1.546A6.001 6.001 0 0016 10v-.357a.75.75 0 00-1.5 0V10a4.5 4.5 0 01-9 0v-.357z" />
                </svg>
                Describe Problem
              </h2>
              <VoiceInput
                onResult={(transcript) => setVoiceTranscript(transcript)}
                transcript={voiceTranscript}
                onClear={() => setVoiceTranscript('')}
              />
            </div>

            {/* Text Input */}
            <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700/50">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5 text-emerald-400">
                  <path fillRule="evenodd" d="M2 3.5A1.5 1.5 0 013.5 2h9A1.5 1.5 0 0114 3.5v11.75A2.75 2.75 0 0016.75 18h-12A2.75 2.75 0 012 15.25V3.5zm3.75 7a.75.75 0 000 1.5h4.5a.75.75 0 000-1.5h-4.5zm0 3a.75.75 0 000 1.5h4.5a.75.75 0 000-1.5h-4.5zM5 5.75A.75.75 0 015.75 5h4.5a.75.75 0 01.75.75v2.5a.75.75 0 01-.75.75h-4.5A.75.75 0 015 8.25v-2.5z" clipRule="evenodd" />
                  <path d="M16.5 6.5h-1v8.75a1.25 1.25 0 102.5 0V8a1.5 1.5 0 00-1.5-1.5z" />
                </svg>
                Or Type It
              </h2>
              <textarea
                value={textDescription}
                onChange={(e) => setTextDescription(e.target.value)}
                placeholder="What is the customer describing? (e.g., 'dripping faucet in kitchen')"
                className="w-full h-32 px-4 py-3 bg-slate-900/50 border border-slate-600 rounded-xl text-white placeholder:text-slate-500 focus:border-emerald-500 focus:ring-2 focus:ring-emerald-500/20 resize-none"
              />
            </div>

            {/* Diagnose Button */}
            <button
              onClick={handleDiagnose}
              disabled={!textDescription && !voiceTranscript && !imageBase64}
              className="w-full bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-bold text-xl py-5 rounded-2xl hover:from-emerald-600 hover:to-emerald-700 transition transform hover:scale-[1.02] shadow-lg shadow-emerald-500/25 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
            >
              What's the Problem?
            </button>
          </div>
        )}

        {/* Diagnosing View */}
        {view === 'diagnosing' && (
          <div className="flex flex-col items-center justify-center py-20">
            <div className="w-16 h-16 border-4 border-emerald-500/30 border-t-emerald-500 rounded-full animate-spin mb-6" />
            <h2 className="text-xl font-semibold mb-2">Analyzing...</h2>
            <p className="text-slate-400">Looking at the symptoms</p>
          </div>
        )}

        {/* Results View */}
        {view === 'results' && diagnosis && (
          <div className="space-y-6">
            {/* Diagnosis Card */}
            <DiagnosisCard
              diagnosis={diagnosis}
              onGetSolution={() => setView('solution')}
            />

            {/* Learning Note */}
            <div className="bg-slate-800/50 rounded-2xl p-5 border border-slate-700/50">
              <h3 className="font-semibold mb-2 flex items-center gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5 text-yellow-400">
                  <path d="M10 1a6 6 0 00-3.815 10.631C7.237 12.5 8 13.443 8 14.456v.644a.75.75 0 00.572.729 6.016 6.016 0 002.856 0A.75.75 0 0012 15.1v-.644c0-1.013.762-1.957 1.815-2.825A6 6 0 0010 1zM8.863 17.414a.75.75 0 00-.226 1.483 9.066 9.066 0 002.726 0 .75.75 0 00-.226-1.483 7.553 7.553 0 01-2.274 0z" />
                </svg>
                Learning Tip
              </h3>
              <p className="text-slate-400 text-sm">
                Notice how the AI weighs different possibilities.
                The confidence score tells you how certain it is.
                Lower confidence means the problem could be several things.
              </p>
            </div>

            {/* Safety Concerns */}
            {diagnosis.safety_concerns && diagnosis.safety_concerns.length > 0 && (
              <div className="bg-red-500/10 border border-red-500/30 rounded-2xl p-5">
                <h3 className="font-semibold text-red-400 mb-2">Safety First</h3>
                <ul className="space-y-1 text-sm text-red-300">
                  {diagnosis.safety_concerns.map((concern, idx) => (
                    <li key={idx}>• {concern}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Action Buttons */}
            <div className="grid grid-cols-2 gap-4">
              <button
                onClick={() => setView('feedback')}
                className="px-6 py-4 bg-slate-700 text-white rounded-xl hover:bg-slate-600 transition flex items-center justify-center gap-2"
              >
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
                  <path d="M5.433 13.917l1.262-3.155A4 4 0 017.58 9.42l6.92-6.918a2.121 2.121 0 013 3l-6.92 6.918c-.383.383-.84.685-1.343.886l-3.154 1.262a.5.5 0 01-.65-.65z" />
                </svg>
                AI Got It Wrong?
              </button>
              <button
                onClick={() => setView('solution')}
                className="px-6 py-4 bg-emerald-500 text-white font-bold rounded-xl hover:bg-emerald-600 transition"
              >
                Show Solution
              </button>
            </div>
          </div>
        )}

        {/* Feedback View */}
        {view === 'feedback' && diagnosis && (
          <div className="space-y-6">
            <button
              onClick={() => setView('results')}
              className="flex items-center gap-2 text-slate-400 hover:text-white transition"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
                <path fillRule="evenodd" d="M17 10a.75.75 0 01-.75.75H5.612l4.158 3.96a.75.75 0 11-1.04 1.08l-5.5-5.25a.75.75 0 010-1.08l5.5-5.25a.75.75 0 111.04 1.08L5.612 9.25H16.25A.75.75 0 0117 10z" clipRule="evenodd" />
              </svg>
              Back
            </button>

            <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700/50">
              <h2 className="text-xl font-semibold mb-2">Help Improve the AI</h2>
              <p className="text-slate-400 mb-6">
                Your real-world knowledge makes this tool better for everyone.
                What do you think the actual problem is?
              </p>

              {/* Category Selection */}
              <div className="space-y-3 mb-6">
                {diagnosis.hypotheses.map((h) => (
                  <button
                    key={h.category_slug}
                    onClick={() => setSelectedCorrection(h.category_slug)}
                    className={`w-full text-left px-4 py-3 rounded-xl border transition ${
                      selectedCorrection === h.category_slug
                        ? 'bg-emerald-500/20 border-emerald-500'
                        : 'bg-slate-900/50 border-slate-600 hover:border-slate-500'
                    }`}
                  >
                    <span className="font-medium">{h.category_name}</span>
                    {h.category_slug === diagnosis.top_hypothesis.category_slug && (
                      <span className="ml-2 text-xs text-slate-500">(AI's guess)</span>
                    )}
                  </button>
                ))}

                <button
                  onClick={() => setSelectedCorrection('other')}
                  className={`w-full text-left px-4 py-3 rounded-xl border transition ${
                    selectedCorrection === 'other'
                      ? 'bg-emerald-500/20 border-emerald-500'
                      : 'bg-slate-900/50 border-slate-600 hover:border-slate-500'
                  }`}
                >
                  <span className="font-medium">Something else</span>
                </button>
              </div>

              {/* Notes */}
              <div className="mb-6">
                <label className="block text-sm text-slate-400 mb-2">
                  What did you notice? (optional)
                </label>
                <textarea
                  value={correctionNotes}
                  onChange={(e) => setCorrectionNotes(e.target.value)}
                  placeholder="E.g., 'The discoloration pattern looks more like a leak than corrosion...'"
                  className="w-full h-24 px-4 py-3 bg-slate-900/50 border border-slate-600 rounded-xl text-white placeholder:text-slate-500 focus:border-emerald-500 focus:ring-2 focus:ring-emerald-500/20 resize-none"
                />
              </div>

              {/* Submit */}
              <button
                onClick={handleFeedback}
                disabled={!selectedCorrection}
                className="w-full bg-emerald-500 text-white font-bold py-4 rounded-xl hover:bg-emerald-600 transition disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Share Feedback
              </button>
            </div>
          </div>
        )}

        {/* Solution View */}
        {view === 'solution' && diagnosis && (
          <div className="space-y-6">
            <button
              onClick={() => setView('results')}
              className="flex items-center gap-2 text-slate-400 hover:text-white transition"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
                <path fillRule="evenodd" d="M17 10a.75.75 0 01-.75.75H5.612l4.158 3.96a.75.75 0 11-1.04 1.08l-5.5-5.25a.75.75 0 010-1.08l5.5-5.25a.75.75 0 111.04 1.08L5.612 9.25H16.25A.75.75 0 0117 10z" clipRule="evenodd" />
              </svg>
              Back to Diagnosis
            </button>

            <SolutionView
              problemId={diagnosis.problem_id}
              categorySlug={diagnosis.top_hypothesis.category_slug}
              categoryName={diagnosis.top_hypothesis.category_name}
            />

            {/* Learning Note */}
            <div className="bg-blue-500/10 border border-blue-500/30 rounded-2xl p-5">
              <h3 className="font-semibold text-blue-300 mb-2">Study These Steps</h3>
              <p className="text-blue-200/80 text-sm">
                Walk through this with the customer. Each repair you help with
                builds your expertise for the next one.
              </p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <button
                onClick={() => setView('feedback')}
                className="px-6 py-4 bg-slate-700 text-white rounded-xl hover:bg-slate-600 transition"
              >
                Something Wrong?
              </button>
              <button
                onClick={handleReset}
                className="px-6 py-4 bg-emerald-500 text-white font-bold rounded-xl hover:bg-emerald-600 transition"
              >
                Help Next Customer
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
