'use client'

import { useState, useRef } from 'react'
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

type ViewState = 'input' | 'diagnosing' | 'results' | 'solution'

export default function RepairAssistantPage() {
  const [view, setView] = useState<ViewState>('input')
  const [textDescription, setTextDescription] = useState('')
  const [voiceTranscript, setVoiceTranscript] = useState('')
  const [imageBase64, setImageBase64] = useState<string | null>(null)
  const [diagnosis, setDiagnosis] = useState<DiagnoseResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const { addToast } = useToast()

  // Demo store ID - would come from auth in production
  const storeId = 'demo-store-001'

  const handleDiagnose = async () => {
    if (!textDescription && !voiceTranscript && !imageBase64) {
      addToast({
        type: 'error',
        message: 'Please provide a description, voice input, or photo',
      })
      return
    }

    setView('diagnosing')
    setError(null)

    try {
      const response = await fetch(`${API_URL}/repair/diagnose`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text_description: textDescription || undefined,
          voice_transcript: voiceTranscript || undefined,
          image_base64: imageBase64 || undefined,
          store_id: storeId,
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
      setError(err instanceof Error ? err.message : 'Something went wrong')
      setView('input')
      addToast({
        type: 'error',
        message: err instanceof Error ? err.message : 'Diagnosis failed',
      })
    }
  }

  const handleReset = () => {
    setView('input')
    setTextDescription('')
    setVoiceTranscript('')
    setImageBase64(null)
    setDiagnosis(null)
    setError(null)
  }

  const handleVoiceResult = (transcript: string) => {
    setVoiceTranscript(transcript)
    addToast({
      type: 'success',
      message: 'Voice captured',
      duration: 2000,
    })
  }

  const handleImageCapture = (base64: string) => {
    setImageBase64(base64)
    addToast({
      type: 'success',
      message: 'Photo captured',
      duration: 2000,
    })
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 text-slate-200">
      {/* Header */}
      <header className="py-8 px-6 text-center border-b border-slate-700/50">
        <div className="flex items-center justify-center gap-3 mb-2">
          <div className="w-10 h-10 rounded-lg bg-emerald-500/20 flex items-center justify-center">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6 text-emerald-400">
              <path fillRule="evenodd" d="M12 6.75a5.25 5.25 0 016.775-5.025.75.75 0 01.313 1.248l-3.32 3.319c.063.475.276.934.641 1.299.365.365.824.578 1.3.64l3.318-3.319a.75.75 0 011.248.313 5.25 5.25 0 01-5.472 6.756c-1.018-.086-1.87.1-2.309.634L7.344 21.3A3.298 3.298 0 112.7 16.657l8.684-7.151c.533-.44.72-1.291.634-2.309A5.342 5.342 0 0112 6.75zM4.117 19.125a.75.75 0 01.75-.75h.008a.75.75 0 01.75.75v.008a.75.75 0 01-.75.75h-.008a.75.75 0 01-.75-.75v-.008z" clipRule="evenodd" />
            </svg>
          </div>
          <h1 className="text-2xl font-bold">Repair Assistant</h1>
        </div>
        <p className="text-slate-400">
          Describe your problem or take a photo - we'll help you fix it
        </p>
      </header>

      {/* Main Content */}
      <main className="max-w-2xl mx-auto px-6 py-8">
        {/* Input View */}
        {view === 'input' && (
          <div className="space-y-6">
            {/* Photo Upload */}
            <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700/50">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5 text-blue-400">
                  <path fillRule="evenodd" d="M1 5.25A2.25 2.25 0 013.25 3h13.5A2.25 2.25 0 0119 5.25v9.5A2.25 2.25 0 0116.75 17H3.25A2.25 2.25 0 011 14.75v-9.5zm1.5 5.81v3.69c0 .414.336.75.75.75h13.5a.75.75 0 00.75-.75v-2.69l-2.22-2.219a.75.75 0 00-1.06 0l-1.91 1.909.47.47a.75.75 0 11-1.06 1.06L6.53 8.091a.75.75 0 00-1.06 0l-2.97 2.97zM12 7a1 1 0 11-2 0 1 1 0 012 0z" clipRule="evenodd" />
                </svg>
                Take a Photo
              </h2>
              <PhotoUpload
                onCapture={handleImageCapture}
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
                Describe by Voice
              </h2>
              <VoiceInput
                onResult={handleVoiceResult}
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
                Type Description
              </h2>
              <textarea
                value={textDescription}
                onChange={(e) => setTextDescription(e.target.value)}
                placeholder="What's the problem? (e.g., 'My kitchen faucet is dripping')"
                className="w-full h-32 px-4 py-3 bg-slate-900/50 border border-slate-600 rounded-xl text-white placeholder:text-slate-500 focus:border-emerald-500 focus:ring-2 focus:ring-emerald-500/20 resize-none"
              />
            </div>

            {/* Error Message */}
            {error && (
              <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-red-400 text-center">
                {error}
              </div>
            )}

            {/* Diagnose Button */}
            <button
              onClick={handleDiagnose}
              disabled={!textDescription && !voiceTranscript && !imageBase64}
              className="w-full bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-bold text-xl py-5 rounded-2xl hover:from-emerald-600 hover:to-emerald-700 transition transform hover:scale-[1.02] shadow-lg shadow-emerald-500/25 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
            >
              What Do I Need to Fix This?
            </button>

            {/* Helper Text */}
            <p className="text-center text-slate-500 text-sm">
              Your data is processed locally and never stored
            </p>
          </div>
        )}

        {/* Diagnosing View */}
        {view === 'diagnosing' && (
          <div className="flex flex-col items-center justify-center py-20">
            <div className="w-16 h-16 border-4 border-emerald-500/30 border-t-emerald-500 rounded-full animate-spin mb-6" />
            <h2 className="text-xl font-semibold mb-2">Analyzing Your Problem...</h2>
            <p className="text-slate-400 text-center">
              Our AI is examining your input to identify the issue
            </p>
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

            {/* Safety Concerns */}
            {diagnosis.safety_concerns && diagnosis.safety_concerns.length > 0 && (
              <div className="bg-red-500/10 border border-red-500/30 rounded-2xl p-6">
                <h3 className="text-lg font-semibold text-red-400 mb-3 flex items-center gap-2">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
                    <path fillRule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
                  </svg>
                  Safety Concerns
                </h3>
                <ul className="space-y-2">
                  {diagnosis.safety_concerns.map((concern, idx) => (
                    <li key={idx} className="flex items-start gap-2 text-red-300">
                      <span className="text-red-400 mt-0.5">!</span>
                      {concern}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Parts Needed */}
            {diagnosis.likely_parts_needed && diagnosis.likely_parts_needed.length > 0 && (
              <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700/50">
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5 text-amber-400">
                    <path d="M1 4.75C1 3.784 1.784 3 2.75 3h14.5c.966 0 1.75.784 1.75 1.75v10.515a1.75 1.75 0 01-1.75 1.75h-1.5v-1.5h1.5a.25.25 0 00.25-.25V4.75a.25.25 0 00-.25-.25H2.75a.25.25 0 00-.25.25v10.515c0 .138.112.25.25.25h1.5v1.5h-1.5A1.75 1.75 0 011 15.265V4.75z" />
                    <path d="M10 6a.75.75 0 01.75.75v3.5a.75.75 0 01-.75.75H6.25a.75.75 0 010-1.5h2.94L5.47 5.78a.75.75 0 011.06-1.06l3.72 3.72V4.25A.75.75 0 0110 3.5zM10 14.5a1.5 1.5 0 100-3 1.5 1.5 0 000 3z" />
                  </svg>
                  Parts You May Need
                </h3>
                <div className="flex flex-wrap gap-2">
                  {diagnosis.likely_parts_needed.map((part, idx) => (
                    <span
                      key={idx}
                      className="px-3 py-1.5 bg-amber-500/20 text-amber-300 rounded-lg text-sm"
                    >
                      {part}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Tools Needed */}
            {diagnosis.tools_needed && diagnosis.tools_needed.length > 0 && (
              <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700/50">
                <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5 text-blue-400">
                    <path fillRule="evenodd" d="M14.5 10a4.5 4.5 0 004.284-5.882c-.105-.324-.51-.391-.752-.15L15.34 6.66a.454.454 0 01-.493.11 3.01 3.01 0 01-1.618-1.616.455.455 0 01.11-.494l2.694-2.692c.24-.241.174-.647-.15-.752a4.5 4.5 0 00-5.873 4.575c.055.873-.128 1.808-.8 2.368l-7.23 6.024a2.724 2.724 0 103.837 3.837l6.024-7.23c.56-.672 1.495-.855 2.368-.8.096.007.193.01.291.01zM5 16a1 1 0 11-2 0 1 1 0 012 0z" clipRule="evenodd" />
                  </svg>
                  Tools Needed
                </h3>
                <div className="flex flex-wrap gap-2">
                  {diagnosis.tools_needed.map((tool, idx) => (
                    <span
                      key={idx}
                      className="px-3 py-1.5 bg-blue-500/20 text-blue-300 rounded-lg text-sm"
                    >
                      {tool}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Professional Recommendation */}
            {diagnosis.professional_recommended && (
              <div className="bg-amber-500/10 border border-amber-500/30 rounded-2xl p-6 text-center">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-8 h-8 text-amber-400 mx-auto mb-2">
                  <path d="M10 1a6 6 0 00-3.815 10.631C7.237 12.5 8 13.443 8 14.456v.644a.75.75 0 00.572.729 6.016 6.016 0 002.856 0A.75.75 0 0012 15.1v-.644c0-1.013.762-1.957 1.815-2.825A6 6 0 0010 1zM8.863 17.414a.75.75 0 00-.226 1.483 9.066 9.066 0 002.726 0 .75.75 0 00-.226-1.483 7.553 7.553 0 01-2.274 0z" />
                </svg>
                <p className="text-amber-300 font-medium">
                  We recommend consulting a professional for this repair
                </p>
                <p className="text-slate-400 text-sm mt-1">
                  This job may require specialized tools or expertise
                </p>
              </div>
            )}

            {/* Actions */}
            <div className="flex gap-4">
              <button
                onClick={handleReset}
                className="flex-1 px-6 py-4 bg-slate-700 text-white rounded-xl hover:bg-slate-600 transition"
              >
                Start Over
              </button>
              <button
                onClick={() => setView('solution')}
                className="flex-1 px-6 py-4 bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-bold rounded-xl hover:from-emerald-600 hover:to-emerald-700 transition"
              >
                Get Step-by-Step Guide
              </button>
            </div>
          </div>
        )}

        {/* Solution View */}
        {view === 'solution' && diagnosis && (
          <div className="space-y-6">
            <SolutionView
              problemId={diagnosis.problem_id}
              categorySlug={diagnosis.top_hypothesis.category_slug}
              categoryName={diagnosis.top_hypothesis.category_name}
            />

            <button
              onClick={handleReset}
              className="w-full px-6 py-4 bg-slate-700 text-white rounded-xl hover:bg-slate-600 transition"
            >
              Start New Diagnosis
            </button>
          </div>
        )}
      </main>
    </div>
  )
}
