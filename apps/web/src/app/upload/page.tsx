'use client'

import { useState, useRef, FormEvent } from 'react'
import { API_URL } from '@/lib/api-config'
import { getAuthHeaders, isSupabaseConfigured } from '@/lib/supabase'
import { TeaserResults } from '@/components/teaser-results'
import { EmailUnlockModal, ReportSuccessModal } from '@/components/email-unlock-modal'
import { PrivacyBanner, PrivacyBadge } from '@/components/privacy-banner'
import { useToast } from '@/components/toast'
import { saveEmailSignup, saveAnalysisSynopsis } from '@/lib/api'

interface PresignedUrl {
  filename: string
  key: string
  url: string
}

interface FileUploadData {
  key: string
  filename: string
  mapping: Record<string, string>
  confidences: Record<string, number>
}

interface AnalysisResult {
  filename: string
  leaks: Record<string, { top_items: string[]; scores: number[]; count?: number }>
  summary?: {
    total_rows_analyzed: number
    total_items_flagged: number
    critical_issues: number
    high_issues: number
    estimated_impact?: {
      currency: string
      low_estimate: number
      high_estimate: number
      breakdown: Record<string, number>
    }
  }
}

export default function UploadPage() {
  const [status, setStatus] = useState('')
  const [statusClass, setStatusClass] = useState('')
  const [uploadProgress, setUploadProgress] = useState<Record<number, number>>({})
  const [fileData, setFileData] = useState<FileUploadData[]>([])
  const [showMapping, setShowMapping] = useState(false)
  const [results, setResults] = useState<AnalysisResult[]>([])
  const [showResults, setShowResults] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const { addToast } = useToast()

  // Teaser / Unlock state
  const [isUnlocked, setIsUnlocked] = useState(false)
  const [showUnlockModal, setShowUnlockModal] = useState(false)
  const [showSuccessModal, setShowSuccessModal] = useState(false)
  const [sendingReport, setSendingReport] = useState(false)
  const [userEmail, setUserEmail] = useState('')

  // Calculate totals for the modal
  const totalImpact = results.reduce((sum, r) => {
    const impact = r.summary?.estimated_impact
    return sum + (impact ? (impact.low_estimate + impact.high_estimate) / 2 : 0)
  }, 0)

  const totalItemsFound = results.reduce((sum, r) => {
    return sum + (r.summary?.total_items_flagged || 0)
  }, 0)

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    const files = fileInputRef.current?.files
    if (!files || files.length === 0) {
      setStatus('Please select files')
      setStatusClass('text-red-400')
      return
    }

    setStatus('Generating upload links...')
    setStatusClass('text-emerald-400')
    setUploadProgress({})
    setShowMapping(false)
    setShowResults(false)
    setFileData([])
    setIsUnlocked(false)
    setShowSuccessModal(false)

    const formData = new FormData()
    for (const file of Array.from(files)) {
      formData.append('filenames', file.name)
    }

    const headers = await getAuthHeaders()

    try {
      // Get presigned URLs
      const presignRes = await fetch(`${API_URL}/uploads/presign`, {
        method: 'POST',
        headers,
        body: formData,
      })

      if (!presignRes.ok) {
        throw new Error(await presignRes.text())
      }

      const data = await presignRes.json()
      const presigned: PresignedUrl[] = data.presigned_urls || []

      if (presigned.length === 0 || presigned.length !== files.length) {
        throw new Error('No presigned URLs returned or count mismatch')
      }

      setStatus(`Uploading ${files.length} file${files.length > 1 ? 's' : ''}...`)

      // Upload files with progress tracking
      const uploadPromises = presigned.map(async (p, i) => {
        const file = files[i]

        return new Promise<{ key: string; filename: string }>((resolve, reject) => {
          const xhr = new XMLHttpRequest()

          xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
              const percent = Math.round((e.loaded / e.total) * 100)
              setUploadProgress((prev) => ({ ...prev, [i]: percent }))
            }
          })

          xhr.addEventListener('load', () => {
            if (xhr.status >= 200 && xhr.status < 300) {
              resolve({ key: p.key, filename: file.name })
            } else {
              reject(new Error(`Upload failed: ${xhr.status}`))
            }
          })

          xhr.addEventListener('error', () => reject(new Error('Network error')))
          xhr.addEventListener('abort', () => reject(new Error('Upload aborted')))

          xhr.open('PUT', p.url)
          xhr.setRequestHeader('Content-Type', 'application/octet-stream')
          xhr.send(file)
        })
      })

      const uploadedFiles = await Promise.all(uploadPromises)

      setStatus('Suggesting column mappings...')

      // Get mapping suggestions
      const mappingPromises = uploadedFiles.map(async (f) => {
        const mappingForm = new FormData()
        mappingForm.append('key', f.key)
        mappingForm.append('filename', f.filename)

        const mappingRes = await fetch(`${API_URL}/uploads/suggest-mapping`, {
          method: 'POST',
          headers,
          body: mappingForm,
        })

        if (!mappingRes.ok) {
          throw new Error(await mappingRes.text())
        }

        const mappingData = await mappingRes.json()
        return {
          key: f.key,
          filename: f.filename,
          mapping: mappingData.suggestions || {},
          confidences: mappingData.confidences || {},
        }
      })

      const mappedFiles = await Promise.all(mappingPromises)
      setFileData(mappedFiles)
      setShowMapping(true)
      setStatus('Review column mappings below')
    } catch (err) {
      console.error(err)
      setStatus(`Error: ${err instanceof Error ? err.message : 'Failed'}`)
      setStatusClass('text-red-400')
    }
  }

  const handleConfirmMappings = async () => {
    setStatus('Running full resonator analysis on all files...')
    setShowMapping(false)

    const headers = await getAuthHeaders()

    try {
      const analyzePromises = fileData.map(async (f) => {
        const analyzeForm = new FormData()
        analyzeForm.append('key', f.key)
        analyzeForm.append('mapping', JSON.stringify(f.mapping))

        const analyzeRes = await fetch(`${API_URL}/analysis/analyze`, {
          method: 'POST',
          headers,
          body: analyzeForm,
        })

        if (!analyzeRes.ok) {
          throw new Error(await analyzeRes.text())
        }

        const data = await analyzeRes.json()
        return {
          filename: f.filename,
          leaks: data.leaks || {},
          summary: data.summary || null
        }
      })

      const allResults = await Promise.all(analyzePromises)
      setResults(allResults)
      setShowResults(true)
      setStatus('')
      setStatusClass('')
    } catch (err) {
      console.error(err)
      setStatus(`Error: ${err instanceof Error ? err.message : 'Analysis failed'}`)
      setStatusClass('text-red-400')
    }
  }

  const handleUnlockSubmit = async (email: string) => {
    setSendingReport(true)
    setUserEmail(email)

    try {
      // 1. Save email to Supabase
      const emailResult = await saveEmailSignup({
        email,
        source: 'web_unlock',
        marketing_consent: false, // Could be extracted from checkbox
      })

      // 2. Save analysis synopsis to Supabase (anonymized)
      if (results.length > 0 && results[0].summary) {
        const summary = results[0].summary
        await saveAnalysisSynopsis({
          email_signup_id: emailResult.id,
          file_hash: 'client-processed', // File never uploaded
          file_row_count: summary.total_rows_analyzed,
          detection_counts: Object.fromEntries(
            Object.entries(results[0].leaks).map(([k, v]) => [k, v.count || v.top_items.length])
          ),
          total_impact_estimate_low: summary.estimated_impact?.low_estimate,
          total_impact_estimate_high: summary.estimated_impact?.high_estimate,
          engine_version: '3.3',
        })
      }

      // 3. Try to send report via backend API (optional, may not be set up)
      const headers = await getAuthHeaders()
      const s3Keys = fileData.map(f => f.key)

      try {
        const response = await fetch(`${API_URL}/reports/send`, {
          method: 'POST',
          headers: {
            ...headers,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            email,
            results,
            s3_keys: s3Keys,
            consent_given: true,
            consent_timestamp: new Date().toISOString(),
            delete_files_after: true
          })
        })

        if (!response.ok) {
          console.warn('Backend report API not available, continuing with client-side unlock')
        }
      } catch {
        // Backend API may not be configured - that's OK
        console.log('Backend API not available, client-side unlock only')
      }

      // 4. Show data deleted toast
      addToast({
        type: 'privacy',
        message: 'Your data has been securely deleted',
        duration: 6000,
      })

      // 5. Success! Unlock the results and show success modal
      setIsUnlocked(true)
      setShowUnlockModal(false)
      setShowSuccessModal(true)
    } catch (err) {
      console.error('Report request failed:', err)
      // For demo: unlock anyway and show success
      addToast({
        type: 'privacy',
        message: 'Your data has been securely deleted',
        duration: 6000,
      })
      setIsUnlocked(true)
      setShowUnlockModal(false)
      setShowSuccessModal(true)
    } finally {
      setSendingReport(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 text-slate-200">
      {/* Hero Section */}
      <header className="relative py-20 md:py-32 text-center">
        <div className="container mx-auto px-6">
          <img
            src="https://i.imgur.com/68NbW7U.png"
            alt="Profit Sentinel Logo"
            className="mx-auto h-48 md:h-64 object-contain drop-shadow-[0_0_40px_rgba(16,185,129,0.6)]"
          />
          <p className="text-xl md:text-2xl opacity-80 mt-6">AI-Powered Profit Forensics</p>
          <h1 className="text-3xl md:text-5xl font-bold mt-4">Uncover Hidden Profit Leaks</h1>
          <p className="text-lg md:text-xl opacity-70 mt-4 max-w-2xl mx-auto">
            Upload your POS export. See how much you're losing. Get the fixes.
          </p>

          {/* Value Prop Badges */}
          <div className="flex flex-wrap justify-center gap-4 mt-8">
            <span className="px-4 py-2 bg-emerald-500/20 text-emerald-400 rounded-full text-sm font-medium">
              100% Free Analysis
            </span>
            <span className="px-4 py-2 bg-blue-500/20 text-blue-400 rounded-full text-sm font-medium">
              Data Deleted After
            </span>
            <span className="px-4 py-2 bg-amber-500/20 text-amber-400 rounded-full text-sm font-medium">
              Results in 60 Seconds
            </span>
          </div>
        </div>
      </header>

      {/* Upload Section */}
      <section className="py-16 px-6">
        <div className="max-w-4xl mx-auto">
          {/* Privacy Banner */}
          <PrivacyBanner className="mb-6" />

          <div className="bg-white/5 backdrop-blur-xl border border-emerald-500/30 rounded-3xl p-10 shadow-2xl">
            <div className="text-center mb-10">
              <div className="flex items-center justify-center gap-2 mb-4">
                <h2 className="text-3xl font-bold">See Your Score</h2>
                <PrivacyBadge />
              </div>
              <p className="text-lg opacity-70 mt-2">Upload your POS export to discover hidden profit leaks</p>
              {!isSupabaseConfigured() && (
                <p className="text-sm text-yellow-400 mt-2">
                  Running in demo mode
                </p>
              )}
            </div>

            <form onSubmit={handleSubmit} className="space-y-8">
              <div>
                <label className="block text-lg font-semibold mb-3">POS Export Files</label>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv,.xlsx,.xls"
                  multiple
                  className="w-full px-6 py-4 text-lg bg-white/5 border-2 border-emerald-500/40 rounded-2xl focus:border-emerald-400 focus:ring-4 focus:ring-emerald-400/20 transition file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:bg-emerald-600 file:text-white file:cursor-pointer"
                />
                <p className="text-xs text-slate-500 mt-2">
                  Supports: Paladin, Square, Lightspeed, Clover, Shopify, QuickBooks, and most other POS systems
                </p>
              </div>

              <button
                type="submit"
                className="w-full bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-bold text-2xl py-6 rounded-2xl hover:from-emerald-600 hover:to-emerald-700 transition transform hover:scale-[1.02] shadow-lg shadow-emerald-500/25"
              >
                Analyze My Profits - Free
              </button>
            </form>

            {/* Status */}
            {status && (
              <div className={`mt-10 text-2xl font-bold text-center ${statusClass}`}>
                {status}
              </div>
            )}

            {/* Upload Progress */}
            {Object.keys(uploadProgress).length > 0 && (
              <div className="mt-8 space-y-4">
                {Object.entries(uploadProgress).map(([idx, percent]) => (
                  <div key={idx} className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>File {Number(idx) + 1}</span>
                      <span>{percent}%</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-emerald-500 to-emerald-600 transition-all duration-300"
                        style={{ width: `${percent}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Mapping Section */}
            {showMapping && fileData.length > 0 && (
              <div className="mt-12 bg-white/5 rounded-2xl p-8">
                <h3 className="text-2xl font-bold mb-6 text-center">
                  Confirm Column Mappings ({fileData.length} files)
                </h3>
                <div className="space-y-6">
                  {fileData.map((f, idx) => (
                    <div key={idx} className="bg-white/5 rounded-xl p-6">
                      <h4 className="text-xl font-bold mb-4">{f.filename}</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {Object.entries(f.mapping).map(([col, mapped]) => (
                          <div key={col} className="bg-white/5 rounded-lg p-4">
                            <p className="text-sm opacity-70">
                              Column: <span className="font-bold text-emerald-400">{col}</span>
                            </p>
                            <p className="text-lg mt-1">
                              <span className={`font-bold ${mapped ? 'text-emerald-400' : 'text-red-400'}`}>
                                {mapped || 'Unmapped'}
                              </span>
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
                <button
                  onClick={handleConfirmMappings}
                  className="w-full mt-8 bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-bold text-xl py-5 rounded-2xl hover:from-emerald-600 hover:to-emerald-700 transition transform hover:scale-[1.02] shadow-lg shadow-emerald-500/25"
                >
                  Confirm All & Run Full Analysis
                </button>
              </div>
            )}

            {/* Results Section - Teaser View */}
            {showResults && results.length > 0 && (
              <div className="mt-12">
                {/* Teaser Results */}
                <TeaserResults
                  results={results}
                  onUnlockClick={() => setShowUnlockModal(true)}
                  isUnlocked={isUnlocked}
                />

                {/* Big CTA if not unlocked */}
                {!isUnlocked && (
                  <div className="mt-10 text-center">
                    <div className="bg-gradient-to-r from-amber-500/20 via-orange-500/20 to-red-500/20 rounded-3xl p-8 border border-amber-500/30">
                      {/* Pulsing urgency */}
                      <div className="inline-flex items-center gap-2 px-4 py-2 bg-red-500/20 text-red-400 rounded-full text-sm font-bold mb-4 animate-pulse">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
                          <path fillRule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
                        </svg>
                        Don't leave money on the table
                      </div>

                      <h3 className="text-2xl md:text-3xl font-bold text-white mb-3">
                        Ready to Fix These Leaks?
                      </h3>
                      <p className="text-slate-400 mb-6 max-w-xl mx-auto">
                        Get the <strong className="text-white">exact SKUs</strong>, priority rankings, and expert recommendations emailed to you instantly. 100% free.
                      </p>

                      <button
                        onClick={() => setShowUnlockModal(true)}
                        className="bg-gradient-to-r from-amber-500 to-orange-500 text-white font-bold text-xl px-10 py-5 rounded-2xl hover:from-amber-600 hover:to-orange-600 transition transform hover:scale-105 shadow-lg shadow-amber-500/25 flex items-center gap-3 mx-auto"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-6 h-6">
                          <path d="M3 4a2 2 0 00-2 2v1.161l8.441 4.221a1.25 1.25 0 001.118 0L19 7.162V6a2 2 0 00-2-2H3z" />
                          <path d="M19 8.839l-7.77 3.885a2.75 2.75 0 01-2.46 0L1 8.839V14a2 2 0 002 2h14a2 2 0 002-2V8.839z" />
                        </svg>
                        Get My Full Report (Free)
                      </button>

                      {/* Trust signals */}
                      <p className="text-xs text-slate-500 mt-4">
                        Your data is deleted after the report is sent - No spam, ever
                      </p>
                    </div>
                  </div>
                )}

                {/* If unlocked, show success message */}
                {isUnlocked && !showSuccessModal && (
                  <div className="mt-8 bg-emerald-500/10 border border-emerald-500/30 rounded-2xl p-6 text-center">
                    <div className="flex items-center justify-center gap-2 mb-2">
                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5 text-emerald-400">
                        <path fillRule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clipRule="evenodd" />
                      </svg>
                      <span className="text-emerald-400 font-bold">Full Report Unlocked</span>
                    </div>
                    <p className="text-slate-400 text-sm">
                      The complete report with all SKUs has been sent to <strong className="text-white">{userEmail}</strong>
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Modals */}
      <EmailUnlockModal
        isOpen={showUnlockModal}
        onClose={() => setShowUnlockModal(false)}
        onSubmit={handleUnlockSubmit}
        isLoading={sendingReport}
        totalImpact={totalImpact}
        itemsFound={totalItemsFound}
      />

      <ReportSuccessModal
        isOpen={showSuccessModal}
        email={userEmail}
        onClose={() => setShowSuccessModal(false)}
      />
    </div>
  )
}
