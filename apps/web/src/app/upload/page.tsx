'use client'

import { useState, useRef, FormEvent } from 'react'
import { API_URL } from '@/lib/api-config'
import { getAuthHeaders, isSupabaseConfigured } from '@/lib/supabase'

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
  leaks: Record<string, { top_items: string[]; scores: number[] }>
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
        return { filename: f.filename, leaks: data.leaks || {} }
      })

      const allResults = await Promise.all(analyzePromises)
      setResults(allResults)
      setShowResults(true)
      setStatus('Analysis Complete!')
      setStatusClass('text-emerald-400')
    } catch (err) {
      console.error(err)
      setStatus(`Error: ${err instanceof Error ? err.message : 'Analysis failed'}`)
      setStatusClass('text-red-400')
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
            Instant AI-powered forensic analysis of your POS exports. Free beta.
          </p>
        </div>
      </header>

      {/* Upload Section */}
      <section className="py-16 px-6">
        <div className="max-w-4xl mx-auto">
          <div className="bg-white/5 backdrop-blur-xl border border-emerald-500/30 rounded-3xl p-10 shadow-2xl">
            <div className="text-center mb-10">
              <h2 className="text-3xl font-bold">Get Your Free Report</h2>
              <p className="text-lg opacity-70 mt-2">Upload POS exports below for immediate analysis</p>
              {!isSupabaseConfigured() && (
                <p className="text-sm text-yellow-400 mt-2">
                  Running in anonymous mode (Supabase not configured)
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
              </div>

              <div>
                <label className="block text-lg font-semibold mb-3">
                  Email (for detailed report)
                </label>
                <input
                  type="email"
                  placeholder="optional@yourstore.com"
                  className="w-full px-6 py-4 text-lg bg-white/5 border-2 border-gray-500/40 rounded-2xl focus:border-emerald-400 focus:ring-4 focus:ring-emerald-400/20 transition"
                />
              </div>

              <button
                type="submit"
                className="w-full bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-bold text-2xl py-6 rounded-2xl hover:from-emerald-600 hover:to-emerald-700 transition transform hover:scale-[1.02] shadow-lg shadow-emerald-500/25"
              >
                Analyze Profits Now
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

            {/* Results Section */}
            {showResults && results.length > 0 && (
              <div className="mt-12 bg-white/5 rounded-2xl p-8">
                <h2 className="text-3xl font-bold mb-8 text-center">
                  Profit Leak Analysis ({results.length} files)
                </h2>
                <div className="space-y-10">
                  {results.map((res, idx) => (
                    <div key={idx} className="bg-white/5 rounded-xl p-6">
                      <h3 className="text-2xl font-bold mb-6">{res.filename}</h3>
                      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
                        {Object.entries(res.leaks).map(([primitive, data]) => (
                          <div key={primitive}>
                            <h4 className="text-lg font-semibold mb-3 capitalize">
                              {primitive.replace('_', ' ')}
                            </h4>
                            <ol className="space-y-2 text-sm">
                              {data.top_items.slice(0, 5).map((item, i) => (
                                <li key={i} className="flex justify-between">
                                  <span>{item || 'Unknown'}</span>
                                  <span className="text-emerald-400">
                                    {data.scores[i]?.toFixed(2) || 'N/A'}
                                  </span>
                                </li>
                              ))}
                            </ol>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </section>
    </div>
  )
}
