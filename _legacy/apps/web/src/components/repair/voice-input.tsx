'use client'

import { useState, useCallback, useEffect, useRef } from 'react'

interface VoiceInputProps {
  onResult: (transcript: string) => void
  transcript: string
  onClear: () => void
}

// Web Speech API types
interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList
  resultIndex: number
}

interface SpeechRecognitionResultList {
  length: number
  item(index: number): SpeechRecognitionResult
  [index: number]: SpeechRecognitionResult
}

interface SpeechRecognitionResult {
  isFinal: boolean
  length: number
  item(index: number): SpeechRecognitionAlternative
  [index: number]: SpeechRecognitionAlternative
}

interface SpeechRecognitionAlternative {
  transcript: string
  confidence: number
}

interface SpeechRecognition extends EventTarget {
  continuous: boolean
  interimResults: boolean
  lang: string
  start(): void
  stop(): void
  abort(): void
  onresult: ((event: SpeechRecognitionEvent) => void) | null
  onerror: ((event: Event) => void) | null
  onend: (() => void) | null
}

declare global {
  interface Window {
    SpeechRecognition?: new () => SpeechRecognition
    webkitSpeechRecognition?: new () => SpeechRecognition
  }
}

export function VoiceInput({ onResult, transcript, onClear }: VoiceInputProps) {
  const [isListening, setIsListening] = useState(false)
  const [isSupported, setIsSupported] = useState(false)
  const [interimTranscript, setInterimTranscript] = useState('')
  const [error, setError] = useState<string | null>(null)
  const recognitionRef = useRef<SpeechRecognition | null>(null)

  // Check for browser support
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    setIsSupported(!!SpeechRecognition)
  }, [])

  const startListening = useCallback(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SpeechRecognition) {
      setError('Speech recognition not supported in this browser')
      return
    }

    const recognition = new SpeechRecognition()
    recognitionRef.current = recognition

    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = 'en-US'

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      let interim = ''
      let final = ''

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i]
        if (result.isFinal) {
          final += result[0].transcript
        } else {
          interim += result[0].transcript
        }
      }

      setInterimTranscript(interim)

      if (final) {
        onResult(transcript ? `${transcript} ${final}` : final)
        setInterimTranscript('')
      }
    }

    recognition.onerror = (event: Event) => {
      console.error('Speech recognition error:', event)
      setError('Voice recognition failed. Try again.')
      setIsListening(false)
    }

    recognition.onend = () => {
      setIsListening(false)
    }

    try {
      recognition.start()
      setIsListening(true)
      setError(null)
    } catch (err) {
      console.error('Failed to start recognition:', err)
      setError('Could not start voice input')
    }
  }, [onResult, transcript])

  const stopListening = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.stop()
    }
    setIsListening(false)
    setInterimTranscript('')
  }, [])

  // Show transcript if captured
  if (transcript) {
    return (
      <div className="space-y-3">
        <div className="relative p-4 bg-slate-900/50 border border-slate-600 rounded-xl">
          <p className="text-white pr-8">{transcript}</p>
          <button
            onClick={onClear}
            className="absolute top-2 right-2 p-1.5 text-slate-400 hover:text-red-400 transition"
            title="Clear transcript"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
              <path d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z" />
            </svg>
          </button>
        </div>

        {/* Add more button */}
        <button
          onClick={startListening}
          disabled={isListening}
          className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-purple-500/20 text-purple-300 rounded-lg hover:bg-purple-500/30 transition disabled:opacity-50"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
            <path d="M7 4a3 3 0 016 0v6a3 3 0 11-6 0V4z" />
            <path d="M5.5 9.643a.75.75 0 00-1.5 0V10c0 3.06 2.29 5.585 5.25 5.954V17.5h-1.5a.75.75 0 000 1.5h4.5a.75.75 0 000-1.5h-1.5v-1.546A6.001 6.001 0 0016 10v-.357a.75.75 0 00-1.5 0V10a4.5 4.5 0 01-9 0v-.357z" />
          </svg>
          Add More
        </button>
      </div>
    )
  }

  // Not supported
  if (!isSupported) {
    return (
      <div className="p-4 bg-slate-900/50 border border-slate-600 rounded-xl text-center">
        <p className="text-slate-400 text-sm">
          Voice input not supported in this browser.
        </p>
        <p className="text-slate-500 text-xs mt-1">
          Try Chrome or Safari for voice features.
        </p>
      </div>
    )
  }

  // Listening state
  if (isListening) {
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-center gap-4 p-8 bg-purple-500/10 border border-purple-500/30 rounded-xl">
          {/* Animated mic icon */}
          <div className="relative">
            <div className="absolute inset-0 animate-ping bg-purple-500/30 rounded-full" />
            <div className="relative p-4 bg-purple-500 rounded-full">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-8 h-8 text-white">
                <path d="M7 4a3 3 0 016 0v6a3 3 0 11-6 0V4z" />
                <path d="M5.5 9.643a.75.75 0 00-1.5 0V10c0 3.06 2.29 5.585 5.25 5.954V17.5h-1.5a.75.75 0 000 1.5h4.5a.75.75 0 000-1.5h-1.5v-1.546A6.001 6.001 0 0016 10v-.357a.75.75 0 00-1.5 0V10a4.5 4.5 0 01-9 0v-.357z" />
              </svg>
            </div>
          </div>
          <div>
            <p className="text-purple-300 font-medium">Listening...</p>
            <p className="text-slate-400 text-sm">Speak clearly about the problem</p>
          </div>
        </div>

        {/* Interim transcript */}
        {interimTranscript && (
          <div className="p-3 bg-slate-900/50 border border-slate-600 rounded-lg">
            <p className="text-slate-400 italic">{interimTranscript}</p>
          </div>
        )}

        {/* Stop button */}
        <button
          onClick={stopListening}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-red-500/20 text-red-300 rounded-xl hover:bg-red-500/30 transition"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
            <path fillRule="evenodd" d="M2 10a8 8 0 1116 0 8 8 0 01-16 0zm5-2.25A.75.75 0 017.75 7h4.5a.75.75 0 01.75.75v4.5a.75.75 0 01-.75.75h-4.5a.75.75 0 01-.75-.75v-4.5z" clipRule="evenodd" />
          </svg>
          Stop Recording
        </button>
      </div>
    )
  }

  // Default: show start button
  return (
    <div className="space-y-3">
      {error && (
        <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
          {error}
        </div>
      )}

      <button
        onClick={startListening}
        className="w-full flex items-center justify-center gap-3 p-6 bg-slate-900/50 border-2 border-dashed border-slate-600 rounded-xl hover:border-purple-500 hover:bg-purple-500/5 transition"
      >
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-10 h-10 text-purple-400">
          <path d="M8.25 4.5a3.75 3.75 0 117.5 0v8.25a3.75 3.75 0 11-7.5 0V4.5z" />
          <path d="M6 10.5a.75.75 0 01.75.75v1.5a5.25 5.25 0 1010.5 0v-1.5a.75.75 0 011.5 0v1.5a6.751 6.751 0 01-6 6.709v2.291h3a.75.75 0 010 1.5h-7.5a.75.75 0 010-1.5h3v-2.291a6.751 6.751 0 01-6-6.709v-1.5A.75.75 0 016 10.5z" />
        </svg>
        <span className="text-slate-300 font-medium">Tap to Speak</span>
      </button>

      <p className="text-slate-500 text-xs text-center">
        Speech is processed locally - no audio is recorded or stored
      </p>
    </div>
  )
}
