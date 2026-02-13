'use client'

import { useState } from 'react'

interface ContactFormProps {
  /** Pre-selected contact type */
  defaultType?: 'support' | 'feature_request' | 'feedback' | 'privacy'
  /** Pre-filled subject */
  defaultSubject?: string
  /** Callback on successful submission */
  onSuccess?: (referenceId: string) => void
  /** Callback on error */
  onError?: (error: string) => void
}

const CONTACT_TYPES = [
  { key: 'support', label: 'Support', description: 'Get help with using Profit Sentinel' },
  { key: 'feature_request', label: 'Feature Request', description: 'Suggest a new feature or improvement' },
  { key: 'feedback', label: 'Feedback', description: 'Share your experience or suggestions' },
  { key: 'privacy', label: 'Privacy Request', description: 'Data deletion, access, or privacy concerns' },
]

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

/**
 * ContactForm - Reusable contact form component
 *
 * Sends submissions to the backend /contact/submit endpoint.
 * Falls back gracefully if API is unavailable (shows mailto link).
 */
export function ContactForm({
  defaultType = 'support',
  defaultSubject = '',
  onSuccess,
  onError,
}: ContactFormProps) {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [type, setType] = useState(defaultType)
  const [subject, setSubject] = useState(defaultSubject)
  const [message, setMessage] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [submitted, setSubmitted] = useState(false)
  const [referenceId, setReferenceId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsSubmitting(true)
    setError(null)

    try {
      const response = await fetch(`${API_URL}/contact/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, email, type, subject, message }),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const data = await response.json()

      if (data.success) {
        setSubmitted(true)
        setReferenceId(data.reference_id)
        onSuccess?.(data.reference_id)
      } else {
        throw new Error(data.message || 'Failed to send message')
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to send message'
      setError(errorMessage)
      onError?.(errorMessage)
    } finally {
      setIsSubmitting(false)
    }
  }

  if (submitted) {
    return (
      <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-8 text-center">
        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-emerald-500/20 flex items-center justify-center">
          <CheckIcon className="w-8 h-8 text-emerald-400" />
        </div>
        <h3 className="text-xl font-bold text-emerald-400 mb-2">Message Sent!</h3>
        <p className="text-slate-400 mb-4">
          We'll get back to you within 24-48 hours.
        </p>
        {referenceId && (
          <p className="text-xs text-slate-500">
            Reference: <span className="font-mono text-slate-400">{referenceId}</span>
          </p>
        )}
        <button
          onClick={() => {
            setSubmitted(false)
            setName('')
            setEmail('')
            setSubject('')
            setMessage('')
          }}
          className="mt-6 text-sm text-emerald-400 hover:text-emerald-300 underline"
        >
          Send another message
        </button>
      </div>
    )
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-red-400 text-sm">
          {error}
          <p className="mt-2 text-xs">
            You can also email us directly at{' '}
            <a
              href={`mailto:support@profitsentinel.com?subject=${encodeURIComponent(subject)}`}
              className="underline"
            >
              support@profitsentinel.com
            </a>
          </p>
        </div>
      )}

      {/* Contact Type */}
      <div>
        <label className="block text-sm font-medium text-slate-300 mb-2">
          What can we help with?
        </label>
        <div className="grid grid-cols-2 gap-2">
          {CONTACT_TYPES.map((t) => (
            <button
              key={t.key}
              type="button"
              onClick={() => setType(t.key as typeof type)}
              className={`p-3 rounded-lg border text-left transition ${
                type === t.key
                  ? 'border-emerald-500/50 bg-emerald-500/10'
                  : 'border-slate-700 bg-white/5 hover:bg-white/10'
              }`}
            >
              <p className={`text-sm font-medium ${type === t.key ? 'text-emerald-400' : 'text-slate-300'}`}>
                {t.label}
              </p>
              <p className="text-xs text-slate-500 mt-0.5">{t.description}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Name & Email */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label htmlFor="name" className="block text-sm font-medium text-slate-300 mb-2">
            Name <span className="text-red-400">*</span>
          </label>
          <input
            type="text"
            id="name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
            className="w-full px-4 py-3 bg-white/5 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/50 transition"
            placeholder="Your name"
          />
        </div>
        <div>
          <label htmlFor="email" className="block text-sm font-medium text-slate-300 mb-2">
            Email <span className="text-red-400">*</span>
          </label>
          <input
            type="email"
            id="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            className="w-full px-4 py-3 bg-white/5 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/50 transition"
            placeholder="you@company.com"
          />
        </div>
      </div>

      {/* Subject */}
      <div>
        <label htmlFor="subject" className="block text-sm font-medium text-slate-300 mb-2">
          Subject <span className="text-red-400">*</span>
        </label>
        <input
          type="text"
          id="subject"
          value={subject}
          onChange={(e) => setSubject(e.target.value)}
          required
          className="w-full px-4 py-3 bg-white/5 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/50 transition"
          placeholder="Brief description of your request"
        />
      </div>

      {/* Message */}
      <div>
        <label htmlFor="message" className="block text-sm font-medium text-slate-300 mb-2">
          Message <span className="text-red-400">*</span>
        </label>
        <textarea
          id="message"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          required
          rows={5}
          minLength={10}
          className="w-full px-4 py-3 bg-white/5 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/50 transition resize-none"
          placeholder="Tell us more about your request..."
        />
      </div>

      {/* Submit */}
      <button
        type="submit"
        disabled={isSubmitting}
        className="w-full py-3 px-6 bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-bold rounded-xl hover:from-emerald-600 hover:to-emerald-700 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
      >
        {isSubmitting ? (
          <>
            <LoadingSpinner className="w-5 h-5" />
            Sending...
          </>
        ) : (
          <>
            <MailIcon className="w-5 h-5" />
            Send Message
          </>
        )}
      </button>
    </form>
  )
}

// Icons
function CheckIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className={className}>
      <path fillRule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clipRule="evenodd" />
    </svg>
  )
}

function MailIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className={className}>
      <path d="M3 4a2 2 0 00-2 2v1.161l8.441 4.221a1.25 1.25 0 001.118 0L19 7.162V6a2 2 0 00-2-2H3z" />
      <path d="M19 8.839l-7.77 3.885a2.75 2.75 0 01-2.46 0L1 8.839V14a2 2 0 002 2h14a2 2 0 002-2V8.839z" />
    </svg>
  )
}

function LoadingSpinner({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" className={`animate-spin ${className}`}>
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeOpacity="0.25" />
      <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  )
}

export default ContactForm
