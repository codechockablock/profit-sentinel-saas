'use client'

import { useState, useEffect } from 'react'
import { formatDollarImpact } from '@/lib/leak-metadata'

interface EmailUnlockModalProps {
  isOpen: boolean
  onClose: () => void
  onSubmit: (email: string) => void
  isLoading: boolean
  totalImpact?: number
  itemsFound?: number
}

/**
 * High-Converting Email Unlock Modal
 *
 * Conversion psychology:
 * - Urgency: Animated countdown, pulsing elements
 * - Value prop: Show exactly what they'll get
 * - Social proof: "Join 1000+ store owners"
 * - Loss aversion: "Don't let $X walk out the door"
 * - Trust: Privacy badges, GDPR compliance
 */
export function EmailUnlockModal({
  isOpen,
  onClose,
  onSubmit,
  isLoading,
  totalImpact = 0,
  itemsFound = 0
}: EmailUnlockModalProps) {
  const [email, setEmail] = useState('')
  const [agreed, setAgreed] = useState(false)
  const [showDetails, setShowDetails] = useState(false)
  const [shake, setShake] = useState(false)

  // Prevent body scroll when modal is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = 'unset'
    }
    return () => {
      document.body.style.overflow = 'unset'
    }
  }, [isOpen])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!email || !agreed) {
      setShake(true)
      setTimeout(() => setShake(false), 500)
      return
    }
    onSubmit(email)
  }

  const isValidEmail = (email: string) => {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/80 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div
        className={`relative w-full max-w-lg bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 rounded-3xl border border-emerald-500/30 shadow-2xl shadow-emerald-500/20 overflow-hidden ${shake ? 'animate-shake' : ''}`}
      >
        {/* Animated background glow */}
        <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/5 via-transparent to-amber-500/5 animate-pulse" />

        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-2 text-slate-400 hover:text-white transition z-10"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
            <path d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z" />
          </svg>
        </button>

        <div className="relative p-8">
          {/* Header with Impact */}
          <div className="text-center mb-8">
            {/* Urgency badge */}
            <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-red-500/20 text-red-400 rounded-full text-sm font-bold mb-4 animate-pulse">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
                <path fillRule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
              </svg>
              {itemsFound} Problems Detected
            </div>

            <h2 className="text-2xl md:text-3xl font-bold text-white mb-2">
              Unlock Your Full Report
            </h2>

            {/* Big Impact Number */}
            {totalImpact > 0 && (
              <div className="mt-4 mb-2">
                <p className="text-slate-400 text-sm">You could be losing up to</p>
                <p className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-red-400 via-orange-400 to-amber-400">
                  {formatDollarImpact(totalImpact)}/year
                </p>
              </div>
            )}

            <p className="text-slate-400">
              Get specific SKUs, fix instructions & priority action list
            </p>
          </div>

          {/* What You'll Get */}
          <div className="bg-white/5 rounded-xl p-4 mb-6">
            <p className="text-sm font-semibold text-emerald-400 mb-3 flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
                <path fillRule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clipRule="evenodd" />
              </svg>
              Your Full Report Includes:
            </p>
            <ul className="space-y-2 text-sm">
              <li className="flex items-center gap-2 text-slate-300">
                <span className="text-emerald-400">-</span>
                <strong>Exact SKUs</strong> for every flagged item
              </li>
              <li className="flex items-center gap-2 text-slate-300">
                <span className="text-emerald-400">-</span>
                <strong>Dollar impact</strong> per item & category
              </li>
              <li className="flex items-center gap-2 text-slate-300">
                <span className="text-emerald-400">-</span>
                <strong>Priority action list</strong> - what to fix first
              </li>
              <li className="flex items-center gap-2 text-slate-300">
                <span className="text-emerald-400">-</span>
                <strong>Expert recommendations</strong> for each leak type
              </li>
            </ul>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Email Input */}
            <div>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="your@email.com"
                disabled={isLoading}
                className="w-full px-4 py-4 text-lg bg-white/5 border-2 border-slate-600 rounded-xl focus:border-emerald-400 focus:ring-4 focus:ring-emerald-400/20 transition disabled:opacity-50"
              />
              {email && !isValidEmail(email) && (
                <p className="text-xs text-red-400 mt-1">Please enter a valid email</p>
              )}
            </div>

            {/* Consent Checkbox */}
            <div className="flex items-start gap-3">
              <input
                type="checkbox"
                id="consent"
                checked={agreed}
                onChange={(e) => setAgreed(e.target.checked)}
                disabled={isLoading}
                className="mt-1 w-5 h-5 rounded border-slate-500 bg-white/10 text-emerald-500 focus:ring-emerald-500 cursor-pointer"
              />
              <label htmlFor="consent" className="text-sm text-slate-300 cursor-pointer">
                <span className="font-medium text-white">
                  Yes, send my free report
                </span>
                <br />
                <span className="text-slate-400">
                  I agree to receive my analysis report via email.{' '}
                  <button
                    type="button"
                    onClick={() => setShowDetails(!showDetails)}
                    className="text-emerald-400 hover:text-emerald-300 underline"
                  >
                    {showDetails ? 'Hide' : 'View'} details
                  </button>
                </span>
              </label>
            </div>

            {/* Expanded Consent Details */}
            {showDetails && (
              <div className="bg-slate-800/50 rounded-lg p-3 text-xs text-slate-400 space-y-2">
                <p>
                  <strong className="text-white">What we send:</strong> One email containing your full profit leak analysis report with specific SKUs, recommendations, and action items.
                </p>
                <p>
                  <strong className="text-white">Your data:</strong> Your uploaded file is processed, the report is generated, then your raw data is permanently deleted. We only keep anonymized, aggregated statistics (no SKUs or emails linked).
                </p>
                <p>
                  <strong className="text-white">Privacy:</strong> We never sell or share your email. You can unsubscribe anytime via the link in the email or by contacting{' '}
                  <a href="mailto:privacy@profitsentinel.com" className="text-emerald-400">privacy@profitsentinel.com</a>.
                </p>
                <p className="pt-1 border-t border-slate-700">
                  By checking this box, you consent to receive this one-time analysis report email. See our{' '}
                  <a href="/privacy" className="text-emerald-400 hover:underline">Privacy Policy</a> for full details. GDPR, CCPA & CAN-SPAM compliant.
                </p>
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isLoading || !email || !agreed || !isValidEmail(email)}
              className="w-full bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-bold text-xl py-4 rounded-xl hover:from-emerald-600 hover:to-emerald-700 transition transform hover:scale-[1.02] shadow-lg shadow-emerald-500/25 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none flex items-center justify-center gap-3"
            >
              {isLoading ? (
                <>
                  <svg className="animate-spin h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Generating Report...
                </>
              ) : (
                <>
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-6 h-6">
                    <path fillRule="evenodd" d="M10 1a4.5 4.5 0 00-4.5 4.5V9H5a2 2 0 00-2 2v6a2 2 0 002 2h10a2 2 0 002-2v-6a2 2 0 00-2-2h-.5V5.5A4.5 4.5 0 0010 1zm3 8V5.5a3 3 0 10-6 0V9h6z" clipRule="evenodd" />
                  </svg>
                  Unlock My Full Report (Free)
                </>
              )}
            </button>
          </form>

          {/* Trust Badges */}
          <div className="mt-6 flex items-center justify-center gap-4 text-xs text-slate-500">
            <div className="flex items-center gap-1">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 text-emerald-500">
                <path fillRule="evenodd" d="M10 1a4.5 4.5 0 00-4.5 4.5V9H5a2 2 0 00-2 2v6a2 2 0 002 2h10a2 2 0 002-2v-6a2 2 0 00-2-2h-.5V5.5A4.5 4.5 0 0010 1zm3 8V5.5a3 3 0 10-6 0V9h6z" clipRule="evenodd" />
              </svg>
              <span>256-bit SSL</span>
            </div>
            <span className="text-slate-700">|</span>
            <div className="flex items-center gap-1">
              <span>GDPR</span>
            </div>
            <span className="text-slate-700">|</span>
            <div className="flex items-center gap-1">
              <span>CCPA</span>
            </div>
            <span className="text-slate-700">|</span>
            <div className="flex items-center gap-1">
              <span>No Spam</span>
            </div>
          </div>

          {/* Social Proof */}
          <p className="text-center text-xs text-slate-500 mt-4">
            Join 1,000+ retail store owners who've found hidden profits
          </p>
        </div>
      </div>

      {/* CSS for shake animation */}
      <style jsx>{`
        @keyframes shake {
          0%, 100% { transform: translateX(0); }
          10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
          20%, 40%, 60%, 80% { transform: translateX(5px); }
        }
        .animate-shake {
          animation: shake 0.5s ease-in-out;
        }
      `}</style>
    </div>
  )
}

/**
 * Success Modal after email submission
 */
export function ReportSuccessModal({
  isOpen,
  email,
  onClose
}: {
  isOpen: boolean
  email: string
  onClose: () => void
}) {
  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/80 backdrop-blur-sm" onClick={onClose} />

      <div className="relative w-full max-w-md bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 rounded-3xl border border-emerald-500/30 shadow-2xl p-8 text-center">
        {/* Success Icon */}
        <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-emerald-500/20 flex items-center justify-center">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-10 h-10 text-emerald-400">
            <path fillRule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clipRule="evenodd" />
          </svg>
        </div>

        <h2 className="text-2xl font-bold text-emerald-400 mb-2">Report on the Way!</h2>

        <p className="text-slate-300 mb-6">
          Your full profit leak analysis is being sent to{' '}
          <span className="font-medium text-white">{email}</span>
        </p>

        <div className="bg-white/5 rounded-xl p-4 mb-6 text-left">
          <p className="text-sm font-semibold text-white mb-2">What's Next:</p>
          <ol className="text-sm text-slate-400 space-y-2">
            <li className="flex items-start gap-2">
              <span className="text-emerald-400 font-bold">1.</span>
              Check your inbox in 2-5 minutes
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-400 font-bold">2.</span>
              Check spam/promotions if you don't see it
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-400 font-bold">3.</span>
              Start fixing high-priority leaks first
            </li>
          </ol>
        </div>

        <button
          onClick={onClose}
          className="w-full bg-emerald-500/20 text-emerald-400 font-bold py-3 rounded-xl hover:bg-emerald-500/30 transition"
        >
          Got it!
        </button>

        <p className="text-xs text-slate-500 mt-4">
          Your uploaded data has been deleted per our privacy policy.
        </p>
      </div>
    </div>
  )
}

export default EmailUnlockModal
