'use client'

import { useState } from 'react'

interface EmailOptInProps {
  email: string
  onEmailChange: (email: string) => void
  optedIn: boolean
  onOptInChange: (optedIn: boolean) => void
  disabled?: boolean
}

/**
 * Email Opt-In Component with GDPR/CCPA Compliant Disclaimer
 *
 * Compliance requirements implemented:
 * - Explicit opt-in (checkbox NOT pre-checked)
 * - Clear explanation of what user is consenting to
 * - Easy way to withdraw consent (unsubscribe mention)
 * - Privacy policy link
 * - Data handling transparency
 */
export function EmailOptIn({
  email,
  onEmailChange,
  optedIn,
  onOptInChange,
  disabled = false
}: EmailOptInProps) {
  const [showDetails, setShowDetails] = useState(false)

  return (
    <div className="bg-white/5 rounded-2xl border border-slate-700 p-6 mt-6">
      {/* Email Input */}
      <div className="mb-4">
        <label className="block text-lg font-semibold mb-2 text-slate-200">
          Email Address
        </label>
        <input
          type="email"
          value={email}
          onChange={(e) => onEmailChange(e.target.value)}
          placeholder="you@yourstore.com"
          disabled={disabled}
          className="w-full px-4 py-3 text-lg bg-white/5 border-2 border-slate-600 rounded-xl
                     focus:border-emerald-400 focus:ring-4 focus:ring-emerald-400/20 transition
                     disabled:opacity-50 disabled:cursor-not-allowed"
        />
      </div>

      {/* Opt-in Checkbox */}
      <div className="flex items-start gap-3">
        <input
          type="checkbox"
          id="email-opt-in"
          checked={optedIn}
          onChange={(e) => onOptInChange(e.target.checked)}
          disabled={disabled || !email}
          className="mt-1 w-5 h-5 rounded border-slate-500 bg-white/10
                     text-emerald-500 focus:ring-emerald-500 focus:ring-offset-0
                     disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
        />
        <label htmlFor="email-opt-in" className="text-sm text-slate-300 cursor-pointer">
          <span className="font-medium text-white">
            Yes, send me a detailed profit leak report via email
          </span>
          <br />
          <span className="text-slate-400">
            I agree to receive a one-time analysis report with actionable recommendations.
            {' '}
            <button
              type="button"
              onClick={() => setShowDetails(!showDetails)}
              className="text-emerald-400 hover:text-emerald-300 underline"
            >
              {showDetails ? 'Hide details' : 'Learn more'}
            </button>
          </span>
        </label>
      </div>

      {/* Expanded Details */}
      {showDetails && (
        <div className="mt-4 p-4 bg-slate-800/50 rounded-xl text-sm text-slate-300 space-y-3">
          <div>
            <p className="font-medium text-white mb-1">What you'll receive:</p>
            <ul className="list-disc list-inside space-y-1 text-slate-400">
              <li>Full PDF/HTML report of your profit leak analysis</li>
              <li>Detailed breakdown of each leak category</li>
              <li>Prioritized action items based on estimated impact</li>
              <li>Comparison benchmarks (anonymized industry data)</li>
            </ul>
          </div>

          <div>
            <p className="font-medium text-white mb-1">How we handle your data:</p>
            <ul className="list-disc list-inside space-y-1 text-slate-400">
              <li>Your uploaded file data is processed and then deleted</li>
              <li>We store only anonymized, aggregated statistics</li>
              <li>Your email is used solely to send this report</li>
              <li>We do not sell or share your data with third parties</li>
              <li>You can request data deletion at any time</li>
            </ul>
          </div>

          <div className="pt-2 border-t border-slate-700">
            <p className="text-xs text-slate-500">
              By checking this box, you consent to receive email communications from
              Profit Sentinel regarding your analysis results. You may unsubscribe at
              any time by clicking the unsubscribe link in the email or by contacting
              {' '}<a href="mailto:privacy@profitsentinel.com" className="text-emerald-400 hover:underline">
                privacy@profitsentinel.com
              </a>.
              See our{' '}
              <a href="/privacy" className="text-emerald-400 hover:underline">
                Privacy Policy
              </a>{' '}
              for more information on how we handle your data.
            </p>
          </div>
        </div>
      )}

      {/* Compliance Notice */}
      <div className="mt-4 flex items-center gap-2 text-xs text-slate-500">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
          <path fillRule="evenodd" d="M10 1a4.5 4.5 0 00-4.5 4.5V9H5a2 2 0 00-2 2v6a2 2 0 002 2h10a2 2 0 002-2v-6a2 2 0 00-2-2h-.5V5.5A4.5 4.5 0 0010 1zm3 8V5.5a3 3 0 10-6 0V9h6z" clipRule="evenodd" />
        </svg>
        <span>
          GDPR, CCPA & CAN-SPAM compliant. Your data is encrypted and secure.
        </span>
      </div>
    </div>
  )
}

/**
 * Report Request Success Message
 */
export function ReportRequestSuccess({ email }: { email: string }) {
  return (
    <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-2xl p-6 mt-6">
      <div className="flex items-start gap-4">
        <div className="flex-shrink-0 w-10 h-10 rounded-full bg-emerald-500/20 flex items-center justify-center">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5 text-emerald-400">
            <path fillRule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clipRule="evenodd" />
          </svg>
        </div>
        <div>
          <h4 className="text-lg font-bold text-emerald-400">Report Requested!</h4>
          <p className="text-slate-300 mt-1">
            We're generating your detailed profit leak report and will send it to{' '}
            <span className="font-medium text-white">{email}</span> shortly.
          </p>
          <p className="text-sm text-slate-400 mt-2">
            Check your inbox in 2-5 minutes. Don't forget to check spam/promotions folders.
          </p>
        </div>
      </div>
    </div>
  )
}

export default EmailOptIn
