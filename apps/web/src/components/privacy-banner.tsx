'use client'

import { useState } from 'react'

interface PrivacyBannerProps {
  /** Variant: 'default' shows full banner, 'compact' shows minimal */
  variant?: 'default' | 'compact'
  /** Whether to show the banner initially */
  defaultVisible?: boolean
  /** Callback when user dismisses the banner */
  onDismiss?: () => void
  /** Custom class name */
  className?: string
}

/**
 * PrivacyBanner - Communicates data handling practices
 *
 * Shows users that:
 * 1. Files are encrypted in transit and at rest
 * 2. Files are auto-deleted within 24 hours
 * 3. Only aggregate statistics are retained (no SKUs)
 *
 * Two variants:
 * - default: Full banner with icon and detailed message
 * - compact: Minimal inline text for space-constrained areas
 */
export function PrivacyBanner({
  variant = 'default',
  defaultVisible = true,
  onDismiss,
  className = '',
}: PrivacyBannerProps) {
  const [visible, setVisible] = useState(defaultVisible)

  const handleDismiss = () => {
    setVisible(false)
    onDismiss?.()
  }

  if (!visible) return null

  if (variant === 'compact') {
    return (
      <div className={`flex items-center gap-2 text-xs text-slate-400 ${className}`}>
        <ShieldIcon className="w-3.5 h-3.5 text-emerald-500/70" />
        <span>
          Files encrypted & {' '}
          <span className="text-emerald-400/80">auto-deleted in 24 hours.</span>
        </span>
      </div>
    )
  }

  return (
    <div
      className={`relative bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-4 ${className}`}
    >
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0 mt-0.5">
          <ShieldIcon className="w-5 h-5 text-emerald-400" />
        </div>
        <div className="flex-1 min-w-0">
          <h4 className="text-sm font-semibold text-emerald-300 mb-1">
            Your Data Stays Protected
          </h4>
          <p className="text-xs text-slate-300 leading-relaxed">
            Files are <span className="text-emerald-400 font-medium">encrypted in transit and at rest</span>,
            then automatically deleted within 24 hours. We only retain aggregate statistics
            (counts, not SKUs) to improve our detection algorithms.
          </p>
          <div className="mt-3 flex flex-wrap gap-3 text-xs">
            <PrivacyPoint icon={<LockClosedIcon />} text="Encrypted storage" />
            <PrivacyPoint icon={<ClockIcon />} text="Auto-deleted in 24 hours" />
            <PrivacyPoint icon={<EyeOffIcon />} text="No SKUs retained" />
          </div>
        </div>
        {onDismiss && (
          <button
            onClick={handleDismiss}
            className="flex-shrink-0 p-1 text-slate-400 hover:text-white transition rounded"
            aria-label="Dismiss"
          >
            <XIcon className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  )
}

function PrivacyPoint({ icon, text }: { icon: React.ReactNode; text: string }) {
  return (
    <div className="flex items-center gap-1.5 text-slate-400">
      <span className="text-emerald-500/70">{icon}</span>
      <span>{text}</span>
    </div>
  )
}

/**
 * PrivacyBadge - Minimal inline privacy indicator
 *
 * Use next to file upload areas or sensitive data displays
 */
export function PrivacyBadge({ className = '' }: { className?: string }) {
  return (
    <div
      className={`inline-flex items-center gap-1.5 px-2 py-1 bg-emerald-500/10 border border-emerald-500/30 rounded-full text-xs text-emerald-400 ${className}`}
    >
      <ShieldIcon className="w-3 h-3" />
      <span>Encrypted & auto-deleted</span>
    </div>
  )
}

/**
 * PrivacyFooter - Footer text for forms and modals
 */
export function PrivacyFooter({ className = '' }: { className?: string }) {
  return (
    <p className={`text-xs text-slate-500 ${className}`}>
      <ShieldIcon className="w-3 h-3 inline-block mr-1 text-emerald-500/50" />
      We respect your privacy. Your files are encrypted and automatically deleted within 24 hours.
      We only retain anonymized statistics. Your email is stored only if you opt in.{' '}
      <a
        href="/privacy"
        className="text-emerald-400/70 hover:text-emerald-400 underline"
      >
        Privacy Policy
      </a>
    </p>
  )
}

// Icons
function ShieldIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 20 20"
      fill="currentColor"
      className={className}
    >
      <path
        fillRule="evenodd"
        d="M9.661 2.237a.531.531 0 01.678 0 11.947 11.947 0 007.078 2.749.5.5 0 01.479.425c.069.52.104 1.05.104 1.59 0 5.162-3.26 9.563-7.834 11.256a.48.48 0 01-.332 0C5.26 16.564 2 12.163 2 7c0-.538.035-1.069.104-1.589a.5.5 0 01.48-.425 11.947 11.947 0 007.077-2.75z"
        clipRule="evenodd"
      />
    </svg>
  )
}

function ClockIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 16 16"
      fill="currentColor"
      className="w-3.5 h-3.5"
    >
      <path
        fillRule="evenodd"
        d="M8 15A7 7 0 1 0 8 1a7 7 0 0 0 0 14Zm.75-10.25a.75.75 0 0 0-1.5 0v4c0 .414.336.75.75.75h3a.75.75 0 0 0 0-1.5H8.75v-3.25Z"
        clipRule="evenodd"
      />
    </svg>
  )
}

function LockClosedIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 16 16"
      fill="currentColor"
      className="w-3.5 h-3.5"
    >
      <path
        fillRule="evenodd"
        d="M8 1a3.5 3.5 0 0 0-3.5 3.5V7A1.5 1.5 0 0 0 3 8.5v5A1.5 1.5 0 0 0 4.5 15h7a1.5 1.5 0 0 0 1.5-1.5v-5A1.5 1.5 0 0 0 11.5 7V4.5A3.5 3.5 0 0 0 8 1Zm2 6V4.5a2 2 0 1 0-4 0V7h4Z"
        clipRule="evenodd"
      />
    </svg>
  )
}

function EyeOffIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 16 16"
      fill="currentColor"
      className="w-3.5 h-3.5"
    >
      <path
        fillRule="evenodd"
        d="M3.28 2.22a.75.75 0 0 0-1.06 1.06l10.5 10.5a.75.75 0 1 0 1.06-1.06l-1.527-1.527A6.97 6.97 0 0 0 15 8c-1.146-3.322-4.095-5.5-7-5.5-1.1 0-2.168.271-3.146.76L3.28 2.22ZM8 4.5a4.019 4.019 0 0 1 2.094.586L4.586 10.594A4.019 4.019 0 0 1 4 8.5C4 6.015 5.79 4.5 8 4.5Z"
        clipRule="evenodd"
      />
      <path d="M7.557 13.479A6.975 6.975 0 0 1 1 8c.37-1.07.955-2.036 1.706-2.843L3.768 6.22A4.974 4.974 0 0 0 3 8.5C3 10.985 4.79 12.5 8 12.5c.471 0 .924-.04 1.357-.115l1.063 1.063a6.95 6.95 0 0 1-2.863.031Z" />
    </svg>
  )
}

function XIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 20 20"
      fill="currentColor"
      className={className}
    >
      <path d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z" />
    </svg>
  )
}

export default PrivacyBanner
