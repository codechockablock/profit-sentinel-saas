'use client'

import Link from 'next/link'

/**
 * Footer - Site-wide footer with trust signals
 *
 * Includes:
 * - Terms, Privacy, Support email links
 * - Copyright
 * - LinkedIn social link
 * - Privacy badge
 */
export function Footer() {
  const currentYear = new Date().getFullYear()

  return (
    <footer className="border-t border-slate-800 bg-slate-900/50">
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Main Footer Content */}
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          {/* Logo and Tagline */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-emerald-400 to-emerald-600 flex items-center justify-center">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 20 20"
                fill="currentColor"
                className="w-5 h-5 text-white"
              >
                <path
                  fillRule="evenodd"
                  d="M9.661 2.237a.531.531 0 01.678 0 11.947 11.947 0 007.078 2.749.5.5 0 01.479.425c.069.52.104 1.05.104 1.59 0 5.162-3.26 9.563-7.834 11.256a.48.48 0 01-.332 0C5.26 16.564 2 12.163 2 7c0-.538.035-1.069.104-1.589a.5.5 0 01.48-.425 11.947 11.947 0 007.077-2.75z"
                  clipRule="evenodd"
                />
              </svg>
            </div>
            <div>
              <span className="text-lg font-bold text-white">Profit Sentinel</span>
              <p className="text-xs text-slate-500">Find the profit leaks hiding in your inventory</p>
            </div>
          </div>

          {/* Links */}
          <div className="flex flex-wrap items-center justify-center gap-6 text-sm">
            <Link
              href="/terms"
              className="text-slate-400 hover:text-emerald-400 transition"
            >
              Terms
            </Link>
            <Link
              href="/privacy"
              className="text-slate-400 hover:text-emerald-400 transition"
            >
              Privacy
            </Link>
            <Link
              href="/about"
              className="text-slate-400 hover:text-emerald-400 transition"
            >
              About
            </Link>
            <a
              href="mailto:support@profitsentinel.com"
              className="text-slate-400 hover:text-emerald-400 transition"
            >
              Support
            </a>
          </div>

          {/* Social & Copyright */}
          <div className="flex items-center gap-4">
            {/* LinkedIn */}
            <a
              href="https://www.linkedin.com/in/joseph-hopkins-716b65173/"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 text-slate-400 hover:text-emerald-400 transition rounded-lg hover:bg-slate-800"
              aria-label="LinkedIn"
            >
              <LinkedInIcon className="w-5 h-5" />
            </a>
          </div>
        </div>

        {/* Privacy Badge Row */}
        <div className="mt-6 pt-6 border-t border-slate-800 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-4 text-xs text-slate-500">
            <PrivacyBadge />
            <span className="hidden md:inline">|</span>
            <span>GDPR Compliant</span>
            <span className="hidden md:inline">|</span>
            <span>CCPA Compliant</span>
          </div>

          <p className="text-xs text-slate-500">
            &copy; {currentYear} Profit Sentinel. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  )
}

/**
 * Privacy Badge - Shows data handling commitment
 */
function PrivacyBadge() {
  return (
    <div className="inline-flex items-center gap-1.5 text-emerald-400/80">
      <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 20 20"
        fill="currentColor"
        className="w-4 h-4"
      >
        <path
          fillRule="evenodd"
          d="M9.661 2.237a.531.531 0 01.678 0 11.947 11.947 0 007.078 2.749.5.5 0 01.479.425c.069.52.104 1.05.104 1.59 0 5.162-3.26 9.563-7.834 11.256a.48.48 0 01-.332 0C5.26 16.564 2 12.163 2 7c0-.538.035-1.069.104-1.589a.5.5 0 01.48-.425 11.947 11.947 0 007.077-2.75z"
          clipRule="evenodd"
        />
      </svg>
      <span>Files encrypted & auto-deleted</span>
    </div>
  )
}

function LinkedInIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="currentColor"
      className={className}
    >
      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
    </svg>
  )
}

export default Footer
