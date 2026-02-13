'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { HeaderNav } from './header-nav'
import { Footer } from './footer'
import { PrivacyBanner } from './privacy-banner'

interface AppShellProps {
  children: React.ReactNode
  /** Show sidebar (for chat/channel views) */
  showSidebar?: boolean
  /** Show privacy banner at top */
  showPrivacyBanner?: boolean
  /** Show footer */
  showFooter?: boolean
}

/**
 * AppShell - Main layout wrapper for the application
 *
 * Provides consistent header, footer, and privacy messaging.
 */
export function AppShell({
  children,
  showSidebar: _showSidebar = false,
  showPrivacyBanner = false,
  showFooter = true,
}: AppShellProps) {
  const pathname = usePathname()

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-b from-slate-900 via-slate-900 to-slate-800">
      {/* Header */}
      <header className="sticky top-0 z-40 border-b border-slate-800 bg-slate-900/90 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <Link href="/" className="flex items-center gap-3 group">
              <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-emerald-400 to-emerald-600 flex items-center justify-center group-hover:scale-110 transition-transform">
                <ShieldIcon className="w-5 h-5 text-white" />
              </div>
              <div>
                <span className="text-xl font-bold text-white">
                  Profit <span className="text-emerald-400">Sentinel</span>
                </span>
              </div>
            </Link>

            {/* Navigation */}
            <HeaderNav currentPath={pathname} />
          </div>
        </div>
      </header>

      {/* Privacy Banner */}
      {showPrivacyBanner && (
        <div className="max-w-7xl mx-auto px-4 pt-4">
          <PrivacyBanner />
        </div>
      )}

      {/* Main Content */}
      <main className="flex-1">
        {children}
      </main>

      {/* Footer */}
      {showFooter && <Footer />}
    </div>
  )
}

function ShieldIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className={className}>
      <path fillRule="evenodd" d="M9.661 2.237a.531.531 0 01.678 0 11.947 11.947 0 007.078 2.749.5.5 0 01.479.425c.069.52.104 1.05.104 1.59 0 5.162-3.26 9.563-7.834 11.256a.48.48 0 01-.332 0C5.26 16.564 2 12.163 2 7c0-.538.035-1.069.104-1.589a.5.5 0 01.48-.425 11.947 11.947 0 007.077-2.75z" clipRule="evenodd" />
    </svg>
  )
}

export default AppShell
