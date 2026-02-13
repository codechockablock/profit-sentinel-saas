'use client'

import Link from 'next/link'
import { useState, useEffect } from 'react'
import { getSupabase } from '@/lib/supabase'
import { robustSignOut } from '@/lib/auth-helpers'
import { AuthModal } from '@/components/auth/AuthModal'

interface HeaderNavProps {
  /** Current page path for active state */
  currentPath?: string
}

/**
 * HeaderNav - Top-right navigation menu with auth controls
 *
 * Links: Home, Analyze, Dashboard, Roadmap, About, Contact
 * Auth: Sign In / Sign Out based on session state
 * Mobile: Hamburger menu
 */
export function HeaderNav({ currentPath }: HeaderNavProps) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [userEmail, setUserEmail] = useState<string | null>(null)
  const [showAuthModal, setShowAuthModal] = useState(false)
  const [authModalMode, setAuthModalMode] = useState<'login' | 'signup'>('login')
  const [signingOut, setSigningOut] = useState(false)

  useEffect(() => {
    const supabase = getSupabase()
    if (!supabase) return

    supabase.auth.getSession().then(({ data: { session } }: { data: { session: { user?: { email?: string } } | null } }) => {
      setIsAuthenticated(!!session)
      setUserEmail(session?.user?.email ?? null)
    })

    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event: string, session: { user?: { email?: string } } | null) => {
        setIsAuthenticated(!!session)
        setUserEmail(session?.user?.email ?? null)
      }
    )

    return () => subscription.unsubscribe()
  }, [])

  const handleSignOut = async () => {
    setSigningOut(true)
    await robustSignOut()
    setIsAuthenticated(false)
    setUserEmail(null)
    setSigningOut(false)
  }

  const navLinks = [
    { href: '/analyze', label: 'Analyze' },
    { href: '/dashboard', label: 'Dashboard' },
    { href: '/roadmap', label: 'Roadmap' },
    { href: '/about', label: 'About' },
    { href: '/contact', label: 'Contact' },
  ]

  return (
    <>
      <nav className="relative">
        {/* Desktop Navigation */}
        <div className="hidden md:flex items-center gap-6">
          {navLinks.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={`text-sm font-medium transition-colors hover:text-emerald-400 ${
                currentPath === link.href
                  ? 'text-emerald-400'
                  : 'text-slate-300'
              }`}
            >
              {link.label}
            </Link>
          ))}

          {/* Auth controls */}
          {isAuthenticated ? (
            <div className="flex items-center gap-3 ml-2">
              <span className="text-xs text-slate-500 truncate max-w-[140px]" title={userEmail ?? undefined}>
                {userEmail}
              </span>
              <button
                onClick={handleSignOut}
                disabled={signingOut}
                className="text-sm text-slate-400 hover:text-red-400 transition-colors disabled:opacity-50"
              >
                {signingOut ? '...' : 'Sign Out'}
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-2 ml-2">
              <button
                onClick={() => { setAuthModalMode('login'); setShowAuthModal(true) }}
                className="text-sm font-medium text-slate-300 hover:text-emerald-400 transition-colors"
              >
                Sign In
              </button>
              <Link
                href="/analyze"
                className="px-4 py-2 bg-gradient-to-r from-emerald-500 to-emerald-600 text-white text-sm font-bold rounded-lg hover:from-emerald-600 hover:to-emerald-700 transition transform hover:scale-105 shadow-lg shadow-emerald-500/25"
              >
                Analyze Free
              </Link>
            </div>
          )}
        </div>

        {/* Mobile Menu Button */}
        <button
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="md:hidden p-2 text-slate-300 hover:text-white transition"
          aria-label="Toggle menu"
        >
          {mobileMenuOpen ? (
            <XIcon className="w-6 h-6" />
          ) : (
            <MenuIcon className="w-6 h-6" />
          )}
        </button>

        {/* Mobile Menu Dropdown */}
        {mobileMenuOpen && (
          <div className="absolute top-full right-0 mt-2 w-56 bg-slate-800 border border-slate-700 rounded-xl shadow-xl overflow-hidden md:hidden z-50">
            {navLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                onClick={() => setMobileMenuOpen(false)}
                className={`block px-4 py-3 text-sm font-medium transition-colors hover:bg-slate-700 ${
                  currentPath === link.href
                    ? 'text-emerald-400 bg-slate-700/50'
                    : 'text-slate-300'
                }`}
              >
                {link.label}
              </Link>
            ))}

            <div className="border-t border-slate-700 p-3 space-y-2">
              {isAuthenticated ? (
                <>
                  <div className="px-1 text-xs text-slate-500 truncate">{userEmail}</div>
                  <button
                    onClick={() => { handleSignOut(); setMobileMenuOpen(false) }}
                    disabled={signingOut}
                    className="block w-full text-left px-4 py-2 text-sm text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
                  >
                    {signingOut ? 'Signing out...' : 'Sign Out'}
                  </button>
                </>
              ) : (
                <>
                  <button
                    onClick={() => { setAuthModalMode('login'); setShowAuthModal(true); setMobileMenuOpen(false) }}
                    className="block w-full text-center px-4 py-2 text-sm text-slate-300 border border-slate-600 rounded-lg hover:bg-slate-700 transition-colors"
                  >
                    Sign In
                  </button>
                  <Link
                    href="/analyze"
                    onClick={() => setMobileMenuOpen(false)}
                    className="block w-full text-center px-4 py-2 bg-gradient-to-r from-emerald-500 to-emerald-600 text-white text-sm font-bold rounded-lg hover:from-emerald-600 hover:to-emerald-700 transition"
                  >
                    Analyze Free
                  </Link>
                </>
              )}
            </div>
          </div>
        )}
      </nav>

      <AuthModal
        isOpen={showAuthModal}
        onClose={() => setShowAuthModal(false)}
        onSuccess={() => { setShowAuthModal(false) }}
        defaultMode={authModalMode}
      />
    </>
  )
}

// Icons
function MenuIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <line x1="3" y1="6" x2="21" y2="6" />
      <line x1="3" y1="12" x2="21" y2="12" />
      <line x1="3" y1="18" x2="21" y2="18" />
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

export default HeaderNav
