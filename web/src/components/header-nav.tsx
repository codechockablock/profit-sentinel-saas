'use client'

import Link from 'next/link'
import { useState } from 'react'

interface HeaderNavProps {
  /** Current page path for active state */
  currentPath?: string
}

/**
 * HeaderNav - Top-right navigation menu
 *
 * Links: About, Roadmap, Privacy
 * Mobile: Hamburger menu
 */
export function HeaderNav({ currentPath }: HeaderNavProps) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const navLinks = [
    { href: '/dashboard', label: 'Dashboard' },
    { href: '/about', label: 'About' },
    { href: '/diagnostic', label: 'Shrinkage Diagnostic' },
    { href: '/roadmap', label: 'Roadmap' },
    { href: '/privacy', label: 'Privacy' },
  ]

  return (
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

        {/* CTA Button */}
        <Link
          href="/analyze"
          className="ml-2 px-4 py-2 bg-gradient-to-r from-emerald-500 to-emerald-600 text-white text-sm font-bold rounded-lg hover:from-emerald-600 hover:to-emerald-700 transition transform hover:scale-105 shadow-lg shadow-emerald-500/25"
        >
          Analyze Free
        </Link>
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
        <div className="absolute top-full right-0 mt-2 w-48 bg-slate-800 border border-slate-700 rounded-xl shadow-xl overflow-hidden md:hidden z-50">
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
          <div className="border-t border-slate-700 p-3">
            <Link
              href="/analyze"
              onClick={() => setMobileMenuOpen(false)}
              className="block w-full text-center px-4 py-2 bg-gradient-to-r from-emerald-500 to-emerald-600 text-white text-sm font-bold rounded-lg hover:from-emerald-600 hover:to-emerald-700 transition"
            >
              Analyze Free
            </Link>
          </div>
        </div>
      )}
    </nav>
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
