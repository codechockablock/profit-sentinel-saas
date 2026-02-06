'use client'

import * as React from 'react'

interface TooltipProps {
  children: React.ReactNode
  content: React.ReactNode
  position?: 'top' | 'bottom' | 'left' | 'right'
  maxWidth?: string
  /** Delay in ms before showing tooltip (prevents flicker on sweep) */
  delay?: number
}

/**
 * Lightweight Tooltip Component
 * No external dependencies - pure CSS hover-based tooltip
 */
export function Tooltip({
  children,
  content,
  position = 'top',
  maxWidth = '320px',
  delay = 0,
}: TooltipProps) {
  const [isHovered, setIsHovered] = React.useState(false)
  const [isVisible, setIsVisible] = React.useState(false)
  const timerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)

  React.useEffect(() => {
    if (isHovered) {
      if (delay > 0) {
        timerRef.current = setTimeout(() => setIsVisible(true), delay)
      } else {
        setIsVisible(true)
      }
    } else {
      if (timerRef.current) {
        clearTimeout(timerRef.current)
        timerRef.current = null
      }
      setIsVisible(false)
    }
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [isHovered, delay])

  const positionClasses = {
    top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
    left: 'right-full top-1/2 -translate-y-1/2 mr-2',
    right: 'left-full top-1/2 -translate-y-1/2 ml-2'
  }

  const arrowClasses = {
    top: 'top-full left-1/2 -translate-x-1/2 border-t-slate-700 border-x-transparent border-b-transparent',
    bottom: 'bottom-full left-1/2 -translate-x-1/2 border-b-slate-700 border-x-transparent border-t-transparent',
    left: 'left-full top-1/2 -translate-y-1/2 border-l-slate-700 border-y-transparent border-r-transparent',
    right: 'right-full top-1/2 -translate-y-1/2 border-r-slate-700 border-y-transparent border-l-transparent'
  }

  return (
    <div
      className="relative inline-flex items-center"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onFocus={() => setIsHovered(true)}
      onBlur={() => setIsHovered(false)}
    >
      {children}

      {isVisible && content && (
        <div
          className={`absolute z-50 ${positionClasses[position]} transition-all duration-150 ease-out`}
          style={{ maxWidth }}
          role="tooltip"
        >
          <div className="bg-slate-700 text-slate-100 text-sm rounded-lg shadow-xl border border-slate-600 p-3">
            {content}
          </div>
          <div
            className={`absolute w-0 h-0 border-[6px] ${arrowClasses[position]}`}
          />
        </div>
      )}
    </div>
  )
}

/**
 * Info Icon with built-in tooltip
 */
export function InfoTooltip({
  content,
  className = ''
}: {
  content: React.ReactNode
  className?: string
}) {
  return (
    <Tooltip content={content} position="top" maxWidth="360px">
      <button
        type="button"
        className={`inline-flex items-center justify-center w-5 h-5 rounded-full bg-white/10 hover:bg-white/20 text-slate-400 hover:text-slate-200 transition-colors cursor-help ${className}`}
        aria-label="More information"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 20 20"
          fill="currentColor"
          className="w-3.5 h-3.5"
        >
          <path
            fillRule="evenodd"
            d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a.75.75 0 000 1.5h.253a.25.25 0 01.244.304l-.459 2.066A1.75 1.75 0 0010.747 15H11a.75.75 0 000-1.5h-.253a.25.25 0 01-.244-.304l.459-2.066A1.75 1.75 0 009.253 9H9z"
            clipRule="evenodd"
          />
        </svg>
      </button>
    </Tooltip>
  )
}
