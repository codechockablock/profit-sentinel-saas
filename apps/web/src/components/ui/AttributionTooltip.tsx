'use client'

import * as React from 'react'
import type { ColumnAttribution } from '@/lib/column-attribution'

interface AttributionTooltipProps {
  /** Attribution data for this field. If undefined, renders nothing. */
  attribution: ColumnAttribution | undefined
  className?: string
}

/**
 * Attribution Tooltip — shows data lineage for a single analysis field.
 *
 * Renders a subtle ⓘ icon that reveals a tooltip on hover (desktop),
 * tap (mobile), or focus (keyboard). The tooltip shows where the data
 * came from in the user's CSV, how derived fields are calculated, and
 * what the field is used to detect.
 *
 * Three rendering modes:
 * 1. Direct mapping — "Source: In Stock Qty. (Column G)"
 * 2. Derived/calculated — "Calculated from: Cost, Retail" with formula
 * 3. Defaulted — "Not in your data. Defaulted to: default"
 */
export function AttributionTooltip({ attribution, className = '' }: AttributionTooltipProps) {
  const [isVisible, setIsVisible] = React.useState(false)
  const containerRef = React.useRef<HTMLDivElement>(null)
  const timerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)

  // Click-outside to dismiss (for mobile tap-to-toggle)
  React.useEffect(() => {
    if (!isVisible) return

    function handleClickOutside(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsVisible(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [isVisible])

  // Keyboard: Escape to dismiss
  React.useEffect(() => {
    if (!isVisible) return

    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        setIsVisible(false)
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [isVisible])

  if (!attribution) return null

  const handleMouseEnter = () => {
    timerRef.current = setTimeout(() => setIsVisible(true), 150)
  }

  const handleMouseLeave = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }
    setIsVisible(false)
  }

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    setIsVisible((prev) => !prev)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      setIsVisible((prev) => !prev)
    }
  }

  const tooltipId = `attr-tip-${attribution.normalizedField}`

  return (
    <div
      ref={containerRef}
      className={`relative inline-flex items-center ${className}`}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {/* Trigger: subtle info icon */}
      <button
        type="button"
        className="inline-flex items-center justify-center w-3.5 h-3.5 rounded-full text-slate-600 hover:text-slate-400 hover:bg-white/10 transition-all duration-150 cursor-help opacity-0 group-hover/metrics:opacity-40 hover:!opacity-100 focus:opacity-100"
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        aria-label={`Data source for ${attribution.displayLabelFull}`}
        aria-describedby={isVisible ? tooltipId : undefined}
        tabIndex={0}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 20 20"
          fill="currentColor"
          className="w-3 h-3"
        >
          <path
            fillRule="evenodd"
            d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a.75.75 0 000 1.5h.253a.25.25 0 01.244.304l-.459 2.066A1.75 1.75 0 0010.747 15H11a.75.75 0 000-1.5h-.253a.25.25 0 01-.244-.304l.459-2.066A1.75 1.75 0 009.253 9H9z"
            clipRule="evenodd"
          />
        </svg>
      </button>

      {/* Tooltip Panel */}
      {isVisible && (
        <div
          id={tooltipId}
          role="tooltip"
          className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 transition-all duration-150 ease-out"
          style={{ maxWidth: '340px', minWidth: '260px' }}
        >
          <div className="bg-slate-800 text-slate-200 text-xs rounded-lg shadow-2xl border border-slate-600/80 overflow-hidden">
            {/* Header */}
            <div className="px-3 pt-3 pb-2">
              <div className="font-semibold text-slate-100 text-sm">
                {attribution.displayLabelFull}
              </div>
              <div className="mt-1.5 h-px bg-slate-700" />
            </div>

            <div className="px-3 pb-3 space-y-2.5">
              {/* Source Section */}
              {!attribution.isDefaulted && attribution.sourceColumns.length > 0 && (
                <SourceSection attribution={attribution} />
              )}

              {/* Formula Section (derived fields) */}
              {attribution.isDerived && !attribution.isDefaulted && attribution.formula && (
                <FormulaSection attribution={attribution} />
              )}

              {/* Defaulted Section */}
              {attribution.isDefaulted && (
                <DefaultedSection attribution={attribution} />
              )}

              {/* Explanation */}
              {attribution.detects.length > 0 && (
                <div className="text-slate-400 leading-relaxed">
                  <span className="text-slate-500">Used to detect: </span>
                  {attribution.detects.join(', ')}
                </div>
              )}

              {/* Benchmark */}
              {attribution.benchmark && (
                <div className="pt-1.5 border-t border-slate-700/60">
                  <span className="text-emerald-400/80">
                    {'📈 '}
                  </span>
                  <span className="text-slate-400">
                    Benchmark: {attribution.benchmark}
                  </span>
                  {attribution.benchmarkSource && (
                    <span className="text-slate-600 ml-1">
                      ({attribution.benchmarkSource})
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Arrow */}
          <div className="absolute top-full left-1/2 -translate-x-1/2 w-0 h-0 border-[6px] border-t-slate-800 border-x-transparent border-b-transparent" />
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Sub-sections
// ---------------------------------------------------------------------------

function SourceSection({ attribution }: { attribution: ColumnAttribution }) {
  return (
    <div className="text-slate-300 leading-relaxed">
      <span className="text-slate-500/80">{'📊 '}</span>
      <span className="text-slate-500">Source: </span>
      {attribution.sourceColumns.map((col, i) => (
        <span key={col}>
          {i > 0 && <span className="text-slate-600"> and </span>}
          <span className="text-slate-200 font-medium">&ldquo;{col}&rdquo;</span>
          {attribution.sourceColumnLetters[i] && (
            <span className="text-slate-500 ml-1">
              (Column {attribution.sourceColumnLetters[i]})
            </span>
          )}
        </span>
      ))}
    </div>
  )
}

function FormulaSection({ attribution }: { attribution: ColumnAttribution }) {
  return (
    <div className="bg-slate-900/60 rounded px-2.5 py-1.5 font-mono text-[11px] text-emerald-300/80">
      <span className="text-slate-500 not-italic mr-1.5">ƒ</span>
      {attribution.formulaDisplay ?? attribution.formula}
    </div>
  )
}

function DefaultedSection({ attribution }: { attribution: ColumnAttribution }) {
  return (
    <div className="space-y-1.5">
      <div className="text-amber-400/80 leading-relaxed">
        <span>{'⚠️ '}</span>
        Not found in your data
      </div>
      {attribution.searchedColumns && attribution.searchedColumns.length > 0 && (
        <div className="text-slate-500 leading-relaxed">
          We looked for:{' '}
          {attribution.searchedColumns.map((col, i) => (
            <span key={col}>
              {i > 0 && ', '}
              &ldquo;{col}&rdquo;
            </span>
          ))}
        </div>
      )}
      {attribution.defaultValue !== null && (
        <div className="text-slate-400 leading-relaxed">
          Defaulted to:{' '}
          <span className="font-mono text-slate-300">{attribution.defaultValue}</span>
        </div>
      )}
    </div>
  )
}
