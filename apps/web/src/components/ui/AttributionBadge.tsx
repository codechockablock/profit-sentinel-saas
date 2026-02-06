'use client'

import type { ColumnAttribution } from '@/lib/column-attribution'

interface AttributionBadgeProps {
  /** Attribution data for this field. If undefined, renders nothing. */
  attribution: ColumnAttribution | undefined
  className?: string
}

/**
 * Attribution Badge — tiny inline pill showing data provenance at a glance.
 *
 * Three variants:
 * - Direct: "Col G" — which CSV column this came from
 * - Derived: "Calc" — calculated from other fields
 * - Defaulted: "Default" — not in the user's data, using a default
 *
 * Designed to be barely noticeable until the user looks for it.
 * Complements AttributionTooltip for full progressive disclosure.
 */
export function AttributionBadge({ attribution, className = '' }: AttributionBadgeProps) {
  if (!attribution) return null

  let label: string
  let badgeClass: string

  if (attribution.isDefaulted) {
    label = 'Default'
    badgeClass = 'text-amber-600/60 bg-amber-500/5'
  } else if (attribution.isDerived && attribution.sourceColumns.length === 0) {
    label = 'Calc'
    badgeClass = 'text-emerald-600/60 bg-emerald-500/5'
  } else if (attribution.sourceColumnLetters.length > 0) {
    label = `Col ${attribution.sourceColumnLetters[0]}`
    badgeClass = 'text-slate-500/60 bg-slate-500/5'
  } else {
    return null
  }

  return (
    <span
      className={`inline-block text-[9px] leading-none font-mono px-1 py-0.5 rounded opacity-0 group-hover/metrics:opacity-60 transition-opacity duration-150 ${badgeClass} ${className}`}
      aria-hidden="true"
    >
      {label}
    </span>
  )
}
