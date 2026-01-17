'use client'

import { useState } from 'react'
import { InfoTooltip } from './ui/tooltip'
import {
  LEAK_METADATA,
  getSeverityBadge,
  scoreToRiskLabel,
  formatDollarImpact,
  estimateItemImpact
} from '@/lib/leak-metadata'

interface BlurredLeakCardProps {
  primitive: string
  items: string[]
  scores: number[]
  count?: number
  impactEstimate?: number
  isUnlocked: boolean
  onUnlockClick: () => void
  /** Number of items to reveal (default: 1) */
  revealCount?: number
  /** Number of items to show in preview (default: 5) */
  previewCount?: number
  /** Whether to show impact estimates */
  showImpact?: boolean
  /** Whether to show recommendations */
  showRecommendations?: boolean
}

/**
 * Generate anonymized item label (Item A, Item B, etc.)
 */
function anonymizeItem(index: number): string {
  const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
  return `Item ${letters[index % 26]}${index >= 26 ? Math.floor(index / 26) : ''}`
}

/**
 * BlurredLeakCard - Reusable component with trust hook pattern
 *
 * Shows 4 blurred + 1 revealed item by default to demonstrate value
 * and encourage email signup for full access.
 *
 * Trust Hook Flow:
 * 1. User uploads file, sees analysis
 * 2. 4 items blurred, 1 revealed (teaser)
 * 3. "Unlock full report" CTA
 * 4. Email capture modal
 * 5. Full report access
 */
export function BlurredLeakCard({
  primitive,
  items,
  scores,
  count,
  impactEstimate,
  isUnlocked,
  onUnlockClick,
  revealCount = 1,
  previewCount = 5,
  showImpact = true,
  showRecommendations = true,
}: BlurredLeakCardProps) {
  const [expanded, setExpanded] = useState(true)
  const metadata = LEAK_METADATA[primitive]

  if (!metadata) {
    return (
      <div className="bg-red-500/10 border border-red-500/50 rounded-lg p-4">
        <p className="text-red-400 text-sm">Unknown primitive: {primitive}</p>
      </div>
    )
  }

  const severityBadge = getSeverityBadge(metadata.severity)
  const itemCount = count ?? items.length
  const previewItems = items.slice(0, previewCount)
  const hasItems = previewItems.length > 0

  return (
    <div
      className="bg-white/5 rounded-xl border overflow-hidden transition-all duration-200 hover:bg-white/[0.07]"
      style={{ borderColor: metadata.borderColor }}
    >
      {/* Header */}
      <div className="p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-2">
            <h4 className="text-lg font-semibold" style={{ color: metadata.color }}>
              {metadata.title}
            </h4>
            <InfoTooltip
              content={
                <div className="space-y-3 max-w-xs">
                  <p className="font-medium text-white">{metadata.title}</p>
                  <p className="text-slate-300 text-sm">{metadata.plainEnglish}</p>
                </div>
              }
            />
          </div>
          <span className={`px-2 py-0.5 text-xs font-bold rounded border ${severityBadge.className}`}>
            {severityBadge.label}
          </span>
        </div>

        {/* Stats row */}
        <div className="mt-3 flex items-center gap-4 text-sm">
          <div className="flex items-center gap-1.5">
            <span className="text-slate-400">Items Found:</span>
            <span className="font-bold text-white">{itemCount}</span>
          </div>
          {showImpact && impactEstimate && impactEstimate > 0 && (
            <div className="flex items-center gap-1.5">
              <span className="text-slate-400">Est. Impact:</span>
              <span className="font-bold text-emerald-400 text-lg">
                {formatDollarImpact(impactEstimate)}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Items list */}
      {hasItems && (
        <div className="border-t border-white/10">
          <button
            onClick={() => setExpanded(!expanded)}
            className="w-full px-4 py-2 flex items-center justify-between text-sm text-slate-400 hover:bg-white/5 transition-colors"
          >
            <span>{expanded ? 'Hide preview' : 'Show preview'}</span>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 20 20"
              fill="currentColor"
              className={`w-4 h-4 transition-transform ${expanded ? 'rotate-180' : ''}`}
            >
              <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" clipRule="evenodd" />
            </svg>
          </button>

          {expanded && (
            <div className="px-4 pb-4 space-y-2">
              {previewItems.map((item, i) => {
                const score = scores[i] ?? 0
                const risk = scoreToRiskLabel(score)
                const impact = showImpact ? estimateItemImpact(primitive, score) : null
                const isRevealed = isUnlocked || i < revealCount

                return (
                  <div
                    key={i}
                    className="relative flex items-center justify-between py-2 px-3 bg-white/5 rounded-lg overflow-hidden"
                  >
                    {/* Item name - revealed or anonymized */}
                    <div className="flex items-center gap-3">
                      {isRevealed ? (
                        <span className="font-mono text-sm text-slate-200">
                          {item || 'Unknown'}
                        </span>
                      ) : (
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-sm text-amber-400/60 blur-[2px]">
                            {item?.slice(0, 8) || 'XXXX'}...
                          </span>
                          <span className="text-xs text-slate-500 italic">
                            (locked)
                          </span>
                        </div>
                      )}
                    </div>

                    {/* Score and impact */}
                    <div className="flex items-center gap-3">
                      {impact && (
                        <span className="text-xs text-emerald-400/80 font-semibold">
                          ~${impact.low}-{impact.high}
                        </span>
                      )}
                      <span className={`px-2 py-0.5 text-xs font-bold rounded border ${risk.className}`}>
                        {risk.percentage}
                      </span>
                    </div>

                    {/* Blur overlay for locked items */}
                    {!isRevealed && (
                      <div className="absolute inset-0 bg-gradient-to-r from-slate-900/40 via-slate-900/60 to-slate-900/80 backdrop-blur-[1px] pointer-events-none" />
                    )}
                  </div>
                )
              })}

              {/* "More items" indicator */}
              {itemCount > previewCount && (
                <div className="relative">
                  <div className="py-2 px-3 bg-white/5 rounded-lg text-center">
                    <span className="text-sm text-slate-400">
                      +{itemCount - previewCount} more items
                    </span>
                  </div>
                  {!isUnlocked && (
                    <div className="absolute inset-0 bg-slate-900/70 backdrop-blur-[2px] rounded-lg flex items-center justify-center">
                      <button
                        onClick={onUnlockClick}
                        className="flex items-center gap-1.5 text-amber-400 hover:text-amber-300 transition text-sm font-medium"
                      >
                        <LockIcon className="w-4 h-4" />
                        Unlock Full Report
                      </button>
                    </div>
                  )}
                </div>
              )}

              {/* Recommendations section */}
              {showRecommendations && (
                <div className="mt-4 pt-4 border-t border-white/10 relative">
                  {isUnlocked ? (
                    <>
                      <p className="text-xs font-semibold text-slate-400 mb-2">
                        Recommended Actions:
                      </p>
                      <ul className="space-y-1">
                        {metadata.recommendations.slice(0, 3).map((rec, i) => (
                          <li key={i} className="flex items-start gap-2 text-xs text-slate-300">
                            <span className="text-emerald-500 mt-0.5">-</span>
                            {rec}
                          </li>
                        ))}
                      </ul>
                    </>
                  ) : (
                    <>
                      <div className="blur-sm select-none pointer-events-none">
                        <p className="text-xs font-semibold text-slate-400 mb-2">
                          Expert Recommendations:
                        </p>
                        <ul className="space-y-1">
                          {metadata.recommendations.slice(0, 2).map((rec, i) => (
                            <li key={i} className="flex items-start gap-2 text-xs text-slate-300">
                              <span className="text-emerald-500 mt-0.5">-</span>
                              {rec}
                            </li>
                          ))}
                        </ul>
                      </div>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <span className="text-xs text-amber-400 bg-slate-900/90 px-3 py-1 rounded-full">
                          Unlock for action items
                        </span>
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Unlock CTA for empty cards */}
      {!hasItems && !isUnlocked && (
        <div className="p-4 border-t border-white/10">
          <button
            onClick={onUnlockClick}
            className="w-full py-2 px-4 bg-amber-500/20 hover:bg-amber-500/30 text-amber-400 rounded-lg transition flex items-center justify-center gap-2 text-sm font-medium"
          >
            <LockIcon className="w-4 h-4" />
            Unlock Full Analysis
          </button>
        </div>
      )}
    </div>
  )
}

function LockIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 20 20"
      fill="currentColor"
      className={className}
    >
      <path
        fillRule="evenodd"
        d="M10 1a4.5 4.5 0 00-4.5 4.5V9H5a2 2 0 00-2 2v6a2 2 0 002 2h10a2 2 0 002-2v-6a2 2 0 00-2-2h-.5V5.5A4.5 4.5 0 0010 1zm3 8V5.5a3 3 0 10-6 0V9h6z"
        clipRule="evenodd"
      />
    </svg>
  )
}

export default BlurredLeakCard
