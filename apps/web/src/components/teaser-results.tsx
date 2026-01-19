'use client'

import { useState, useEffect } from 'react'
import { InfoTooltip } from './ui/tooltip'
import {
  LEAK_METADATA,
  getSeverityBadge,
  scoreToRiskLabel,
  formatDollarImpact,
  estimateItemImpact
} from '@/lib/leak-metadata'

interface LeakData {
  top_items: string[]
  scores: number[]
  count?: number
  severity?: string
  title?: string
}

interface AnalysisResult {
  filename: string
  leaks: Record<string, LeakData>
  summary?: {
    total_rows_analyzed: number
    total_items_flagged: number
    critical_issues: number
    high_issues: number
    estimated_impact?: {
      currency: string
      low_estimate: number
      high_estimate: number
      breakdown: Record<string, number>
    }
  }
}

interface TeaserResultsProps {
  results: AnalysisResult[]
  onUnlockClick: () => void
  isUnlocked: boolean
}

/**
 * Anonymize an item name for teaser view
 */
function anonymizeItem(index: number): string {
  const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
  return `Item ${letters[index % 26]}${index >= 26 ? Math.floor(index / 26) : ''}`
}

/**
 * Teaser Leak Card - Shows anonymized preview with blur effect
 */
function TeaserLeakCard({
  primitive,
  data,
  impactBreakdown,
  isUnlocked,
  onUnlockClick
}: {
  primitive: string
  data: LeakData
  impactBreakdown?: number
  isUnlocked: boolean
  onUnlockClick: () => void
}) {
  const metadata = LEAK_METADATA[primitive]
  const [expanded, setExpanded] = useState(true) // Default expanded for teaser

  if (!metadata) return null

  const severityBadge = getSeverityBadge(metadata.severity)
  const hasItems = data.top_items && data.top_items.length > 0
  const itemCount = data.count || data.top_items?.length || 0
  const previewItems = data.top_items.slice(0, 5) // Show 5 items in teaser

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
                <div className="space-y-3">
                  <p className="font-medium text-white">{metadata.title}</p>
                  <p className="text-slate-300">{metadata.plainEnglish}</p>
                </div>
              }
            />
          </div>
          <span className={`px-2 py-0.5 text-xs font-bold rounded border ${severityBadge.className}`}>
            {severityBadge.label}
          </span>
        </div>

        {/* Quick Stats - Always visible */}
        <div className="mt-3 flex items-center gap-4 text-sm">
          <div className="flex items-center gap-1.5">
            <span className="text-slate-400">Items Found:</span>
            <span className="font-bold text-white">{itemCount}</span>
          </div>
          {impactBreakdown && impactBreakdown > 0 && (
            <div className="flex items-center gap-1.5">
              <span className="text-slate-400">Est. Impact:</span>
              <span className="font-bold text-emerald-400 text-lg">
                {formatDollarImpact(impactBreakdown)}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Teaser Items Preview */}
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
            <div className="px-4 pb-4">
              <div className="space-y-2">
                {previewItems.map((item, i) => {
                  const score = data.scores[i] || 0
                  const risk = scoreToRiskLabel(score)
                  const impact = estimateItemImpact(primitive, score)

                  return (
                    <div
                      key={i}
                      className="relative flex items-center justify-between py-2 px-3 bg-white/5 rounded-lg overflow-hidden"
                    >
                      {/* Anonymized or Real Item - Show first item's real SKU as teaser */}
                      <div className="flex items-center gap-3">
                        {isUnlocked || i === 0 ? (
                          <span className={`font-mono text-sm ${i === 0 && !isUnlocked ? 'text-emerald-400 font-semibold' : 'text-slate-200'}`}>
                            {item || 'Unknown'}
                          </span>
                        ) : (
                          <div className="flex items-center gap-2">
                            <span className="font-mono text-sm text-amber-400">
                              {anonymizeItem(i)}
                            </span>
                            <span className="text-xs text-slate-500 italic">
                              (SKU hidden)
                            </span>
                          </div>
                        )}
                      </div>

                      <div className="flex items-center gap-3">
                        <span className="text-xs text-emerald-400/80 font-semibold">
                          ~${impact.low}-{impact.high}
                        </span>
                        <span className={`px-2 py-0.5 text-xs font-bold rounded border ${risk.className}`}>
                          {risk.percentage}
                        </span>
                      </div>

                      {/* Blur overlay for locked items - skip first item to show real SKU */}
                      {!isUnlocked && i !== 0 && (
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-transparent to-slate-900/80 pointer-events-none" />
                      )}
                    </div>
                  )
                })}

                {/* "More items" teaser */}
                {itemCount > 5 && (
                  <div className="relative">
                    <div className="py-2 px-3 bg-white/5 rounded-lg text-center">
                      <span className="text-sm text-slate-400">
                        +{itemCount - 5} more items with specific SKUs
                      </span>
                    </div>
                    {!isUnlocked && (
                      <div className="absolute inset-0 bg-slate-900/60 backdrop-blur-[2px] rounded-lg flex items-center justify-center">
                        <button
                          onClick={onUnlockClick}
                          className="flex items-center gap-1.5 text-amber-400 hover:text-amber-300 transition text-sm font-medium"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
                            <path fillRule="evenodd" d="M10 1a4.5 4.5 0 00-4.5 4.5V9H5a2 2 0 00-2 2v6a2 2 0 002 2h10a2 2 0 002-2v-6a2 2 0 00-2-2h-.5V5.5A4.5 4.5 0 0010 1zm3 8V5.5a3 3 0 10-6 0V9h6z" clipRule="evenodd" />
                          </svg>
                          Unlock Full Report
                        </button>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Blurred recommendations teaser */}
              {!isUnlocked && (
                <div className="mt-4 pt-4 border-t border-white/10 relative">
                  <div className="blur-sm select-none pointer-events-none">
                    <p className="text-xs font-semibold text-slate-400 mb-2">Expert Recommendations:</p>
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
                </div>
              )}

              {/* Show recommendations if unlocked */}
              {isUnlocked && (
                <div className="mt-4 pt-4 border-t border-white/10">
                  <p className="text-xs font-semibold text-slate-400 mb-2">Recommended Actions:</p>
                  <ul className="space-y-1">
                    {metadata.recommendations.slice(0, 3).map((rec, i) => (
                      <li key={i} className="flex items-start gap-2 text-xs text-slate-300">
                        <span className="text-emerald-500 mt-0.5">-</span>
                        {rec}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/**
 * Animated Impact Counter
 */
function AnimatedImpact({ value, duration = 2000 }: { value: number; duration?: number }) {
  const [displayValue, setDisplayValue] = useState(0)

  useEffect(() => {
    let startTime: number
    let animationFrame: number

    const animate = (timestamp: number) => {
      if (!startTime) startTime = timestamp
      const progress = Math.min((timestamp - startTime) / duration, 1)
      setDisplayValue(Math.floor(progress * value))

      if (progress < 1) {
        animationFrame = requestAnimationFrame(animate)
      }
    }

    animationFrame = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(animationFrame)
  }, [value, duration])

  return <>{formatDollarImpact(displayValue)}</>
}

/**
 * Summary Card with Urgency
 */
function TeaserSummaryCard({
  summary,
  onUnlockClick,
  isUnlocked
}: {
  summary: AnalysisResult['summary']
  onUnlockClick: () => void
  isUnlocked: boolean
}) {
  if (!summary) return null

  const impact = summary.estimated_impact
  const avgImpact = impact ? (impact.low_estimate + impact.high_estimate) / 2 : 0

  return (
    <div className="bg-gradient-to-br from-red-500/10 via-orange-500/10 to-amber-500/10 rounded-2xl border border-orange-500/30 p-6 mb-8 relative overflow-hidden">
      {/* Animated background pulse */}
      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-orange-500/5 to-transparent animate-pulse" />

      <div className="relative">
        {/* Urgency Header */}
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold text-orange-400 flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-6 h-6 animate-pulse">
              <path fillRule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
            </svg>
            Profit Leaks Detected!
          </h3>
          <span className="px-3 py-1 bg-red-500/20 text-red-400 text-xs font-bold rounded-full animate-pulse">
            {summary.critical_issues} CRITICAL
          </span>
        </div>

        {/* Big Impact Number */}
        {impact && (
          <div className="text-center py-6">
            <p className="text-slate-400 text-sm mb-2">You're losing approximately</p>
            <p className="text-5xl md:text-6xl font-black text-transparent bg-clip-text bg-gradient-to-r from-red-400 via-orange-400 to-amber-400">
              <AnimatedImpact value={avgImpact} />
              <span className="text-2xl font-normal text-slate-400">/year</span>
            </p>
            <p className="text-slate-500 text-sm mt-2">
              Range: {formatDollarImpact(impact.low_estimate)} - {formatDollarImpact(impact.high_estimate)}
            </p>
          </div>
        )}

        {/* Stats Grid */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="bg-white/5 rounded-xl p-3 text-center">
            <p className="text-2xl font-bold text-white">{summary.total_rows_analyzed.toLocaleString()}</p>
            <p className="text-xs text-slate-400">Items Scanned</p>
          </div>
          <div className="bg-white/5 rounded-xl p-3 text-center">
            <p className="text-2xl font-bold text-orange-400">{summary.total_items_flagged}</p>
            <p className="text-xs text-slate-400">Problems Found</p>
          </div>
          <div className="bg-white/5 rounded-xl p-3 text-center">
            <p className="text-2xl font-bold text-red-400">{summary.critical_issues + summary.high_issues}</p>
            <p className="text-xs text-slate-400">Urgent Issues</p>
          </div>
        </div>

        {/* CTA Banner */}
        {!isUnlocked && (
          <div className="bg-gradient-to-r from-amber-500/20 to-orange-500/20 rounded-xl p-4 border border-amber-500/30">
            <div className="flex flex-col md:flex-row items-center justify-between gap-4">
              <div className="text-center md:text-left">
                <p className="text-amber-400 font-bold">
                  Want the exact SKUs causing these leaks?
                </p>
                <p className="text-sm text-slate-400">
                  Get specific item numbers, fix instructions & priority list
                </p>
              </div>
              <button
                onClick={onUnlockClick}
                className="whitespace-nowrap bg-gradient-to-r from-amber-500 to-orange-500 text-white font-bold px-6 py-3 rounded-xl hover:from-amber-600 hover:to-orange-600 transition transform hover:scale-105 shadow-lg shadow-amber-500/25 flex items-center gap-2"
              >
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
                  <path d="M3 4a2 2 0 00-2 2v1.161l8.441 4.221a1.25 1.25 0 001.118 0L19 7.162V6a2 2 0 00-2-2H3z" />
                  <path d="M19 8.839l-7.77 3.885a2.75 2.75 0 01-2.46 0L1 8.839V14a2 2 0 002 2h14a2 2 0 002-2V8.839z" />
                </svg>
                Get Full Report Free
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

/**
 * Main Teaser Results Component
 */
export function TeaserResults({ results, onUnlockClick, isUnlocked }: TeaserResultsProps) {
  const priorityOrder = [
    'high_margin_leak',
    'negative_inventory',
    'low_stock',
    'shrinkage_pattern',
    'margin_erosion',
    'dead_item',
    'overstock',
    'price_discrepancy'
  ]

  return (
    <div className="space-y-8">
      {results.map((result, resultIdx) => {
        const impactBreakdown = result.summary?.estimated_impact?.breakdown || {}
        const sortedLeaks = Object.entries(result.leaks).sort((a, b) => {
          const aIdx = priorityOrder.indexOf(a[0])
          const bIdx = priorityOrder.indexOf(b[0])
          return (aIdx === -1 ? 99 : aIdx) - (bIdx === -1 ? 99 : bIdx)
        })

        return (
          <div key={resultIdx}>
            {/* Summary with Impact */}
            {result.summary && (
              <TeaserSummaryCard
                summary={result.summary}
                onUnlockClick={onUnlockClick}
                isUnlocked={isUnlocked}
              />
            )}

            {/* Leak Categories */}
            <div className="grid gap-4 md:grid-cols-2">
              {sortedLeaks.map(([primitive, data]) => (
                <TeaserLeakCard
                  key={primitive}
                  primitive={primitive}
                  data={data}
                  impactBreakdown={impactBreakdown[primitive]}
                  isUnlocked={isUnlocked}
                  onUnlockClick={onUnlockClick}
                />
              ))}
            </div>

            {/* Empty State */}
            {sortedLeaks.length === 0 && (
              <div className="text-center py-12 bg-white/5 rounded-2xl">
                <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-emerald-500/20 flex items-center justify-center">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-8 h-8 text-emerald-400">
                    <path fillRule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clipRule="evenodd" />
                  </svg>
                </div>
                <h4 className="text-xl font-bold text-emerald-400">Looking Good!</h4>
                <p className="text-slate-400 mt-2">No significant profit leaks detected</p>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

export default TeaserResults
