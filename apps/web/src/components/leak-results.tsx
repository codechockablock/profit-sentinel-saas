'use client'

import { useState } from 'react'
import { Tooltip, InfoTooltip } from './ui/tooltip'
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

interface LeakResultsProps {
  results: AnalysisResult[]
  onRequestReport?: (email: string) => void
}

/**
 * Leak Category Card - Shows leak type with tooltip, items, and impact
 */
function LeakCategoryCard({
  primitive,
  data,
  impactBreakdown
}: {
  primitive: string
  data: LeakData
  impactBreakdown?: number
}) {
  const metadata = LEAK_METADATA[primitive]
  const [expanded, setExpanded] = useState(false)

  if (!metadata) {
    return null
  }

  const severityBadge = getSeverityBadge(metadata.severity)
  const hasItems = data.top_items && data.top_items.length > 0
  const itemCount = data.count || data.top_items?.length || 0

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
                  <div className="pt-2 border-t border-slate-600">
                    <p className="text-xs text-slate-400 mb-1">Examples:</p>
                    <ul className="text-xs text-slate-300 space-y-1">
                      {metadata.tooltipExamples.map((ex, i) => (
                        <li key={i} className="flex items-start gap-1">
                          <span className="text-slate-500">-</span> {ex}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              }
            />
          </div>

          <span
            className={`px-2 py-0.5 text-xs font-bold rounded border ${severityBadge.className}`}
          >
            {severityBadge.label}
          </span>
        </div>

        {/* Quick Stats */}
        <div className="mt-3 flex items-center gap-4 text-sm">
          <div className="flex items-center gap-1.5">
            <span className="text-slate-400">Items:</span>
            <span className="font-bold text-white">{itemCount}</span>
          </div>
          {impactBreakdown && impactBreakdown > 0 && (
            <div className="flex items-center gap-1.5">
              <span className="text-slate-400">Est. Impact:</span>
              <span className="font-bold text-emerald-400">
                {formatDollarImpact(impactBreakdown)}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Items List */}
      {hasItems && (
        <div className="border-t border-white/10">
          <button
            onClick={() => setExpanded(!expanded)}
            className="w-full px-4 py-2 flex items-center justify-between text-sm text-slate-400 hover:bg-white/5 transition-colors"
          >
            <span>{expanded ? 'Hide items' : 'Show flagged items'}</span>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 20 20"
              fill="currentColor"
              className={`w-4 h-4 transition-transform ${expanded ? 'rotate-180' : ''}`}
            >
              <path
                fillRule="evenodd"
                d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z"
                clipRule="evenodd"
              />
            </svg>
          </button>

          {expanded && (
            <div className="px-4 pb-4">
              <div className="space-y-2">
                {data.top_items.slice(0, 10).map((item, i) => {
                  const score = data.scores[i] || 0
                  const risk = scoreToRiskLabel(score)
                  const impact = estimateItemImpact(primitive, score)

                  return (
                    <div
                      key={i}
                      className="flex items-center justify-between py-2 px-3 bg-white/5 rounded-lg"
                    >
                      <div className="flex items-center gap-3">
                        <span className="font-mono text-sm text-slate-200">
                          {item || 'Unknown'}
                        </span>
                      </div>

                      <div className="flex items-center gap-3">
                        {/* Dollar Impact Estimate */}
                        <Tooltip
                          content={
                            <div className="text-center">
                              <p className="text-xs text-slate-400">Estimated Impact</p>
                              <p className="font-bold text-emerald-400">
                                ${impact.low} - ${impact.high}
                              </p>
                            </div>
                          }
                          position="top"
                        >
                          <span className="text-xs text-emerald-400/80">
                            ~${impact.low}-{impact.high}
                          </span>
                        </Tooltip>

                        {/* Risk Badge */}
                        <Tooltip
                          content={
                            <div className="text-center">
                              <p className="text-xs text-slate-400">Risk Score</p>
                              <p className="font-bold">{risk.percentage}</p>
                              <p className="text-xs text-slate-400 mt-1">
                                Raw score: {score.toFixed(3)}
                              </p>
                            </div>
                          }
                          position="top"
                        >
                          <span
                            className={`px-2 py-0.5 text-xs font-bold rounded border ${risk.className}`}
                          >
                            {risk.label} ({risk.percentage})
                          </span>
                        </Tooltip>
                      </div>
                    </div>
                  )
                })}

                {data.top_items.length > 10 && (
                  <p className="text-xs text-slate-500 text-center pt-2">
                    +{data.top_items.length - 10} more items in full report
                  </p>
                )}
              </div>

              {/* Recommendations */}
              <div className="mt-4 pt-4 border-t border-white/10">
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
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/**
 * Summary Card - Overall analysis stats
 */
function SummaryCard({ summary }: { summary: AnalysisResult['summary'] }) {
  if (!summary) return null

  const impact = summary.estimated_impact

  return (
    <div className="bg-gradient-to-br from-emerald-500/10 to-emerald-600/5 rounded-2xl border border-emerald-500/30 p-6 mb-8">
      <h3 className="text-xl font-bold text-emerald-400 mb-4 flex items-center gap-2">
        Analysis Summary
        <InfoTooltip
          content={
            <div>
              <p className="font-medium text-white mb-2">How We Calculate Impact</p>
              <p className="text-sm text-slate-300">
                Dollar estimates are based on item costs, margins, and industry
                benchmarks. Actual impact may vary - use these as a starting point
                for investigation.
              </p>
            </div>
          }
        />
      </h3>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white/5 rounded-xl p-4 text-center">
          <p className="text-2xl font-bold text-white">
            {summary.total_rows_analyzed.toLocaleString()}
          </p>
          <p className="text-xs text-slate-400 mt-1">Items Analyzed</p>
        </div>

        <div className="bg-white/5 rounded-xl p-4 text-center">
          <p className="text-2xl font-bold text-orange-400">
            {summary.total_items_flagged}
          </p>
          <p className="text-xs text-slate-400 mt-1">Items Flagged</p>
        </div>

        <div className="bg-white/5 rounded-xl p-4 text-center">
          <p className="text-2xl font-bold text-red-400">{summary.critical_issues}</p>
          <p className="text-xs text-slate-400 mt-1">Critical Issues</p>
        </div>

        {impact && (
          <div className="bg-white/5 rounded-xl p-4 text-center">
            <p className="text-2xl font-bold text-emerald-400">
              {formatDollarImpact(impact.low_estimate)} - {formatDollarImpact(impact.high_estimate)}
            </p>
            <p className="text-xs text-slate-400 mt-1">Est. Total Impact</p>
          </div>
        )}
      </div>

      {impact && impact.low_estimate > 0 && (
        <div className="mt-4 p-3 bg-emerald-500/10 rounded-lg border border-emerald-500/20">
          <p className="text-sm text-emerald-300 flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a.75.75 0 000 1.5h.253a.25.25 0 01.244.304l-.459 2.066A1.75 1.75 0 0010.747 15H11a.75.75 0 000-1.5h-.253a.25.25 0 01-.244-.304l.459-2.066A1.75 1.75 0 009.253 9H9z" clipRule="evenodd" />
            </svg>
            <span>
              <strong>Potential savings:</strong> Addressing these leaks could recover{' '}
              <strong>{formatDollarImpact(impact.low_estimate)}</strong> to{' '}
              <strong>{formatDollarImpact(impact.high_estimate)}</strong> annually
            </span>
          </p>
        </div>
      )}
    </div>
  )
}

/**
 * Main Leak Results Component
 */
export function LeakResults({ results, onRequestReport }: LeakResultsProps) {
  // Sort leaks by severity
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
    <div className="space-y-10">
      {results.map((result, resultIdx) => {
        // Get impact breakdown from summary if available
        const impactBreakdown = result.summary?.estimated_impact?.breakdown || {}

        // Sort leaks by priority
        const sortedLeaks = Object.entries(result.leaks).sort((a, b) => {
          const aIdx = priorityOrder.indexOf(a[0])
          const bIdx = priorityOrder.indexOf(b[0])
          return (aIdx === -1 ? 99 : aIdx) - (bIdx === -1 ? 99 : bIdx)
        })

        return (
          <div key={resultIdx} className="bg-white/5 rounded-2xl p-6 md:p-8">
            {/* File Header */}
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-2xl font-bold text-white flex items-center gap-3">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-6 h-6 text-emerald-400">
                  <path d="M3 3.5A1.5 1.5 0 014.5 2h6.879a1.5 1.5 0 011.06.44l4.122 4.12A1.5 1.5 0 0117 7.622V16.5a1.5 1.5 0 01-1.5 1.5h-11A1.5 1.5 0 013 16.5v-13z" />
                </svg>
                {result.filename}
              </h3>
            </div>

            {/* Summary */}
            {result.summary && <SummaryCard summary={result.summary} />}

            {/* Leak Categories Grid */}
            <div className="grid gap-4 md:grid-cols-2">
              {sortedLeaks.map(([primitive, data]) => (
                <LeakCategoryCard
                  key={primitive}
                  primitive={primitive}
                  data={data}
                  impactBreakdown={impactBreakdown[primitive]}
                />
              ))}
            </div>

            {/* Empty State */}
            {sortedLeaks.length === 0 && (
              <div className="text-center py-12">
                <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-emerald-500/20 flex items-center justify-center">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-8 h-8 text-emerald-400">
                    <path fillRule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clipRule="evenodd" />
                  </svg>
                </div>
                <h4 className="text-xl font-bold text-emerald-400">All Clear!</h4>
                <p className="text-slate-400 mt-2">No significant profit leaks detected</p>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

export default LeakResults
