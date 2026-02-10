'use client'

import { useState, useEffect } from 'react'

interface SolutionStep {
  order: number
  instruction: string
  tip?: string
  caution?: string
}

interface SolutionPart {
  part_name: string
  quantity: number
  is_required: boolean
  in_stock?: boolean
  unit_price?: number
}

interface SolutionViewProps {
  problemId: string
  categorySlug: string
  categoryName: string
}

// Mock solution data for demo (would come from API in production)
const MOCK_SOLUTIONS: Record<string, {
  title: string
  summary: string
  steps: SolutionStep[]
  parts: SolutionPart[]
  tools: string[]
  time_minutes: number
  difficulty: number
  video_url?: string
}> = {
  'plumbing-faucet': {
    title: 'Fix a Leaky Faucet',
    summary: 'Most leaky faucets can be fixed by replacing the washer, O-ring, or cartridge. This is a straightforward DIY repair that takes about 30 minutes.',
    steps: [
      {
        order: 1,
        instruction: 'Turn off the water supply valves under the sink. Turn them clockwise until tight.',
        tip: 'Open the faucet to release any remaining pressure.',
      },
      {
        order: 2,
        instruction: 'Remove the faucet handle. Look for a screw under a decorative cap or on the side.',
        tip: 'Use a flathead screwdriver to pry off decorative caps.',
      },
      {
        order: 3,
        instruction: 'Remove the packing nut using an adjustable wrench. Turn counterclockwise.',
        caution: 'Wrap the wrench jaws with tape to prevent scratching the finish.',
      },
      {
        order: 4,
        instruction: 'Pull out the stem/cartridge and inspect the washer and O-rings for damage.',
        tip: 'Take the old parts to the hardware store to find exact replacements.',
      },
      {
        order: 5,
        instruction: 'Replace worn parts. Apply plumber\'s grease to O-rings before installation.',
      },
      {
        order: 6,
        instruction: 'Reassemble in reverse order. Tighten snugly but don\'t over-tighten.',
      },
      {
        order: 7,
        instruction: 'Turn water back on slowly and check for leaks.',
        tip: 'Let water run for a minute to flush any debris.',
      },
    ],
    parts: [
      { part_name: 'Faucet washer kit', quantity: 1, is_required: true, in_stock: true, unit_price: 4.99 },
      { part_name: 'O-ring assortment', quantity: 1, is_required: true, in_stock: true, unit_price: 3.49 },
      { part_name: 'Plumber\'s grease', quantity: 1, is_required: false, in_stock: true, unit_price: 5.99 },
    ],
    tools: ['Adjustable wrench', 'Flathead screwdriver', 'Phillips screwdriver'],
    time_minutes: 30,
    difficulty: 2,
    video_url: 'https://www.youtube.com/watch?v=example',
  },
  'plumbing': {
    title: 'General Plumbing Repair',
    summary: 'Plumbing repairs vary based on the specific issue. This guide covers common troubleshooting steps.',
    steps: [
      {
        order: 1,
        instruction: 'Identify the source of the problem - is it a leak, clog, or pressure issue?',
      },
      {
        order: 2,
        instruction: 'Turn off the water supply if there\'s an active leak.',
        caution: 'Know where your main shutoff valve is located.',
      },
      {
        order: 3,
        instruction: 'Inspect visible pipes and connections for damage or corrosion.',
      },
      {
        order: 4,
        instruction: 'For clogs, try a plunger before using chemical drain cleaners.',
        tip: 'A drain snake is more effective than chemicals for tough clogs.',
      },
      {
        order: 5,
        instruction: 'Check for simple fixes like tightening connections or replacing washers.',
      },
    ],
    parts: [
      { part_name: 'Pipe tape (Teflon)', quantity: 1, is_required: true, in_stock: true, unit_price: 2.99 },
      { part_name: 'Basic washer kit', quantity: 1, is_required: false, in_stock: true, unit_price: 4.99 },
    ],
    tools: ['Adjustable wrench', 'Plunger', 'Bucket', 'Flashlight'],
    time_minutes: 45,
    difficulty: 2,
  },
  'electrical': {
    title: 'Electrical Troubleshooting',
    summary: 'Many electrical issues are simple fixes, but safety is paramount. Always turn off power before working.',
    steps: [
      {
        order: 1,
        instruction: 'Turn off power at the breaker box before any work.',
        caution: 'Use a voltage tester to confirm power is off.',
      },
      {
        order: 2,
        instruction: 'Check if the issue is isolated to one outlet/switch or affects multiple areas.',
      },
      {
        order: 3,
        instruction: 'For dead outlets, check if GFCI outlets need to be reset.',
        tip: 'GFCI outlets have TEST and RESET buttons.',
      },
      {
        order: 4,
        instruction: 'Inspect outlet/switch for visible damage, burn marks, or loose wires.',
        caution: 'If you see burn marks, call a licensed electrician.',
      },
      {
        order: 5,
        instruction: 'Replace damaged outlets or switches with matching replacements.',
      },
    ],
    parts: [
      { part_name: '15A outlet (if replacing)', quantity: 1, is_required: false, in_stock: true, unit_price: 3.99 },
      { part_name: 'Wall plate', quantity: 1, is_required: false, in_stock: true, unit_price: 1.49 },
    ],
    tools: ['Voltage tester', 'Screwdriver set', 'Wire strippers'],
    time_minutes: 30,
    difficulty: 3,
  },
}

function getDifficultyLabel(level: number): string {
  if (level <= 2) return 'Easy'
  if (level <= 3) return 'Moderate'
  return 'Difficult'
}

function getDifficultyColor(level: number): string {
  if (level <= 2) return 'text-emerald-400 bg-emerald-500/10'
  if (level <= 3) return 'text-amber-400 bg-amber-500/10'
  return 'text-red-400 bg-red-500/10'
}

export function SolutionView({ problemId, categorySlug, categoryName }: SolutionViewProps) {
  const [expandedStep, setExpandedStep] = useState<number | null>(null)
  const [completedSteps, setCompletedSteps] = useState<Set<number>>(new Set())

  // Get solution (mock for demo, would be API call)
  const solution = MOCK_SOLUTIONS[categorySlug] || MOCK_SOLUTIONS['plumbing']

  const toggleStep = (order: number) => {
    setExpandedStep(expandedStep === order ? null : order)
  }

  const toggleComplete = (order: number) => {
    const newCompleted = new Set(completedSteps)
    if (newCompleted.has(order)) {
      newCompleted.delete(order)
    } else {
      newCompleted.add(order)
    }
    setCompletedSteps(newCompleted)
  }

  const partsTotal = solution.parts
    .filter(p => p.is_required && p.unit_price)
    .reduce((sum, p) => sum + (p.unit_price || 0) * p.quantity, 0)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-emerald-500/10 to-blue-500/10 rounded-2xl p-6 border border-slate-700/50">
        <h1 className="text-2xl font-bold text-white mb-2">{solution.title}</h1>
        <p className="text-slate-400">{solution.summary}</p>

        {/* Quick stats */}
        <div className="flex flex-wrap gap-3 mt-4">
          <span className={`px-3 py-1.5 rounded-lg text-sm font-medium ${getDifficultyColor(solution.difficulty)}`}>
            {getDifficultyLabel(solution.difficulty)}
          </span>
          <span className="px-3 py-1.5 bg-slate-700 text-slate-300 rounded-lg text-sm">
            ~{solution.time_minutes} min
          </span>
          <span className="px-3 py-1.5 bg-slate-700 text-slate-300 rounded-lg text-sm">
            {solution.steps.length} steps
          </span>
        </div>
      </div>

      {/* Tools Needed */}
      <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700/50">
        <h2 className="text-lg font-semibold mb-3 flex items-center gap-2">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5 text-blue-400">
            <path fillRule="evenodd" d="M14.5 10a4.5 4.5 0 004.284-5.882c-.105-.324-.51-.391-.752-.15L15.34 6.66a.454.454 0 01-.493.11 3.01 3.01 0 01-1.618-1.616.455.455 0 01.11-.494l2.694-2.692c.24-.241.174-.647-.15-.752a4.5 4.5 0 00-5.873 4.575c.055.873-.128 1.808-.8 2.368l-7.23 6.024a2.724 2.724 0 103.837 3.837l6.024-7.23c.56-.672 1.495-.855 2.368-.8.096.007.193.01.291.01zM5 16a1 1 0 11-2 0 1 1 0 012 0z" clipRule="evenodd" />
          </svg>
          Tools Needed
        </h2>
        <div className="flex flex-wrap gap-2">
          {solution.tools.map((tool, idx) => (
            <span key={idx} className="px-3 py-1.5 bg-blue-500/10 text-blue-300 rounded-lg text-sm">
              {tool}
            </span>
          ))}
        </div>
      </div>

      {/* Parts Needed */}
      <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700/50">
        <h2 className="text-lg font-semibold mb-3 flex items-center gap-2">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5 text-amber-400">
            <path d="M2.5 3A1.5 1.5 0 001 4.5v4A1.5 1.5 0 002.5 10h6A1.5 1.5 0 0010 8.5v-4A1.5 1.5 0 008.5 3h-6zm11 2A1.5 1.5 0 0012 6.5v7a1.5 1.5 0 001.5 1.5h4a1.5 1.5 0 001.5-1.5v-7A1.5 1.5 0 0017.5 5h-4zm-10 7A1.5 1.5 0 002 13.5v2A1.5 1.5 0 003.5 17h6a1.5 1.5 0 001.5-1.5v-2A1.5 1.5 0 009.5 12h-6z" />
          </svg>
          Parts Needed
        </h2>
        <div className="space-y-2">
          {solution.parts.map((part, idx) => (
            <div key={idx} className="flex items-center justify-between p-3 bg-slate-900/50 rounded-lg">
              <div className="flex items-center gap-3">
                <span className={`px-2 py-0.5 rounded text-xs font-medium ${part.is_required ? 'bg-red-500/20 text-red-300' : 'bg-slate-600 text-slate-400'}`}>
                  {part.is_required ? 'Required' : 'Optional'}
                </span>
                <span className="text-slate-200">
                  {part.part_name}
                  {part.quantity > 1 && <span className="text-slate-400 ml-1">(x{part.quantity})</span>}
                </span>
              </div>
              <div className="flex items-center gap-3">
                {part.in_stock !== undefined && (
                  <span className={`text-xs ${part.in_stock ? 'text-emerald-400' : 'text-red-400'}`}>
                    {part.in_stock ? 'In Stock' : 'Out of Stock'}
                  </span>
                )}
                {part.unit_price && (
                  <span className="text-slate-400 text-sm">
                    ${(part.unit_price * part.quantity).toFixed(2)}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
        {partsTotal > 0 && (
          <div className="mt-3 pt-3 border-t border-slate-700 flex justify-between">
            <span className="text-slate-400">Estimated Parts Cost</span>
            <span className="text-emerald-400 font-bold">${partsTotal.toFixed(2)}</span>
          </div>
        )}
      </div>

      {/* Steps */}
      <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700/50">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5 text-emerald-400">
              <path fillRule="evenodd" d="M10 2a.75.75 0 01.75.75v.258a33.186 33.186 0 016.668.83.75.75 0 01-.336 1.461 31.28 31.28 0 00-1.103-.232l1.702 7.545a.75.75 0 01-.387.832A4.981 4.981 0 0115 14c-.825 0-1.606-.2-2.294-.556a.75.75 0 01-.387-.832l1.77-7.849a31.743 31.743 0 00-3.339-.254v11.505a20.01 20.01 0 013.78.501.75.75 0 11-.339 1.462A18.558 18.558 0 0010 17.5c-1.442 0-2.845.165-4.191.477a.75.75 0 01-.338-1.462 20.01 20.01 0 013.779-.501V4.509c-1.129.026-2.243.112-3.34.254l1.771 7.85a.75.75 0 01-.387.831A4.981 4.981 0 015 14a4.98 4.98 0 01-2.294-.556.75.75 0 01-.387-.832L4.02 5.067c-.37.07-.738.148-1.103.232a.75.75 0 01-.336-1.462 33.053 33.053 0 016.668-.829V2.75A.75.75 0 0110 2zM5 12.154l-1.182-5.24a32.38 32.38 0 012.364 0L5 12.154zm10 0l-1.182-5.24a32.38 32.38 0 012.364 0L15 12.154z" clipRule="evenodd" />
            </svg>
            Step-by-Step Instructions
          </h2>
          <span className="text-slate-500 text-sm">
            {completedSteps.size}/{solution.steps.length} done
          </span>
        </div>

        <div className="space-y-3">
          {solution.steps.map((step) => (
            <div
              key={step.order}
              className={`rounded-xl border transition ${completedSteps.has(step.order) ? 'bg-emerald-500/10 border-emerald-500/30' : 'bg-slate-900/50 border-slate-700'}`}
            >
              <button
                onClick={() => toggleStep(step.order)}
                className="w-full flex items-start gap-4 p-4 text-left"
              >
                {/* Step number / check */}
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    toggleComplete(step.order)
                  }}
                  className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center font-bold transition ${
                    completedSteps.has(step.order)
                      ? 'bg-emerald-500 text-white'
                      : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                  }`}
                >
                  {completedSteps.has(step.order) ? (
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
                      <path fillRule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clipRule="evenodd" />
                    </svg>
                  ) : (
                    step.order
                  )}
                </button>

                {/* Instruction */}
                <div className="flex-1">
                  <p className={`${completedSteps.has(step.order) ? 'text-slate-400 line-through' : 'text-white'}`}>
                    {step.instruction}
                  </p>
                </div>

                {/* Expand indicator */}
                {(step.tip || step.caution) && (
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                    className={`w-5 h-5 text-slate-500 transition ${expandedStep === step.order ? 'rotate-180' : ''}`}
                  >
                    <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" clipRule="evenodd" />
                  </svg>
                )}
              </button>

              {/* Expanded content */}
              {expandedStep === step.order && (step.tip || step.caution) && (
                <div className="px-4 pb-4 pl-16 space-y-2">
                  {step.tip && (
                    <div className="flex items-start gap-2 p-2 bg-blue-500/10 rounded-lg text-sm">
                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 text-blue-400 mt-0.5 flex-shrink-0">
                        <path d="M10 1a6 6 0 00-3.815 10.631C7.237 12.5 8 13.443 8 14.456v.644a.75.75 0 00.572.729 6.016 6.016 0 002.856 0A.75.75 0 0012 15.1v-.644c0-1.013.762-1.957 1.815-2.825A6 6 0 0010 1zM8.863 17.414a.75.75 0 00-.226 1.483 9.066 9.066 0 002.726 0 .75.75 0 00-.226-1.483 7.553 7.553 0 01-2.274 0z" />
                      </svg>
                      <span className="text-blue-300">{step.tip}</span>
                    </div>
                  )}
                  {step.caution && (
                    <div className="flex items-start gap-2 p-2 bg-amber-500/10 rounded-lg text-sm">
                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 text-amber-400 mt-0.5 flex-shrink-0">
                        <path fillRule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
                      </svg>
                      <span className="text-amber-300">{step.caution}</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Progress indicator */}
        <div className="mt-4 pt-4 border-t border-slate-700">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-slate-400 text-sm">Progress</span>
            <span className="text-emerald-400 text-sm font-medium">
              {Math.round((completedSteps.size / solution.steps.length) * 100)}%
            </span>
          </div>
          <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-emerald-500 rounded-full transition-all duration-300"
              style={{ width: `${(completedSteps.size / solution.steps.length) * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Video Link */}
      {solution.video_url && (
        <a
          href={solution.video_url}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center justify-center gap-3 p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-300 hover:bg-red-500/20 transition"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6">
            <path fillRule="evenodd" d="M4.5 5.653c0-1.426 1.529-2.33 2.779-1.643l11.54 6.348c1.295.712 1.295 2.573 0 3.285L7.28 19.991c-1.25.687-2.779-.217-2.779-1.643V5.653z" clipRule="evenodd" />
          </svg>
          Watch Video Tutorial
        </a>
      )}
    </div>
  )
}
