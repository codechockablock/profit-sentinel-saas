'use client'

interface Correction {
  correction_id: string
  problem_id: string
  original_category: string
  corrected_category: string
  correction_notes: string | null
  created_at: string
  was_accepted: boolean
}

interface CorrectionHistoryProps {
  corrections: Correction[]
}

export function CorrectionHistory({ corrections }: CorrectionHistoryProps) {
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  const formatCategory = (slug: string) => {
    return slug.replace(/-/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
  }

  if (corrections.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow p-6 text-center">
        <svg
          className="w-12 h-12 mx-auto text-gray-400 mb-4"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
          />
        </svg>
        <p className="text-gray-500">No corrections yet</p>
        <p className="text-sm text-gray-400 mt-1">
          When you correct AI diagnoses, they will appear here
        </p>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      <ul className="divide-y divide-gray-200">
        {corrections.map((correction) => (
          <li key={correction.correction_id} className="p-4 hover:bg-gray-50">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  {/* Status indicator */}
                  <span
                    className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                      correction.was_accepted
                        ? 'bg-green-100 text-green-800'
                        : 'bg-yellow-100 text-yellow-800'
                    }`}
                  >
                    {correction.was_accepted ? 'Accepted' : 'Pending'}
                  </span>

                  {/* Category change */}
                  <span className="text-sm text-gray-600">
                    {formatCategory(correction.original_category)}
                  </span>
                  <svg
                    className="w-4 h-4 text-gray-400"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M14 5l7 7m0 0l-7 7m7-7H3"
                    />
                  </svg>
                  <span className="text-sm font-medium text-gray-900">
                    {formatCategory(correction.corrected_category)}
                  </span>
                </div>

                {/* Notes */}
                {correction.correction_notes && (
                  <p className="mt-1 text-sm text-gray-500">
                    {correction.correction_notes}
                  </p>
                )}

                {/* Problem ID */}
                <p className="mt-1 text-xs text-gray-400">
                  Problem: {correction.problem_id.slice(0, 8)}...
                </p>
              </div>

              {/* Timestamp */}
              <div className="ml-4 text-right">
                <p className="text-sm text-gray-500">
                  {formatDate(correction.created_at)}
                </p>
              </div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}
