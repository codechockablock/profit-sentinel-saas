'use client'

import { useState, useEffect } from 'react'
import { API_URL } from '@/lib/api-config'
import { useToast } from '@/components/toast'
import { EmployeeStatsCard } from '@/components/employee/stats-card'
import { CorrectionHistory } from '@/components/employee/correction-history'

interface EmployeeStats {
  employee_id: string
  total_assists: number
  total_corrections: number
  corrections_accepted: number
  accuracy_rate: number
  categories_helped: string[]
  last_activity: string | null
}

interface Employee {
  employee_id: string
  store_id: string
  name: string
  email: string | null
  role: string
  created_at: string
  is_active: boolean
}

interface Correction {
  correction_id: string
  problem_id: string
  original_category: string
  corrected_category: string
  correction_notes: string | null
  created_at: string
  was_accepted: boolean
}

export default function EmployeeDashboardPage() {
  const [employee, setEmployee] = useState<Employee | null>(null)
  const [stats, setStats] = useState<EmployeeStats | null>(null)
  const [corrections, setCorrections] = useState<Correction[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const { addToast } = useToast()

  // Demo employee - would come from auth in production
  const employeeId = 'demo-employee-001'
  const storeId = 'demo-store-001'

  useEffect(() => {
    loadEmployeeData()
  }, [])

  const loadEmployeeData = async () => {
    setLoading(true)
    setError(null)

    try {
      // Load employee profile
      const profileRes = await fetch(
        `${API_URL}/employee/${employeeId}?store_id=${storeId}`
      )

      if (profileRes.status === 404) {
        // Register new employee for demo
        const registerRes = await fetch(`${API_URL}/employee/register`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            employee_id: employeeId,
            store_id: storeId,
            name: 'Demo Employee',
            role: 'associate',
          }),
        })

        if (!registerRes.ok) {
          throw new Error('Failed to register employee')
        }

        const newEmployee = await registerRes.json()
        setEmployee(newEmployee)
      } else if (profileRes.ok) {
        setEmployee(await profileRes.json())
      } else {
        throw new Error('Failed to load employee')
      }

      // Load stats
      const statsRes = await fetch(
        `${API_URL}/employee/${employeeId}/stats?store_id=${storeId}`
      )
      if (statsRes.ok) {
        setStats(await statsRes.json())
      }

      // Load correction history
      const correctionsRes = await fetch(
        `${API_URL}/employee/${employeeId}/corrections?limit=10`
      )
      if (correctionsRes.ok) {
        const data = await correctionsRes.json()
        setCorrections(data.corrections)
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load data'
      setError(message)
      addToast({ type: 'error', message })
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="bg-white p-6 rounded-lg shadow-md max-w-md">
          <h2 className="text-xl font-semibold text-red-600 mb-2">Error</h2>
          <p className="text-gray-600">{error}</p>
          <button
            onClick={loadEmployeeData}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Employee Dashboard
              </h1>
              {employee && (
                <p className="text-sm text-gray-500">
                  Welcome, {employee.name} ({employee.role})
                </p>
              )}
            </div>
            <a
              href="/repair"
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
            >
              Start Assisting
            </a>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        {/* Stats Cards */}
        {stats && <EmployeeStatsCard stats={stats} />}

        {/* Correction History */}
        <div className="mt-8">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Recent Corrections
          </h2>
          <CorrectionHistory corrections={corrections} />
        </div>

        {/* Categories Helped */}
        {stats && stats.categories_helped.length > 0 && (
          <div className="mt-8">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">
              Categories Helped
            </h2>
            <div className="flex flex-wrap gap-2">
              {stats.categories_helped.map((category) => (
                <span
                  key={category}
                  className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm"
                >
                  {category.replace(/-/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                </span>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
