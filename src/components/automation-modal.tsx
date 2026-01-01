// src/components/automation-modal.tsx
'use client'

import { useState } from 'react'

export default function AutomationModal({ onClose }: { onClose: () => void }) {
  const [description, setDescription] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!description.trim()) return

    // For now, just show an alert — later we'll send to Grok
    alert(`Automation requested:\n"${description}"\n\nGrok will build this soon!`)
    setDescription('')
    onClose()
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-2xl p-8 w-full max-w-2xl shadow-2xl">
        <h2 className="text-3xl font-bold mb-4">Create New Automation</h2>
        <p className="text-gray-600 mb-6">
          Describe what you want in plain English. Grok will build it for you.
        </p>

        <form onSubmit={handleSubmit} className="space-y-6">
          <textarea
            value={description}
            onChange={(e: { target: { value: any } }) => setDescription(e.target.value)}
            placeholder='e.g. "Every morning at 9AM, send me a summary of new leads from the CRM"'
            className="w-full h-40 px-5 py-4 rounded-xl border-2 border-gray-200 focus:border-green-500 focus:outline-none resize-none text-lg"
            required
          />

          <div className="flex gap-4">
            <button
              type="submit"
              className="flex-1 py-4 bg-green-600 text-white font-semibold rounded-xl hover:bg-green-700 transition shadow-lg"
            >
              Build with Grok
            </button>
            <button
              type="button"
              onClick={onClose}
              className="px-8 py-4 bg-gray-200 text-gray-700 font-medium rounded-xl hover:bg-gray-300 transition"
            >
              Cancel
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}