// src/components/sidebar.tsx
'use client'

import { useState } from 'react'
import GrokChat from './grok-chat'
import AutomationModal from './automation-modal' // We'll add this soon

export default function Sidebar() {
  const [showGrok, setShowGrok] = useState(true)
  const [showAutomationModal, setShowAutomationModal] = useState(false)

  return (
    <div className="w-80 border-r bg-white flex flex-col h-full">
      {/* Header */}
      <div className="p-6 border-b">
        <h1 className="text-2xl font-bold text-blue-600">MySaaS</h1>
        <p className="text-sm text-gray-500 mt-1">Default Workspace</p>
      </div>

      {/* Channels + New Automation Button */}
      <div className="flex-1 p-4 overflow-y-auto">
        <h3 className="font-semibold text-gray-700 mb-3">Channels</h3>
        <ul className="space-y-1 mb-8">
          <li className="px-4 py-2 rounded-lg bg-blue-100 font-medium"># general</li>
          <li className="px-4 py-2 rounded-lg hover:bg-gray-100 cursor-pointer"># random</li>
          <li className="px-4 py-2 rounded-lg hover:bg-gray-100 cursor-pointer"># ideas</li>
        </ul>

        {/* New Automation Button */}
        <button
          onClick={() => setShowAutomationModal(true)}
          className="w-full px-4 py-3 bg-green-600 text-white font-medium rounded-lg hover:bg-green-700 transition shadow-sm"
        >
          + New Automation
        </button>
      </div>

      {/* Grok Assistant Toggle */}
      <div className="border-t">
        <button
          onClick={() => setShowGrok(!showGrok)}
          className="w-full px-4 py-3 text-left font-medium bg-gray-50 hover:bg-gray-100 flex justify-between items-center"
        >
          Grok Assistant {showGrok ? '▼' : '▲'}
        </button>
        {showGrok && <GrokChat />}
      </div>

      {/* Automation Modal (temporary placeholder until we build it) */}
      {showAutomationModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-2xl p-8 w-full max-w-2xl shadow-2xl">
            <h2 className="text-3xl font-bold mb-4">Coming Soon!</h2>
            <p className="text-gray-600 mb-6">
              Grok-powered no-code automation builder is next.
            </p>
            <button
              onClick={() => setShowAutomationModal(false)}
              className="px-8 py-4 bg-gray-200 text-gray-700 font-medium rounded-xl hover:bg-gray-300"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  )
}