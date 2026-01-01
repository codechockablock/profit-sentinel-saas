// src/components/sidebar.tsx
'use client'

import { useState } from 'react'
import GrokChat from './grok-chat'
import { useTheme } from 'next-themes'

export default function Sidebar() {
  const [showGrok, setShowGrok] = useState(true)
  const { theme, setTheme, systemTheme } = useTheme()

  const currentTheme = theme === 'system' ? systemTheme : theme

  return (
    <div className="w-80 border-r border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-950 flex flex-col h-full">
      {/* Header */}
      <div className="p-8 border-b border-gray-200 dark:border-gray-800">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">MySaaS</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">Default Workspace</p>
      </div>

      {/* Channels */}
      <div className="flex-1 p-6 overflow-y-auto">
        <h3 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-4">
          Channels
        </h3>
        <ul className="space-y-2">
          <li className="px-5 py-3 rounded-xl bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 font-medium">
            # general
          </li>
          <li className="px-5 py-3 rounded-xl hover:bg-gray-100 dark:hover:bg-gray-800 cursor-pointer font-medium text-gray-700 dark:text-gray-300">
            # random
          </li>
          <li className="px-5 py-3 rounded-xl hover:bg-gray-100 dark:hover:bg-gray-800 cursor-pointer font-medium text-gray-700 dark:text-gray-300">
            # ideas
          </li>
        </ul>
      </div>

      {/* Bottom Controls */}
      <div className="border-t border-gray-200 dark:border-gray-800 p-4 space-y-2">
        {/* Dark Mode Toggle */}
        <button
          onClick={() => setTheme(currentTheme === 'dark' ? 'light' : 'dark')}
          className="w-full px-5 py-4 rounded-xl bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 flex items-center justify-between font-medium text-gray-700 dark:text-gray-300 transition"
        >
          <span>{currentTheme === 'dark' ? 'Light Mode ☀️' : 'Dark Mode 🌙'}</span>
        </button>

        {/* Grok Assistant Toggle */}
        <button
          onClick={() => setShowGrok(!showGrok)}
          className="w-full px-5 py-4 rounded-xl bg-gradient-to-r from-blue-600 to-purple-600 text-white font-medium flex items-center justify-between hover:opacity-90 transition"
        >
          <span>Grok Assistant</span>
          <span>{showGrok ? '−' : '+'}</span>
        </button>
        {showGrok && <GrokChat />}
      </div>
    </div>
  )
}