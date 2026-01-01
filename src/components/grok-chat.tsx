// src/components/grok-chat.tsx
'use client'

import { useChat } from '@ai-sdk/react'
import { useState } from 'react'

export default function GrokChat() {
  const [localInput, setLocalInput] = useState('')

  const {
    messages,
    isLoading,
    error,
    sendMessage,
    setInput, // Optional: sync with hook if it provides it
  } = useChat({
    api: '/api/grok', // Required for your custom route
  })

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!localInput.trim()) return

    sendMessage(localInput)
    setLocalInput('')
    // Optional sync
    if (setInput) setInput('')
  }

  if (error) {
    return (
      <div className="flex h-full items-center justify-center p-8 text-center">
        <p className="text-red-500">Grok unavailable – check console</p>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-96 bg-transparent">
      <div className="flex-1 overflow-y-auto p-8 space-y-6">
        {messages.length === 0 && (
          <p className="text-center text-gray-500 dark:text-gray-400 text-lg py-12">
            Ask Grok anything...
          </p>
        )}

        {messages.map((m) => (
          <div
            key={m.id}
            className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-lg px-8 py-5 rounded-3xl shadow-xl ${
                m.role === 'user'
                  ? 'bg-gradient-to-br from-blue-600 to-purple-600 text-white'
                  : 'bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800'
              }`}
            >
              <p className="text-lg leading-relaxed">{m.content || ''}</p>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="px-8 py-5 rounded-3xl bg-white dark:bg-gray-900 shadow-xl border border-gray-200 dark:border-gray-800">
              <span className="text-gray-500 animate-pulse">Grok is thinking...</span>
            </div>
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} className="p-6 bg-white/80 dark:bg-black/80 backdrop-blur-xl border-t border-gray-200 dark:border-gray-800">
        <div className="flex gap-4">
          <input
            value={localInput}
            onChange={(e) => {
              setLocalInput(e.target.value)
              if (setInput) setInput(e.target.value)
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                handleSubmit(e as any)
              }
            }}
            placeholder="Message Grok..."
            className="flex-1 px-8 py-5 rounded-full bg-gray-100 dark:bg-gray-900 focus:bg-white dark:focus:bg-gray-800 focus:outline-none focus:ring-4 focus:ring-blue-500/30 text-lg"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !localInput.trim()}
            className="px-10 py-5 bg-gradient-to-br from-blue-600 to-purple-600 text-white font-semibold rounded-full hover:shadow-2xl disabled:opacity-50 transition"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  )
}