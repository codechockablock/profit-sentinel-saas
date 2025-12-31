// src/components/grok-chat.tsx
'use client'

import { useChat } from '@ai-sdk/react'

// Define types once at the top
interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content?: string
}

interface ChatHook {
  messages: ChatMessage[]
  input?: string
  isLoading?: boolean
  handleSubmit?: (e: React.FormEvent<HTMLFormElement>) => void
  handleInputChange?: (e: React.ChangeEvent<HTMLInputElement>) => void
}

export default function GrokChat() {
  const chat = useChat({
    api: '/api/grok',
    onError: (error: any) => {
      console.error('Grok API error:', error)
    },
  } as any) as unknown as ChatHook

  const isLoading = !!chat.isLoading

  const handleFormSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    chat.handleSubmit?.(e)
  }

  return (
    <div className="flex flex-col h-96 bg-gray-50 border-t">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {chat.messages.length === 0 && (
          <p className="text-center text-gray-500 py-8">
            Ask Grok anything — e.g. "create a daily summary automation" or "summarize this channel"
          </p>
        )}

        {chat.messages.map((m) => (
          <div
            key={m.id}
            className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] px-4 py-3 rounded-2xl ${
                m.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white shadow-sm border'
              }`}
            >
              {m.content || ''}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-white shadow-sm px-4 py-3 rounded-2xl">
              <span className="text-gray-500">Grok is thinking...</span>
            </div>
          </div>
        )}
      </div>

      <form onSubmit={handleFormSubmit} className="p-4 border-t bg-white">
        <div className="flex gap-2">
          <input
            value={chat.input || ''}
            onChange={chat.handleInputChange}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                const form = e.currentTarget.form
                if (form) {
                  if ((form as HTMLFormElement).requestSubmit) {
                    (form as HTMLFormElement).requestSubmit()
                  } else {
                    (form as HTMLFormElement).submit()
                  }
                }
              }
            }}
            placeholder="Ask Grok anything..."
            className="flex-1 px-4 py-3 rounded-xl border focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !chat.input?.trim()}
            className="px-6 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-50 transition"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  )
}