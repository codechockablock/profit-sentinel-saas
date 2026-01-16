// src/components/message-input.tsx
'use client'

import { useState } from 'react'
import { getSupabase } from '@/lib/supabase'

export default function MessageInput({ channelId }: { channelId: string }) {
  const [content, setContent] = useState('')

  const sendMessage = async () => {
    if (!content.trim()) return

    const supabase = getSupabase()
    await supabase.from('messages').insert({
      channel_id: channelId,
      user_name: 'You', // We'll make this dynamic later
      content: content.trim(),
    })

    setContent('')
  }

  return (
    <div className="p-6 border-t bg-white">
      <div className="flex gap-3">
        <input
          value={content}
          onChange={(e) => setContent(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), sendMessage())}
          placeholder="Type a message..."
          className="flex-1 px-4 py-3 rounded-lg border focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button
          onClick={sendMessage}
          className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          Send
        </button>
      </div>
    </div>
  )
}