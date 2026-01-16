// src/components/channel-messages.tsx
'use client'

import { useEffect, useState, useRef } from 'react'
import type { RealtimePostgresChangesPayload } from '@supabase/supabase-js'
import { getSupabase } from '@/lib/supabase'

interface Message {
  id: string
  channel_id: string
  user_name: string
  content: string
  created_at: string
}

export default function ChannelMessages({ channelId }: { channelId: string }) {
  const [messages, setMessages] = useState<Message[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const supabase = getSupabase()

    // Load initial messages
    const loadMessages = async () => {
      const { data } = await supabase
        .from('messages')
        .select('*')
        .eq('channel_id', channelId)
        .order('created_at', { ascending: true })
      setMessages((data as Message[]) || [])
    }
    loadMessages()

    // Realtime subscription
    const subscription = supabase
      .channel('messages')
      .on(
        'postgres_changes',
        {
          event: 'INSERT',
          schema: 'public',
          table: 'messages',
          filter: `channel_id=eq.${channelId}`,
        },
        (payload: RealtimePostgresChangesPayload<Message>) => {
          if (payload.new && 'id' in payload.new) {
            setMessages((prev) => [...prev, payload.new as Message])
          }
        }
      )
      .subscribe()

    return () => {
      supabase.removeChannel(subscription)
    }
  }, [channelId])

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div className="space-y-4">
      {messages.map((msg) => (
        <div key={msg.id} className="flex items-start gap-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold">
            {msg.user_name?.[0]?.toUpperCase() ?? '?'}
          </div>
          <div>
            <p className="font-medium">{msg.user_name}</p>
            <p className="text-gray-800">{msg.content}</p>
          </div>
        </div>
      ))}
      <div ref={messagesEndRef} />
    </div>
  )
}