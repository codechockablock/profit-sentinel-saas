// src/app/api/grok/route.ts
import { streamText } from 'ai'
import { xai } from '@ai-sdk/xai'

export const maxDuration = 60 // Allows longer responses if needed

export async function POST(req: Request) {
  const { messages } = await req.json()

  // Stream the response from Grok
  const result = await streamText({
    model: xai('grok-beta'),
    messages,
    system: 'You are Grok, a helpful and witty AI assistant built by xAI. Be concise and maximally helpful.',
  })

  return result.toTextStreamResponse()
}