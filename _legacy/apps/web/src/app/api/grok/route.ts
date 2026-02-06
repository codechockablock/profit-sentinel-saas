// src/app/api/grok/route.ts
import { streamText } from 'ai'
import { xai } from '@ai-sdk/xai'
import { NextResponse } from 'next/server'

export const maxDuration = 60 // Allows longer responses if needed

/**
 * Grok AI Chat API Route
 *
 * Requires XAI_API_KEY environment variable to be set.
 * Get your API key at https://x.ai/api
 *
 * The @ai-sdk/xai package automatically reads from XAI_API_KEY.
 */
export async function POST(req: Request) {
  // Validate API key is configured (fail-fast)
  if (!process.env.XAI_API_KEY) {
    console.error(
      'XAI_API_KEY environment variable is not configured. ' +
      'Get your API key at https://x.ai/api'
    )
    return NextResponse.json(
      {
        error: 'AI service not configured',
        message: 'The server is missing the required XAI_API_KEY configuration.',
      },
      { status: 503 }
    )
  }

  try {
    const { messages } = await req.json()

    // Stream the response from Grok
    // The xai() function automatically uses process.env.XAI_API_KEY
    const result = await streamText({
      model: xai('grok-beta'),
      messages,
      system: 'You are Grok, a helpful and witty AI assistant built by xAI. Be concise and maximally helpful.',
    })

    return result.toTextStreamResponse()
  } catch (error) {
    console.error('Grok API error:', error)

    // Handle specific error types
    if (error instanceof Error) {
      if (error.message.includes('API key')) {
        return NextResponse.json(
          { error: 'Invalid API key', message: 'Check your XAI_API_KEY configuration.' },
          { status: 401 }
        )
      }
    }

    return NextResponse.json(
      { error: 'AI request failed', message: 'An error occurred while processing your request.' },
      { status: 500 }
    )
  }
}
