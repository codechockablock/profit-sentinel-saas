// src/app/api/grok/route.ts
import { streamText } from 'ai'
import { xai } from '@ai-sdk/xai'
import { NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'

export const maxDuration = 60 // Allows longer responses if needed

// ---------------------------------------------------------------------------
// Per-user rate limiting (in-memory, per-process)
// ---------------------------------------------------------------------------
const RATE_LIMIT_WINDOW_MS = 60_000 // 1 minute
const RATE_LIMIT_MAX = 10 // 10 requests per minute per user

const rateLimitMap = new Map<string, number[]>()

function checkRateLimit(userId: string): boolean {
  const now = Date.now()
  const windowStart = now - RATE_LIMIT_WINDOW_MS
  const timestamps = (rateLimitMap.get(userId) ?? []).filter(t => t > windowStart)
  if (timestamps.length >= RATE_LIMIT_MAX) {
    rateLimitMap.set(userId, timestamps)
    return false
  }
  timestamps.push(now)
  rateLimitMap.set(userId, timestamps)
  return true
}

// Periodic cleanup to prevent memory leaks (every 5 minutes)
setInterval(() => {
  const cutoff = Date.now() - RATE_LIMIT_WINDOW_MS
  for (const [key, timestamps] of rateLimitMap.entries()) {
    const valid = timestamps.filter(t => t > cutoff)
    if (valid.length === 0) {
      rateLimitMap.delete(key)
    } else {
      rateLimitMap.set(key, valid)
    }
  }
}, 5 * 60_000)

// Allowed model IDs (restrict what callers can request)
const ALLOWED_MODELS = new Set(['grok-beta'])

/**
 * Grok AI Chat API Route
 *
 * Requires:
 *   - XAI_API_KEY environment variable
 *   - Valid Supabase auth token (Bearer header)
 *
 * Rate limited to 10 requests/minute per authenticated user.
 */
export async function POST(req: Request) {
  // 1. Validate API key is configured (fail-fast)
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

  // 2. Require Supabase auth token
  const authHeader = req.headers.get('authorization') ?? ''
  if (!authHeader.startsWith('Bearer ')) {
    return NextResponse.json(
      { error: 'Unauthorized', message: 'Authentication required.' },
      { status: 401 }
    )
  }

  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
  const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY
  if (!supabaseUrl || !supabaseServiceKey) {
    return NextResponse.json(
      { error: 'Auth service not configured', message: 'Server auth configuration missing.' },
      { status: 503 }
    )
  }

  const token = authHeader.slice(7)
  let userId: string
  try {
    const supabase = createClient(supabaseUrl, supabaseServiceKey)
    const { data: { user }, error } = await supabase.auth.getUser(token)
    if (error || !user) {
      return NextResponse.json(
        { error: 'Unauthorized', message: 'Invalid or expired token.' },
        { status: 401 }
      )
    }
    userId = user.id
  } catch {
    return NextResponse.json(
      { error: 'Unauthorized', message: 'Token validation failed.' },
      { status: 401 }
    )
  }

  // 3. Per-user rate limiting
  if (!checkRateLimit(userId)) {
    return NextResponse.json(
      { error: 'Rate limited', message: 'Too many requests. Please try again in a minute.' },
      { status: 429 }
    )
  }

  try {
    const { messages, model } = await req.json()

    // 4. Restrict allowed models server-side
    const requestedModel = model ?? 'grok-beta'
    if (!ALLOWED_MODELS.has(requestedModel)) {
      return NextResponse.json(
        { error: 'Invalid model', message: `Model '${requestedModel}' is not allowed.` },
        { status: 400 }
      )
    }

    // Stream the response from Grok
    // The xai() function automatically uses process.env.XAI_API_KEY
    const result = await streamText({
      model: xai(requestedModel),
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
