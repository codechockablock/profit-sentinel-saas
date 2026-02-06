// src/lib/supabase.ts
import { createBrowserClient } from '@supabase/ssr'

/**
 * Supabase client for the frontend.
 *
 * IMPORTANT: In production (Vercel), ensure these environment variables are set:
 *   - NEXT_PUBLIC_SUPABASE_URL: Your Supabase project URL
 *   - NEXT_PUBLIC_SUPABASE_ANON_KEY: Your Supabase anon/public key
 *
 * These must be added in Vercel Dashboard > Settings > Environment Variables
 * with "Production" scope enabled.
 */

// Read env vars at module load time (works in both SSR and client)
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

// Singleton instance for client-side use
let _supabaseClient: ReturnType<typeof createBrowserClient> | null = null

/**
 * Check if Supabase is properly configured.
 */
export function isSupabaseConfigured(): boolean {
  return Boolean(supabaseUrl && supabaseAnonKey)
}

/**
 * Get or create the Supabase client instance.
 * Safe for SSR - only creates client when env vars are available.
 *
 * Returns null if env vars are missing (production-safe).
 * Only logs warning in development mode.
 */
export function getSupabase() {
  if (!supabaseUrl || !supabaseAnonKey) {
    // Only warn in development, not during production build
    if (process.env.NODE_ENV === 'development') {
      console.warn(
        '[Supabase] Environment variables not configured. ' +
        'Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY.'
      )
    }
    return null
  }

  // Only create client in browser context (client-side)
  if (typeof window !== 'undefined' && !_supabaseClient) {
    _supabaseClient = createBrowserClient(supabaseUrl, supabaseAnonKey)
  }

  return _supabaseClient
}

/**
 * Get auth headers for API requests.
 * Returns empty object if Supabase is not configured or user is not logged in.
 */
export async function getAuthHeaders(): Promise<Record<string, string>> {
  const client = getSupabase()
  if (!client) return {}

  try {
    const { data: { session } } = await client.auth.getSession()
    if (session?.access_token) {
      return { Authorization: `Bearer ${session.access_token}` }
    }
  } catch {
    // Silently fail - auth is optional
  }

  return {}
}

/**
 * Direct export for backward compatibility.
 * Returns null if not configured (safe for SSR/build).
 */
export const supabase = isSupabaseConfigured()
  ? createBrowserClient(supabaseUrl!, supabaseAnonKey!)
  : null
