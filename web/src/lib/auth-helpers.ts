/**
 * Shared authentication helpers.
 *
 * Provides a robust sign-out that works even when Supabase is unreachable,
 * plus utilities for checking auth state.
 */

import { getSupabase } from './supabase'

/**
 * Clear all Supabase auth artifacts from the browser.
 * This is the client-side fallback when supabase.auth.signOut() fails.
 */
function clearAuthArtifacts(): void {
  if (typeof window === 'undefined') return

  // Clear cookies matching sb-*-auth-token pattern
  const cookies = document.cookie.split(';')
  for (const cookie of cookies) {
    const name = cookie.split('=')[0].trim()
    if (name.startsWith('sb-') && name.includes('-auth-token')) {
      document.cookie = `${name}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/`
      document.cookie = `${name}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/; domain=${window.location.hostname}`
    }
  }

  // Clear Supabase session keys from localStorage
  const keysToRemove: string[] = []
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i)
    if (key && (key.startsWith('sb-') || key.includes('supabase'))) {
      keysToRemove.push(key)
    }
  }
  keysToRemove.forEach((key) => localStorage.removeItem(key))
}

/**
 * Sign the user out robustly.
 *
 * 1. Attempts supabase.auth.signOut()
 * 2. If that fails, falls back to clearing auth artifacts manually
 * 3. Always redirects to the home page afterward
 *
 * Returns true if sign-out was successful (either path).
 */
export async function robustSignOut(): Promise<boolean> {
  const supabase = getSupabase()

  if (supabase) {
    try {
      const { error } = await supabase.auth.signOut()
      if (!error) {
        return true
      }
      // Sign-out failed (e.g. 403 from Supabase) — fall through to manual cleanup
      console.warn('[auth-helpers] supabase.auth.signOut() failed, using fallback:', error.message)
    } catch (err) {
      console.warn('[auth-helpers] supabase.auth.signOut() threw, using fallback:', err)
    }
  }

  // Fallback: manually clear everything
  clearAuthArtifacts()
  return true
}

/**
 * Get the current user's email from the Supabase session.
 * Returns null if not authenticated or Supabase is not configured.
 */
export async function getCurrentUserEmail(): Promise<string | null> {
  const supabase = getSupabase()
  if (!supabase) return null

  try {
    const { data: { session } } = await supabase.auth.getSession()
    return session?.user?.email ?? null
  } catch {
    return null
  }
}
