// src/lib/api-config.ts
/**
 * API configuration for the frontend.
 *
 * Uses NEXT_PUBLIC_API_URL from environment variables with
 * production fallback to prevent deployment failures.
 */

/**
 * Get the backend API URL.
 * Reads from NEXT_PUBLIC_API_URL environment variable.
 */
export function getApiUrl(): string {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL?.trim()

  if (apiUrl) {
    return apiUrl
  }

  // Production fallback — keeps deploys from breaking
  if (process.env.NODE_ENV === 'production') {
    return 'https://api.profitsentinel.com'
  }

  return 'http://localhost:8000'
}

/**
 * Pre-computed API URL for use in components.
 */
export const API_URL = getApiUrl()
