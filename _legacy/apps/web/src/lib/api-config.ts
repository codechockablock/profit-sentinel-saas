// src/lib/api-config.ts
/**
 * API configuration for the frontend.
 *
 * Uses NEXT_PUBLIC_API_URL from environment variables.
 * Falls back to production API URL if not set.
 */

const DEFAULT_API_URL = 'https://api.profitsentinel.com'

/**
 * Get the backend API URL.
 * Reads from NEXT_PUBLIC_API_URL environment variable.
 */
export function getApiUrl(): string {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL

  if (!apiUrl) {
    // Only warn in development
    if (process.env.NODE_ENV === 'development') {
      console.warn(
        '[API Config] NEXT_PUBLIC_API_URL not set, using default:',
        DEFAULT_API_URL
      )
    }
    return DEFAULT_API_URL
  }

  return apiUrl
}

/**
 * Pre-computed API URL for use in components.
 */
export const API_URL = getApiUrl()
