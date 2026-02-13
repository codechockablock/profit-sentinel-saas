// src/lib/api-config.ts
/**
 * API configuration for the frontend.
 *
 * Uses NEXT_PUBLIC_API_URL from environment variables.
 * Throws if not set to prevent accidental production fallback.
 */

/**
 * Get the backend API URL.
 * Reads from NEXT_PUBLIC_API_URL environment variable.
 */
export function getApiUrl(): string {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL?.trim()

  if (!apiUrl) {
    throw new Error(
      '[API Config] NEXT_PUBLIC_API_URL is required. Set it explicitly for each deployment.'
    )
  }

  return apiUrl
}

/**
 * Pre-computed API URL for use in components.
 */
export const API_URL = getApiUrl()
