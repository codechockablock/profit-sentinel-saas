// src/lib/api.ts
// API helpers for Supabase operations

import { getSupabase, isSupabaseConfigured } from './supabase'

/**
 * Email Signup - captures email during unlock flow
 */
export interface EmailSignupData {
  email: string
  source?: 'web_unlock' | 'landing_page' | 'demo_request'
  company_name?: string
  role?: string
  store_count?: number
  pos_system?: string
  marketing_consent?: boolean
  utm_source?: string
  utm_medium?: string
  utm_campaign?: string
}

export async function saveEmailSignup(data: EmailSignupData): Promise<{ success: boolean; id?: string; error?: string }> {
  const supabase = getSupabase()

  if (!supabase) {
    console.warn('[API] Supabase not configured, skipping email save')
    return { success: true } // Gracefully degrade
  }

  try {
    const { data: result, error } = await supabase
      .from('email_signups')
      .insert({
        email: data.email,
        source: data.source || 'web_unlock',
        company_name: data.company_name,
        role: data.role,
        store_count: data.store_count,
        pos_system: data.pos_system,
        marketing_consent: data.marketing_consent || false,
        utm_source: data.utm_source,
        utm_medium: data.utm_medium,
        utm_campaign: data.utm_campaign,
        user_agent: typeof window !== 'undefined' ? window.navigator.userAgent : null,
        referrer: typeof document !== 'undefined' ? document.referrer || null : null,
      })
      .select('id')
      .single()

    if (error) {
      // Handle duplicate email gracefully
      if (error.code === '23505') {
        // Unique violation - email already exists for this source
        return { success: true } // Still count as success
      }
      throw error
    }

    return { success: true, id: result?.id }
  } catch (err) {
    console.error('[API] Failed to save email signup:', err)
    return { success: false, error: (err as Error).message }
  }
}

/**
 * Analysis Synopsis - stores aggregate analysis data
 */
export interface AnalysisSynopsisData {
  email_signup_id?: string
  file_hash: string
  file_row_count: number
  file_column_count?: number
  detection_counts: Record<string, number>
  top_leaks_by_primitive?: Record<string, Array<{ sku: string; score: number }>>
  total_impact_estimate_low?: number
  total_impact_estimate_high?: number
  currency?: string
  dataset_stats?: Record<string, unknown>
  seeding_summary?: Record<string, unknown>
  dimensions_used?: number
  codebook_size?: number
  processing_time_seconds?: number
  peak_memory_mb?: number
  engine_version?: string
}

export async function saveAnalysisSynopsis(
  data: AnalysisSynopsisData
): Promise<{ success: boolean; id?: string; error?: string }> {
  const supabase = getSupabase()

  if (!supabase) {
    console.warn('[API] Supabase not configured, skipping synopsis save')
    return { success: true }
  }

  try {
    const { data: result, error } = await supabase
      .from('analysis_synopses')
      .insert({
        email_signup_id: data.email_signup_id,
        file_hash: data.file_hash,
        file_row_count: data.file_row_count,
        file_column_count: data.file_column_count,
        detection_counts: data.detection_counts,
        top_leaks_by_primitive: data.top_leaks_by_primitive,
        total_impact_estimate_low: data.total_impact_estimate_low,
        total_impact_estimate_high: data.total_impact_estimate_high,
        currency: data.currency || 'USD',
        dataset_stats: data.dataset_stats,
        seeding_summary: data.seeding_summary,
        dimensions_used: data.dimensions_used,
        codebook_size: data.codebook_size,
        processing_time_seconds: data.processing_time_seconds,
        peak_memory_mb: data.peak_memory_mb,
        engine_version: data.engine_version,
      })
      .select('id')
      .single()

    if (error) throw error

    return { success: true, id: result?.id }
  } catch (err) {
    console.error('[API] Failed to save analysis synopsis:', err)
    return { success: false, error: (err as Error).message }
  }
}

/**
 * Get past analyses for a user (by email)
 */
export async function getPastAnalyses(email: string): Promise<{
  success: boolean
  analyses?: Array<{
    id: string
    file_hash: string
    file_row_count: number
    detection_counts: Record<string, number>
    created_at: string
  }>
  error?: string
}> {
  const supabase = getSupabase()

  if (!supabase) {
    return { success: false, error: 'Supabase not configured' }
  }

  try {
    // First get the email signup IDs for this email
    const { data: signups, error: signupError } = await supabase
      .from('email_signups')
      .select('id')
      .eq('email', email)

    if (signupError) throw signupError
    if (!signups || signups.length === 0) {
      return { success: true, analyses: [] }
    }

    const signupIds = signups.map((s: { id: string }) => s.id)

    // Then get analyses linked to those signups
    const { data: analyses, error: analysisError } = await supabase
      .from('analysis_synopses')
      .select('id, file_hash, file_row_count, detection_counts, created_at')
      .in('email_signup_id', signupIds)
      .order('created_at', { ascending: false })
      .limit(20)

    if (analysisError) throw analysisError

    return { success: true, analyses: analyses || [] }
  } catch (err) {
    console.error('[API] Failed to get past analyses:', err)
    return { success: false, error: (err as Error).message }
  }
}

/**
 * Check if Supabase is available (for conditional UI)
 */
export function isBackendAvailable(): boolean {
  return isSupabaseConfigured()
}
