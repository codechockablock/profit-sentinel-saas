/**
 * Tests for Supabase client
 *
 * Tests client initialization and environment variable handling
 */

describe('Supabase Client', () => {
  const originalEnv = process.env

  beforeEach(() => {
    // Reset module cache to test different env configurations
    jest.resetModules()
    process.env = { ...originalEnv }
  })

  afterAll(() => {
    process.env = originalEnv
  })

  describe('getSupabase function', () => {
    it('returns null when NEXT_PUBLIC_SUPABASE_URL is missing', () => {
      delete process.env.NEXT_PUBLIC_SUPABASE_URL
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY = 'test-key'

      const { getSupabase } = require('@/lib/supabase')

      expect(getSupabase()).toBeNull()
    })

    it('returns null when NEXT_PUBLIC_SUPABASE_ANON_KEY is missing', () => {
      process.env.NEXT_PUBLIC_SUPABASE_URL = 'https://test.supabase.co'
      delete process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

      const { getSupabase } = require('@/lib/supabase')

      expect(getSupabase()).toBeNull()
    })

    it('creates client when both env vars are present', () => {
      process.env.NEXT_PUBLIC_SUPABASE_URL = 'https://test.supabase.co'
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY = 'test-anon-key'

      // Mock createBrowserClient
      jest.mock('@supabase/ssr', () => ({
        createBrowserClient: jest.fn(() => ({ from: jest.fn() }))
      }))

      const { getSupabase } = require('@/lib/supabase')

      // Should not throw
      expect(() => getSupabase()).not.toThrow()
    })

    it('returns same instance on multiple calls (singleton)', () => {
      process.env.NEXT_PUBLIC_SUPABASE_URL = 'https://test.supabase.co'
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY = 'test-anon-key'

      jest.mock('@supabase/ssr', () => ({
        createBrowserClient: jest.fn(() => ({ from: jest.fn() }))
      }))

      const { getSupabase } = require('@/lib/supabase')

      const client1 = getSupabase()
      const client2 = getSupabase()

      expect(client1).toBe(client2)
    })
  })

  describe('supabase export', () => {
    it('is null when env vars are missing', () => {
      delete process.env.NEXT_PUBLIC_SUPABASE_URL
      delete process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

      const { supabase } = require('@/lib/supabase')

      expect(supabase).toBeNull()
    })
  })
})
