/**
 * Tests for Grok API route
 * 
 * Tests the /api/grok endpoint handler
 */

import { POST } from '@/app/api/grok/route'
import { NextRequest } from 'next/server'

// Mock the AI SDK
jest.mock('@ai-sdk/xai', () => ({
  xai: jest.fn(() => 'mock-model')
}))

jest.mock('ai', () => ({
  streamText: jest.fn()
}))

describe('Grok API Route', () => {
  const originalEnv = process.env

  beforeEach(() => {
    jest.clearAllMocks()
    process.env = { ...originalEnv }
  })

  afterAll(() => {
    process.env = originalEnv
  })

  describe('Without API Key', () => {
    it('returns 503 when XAI_API_KEY is not set', async () => {
      delete process.env.XAI_API_KEY
      
      const request = new NextRequest('http://localhost:3000/api/grok', {
        method: 'POST',
        body: JSON.stringify({ messages: [{ role: 'user', content: 'Hello' }] })
      })
      
      const response = await POST(request)
      
      expect(response.status).toBe(503)
      const data = await response.json()
      expect(data.error).toBe('AI service not configured')
    })
  })

  describe('With API Key', () => {
    beforeEach(() => {
      process.env.XAI_API_KEY = 'test-api-key'
    })

    it('calls streamText with correct parameters', async () => {
      const { streamText } = require('ai')
      const mockResponse = {
        toTextStreamResponse: jest.fn(() => new Response('stream'))
      }
      streamText.mockResolvedValue(mockResponse)
      
      const request = new NextRequest('http://localhost:3000/api/grok', {
        method: 'POST',
        body: JSON.stringify({ 
          messages: [{ role: 'user', content: 'Hello Grok' }] 
        })
      })
      
      await POST(request)
      
      expect(streamText).toHaveBeenCalledWith(expect.objectContaining({
        messages: [{ role: 'user', content: 'Hello Grok' }],
        system: expect.stringContaining('Grok')
      }))
    })

    it('returns stream response on success', async () => {
      const { streamText } = require('ai')
      const mockStreamResponse = new Response('stream content')
      streamText.mockResolvedValue({
        toTextStreamResponse: () => mockStreamResponse
      })
      
      const request = new NextRequest('http://localhost:3000/api/grok', {
        method: 'POST',
        body: JSON.stringify({ 
          messages: [{ role: 'user', content: 'Hello' }] 
        })
      })
      
      const response = await POST(request)
      
      expect(response).toBe(mockStreamResponse)
    })

    it('returns 500 on general error', async () => {
      const { streamText } = require('ai')
      streamText.mockRejectedValue(new Error('Something went wrong'))
      
      const request = new NextRequest('http://localhost:3000/api/grok', {
        method: 'POST',
        body: JSON.stringify({ 
          messages: [{ role: 'user', content: 'Hello' }] 
        })
      })
      
      const response = await POST(request)
      
      expect(response.status).toBe(500)
      const data = await response.json()
      expect(data.error).toBe('AI request failed')
    })

    it('returns 401 on API key error', async () => {
      const { streamText } = require('ai')
      streamText.mockRejectedValue(new Error('Invalid API key'))
      
      const request = new NextRequest('http://localhost:3000/api/grok', {
        method: 'POST',
        body: JSON.stringify({ 
          messages: [{ role: 'user', content: 'Hello' }] 
        })
      })
      
      const response = await POST(request)
      
      expect(response.status).toBe(401)
      const data = await response.json()
      expect(data.error).toBe('Invalid API key')
    })
  })

  describe('Request Validation', () => {
    beforeEach(() => {
      process.env.XAI_API_KEY = 'test-api-key'
    })

    it('handles malformed JSON', async () => {
      const request = new NextRequest('http://localhost:3000/api/grok', {
        method: 'POST',
        body: 'not valid json'
      })
      
      const response = await POST(request)
      
      // Should return error status (500 for general error)
      expect(response.status).toBe(500)
    })
  })
})
