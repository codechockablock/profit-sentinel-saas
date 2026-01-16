/**
 * Tests for GrokChat component
 * 
 * Tests rendering, user interactions, API calls, and error handling
 */

import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import GrokChat from '@/components/grok-chat'

// Mock fetch globally
const mockFetch = jest.fn()
global.fetch = mockFetch

describe('GrokChat Component', () => {
  beforeEach(() => {
    mockFetch.mockClear()
  })

  describe('Rendering', () => {
    it('renders initial state correctly', () => {
      render(<GrokChat />)
      
      expect(screen.getByPlaceholderText(/message grok/i)).toBeInTheDocument()
      expect(screen.getByRole('button', { name: /send/i })).toBeInTheDocument()
      expect(screen.getByText(/ask grok anything/i)).toBeInTheDocument()
    })

    it('renders input field as enabled', () => {
      render(<GrokChat />)
      
      const input = screen.getByPlaceholderText(/message grok/i)
      expect(input).not.toBeDisabled()
    })

    it('renders send button as disabled when input is empty', () => {
      render(<GrokChat />)
      
      const button = screen.getByRole('button', { name: /send/i })
      expect(button).toBeDisabled()
    })
  })

  describe('User Input', () => {
    it('enables send button when user types', async () => {
      const user = userEvent.setup()
      render(<GrokChat />)
      
      const input = screen.getByPlaceholderText(/message grok/i)
      await user.type(input, 'Hello Grok')
      
      const button = screen.getByRole('button', { name: /send/i })
      expect(button).not.toBeDisabled()
    })

    it('clears input after sending message', async () => {
      const user = userEvent.setup()
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ content: 'Hello!' })
      })
      
      render(<GrokChat />)
      
      const input = screen.getByPlaceholderText(/message grok/i) as HTMLInputElement
      await user.type(input, 'Hello Grok')
      await user.click(screen.getByRole('button', { name: /send/i }))
      
      // Input should be cleared
      await waitFor(() => {
        expect(input.value).toBe('')
      })
    })

    it('trims whitespace from messages', async () => {
      const user = userEvent.setup()
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ content: 'Response' })
      })
      
      render(<GrokChat />)
      
      const input = screen.getByPlaceholderText(/message grok/i)
      await user.type(input, '   Hello   ')
      await user.click(screen.getByRole('button', { name: /send/i }))
      
      await waitFor(() => {
        expect(screen.getByText('Hello')).toBeInTheDocument()
      })
    })

    it('does not send empty messages', async () => {
      const user = userEvent.setup()
      render(<GrokChat />)
      
      const input = screen.getByPlaceholderText(/message grok/i)
      await user.type(input, '   ')
      
      const button = screen.getByRole('button', { name: /send/i })
      expect(button).toBeDisabled()
    })
  })

  describe('API Integration', () => {
    it('sends message to API and displays response', async () => {
      const user = userEvent.setup()
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ content: 'I am Grok, your AI assistant!' })
      })
      
      render(<GrokChat />)
      
      const input = screen.getByPlaceholderText(/message grok/i)
      await user.type(input, 'Who are you?')
      await user.click(screen.getByRole('button', { name: /send/i }))
      
      // Should show user message
      await waitFor(() => {
        expect(screen.getByText('Who are you?')).toBeInTheDocument()
      })
      
      // Should show API response
      await waitFor(() => {
        expect(screen.getByText('I am Grok, your AI assistant!')).toBeInTheDocument()
      })
    })

    it('calls correct API endpoint', async () => {
      const user = userEvent.setup()
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ content: 'Response' })
      })
      
      render(<GrokChat />)
      
      const input = screen.getByPlaceholderText(/message grok/i)
      await user.type(input, 'Test message')
      await user.click(screen.getByRole('button', { name: /send/i }))
      
      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith('/api/grok', expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' }
        }))
      })
    })

    it('includes message history in API request', async () => {
      const user = userEvent.setup()
      
      // First message
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ content: 'Response 1' })
      })
      
      render(<GrokChat />)
      
      const input = screen.getByPlaceholderText(/message grok/i)
      await user.type(input, 'First message')
      await user.click(screen.getByRole('button', { name: /send/i }))
      
      await waitFor(() => {
        expect(screen.getByText('Response 1')).toBeInTheDocument()
      })
      
      // Second message
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ content: 'Response 2' })
      })
      
      await user.type(input, 'Second message')
      await user.click(screen.getByRole('button', { name: /send/i }))
      
      await waitFor(() => {
        // Check that second call includes previous messages
        const lastCall = mockFetch.mock.calls[1]
        const body = JSON.parse(lastCall[1].body)
        expect(body.messages.length).toBe(3) // user1, assistant1, user2
      })
    })
  })

  describe('Loading State', () => {
    it('shows loading indicator while waiting for response', async () => {
      const user = userEvent.setup()
      
      // Create a promise that doesn't resolve immediately
      let resolvePromise: (value: any) => void
      const pendingPromise = new Promise(resolve => {
        resolvePromise = resolve
      })
      
      mockFetch.mockReturnValueOnce(pendingPromise)
      
      render(<GrokChat />)
      
      const input = screen.getByPlaceholderText(/message grok/i)
      await user.type(input, 'Hello')
      await user.click(screen.getByRole('button', { name: /send/i }))
      
      // Should show loading state
      await waitFor(() => {
        expect(screen.getByText(/grok is thinking/i)).toBeInTheDocument()
      })
      
      // Resolve the promise
      resolvePromise!({
        ok: true,
        json: () => Promise.resolve({ content: 'Done!' })
      })
    })

    it('disables input while loading', async () => {
      const user = userEvent.setup()
      
      let resolvePromise: (value: any) => void
      const pendingPromise = new Promise(resolve => {
        resolvePromise = resolve
      })
      
      mockFetch.mockReturnValueOnce(pendingPromise)
      
      render(<GrokChat />)
      
      const input = screen.getByPlaceholderText(/message grok/i)
      await user.type(input, 'Hello')
      await user.click(screen.getByRole('button', { name: /send/i }))
      
      await waitFor(() => {
        expect(input).toBeDisabled()
      })
      
      // Cleanup
      resolvePromise!({
        ok: true,
        json: () => Promise.resolve({ content: 'Done!' })
      })
    })
  })

  describe('Error Handling', () => {
    it('displays error message on API failure', async () => {
      const user = userEvent.setup()
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500
      })
      
      render(<GrokChat />)
      
      const input = screen.getByPlaceholderText(/message grok/i)
      await user.type(input, 'Hello')
      await user.click(screen.getByRole('button', { name: /send/i }))
      
      await waitFor(() => {
        expect(screen.getByText(/grok unavailable/i)).toBeInTheDocument()
      })
    })

    it('shows try again button on error', async () => {
      const user = userEvent.setup()
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500
      })
      
      render(<GrokChat />)
      
      const input = screen.getByPlaceholderText(/message grok/i)
      await user.type(input, 'Hello')
      await user.click(screen.getByRole('button', { name: /send/i }))
      
      await waitFor(() => {
        expect(screen.getByRole('button', { name: /try again/i })).toBeInTheDocument()
      })
    })

    it('handles network errors gracefully', async () => {
      const user = userEvent.setup()
      mockFetch.mockRejectedValueOnce(new Error('Network error'))
      
      render(<GrokChat />)
      
      const input = screen.getByPlaceholderText(/message grok/i)
      await user.type(input, 'Hello')
      await user.click(screen.getByRole('button', { name: /send/i }))
      
      await waitFor(() => {
        expect(screen.getByText(/grok unavailable/i)).toBeInTheDocument()
      })
    })

    it('clears error state when try again is clicked', async () => {
      const user = userEvent.setup()
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500
      })
      
      render(<GrokChat />)
      
      const input = screen.getByPlaceholderText(/message grok/i)
      await user.type(input, 'Hello')
      await user.click(screen.getByRole('button', { name: /send/i }))
      
      await waitFor(() => {
        expect(screen.getByText(/grok unavailable/i)).toBeInTheDocument()
      })
      
      await user.click(screen.getByRole('button', { name: /try again/i }))
      
      await waitFor(() => {
        expect(screen.queryByText(/grok unavailable/i)).not.toBeInTheDocument()
      })
    })
  })

  describe('Message Display', () => {
    it('displays user messages with correct styling', async () => {
      const user = userEvent.setup()
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ content: 'Response' })
      })
      
      render(<GrokChat />)
      
      const input = screen.getByPlaceholderText(/message grok/i)
      await user.type(input, 'User message')
      await user.click(screen.getByRole('button', { name: /send/i }))
      
      await waitFor(() => {
        const userMessage = screen.getByText('User message')
        // Check parent has justify-end class (user messages are right-aligned)
        expect(userMessage.closest('div')?.parentElement).toHaveClass('justify-end')
      })
    })

    it('displays assistant messages with correct styling', async () => {
      const user = userEvent.setup()
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ content: 'Assistant response' })
      })
      
      render(<GrokChat />)
      
      const input = screen.getByPlaceholderText(/message grok/i)
      await user.type(input, 'Hello')
      await user.click(screen.getByRole('button', { name: /send/i }))
      
      await waitFor(() => {
        const assistantMessage = screen.getByText('Assistant response')
        expect(assistantMessage.closest('div')?.parentElement).toHaveClass('justify-start')
      })
    })
  })

  describe('Keyboard Navigation', () => {
    it('submits on Enter key press', async () => {
      const user = userEvent.setup()
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ content: 'Response' })
      })
      
      render(<GrokChat />)
      
      const input = screen.getByPlaceholderText(/message grok/i)
      await user.type(input, 'Hello{enter}')
      
      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalled()
      })
    })

    it('does not submit on Shift+Enter', async () => {
      const user = userEvent.setup()
      render(<GrokChat />)
      
      const input = screen.getByPlaceholderText(/message grok/i)
      await user.type(input, 'Hello')
      await user.keyboard('{Shift>}{Enter}{/Shift}')
      
      // Fetch should not be called
      expect(mockFetch).not.toHaveBeenCalled()
    })
  })
})
