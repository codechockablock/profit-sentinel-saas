/**
 * Tests for Sidebar component
 * 
 * Tests rendering, theme toggle, and GrokChat integration
 */

import React from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ThemeProvider } from 'next-themes'
import Sidebar from '@/components/sidebar'

// Mock GrokChat component
jest.mock('@/components/grok-chat', () => {
  return function MockGrokChat() {
    return <div data-testid="grok-chat-mock">GrokChat Component</div>
  }
})

// Mock next-themes
const mockSetTheme = jest.fn()
jest.mock('next-themes', () => ({
  ...jest.requireActual('next-themes'),
  useTheme: () => ({
    theme: 'light',
    setTheme: mockSetTheme,
    systemTheme: 'light'
  })
}))

// Wrapper component for testing
const renderWithProviders = (ui: React.ReactElement) => {
  return render(
    <ThemeProvider attribute="class">
      {ui}
    </ThemeProvider>
  )
}

describe('Sidebar Component', () => {
  beforeEach(() => {
    mockSetTheme.mockClear()
  })

  describe('Rendering', () => {
    it('renders sidebar with title', () => {
      renderWithProviders(<Sidebar />)
      
      expect(screen.getByText('MySaaS')).toBeInTheDocument()
    })

    it('renders workspace name', () => {
      renderWithProviders(<Sidebar />)
      
      expect(screen.getByText('Default Workspace')).toBeInTheDocument()
    })

    it('renders channel list', () => {
      renderWithProviders(<Sidebar />)
      
      expect(screen.getByText('# general')).toBeInTheDocument()
      expect(screen.getByText('# random')).toBeInTheDocument()
      expect(screen.getByText('# ideas')).toBeInTheDocument()
    })

    it('renders Channels header', () => {
      renderWithProviders(<Sidebar />)
      
      expect(screen.getByText('Channels')).toBeInTheDocument()
    })
  })

  describe('Theme Toggle', () => {
    it('renders dark mode toggle button', () => {
      renderWithProviders(<Sidebar />)
      
      expect(screen.getByText(/dark mode/i)).toBeInTheDocument()
    })

    it('calls setTheme when toggle clicked', async () => {
      const user = userEvent.setup()
      renderWithProviders(<Sidebar />)
      
      const toggleButton = screen.getByText(/dark mode/i).closest('button')
      expect(toggleButton).toBeInTheDocument()
      
      if (toggleButton) {
        await user.click(toggleButton)
        expect(mockSetTheme).toHaveBeenCalled()
      }
    })
  })

  describe('Grok Assistant Toggle', () => {
    it('renders Grok Assistant button', () => {
      renderWithProviders(<Sidebar />)
      
      expect(screen.getByText('Grok Assistant')).toBeInTheDocument()
    })

    it('shows GrokChat by default', () => {
      renderWithProviders(<Sidebar />)
      
      expect(screen.getByTestId('grok-chat-mock')).toBeInTheDocument()
    })

    it('toggles GrokChat visibility when button clicked', async () => {
      const user = userEvent.setup()
      renderWithProviders(<Sidebar />)
      
      // Initially visible
      expect(screen.getByTestId('grok-chat-mock')).toBeInTheDocument()
      
      // Click to hide
      const toggleButton = screen.getByText('Grok Assistant').closest('button')
      if (toggleButton) {
        await user.click(toggleButton)
        expect(screen.queryByTestId('grok-chat-mock')).not.toBeInTheDocument()
        
        // Click to show again
        await user.click(toggleButton)
        expect(screen.getByTestId('grok-chat-mock')).toBeInTheDocument()
      }
    })

    it('shows minus sign when GrokChat is visible', () => {
      renderWithProviders(<Sidebar />)

      const toggleButton = screen.getByText('Grok Assistant').closest('button')
      // Component uses Unicode minus sign (U+2212) not hyphen (U+002D)
      expect(toggleButton).toHaveTextContent('−')
    })
  })

  describe('Active Channel', () => {
    it('shows general channel as active', () => {
      renderWithProviders(<Sidebar />)
      
      const generalChannel = screen.getByText('# general')
      // Check it has active styling (blue background)
      expect(generalChannel.closest('li')).toHaveClass('bg-blue-100')
    })
  })

  describe('Accessibility', () => {
    it('all buttons are keyboard accessible', () => {
      renderWithProviders(<Sidebar />)
      
      const buttons = screen.getAllByRole('button')
      buttons.forEach(button => {
        expect(button).not.toHaveAttribute('tabindex', '-1')
      })
    })
  })
})
