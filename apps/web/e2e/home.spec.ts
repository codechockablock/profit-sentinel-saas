/**
 * E2E Tests for Home Page
 *
 * Tests the main user flows using Playwright
 *
 * NOTE: Tests for Grok Chat features are skipped until the UI is implemented.
 * See: https://github.com/codechockablock/profit-sentinel-saas/issues/XX
 */

import { test, expect } from '@playwright/test'

test.describe('Home Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('has correct title', async ({ page }) => {
    await expect(page).toHaveTitle(/Profit Sentinel|MySaaS/)
  })

  // Skip tests that depend on sidebar elements not yet implemented
  test.skip('displays sidebar with navigation', async ({ page }) => {
    // Check sidebar exists
    await expect(page.locator('text=MySaaS')).toBeVisible()

    // Check channel list
    await expect(page.locator('text=# general')).toBeVisible()
    await expect(page.locator('text=# random')).toBeVisible()
  })

  // Skip - Grok Assistant UI not yet implemented
  test.skip('displays Grok Assistant toggle', async ({ page }) => {
    await expect(page.locator('text=Grok Assistant')).toBeVisible()
  })

  // Skip - Dark mode toggle button not yet implemented
  test.skip('can toggle dark mode', async ({ page }) => {
    // Find and click theme toggle
    const themeButton = page.locator('button:has-text("Dark Mode"), button:has-text("Light Mode")')
    await expect(themeButton).toBeVisible()

    // Get initial state
    const initialText = await themeButton.textContent()

    // Click toggle
    await themeButton.click()

    // Verify state changed
    await expect(themeButton).not.toHaveText(initialText!)
  })

  // Skip - Grok chat panel not yet implemented
  test.skip('can toggle Grok chat panel', async ({ page }) => {
    const grokButton = page.locator('button:has-text("Grok Assistant")')

    // Initially, the chat should be visible (default state)
    // Toggle to hide
    await grokButton.click()

    // The input field should not be visible when hidden
    await expect(page.locator('input[placeholder*="Message Grok"]')).not.toBeVisible()

    // Toggle to show again
    await grokButton.click()

    // Chat input should be visible
    await expect(page.locator('input[placeholder*="Message Grok"]')).toBeVisible()
  })
})

// Skip entire Grok Chat test suite - UI not yet implemented
test.describe.skip('Grok Chat', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('displays empty state message', async ({ page }) => {
    await expect(page.locator('text=Ask Grok anything')).toBeVisible()
  })

  test('has input field and send button', async ({ page }) => {
    await expect(page.locator('input[placeholder*="Message Grok"]')).toBeVisible()
    await expect(page.locator('button:has-text("Send")')).toBeVisible()
  })

  test('send button is disabled when input is empty', async ({ page }) => {
    const sendButton = page.locator('button:has-text("Send")')
    await expect(sendButton).toBeDisabled()
  })

  test('send button is enabled when input has text', async ({ page }) => {
    const input = page.locator('input[placeholder*="Message Grok"]')
    const sendButton = page.locator('button:has-text("Send")')

    await input.fill('Hello Grok')

    await expect(sendButton).not.toBeDisabled()
  })

  test('can type message and see it appear', async ({ page }) => {
    const input = page.locator('input[placeholder*="Message Grok"]')

    await input.fill('Test message')

    await expect(input).toHaveValue('Test message')
  })

  test('submitting message shows user message', async ({ page }) => {
    const input = page.locator('input[placeholder*="Message Grok"]')
    const sendButton = page.locator('button:has-text("Send")')

    await input.fill('Hello from E2E test')
    await sendButton.click()

    // User message should appear
    await expect(page.locator('text=Hello from E2E test')).toBeVisible()
  })
})

test.describe('Responsive Design', () => {
  // Skip - sidebar not yet implemented
  test.skip('sidebar is visible on desktop', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 720 })
    await page.goto('/')

    await expect(page.locator('text=MySaaS')).toBeVisible()
  })

  test('main content area exists', async ({ page }) => {
    await page.goto('/')

    // Check for main content wrapper
    const mainContent = page.locator('main, [role="main"], .flex-1')
    await expect(mainContent.first()).toBeVisible()
  })
})

test.describe('Accessibility', () => {
  test('page has no automatically detectable a11y issues', async ({ page }) => {
    await page.goto('/')

    // Basic accessibility checks
    // Check that all buttons have accessible names
    const buttons = await page.locator('button').all()
    for (const button of buttons) {
      const text = await button.textContent()
      const ariaLabel = await button.getAttribute('aria-label')
      expect(text || ariaLabel).toBeTruthy()
    }
  })

  // Skip - Grok input not yet implemented
  test.skip('input fields have proper labels or placeholders', async ({ page }) => {
    await page.goto('/')

    const input = page.locator('input[placeholder*="Message Grok"]')
    const placeholder = await input.getAttribute('placeholder')

    expect(placeholder).toBeTruthy()
  })

  // Skip - Grok input not yet implemented
  test.skip('focus is visible on interactive elements', async ({ page }) => {
    await page.goto('/')

    const input = page.locator('input[placeholder*="Message Grok"]')
    await input.focus()

    // Input should be focused
    await expect(input).toBeFocused()
  })
})

test.describe('Error Handling', () => {
  test('page loads without JavaScript errors', async ({ page }) => {
    const errors: string[] = []
    page.on('pageerror', err => errors.push(err.message))

    await page.goto('/')

    // Wait for page to settle
    await page.waitForLoadState('networkidle')

    expect(errors).toHaveLength(0)
  })

  // Skip - Grok API handling not yet implemented
  test.skip('handles missing API gracefully', async ({ page }) => {
    // Mock failed API response
    await page.route('/api/grok', route => {
      route.fulfill({
        status: 503,
        body: JSON.stringify({ error: 'Service unavailable' })
      })
    })

    await page.goto('/')

    const input = page.locator('input[placeholder*="Message Grok"]')
    const sendButton = page.locator('button:has-text("Send")')

    await input.fill('Test')
    await sendButton.click()

    // Should show error state
    await expect(page.locator('text=/unavailable|error/i')).toBeVisible({ timeout: 5000 })
  })
})
