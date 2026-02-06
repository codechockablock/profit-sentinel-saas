// src/components/theme-provider.tsx
'use client'

import { ThemeProvider } from 'next-themes'
import { useEffect, useState } from 'react'
import { ToastProvider } from './toast'

export default function ThemeProviderWrapper({ children }: { children: React.ReactNode }) {
  const [mounted, setMounted] = useState(false)

  useEffect(() => setMounted(true), [])

  // ToastProvider must always wrap children (works on server and client)
  // ThemeProvider needs client mount check to avoid hydration mismatch
  return (
    <ToastProvider>
      {mounted ? (
        <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
          {children}
        </ThemeProvider>
      ) : (
        <>{children}</>
      )}
    </ToastProvider>
  )
}
