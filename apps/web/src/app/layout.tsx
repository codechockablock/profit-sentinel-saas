// src/app/layout.tsx
import './globals.css'
import { Inter } from 'next/font/google'
import Providers from '@/components/theme-provider'
import { AppShell } from '@/components/app-shell'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'Profit Sentinel | Find Hidden Profit Leaks in Your Inventory',
  description: 'AI-powered inventory analysis that finds the profit leaks hiding in your retail data. Free analysis, no credit card required.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning className="dark">
      <body className={`${inter.className} bg-slate-900 text-slate-100 antialiased`}>
        <Providers>
          <AppShell showFooter={true}>
            {children}
          </AppShell>
        </Providers>
      </body>
    </html>
  )
}
