// src/app/layout.tsx
import './globals.css'
import Providers from '@/components/theme-provider'
import { AppShell } from '@/components/app-shell'

export const metadata = {
  title: 'Profit Sentinel | Find Hidden Profit Leaks in Your Inventory',
  description: 'Deterministic analysis engine detects 11 types of profit leaks in your inventory. 36K SKUs in under 3 seconds. Free analysis, no credit card required.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning className="dark">
      <body className="bg-slate-900 text-slate-100 antialiased">
        <Providers>
          <AppShell showFooter={true}>
            {children}
          </AppShell>
        </Providers>
      </body>
    </html>
  )
}
