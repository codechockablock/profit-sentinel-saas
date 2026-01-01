// src/app/layout.tsx
import './globals.css'
import { Inter } from 'next/font/google'
import Sidebar from '@/components/sidebar'
import Providers from '@/components/theme-provider'

const inter = Inter({ subsets: ['latin'] })

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <Providers>
          <div className="flex h-screen">
            <Sidebar />
            <main className="flex-1 overflow-hidden">
              {children}
            </main>
          </div>
        </Providers>
      </body>
    </html>
  )
}