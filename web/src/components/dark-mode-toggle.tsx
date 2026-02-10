// src/components/dark-mode-toggle.tsx
'use client'

export default function DarkModeToggle() {
  return (
    <button
      onClick={() => {
        document.documentElement.classList.toggle('dark')
        // Optional: persist preference
        localStorage.theme = document.documentElement.classList.contains('dark') ? 'dark' : 'light'
      }}
      className="fixed bottom-8 right-8 z-[9999] flex h-16 w-16 items-center justify-center rounded-full bg-gray-800 text-white shadow-2xl ring-4 ring-white dark:bg-yellow-400 dark:text-gray-900 dark:ring-gray-900 md:h-20 md:w-20"
      aria-label="Toggle dark mode"
    >
      <span className="text-4xl md:text-5xl">🌙</span>
    </button>
  )
}
