// src/app/page.tsx
import Link from 'next/link'

export default function Home() {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative py-20 md:py-32">
        <div className="max-w-6xl mx-auto px-4 text-center">
          {/* Logo */}
          <div className="mb-8">
            <img
              src="https://i.imgur.com/68NbW7U.png"
              alt="Profit Sentinel"
              className="mx-auto h-40 md:h-56 object-contain drop-shadow-[0_0_40px_rgba(16,185,129,0.5)]"
            />
          </div>

          {/* Headline */}
          <h1 className="text-4xl md:text-6xl font-bold text-white mb-6">
            Find the{' '}
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-emerald-600">
              Profit Leaks
            </span>{' '}
            Hiding in Your Inventory
          </h1>

          <p className="text-xl md:text-2xl text-slate-400 max-w-3xl mx-auto mb-8">
            AI-powered analysis finds dead stock, margin erosion, shrinkage patterns, and 5 other profit-draining issues in your POS data.
          </p>

          {/* CTA */}
          <Link
            href="/upload"
            className="inline-flex items-center gap-3 px-10 py-5 bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-bold text-xl rounded-2xl hover:from-emerald-600 hover:to-emerald-700 transition transform hover:scale-105 shadow-lg shadow-emerald-500/25"
          >
            <UploadIcon className="w-6 h-6" />
            Analyze My Inventory Free
          </Link>

          {/* Trust Badges */}
          <div className="flex flex-wrap justify-center gap-6 mt-10 text-sm text-slate-400">
            <Badge icon={<ShieldIcon />} text="Data never leaves your browser" />
            <Badge icon={<ClockIcon />} text="Results in 60 seconds" />
            <Badge icon={<CheckIcon />} text="No credit card required" />
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-20 bg-slate-800/30">
        <div className="max-w-6xl mx-auto px-4">
          <h2 className="text-3xl md:text-4xl font-bold text-center text-white mb-4">
            How It Works
          </h2>
          <p className="text-xl text-slate-400 text-center mb-12 max-w-2xl mx-auto">
            Three steps to find thousands in hidden profits
          </p>

          <div className="grid md:grid-cols-3 gap-8">
            <StepCard
              number="1"
              title="Upload Your Export"
              description="Export your inventory from any POS system - Paladin, Square, Lightspeed, Clover, or custom CSV."
            />
            <StepCard
              number="2"
              title="AI Scans for Leaks"
              description="Our engine checks for 8 types of profit leaks: dead stock, margin erosion, shrinkage, and more."
            />
            <StepCard
              number="3"
              title="Get Actionable Results"
              description="See exactly which SKUs need attention, with dollar impact estimates and fix recommendations."
            />
          </div>
        </div>
      </section>

      {/* Leak Types */}
      <section className="py-20">
        <div className="max-w-6xl mx-auto px-4">
          <h2 className="text-3xl md:text-4xl font-bold text-center text-white mb-4">
            What We Find
          </h2>
          <p className="text-xl text-slate-400 text-center mb-12 max-w-2xl mx-auto">
            8 types of profit leaks, detected automatically
          </p>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            <LeakTypeCard
              title="Dead Inventory"
              description="Items sitting on shelves with no sales velocity"
              color="text-red-400"
              bgColor="bg-red-500/10"
              borderColor="border-red-500/30"
            />
            <LeakTypeCard
              title="Margin Erosion"
              description="Items selling below target margin"
              color="text-orange-400"
              bgColor="bg-orange-500/10"
              borderColor="border-orange-500/30"
            />
            <LeakTypeCard
              title="Low Stock"
              description="Fast sellers about to stock out"
              color="text-amber-400"
              bgColor="bg-amber-500/10"
              borderColor="border-amber-500/30"
            />
            <LeakTypeCard
              title="Shrinkage"
              description="Inventory discrepancies indicating theft or loss"
              color="text-purple-400"
              bgColor="bg-purple-500/10"
              borderColor="border-purple-500/30"
            />
            <LeakTypeCard
              title="Overstock"
              description="Capital tied up in slow-moving inventory"
              color="text-blue-400"
              bgColor="bg-blue-500/10"
              borderColor="border-blue-500/30"
            />
            <LeakTypeCard
              title="Price Issues"
              description="Items priced below suggested retail"
              color="text-pink-400"
              bgColor="bg-pink-500/10"
              borderColor="border-pink-500/30"
            />
            <LeakTypeCard
              title="Negative Inventory"
              description="System errors showing impossible stock levels"
              color="text-cyan-400"
              bgColor="bg-cyan-500/10"
              borderColor="border-cyan-500/30"
            />
            <LeakTypeCard
              title="Margin Leaks"
              description="High-volume items with below-cost pricing"
              color="text-emerald-400"
              bgColor="bg-emerald-500/10"
              borderColor="border-emerald-500/30"
            />
          </div>
        </div>
      </section>

      {/* Privacy Section */}
      <section className="py-20 bg-emerald-500/5 border-y border-emerald-500/20">
        <div className="max-w-4xl mx-auto px-4 text-center">
          <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-emerald-500/20 flex items-center justify-center">
            <ShieldIcon className="w-8 h-8 text-emerald-400" />
          </div>
          <h2 className="text-3xl font-bold text-white mb-4">
            Your Data Stays Private
          </h2>
          <p className="text-lg text-slate-400 max-w-2xl mx-auto mb-8">
            All analysis happens <strong className="text-emerald-400">locally in your browser</strong>.
            Your inventory files are never uploaded to our servers. We only store
            anonymized aggregate statistics - never your actual SKUs or item names.
          </p>
          <Link
            href="/privacy"
            className="inline-flex items-center gap-2 text-emerald-400 hover:text-emerald-300 font-medium"
          >
            Read our Privacy Policy
            <ArrowRightIcon className="w-4 h-4" />
          </Link>
        </div>
      </section>

      {/* Final CTA */}
      <section className="py-20">
        <div className="max-w-4xl mx-auto px-4 text-center">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Ready to Find Your Hidden Profits?
          </h2>
          <p className="text-xl text-slate-400 mb-8">
            Free analysis. Results in under 2 minutes. No credit card required.
          </p>
          <Link
            href="/upload"
            className="inline-flex items-center gap-3 px-10 py-5 bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-bold text-xl rounded-2xl hover:from-emerald-600 hover:to-emerald-700 transition transform hover:scale-105 shadow-lg shadow-emerald-500/25"
          >
            <UploadIcon className="w-6 h-6" />
            Start My Free Analysis
          </Link>
        </div>
      </section>
    </div>
  )
}

// Components
function Badge({ icon, text }: { icon: React.ReactNode; text: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-emerald-400">{icon}</span>
      <span>{text}</span>
    </div>
  )
}

function StepCard({ number, title, description }: { number: string; title: string; description: string }) {
  return (
    <div className="bg-white/5 rounded-2xl border border-slate-700 p-8 relative text-center">
      <div className="absolute -top-4 left-1/2 -translate-x-1/2 w-10 h-10 rounded-full bg-emerald-500 text-white font-bold flex items-center justify-center text-lg">
        {number}
      </div>
      <h3 className="text-xl font-bold text-white mb-3 mt-4">{title}</h3>
      <p className="text-slate-400">{description}</p>
    </div>
  )
}

function LeakTypeCard({
  title,
  description,
  color,
  bgColor,
  borderColor,
}: {
  title: string
  description: string
  color: string
  bgColor: string
  borderColor: string
}) {
  return (
    <div className={`${bgColor} rounded-xl border ${borderColor} p-5`}>
      <h3 className={`text-lg font-bold ${color} mb-2`}>{title}</h3>
      <p className="text-sm text-slate-400">{description}</p>
    </div>
  )
}

// Icons
function UploadIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  )
}

function ShieldIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className={className || "w-4 h-4"}>
      <path fillRule="evenodd" d="M9.661 2.237a.531.531 0 01.678 0 11.947 11.947 0 007.078 2.749.5.5 0 01.479.425c.069.52.104 1.05.104 1.59 0 5.162-3.26 9.563-7.834 11.256a.48.48 0 01-.332 0C5.26 16.564 2 12.163 2 7c0-.538.035-1.069.104-1.589a.5.5 0 01.48-.425 11.947 11.947 0 007.077-2.75z" clipRule="evenodd" />
    </svg>
  )
}

function ClockIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className={className || "w-4 h-4"}>
      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm.75-13a.75.75 0 00-1.5 0v5c0 .414.336.75.75.75h4a.75.75 0 000-1.5h-3.25V5z" clipRule="evenodd" />
    </svg>
  )
}

function CheckIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className={className || "w-4 h-4"}>
      <path fillRule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clipRule="evenodd" />
    </svg>
  )
}

function ArrowRightIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className={className}>
      <path fillRule="evenodd" d="M3 10a.75.75 0 01.75-.75h10.638L10.23 5.29a.75.75 0 111.04-1.08l5.5 5.25a.75.75 0 010 1.08l-5.5 5.25a.75.75 0 11-1.04-1.08l4.158-3.96H3.75A.75.75 0 013 10z" clipRule="evenodd" />
    </svg>
  )
}
