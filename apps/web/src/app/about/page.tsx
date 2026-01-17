import Link from 'next/link'

export const metadata = {
  title: 'About | Profit Sentinel',
  description: 'Learn about Profit Sentinel - the AI-powered tool that finds profit leaks hiding in your retail inventory data.',
}

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-900 to-slate-800">
      <div className="max-w-4xl mx-auto px-4 py-16">
        {/* Header */}
        <div className="text-center mb-16">
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
            About <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-emerald-600">Profit Sentinel</span>
          </h1>
          <p className="text-xl text-slate-400 max-w-2xl mx-auto">
            We help retailers find the profit leaks hiding in plain sight
          </p>
        </div>

        {/* Mission */}
        <section className="mb-16">
          <h2 className="text-2xl font-bold text-emerald-400 mb-4">Our Mission</h2>
          <div className="bg-white/5 rounded-2xl border border-emerald-500/20 p-6">
            <p className="text-lg text-slate-300 leading-relaxed">
              Every retail business has profit leaking out through inefficiencies they can't see.
              Dead inventory, margin erosion, stock imbalances - these problems hide in spreadsheets
              and cost businesses thousands of dollars every year.
            </p>
            <p className="text-lg text-slate-300 leading-relaxed mt-4">
              <strong className="text-white">Profit Sentinel</strong> uses advanced pattern recognition
              to analyze your inventory data and surface the specific SKUs that need attention.
              No more guessing. No more missed opportunities.
            </p>
          </div>
        </section>

        {/* How It Works */}
        <section className="mb-16">
          <h2 className="text-2xl font-bold text-emerald-400 mb-6">How It Works</h2>
          <div className="grid gap-4 md:grid-cols-3">
            <StepCard
              number="1"
              title="Upload Your Data"
              description="Export your inventory report as CSV or Excel. We support most POS system formats."
            />
            <StepCard
              number="2"
              title="AI Analysis"
              description="Our engine scans for 8 types of profit leaks: dead stock, margin issues, shrinkage patterns, and more."
            />
            <StepCard
              number="3"
              title="Get Actionable Results"
              description="See exactly which SKUs need attention, prioritized by potential dollar impact."
            />
          </div>
        </section>

        {/* Privacy First */}
        <section className="mb-16">
          <h2 className="text-2xl font-bold text-emerald-400 mb-4">Privacy First</h2>
          <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-2xl p-6">
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0">
                <ShieldIcon className="w-8 h-8 text-emerald-400" />
              </div>
              <div>
                <p className="text-lg text-slate-300 leading-relaxed">
                  <strong className="text-white">Your data stays private.</strong> All analysis
                  happens locally in your browser. Your inventory files are never uploaded
                  to our servers. We only store anonymized aggregate statistics to improve
                  our detection algorithms - never your actual SKUs or item names.
                </p>
                <Link
                  href="/privacy"
                  className="inline-flex items-center gap-1 text-emerald-400 hover:text-emerald-300 mt-4 text-sm font-medium"
                >
                  Read our full Privacy Policy
                  <ArrowRightIcon className="w-4 h-4" />
                </Link>
              </div>
            </div>
          </div>
        </section>

        {/* The Technology */}
        <section className="mb-16">
          <h2 className="text-2xl font-bold text-emerald-400 mb-4">The Technology</h2>
          <div className="bg-white/5 rounded-2xl border border-slate-700 p-6">
            <p className="text-slate-300 leading-relaxed mb-4">
              Profit Sentinel is powered by <strong className="text-white">Vector Symbolic Architecture (VSA)</strong>,
              an AI technique that encodes complex business rules into high-dimensional vectors.
              This allows us to detect subtle patterns that simple threshold-based rules miss.
            </p>
            <p className="text-slate-300 leading-relaxed">
              Our engine processes <strong className="text-white">150,000+ rows in under 60 seconds</strong>,
              making it practical for even large retail operations. Results are deterministic and
              reproducible - the same data always produces the same findings.
            </p>
          </div>
        </section>

        {/* CTA */}
        <section className="text-center">
          <div className="bg-gradient-to-r from-emerald-500/20 to-emerald-600/20 rounded-2xl border border-emerald-500/30 p-8">
            <h2 className="text-2xl font-bold text-white mb-4">
              Ready to find your hidden profit leaks?
            </h2>
            <p className="text-slate-400 mb-6">
              Free analysis. No credit card required. Results in under 2 minutes.
            </p>
            <Link
              href="/upload"
              className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-bold text-lg rounded-xl hover:from-emerald-600 hover:to-emerald-700 transition transform hover:scale-105 shadow-lg shadow-emerald-500/25"
            >
              Analyze My Inventory Free
              <ArrowRightIcon className="w-5 h-5" />
            </Link>
          </div>
        </section>
      </div>
    </div>
  )
}

function StepCard({ number, title, description }: { number: string; title: string; description: string }) {
  return (
    <div className="bg-white/5 rounded-xl border border-slate-700 p-5 relative">
      <div className="absolute -top-3 -left-3 w-8 h-8 rounded-full bg-emerald-500 text-white font-bold flex items-center justify-center text-sm">
        {number}
      </div>
      <h3 className="text-lg font-semibold text-white mb-2 mt-2">{title}</h3>
      <p className="text-sm text-slate-400">{description}</p>
    </div>
  )
}

function ShieldIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className={className}>
      <path fillRule="evenodd" d="M9.661 2.237a.531.531 0 01.678 0 11.947 11.947 0 007.078 2.749.5.5 0 01.479.425c.069.52.104 1.05.104 1.59 0 5.162-3.26 9.563-7.834 11.256a.48.48 0 01-.332 0C5.26 16.564 2 12.163 2 7c0-.538.035-1.069.104-1.589a.5.5 0 01.48-.425 11.947 11.947 0 007.077-2.75z" clipRule="evenodd" />
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
