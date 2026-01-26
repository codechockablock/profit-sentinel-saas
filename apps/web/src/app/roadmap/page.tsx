export const metadata = {
  title: 'Roadmap | Profit Sentinel',
  description: 'See what features are coming next to Profit Sentinel - integrations, automation, and more.',
}

interface RoadmapItem {
  status: 'shipped' | 'in-progress' | 'planned' | 'exploring'
  title: string
  description: string
  eta?: string
}

const roadmapItems: RoadmapItem[] = [
  // Shipped
  {
    status: 'shipped',
    title: 'Core Leak Detection Engine',
    description: '11 profit leak types: low stock, dead inventory, margin leak, negative inventory, overstock, price discrepancy, shrinkage pattern, margin erosion, zero cost anomaly, negative profit, and severe inventory deficit.',
  },
  {
    status: 'shipped',
    title: 'Large File Support',
    description: 'Process 156,000+ SKUs in seconds with sub-second analysis.',
  },
  {
    status: 'shipped',
    title: 'Deterministic Results',
    description: 'Same data always produces the same findings. Audit-ready with reproducible analysis.',
  },
  {
    status: 'shipped',
    title: 'Privacy-First Architecture',
    description: 'Files encrypted in transit and at rest. Auto-deleted within 24 hours.',
  },
  {
    status: 'shipped',
    title: 'Email Reports',
    description: 'Get your full analysis report delivered to your inbox with actionable recommendations.',
  },
  {
    status: 'shipped',
    title: 'PDF Report Generation',
    description: 'CFO-ready PDF reports with 100+ pages of detailed analysis.',
  },
  {
    status: 'shipped',
    title: 'Multi-File Vendor Correlation',
    description: 'Upload up to 200 vendor invoices. Cross-reference to find short ships & cost variances.',
    eta: 'Premium Preview',
  },

  // In Progress
  {
    status: 'in-progress',
    title: 'Cross-Report Pattern Detection',
    description: 'Identify patterns across multiple analyses to spot recurring issues.',
    eta: 'Q1 2026',
  },

  // Planned
  {
    status: 'planned',
    title: 'Automated Vendor Performance Scoring',
    description: 'Score vendors based on delivery accuracy, pricing consistency, and short-ship history.',
    eta: 'Q2 2026',
  },
  {
    status: 'planned',
    title: 'Predictive Inventory Alerts',
    description: 'AI-powered predictions for stockouts and overstock situations before they happen.',
    eta: 'Q3 2026',
  },
  {
    status: 'planned',
    title: 'API Access for Enterprise',
    description: 'REST API for integrating Profit Sentinel into your existing workflows.',
    eta: 'Q4 2026',
  },
  {
    status: 'planned',
    title: 'POS System Integrations',
    description: 'Direct connections to Square, Lightspeed, Clover, and Shopify POS for automatic data sync.',
    eta: 'Q4 2026',
  },

  // Exploring
  {
    status: 'exploring',
    title: 'Multi-Location Support',
    description: 'Analyze inventory across multiple store locations with cross-store insights.',
  },
  {
    status: 'exploring',
    title: 'Scheduled Analysis',
    description: 'Set up weekly or monthly automatic scans with email alerts for new issues.',
  },
  {
    status: 'exploring',
    title: 'Mobile App',
    description: 'Scan inventory and get alerts on the go.',
  },
]

const statusConfig = {
  shipped: {
    label: 'Shipped',
    color: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    icon: CheckIcon,
  },
  'in-progress': {
    label: 'In Progress',
    color: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
    icon: LoadingIcon,
  },
  planned: {
    label: 'Planned',
    color: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
    icon: CalendarIcon,
  },
  exploring: {
    label: 'Exploring',
    color: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
    icon: SparklesIcon,
  },
}

export default function RoadmapPage() {
  const groupedItems = {
    shipped: roadmapItems.filter((item) => item.status === 'shipped'),
    'in-progress': roadmapItems.filter((item) => item.status === 'in-progress'),
    planned: roadmapItems.filter((item) => item.status === 'planned'),
    exploring: roadmapItems.filter((item) => item.status === 'exploring'),
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-900 to-slate-800">
      <div className="max-w-4xl mx-auto px-4 py-16">
        {/* Header */}
        <div className="text-center mb-16">
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Product <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-emerald-600">Roadmap</span>
          </h1>
          <p className="text-xl text-slate-400 max-w-2xl mx-auto">
            See what we're building next. Have a feature request?{' '}
            <a
              href="/contact?type=feature_request"
              className="text-emerald-400 hover:text-emerald-300 underline"
            >
              Let us know
            </a>
          </p>
        </div>

        {/* Roadmap Sections */}
        <div className="space-y-12">
          {(Object.keys(groupedItems) as Array<keyof typeof groupedItems>).map((status) => {
            const config = statusConfig[status]
            const items = groupedItems[status]
            if (items.length === 0) return null

            return (
              <section key={status}>
                <div className="flex items-center gap-3 mb-6">
                  <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border ${config.color}`}>
                    <config.icon className="w-4 h-4" />
                    <span className="text-sm font-bold">{config.label}</span>
                  </div>
                  <div className="flex-1 h-px bg-slate-700" />
                </div>

                <div className="grid gap-4">
                  {items.map((item, index) => (
                    <RoadmapCard key={index} item={item} />
                  ))}
                </div>
              </section>
            )
          })}
        </div>

        {/* Feedback CTA */}
        <section className="mt-16">
          <div className="bg-gradient-to-r from-emerald-500/10 to-emerald-600/10 rounded-2xl border border-emerald-500/30 p-8 text-center">
            <h2 className="text-2xl font-bold text-white mb-4">
              Want to influence what we build next?
            </h2>
            <p className="text-slate-400 mb-6 max-w-xl mx-auto">
              We prioritize features based on customer feedback. Tell us what would
              make the biggest difference for your business.
            </p>
            <a
              href="/contact?type=feature_request"
              className="inline-flex items-center gap-2 px-6 py-3 bg-emerald-500/20 text-emerald-400 font-bold rounded-xl hover:bg-emerald-500/30 transition border border-emerald-500/30"
            >
              <MailIcon className="w-5 h-5" />
              Submit Feature Request
            </a>
          </div>
        </section>
      </div>
    </div>
  )
}

function RoadmapCard({ item }: { item: RoadmapItem }) {
  return (
    <div className="bg-white/5 rounded-xl border border-slate-700 p-5 hover:bg-white/[0.07] transition">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className="text-lg font-semibold text-white mb-1">{item.title}</h3>
          <p className="text-sm text-slate-400">{item.description}</p>
        </div>
        {item.eta && (
          <span className="flex-shrink-0 text-xs text-slate-500 bg-slate-800 px-2 py-1 rounded">
            {item.eta}
          </span>
        )}
      </div>
    </div>
  )
}

// Icons
function CheckIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className={className}>
      <path fillRule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clipRule="evenodd" />
    </svg>
  )
}

function LoadingIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className={className}>
      <path fillRule="evenodd" d="M15.312 11.424a5.5 5.5 0 01-9.201 2.466l-.312-.311h2.433a.75.75 0 000-1.5H3.989a.75.75 0 00-.75.75v4.242a.75.75 0 001.5 0v-2.43l.31.31a7 7 0 0011.712-3.138.75.75 0 00-1.449-.39zm1.23-3.723a.75.75 0 00.219-.53V2.929a.75.75 0 00-1.5 0V5.36l-.31-.31A7 7 0 003.239 8.188a.75.75 0 101.448.389A5.5 5.5 0 0113.89 6.11l.311.31h-2.432a.75.75 0 000 1.5h4.243a.75.75 0 00.53-.219z" clipRule="evenodd" />
    </svg>
  )
}

function CalendarIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className={className}>
      <path fillRule="evenodd" d="M5.75 2a.75.75 0 01.75.75V4h7V2.75a.75.75 0 011.5 0V4h.25A2.75 2.75 0 0118 6.75v8.5A2.75 2.75 0 0115.25 18H4.75A2.75 2.75 0 012 15.25v-8.5A2.75 2.75 0 014.75 4H5V2.75A.75.75 0 015.75 2zm-1 5.5c-.69 0-1.25.56-1.25 1.25v6.5c0 .69.56 1.25 1.25 1.25h10.5c.69 0 1.25-.56 1.25-1.25v-6.5c0-.69-.56-1.25-1.25-1.25H4.75z" clipRule="evenodd" />
    </svg>
  )
}

function SparklesIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className={className}>
      <path d="M10.868 2.884c-.321-.772-1.415-.772-1.736 0l-1.83 4.401-4.753.381c-.833.067-1.171 1.107-.536 1.651l3.62 3.102-1.106 4.637c-.194.813.691 1.456 1.405 1.02L10 15.591l4.069 2.485c.713.436 1.598-.207 1.404-1.02l-1.106-4.637 3.62-3.102c.635-.544.297-1.584-.536-1.65l-4.752-.382-1.831-4.401z" />
    </svg>
  )
}

function MailIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className={className}>
      <path d="M3 4a2 2 0 00-2 2v1.161l8.441 4.221a1.25 1.25 0 001.118 0L19 7.162V6a2 2 0 00-2-2H3z" />
      <path d="M19 8.839l-7.77 3.885a2.75 2.75 0 01-2.46 0L1 8.839V14a2 2 0 002 2h14a2 2 0 002-2V8.839z" />
    </svg>
  )
}
