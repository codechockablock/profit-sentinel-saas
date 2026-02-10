import Link from 'next/link'

export const metadata = {
  title: 'Terms of Service | Profit Sentinel',
  description: 'Terms of Service for using Profit Sentinel inventory analysis platform.',
}

export default function TermsPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-900 to-slate-800">
      <div className="max-w-4xl mx-auto px-4 py-16">
        {/* Header */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold text-white mb-4">Terms of Service</h1>
          <p className="text-slate-400">
            Last updated: January 17, 2026
          </p>
        </div>

        {/* Content */}
        <div className="prose prose-invert prose-slate max-w-none">
          <section className="mb-10">
            <h2 className="text-2xl font-bold text-emerald-400 mb-4">1. Agreement to Terms</h2>
            <p className="text-slate-300 leading-relaxed">
              By accessing or using Profit Sentinel ("the Service"), you agree to be bound by these
              Terms of Service. If you disagree with any part of the terms, you may not access
              the Service.
            </p>
          </section>

          <section className="mb-10">
            <h2 className="text-2xl font-bold text-emerald-400 mb-4">2. Description of Service</h2>
            <p className="text-slate-300 leading-relaxed">
              Profit Sentinel provides inventory analysis tools that help retailers identify
              profit leaks in their data. The Service analyzes uploaded inventory data and
              generates reports highlighting potential issues.
            </p>
            <p className="text-slate-300 leading-relaxed mt-4">
              The Service is provided "as is" and is intended for informational purposes only.
              Analysis results are algorithmic suggestions and should be verified by qualified
              personnel before taking business action.
            </p>
          </section>

          <section className="mb-10">
            <h2 className="text-2xl font-bold text-emerald-400 mb-4">3. Your Data</h2>
            <p className="text-slate-300 leading-relaxed">
              <strong className="text-white">Privacy:</strong> Your inventory data is uploaded
              securely to encrypted storage for analysis, then automatically deleted within 24 hours.
              We only retain anonymized, aggregate statistics — your actual SKUs and item names are never stored permanently. See our{' '}
              <Link href="/privacy" className="text-emerald-400 hover:text-emerald-300">
                Privacy Policy
              </Link>{' '}
              for complete details.
            </p>
            <p className="text-slate-300 leading-relaxed mt-4">
              <strong className="text-white">Responsibility:</strong> You are responsible for
              ensuring you have the right to analyze any data you upload. Do not upload data
              containing information you do not have permission to process.
            </p>
          </section>

          <section className="mb-10">
            <h2 className="text-2xl font-bold text-emerald-400 mb-4">4. Account Terms</h2>
            <ul className="list-disc list-inside text-slate-300 space-y-2">
              <li>You must provide a valid email address to unlock full reports</li>
              <li>You are responsible for maintaining the security of your email account</li>
              <li>You must not use the Service for any illegal or unauthorized purpose</li>
              <li>You must not transmit any malware or code of a destructive nature</li>
            </ul>
          </section>

          <section className="mb-10">
            <h2 className="text-2xl font-bold text-emerald-400 mb-4">5. Acceptable Use</h2>
            <p className="text-slate-300 leading-relaxed">You agree not to:</p>
            <ul className="list-disc list-inside text-slate-300 space-y-2 mt-2">
              <li>Attempt to reverse engineer, decompile, or hack the Service</li>
              <li>Use the Service to analyze data belonging to others without permission</li>
              <li>Share or resell access to the Service without authorization</li>
              <li>Overload, flood, or spam the Service with excessive requests</li>
              <li>Use automated tools to access the Service without our consent</li>
            </ul>
          </section>

          <section className="mb-10">
            <h2 className="text-2xl font-bold text-emerald-400 mb-4">6. Intellectual Property</h2>
            <p className="text-slate-300 leading-relaxed">
              The Service, including its original content, features, and functionality, is owned
              by Profit Sentinel and is protected by international copyright, trademark, and
              other intellectual property laws.
            </p>
            <p className="text-slate-300 leading-relaxed mt-4">
              You retain ownership of any data you upload. By using the Service, you grant us
              a limited license to process your data solely for the purpose of providing
              analysis results to you.
            </p>
          </section>

          <section className="mb-10">
            <h2 className="text-2xl font-bold text-emerald-400 mb-4">7. Disclaimer of Warranties</h2>
            <p className="text-slate-300 leading-relaxed">
              THE SERVICE IS PROVIDED "AS IS" WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED.
              WE DO NOT WARRANT THAT THE SERVICE WILL BE UNINTERRUPTED, SECURE, OR ERROR-FREE.
            </p>
            <p className="text-slate-300 leading-relaxed mt-4">
              Analysis results are algorithmic suggestions based on patterns in your data.
              Results should be verified by qualified personnel. We are not responsible for
              business decisions made based on our analysis.
            </p>
          </section>

          <section className="mb-10">
            <h2 className="text-2xl font-bold text-emerald-400 mb-4">8. Limitation of Liability</h2>
            <p className="text-slate-300 leading-relaxed">
              TO THE MAXIMUM EXTENT PERMITTED BY LAW, IN NO EVENT SHALL PROFIT SENTINEL BE LIABLE
              FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES, INCLUDING
              BUT NOT LIMITED TO LOSS OF PROFITS, DATA, OR OTHER INTANGIBLE LOSSES.
            </p>
          </section>

          <section className="mb-10">
            <h2 className="text-2xl font-bold text-emerald-400 mb-4">9. Changes to Terms</h2>
            <p className="text-slate-300 leading-relaxed">
              We reserve the right to modify or replace these Terms at any time. If a revision
              is material, we will provide at least 30 days' notice prior to any new terms
              taking effect.
            </p>
          </section>

          <section className="mb-10">
            <h2 className="text-2xl font-bold text-emerald-400 mb-4">10. Contact Us</h2>
            <p className="text-slate-300 leading-relaxed">
              If you have any questions about these Terms, please contact us at:{' '}
              <a href="mailto:legal@profitsentinel.com" className="text-emerald-400 hover:text-emerald-300">
                legal@profitsentinel.com
              </a>
            </p>
          </section>
        </div>

        {/* Back Link */}
        <div className="mt-12 pt-8 border-t border-slate-800">
          <Link
            href="/"
            className="inline-flex items-center gap-2 text-slate-400 hover:text-emerald-400 transition"
          >
            <ArrowLeftIcon className="w-4 h-4" />
            Back to Home
          </Link>
        </div>
      </div>
    </div>
  )
}

function ArrowLeftIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className={className}>
      <path fillRule="evenodd" d="M17 10a.75.75 0 01-.75.75H5.612l4.158 3.96a.75.75 0 11-1.04 1.08l-5.5-5.25a.75.75 0 010-1.08l5.5-5.25a.75.75 0 111.04 1.08L5.612 9.25H16.25A.75.75 0 0117 10z" clipRule="evenodd" />
    </svg>
  )
}
