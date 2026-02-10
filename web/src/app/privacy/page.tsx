'use client'

export default function PrivacyPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 text-slate-200">
      <div className="container mx-auto px-6 py-16 max-w-4xl">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-emerald-400 mb-4">Privacy Policy</h1>
          <p className="text-slate-400">Last updated: January 2026</p>
        </div>

        {/* Content */}
        <div className="prose prose-invert prose-emerald max-w-none">
          <section className="mb-12 bg-white/5 rounded-2xl p-8">
            <h2 className="text-2xl font-bold text-emerald-400 mb-4">Our Commitment to Your Privacy</h2>
            <p className="text-slate-300 leading-relaxed">
              Profit Sentinel is built with privacy as a core principle. We process your POS data to detect
              profit leaks, then delete the raw data within 24 hours. We only retain anonymized, aggregated statistics
              to improve our service. We never sell your data. We share limited data with third-party
              services only as necessary to operate the platform (see Third-Party Services below).
            </p>
          </section>

          <section className="mb-12 bg-white/5 rounded-2xl p-8">
            <h2 className="text-2xl font-bold text-emerald-400 mb-4">What Data We Collect</h2>

            <h3 className="text-xl font-semibold text-slate-200 mt-6 mb-3">Uploaded Files</h3>
            <ul className="list-disc list-inside text-slate-300 space-y-2">
              <li>POS export files (CSV, Excel) that you upload for analysis</li>
              <li>Files are processed in-memory and stored temporarily in encrypted S3 storage</li>
              <li><strong className="text-emerald-400">Automatically deleted within 24 hours of processing</strong></li>
            </ul>

            <h3 className="text-xl font-semibold text-slate-200 mt-6 mb-3">Email Address (Optional)</h3>
            <ul className="list-disc list-inside text-slate-300 space-y-2">
              <li>Only collected if you opt-in to receive the full detailed report</li>
              <li>Used solely to send your complete analysis report with specific SKUs</li>
              <li>You can unsubscribe at any time</li>
            </ul>

            <h3 className="text-xl font-semibold text-slate-200 mt-6 mb-3">Technical Information</h3>
            <ul className="list-disc list-inside text-slate-300 space-y-2">
              <li>IP address - used for security, rate limiting, and fraud prevention</li>
              <li>User agent (browser/device info) - used for compatibility and debugging</li>
              <li>This data is retained for 7 days for security purposes, then deleted</li>
            </ul>

            <h3 className="text-xl font-semibold text-slate-200 mt-6 mb-3">Anonymized Analytics</h3>
            <ul className="list-disc list-inside text-slate-300 space-y-2">
              <li>Aggregate statistics (leak counts, averages) with no PII</li>
              <li>Used to improve detection algorithms</li>
              <li>Cannot be linked back to your business</li>
            </ul>
          </section>

          <section className="mb-12 bg-white/5 rounded-2xl p-8">
            <h2 className="text-2xl font-bold text-emerald-400 mb-4">Preview vs. Full Report</h2>
            <p className="text-slate-300 mb-6">
              We provide two levels of analysis results so you can verify findings before opting in:
            </p>

            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700">
                <h3 className="font-bold text-emerald-400 mb-3 flex items-center gap-2">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
                    <path d="M10 12.5a2.5 2.5 0 100-5 2.5 2.5 0 000 5z" />
                    <path fillRule="evenodd" d="M.664 10.59a1.651 1.651 0 010-1.186A10.004 10.004 0 0110 3c4.257 0 7.893 2.66 9.336 6.41.147.381.146.804 0 1.186A10.004 10.004 0 0110 17c-4.257 0-7.893-2.66-9.336-6.41zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clipRule="evenodd" />
                  </svg>
                  Preview Report (On-Screen)
                </h3>
                <ul className="text-sm text-slate-400 space-y-2">
                  <li>Shows your real SKU names and product details so you can verify findings against your actual data</li>
                  <li>Displays leak categories, severity levels, and estimated dollar impact</li>
                  <li>Top items shown per leak type (not the full inventory)</li>
                  <li>No email required</li>
                  <li>Preview data exists only in your browser session and is not stored on our servers</li>
                </ul>
              </div>

              <div className="bg-emerald-900/30 rounded-xl p-6 border border-emerald-500/30">
                <h3 className="font-bold text-emerald-400 mb-3 flex items-center gap-2">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
                    <path fillRule="evenodd" d="M10 1a4.5 4.5 0 00-4.5 4.5V9H5a2 2 0 00-2 2v6a2 2 0 002 2h10a2 2 0 002-2v-6a2 2 0 00-2-2h-.5V5.5A4.5 4.5 0 0010 1zm3 8V5.5a3 3 0 10-6 0V9h6z" clipRule="evenodd" />
                  </svg>
                  Full Report (Email Delivery)
                </h3>
                <ul className="text-sm text-slate-400 space-y-2">
                  <li>Complete analysis across all SKUs with full product names</li>
                  <li>Specific actionable recommendations per leak type</li>
                  <li>Detailed breakdown by category with financial impact</li>
                  <li>Delivered as a PDF attachment to your email</li>
                  <li>Requires email opt-in with GDPR consent</li>
                  <li>Source files deleted within 24 hours</li>
                </ul>
              </div>
            </div>

            <p className="text-slate-400 mt-6 text-sm">
              The on-screen preview shows real SKU names so you can verify the analysis is accurate.
              The full report via email includes the complete inventory analysis across all flagged items.
              After report delivery, source files are deleted and only anonymized aggregate statistics are retained.
            </p>
          </section>

          <section className="mb-12 bg-white/5 rounded-2xl p-8">
            <h2 className="text-2xl font-bold text-emerald-400 mb-4">What We Do NOT Collect</h2>
            <ul className="list-disc list-inside text-slate-300 space-y-2">
              <li>Customer names, addresses, or contact information from your files</li>
              <li>Payment or credit card information</li>
              <li>Social security numbers or government IDs</li>
              <li>Any data you don't explicitly upload</li>
            </ul>
          </section>

          <section className="mb-12 bg-white/5 rounded-2xl p-8">
            <h2 className="text-2xl font-bold text-emerald-400 mb-4">How We Protect Your Data</h2>

            <div className="grid md:grid-cols-2 gap-6 mt-6">
              <div className="bg-slate-800/50 rounded-xl p-6">
                <h3 className="font-bold text-emerald-400 mb-2">Encryption in Transit</h3>
                <p className="text-sm text-slate-400">
                  All data transferred via HTTPS/TLS 1.2+. Your files are encrypted from your browser to our servers.
                </p>
              </div>

              <div className="bg-slate-800/50 rounded-xl p-6">
                <h3 className="font-bold text-emerald-400 mb-2">Encryption at Rest</h3>
                <p className="text-sm text-slate-400">
                  Files stored with AES-256 server-side encryption in AWS S3. Even if accessed, data is unreadable.
                </p>
              </div>

              <div className="bg-slate-800/50 rounded-xl p-6">
                <h3 className="font-bold text-emerald-400 mb-2">Auto-Deletion</h3>
                <p className="text-sm text-slate-400">
                  Raw files are automatically deleted within 24 hours of analysis. No manual intervention needed.
                </p>
              </div>

              <div className="bg-slate-800/50 rounded-xl p-6">
                <h3 className="font-bold text-emerald-400 mb-2">PII Stripping</h3>
                <p className="text-sm text-slate-400">
                  Before any analytics are stored, we automatically detect and remove personal information.
                </p>
              </div>
            </div>
          </section>

          <section className="mb-12 bg-white/5 rounded-2xl p-8">
            <h2 className="text-2xl font-bold text-emerald-400 mb-4">Your Rights (GDPR/CCPA)</h2>
            <p className="text-slate-300 mb-4">
              Under GDPR (EU) and CCPA (California), you have the following rights:
            </p>
            <ul className="list-disc list-inside text-slate-300 space-y-2">
              <li><strong>Right to Access:</strong> Request a copy of any data we hold about you</li>
              <li><strong>Right to Deletion:</strong> Request deletion of your data at any time</li>
              <li><strong>Right to Opt-Out:</strong> Decline email communications at any time</li>
              <li><strong>Right to Portability:</strong> Receive your data in a machine-readable format</li>
              <li><strong>Right to Correction:</strong> Request correction of inaccurate data</li>
            </ul>
            <p className="text-slate-400 mt-4">
              To exercise any of these rights, contact us at{' '}
              <a href="mailto:privacy@profitsentinel.com" className="text-emerald-400 hover:underline">
                privacy@profitsentinel.com
              </a>
            </p>
          </section>

          <section className="mb-12 bg-white/5 rounded-2xl p-8">
            <h2 className="text-2xl font-bold text-emerald-400 mb-4">Email Communications</h2>
            <p className="text-slate-300 mb-4">
              We follow CAN-SPAM, GDPR, and CCPA requirements for email:
            </p>
            <ul className="list-disc list-inside text-slate-300 space-y-2">
              <li>We only send emails if you explicitly opt-in</li>
              <li>Every email contains an unsubscribe link</li>
              <li>We honor unsubscribe requests within 24 hours</li>
              <li>We never sell or share your email with third parties</li>
            </ul>
          </section>

          <section className="mb-12 bg-white/5 rounded-2xl p-8">
            <h2 className="text-2xl font-bold text-emerald-400 mb-4">Third-Party Services</h2>
            <p className="text-slate-300 mb-4">
              We use the following services to operate Profit Sentinel:
            </p>

            <div className="space-y-4">
              <div className="border-l-4 border-emerald-500 pl-4">
                <h3 className="font-bold text-slate-200">Amazon Web Services (AWS)</h3>
                <p className="text-sm text-slate-400">Secure file storage and computing infrastructure</p>
              </div>

              <div className="border-l-4 border-emerald-500 pl-4">
                <h3 className="font-bold text-slate-200">Supabase</h3>
                <p className="text-sm text-slate-400">Authentication and database (anonymized analytics only)</p>
              </div>

              <div className="border-l-4 border-emerald-500 pl-4">
                <h3 className="font-bold text-slate-200">Resend / SendGrid</h3>
                <p className="text-sm text-slate-400">Email delivery for analysis reports (if opted-in)</p>
              </div>

              <div className="border-l-4 border-emerald-500 pl-4">
                <h3 className="font-bold text-slate-200">Vercel</h3>
                <p className="text-sm text-slate-400">Website hosting and deployment</p>
              </div>

              <div className="border-l-4 border-emerald-500 pl-4">
                <h3 className="font-bold text-slate-200">xAI</h3>
                <p className="text-sm text-slate-400">AI-powered column mapping to identify data fields in your uploads (receives column headers and a small data sample)</p>
              </div>

              <div className="border-l-4 border-emerald-500 pl-4">
                <h3 className="font-bold text-slate-200">Anthropic</h3>
                <p className="text-sm text-slate-400">AI-powered column mapping fallback (receives column headers and a small data sample when primary service is unavailable)</p>
              </div>
            </div>
          </section>

          <section className="mb-12 bg-white/5 rounded-2xl p-8">
            <h2 className="text-2xl font-bold text-emerald-400 mb-4">Cookies & Tracking</h2>
            <p className="text-slate-300">
              We use minimal, essential cookies only:
            </p>
            <ul className="list-disc list-inside text-slate-300 space-y-2 mt-4">
              <li><strong>Session cookies:</strong> Keep you logged in during your visit</li>
              <li><strong>Preference cookies:</strong> Remember your theme preference (dark/light)</li>
            </ul>
            <p className="text-slate-400 mt-4">
              We do NOT use advertising cookies or third-party tracking pixels.
            </p>
          </section>

          <section className="mb-12 bg-white/5 rounded-2xl p-8">
            <h2 className="text-2xl font-bold text-emerald-400 mb-4">Data Retention</h2>

            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead>
                  <tr className="border-b border-slate-700">
                    <th className="py-3 px-4 text-emerald-400">Data Type</th>
                    <th className="py-3 px-4 text-emerald-400">Retention Period</th>
                  </tr>
                </thead>
                <tbody className="text-slate-300">
                  <tr className="border-b border-slate-800">
                    <td className="py-3 px-4">Uploaded files</td>
                    <td className="py-3 px-4">Deleted within 24 hours of processing (typically within minutes)</td>
                  </tr>
                  <tr className="border-b border-slate-800">
                    <td className="py-3 px-4">Email address (if opted-in)</td>
                    <td className="py-3 px-4">Until you unsubscribe or request deletion</td>
                  </tr>
                  <tr className="border-b border-slate-800">
                    <td className="py-3 px-4">Anonymized analytics</td>
                    <td className="py-3 px-4">Indefinitely (no PII)</td>
                  </tr>
                  <tr className="border-b border-slate-800">
                    <td className="py-3 px-4">Session data</td>
                    <td className="py-3 px-4">24 hours</td>
                  </tr>
                  <tr>
                    <td className="py-3 px-4">Preview/teaser report data</td>
                    <td className="py-3 px-4">Browser session only (not stored on servers)</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>

          <section className="mb-12 bg-white/5 rounded-2xl p-8">
            <h2 className="text-2xl font-bold text-emerald-400 mb-4">Contact Us</h2>
            <p className="text-slate-300 mb-4">
              For privacy-related questions or to exercise your rights:
            </p>
            <div className="bg-slate-800/50 rounded-xl p-6">
              <p className="text-slate-300">
                <strong className="text-emerald-400">Email:</strong>{' '}
                <a href="mailto:privacy@profitsentinel.com" className="text-emerald-400 hover:underline">
                  privacy@profitsentinel.com
                </a>
              </p>
              <p className="text-slate-300 mt-2">
                <strong className="text-emerald-400">Response Time:</strong> Within 48 hours
              </p>
            </div>
          </section>

          <section className="bg-emerald-500/10 border border-emerald-500/30 rounded-2xl p-8">
            <h2 className="text-2xl font-bold text-emerald-400 mb-4">Changes to This Policy</h2>
            <p className="text-slate-300">
              We may update this privacy policy from time to time. We will notify you of any material
              changes by posting the new policy on this page and updating the "Last updated" date.
              We encourage you to review this policy periodically.
            </p>
          </section>
        </div>

        {/* Back Link */}
        <div className="text-center mt-12">
          <a
            href="/diagnostic"
            className="inline-flex items-center gap-2 text-emerald-400 hover:text-emerald-300 transition"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
              <path fillRule="evenodd" d="M17 10a.75.75 0 01-.75.75H5.612l4.158 3.96a.75.75 0 11-1.04 1.08l-5.5-5.25a.75.75 0 010-1.08l5.5-5.25a.75.75 0 111.04 1.08L5.612 9.25H16.25A.75.75 0 0117 10z" clipRule="evenodd" />
            </svg>
            Back to Diagnostic
          </a>
        </div>
      </div>
    </div>
  )
}
