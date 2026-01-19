import { ContactForm } from '@/components/contact-form'

export const metadata = {
  title: 'Contact Us | Profit Sentinel',
  description: 'Get in touch with the Profit Sentinel team for support, feature requests, or feedback.',
}

export default function ContactPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-900 to-slate-800">
      <div className="max-w-2xl mx-auto px-4 py-16">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Get in <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-emerald-600">Touch</span>
          </h1>
          <p className="text-xl text-slate-400 max-w-lg mx-auto">
            Have a question, feature idea, or need help? We'd love to hear from you.
          </p>
        </div>

        {/* Form */}
        <div className="bg-white/5 border border-slate-700 rounded-2xl p-8">
          <ContactForm />
        </div>

        {/* Alternative Contact */}
        <div className="mt-8 text-center text-sm text-slate-500">
          <p>
            You can also reach us directly at{' '}
            <a href="mailto:support@profitsentinel.com" className="text-emerald-400 hover:text-emerald-300 underline">
              support@profitsentinel.com
            </a>
          </p>
        </div>
      </div>
    </div>
  )
}
