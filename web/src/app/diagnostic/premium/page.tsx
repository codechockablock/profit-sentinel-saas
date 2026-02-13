import type { Metadata } from 'next';
import PremiumDiagnosticClient from './client';

export const metadata: Metadata = {
  title: 'Premium Diagnostic | Profit Sentinel',
  description: 'Multi-file vendor correlation - cross-reference invoices to find short ships and cost variances.',
};

export default function PremiumDiagnosticPage() {
  return <PremiumDiagnosticClient />;
}
