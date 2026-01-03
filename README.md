# Profit Sentinel

**Your AI Profit Sentinel – Vigilant Protection for Retail Margins**

Profit Sentinel is a powerful SaaS tool designed to uncover hidden profit leaks in your POS data while providing actionable inventory management tools. Simply upload your exports, and get instant forensic insights: unrecorded COGS, negative inventory, true margins, category/vendor breakdowns, and smart recommendations to fix and prevent issues.

Built for independent retailers (hardware, lumber, garden centers, auto parts, and more)—a lightweight, non-disruptive add-on to your existing POS system (RockSolid, Spruce, Paladin, Square, and others).

## Why Profit Sentinel?

Retail margins are thin (typically 3–5%), and silent leaks from skipped receiving, negatives, or unrecorded purchases can cost tens or hundreds of thousands annually. Most POS tools assume clean data—Profit Sentinel not only exposes the truth but helps you manage inventory smarter to protect and grow profits.

- **Forensic Leak Detection**: Find $100k+ hidden shrink others miss.
- **Smart Inventory Management**: Reorder alerts, dead stock optimization, cycle counting guidance, vendor scorecards.
- **AI-Powered Insights**: "Ask Sentinel" chat for natural language queries and recommendations.
- **Privacy-Focused**: Secure AWS infrastructure with encryption everywhere.
- **No Disruption**: Works alongside your current POS—no migration needed.

## Features

### Current
- POS/payroll export upload (CSV/Excel) → instant diagnostic reports.
- True margin correction and profit leak quantification.
- Category, vendor, and customer cross-insights.
- Multi-location roll-ups for chains.

### Coming Soon
- **Smart Reorder & Low-Stock Alerts**: Dynamic reorder points based on sales velocity and leak-adjusted stock.
- **Dead/Slow-Moving Stock Optimizer**: Identify capital-tied items with markdown/clearance recommendations.
- **Cycle Counting & Receiving Assistant**: Guided tools to fix root causes of leaks.
- **Vendor Scorecard**: Performance tracking and negotiation insights.
- **Inventory Health Scorecard**: Overall score with peer benchmarks.
- Seamless POS integrations (RockSolid, Spruce, Square).
- Predictive forecasting and scenario modeling.
- Accounting exports (corrected margins to QuickBooks/Xero).
- Training modules for staff.
- "Sentinel Council" – hierarchical AI debate for bias-free decisions.

## Data Privacy & Security

Your data is sensitive—we treat it that way.

- **Encryption**: All data at rest (S3/RDS encrypted) and in transit (HTTPS).
- **Isolation**: Multi-tenant design with row-level security—no mixing of customer data.
- **AWS Best Practices**: Private networking, least-privilege access, no public exposure.
- **Future Enhancements**: Local/on-device processing options for maximum privacy.

No data is sold or shared. Profit Sentinel guards your profits—and your trust.

## Quick Start (Local Development)

### Prerequisites
- AWS account (with infra applied)
- Docker
- Node.js (for future frontend)

### Run Backend Locally
```bash
cd backend
docker build -t profit-sentinel-api .
docker run -p 8000:8000 profit-sentinel-api