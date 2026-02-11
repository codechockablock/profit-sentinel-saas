# Onboarding Flow Specification

> Spec for the first-run onboarding experience that configures the analysis engine for the user's store type.

## Problem

Currently, new users land directly on the upload page with no context about what Profit Sentinel does or how to configure it for their business. The dead stock detection engine uses configurable thresholds with industry-specific presets, but there's no UI to select them.

## Flow

### Screen 1: Welcome
- Headline: "Let's set up Profit Sentinel for your store"
- Subtext: "This takes about 30 seconds and helps us give you better results"
- CTA: "Get Started"

### Screen 2: Store Type Selection
- Headline: "What kind of store do you run?"
- Cards (one per preset):

| Card | Preset | Description |
|------|--------|-------------|
| 🔧 Hardware Store | `hardware_store` | General hardware, tools, fasteners, plumbing, electrical. Seasonal items tracked separately. |
| 🌿 Garden Center | `garden_center` | Plants, soil, outdoor living. Heavy seasonal variation with shorter dead stock windows. |
| 🪵 Lumber Yard | `lumber_yard` | Building materials, lumber, commercial hardware. Longer acceptable stock holding periods. |
| 🏪 Convenience Store | `convenience_store` | Fast-moving consumer goods. Short dead stock windows, high velocity expectations. |
| ⚙️ Custom | (manual) | "I'll configure thresholds myself" — skips to Settings page after upload. |

- Each card shows the key thresholds:
  - "Dead stock alert after: N days"
  - "Watchlist after: N days"

### Screen 3: Multi-Store Setup (Optional)
- Headline: "Do you have multiple locations?"
- Options:
  - "Just one store" → skip
  - "Multiple stores" → enter store names/IDs (used for transfer matching)
- Subtext: "Multi-store enables cross-location transfer recommendations"

### Screen 4: Upload
- Redirect to existing `/diagnostic` upload flow
- Pre-configured with selected preset

## Data Flow

1. User selects store type → frontend stores preset name in React state
2. On analysis, include `preset: "hardware_store"` in the POST to `/analysis/analyze`
3. Backend creates `DeadStockConfig` from preset and passes to pipeline
4. Config persisted to Supabase for authenticated users (new `user_preferences` table)

## API Changes

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `GET /api/v1/config` | GET | Fetch user's current config (or default) |
| `PUT /api/v1/config` | PUT | Save user's config preset + overrides |

### PUT /api/v1/config Request Body
```json
{
  "preset": "hardware_store",
  "overrides": {
    "category_overrides": {
      "Seasonal": {
        "watchlist_days": 30,
        "attention_days": 60
      }
    }
  }
}
```

## Frontend Files to Create

| File | Purpose |
|------|---------|
| `web/src/app/onboarding/page.tsx` | Main onboarding page with step wizard |
| `web/src/components/onboarding/StoreTypeCard.tsx` | Selectable store type card |
| `web/src/components/onboarding/MultiStoreSetup.tsx` | Optional multi-store input |
| `web/src/lib/config-presets.ts` | Frontend mirror of Python ConfigPresets data |

## Implementation Priority

- **P0:** Store type selection (Screen 2) + pass preset to analysis
- **P1:** Welcome screen (Screen 1) + redirect logic for first-time users
- **P2:** Multi-store setup (Screen 3)
- **P3:** Custom threshold configuration (links to Settings page)
