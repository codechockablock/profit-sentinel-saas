# Settings Page Specification

> Spec for the settings page that exposes configurable dead stock thresholds and per-category overrides.

## Problem

The dead stock detection engine (`world_model/config.py`) supports configurable thresholds with 4 tiers (Watchlist → Attention → Action Required → Write-off), per-category overrides, and capital threshold filtering. None of this is exposed in the UI. Users cannot customize detection sensitivity for their business.

## Current State

- No settings page exists at `/dashboard/settings`
- A sidebar link for "Settings" exists in the dashboard layout but leads to a placeholder
- All thresholds are hardcoded defaults (60/120/180/360 days)

## Settings Sections

### Section 1: Detection Thresholds (Global)

Headline: "When should we alert you about dead stock?"

| Field | Default | Description | Input Type |
|-------|---------|-------------|------------|
| Watchlist | 60 days | Items enter the watchlist | Number input + slider |
| Needs Attention | 120 days | Escalated — review recommended | Number input + slider |
| Action Required | 180 days | Clearance or transfer recommended | Number input + slider |
| Write-off Candidate | 360 days | Consider writing off | Number input + slider |

**Validation rules** (enforced client-side and server-side):
- Each threshold must be greater than the previous
- Minimum value: 7 days (Watchlist)
- Maximum value: 730 days (Write-off)

Visual: Horizontal timeline bar showing the 4 thresholds with colored segments (green → yellow → orange → red).

### Section 2: Category Overrides

Headline: "Different rules for different departments"

- Table showing category name + override thresholds
- "Add Override" button opens a modal:
  - Category name (text input or dropdown from previously seen categories)
  - 4 threshold fields (same as global, pre-filled with global values)
- Each row has an "Edit" and "Remove" button
- Pre-populated from preset if user selected one during onboarding

Example presets:

| Preset | Category | Watchlist | Attention | Action | Write-off |
|--------|----------|-----------|-----------|--------|-----------|
| Hardware Store | Seasonal | 30 | 60 | 90 | 180 |
| Hardware Store | Commercial Hardware | 90 | 180 | 270 | 450 |
| Hardware Store | Fasteners | 90 | 180 | 270 | 540 |
| Garden Center | Live Plants | 14 | 30 | 45 | 90 |
| Garden Center | Hard Goods | 45 | 90 | 150 | 300 |

### Section 3: Capital Threshold

Headline: "Minimum dollar amount to alert"

- Single numeric input: "Only alert when capital at risk exceeds: $___"
- Default: $0 (alert on everything)
- Subtext: "Set this higher to focus on bigger-dollar items and reduce noise"

### Section 4: Velocity Threshold

Headline: "What counts as healthy velocity?"

- Single numeric input: "Minimum weekly units sold to consider healthy: ___"
- Default: 0.5 units/week
- Subtext: "Items below this velocity are flagged for review"

### Section 5: Preset Quick-Select

- Dropdown or card row: "Quick setup for your store type"
- Selecting a preset populates all fields (global + category overrides)
- Shows a diff if current values differ from preset
- "Reset to Preset" button

## API Integration

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `GET /api/v1/config` | GET | Load current user config |
| `PUT /api/v1/config` | PUT | Save updated config |

### GET /api/v1/config Response
```json
{
  "preset": "hardware_store",
  "global_thresholds": {
    "watchlist_days": 60,
    "attention_days": 120,
    "action_days": 180,
    "writeoff_days": 360
  },
  "category_overrides": {
    "Seasonal": {
      "watchlist_days": 30,
      "attention_days": 60,
      "action_days": 90,
      "writeoff_days": 180
    }
  },
  "min_capital_threshold": 0.0,
  "min_healthy_velocity": 0.5
}
```

## Frontend Files

| File | Purpose |
|------|---------|
| `web/src/app/dashboard/settings/page.tsx` | Settings page |
| `web/src/components/settings/ThresholdSlider.tsx` | Threshold input with validation |
| `web/src/components/settings/CategoryOverrideTable.tsx` | Category override CRUD |
| `web/src/components/settings/ThresholdTimeline.tsx` | Visual timeline bar |
| `web/src/components/settings/PresetSelector.tsx` | Preset quick-select |

## Implementation Priority

- **P0:** Global threshold editing + save (Section 1)
- **P1:** Category overrides (Section 2) + Capital threshold (Section 3)
- **P2:** Preset quick-select (Section 5) + Velocity threshold (Section 4)
- **P3:** Visual timeline bar, diff view
