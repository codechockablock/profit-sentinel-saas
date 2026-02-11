# Dashboard Specification

> Production readiness audit — gap analysis and requirements for the main dashboard.

## Current State (as of audit)

| # | Requirement | Status | Notes |
|---|------------|--------|-------|
| 1 | Dollar amount at top showing potential recovery | PARTIAL | Shows "Dollar Impact" — needs "potential recovery identified" phrasing |
| 2 | Traffic-light status per department | NOT MET | No department-level status indicators exist |
| 3 | Finding cards ranked by dollar impact | PARTIAL | Sorted by `priority_score` (composite), not pure dollar amount |
| 4 | Finding card shows type icon + recommended action | PARTIAL | Has severity badge + description, missing type icon and inline action |
| 5 | Acknowledged/dismissed section | NOT MET | No mechanism for acknowledging findings |
| 6 | Trend indicators with sparklines | PARTIAL | Up/down arrows exist; no inline sparkline charts |
| 7 | Active prediction count on main dashboard | PARTIAL | Exists on `/dashboard/predictions` sub-page, not main view |
| 8 | No internal terminology exposed | MOSTLY MET | "Pipeline" metric leaks in one place |
| 9 | Mobile-responsive layout | PARTIAL | Uses responsive grid but no hamburger menu for sidebar on mobile |

## Required Changes

### 1. Recovery Summary Banner
**File:** `web/src/app/dashboard/page.tsx`

Replace the current "Dollar Impact" header with a recovery-focused summary:
- Display: "**$X,XXX** in potential recovery identified"
- Sum dollar amounts from all active (unacknowledged) findings
- Include subtext: "across N findings in M departments"

### 2. Department Traffic-Light Status
**New component:** `web/src/components/dashboard/DepartmentStatus.tsx`

- Row of department badges, each showing GREEN / YELLOW / RED
- GREEN: No active findings in department
- YELLOW: Findings exist, all below $500 impact
- RED: Any finding above $500 or multiple findings
- Data source: Group findings by department/category from the analysis result
- Click a department badge to filter the findings list

### 3. Dollar-Impact Sort
**File:** `web/src/app/dashboard/page.tsx`

- Change sort key from `priority_score` to estimated dollar impact
- Add a toggle: "Sort by: Dollar Impact | Priority Score | Date"
- Default to dollar impact

### 4. Enhanced Finding Cards
**File:** `web/src/components/dashboard/IssueCard.tsx` (or equivalent)

Each finding card must show:
- **Type icon** (left side): 📦 Dead Stock, 📉 Margin Erosion, 👻 Phantom Inventory, 📊 Shrinkage, ⚠️ Overstock, 🔄 Vendor Anomaly
- **Title**: One-line summary
- **Dollar amount**: Prominent, right-aligned
- **Severity badge**: Critical / High / Medium / Low (already exists)
- **Recommended action**: One-line inline text (e.g., "Review pricing on copper pipe category")
- **Acknowledge button**: Moves card to acknowledged section

### 5. Acknowledged Section
**New component:** `web/src/components/dashboard/AcknowledgedFindings.tsx`

- Collapsible section at bottom of dashboard: "Acknowledged (N)"
- When user clicks "Acknowledge" on a finding card, it moves here
- Acknowledged findings are dimmed/muted but still visible
- "Restore" button to move back to active
- Persist acknowledged state in localStorage (guest) or Supabase (authenticated)

### 6. Sparkline Trend Indicators
**New component:** `web/src/components/dashboard/Sparkline.tsx`

- Small inline SVG sparkline (last 7 data points)
- Show next to velocity, margin, and stock-level metrics
- Use the temporal data from predictions endpoint if available
- Fallback: show up/down arrow if insufficient data points

### 7. Prediction Count on Main Dashboard
**File:** `web/src/app/dashboard/page.tsx`

- Add a summary card: "N active predictions" with link to `/dashboard/predictions`
- Show the count of predictions with confidence > 0.7
- Include a mini indicator of how many are improving vs declining

### 8. Terminology Cleanup
**File:** `web/src/app/dashboard/page.tsx`

- Replace "Pipeline" metric label with "Analysis Engine" or "System Status"
- Grep for other internal terms: VSA, phasor, resonator, battery, primitive
- Ensure none appear in any user-facing string

### 9. Mobile Sidebar
**File:** `web/src/app/dashboard/layout.tsx`

- Add hamburger menu icon (visible at `md` breakpoint and below)
- Clicking opens sidebar as overlay
- Clicking outside or on a link closes it
- Use existing Tailwind responsive classes

## API Dependencies

The dashboard currently reads from the analysis result stored in React state (for guests) or fetched from Supabase (for authenticated users). The following new endpoints are needed:

| Endpoint | Purpose | Priority |
|----------|---------|----------|
| `GET /api/v1/findings` | Paginated findings with acknowledge status | HIGH |
| `GET /api/v1/dashboard` | Pre-computed dashboard summary (recovery $, dept status) | HIGH |
| `GET /api/v1/predictions` | Active predictions with confidence scores | MEDIUM |

## Implementation Priority

1. **P0 (launch blocker):** Terminology cleanup (#8), Recovery summary banner (#1)
2. **P1 (first sprint):** Department traffic-light (#2), Dollar-impact sort (#3), Enhanced finding cards (#4)
3. **P2 (second sprint):** Acknowledged section (#5), Mobile sidebar (#9), Prediction count (#7)
4. **P3 (polish):** Sparklines (#6)
