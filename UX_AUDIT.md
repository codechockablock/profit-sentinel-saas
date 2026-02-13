# Profit Sentinel — Production UX Audit

**Date:** 2026-02-12
**Site:** https://www.profitsentinel.com
**Reviewed by:** Claude (Opus 4.6)

---

## Critical Issues

### 1. Two Competing Upload Flows — Confusing Entry Points
- `/diagnostic` has a drag-and-drop upload area
- `/analyze` has a separate drag-and-drop upload area with Turnstile captcha
- Both do the same thing but live at different URLs with different UX
- **Landing page "Upload P&L" button → `/diagnostic`**, but `/analyze` also exists
- Users can stumble into either path with no guidance on which is canonical

**Recommendation:** Consolidate into a single upload flow at `/analyze` (which has the captcha gate). Redirect `/diagnostic` → `/analyze` or repurpose `/diagnostic` as a results-only view.

### 2. Dashboard Link is a Dead End
- Nav bar has "Dashboard" link pointing to `/dashboard`
- `/dashboard` shows an empty state with no data, no onboarding, no explanation
- First-time visitors clicking Dashboard get nothing useful

**Recommendation:** Either hide the Dashboard link for unauthenticated/new users, or show an onboarding prompt explaining what the dashboard does and how to populate it.

---

## High Priority

### 3. Landing Page Says "3 Steps" but Shows 4
- Hero section says "Analyze your P&L in 3 simple steps"
- The steps section below actually lists 4 steps
- Inconsistency undermines trust

**Recommendation:** Update the hero copy to match the actual step count, or consolidate to 3 steps.

### 4. Landing Page Time Claim Mismatch
- Hero says "60-second P&L analysis"
- Another section says "under 2 minutes"
- Pick one consistent time claim

**Recommendation:** Standardize on a single time claim across the page.

### 5. "Get Started Free" Button Scrolls Instead of Navigating
- The hero CTA "Get Started Free" scrolls down the page to the features section
- Users expect this to start the upload flow
- The actual upload CTA ("Upload P&L") is in the nav bar

**Recommendation:** "Get Started Free" should navigate to `/analyze` (the upload flow).

### 6. About Page Lists "Exposed Margin Blind Spots for 200+ Businesses"
- The product is new — this claim may not be accurate yet
- Could damage credibility if questioned

**Recommendation:** Use honest, forward-looking language or remove the specific number until it's real.

---

## Medium Priority

### 7. Turnstile Widget Placement on /analyze
- The Turnstile captcha widget appears at the bottom of the upload card
- It shows "Success!" immediately on page load before the user does anything
- Feels out of place — should appear contextually (e.g., right before or during upload)

**Recommendation:** Move Turnstile verification to trigger on upload submit rather than on page load, or place it closer to the upload button.

### 8. /diagnostic/premium Shows Upgrade Paywall with No Context
- Visiting `/diagnostic/premium` directly shows a paywall modal
- No explanation of what premium features are or what the user gets
- No sample or preview

**Recommendation:** Add a feature comparison or value proposition before the paywall.

### 9. /repair Page Feels Orphaned
- `/repair` exists but isn't linked from main navigation
- Content is sparse — feels like an incomplete page
- No clear path to get there from the normal user flow

**Recommendation:** Either integrate repair recommendations into the main results flow or add it to navigation with proper content.

### 10. Page Titles Are Generic
- Multiple pages use "Profit Sentinel" as the sole page title
- `/analyze` should say "Upload P&L — Profit Sentinel"
- `/diagnostic` should say "Diagnostic — Profit Sentinel"
- Bad for SEO and browser tab identification

**Recommendation:** Add unique, descriptive `<title>` tags per page.

### 11. Contact Page Form Doesn't Indicate Required Fields
- `/contact` form has no asterisks or "required" indicators
- No inline validation feedback
- Submit button doesn't clearly indicate what happens next

**Recommendation:** Add required field indicators and inline validation.

---

## Low Priority

### 12. Roadmap Page Could Be More Interactive
- `/roadmap` is a static list
- No timeline, no progress indicators, no ETA ranges
- Feels like a placeholder

**Recommendation:** Add visual timeline or progress bars to make it feel alive.

### 13. Footer Links Consistency
- Footer has links to Terms and Privacy, which is good
- But some footer links open in same tab while others don't — inconsistent behavior

**Recommendation:** Standardize link behavior (all same-tab for internal, new-tab for external).

### 14. Mobile Nav Hamburger Menu
- Mobile hamburger menu works but animation is abrupt
- Menu items could use better spacing on small screens

**Recommendation:** Smooth the animation and increase tap targets.

---

## Forward-Looking: Engine 3 — Counterfactual World Model

> **Note:** This is out of scope for the current audit but critical for the upcoming upload flow consolidation.

When Engine 3 (Counterfactual World Model) ships, the `/analyze` results page will need to display **cost-of-inaction data** — i.e., "what happens if you do nothing." This is a new data dimension that doesn't have a home in the current results layout.

When consolidating the upload flow (Issue #1 above), ensure the results page architecture has room for an **Engine 3 summary line** showing counterfactual projections. This should be planned into the layout now rather than bolted on after the fact.

---

## Priority Summary

| # | Issue | Severity | Effort |
|---|-------|----------|--------|
| 1 | Two competing upload flows | Critical | Medium |
| 2 | Dashboard dead end | Critical | Low |
| 3 | "3 steps" but shows 4 | High | Low |
| 4 | Time claim mismatch | High | Low |
| 5 | "Get Started Free" scrolls instead of navigating | High | Low |
| 6 | Unverified business count claim | High | Low |
| 7 | Turnstile widget placement | Medium | Low |
| 8 | Premium paywall without context | Medium | Medium |
| 9 | Orphaned /repair page | Medium | Medium |
| 10 | Generic page titles | Medium | Low |
| 11 | Contact form UX | Medium | Low |
| 12 | Static roadmap | Low | Medium |
| 13 | Footer link behavior | Low | Low |
| 14 | Mobile nav polish | Low | Low |

---

## What's Working Well

- **Landing page design** is clean and professional
- **Upload flow on /analyze** works end-to-end with Turnstile protection
- **Security headers** are comprehensive (CSP, HSTS, X-Frame-Options, etc.)
- **About page** tells a clear story
- **Terms and Privacy pages** are thorough
- **Responsive layout** works across breakpoints
- **Cloudflare Turnstile** is live and verifying on the upload flow
