# Fixes Applied During Production Audit

**Date:** 2026-02-11

---

## 1. Removed Unverified "71% Shrinkage Reduction" Claim

**File:** `web/src/app/page.tsx` (line 33)
**File:** `web/src/app/layout.tsx` (line 15)

**Before:**
> "Our proprietary AI engine analyzes 156,000+ SKUs in seconds, detecting 11 types of profit leaks that humans miss. Retailers see 71% average shrinkage reduction."

**After:**
> "Our deterministic analysis engine detects 11 types of profit leaks that humans miss. Built in Rust for speed — 36,000 SKUs analyzed in under 3 seconds."

**Reason:** The "71% average shrinkage reduction" figure has no supporting data in the codebase — no test results, no customer data, no benchmarks. The "156,000+ SKUs" claim was an unvalidated extrapolation from a 36K-row benchmark. Replaced with the verified 36K/3s E2E benchmark.

---

## 2. Removed "VSA" Internal Terminology from About Page

**File:** `web/src/app/about/page.tsx` (line 93-94)

**Before:**
> "...built in Rust that uses Vector Symbolic Architecture (VSA) and rule-based pattern recognition to find profit leaks."

**After:**
> "...built in Rust that uses advanced pattern recognition to find profit leaks."

**Reason:** "Vector Symbolic Architecture (VSA)" is internal technical terminology that should never appear in customer-facing copy. Customers should see: "pattern recognition", "analysis", "detection".

---

## 3. Replaced "11-Primitive" with Customer-Friendly Term

**File:** `web/src/components/diagnostic/AnalysisDashboard.tsx` (line 369)

**Before:**
> "Full 11-Primitive Analysis"

**After:**
> "Full 11-Type Profit Leak Analysis"

**Reason:** "Primitive" is internal VSA terminology. Customers understand "type" or "category".

---

## 4. Fixed POS Integrations Roadmap Status (shipped → in-progress)

**File:** `web/src/app/roadmap/page.tsx` (line 106-108)

**Before:** Status: `shipped`
> "Direct connections to Square, Lightspeed, Clover, and Shopify POS with OAuth2 sync and connection lifecycle management."

**After:** Status: `in-progress`, eta: "Coming Soon"
> "Connect to Square, Lightspeed, Clover, and Shopify POS for automatic inventory sync."

**Reason:** Code audit reveals POS integrations are **stubbed** — the UI exists but `sync` always returns 0 rows, there is no actual OAuth2 implementation, and no API client code for any POS system exists. The `InMemoryPosConnectionStore` creates in-memory records only.

---

## 5. Fixed Multi-File Vendor Correlation Roadmap Status (shipped → in-progress)

**File:** `web/src/app/roadmap/page.tsx` (line 51-54)

**Before:** Status: `shipped`
> "Upload up to 200 vendor invoices. Cross-reference to find short ships & cost variances."

**After:** Status: `in-progress`, eta: "Coming Soon"
> "Upload vendor invoices and cross-reference to find short ships & cost variances."

**Reason:** No cross-file correlation logic exists. The system processes individual CSV files but does not correlate across multiple vendor invoices. The "200 vendor invoices" claim was fabricated.

---

## 6. Softened "100+ Pages" PDF Report Claim

**File:** `web/src/app/roadmap/page.tsx` (line 48)

**Before:**
> "CFO-ready PDF reports with 100+ pages of detailed analysis."

**After:**
> "CFO-ready PDF reports with detailed analysis, financial impact, and prioritized action items."

**Reason:** While the PDF generation code can produce multi-page reports, no benchmark validates that reports reach 100+ pages. Page count depends on data volume.

---

## 7. Fixed "156,000+ SKUs" Claim on Roadmap

**File:** `web/src/app/roadmap/page.tsx` (line 28)

**Before:**
> "Process 156,000+ SKUs in seconds with sub-second analysis."

**After:**
> "Process large inventory files with the Rust-powered analysis engine. 36K SKUs in under 3 seconds."

**Reason:** The 156K claim is an unvalidated extrapolation. The verified benchmark is 36K rows in ~3s E2E.

---

## 8. Renamed "Symbolic Reasoning & Proof Trees" on Roadmap

**File:** `web/src/app/roadmap/page.tsx` (line 67)

**Before:**
> "Symbolic Reasoning & Proof Trees"

**After:**
> "Transparent Explanations"

**Reason:** "Symbolic reasoning" and "proof trees" are internal technical terminology. Customers should see plain language describing the capability.

---

## 9. Updated Meta Description

**File:** `web/src/app/layout.tsx` (line 15)

**Before:**
> "Proprietary AI engine analyzes 156,000+ SKUs in seconds, detecting 11 types of profit leaks. 71% average shrinkage reduction. Free analysis, no credit card required."

**After:**
> "Deterministic analysis engine detects 11 types of profit leaks in your inventory. 36K SKUs in under 3 seconds. Free analysis, no credit card required."

**Reason:** Same as fix #1 — removed unverified claims from meta description.

---

## Summary

| # | File | Type | Severity |
|---|------|------|----------|
| 1 | page.tsx, layout.tsx | Unverified claim removed | HIGH |
| 2 | about/page.tsx | Internal terminology removed | HIGH |
| 3 | AnalysisDashboard.tsx | Internal terminology replaced | HIGH |
| 4 | roadmap/page.tsx | False "shipped" status corrected | HIGH |
| 5 | roadmap/page.tsx | False "shipped" status corrected | HIGH |
| 6 | roadmap/page.tsx | Unvalidated claim softened | MEDIUM |
| 7 | roadmap/page.tsx | Unvalidated claim fixed | MEDIUM |
| 8 | roadmap/page.tsx | Internal terminology replaced | MEDIUM |
| 9 | layout.tsx | Meta description updated | MEDIUM |
