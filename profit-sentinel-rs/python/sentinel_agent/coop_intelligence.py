"""Co-op Intelligence Engine.

Detects patronage leakage, calculates consolidation opportunities,
and tracks tier progress for Do It Best / Ace / Orgill members.

Every opportunity is quantified in dollars. "Shift $18,400 to co-op
warehouse, gain $2,412/year" — not "consider consolidating vendors."
"""

from __future__ import annotations

from collections import defaultdict

from .coop_models import (
    PATRONAGE_RATES,
    WAREHOUSE_CASH_DISCOUNT,
    ConsolidationOpportunity,
    CoopAffiliation,
    CoopAlert,
    CoopAlertType,
    CoopType,
    PatronageCategory,
    PatronageLeakage,
    VendorPurchase,
)

# ---------------------------------------------------------------------------
# Known co-op vendor prefixes (stub — production would use real catalog)
# ---------------------------------------------------------------------------

# SKU prefixes that are Do It Best warehouse items
_COOP_WAREHOUSE_PREFIXES: set[str] = {
    "DIB",
    "WHS",
    "HRD",
    "FLR",
    "TLS",
    "PLB",
    "ELC",
    "PNT",
    "SSN",
}

# Known non-co-op (outside buy) vendor prefixes
_NON_COOP_PREFIXES: set[str] = {
    "DMG",
    "EXT",
    "LOC",
    "OEM",
}

# Categories where co-op warehouse equivalents typically exist
_COOP_EQUIVALENT_CATEGORIES: set[str] = {
    "Paint",
    "Electrical",
    "Plumbing",
    "Hand Tools",
    "Fasteners",
    "General Hardware",
    "Seasonal",
    "Power Tools",
    "Lawn & Garden",
}


class CoopIntelligence:
    """Analyze purchasing patterns for co-op optimization.

    Usage:
        coop = CoopIntelligence(
            affiliation=CoopAffiliation(store_id="store-7", coop_type=CoopType.DO_IT_BEST),
        )
        purchases = [VendorPurchase(...), ...]
        leakages = coop.detect_patronage_leakage(purchases)
        consolidations = coop.find_consolidation_opportunities(purchases)
    """

    def __init__(self, affiliation: CoopAffiliation):
        self.affiliation = affiliation

    # -----------------------------------------------------------------
    # Patronage Leakage Detection
    # -----------------------------------------------------------------

    def detect_patronage_leakage(
        self,
        purchases: list[VendorPurchase],
    ) -> list[PatronageLeakage]:
        """Find purchases from non-co-op vendors where co-op alternatives exist.

        For each non-co-op vendor + category combination, calculates:
        - Annual spend going to non-co-op sources
        - Rebate rate differential (co-op warehouse vs current)
        - Annual dollar leakage = spend × rate differential

        Args:
            purchases: List of vendor purchase records.

        Returns:
            List of PatronageLeakage opportunities, sorted by leakage descending.
        """
        # Group non-co-op purchases by vendor+category
        leakage_map: dict[tuple[str, str], list[VendorPurchase]] = defaultdict(list)

        for purchase in purchases:
            if self._is_non_coop_purchase(purchase):
                key = (purchase.vendor_name, purchase.category)
                leakage_map[key].append(purchase)

        leakages: list[PatronageLeakage] = []
        for (vendor_name, category), vendor_purchases in leakage_map.items():
            total_spend = sum(p.total_cost for p in vendor_purchases)
            coop_rate = self._get_coop_rate(category)
            has_equivalent = category in _COOP_EQUIVALENT_CATEGORIES

            annual_leakage = total_spend * coop_rate if has_equivalent else 0.0

            leakages.append(
                PatronageLeakage(
                    store_id=self.affiliation.store_id,
                    vendor_name=vendor_name,
                    category=category,
                    non_coop_spend=total_spend,
                    coop_equivalent_available=has_equivalent,
                    current_rebate_rate=0.0,
                    coop_rebate_rate=coop_rate,
                    annual_leakage=annual_leakage,
                    affected_skus=[p.sku_id for p in vendor_purchases],
                )
            )

        # Sort by annual leakage descending
        leakages.sort(key=lambda x: x.annual_leakage, reverse=True)
        return leakages

    def _is_non_coop_purchase(self, purchase: VendorPurchase) -> bool:
        """Determine if a purchase is from a non-co-op vendor."""
        if purchase.is_coop_available:
            return False  # Already buying through co-op

        prefix = purchase.sku_id.split("-")[0] if "-" in purchase.sku_id else ""
        if prefix in _NON_COOP_PREFIXES:
            return True
        if prefix in _COOP_WAREHOUSE_PREFIXES:
            return False

        # Default: if vendor is not in co-op catalog, it's non-co-op
        return not purchase.is_coop_available

    def _get_coop_rate(self, category: str) -> float:
        """Get the appropriate co-op patronage rate for a category."""
        if self.affiliation.coop_type == CoopType.DO_IT_BEST:
            if category == "Lumber":
                return PATRONAGE_RATES[PatronageCategory.LUMBER]
            # Default to regular warehouse rate for most categories
            return PATRONAGE_RATES[PatronageCategory.REGULAR_WAREHOUSE]
        elif self.affiliation.coop_type == CoopType.ACE:
            # Ace uses a different rebate structure (simplified)
            return 0.05  # ~5% average Ace rebate
        else:
            # Orgill (simplified)
            return 0.03  # ~3% average Orgill rebate

    # -----------------------------------------------------------------
    # Vendor Consolidation Opportunities
    # -----------------------------------------------------------------

    def find_consolidation_opportunities(
        self,
        purchases: list[VendorPurchase],
        min_vendor_count: int = 2,
        min_category_spend: float = 1000.0,
    ) -> list[ConsolidationOpportunity]:
        """Find categories where multiple vendors could be consolidated.

        When a store buys the same category from 3+ vendors, there's
        usually opportunity to consolidate to co-op warehouse and earn
        patronage + cash discount.

        Args:
            purchases: All purchase records.
            min_vendor_count: Minimum vendors in a category to flag.
            min_category_spend: Minimum annual spend to flag.

        Returns:
            List of ConsolidationOpportunity, sorted by benefit descending.
        """
        # Group by category
        category_purchases: dict[str, list[VendorPurchase]] = defaultdict(list)
        for p in purchases:
            category_purchases[p.category].append(p)

        opportunities: list[ConsolidationOpportunity] = []

        for category, cat_purchases in category_purchases.items():
            # Count unique vendors
            vendors = list({p.vendor_name for p in cat_purchases})
            if len(vendors) < min_vendor_count:
                continue

            total_spend = sum(p.total_cost for p in cat_purchases)
            if total_spend < min_category_spend:
                continue

            # Calculate shiftable spend (non-co-op purchases)
            non_coop_spend = sum(
                p.total_cost for p in cat_purchases if self._is_non_coop_purchase(p)
            )

            coop_rate = self._get_coop_rate(category)
            cash_discount = WAREHOUSE_CASH_DISCOUNT

            # Annual benefit = shiftable_spend × (patronage_rate + cash_discount)
            annual_benefit = non_coop_spend * (coop_rate + cash_discount)

            recommendation = self._build_consolidation_recommendation(
                category,
                vendors,
                non_coop_spend,
                annual_benefit,
            )

            opportunities.append(
                ConsolidationOpportunity(
                    store_id=self.affiliation.store_id,
                    category=category,
                    current_vendor_count=len(vendors),
                    vendors=vendors,
                    total_category_spend=total_spend,
                    shiftable_spend=non_coop_spend,
                    coop_rebate_rate=coop_rate,
                    cash_discount_rate=cash_discount,
                    annual_benefit=annual_benefit,
                    recommendation=recommendation,
                )
            )

        # Sort by annual benefit descending
        opportunities.sort(key=lambda x: x.annual_benefit, reverse=True)
        return opportunities

    def _build_consolidation_recommendation(
        self,
        category: str,
        vendors: list[str],
        shiftable_spend: float,
        annual_benefit: float,
    ) -> str:
        """Build a specific, dollar-quantified recommendation."""
        if annual_benefit <= 0:
            return f"Monitor {category} — no immediate benefit from consolidation."

        return (
            f"Shift ${shiftable_spend:,.0f} in {category} purchases to "
            f"co-op warehouse. Consolidate from {len(vendors)} vendors to "
            f"primary co-op source. Annual gain: ${annual_benefit:,.0f}/year "
            f"(patronage + cash discount)."
        )

    # -----------------------------------------------------------------
    # Tier Progress Tracking
    # -----------------------------------------------------------------

    def calculate_tier_progress(
        self,
        purchases: list[VendorPurchase],
    ) -> dict[PatronageCategory, dict]:
        """Calculate YTD progress by patronage category.

        Returns a dict of category → {ytd_spend, patronage_earned,
        cash_discount_earned, total_earned}.
        """
        category_spend: dict[PatronageCategory, float] = defaultdict(float)

        for p in purchases:
            if p.is_coop_available or self._is_coop_sku(p.sku_id):
                pat_cat = self._classify_patronage_category(p)
                category_spend[pat_cat] += p.total_cost

        progress: dict[PatronageCategory, dict] = {}
        for pat_cat, spend in category_spend.items():
            rate = PATRONAGE_RATES.get(pat_cat, 0.0)
            patronage = spend * rate

            # Cash discount applies to warehouse categories
            cash_discount = 0.0
            if pat_cat in (
                PatronageCategory.REGULAR_WAREHOUSE,
                PatronageCategory.PROMOTIONAL_WAREHOUSE,
            ):
                cash_discount = spend * WAREHOUSE_CASH_DISCOUNT

            progress[pat_cat] = {
                "ytd_spend": spend,
                "patronage_rate": rate,
                "patronage_earned": patronage,
                "cash_discount_earned": cash_discount,
                "total_earned": patronage + cash_discount,
            }

        return progress

    def _is_coop_sku(self, sku_id: str) -> bool:
        """Check if an SKU is a co-op warehouse item."""
        prefix = sku_id.split("-")[0] if "-" in sku_id else ""
        return prefix in _COOP_WAREHOUSE_PREFIXES

    def _classify_patronage_category(
        self,
        purchase: VendorPurchase,
    ) -> PatronageCategory:
        """Classify a purchase into a patronage category."""
        if purchase.category == "Lumber":
            return PatronageCategory.LUMBER
        # Could use purchase metadata to distinguish promotional vs regular
        # For now, default to regular warehouse
        return PatronageCategory.REGULAR_WAREHOUSE

    # -----------------------------------------------------------------
    # Alert Generation
    # -----------------------------------------------------------------

    def generate_alerts(
        self,
        purchases: list[VendorPurchase],
    ) -> list[CoopAlert]:
        """Run full analysis and generate actionable alerts.

        Combines leakage detection, consolidation, and tier progress
        into a unified alert stream.

        Returns:
            List of CoopAlert, sorted by dollar impact descending.
        """
        alerts: list[CoopAlert] = []

        # 1. Patronage leakage alerts
        leakages = self.detect_patronage_leakage(purchases)
        for leakage in leakages:
            if leakage.annual_leakage >= 100:  # $100 minimum threshold
                alerts.append(
                    CoopAlert(
                        alert_type=CoopAlertType.PATRONAGE_LEAKAGE,
                        store_id=self.affiliation.store_id,
                        title=(
                            f"Patronage Leakage: ${leakage.non_coop_spend:,.0f} "
                            f"in {leakage.category} to {leakage.vendor_name}"
                        ),
                        dollar_impact=leakage.annual_leakage,
                        detail=(
                            f"Buying {leakage.category} from {leakage.vendor_name} "
                            f"instead of co-op warehouse. "
                            f"{len(leakage.affected_skus)} SKUs affected."
                        ),
                        recommendation=(
                            f"Shift ${leakage.non_coop_spend:,.0f} to co-op warehouse "
                            f"({leakage.coop_rebate_rate * 100:.1f}% patronage). "
                            f"Annual gain: ${leakage.annual_leakage:,.0f}."
                        ),
                        confidence=0.85,
                    )
                )

        # 2. Consolidation opportunity alerts
        consolidations = self.find_consolidation_opportunities(purchases)
        for opp in consolidations:
            if opp.annual_benefit >= 200:  # $200 minimum threshold
                alerts.append(
                    CoopAlert(
                        alert_type=CoopAlertType.CONSOLIDATION_OPPORTUNITY,
                        store_id=self.affiliation.store_id,
                        title=(
                            f"Consolidation: {opp.category} "
                            f"({opp.current_vendor_count} vendors)"
                        ),
                        dollar_impact=opp.annual_benefit,
                        detail=(
                            f"{opp.current_vendor_count} vendors in {opp.category} "
                            f"with ${opp.total_category_spend:,.0f} total spend. "
                            f"${opp.shiftable_spend:,.0f} shiftable to co-op."
                        ),
                        recommendation=opp.recommendation,
                        confidence=0.75,
                    )
                )

        # Sort by dollar impact descending
        alerts.sort(key=lambda a: a.dollar_impact, reverse=True)
        return alerts
