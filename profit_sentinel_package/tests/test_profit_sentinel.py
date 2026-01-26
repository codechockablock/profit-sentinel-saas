"""
PROFIT SENTINEL COMPREHENSIVE TEST
==================================

Test the agent with realistic retail inventory scenarios for all 11 leak types.
"""

import json
import random
from datetime import datetime, timedelta

# Import Dorian components
from dorian_core import DorianCore
from dorian_economics import load_economics_into_core
from profit_sentinel_agent import ProfitLeakType, ProfitSentinelAgent


def generate_realistic_inventory(n_skus: int = 100) -> list:
    """Generate realistic inventory data with various profit leak scenarios."""

    categories = [
        "electronics",
        "apparel",
        "grocery",
        "home",
        "toys",
        "automotive",
        "sports",
        "beauty",
    ]
    seasons = ["winter", "spring", "summer", "fall"]
    season_items = {
        "winter": ["winter coat", "snow boots", "heated blanket", "christmas lights"],
        "spring": ["garden tools", "rain jacket", "allergy medicine", "easter candy"],
        "summer": ["beach towel", "sunscreen", "pool float", "bbq grill"],
        "fall": ["halloween costume", "thanksgiving decor", "back to school supplies"],
    }

    inventory = []

    for i in range(n_skus):
        sku_id = f"SKU{i:05d}"
        category = random.choice(categories)

        # Base metrics
        unit_cost = random.uniform(5, 200)
        base_margin = random.uniform(0.15, 0.45)
        retail_price = unit_cost * (1 + base_margin)

        # Determine scenario type - now with all 11 types
        scenario = random.choices(
            [
                "healthy",
                "dead",
                "margin_erosion",
                "overstock",
                "stockout",
                "shrinkage",
                "pricing_error",
                "markdown_timing",
                "seasonal_mismatch",
                "cannibalization",
                "cost_spike",
            ],
            weights=[30, 8, 10, 10, 8, 5, 6, 6, 6, 5, 6],
        )[0]

        # Common base values
        item = {
            "sku_id": sku_id,
            "category": category,
            "unit_cost": unit_cost,
            "retail_price": retail_price,
            "name": f"{category} item {i}",
            "tags": [],
        }

        if scenario == "healthy":
            item.update(
                {
                    "days_since_sale": random.randint(0, 14),
                    "margin_percent": base_margin * 100,
                    "prev_margin_percent": base_margin * 100 + random.uniform(-2, 2),
                    "quantity_on_hand": random.randint(20, 100),
                    "avg_daily_sales": random.uniform(2, 10),
                    "revenue": random.uniform(1000, 10000),
                }
            )

        elif scenario == "dead":
            item.update(
                {
                    "days_since_sale": random.randint(91, 365),
                    "margin_percent": random.uniform(5, 25),
                    "prev_margin_percent": random.uniform(5, 25),
                    "quantity_on_hand": random.randint(50, 500),
                    "avg_daily_sales": 0,
                    "revenue": 0,
                }
            )

        elif scenario == "margin_erosion":
            item.update(
                {
                    "days_since_sale": random.randint(0, 14),
                    "margin_percent": random.uniform(2, 12),
                    "prev_margin_percent": random.uniform(20, 35),
                    "quantity_on_hand": random.randint(20, 100),
                    "avg_daily_sales": random.uniform(1, 5),
                    "revenue": random.uniform(500, 5000),
                }
            )

        elif scenario == "overstock":
            item.update(
                {
                    "days_since_sale": random.randint(0, 7),
                    "margin_percent": base_margin * 100,
                    "prev_margin_percent": base_margin * 100,
                    "quantity_on_hand": random.randint(200, 500),
                    "avg_daily_sales": random.uniform(0.5, 2),
                    "revenue": random.uniform(500, 2000),
                }
            )

        elif scenario == "stockout":
            item.update(
                {
                    "days_since_sale": 0,
                    "margin_percent": base_margin * 100,
                    "prev_margin_percent": base_margin * 100,
                    "quantity_on_hand": 0,
                    "avg_daily_sales": random.uniform(5, 20),
                    "revenue": 0,
                }
            )

        elif scenario == "shrinkage":
            item.update(
                {
                    "days_since_sale": random.randint(0, 30),
                    "margin_percent": base_margin * 100,
                    "prev_margin_percent": base_margin * 100,
                    "quantity_on_hand": random.randint(10, 50),
                    "avg_daily_sales": random.uniform(1, 5),
                    "revenue": random.uniform(500, 3000),
                }
            )

        elif scenario == "pricing_error":
            # Price below cost or way below category average
            bad_price = unit_cost * random.uniform(0.7, 0.95)  # Below cost
            item.update(
                {
                    "retail_price": bad_price,
                    "days_since_sale": random.randint(0, 7),
                    "margin_percent": ((bad_price - unit_cost) / bad_price * 100)
                    if bad_price > 0
                    else -10,
                    "prev_margin_percent": base_margin * 100,
                    "quantity_on_hand": random.randint(20, 100),
                    "avg_daily_sales": random.uniform(
                        3, 10
                    ),  # Selling fast due to low price
                    "revenue": random.uniform(1000, 5000),
                }
            )

        elif scenario == "markdown_timing":
            # Item on markdown with timing issues
            original_price = retail_price * random.uniform(1.3, 1.8)
            item.update(
                {
                    "original_price": original_price,
                    "retail_price": retail_price,
                    "original_margin_pct": 35,
                    "days_since_sale": random.randint(45, 90),  # Sat too long
                    "margin_percent": base_margin * 100,
                    "prev_margin_percent": 35,
                    "quantity_on_hand": random.randint(50, 200),
                    "avg_daily_sales": random.uniform(0.5, 2),
                    "revenue": random.uniform(500, 2000),
                }
            )

        elif scenario == "seasonal_mismatch":
            # Pick a season item that's out of season
            item_season = random.choice(seasons)
            item_name = random.choice(season_items[item_season])
            item.update(
                {
                    "name": item_name,
                    "tags": [item_season],
                    "days_since_sale": random.randint(10, 60),
                    "margin_percent": base_margin * 100,
                    "prev_margin_percent": base_margin * 100,
                    "quantity_on_hand": random.randint(50, 300),
                    "avg_daily_sales": random.uniform(0.2, 1),
                    "revenue": random.uniform(200, 1000),
                }
            )

        elif scenario == "cannibalization":
            # Item that's cannibalizing others (low margin, high sales)
            item.update(
                {
                    "days_since_sale": random.randint(0, 3),
                    "margin_percent": random.uniform(5, 12),  # Low margin
                    "prev_margin_percent": random.uniform(5, 12),
                    "quantity_on_hand": random.randint(50, 150),
                    "avg_daily_sales": random.uniform(8, 20),  # High velocity
                    "prev_avg_daily_sales": random.uniform(5, 10),
                    "is_promoted": True,
                    "revenue": random.uniform(2000, 8000),
                }
            )

        elif scenario == "cost_spike":
            # Cost increased but price didn't
            prev_cost = unit_cost * random.uniform(0.75, 0.9)
            item.update(
                {
                    "unit_cost": unit_cost,
                    "prev_unit_cost": prev_cost,
                    "retail_price": retail_price,
                    "prev_retail_price": retail_price,  # Price stayed same
                    "days_since_sale": random.randint(0, 14),
                    "margin_percent": (retail_price - unit_cost) / retail_price * 100,
                    "prev_margin_percent": (retail_price - prev_cost)
                    / retail_price
                    * 100,
                    "quantity_on_hand": random.randint(30, 100),
                    "avg_daily_sales": random.uniform(2, 8),
                    "revenue": random.uniform(1000, 5000),
                }
            )

        item["_scenario"] = scenario  # For validation
        inventory.append(item)

    return inventory


def generate_adjustments(inventory: list, n_adjustments: int = 20) -> list:
    """Generate inventory adjustments (some indicating shrinkage)."""

    adjustments = []

    for _ in range(n_adjustments):
        sku = random.choice(inventory)

        # 70% negative adjustments (potential shrinkage)
        if random.random() < 0.7:
            qty = -random.randint(1, 10)
        else:
            qty = random.randint(1, 5)

        adjustments.append(
            {
                "sku_id": sku["sku_id"],
                "quantity": qty,
                "unit_cost": sku["unit_cost"],
                "timestamp": datetime.now() - timedelta(days=random.randint(0, 30)),
            }
        )

    return adjustments


def generate_vendor_data(inventory: list) -> tuple:
    """Generate receipts and POs with some compliance issues."""

    vendors = ["VENDOR_A", "VENDOR_B", "VENDOR_C", "VENDOR_D"]

    # Assign vendors to items
    for item in inventory:
        item["vendor_id"] = random.choice(vendors)

    purchase_orders = []
    receipts = []

    for i, item in enumerate(random.sample(inventory, min(50, len(inventory)))):
        po_id = f"PO{i:05d}"
        ordered_qty = random.randint(20, 100)
        agreed_cost = item["unit_cost"] * random.uniform(0.9, 1.0)

        purchase_orders.append(
            {
                "po_id": po_id,
                "sku_id": item["sku_id"],
                "vendor_id": item["vendor_id"],
                "quantity_ordered": ordered_qty,
                "agreed_unit_cost": agreed_cost,
            }
        )

        # Some receipts have issues
        if random.random() < 0.3:
            # Short shipment
            received_qty = int(ordered_qty * random.uniform(0.7, 0.9))
        else:
            received_qty = ordered_qty

        if random.random() < 0.2:
            # Price overcharge
            received_cost = agreed_cost * random.uniform(1.05, 1.15)
        else:
            received_cost = agreed_cost

        receipts.append(
            {
                "po_id": po_id,
                "sku_id": item["sku_id"],
                "vendor_id": item["vendor_id"],
                "quantity_received": received_qty,
                "unit_cost": received_cost,
            }
        )

    return receipts, purchase_orders


def main():
    print("=" * 70)
    print("PROFIT SENTINEL COMPREHENSIVE TEST - ALL 11 LEAK TYPES")
    print("=" * 70)

    # Initialize core
    print("\n[1/4] Initializing Dorian Knowledge Brain...")
    core = DorianCore(dim=256, load_ontology=True)
    core.bootstrap_ontology()

    # Load economics domain
    print("\n[2/4] Loading economics knowledge...")
    load_economics_into_core(core)
    core.train(verbose=False)

    # Create agent
    print("\n[3/4] Initializing Profit Sentinel Agent...")
    agent = ProfitSentinelAgent(core)

    # Generate test data
    print("\n[4/4] Generating test data...")
    inventory = generate_realistic_inventory(150)
    adjustments = generate_adjustments(inventory, 40)
    receipts, purchase_orders = generate_vendor_data(inventory)

    # Count expected scenarios
    scenario_counts = {}
    for item in inventory:
        s = item["_scenario"]
        scenario_counts[s] = scenario_counts.get(s, 0) + 1

    print("\nExpected scenarios in test data:")
    for scenario, count in sorted(scenario_counts.items()):
        print(f"  {scenario}: {count}")

    # Run analysis
    print("\n" + "=" * 70)
    report = agent.analyze_inventory(
        inventory,
        adjustments=adjustments,
        receipts=receipts,
        purchase_orders=purchase_orders,
        current_season="winter",
    )

    # Detailed results
    print("\n" + "=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)

    print(f"\nTotal SKUs analyzed: {report['skus_analyzed']}")
    print(f"Total leaks detected: {report['total_leaks_detected']}")
    print(f"Total estimated impact: ${report['total_estimated_impact']:,.2f}")

    print("\nLeaks by type:")
    for leak_type, data in sorted(
        report["leaks_by_type"].items(), key=lambda x: -x[1]["total_impact"]
    ):
        print(f"  {leak_type}:")
        print(f"    Count: {data['count']}")
        print(f"    Impact: ${data['total_impact']:,.2f}")
        print(f"    Avg confidence: {data['avg_confidence']:.2f}")

    print("\nTop 10 highest impact leaks:")
    for i, leak in enumerate(report["top_leaks"][:10], 1):
        print(f"  {i}. [{leak['type']}] SKU {leak['affected_skus'][0]}")
        print(
            f"     Impact: ${leak['estimated_impact']:,.2f} | Confidence: {leak['confidence']:.2f}"
        )

    print("\nCommon causal patterns:")
    for pattern, count in list(report["causal_patterns"].items())[:8]:
        print(f"  {pattern}: {count} occurrences")

    # Coverage check
    print("\n" + "=" * 70)
    print("DETECTOR COVERAGE CHECK")
    print("=" * 70)

    detected_types = set(report["leaks_by_type"].keys())
    all_types = {lt.value for lt in ProfitLeakType}
    missing = all_types - detected_types

    print(f"\nDetected {len(detected_types)}/11 leak types")
    if missing:
        print(f"Not detected: {missing}")
    else:
        print("✓ All 11 leak types detected!")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

    return report


if __name__ == "__main__":
    report = main()
