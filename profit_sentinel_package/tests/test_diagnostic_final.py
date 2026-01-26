"""Final diagnostic test - ALL rules from our conversation."""

import csv
import re
from collections import defaultdict


def parse_number(val, default=0):
    if not val or val.strip() == "":
        return default
    try:
        return float(val.replace(",", "").replace('"', "").strip())
    except:
        return default


def load_real_data():
    items = []
    with open(
        "/mnt/user-data/uploads/Inventory_Report_Audit_Adjust.csv",
        encoding="utf-8",
        errors="replace",
    ) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sku = row.get("SKU", "").strip()
            if not sku:
                continue
            item = {
                "sku": sku,
                "description": row.get("Description ", "").strip(),
                "in_stock_qty": parse_number(row.get("In Stock Qty.", "0")),
                "cost": parse_number(row.get("Cost", "0")),
                "retail": parse_number(row.get("Retail", "0")),
                "vendor": row.get("Vendor", "").strip(),
                "sold": parse_number(row.get("Sold", "0")),
                "purchased": parse_number(row.get("Pur.", "0")),
            }
            items.append(item)
    return items


# COMPLETE rules from our diagnostic conversation
LEARNED_RULES = {
    # Non-tracked by design
    "non_tracked": [
        (r"^[A-Z]{1,3}$", "Single-letter package fasteners"),
        (r"PACKAGE FASTENER|PACKAGE ITEM", "Package fasteners"),
        (r"MNB|NUTS.*BOLTS|MISC NUTS", "Nuts and bolts"),
        (r"^K1$|^KF1$|SINGLE SIDE KEY|FANCY KEY", "Individual keys"),
        (r"BY THE FOOT|BY FOOT", "Cut-to-length items"),
        (r"^WIRE BY FOOT$|^TUBING$|^ROPE & CHAIN$", "Cut-to-length"),
        (r"^SEED$", "Seeds"),
        (r"^STAPLES$|LANDSCAPE STAPLE", "Landscape staples"),  # Added!
    ],
    # Sold but not received (building materials)
    "receiving_gap": [
        (r"2X[0-9]|4X4|1X[0-9]", "Dimensional lumber"),
        (r"BOARD|STUD|LUMBER|FRAMING|SPRUCE|PINE|POPLAR|OAK", "Lumber/boards"),
        (r"PLYWOOD|DRYWALL|OSB|SHEATHING|LUAN|BIRCH.*PLY", "Sheet goods"),
        (
            r"MOULDING|TRIM|BASE|CROWN|PRIMED|PREPRIMED|QUARTERROUND|SHOE|CASING",
            "Moulding/trim",
        ),
        (r"DECK\s*BOARD|DECKING|5/4X", "Decking"),
        (r"PRESSURE\s*TREATED|[0-9]X[0-9].*PT$|\sPT$", "Pressure treated"),
        (r"CONCRETE|MORTAR|CEMENT", "Concrete/masonry"),
        (
            r"SAND\s*[0-9]+|MULTI.*SAND|PLAY.*SAND|FAST\s*SET",
            "Sand/concrete products",
        ),  # Added!
        (r"PELLET|WOOD\s*PEL", "Wood pellets"),
        (r"FILTER|FURNACE", "Air filters"),  # Added! (per our conversation)
        (r"DEER\s*CORN", "Deer corn"),  # Added!
        (r"PREMIX|PRE\s*MIX", "Premix fuel"),  # Added!
    ],
    # Expiration/damage
    "expiration": [
        (r"COKE|PEPSI|MONSTER|DR PEPPER|DASANI|POWERADE|GATORADE", "Beverages"),
        (r"SMART\s*WATER|WATER.*[0-9]+\s*(OZ|FL)", "Bottled water"),
        (r"DIET\s*(COKE|PEPSI|DR)", "Diet drinks"),
        (r"SODA|SOFT DRINK", "Soft drinks"),
        (r"SLIM\s*JIM|PEPPERMINT|CANDY", "Snacks"),  # Added!
    ],
    # Vendor managed
    "vendor_managed": [
        (r"SCOTTS|MIRACLE.?GRO|TURF BUILDER", "Scotts products"),
        (r"WEED.*FEED|FERTILIZER", "Lawn chemicals"),
        (r"MULCH", "Mulch"),
        (r"POTTING|SOIL|TOP\s*SOIL", "Soil products"),
        (r"GRASS\s*SEED", "Grass seed"),  # Added!
        (r"ICE\s*MELT|MAJESTIC|ROCK\s*SALT", "Ice melt"),  # Added!
        (r"WORM|BAIT", "Bait products"),  # Added!
    ],
}


def classify_item(item):
    """Classify item based on learned rules."""
    sku = item["sku"].upper()
    desc = item["description"].upper()
    combined = f"{sku} {desc}"

    for category, patterns in LEARNED_RULES.items():
        for pattern, name in patterns:
            if re.search(pattern, combined, re.IGNORECASE):
                return category, name

    return "unknown", None


def main():
    print("=" * 70)
    print("PROFIT SENTINEL DIAGNOSTIC - FINAL ANALYSIS")
    print("(Applying ALL rules learned from our conversation)")
    print("=" * 70)

    # Load data
    data = load_real_data()
    negative = [i for i in data if i["in_stock_qty"] < 0]

    total_value = sum(abs(i["in_stock_qty"]) * i["cost"] for i in negative)
    print(f"\nTotal SKUs: {len(data):,}")
    print(f"Negative stock: {len(negative):,} SKUs")
    print(f"Original 'shrinkage': ${total_value:,.2f}")

    # Classify all negative items
    classified = defaultdict(list)

    for item in negative:
        category, rule_name = classify_item(item)
        item["_category"] = category
        item["_rule"] = rule_name
        classified[category].append(item)

    # Calculate values
    print("\n" + "=" * 70)
    print("BREAKDOWN BY CATEGORY")
    print("=" * 70)

    categories_summary = {}
    for cat, items in classified.items():
        value = sum(abs(i["in_stock_qty"]) * i["cost"] for i in items)
        categories_summary[cat] = {"count": len(items), "value": value, "items": items}

    explained_value = 0

    # Non-tracked
    if "non_tracked" in categories_summary:
        d = categories_summary["non_tracked"]
        explained_value += d["value"]
        print(f"\n✓ NON-TRACKED (By Design): ${d['value']:,.2f} ({d['count']} SKUs)")
        sub = defaultdict(float)
        for i in d["items"]:
            sub[i["_rule"] or "other"] += abs(i["in_stock_qty"]) * i["cost"]
        for rule, val in sorted(sub.items(), key=lambda x: -x[1])[:5]:
            print(f"    {rule}: ${val:,.2f}")

    # Receiving gaps
    if "receiving_gap" in categories_summary:
        d = categories_summary["receiving_gap"]
        explained_value += d["value"]
        print(
            f"\n✓ RECEIVING GAPS (Not Entered): ${d['value']:,.2f} ({d['count']} SKUs)"
        )
        sub = defaultdict(float)
        for i in d["items"]:
            sub[i["_rule"] or "other"] += abs(i["in_stock_qty"]) * i["cost"]
        for rule, val in sorted(sub.items(), key=lambda x: -x[1]):
            print(f"    {rule}: ${val:,.2f}")

    # Expiration
    if "expiration" in categories_summary:
        d = categories_summary["expiration"]
        explained_value += d["value"]
        print(f"\n✓ EXPIRATION/DAMAGE: ${d['value']:,.2f} ({d['count']} SKUs)")
        sub = defaultdict(float)
        for i in d["items"]:
            sub[i["_rule"] or "other"] += abs(i["in_stock_qty"]) * i["cost"]
        for rule, val in sorted(sub.items(), key=lambda x: -x[1]):
            print(f"    {rule}: ${val:,.2f}")

    # Vendor managed
    if "vendor_managed" in categories_summary:
        d = categories_summary["vendor_managed"]
        explained_value += d["value"]
        print(f"\n✓ VENDOR MANAGED: ${d['value']:,.2f} ({d['count']} SKUs)")
        sub = defaultdict(float)
        for i in d["items"]:
            sub[i["_rule"] or "other"] += abs(i["in_stock_qty"]) * i["cost"]
        for rule, val in sorted(sub.items(), key=lambda x: -x[1]):
            print(f"    {rule}: ${val:,.2f}")

    # Unknown
    unexplained = 0
    if "unknown" in categories_summary:
        d = categories_summary["unknown"]
        unexplained = d["value"]
        print(f"\n❓ UNEXPLAINED: ${d['value']:,.2f} ({d['count']} SKUs)")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                  DIAGNOSTIC RESULTS                           ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║                                                               ║
    ║  ORIGINAL "SHRINKAGE":         ${total_value:>14,.2f}            ║
    ║                                                               ║
    ║  EXPLAINED (Process Issues):   ${explained_value:>14,.2f}            ║
    ║    • Non-tracked by design     ${categories_summary.get('non_tracked', {}).get('value', 0):>14,.2f}            ║
    ║    • Receiving gaps            ${categories_summary.get('receiving_gap', {}).get('value', 0):>14,.2f}            ║
    ║    • Expiration/damage         ${categories_summary.get('expiration', {}).get('value', 0):>14,.2f}            ║
    ║    • Vendor managed            ${categories_summary.get('vendor_managed', {}).get('value', 0):>14,.2f}            ║
    ║                                                               ║
    ║  TRUE SHRINKAGE:               ${unexplained:>14,.2f}            ║
    ║                                                               ║
    ║  REDUCTION:                    {explained_value/total_value*100:>13.1f}%             ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝

    ACTUAL SHRINK RATE: {unexplained/3200000*100:.2f}% (${unexplained:,.0f} / $3.2M)
    Industry average: 1.4%

    {'✅ WITHIN NORMAL RANGE!' if unexplained/3200000*100 < 2.0 else '⚠️ Above average - investigate top items'}
    """
    )

    # Journey simulation
    print("\n" + "-" * 70)
    print("DIAGNOSTIC JOURNEY (how we got here)")
    print("-" * 70)
    print(
        """
    Iteration 1: $726,749 "shrinkage" detected
        ↓ User: "nuts and bolts aren't tracked"
    Iteration 2: $688,973 remaining
        ↓ User: "lumber isn't received either"
    Iteration 3: $325,000 remaining
        ↓ User: "same with plywood, drywall"
    Iteration 4: $200,000 remaining
        ↓ User: "beverages expire, scotts ships direct"
    Iteration 5: $175,000 remaining
        ↓ User: "pellets, filters, staples same deal"
    Iteration 6: ${unexplained:,.0f} remaining ← TRUE SHRINKAGE

    RULES LEARNED: {len(sum([p for p in LEARNED_RULES.values()], []))} patterns across 4 categories
    """
    )

    # Top unexplained
    if "unknown" in categories_summary:
        print("\n" + "-" * 70)
        print("REMAINING UNEXPLAINED ITEMS (potential theft)")
        print("-" * 70)

        unknown_items = sorted(
            categories_summary["unknown"]["items"],
            key=lambda x: abs(x["in_stock_qty"]) * x["cost"],
            reverse=True,
        )

        print(f"\n{'SKU':<18} {'Description':<32} {'Gap':>8} {'$ Loss':>12}")
        print("-" * 75)

        for item in unknown_items[:15]:
            desc = (
                item["description"][:30] + ".."
                if len(item["description"]) > 32
                else item["description"]
            )
            value = abs(item["in_stock_qty"]) * item["cost"]
            print(
                f"{item['sku']:<18} {desc:<32} {abs(item['in_stock_qty']):>8,.0f} ${value:>10,.2f}"
            )

        print(f"\n... and {len(unknown_items) - 15} more items")


if __name__ == "__main__":
    main()
