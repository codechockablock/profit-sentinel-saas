#!/usr/bin/env python3
"""
Profit Sentinel - Synthetic Data Generator

Generates realistic retail POS data for testing without using real customer data.

Usage:
    python tools/generate_synthetic_data.py
    python tools/generate_synthetic_data.py --items 5000 --transactions 20000
    python tools/generate_synthetic_data.py --output ./my_test_data

Prerequisites:
    pip install pandas numpy faker

Output:
    - inventory.csv: Inventory items with SKUs, descriptions, costs, prices, quantities
    - sales.csv: Sales transaction history
    - Both files include realistic "profit leaks" for testing detection

IMPORTANT: Use this for ALL testing. Never use real customer data!
"""

import argparse
import os
import random
from datetime import datetime, timedelta
from typing import Optional

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Run: pip install pandas numpy")
    exit(1)

# Try to import faker for realistic names, fall back to simple generation
try:
    from faker import Faker
    fake = Faker()
    HAS_FAKER = True
except ImportError:
    HAS_FAKER = False
    print("NOTE: faker not installed. Using basic name generation.")
    print("For more realistic data, run: pip install faker")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Retail categories typical for hardware/home improvement stores
CATEGORIES = [
    "Hardware",
    "Lumber",
    "Plumbing",
    "Electrical",
    "Paint",
    "Tools",
    "Outdoor",
    "Flooring",
    "Lighting",
    "HVAC",
    "Fasteners",
    "Safety",
]

# Vendor names
VENDORS = [
    "ABC Supply Co",
    "Builder's Best",
    "Direct Wholesale",
    "Eastern Hardware",
    "FastTrack Distribution",
    "General Supply",
    "Home Pros Inc",
    "Industrial Direct",
]

# Product name templates per category
PRODUCT_TEMPLATES = {
    "Hardware": ["Lock Set", "Door Hinge", "Cabinet Knob", "Drawer Pull", "Chain Link", "Padlock"],
    "Lumber": ["2x4x8 Pine", "4x4x10 Treated", "Plywood Sheet", "MDF Board", "Cedar Board"],
    "Plumbing": ["PVC Elbow", "Copper Fitting", "Drain Pipe", "Faucet", "Valve", "Trap"],
    "Electrical": ["Wire Spool", "Outlet", "Switch", "Junction Box", "Conduit", "Breaker"],
    "Paint": ["Interior Latex", "Exterior Paint", "Primer", "Stain", "Brush Set", "Roller"],
    "Tools": ["Drill", "Saw", "Hammer", "Screwdriver Set", "Level", "Tape Measure"],
    "Outdoor": ["Garden Hose", "Sprinkler", "Rake", "Shovel", "Wheelbarrow", "Planter"],
    "Flooring": ["Tile", "Laminate", "Vinyl Plank", "Carpet Square", "Underlayment", "Grout"],
    "Lighting": ["LED Bulb", "Fixture", "Lamp", "Floodlight", "Motion Sensor", "Dimmer"],
    "HVAC": ["Filter", "Thermostat", "Duct Tape", "Vent Cover", "Insulation", "Fan"],
    "Fasteners": ["Screw Box", "Nail Box", "Bolt Set", "Anchor Kit", "Staples", "Washer Pack"],
    "Safety": ["Gloves", "Goggles", "Mask", "Hard Hat", "Vest", "First Aid Kit"],
}


# =============================================================================
# DATA GENERATION
# =============================================================================


def generate_sku(index: int, category: str) -> str:
    """Generate a realistic SKU."""
    prefix = category[:3].upper()
    return f"{prefix}-{index:06d}"


def generate_upc(index: int) -> str:
    """Generate a realistic UPC barcode."""
    base = f"0{random.randint(10000, 99999)}{index:05d}"
    return base[:12]


def generate_description(category: str) -> str:
    """Generate a product description."""
    templates = PRODUCT_TEMPLATES.get(category, ["Item"])
    base = random.choice(templates)

    # Add modifiers
    modifiers = ["", "Pro", "Premium", "Standard", "Economy", "Heavy Duty"]
    sizes = ["", "Small", "Medium", "Large", "XL"]
    colors = ["", "Black", "White", "Gray", "Blue", "Red"]

    modifier = random.choice(modifiers)
    size = random.choice(sizes)
    color = random.choice(colors)

    parts = [p for p in [modifier, size, color, base] if p]
    return " ".join(parts)


def generate_inventory_items(num_items: int = 1000) -> pd.DataFrame:
    """Generate synthetic inventory items."""
    print(f"Generating {num_items} inventory items...")

    items = []
    for i in range(num_items):
        category = random.choice(CATEGORIES)
        vendor = random.choice(VENDORS)

        # Generate cost with realistic distribution
        cost = round(random.lognormvariate(2.5, 1.0), 2)  # Most items $5-50, some expensive
        cost = min(max(cost, 0.50), 500.00)  # Clamp to reasonable range

        # Generate markup (typically 20-80%)
        markup = random.uniform(1.2, 1.8)
        price = round(cost * markup, 2)

        # Generate quantities
        qty_on_hand = random.randint(0, 100)
        qty_on_order = random.randint(0, 50) if random.random() > 0.7 else 0
        reorder_point = random.randint(5, 20)

        # Sales velocity (items sold per day, average)
        sales_velocity = random.uniform(0, 2)

        items.append({
            "sku": generate_sku(i, category),
            "upc": generate_upc(i),
            "description": generate_description(category),
            "category": category,
            "vendor": vendor,
            "cost": cost,
            "price": price,
            "quantity_on_hand": qty_on_hand,
            "quantity_on_order": qty_on_order,
            "reorder_point": reorder_point,
            "last_received": (datetime.now() - timedelta(days=random.randint(1, 180))).strftime("%Y-%m-%d"),
            "last_sold": (datetime.now() - timedelta(days=random.randint(0, 90))).strftime("%Y-%m-%d"),
            "_sales_velocity": sales_velocity,  # Internal use for leak injection
        })

    return pd.DataFrame(items)


def inject_profit_leaks(df: pd.DataFrame, leak_config: Optional[dict] = None) -> pd.DataFrame:
    """
    Inject realistic profit leaks into the inventory data.

    This makes the synthetic data useful for testing the detection engine.
    """
    if leak_config is None:
        leak_config = {
            "dead_stock_pct": 0.15,      # 15% dead stock
            "shrinkage_pct": 0.08,        # 8% shrinkage
            "negative_inventory_pct": 0.03,  # 3% negative inventory
            "margin_erosion_pct": 0.05,   # 5% margin erosion
            "low_stock_pct": 0.10,        # 10% low stock
            "overstock_pct": 0.07,        # 7% overstock
        }

    df = df.copy()
    n = len(df)

    print("Injecting profit leaks...")

    # Dead Stock - items with no recent sales
    dead_count = int(n * leak_config.get("dead_stock_pct", 0.15))
    dead_indices = np.random.choice(n, size=dead_count, replace=False)
    df.loc[dead_indices, "last_sold"] = (datetime.now() - timedelta(days=random.randint(120, 365))).strftime("%Y-%m-%d")
    df.loc[dead_indices, "quantity_on_hand"] = df.loc[dead_indices, "quantity_on_hand"].apply(
        lambda x: max(x, random.randint(20, 50))  # Ensure they have stock
    )
    print(f"  - Dead stock: {dead_count} items")

    # Shrinkage - quantity discrepancies (unexplained losses)
    shrink_count = int(n * leak_config.get("shrinkage_pct", 0.08))
    shrink_indices = np.random.choice(n, size=shrink_count, replace=False)
    # Reduce quantity without explanation
    df.loc[shrink_indices, "quantity_on_hand"] = df.loc[shrink_indices, "quantity_on_hand"].apply(
        lambda x: max(0, x - random.randint(1, 10))
    )
    print(f"  - Shrinkage patterns: {shrink_count} items")

    # Negative Inventory - system errors (sold without being received)
    negative_count = int(n * leak_config.get("negative_inventory_pct", 0.03))
    negative_indices = np.random.choice(n, size=negative_count, replace=False)
    df.loc[negative_indices, "quantity_on_hand"] = -random.randint(1, 15)
    print(f"  - Negative inventory: {negative_count} items")

    # Margin Erosion - items priced below acceptable margin
    margin_count = int(n * leak_config.get("margin_erosion_pct", 0.05))
    margin_indices = np.random.choice(n, size=margin_count, replace=False)
    # Set price too close to cost (less than 10% margin)
    df.loc[margin_indices, "price"] = df.loc[margin_indices, "cost"].apply(
        lambda c: round(c * random.uniform(0.95, 1.08), 2)
    )
    print(f"  - Margin erosion: {margin_count} items")

    # Low Stock - items below reorder point
    low_count = int(n * leak_config.get("low_stock_pct", 0.10))
    low_indices = np.random.choice(n, size=low_count, replace=False)
    df.loc[low_indices, "quantity_on_hand"] = df.loc[low_indices, "reorder_point"].apply(
        lambda rp: random.randint(0, max(0, rp - 3))
    )
    print(f"  - Low stock: {low_count} items")

    # Overstock - excessive inventory
    over_count = int(n * leak_config.get("overstock_pct", 0.07))
    over_indices = np.random.choice(n, size=over_count, replace=False)
    df.loc[over_indices, "quantity_on_hand"] = df.loc[over_indices, "quantity_on_hand"].apply(
        lambda x: x + random.randint(50, 200)
    )
    df.loc[over_indices, "_sales_velocity"] = df.loc[over_indices, "_sales_velocity"].apply(
        lambda v: v * 0.2  # Low sales velocity for overstock
    )
    print(f"  - Overstock: {over_count} items")

    return df


def generate_sales_transactions(
    inventory_df: pd.DataFrame,
    num_transactions: int = 5000,
    days_back: int = 90
) -> pd.DataFrame:
    """Generate synthetic sales transaction history."""
    print(f"Generating {num_transactions} sales transactions...")

    transactions = []
    start_date = datetime.now() - timedelta(days=days_back)

    for i in range(num_transactions):
        # Random date within range
        transaction_date = start_date + timedelta(
            days=random.randint(0, days_back),
            hours=random.randint(8, 20),  # Business hours
            minutes=random.randint(0, 59)
        )

        # Random number of items in transaction (1-5)
        num_items = random.randint(1, 5)

        # Select random items, weighted by sales velocity
        weights = inventory_df["_sales_velocity"].values + 0.1  # Avoid zero weights
        weights = weights / weights.sum()

        selected_indices = np.random.choice(
            len(inventory_df),
            size=num_items,
            replace=False,
            p=weights
        )

        for idx in selected_indices:
            item = inventory_df.iloc[idx]
            quantity = random.randint(1, 5)

            # Occasional discount
            discount_pct = random.choice([0, 0, 0, 0, 5, 10, 15, 20])  # Mostly no discount
            unit_price = item["price"] * (1 - discount_pct / 100)

            transactions.append({
                "transaction_id": f"TXN-{i:07d}",
                "transaction_date": transaction_date.strftime("%Y-%m-%d %H:%M:%S"),
                "sku": item["sku"],
                "description": item["description"],
                "quantity": quantity,
                "unit_price": round(unit_price, 2),
                "discount_pct": discount_pct,
                "line_total": round(quantity * unit_price, 2),
                "cost": item["cost"],
                "margin": round((unit_price - item["cost"]) / unit_price * 100, 1) if unit_price > 0 else 0,
            })

    return pd.DataFrame(transactions)


def export_data(
    inventory_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    output_dir: str
):
    """Export generated data to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    # Remove internal columns before export
    inventory_export = inventory_df.drop(columns=["_sales_velocity"], errors="ignore")

    inventory_path = os.path.join(output_dir, "inventory.csv")
    sales_path = os.path.join(output_dir, "sales.csv")

    inventory_export.to_csv(inventory_path, index=False)
    sales_df.to_csv(sales_path, index=False)

    print(f"\n{'=' * 50}")
    print("SYNTHETIC DATA GENERATED SUCCESSFULLY")
    print("=" * 50)
    print(f"\nOutput directory: {output_dir}")
    print(f"  - inventory.csv: {len(inventory_df):,} items")
    print(f"  - sales.csv: {len(sales_df):,} transactions")

    # Summary of injected leaks
    print("\nInjected profit leaks for testing:")
    negative_count = (inventory_export["quantity_on_hand"] < 0).sum()
    low_margin = (inventory_export["price"] < inventory_export["cost"] * 1.1).sum()
    print(f"  - Negative inventory items: {negative_count}")
    print(f"  - Low margin items (<10%): {low_margin}")
    print(f"  - Dead stock (90+ days): ~{int(len(inventory_df) * 0.15)}")

    print("\nUsage:")
    print(f"  1. Upload {inventory_path} to Profit Sentinel")
    print(f"  2. Run analysis to test detection engine")
    print(f"  3. Optionally upload {sales_path} for sales history analysis")

    print("\n" + "=" * 50)
    print("IMPORTANT: Use this data for testing ONLY.")
    print("Never use real customer data for testing!")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic retail POS data for testing"
    )
    parser.add_argument(
        "--items",
        type=int,
        default=1000,
        help="Number of inventory items to generate (default: 1000)"
    )
    parser.add_argument(
        "--transactions",
        type=int,
        default=5000,
        help="Number of sales transactions to generate (default: 5000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./synthetic_data",
        help="Output directory for generated files (default: ./synthetic_data)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation"
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    print("\n" + "=" * 50)
    print("PROFIT SENTINEL - SYNTHETIC DATA GENERATOR")
    print("=" * 50)

    # Generate data
    inventory = generate_inventory_items(args.items)
    inventory = inject_profit_leaks(inventory)
    sales = generate_sales_transactions(inventory, args.transactions)

    # Export
    export_data(inventory, sales, args.output)


if __name__ == "__main__":
    main()
