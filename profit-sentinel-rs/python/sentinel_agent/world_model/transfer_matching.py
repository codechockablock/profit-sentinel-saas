"""
Cross-Store Transfer Matching
================================

The first feature that REQUIRES multi-agent architecture.

Single store finds dead stock → recommendation is markdown/clearance.
Multi-store finds dead stock → recommendation is TRANSFER to the
store that's actually selling it, at full margin.

The difference between "sell it at a loss" and "move it to where
the demand is" is the entire value proposition for multi-store
customers.

How it works:
1. Source store agent detects dead stock (sustained zero velocity)
2. Orchestrator takes the dead item's VSA encoding
3. Searches all other stores' published velocity primitives
4. Matches at three hierarchy levels: exact SKU, subcategory, category
5. Ranks destinations by velocity, margin, and confidence
6. Generates transfer recommendation with dollar impact

The algebra does the heavy lifting:
- Entity hierarchy (SKU → subcategory → category) is VSA bundling
- Cross-store matching is resonator search over shared codebook
- Confidence comes from the battery (how healthy are the bindings?)

Author: Joseph + Claude
Date: 2026-02-10
"""

import hashlib
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import DeadStockConfig
from .core import PhasorAlgebra, StateVector, WorldModelConfig

# Optional Rust-accelerated backend
try:
    from .rust_algebra import RUST_AVAILABLE, RustPhasorAlgebra
except ImportError:
    RUST_AVAILABLE = False


# =============================================================================
# DATA TYPES
# =============================================================================


class MatchLevel(Enum):
    """How precisely the demand match was found."""

    EXACT_SKU = "exact_sku"  # Same SKU at another store
    SUBCATEGORY = "subcategory"  # Similar items in same subcategory
    CATEGORY = "category"  # Broader category match


@dataclass
class SKUProfile:
    """Everything we know about a SKU at a specific store."""

    sku_id: str
    store_id: str
    description: str
    subcategory: str
    category: str
    current_stock: int
    avg_weekly_velocity: float  # Units sold per week
    cost: float
    price: float
    margin_pct: float
    days_since_last_sale: int
    vector: np.ndarray  # VSA encoding
    subcategory_vector: np.ndarray  # Category-level encoding
    category_vector: np.ndarray  # Broad category encoding


@dataclass
class TransferRecommendation:
    """A specific transfer recommendation with financial impact."""

    # Source
    source_store: str
    source_sku: str
    source_description: str
    source_stock: int
    source_days_dead: int

    # Destination
    dest_store: str
    dest_sku: str  # May differ from source if subcategory match
    dest_description: str
    dest_weekly_velocity: float
    dest_current_stock: int

    # Match details
    match_level: MatchLevel
    match_confidence: float  # 0-1, from resonator convergence

    # Transfer details
    units_to_transfer: int

    # Financial impact
    unit_cost: float
    unit_price: float
    margin_per_unit: float

    clearance_recovery: float  # What you'd get at 50% markdown
    transfer_recovery: float  # What you'd get at full margin
    net_benefit: float  # transfer - clearance = money saved

    # Context
    demand_pattern: str  # Why the destination is selling
    estimated_weeks_to_sell: float  # How long it'll take at dest velocity

    def to_dict(self) -> dict:
        return {
            "source_store": self.source_store,
            "source_sku": self.source_sku,
            "source_description": self.source_description,
            "units_to_transfer": self.units_to_transfer,
            "dest_store": self.dest_store,
            "dest_sku": self.dest_sku,
            "dest_description": self.dest_description,
            "match_level": self.match_level.value,
            "match_confidence": round(self.match_confidence, 3),
            "clearance_recovery": round(self.clearance_recovery, 2),
            "transfer_recovery": round(self.transfer_recovery, 2),
            "net_benefit": round(self.net_benefit, 2),
            "demand_pattern": self.demand_pattern,
            "estimated_weeks_to_sell": round(self.estimated_weeks_to_sell, 1),
        }

    def summary(self) -> str:
        return (
            f"TRANSFER: {self.units_to_transfer}x {self.source_description} "
            f"({self.source_sku})\n"
            f"  From: {self.source_store} (dead {self.source_days_dead} days)\n"
            f"  To:   {self.dest_store} "
            f"(selling {self.dest_weekly_velocity:.1f}/week)\n"
            f"  Match: {self.match_level.value} "
            f"(confidence: {self.match_confidence:.0%})\n"
            f"  Recovery at clearance (50%): ${self.clearance_recovery:,.2f}\n"
            f"  Recovery at transfer (full): ${self.transfer_recovery:,.2f}\n"
            f"  NET BENEFIT: ${self.net_benefit:,.2f}\n"
            f"  Estimated sell-through: {self.estimated_weeks_to_sell:.1f} weeks\n"
            f"  Demand pattern: {self.demand_pattern}"
        )


# =============================================================================
# ENTITY HIERARCHY
# =============================================================================


class EntityHierarchy:
    """
    Three-level entity hierarchy encoded in VSA.

    SKU → Subcategory → Category

    Each level is a VSA vector. Higher levels are bundles of
    lower-level vectors. This lets the resonator match at any
    granularity: exact SKU, similar items, or broad category.

    The hierarchy is built from the store's actual inventory data,
    not from a fixed taxonomy. Structure learning can discover
    that two subcategories should merge or one should split.
    """

    def __init__(self, algebra: PhasorAlgebra):
        self.algebra = algebra

        # Role vectors for hierarchy levels
        self.role_sku = algebra.get_or_create("role_sku")
        self.role_subcategory = algebra.get_or_create("role_subcategory")
        self.role_category = algebra.get_or_create("role_category")
        self.role_velocity = algebra.get_or_create("role_velocity")
        self.role_margin = algebra.get_or_create("role_margin")
        self.role_stock = algebra.get_or_create("role_stock")

        # Codebooks at each level
        self.sku_vectors: dict[str, np.ndarray] = {}
        self.subcategory_vectors: dict[str, np.ndarray] = {}
        self.category_vectors: dict[str, np.ndarray] = {}

        # Velocity encoding
        self.velocity_levels = {}
        for level in ["dead", "slow", "moderate", "fast", "hot"]:
            self.velocity_levels[level] = algebra.get_or_create(f"vel_{level}")

        # Membership tracking
        self.sku_to_subcategory: dict[str, str] = {}
        self.subcategory_to_category: dict[str, str] = {}
        self.subcategory_members: dict[str, list[str]] = defaultdict(list)
        self.category_members: dict[str, list[str]] = defaultdict(list)

    def register_sku(self, sku_id: str, subcategory: str, category: str):
        """Register a SKU in the hierarchy."""
        # Create or retrieve vectors at each level
        if sku_id not in self.sku_vectors:
            self.sku_vectors[sku_id] = self.algebra.get_or_create(sku_id)

        if subcategory not in self.subcategory_vectors:
            self.subcategory_vectors[subcategory] = self.algebra.get_or_create(
                f"subcat_{subcategory}"
            )

        if category not in self.category_vectors:
            self.category_vectors[category] = self.algebra.get_or_create(
                f"cat_{category}"
            )

        # Track membership
        self.sku_to_subcategory[sku_id] = subcategory
        self.subcategory_to_category[subcategory] = category
        if sku_id not in self.subcategory_members[subcategory]:
            self.subcategory_members[subcategory].append(sku_id)
        if subcategory not in self.category_members[category]:
            self.category_members[category].append(subcategory)

    def _deterministic_vector(self, key: str) -> np.ndarray:
        """Generate a deterministic fallback vector from a key string."""
        seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        phases = rng.uniform(0, 2 * np.pi, self.algebra.dim)
        return np.exp(1j * phases)

    def encode_sku_profile(self, profile: SKUProfile) -> dict[str, np.ndarray]:
        """
        Encode a SKU profile at all hierarchy levels.

        Returns vectors for exact, subcategory, and category matching.
        """
        a = self.algebra

        # Velocity encoding
        vel = profile.avg_weekly_velocity
        if vel == 0:
            vel_vec = self.velocity_levels["dead"]
        elif vel < 2:
            vel_vec = self.velocity_levels["slow"]
        elif vel < 8:
            vel_vec = self.velocity_levels["moderate"]
        elif vel < 20:
            vel_vec = self.velocity_levels["fast"]
        else:
            vel_vec = self.velocity_levels["hot"]

        sku_vec = self.sku_vectors.get(
            profile.sku_id, self._deterministic_vector(f"sku:{profile.sku_id}")
        )
        subcat_vec = self.subcategory_vectors.get(
            profile.subcategory,
            self._deterministic_vector(f"subcat:{profile.subcategory}"),
        )
        cat_vec = self.category_vectors.get(
            profile.category,
            self._deterministic_vector(f"cat:{profile.category}"),
        )

        # Exact SKU encoding: entity ⊗ velocity
        exact = a.bind(
            a.bind(self.role_sku, sku_vec), a.bind(self.role_velocity, vel_vec)
        )

        # Subcategory encoding: subcategory ⊗ velocity
        subcat = a.bind(
            a.bind(self.role_subcategory, subcat_vec),
            a.bind(self.role_velocity, vel_vec),
        )

        # Category encoding: category ⊗ velocity
        cat = a.bind(
            a.bind(self.role_category, cat_vec), a.bind(self.role_velocity, vel_vec)
        )

        return {
            "exact": exact,
            "subcategory": subcat,
            "category": cat,
        }


# =============================================================================
# STORE AGENT
# =============================================================================


class StoreAgent:
    """
    Single store agent with inventory VSA stack.

    Maintains SKU profiles, detects dead stock, and publishes
    velocity primitives to the shared codebook.
    """

    def __init__(
        self,
        store_id: str,
        algebra: PhasorAlgebra,
        hierarchy: EntityHierarchy,
        dead_stock_config: DeadStockConfig = None,
    ):
        self.store_id = store_id
        self.algebra = algebra
        self.hierarchy = hierarchy

        # Inventory state
        self.profiles: dict[str, SKUProfile] = {}

        # Published primitives (shared with other agents)
        self.published_primitives: dict[str, np.ndarray] = {}

        # Dead stock configuration — use provided config or sensible defaults
        self.dead_stock_config = dead_stock_config or DeadStockConfig()

    def ingest_sku(
        self,
        sku_id: str,
        description: str,
        subcategory: str,
        category: str,
        stock: int,
        weekly_velocity: float,
        cost: float,
        price: float,
        days_since_last_sale: int,
    ):
        """Ingest or update a SKU profile."""
        # Register in hierarchy
        self.hierarchy.register_sku(sku_id, subcategory, category)

        margin_pct = (price - cost) / price if price > 0 else 0

        # Create vectors
        encodings = self.hierarchy.encode_sku_profile(
            SKUProfile(
                sku_id=sku_id,
                store_id=self.store_id,
                description=description,
                subcategory=subcategory,
                category=category,
                current_stock=stock,
                avg_weekly_velocity=weekly_velocity,
                cost=cost,
                price=price,
                margin_pct=margin_pct,
                days_since_last_sale=days_since_last_sale,
                vector=np.array([]),  # placeholder
                subcategory_vector=np.array([]),
                category_vector=np.array([]),
            )
        )

        profile = SKUProfile(
            sku_id=sku_id,
            store_id=self.store_id,
            description=description,
            subcategory=subcategory,
            category=category,
            current_stock=stock,
            avg_weekly_velocity=weekly_velocity,
            cost=cost,
            price=price,
            margin_pct=margin_pct,
            days_since_last_sale=days_since_last_sale,
            vector=encodings["exact"],
            subcategory_vector=encodings["subcategory"],
            category_vector=encodings["category"],
        )

        self.profiles[sku_id] = profile

        # Publish velocity primitives for active items
        if weekly_velocity > 0:
            self.published_primitives[f"{self.store_id}:{sku_id}"] = encodings["exact"]
            self.published_primitives[f"{self.store_id}:subcat:{subcategory}"] = (
                encodings["subcategory"]
            )
            self.published_primitives[f"{self.store_id}:cat:{category}"] = encodings[
                "category"
            ]

    def find_dead_stock(self) -> list[SKUProfile]:
        """Find all dead/slow stock items with remaining inventory."""
        dead = []
        for profile in self.profiles.values():
            if profile.current_stock <= 0:
                continue
            result = self.dead_stock_config.classify(
                days_since_last_sale=profile.days_since_last_sale,
                current_stock=profile.current_stock,
                unit_cost=profile.cost,
                velocity=profile.avg_weekly_velocity,
                category=profile.category,
            )
            if result.should_alert:
                dead.append(profile)

        # Sort by capital tied up (stock × cost), descending
        dead.sort(key=lambda p: p.current_stock * p.cost, reverse=True)
        return dead

    def find_active_demand(
        self, subcategory: str = None, category: str = None, min_velocity: float = 1.0
    ) -> list[SKUProfile]:
        """Find items with active demand, optionally filtered."""
        matches = []
        for profile in self.profiles.values():
            if profile.avg_weekly_velocity < min_velocity:
                continue
            if subcategory and profile.subcategory != subcategory:
                continue
            if category and profile.category != category:
                continue
            matches.append(profile)

        matches.sort(key=lambda p: p.avg_weekly_velocity, reverse=True)
        return matches


# =============================================================================
# TRANSFER MATCHER (The Orchestrator Feature)
# =============================================================================


class TransferMatcher:
    """
    Cross-store transfer matching engine.

    Takes dead stock from any store, searches all other stores
    for demand, and generates transfer recommendations.

    Uses the VSA hierarchy for matching:
    - Level 1: Exact SKU match (strongest signal)
    - Level 2: Subcategory match (same type of item)
    - Level 3: Category match (broad match, lower confidence)

    Uses the resonator for finding matches in the shared codebook,
    and the battery for confidence scoring.
    """

    def __init__(self, algebra: PhasorAlgebra, hierarchy: EntityHierarchy):
        self.algebra = algebra
        self.hierarchy = hierarchy
        self.agents: dict[str, StoreAgent] = {}

        # Shared codebook: all published primitives from all stores
        self.shared_codebook: dict[str, np.ndarray] = {}

        # Transfer history for learning (bounded)
        self.transfer_history: deque[dict] = deque(maxlen=5000)

        # Configuration
        self.clearance_discount = 0.50  # 50% markdown for clearance
        self.min_transfer_units = 3  # Don't transfer tiny quantities
        self.max_weeks_supply = 8  # Don't overship destination
        self.min_confidence = 0.05  # Minimum match confidence

    def register_agent(self, agent: StoreAgent):
        """Add a store agent to the network."""
        self.agents[agent.store_id] = agent
        self._rebuild_shared_codebook()

    def _rebuild_shared_codebook(self):
        """Rebuild shared codebook from all agents' published primitives."""
        self.shared_codebook = {}
        for agent in self.agents.values():
            self.shared_codebook.update(agent.published_primitives)

    def find_transfers(
        self, source_store_id: str, max_recommendations: int = 10
    ) -> list[TransferRecommendation]:
        """
        Find all transfer opportunities for dead stock at one store.

        Returns ranked recommendations with financial impact.
        """
        source_agent = self.agents.get(source_store_id)
        if not source_agent:
            return []

        dead_items = source_agent.find_dead_stock()
        if not dead_items:
            return []

        self._rebuild_shared_codebook()

        all_recommendations = []

        for dead_item in dead_items:
            # Search for demand at each hierarchy level
            matches = self._find_demand_matches(dead_item, source_store_id)

            for match in matches:
                rec = self._build_recommendation(dead_item, match)
                if (
                    rec
                    and rec.confidence >= self.min_confidence
                    and rec.net_benefit > 0
                ):
                    all_recommendations.append(rec)

        # Sort by net benefit (most money saved first)
        all_recommendations.sort(key=lambda r: -r.net_benefit)

        return all_recommendations[:max_recommendations]

    def find_all_transfers(
        self, max_per_store: int = 10
    ) -> dict[str, list[TransferRecommendation]]:
        """Find transfers across all stores."""
        results = {}
        for store_id in self.agents:
            recs = self.find_transfers(store_id, max_per_store)
            if recs:
                results[store_id] = recs
        return results

    def _find_demand_matches(
        self, dead_item: SKUProfile, source_store_id: str
    ) -> list[dict]:
        """
        Find demand matches for a dead item across all other stores.

        Searches at three hierarchy levels:
        1. Exact SKU — does another store sell this exact item?
        2. Subcategory — does another store sell similar items?
        3. Category — does another store have demand in this category?

        Uses Rust matrix_similarity for batch confidence scoring when available.
        """
        matches = []
        use_matrix = hasattr(self.algebra, "matrix_similarity")

        for store_id, agent in self.agents.items():
            if store_id == source_store_id:
                continue

            # Level 1: Exact SKU match
            if dead_item.sku_id in agent.profiles:
                dest_profile = agent.profiles[dead_item.sku_id]
                if dest_profile.avg_weekly_velocity > 0:
                    confidence = self._compute_match_confidence(
                        dead_item.vector, dest_profile.vector
                    )
                    matches.append(
                        {
                            "level": MatchLevel.EXACT_SKU,
                            "dest_store": store_id,
                            "dest_profile": dest_profile,
                            "confidence": confidence,
                            "pattern": (
                                f"Exact SKU match — {store_id} sells "
                                f"{dest_profile.avg_weekly_velocity:.1f}/week"
                            ),
                        }
                    )

            # Level 2: Subcategory match — batch similarity with matrix op
            subcat_matches = agent.find_active_demand(
                subcategory=dead_item.subcategory, min_velocity=1.0
            )
            # Filter out exact SKU (already caught at Level 1)
            subcat_matches = [p for p in subcat_matches if p.sku_id != dead_item.sku_id]

            if subcat_matches and use_matrix and len(subcat_matches) >= 2:
                # Batch: compute all subcategory confidences in one Rust call
                dest_vecs = np.array([p.subcategory_vector for p in subcat_matches])
                scores = self.algebra.matrix_similarity(
                    dead_item.subcategory_vector, dest_vecs
                )
                for dest_profile, conf in zip(subcat_matches, scores):
                    matches.append(
                        {
                            "level": MatchLevel.SUBCATEGORY,
                            "dest_store": store_id,
                            "dest_profile": dest_profile,
                            "confidence": float(conf),
                            "pattern": (
                                f"Subcategory match ({dead_item.subcategory}) — "
                                f"{store_id} sells {dest_profile.description} "
                                f"at {dest_profile.avg_weekly_velocity:.1f}/week"
                            ),
                        }
                    )
            else:
                for dest_profile in subcat_matches:
                    confidence = self._compute_match_confidence(
                        dead_item.subcategory_vector, dest_profile.subcategory_vector
                    )
                    matches.append(
                        {
                            "level": MatchLevel.SUBCATEGORY,
                            "dest_store": store_id,
                            "dest_profile": dest_profile,
                            "confidence": confidence,
                            "pattern": (
                                f"Subcategory match ({dead_item.subcategory}) — "
                                f"{store_id} sells {dest_profile.description} "
                                f"at {dest_profile.avg_weekly_velocity:.1f}/week"
                            ),
                        }
                    )

            # Level 3: Category match (only if no better matches found)
            if not any(
                m["dest_store"] == store_id
                and m["level"] in (MatchLevel.EXACT_SKU, MatchLevel.SUBCATEGORY)
                for m in matches
            ):
                cat_matches = agent.find_active_demand(
                    category=dead_item.category,
                    min_velocity=2.0,  # Higher threshold for broad match
                )[
                    :3
                ]  # Limit broad matches

                if cat_matches and use_matrix and len(cat_matches) >= 2:
                    dest_vecs = np.array([p.category_vector for p in cat_matches])
                    scores = self.algebra.matrix_similarity(
                        dead_item.category_vector, dest_vecs
                    )
                    for dest_profile, conf in zip(cat_matches, scores):
                        matches.append(
                            {
                                "level": MatchLevel.CATEGORY,
                                "dest_store": store_id,
                                "dest_profile": dest_profile,
                                "confidence": float(conf) * 0.7,  # Penalize broad match
                                "pattern": (
                                    f"Category match ({dead_item.category}) — "
                                    f"{store_id} sells {dest_profile.description} "
                                    f"at {dest_profile.avg_weekly_velocity:.1f}/week"
                                ),
                            }
                        )
                else:
                    for dest_profile in cat_matches:
                        confidence = self._compute_match_confidence(
                            dead_item.category_vector, dest_profile.category_vector
                        )
                        matches.append(
                            {
                                "level": MatchLevel.CATEGORY,
                                "dest_store": store_id,
                                "dest_profile": dest_profile,
                                "confidence": confidence * 0.7,  # Penalize broad match
                                "pattern": (
                                    f"Category match ({dead_item.category}) — "
                                    f"{store_id} sells {dest_profile.description} "
                                    f"at {dest_profile.avg_weekly_velocity:.1f}/week"
                                ),
                            }
                        )

        # Sort: exact > subcategory > category, then by confidence
        level_priority = {
            MatchLevel.EXACT_SKU: 0,
            MatchLevel.SUBCATEGORY: 1,
            MatchLevel.CATEGORY: 2,
        }
        matches.sort(key=lambda m: (level_priority[m["level"]], -m["confidence"]))

        return matches

    def _compute_match_confidence(
        self, source_vec: np.ndarray, dest_vec: np.ndarray
    ) -> float:
        """
        Compute match confidence using VSA similarity.

        This is where the algebra does the matching — not string
        comparison, not rule-based, but geometric similarity in
        the hyperdimensional space.
        """
        return self.algebra.similarity(source_vec, dest_vec)

    def _build_recommendation(
        self, dead_item: SKUProfile, match: dict
    ) -> TransferRecommendation | None:
        """Build a complete transfer recommendation with financials."""
        dest_profile = match["dest_profile"]

        # How many units to transfer?
        # Don't transfer more than the destination can sell in max_weeks
        dest_capacity = int(dest_profile.avg_weekly_velocity * self.max_weeks_supply)
        # Don't transfer more than we have
        units = min(
            dead_item.current_stock, max(dest_capacity, self.min_transfer_units)
        )

        if units < self.min_transfer_units:
            return None

        # Financials
        unit_cost = dead_item.cost
        unit_price = dead_item.price
        margin_per_unit = unit_price - unit_cost

        clearance_recovery = units * unit_price * self.clearance_discount
        transfer_recovery = units * unit_price  # Full price at destination
        net_benefit = transfer_recovery - clearance_recovery

        # Estimated sell-through time
        weeks_to_sell = (
            units / dest_profile.avg_weekly_velocity
            if dest_profile.avg_weekly_velocity > 0
            else float("inf")
        )

        return TransferRecommendation(
            source_store=dead_item.store_id,
            source_sku=dead_item.sku_id,
            source_description=dead_item.description,
            source_stock=dead_item.current_stock,
            source_days_dead=dead_item.days_since_last_sale,
            dest_store=match["dest_store"],
            dest_sku=dest_profile.sku_id,
            dest_description=dest_profile.description,
            dest_weekly_velocity=dest_profile.avg_weekly_velocity,
            dest_current_stock=dest_profile.current_stock,
            match_level=match["level"],
            match_confidence=match["confidence"],
            units_to_transfer=units,
            unit_cost=unit_cost,
            unit_price=unit_price,
            margin_per_unit=margin_per_unit,
            clearance_recovery=clearance_recovery,
            transfer_recovery=transfer_recovery,
            net_benefit=net_benefit,
            demand_pattern=match["pattern"],
            estimated_weeks_to_sell=weeks_to_sell,
        )

    def network_summary(self) -> dict:
        """Summary of the entire store network."""
        total_dead_capital = 0
        total_potential_recovery = 0

        all_recs = self.find_all_transfers()

        for store_id, recs in all_recs.items():
            for rec in recs:
                total_dead_capital += rec.source_stock * rec.unit_cost
                total_potential_recovery += rec.net_benefit

        return {
            "n_stores": len(self.agents),
            "n_skus_total": sum(len(a.profiles) for a in self.agents.values()),
            "n_dead_items": sum(len(a.find_dead_stock()) for a in self.agents.values()),
            "total_transfer_opportunities": sum(len(r) for r in all_recs.values()),
            "total_potential_recovery": total_potential_recovery,
            "recommendations_by_store": {
                sid: len(recs) for sid, recs in all_recs.items()
            },
        }


# =============================================================================
# TEST WITH REALISTIC HARDWARE STORE DATA
# =============================================================================


def run_transfer_test():
    """
    Test with realistic hardware store inventory data.

    Simulates a 3-store chain where each store has different
    demand patterns based on location/customer base.
    """
    print("=" * 70)
    print("CROSS-STORE TRANSFER MATCHING TEST")
    print("=" * 70)
    print()

    algebra = PhasorAlgebra(dim=4096, seed=42)
    hierarchy = EntityHierarchy(algebra)
    matcher = TransferMatcher(algebra, hierarchy)

    # =========================================================
    # STORE 1: Suburban location, homeowner-focused
    # Has dead stock in contractor-grade items
    # =========================================================
    store1 = StoreAgent("Store_1_Suburban", algebra, hierarchy)

    store1_inventory = [
        # Dead stock — contractor items that homeowners don't buy
        (
            "SKU-7742",
            "Stainless Cabinet Pulls (25pk)",
            "Cabinet Hardware",
            "Hardware",
            47,
            0.0,
            12.50,
            24.99,
            95,
        ),
        (
            "SKU-3301",
            "Commercial Grade Deadbolt",
            "Locks",
            "Hardware",
            23,
            0.0,
            34.00,
            59.99,
            120,
        ),
        (
            "SKU-5510",
            "3/4 inch Copper Pipe 10ft",
            "Copper Pipe",
            "Plumbing",
            85,
            0.2,
            8.75,
            15.99,
            45,
        ),
        (
            "SKU-8820",
            "Concrete Anchors 50pk",
            "Anchors",
            "Fasteners",
            30,
            0.1,
            18.00,
            32.99,
            78,
        ),
        # Active items — homeowner staples
        (
            "SKU-1100",
            "Interior Latex Paint Gallon",
            "Interior Paint",
            "Paint",
            120,
            15.0,
            22.00,
            38.99,
            2,
        ),
        (
            "SKU-1205",
            "Paint Roller Kit",
            "Paint Supplies",
            "Paint",
            60,
            8.0,
            6.50,
            12.99,
            1,
        ),
        (
            "SKU-2200",
            "LED Bulb 4-pack",
            "Light Bulbs",
            "Electrical",
            200,
            20.0,
            4.50,
            9.99,
            0,
        ),
        (
            "SKU-4400",
            "Garden Hose 50ft",
            "Garden Hose",
            "Garden",
            40,
            5.0,
            15.00,
            29.99,
            3,
        ),
    ]

    for item in store1_inventory:
        store1.ingest_sku(*item)

    matcher.register_agent(store1)

    # =========================================================
    # STORE 2: Near construction corridor, contractor-heavy
    # High demand for the items Store 1 can't sell
    # =========================================================
    store2 = StoreAgent("Store_2_Contractor", algebra, hierarchy)

    store2_inventory = [
        # High velocity contractor items
        (
            "SKU-7742",
            "Stainless Cabinet Pulls (25pk)",
            "Cabinet Hardware",
            "Hardware",
            8,
            12.0,
            12.50,
            24.99,
            1,
        ),  # Same SKU, high velocity!
        (
            "SKU-3301",
            "Commercial Grade Deadbolt",
            "Locks",
            "Hardware",
            5,
            6.0,
            34.00,
            59.99,
            2,
        ),  # Same SKU, selling well
        (
            "SKU-5515",
            "1 inch Copper Pipe 10ft",
            "Copper Pipe",
            "Plumbing",
            30,
            8.0,
            11.25,
            19.99,
            1,
        ),  # Different SKU, same subcat
        (
            "SKU-8825",
            "Wedge Anchors 25pk",
            "Anchors",
            "Fasteners",
            15,
            4.0,
            22.00,
            39.99,
            3,
        ),  # Different SKU, same subcat
        # Moderate homeowner items
        (
            "SKU-1100",
            "Interior Latex Paint Gallon",
            "Interior Paint",
            "Paint",
            30,
            3.0,
            22.00,
            38.99,
            5,
        ),
        (
            "SKU-6600",
            "Framing Lumber 2x4x8",
            "Dimensional Lumber",
            "Lumber",
            500,
            50.0,
            3.25,
            5.99,
            0,
        ),
        (
            "SKU-6610",
            "Treated Lumber 2x4x8",
            "Treated Lumber",
            "Lumber",
            300,
            30.0,
            4.50,
            8.49,
            0,
        ),
    ]

    for item in store2_inventory:
        store2.ingest_sku(*item)

    matcher.register_agent(store2)

    # =========================================================
    # STORE 3: Rural location, mixed customer base
    # Some unique demand, some overlap
    # =========================================================
    store3 = StoreAgent("Store_3_Rural", algebra, hierarchy)

    store3_inventory = [
        # Strong in outdoor/garden
        (
            "SKU-4400",
            "Garden Hose 50ft",
            "Garden Hose",
            "Garden",
            15,
            2.0,
            15.00,
            29.99,
            4,
        ),
        (
            "SKU-4410",
            "Sprinkler System Kit",
            "Irrigation",
            "Garden",
            10,
            3.0,
            45.00,
            79.99,
            2,
        ),
        # Some hardware demand
        (
            "SKU-7750",
            "Brushed Nickel Cabinet Knobs (10pk)",
            "Cabinet Hardware",
            "Hardware",
            20,
            3.0,
            8.00,
            16.99,
            5,
        ),  # Different SKU, same subcat
        # Dead stock — wrong items for rural
        (
            "SKU-9900",
            "Smart Doorbell Camera",
            "Smart Home",
            "Electrical",
            12,
            0.0,
            85.00,
            149.99,
            150,
        ),
        (
            "SKU-9910",
            "WiFi Thermostat",
            "Smart Home",
            "Electrical",
            8,
            0.0,
            65.00,
            119.99,
            130,
        ),
        # Active basics
        (
            "SKU-2200",
            "LED Bulb 4-pack",
            "Light Bulbs",
            "Electrical",
            100,
            10.0,
            4.50,
            9.99,
            1,
        ),
        (
            "SKU-6600",
            "Framing Lumber 2x4x8",
            "Dimensional Lumber",
            "Lumber",
            200,
            15.0,
            3.25,
            5.99,
            0,
        ),
    ]

    for item in store3_inventory:
        store3.ingest_sku(*item)

    matcher.register_agent(store3)

    # =========================================================
    # RUN TRANSFER MATCHING
    # =========================================================

    print(f"Network: {len(matcher.agents)} stores")
    print(f"Shared codebook: {len(matcher.shared_codebook)} primitives")
    print()

    # Find dead stock at each store
    for store_id, agent in matcher.agents.items():
        dead = agent.find_dead_stock()
        if dead:
            dead_capital = sum(p.current_stock * p.cost for p in dead)
            print(
                f"{store_id}: {len(dead)} dead items, " f"${dead_capital:,.2f} tied up"
            )
            for item in dead:
                print(
                    f"  - {item.sku_id} ({item.description}): "
                    f"{item.current_stock} units, "
                    f"dead {item.days_since_last_sale} days, "
                    f"${item.current_stock * item.cost:,.2f} capital"
                )

    print()
    print("-" * 70)
    print("TRANSFER RECOMMENDATIONS")
    print("-" * 70)
    print()

    total_benefit = 0
    total_clearance = 0
    total_transfer = 0

    all_recs = matcher.find_all_transfers(max_per_store=10)

    for store_id, recs in all_recs.items():
        print(f"=== {store_id} ===")
        print()
        for i, rec in enumerate(recs, 1):
            print(f"Recommendation #{i}:")
            print(rec.summary())
            print()
            total_benefit += rec.net_benefit
            total_clearance += rec.clearance_recovery
            total_transfer += rec.transfer_recovery

    # Network summary
    print("=" * 70)
    print("NETWORK SUMMARY")
    print("=" * 70)
    print()

    summary = matcher.network_summary()
    print(f"Stores: {summary['n_stores']}")
    print(f"Total SKUs: {summary['n_skus_total']}")
    print(f"Dead stock items: {summary['n_dead_items']}")
    print(f"Transfer opportunities: {summary['total_transfer_opportunities']}")
    print()
    print(f"If all dead stock cleared at 50% markdown: " f"${total_clearance:,.2f}")
    print(f"If transferred to selling stores at full price: " f"${total_transfer:,.2f}")
    print(f"NET BENEFIT OF TRANSFERS: ${total_benefit:,.2f}")
    print()

    if total_clearance > 0:
        improvement = (total_benefit / total_clearance) * 100
        print(f"Transfer recovery vs clearance: " f"+{improvement:.0f}% better")

    return all_recs


if __name__ == "__main__":
    recs = run_transfer_test()
