"""
Dorian Persistence Layer - Supabase + pgvector

Saves and loads the geometric knowledge moat.
Every confirmed pattern makes the system smarter for the next customer.

CRITICAL PRIVACY RULES:
- NO PII ever saved (no emails, no customer names, no store names)
- NO raw SKUs (anonymize to category: "lumber", "electrical", "plumbing")
- NO exact quantities (just pattern: "negative", "overstock", "low")
- ONLY save confirmed patterns (user said "yes this is correct")
- Industry is optional but helpful for segmentation

Usage:
    from sentinel_engine.dorian.persistence import DorianPersistence

    persistence = DorianPersistence()

    # Save a confirmed pattern
    fact_id = persistence.save_fact(
        subject="hardware_store",
        predicate="has_pattern",
        object="receiving_gap",
        vector=encoded_vector.tolist(),
        industry="hardware",
        pattern_type="receiving_gap",
        sku_category="lumber"
    )

    # Find similar patterns
    similar = persistence.find_similar(query_vector, limit=10, threshold=0.8)

    # Increment confirmation when users verify
    persistence.increment_confirmation(fact_id)
"""

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# ANONYMIZATION HELPERS
# =============================================================================

# SKU category mappings for anonymization
SKU_CATEGORY_PATTERNS = {
    # Building materials
    "lumber": [
        r"lumber",
        r"2x4",
        r"2x6",
        r"plywood",
        r"osb",
        r"board",
        r"timber",
        r"stud",
    ],
    "electrical": [
        r"wire",
        r"outlet",
        r"switch",
        r"breaker",
        r"conduit",
        r"romex",
        r"volt",
        r"amp",
    ],
    "plumbing": [
        r"pipe",
        r"pvc",
        r"fitting",
        r"valve",
        r"faucet",
        r"drain",
        r"copper",
        r"cpvc",
    ],
    "hardware": [
        r"screw",
        r"nail",
        r"bolt",
        r"nut",
        r"washer",
        r"anchor",
        r"hinge",
        r"latch",
    ],
    "paint": [r"paint", r"stain", r"primer", r"brush", r"roller", r"gallon", r"quart"],
    "outdoor": [
        r"outdoor",
        r"garden",
        r"lawn",
        r"plant",
        r"soil",
        r"mulch",
        r"fence",
        r"deck",
    ],
    # Retail categories
    "grocery": [
        r"food",
        r"beverage",
        r"snack",
        r"cereal",
        r"dairy",
        r"produce",
        r"meat",
    ],
    "pharmacy": [r"med", r"vitamin", r"supplement", r"otc", r"rx", r"health", r"drug"],
    "apparel": [
        r"shirt",
        r"pants",
        r"dress",
        r"shoe",
        r"jacket",
        r"coat",
        r"clothing",
    ],
    "electronics": [
        r"phone",
        r"cable",
        r"charger",
        r"battery",
        r"usb",
        r"hdmi",
        r"adapter",
    ],
}

# Industry detection patterns
INDUSTRY_PATTERNS = {
    "hardware": [
        r"hardware",
        r"lumber",
        r"building",
        r"home\s*improvement",
        r"contractor",
    ],
    "grocery": [r"grocery", r"supermarket", r"food", r"market"],
    "pharmacy": [r"pharmacy", r"drug", r"rx", r"cvs", r"walgreens"],
    "retail": [r"retail", r"store", r"shop", r"mart"],
    "restaurant": [r"restaurant", r"food\s*service", r"dining", r"cafe"],
}


def anonymize_sku_to_category(sku: str) -> str:
    """
    Anonymize a raw SKU to a category.

    Example:
        "2X4-LUMBER-8FT-TREATED" -> "lumber"
        "ROMEX-14-2-NM" -> "electrical"
    """
    if not sku:
        return "general"

    sku_lower = sku.lower()

    for category, patterns in SKU_CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, sku_lower):
                return category

    return "general"


def detect_industry(store_name: str | None, context: str | None = None) -> str | None:
    """
    Detect industry from store name or context.

    Example:
        "ABC Hardware Supply" -> "hardware"
        "Joe's Pharmacy" -> "pharmacy"
    """
    text = " ".join(filter(None, [store_name, context])).lower()

    if not text:
        return None

    for industry, patterns in INDUSTRY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                return industry

    return None


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class PersistedFact:
    """A fact loaded from persistence."""

    id: str
    subject: str
    predicate: str
    object: str
    vector: list[float] | None
    confidence: float
    confirmations: int
    industry: str | None
    pattern_type: str | None
    sku_category: str | None
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


# =============================================================================
# PERSISTENCE LAYER
# =============================================================================


class DorianPersistence:
    """
    Persistence layer for Dorian facts using Supabase + pgvector.

    This class handles saving and loading anonymized patterns to build
    the data moat. Every confirmed pattern makes the system smarter.
    """

    def __init__(
        self,
        supabase_url: str | None = None,
        supabase_key: str | None = None,
    ):
        """
        Initialize the persistence layer.

        Args:
            supabase_url: Supabase project URL (or SUPABASE_URL env var)
            supabase_key: Supabase service key (or SUPABASE_SERVICE_KEY env var)
        """
        self._url = supabase_url or os.environ.get("SUPABASE_URL")
        self._key = supabase_key or os.environ.get("SUPABASE_SERVICE_KEY")
        self._client = None
        self._enabled = bool(self._url and self._key)

        if not self._enabled:
            logger.info("Dorian persistence disabled (no Supabase credentials)")

    @property
    def client(self):
        """Lazy-load Supabase client."""
        if self._client is None and self._enabled:
            try:
                from supabase import create_client

                self._client = create_client(self._url, self._key)
                logger.info("Dorian persistence connected to Supabase")
            except ImportError:
                logger.warning("supabase package not installed, persistence disabled")
                self._enabled = False
            except Exception as e:
                logger.error(f"Failed to connect to Supabase: {e}")
                self._enabled = False
        return self._client

    @property
    def is_enabled(self) -> bool:
        """Check if persistence is enabled and available."""
        return self._enabled and self.client is not None

    def save_fact(
        self,
        subject: str,
        predicate: str,
        object: str,
        vector: list[float] | None = None,
        confidence: float = 1.0,
        industry: str | None = None,
        pattern_type: str | None = None,
        sku_category: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """
        Save a fact to the moat.

        If a very similar fact exists (vector similarity > 0.95), increments
        the confirmation count instead of creating a duplicate.

        Args:
            subject: Anonymized subject (e.g., "hardware_store")
            predicate: Relationship (e.g., "has_pattern")
            object: Value (e.g., "receiving_gap")
            vector: VSA-encoded vector (512 dimensions)
            confidence: Confidence score 0.0-1.0
            industry: Industry category (optional)
            pattern_type: Pattern classification (optional)
            sku_category: Anonymized SKU category (optional)
            metadata: Additional metadata (optional)

        Returns:
            fact_id if saved, None if persistence is disabled
        """
        if not self.is_enabled:
            logger.debug("Persistence disabled, skipping save")
            return None

        try:
            # Use the upsert function if vector is provided
            if vector is not None:
                result = self.client.rpc(
                    "upsert_dorian_fact",
                    {
                        "p_subject": subject,
                        "p_predicate": predicate,
                        "p_object": object,
                        "p_vector": vector,
                        "p_confidence": confidence,
                        "p_industry": industry,
                        "p_pattern_type": pattern_type,
                        "p_sku_category": sku_category,
                        "p_metadata": metadata or {},
                    },
                ).execute()

                fact_id = result.data
                logger.info(f"Saved/updated fact: {fact_id}")
                return fact_id
            else:
                # Direct insert without vector (no similarity check)
                result = (
                    self.client.table("dorian_facts")
                    .insert(
                        {
                            "subject": subject,
                            "predicate": predicate,
                            "object": object,
                            "confidence": confidence,
                            "industry": industry,
                            "pattern_type": pattern_type,
                            "sku_category": sku_category,
                            "metadata": metadata or {},
                        }
                    )
                    .execute()
                )

                if result.data:
                    fact_id = result.data[0]["id"]
                    logger.info(f"Saved fact (no vector): {fact_id}")
                    return fact_id

        except Exception as e:
            logger.error(f"Failed to save fact: {e}")

        return None

    def increment_confirmation(self, fact_id: str) -> bool:
        """
        Increment confirmation count when users verify a pattern.

        Args:
            fact_id: UUID of the fact to confirm

        Returns:
            True if successful, False otherwise
        """
        if not self.is_enabled:
            return False

        try:
            self.client.rpc(
                "increment_fact_confirmation", {"fact_uuid": fact_id}
            ).execute()
            logger.debug(f"Incremented confirmation for fact: {fact_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to increment confirmation: {e}")
            return False

    def find_similar(
        self,
        vector: list[float],
        limit: int = 10,
        threshold: float = 0.8,
        industry: str | None = None,
        pattern_type: str | None = None,
    ) -> list[PersistedFact]:
        """
        Find similar facts by vector similarity.

        Args:
            vector: Query vector (512 dimensions)
            limit: Maximum number of results
            threshold: Minimum similarity threshold (0.0-1.0)
            industry: Filter by industry (optional)
            pattern_type: Filter by pattern type (optional)

        Returns:
            List of similar facts ordered by similarity
        """
        if not self.is_enabled:
            return []

        try:
            result = self.client.rpc(
                "find_similar_facts",
                {
                    "query_vector": vector,
                    "similarity_threshold": threshold,
                    "max_results": limit,
                    "filter_industry": industry,
                    "filter_pattern_type": pattern_type,
                },
            ).execute()

            facts = []
            for row in result.data or []:
                facts.append(
                    PersistedFact(
                        id=row["fact_id"],
                        subject=row["subject"],
                        predicate=row["predicate"],
                        object=row["object"],
                        vector=None,  # Not returned by similarity search
                        confidence=row.get("similarity", 1.0),
                        confirmations=row.get("confirmations", 1),
                        industry=row.get("industry"),
                        pattern_type=row.get("pattern_type"),
                        sku_category=None,
                        metadata={},
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                    )
                )

            logger.debug(f"Found {len(facts)} similar facts")
            return facts

        except Exception as e:
            logger.error(f"Failed to find similar facts: {e}")
            return []

    def load_all(self, limit: int = 100000) -> list[PersistedFact]:
        """
        Load all active facts from persistence.

        Use sparingly - for initial loading or batch operations.

        Args:
            limit: Maximum number of facts to load

        Returns:
            List of all active facts
        """
        if not self.is_enabled:
            return []

        try:
            result = (
                self.client.table("dorian_facts")
                .select("*")
                .eq("status", "active")
                .order("confirmations", desc=True)
                .limit(limit)
                .execute()
            )

            facts = []
            for row in result.data or []:
                facts.append(self._row_to_fact(row))

            logger.info(f"Loaded {len(facts)} facts from persistence")
            return facts

        except Exception as e:
            logger.error(f"Failed to load facts: {e}")
            return []

    def load_by_industry(
        self, industry: str, limit: int = 10000
    ) -> list[PersistedFact]:
        """
        Load facts for a specific industry.

        Args:
            industry: Industry to filter by
            limit: Maximum number of facts

        Returns:
            List of facts for the industry
        """
        if not self.is_enabled:
            return []

        try:
            result = (
                self.client.table("dorian_facts")
                .select("*")
                .eq("status", "active")
                .eq("industry", industry)
                .order("confirmations", desc=True)
                .limit(limit)
                .execute()
            )

            facts = []
            for row in result.data or []:
                facts.append(self._row_to_fact(row))

            logger.info(f"Loaded {len(facts)} facts for industry: {industry}")
            return facts

        except Exception as e:
            logger.error(f"Failed to load facts for industry {industry}: {e}")
            return []

    def load_by_pattern_type(
        self, pattern_type: str, limit: int = 10000
    ) -> list[PersistedFact]:
        """
        Load facts for a specific pattern type.

        Args:
            pattern_type: Pattern type to filter by
            limit: Maximum number of facts

        Returns:
            List of facts for the pattern type
        """
        if not self.is_enabled:
            return []

        try:
            result = (
                self.client.table("dorian_facts")
                .select("*")
                .eq("status", "active")
                .eq("pattern_type", pattern_type)
                .order("confirmations", desc=True)
                .limit(limit)
                .execute()
            )

            facts = []
            for row in result.data or []:
                facts.append(self._row_to_fact(row))

            logger.info(f"Loaded {len(facts)} facts for pattern: {pattern_type}")
            return facts

        except Exception as e:
            logger.error(f"Failed to load facts for pattern {pattern_type}: {e}")
            return []

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the fact store.

        Returns:
            Dictionary with counts, top patterns, etc.
        """
        if not self.is_enabled:
            return {"enabled": False}

        try:
            # Get pattern stats
            pattern_result = (
                self.client.table("dorian_pattern_stats").select("*").execute()
            )

            # Get industry stats
            industry_result = (
                self.client.table("dorian_industry_stats").select("*").execute()
            )

            # Count total facts
            count_result = (
                self.client.table("dorian_facts")
                .select("id", count="exact")
                .eq("status", "active")
                .execute()
            )

            return {
                "enabled": True,
                "total_facts": count_result.count or 0,
                "patterns": pattern_result.data or [],
                "industries": industry_result.data or [],
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"enabled": True, "error": str(e)}

    def _row_to_fact(self, row: dict) -> PersistedFact:
        """Convert a database row to a PersistedFact."""
        return PersistedFact(
            id=row["id"],
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            vector=row.get("vector"),
            confidence=row.get("confidence", 1.0),
            confirmations=row.get("confirmations", 1),
            industry=row.get("industry"),
            pattern_type=row.get("pattern_type"),
            sku_category=row.get("sku_category"),
            metadata=row.get("metadata", {}),
            created_at=(
                datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
                if row.get("created_at")
                else datetime.now()
            ),
            updated_at=(
                datetime.fromisoformat(row["updated_at"].replace("Z", "+00:00"))
                if row.get("updated_at")
                else datetime.now()
            ),
        )


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_persistence: DorianPersistence | None = None


def get_persistence() -> DorianPersistence:
    """Get or create the singleton persistence instance."""
    global _persistence
    if _persistence is None:
        _persistence = DorianPersistence()
    return _persistence
