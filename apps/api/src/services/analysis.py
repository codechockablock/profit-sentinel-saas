"""
Analysis Service.

Runs VSA-based profit leak detection.
"""

import logging
import time
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service for VSA-based profit leak analysis."""

    # Analysis primitives
    PRIMITIVES = [
        "low_stock",
        "high_margin_leak",
        "dead_item",
        "negative_inventory"
    ]

    def __init__(self):
        """Initialize analysis service with VSA engine."""
        # Import here to avoid circular imports and allow lazy loading
        try:
            from sentinel_engine import bundle_pos_facts, query_bundle
            self._bundle_pos_facts = bundle_pos_facts
            self._query_bundle = query_bundle
            self._engine_available = True
        except ImportError:
            logger.warning("Sentinel engine not available, using mock analysis")
            self._engine_available = False

    def analyze(self, rows: List[Dict]) -> Dict:
        """
        Analyze POS data for profit leaks.

        Args:
            rows: List of row dictionaries from POS data

        Returns:
            Dictionary of leak types with top items and scores
        """
        if not self._engine_available:
            return self._mock_analysis()

        # Bundle facts
        bundle_start = time.time()
        bundle = self._bundle_pos_facts(rows)
        logger.info(f"Bundled facts in {time.time() - bundle_start:.2f}s")

        # Query each primitive
        leaks = {}
        for primitive in self.PRIMITIVES:
            query_start = time.time()
            items, scores = self._query_bundle(bundle, primitive)
            leaks[primitive] = {
                "top_items": items[:20],
                "scores": [float(s) for s in scores[:20]]
            }
            logger.info(f"Queried {primitive} in {time.time() - query_start:.2f}s")

        return leaks

    def _mock_analysis(self) -> Dict:
        """Return mock analysis results when engine is unavailable."""
        return {
            primitive: {
                "top_items": [],
                "scores": []
            }
            for primitive in self.PRIMITIVES
        }
