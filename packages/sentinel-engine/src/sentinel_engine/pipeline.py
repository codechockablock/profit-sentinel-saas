"""
engine/pipeline.py - Tiered Inference Pipeline

Implements a multi-stage pipeline for efficient large-scale inference:

STAGE 1: Statistical Pre-Filter
    Fast statistical checks (z-scores, percentiles, thresholds)
    to quickly identify candidates for deeper analysis.
    O(n) per entity, no VSA operations.

STAGE 2: VSA Candidate Screening
    Approximate nearest neighbor search using FAISS.
    O(log n) per candidate with index.

STAGE 3: VSA Deep Analysis
    Full resonator analysis on top candidates.
    O(d × k × iters) where k << n.

This tiered approach reduces the effective workload:
- 1M entities → ~10k candidates (Stage 1) → ~100 deep analyses (Stage 2+3)
- 100x reduction in resonator calls
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline stage identifiers."""

    PREFILTER = "prefilter"
    SCREENING = "screening"
    DEEP_ANALYSIS = "deep_analysis"


@dataclass
class StageResult:
    """Result from a pipeline stage."""

    stage: PipelineStage
    candidates: list[str]  # Entity IDs that passed
    scores: dict[str, float]  # Entity → score
    elapsed_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Complete pipeline result."""

    entity_id: str
    detected_anomalies: list[str]
    confidence: float
    root_causes: list[dict[str, Any]]
    stage_results: list[StageResult]
    total_elapsed_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PrefilterRule:
    """Statistical prefilter rule."""

    name: str
    field: str
    operator: str  # <, >, <=, >=, ==, between, zscore
    threshold: float
    secondary_threshold: float | None = None  # For 'between'

    def evaluate(self, value: float, stats: dict | None = None) -> bool:
        """Evaluate rule against value."""
        if self.operator == "<":
            return value < self.threshold
        elif self.operator == ">":
            return value > self.threshold
        elif self.operator == "<=":
            return value <= self.threshold
        elif self.operator == ">=":
            return value >= self.threshold
        elif self.operator == "==":
            return abs(value - self.threshold) < 1e-6
        elif self.operator == "between":
            return self.threshold <= value <= (self.secondary_threshold or float("inf"))
        elif self.operator == "zscore" and stats:
            mean = stats.get("mean", 0)
            std = stats.get("std", 1)
            zscore = abs(value - mean) / (std + 1e-8)
            return zscore > self.threshold
        return False


class StatisticalPrefilter:
    """Stage 1: Fast statistical prefiltering.

    Uses simple threshold checks, z-scores, and percentile ranks
    to quickly identify anomaly candidates.
    """

    def __init__(self):
        self.rules: list[PrefilterRule] = []
        self.field_stats: dict[str, dict[str, float]] = {}

    def add_rule(
        self,
        name: str,
        field: str,
        operator: str,
        threshold: float,
        secondary: float | None = None,
    ) -> None:
        """Add a prefilter rule."""
        self.rules.append(
            PrefilterRule(
                name=name,
                field=field,
                operator=operator,
                threshold=threshold,
                secondary_threshold=secondary,
            )
        )

    def compute_statistics(self, data: list[dict[str, Any]]) -> None:
        """Compute field statistics for z-score rules."""
        import numpy as np

        # Collect values by field
        field_values: dict[str, list[float]] = {}
        for record in data:
            for key, value in record.items():
                if isinstance(value, (int, float)):
                    if key not in field_values:
                        field_values[key] = []
                    field_values[key].append(float(value))

        # Compute stats
        for field, values in field_values.items():
            arr = np.array(values)
            self.field_stats[field] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "p95": float(np.percentile(arr, 95)),
            }

    def filter(self, data: list[dict[str, Any]]) -> StageResult:
        """Run prefilter on data.

        Args:
            data: List of records with entity_id field

        Returns:
            StageResult with candidates
        """
        start = time.time()
        candidates = []
        scores = {}

        for record in data:
            entity_id = record.get("entity_id", record.get("sku", str(id(record))))
            triggered_rules = []

            for rule in self.rules:
                value = record.get(rule.field)
                if value is None:
                    continue

                stats = self.field_stats.get(rule.field)
                if rule.evaluate(float(value), stats):
                    triggered_rules.append(rule.name)

            if triggered_rules:
                candidates.append(entity_id)
                scores[entity_id] = len(triggered_rules) / len(self.rules)

        elapsed = (time.time() - start) * 1000

        return StageResult(
            stage=PipelineStage.PREFILTER,
            candidates=candidates,
            scores=scores,
            elapsed_ms=elapsed,
            metadata={
                "total_records": len(data),
                "rules_applied": len(self.rules),
            },
        )


class TieredPipeline:
    """Multi-stage inference pipeline.

    Combines statistical prefiltering with VSA analysis for
    efficient large-scale anomaly detection.

    Example:
        pipeline = TieredPipeline()

        # Configure prefilter
        pipeline.prefilter.add_rule("low_qty", "quantity", "<", 10)
        pipeline.prefilter.add_rule("high_margin", "margin", "<", 0.1)

        # Set VSA components
        pipeline.set_resonator(resonator)
        pipeline.set_primitive_loader(loader)

        # Run pipeline
        results = pipeline.run(data)
    """

    def __init__(
        self,
        max_prefilter_candidates: int = 10000,
        max_deep_analysis: int = 100,
        parallel_workers: int = 4,
    ):
        """Initialize pipeline.

        Args:
            max_prefilter_candidates: Max candidates from Stage 1
            max_deep_analysis: Max entities for Stage 3
            parallel_workers: Thread pool size
        """
        self.prefilter = StatisticalPrefilter()
        self.max_prefilter = max_prefilter_candidates
        self.max_deep = max_deep_analysis
        self.workers = parallel_workers

        self._resonator = None
        self._primitive_loader = None
        self._faiss_index = None
        self._entity_vectors: dict[str, torch.Tensor] = {}

    def set_resonator(self, resonator) -> None:
        """Set VSA resonator for deep analysis."""
        self._resonator = resonator

    def set_primitive_loader(self, loader) -> None:
        """Set primitive loader for vector generation."""
        self._primitive_loader = loader

    def build_index(self, entities: dict[str, torch.Tensor]) -> None:
        """Build FAISS index for fast screening.

        Args:
            entities: Dict mapping entity_id to vector
        """
        try:
            import faiss

            self._entity_vectors = entities
            ids = list(entities.keys())
            vectors = torch.stack([entities[id] for id in ids])

            # Convert to real vectors for FAISS (interleave real/imag)
            real_vectors = torch.view_as_real(vectors).reshape(len(ids), -1)
            real_vectors = real_vectors.cpu().numpy().astype(np.float32)

            # Normalize for cosine similarity
            faiss.normalize_L2(real_vectors)

            # Build index
            d = real_vectors.shape[1]
            self._faiss_index = faiss.IndexFlatIP(
                d
            )  # Inner product after normalization = cosine
            self._faiss_index.add(real_vectors)

            self._index_ids = ids
            logger.info(f"Built FAISS index with {len(ids)} vectors, dim={d}")

        except ImportError:
            logger.warning("FAISS not installed, screening stage will be slower")
            self._faiss_index = None

    def _screening_stage(
        self, candidates: list[str], query_vectors: dict[str, torch.Tensor]
    ) -> StageResult:
        """Stage 2: VSA screening with FAISS.

        Args:
            candidates: Entity IDs from prefilter
            query_vectors: Query vectors for each primitive

        Returns:
            StageResult with top candidates
        """
        start = time.time()
        scores: dict[str, float] = {}

        if self._faiss_index is None:
            # Fallback to brute force
            for entity_id in candidates:
                if entity_id not in self._entity_vectors:
                    continue
                entity_vec = self._entity_vectors[entity_id]
                max_sim = 0.0
                for query in query_vectors.values():
                    sim = float(torch.sum(entity_vec * torch.conj(query)).real)
                    max_sim = max(max_sim, sim)
                scores[entity_id] = max_sim
        else:
            # Use FAISS
            import faiss

            for primitive, query in query_vectors.items():
                # Convert query
                query_np = torch.view_as_real(query).reshape(1, -1)
                query_np = query_np.cpu().numpy().astype(np.float32)
                faiss.normalize_L2(query_np)

                # Search
                k = min(len(candidates), self.max_deep * 2)
                D, I = self._faiss_index.search(query_np, k)

                for dist, idx in zip(D[0], I[0]):
                    if idx < 0:
                        continue
                    entity_id = self._index_ids[idx]
                    if entity_id in candidates:
                        scores[entity_id] = max(scores.get(entity_id, 0), float(dist))

        # Sort and take top
        sorted_candidates = sorted(scores.keys(), key=lambda x: -scores[x])
        top_candidates = sorted_candidates[: self.max_deep]

        elapsed = (time.time() - start) * 1000

        return StageResult(
            stage=PipelineStage.SCREENING,
            candidates=top_candidates,
            scores={k: scores[k] for k in top_candidates},
            elapsed_ms=elapsed,
            metadata={"used_faiss": self._faiss_index is not None},
        )

    def _deep_analysis_stage(
        self, candidates: list[str], data_lookup: dict[str, dict]
    ) -> tuple[StageResult, list[PipelineResult]]:
        """Stage 3: Full VSA resonator analysis.

        Args:
            candidates: Entity IDs for deep analysis
            data_lookup: Entity data by ID

        Returns:
            (StageResult, list of PipelineResults)
        """
        start = time.time()
        results: list[PipelineResult] = []
        scores: dict[str, float] = {}

        for entity_id in candidates:
            if entity_id not in self._entity_vectors:
                continue

            entity_vec = self._entity_vectors[entity_id]

            # Run resonator
            if self._resonator:
                res_result = self._resonator.resonate(entity_vec)

                # Extract anomalies from top matches
                anomalies = [
                    match[0]
                    for match in res_result.top_matches
                    if match[1] > 0.4  # Threshold
                ]

                scores[entity_id] = (
                    res_result.top_matches[0][1] if res_result.top_matches else 0
                )

                results.append(
                    PipelineResult(
                        entity_id=entity_id,
                        detected_anomalies=anomalies,
                        confidence=scores[entity_id],
                        root_causes=[],  # Filled by symbolic reasoning
                        stage_results=[],
                        total_elapsed_ms=res_result.elapsed_ms,
                        metadata={"resonator_iterations": res_result.iterations},
                    )
                )

        elapsed = (time.time() - start) * 1000

        stage_result = StageResult(
            stage=PipelineStage.DEEP_ANALYSIS,
            candidates=list(scores.keys()),
            scores=scores,
            elapsed_ms=elapsed,
            metadata={"analyzed_count": len(results)},
        )

        return stage_result, results

    def run(
        self, data: list[dict[str, Any]], primitives: list[str] | None = None
    ) -> list[PipelineResult]:
        """Run full pipeline on data.

        Args:
            data: List of entity records
            primitives: Primitive names to query for (default: all)

        Returns:
            List of PipelineResults for detected anomalies
        """
        total_start = time.time()

        # Stage 1: Prefilter
        self.prefilter.compute_statistics(data)
        prefilter_result = self.prefilter.filter(data)
        logger.info(
            f"Prefilter: {len(prefilter_result.candidates)}/{len(data)} candidates "
            f"in {prefilter_result.elapsed_ms:.1f}ms"
        )

        if not prefilter_result.candidates:
            return []

        # Build data lookup
        data_lookup = {}
        for record in data:
            entity_id = record.get("entity_id", record.get("sku", str(id(record))))
            data_lookup[entity_id] = record

        # Get query vectors
        query_vectors = {}
        if self._primitive_loader and primitives:
            for prim in primitives:
                try:
                    query_vectors[prim] = self._primitive_loader.get_vector(prim)
                except KeyError:
                    logger.warning(f"Primitive not found: {prim}")

        # Stage 2: Screening
        screening_result = self._screening_stage(
            prefilter_result.candidates, query_vectors
        )
        logger.info(
            f"Screening: {len(screening_result.candidates)} candidates "
            f"in {screening_result.elapsed_ms:.1f}ms"
        )

        # Stage 3: Deep analysis
        deep_result, pipeline_results = self._deep_analysis_stage(
            screening_result.candidates, data_lookup
        )
        logger.info(
            f"Deep analysis: {len(pipeline_results)} results "
            f"in {deep_result.elapsed_ms:.1f}ms"
        )

        # Attach stage results to pipeline results
        for result in pipeline_results:
            result.stage_results = [prefilter_result, screening_result, deep_result]
            result.total_elapsed_ms = (time.time() - total_start) * 1000

        return pipeline_results
