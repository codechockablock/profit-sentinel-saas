"""
vsa_core/resonator.py - Convergence-Lock Resonator for Query Cleanup

The resonator is the core inference mechanism in VSA systems. Given a noisy
query vector, it iteratively refines the query until it converges to a
stable attractor in the representation space.

Algorithm:
    1. Start with noisy query q
    2. Compute similarity to all codebook vectors
    3. Reconstruct cleaned vector using weighted combination
    4. Apply momentum (mix with previous iterate)
    5. Normalize and repeat until convergence

Mathematical Foundation:
    The resonator implements iterative projection onto the manifold of
    "clean" representations. With momentum, it follows a damped trajectory
    that settles into local attractors.

    For complex phasors, convergence means phase alignment with
    codebook entries that explain the query.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from .operators import sparse_resonance_step
from .types import ResonatorConfig
from .vectors import batch_similarity, get_device, normalize


@dataclass
class ResonatorResult:
    """Result from resonator processing."""
    vector: torch.Tensor
    iterations: int
    converged: bool
    convergence_delta: float
    elapsed_ms: float
    top_matches: list[tuple[str, float]]  # [(label, similarity), ...]
    metadata: dict[str, Any]


class Resonator:
    """Convergence-lock resonator for VSA query cleanup.

    The resonator takes a noisy query vector and iteratively refines it
    against a codebook until the query stabilizes on clean primitives.

    Example:
        resonator = Resonator()
        resonator.set_codebook(labels, vectors)

        # Query with noisy fact
        result = resonator.resonate(noisy_query)
        print(result.top_matches)  # [('low_stock', 0.87), ('sku:123', 0.65)]
    """

    def __init__(self, config: ResonatorConfig | None = None):
        """Initialize resonator.

        Args:
            config: Resonator configuration (uses defaults if None)
        """
        self.config = config or ResonatorConfig()
        self.codebook: torch.Tensor | None = None
        self.labels: list[str] = []
        self._device = get_device()

    def set_codebook(
        self,
        labels: list[str],
        vectors: torch.Tensor
    ) -> None:
        """Set the codebook for resonator queries.

        Args:
            labels: Human-readable labels for each vector
            vectors: Codebook matrix of shape (n, d)
        """
        if len(labels) != len(vectors):
            raise ValueError("labels and vectors must have same length")

        self.labels = labels
        self.codebook = vectors.to(self._device)

    def add_to_codebook(self, label: str, vector: torch.Tensor) -> None:
        """Add single vector to codebook.

        Args:
            label: Label for the vector
            vector: Vector to add
        """
        vector = vector.to(self._device)

        if self.codebook is None:
            self.codebook = vector.unsqueeze(0)
            self.labels = [label]
        else:
            self.codebook = torch.cat([
                self.codebook,
                vector.unsqueeze(0)
            ], dim=0)
            self.labels.append(label)

    def resonate(
        self,
        query: torch.Tensor,
        return_trajectory: bool = False
    ) -> ResonatorResult:
        """Run convergence-lock resonator on query.

        Args:
            query: Input query vector
            return_trajectory: If True, include iteration history in metadata

        Returns:
            ResonatorResult with cleaned vector and diagnostics
        """
        if self.codebook is None or len(self.codebook) == 0:
            raise RuntimeError("Codebook not set. Call set_codebook() first.")

        start_time = time.time()
        query = query.to(self._device)

        # Initialize
        x = normalize(query.clone())
        x_prev = x.clone()
        momentum = torch.zeros_like(x)
        trajectory = [x.clone()] if return_trajectory else []

        converged = False
        final_delta = float('inf')

        for iteration in range(self.config.iterations):
            # Multi-step update
            for _ in range(self.config.multi_steps):
                # Sparse resonance step
                x_new = sparse_resonance_step(
                    x,
                    self.codebook,
                    top_k=self.config.top_k,
                    power=self.config.power
                )

                # Apply momentum
                momentum = self.config.alpha * momentum + (1 - self.config.alpha) * (x_new - x)
                x = normalize(x + momentum)

            # Check convergence
            delta = float(torch.norm(x - x_prev).item())

            if return_trajectory:
                trajectory.append(x.clone())

            if self.config.early_exit and delta < self.config.convergence_threshold:
                converged = True
                final_delta = delta
                break

            x_prev = x.clone()
            final_delta = delta

        # Compute final similarities
        similarities = batch_similarity(x, self.codebook)
        top_k = min(10, len(similarities))
        values, indices = torch.topk(similarities, top_k)

        top_matches = [
            (self.labels[idx], float(val))
            for idx, val in zip(indices.tolist(), values.tolist())
        ]

        elapsed = (time.time() - start_time) * 1000

        metadata = {
            "codebook_size": len(self.codebook),
            "config": self.config.model_dump(),
        }
        if return_trajectory:
            metadata["trajectory"] = trajectory

        return ResonatorResult(
            vector=x,
            iterations=iteration + 1,
            converged=converged,
            convergence_delta=final_delta,
            elapsed_ms=elapsed,
            top_matches=top_matches,
            metadata=metadata
        )

    def batch_resonate(
        self,
        queries: torch.Tensor
    ) -> list[ResonatorResult]:
        """Process multiple queries.

        Args:
            queries: Batch of queries (n, d)

        Returns:
            List of results, one per query
        """
        return [self.resonate(q) for q in queries]

    def resonator_attention(
        self,
        query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get attention weights without full resonation.

        Useful for inspecting what the resonator "sees" in a query.

        Args:
            query: Input query

        Returns:
            (weights, labels_indices) of top-k attended vectors
        """
        if self.codebook is None:
            raise RuntimeError("Codebook not set")

        query = query.to(self._device)
        sims = batch_similarity(query, self.codebook)

        k = min(self.config.top_k, len(sims))
        values, indices = torch.topk(sims, k)

        weights = torch.pow(torch.clamp(values, min=0), self.config.power)
        weights = weights / (weights.sum() + 1e-10)

        return weights, indices


class HierarchicalResonator:
    """Multi-level resonator with coarse-to-fine processing.

    Uses a hierarchy of codebooks for efficient large-scale queries:
    1. Quick coarse matching to find relevant region
    2. Fine resonation within selected region

    Useful for codebooks with 100k+ entries.
    """

    def __init__(
        self,
        config: ResonatorConfig | None = None,
        coarse_clusters: int = 100
    ):
        """Initialize hierarchical resonator.

        Args:
            config: Resonator configuration
            coarse_clusters: Number of coarse clusters
        """
        self.config = config or ResonatorConfig()
        self.coarse_clusters = coarse_clusters

        self.coarse_centroids: torch.Tensor | None = None
        self.fine_codebooks: list[torch.Tensor] = []
        self.fine_labels: list[list[str]] = []
        self.cluster_assignments: list[int] = []
        self._device = get_device()

    def build_hierarchy(
        self,
        labels: list[str],
        vectors: torch.Tensor
    ) -> None:
        """Build hierarchical codebook structure.

        Uses k-means to create coarse clusters, then stores
        fine codebooks per cluster.

        Args:
            labels: All labels
            vectors: All vectors
        """
        vectors = vectors.to(self._device)
        n = len(vectors)

        # Simple k-means for coarse clustering
        k = min(self.coarse_clusters, n)

        # Handle complex vectors by using magnitude for distance calculation
        if vectors.is_complex():
            vectors_real = torch.abs(vectors)  # Use magnitude for clustering
        else:
            vectors_real = vectors

        # Initialize centroids randomly
        indices = torch.randperm(n)[:k]
        centroids = vectors[indices].clone()
        centroids_real = vectors_real[indices].clone()

        # K-means iterations
        for _ in range(20):
            # Assign to nearest centroid using real-valued distances
            dists = torch.cdist(vectors_real.float(), centroids_real.float())
            assignments = torch.argmin(dists, dim=1)

            # Update centroids (use original complex vectors)
            new_centroids = torch.zeros_like(centroids)
            new_centroids_real = torch.zeros_like(centroids_real)
            for i in range(k):
                mask = assignments == i
                if mask.any():
                    new_centroids[i] = normalize(vectors[mask].sum(dim=0))
                    new_centroids_real[i] = torch.abs(new_centroids[i]) if vectors.is_complex() else new_centroids[i]
                else:
                    new_centroids[i] = centroids[i]
                    new_centroids_real[i] = centroids_real[i]
            centroids = new_centroids
            centroids_real = new_centroids_real

        self.coarse_centroids = centroids
        self.cluster_assignments = assignments.tolist()

        # Build fine codebooks per cluster
        self.fine_codebooks = []
        self.fine_labels = []

        for i in range(k):
            mask = assignments == i
            cluster_vectors = vectors[mask]
            cluster_labels = [labels[j] for j, m in enumerate(mask.tolist()) if m]

            self.fine_codebooks.append(cluster_vectors)
            self.fine_labels.append(cluster_labels)

    def resonate(self, query: torch.Tensor) -> ResonatorResult:
        """Hierarchical resonation.

        Args:
            query: Input query

        Returns:
            Result from fine-grained resonator
        """
        if self.coarse_centroids is None:
            raise RuntimeError("Hierarchy not built. Call build_hierarchy() first.")

        query = query.to(self._device)

        # Find best cluster with non-empty codebook
        coarse_sims = batch_similarity(query, self.coarse_centroids)

        # Sort clusters by similarity and find first non-empty
        sorted_clusters = torch.argsort(coarse_sims, descending=True)
        best_cluster = None
        for cluster_idx in sorted_clusters.tolist():
            if len(self.fine_labels[cluster_idx]) > 0:
                best_cluster = cluster_idx
                break

        # Fallback: if all clusters empty, return empty result
        if best_cluster is None:
            return ResonatorResult(
                vector=query,
                iterations=0,
                converged=False,
                convergence_delta=float('inf'),
                elapsed_ms=0,
                top_matches=[],
                metadata={"error": "All clusters empty"}
            )

        # Create temporary resonator for fine search
        fine_resonator = Resonator(self.config)
        fine_resonator.set_codebook(
            self.fine_labels[best_cluster],
            self.fine_codebooks[best_cluster]
        )

        result = fine_resonator.resonate(query)
        result.metadata["selected_cluster"] = best_cluster
        result.metadata["coarse_similarity"] = float(coarse_sims[best_cluster])

        return result


# Note: EnsembleResonator was removed in v3.0.0 (unused complexity).
# HierarchicalResonator is kept for Phase 3 multi-store scenarios.
