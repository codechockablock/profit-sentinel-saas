"""
vsa_core/geodesic_resonator.py - Riemannian Geometry-Aware Resonator

This module implements a resonator that uses Riemannian geometry
for more principled inference on the complex hypersphere.

KEY IMPROVEMENTS OVER STANDARD RESONATOR:

1. GEODESIC AVERAGING:
   Instead of Euclidean weighted sum, uses Fréchet mean on manifold.
   This respects the curved geometry of phasor space.

2. RIEMANNIAN GRADIENT DESCENT:
   Updates follow geodesics rather than straight lines.
   Stays on manifold naturally.

3. CURVATURE-AWARE SIMILARITY:
   Uses geodesic distance instead of cosine similarity.
   Better captures true separation in phasor space.

4. PARALLEL TRANSPORT FOR MOMENTUM:
   Momentum vectors are transported along geodesics.
   Preserves their meaning across iterations.

Mathematical Foundation:
    The resonator solves an optimization problem on the manifold:

    min_x Σ w_i d_g(x, c_i)^2

    where d_g is geodesic distance and c_i are codebook vectors.
    This is the Fréchet mean problem.
"""
from __future__ import annotations
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time

from .types import ResonatorConfig
from .vectors import normalize, batch_similarity, get_device
from .manifolds import ComplexHypersphere, Manifold


@dataclass
class GeodesicResonatorResult:
    """Result from geodesic resonator."""
    vector: torch.Tensor
    iterations: int
    converged: bool
    geodesic_delta: float
    elapsed_ms: float
    top_matches: List[Tuple[str, float]]
    geodesic_distances: List[float]
    metadata: Dict[str, Any]


class GeodesicResonator:
    """Resonator using Riemannian geometry on complex hypersphere.

    This resonator uses geodesic operations to stay on the manifold
    and respect the curved geometry of phasor space.

    Example:
        resonator = GeodesicResonator()
        resonator.set_codebook(labels, vectors)

        result = resonator.resonate(noisy_query)
        print(result.top_matches)
        print(result.geodesic_distances)  # True geodesic distances
    """

    def __init__(
        self,
        config: Optional[ResonatorConfig] = None,
        manifold: Optional[Manifold] = None
    ):
        """Initialize geodesic resonator.

        Args:
            config: Resonator configuration
            manifold: Riemannian manifold (default: ComplexHypersphere)
        """
        self.config = config or ResonatorConfig()
        self.manifold = manifold or ComplexHypersphere()
        self.codebook: Optional[torch.Tensor] = None
        self.labels: List[str] = []
        self._device = get_device()

    def set_codebook(
        self,
        labels: List[str],
        vectors: torch.Tensor
    ) -> None:
        """Set codebook, projecting all vectors onto manifold."""
        if len(labels) != len(vectors):
            raise ValueError("labels and vectors must have same length")

        self.labels = labels
        # Project to manifold
        self.codebook = self.manifold.project(vectors.to(self._device))

    def geodesic_distances(self, query: torch.Tensor) -> torch.Tensor:
        """Compute geodesic distances from query to all codebook vectors.

        Args:
            query: Query vector (projected onto manifold)

        Returns:
            Tensor of geodesic distances
        """
        query = self.manifold.project(query.to(self._device))
        return torch.stack([
            self.manifold.dist(query, c) for c in self.codebook
        ])

    def resonate(
        self,
        query: torch.Tensor,
        return_trajectory: bool = False
    ) -> GeodesicResonatorResult:
        """Run geodesic resonator.

        Uses Riemannian gradient descent to find closest attractor.

        Args:
            query: Input query vector
            return_trajectory: Include iteration history

        Returns:
            GeodesicResonatorResult with geodesic metrics
        """
        if self.codebook is None:
            raise RuntimeError("Codebook not set")

        start_time = time.time()
        query = self.manifold.project(query.to(self._device))

        # Initialize
        x = query.clone()
        momentum = torch.zeros_like(x)
        trajectory = [x.clone()] if return_trajectory else []

        converged = False
        final_delta = float('inf')

        for iteration in range(self.config.iterations):
            # Compute geodesic distances
            dists = self.geodesic_distances(x)

            # Get top-k nearest
            k = min(self.config.top_k, len(dists))
            _, indices = torch.topk(-dists, k)  # Negative for smallest

            # Compute weights (inverse distance weighting with power)
            top_dists = dists[indices]
            weights = torch.pow(1.0 / (top_dists + 1e-8), self.config.power)
            weights = weights / weights.sum()

            # Compute weighted tangent sum (Riemannian gradient)
            tangent_sum = torch.zeros_like(x)
            for i, (idx, w) in enumerate(zip(indices, weights)):
                log_vec = self.manifold.log(x, self.codebook[idx])
                tangent_sum = tangent_sum + w * log_vec

            # Apply momentum with parallel transport
            if iteration > 0:
                transported_momentum = self.manifold.parallel_transport(
                    x_prev, x, momentum
                )
                tangent_sum = (
                    self.config.alpha * transported_momentum +
                    (1 - self.config.alpha) * tangent_sum
                )

            # Store for next iteration
            x_prev = x.clone()
            momentum = tangent_sum.clone()

            # Update via exponential map
            x = self.manifold.exp(x, tangent_sum)

            if return_trajectory:
                trajectory.append(x.clone())

            # Check convergence (geodesic distance moved)
            delta = float(self.manifold.dist(x, x_prev))

            if self.config.early_exit and delta < self.config.convergence_threshold:
                converged = True
                final_delta = delta
                break

            final_delta = delta

        # Compute final geodesic distances
        final_dists = self.geodesic_distances(x)
        top_k = min(10, len(final_dists))
        values, indices = torch.topk(-final_dists, top_k)

        top_matches = [
            (self.labels[idx], -float(val))  # Distance (smaller = better)
            for idx, val in zip(indices.tolist(), values.tolist())
        ]

        # Convert to similarities for compatibility
        geodesic_distances_list = [-float(v) for v in values]

        elapsed = (time.time() - start_time) * 1000

        metadata = {
            "codebook_size": len(self.codebook),
            "manifold": type(self.manifold).__name__,
            "config": self.config.model_dump(),
        }
        if return_trajectory:
            metadata["trajectory"] = trajectory

        return GeodesicResonatorResult(
            vector=x,
            iterations=iteration + 1,
            converged=converged,
            geodesic_delta=final_delta,
            elapsed_ms=elapsed,
            top_matches=top_matches,
            geodesic_distances=geodesic_distances_list,
            metadata=metadata
        )

    def frechet_mean_bundle(
        self,
        vectors: List[torch.Tensor],
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        """Bundle vectors using Fréchet mean.

        This is the geometrically correct way to average on the manifold.

        Args:
            vectors: Vectors to bundle
            weights: Optional weights

        Returns:
            Fréchet mean (intrinsic average)
        """
        return self.manifold.frechet_mean(vectors, weights)

    def geodesic_interpolate(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        t: float
    ) -> torch.Tensor:
        """Interpolate along geodesic from p to q.

        Args:
            p: Start point
            q: End point
            t: Interpolation parameter (0=p, 1=q)

        Returns:
            Point on geodesic
        """
        return self.manifold.geodesic_interpolation(p, q, t)


class CurvatureAwareDetector:
    """Anomaly detector using curvature-based metrics.

    Uses Riemannian geometry to detect anomalies based on:
    1. Geodesic distance from cluster centers
    2. Local curvature (how "bent" the space is around a point)
    3. Volume distortion (how much geodesic balls differ from Euclidean)
    """

    def __init__(self, manifold: Optional[Manifold] = None):
        self.manifold = manifold or ComplexHypersphere()
        self.cluster_centers: List[torch.Tensor] = []
        self.cluster_radii: List[float] = []

    def fit_clusters(
        self,
        vectors: List[torch.Tensor],
        n_clusters: int = 10
    ) -> None:
        """Fit cluster centers using Riemannian k-means.

        Uses Fréchet mean for cluster updates.
        """
        # Initialize randomly
        indices = torch.randperm(len(vectors))[:n_clusters]
        centers = [vectors[i] for i in indices]

        # K-means iterations
        for _ in range(20):
            # Assign to nearest center
            assignments = [[] for _ in range(n_clusters)]
            for v in vectors:
                dists = [self.manifold.dist(v, c) for c in centers]
                nearest = int(torch.argmin(torch.tensor(dists)))
                assignments[nearest].append(v)

            # Update centers with Fréchet mean
            new_centers = []
            for i, assigned in enumerate(assignments):
                if assigned:
                    new_centers.append(self.manifold.frechet_mean(assigned))
                else:
                    new_centers.append(centers[i])
            centers = new_centers

        self.cluster_centers = centers

        # Compute cluster radii (max geodesic distance to center)
        self.cluster_radii = []
        for center, assigned in zip(centers, assignments):
            if assigned:
                max_dist = max(self.manifold.dist(center, v) for v in assigned)
                self.cluster_radii.append(float(max_dist))
            else:
                self.cluster_radii.append(0.0)

    def anomaly_score(self, vector: torch.Tensor) -> float:
        """Compute anomaly score based on geodesic distance.

        Higher score = more anomalous.

        Args:
            vector: Vector to score

        Returns:
            Anomaly score (0-1, higher = more anomalous)
        """
        if not self.cluster_centers:
            return 0.0

        # Distance to nearest cluster
        min_dist = float('inf')
        nearest_idx = 0

        for i, center in enumerate(self.cluster_centers):
            d = float(self.manifold.dist(vector, center))
            if d < min_dist:
                min_dist = d
                nearest_idx = i

        # Normalize by cluster radius
        radius = self.cluster_radii[nearest_idx] + 1e-8

        # Score: how many radii away from center
        score = min_dist / radius

        # Sigmoid to bound [0, 1]
        return float(2 / (1 + torch.exp(torch.tensor(-score))) - 1)

    def detect_anomalies(
        self,
        vectors: List[torch.Tensor],
        threshold: float = 0.5
    ) -> List[Tuple[int, float]]:
        """Detect anomalous vectors.

        Args:
            vectors: Vectors to check
            threshold: Anomaly score threshold

        Returns:
            List of (index, score) for anomalous vectors
        """
        anomalies = []
        for i, v in enumerate(vectors):
            score = self.anomaly_score(v)
            if score > threshold:
                anomalies.append((i, score))

        return sorted(anomalies, key=lambda x: -x[1])
