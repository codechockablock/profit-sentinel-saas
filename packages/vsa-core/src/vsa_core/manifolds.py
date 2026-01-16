"""
vsa_core/manifolds.py - Riemannian Geometry for VSA

This module implements Riemannian manifold operations for complex phasor
vectors, enabling geometrically-aware inference.

MATHEMATICAL FOUNDATION:

Complex phasor vectors live on the product manifold (S^1)^d, where each
component lies on the unit circle. This has rich geometric structure:

1. COMPLEX HYPERSPHERE VIEW:
   Treating the real/imaginary parts as coordinates, phasor vectors
   lie on the complex hypersphere S^(2d-1) ⊂ ℂ^d ≅ ℝ^(2d).

2. GEODESICS:
   Shortest paths on the manifold. On S^n, geodesics are great circles.
   For the product manifold, geodesics are products of great circles.

3. EXPONENTIAL MAP:
   exp_p(v): Project tangent vector v at point p onto manifold.
   Moves along geodesic from p in direction v.

4. LOGARITHM MAP:
   log_p(q): Inverse of exp. Returns tangent vector from p to q.

5. FRÉCHET MEAN:
   Riemannian analog of centroid. Minimizes sum of squared geodesic
   distances.

6. PARALLEL TRANSPORT:
   Move tangent vectors along geodesics while preserving inner products.

These operations enable:
- Geodesic interpolation (morphing between vectors)
- Riemannian gradient descent for optimization
- Intrinsic averaging (Fréchet mean bundling)
- Curvature-aware resonance
"""
from __future__ import annotations
import torch
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .vectors import normalize, get_device, get_dtype


# =============================================================================
# ABSTRACT MANIFOLD
# =============================================================================

class Manifold(ABC):
    """Abstract base class for Riemannian manifolds."""

    @abstractmethod
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project point onto manifold."""
        pass

    @abstractmethod
    def exp(self, p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map: move from p along tangent v."""
        pass

    @abstractmethod
    def log(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Logarithm map: tangent vector from p to q."""
        pass

    @abstractmethod
    def dist(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Geodesic distance between points."""
        pass

    @abstractmethod
    def parallel_transport(
        self, p: torch.Tensor, q: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Parallel transport v from tangent space at p to q."""
        pass


# =============================================================================
# COMPLEX HYPERSPHERE MANIFOLD
# =============================================================================

class ComplexHypersphere(Manifold):
    """Complex hypersphere manifold for phasor vectors.

    Each component is constrained to the unit circle: |z_i| = 1.
    This is the product manifold (S^1)^d.

    For operations, we treat this as embedded in the complex hypersphere
    S^(2d-1) ⊂ ℂ^d ≅ ℝ^(2d).
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project onto unit phasors (normalize each component)."""
        return normalize(x, eps=self.eps)

    def inner(
        self, p: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Riemannian inner product at p.

        For complex hypersphere: ⟨u, v⟩_p = Re(u · conj(v))
        """
        return torch.sum(u * torch.conj(v)).real

    def norm(self, p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Riemannian norm of tangent vector."""
        return torch.sqrt(self.inner(p, v, v) + self.eps)

    def project_tangent(self, p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Project v onto tangent space at p.

        Tangent space at p consists of vectors orthogonal to p:
        v_tangent = v - Re(⟨v, p⟩)p / |p|^2
        """
        # For unit phasors, |p|^2 = d
        proj = torch.sum(v * torch.conj(p)).real / (p.numel() + self.eps)
        return v - proj * p

    def exp(self, p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map on complex hypersphere.

        exp_p(v) = cos(||v||)p + sin(||v||)v/||v||

        This follows the geodesic from p in direction v.
        """
        v_norm = self.norm(p, v)

        # Handle zero tangent
        if v_norm < self.eps:
            return p

        # Geodesic formula
        cos_t = torch.cos(v_norm)
        sin_t = torch.sin(v_norm)

        result = cos_t * p + sin_t * (v / v_norm)
        return self.project(result)

    def log(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Logarithm map on complex hypersphere.

        log_p(q) = arccos(⟨p, q⟩) * (q - ⟨p, q⟩p) / ||q - ⟨p, q⟩p||

        Returns tangent vector at p pointing toward q.
        """
        # Inner product (on full vector)
        inner = torch.sum(p * torch.conj(q)).real / p.numel()
        inner = torch.clamp(inner, -1 + self.eps, 1 - self.eps)

        # Distance (angle)
        theta = torch.acos(inner)

        # Direction
        direction = q - inner * p
        dir_norm = torch.norm(direction) + self.eps

        # Handle antipodal or identical points
        if dir_norm < self.eps or theta < self.eps:
            return torch.zeros_like(p)

        return theta * direction / dir_norm

    def dist(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Geodesic distance (arc length).

        d(p, q) = arccos(⟨p, q⟩)
        """
        inner = torch.sum(p * torch.conj(q)).real / p.numel()
        inner = torch.clamp(inner, -1 + self.eps, 1 - self.eps)
        return torch.acos(inner)

    def parallel_transport(
        self, p: torch.Tensor, q: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Parallel transport v from T_p to T_q along geodesic.

        Uses Schild's ladder approximation for efficiency.
        """
        # Get geodesic direction
        log_pq = self.log(p, q)
        d = self.norm(p, log_pq)

        if d < self.eps:
            return v

        # Normalize direction
        u = log_pq / d

        # Transport formula for sphere
        # v_q = v - ⟨log_p(q), v⟩(p + q) / (1 + ⟨p, q⟩)
        inner_pq = torch.sum(p * torch.conj(q)).real / p.numel()
        inner_pq = torch.clamp(inner_pq, -1 + self.eps, 1 - self.eps)

        inner_uv = self.inner(p, u, v)

        transported = v - inner_uv * (p + q) / (1 + inner_pq + self.eps)
        return self.project_tangent(q, transported)

    def frechet_mean(
        self,
        points: List[torch.Tensor],
        weights: Optional[List[float]] = None,
        max_iters: int = 100,
        tol: float = 1e-6
    ) -> torch.Tensor:
        """Compute Fréchet (intrinsic) mean.

        The Fréchet mean minimizes:
            μ = argmin_x Σ w_i d(x, p_i)^2

        Uses gradient descent on the manifold.
        """
        if weights is None:
            weights = [1.0 / len(points)] * len(points)

        weights = torch.tensor(weights, device=points[0].device)
        weights = weights / weights.sum()

        # Initialize at first point
        mean = points[0].clone()

        for _ in range(max_iters):
            # Compute weighted tangent sum
            tangent_sum = torch.zeros_like(mean)
            for p, w in zip(points, weights):
                tangent_sum = tangent_sum + w * self.log(mean, p)

            # Check convergence
            delta = self.norm(mean, tangent_sum)
            if delta < tol:
                break

            # Update mean
            mean = self.exp(mean, tangent_sum)

        return mean

    def geodesic_interpolation(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        t: float
    ) -> torch.Tensor:
        """Interpolate along geodesic from p to q.

        γ(t) = exp_p(t * log_p(q))

        t=0 gives p, t=1 gives q.
        """
        v = self.log(p, q)
        return self.exp(p, t * v)


# =============================================================================
# HYPERBOLIC SPACE (POINCARÉ BALL)
# =============================================================================

class PoincareBall(Manifold):
    """Poincaré ball model of hyperbolic space.

    Points lie in the open unit ball: ||x|| < 1
    Useful for hierarchical embeddings.
    """

    def __init__(self, curvature: float = -1.0, eps: float = 1e-8):
        self.c = abs(curvature)  # Positive value for computation
        self.eps = eps

    def _lambda(self, x: torch.Tensor) -> torch.Tensor:
        """Conformal factor λ_x = 2 / (1 - c||x||^2)"""
        norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
        return 2 / (1 - self.c * norm_sq + self.eps)

    def project(self, x: torch.Tensor, max_norm: float = 0.99) -> torch.Tensor:
        """Project onto open ball."""
        norm = torch.norm(x, dim=-1, keepdim=True)
        max_norm_tensor = torch.tensor(max_norm / math.sqrt(self.c), device=x.device)
        cond = norm > max_norm_tensor
        return torch.where(cond, x * max_norm_tensor / (norm + self.eps), x)

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Möbius addition: x ⊕ y."""
        x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
        y_norm_sq = torch.sum(y * y, dim=-1, keepdim=True)
        xy_inner = torch.sum(x * y, dim=-1, keepdim=True)

        num = (1 + 2 * self.c * xy_inner + self.c * y_norm_sq) * x
        num = num + (1 - self.c * x_norm_sq) * y
        denom = 1 + 2 * self.c * xy_inner + self.c**2 * x_norm_sq * y_norm_sq

        return num / (denom + self.eps)

    def exp(self, p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map in Poincaré ball."""
        v_norm = torch.norm(v, dim=-1, keepdim=True)
        lambda_p = self._lambda(p)

        # exp_p(v) = p ⊕ tanh(√c λ_p ||v|| / 2) * v / (√c ||v||)
        sqrt_c = math.sqrt(self.c)
        coef = torch.tanh(sqrt_c * lambda_p * v_norm / 2)
        direction = v / (sqrt_c * v_norm + self.eps)

        result = self.mobius_add(p, coef * direction)
        return self.project(result)

    def log(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Logarithm map in Poincaré ball."""
        # -p ⊕ q
        neg_p = -p
        add_result = self.mobius_add(neg_p, q)
        add_norm = torch.norm(add_result, dim=-1, keepdim=True)

        lambda_p = self._lambda(p)
        sqrt_c = math.sqrt(self.c)

        # log_p(q) = (2 / √c λ_p) arctanh(√c ||-p ⊕ q||) * (-p ⊕ q) / ||-p ⊕ q||
        coef = 2 / (sqrt_c * lambda_p + self.eps)
        arctanh_arg = torch.clamp(sqrt_c * add_norm, max=1 - self.eps)
        arctanh_val = torch.arctanh(arctanh_arg)

        return coef * arctanh_val * add_result / (add_norm + self.eps)

    def dist(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Geodesic distance in Poincaré ball."""
        sqrt_c = math.sqrt(self.c)
        add_result = self.mobius_add(-p, q)
        add_norm = torch.norm(add_result)

        return 2 / sqrt_c * torch.arctanh(
            torch.clamp(sqrt_c * add_norm, max=1 - self.eps)
        )

    def parallel_transport(
        self, p: torch.Tensor, q: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Parallel transport in Poincaré ball."""
        lambda_p = self._lambda(p)
        lambda_q = self._lambda(q)
        return v * lambda_p / lambda_q


# =============================================================================
# PRODUCT MANIFOLD (MIXED CURVATURE)
# =============================================================================

class ProductManifold(Manifold):
    """Product of multiple manifolds with different curvatures.

    Useful for mixed-curvature embeddings:
    M = S^d × H^d × ℝ^d (spherical × hyperbolic × Euclidean)
    """

    def __init__(
        self,
        manifolds: List[Tuple[Manifold, int]],  # (manifold, dim)
    ):
        """Initialize product manifold.

        Args:
            manifolds: List of (manifold, dimension) pairs
        """
        self.manifolds = manifolds
        self.dims = [d for _, d in manifolds]
        self.total_dim = sum(self.dims)
        self.splits = self._compute_splits()

    def _compute_splits(self) -> List[int]:
        """Compute dimension split points."""
        splits = [0]
        for d in self.dims:
            splits.append(splits[-1] + d)
        return splits

    def _split(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Split tensor into components."""
        return [
            x[..., self.splits[i]:self.splits[i+1]]
            for i in range(len(self.manifolds))
        ]

    def _concat(self, parts: List[torch.Tensor]) -> torch.Tensor:
        """Concatenate components."""
        return torch.cat(parts, dim=-1)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project onto product manifold."""
        parts = self._split(x)
        projected = [
            m.project(p) for (m, _), p in zip(self.manifolds, parts)
        ]
        return self._concat(projected)

    def exp(self, p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Product exponential map."""
        p_parts = self._split(p)
        v_parts = self._split(v)
        result_parts = [
            m.exp(pp, vp) for (m, _), pp, vp in zip(self.manifolds, p_parts, v_parts)
        ]
        return self._concat(result_parts)

    def log(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Product logarithm map."""
        p_parts = self._split(p)
        q_parts = self._split(q)
        result_parts = [
            m.log(pp, qp) for (m, _), pp, qp in zip(self.manifolds, p_parts, q_parts)
        ]
        return self._concat(result_parts)

    def dist(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Product distance (sum of squared component distances)."""
        p_parts = self._split(p)
        q_parts = self._split(q)
        dist_sq = sum(
            m.dist(pp, qp) ** 2
            for (m, _), pp, qp in zip(self.manifolds, p_parts, q_parts)
        )
        return torch.sqrt(dist_sq)

    def parallel_transport(
        self, p: torch.Tensor, q: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Product parallel transport."""
        p_parts = self._split(p)
        q_parts = self._split(q)
        v_parts = self._split(v)
        result_parts = [
            m.parallel_transport(pp, qp, vp)
            for (m, _), pp, qp, vp in zip(self.manifolds, p_parts, q_parts, v_parts)
        ]
        return self._concat(result_parts)


# =============================================================================
# EUCLIDEAN SPACE (FOR PRODUCT MANIFOLDS)
# =============================================================================

class Euclidean(Manifold):
    """Flat Euclidean space (zero curvature)."""

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def exp(self, p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return p + v

    def log(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return q - p

    def dist(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return torch.norm(q - p)

    def parallel_transport(
        self, p: torch.Tensor, q: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return v  # Flat space - transport is identity


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_mixed_curvature_space(
    sphere_dim: int,
    hyperbolic_dim: int,
    euclidean_dim: int,
    curvature: float = -1.0
) -> ProductManifold:
    """Create S^d × H^d × ℝ^d product manifold.

    This mixed-curvature space can represent:
    - Spherical: cyclic/periodic patterns
    - Hyperbolic: hierarchical/tree-like structures
    - Euclidean: linear/additive quantities

    Args:
        sphere_dim: Dimension of spherical component
        hyperbolic_dim: Dimension of hyperbolic component
        euclidean_dim: Dimension of Euclidean component
        curvature: Curvature for hyperbolic space (negative)

    Returns:
        ProductManifold instance
    """
    manifolds = [
        (ComplexHypersphere(), sphere_dim),
        (PoincareBall(curvature), hyperbolic_dim),
        (Euclidean(), euclidean_dim),
    ]
    return ProductManifold(manifolds)
