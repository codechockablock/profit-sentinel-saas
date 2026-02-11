"""
Rust-Accelerated Phasor Algebra (zero-copy numpy)
===================================================

Drop-in replacement for PhasorAlgebra that delegates to the Rust
sentinel_vsa.PhasorAlgebra backend via PyO3 + numpy zero-copy.

Usage:
    from sentinel_agent.world_model.rust_algebra import RustPhasorAlgebra
    algebra = RustPhasorAlgebra(dim=4096, seed=42)

    # Same API as PhasorAlgebra — all vectors are numpy complex128
    a = algebra.random_vector("role_velocity")
    b = algebra.random_vector("filler_high")
    bound = algebra.bind(a, b)
    recovered = algebra.unbind(bound, b)
    sim = algebra.similarity(a, recovered)  # approx 1.0

Strategy:
  - Single-vector ops (bind, unbind, similarity): pure numpy (fast enough)
  - Batch ops over matrices: Rust + Rayon via matrix_similarity / matrix_compile_states
  - The matrix methods accept (N, D) numpy arrays and read the buffer directly
    in Rust with zero per-vector copy — this is where the 5-14x speedup lives.
"""

from typing import List, Optional

import numpy as np

try:
    from sentinel_vsa import PhasorAlgebra as _RustPhasorAlgebra

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


class RustPhasorAlgebra:
    """
    Phasor algebra backed by Rust sentinel-vsa (zero-copy numpy).

    API-compatible with PhasorAlgebra from core.py.
    Single ops stay in numpy. Batch ops use Rust + Rayon matrix methods.
    """

    def __init__(self, dim: int = 4096, seed: int = 42):
        if not RUST_AVAILABLE:
            raise ImportError(
                "sentinel_vsa not installed. Build with: "
                "cd sentinel-vsa && maturin develop --features python --release"
            )
        self.dim = dim
        self.seed = seed
        self._engine = _RustPhasorAlgebra(dimensions=dim, seed=seed)
        # Local numpy cache for hot-path avoidance of Rust->Python round-trips
        self._np_cache = {}

    def random_vector(self, label: str = None) -> np.ndarray:
        """Generate a deterministic random phasor vector (Rust-accelerated)."""
        if label is None:
            label = f"_anon_{np.random.randint(0, 2**63)}"

        if label in self._np_cache:
            return self._np_cache[label]

        # Rust returns numpy complex128 array directly — zero-copy
        v = np.asarray(self._engine.random_vector(label))
        self._np_cache[label] = v
        return v

    def identity(self) -> np.ndarray:
        """All-ones vector (multiplicative identity for bind)."""
        return np.ones(self.dim, dtype=complex)

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise multiply (binding). Pure numpy — faster than FFI."""
        return a * b

    def unbind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Multiply by conjugate (inverse of bind). Pure numpy."""
        return a * np.conj(b)

    def bundle(
        self, vectors: list[np.ndarray], weights: list[float] | None = None
    ) -> np.ndarray:
        """Superposition with optional weights, normalized to unit magnitude."""
        if weights:
            result = sum(w * v for w, v in zip(weights, vectors))
        else:
            result = sum(vectors)
        return result / np.abs(result)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Phasor similarity: abs(mean(a * conj(b))).

        Matches the Python PhasorAlgebra formula — captures magnitude of
        average phase alignment, not just real-axis projection. This is
        intentionally different from standard cosine similarity.
        """
        return float(np.abs(np.mean(a * np.conj(b))))

    def permute(self, v: np.ndarray, k: int = 1) -> np.ndarray:
        """Circular permutation."""
        return np.roll(v, k)

    # ---- Matrix batch ops (Rust + Rayon, true zero-copy) ----
    # These are the fast path: pass (N, D) matrices into Rust,
    # which reads them as contiguous buffers with no per-vector copy.

    def matrix_similarity(self, query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Similarity of query against every row of matrix (N, D).

        Returns numpy float64 array of shape (N,). 5-14x faster than
        Python loop at N >= 1000.
        """
        return np.asarray(
            self._engine.matrix_similarity(query, np.ascontiguousarray(matrix))
        )

    def matrix_compile_states(
        self, roles: np.ndarray, fillers: np.ndarray
    ) -> np.ndarray:
        """Compile N state vectors from roles (R, D) and fillers (N, R, D).

        Returns numpy complex128 array of shape (N, D). 2-4x faster than
        Python loop at N >= 1000.
        """
        roles_2d = np.ascontiguousarray(roles)
        fillers_flat = np.ascontiguousarray(fillers.reshape(-1))
        return np.asarray(self._engine.matrix_compile_states(roles_2d, fillers_flat))

    # ---- List-based batch ops (kept for API compat) ----

    def batch_bind(
        self, a_vecs: list[np.ndarray], b_vecs: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Bind pairs in parallel (Rust + Rayon). Zero-copy numpy."""
        return self._engine.batch_bind(a_vecs, b_vecs)

    def batch_similarity(
        self, query: np.ndarray, candidates: list[np.ndarray]
    ) -> list[float]:
        """Compare query against all candidates in parallel (Rust + Rayon).

        For best performance, use matrix_similarity with a stacked (N, D) array.
        """
        return self._engine.batch_similarity(query, candidates)

    def compile_state(
        self, role_vectors: list[np.ndarray], filler_vectors: list[np.ndarray]
    ) -> np.ndarray:
        """Compile role-filler bindings into state vector (Rust + Rayon)."""
        return np.asarray(self._engine.compile_state(role_vectors, filler_vectors))

    def batch_compile_states(
        self, role_vectors: list[np.ndarray], filler_batches: list[list[np.ndarray]]
    ) -> list[np.ndarray]:
        """Compile N state vectors in parallel (Rust + Rayon).

        For best performance, use matrix_compile_states with stacked arrays.
        """
        return self._engine.batch_compile_states(role_vectors, filler_batches)

    @property
    def cache_size(self) -> int:
        return len(self._np_cache)

    def clear_cache(self):
        self._np_cache.clear()
        self._engine.clear_cache()
