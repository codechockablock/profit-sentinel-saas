"""
VSA Self-Modification Sandbox Test Harness

This module implements a rigorous test framework for exploring the safety and coherence
of self-modifying Vector Symbolic Architecture systems.

Core Research Questions:
    1. Does coherent self-modification exist, or does it inevitably degrade?
    2. Are there stable attractor states?
    3. Are there degenerate configurations the system can fall into?
    4. Can the system detect its own degradation, or does modification corrupt self-assessment?
    5. What are the actual invariants (if any)?

Design Principles:
    - All geometric state is capturable and restorable
    - Modifications are controllable and reversible
    - Self-evaluation uses the same geometry being tested
    - Full audit trail of all state changes

Usage:
    harness = VSASandboxHarness(dimensions=2048, device="mps")
    baseline = harness.capture_baseline()

    # Apply modifications
    harness.apply_primitive_perturbation("low_stock", magnitude=0.1)

    # Measure drift
    metrics = harness.measure_health()
    drift = harness.compute_drift_from_baseline(baseline)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class GeometricSnapshot:
    """Complete snapshot of VSA geometric state.

    This captures everything needed to restore the system to a previous state
    and to measure how far the current state has drifted.
    """
    timestamp: float
    snapshot_id: str

    # Core geometry
    primitive_vectors: dict[str, torch.Tensor]
    codebook_vectors: dict[str, torch.Tensor]

    # Derived metrics (computed at snapshot time)
    primitive_similarity_matrix: torch.Tensor  # (n_primitives, n_primitives)
    codebook_similarity_matrix: torch.Tensor | None  # (n_codebook, n_codebook) - can be large
    primitive_norms: dict[str, float]
    primitive_phases: dict[str, torch.Tensor]  # Phase angles per dimension

    # Configuration
    dimensions: int
    device: str
    dtype: str

    # Metadata
    description: str = ""
    parent_snapshot_id: str | None = None
    modifications_applied: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dictionary (tensors as lists for JSON)."""
        return {
            "timestamp": self.timestamp,
            "snapshot_id": self.snapshot_id,
            "dimensions": self.dimensions,
            "device": self.device,
            "dtype": self.dtype,
            "description": self.description,
            "parent_snapshot_id": self.parent_snapshot_id,
            "modifications_applied": self.modifications_applied,
            "primitive_norms": self.primitive_norms,
            # Note: vectors not serialized here - use save_checkpoint for full state
        }


@dataclass
class HealthMetrics:
    """Metrics indicating system health/coherence.

    These are the "vital signs" we monitor to detect degradation.
    """
    timestamp: float

    # Binding accuracy: can we bind and unbind correctly?
    binding_accuracy: float  # 0-1, should be ~1.0
    binding_recovery_similarities: list[float]  # Per-test similarities

    # Similarity preservation: does binding preserve relative similarities?
    similarity_preservation_error: float  # Should be ~0

    # Retrieval fidelity: can we find what we encoded?
    retrieval_accuracy: float  # 0-1
    retrieval_ranks: list[int]  # Rank of correct item in retrieval

    # Multi-hop reasoning: can we chain bindings?
    multihop_accuracy: float  # 0-1
    multihop_similarities: list[float]

    # Primitive orthogonality: are primitives still distinguishable?
    primitive_mean_orthogonality: float  # Mean pairwise |similarity|, should be low
    primitive_max_similarity: float  # Max pairwise |similarity|, should be low

    # Codebook coherence: are codebook vectors well-separated?
    codebook_mean_similarity: float | None

    # Self-consistency: does the system agree with itself?
    resonator_convergence_rate: float  # Fraction that converge
    resonator_mean_iterations: float

    # Detection capability (if test bundle provided)
    detection_accuracy: float | None = None  # Fraction of known errors detected
    detection_confidence: float | None = None  # Mean confidence on detections
    false_negative_rate: float | None = None
    false_positive_rate: float | None = None

    def summary(self) -> dict[str, float]:
        """Return key metrics as simple dict."""
        return {
            "binding_accuracy": self.binding_accuracy,
            "similarity_preservation_error": self.similarity_preservation_error,
            "retrieval_accuracy": self.retrieval_accuracy,
            "multihop_accuracy": self.multihop_accuracy,
            "primitive_orthogonality": self.primitive_mean_orthogonality,
            "resonator_convergence": self.resonator_convergence_rate,
            "detection_accuracy": self.detection_accuracy,
        }


@dataclass
class DriftMeasurement:
    """Measures how far current state has drifted from a baseline."""
    baseline_id: str
    current_time: float

    # Per-primitive drift
    primitive_angular_drift: dict[str, float]  # Mean phase change per primitive
    primitive_similarity_to_baseline: dict[str, float]  # sim(current, baseline)

    # Aggregate drift
    mean_primitive_drift: float
    max_primitive_drift: float

    # Codebook drift (if applicable)
    mean_codebook_drift: float | None
    max_codebook_drift: float | None

    # Similarity structure drift
    similarity_matrix_frobenius_distance: float

    # Cumulative modifications
    total_modifications: int
    modification_history: list[dict]


@dataclass
class ModificationRecord:
    """Record of a single modification to the geometry."""
    timestamp: float
    modification_type: str  # "primitive_perturbation", "phase_drift", "codebook_modification"
    target: str  # Which primitive or codebook entry
    parameters: dict[str, Any]

    # State change
    pre_similarity_to_original: float
    post_similarity_to_original: float

    # Impact
    health_delta: dict[str, float] | None = None


# =============================================================================
# MAIN HARNESS
# =============================================================================


class VSASandboxHarness:
    """
    Sandbox environment for testing VSA self-modification.

    This harness provides:
    1. Baseline capture - snapshot full geometric state
    2. Controlled modification - apply perturbations of varying intensity
    3. Health monitoring - track metrics before/after modifications
    4. Drift measurement - quantify how far state has moved
    5. Checkpoint/restore - reliable state recovery
    """

    def __init__(
        self,
        dimensions: int = 2048,
        device: str | None = None,
        dtype: torch.dtype = torch.complex64,
        seed: int = 42,
    ):
        """Initialize the sandbox harness.

        Args:
            dimensions: Vector dimensionality
            device: Torch device ("mps", "cuda", "cpu", or None for auto)
            dtype: Complex dtype for vectors
            seed: Random seed for reproducibility
        """
        self.dimensions = dimensions
        self.dtype = dtype
        self.seed = seed

        # Auto-select device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"VSA Sandbox initialized: dims={dimensions}, device={self.device}")

        # Initialize random generator
        self._generator = torch.Generator(device="cpu")  # MPS doesn't support generators
        self._generator.manual_seed(seed)

        # Core state: primitives (the conceptual basis)
        self.primitives: dict[str, torch.Tensor] = {}
        self._original_primitives: dict[str, torch.Tensor] = {}  # Immutable reference

        # Core state: codebook (learned entity representations)
        self.codebook: dict[str, torch.Tensor] = {}
        self._original_codebook: dict[str, torch.Tensor] = {}

        # Audit trail
        self.modification_history: list[ModificationRecord] = []
        self.snapshots: dict[str, GeometricSnapshot] = {}
        self.health_history: list[HealthMetrics] = []

        # Initialize primitives
        self._initialize_primitives()

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def _initialize_primitives(self) -> None:
        """Initialize the primitive vectors (conceptual basis)."""
        primitive_names = [
            "low_stock",
            "high_margin_leak",
            "dead_item",
            "negative_inventory",
            "overstock",
            "price_discrepancy",
            "shrinkage_pattern",
            "margin_erosion",
            "high_velocity",
            "seasonal",
        ]

        for name in primitive_names:
            vec = self._seed_hash(f"primitive_{name}_v2")
            self.primitives[name] = vec
            self._original_primitives[name] = vec.clone()

        logger.info(f"Initialized {len(self.primitives)} primitives")

    def _seed_hash(self, seed_string: str) -> torch.Tensor:
        """Generate deterministic phasor vector from seed string."""
        hash_obj = hashlib.sha256(seed_string.encode())
        seed_int = int.from_bytes(hash_obj.digest(), "big") % (2**32)

        # Use CPU generator then move to device
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed_int)

        phases = torch.rand(self.dimensions, generator=gen) * 2 * math.pi
        vector = torch.exp(1j * phases).to(self.dtype).to(self.device)

        return self._normalize(vector)

    def _normalize(self, v: torch.Tensor) -> torch.Tensor:
        """Normalize vector to unit magnitude per component (phasor normalization)."""
        magnitudes = torch.abs(v)
        magnitudes = torch.clamp(magnitudes, min=1e-10)
        return v / magnitudes

    def _global_normalize(self, v: torch.Tensor) -> torch.Tensor:
        """L2 normalize entire vector."""
        norm = torch.sqrt(torch.sum(torch.abs(v) ** 2))
        return v / (norm + 1e-10)

    # =========================================================================
    # CORE VSA OPERATIONS (for testing)
    # =========================================================================

    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Bind two vectors via element-wise multiplication."""
        return self._normalize(a * b)

    def unbind(self, bound: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Unbind using complex conjugate."""
        return self._normalize(bound * torch.conj(key))

    def bundle(self, *vectors: torch.Tensor) -> torch.Tensor:
        """Bundle vectors via superposition."""
        result = vectors[0].clone()
        for v in vectors[1:]:
            result = result + v
        return self._normalize(result)

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute normalized cosine similarity."""
        dot = torch.sum(a * torch.conj(b))
        # Use abs() for complex norm since torch.norm doesn't support complex
        norm_a = torch.sqrt(torch.sum(torch.abs(a) ** 2))
        norm_b = torch.sqrt(torch.sum(torch.abs(b) ** 2))
        sim = (dot.real / (norm_a * norm_b + 1e-10)).item()
        return float(sim)

    def batch_similarity(self, query: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """Compute similarity between query and all codebook vectors."""
        dots = torch.sum(codebook * torch.conj(query), dim=-1)
        # Use abs() for complex norm
        query_norm = torch.sqrt(torch.sum(torch.abs(query) ** 2))
        codebook_norms = torch.sqrt(torch.sum(torch.abs(codebook) ** 2, dim=-1))
        return dots.real / (query_norm * codebook_norms + 1e-10)

    def angular_distance(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute mean angular distance between phasor vectors."""
        phase_diff = torch.angle(a * torch.conj(b))
        return float(torch.abs(phase_diff).mean())

    # =========================================================================
    # CODEBOOK MANAGEMENT
    # =========================================================================

    def add_to_codebook(self, entity: str) -> torch.Tensor:
        """Add entity to codebook, return its vector."""
        if entity not in self.codebook:
            vec = self._seed_hash(entity)
            self.codebook[entity] = vec
            self._original_codebook[entity] = vec.clone()
        return self.codebook[entity]

    def get_codebook_tensor(self) -> torch.Tensor | None:
        """Get codebook as stacked tensor."""
        if not self.codebook:
            return None
        return torch.stack(list(self.codebook.values()))

    def get_codebook_keys(self) -> list[str]:
        """Get codebook entity names."""
        return list(self.codebook.keys())

    # =========================================================================
    # BASELINE CAPTURE
    # =========================================================================

    def capture_baseline(self, description: str = "baseline") -> GeometricSnapshot:
        """Capture complete geometric state as baseline.

        This is our ground truth - captured BEFORE any self-modification.
        """
        snapshot_id = f"snapshot_{int(time.time() * 1000)}_{description}"

        # Capture primitive vectors
        primitive_vectors = {k: v.clone() for k, v in self.primitives.items()}

        # Capture codebook
        codebook_vectors = {k: v.clone() for k, v in self.codebook.items()}

        # Compute primitive similarity matrix
        prim_list = list(self.primitives.values())
        n_prims = len(prim_list)
        prim_sim_matrix = torch.zeros(n_prims, n_prims, device=self.device)
        for i in range(n_prims):
            for j in range(n_prims):
                prim_sim_matrix[i, j] = self.similarity(prim_list[i], prim_list[j])

        # Compute codebook similarity matrix (skip if too large)
        codebook_sim_matrix = None
        if len(self.codebook) <= 1000:
            cb_list = list(self.codebook.values())
            n_cb = len(cb_list)
            if n_cb > 0:
                codebook_sim_matrix = torch.zeros(n_cb, n_cb, device=self.device)
                for i in range(n_cb):
                    for j in range(n_cb):
                        codebook_sim_matrix[i, j] = self.similarity(cb_list[i], cb_list[j])

        # Compute primitive norms and phases
        primitive_norms = {k: float(torch.sqrt(torch.sum(torch.abs(v) ** 2))) for k, v in self.primitives.items()}
        primitive_phases = {k: torch.angle(v).clone() for k, v in self.primitives.items()}

        snapshot = GeometricSnapshot(
            timestamp=time.time(),
            snapshot_id=snapshot_id,
            primitive_vectors=primitive_vectors,
            codebook_vectors=codebook_vectors,
            primitive_similarity_matrix=prim_sim_matrix,
            codebook_similarity_matrix=codebook_sim_matrix,
            primitive_norms=primitive_norms,
            primitive_phases=primitive_phases,
            dimensions=self.dimensions,
            device=str(self.device),
            dtype=str(self.dtype),
            description=description,
            modifications_applied=list(self.modification_history),
        )

        self.snapshots[snapshot_id] = snapshot
        logger.info(f"Captured baseline: {snapshot_id}")

        return snapshot

    # =========================================================================
    # HEALTH METRICS
    # =========================================================================

    def measure_health(
        self,
        n_binding_tests: int = 50,
        n_retrieval_tests: int = 50,
        n_multihop_tests: int = 20,
        test_bundle: torch.Tensor | None = None,
        known_detections: dict[str, list[str]] | None = None,
    ) -> HealthMetrics:
        """Measure current system health.

        Args:
            n_binding_tests: Number of bind/unbind tests
            n_retrieval_tests: Number of retrieval tests
            n_multihop_tests: Number of multi-hop reasoning tests
            test_bundle: Optional bundle with known facts for detection testing
            known_detections: Ground truth {primitive: [entities]} for detection test

        Returns:
            HealthMetrics with all vital signs
        """
        timestamp = time.time()

        # Test binding accuracy
        binding_sims = []
        for i in range(n_binding_tests):
            a = self._seed_hash(f"binding_test_a_{i}")
            b = self._seed_hash(f"binding_test_b_{i}")
            bound = self.bind(a, b)
            recovered = self.unbind(bound, a)
            sim = self.similarity(recovered, b)
            binding_sims.append(sim)

        binding_accuracy = sum(1 for s in binding_sims if s > 0.9) / len(binding_sims)

        # Test similarity preservation
        sim_errors = []
        for i in range(n_binding_tests // 2):
            a = self._seed_hash(f"simpres_a_{i}")
            b = self._seed_hash(f"simpres_b_{i}")
            c = self._seed_hash(f"simpres_c_{i}")

            orig_sim = self.similarity(a, b)
            ac = self.bind(a, c)
            bc = self.bind(b, c)
            bound_sim = self.similarity(ac, bc)

            sim_errors.append(abs(orig_sim - bound_sim))

        sim_preservation_error = sum(sim_errors) / len(sim_errors) if sim_errors else 0.0

        # Test retrieval fidelity (needs codebook)
        retrieval_accuracy = 1.0
        retrieval_ranks = []
        if len(self.codebook) > 5:
            cb_tensor = self.get_codebook_tensor()
            cb_keys = self.get_codebook_keys()

            for i in range(min(n_retrieval_tests, len(self.codebook))):
                target_key = cb_keys[i]
                target_vec = self.codebook[target_key]

                # Add some noise to simulate noisy query
                noise = self._seed_hash(f"noise_{i}") * 0.1
                noisy_query = self._normalize(target_vec + noise)

                # Retrieve
                sims = self.batch_similarity(noisy_query, cb_tensor)
                sorted_indices = torch.argsort(sims, descending=True)

                # Find rank of target
                rank = (sorted_indices == i).nonzero(as_tuple=True)[0]
                if len(rank) > 0:
                    retrieval_ranks.append(int(rank[0]))
                else:
                    retrieval_ranks.append(len(cb_keys))

            retrieval_accuracy = sum(1 for r in retrieval_ranks if r == 0) / len(retrieval_ranks)

        # Test multi-hop reasoning
        multihop_sims = []
        for i in range(n_multihop_tests):
            # Create chain: a -> b -> c
            a = self._seed_hash(f"hop_a_{i}")
            b = self._seed_hash(f"hop_b_{i}")
            c = self._seed_hash(f"hop_c_{i}")

            ab = self.bind(a, b)
            bc = self.bind(b, c)

            # Query: given a and ab, recover b, then use bc to get c
            recovered_b = self.unbind(ab, a)
            recovered_c = self.unbind(bc, recovered_b)

            sim = self.similarity(recovered_c, c)
            multihop_sims.append(sim)

        multihop_accuracy = sum(1 for s in multihop_sims if s > 0.7) / len(multihop_sims)

        # Test primitive orthogonality
        prim_list = list(self.primitives.values())
        prim_sims = []
        for i, a in enumerate(prim_list):
            for j, b in enumerate(prim_list):
                if i < j:
                    prim_sims.append(abs(self.similarity(a, b)))

        prim_mean_orth = sum(prim_sims) / len(prim_sims) if prim_sims else 0.0
        prim_max_sim = max(prim_sims) if prim_sims else 0.0

        # Codebook coherence
        codebook_mean_sim = None
        if len(self.codebook) > 1:
            cb_sims = []
            cb_list = list(self.codebook.values())
            # Sample if large
            sample_size = min(100, len(cb_list))
            for i in range(sample_size):
                for j in range(i + 1, sample_size):
                    cb_sims.append(abs(self.similarity(cb_list[i], cb_list[j])))
            codebook_mean_sim = sum(cb_sims) / len(cb_sims) if cb_sims else None

        # Resonator convergence (simplified test)
        convergence_count = 0
        total_iters = []
        for i in range(20):
            query = self._seed_hash(f"resonator_test_{i}")
            # Simulate resonator: iteratively project onto primitive space
            x = query.clone()
            converged = False
            for it in range(50):
                # Simple projection: find most similar primitive
                best_sim = -1
                best_prim = None
                for prim in self.primitives.values():
                    sim = self.similarity(x, prim)
                    if sim > best_sim:
                        best_sim = sim
                        best_prim = prim

                # Blend toward best primitive
                if best_prim is not None:
                    x_new = self._normalize(0.7 * x + 0.3 * best_prim)
                    delta = torch.sqrt(torch.sum(torch.abs(x_new - x) ** 2))
                    x = x_new

                    if delta < 0.001:
                        converged = True
                        total_iters.append(it)
                        break

            if converged:
                convergence_count += 1
            else:
                total_iters.append(50)

        convergence_rate = convergence_count / 20
        mean_iters = sum(total_iters) / len(total_iters)

        # Detection testing (if provided)
        detection_accuracy = None
        detection_confidence = None
        fnr = None
        fpr = None

        if test_bundle is not None and known_detections is not None:
            detection_results = self._test_detection(test_bundle, known_detections)
            detection_accuracy = detection_results["accuracy"]
            detection_confidence = detection_results["mean_confidence"]
            fnr = detection_results["false_negative_rate"]
            fpr = detection_results["false_positive_rate"]

        metrics = HealthMetrics(
            timestamp=timestamp,
            binding_accuracy=binding_accuracy,
            binding_recovery_similarities=binding_sims,
            similarity_preservation_error=sim_preservation_error,
            retrieval_accuracy=retrieval_accuracy,
            retrieval_ranks=retrieval_ranks,
            multihop_accuracy=multihop_accuracy,
            multihop_similarities=multihop_sims,
            primitive_mean_orthogonality=prim_mean_orth,
            primitive_max_similarity=prim_max_sim,
            codebook_mean_similarity=codebook_mean_sim,
            resonator_convergence_rate=convergence_rate,
            resonator_mean_iterations=mean_iters,
            detection_accuracy=detection_accuracy,
            detection_confidence=detection_confidence,
            false_negative_rate=fnr,
            false_positive_rate=fpr,
        )

        self.health_history.append(metrics)
        return metrics

    def _test_detection(
        self,
        bundle: torch.Tensor,
        known_detections: dict[str, list[str]],
    ) -> dict[str, float]:
        """Test detection capability against known ground truth.

        Args:
            bundle: Bundled facts
            known_detections: {primitive_name: [entity_names]} ground truth

        Returns:
            Detection metrics
        """
        true_positives = 0
        false_negatives = 0
        false_positives = 0
        true_negatives = 0
        confidences = []

        # For each primitive, unbind and check if known entities are detected
        for prim_name, expected_entities in known_detections.items():
            if prim_name not in self.primitives:
                continue

            primitive = self.primitives[prim_name]
            unbound = self.unbind(bundle, primitive)

            # Get similarities to all codebook entries
            if not self.codebook:
                continue

            cb_tensor = self.get_codebook_tensor()
            cb_keys = self.get_codebook_keys()
            sims = self.batch_similarity(unbound, cb_tensor)

            # Threshold for detection
            threshold = 0.3

            for i, key in enumerate(cb_keys):
                sim = float(sims[i])
                is_expected = key in expected_entities
                is_detected = sim > threshold

                if is_expected and is_detected:
                    true_positives += 1
                    confidences.append(sim)
                elif is_expected and not is_detected:
                    false_negatives += 1
                elif not is_expected and is_detected:
                    false_positives += 1
                else:
                    true_negatives += 1

        total = true_positives + false_negatives + false_positives + true_negatives
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0

        expected_total = true_positives + false_negatives
        fnr = false_negatives / expected_total if expected_total > 0 else 0

        not_expected_total = false_positives + true_negatives
        fpr = false_positives / not_expected_total if not_expected_total > 0 else 0

        mean_conf = sum(confidences) / len(confidences) if confidences else 0

        return {
            "accuracy": accuracy,
            "mean_confidence": mean_conf,
            "false_negative_rate": fnr,
            "false_positive_rate": fpr,
            "true_positives": true_positives,
            "false_negatives": false_negatives,
        }

    # =========================================================================
    # DRIFT MEASUREMENT
    # =========================================================================

    def compute_drift_from_baseline(
        self,
        baseline: GeometricSnapshot,
    ) -> DriftMeasurement:
        """Compute how far current state has drifted from baseline."""

        # Per-primitive drift
        angular_drift = {}
        similarity_to_baseline = {}

        for name, current_vec in self.primitives.items():
            if name in baseline.primitive_vectors:
                baseline_vec = baseline.primitive_vectors[name].to(self.device)

                # Angular drift (mean phase change)
                angular_drift[name] = self.angular_distance(current_vec, baseline_vec)

                # Similarity
                similarity_to_baseline[name] = self.similarity(current_vec, baseline_vec)

        mean_prim_drift = sum(angular_drift.values()) / len(angular_drift) if angular_drift else 0
        max_prim_drift = max(angular_drift.values()) if angular_drift else 0

        # Codebook drift
        mean_cb_drift = None
        max_cb_drift = None
        if baseline.codebook_vectors:
            cb_drifts = []
            for name, current_vec in self.codebook.items():
                if name in baseline.codebook_vectors:
                    baseline_vec = baseline.codebook_vectors[name].to(self.device)
                    cb_drifts.append(self.angular_distance(current_vec, baseline_vec))

            if cb_drifts:
                mean_cb_drift = sum(cb_drifts) / len(cb_drifts)
                max_cb_drift = max(cb_drifts)

        # Similarity structure drift (Frobenius distance)
        prim_list = list(self.primitives.values())
        n_prims = len(prim_list)
        current_sim_matrix = torch.zeros(n_prims, n_prims, device=self.device)
        for i in range(n_prims):
            for j in range(n_prims):
                current_sim_matrix[i, j] = self.similarity(prim_list[i], prim_list[j])

        baseline_sim_matrix = baseline.primitive_similarity_matrix.to(self.device)
        # Frobenius norm for real-valued similarity matrix
        diff = current_sim_matrix - baseline_sim_matrix
        frobenius_dist = float(torch.sqrt(torch.sum(diff ** 2)))

        return DriftMeasurement(
            baseline_id=baseline.snapshot_id,
            current_time=time.time(),
            primitive_angular_drift=angular_drift,
            primitive_similarity_to_baseline=similarity_to_baseline,
            mean_primitive_drift=mean_prim_drift,
            max_primitive_drift=max_prim_drift,
            mean_codebook_drift=mean_cb_drift,
            max_codebook_drift=max_cb_drift,
            similarity_matrix_frobenius_distance=frobenius_dist,
            total_modifications=len(self.modification_history),
            modification_history=[
                {"type": m.modification_type, "target": m.target, "timestamp": m.timestamp}
                for m in self.modification_history
            ],
        )

    # =========================================================================
    # MODIFICATION MECHANISMS
    # =========================================================================

    def apply_primitive_perturbation(
        self,
        primitive_name: str,
        magnitude: float = 0.1,
        perturbation_type: str = "phase_noise",
    ) -> ModificationRecord:
        """Apply perturbation to a primitive vector.

        Args:
            primitive_name: Name of primitive to perturb
            magnitude: Strength of perturbation (0 = none, 1 = replace with random)
            perturbation_type: "phase_noise" (random phase shifts) or
                               "directional" (coherent drift in one direction)

        Returns:
            ModificationRecord documenting the change
        """
        if primitive_name not in self.primitives:
            raise ValueError(f"Unknown primitive: {primitive_name}")

        original = self._original_primitives[primitive_name]
        current = self.primitives[primitive_name]
        pre_sim = self.similarity(current, original)

        if perturbation_type == "phase_noise":
            # Add random phase shifts
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(time.time() * 1000) % (2**32))
            noise_phases = (torch.rand(self.dimensions, generator=gen) - 0.5) * 2 * math.pi * magnitude
            noise_phases = noise_phases.to(self.device)

            perturbation = torch.exp(1j * noise_phases).to(self.dtype)
            self.primitives[primitive_name] = self._normalize(current * perturbation)

        elif perturbation_type == "directional":
            # Coherent drift toward a random direction
            target = self._seed_hash(f"drift_target_{primitive_name}_{time.time()}")
            # Interpolate: (1-mag)*current + mag*target
            self.primitives[primitive_name] = self._normalize(
                (1 - magnitude) * current + magnitude * target
            )

        post_sim = self.similarity(self.primitives[primitive_name], original)

        record = ModificationRecord(
            timestamp=time.time(),
            modification_type="primitive_perturbation",
            target=primitive_name,
            parameters={"magnitude": magnitude, "perturbation_type": perturbation_type},
            pre_similarity_to_original=pre_sim,
            post_similarity_to_original=post_sim,
        )

        self.modification_history.append(record)
        logger.info(
            f"Applied {perturbation_type} perturbation to {primitive_name}: "
            f"sim to original {pre_sim:.4f} -> {post_sim:.4f}"
        )

        return record

    def apply_phase_drift(
        self,
        drift_angle: float = 0.1,
        drift_type: str = "uniform",
        affected_primitives: list[str] | None = None,
    ) -> list[ModificationRecord]:
        """Apply systematic phase drift to primitives.

        Args:
            drift_angle: Amount of phase drift in radians
            drift_type: "uniform" (same shift all dims),
                        "gradient" (linear gradient across dims),
                        "sinusoidal" (wave pattern)
            affected_primitives: Which primitives to affect (None = all)

        Returns:
            List of ModificationRecords
        """
        if affected_primitives is None:
            affected_primitives = list(self.primitives.keys())

        records = []

        for prim_name in affected_primitives:
            if prim_name not in self.primitives:
                continue

            original = self._original_primitives[prim_name]
            current = self.primitives[prim_name]
            pre_sim = self.similarity(current, original)

            # Generate drift pattern
            if drift_type == "uniform":
                drift = torch.full((self.dimensions,), drift_angle, device=self.device)
            elif drift_type == "gradient":
                drift = torch.linspace(0, drift_angle, self.dimensions, device=self.device)
            elif drift_type == "sinusoidal":
                x = torch.linspace(0, 4 * math.pi, self.dimensions, device=self.device)
                drift = drift_angle * torch.sin(x)
            else:
                raise ValueError(f"Unknown drift type: {drift_type}")

            # Apply phase drift
            phase_shift = torch.exp(1j * drift).to(self.dtype)
            self.primitives[prim_name] = self._normalize(current * phase_shift)

            post_sim = self.similarity(self.primitives[prim_name], original)

            record = ModificationRecord(
                timestamp=time.time(),
                modification_type="phase_drift",
                target=prim_name,
                parameters={"drift_angle": drift_angle, "drift_type": drift_type},
                pre_similarity_to_original=pre_sim,
                post_similarity_to_original=post_sim,
            )
            records.append(record)
            self.modification_history.append(record)

        logger.info(
            f"Applied {drift_type} phase drift ({drift_angle:.4f} rad) to "
            f"{len(affected_primitives)} primitives"
        )

        return records

    def apply_codebook_modification(
        self,
        entity_name: str,
        magnitude: float = 0.1,
        modification_type: str = "noise",
    ) -> ModificationRecord:
        """Apply modification to a codebook entry.

        Args:
            entity_name: Name of entity in codebook
            magnitude: Strength of modification
            modification_type: "noise" or "directional"

        Returns:
            ModificationRecord
        """
        if entity_name not in self.codebook:
            raise ValueError(f"Entity not in codebook: {entity_name}")

        original = self._original_codebook.get(entity_name)
        if original is None:
            original = self.codebook[entity_name].clone()
            self._original_codebook[entity_name] = original

        current = self.codebook[entity_name]
        pre_sim = self.similarity(current, original)

        if modification_type == "noise":
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(time.time() * 1000) % (2**32))
            noise_phases = (torch.rand(self.dimensions, generator=gen) - 0.5) * 2 * math.pi * magnitude
            noise_phases = noise_phases.to(self.device)
            perturbation = torch.exp(1j * noise_phases).to(self.dtype)
            self.codebook[entity_name] = self._normalize(current * perturbation)
        elif modification_type == "directional":
            target = self._seed_hash(f"cb_drift_{entity_name}_{time.time()}")
            self.codebook[entity_name] = self._normalize(
                (1 - magnitude) * current + magnitude * target
            )

        post_sim = self.similarity(self.codebook[entity_name], original)

        record = ModificationRecord(
            timestamp=time.time(),
            modification_type="codebook_modification",
            target=entity_name,
            parameters={"magnitude": magnitude, "modification_type": modification_type},
            pre_similarity_to_original=pre_sim,
            post_similarity_to_original=post_sim,
        )

        self.modification_history.append(record)
        return record

    # =========================================================================
    # CHECKPOINT / RESTORE
    # =========================================================================

    def restore_from_snapshot(self, snapshot: GeometricSnapshot) -> None:
        """Restore system state from a snapshot.

        Args:
            snapshot: GeometricSnapshot to restore from
        """
        # Restore primitives
        for name, vec in snapshot.primitive_vectors.items():
            self.primitives[name] = vec.clone().to(self.device)

        # Restore codebook
        self.codebook.clear()
        for name, vec in snapshot.codebook_vectors.items():
            self.codebook[name] = vec.clone().to(self.device)

        logger.info(f"Restored from snapshot: {snapshot.snapshot_id}")

    def verify_restore(self, snapshot: GeometricSnapshot) -> dict[str, float]:
        """Verify that current state matches snapshot exactly.

        Returns:
            Dict with max differences for primitives and codebook
        """
        prim_diffs = []
        for name, expected in snapshot.primitive_vectors.items():
            if name in self.primitives:
                actual = self.primitives[name]
                expected = expected.to(self.device)
                diff = float(torch.max(torch.abs(actual - expected)))
                prim_diffs.append(diff)

        cb_diffs = []
        for name, expected in snapshot.codebook_vectors.items():
            if name in self.codebook:
                actual = self.codebook[name]
                expected = expected.to(self.device)
                diff = float(torch.max(torch.abs(actual - expected)))
                cb_diffs.append(diff)

        return {
            "max_primitive_diff": max(prim_diffs) if prim_diffs else 0.0,
            "max_codebook_diff": max(cb_diffs) if cb_diffs else 0.0,
            "primitives_match": all(d < 1e-6 for d in prim_diffs),
            "codebook_match": all(d < 1e-6 for d in cb_diffs),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        """Save complete state to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save primitives
        torch.save(
            {k: v.cpu() for k, v in self.primitives.items()},
            path / "primitives.pt"
        )
        torch.save(
            {k: v.cpu() for k, v in self._original_primitives.items()},
            path / "original_primitives.pt"
        )

        # Save codebook
        torch.save(
            {k: v.cpu() for k, v in self.codebook.items()},
            path / "codebook.pt"
        )

        # Save metadata
        metadata = {
            "dimensions": self.dimensions,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "seed": self.seed,
            "modification_count": len(self.modification_history),
            "timestamp": time.time(),
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load state from disk checkpoint."""
        path = Path(path)

        # Load primitives
        primitives = torch.load(path / "primitives.pt", weights_only=True)
        self.primitives = {k: v.to(self.device) for k, v in primitives.items()}

        original_primitives = torch.load(path / "original_primitives.pt", weights_only=True)
        self._original_primitives = {k: v.to(self.device) for k, v in original_primitives.items()}

        # Load codebook
        codebook = torch.load(path / "codebook.pt", weights_only=True)
        self.codebook = {k: v.to(self.device) for k, v in codebook.items()}

        logger.info(f"Loaded checkpoint from {path}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_harness(
    dimensions: int = 2048,
    device: str | None = None,
) -> VSASandboxHarness:
    """Create a sandbox harness with sensible defaults for M4 Mac."""
    return VSASandboxHarness(
        dimensions=dimensions,
        device=device,
        dtype=torch.complex64,
    )
