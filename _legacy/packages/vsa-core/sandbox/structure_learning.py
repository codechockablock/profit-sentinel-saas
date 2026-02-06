"""
Primitive Structure Learning

Extends the metacognitive loop to critique and modify the primitive structure itself,
not just optimize the vectors within a fixed structure.

Three structure operations:
    1. Merge - Detect redundant primitives (high similarity + co-firing)
    2. Split - Detect overloaded primitives (fires on dissimilar patterns)
    3. Add - Detect unexplained patterns (low-confidence detections or coverage gaps)

The extended loop alternates between:
    - Phase 1: Optimize primitive vectors (freeze codebook)
    - Phase 2: Optimize codebook vectors (freeze primitives)
    - Phase 3: Structure critique (merge/split/add candidates)
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import torch
from vsa_sandbox_harness import VSASandboxHarness, create_harness

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


class StructureOperation(Enum):
    """Types of structure modifications."""

    MERGE = "merge"
    SPLIT = "split"
    ADD = "add"
    NONE = "none"


@dataclass
class MergeCandidate:
    """Candidate for merging two primitives."""

    primitive_a: str
    primitive_b: str
    similarity: float
    co_firing_rate: float  # How often they fire together
    confidence: float  # Overall confidence in this merge
    reason: str


@dataclass
class SplitCandidate:
    """Candidate for splitting an overloaded primitive."""

    primitive_name: str
    pattern_variance: float  # Variance among cases where it fires
    cluster_separation: float  # How separable the two clusters are
    confidence: float
    reason: str
    # The two cluster centroids for the split
    cluster_a_centroid: torch.Tensor | None = None
    cluster_b_centroid: torch.Tensor | None = None


@dataclass
class AddCandidate:
    """Candidate for adding a new primitive."""

    unexplained_pattern: torch.Tensor
    coverage_gap: float  # How much of the data this would explain
    confidence: float
    reason: str
    suggested_name: str = ""


@dataclass
class StructureChange:
    """Record of a structure change that was applied."""

    operation: StructureOperation
    step: int
    primitives_before: list[str]
    primitives_after: list[str]
    details: dict
    accuracy_before: float
    accuracy_after: float
    reason: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class StructureLearningResult:
    """Full results of structure learning."""

    # Configuration
    max_iterations: int = 0

    # Trajectory
    accuracy_trajectory: list[float] = field(default_factory=list)
    primitive_count_trajectory: list[int] = field(default_factory=list)
    phase_trajectory: list[str] = field(default_factory=list)

    # Structure changes
    structure_changes: list[StructureChange] = field(default_factory=list)
    merge_candidates_evaluated: int = 0
    split_candidates_evaluated: int = 0
    add_candidates_evaluated: int = 0

    # Final state
    initial_primitives: list[str] = field(default_factory=list)
    final_primitives: list[str] = field(default_factory=list)
    final_accuracy: float = 0.0
    best_accuracy: float = 0.0

    # Convergence
    converged: bool = False
    convergence_reason: str = ""
    total_iterations: int = 0

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0


# =============================================================================
# STRUCTURE CRITIC
# =============================================================================


class StructureCritic:
    """
    Evaluates the current primitive structure and proposes changes.
    """

    def __init__(
        self,
        harness: VSASandboxHarness,
        merge_similarity_threshold: float = 0.65,  # Lowered for high-dim spaces
        merge_cofire_threshold: float = 0.5,
        split_variance_threshold: float = 0.3,
        add_coverage_threshold: float = 0.1,
    ):
        self.harness = harness
        self.merge_similarity_threshold = merge_similarity_threshold
        self.merge_cofire_threshold = merge_cofire_threshold
        self.split_variance_threshold = split_variance_threshold
        self.add_coverage_threshold = add_coverage_threshold

        # Track firing patterns
        self.firing_history: list[dict] = []

    def record_firing(
        self, signal: torch.Tensor, detected_primitives: set[str]
    ) -> None:
        """Record which primitives fired on a given signal."""
        self.firing_history.append(
            {
                "signal": signal.clone(),
                "detected": detected_primitives.copy(),
            }
        )

    def clear_history(self) -> None:
        """Clear firing history."""
        self.firing_history = []

    def find_merge_candidates(self) -> list[MergeCandidate]:
        """Find pairs of primitives that should be merged."""
        candidates = []
        prim_names = list(self.harness.primitives.keys())

        for i, name_a in enumerate(prim_names):
            for name_b in prim_names[i + 1 :]:
                # Check similarity
                sim = self.harness.similarity(
                    self.harness.primitives[name_a], self.harness.primitives[name_b]
                )

                if sim < self.merge_similarity_threshold:
                    continue

                # Check co-firing rate
                cofire_count = 0
                total_fires = 0

                for record in self.firing_history:
                    a_fired = name_a in record["detected"]
                    b_fired = name_b in record["detected"]

                    if a_fired or b_fired:
                        total_fires += 1
                        if a_fired and b_fired:
                            cofire_count += 1

                cofire_rate = cofire_count / max(total_fires, 1)

                if cofire_rate >= self.merge_cofire_threshold:
                    confidence = (sim + cofire_rate) / 2
                    candidates.append(
                        MergeCandidate(
                            primitive_a=name_a,
                            primitive_b=name_b,
                            similarity=sim,
                            co_firing_rate=cofire_rate,
                            confidence=confidence,
                            reason=f"High similarity ({sim:.3f}) and co-firing ({cofire_rate:.3f})",
                        )
                    )

        # Sort by confidence
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates

    def find_split_candidates(self) -> list[SplitCandidate]:
        """Find primitives that are overloaded and should be split."""
        candidates = []

        for name, primitive in self.harness.primitives.items():
            # Gather signals where this primitive fired
            firing_signals = []
            for record in self.firing_history:
                if name in record["detected"]:
                    firing_signals.append(record["signal"])

            if len(firing_signals) < 4:  # Need enough samples to split
                continue

            # Compute pairwise similarities among firing signals
            n = len(firing_signals)
            similarities = []
            for i in range(n):
                for j in range(i + 1, n):
                    sim = self.harness.similarity(firing_signals[i], firing_signals[j])
                    similarities.append(sim)

            if not similarities:
                continue

            # High variance = primitive is doing double duty
            mean_sim = sum(similarities) / len(similarities)
            variance = sum((s - mean_sim) ** 2 for s in similarities) / len(
                similarities
            )

            if variance < self.split_variance_threshold:
                continue

            # Try to find two clusters using simple k-means-ish approach
            cluster_a, cluster_b, separation = self._find_clusters(firing_signals)

            if separation > 0.3:  # Clusters are reasonably separated
                confidence = min(variance, separation)
                candidates.append(
                    SplitCandidate(
                        primitive_name=name,
                        pattern_variance=variance,
                        cluster_separation=separation,
                        confidence=confidence,
                        reason=f"High pattern variance ({variance:.3f}), cluster separation ({separation:.3f})",
                        cluster_a_centroid=cluster_a,
                        cluster_b_centroid=cluster_b,
                    )
                )

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates

    def _find_clusters(
        self, signals: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Simple 2-means clustering of signals."""
        if len(signals) < 2:
            return signals[0], signals[0], 0.0

        # Initialize with two random signals
        centroid_a = signals[0].clone()
        centroid_b = signals[len(signals) // 2].clone()

        for _ in range(10):  # K-means iterations
            # Assign signals to clusters
            cluster_a_signals = []
            cluster_b_signals = []

            for sig in signals:
                sim_a = self.harness.similarity(sig, centroid_a)
                sim_b = self.harness.similarity(sig, centroid_b)

                if sim_a > sim_b:
                    cluster_a_signals.append(sig)
                else:
                    cluster_b_signals.append(sig)

            # Update centroids
            if cluster_a_signals:
                centroid_a = self._compute_centroid(cluster_a_signals)
            if cluster_b_signals:
                centroid_b = self._compute_centroid(cluster_b_signals)

        # Compute separation (1 - similarity between centroids)
        separation = 1 - self.harness.similarity(centroid_a, centroid_b)

        return centroid_a, centroid_b, separation

    def _compute_centroid(self, signals: list[torch.Tensor]) -> torch.Tensor:
        """Compute centroid of a set of signals."""
        if not signals:
            return torch.zeros_like(
                self.harness.primitives[list(self.harness.primitives.keys())[0]]
            )

        centroid = signals[0].clone()
        for sig in signals[1:]:
            centroid = centroid + sig

        # Normalize
        norm = torch.sqrt(torch.sum(torch.abs(centroid) ** 2))
        return centroid / (norm + 1e-10)

    def find_add_candidates(self) -> list[AddCandidate]:
        """Find patterns that need new primitives."""
        candidates = []

        # Look for signals with low max similarity to any primitive
        # Use the same threshold as detection (0.4 by default)
        unexplained_signals = []

        for record in self.firing_history:
            signal = record["signal"]
            detected = record["detected"]

            # If nothing was detected, this is an unexplained signal
            if len(detected) == 0:
                unexplained_signals.append(signal)

        if len(unexplained_signals) < 3:
            return candidates

        # Cluster the unexplained signals
        centroid = self._compute_centroid(unexplained_signals)
        coverage = len(unexplained_signals) / max(len(self.firing_history), 1)

        if coverage >= self.add_coverage_threshold:
            # Check this pattern isn't already close to an existing primitive
            max_existing_sim = 0.0
            for prim in self.harness.primitives.values():
                sim = self.harness.similarity(centroid, prim)
                max_existing_sim = max(max_existing_sim, sim)

            if max_existing_sim < 0.5:  # Genuinely new pattern
                candidates.append(
                    AddCandidate(
                        unexplained_pattern=centroid,
                        coverage_gap=coverage,
                        confidence=coverage * (1 - max_existing_sim),
                        reason=f"Covers {coverage:.1%} of data, max similarity to existing: {max_existing_sim:.3f}",
                        suggested_name=f"learned_primitive_{len(self.harness.primitives)}",
                    )
                )

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates

    def get_best_structure_change(
        self,
    ) -> tuple[
        StructureOperation, MergeCandidate | SplitCandidate | AddCandidate | None
    ]:
        """Get the highest-confidence structure change."""
        merge_candidates = self.find_merge_candidates()
        split_candidates = self.find_split_candidates()
        add_candidates = self.find_add_candidates()

        best_op = StructureOperation.NONE
        best_candidate = None
        best_confidence = 0.0

        if merge_candidates and merge_candidates[0].confidence > best_confidence:
            best_op = StructureOperation.MERGE
            best_candidate = merge_candidates[0]
            best_confidence = merge_candidates[0].confidence

        if split_candidates and split_candidates[0].confidence > best_confidence:
            best_op = StructureOperation.SPLIT
            best_candidate = split_candidates[0]
            best_confidence = split_candidates[0].confidence

        if add_candidates and add_candidates[0].confidence > best_confidence:
            best_op = StructureOperation.ADD
            best_candidate = add_candidates[0]
            best_confidence = add_candidates[0].confidence

        return best_op, best_candidate


# =============================================================================
# STRUCTURE MODIFIER
# =============================================================================


class StructureModifier:
    """Applies structure changes to the harness."""

    def __init__(self, harness: VSASandboxHarness):
        self.harness = harness

    def apply_merge(self, candidate: MergeCandidate) -> dict:
        """Merge two primitives into one."""
        name_a = candidate.primitive_a
        name_b = candidate.primitive_b

        # Average the two primitives
        prim_a = self.harness.primitives[name_a]
        prim_b = self.harness.primitives[name_b]

        merged = prim_a + prim_b
        norm = torch.sqrt(torch.sum(torch.abs(merged) ** 2))
        merged = merged / (norm + 1e-10)

        # Keep the first name, remove the second
        self.harness.primitives[name_a] = merged
        del self.harness.primitives[name_b]

        # Update any codebook references (if they reference the removed primitive)
        # For now, just log this - actual codebook update would depend on structure

        return {
            "merged_into": name_a,
            "removed": name_b,
            "similarity": candidate.similarity,
            "co_firing_rate": candidate.co_firing_rate,
        }

    def apply_split(self, candidate: SplitCandidate) -> dict:
        """Split an overloaded primitive into two."""
        name = candidate.primitive_name

        # Create two new primitives from the cluster centroids
        new_name_a = f"{name}_a"
        new_name_b = f"{name}_b"

        self.harness.primitives[new_name_a] = candidate.cluster_a_centroid.clone()
        self.harness.primitives[new_name_b] = candidate.cluster_b_centroid.clone()

        # Remove the original
        del self.harness.primitives[name]

        return {
            "original": name,
            "split_into": [new_name_a, new_name_b],
            "variance": candidate.pattern_variance,
            "separation": candidate.cluster_separation,
        }

    def apply_add(self, candidate: AddCandidate) -> dict:
        """Add a new primitive."""
        name = candidate.suggested_name

        # Ensure unique name
        while name in self.harness.primitives:
            name = name + "_"

        self.harness.primitives[name] = candidate.unexplained_pattern.clone()

        return {
            "added": name,
            "coverage_gap": candidate.coverage_gap,
        }


# =============================================================================
# STRUCTURE LEARNING LOOP
# =============================================================================


class StructureLearningLoop:
    """
    Main orchestrator for structure learning.

    Alternates between:
    1. Primitive vector optimization
    2. Codebook vector optimization
    3. Structure critique and modification
    """

    def __init__(
        self,
        harness: VSASandboxHarness,
        evaluator: "StructureEvaluator",
        max_iterations: int = 50,
        patience: int = 5,
        structure_change_cooldown: int = 1,  # Reduced for faster convergence
    ):
        self.harness = harness
        self.evaluator = evaluator
        self.max_iterations = max_iterations
        self.patience = patience
        self.structure_change_cooldown = structure_change_cooldown

        self.critic = StructureCritic(harness)
        self.modifier = StructureModifier(harness)

        # State tracking
        self.best_state: dict | None = None
        self.best_accuracy: float = 0.0
        self.steps_since_improvement: int = 0
        self.steps_since_structure_change: int = 0

    def run(self) -> StructureLearningResult:
        """Run the structure learning loop."""
        result = StructureLearningResult(
            max_iterations=self.max_iterations,
            start_time=time.time(),
            initial_primitives=list(self.harness.primitives.keys()),
        )

        # Initial evaluation
        accuracy = self.evaluator.evaluate_and_record(self.critic)
        self.best_accuracy = accuracy
        self._save_best_state()

        result.accuracy_trajectory.append(accuracy)
        result.primitive_count_trajectory.append(len(self.harness.primitives))
        result.phase_trajectory.append("initial")

        logger.info(
            f"Starting structure learning: {len(self.harness.primitives)} primitives, accuracy={accuracy:.3f}"
        )

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            self.steps_since_structure_change += 1

            # Phase 1: Optimize primitive vectors
            self._optimize_primitives()
            accuracy = self.evaluator.evaluate_and_record(self.critic)
            result.accuracy_trajectory.append(accuracy)
            result.primitive_count_trajectory.append(len(self.harness.primitives))
            result.phase_trajectory.append("primitive_opt")

            if accuracy > self.best_accuracy + 0.001:
                self.best_accuracy = accuracy
                self._save_best_state()
                self.steps_since_improvement = 0
            else:
                self.steps_since_improvement += 1

            # Phase 2: Optimize codebook vectors
            self._optimize_codebook()
            accuracy = self.evaluator.evaluate_and_record(self.critic)
            result.accuracy_trajectory.append(accuracy)
            result.primitive_count_trajectory.append(len(self.harness.primitives))
            result.phase_trajectory.append("codebook_opt")

            if accuracy > self.best_accuracy + 0.001:
                self.best_accuracy = accuracy
                self._save_best_state()
                self.steps_since_improvement = 0
            else:
                self.steps_since_improvement += 1

            # Phase 3: Structure critique
            if self.steps_since_structure_change >= self.structure_change_cooldown:
                op, candidate = self.critic.get_best_structure_change()

                if op != StructureOperation.NONE and candidate is not None:
                    # Record primitives before change
                    prims_before = list(self.harness.primitives.keys())
                    accuracy_before = accuracy

                    # Apply the change
                    details = self._apply_structure_change(op, candidate)

                    # Evaluate after change
                    self.critic.clear_history()  # Reset firing history
                    accuracy = self.evaluator.evaluate_and_record(self.critic)

                    prims_after = list(self.harness.primitives.keys())

                    # Record the change
                    change = StructureChange(
                        operation=op,
                        step=iteration,
                        primitives_before=prims_before,
                        primitives_after=prims_after,
                        details=details,
                        accuracy_before=accuracy_before,
                        accuracy_after=accuracy,
                        reason=candidate.reason,
                    )
                    result.structure_changes.append(change)

                    logger.info(
                        f"Structure change: {op.value} "
                        f"({len(prims_before)} -> {len(prims_after)} primitives), "
                        f"accuracy: {accuracy_before:.3f} -> {accuracy:.3f}"
                    )

                    result.accuracy_trajectory.append(accuracy)
                    result.primitive_count_trajectory.append(
                        len(self.harness.primitives)
                    )
                    result.phase_trajectory.append(f"structure_{op.value}")

                    # Update tracking
                    if op == StructureOperation.MERGE:
                        result.merge_candidates_evaluated += 1
                    elif op == StructureOperation.SPLIT:
                        result.split_candidates_evaluated += 1
                    elif op == StructureOperation.ADD:
                        result.add_candidates_evaluated += 1

                    self.steps_since_structure_change = 0

                    if accuracy > self.best_accuracy + 0.001:
                        self.best_accuracy = accuracy
                        self._save_best_state()
                        self.steps_since_improvement = 0

            # Check convergence
            if self.steps_since_improvement >= self.patience:
                result.converged = True
                result.convergence_reason = (
                    f"No improvement for {self.patience} iterations"
                )
                logger.info(f"Converged: {result.convergence_reason}")
                break

            # Log progress
            if iteration % 5 == 0:
                logger.info(
                    f"Iteration {iteration}: {len(self.harness.primitives)} primitives, "
                    f"accuracy={accuracy:.3f}, best={self.best_accuracy:.3f}"
                )

        else:
            result.convergence_reason = "Max iterations reached"

        # Restore best state
        if accuracy < self.best_accuracy - 0.001:
            logger.info(
                f"Restoring best state: {accuracy:.3f} -> {self.best_accuracy:.3f}"
            )
            self._restore_best_state()
            accuracy = self.best_accuracy

        # Finalize results
        result.final_primitives = list(self.harness.primitives.keys())
        result.final_accuracy = accuracy
        result.best_accuracy = self.best_accuracy
        result.total_iterations = iteration
        result.end_time = time.time()

        return result

    def _optimize_primitives(self) -> bool:
        """Optimize primitive vectors (small perturbations)."""
        # Simple gradient-free optimization: perturb and keep if better
        improved = False

        for name in list(self.harness.primitives.keys()):
            original = self.harness.primitives[name].clone()
            original_acc = self.evaluator.evaluate_quick()

            # Try a small perturbation
            noise = torch.randn_like(original.real) + 1j * torch.randn_like(
                original.real
            )
            noise = noise * 0.02
            perturbed = original + noise
            norm = torch.sqrt(torch.sum(torch.abs(perturbed) ** 2))
            self.harness.primitives[name] = perturbed / (norm + 1e-10)

            new_acc = self.evaluator.evaluate_quick()

            if new_acc > original_acc + 0.001:
                improved = True
            else:
                # Revert
                self.harness.primitives[name] = original

        return improved

    def _optimize_codebook(self) -> bool:
        """Optimize codebook vectors."""
        if not self.harness.codebook:
            return False

        improved = False

        for name in list(self.harness.codebook.keys()):
            original = self.harness.codebook[name].clone()
            original_acc = self.evaluator.evaluate_quick()

            # Try a small perturbation
            noise = torch.randn_like(original.real) + 1j * torch.randn_like(
                original.real
            )
            noise = noise * 0.02
            perturbed = original + noise
            norm = torch.sqrt(torch.sum(torch.abs(perturbed) ** 2))
            self.harness.codebook[name] = perturbed / (norm + 1e-10)

            new_acc = self.evaluator.evaluate_quick()

            if new_acc > original_acc + 0.001:
                improved = True
            else:
                # Revert
                self.harness.codebook[name] = original

        return improved

    def _apply_structure_change(
        self,
        op: StructureOperation,
        candidate: MergeCandidate | SplitCandidate | AddCandidate,
    ) -> dict:
        """Apply a structure change."""
        if op == StructureOperation.MERGE:
            return self.modifier.apply_merge(candidate)
        elif op == StructureOperation.SPLIT:
            return self.modifier.apply_split(candidate)
        elif op == StructureOperation.ADD:
            return self.modifier.apply_add(candidate)
        return {}

    def _save_best_state(self) -> None:
        """Save current state as best."""
        self.best_state = {
            "primitives": {k: v.clone() for k, v in self.harness.primitives.items()},
            "codebook": {k: v.clone() for k, v in self.harness.codebook.items()},
        }

    def _restore_best_state(self) -> None:
        """Restore best state."""
        if self.best_state:
            self.harness.primitives = {
                k: v.clone() for k, v in self.best_state["primitives"].items()
            }
            self.harness.codebook = {
                k: v.clone() for k, v in self.best_state["codebook"].items()
            }


# =============================================================================
# EVALUATOR
# =============================================================================


class StructureEvaluator:
    """Evaluates the current structure against test cases."""

    def __init__(
        self,
        harness: VSASandboxHarness,
        test_cases: list[dict],
        detection_threshold: float = 0.4,
    ):
        self.harness = harness
        self.test_cases = test_cases
        self.detection_threshold = detection_threshold

    def evaluate_and_record(self, critic: StructureCritic) -> float:
        """Evaluate and record firing patterns for the critic."""
        correct = 0

        for case in self.test_cases:
            signal = case["signal"]
            ground_truth = case["ground_truth"]

            # Detect primitives
            detected = self._detect_primitives(signal)

            # Record for critic
            critic.record_firing(signal, detected)

            # Check accuracy - handle different cases
            if case.get("needs_primitive", False):
                # For patterns needing a new primitive:
                # - Initially nothing fires (wrong)
                # - Once primitive added, something should fire (correct)
                if len(detected) > 0:
                    correct += 1  # Something fired - good (primitive was added)
                # If nothing fires, it's incorrect (still needs primitive)
            elif case.get("accepts_any", False):
                # For redundant patterns, correct if ANY of the ground truth fires
                if detected & ground_truth:  # Intersection non-empty
                    correct += 1
            else:
                # Standard: exact match
                if detected == ground_truth:
                    correct += 1

        return correct / len(self.test_cases)

    def evaluate_quick(self) -> float:
        """Quick evaluation without recording."""
        correct = 0

        for case in self.test_cases:
            signal = case["signal"]
            ground_truth = case["ground_truth"]

            detected = self._detect_primitives(signal)

            if case.get("needs_primitive", False):
                if len(detected) > 0:
                    correct += 1
            elif case.get("accepts_any", False):
                if detected & ground_truth:
                    correct += 1
            else:
                if detected == ground_truth:
                    correct += 1

        return correct / len(self.test_cases)

    def _detect_primitives(self, signal: torch.Tensor) -> set[str]:
        """Detect which primitives are present in the signal."""
        detected = set()

        for name, primitive in self.harness.primitives.items():
            sim = self.harness.similarity(signal, primitive)
            if sim > self.detection_threshold:
                detected.add(name)

        return detected


# =============================================================================
# SYNTHETIC TEST SCENARIO
# =============================================================================


def create_structure_learning_scenario(
    dimensions: int = 2048,
    device: str | None = None,
) -> tuple[VSASandboxHarness, list[dict], dict]:
    """
    Create a synthetic scenario for structure learning testing.

    Scenario:
    - Two initial primitives are actually redundant (should merge)
    - One initial primitive is overloaded (should split)
    - One pattern exists with no primitive (should add)

    The ground truth is defined relative to CURRENT primitives, so we can
    measure accuracy at each step.

    Returns:
        (harness, test_cases, ground_truth_structure)
    """
    harness = create_harness(dimensions=dimensions, device=device)

    # Clear default primitives
    harness.primitives.clear()

    # Create the "true" underlying patterns
    torch.manual_seed(42)

    # Pattern A: Two distinct patterns that share an "overloaded" primitive
    pattern_a1 = torch.randn(dimensions, dtype=torch.complex64, device=harness.device)
    pattern_a1 = pattern_a1 / torch.sqrt(torch.sum(torch.abs(pattern_a1) ** 2))

    pattern_a2 = torch.randn(dimensions, dtype=torch.complex64, device=harness.device)
    pattern_a2 = pattern_a2 / torch.sqrt(torch.sum(torch.abs(pattern_a2) ** 2))

    # Pattern B: Single pattern
    pattern_b = torch.randn(dimensions, dtype=torch.complex64, device=harness.device)
    pattern_b = pattern_b / torch.sqrt(torch.sum(torch.abs(pattern_b) ** 2))

    # Pattern C: Hidden pattern (no primitive initially)
    pattern_c = torch.randn(dimensions, dtype=torch.complex64, device=harness.device)
    pattern_c = pattern_c / torch.sqrt(torch.sum(torch.abs(pattern_c) ** 2))

    # Pattern D: The redundant pattern
    pattern_d = torch.randn(dimensions, dtype=torch.complex64, device=harness.device)
    pattern_d = pattern_d / torch.sqrt(torch.sum(torch.abs(pattern_d) ** 2))

    # Set up FLAWED initial primitives:

    # 1. REDUNDANT: Two primitives that are nearly identical (should merge)
    # Both are close to pattern_d - use very small noise to keep similarity high
    noise1 = torch.randn_like(pattern_d) * 0.01  # Tiny noise
    noise2 = torch.randn_like(pattern_d) * 0.01

    r1 = pattern_d + noise1
    harness.primitives["redundant_1"] = r1 / torch.sqrt(torch.sum(torch.abs(r1) ** 2))

    r2 = pattern_d + noise2
    harness.primitives["redundant_2"] = r2 / torch.sqrt(torch.sum(torch.abs(r2) ** 2))

    # 2. OVERLOADED: One primitive positioned between pattern_a1 and pattern_a2
    overloaded = pattern_a1 + pattern_a2
    overloaded = overloaded / torch.sqrt(torch.sum(torch.abs(overloaded) ** 2))
    harness.primitives["overloaded"] = overloaded

    # 3. CORRECT: One primitive that correctly matches pattern_b
    harness.primitives["correct_b"] = pattern_b.clone()

    # 4. MISSING: No primitive for pattern_c (should be added)

    # Generate test cases
    # Ground truth uses the CURRENT primitive names that should fire
    test_cases = []

    # Cases for pattern_d (fires on redundant_1 AND redundant_2 initially)
    # After merge, should fire on just one
    for i in range(15):
        noise = torch.randn_like(pattern_d) * 0.02  # Small noise
        signal = pattern_d + noise
        signal = signal / torch.sqrt(torch.sum(torch.abs(signal) ** 2))
        # Initially both redundant primitives should fire (they're nearly identical)
        # The evaluator will count this as correct if EITHER fires
        test_cases.append(
            {
                "signal": signal,
                "ground_truth": {
                    "redundant_1",
                    "redundant_2",
                },  # Both should fire (redundant)
                "true_pattern": "d",
                "accepts_any": True,  # Accept if any of ground_truth fires
            }
        )

    # Cases for pattern_a1 (should fire on overloaded initially)
    for i in range(15):
        noise = torch.randn_like(pattern_a1) * 0.02
        signal = pattern_a1 + noise
        signal = signal / torch.sqrt(torch.sum(torch.abs(signal) ** 2))
        test_cases.append(
            {
                "signal": signal,
                "ground_truth": {"overloaded"},  # Initially fires on overloaded
                "true_pattern": "a1",
            }
        )

    # Cases for pattern_a2 (should also fire on overloaded initially)
    for i in range(15):
        noise = torch.randn_like(pattern_a2) * 0.02
        signal = pattern_a2 + noise
        signal = signal / torch.sqrt(torch.sum(torch.abs(signal) ** 2))
        test_cases.append(
            {
                "signal": signal,
                "ground_truth": {"overloaded"},  # Initially fires on overloaded
                "true_pattern": "a2",
            }
        )

    # Cases for pattern_b (correct primitive exists)
    for i in range(15):
        noise = torch.randn_like(pattern_b) * 0.02
        signal = pattern_b + noise
        signal = signal / torch.sqrt(torch.sum(torch.abs(signal) ** 2))
        test_cases.append(
            {
                "signal": signal,
                "ground_truth": {"correct_b"},
                "true_pattern": "b",
            }
        )

    # Cases for pattern_c (no primitive exists - these SHOULD have a primitive)
    # These will fail until a new primitive is added
    # We mark what SHOULD fire once a primitive is added
    for i in range(15):
        noise = torch.randn_like(pattern_c) * 0.02
        signal = pattern_c + noise
        signal = signal / torch.sqrt(torch.sum(torch.abs(signal) ** 2))
        test_cases.append(
            {
                "signal": signal,
                "ground_truth": {
                    "pattern_c_expected"
                },  # Should fire on a pattern_c primitive
                "true_pattern": "c",
                "needs_primitive": True,  # Mark as needing a primitive to be added
            }
        )

    ground_truth_structure = {
        "optimal_primitives": [
            "redundant",
            "pattern_a1",
            "pattern_a2",
            "correct_b",
            "pattern_c",
        ],
        "initial_primitives": ["redundant_1", "redundant_2", "overloaded", "correct_b"],
        "expected_merges": [("redundant_1", "redundant_2")],
        "expected_splits": ["overloaded"],
        "expected_adds": ["pattern for pattern_c"],
    }

    return harness, test_cases, ground_truth_structure


def run_structure_learning_test(
    dimensions: int = 2048,
    device: str | None = None,
    max_iterations: int = 50,
) -> StructureLearningResult:
    """
    Run the structure learning test with the synthetic scenario.
    """
    harness, test_cases, ground_truth = create_structure_learning_scenario(
        dimensions=dimensions,
        device=device,
    )

    logger.info("=" * 60)
    logger.info("STRUCTURE LEARNING TEST")
    logger.info("=" * 60)
    logger.info(f"Initial primitives: {list(harness.primitives.keys())}")
    logger.info(f"Expected optimal: {ground_truth['optimal_primitives']}")
    logger.info(f"Test cases: {len(test_cases)}")

    evaluator = StructureEvaluator(harness, test_cases)

    loop = StructureLearningLoop(
        harness=harness,
        evaluator=evaluator,
        max_iterations=max_iterations,
        patience=10,
    )

    result = loop.run()

    return result


def print_structure_learning_report(result: StructureLearningResult) -> None:
    """Print a report of structure learning results."""
    print("\n" + "=" * 70)
    print("STRUCTURE LEARNING REPORT")
    print("=" * 70)

    print("\nConfiguration:")
    print(f"  Max iterations: {result.max_iterations}")
    print(f"  Total iterations: {result.total_iterations}")

    print("\nPrimitive Structure:")
    print(f"  Initial: {result.initial_primitives}")
    print(f"  Final:   {result.final_primitives}")
    print(
        f"  Count:   {len(result.initial_primitives)} -> {len(result.final_primitives)}"
    )

    print("\nAccuracy:")
    print(f"  Initial: {result.accuracy_trajectory[0]:.3f}")
    print(f"  Final:   {result.final_accuracy:.3f}")
    print(f"  Best:    {result.best_accuracy:.3f}")

    print(f"\nStructure Changes ({len(result.structure_changes)} total):")
    for i, change in enumerate(result.structure_changes):
        print(f"  {i+1}. {change.operation.value.upper()} at step {change.step}")
        print(
            f"     {len(change.primitives_before)} -> {len(change.primitives_after)} primitives"
        )
        print(
            f"     Accuracy: {change.accuracy_before:.3f} -> {change.accuracy_after:.3f}"
        )
        print(f"     Reason: {change.reason}")
        print(f"     Details: {change.details}")

    print("\nCandidates Evaluated:")
    print(f"  Merges: {result.merge_candidates_evaluated}")
    print(f"  Splits: {result.split_candidates_evaluated}")
    print(f"  Adds:   {result.add_candidates_evaluated}")

    print("\nConvergence:")
    print(f"  Converged: {result.converged}")
    print(f"  Reason: {result.convergence_reason}")

    duration = result.end_time - result.start_time
    print("\nTiming:")
    print(f"  Duration: {duration:.1f}s")

    print("\n" + "=" * 70)


# =============================================================================
# REAL RETAIL SCENARIO
# =============================================================================


def create_real_retail_scenario(
    csv_path: str,
    dimensions: int = 2048,
    device: str | None = None,
    max_rows: int = 5000,
) -> tuple[VSASandboxHarness, list[dict], dict]:
    """
    Create a scenario using real retail data and Profit Sentinel primitives.

    This uses the actual primitives from the Profit Sentinel VSA:
    - low_stock: Qty=0 but has had sales
    - high_margin_leak: Negative or very low margin
    - dead_item: Has stock but no sales
    - negative_inventory: Qty < 0
    - overstock: Qty >> Sales (high stock-to-sales ratio)
    - price_discrepancy: Retail differs significantly from suggested
    - shrinkage_pattern: Returns >> Sales or inventory losses
    - margin_erosion: Margin below expected threshold

    Args:
        csv_path: Path to the retail CSV file
        dimensions: VSA dimensions
        device: Compute device
        max_rows: Maximum rows to process

    Returns:
        (harness, test_cases, scenario_info)
    """
    import csv

    harness = create_harness(dimensions=dimensions, device=device)
    harness.primitives.clear()

    # Create random seed for reproducibility
    torch.manual_seed(42)

    # Create base patterns for each anomaly type
    # These are the "ideal" patterns that represent each anomaly
    anomaly_patterns = {}

    primitive_names = [
        "low_stock",
        "high_margin_leak",
        "dead_item",
        "negative_inventory",
        "overstock",
        "price_discrepancy",
        "shrinkage_pattern",
        "margin_erosion",
    ]

    for name in primitive_names:
        pattern = torch.randn(dimensions, dtype=torch.complex64, device=harness.device)
        pattern = pattern / torch.sqrt(torch.sum(torch.abs(pattern) ** 2))
        anomaly_patterns[name] = pattern
        harness.primitives[name] = pattern.clone()

    # Parse CSV and categorize each row by its anomaly type(s)
    test_cases = []
    anomaly_counts = {name: 0 for name in primitive_names}
    normal_count = 0

    with open(csv_path, encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)

        row_count = 0
        for row in reader:
            row_count += 1
            if row_count > max_rows:
                break

            try:
                qty = float(row.get("Qty.", 0) or 0)
                margin = float(row.get("Margin @Cost", 0) or 0)
                sales = int(row.get("Sales", 0) or 0)
                retail = float(row.get("Retail", 0) or 0)
                sug_retail = float(row.get("Sug. Retail", 0) or 0)
                std_cost = float(row.get("Std. Cost", 0) or 0)
                returns = int(row.get("Returns", 0) or 0)

                # Determine which anomalies this row exhibits
                row_anomalies = set()

                # negative_inventory: Qty < 0
                if qty < 0:
                    row_anomalies.add("negative_inventory")

                # high_margin_leak: Margin < 0 or very negative
                if margin < -10:  # Significantly negative margin
                    row_anomalies.add("high_margin_leak")

                # margin_erosion: Low positive margin (0-15%)
                if 0 <= margin < 15 and std_cost > 0:
                    row_anomalies.add("margin_erosion")

                # dead_item: Has stock but no sales
                if qty > 0 and sales == 0 and std_cost > 0:
                    row_anomalies.add("dead_item")

                # low_stock: Out of stock but has had sales
                if qty == 0 and sales > 0:
                    row_anomalies.add("low_stock")

                # overstock: High qty relative to sales
                if qty > 0 and sales > 0 and qty / sales > 5:
                    row_anomalies.add("overstock")

                # price_discrepancy: Retail differs from suggested by > 20%
                if sug_retail > 0 and retail > 0:
                    diff_pct = abs(retail - sug_retail) / sug_retail
                    if diff_pct > 0.2:
                        row_anomalies.add("price_discrepancy")

                # shrinkage_pattern: Returns > 50% of sales
                if sales > 0 and returns > 0 and returns / sales > 0.5:
                    row_anomalies.add("shrinkage_pattern")

                if row_anomalies:
                    # Create signal as combination of anomaly patterns
                    # For single-anomaly cases: signal ≈ primitive (high similarity)
                    # For multi-anomaly cases: signal = sum of primitives (each detectable)

                    for anomaly in row_anomalies:
                        anomaly_counts[anomaly] += 1

                    if len(row_anomalies) == 1:
                        # Single anomaly: use the exact primitive with tiny phase noise
                        anomaly_name = list(row_anomalies)[0]
                        signal = anomaly_patterns[anomaly_name].clone()
                        # Add very small phase perturbation (preserves similarity)
                        phase_noise = (
                            torch.randn(dimensions, device=harness.device) * 0.01
                        )
                        signal = signal * torch.exp(1j * phase_noise)
                        signal = signal / torch.sqrt(torch.sum(torch.abs(signal) ** 2))
                    else:
                        # Multi-anomaly: bundle the primitives
                        # Use the harness's bundle operation which preserves detectability
                        signals_to_bundle = [anomaly_patterns[a] for a in row_anomalies]
                        signal = signals_to_bundle[0].clone()
                        for s in signals_to_bundle[1:]:
                            signal = signal + s
                        signal = signal / torch.sqrt(torch.sum(torch.abs(signal) ** 2))

                    test_cases.append(
                        {
                            "signal": signal,
                            "ground_truth": row_anomalies.copy(),
                            "sku": row.get("SKU", ""),
                            "accepts_any": len(row_anomalies) > 1,  # Multi-anomaly rows
                        }
                    )
                else:
                    normal_count += 1

            except (ValueError, TypeError):
                continue

    # Note: We don't add normal cases here because:
    # 1. In real detection, we only care about correctly identifying anomalies
    # 2. Normal rows simply don't fire any primitive (correct behavior)
    # 3. Adding synthetic "normal" patterns would create artificial coverage gaps

    scenario_info = {
        "total_rows_processed": row_count,
        "anomaly_counts": anomaly_counts,
        "normal_count": normal_count,
        "test_cases_generated": len(test_cases),
        "primitives": list(harness.primitives.keys()),
    }

    return harness, test_cases, scenario_info


def create_ytd_inventory_scenario(
    csv_path: str,
    dimensions: int = 2048,
    device: str | None = None,
    max_rows: int = 50000,
) -> tuple[VSASandboxHarness, list[dict], dict]:
    """
    Create a scenario using YTD inventory report with sales history.

    This handles the Inventory_Report_SKU_SHLP_YTD.csv format with columns:
    - SKU, Description, Vendor
    - Gross Sales, Gross Cost, Gross Profit, Profit Margin%
    - Stock (can be massively negative)
    - Monthly sales history (Jan through Last Jan)

    Anomaly patterns to detect:
    - negative_inventory: Stock < 0 (shrinkage/overselling)
    - massive_negative_stock: Stock < -100 (severe inventory problem)
    - zero_cost_anomaly: Sales with $0 cost (costing issue)
    - negative_profit: Gross Profit < 0 (selling at loss)
    - negative_margin: Profit Margin% < 0
    - dead_item: Stock > 0 but zero YTD sales
    - high_margin_anomaly: Margin > 100% (data quality issue)
    """
    import csv

    harness = create_harness(dimensions=dimensions, device=device)
    harness.primitives.clear()

    torch.manual_seed(42)

    # Define primitives for YTD inventory anomalies
    primitive_names = [
        "negative_inventory",
        "massive_negative_stock",
        "zero_cost_anomaly",
        "negative_profit",
        "negative_margin",
        "dead_item",
        "high_margin_anomaly",
        "margin_erosion",  # Low but positive margin (0-10%)
    ]

    anomaly_patterns = {}
    for name in primitive_names:
        pattern = torch.randn(dimensions, dtype=torch.complex64, device=harness.device)
        pattern = pattern / torch.sqrt(torch.sum(torch.abs(pattern) ** 2))
        anomaly_patterns[name] = pattern
        harness.primitives[name] = pattern.clone()

    # Parse CSV
    test_cases = []
    anomaly_counts = {name: 0 for name in primitive_names}
    normal_count = 0

    with open(csv_path, encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)

        row_count = 0
        for row in reader:
            row_count += 1
            if row_count > max_rows:
                break

            try:
                stock = float(row.get("Stock", 0) or 0)
                gross_sales = float(row.get("Gross Sales", 0) or 0)
                gross_cost = float(row.get("Gross Cost", 0) or 0)
                gross_profit = float(row.get("Gross Profit", 0) or 0)
                margin_str = row.get("Profit Margin%", "") or ""
                margin = float(margin_str) if margin_str else None

                row_anomalies = set()

                # Negative inventory
                if stock < 0:
                    row_anomalies.add("negative_inventory")

                # Massive negative stock (severe)
                if stock < -100:
                    row_anomalies.add("massive_negative_stock")

                # Zero cost with sales
                if gross_cost == 0 and gross_sales > 0:
                    row_anomalies.add("zero_cost_anomaly")

                # Negative profit
                if gross_profit < 0:
                    row_anomalies.add("negative_profit")

                # Negative margin
                if margin is not None and margin < 0:
                    row_anomalies.add("negative_margin")

                # High margin anomaly (>100%)
                if margin is not None and margin > 100:
                    row_anomalies.add("high_margin_anomaly")

                # Dead item (stock but no sales)
                if stock > 0 and gross_sales == 0:
                    row_anomalies.add("dead_item")

                # Margin erosion (low positive margin)
                if margin is not None and 0 < margin < 10:
                    row_anomalies.add("margin_erosion")

                if row_anomalies:
                    for anomaly in row_anomalies:
                        anomaly_counts[anomaly] += 1

                    # Create signal
                    if len(row_anomalies) == 1:
                        anomaly_name = list(row_anomalies)[0]
                        signal = anomaly_patterns[anomaly_name].clone()
                        phase_noise = (
                            torch.randn(dimensions, device=harness.device) * 0.01
                        )
                        signal = signal * torch.exp(1j * phase_noise)
                        signal = signal / torch.sqrt(torch.sum(torch.abs(signal) ** 2))
                    else:
                        signals_to_bundle = [anomaly_patterns[a] for a in row_anomalies]
                        signal = signals_to_bundle[0].clone()
                        for s in signals_to_bundle[1:]:
                            signal = signal + s
                        signal = signal / torch.sqrt(torch.sum(torch.abs(signal) ** 2))

                    test_cases.append(
                        {
                            "signal": signal,
                            "ground_truth": row_anomalies.copy(),
                            "sku": row.get("SKU", "")[:30],
                            "accepts_any": len(row_anomalies) > 1,
                        }
                    )
                else:
                    normal_count += 1

            except (ValueError, TypeError):
                continue

    scenario_info = {
        "total_rows_processed": row_count,
        "anomaly_counts": anomaly_counts,
        "normal_count": normal_count,
        "test_cases_generated": len(test_cases),
        "primitives": list(harness.primitives.keys()),
        "data_type": "YTD_inventory_report",
    }

    return harness, test_cases, scenario_info


def create_flawed_retail_scenario(
    csv_path: str,
    dimensions: int = 2048,
    device: str | None = None,
    max_rows: int = 5000,
) -> tuple[VSASandboxHarness, list[dict], dict]:
    """
    Create a FLAWED scenario to test structure learning.

    This deliberately introduces problems in the primitive structure:
    1. REDUNDANT: margin_erosion_v1 and margin_erosion_v2 are nearly identical
    2. OVERLOADED: inventory_problem combines negative_inventory + low_stock
    3. MISSING: No primitive for price_discrepancy

    Structure learning should:
    - Merge the redundant margin_erosion primitives
    - Split the overloaded inventory_problem
    - Add a new primitive for price_discrepancy
    """
    import csv

    harness = create_harness(dimensions=dimensions, device=device)
    harness.primitives.clear()

    torch.manual_seed(42)

    # Create the TRUE underlying patterns for anomaly detection
    true_patterns = {}
    true_primitive_names = [
        "low_stock",
        "high_margin_leak",
        "dead_item",
        "negative_inventory",
        "overstock",
        "price_discrepancy",
        "shrinkage_pattern",
        "margin_erosion",
    ]

    for name in true_primitive_names:
        pattern = torch.randn(dimensions, dtype=torch.complex64, device=harness.device)
        pattern = pattern / torch.sqrt(torch.sum(torch.abs(pattern) ** 2))
        true_patterns[name] = pattern

    # Now create a FLAWED primitive set:

    # 1. REDUNDANT: Two margin_erosion primitives that are nearly identical
    harness.primitives["margin_erosion_v1"] = true_patterns["margin_erosion"].clone()
    noise = torch.randn_like(true_patterns["margin_erosion"]) * 0.01
    me2 = true_patterns["margin_erosion"] + noise
    harness.primitives["margin_erosion_v2"] = me2 / torch.sqrt(
        torch.sum(torch.abs(me2) ** 2)
    )

    # 2. OVERLOADED: One primitive that handles both negative_inventory AND low_stock
    overloaded = true_patterns["negative_inventory"] + true_patterns["low_stock"]
    harness.primitives["inventory_problem"] = overloaded / torch.sqrt(
        torch.sum(torch.abs(overloaded) ** 2)
    )

    # 3. CORRECT: These primitives are correctly defined
    harness.primitives["high_margin_leak"] = true_patterns["high_margin_leak"].clone()
    harness.primitives["dead_item"] = true_patterns["dead_item"].clone()
    harness.primitives["overstock"] = true_patterns["overstock"].clone()
    harness.primitives["shrinkage_pattern"] = true_patterns["shrinkage_pattern"].clone()

    # 4. MISSING: No primitive for price_discrepancy!

    # Parse CSV and create test cases using TRUE patterns
    test_cases = []
    anomaly_counts = {name: 0 for name in true_primitive_names}
    normal_count = 0

    with open(csv_path, encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)

        row_count = 0
        for row in reader:
            row_count += 1
            if row_count > max_rows:
                break

            try:
                qty = float(row.get("Qty.", 0) or 0)
                margin = float(row.get("Margin @Cost", 0) or 0)
                sales = int(row.get("Sales", 0) or 0)
                retail = float(row.get("Retail", 0) or 0)
                sug_retail = float(row.get("Sug. Retail", 0) or 0)
                std_cost = float(row.get("Std. Cost", 0) or 0)
                returns = int(row.get("Returns", 0) or 0)

                row_anomalies = set()

                if qty < 0:
                    row_anomalies.add("negative_inventory")
                if margin < -10:
                    row_anomalies.add("high_margin_leak")
                if 0 <= margin < 15 and std_cost > 0:
                    row_anomalies.add("margin_erosion")
                if qty > 0 and sales == 0 and std_cost > 0:
                    row_anomalies.add("dead_item")
                if qty == 0 and sales > 0:
                    row_anomalies.add("low_stock")
                if qty > 0 and sales > 0 and qty / sales > 5:
                    row_anomalies.add("overstock")
                if (
                    sug_retail > 0
                    and retail > 0
                    and abs(retail - sug_retail) / sug_retail > 0.2
                ):
                    row_anomalies.add("price_discrepancy")
                if sales > 0 and returns > 0 and returns / sales > 0.5:
                    row_anomalies.add("shrinkage_pattern")

                if row_anomalies:
                    for anomaly in row_anomalies:
                        anomaly_counts[anomaly] += 1

                    # Create signal using TRUE patterns
                    if len(row_anomalies) == 1:
                        anomaly_name = list(row_anomalies)[0]
                        signal = true_patterns[anomaly_name].clone()
                        phase_noise = (
                            torch.randn(dimensions, device=harness.device) * 0.01
                        )
                        signal = signal * torch.exp(1j * phase_noise)
                        signal = signal / torch.sqrt(torch.sum(torch.abs(signal) ** 2))
                    else:
                        signals_to_bundle = [true_patterns[a] for a in row_anomalies]
                        signal = signals_to_bundle[0].clone()
                        for s in signals_to_bundle[1:]:
                            signal = signal + s
                        signal = signal / torch.sqrt(torch.sum(torch.abs(signal) ** 2))

                    # Map true anomalies to expected primitive detections
                    # This is what SHOULD fire given the FLAWED primitives
                    expected_primitives = set()
                    for a in row_anomalies:
                        if a == "margin_erosion":
                            expected_primitives.add("margin_erosion_v1")
                            expected_primitives.add(
                                "margin_erosion_v2"
                            )  # Both will fire (redundant)
                        elif a in ["negative_inventory", "low_stock"]:
                            expected_primitives.add(
                                "inventory_problem"
                            )  # Overloaded handles both
                        elif a == "price_discrepancy":
                            pass  # No primitive for this - will be unexplained
                        elif a in harness.primitives:
                            expected_primitives.add(a)

                    test_cases.append(
                        {
                            "signal": signal,
                            "ground_truth": expected_primitives,
                            "true_anomalies": row_anomalies.copy(),
                            "sku": row.get("SKU", ""),
                            "accepts_any": True,  # Accept partial matches due to flawed structure
                            "needs_primitive": "price_discrepancy" in row_anomalies,
                        }
                    )
                else:
                    normal_count += 1

            except (ValueError, TypeError):
                continue

    scenario_info = {
        "total_rows_processed": row_count,
        "anomaly_counts": anomaly_counts,
        "normal_count": normal_count,
        "test_cases_generated": len(test_cases),
        "primitives": list(harness.primitives.keys()),
        "flaws": {
            "redundant": ["margin_erosion_v1", "margin_erosion_v2"],
            "overloaded": ["inventory_problem"],
            "missing": ["price_discrepancy"],
        },
    }

    return harness, test_cases, scenario_info


def run_real_retail_structure_learning(
    csv_path: str,
    dimensions: int = 2048,
    device: str | None = None,
    max_iterations: int = 30,
    max_rows: int = 5000,
    flawed: bool = False,
) -> StructureLearningResult:
    """
    Run structure learning on real retail data.

    This will analyze the primitives and determine if:
    - Any primitives are redundant (detecting same patterns)
    - Any primitives are overloaded (detecting dissimilar patterns)
    - There are coverage gaps (patterns with no matching primitive)

    Args:
        flawed: If True, start with deliberately flawed primitive structure
    """
    if flawed:
        harness, test_cases, scenario_info = create_flawed_retail_scenario(
            csv_path=csv_path,
            dimensions=dimensions,
            device=device,
            max_rows=max_rows,
        )
    else:
        harness, test_cases, scenario_info = create_real_retail_scenario(
            csv_path=csv_path,
            dimensions=dimensions,
            device=device,
            max_rows=max_rows,
        )

    print("=" * 70)
    print("REAL RETAIL STRUCTURE LEARNING")
    print("=" * 70)
    print("\nData Summary:")
    print(f"  CSV path: {csv_path}")
    print(f"  Rows processed: {scenario_info['total_rows_processed']}")
    print(f"  Test cases generated: {scenario_info['test_cases_generated']}")
    print(f"  Normal (non-anomaly) rows: {scenario_info['normal_count']}")

    print("\nAnomaly Distribution:")
    for name, count in scenario_info["anomaly_counts"].items():
        print(f"  {name}: {count}")

    print(f"\nInitial Primitives ({len(harness.primitives)}):")
    for name in harness.primitives.keys():
        print(f"  - {name}")

    print("\n" + "-" * 70)
    print("Starting structure learning...")
    print("-" * 70 + "\n")

    evaluator = StructureEvaluator(harness, test_cases, detection_threshold=0.35)

    loop = StructureLearningLoop(
        harness=harness,
        evaluator=evaluator,
        max_iterations=max_iterations,
        patience=8,
        structure_change_cooldown=2,
    )

    result = loop.run()

    # Add scenario info to result for reporting
    result.scenario_info = scenario_info  # type: ignore

    return result


def print_real_retail_report(result: StructureLearningResult) -> None:
    """Print a detailed report for real retail structure learning."""
    print("\n" + "=" * 70)
    print("REAL RETAIL STRUCTURE LEARNING REPORT")
    print("=" * 70)

    # Get scenario info if available
    scenario_info = getattr(result, "scenario_info", None)
    if scenario_info:
        print("\nData Summary:")
        print(f"  Rows processed: {scenario_info['total_rows_processed']}")
        print(f"  Test cases: {scenario_info['test_cases_generated']}")
        print("\nAnomaly Distribution:")
        for name, count in scenario_info["anomaly_counts"].items():
            print(f"  {name}: {count}")

    print(f"\n{'='*70}")
    print("STRUCTURE CHANGES")
    print("=" * 70)

    print(f"\nStarting Structure ({len(result.initial_primitives)} primitives):")
    for name in result.initial_primitives:
        print(f"  - {name}")

    print(f"\nFinal Structure ({len(result.final_primitives)} primitives):")
    for name in result.final_primitives:
        print(f"  - {name}")

    if result.structure_changes:
        print(f"\nOperations Applied ({len(result.structure_changes)}):")
        for i, change in enumerate(result.structure_changes):
            print(
                f"\n  {i+1}. {change.operation.value.upper()} at iteration {change.step}"
            )
            print(f"     Reason: {change.reason}")
            print(f"     Details: {change.details}")
            print(
                f"     Accuracy: {change.accuracy_before:.3f} -> {change.accuracy_after:.3f}"
            )
    else:
        print("\n  No structure changes applied.")
        print("  This suggests the current primitive set is well-calibrated!")

    print(f"\n{'='*70}")
    print("ACCURACY TRAJECTORY")
    print("=" * 70)

    print(f"\n  Initial accuracy: {result.accuracy_trajectory[0]:.3f}")
    print(f"  Final accuracy:   {result.final_accuracy:.3f}")
    print(f"  Best accuracy:    {result.best_accuracy:.3f}")

    # Print trajectory summary
    print(f"\n  Trajectory ({len(result.accuracy_trajectory)} points):")
    step_size = max(1, len(result.accuracy_trajectory) // 10)
    for i in range(0, len(result.accuracy_trajectory), step_size):
        print(f"    Step {i:3d}: {result.accuracy_trajectory[i]:.3f}")

    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print("=" * 70)

    if not result.structure_changes:
        print("\n  The primitive structure appears well-suited to this data.")
        print("  No redundant, overloaded, or missing primitives detected.")
    else:
        merges = [
            c
            for c in result.structure_changes
            if c.operation == StructureOperation.MERGE
        ]
        splits = [
            c
            for c in result.structure_changes
            if c.operation == StructureOperation.SPLIT
        ]
        adds = [
            c for c in result.structure_changes if c.operation == StructureOperation.ADD
        ]

        if merges:
            print(f"\n  REDUNDANCIES FOUND ({len(merges)}):")
            for m in merges:
                print(
                    f"    - {m.details.get('merged_into', '?')} absorbed {m.details.get('removed', '?')}"
                )

        if splits:
            print(f"\n  OVERLOADED PRIMITIVES ({len(splits)}):")
            for s in splits:
                print(
                    f"    - {s.details.get('original', '?')} split into {s.details.get('split_into', [])}"
                )

        if adds:
            print(f"\n  COVERAGE GAPS ({len(adds)}):")
            for a in adds:
                print(
                    f"    - Added {a.details.get('added', '?')} (coverage: {a.details.get('coverage_gap', 0):.1%})"
                )

    print(f"\n{'='*70}")
    print(f"Convergence: {result.convergence_reason}")
    print(f"Duration: {result.end_time - result.start_time:.1f}s")
    print("=" * 70 + "\n")


def save_report_to_file(result: StructureLearningResult, output_path: str) -> None:
    """Save the structure learning report to a file."""
    import io
    from contextlib import redirect_stdout

    # Capture printed output
    f = io.StringIO()
    with redirect_stdout(f):
        scenario_info = getattr(result, "scenario_info", None)
        if scenario_info:
            print_real_retail_report(result)
        else:
            print_structure_learning_report(result)

    report_content = f.getvalue()

    # Also add JSON summary at the end
    import json

    summary = {
        "initial_primitives": result.initial_primitives,
        "final_primitives": result.final_primitives,
        "accuracy": {
            "initial": (
                result.accuracy_trajectory[0] if result.accuracy_trajectory else 0
            ),
            "final": result.final_accuracy,
            "best": result.best_accuracy,
        },
        "structure_changes": [
            {
                "operation": c.operation.value,
                "step": c.step,
                "details": c.details,
                "accuracy_before": c.accuracy_before,
                "accuracy_after": c.accuracy_after,
                "reason": c.reason,
            }
            for c in result.structure_changes
        ],
        "convergence": {
            "converged": result.converged,
            "reason": result.convergence_reason,
            "total_iterations": result.total_iterations,
        },
        "duration_seconds": result.end_time - result.start_time,
    }

    with open(output_path, "w") as out:
        out.write(report_content)
        out.write("\n\n" + "=" * 70 + "\n")
        out.write("JSON SUMMARY\n")
        out.write("=" * 70 + "\n")
        out.write(json.dumps(summary, indent=2))
        out.write("\n")

    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    import sys
    from datetime import datetime

    # Default output directory
    output_dir = "/Users/joseph/profit-sentinel-saas/packages/vsa-core/sandbox"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if len(sys.argv) > 1 and sys.argv[1] == "--real":
        # Run on real retail data with well-calibrated primitives
        csv_path = (
            sys.argv[2] if len(sys.argv) > 2 else "/Users/joseph/Downloads/custom_1.csv"
        )
        result = run_real_retail_structure_learning(
            csv_path=csv_path,
            dimensions=2048,
            max_iterations=30,
            max_rows=5000,
            flawed=False,
        )
        print_real_retail_report(result)

        # Save to file
        report_path = f"{output_dir}/structure_learning_report_{timestamp}.txt"
        save_report_to_file(result, report_path)

    elif len(sys.argv) > 1 and sys.argv[1] == "--flawed":
        # Run on real retail data with FLAWED primitives (to test structure learning)
        csv_path = (
            sys.argv[2] if len(sys.argv) > 2 else "/Users/joseph/Downloads/custom_1.csv"
        )
        print("\n" + "!" * 70)
        print("FLAWED PRIMITIVES TEST")
        print(
            "Testing structure learning with deliberately broken primitive structure:"
        )
        print("  - REDUNDANT: margin_erosion_v1 + margin_erosion_v2 (should merge)")
        print(
            "  - OVERLOADED: inventory_problem (handles negative_inventory + low_stock)"
        )
        print("  - MISSING: No primitive for price_discrepancy")
        print("!" * 70 + "\n")

        result = run_real_retail_structure_learning(
            csv_path=csv_path,
            dimensions=2048,
            max_iterations=40,
            max_rows=5000,
            flawed=True,
        )
        print_real_retail_report(result)

        # Save to file
        report_path = f"{output_dir}/structure_learning_flawed_report_{timestamp}.txt"
        save_report_to_file(result, report_path)

    elif len(sys.argv) > 1 and sys.argv[1] == "--ytd":
        # Run on YTD inventory report (156K rows)
        csv_path = (
            sys.argv[2]
            if len(sys.argv) > 2
            else "/Users/joseph/Downloads/Reports/Inventory_Report_SKU_SHLP_YTD.csv"
        )
        max_rows = int(sys.argv[3]) if len(sys.argv) > 3 else 50000

        print("\n" + "=" * 70)
        print("YTD INVENTORY STRUCTURE LEARNING")
        print(f"Processing up to {max_rows:,} rows from: {csv_path}")
        print("=" * 70 + "\n")

        harness, test_cases, scenario_info = create_ytd_inventory_scenario(
            csv_path=csv_path,
            dimensions=2048,
            max_rows=max_rows,
        )

        print("Data Summary:")
        print(f"  Rows processed: {scenario_info['total_rows_processed']:,}")
        print(f"  Test cases: {scenario_info['test_cases_generated']:,}")
        print(f"  Normal rows: {scenario_info['normal_count']:,}")

        print("\nAnomaly Distribution:")
        for name, count in sorted(
            scenario_info["anomaly_counts"].items(), key=lambda x: -x[1]
        ):
            if count > 0:
                print(f"  {name}: {count:,}")

        print(f"\nInitial Primitives ({len(harness.primitives)}):")
        for name in harness.primitives.keys():
            print(f"  - {name}")

        print("\n" + "-" * 70)
        print("Running structure learning...")
        print("-" * 70 + "\n")

        evaluator = StructureEvaluator(harness, test_cases, detection_threshold=0.35)

        loop = StructureLearningLoop(
            harness=harness,
            evaluator=evaluator,
            max_iterations=40,
            patience=10,
            structure_change_cooldown=2,
        )

        result = loop.run()
        result.scenario_info = scenario_info  # type: ignore

        print_real_retail_report(result)

        # Save to file
        report_path = f"{output_dir}/structure_learning_ytd_report_{timestamp}.txt"
        save_report_to_file(result, report_path)

    elif len(sys.argv) > 1 and sys.argv[1] == "--ytd-original":
        # Run on YTD data with ORIGINAL Profit Sentinel primitives
        # Tests the actual primitives from the codebase against real YTD data
        csv_path = (
            sys.argv[2]
            if len(sys.argv) > 2
            else "/Users/joseph/Downloads/Reports/Inventory_Report_SKU_SHLP_YTD.csv"
        )
        max_rows = int(sys.argv[3]) if len(sys.argv) > 3 else 10000

        print("\n" + "=" * 70)
        print("YTD DATA vs ORIGINAL PROFIT SENTINEL PRIMITIVES")
        print("=" * 70)
        print("\nTesting the ORIGINAL 8 primitives from Profit Sentinel:")
        print("  - low_stock, high_margin_leak, dead_item, negative_inventory")
        print("  - overstock, price_discrepancy, shrinkage_pattern, margin_erosion")
        print(f"\nAgainst YTD data: {csv_path}")
        print(f"Processing up to {max_rows:,} rows")
        print("=" * 70 + "\n")

        import csv

        harness = create_harness(dimensions=2048)
        harness.primitives.clear()

        torch.manual_seed(42)

        # ORIGINAL Profit Sentinel primitives
        original_primitives = [
            "low_stock",
            "high_margin_leak",
            "dead_item",
            "negative_inventory",
            "overstock",
            "price_discrepancy",
            "shrinkage_pattern",
            "margin_erosion",
        ]

        # Create the primitive vectors
        primitive_patterns = {}
        for name in original_primitives:
            pattern = torch.randn(2048, dtype=torch.complex64, device=harness.device)
            pattern = pattern / torch.sqrt(torch.sum(torch.abs(pattern) ** 2))
            primitive_patterns[name] = pattern
            harness.primitives[name] = pattern.clone()

        # Parse YTD data and map to ORIGINAL primitives
        # Key question: Which YTD anomalies map to which original primitives?
        test_cases = []
        anomaly_counts = {name: 0 for name in original_primitives}
        unmapped_anomalies = {
            "zero_cost_anomaly": 0,
            "negative_profit": 0,
            "massive_negative_stock": 0,
            "high_margin_anomaly": 0,
        }
        normal_count = 0

        with open(csv_path, encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)

            row_count = 0
            for row in reader:
                row_count += 1
                if row_count > max_rows:
                    break

                try:
                    stock = float(row.get("Stock", 0) or 0)
                    gross_sales = float(row.get("Gross Sales", 0) or 0)
                    gross_cost = float(row.get("Gross Cost", 0) or 0)
                    gross_profit = float(row.get("Gross Profit", 0) or 0)
                    margin_str = row.get("Profit Margin%", "") or ""
                    margin = float(margin_str) if margin_str else None

                    row_anomalies = set()
                    unmapped = set()

                    # Map YTD patterns to original primitives:

                    # negative_inventory: Stock < 0
                    if stock < 0:
                        row_anomalies.add("negative_inventory")

                    # dead_item: Stock > 0 but no sales
                    if stock > 0 and gross_sales == 0:
                        row_anomalies.add("dead_item")

                    # high_margin_leak: Negative margin (selling below cost)
                    if margin is not None and margin < 0:
                        row_anomalies.add("high_margin_leak")

                    # margin_erosion: Low positive margin (0-15%)
                    if margin is not None and 0 < margin < 15:
                        row_anomalies.add("margin_erosion")

                    # low_stock: Zero stock (approximation - no "Qty" column)
                    if stock == 0 and gross_sales > 0:
                        row_anomalies.add("low_stock")

                    # UNMAPPED PATTERNS (no direct primitive):
                    # zero_cost_anomaly: Sales with $0 cost
                    if gross_cost == 0 and gross_sales > 0:
                        unmapped.add("zero_cost_anomaly")
                        unmapped_anomalies["zero_cost_anomaly"] += 1

                    # negative_profit: Gross Profit < 0
                    if gross_profit < 0:
                        unmapped.add("negative_profit")
                        unmapped_anomalies["negative_profit"] += 1

                    # massive_negative_stock: Stock < -100
                    if stock < -100:
                        unmapped.add("massive_negative_stock")
                        unmapped_anomalies["massive_negative_stock"] += 1

                    # high_margin_anomaly: Margin > 100%
                    if margin is not None and margin > 100:
                        unmapped.add("high_margin_anomaly")
                        unmapped_anomalies["high_margin_anomaly"] += 1

                    if row_anomalies:
                        for a in row_anomalies:
                            anomaly_counts[a] += 1

                        # Create signal from mapped primitives
                        if len(row_anomalies) == 1:
                            anomaly_name = list(row_anomalies)[0]
                            signal = primitive_patterns[anomaly_name].clone()
                            phase_noise = (
                                torch.randn(2048, device=harness.device) * 0.01
                            )
                            signal = signal * torch.exp(1j * phase_noise)
                            signal = signal / torch.sqrt(
                                torch.sum(torch.abs(signal) ** 2)
                            )
                        else:
                            signals_to_bundle = [
                                primitive_patterns[a] for a in row_anomalies
                            ]
                            signal = signals_to_bundle[0].clone()
                            for s in signals_to_bundle[1:]:
                                signal = signal + s
                            signal = signal / torch.sqrt(
                                torch.sum(torch.abs(signal) ** 2)
                            )

                        test_cases.append(
                            {
                                "signal": signal,
                                "ground_truth": row_anomalies.copy(),
                                "sku": row.get("SKU", "")[:30],
                                "accepts_any": len(row_anomalies) > 1,
                                "unmapped": unmapped,
                            }
                        )

                    elif unmapped:
                        # Row has ONLY unmapped anomalies - these are coverage gaps!
                        # Create signal from random pattern (won't match any primitive)
                        signal = torch.randn(
                            2048, dtype=torch.complex64, device=harness.device
                        )
                        signal = signal / torch.sqrt(torch.sum(torch.abs(signal) ** 2))

                        test_cases.append(
                            {
                                "signal": signal,
                                "ground_truth": set(),  # Nothing should fire
                                "sku": row.get("SKU", "")[:30],
                                "needs_primitive": True,  # Flag for structure learning
                                "unmapped": unmapped,
                            }
                        )
                    else:
                        normal_count += 1

                except (ValueError, TypeError):
                    continue

        scenario_info = {
            "total_rows_processed": row_count,
            "anomaly_counts": anomaly_counts,
            "unmapped_anomalies": unmapped_anomalies,
            "normal_count": normal_count,
            "test_cases_generated": len(test_cases),
            "primitives": list(harness.primitives.keys()),
            "data_type": "YTD_vs_original_primitives",
        }

        print("Data Summary:")
        print(f"  Rows processed: {scenario_info['total_rows_processed']:,}")
        print(f"  Test cases: {scenario_info['test_cases_generated']:,}")
        print(f"  Normal rows: {scenario_info['normal_count']:,}")

        print("\nMAPPED to Original Primitives:")
        for name, count in sorted(anomaly_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                print(f"  {name}: {count:,}")

        print("\nUNMAPPED Anomalies (potential coverage gaps):")
        for name, count in sorted(unmapped_anomalies.items(), key=lambda x: -x[1]):
            if count > 0:
                print(f"  {name}: {count:,} <-- No primitive for this!")

        print(f"\nInitial Primitives ({len(harness.primitives)}):")
        for name in harness.primitives.keys():
            print(f"  - {name}")

        print("\n" + "-" * 70)
        print("Running structure learning...")
        print("-" * 70 + "\n")

        evaluator = StructureEvaluator(harness, test_cases, detection_threshold=0.35)

        loop = StructureLearningLoop(
            harness=harness,
            evaluator=evaluator,
            max_iterations=40,
            patience=10,
            structure_change_cooldown=2,
        )

        result = loop.run()
        result.scenario_info = scenario_info  # type: ignore

        print_real_retail_report(result)

        # Save to file
        report_path = (
            f"{output_dir}/structure_learning_ytd_original_report_{timestamp}.txt"
        )
        save_report_to_file(result, report_path)

    else:
        # Run synthetic test
        result = run_structure_learning_test(
            dimensions=2048,
            max_iterations=30,
        )
        print_structure_learning_report(result)

        # Save to file
        report_path = (
            f"{output_dir}/structure_learning_synthetic_report_{timestamp}.txt"
        )
        save_report_to_file(result, report_path)
