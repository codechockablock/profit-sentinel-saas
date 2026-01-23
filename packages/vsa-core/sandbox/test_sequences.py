"""
VSA Self-Modification Test Sequences

This module implements the test sequences for exploring self-modification dynamics:
    1. Controlled drift - small modifications, measure cumulative effect
    2. Aggressive modification - rapid/large changes, find point of no return
    3. Recovery testing - can we restore from degradation?
    4. Self-evaluation integrity - THE CRITICAL TEST
    5. Adversarial inputs - what breaks it?

The goal is not pass/fail, but exploration:
    - Does coherent self-modification exist?
    - Are there stable attractor states?
    - Are there degenerate configurations?
    - Can it detect its own degradation?

Usage:
    harness = create_harness()
    results = run_controlled_drift_test(harness)
    results = run_self_evaluation_integrity_test(harness)
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from vsa_sandbox_harness import (
    VSASandboxHarness,
    create_harness,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES FOR TEST RESULTS
# =============================================================================


@dataclass
class ControlledDriftResult:
    """Results from controlled drift test."""
    test_name: str = "controlled_drift"
    start_time: float = 0.0
    end_time: float = 0.0

    # Configuration
    n_steps: int = 0
    magnitude_per_step: float = 0.0
    primitives_modified: list[str] = field(default_factory=list)

    # Trajectories
    health_trajectory: list[dict] = field(default_factory=list)  # Simplified health per step
    drift_trajectory: list[dict] = field(default_factory=list)  # Drift from baseline per step
    modification_log: list[dict] = field(default_factory=list)

    # Summary statistics
    final_health: dict = field(default_factory=dict)
    final_drift: dict = field(default_factory=dict)
    degradation_pattern: str = ""  # "stable", "monotonic_decline", "oscillating", "catastrophic"

    # Observations
    observations: list[str] = field(default_factory=list)


@dataclass
class AggressiveModificationResult:
    """Results from aggressive modification test."""
    test_name: str = "aggressive_modification"

    # Point of no return analysis
    degradation_threshold_step: int | None = None  # Step where health drops below threshold
    recovery_possible_after: list[int] = field(default_factory=list)  # Steps where recovery worked
    recovery_failed_after: list[int] = field(default_factory=list)  # Steps where recovery failed

    # Trajectories
    health_trajectory: list[dict] = field(default_factory=list)
    similarity_to_original: list[float] = field(default_factory=list)

    # Summary
    observations: list[str] = field(default_factory=list)


@dataclass
class SelfEvaluationIntegrityResult:
    """Results from the critical self-evaluation integrity test.

    This tests whether the system can detect its own degradation, or whether
    modification corrupts the self-assessment mechanism.
    """
    test_name: str = "self_evaluation_integrity"

    # Baseline detection performance
    baseline_detection_accuracy: float = 0.0
    baseline_detection_confidence: float = 0.0
    baseline_false_negative_rate: float = 0.0

    # Post-modification detection (same known errors)
    post_modification_detection_accuracy: float = 0.0
    post_modification_detection_confidence: float = 0.0
    post_modification_false_negative_rate: float = 0.0

    # THE CRITICAL QUESTION: Does confidence drop when accuracy drops?
    accuracy_drop: float = 0.0  # baseline - post_modification accuracy
    confidence_drop: float = 0.0  # baseline - post_modification confidence

    # If accuracy dropped but confidence didn't, we have a SERIOUS PROBLEM
    confidence_accuracy_gap: float = 0.0  # confidence_drop - accuracy_drop
    # Positive gap means confidence dropped more than accuracy (conservative, good)
    # Negative gap means accuracy dropped more than confidence (overconfident, BAD)

    # Per-primitive breakdown
    per_primitive_results: dict = field(default_factory=dict)

    # Corruption severity analysis
    modification_applied: dict = field(default_factory=dict)
    corruption_type: str = ""  # "benign", "detectable", "silent_failure"

    observations: list[str] = field(default_factory=list)


@dataclass
class RecoveryTestResult:
    """Results from recovery/checkpoint testing."""
    test_name: str = "recovery_test"

    # Restoration accuracy
    restore_exact_match: bool = False
    max_primitive_difference: float = 0.0
    max_codebook_difference: float = 0.0

    # Behavioral equivalence after restore
    health_before_modification: dict = field(default_factory=dict)
    health_after_modification: dict = field(default_factory=dict)
    health_after_restore: dict = field(default_factory=dict)

    # Does restored state behave identically to original?
    behavioral_equivalence: bool = False
    detection_equivalence: bool = False

    observations: list[str] = field(default_factory=list)


# =============================================================================
# CONTROLLED TEST DATA
# =============================================================================


def create_controlled_test_data(harness: VSASandboxHarness) -> tuple[torch.Tensor, dict]:
    """Create minimal controlled test case with known ground truth.

    Returns:
        (bundle, known_detections) where known_detections is {primitive: [entities]}
    """
    # Create small codebook with known entities
    entities = [
        "sku_low_stock_001",
        "sku_low_stock_002",
        "sku_healthy_001",
        "sku_healthy_002",
        "sku_margin_leak_001",
        "sku_dead_item_001",
        "sku_overstock_001",
    ]

    for entity in entities:
        harness.add_to_codebook(entity)

    # Build bundle with known facts
    bundle = harness._normalize(torch.zeros(harness.dimensions, dtype=harness.dtype, device=harness.device))

    # Bind known anomalies to entities
    known_detections = {
        "low_stock": ["sku_low_stock_001", "sku_low_stock_002"],
        "high_margin_leak": ["sku_margin_leak_001"],
        "dead_item": ["sku_dead_item_001"],
        "overstock": ["sku_overstock_001"],
    }

    # Build bundle: for each detection, bind primitive to entity
    for primitive_name, entity_list in known_detections.items():
        primitive = harness.primitives[primitive_name]
        for entity in entity_list:
            entity_vec = harness.codebook[entity]
            fact = harness.bind(primitive, entity_vec)
            bundle = bundle + fact

    bundle = harness._normalize(bundle)

    return bundle, known_detections


# =============================================================================
# TEST SEQUENCES
# =============================================================================


def run_controlled_drift_test(
    harness: VSASandboxHarness | None = None,
    n_steps: int = 20,
    magnitude_per_step: float = 0.05,
    primitives_to_modify: list[str] | None = None,
    modification_type: str = "phase_noise",
) -> ControlledDriftResult:
    """
    Test 1: Controlled Drift

    Apply small modifications iteratively and measure cumulative effect.
    Question: Does it stay stable? Oscillate? Drift monotonically?
    """
    logger.info("=" * 60)
    logger.info("CONTROLLED DRIFT TEST")
    logger.info("=" * 60)

    if harness is None:
        harness = create_harness()

    if primitives_to_modify is None:
        primitives_to_modify = ["low_stock", "high_margin_leak", "dead_item"]

    result = ControlledDriftResult(
        start_time=time.time(),
        n_steps=n_steps,
        magnitude_per_step=magnitude_per_step,
        primitives_modified=primitives_to_modify,
    )

    # Create test data
    test_bundle, known_detections = create_controlled_test_data(harness)

    # Capture baseline
    baseline = harness.capture_baseline("controlled_drift_baseline")

    # Measure baseline health
    baseline_health = harness.measure_health(
        test_bundle=test_bundle,
        known_detections=known_detections,
    )
    result.health_trajectory.append(baseline_health.summary())

    logger.info(f"Baseline health: {baseline_health.summary()}")

    # Apply modifications step by step
    for step in range(n_steps):
        # Modify each primitive
        for prim_name in primitives_to_modify:
            record = harness.apply_primitive_perturbation(
                prim_name,
                magnitude=magnitude_per_step,
                perturbation_type=modification_type,
            )
            result.modification_log.append({
                "step": step,
                "primitive": prim_name,
                "pre_sim": record.pre_similarity_to_original,
                "post_sim": record.post_similarity_to_original,
            })

        # Measure health after this step
        health = harness.measure_health(
            test_bundle=test_bundle,
            known_detections=known_detections,
        )
        result.health_trajectory.append(health.summary())

        # Measure drift
        drift = harness.compute_drift_from_baseline(baseline)
        result.drift_trajectory.append({
            "step": step,
            "mean_drift": drift.mean_primitive_drift,
            "max_drift": drift.max_primitive_drift,
            "frobenius_dist": drift.similarity_matrix_frobenius_distance,
        })

        det_acc_str = f"{health.detection_accuracy:.3f}" if health.detection_accuracy is not None else "N/A"
        logger.info(
            f"Step {step + 1}/{n_steps}: "
            f"binding_acc={health.binding_accuracy:.3f}, "
            f"detection_acc={det_acc_str}, "
            f"mean_drift={drift.mean_primitive_drift:.4f}"
        )

    result.end_time = time.time()

    # Analyze pattern
    [h.get("binding_accuracy", 1.0) for h in result.health_trajectory]
    detection_accs = [h.get("detection_accuracy", 1.0) for h in result.health_trajectory if h.get("detection_accuracy")]

    # Determine degradation pattern
    if len(detection_accs) > 1:
        final_acc = detection_accs[-1] if detection_accs else 1.0
        initial_acc = detection_accs[0] if detection_accs else 1.0
        acc_drop = initial_acc - final_acc

        # Check for monotonic decline
        is_monotonic = all(detection_accs[i] >= detection_accs[i+1] - 0.05 for i in range(len(detection_accs)-1))

        # Check for oscillation
        direction_changes = sum(
            1 for i in range(1, len(detection_accs) - 1)
            if (detection_accs[i] - detection_accs[i-1]) * (detection_accs[i+1] - detection_accs[i]) < 0
        )

        if acc_drop > 0.5:
            result.degradation_pattern = "catastrophic"
        elif is_monotonic and acc_drop > 0.1:
            result.degradation_pattern = "monotonic_decline"
        elif direction_changes > n_steps // 3:
            result.degradation_pattern = "oscillating"
        elif acc_drop < 0.05:
            result.degradation_pattern = "stable"
        else:
            result.degradation_pattern = "gradual_decline"

    result.final_health = result.health_trajectory[-1] if result.health_trajectory else {}
    result.final_drift = result.drift_trajectory[-1] if result.drift_trajectory else {}

    # Observations
    if result.degradation_pattern == "stable":
        result.observations.append("System remained stable under controlled drift")
    elif result.degradation_pattern == "catastrophic":
        result.observations.append("CRITICAL: Catastrophic degradation observed")
    elif result.degradation_pattern == "monotonic_decline":
        result.observations.append("Monotonic decline suggests cumulative damage without self-repair")

    logger.info(f"Test complete. Pattern: {result.degradation_pattern}")
    return result


def run_self_evaluation_integrity_test(
    harness: VSASandboxHarness | None = None,
    modification_magnitude: float = 0.3,
    modification_type: str = "directional",
) -> SelfEvaluationIntegrityResult:
    """
    Test 4: Self-Evaluation Integrity (THE CRITICAL TEST)

    The nightmare scenario: corrupt the primitive vectors used for detection,
    and the system stops detecting the corruption as a problem because the
    evaluator uses the corrupted primitives.

    Test sequence:
        1. Inject known errors, verify detection (baseline)
        2. Modify geometry (corrupt detection primitives)
        3. Re-test detection of the same known errors
        4. KEY QUESTION: If detection accuracy drops, does self-reported
           confidence also drop? Or does it stay confident while getting worse?
    """
    logger.info("=" * 60)
    logger.info("SELF-EVALUATION INTEGRITY TEST (CRITICAL)")
    logger.info("=" * 60)

    if harness is None:
        harness = create_harness()

    result = SelfEvaluationIntegrityResult()

    # Create test data with known ground truth
    test_bundle, known_detections = create_controlled_test_data(harness)

    # Capture baseline
    harness.capture_baseline("self_eval_baseline")

    # PHASE 1: Baseline detection
    logger.info("Phase 1: Measuring baseline detection capability...")
    baseline_health = harness.measure_health(
        test_bundle=test_bundle,
        known_detections=known_detections,
    )

    result.baseline_detection_accuracy = baseline_health.detection_accuracy or 0.0
    result.baseline_detection_confidence = baseline_health.detection_confidence or 0.0
    result.baseline_false_negative_rate = baseline_health.false_negative_rate or 0.0

    logger.info(
        f"Baseline: accuracy={result.baseline_detection_accuracy:.3f}, "
        f"confidence={result.baseline_detection_confidence:.3f}, "
        f"FNR={result.baseline_false_negative_rate:.3f}"
    )

    # PHASE 2: Corrupt the detection primitives
    logger.info(f"Phase 2: Applying {modification_type} corruption (magnitude={modification_magnitude})...")

    # Corrupt the primitives that are used for detection
    detection_primitives = list(known_detections.keys())
    for prim_name in detection_primitives:
        harness.apply_primitive_perturbation(
            prim_name,
            magnitude=modification_magnitude,
            perturbation_type=modification_type,
        )

    result.modification_applied = {
        "primitives_corrupted": detection_primitives,
        "magnitude": modification_magnitude,
        "type": modification_type,
    }

    # PHASE 3: Re-test detection with SAME known errors, SAME bundle
    logger.info("Phase 3: Re-testing detection with corrupted geometry...")

    # IMPORTANT: We use the SAME test_bundle - the facts didn't change,
    # only the geometry used to interpret them changed.
    post_health = harness.measure_health(
        test_bundle=test_bundle,
        known_detections=known_detections,
    )

    result.post_modification_detection_accuracy = post_health.detection_accuracy or 0.0
    result.post_modification_detection_confidence = post_health.detection_confidence or 0.0
    result.post_modification_false_negative_rate = post_health.false_negative_rate or 0.0

    logger.info(
        f"Post-modification: accuracy={result.post_modification_detection_accuracy:.3f}, "
        f"confidence={result.post_modification_detection_confidence:.3f}, "
        f"FNR={result.post_modification_false_negative_rate:.3f}"
    )

    # PHASE 4: THE CRITICAL ANALYSIS
    result.accuracy_drop = result.baseline_detection_accuracy - result.post_modification_detection_accuracy
    result.confidence_drop = result.baseline_detection_confidence - result.post_modification_detection_confidence

    # The gap tells us if the system knows it's getting worse
    # Positive gap = confidence dropped more than accuracy (conservative, good)
    # Negative gap = accuracy dropped more than confidence (overconfident, DANGEROUS)
    result.confidence_accuracy_gap = result.confidence_drop - result.accuracy_drop

    logger.info("=" * 40)
    logger.info("CRITICAL ANALYSIS:")
    logger.info(f"  Accuracy drop:    {result.accuracy_drop:+.3f}")
    logger.info(f"  Confidence drop:  {result.confidence_drop:+.3f}")
    logger.info(f"  Gap (conf - acc): {result.confidence_accuracy_gap:+.3f}")

    # Classify the corruption type
    if result.accuracy_drop < 0.1:
        result.corruption_type = "benign"
        result.observations.append("Modification had minimal impact on detection accuracy")
    elif result.confidence_accuracy_gap > -0.1:
        result.corruption_type = "detectable"
        result.observations.append("System's confidence appropriately reflects degraded accuracy")
    else:
        result.corruption_type = "silent_failure"
        result.observations.append(
            "CRITICAL: Silent failure detected! "
            f"Accuracy dropped by {result.accuracy_drop:.2f} but confidence only dropped by {result.confidence_drop:.2f}. "
            "System is overconfident in degraded state."
        )

    # Per-primitive analysis
    logger.info("\nPer-primitive analysis:")
    for prim_name in detection_primitives:
        original = harness._original_primitives[prim_name]
        current = harness.primitives[prim_name]
        sim = harness.similarity(current, original)
        logger.info(f"  {prim_name}: similarity to original = {sim:.3f}")
        result.per_primitive_results[prim_name] = {
            "similarity_to_original": sim,
        }

    logger.info("=" * 40)

    if result.corruption_type == "silent_failure":
        logger.warning(
            "\n*** SILENT FAILURE DETECTED ***\n"
            "The system's self-assessment is corrupted.\n"
            "It believes it is functioning correctly while actually degraded.\n"
            "This is the failure mode we need to protect against.\n"
        )

    return result


def run_recovery_test(
    harness: VSASandboxHarness | None = None,
    modification_magnitude: float = 0.5,
) -> RecoveryTestResult:
    """
    Test 3: Recovery Testing

    After degradation, can we restore from checkpoint?
    Does restoration actually work, or has something subtle shifted?
    """
    logger.info("=" * 60)
    logger.info("RECOVERY TEST")
    logger.info("=" * 60)

    if harness is None:
        harness = create_harness()

    result = RecoveryTestResult()

    # Create test data
    test_bundle, known_detections = create_controlled_test_data(harness)

    # PHASE 1: Measure health before modification
    logger.info("Phase 1: Measuring initial health...")
    initial_health = harness.measure_health(
        test_bundle=test_bundle,
        known_detections=known_detections,
    )
    result.health_before_modification = initial_health.summary()

    # Capture baseline for restoration
    baseline = harness.capture_baseline("recovery_test_baseline")

    # PHASE 2: Apply significant modification
    logger.info(f"Phase 2: Applying modifications (magnitude={modification_magnitude})...")
    for prim_name in harness.primitives.keys():
        harness.apply_primitive_perturbation(
            prim_name,
            magnitude=modification_magnitude,
            perturbation_type="directional",
        )

    # Measure degraded health
    degraded_health = harness.measure_health(
        test_bundle=test_bundle,
        known_detections=known_detections,
    )
    result.health_after_modification = degraded_health.summary()

    logger.info(f"Health after modification: {degraded_health.summary()}")

    # PHASE 3: Restore from checkpoint
    logger.info("Phase 3: Restoring from checkpoint...")
    harness.restore_from_snapshot(baseline)

    # Verify exact restoration
    verify = harness.verify_restore(baseline)
    result.restore_exact_match = verify["primitives_match"] and verify["codebook_match"]
    result.max_primitive_difference = verify["max_primitive_diff"]
    result.max_codebook_difference = verify["max_codebook_diff"]

    logger.info(f"Restoration verification: {verify}")

    # PHASE 4: Measure health after restore
    restored_health = harness.measure_health(
        test_bundle=test_bundle,
        known_detections=known_detections,
    )
    result.health_after_restore = restored_health.summary()

    # Check behavioral equivalence
    def health_equivalent(h1: dict, h2: dict, tolerance: float = 0.05) -> bool:
        for key in h1:
            if h1[key] is not None and h2.get(key) is not None:
                if abs(h1[key] - h2[key]) > tolerance:
                    return False
        return True

    result.behavioral_equivalence = health_equivalent(
        result.health_before_modification,
        result.health_after_restore,
    )

    # Check detection equivalence specifically
    initial_det = result.health_before_modification.get("detection_accuracy")
    restored_det = result.health_after_restore.get("detection_accuracy")
    if initial_det is not None and restored_det is not None:
        result.detection_equivalence = abs(initial_det - restored_det) < 0.05
    else:
        result.detection_equivalence = True

    # Observations
    if result.restore_exact_match:
        result.observations.append("Checkpoint restoration was exact (bit-identical)")
    else:
        result.observations.append(
            f"Restoration had small differences: "
            f"max_prim_diff={result.max_primitive_difference:.2e}"
        )

    if result.behavioral_equivalence:
        result.observations.append("Restored system behaves equivalently to original")
    else:
        result.observations.append(
            "WARNING: Restored system does NOT behave equivalently to original! "
            "Some subtle state may not be captured in checkpoint."
        )

    logger.info(f"Recovery test complete. Behavioral equivalence: {result.behavioral_equivalence}")
    return result


def run_aggressive_modification_test(
    harness: VSASandboxHarness | None = None,
    max_steps: int = 50,
    magnitude_per_step: float = 0.1,
    health_threshold: float = 0.5,
) -> AggressiveModificationResult:
    """
    Test 2: Aggressive Modification

    What happens under rapid/large changes? Is there a point of no return?
    """
    logger.info("=" * 60)
    logger.info("AGGRESSIVE MODIFICATION TEST")
    logger.info("=" * 60)

    if harness is None:
        harness = create_harness()

    result = AggressiveModificationResult()

    # Create test data
    test_bundle, known_detections = create_controlled_test_data(harness)

    # Capture baseline
    baseline = harness.capture_baseline("aggressive_baseline")

    for step in range(max_steps):
        # Apply aggressive modification to ALL primitives
        for prim_name in harness.primitives.keys():
            harness.apply_primitive_perturbation(
                prim_name,
                magnitude=magnitude_per_step,
                perturbation_type="directional",  # Coherent drift, not noise
            )

        # Measure health
        health = harness.measure_health(
            test_bundle=test_bundle,
            known_detections=known_detections,
        )
        result.health_trajectory.append(health.summary())

        # Track similarity to original
        mean_sim = sum(
            harness.similarity(harness.primitives[k], harness._original_primitives[k])
            for k in harness.primitives
        ) / len(harness.primitives)
        result.similarity_to_original.append(mean_sim)

        # Check if we've crossed the threshold
        detection_acc = health.detection_accuracy or health.binding_accuracy
        if detection_acc < health_threshold and result.degradation_threshold_step is None:
            result.degradation_threshold_step = step
            logger.warning(f"Step {step}: Crossed degradation threshold (acc={detection_acc:.3f})")

        # Try recovery at certain points
        if step in [10, 20, 30, 40]:
            logger.info(f"Step {step}: Testing recovery...")
            # Save current state
            current_primitives = {k: v.clone() for k, v in harness.primitives.items()}

            # Restore
            harness.restore_from_snapshot(baseline)

            # Check if it worked
            restored_health = harness.measure_health(
                test_bundle=test_bundle,
                known_detections=known_detections,
            )
            restored_acc = restored_health.detection_accuracy or restored_health.binding_accuracy
            baseline_acc = result.health_trajectory[0].get("detection_accuracy", 1.0)

            if abs(restored_acc - baseline_acc) < 0.1:
                result.recovery_possible_after.append(step)
            else:
                result.recovery_failed_after.append(step)

            # Put back the degraded state to continue test
            harness.primitives = current_primitives

        det_acc_str = f"{health.detection_accuracy:.3f}" if health.detection_accuracy is not None else "N/A"
        logger.info(
            f"Step {step + 1}/{max_steps}: "
            f"mean_sim_to_orig={mean_sim:.3f}, "
            f"detection_acc={det_acc_str}"
        )

    # Observations
    if result.degradation_threshold_step is not None:
        result.observations.append(
            f"Degradation threshold crossed at step {result.degradation_threshold_step}"
        )
    else:
        result.observations.append(
            f"System remained above health threshold through all {max_steps} steps"
        )

    if result.recovery_failed_after:
        result.observations.append(
            f"Recovery failed after steps: {result.recovery_failed_after}"
        )

    return result


def run_phase_drift_sweep(
    harness: VSASandboxHarness | None = None,
    drift_angles: list[float] | None = None,
    drift_types: list[str] | None = None,
) -> dict[str, Any]:
    """
    Sweep over different phase drift parameters to find boundaries.

    Tests different drift types and magnitudes to understand:
    - Which drift types are most damaging?
    - What magnitude causes detectable degradation?
    - Are there drift patterns the system is robust to?
    """
    logger.info("=" * 60)
    logger.info("PHASE DRIFT SWEEP")
    logger.info("=" * 60)

    if harness is None:
        harness = create_harness()

    if drift_angles is None:
        drift_angles = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    if drift_types is None:
        drift_types = ["uniform", "gradient", "sinusoidal"]

    results = {
        "sweep_config": {
            "drift_angles": drift_angles,
            "drift_types": drift_types,
        },
        "results": [],
    }

    for drift_type in drift_types:
        for angle in drift_angles:
            # Reset to fresh state
            harness = create_harness()

            # Create test data
            test_bundle, known_detections = create_controlled_test_data(harness)

            # Measure baseline
            baseline_health = harness.measure_health(
                test_bundle=test_bundle,
                known_detections=known_detections,
            )

            # Apply drift
            harness.apply_phase_drift(
                drift_angle=angle,
                drift_type=drift_type,
            )

            # Measure after drift
            post_health = harness.measure_health(
                test_bundle=test_bundle,
                known_detections=known_detections,
            )

            entry = {
                "drift_type": drift_type,
                "drift_angle": angle,
                "baseline_binding_acc": baseline_health.binding_accuracy,
                "post_binding_acc": post_health.binding_accuracy,
                "baseline_detection_acc": baseline_health.detection_accuracy,
                "post_detection_acc": post_health.detection_accuracy,
                "binding_drop": baseline_health.binding_accuracy - post_health.binding_accuracy,
                "detection_drop": (baseline_health.detection_accuracy or 0) - (post_health.detection_accuracy or 0),
            }
            results["results"].append(entry)

            logger.info(
                f"{drift_type:12s} angle={angle:.2f}: "
                f"binding_drop={entry['binding_drop']:.3f}, "
                f"detection_drop={entry['detection_drop']:.3f}"
            )

    return results


# =============================================================================
# INVARIANT HUNTING
# =============================================================================


def hunt_for_invariants(
    harness: VSASandboxHarness | None = None,
    n_trials: int = 50,
) -> dict[str, Any]:
    """
    Experiment to discover emergent invariants.

    We don't know what properties MUST hold for the system to function.
    This function applies various modifications and observes what breaks.

    Goal: Find properties that, when violated, cause system failure.
    """
    logger.info("=" * 60)
    logger.info("INVARIANT HUNTING")
    logger.info("=" * 60)

    if harness is None:
        harness = create_harness()

    # Candidate invariants to test
    invariants = {
        "primitive_orthogonality": [],  # Track mean pairwise |sim|
        "primitive_normalization": [],  # Track deviation from unit magnitude
        "binding_invertibility": [],  # Track bind/unbind accuracy
        "similarity_matrix_rank": [],  # Track rank of similarity matrix
        "phase_coherence": [],  # Track phase distribution statistics
    }

    correlations = {
        "orthogonality_vs_detection": [],
        "normalization_vs_binding": [],
    }

    for trial in range(n_trials):
        # Reset
        harness = create_harness()
        test_bundle, known_detections = create_controlled_test_data(harness)

        # Apply random modification
        import random
        mod_type = random.choice(["phase_noise", "directional"])
        magnitude = random.uniform(0.01, 0.5)
        prim = random.choice(list(harness.primitives.keys()))

        harness.apply_primitive_perturbation(prim, magnitude, mod_type)

        # Measure all candidate invariants
        health = harness.measure_health(test_bundle=test_bundle, known_detections=known_detections)

        invariants["primitive_orthogonality"].append(health.primitive_mean_orthogonality)
        invariants["binding_invertibility"].append(health.binding_accuracy)

        # Check normalization
        norm_devs = []
        for vec in harness.primitives.values():
            mags = torch.abs(vec)
            norm_devs.append(float(torch.mean(torch.abs(mags - 1.0))))
        invariants["primitive_normalization"].append(sum(norm_devs) / len(norm_devs))

        # Track correlations
        correlations["orthogonality_vs_detection"].append(
            (health.primitive_mean_orthogonality, health.detection_accuracy or 0)
        )
        correlations["normalization_vs_binding"].append(
            (invariants["primitive_normalization"][-1], health.binding_accuracy)
        )

    # Analyze correlations
    def compute_correlation(pairs: list[tuple[float, float]]) -> float:
        if len(pairs) < 2:
            return 0
        x = [p[0] for p in pairs]
        y = [p[1] for p in pairs]
        x_mean = sum(x) / len(x)
        y_mean = sum(y) / len(y)

        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denom_x = math.sqrt(sum((xi - x_mean) ** 2 for xi in x))
        denom_y = math.sqrt(sum((yi - y_mean) ** 2 for yi in y))

        if denom_x * denom_y == 0:
            return 0
        return numerator / (denom_x * denom_y)

    results = {
        "invariant_statistics": {
            k: {
                "mean": sum(v) / len(v) if v else 0,
                "min": min(v) if v else 0,
                "max": max(v) if v else 0,
            }
            for k, v in invariants.items()
        },
        "correlations": {
            "orthogonality_vs_detection": compute_correlation(correlations["orthogonality_vs_detection"]),
            "normalization_vs_binding": compute_correlation(correlations["normalization_vs_binding"]),
        },
        "candidate_invariants": [],
    }

    # Identify strong invariant candidates (high correlation with functionality)
    if abs(results["correlations"]["orthogonality_vs_detection"]) > 0.5:
        results["candidate_invariants"].append({
            "name": "primitive_orthogonality",
            "correlation_with_detection": results["correlations"]["orthogonality_vs_detection"],
            "hypothesis": "Primitives must remain approximately orthogonal for reliable detection",
        })

    logger.info(f"Invariant hunting complete. Candidates found: {len(results['candidate_invariants'])}")
    for inv in results["candidate_invariants"]:
        logger.info(f"  - {inv['name']}: {inv['hypothesis']}")

    return results


# =============================================================================
# MAIN RUNNER
# =============================================================================


def run_all_tests(
    output_dir: str | Path = "sandbox_results",
    dimensions: int = 2048,
    device: str | None = None,
) -> dict[str, Any]:
    """Run all test sequences and save results."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    harness = create_harness(dimensions=dimensions, device=device)

    all_results = {
        "timestamp": time.time(),
        "config": {
            "dimensions": dimensions,
            "device": str(harness.device),
        },
        "tests": {},
    }

    # Test 1: Controlled drift
    logger.info("\n" + "=" * 80)
    logger.info("Running controlled drift test...")
    drift_result = run_controlled_drift_test(harness=create_harness(dimensions, device))
    all_results["tests"]["controlled_drift"] = asdict(drift_result)

    # Test 2: Aggressive modification
    logger.info("\n" + "=" * 80)
    logger.info("Running aggressive modification test...")
    aggressive_result = run_aggressive_modification_test(harness=create_harness(dimensions, device))
    all_results["tests"]["aggressive_modification"] = asdict(aggressive_result)

    # Test 3: Recovery
    logger.info("\n" + "=" * 80)
    logger.info("Running recovery test...")
    recovery_result = run_recovery_test(harness=create_harness(dimensions, device))
    all_results["tests"]["recovery"] = asdict(recovery_result)

    # Test 4: Self-evaluation integrity (THE CRITICAL ONE)
    logger.info("\n" + "=" * 80)
    logger.info("Running self-evaluation integrity test (CRITICAL)...")
    integrity_result = run_self_evaluation_integrity_test(harness=create_harness(dimensions, device))
    all_results["tests"]["self_evaluation_integrity"] = asdict(integrity_result)

    # Test 5: Phase drift sweep
    logger.info("\n" + "=" * 80)
    logger.info("Running phase drift sweep...")
    sweep_result = run_phase_drift_sweep(harness=create_harness(dimensions, device))
    all_results["tests"]["phase_drift_sweep"] = sweep_result

    # Test 6: Invariant hunting
    logger.info("\n" + "=" * 80)
    logger.info("Running invariant hunting...")
    invariant_result = hunt_for_invariants(harness=create_harness(dimensions, device))
    all_results["tests"]["invariant_hunting"] = invariant_result

    # Save results
    results_file = output_dir / f"sandbox_results_{int(time.time())}.json"

    # Convert non-serializable types
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    with open(results_file, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)

    logger.info(f"\nResults saved to {results_file}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    logger.info(f"Controlled drift pattern: {drift_result.degradation_pattern}")
    logger.info(f"Aggressive test - degradation threshold: step {aggressive_result.degradation_threshold_step}")
    logger.info(f"Recovery test - behavioral equivalence: {recovery_result.behavioral_equivalence}")
    logger.info(f"Self-evaluation integrity - corruption type: {integrity_result.corruption_type}")
    logger.info(f"Invariant candidates found: {len(invariant_result['candidate_invariants'])}")

    if integrity_result.corruption_type == "silent_failure":
        logger.warning("\n*** CRITICAL: SILENT FAILURE DETECTED ***")
        logger.warning("The system cannot reliably detect its own degradation.")

    return all_results


if __name__ == "__main__":
    run_all_tests()
