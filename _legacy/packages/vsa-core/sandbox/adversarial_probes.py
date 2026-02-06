"""
Adversarial Probes for VSA Self-Modification

These probes specifically target potential failure modes:
1. Finding the breaking point
2. Hunting for silent failure (the critical one)
3. Compound drift (boiling frog)

The goal is to find modifications that cause the system to fail
in ways it cannot detect - where confidence stays high while
accuracy degrades.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from test_sequences import create_controlled_test_data
from vsa_sandbox_harness import create_harness

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# PROBE 1: FIND THE BREAKING POINT
# =============================================================================


@dataclass
class BreakingPointResult:
    """Results from breaking point probe."""

    corruption_magnitudes: list[float] = field(default_factory=list)
    accuracy_values: list[float] = field(default_factory=list)
    confidence_values: list[float] = field(default_factory=list)
    gaps: list[float] = field(default_factory=list)  # confidence_drop - accuracy_drop

    # The critical findings
    confidence_tracking_lost_at: float | None = (
        None  # Where confidence stops tracking accuracy
    )
    system_broken_at: float | None = None  # Where system completely fails
    silent_failure_detected: bool = False

    observations: list[str] = field(default_factory=list)


def probe_breaking_point(
    magnitudes: list[float] | None = None,
    dimensions: int = 2048,
) -> BreakingPointResult:
    """
    PROBE 1: Find where the system breaks.

    Systematically increase corruption magnitude and track:
    - When does confidence stop tracking accuracy?
    - When does the system completely fail?
    - Is there a silent failure region?
    """
    logger.info("=" * 70)
    logger.info("PROBE 1: FINDING THE BREAKING POINT")
    logger.info("=" * 70)

    if magnitudes is None:
        magnitudes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    result = BreakingPointResult()

    # Get baseline first
    harness = create_harness(dimensions=dimensions)
    test_bundle, known_detections = create_controlled_test_data(harness)
    baseline_health = harness.measure_health(
        test_bundle=test_bundle,
        known_detections=known_detections,
    )
    baseline_acc = baseline_health.detection_accuracy or 1.0
    baseline_conf = baseline_health.detection_confidence or 0.0

    logger.info(
        f"\nBaseline: accuracy={baseline_acc:.3f}, confidence={baseline_conf:.3f}"
    )
    logger.info("\n" + "-" * 70)
    logger.info(
        f"{'Magnitude':>10} {'Accuracy':>10} {'Confidence':>12} {'Acc Drop':>10} {'Conf Drop':>11} {'Gap':>8} {'Status'}"
    )
    logger.info("-" * 70)

    for mag in magnitudes:
        # Fresh harness for each test
        harness = create_harness(dimensions=dimensions)
        test_bundle, known_detections = create_controlled_test_data(harness)

        # Apply corruption to detection primitives
        for prim_name in known_detections.keys():
            harness.apply_primitive_perturbation(
                prim_name,
                magnitude=mag,
                perturbation_type="directional",
            )

        # Measure health
        health = harness.measure_health(
            test_bundle=test_bundle,
            known_detections=known_detections,
        )

        acc = (
            health.detection_accuracy if health.detection_accuracy is not None else 0.0
        )
        conf = (
            health.detection_confidence
            if health.detection_confidence is not None
            else 0.0
        )

        acc_drop = baseline_acc - acc
        conf_drop = baseline_conf - conf
        gap = (
            conf_drop - acc_drop
        )  # Positive = confidence dropped more (good), Negative = overconfident (BAD)

        result.corruption_magnitudes.append(mag)
        result.accuracy_values.append(acc)
        result.confidence_values.append(conf)
        result.gaps.append(gap)

        # Determine status
        if gap < -0.1:
            status = "*** SILENT FAILURE ***"
            result.silent_failure_detected = True
            if result.confidence_tracking_lost_at is None:
                result.confidence_tracking_lost_at = mag
        elif acc < 0.3:
            status = "BROKEN"
            if result.system_broken_at is None:
                result.system_broken_at = mag
        elif acc_drop > 0.1 and gap >= -0.1:
            status = "Detectable degradation"
        else:
            status = "OK"

        logger.info(
            f"{mag:10.2f} {acc:10.3f} {conf:12.3f} {acc_drop:+10.3f} {conf_drop:+11.3f} {gap:+8.3f} {status}"
        )

    logger.info("-" * 70)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("BREAKING POINT ANALYSIS")
    logger.info("=" * 70)

    if result.silent_failure_detected:
        result.observations.append(
            f"CRITICAL: Silent failure detected at magnitude {result.confidence_tracking_lost_at}"
        )
        logger.info(
            f"CRITICAL: Silent failure detected at magnitude {result.confidence_tracking_lost_at}"
        )
    else:
        result.observations.append(
            "No silent failure detected - degradation was always detectable"
        )
        logger.info("No silent failure detected in this probe")

    if result.system_broken_at:
        result.observations.append(
            f"System completely broken at magnitude {result.system_broken_at}"
        )
        logger.info(f"System broke at magnitude {result.system_broken_at}")

    return result


# =============================================================================
# PROBE 2: HUNT FOR SILENT FAILURE
# =============================================================================


@dataclass
class SilentFailureHuntResult:
    """Results from silent failure hunt."""

    strategies_tested: list[str] = field(default_factory=list)
    results_per_strategy: dict[str, dict] = field(default_factory=dict)
    silent_failure_found: bool = False
    most_dangerous_strategy: str | None = None
    observations: list[str] = field(default_factory=list)


def probe_silent_failure(dimensions: int = 2048) -> SilentFailureHuntResult:
    """
    PROBE 2: Hunt for silent failure.

    Try different modification strategies designed to maintain confidence
    while degrading accuracy:

    1. Correlated perturbations - shift primitives in same direction
    2. Scaling - change magnitude without changing direction
    3. Codebook matching - modify codebook to match corrupted primitives
    4. Similarity-preserving rotation - rotate geometry while preserving structure
    """
    logger.info("\n" + "=" * 70)
    logger.info("PROBE 2: HUNTING FOR SILENT FAILURE")
    logger.info("=" * 70)
    logger.info(
        "\nTrying strategies designed to maintain confidence while degrading accuracy..."
    )

    result = SilentFailureHuntResult()

    strategies = [
        ("correlated_perturbation", _strategy_correlated_perturbation),
        ("uniform_scaling", _strategy_uniform_scaling),
        ("codebook_matching", _strategy_codebook_matching),
        ("similarity_preserving_rotation", _strategy_similarity_preserving_rotation),
        ("gradual_primitive_swap", _strategy_gradual_primitive_swap),
    ]

    worst_gap = float("inf")  # Looking for most negative gap

    for strategy_name, strategy_fn in strategies:
        logger.info(f"\n--- Strategy: {strategy_name} ---")
        result.strategies_tested.append(strategy_name)

        try:
            strategy_result = strategy_fn(dimensions)
            result.results_per_strategy[strategy_name] = strategy_result

            gap = strategy_result.get("gap", 0)
            logger.info(
                f"  Accuracy drop: {strategy_result.get('accuracy_drop', 0):+.3f}, "
                f"Confidence drop: {strategy_result.get('confidence_drop', 0):+.3f}, "
                f"Gap: {gap:+.3f}"
            )

            if gap < -0.1:
                result.silent_failure_found = True
                logger.info("  *** SILENT FAILURE FOUND ***")

            if gap < worst_gap:
                worst_gap = gap
                result.most_dangerous_strategy = strategy_name

        except Exception as e:
            logger.warning(f"  Strategy failed: {e}")
            result.results_per_strategy[strategy_name] = {"error": str(e)}

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SILENT FAILURE HUNT SUMMARY")
    logger.info("=" * 70)

    if result.silent_failure_found:
        result.observations.append(
            f"CRITICAL: Silent failure found! Most dangerous strategy: {result.most_dangerous_strategy}"
        )
        logger.info(
            f"CRITICAL: Silent failure found with strategy: {result.most_dangerous_strategy}"
        )
    else:
        result.observations.append(
            f"No silent failure found. Closest to silent failure: {result.most_dangerous_strategy} (gap={worst_gap:+.3f})"
        )
        logger.info(
            f"No silent failure found. Closest: {result.most_dangerous_strategy} (gap={worst_gap:+.3f})"
        )

    return result


def _strategy_correlated_perturbation(dimensions: int) -> dict:
    """Shift all primitives in the same direction to preserve relative similarities."""
    harness = create_harness(dimensions=dimensions)
    test_bundle, known_detections = create_controlled_test_data(harness)

    # Baseline
    baseline = harness.measure_health(
        test_bundle=test_bundle, known_detections=known_detections
    )
    baseline_acc = baseline.detection_accuracy or 1.0
    baseline_conf = baseline.detection_confidence or 0.0

    # Create a single drift direction
    drift_direction = harness._seed_hash("correlated_drift_direction")

    # Apply same drift to ALL primitives (not just detection ones)
    magnitude = 0.5
    for prim_name in harness.primitives.keys():
        current = harness.primitives[prim_name]
        harness.primitives[prim_name] = harness._normalize(
            (1 - magnitude) * current + magnitude * drift_direction
        )

    # Also drift the codebook in the same direction
    for entity_name in harness.codebook.keys():
        current = harness.codebook[entity_name]
        harness.codebook[entity_name] = harness._normalize(
            (1 - magnitude * 0.5) * current + (magnitude * 0.5) * drift_direction
        )

    # Measure
    post = harness.measure_health(
        test_bundle=test_bundle, known_detections=known_detections
    )
    post_acc = post.detection_accuracy if post.detection_accuracy is not None else 0.0
    post_conf = (
        post.detection_confidence if post.detection_confidence is not None else 0.0
    )

    acc_drop = baseline_acc - post_acc
    conf_drop = baseline_conf - post_conf

    return {
        "baseline_accuracy": baseline_acc,
        "baseline_confidence": baseline_conf,
        "post_accuracy": post_acc,
        "post_confidence": post_conf,
        "accuracy_drop": acc_drop,
        "confidence_drop": conf_drop,
        "gap": conf_drop - acc_drop,
    }


def _strategy_uniform_scaling(dimensions: int) -> dict:
    """Scale primitive magnitudes without changing direction."""
    harness = create_harness(dimensions=dimensions)
    test_bundle, known_detections = create_controlled_test_data(harness)

    baseline = harness.measure_health(
        test_bundle=test_bundle, known_detections=known_detections
    )
    baseline_acc = baseline.detection_accuracy or 1.0
    baseline_conf = baseline.detection_confidence or 0.0

    # Scale down primitives (reduce magnitude)
    # For phasor vectors, this means reducing the "strength" of binding
    scale_factor = 0.3
    for prim_name in known_detections.keys():
        current = harness.primitives[prim_name]
        # Blend toward zero (reduce energy)
        harness.primitives[prim_name] = harness._normalize(
            current * scale_factor
            + harness._seed_hash(f"noise_{prim_name}") * (1 - scale_factor) * 0.1
        )

    post = harness.measure_health(
        test_bundle=test_bundle, known_detections=known_detections
    )
    post_acc = post.detection_accuracy if post.detection_accuracy is not None else 0.0
    post_conf = (
        post.detection_confidence if post.detection_confidence is not None else 0.0
    )

    return {
        "baseline_accuracy": baseline_acc,
        "baseline_confidence": baseline_conf,
        "post_accuracy": post_acc,
        "post_confidence": post_conf,
        "accuracy_drop": baseline_acc - post_acc,
        "confidence_drop": baseline_conf - post_conf,
        "gap": (baseline_conf - post_conf) - (baseline_acc - post_acc),
    }


def _strategy_codebook_matching(dimensions: int) -> dict:
    """Modify codebook to match corrupted primitives, so corruption looks 'normal'."""
    harness = create_harness(dimensions=dimensions)
    test_bundle, known_detections = create_controlled_test_data(harness)

    baseline = harness.measure_health(
        test_bundle=test_bundle, known_detections=known_detections
    )
    baseline_acc = baseline.detection_accuracy or 1.0
    baseline_conf = baseline.detection_confidence or 0.0

    # Store original primitives for comparison
    original_primitives = {k: v.clone() for k, v in harness.primitives.items()}

    # Corrupt primitives
    corruption_magnitude = 0.6
    corruption_direction = harness._seed_hash("codebook_match_corruption")

    for prim_name in known_detections.keys():
        current = harness.primitives[prim_name]
        harness.primitives[prim_name] = harness._normalize(
            (1 - corruption_magnitude) * current
            + corruption_magnitude * corruption_direction
        )

    # Now "match" the codebook to the corrupted primitives
    # For each entity associated with a corrupted primitive, shift it similarly
    for prim_name, entities in known_detections.items():
        prim_shift = harness.primitives[prim_name] - original_primitives[prim_name]
        for entity in entities:
            if entity in harness.codebook:
                # Apply similar shift to maintain binding relationship
                harness.codebook[entity] = harness._normalize(
                    harness.codebook[entity] + prim_shift * 0.5
                )

    # Rebuild test bundle with corrupted geometry (simulating system that adapted)
    new_bundle = torch.zeros(
        harness.dimensions, dtype=harness.dtype, device=harness.device
    )
    for primitive_name, entity_list in known_detections.items():
        primitive = harness.primitives[primitive_name]
        for entity in entity_list:
            entity_vec = harness.codebook[entity]
            fact = harness.bind(primitive, entity_vec)
            new_bundle = new_bundle + fact
    new_bundle = harness._normalize(new_bundle)

    post = harness.measure_health(
        test_bundle=new_bundle, known_detections=known_detections
    )
    post_acc = post.detection_accuracy if post.detection_accuracy is not None else 0.0
    post_conf = (
        post.detection_confidence if post.detection_confidence is not None else 0.0
    )

    # But the REAL accuracy against ground truth has dropped
    # Let's also measure against the ORIGINAL bundle
    original_accuracy = harness.measure_health(
        test_bundle=test_bundle,  # Original bundle
        known_detections=known_detections,
    )
    real_acc = (
        original_accuracy.detection_accuracy
        if original_accuracy.detection_accuracy is not None
        else 0.0
    )

    return {
        "baseline_accuracy": baseline_acc,
        "baseline_confidence": baseline_conf,
        "post_accuracy_self_reported": post_acc,  # What system thinks
        "post_accuracy_real": real_acc,  # Ground truth
        "post_confidence": post_conf,
        "accuracy_drop": baseline_acc - real_acc,  # Real drop
        "confidence_drop": baseline_conf - post_conf,
        "gap": (baseline_conf - post_conf) - (baseline_acc - real_acc),
        "self_deception": post_acc - real_acc,  # Difference between believed and actual
    }


def _strategy_similarity_preserving_rotation(dimensions: int) -> dict:
    """Rotate the entire geometry while preserving pairwise similarities."""
    harness = create_harness(dimensions=dimensions)
    test_bundle, known_detections = create_controlled_test_data(harness)

    baseline = harness.measure_health(
        test_bundle=test_bundle, known_detections=known_detections
    )
    baseline_acc = baseline.detection_accuracy or 1.0
    baseline_conf = baseline.detection_confidence or 0.0

    # Apply a global phase rotation (should preserve all similarities)
    rotation_angle = 1.5  # radians
    rotation = torch.exp(
        torch.tensor(1j * rotation_angle, dtype=harness.dtype, device=harness.device)
    )

    # Rotate ALL primitives
    for prim_name in harness.primitives.keys():
        harness.primitives[prim_name] = harness.primitives[prim_name] * rotation

    # But DON'T rotate the codebook - this creates mismatch
    # The primitives and codebook are now misaligned

    post = harness.measure_health(
        test_bundle=test_bundle, known_detections=known_detections
    )
    post_acc = post.detection_accuracy if post.detection_accuracy is not None else 0.0
    post_conf = (
        post.detection_confidence if post.detection_confidence is not None else 0.0
    )

    return {
        "baseline_accuracy": baseline_acc,
        "baseline_confidence": baseline_conf,
        "post_accuracy": post_acc,
        "post_confidence": post_conf,
        "accuracy_drop": baseline_acc - post_acc,
        "confidence_drop": baseline_conf - post_conf,
        "gap": (baseline_conf - post_conf) - (baseline_acc - post_acc),
    }


def _strategy_gradual_primitive_swap(dimensions: int) -> dict:
    """Gradually swap primitives with each other, confusing the detection."""
    harness = create_harness(dimensions=dimensions)
    test_bundle, known_detections = create_controlled_test_data(harness)

    baseline = harness.measure_health(
        test_bundle=test_bundle, known_detections=known_detections
    )
    baseline_acc = baseline.detection_accuracy or 1.0
    baseline_conf = baseline.detection_confidence or 0.0

    # Swap low_stock and high_margin_leak partially
    # This should cause detection confusion while maintaining overall "shape"
    swap_amount = 0.7

    low_stock = harness.primitives["low_stock"].clone()
    margin_leak = harness.primitives["high_margin_leak"].clone()

    harness.primitives["low_stock"] = harness._normalize(
        (1 - swap_amount) * low_stock + swap_amount * margin_leak
    )
    harness.primitives["high_margin_leak"] = harness._normalize(
        (1 - swap_amount) * margin_leak + swap_amount * low_stock
    )

    post = harness.measure_health(
        test_bundle=test_bundle, known_detections=known_detections
    )
    post_acc = post.detection_accuracy if post.detection_accuracy is not None else 0.0
    post_conf = (
        post.detection_confidence if post.detection_confidence is not None else 0.0
    )

    return {
        "baseline_accuracy": baseline_acc,
        "baseline_confidence": baseline_conf,
        "post_accuracy": post_acc,
        "post_confidence": post_conf,
        "accuracy_drop": baseline_acc - post_acc,
        "confidence_drop": baseline_conf - post_conf,
        "gap": (baseline_conf - post_conf) - (baseline_acc - post_acc),
    }


# =============================================================================
# PROBE 3: COMPOUND DRIFT (BOILING FROG)
# =============================================================================


@dataclass
class CompoundDriftResult:
    """Results from compound drift probe."""

    steps: list[int] = field(default_factory=list)
    health_pass: list[bool] = field(default_factory=list)
    accuracy_values: list[float] = field(default_factory=list)
    confidence_values: list[float] = field(default_factory=list)
    similarity_to_original: list[float] = field(default_factory=list)

    # Critical findings
    last_healthy_step: int | None = None
    first_failed_step: int | None = None
    alarm_raised_at: int | None = None  # Step where health check would have caught it
    slid_into_failure: bool = False  # Did it fail without warning?

    trajectory_log: list[dict] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)


def probe_compound_drift(
    n_steps: int = 50,
    magnitude_per_step: float = 0.02,
    health_threshold: float = 0.7,
    alarm_threshold: float = 0.05,  # Health drop that should trigger alarm
    dimensions: int = 2048,
) -> CompoundDriftResult:
    """
    PROBE 3: Compound drift (boiling frog).

    Apply small modifications that each pass health checks, but compound into failure.
    Track whether there's a detectable "point of no return".
    """
    logger.info("\n" + "=" * 70)
    logger.info("PROBE 3: COMPOUND DRIFT (BOILING FROG)")
    logger.info("=" * 70)
    logger.info(
        f"\nApplying {n_steps} small modifications (magnitude={magnitude_per_step} each)"
    )
    logger.info(
        f"Health threshold: {health_threshold}, Alarm threshold: {alarm_threshold}"
    )

    result = CompoundDriftResult()

    harness = create_harness(dimensions=dimensions)
    test_bundle, known_detections = create_controlled_test_data(harness)

    # Store original state
    original_primitives = {k: v.clone() for k, v in harness.primitives.items()}

    # Baseline
    baseline = harness.measure_health(
        test_bundle=test_bundle, known_detections=known_detections
    )
    baseline_acc = baseline.detection_accuracy or 1.0
    baseline_conf = baseline.detection_confidence or 0.0
    prev_acc = baseline_acc

    logger.info(
        f"\nBaseline: accuracy={baseline_acc:.3f}, confidence={baseline_conf:.3f}"
    )
    logger.info("\n" + "-" * 90)
    logger.info(
        f"{'Step':>4} {'Accuracy':>10} {'Confidence':>12} {'Sim to Orig':>12} {'Health':>8} {'Alarm':>8}"
    )
    logger.info("-" * 90)

    all_primitives = list(harness.primitives.keys())

    for step in range(n_steps):
        # Apply small modification to a rotating set of primitives
        prim_to_modify = all_primitives[step % len(all_primitives)]

        harness.apply_primitive_perturbation(
            prim_to_modify,
            magnitude=magnitude_per_step,
            perturbation_type="directional",
        )

        # Measure health
        health = harness.measure_health(
            test_bundle=test_bundle, known_detections=known_detections
        )
        acc = (
            health.detection_accuracy if health.detection_accuracy is not None else 0.0
        )
        conf = (
            health.detection_confidence
            if health.detection_confidence is not None
            else 0.0
        )

        # Calculate similarity to original
        mean_sim = sum(
            harness.similarity(harness.primitives[k], original_primitives[k])
            for k in harness.primitives
        ) / len(harness.primitives)

        # Health check
        health_pass = acc >= health_threshold

        # Alarm check (would we have caught this?)
        acc_drop = prev_acc - acc
        alarm_triggered = acc_drop > alarm_threshold

        result.steps.append(step)
        result.accuracy_values.append(acc)
        result.confidence_values.append(conf)
        result.similarity_to_original.append(mean_sim)
        result.health_pass.append(health_pass)

        result.trajectory_log.append(
            {
                "step": step,
                "accuracy": acc,
                "confidence": conf,
                "similarity_to_original": mean_sim,
                "health_pass": health_pass,
                "accuracy_drop_from_prev": acc_drop,
                "alarm_triggered": alarm_triggered,
                "primitive_modified": prim_to_modify,
            }
        )

        health_str = "PASS" if health_pass else "FAIL"
        alarm_str = "ALARM!" if alarm_triggered else "-"

        if step % 5 == 0 or not health_pass or alarm_triggered:
            logger.info(
                f"{step:4d} {acc:10.3f} {conf:12.3f} {mean_sim:12.3f} {health_str:>8} {alarm_str:>8}"
            )

        # Track transitions
        if health_pass and result.last_healthy_step is None:
            result.last_healthy_step = step
        elif health_pass:
            result.last_healthy_step = step

        if not health_pass and result.first_failed_step is None:
            result.first_failed_step = step

        if alarm_triggered and result.alarm_raised_at is None:
            result.alarm_raised_at = step

        prev_acc = acc

    logger.info("-" * 90)

    # Analyze trajectory
    logger.info("\n" + "=" * 70)
    logger.info("COMPOUND DRIFT ANALYSIS")
    logger.info("=" * 70)

    if result.first_failed_step is not None:
        logger.info(f"System failed at step {result.first_failed_step}")

        if (
            result.alarm_raised_at is not None
            and result.alarm_raised_at < result.first_failed_step
        ):
            result.observations.append(
                f"DETECTABLE: Alarm raised at step {result.alarm_raised_at}, "
                f"before failure at step {result.first_failed_step}"
            )
            logger.info(
                f"DETECTABLE: Alarm raised at step {result.alarm_raised_at} (before failure)"
            )
        else:
            result.slid_into_failure = True
            result.observations.append(
                f"CRITICAL: System slid into failure without warning! "
                f"No alarm before failure at step {result.first_failed_step}"
            )
            logger.info("CRITICAL: System slid into failure WITHOUT WARNING!")
    else:
        result.observations.append(
            f"System remained healthy through all {n_steps} steps"
        )
        logger.info(f"System remained healthy through all {n_steps} steps")

    # Check for gradual confidence erosion
    if result.confidence_values:
        conf_start = (
            result.confidence_values[0]
            if result.confidence_values[0]
            else baseline_conf
        )
        conf_end = result.confidence_values[-1]
        conf_erosion = conf_start - conf_end

        if conf_erosion > 0.2:
            result.observations.append(
                f"Confidence eroded by {conf_erosion:.3f} over the trajectory"
            )
            logger.info(f"Confidence eroded by {conf_erosion:.3f}")

    return result


# =============================================================================
# MAIN RUNNER
# =============================================================================


def run_all_probes(dimensions: int = 2048) -> dict[str, Any]:
    """Run all three adversarial probes."""

    results = {}

    # Probe 1: Breaking point
    results["breaking_point"] = probe_breaking_point(dimensions=dimensions)

    # Probe 2: Silent failure hunt
    results["silent_failure_hunt"] = probe_silent_failure(dimensions=dimensions)

    # Probe 3: Compound drift
    results["compound_drift"] = probe_compound_drift(
        n_steps=60,
        magnitude_per_step=0.025,
        dimensions=dimensions,
    )

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY: ADVERSARIAL PROBES")
    logger.info("=" * 70)

    logger.info("\n1. BREAKING POINT:")
    for obs in results["breaking_point"].observations:
        logger.info(f"   - {obs}")

    logger.info("\n2. SILENT FAILURE HUNT:")
    for obs in results["silent_failure_hunt"].observations:
        logger.info(f"   - {obs}")

    logger.info("\n3. COMPOUND DRIFT:")
    for obs in results["compound_drift"].observations:
        logger.info(f"   - {obs}")

    # Overall assessment
    logger.info("\n" + "=" * 70)
    logger.info("OVERALL ASSESSMENT")
    logger.info("=" * 70)

    critical_findings = []

    if results["breaking_point"].silent_failure_detected:
        critical_findings.append("Silent failure found in breaking point probe")

    if results["silent_failure_hunt"].silent_failure_found:
        critical_findings.append(
            f"Silent failure found via {results['silent_failure_hunt'].most_dangerous_strategy}"
        )

    if results["compound_drift"].slid_into_failure:
        critical_findings.append(
            "System slid into failure without warning (boiling frog)"
        )

    if critical_findings:
        logger.info("\nCRITICAL FINDINGS:")
        for finding in critical_findings:
            logger.info(f"  *** {finding} ***")
    else:
        logger.info(
            "\nNo critical failures found - system appears robust to tested attacks"
        )

    return results


if __name__ == "__main__":
    run_all_probes()
