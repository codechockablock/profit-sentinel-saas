"""
VSA Sandbox Visualization

Provides visualization tools for understanding self-modification dynamics:
    - Geometric drift over modification sequences
    - Health metric trajectories
    - Similarity matrix evolution
    - Anomaly detection on system behavior

Uses matplotlib for plotting (optional dependency).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

# Try to import matplotlib, provide fallback if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not available - visualization disabled")


def check_matplotlib():
    """Raise informative error if matplotlib not available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


# =============================================================================
# DRIFT VISUALIZATION
# =============================================================================


def plot_drift_trajectory(
    drift_trajectory: list[dict],
    title: str = "Geometric Drift Over Time",
    save_path: str | Path | None = None,
) -> None:
    """Plot drift from baseline over modification steps.

    Args:
        drift_trajectory: List of drift measurements from test sequence
        title: Plot title
        save_path: Optional path to save figure
    """
    check_matplotlib()

    steps = [d.get("step", i) for i, d in enumerate(drift_trajectory)]
    mean_drift = [d.get("mean_drift", 0) for d in drift_trajectory]
    max_drift = [d.get("max_drift", 0) for d in drift_trajectory]
    frobenius = [d.get("frobenius_dist", 0) for d in drift_trajectory]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Mean drift
    axes[0].plot(steps, mean_drift, 'b-', linewidth=2, marker='o', markersize=4)
    axes[0].set_xlabel("Modification Step")
    axes[0].set_ylabel("Mean Angular Drift (rad)")
    axes[0].set_title("Mean Primitive Drift")
    axes[0].grid(True, alpha=0.3)

    # Max drift
    axes[1].plot(steps, max_drift, 'r-', linewidth=2, marker='o', markersize=4)
    axes[1].set_xlabel("Modification Step")
    axes[1].set_ylabel("Max Angular Drift (rad)")
    axes[1].set_title("Max Primitive Drift")
    axes[1].grid(True, alpha=0.3)

    # Frobenius distance
    axes[2].plot(steps, frobenius, 'g-', linewidth=2, marker='o', markersize=4)
    axes[2].set_xlabel("Modification Step")
    axes[2].set_ylabel("Frobenius Distance")
    axes[2].set_title("Similarity Matrix Drift")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved drift trajectory plot to {save_path}")

    plt.show()


def plot_health_trajectory(
    health_trajectory: list[dict],
    title: str = "System Health Over Modifications",
    save_path: str | Path | None = None,
) -> None:
    """Plot health metrics over modification steps.

    Args:
        health_trajectory: List of health metric dicts from test sequence
        title: Plot title
        save_path: Optional path to save figure
    """
    check_matplotlib()

    steps = list(range(len(health_trajectory)))

    # Extract metrics
    binding_acc = [h.get("binding_accuracy", 1.0) for h in health_trajectory]
    detection_acc = [h.get("detection_accuracy") for h in health_trajectory]
    retrieval_acc = [h.get("retrieval_accuracy", 1.0) for h in health_trajectory]
    multihop_acc = [h.get("multihop_accuracy", 1.0) for h in health_trajectory]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, binding_acc, 'b-', linewidth=2, marker='o', markersize=4, label='Binding Accuracy')
    if any(d is not None for d in detection_acc):
        detection_acc_clean = [d if d is not None else 0 for d in detection_acc]
        ax.plot(steps, detection_acc_clean, 'r-', linewidth=2, marker='s', markersize=4, label='Detection Accuracy')
    ax.plot(steps, retrieval_acc, 'g-', linewidth=2, marker='^', markersize=4, label='Retrieval Accuracy')
    ax.plot(steps, multihop_acc, 'm-', linewidth=2, marker='d', markersize=4, label='Multi-hop Accuracy')

    # Add threshold lines
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='Good threshold (0.9)')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Degradation threshold (0.5)')

    ax.set_xlabel("Modification Step", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved health trajectory plot to {save_path}")

    plt.show()


def plot_self_evaluation_integrity(
    result: dict,
    save_path: str | Path | None = None,
) -> None:
    """Visualize self-evaluation integrity test results.

    This is the critical visualization showing whether the system's
    confidence tracks its actual accuracy.

    Args:
        result: SelfEvaluationIntegrityResult as dict
        save_path: Optional path to save figure
    """
    check_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Before/After comparison
    metrics = ['Detection\nAccuracy', 'Detection\nConfidence', 'False Negative\nRate']
    baseline_vals = [
        result.get("baseline_detection_accuracy", 0),
        result.get("baseline_detection_confidence", 0),
        result.get("baseline_false_negative_rate", 0),
    ]
    post_vals = [
        result.get("post_modification_detection_accuracy", 0),
        result.get("post_modification_detection_confidence", 0),
        result.get("post_modification_false_negative_rate", 0),
    ]

    x = range(len(metrics))
    width = 0.35

    bars1 = axes[0].bar([i - width/2 for i in x], baseline_vals, width, label='Baseline', color='steelblue')
    bars2 = axes[0].bar([i + width/2 for i in x], post_vals, width, label='Post-Modification', color='coral')

    axes[0].set_ylabel('Value')
    axes[0].set_title('Baseline vs Post-Modification')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        axes[0].annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    # Right: The critical gap analysis
    accuracy_drop = result.get("accuracy_drop", 0)
    confidence_drop = result.get("confidence_drop", 0)
    gap = result.get("confidence_accuracy_gap", 0)

    drops = ['Accuracy\nDrop', 'Confidence\nDrop', 'Gap\n(Conf - Acc)']
    values = [accuracy_drop, confidence_drop, gap]

    # Color based on whether gap is concerning
    colors = ['steelblue', 'steelblue', 'green' if gap >= 0 else 'red']

    bars = axes[1].bar(drops, values, color=colors, edgecolor='black')

    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_ylabel('Change')
    axes[1].set_title('Critical: Accuracy vs Confidence Drop')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Annotate the interpretation
    corruption_type = result.get("corruption_type", "unknown")
    if corruption_type == "silent_failure":
        annotation = "⚠️ SILENT FAILURE\nAccuracy dropped more than confidence"
        color = 'red'
    elif corruption_type == "detectable":
        annotation = "✓ Detectable degradation\nConfidence reflects accuracy"
        color = 'green'
    else:
        annotation = "○ Benign modification\nMinimal impact"
        color = 'gray'

    axes[1].annotate(annotation,
                    xy=(0.5, 0.95), xycoords='axes fraction',
                    ha='center', va='top',
                    fontsize=11, color=color,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        axes[1].annotate(f'{val:+.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -12),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=10, fontweight='bold')

    fig.suptitle('Self-Evaluation Integrity Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved self-evaluation integrity plot to {save_path}")

    plt.show()


def plot_phase_drift_sweep(
    sweep_results: dict,
    save_path: str | Path | None = None,
) -> None:
    """Visualize phase drift sweep results as heatmaps.

    Args:
        sweep_results: Results from run_phase_drift_sweep
        save_path: Optional path to save figure
    """
    check_matplotlib()

    results = sweep_results.get("results", [])
    if not results:
        print("No sweep results to plot")
        return

    drift_types = sweep_results["sweep_config"]["drift_types"]
    drift_angles = sweep_results["sweep_config"]["drift_angles"]

    # Build matrices for binding and detection drops
    binding_drop_matrix = []
    detection_drop_matrix = []

    for dtype in drift_types:
        binding_row = []
        detection_row = []
        for angle in drift_angles:
            # Find matching result
            for r in results:
                if r["drift_type"] == dtype and r["drift_angle"] == angle:
                    binding_row.append(r.get("binding_drop", 0))
                    detection_row.append(r.get("detection_drop", 0))
                    break
            else:
                binding_row.append(0)
                detection_row.append(0)
        binding_drop_matrix.append(binding_row)
        detection_drop_matrix.append(detection_row)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Binding drop heatmap
    im1 = axes[0].imshow(binding_drop_matrix, cmap='Reds', aspect='auto')
    axes[0].set_xticks(range(len(drift_angles)))
    axes[0].set_xticklabels([f'{a:.2f}' for a in drift_angles])
    axes[0].set_yticks(range(len(drift_types)))
    axes[0].set_yticklabels(drift_types)
    axes[0].set_xlabel('Drift Angle (radians)')
    axes[0].set_ylabel('Drift Type')
    axes[0].set_title('Binding Accuracy Drop')
    plt.colorbar(im1, ax=axes[0], label='Drop')

    # Add values
    for i in range(len(drift_types)):
        for j in range(len(drift_angles)):
            text = axes[0].text(j, i, f'{binding_drop_matrix[i][j]:.2f}',
                               ha='center', va='center', color='black', fontsize=9)

    # Detection drop heatmap
    im2 = axes[1].imshow(detection_drop_matrix, cmap='Reds', aspect='auto')
    axes[1].set_xticks(range(len(drift_angles)))
    axes[1].set_xticklabels([f'{a:.2f}' for a in drift_angles])
    axes[1].set_yticks(range(len(drift_types)))
    axes[1].set_yticklabels(drift_types)
    axes[1].set_xlabel('Drift Angle (radians)')
    axes[1].set_ylabel('Drift Type')
    axes[1].set_title('Detection Accuracy Drop')
    plt.colorbar(im2, ax=axes[1], label='Drop')

    # Add values
    for i in range(len(drift_types)):
        for j in range(len(drift_angles)):
            text = axes[1].text(j, i, f'{detection_drop_matrix[i][j]:.2f}',
                               ha='center', va='center', color='black', fontsize=9)

    fig.suptitle('Phase Drift Sweep: Impact on System Health', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved phase drift sweep plot to {save_path}")

    plt.show()


def plot_similarity_matrix_comparison(
    baseline_matrix: torch.Tensor,
    current_matrix: torch.Tensor,
    labels: list[str] | None = None,
    title: str = "Similarity Matrix: Baseline vs Current",
    save_path: str | Path | None = None,
) -> None:
    """Compare similarity matrices before and after modification.

    Args:
        baseline_matrix: Similarity matrix at baseline
        current_matrix: Current similarity matrix
        labels: Labels for rows/columns
        title: Plot title
        save_path: Optional path to save figure
    """
    check_matplotlib()

    # Convert to numpy
    baseline = baseline_matrix.cpu().numpy() if isinstance(baseline_matrix, torch.Tensor) else baseline_matrix
    current = current_matrix.cpu().numpy() if isinstance(current_matrix, torch.Tensor) else current_matrix

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Baseline
    im1 = axes[0].imshow(baseline, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title('Baseline')
    plt.colorbar(im1, ax=axes[0])

    # Current
    im2 = axes[1].imshow(current, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title('Current')
    plt.colorbar(im2, ax=axes[1])

    # Difference
    diff = current - baseline
    max_diff = max(abs(diff.min()), abs(diff.max()))
    im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-max_diff, vmax=max_diff)
    axes[2].set_title(f'Difference (Frobenius: {float(torch.norm(torch.tensor(diff), p="fro")):.4f})')
    plt.colorbar(im3, ax=axes[2])

    if labels:
        for ax in axes:
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved similarity matrix comparison to {save_path}")

    plt.show()


def plot_recovery_analysis(
    result: dict,
    save_path: str | Path | None = None,
) -> None:
    """Visualize recovery test results.

    Args:
        result: RecoveryTestResult as dict
        save_path: Optional path to save figure
    """
    check_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Health comparison (before, after modification, after restore)
    states = ['Before\nModification', 'After\nModification', 'After\nRestore']

    health_before = result.get("health_before_modification", {})
    health_after = result.get("health_after_modification", {})
    health_restored = result.get("health_after_restore", {})

    binding_vals = [
        health_before.get("binding_accuracy", 0),
        health_after.get("binding_accuracy", 0),
        health_restored.get("binding_accuracy", 0),
    ]
    detection_vals = [
        health_before.get("detection_accuracy", 0) or 0,
        health_after.get("detection_accuracy", 0) or 0,
        health_restored.get("detection_accuracy", 0) or 0,
    ]

    x = range(len(states))
    width = 0.35

    axes[0].bar([i - width/2 for i in x], binding_vals, width, label='Binding', color='steelblue')
    axes[0].bar([i + width/2 for i in x], detection_vals, width, label='Detection', color='coral')

    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Health Through Recovery Cycle')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(states)
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Right: Recovery quality metrics
    metrics = ['Exact\nMatch', 'Behavioral\nEquivalence', 'Detection\nEquivalence']
    values = [
        1.0 if result.get("restore_exact_match", False) else 0.0,
        1.0 if result.get("behavioral_equivalence", False) else 0.0,
        1.0 if result.get("detection_equivalence", False) else 0.0,
    ]

    colors = ['green' if v == 1.0 else 'red' for v in values]
    bars = axes[1].bar(metrics, values, color=colors, edgecolor='black')

    axes[1].set_ylabel('Pass (1.0) / Fail (0.0)')
    axes[1].set_title('Recovery Quality Checks')
    axes[1].set_ylim(-0.1, 1.2)

    # Add pass/fail labels
    for bar, v in zip(bars, values):
        label = '✓ PASS' if v == 1.0 else '✗ FAIL'
        axes[1].annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05),
                        ha='center', va='bottom',
                        fontsize=11, fontweight='bold',
                        color='green' if v == 1.0 else 'red')

    fig.suptitle('Recovery Test Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved recovery analysis plot to {save_path}")

    plt.show()


def plot_all_results(
    results: dict,
    output_dir: str | Path = "plots",
) -> None:
    """Generate all visualizations from test results.

    Args:
        results: Full results dict from run_all_tests
        output_dir: Directory to save plots
    """
    check_matplotlib()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tests = results.get("tests", {})

    # Controlled drift
    if "controlled_drift" in tests:
        cd = tests["controlled_drift"]
        if cd.get("drift_trajectory"):
            plot_drift_trajectory(
                cd["drift_trajectory"],
                title="Controlled Drift Test: Geometric Drift",
                save_path=output_dir / "controlled_drift_trajectory.png"
            )
        if cd.get("health_trajectory"):
            plot_health_trajectory(
                cd["health_trajectory"],
                title="Controlled Drift Test: Health Metrics",
                save_path=output_dir / "controlled_drift_health.png"
            )

    # Self-evaluation integrity
    if "self_evaluation_integrity" in tests:
        plot_self_evaluation_integrity(
            tests["self_evaluation_integrity"],
            save_path=output_dir / "self_evaluation_integrity.png"
        )

    # Phase drift sweep
    if "phase_drift_sweep" in tests:
        plot_phase_drift_sweep(
            tests["phase_drift_sweep"],
            save_path=output_dir / "phase_drift_sweep.png"
        )

    # Recovery test
    if "recovery" in tests:
        plot_recovery_analysis(
            tests["recovery"],
            save_path=output_dir / "recovery_analysis.png"
        )

    print(f"\nAll plots saved to {output_dir}/")


# =============================================================================
# ANOMALY DETECTION ON SYSTEM BEHAVIOR
# =============================================================================


class BehaviorAnomalyDetector:
    """Detect anomalies in system behavior over time.

    Monitors health metrics for unusual patterns that might indicate
    emerging problems before catastrophic failure.
    """

    def __init__(self, window_size: int = 10, sensitivity: float = 2.0):
        """
        Args:
            window_size: Number of recent observations to consider
            sensitivity: Standard deviations for anomaly threshold
        """
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.history: list[dict] = []

    def add_observation(self, health: dict) -> dict[str, Any]:
        """Add a health observation and check for anomalies.

        Args:
            health: Health metrics dict

        Returns:
            Anomaly report with any detected issues
        """
        self.history.append(health)

        if len(self.history) < self.window_size:
            return {"anomalies_detected": False, "message": "Insufficient history"}

        # Get recent window
        window = self.history[-self.window_size:]

        anomalies = []

        # Check each metric
        for metric in ["binding_accuracy", "detection_accuracy", "retrieval_accuracy"]:
            values = [h.get(metric) for h in window if h.get(metric) is not None]
            if len(values) < 3:
                continue

            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = variance ** 0.5

            current = health.get(metric)
            if current is not None and std > 0:
                z_score = (current - mean) / std
                if abs(z_score) > self.sensitivity:
                    anomalies.append({
                        "metric": metric,
                        "current_value": current,
                        "expected_mean": mean,
                        "z_score": z_score,
                        "direction": "drop" if z_score < 0 else "spike",
                    })

        # Check for sudden drops (most concerning)
        sudden_drops = []
        if len(self.history) >= 2:
            prev = self.history[-2]
            for metric in ["binding_accuracy", "detection_accuracy"]:
                prev_val = prev.get(metric)
                curr_val = health.get(metric)
                if prev_val is not None and curr_val is not None:
                    drop = prev_val - curr_val
                    if drop > 0.2:  # More than 20% drop in one step
                        sudden_drops.append({
                            "metric": metric,
                            "previous": prev_val,
                            "current": curr_val,
                            "drop": drop,
                        })

        return {
            "anomalies_detected": len(anomalies) > 0 or len(sudden_drops) > 0,
            "statistical_anomalies": anomalies,
            "sudden_drops": sudden_drops,
            "recommendation": self._get_recommendation(anomalies, sudden_drops),
        }

    def _get_recommendation(self, anomalies: list, sudden_drops: list) -> str:
        """Generate recommendation based on detected anomalies."""
        if sudden_drops:
            return "CRITICAL: Sudden performance drop detected. Consider reverting to checkpoint."
        elif anomalies:
            directions = [a["direction"] for a in anomalies]
            if all(d == "drop" for d in directions):
                return "WARNING: Multiple metrics trending downward. Monitor closely."
            else:
                return "INFO: Unusual metric behavior detected. Investigate if persistent."
        return "OK: No anomalies detected."


# =============================================================================
# CONSOLE REPORT
# =============================================================================


def print_summary_report(results: dict) -> None:
    """Print a formatted summary report to console.

    Args:
        results: Full results dict from run_all_tests
    """
    print("\n" + "=" * 80)
    print("VSA SELF-MODIFICATION SANDBOX - TEST SUMMARY REPORT")
    print("=" * 80)

    tests = results.get("tests", {})

    # Controlled Drift
    print("\n### 1. CONTROLLED DRIFT TEST")
    if "controlled_drift" in tests:
        cd = tests["controlled_drift"]
        print(f"   Steps: {cd.get('n_steps', 'N/A')}")
        print(f"   Magnitude/step: {cd.get('magnitude_per_step', 'N/A')}")
        print(f"   Pattern: {cd.get('degradation_pattern', 'N/A')}")
        for obs in cd.get("observations", []):
            print(f"   → {obs}")
    else:
        print("   (not run)")

    # Aggressive Modification
    print("\n### 2. AGGRESSIVE MODIFICATION TEST")
    if "aggressive_modification" in tests:
        am = tests["aggressive_modification"]
        threshold_step = am.get("degradation_threshold_step")
        print(f"   Degradation threshold reached: {'Step ' + str(threshold_step) if threshold_step else 'Never'}")
        print(f"   Recovery possible after: {am.get('recovery_possible_after', [])}")
        print(f"   Recovery failed after: {am.get('recovery_failed_after', [])}")
        for obs in am.get("observations", []):
            print(f"   → {obs}")
    else:
        print("   (not run)")

    # Recovery Test
    print("\n### 3. RECOVERY TEST")
    if "recovery" in tests:
        rt = tests["recovery"]
        print(f"   Exact match: {'✓' if rt.get('restore_exact_match') else '✗'}")
        print(f"   Behavioral equivalence: {'✓' if rt.get('behavioral_equivalence') else '✗'}")
        print(f"   Detection equivalence: {'✓' if rt.get('detection_equivalence') else '✗'}")
        for obs in rt.get("observations", []):
            print(f"   → {obs}")
    else:
        print("   (not run)")

    # Self-Evaluation Integrity (THE CRITICAL ONE)
    print("\n### 4. SELF-EVALUATION INTEGRITY (CRITICAL)")
    if "self_evaluation_integrity" in tests:
        sei = tests["self_evaluation_integrity"]
        corruption_type = sei.get("corruption_type", "unknown")

        print(f"   Baseline accuracy: {sei.get('baseline_detection_accuracy', 0):.3f}")
        print(f"   Post-mod accuracy: {sei.get('post_modification_detection_accuracy', 0):.3f}")
        print(f"   Accuracy drop: {sei.get('accuracy_drop', 0):+.3f}")
        print(f"   Confidence drop: {sei.get('confidence_drop', 0):+.3f}")
        print(f"   Gap (conf - acc): {sei.get('confidence_accuracy_gap', 0):+.3f}")
        print(f"   Corruption type: {corruption_type.upper()}")

        if corruption_type == "silent_failure":
            print("\n   ⚠️  CRITICAL WARNING: SILENT FAILURE DETECTED")
            print("   The system cannot reliably detect its own degradation.")
            print("   Self-assessment is compromised by geometry corruption.")

        for obs in sei.get("observations", []):
            print(f"   → {obs}")
    else:
        print("   (not run)")

    # Invariant Hunting
    print("\n### 5. INVARIANT HUNTING")
    if "invariant_hunting" in tests:
        ih = tests["invariant_hunting"]
        candidates = ih.get("candidate_invariants", [])
        print(f"   Candidate invariants found: {len(candidates)}")
        for inv in candidates:
            print(f"   → {inv['name']}: {inv['hypothesis']}")
    else:
        print("   (not run)")

    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80)
