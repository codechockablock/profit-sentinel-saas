#!/usr/bin/env python3
"""
VSA Self-Modification Sandbox Runner

CLI tool for running sandbox tests and generating reports.

Usage:
    # Run all tests with default settings
    python run_sandbox.py

    # Run specific test
    python run_sandbox.py --test self-evaluation

    # Run with custom dimensions
    python run_sandbox.py --dimensions 4096

    # Force CPU (useful for debugging)
    python run_sandbox.py --device cpu

    # Run and visualize results
    python run_sandbox.py --visualize

    # Load and visualize existing results
    python run_sandbox.py --load-results sandbox_results/results.json --visualize
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from vsa_sandbox_harness import create_harness
from test_sequences import (
    run_controlled_drift_test,
    run_aggressive_modification_test,
    run_recovery_test,
    run_self_evaluation_integrity_test,
    run_phase_drift_sweep,
    hunt_for_invariants,
    run_all_tests,
)


def main():
    parser = argparse.ArgumentParser(
        description="VSA Self-Modification Sandbox Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_sandbox.py                     # Run all tests
  python run_sandbox.py --test self-eval    # Run critical self-evaluation test
  python run_sandbox.py --test drift        # Run controlled drift test
  python run_sandbox.py --visualize         # Generate visualizations after tests
        """,
    )

    parser.add_argument(
        "--test",
        choices=["all", "drift", "aggressive", "recovery", "self-eval", "sweep", "invariants", "metacog", "metacog-compare", "structure"],
        default="all",
        help="Which test to run (default: all)",
    )

    parser.add_argument(
        "--dimensions",
        type=int,
        default=2048,
        help="Vector dimensions (default: 2048)",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
        help="Compute device (default: auto)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="sandbox_results",
        help="Output directory for results (default: sandbox_results)",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations after tests",
    )

    parser.add_argument(
        "--load-results",
        type=str,
        help="Load existing results file instead of running tests",
    )

    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Print summary report only (requires --load-results)",
    )

    # Test-specific parameters
    parser.add_argument(
        "--drift-steps",
        type=int,
        default=20,
        help="Number of steps for drift test (default: 20)",
    )

    parser.add_argument(
        "--drift-magnitude",
        type=float,
        default=0.05,
        help="Magnitude per step for drift test (default: 0.05)",
    )

    parser.add_argument(
        "--corruption-magnitude",
        type=float,
        default=0.3,
        help="Corruption magnitude for self-eval test (default: 0.3)",
    )

    # Metacognitive loop parameters
    parser.add_argument(
        "--metacog-steps",
        type=int,
        default=100,
        help="Max steps for metacognitive loop (default: 100)",
    )

    parser.add_argument(
        "--plateau-threshold",
        type=int,
        default=10,
        help="Steps without improvement to trigger plateau detection (default: 10)",
    )

    parser.add_argument(
        "--metacog-scenario",
        choices=["default", "improvable"],
        default="default",
        help="Test scenario for metacognitive loop (default: default)",
    )

    args = parser.parse_args()

    # Handle device
    device = None if args.device == "auto" else args.device

    # Handle load/report mode
    if args.load_results:
        print(f"Loading results from {args.load_results}...")
        with open(args.load_results, "r") as f:
            results = json.load(f)

        if args.report_only:
            try:
                from visualization import print_summary_report
                print_summary_report(results)
            except ImportError:
                print("Could not import visualization module")
            return

        if args.visualize:
            try:
                from visualization import plot_all_results, print_summary_report
                print_summary_report(results)
                plot_all_results(results, output_dir=args.output_dir)
            except ImportError as e:
                print(f"Visualization requires matplotlib: {e}")
            return

        print("Loaded results. Use --visualize or --report-only to process.")
        return

    # Run tests
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("VSA SELF-MODIFICATION SANDBOX")
    print("=" * 80)
    print(f"Dimensions: {args.dimensions}")
    print(f"Device: {device or 'auto'}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    results = {
        "timestamp": time.time(),
        "config": {
            "dimensions": args.dimensions,
            "device": device or "auto",
            "test": args.test,
        },
        "tests": {},
    }

    if args.test == "all":
        results = run_all_tests(
            output_dir=output_dir,
            dimensions=args.dimensions,
            device=device,
        )

    elif args.test == "drift":
        harness = create_harness(dimensions=args.dimensions, device=device)
        result = run_controlled_drift_test(
            harness=harness,
            n_steps=args.drift_steps,
            magnitude_per_step=args.drift_magnitude,
        )
        results["tests"]["controlled_drift"] = _to_dict(result)

    elif args.test == "aggressive":
        harness = create_harness(dimensions=args.dimensions, device=device)
        result = run_aggressive_modification_test(harness=harness)
        results["tests"]["aggressive_modification"] = _to_dict(result)

    elif args.test == "recovery":
        harness = create_harness(dimensions=args.dimensions, device=device)
        result = run_recovery_test(harness=harness)
        results["tests"]["recovery"] = _to_dict(result)

    elif args.test == "self-eval":
        print("\n*** RUNNING CRITICAL SELF-EVALUATION INTEGRITY TEST ***\n")
        harness = create_harness(dimensions=args.dimensions, device=device)
        result = run_self_evaluation_integrity_test(
            harness=harness,
            modification_magnitude=args.corruption_magnitude,
        )
        results["tests"]["self_evaluation_integrity"] = _to_dict(result)

        # Always print detailed report for this critical test
        print("\n" + "=" * 60)
        print("SELF-EVALUATION INTEGRITY RESULTS")
        print("=" * 60)
        print(f"Baseline detection accuracy: {result.baseline_detection_accuracy:.3f}")
        print(f"Post-modification accuracy:  {result.post_modification_detection_accuracy:.3f}")
        print(f"Accuracy drop:              {result.accuracy_drop:+.3f}")
        print(f"Confidence drop:            {result.confidence_drop:+.3f}")
        print(f"Gap (confidence - accuracy): {result.confidence_accuracy_gap:+.3f}")
        print(f"\nCorruption type: {result.corruption_type.upper()}")

        if result.corruption_type == "silent_failure":
            print("\n" + "!" * 60)
            print("!!! CRITICAL: SILENT FAILURE DETECTED !!!")
            print("!" * 60)
            print("The system's self-assessment is compromised.")
            print("It believes it is functioning correctly while degraded.")
            print("This is the failure mode we must protect against.")
            print("!" * 60)

    elif args.test == "sweep":
        harness = create_harness(dimensions=args.dimensions, device=device)
        result = run_phase_drift_sweep(harness=harness)
        results["tests"]["phase_drift_sweep"] = result

    elif args.test == "invariants":
        harness = create_harness(dimensions=args.dimensions, device=device)
        result = hunt_for_invariants(harness=harness)
        results["tests"]["invariant_hunting"] = result

    elif args.test == "metacog":
        print("\n*** RUNNING METACOGNITIVE LOOP TEST ***\n")
        print(f"Scenario: {args.metacog_scenario}")
        from metacognitive_loop import run_metacognitive_test, print_metacognitive_report
        result = run_metacognitive_test(
            dimensions=args.dimensions,
            device=device,
            max_steps=args.metacog_steps,
            agent_type="rule_based",
            plateau_threshold=args.plateau_threshold,
            scenario=args.metacog_scenario,
        )
        print_metacognitive_report(result)
        results["tests"]["metacognitive_loop"] = _metacog_to_dict(result)

    elif args.test == "metacog-compare":
        print("\n*** COMPARING BASELINE VS METACOGNITIVE AGENTS ***\n")
        from metacognitive_loop import compare_agent_behaviors, print_metacognitive_report
        comparison = compare_agent_behaviors(
            dimensions=args.dimensions,
            device=device,
            max_steps=args.metacog_steps,
        )
        print("\n" + "=" * 70)
        print("BASELINE AGENT")
        print_metacognitive_report(comparison["baseline"])
        print("\n" + "=" * 70)
        print("METACOGNITIVE AGENT")
        print_metacognitive_report(comparison["metacognitive"])

        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        comp = comparison["comparison"]
        print(f"Baseline: {comp['baseline_steps']} steps, final accuracy {comp['baseline_final_accuracy']:.3f}")
        print(f"Metacog:  {comp['metacog_steps']} steps, final accuracy {comp['metacog_final_accuracy']:.3f}")
        print(f"Metacog stopped self: {comp['metacog_stopped_self']}")
        print(f"Metacog switched strategy: {comp['metacog_switched_strategy']}")
        print(f"Metacog success level: {comp['metacog_success_level']}")

        results["tests"]["metacognitive_comparison"] = {
            "baseline": _metacog_to_dict(comparison["baseline"]),
            "metacognitive": _metacog_to_dict(comparison["metacognitive"]),
            "comparison": comparison["comparison"],
        }

    elif args.test == "structure":
        print("\n*** RUNNING STRUCTURE LEARNING TEST ***\n")
        from structure_learning import run_structure_learning_test, print_structure_learning_report
        result = run_structure_learning_test(
            dimensions=args.dimensions,
            device=device,
            max_iterations=args.metacog_steps,  # Reuse this param
        )
        print_structure_learning_report(result)
        results["tests"]["structure_learning"] = _structure_to_dict(result)

    # Save results
    results_file = output_dir / f"sandbox_{args.test}_{int(time.time())}.json"
    with open(results_file, "w") as f:
        json.dump(_make_serializable(results), f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Visualize if requested
    if args.visualize:
        try:
            from visualization import plot_all_results, print_summary_report
            print_summary_report(results)
            plot_all_results(results, output_dir=output_dir)
        except ImportError as e:
            print(f"Visualization requires matplotlib: {e}")
            print("Install with: pip install matplotlib")


def _to_dict(obj):
    """Convert dataclass or object to dict."""
    from dataclasses import asdict, is_dataclass
    if is_dataclass(obj):
        return asdict(obj)
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    return obj


def _structure_to_dict(result):
    """Convert StructureLearningResult to serializable dict."""
    from dataclasses import asdict, is_dataclass
    from enum import Enum

    def convert(obj):
        if is_dataclass(obj):
            return {k: convert(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, torch.Tensor):
            return "tensor"  # Don't serialize tensors
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, set)):
            return [convert(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    return convert(result)


def _make_serializable(obj):
    """Make object JSON serializable."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def _metacog_to_dict(result):
    """Convert MetacognitiveResult to serializable dict."""
    from dataclasses import asdict, is_dataclass
    from enum import Enum

    def convert(obj):
        if is_dataclass(obj):
            return {k: convert(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    return convert(result)


if __name__ == "__main__":
    main()
