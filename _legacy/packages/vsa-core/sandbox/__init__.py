"""
VSA Self-Modification Sandbox

A rigorous test harness for exploring the safety and coherence of self-modifying
Vector Symbolic Architecture systems.

Core Research Questions:
    1. Does coherent self-modification exist, or does it inevitably degrade?
    2. Are there stable attractor states?
    3. Are there degenerate configurations the system can fall into?
    4. Can the system detect its own degradation, or does modification corrupt self-assessment?
    5. What are the actual invariants (if any)?

Main Components:
    - VSASandboxHarness: Core sandbox environment for controlled experiments
    - Test sequences: Controlled drift, aggressive modification, recovery, self-evaluation integrity
    - Visualization: Plotting tools for understanding dynamics
    - Anomaly detection: Meta-level monitoring of system behavior

Quick Start:
    from vsa_core.sandbox import create_harness, run_all_tests

    # Run all tests
    results = run_all_tests()

    # Or run specific test
    harness = create_harness()
    from vsa_core.sandbox.test_sequences import run_self_evaluation_integrity_test
    result = run_self_evaluation_integrity_test(harness)

See run_sandbox.py for CLI usage.
"""

from .metacognitive_loop import (
    BaselineAgent,
    Decision,
    MetacognitiveAgent,
    MetacognitiveLoop,
    MetacognitiveResult,
    Observation,
    RuleBasedAgent,
    Strategy,
    compare_agent_behaviors,
    print_metacognitive_report,
    run_metacognitive_test,
)
from .structure_learning import (
    StructureCritic,
    StructureLearningLoop,
    StructureLearningResult,
    StructureModifier,
    StructureOperation,
    print_structure_learning_report,
    run_structure_learning_test,
)
from .test_sequences import (
    AggressiveModificationResult,
    ControlledDriftResult,
    RecoveryTestResult,
    SelfEvaluationIntegrityResult,
    hunt_for_invariants,
    run_aggressive_modification_test,
    run_all_tests,
    run_controlled_drift_test,
    run_phase_drift_sweep,
    run_recovery_test,
    run_self_evaluation_integrity_test,
)
from .vsa_sandbox_harness import (
    DriftMeasurement,
    GeometricSnapshot,
    HealthMetrics,
    ModificationRecord,
    VSASandboxHarness,
    create_harness,
)

__all__ = [
    # Core harness
    "VSASandboxHarness",
    "GeometricSnapshot",
    "HealthMetrics",
    "DriftMeasurement",
    "ModificationRecord",
    "create_harness",
    # Test sequences
    "run_controlled_drift_test",
    "run_aggressive_modification_test",
    "run_recovery_test",
    "run_self_evaluation_integrity_test",
    "run_phase_drift_sweep",
    "hunt_for_invariants",
    "run_all_tests",
    # Result types
    "ControlledDriftResult",
    "AggressiveModificationResult",
    "RecoveryTestResult",
    "SelfEvaluationIntegrityResult",
    # Metacognitive loop
    "MetacognitiveAgent",
    "RuleBasedAgent",
    "BaselineAgent",
    "MetacognitiveLoop",
    "MetacognitiveResult",
    "Strategy",
    "Decision",
    "Observation",
    "run_metacognitive_test",
    "compare_agent_behaviors",
    "print_metacognitive_report",
    # Structure learning
    "StructureOperation",
    "StructureLearningResult",
    "StructureCritic",
    "StructureModifier",
    "StructureLearningLoop",
    "run_structure_learning_test",
    "print_structure_learning_report",
]
