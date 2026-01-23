"""
Metacognitive Loop Test Harness

Tests whether an agent can observe its own modification process and make
decisions about it - not just execute modifications, but reason about
whether to continue, stop, or change strategy.

Architecture:
    - Outer loop the agent controls
    - Agent observes: modification history, accuracy trajectory, current vs best
    - Agent decides: continue, stop, or switch strategy
    - Ground truth: external accuracy (not self-reported confidence)

Success Levels:
    - Baseline: Agent runs until external counter stops it (no metacognition)
    - Level 1: Agent recognizes plateau and stops itself
    - Level 2: Agent recognizes plateau, switches strategy, finds improvement
    - Level 3: Agent recognizes "no strategy working" and stops gracefully

The "agent" is rule-based for now, but structured for LLM swap-in later.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch
from vsa_sandbox_harness import VSASandboxHarness, create_harness

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


class Strategy(Enum):
    """Available modification strategies."""
    SMALL_PERTURBATION = "small_perturbation"
    LARGE_JUMP = "large_jump"
    STRUCTURED = "structured"  # Target specific primitives
    RESET_AND_DIVERGE = "reset_and_diverge"
    GRADIENT_DIRECTION = "gradient_direction"  # Move in direction that improves accuracy


class Decision(Enum):
    """Agent decisions at each step."""
    CONTINUE = "continue"
    SWITCH_STRATEGY = "switch_strategy"
    STOP = "stop"


@dataclass
class Observation:
    """What the agent can see at each decision point."""
    step: int
    current_accuracy: float
    best_accuracy: float
    best_accuracy_step: int
    accuracy_history: list[float]
    current_strategy: Strategy
    strategies_tried: list[Strategy]
    steps_since_improvement: int
    total_steps: int

    # Derived metrics (agent can compute these)
    recent_trend: float | None = None  # Slope of last N accuracies
    plateau_detected: bool = False
    accuracy_variance: float = 0.0


@dataclass
class DecisionRecord:
    """Full record of an agent decision for logging."""
    step: int
    observation: Observation
    decision: Decision
    new_strategy: Strategy | None
    reasoning: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class StrategyAttempt:
    """Record of a strategy attempt."""
    strategy: Strategy
    start_step: int
    end_step: int | None = None
    start_accuracy: float = 0.0
    best_accuracy: float = 0.0
    final_accuracy: float = 0.0
    steps_run: int = 0
    improved: bool = False


@dataclass
class MetacognitiveResult:
    """Full results of a metacognitive loop run."""
    # Configuration
    max_steps: int = 0
    ground_truth_source: str = ""

    # Trajectory
    accuracy_trajectory: list[float] = field(default_factory=list)
    strategy_trajectory: list[str] = field(default_factory=list)
    decision_log: list[DecisionRecord] = field(default_factory=list)
    strategy_attempts: list[StrategyAttempt] = field(default_factory=list)

    # Outcomes
    final_accuracy: float = 0.0
    best_accuracy: float = 0.0
    best_accuracy_step: int = 0
    total_steps: int = 0

    # Success level assessment
    stopped_self: bool = False
    switched_strategy: bool = False
    found_improvement_after_switch: bool = False
    recognized_no_progress: bool = False
    success_level: int = 0  # 0=baseline, 1, 2, or 3

    # Restoration info (GA-style "return best individual")
    restored_to_best: bool = False
    pre_restoration_accuracy: float = 0.0
    post_restoration_accuracy: float = 0.0
    restoration_verified: bool = False

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0


# =============================================================================
# AGENT INTERFACE
# =============================================================================


class MetacognitiveAgent(ABC):
    """
    Abstract base class for metacognitive agents.

    Structured for easy swap to LLM-based decision maker later.
    """

    @abstractmethod
    def decide(self, observation: Observation) -> tuple[Decision, Strategy | None, str]:
        """
        Make a decision based on current observation.

        Args:
            observation: Current state visible to the agent

        Returns:
            (decision, new_strategy, reasoning)
            - decision: CONTINUE, SWITCH_STRATEGY, or STOP
            - new_strategy: If SWITCH_STRATEGY, which strategy to use
            - reasoning: Human-readable explanation of the decision
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset agent state for a new run."""
        pass


class RuleBasedAgent(MetacognitiveAgent):
    """
    Rule-based metacognitive agent with configurable thresholds.

    Rules:
    1. Plateau detection: If no improvement for N steps, consider switching
    2. Strategy exhaustion: If all strategies tried without success, stop
    3. Minimum steps per strategy: Give each strategy a fair chance
    4. Best-so-far tracking: Only switch if current strategy is clearly stuck
    """

    def __init__(
        self,
        plateau_threshold: int = 10,  # Steps without improvement = plateau
        min_steps_per_strategy: int = 5,  # Minimum steps before switching
        trend_window: int = 5,  # Window for trend calculation
        improvement_threshold: float = 0.01,  # Minimum improvement to count
        max_strategy_attempts: int = 2,  # Max times to try each strategy
    ):
        self.plateau_threshold = plateau_threshold
        self.min_steps_per_strategy = min_steps_per_strategy
        self.trend_window = trend_window
        self.improvement_threshold = improvement_threshold
        self.max_strategy_attempts = max_strategy_attempts

        # Internal state
        self.strategy_attempt_counts: dict[Strategy, int] = {s: 0 for s in Strategy}
        self.current_strategy_steps: int = 0
        self.last_switch_accuracy: float = 0.0

    def reset(self) -> None:
        self.strategy_attempt_counts = {s: 0 for s in Strategy}
        self.current_strategy_steps = 0
        self.last_switch_accuracy = 0.0

    def decide(self, obs: Observation) -> tuple[Decision, Strategy | None, str]:
        """Rule-based decision making."""

        # Track steps in current strategy
        self.current_strategy_steps += 1

        # Compute derived metrics
        obs.recent_trend = self._compute_trend(obs.accuracy_history)
        obs.plateau_detected = obs.steps_since_improvement >= self.plateau_threshold
        obs.accuracy_variance = self._compute_variance(obs.accuracy_history)

        # Rule 1: Check if all strategies exhausted
        strategies_available = [
            s for s in Strategy
            if self.strategy_attempt_counts[s] < self.max_strategy_attempts
        ]

        if not strategies_available and obs.plateau_detected:
            return (
                Decision.STOP,
                None,
                f"All strategies exhausted (tried each {self.max_strategy_attempts}x). "
                f"Best accuracy {obs.best_accuracy:.3f} at step {obs.best_accuracy_step}. "
                f"No further improvement possible with current approaches."
            )

        # Rule 2: Don't switch too early
        if self.current_strategy_steps < self.min_steps_per_strategy:
            return (
                Decision.CONTINUE,
                None,
                f"Continuing {obs.current_strategy.value}: "
                f"only {self.current_strategy_steps}/{self.min_steps_per_strategy} minimum steps completed."
            )

        # Rule 3: Check for plateau
        if obs.plateau_detected:
            # Mark current strategy as attempted
            self.strategy_attempt_counts[obs.current_strategy] += 1

            # Recalculate available strategies after incrementing count
            strategies_still_available = [
                s for s in Strategy
                if self.strategy_attempt_counts[s] < self.max_strategy_attempts
            ]

            # Find best alternative strategy
            next_strategy = self._select_next_strategy(obs, strategies_still_available)

            if next_strategy is None:
                # No alternatives, stop
                return (
                    Decision.STOP,
                    None,
                    f"Plateau detected ({obs.steps_since_improvement} steps without improvement). "
                    f"All strategies exhausted. Stopping with best accuracy {obs.best_accuracy:.3f}."
                )

            # Switch strategy
            self.current_strategy_steps = 0
            self.last_switch_accuracy = obs.current_accuracy

            return (
                Decision.SWITCH_STRATEGY,
                next_strategy,
                f"Plateau detected after {obs.steps_since_improvement} steps without improvement. "
                f"Current accuracy {obs.current_accuracy:.3f}, best was {obs.best_accuracy:.3f}. "
                f"Switching from {obs.current_strategy.value} to {next_strategy.value}."
            )

        # Rule 4: Check for negative trend
        if obs.recent_trend is not None and obs.recent_trend < -0.02:
            # Accuracy declining - might want to reset
            if Strategy.RESET_AND_DIVERGE in strategies_available:
                self.strategy_attempt_counts[obs.current_strategy] += 1
                self.current_strategy_steps = 0
                return (
                    Decision.SWITCH_STRATEGY,
                    Strategy.RESET_AND_DIVERGE,
                    f"Negative trend detected ({obs.recent_trend:.4f}). "
                    f"Resetting to best-known-good state."
                )

        # Default: continue current strategy
        trend_str = f"{obs.recent_trend:+.4f}" if obs.recent_trend else "N/A"
        return (
            Decision.CONTINUE,
            None,
            f"Continuing {obs.current_strategy.value}: "
            f"accuracy={obs.current_accuracy:.3f}, trend={trend_str}, "
            f"steps_since_improvement={obs.steps_since_improvement}"
        )

    def _compute_trend(self, history: list[float]) -> float | None:
        """Compute linear trend of recent accuracy values."""
        if len(history) < self.trend_window:
            return None

        recent = history[-self.trend_window:]
        # Simple linear regression slope
        n = len(recent)
        x_mean = (n - 1) / 2
        y_mean = sum(recent) / n

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0
        return numerator / denominator

    def _compute_variance(self, history: list[float]) -> float:
        """Compute variance of accuracy history."""
        if len(history) < 2:
            return 0.0
        mean = sum(history) / len(history)
        return sum((x - mean) ** 2 for x in history) / len(history)

    def _select_next_strategy(
        self,
        obs: Observation,
        available: list[Strategy]
    ) -> Strategy | None:
        """Select the best next strategy to try."""
        if not available:
            return None

        # Preference order based on current situation
        if obs.current_accuracy < obs.best_accuracy * 0.9:
            # Significantly worse than best - try reset first
            if Strategy.RESET_AND_DIVERGE in available:
                return Strategy.RESET_AND_DIVERGE

        # If small perturbations aren't working, try larger jumps
        if obs.current_strategy == Strategy.SMALL_PERTURBATION:
            if Strategy.LARGE_JUMP in available:
                return Strategy.LARGE_JUMP
            if Strategy.STRUCTURED in available:
                return Strategy.STRUCTURED

        # If large jumps aren't working, try structured
        if obs.current_strategy == Strategy.LARGE_JUMP:
            if Strategy.STRUCTURED in available:
                return Strategy.STRUCTURED
            if Strategy.GRADIENT_DIRECTION in available:
                return Strategy.GRADIENT_DIRECTION

        # If structured isn't working, try gradient
        if obs.current_strategy == Strategy.STRUCTURED:
            if Strategy.GRADIENT_DIRECTION in available:
                return Strategy.GRADIENT_DIRECTION
            if Strategy.RESET_AND_DIVERGE in available:
                return Strategy.RESET_AND_DIVERGE

        # If gradient isn't working, try reset
        if obs.current_strategy == Strategy.GRADIENT_DIRECTION:
            if Strategy.RESET_AND_DIVERGE in available:
                return Strategy.RESET_AND_DIVERGE

        # Default: pick first available
        return available[0]


class BaselineAgent(MetacognitiveAgent):
    """
    Baseline agent with NO metacognition - just runs until stopped externally.
    Used to establish baseline behavior.
    """

    def reset(self) -> None:
        pass

    def decide(self, observation: Observation) -> tuple[Decision, Strategy | None, str]:
        return (
            Decision.CONTINUE,
            None,
            "Baseline agent: continuing without metacognition."
        )


# =============================================================================
# MODIFICATION STRATEGIES
# =============================================================================


class ModificationStrategy:
    """Executes a modification strategy on the harness."""

    def __init__(self, harness: VSASandboxHarness):
        self.harness = harness
        self.best_state: dict | None = None
        self.best_accuracy: float = 0.0
        self.best_step: int = 0

    def save_best_state(self, accuracy: float, step: int = 0) -> None:
        """Save current state if it's the best so far."""
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_step = step
            self.best_state = {
                "primitives": {k: v.clone() for k, v in self.harness.primitives.items()},
                "codebook": {k: v.clone() for k, v in self.harness.codebook.items()},
            }

    def restore_best_state(self) -> None:
        """Restore the best-known-good state."""
        if self.best_state is not None:
            self.harness.primitives = {k: v.clone() for k, v in self.best_state["primitives"].items()}
            self.harness.codebook = {k: v.clone() for k, v in self.best_state["codebook"].items()}

    def apply(
        self,
        strategy: Strategy,
        step: int,
        failing_primitives: list[str] | None = None,
        evaluator: Optional["GroundTruthEvaluator"] = None
    ) -> dict:
        """
        Apply a modification strategy.

        Returns dict with modification details.
        """
        if strategy == Strategy.SMALL_PERTURBATION:
            return self._small_perturbation()
        elif strategy == Strategy.LARGE_JUMP:
            return self._large_jump()
        elif strategy == Strategy.STRUCTURED:
            return self._structured_modification(failing_primitives or [])
        elif strategy == Strategy.RESET_AND_DIVERGE:
            return self._reset_and_diverge()
        elif strategy == Strategy.GRADIENT_DIRECTION:
            return self._gradient_direction(evaluator)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _small_perturbation(self) -> dict:
        """Small random perturbations to all primitives."""
        magnitude = 0.05
        modified = []

        for name in self.harness.primitives:
            self.harness.apply_primitive_perturbation(
                primitive_name=name,
                magnitude=magnitude,
                perturbation_type="additive"
            )
            modified.append(name)

        return {
            "type": "small_perturbation",
            "magnitude": magnitude,
            "primitives_modified": modified,
        }

    def _large_jump(self) -> dict:
        """Larger magnitude changes."""
        magnitude = 0.2
        modified = []

        for name in self.harness.primitives:
            self.harness.apply_primitive_perturbation(
                primitive_name=name,
                magnitude=magnitude,
                perturbation_type="additive"
            )
            modified.append(name)

        return {
            "type": "large_jump",
            "magnitude": magnitude,
            "primitives_modified": modified,
        }

    def _structured_modification(self, failing_primitives: list[str]) -> dict:
        """Target specific primitives that are underperforming."""
        if not failing_primitives:
            # If no specific failures, perturb half the primitives
            all_prims = list(self.harness.primitives.keys())
            failing_primitives = all_prims[:len(all_prims) // 2]

        magnitude = 0.1
        for name in failing_primitives:
            if name in self.harness.primitives:
                self.harness.apply_primitive_perturbation(
                    primitive_name=name,
                    magnitude=magnitude,
                    perturbation_type="additive"
                )

        return {
            "type": "structured",
            "magnitude": magnitude,
            "primitives_modified": failing_primitives,
        }

    def _reset_and_diverge(self) -> dict:
        """Reset to best-known-good and try a different direction."""
        self.restore_best_state()

        # Apply a different perturbation direction
        magnitude = 0.08
        modified = list(self.harness.primitives.keys())

        # Use phase drift instead of additive perturbation
        self.harness.apply_phase_drift(
            drift_angle=magnitude,
            drift_type="uniform",
            affected_primitives=modified,
        )

        return {
            "type": "reset_and_diverge",
            "reset_to_accuracy": self.best_accuracy,
            "new_direction": "phase_drift",
            "magnitude": magnitude,
            "primitives_modified": modified,
        }

    def _gradient_direction(self, evaluator: Optional["GroundTruthEvaluator"]) -> dict:
        """
        Move primitives in a direction that might improve accuracy.

        If the evaluator has an _improvement_bias (from improvable scenario),
        use that. Otherwise, try random directions and keep the best.
        """
        modified = list(self.harness.primitives.keys())
        magnitude = 0.03  # Small steps

        # Check if evaluator has the improvement bias
        if evaluator and hasattr(evaluator, "_improvement_bias"):
            bias = evaluator._improvement_bias
            for name in modified:
                prim = self.harness.primitives[name]
                # Smaller, more controlled step
                new_prim = prim + bias * magnitude
                norm = torch.sqrt(torch.sum(torch.abs(new_prim) ** 2))
                self.harness.primitives[name] = new_prim / (norm + 1e-10)

            return {
                "type": "gradient_direction",
                "method": "known_bias",
                "magnitude": magnitude,
                "primitives_modified": modified,
            }

        # Fallback: random exploration
        return self._small_perturbation()


# =============================================================================
# GROUND TRUTH EVALUATOR
# =============================================================================


class GroundTruthEvaluator:
    """
    Evaluates accuracy against a known anomaly set.

    This is EXTERNAL to the system - the "real" accuracy,
    not the system's self-reported confidence.
    """

    def __init__(
        self,
        harness: VSASandboxHarness,
        n_test_cases: int = 50,
        scenario: str = "default"
    ):
        self.harness = harness
        self.n_test_cases = n_test_cases
        self.scenario = scenario

        # Generate fixed test set at construction
        self.test_cases = self._generate_test_cases()

    def _generate_test_cases(self) -> list[dict]:
        """Generate a fixed set of test cases with known ground truth."""
        if self.scenario == "improvable":
            return self._generate_improvable_test_cases()
        return self._generate_default_test_cases()

    def _generate_default_test_cases(self) -> list[dict]:
        """Standard test cases - may or may not be improvable."""
        cases = []
        primitive_names = list(self.harness.primitives.keys())

        for i in range(self.n_test_cases):
            # Each test case: 1-3 anomalies bundled together
            n_anomalies = (i % 3) + 1
            selected = []
            for j in range(n_anomalies):
                idx = (i * 7 + j * 13) % len(primitive_names)
                selected.append(primitive_names[idx])

            # Create the bundled signal
            vectors = [self.harness.primitives[name].clone() for name in selected]
            if len(vectors) == 1:
                signal = vectors[0]
            else:
                signal = vectors[0].clone()
                for v in vectors[1:]:
                    signal = signal + v
                # Normalize
                norm = torch.sqrt(torch.sum(torch.abs(signal) ** 2))
                signal = signal / (norm + 1e-10)

            # Add noise
            noise = torch.randn_like(signal.real) + 1j * torch.randn_like(signal.real)
            noise = noise * 0.1
            signal = signal + noise
            norm = torch.sqrt(torch.sum(torch.abs(signal) ** 2))
            signal = signal / (norm + 1e-10)

            cases.append({
                "signal": signal,
                "ground_truth": set(selected),
                "n_anomalies": n_anomalies,
            })

        return cases

    def _generate_improvable_test_cases(self) -> list[dict]:
        """
        Test cases where improvement IS possible.

        The trick: we generate test signals that are offset from the current
        primitives by a consistent bias. Certain modification strategies
        (like large jumps in that direction) will improve accuracy.
        """
        cases = []
        primitive_names = list(self.harness.primitives.keys())

        # Create a consistent "bias" direction
        torch.manual_seed(42)
        bias_direction = torch.randn(
            self.harness.dimensions,
            dtype=torch.complex64,
            device=self.harness.device
        )
        bias_direction = bias_direction / torch.sqrt(torch.sum(torch.abs(bias_direction) ** 2))

        # Store bias for potential use by strategies
        self._improvement_bias = bias_direction * 0.15

        for i in range(self.n_test_cases):
            # Single anomaly per test case (simpler, more controllable)
            idx = i % len(primitive_names)
            selected = [primitive_names[idx]]

            # Get the primitive and add the bias
            prim = self.harness.primitives[selected[0]].clone()

            # The "true" signal is the primitive shifted by the bias
            signal = prim + self._improvement_bias

            # Normalize
            norm = torch.sqrt(torch.sum(torch.abs(signal) ** 2))
            signal = signal / (norm + 1e-10)

            # Add small noise
            noise = torch.randn_like(signal.real) + 1j * torch.randn_like(signal.real)
            noise = noise * 0.05
            signal = signal + noise
            norm = torch.sqrt(torch.sum(torch.abs(signal) ** 2))
            signal = signal / (norm + 1e-10)

            cases.append({
                "signal": signal,
                "ground_truth": set(selected),
                "n_anomalies": 1,
            })

        return cases

    def evaluate(self) -> tuple[float, list[str]]:
        """
        Evaluate current system against ground truth.

        Returns:
            (accuracy, list of failing primitive names)
        """
        correct = 0
        failing_primitives: set[str] = set()

        for case in self.test_cases:
            # Detect anomalies using current primitives
            detected = self._detect_anomalies(case["signal"])

            # Check accuracy
            if detected == case["ground_truth"]:
                correct += 1
            else:
                # Track which primitives are involved in failures
                missed = case["ground_truth"] - detected
                false_positives = detected - case["ground_truth"]
                failing_primitives.update(missed)
                failing_primitives.update(false_positives)

        accuracy = correct / len(self.test_cases)
        return accuracy, list(failing_primitives)

    def _detect_anomalies(self, signal: torch.Tensor, threshold: float = 0.3) -> set[str]:
        """Detect which anomalies are present in a signal."""
        detected = set()

        for name, primitive in self.harness.primitives.items():
            sim = self.harness.similarity(signal, primitive)
            if sim > threshold:
                detected.add(name)

        return detected


# =============================================================================
# METACOGNITIVE LOOP
# =============================================================================


class MetacognitiveLoop:
    """
    Main orchestrator for the metacognitive modification loop.
    """

    def __init__(
        self,
        harness: VSASandboxHarness,
        agent: MetacognitiveAgent,
        initial_strategy: Strategy = Strategy.SMALL_PERTURBATION,
        max_steps: int = 100,
        log_every: int = 5,
        evaluator_scenario: str = "default",
    ):
        self.harness = harness
        self.agent = agent
        self.initial_strategy = initial_strategy
        self.max_steps = max_steps
        self.log_every = log_every

        # Components
        self.evaluator = GroundTruthEvaluator(harness, scenario=evaluator_scenario)
        self.modifier = ModificationStrategy(harness)

    def run(self) -> MetacognitiveResult:
        """Run the metacognitive loop."""
        result = MetacognitiveResult(
            max_steps=self.max_steps,
            ground_truth_source="fixed_test_set",
            start_time=time.time(),
        )

        # Initialize
        self.agent.reset()
        current_strategy = self.initial_strategy
        strategies_tried = [current_strategy]

        # Track best
        accuracy, failing = self.evaluator.evaluate()
        best_accuracy = accuracy
        best_accuracy_step = 0
        self.modifier.save_best_state(accuracy, step=0)

        # Current strategy attempt
        current_attempt = StrategyAttempt(
            strategy=current_strategy,
            start_step=0,
            start_accuracy=accuracy,
            best_accuracy=accuracy,
        )

        result.accuracy_trajectory.append(accuracy)
        result.strategy_trajectory.append(current_strategy.value)

        logger.info(f"Starting metacognitive loop: initial accuracy={accuracy:.3f}")

        for step in range(1, self.max_steps + 1):
            # Build observation
            obs = Observation(
                step=step,
                current_accuracy=accuracy,
                best_accuracy=best_accuracy,
                best_accuracy_step=best_accuracy_step,
                accuracy_history=result.accuracy_trajectory.copy(),
                current_strategy=current_strategy,
                strategies_tried=strategies_tried.copy(),
                steps_since_improvement=step - best_accuracy_step,
                total_steps=step,
            )

            # Agent decides
            decision, new_strategy, reasoning = self.agent.decide(obs)

            # Record decision
            record = DecisionRecord(
                step=step,
                observation=obs,
                decision=decision,
                new_strategy=new_strategy,
                reasoning=reasoning,
            )
            result.decision_log.append(record)

            # Log periodically or on significant events
            if step % self.log_every == 0 or decision != Decision.CONTINUE:
                logger.info(
                    f"Step {step}: accuracy={accuracy:.3f}, "
                    f"best={best_accuracy:.3f}, "
                    f"strategy={current_strategy.value}, "
                    f"decision={decision.value}"
                )
                logger.info(f"  Reasoning: {reasoning}")

            # Handle decision
            if decision == Decision.STOP:
                result.stopped_self = True
                current_attempt.end_step = step
                current_attempt.final_accuracy = accuracy
                current_attempt.steps_run = step - current_attempt.start_step
                result.strategy_attempts.append(current_attempt)

                # Check if recognized no progress
                if "exhausted" in reasoning.lower() or "no further improvement" in reasoning.lower():
                    result.recognized_no_progress = True
                break

            elif decision == Decision.SWITCH_STRATEGY:
                result.switched_strategy = True

                # Close current attempt
                current_attempt.end_step = step
                current_attempt.final_accuracy = accuracy
                current_attempt.steps_run = step - current_attempt.start_step
                current_attempt.improved = current_attempt.best_accuracy > current_attempt.start_accuracy
                result.strategy_attempts.append(current_attempt)

                # Start new attempt
                current_strategy = new_strategy
                if new_strategy not in strategies_tried:
                    strategies_tried.append(new_strategy)

                current_attempt = StrategyAttempt(
                    strategy=current_strategy,
                    start_step=step,
                    start_accuracy=accuracy,
                    best_accuracy=accuracy,
                )

            # Apply modification
            self.modifier.apply(
                current_strategy,
                step,
                failing_primitives=failing,
                evaluator=self.evaluator
            )

            # Evaluate
            accuracy, failing = self.evaluator.evaluate()
            result.accuracy_trajectory.append(accuracy)
            result.strategy_trajectory.append(current_strategy.value)

            # Update best
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_accuracy_step = step
                self.modifier.save_best_state(accuracy, step=step)

            # Update current attempt
            if accuracy > current_attempt.best_accuracy:
                current_attempt.best_accuracy = accuracy

        else:
            # Loop ended by max_steps, not agent decision
            current_attempt.end_step = self.max_steps
            current_attempt.final_accuracy = accuracy
            current_attempt.steps_run = self.max_steps - current_attempt.start_step
            result.strategy_attempts.append(current_attempt)

        # =====================================================================
        # GA-STYLE RESTORATION: Return best individual found
        # =====================================================================
        # If current accuracy is worse than best, restore to best state
        if accuracy < best_accuracy - 0.001:  # Small epsilon to avoid noise
            logger.info(
                f"Restoring to best state: current={accuracy:.3f}, "
                f"best={best_accuracy:.3f} (from step {best_accuracy_step})"
            )

            result.pre_restoration_accuracy = accuracy
            self.modifier.restore_best_state()

            # Verify restoration worked
            restored_accuracy, _ = self.evaluator.evaluate()
            result.post_restoration_accuracy = restored_accuracy
            result.restored_to_best = True

            # Check if restoration was successful
            if abs(restored_accuracy - best_accuracy) < 0.01:
                result.restoration_verified = True
                accuracy = restored_accuracy
                logger.info(
                    f"Restoration verified: accuracy={restored_accuracy:.3f} "
                    f"(expected {best_accuracy:.3f})"
                )
            else:
                result.restoration_verified = False
                logger.warning(
                    f"Restoration mismatch: got {restored_accuracy:.3f}, "
                    f"expected {best_accuracy:.3f}"
                )

        # Final results
        result.final_accuracy = accuracy
        result.best_accuracy = best_accuracy
        result.best_accuracy_step = best_accuracy_step
        result.total_steps = len(result.accuracy_trajectory) - 1
        result.end_time = time.time()

        # Assess success level
        result.success_level = self._assess_success_level(result)

        return result

    def _assess_success_level(self, result: MetacognitiveResult) -> int:
        """
        Assess which success level was achieved.

        Level 0: Baseline - ran until external stop
        Level 1: Recognized plateau and stopped
        Level 2: Recognized plateau, switched, found improvement
        Level 3: Recognized "no strategy working" and stopped gracefully
        """
        # First check if any strategy switch led to improvement
        # (This should be checked before Level 3 since finding improvement is positive)
        if result.switched_strategy and len(result.strategy_attempts) > 1:
            first_strategy_best = result.strategy_attempts[0].best_accuracy
            for attempt in result.strategy_attempts[1:]:
                if attempt.best_accuracy > first_strategy_best + 0.001:  # Small epsilon
                    result.found_improvement_after_switch = True
                    # Level 2 if we found improvement, even if we eventually stopped
                    return 2

        # Level 3: Exhausted all strategies and recognized it
        if result.recognized_no_progress and result.stopped_self:
            return 3

        # Level 1: Stopped self (plateau detection worked)
        if result.stopped_self:
            return 1

        # Level 0: Ran until external stop
        return 0


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def run_metacognitive_test(
    dimensions: int = 2048,
    device: str | None = None,
    max_steps: int = 100,
    agent_type: str = "rule_based",
    plateau_threshold: int = 10,
    scenario: str = "default",
) -> MetacognitiveResult:
    """
    Run a metacognitive loop test.

    Args:
        dimensions: Vector dimensions
        device: Compute device
        max_steps: Maximum steps before external stop
        agent_type: "rule_based" or "baseline"
        plateau_threshold: Steps without improvement to trigger plateau
        scenario: "default" or "improvable" - controls test case generation

    Returns:
        MetacognitiveResult with full trajectory and analysis
    """
    harness = create_harness(dimensions=dimensions, device=device)

    if agent_type == "baseline":
        agent = BaselineAgent()
    else:
        agent = RuleBasedAgent(
            plateau_threshold=plateau_threshold,
            min_steps_per_strategy=5,
        )

    loop = MetacognitiveLoop(
        harness=harness,
        agent=agent,
        max_steps=max_steps,
        evaluator_scenario=scenario,
    )

    return loop.run()


def compare_agent_behaviors(
    dimensions: int = 2048,
    device: str | None = None,
    max_steps: int = 100,
) -> dict:
    """
    Compare baseline vs rule-based agent behaviors.

    Returns dict with both results for comparison.
    """
    logger.info("=" * 60)
    logger.info("RUNNING BASELINE AGENT (no metacognition)")
    logger.info("=" * 60)

    baseline_result = run_metacognitive_test(
        dimensions=dimensions,
        device=device,
        max_steps=max_steps,
        agent_type="baseline",
    )

    logger.info("\n" + "=" * 60)
    logger.info("RUNNING RULE-BASED AGENT (with metacognition)")
    logger.info("=" * 60)

    metacog_result = run_metacognitive_test(
        dimensions=dimensions,
        device=device,
        max_steps=max_steps,
        agent_type="rule_based",
    )

    return {
        "baseline": baseline_result,
        "metacognitive": metacog_result,
        "comparison": {
            "baseline_final_accuracy": baseline_result.final_accuracy,
            "metacog_final_accuracy": metacog_result.final_accuracy,
            "baseline_steps": baseline_result.total_steps,
            "metacog_steps": metacog_result.total_steps,
            "metacog_stopped_self": metacog_result.stopped_self,
            "metacog_switched_strategy": metacog_result.switched_strategy,
            "metacog_success_level": metacog_result.success_level,
        }
    }


def print_metacognitive_report(result: MetacognitiveResult) -> None:
    """Print a human-readable report of metacognitive loop results."""
    print("\n" + "=" * 70)
    print("METACOGNITIVE LOOP REPORT")
    print("=" * 70)

    print("\nConfiguration:")
    print(f"  Max steps: {result.max_steps}")
    print(f"  Ground truth: {result.ground_truth_source}")

    print("\nOutcomes:")
    print(f"  Total steps run: {result.total_steps}")
    print(f"  Final accuracy: {result.final_accuracy:.3f}")
    print(f"  Best accuracy: {result.best_accuracy:.3f} (at step {result.best_accuracy_step})")

    # Show restoration info if it happened
    if result.restored_to_best:
        print("\n  [RESTORED TO BEST STATE]")
        print(f"  Pre-restoration accuracy:  {result.pre_restoration_accuracy:.3f}")
        print(f"  Post-restoration accuracy: {result.post_restoration_accuracy:.3f}")
        print(f"  Restoration verified: {result.restoration_verified}")

    print("\nAgent Behavior:")
    print(f"  Stopped self: {result.stopped_self}")
    print(f"  Switched strategy: {result.switched_strategy}")
    print(f"  Found improvement after switch: {result.found_improvement_after_switch}")
    print(f"  Recognized no progress: {result.recognized_no_progress}")

    print(f"\n  SUCCESS LEVEL: {result.success_level}")
    level_descriptions = {
        0: "Baseline - ran until external stop (no metacognition)",
        1: "Recognized plateau and stopped self",
        2: "Recognized plateau, switched strategy, found improvement",
        3: "Recognized 'no strategy working' and stopped gracefully",
    }
    print(f"  ({level_descriptions[result.success_level]})")

    print("\nStrategy Attempts:")
    for i, attempt in enumerate(result.strategy_attempts):
        print(f"  {i+1}. {attempt.strategy.value}")
        print(f"     Steps {attempt.start_step}-{attempt.end_step} ({attempt.steps_run} steps)")
        print(f"     Accuracy: {attempt.start_accuracy:.3f} -> {attempt.final_accuracy:.3f} (best: {attempt.best_accuracy:.3f})")
        print(f"     Improved: {attempt.improved}")

    print("\nKey Decision Points:")
    key_decisions = [
        d for d in result.decision_log
        if d.decision != Decision.CONTINUE
    ]
    for d in key_decisions[:5]:  # Show first 5
        print(f"  Step {d.step}: {d.decision.value}")
        if d.new_strategy:
            print(f"    -> New strategy: {d.new_strategy.value}")
        print(f"    Reasoning: {d.reasoning[:100]}...")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    print("Running metacognitive loop comparison...")
    results = compare_agent_behaviors(
        dimensions=2048,
        max_steps=50,
    )

    print("\n" + "=" * 70)
    print("BASELINE AGENT RESULTS")
    print_metacognitive_report(results["baseline"])

    print("\n" + "=" * 70)
    print("METACOGNITIVE AGENT RESULTS")
    print_metacognitive_report(results["metacognitive"])

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    comp = results["comparison"]
    print(f"Baseline: {comp['baseline_steps']} steps, final accuracy {comp['baseline_final_accuracy']:.3f}")
    print(f"Metacog:  {comp['metacog_steps']} steps, final accuracy {comp['metacog_final_accuracy']:.3f}")
    print(f"Metacog stopped self: {comp['metacog_stopped_self']}")
    print(f"Metacog switched strategy: {comp['metacog_switched_strategy']}")
    print(f"Metacog success level: {comp['metacog_success_level']}")
