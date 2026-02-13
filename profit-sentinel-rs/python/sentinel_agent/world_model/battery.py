"""
Behavioral Battery for VSA World Model
========================================

Adapted from the meta-diagonal research (Sessions 1-8, 18c).

The battery asks GEOMETRIC questions about how the algebra is behaving.
Applied to the world model, it measures the HEALTH of state transitions:

- Are bindings recovering cleanly? (chain integrity)
- Is bundling maintaining capacity? (state not saturated)
- Are transitions compositional? (operations distribute)
- Is the resonator converging? (dynamics are stable)

These measurements feed proprioception: the model knows when
its own operations are degrading.

Key lesson from Session 8: "The algebra processes geometry, not meaning."
The battery measures what operations DO to structure, not what they mean.
That's exactly right for proprioception — we want to know if the
machinery is working, not if the content is true.

Validated measurements:
- 22 behavioral measurements across 7+ VSA engines (Sessions 1-8)
- Population-level separation d=0.92-1.10 (Session 8)
- Asymmetry detection is a genuine geometric property (Session 8)
- 490+ passing tests across the battery framework

Author: Joseph + Claude
Date: 2026-02-08
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .core import safe_normalize


@dataclass
class BatteryResult:
    """Result of running the behavioral battery."""

    measurements: np.ndarray  # Raw measurement vector
    measurement_names: list[str]  # Names for each measurement
    health_scores: dict[str, float]  # Derived health indicators
    anomalies: list[str]  # Detected issues
    timestamp: float = 0.0


def get_measurement_names() -> list[str]:
    """Names for all 22 measurements in the behavioral battery."""
    return [
        # Group 1: Binding geometry (4)
        "bind_preserves_a",  # sim(bind(a,b), a)
        "bind_preserves_b",  # sim(bind(a,b), b)
        "shared_operand_similarity",  # sim(bind(a,b), bind(a,c))
        "unrelated_binding_orthog",  # sim(bind(a,b), bind(c,d))
        # Group 2: Recovery fidelity (4)
        "single_recovery",  # sim(unbind(bind(a,b),b), a)
        "nested_recovery",  # sim(unbind(bind(bind(a,b),c),c), bind(a,b))
        "second_operand_recovery",  # sim(unbind(bind(a,b),a), b)
        "chain_degradation",  # recovery after 4 nested bindings
        # Group 3: Bundling geometry (4)
        "bundle_2_retrieval",  # sim(bundle([a,b]), a)
        "bundle_3_retrieval",  # sim(bundle([a,b,c]), a)
        "bundle_5_retrieval",  # sim(bundle(5 vecs), component)
        "bundle_10_retrieval",  # sim(bundle(10 vecs), component)
        # Group 4: Algebraic properties (2)
        "commutativity",  # sim(bind(a,b), bind(b,a))
        "associativity",  # sim(bind(bind(a,b),c), bind(a,bind(b,c)))
        # Group 5: Distribution (2)
        "bind_distributes_bundle",  # sim(bind(a,bundle(b,c)), bundle(bind(a,b),bind(a,c)))
        "reverse_distributivity",  # sim(bundle(bind(a,x),bind(b,x)), bind(bundle(a,b),x))
        # Group 6: Scaling (2)
        "orthogonality_concentration",  # std of random pair similarities
        "bundle_capacity",  # max k where retrieval > threshold
        # Group 7: Transition-specific (4) — NEW for world model
        "transition_recovery",  # can we recover the transition from before/after?
        "transition_composition",  # do sequential transitions compose?
        "prediction_self_consistency",  # predict → observe → predict again: stable?
        "resonator_convergence_rate",  # what fraction of transitions converge?
    ]


class WorldModelBattery:
    """
    Behavioral battery adapted for world model diagnostics.

    Runs the full 22-measurement battery on the model's algebra,
    plus 4 transition-specific measurements.

    Returns a health profile that feeds proprioception.
    """

    def __init__(self, algebra, n_trials: int = 10, seed: int = 42):
        """
        Args:
            algebra: PhasorAlgebra instance from the world model
            n_trials: Number of random trials per measurement
            seed: Random seed for reproducibility
        """
        self.algebra = algebra
        self.n_trials = n_trials
        self.rng = np.random.default_rng(seed)

        # Baseline measurements (established during first run)
        MAX_BATTERY_HISTORY = 200
        self.baseline: np.ndarray | None = None
        self.history: deque[np.ndarray] = deque(maxlen=MAX_BATTERY_HISTORY)

    def run_structural_battery(self) -> np.ndarray:
        """
        Run the full 18-measurement structural battery.

        Uses only random vectors — measures algebraic properties,
        not content. This characterizes how the algebra itself
        is functioning.
        """
        measurements = []
        a = self.algebra

        for trial in range(self.n_trials):
            trial_measurements = []

            # Generate fresh random vectors for each trial
            v_a = a.random_vector()
            v_b = a.random_vector()
            v_c = a.random_vector()
            v_d = a.random_vector()
            v_x = a.random_vector()

            # === GROUP 1: Binding geometry ===
            bound_ab = a.bind(v_a, v_b)
            bound_ac = a.bind(v_a, v_c)
            bound_cd = a.bind(v_c, v_d)

            trial_measurements.append(a.similarity(bound_ab, v_a))
            trial_measurements.append(a.similarity(bound_ab, v_b))
            trial_measurements.append(a.similarity(bound_ab, bound_ac))
            trial_measurements.append(a.similarity(bound_ab, bound_cd))

            # === GROUP 2: Recovery fidelity ===
            recovered_a = a.unbind(bound_ab, v_b)
            trial_measurements.append(a.similarity(recovered_a, v_a))

            bound_abc = a.bind(bound_ab, v_c)
            recovered_ab = a.unbind(bound_abc, v_c)
            trial_measurements.append(a.similarity(recovered_ab, bound_ab))

            recovered_b = a.unbind(bound_ab, v_a)
            trial_measurements.append(a.similarity(recovered_b, v_b))

            # Chain degradation: 4 nested bindings then recover
            chain = v_a
            keys = []
            for _ in range(4):
                key = a.random_vector()
                keys.append(key)
                chain = a.bind(chain, key)
            # Recover by unbinding in reverse
            recovered = chain
            for key in reversed(keys):
                recovered = a.unbind(recovered, key)
            trial_measurements.append(a.similarity(recovered, v_a))

            # === GROUP 3: Bundling geometry ===
            vecs_2 = [v_a, v_b]
            vecs_3 = [v_a, v_b, v_c]
            vecs_5 = [a.random_vector() for _ in range(4)] + [v_a]
            vecs_10 = [a.random_vector() for _ in range(9)] + [v_a]

            trial_measurements.append(a.similarity(a.bundle(vecs_2), v_a))
            trial_measurements.append(a.similarity(a.bundle(vecs_3), v_a))
            trial_measurements.append(a.similarity(a.bundle(vecs_5), v_a))
            trial_measurements.append(a.similarity(a.bundle(vecs_10), v_a))

            # === GROUP 4: Algebraic properties ===
            bound_ba = a.bind(v_b, v_a)
            trial_measurements.append(a.similarity(bound_ab, bound_ba))

            left = a.bind(a.bind(v_a, v_b), v_c)
            right = a.bind(v_a, a.bind(v_b, v_c))
            trial_measurements.append(a.similarity(left, right))

            # === GROUP 5: Distribution ===
            bundle_bc = a.bundle([v_b, v_c])
            bind_a_bundle = a.bind(v_a, bundle_bc)
            bundle_binds = a.bundle([a.bind(v_a, v_b), a.bind(v_a, v_c)])
            trial_measurements.append(a.similarity(bind_a_bundle, bundle_binds))

            bind_ax = a.bind(v_a, v_x)
            bind_bx = a.bind(v_b, v_x)
            bundle_ab = a.bundle([v_a, v_b])
            bind_bundle_x = a.bind(bundle_ab, v_x)
            trial_measurements.append(
                a.similarity(a.bundle([bind_ax, bind_bx]), bind_bundle_x)
            )

            # === GROUP 6: Scaling ===
            # Orthogonality concentration
            pair_sims = []
            for _ in range(50):
                r1 = a.random_vector()
                r2 = a.random_vector()
                pair_sims.append(a.similarity(r1, r2))
            trial_measurements.append(float(np.std(pair_sims)))

            # Bundle capacity: find max k where retrieval > 0.1
            target = a.random_vector()
            max_k = 1
            for k in [2, 5, 10, 20, 50, 100]:
                others = [a.random_vector() for _ in range(k - 1)]
                bundled = a.bundle([target] + others)
                if a.similarity(bundled, target) > 0.1:
                    max_k = k
            trial_measurements.append(float(max_k))

            measurements.append(trial_measurements)

        # Average over trials
        return np.mean(measurements, axis=0)

    def run_transition_battery(
        self,
        transition_model,
        recent_states: list[np.ndarray],
        recent_observations: list[np.ndarray],
    ) -> np.ndarray:
        """
        Run transition-specific measurements.

        These measure how well the DYNAMICS MODEL is functioning:
        - Can transitions be recovered from before/after states?
        - Do sequential transitions compose correctly?
        - Are predictions self-consistent?
        - Is the resonator converging?
        """
        a = self.algebra
        measurements = []

        # 1. Transition recovery
        # Given state_before and state_after, can we extract the transition?
        recovery_scores = []
        for i in range(min(len(recent_states) - 1, self.n_trials)):
            before = recent_states[i]
            after = recent_states[i + 1]
            # Extract transition
            transition = a.bind(after, np.conj(before))
            # Re-apply transition to before
            reconstructed = a.bind(before, transition)
            recovery_scores.append(a.similarity(reconstructed, after))
        measurements.append(float(np.mean(recovery_scores)) if recovery_scores else 0.0)

        # 2. Transition composition
        # Does T1 followed by T2 equal T_composed?
        composition_scores = []
        for i in range(min(len(recent_states) - 2, self.n_trials)):
            s0, s1, s2 = recent_states[i], recent_states[i + 1], recent_states[i + 2]
            t1 = a.bind(s1, np.conj(s0))
            t2 = a.bind(s2, np.conj(s1))
            t_composed = a.bind(s2, np.conj(s0))
            t_sequential = a.bind(t1, t2)
            composition_scores.append(a.similarity(t_sequential, t_composed))
        measurements.append(
            float(np.mean(composition_scores)) if composition_scores else 0.0
        )

        # 3. Prediction self-consistency
        # Predict from current state, then predict again from prediction
        consistency_scores = []
        if recent_states and recent_observations:
            for i in range(min(len(recent_states), self.n_trials)):
                state = recent_states[i]
                obs = recent_observations[min(i, len(recent_observations) - 1)]
                pred1, _ = transition_model.predict_next_state(state, obs)
                pred2, _ = transition_model.predict_next_state(pred1, obs)
                # If consistent, pred2 should be a smooth continuation
                # not wildly different from pred1
                consistency_scores.append(a.similarity(pred1, pred2))
        measurements.append(
            float(np.mean(consistency_scores)) if consistency_scores else 0.0
        )

        # 4. Resonator convergence rate
        total = 0
        for exp in list(transition_model.experience_buffer)[-100:]:
            total += 1
        # Use the monitor's tracking instead
        if (
            hasattr(transition_model, "prediction_errors")
            and transition_model.prediction_errors
        ):
            # Proxy: low error = good convergence
            recent_errors = list(transition_model.prediction_errors)[-100:]
            measurements.append(
                float(np.mean([1.0 if e < 0.5 else 0.0 for e in recent_errors]))
            )
        else:
            measurements.append(0.0)

        return np.array(measurements)

    def run_full_battery(
        self,
        transition_model=None,
        recent_states: list[np.ndarray] = None,
        recent_observations: list[np.ndarray] = None,
    ) -> BatteryResult:
        """
        Run the complete battery: structural + transition measurements.

        Returns a BatteryResult with health scores and anomaly detection.
        """
        import time

        # Structural measurements (18)
        structural = self.run_structural_battery()

        # Transition measurements (4) — only if model is running
        if (
            transition_model is not None
            and recent_states is not None
            and len(recent_states) > 2
        ):
            transition = self.run_transition_battery(
                transition_model, recent_states, recent_observations or []
            )
        else:
            transition = np.zeros(4)

        # Combine
        all_measurements = np.concatenate([structural, transition])

        # Store in history
        self.history.append(all_measurements)

        # Establish baseline on first run
        if self.baseline is None:
            self.baseline = all_measurements.copy()

        # Compute health scores
        health_scores = self._compute_health(all_measurements)

        # Detect anomalies (deviations from baseline)
        anomalies = self._detect_anomalies(all_measurements)

        return BatteryResult(
            measurements=all_measurements,
            measurement_names=get_measurement_names(),
            health_scores=health_scores,
            anomalies=anomalies,
            timestamp=time.time(),
        )

    def _compute_health(self, measurements: np.ndarray) -> dict[str, float]:
        """
        Derive interpretable health scores from raw measurements.

        These are the numbers that matter for proprioception.
        """
        names = get_measurement_names()
        m = dict(zip(names, measurements))

        return {
            # Binding health: are bindings working correctly?
            "binding_health": float(
                np.mean(
                    [
                        m.get("single_recovery", 0),
                        m.get("nested_recovery", 0),
                        m.get("second_operand_recovery", 0),
                    ]
                )
            ),
            # Chain health: do long chains degrade?
            "chain_health": float(m.get("chain_degradation", 0)),
            # Bundling capacity: can the state hold multiple items?
            "bundling_health": float(
                np.mean(
                    [
                        m.get("bundle_2_retrieval", 0),
                        m.get("bundle_5_retrieval", 0),
                    ]
                )
            ),
            # Algebraic integrity: do the algebraic laws hold?
            "algebraic_health": float(
                np.mean(
                    [
                        m.get("commutativity", 0),
                        m.get("associativity", 0),
                        m.get("bind_distributes_bundle", 0),
                    ]
                )
            ),
            # Transition health: are dynamics working?
            "transition_health": float(
                np.mean(
                    [
                        m.get("transition_recovery", 0),
                        m.get("transition_composition", 0),
                        m.get("prediction_self_consistency", 0),
                    ]
                )
            ),
            # Convergence health: is the resonator working?
            "convergence_health": float(m.get("resonator_convergence_rate", 0)),
        }

    def _detect_anomalies(self, measurements: np.ndarray) -> list[str]:
        """
        Detect anomalies by comparing to baseline and history.

        Uses the adversarial probe insight: silent failures are
        when confidence stays high but accuracy degrades.
        """
        anomalies = []

        if self.baseline is None:
            return anomalies

        # Deviation from baseline
        deviation = np.abs(measurements - self.baseline)

        # Check each measurement group
        names = get_measurement_names()

        # Recovery degradation
        recovery_idx = [
            names.index(n)
            for n in ["single_recovery", "nested_recovery", "chain_degradation"]
            if n in names
        ]
        if recovery_idx:
            recovery_deviation = np.mean(deviation[recovery_idx])
            if recovery_deviation > 0.1:
                anomalies.append(
                    f"RECOVERY_DEGRADATION: binding recovery dropped "
                    f"by {recovery_deviation:.3f} from baseline"
                )

        # Bundling capacity loss
        bundle_idx = [
            names.index(n)
            for n in ["bundle_5_retrieval", "bundle_10_retrieval"]
            if n in names
        ]
        if bundle_idx:
            bundle_current = np.mean(measurements[bundle_idx])
            if bundle_current < 0.1:
                anomalies.append(
                    f"BUNDLE_SATURATION: bundling retrieval at "
                    f"{bundle_current:.3f}, state may be saturated"
                )

        # Algebraic integrity violation
        alg_idx = [
            names.index(n) for n in ["commutativity", "associativity"] if n in names
        ]
        if alg_idx:
            alg_deviation = np.mean(deviation[alg_idx])
            if alg_deviation > 0.05:
                anomalies.append(
                    f"ALGEBRAIC_INTEGRITY: algebraic properties shifted "
                    f"by {alg_deviation:.3f} — numerical issues?"
                )

        # Trend detection over history
        if len(self.history) >= 5:
            recent = np.array(list(self.history)[-5:])
            for i, name in enumerate(names):
                if i < recent.shape[1]:
                    trend = np.polyfit(range(5), recent[:, i], 1)[0]
                    if trend < -0.02 and "recovery" in name.lower():
                        anomalies.append(
                            f"DECLINING_{name.upper()}: " f"trend={trend:.4f}/step"
                        )

        return anomalies


class WarmupPhase:
    """
    Warmup phase that uses the battery to bootstrap transition primitives.

    Instead of starting with random primitives (which caused the collapse
    to a single attractor in the first test), observe transitions for N
    steps, cluster them, and create primitives from actual patterns.

    This is the structure_learning.py approach applied to transitions.
    """

    def __init__(self, algebra, n_warmup_steps: int = 50):
        self.algebra = algebra
        self.n_warmup_steps = n_warmup_steps
        self.observed_transitions: list[np.ndarray] = []
        self.transition_contexts: list[np.ndarray] = []

    def collect_transition(
        self, state_before: np.ndarray, observation: np.ndarray, state_after: np.ndarray
    ):
        """Record an observed transition during warmup."""
        transition = self.algebra.bind(state_after, np.conj(state_before))
        context = self.algebra.bind(state_before, observation)
        self.observed_transitions.append(transition)
        self.transition_contexts.append(context)

    def is_complete(self) -> bool:
        """Has warmup collected enough data?"""
        return len(self.observed_transitions) >= self.n_warmup_steps

    def derive_primitives(self, max_primitives: int = 12) -> dict[str, np.ndarray]:
        """
        Cluster observed transitions and create primitives from clusters.

        Uses the geometry of the transitions themselves to find natural
        groups — the same principle as Level 5 blind self-grounding,
        but applied to state transitions instead of inventory data.
        """
        if not self.observed_transitions:
            return {}

        transitions = np.array(self.observed_transitions)
        n = len(transitions)

        # Compute pairwise similarity matrix
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                sim = self.algebra.similarity(transitions[i], transitions[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        # Simple greedy clustering:
        # Pick the transition farthest from all existing centroids
        # This avoids the random initialization problem
        centroids = [transitions[0]]
        centroid_indices = [0]

        for _ in range(max_primitives - 1):
            # Find the transition most dissimilar from all centroids
            min_max_sims = []
            for i in range(n):
                if i in centroid_indices:
                    min_max_sims.append(float("inf"))
                    continue
                max_sim_to_centroid = max(sim_matrix[i, ci] for ci in centroid_indices)
                min_max_sims.append(max_sim_to_centroid)

            # The most dissimilar transition becomes a new centroid
            new_idx = np.argmin(min_max_sims)
            if min_max_sims[new_idx] > 0.9:
                # Remaining transitions too similar to existing centroids
                break

            centroids.append(transitions[new_idx])
            centroid_indices.append(new_idx)

        # Name and return primitives
        primitives = {}
        for i, centroid in enumerate(centroids):
            # Count how many transitions cluster with this centroid
            cluster_size = sum(
                1
                for j in range(n)
                if np.argmax(
                    [self.algebra.similarity(transitions[j], c) for c in centroids]
                )
                == i
            )
            name = f"T_{i:02d}_n{cluster_size}"
            primitives[name] = safe_normalize(centroid)

        return primitives
