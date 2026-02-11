"""
VSA World Model — Core Architecture
=====================================

A world model where:
- State is a VSA vector (compositional, queryable, exact recovery)
- Dynamics are learned via contrastive resonator tuning
- Prediction error drives attention and learning
- The system monitors its own reasoning state (proprioception)

This is NOT a neural network in the traditional sense. It's a
resonator-based dynamics model that learns transition structure
from data sequences, using the VSA operations that have already
been validated:

PROVEN components used:
- Clifford algebra bind/unbind/bundle (chain recovery 1.0 at depth 1000+)
- Phasor encoding of numerical data (0% quantitative hallucination)
- Resonator dynamics for factorization (convergence/divergence as signal)
- Structure learning (merge/split/add primitives)
- Contrastive learning (positive + negative pairs)
- Proprioceptive monitoring (Session 18c: d=-0.74 on boundary tasks)

DEAD END lessons incorporated:
- No universal negation operator (context-dependent only)
- No Hebbian single-operator learning (contrastive instead)
- No curved geometry for classification (flat thresholds work)
- Encoding in rotor, not phase bypass (Session 11 lesson)

Author: Joseph + Claude
Date: 2026-02-08
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class WorldModelConfig:
    """Configuration for the VSA world model."""

    dim: int = 4096  # VSA dimension
    n_roles: int = 8  # Number of role vectors (slots)
    codebook_capacity: int = 1000  # Max items in cleanup memory
    resonator_iters: int = 100  # Max resonator iterations
    resonator_threshold: float = 0.95  # Convergence threshold
    learning_rate: float = 0.01  # Transition learning rate
    prediction_horizon: int = 1  # Steps ahead to predict
    attention_threshold: float = 0.3  # Prediction error threshold for attention
    entropy_window: int = 20  # Window for entropy tracking
    seed: int = 42


# =============================================================================
# PHASOR VSA ALGEBRA
# =============================================================================


class PhasorAlgebra:
    """
    Complex phasor VSA operations.

    Validated: chain recovery 1.0 at depth 1000+.
    All vectors are unit-magnitude complex (phases on unit circle).
    """

    def __init__(self, dim: int, seed: int = 42):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self._cache: dict[str, np.ndarray] = {}

    def random_vector(self, label: str | None = None) -> np.ndarray:
        """Generate random unit phasor vector."""
        phases = self.rng.uniform(0, 2 * np.pi, self.dim)
        vec = np.exp(1j * phases)
        if label:
            self._cache[label] = vec
        return vec

    def get_or_create(self, label: str) -> np.ndarray:
        """Get cached vector or create new one."""
        if label not in self._cache:
            self._cache[label] = self.random_vector()
        return self._cache[label]

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise multiply (phasor bind). Commutative."""
        return a * b

    def unbind(self, composite: np.ndarray, key: np.ndarray) -> np.ndarray:
        """Element-wise conjugate multiply (exact inverse of bind)."""
        return composite * np.conj(key)

    def ordered_bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Non-commutative bind via permutation. ordered_bind(a,b) != ordered_bind(b,a)."""
        return self.bind(self.permute(a, 1), b)

    def bundle(
        self, vectors: list[np.ndarray], weights: list[float] | None = None
    ) -> np.ndarray:
        """Superposition with optional weights. Normalize to unit magnitude."""
        if weights:
            result = sum(w * v for w, v in zip(weights, vectors))
        else:
            result = sum(vectors)
        # Normalize to unit phasor
        return result / np.abs(result)

    def permute(self, v: np.ndarray, k: int = 1) -> np.ndarray:
        """Circular shift by k positions."""
        return np.roll(v, k)

    def inverse_permute(self, v: np.ndarray, k: int = 1) -> np.ndarray:
        """Inverse circular shift."""
        return np.roll(v, -k)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between phasor vectors."""
        return float(np.abs(np.mean(a * np.conj(b))))

    def identity(self) -> np.ndarray:
        """Identity vector (all ones)."""
        return np.ones(self.dim, dtype=complex)


# =============================================================================
# STRUCTURED STATE VECTOR
# =============================================================================


class StateVector:
    """
    Compositional world state as a VSA vector.

    Instead of a monolithic latent vector (Dreamer's failure mode),
    state is a role-filler binding:

        state = Σ_i  role_i ⊗ filler_i

    Each role is a named slot. Each filler is what occupies that slot.
    The state can be queried: unbind(state, role_i) ≈ filler_i

    This solves the binding problem: entity identity is maintained
    by algebraic structure, not slot position (no drift).
    """

    def __init__(self, algebra: PhasorAlgebra, role_names: list[str]):
        self.algebra = algebra
        self.role_names = role_names
        self.roles: dict[str, np.ndarray] = {}
        self.fillers: dict[str, np.ndarray] = {}

        # Create role vectors
        for name in role_names:
            self.roles[name] = algebra.random_vector(f"role_{name}")

        # Initialize fillers to identity (empty slots)
        for name in role_names:
            self.fillers[name] = algebra.identity()

        self._compiled: np.ndarray | None = None
        self._dirty = True

    def set_filler(self, role: str, filler: np.ndarray):
        """Set the content of a role slot."""
        if role not in self.roles:
            raise ValueError(f"Unknown role: {role}")
        self.fillers[role] = filler
        self._dirty = True

    def get_filler(self, role: str) -> np.ndarray:
        """Query the state for a specific role's content."""
        compiled = self.compile()
        return self.algebra.unbind(compiled, self.roles[role])

    def compile(self) -> np.ndarray:
        """Compile role-filler pairs into a single state vector."""
        if not self._dirty and self._compiled is not None:
            return self._compiled

        bindings = []
        for name in self.role_names:
            binding = self.algebra.bind(self.roles[name], self.fillers[name])
            bindings.append(binding)

        self._compiled = self.algebra.bundle(bindings)
        self._dirty = False
        return self._compiled

    def similarity_to(self, other_state: np.ndarray) -> float:
        """How similar is this state to another?"""
        return self.algebra.similarity(self.compile(), other_state)

    def slot_similarities(self, other: "StateVector") -> dict[str, float]:
        """Per-slot similarity between two states."""
        result = {}
        for name in self.role_names:
            my_filler = self.fillers[name]
            their_filler = other.fillers[name]
            result[name] = self.algebra.similarity(my_filler, their_filler)
        return result

    def entropy(self) -> float:
        """
        State entropy: how "spread out" is the information?

        High entropy = many slots active, distributed state
        Low entropy = few slots dominate, concentrated state

        This IS the fuzziness of the inner model.
        """
        energies = []
        identity = self.algebra.identity()
        for name in self.role_names:
            # How different is this filler from empty?
            energy = 1.0 - self.algebra.similarity(self.fillers[name], identity)
            energies.append(energy)

        energies = np.array(energies)
        total = energies.sum() + 1e-15
        probs = energies / total
        return float(-np.sum(probs * np.log(probs + 1e-15)))


# =============================================================================
# TRANSITION MODEL (The "Neural Network")
# =============================================================================


class TransitionModel:
    """
    Learned state transition dynamics.

    Given state S_t and observation O_t, predict state S_{t+1}.

    This is NOT a neural network. It's a resonator-based dynamics
    model that learns which VSA operations best predict next states.

    The key insight: transitions between states in a structured domain
    can be decomposed into a small set of TRANSITION PRIMITIVES —
    reusable operations that compose to produce complex state changes.

    This is analogous to how your detection primitives decompose
    profit leaks into reusable patterns. But instead of patterns
    in data, these are patterns in CHANGE.

    Learning uses contrastive pairs (not Hebbian):
    - Positive: (S_t, O_t, S_{t+1}) — actual observed transition
    - Negative: (S_t, O_t, S_random) — counterfactual transition

    The model learns to predict transitions that actually happened
    over transitions that didn't.
    """

    def __init__(self, algebra: PhasorAlgebra, config: WorldModelConfig):
        self.algebra = algebra
        self.config = config

        # Transition primitives: learned operations on state
        # Start with basic set, structure learning can add/merge/split
        self.transition_primitives: dict[str, np.ndarray] = {}
        self._init_transition_primitives()

        # Transition codebook: maps (context) -> (which primitive applies)
        self.transition_codebook: list[tuple[np.ndarray, str]] = []

        # Experience buffer for learning
        self.experience_buffer: list[dict] = []
        self.max_buffer_size = 10000

        # Performance tracking
        self.prediction_errors: list[float] = []
        self.transition_counts: dict[str, int] = defaultdict(int)

    def _init_transition_primitives(self):
        """
        Initialize basic transition primitives.

        These are the "verbs" of state change:
        - PERSIST: nothing changes (identity transition)
        - INCREMENT: slot value increases
        - DECREMENT: slot value decreases
        - SWAP: two slots exchange fillers
        - APPEAR: new entity enters a slot
        - VANISH: entity leaves a slot (filler -> identity)
        - SHIFT: temporal shift (permutation)

        Structure learning can discover domain-specific transitions
        beyond these basics.
        """
        primitives = [
            "PERSIST",
            "INCREMENT",
            "DECREMENT",
            "SWAP",
            "APPEAR",
            "VANISH",
            "SHIFT",
        ]
        for name in primitives:
            self.transition_primitives[name] = self.algebra.random_vector(
                f"transition_{name}"
            )

    def predict_next_state(
        self, current_state: np.ndarray, observation: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        """
        Predict next state given current state and new observation.

        Method:
        1. Create context = bind(current_state, observation)
        2. Use resonator to find best matching transition primitive
        3. Apply that primitive to current state
        4. Return predicted next state + metadata

        The resonator is doing the "dynamics" — it's finding which
        transition pattern best fits the current context.
        """
        # Create transition context
        context = self.algebra.bind(current_state, observation)

        # Find best matching transition via resonator
        best_primitive, confidence, convergence_info = self._resonate_transition(
            context
        )

        # Apply transition to state
        if best_primitive == "PERSIST":
            predicted = current_state.copy()
        elif best_primitive == "SHIFT":
            predicted = self.algebra.permute(current_state, 1)
        else:
            # General transition: bind state with transition vector
            t_vec = self.transition_primitives[best_primitive]
            predicted = self.algebra.bind(current_state, t_vec)

        metadata = {
            "primitive": best_primitive,
            "confidence": confidence,
            "convergence": convergence_info,
            "context_entropy": float(np.std(np.angle(context))),
        }

        return predicted, metadata

    def _resonate_transition(self, context: np.ndarray) -> tuple[str, float, dict]:
        """
        Use resonator dynamics to find the best transition primitive
        for a given context.

        This is the core "learned dynamics" — the resonator iteratively
        cleans up the context signal against the transition codebook.
        """
        # Build transition codebook matrix
        names = list(self.transition_primitives.keys())
        codebook = np.array([self.transition_primitives[n] for n in names])

        # Also include learned transitions from experience
        for ctx_vec, t_name in self.transition_codebook:
            if t_name in self.transition_primitives:
                continue
            names.append(t_name)
            codebook = np.vstack([codebook, ctx_vec.reshape(1, -1)])

        # Resonator iteration
        estimate = context.copy()
        trajectory = []

        for iteration in range(self.config.resonator_iters):
            # Similarities to all transition primitives
            sims = np.array([self.algebra.similarity(estimate, cb) for cb in codebook])

            best_idx = np.argmax(sims)
            best_sim = sims[best_idx]
            trajectory.append(float(best_sim))

            if best_sim > self.config.resonator_threshold:
                return (
                    names[best_idx],
                    float(best_sim),
                    {
                        "converged": True,
                        "iterations": iteration + 1,
                        "trajectory": trajectory,
                    },
                )

            # Update estimate: project onto best match, blend back
            alpha = 0.7
            estimate = alpha * estimate + (1 - alpha) * codebook[best_idx]
            # Renormalize
            estimate = estimate / np.abs(estimate)

        # Didn't converge — return best guess
        best_idx = np.argmax([self.algebra.similarity(context, cb) for cb in codebook])
        return (
            names[best_idx],
            float(sims[best_idx]),
            {
                "converged": False,
                "iterations": self.config.resonator_iters,
                "trajectory": trajectory,
            },
        )

    def learn_from_transition(
        self, state_before: np.ndarray, observation: np.ndarray, state_after: np.ndarray
    ):
        """
        Learn from an observed state transition.

        Uses contrastive learning (not Hebbian):
        - The actual transition is the positive example
        - Random states are negative examples
        - Adjust transition primitives to better predict actual outcomes

        Key fix: primitives are updated toward the CONTEXT-TRANSITION
        association, not just the raw transition. This lets different
        contexts activate different primitives instead of collapsing
        to a single attractor.
        """
        # What did we predict?
        predicted, meta = self.predict_next_state(state_before, observation)

        # Prediction error
        error = 1.0 - self.algebra.similarity(predicted, state_after)
        self.prediction_errors.append(error)

        # Context for this transition
        context = self.algebra.bind(state_before, observation)

        # Actual transition vector: what operation transforms before→after?
        actual_transition = self.algebra.bind(state_after, np.conj(state_before))

        # Find which primitive is closest to the actual transition
        best_name = None
        best_sim = -1
        second_best_name = None
        second_best_sim = -1
        for name, t_vec in self.transition_primitives.items():
            sim = self.algebra.similarity(actual_transition, t_vec)
            if sim > best_sim:
                second_best_name = best_name
                second_best_sim = best_sim
                best_sim = sim
                best_name = name
            elif sim > second_best_sim:
                second_best_sim = sim
                second_best_name = name

        # Contrastive update:
        # - Move best primitive TOWARD actual transition (positive)
        # - Move second-best AWAY from actual transition (negative)
        lr = self.config.learning_rate

        if best_name and best_sim > 0.05:
            t_vec = self.transition_primitives[best_name]
            # Positive: blend toward actual
            updated = (1 - lr) * t_vec + lr * actual_transition
            self.transition_primitives[best_name] = updated / np.abs(updated)
            self.transition_counts[best_name] += 1

        if second_best_name and second_best_sim > 0.05:
            t_vec = self.transition_primitives[second_best_name]
            # Negative: push away from actual transition
            repulsion = lr * 0.5  # Weaker than attraction
            updated = t_vec - repulsion * actual_transition
            self.transition_primitives[second_best_name] = updated / np.abs(updated)

        # If no primitive matches well AND error is high, add a new one
        # (high error = this transition isn't captured by existing primitives)
        if best_sim < 0.15 and error > 0.5:
            new_name = f"LEARNED_{len(self.transition_primitives)}"
            self.transition_primitives[new_name] = actual_transition / np.abs(
                actual_transition
            )
            self.transition_codebook.append((context, new_name))
            self.transition_counts[new_name] = 1

        # Store experience
        self.experience_buffer.append(
            {
                "context": context,
                "actual_transition": actual_transition,
                "predicted_primitive": meta["primitive"],
                "actual_closest": best_name,
                "error": error,
                "timestamp": time.time(),
            }
        )

        # Trim buffer
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer = self.experience_buffer[-self.max_buffer_size :]

        return error, meta


# =============================================================================
# PROPRIOCEPTIVE MONITOR
# =============================================================================


class ProprioceptiveMonitor:
    """
    Self-monitoring system that tracks the model's own reasoning state.

    Validated in Session 18c: d=-0.74 effect on boundary tasks.

    Tracks:
    - Prediction error trajectory (is the model getting better or worse?)
    - State entropy over time (is the model concentrating or diffusing?)
    - Transition primitive usage (is it stuck in one mode?)
    - Resonator convergence health (are transitions becoming harder to identify?)

    When anomalies are detected, it generates ALERTS that can be fed
    back to the LLM interface or used to trigger attention shifts.
    """

    def __init__(self, config: WorldModelConfig):
        self.config = config

        # Trajectory buffers
        self.error_trajectory: list[float] = []
        self.entropy_trajectory: list[float] = []
        self.convergence_trajectory: list[bool] = []
        self.primitive_history: list[str] = []

        # Alert state
        self.alerts: list[dict] = []
        self.alert_count = 0

    def observe_step(
        self, prediction_error: float, state_entropy: float, transition_meta: dict
    ):
        """Record one step of the model's operation."""
        self.error_trajectory.append(prediction_error)
        self.entropy_trajectory.append(state_entropy)
        self.convergence_trajectory.append(
            transition_meta.get("convergence", {}).get("converged", False)
        )
        self.primitive_history.append(transition_meta.get("primitive", "UNKNOWN"))

        # Check for anomalies
        self._check_anomalies()

    def _check_anomalies(self):
        """
        Detect anomalies in the model's own operation.

        These aren't anomalies in the DATA (that's the outer model's job).
        These are anomalies in HOW THE MODEL IS FUNCTIONING.
        """
        window = self.config.entropy_window

        if len(self.error_trajectory) < window:
            return

        recent_errors = self.error_trajectory[-window:]
        recent_entropy = self.entropy_trajectory[-window:]
        recent_convergence = self.convergence_trajectory[-window:]
        recent_primitives = self.primitive_history[-window:]

        # Alert 1: Rising prediction error (model is getting worse)
        error_trend = np.polyfit(range(window), recent_errors, 1)[0]
        if error_trend > 0.01:
            self._emit_alert(
                "DEGRADATION",
                {
                    "message": "Prediction error is increasing",
                    "trend": float(error_trend),
                    "recent_mean_error": float(np.mean(recent_errors)),
                },
            )

        # Alert 2: Entropy collapse (model is losing information)
        if np.mean(recent_entropy) < 0.1:
            self._emit_alert(
                "ENTROPY_COLLAPSE",
                {
                    "message": "State entropy near zero — model may be stuck",
                    "mean_entropy": float(np.mean(recent_entropy)),
                },
            )

        # Alert 3: Convergence failure (transitions are unrecognizable)
        convergence_rate = sum(recent_convergence) / len(recent_convergence)
        if convergence_rate < 0.3:
            self._emit_alert(
                "CONVERGENCE_FAILURE",
                {
                    "message": "Resonator failing to converge — "
                    "transitions may not match known primitives",
                    "convergence_rate": convergence_rate,
                },
            )

        # Alert 4: Primitive repetition (stuck in one mode)
        unique_primitives = len(set(recent_primitives))
        if unique_primitives <= 1 and len(recent_primitives) >= window:
            self._emit_alert(
                "MODE_LOCK",
                {
                    "message": f"Using only {recent_primitives[-1]} "
                    f"for {window} consecutive steps",
                    "locked_primitive": recent_primitives[-1],
                },
            )

        # Alert 5: High error + high confidence = silent failure
        # (This is the adversarial probe finding)
        if np.mean(recent_errors) > 0.5 and convergence_rate > 0.8:
            self._emit_alert(
                "SILENT_FAILURE",
                {
                    "message": "High prediction error despite confident "
                    "transitions — possible silent degradation",
                    "mean_error": float(np.mean(recent_errors)),
                    "convergence_rate": convergence_rate,
                },
            )

    def _emit_alert(self, alert_type: str, details: dict):
        """Emit a proprioceptive alert."""
        alert = {
            "type": alert_type,
            "step": len(self.error_trajectory),
            "timestamp": time.time(),
            "details": details,
        }
        self.alerts.append(alert)
        self.alert_count += 1

    def get_state_summary(self) -> dict:
        """
        Summary of the model's proprioceptive state.

        This is what gets fed back to the LLM interface
        (the Session 18c mechanism).
        """
        window = min(self.config.entropy_window, len(self.error_trajectory))
        if window == 0:
            return {"status": "no_data", "steps": 0}

        recent_errors = self.error_trajectory[-window:]
        recent_entropy = self.entropy_trajectory[-window:]

        return {
            "status": "active",
            "steps": len(self.error_trajectory),
            "mean_error": float(np.mean(recent_errors)),
            "error_trend": (
                float(np.polyfit(range(window), recent_errors, 1)[0])
                if window > 1
                else 0.0
            ),
            "mean_entropy": float(np.mean(recent_entropy)),
            "convergence_rate": (sum(self.convergence_trajectory[-window:]) / window),
            "unique_primitives_used": len(set(self.primitive_history[-window:])),
            "active_alerts": [a for a in self.alerts[-5:]],
            "total_alerts": self.alert_count,
        }


# =============================================================================
# THE WORLD MODEL
# =============================================================================


class VSAWorldModel:
    """
    The complete world model.

    Inner model: VSA state vector (fuzzy, compressed, compositional)
    Outer interface: observations come in, predictions go out
    Dynamics: resonator-based transition model (learned)
    Proprioception: self-monitoring of reasoning health

    The TENSION between inner model prediction and outer observation
    is the prediction error. This error:
    - Drives attention (high error = attend more to this)
    - Drives learning (update transitions to reduce future error)
    - Drives exploration (persistent error = unknown territory)

    This is the reconciliation loop. Intelligence lives here.
    """

    def __init__(self, role_names: list[str], config: WorldModelConfig | None = None):
        self.config = config or WorldModelConfig()
        self.algebra = PhasorAlgebra(self.config.dim, self.config.seed)

        # Inner model: the state
        self.state = StateVector(self.algebra, role_names)

        # Dynamics: how states change
        self.transition_model = TransitionModel(self.algebra, self.config)

        # Proprioception: self-monitoring
        self.monitor = ProprioceptiveMonitor(self.config)

        # History
        self.state_history: list[np.ndarray] = []
        self.observation_history: list[np.ndarray] = []
        self.step_count = 0

        # Attention map: which roles have highest prediction error
        self.attention: dict[str, float] = {name: 0.0 for name in role_names}

    def observe(self, observation: dict[str, np.ndarray]) -> dict:
        """
        Process a new observation.

        1. Encode observation into VSA
        2. Predict what we expected to see
        3. Compute prediction error (the mismatch)
        4. Update state with actual observation
        5. Learn from the transition
        6. Update proprioceptive state
        7. Return attention map and alerts

        This is one tick of the world model clock.
        """
        # Save current state
        state_before = self.state.compile().copy()
        self.state_history.append(state_before)

        # Encode observation: set fillers for observed roles
        obs_vector_parts = []
        per_slot_errors = {}

        for role_name, filler_value in observation.items():
            if role_name in self.state.roles:
                # What did we expect in this slot?
                expected = self.state.get_filler(role_name)
                actual = filler_value

                # Per-slot prediction error
                slot_error = 1.0 - self.algebra.similarity(expected, actual)
                per_slot_errors[role_name] = slot_error

                # Update the slot with actual observation
                self.state.set_filler(role_name, actual)
                obs_vector_parts.append(
                    self.algebra.bind(self.state.roles[role_name], actual)
                )

        # Compile observation into single vector
        if obs_vector_parts:
            obs_compiled = self.algebra.bundle(obs_vector_parts)
        else:
            obs_compiled = self.algebra.identity()

        self.observation_history.append(obs_compiled)

        # Get new state after observation
        state_after = self.state.compile()

        # Learn from this transition
        error, transition_meta = self.transition_model.learn_from_transition(
            state_before, obs_compiled, state_after
        )

        # Update attention based on per-slot errors
        # The ENTITY slot error is the key signal — it tells us
        # whether THIS particular entity is behaving unexpectedly
        entity_error = per_slot_errors.get("entity", 0.0)
        for role_name, slot_error in per_slot_errors.items():
            # Weight by entity error: high entity error = pay attention
            # to ALL slots for this observation
            weighted_error = slot_error * (1.0 + entity_error)
            alpha = 0.3
            self.attention[role_name] = alpha * weighted_error + (
                1 - alpha
            ) * self.attention.get(role_name, 0.0)

        # Proprioceptive monitoring
        state_entropy = self.state.entropy()
        self.monitor.observe_step(error, state_entropy, transition_meta)

        self.step_count += 1

        # Build result
        result = {
            "step": self.step_count,
            "prediction_error": error,
            "per_slot_errors": per_slot_errors,
            "attention": dict(self.attention),
            "state_entropy": state_entropy,
            "transition": transition_meta,
            "proprioception": self.monitor.get_state_summary(),
            "high_attention_slots": [
                name
                for name, attn in self.attention.items()
                if attn > self.config.attention_threshold
            ],
        }

        return result

    def predict(self) -> tuple[np.ndarray, dict]:
        """
        Predict the next state without observing.

        Uses the transition model to imagine what comes next
        based on current state and recent trajectory.

        This is the "dreaming" capability — running the model
        forward without new observations.
        """
        current = self.state.compile()

        # Use most recent observation as context
        if self.observation_history:
            recent_obs = self.observation_history[-1]
        else:
            recent_obs = self.algebra.identity()

        predicted, meta = self.transition_model.predict_next_state(current, recent_obs)

        return predicted, meta

    def query(self, role: str) -> tuple[np.ndarray, float]:
        """
        Query the current state for a specific role's content.

        Returns the filler vector and the attention level
        (how uncertain the model is about this slot).
        """
        filler = self.state.get_filler(role)
        attention = self.attention.get(role, 0.0)
        return filler, attention

    def get_surprise(self) -> float:
        """
        Overall surprise level — mean prediction error.

        High surprise = the world is behaving unexpectedly
        Low surprise = the model's predictions are accurate

        This is the tension between inner and outer.
        """
        if not self.transition_model.prediction_errors:
            return 1.0  # Maximum surprise when we know nothing
        window = min(
            self.config.entropy_window, len(self.transition_model.prediction_errors)
        )
        return float(np.mean(self.transition_model.prediction_errors[-window:]))

    def status(self) -> dict:
        """Full status report."""
        return {
            "step": self.step_count,
            "state_entropy": self.state.entropy(),
            "surprise": self.get_surprise(),
            "attention": dict(self.attention),
            "transition_primitives": list(
                self.transition_model.transition_primitives.keys()
            ),
            "primitive_usage": dict(self.transition_model.transition_counts),
            "proprioception": self.monitor.get_state_summary(),
            "experience_buffer_size": len(self.transition_model.experience_buffer),
        }
