"""
VSA World Model — Battery-Driven Test
=======================================

Uses the behavioral battery not just for evaluation but as the
actual learning mechanism:

1. WARMUP: Observe transitions, cluster them, derive primitives
   (Level 5 self-grounding applied to transitions)

2. RUN: Process observations, predict, learn, measure battery health

3. DIAGNOSE: Battery detects when operations are degrading

The battery IS the proprioceptive system.
"""

import time
from collections import defaultdict
from typing import Dict, List

import numpy as np

from ..battery import WarmupPhase, WorldModelBattery
from ..core import PhasorAlgebra, TransitionModel, VSAWorldModel, WorldModelConfig


class InventoryEnvironment:
    """Same environment, extracted for clarity."""

    def __init__(self, n_skus: int = 50, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.n_skus = n_skus
        self.algebra = PhasorAlgebra(dim=4096, seed=seed)

        self.sku_vectors = {}
        for i in range(n_skus):
            self.sku_vectors[f"SKU_{i:04d}"] = self.algebra.random_vector()

        self.value_vectors = {}
        for val in ["high", "medium", "low", "zero", "negative"]:
            self.value_vectors[val] = self.algebra.random_vector()

        self.true_state = {}
        for sku_id in self.sku_vectors:
            cost = round(self.rng.uniform(5, 100), 2)
            margin = self.rng.uniform(0.2, 0.6)
            self.true_state[sku_id] = {
                "stock": int(self.rng.integers(10, 200)),
                "cost": cost,
                "price": round(cost * (1 + margin), 2),
                "sales_velocity": int(self.rng.integers(1, 20)),
                "anomaly": None,
            }

        self.anomalous_skus = {}
        self._inject_anomalies()
        self.step = 0

    def _inject_anomalies(self):
        sku_ids = list(self.sku_vectors.keys())
        n_anom = max(1, self.n_skus // 5)
        anom_ids = self.rng.choice(sku_ids, n_anom, replace=False)
        types = ["SHRINKAGE", "MARGIN_EROSION", "DEAD_STOCK", "PHANTOM_INVENTORY"]

        for sid in anom_ids:
            atype = self.rng.choice(types)
            self.anomalous_skus[str(sid)] = atype
            self.true_state[str(sid)]["anomaly"] = atype
            if atype == "PHANTOM_INVENTORY":
                self.true_state[str(sid)]["stock"] = -int(self.rng.integers(10, 100))
            elif atype == "DEAD_STOCK":
                self.true_state[str(sid)]["sales_velocity"] = 0
            elif atype == "MARGIN_EROSION":
                self.true_state[str(sid)]["cost"] = (
                    self.true_state[str(sid)]["price"] * 1.1
                )

    def _quantize(self, value, domain):
        if domain == "stock":
            if value < 0:
                return "negative"
            if value == 0:
                return "zero"
            if value < 20:
                return "low"
            if value < 100:
                return "medium"
            return "high"
        elif domain == "margin":
            if value < 0:
                return "negative"
            if value < 0.1:
                return "low"
            if value < 0.3:
                return "medium"
            return "high"
        elif domain == "velocity":
            if value == 0:
                return "zero"
            if value < 5:
                return "low"
            if value < 15:
                return "medium"
            return "high"
        return "medium"

    def get_observation(self, sku_id):
        state = self.true_state[sku_id]

        obs_stock = state["stock"] + int(self.rng.integers(-3, 4))
        obs_cost = state["cost"] * (1 + self.rng.uniform(-0.02, 0.02))
        obs_price = state["price"]

        if state["anomaly"] == "SHRINKAGE":
            obs_stock -= int(self.rng.integers(5, 20))
        elif state["anomaly"] == "MARGIN_EROSION":
            obs_cost *= 1 + self.rng.uniform(0.01, 0.05)
            self.true_state[sku_id]["cost"] = obs_cost

        margin = (obs_price - obs_cost) / obs_price if obs_price > 0 else 0

        return {
            "entity": self.sku_vectors[sku_id],
            "stock_level": self.value_vectors[self._quantize(obs_stock, "stock")],
            "margin": self.value_vectors[self._quantize(margin, "margin")],
            "velocity": self.value_vectors[
                self._quantize(state["sales_velocity"], "velocity")
            ],
        }

    def step_environment(self):
        self.step += 1
        n_obs = min(10, self.n_skus)
        observed = self.rng.choice(list(self.sku_vectors.keys()), n_obs, replace=False)

        results = []
        for sid in observed:
            obs = self.get_observation(str(sid))
            obs["_sku_id"] = str(sid)
            obs["_anomaly"] = self.true_state[str(sid)]["anomaly"]
            results.append(obs)
        return results


def run_battery_driven_test(n_warmup: int = 30, n_steps: int = 100, n_skus: int = 50):
    """
    Run the world model with battery-driven warmup and diagnostics.
    """
    print("=" * 70)
    print("VSA WORLD MODEL — BATTERY-DRIVEN TEST")
    print("=" * 70)
    print()

    config = WorldModelConfig(
        dim=4096,
        n_roles=8,
        resonator_iters=50,
        resonator_threshold=0.9,
        learning_rate=0.02,
        attention_threshold=0.3,
        entropy_window=10,
    )

    env = InventoryEnvironment(n_skus=n_skus, seed=42)
    role_names = [
        "entity",
        "stock_level",
        "margin",
        "velocity",
        "trend",
        "category",
        "supplier",
        "seasonality",
    ]
    model = VSAWorldModel(role_names, config)

    battery = WorldModelBattery(model.algebra, n_trials=5, seed=42)
    warmup = WarmupPhase(model.algebra, n_warmup_steps=n_warmup * 10)

    print(f"Environment: {n_skus} SKUs, {len(env.anomalous_skus)} anomalous")
    print(f"Anomalies: {dict(sorted(env.anomalous_skus.items()))}")
    print()

    # =========================================================================
    # PHASE 1: BASELINE BATTERY
    # =========================================================================
    print("PHASE 1: BASELINE BATTERY")
    print("-" * 40)

    baseline_result = battery.run_full_battery()
    health = baseline_result.health_scores

    print(f"  Binding health:    {health['binding_health']:.4f}")
    print(f"  Chain health:      {health['chain_health']:.4f}")
    print(f"  Bundling health:   {health['bundling_health']:.4f}")
    print(f"  Algebraic health:  {health['algebraic_health']:.4f}")
    print()

    # =========================================================================
    # PHASE 2: WARMUP — Observe transitions, derive primitives
    # =========================================================================
    print(f"PHASE 2: WARMUP ({n_warmup} steps)")
    print("-" * 40)

    t_start = time.time()

    for step in range(n_warmup):
        observations = env.step_environment()
        for obs in observations:
            sku_id = obs.pop("_sku_id")
            anomaly = obs.pop("_anomaly")

            state_before = model.state.compile().copy()

            # Apply observation to state (no prediction yet)
            for role_name, filler_value in obs.items():
                if role_name in model.state.roles:
                    model.state.set_filler(role_name, filler_value)

            state_after = model.state.compile()

            # Collect transition for warmup
            obs_compiled = model.algebra.bundle(
                [
                    model.algebra.bind(model.state.roles[r], v)
                    for r, v in obs.items()
                    if r in model.state.roles
                ]
            )
            warmup.collect_transition(state_before, obs_compiled, state_after)

    warmup_time = time.time() - t_start

    # Derive primitives from observed transitions
    learned_primitives = warmup.derive_primitives(max_primitives=12)

    print(
        f"  Collected {len(warmup.observed_transitions)} transitions "
        f"in {warmup_time:.1f}s"
    )
    print(f"  Derived {len(learned_primitives)} transition primitives:")
    for name in sorted(learned_primitives.keys()):
        print(f"    {name}")
    print()

    # Install learned primitives into the transition model
    model.transition_model.transition_primitives = learned_primitives
    model.transition_model.transition_counts = defaultdict(int)

    # =========================================================================
    # PHASE 3: RUN — Process observations with learned dynamics
    # =========================================================================
    print(f"PHASE 3: RUN ({n_steps} steps with learned primitives)")
    print("-" * 40)

    errors_normal = []
    errors_anomalous = []
    sku_error_acc = defaultdict(list)
    sku_attention_acc = defaultdict(list)

    t_start = time.time()

    for step in range(n_steps):
        observations = env.step_environment()

        for obs in observations:
            sku_id = obs.pop("_sku_id")
            anomaly = obs.pop("_anomaly")

            result = model.observe(obs)
            error = result["prediction_error"]

            sku_error_acc[sku_id].append(error)
            mean_attn = np.mean(list(result["attention"].values()))
            sku_attention_acc[sku_id].append(mean_attn)

            if anomaly:
                errors_anomalous.append(error)
            else:
                errors_normal.append(error)

        if (step + 1) % 25 == 0:
            status = model.status()
            print(
                f"  Step {step+1}/{n_steps} | "
                f"Surprise: {status['surprise']:.3f} | "
                f"Primitives: {len(status['transition_primitives'])} | "
                f"Alerts: {status['proprioception'].get('total_alerts', 0)}"
            )

    run_time = time.time() - t_start

    # =========================================================================
    # PHASE 4: POST-RUN BATTERY
    # =========================================================================
    print()
    print("PHASE 4: POST-RUN BATTERY")
    print("-" * 40)

    post_result = battery.run_full_battery(
        transition_model=model.transition_model,
        recent_states=model.state_history[-100:],
        recent_observations=model.observation_history[-100:],
    )

    post_health = post_result.health_scores
    print(
        f"  Binding health:    {post_health['binding_health']:.4f} "
        f"(baseline: {health['binding_health']:.4f})"
    )
    print(
        f"  Chain health:      {post_health['chain_health']:.4f} "
        f"(baseline: {health['chain_health']:.4f})"
    )
    print(f"  Transition health: {post_health['transition_health']:.4f}")
    print(f"  Convergence:       {post_health['convergence_health']:.4f}")

    if post_result.anomalies:
        print("\n  Battery anomalies detected:")
        for a in post_result.anomalies:
            print(f"    ⚠ {a}")

    # =========================================================================
    # PHASE 5: RESULTS
    # =========================================================================
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    print(
        f"Time: warmup {warmup_time:.1f}s + run {run_time:.1f}s "
        f"= {warmup_time + run_time:.1f}s total"
    )
    print()

    # Error separation
    mean_err_norm = np.mean(errors_normal) if errors_normal else 0
    mean_err_anom = np.mean(errors_anomalous) if errors_anomalous else 0

    print("1. PREDICTION ERROR SEPARATION")
    print(f"   Normal:    {mean_err_norm:.4f} (n={len(errors_normal)})")
    print(f"   Anomalous: {mean_err_anom:.4f} (n={len(errors_anomalous)})")

    if errors_normal and errors_anomalous:
        pooled_std = np.sqrt(
            (np.std(errors_normal) ** 2 + np.std(errors_anomalous) ** 2) / 2
        )
        if pooled_std > 0:
            d = (mean_err_anom - mean_err_norm) / pooled_std
            print(f"   Cohen's d = {d:.3f}", end="")
            if d > 0.8:
                print(" → LARGE")
            elif d > 0.5:
                print(" → MEDIUM")
            elif d > 0.2:
                print(" → SMALL")
            else:
                print(" → NONE")

    print()

    # Per-SKU ranking
    print("2. PER-SKU ERROR RANKING (top 15)")
    sku_means = {
        s: np.mean(errs) for s, errs in sku_error_acc.items() if len(errs) >= 3
    }
    sorted_skus = sorted(sku_means.items(), key=lambda x: -x[1])[:15]

    anom_in_top = 0
    for rank, (sid, merr) in enumerate(sorted_skus, 1):
        is_anom = sid in env.anomalous_skus
        atype = env.anomalous_skus.get(sid, "-")
        marker = " ← ANOMALY" if is_anom else ""
        print(f"   {rank:2d}. {sid}: error={merr:.4f} [{atype}]{marker}")
        if is_anom:
            anom_in_top += 1

    n_anom = len(env.anomalous_skus)
    expected = min(15, n_anom) * 15 / n_skus
    print(
        f"\n   Anomalous in top 15: {anom_in_top} "
        f"(expected by chance: {expected:.1f})"
    )
    enrichment_ratio = anom_in_top / (expected + 0.01)
    if enrichment_ratio > 2:
        print("   → STRONG enrichment")
    elif enrichment_ratio > 1.3:
        print("   → MODERATE enrichment")
    else:
        print("   → WEAK/NO enrichment")

    print()

    # Primitive usage distribution
    print("3. TRANSITION PRIMITIVE USAGE")
    status = model.status()
    total_prims = len(status["transition_primitives"])
    usage = status["primitive_usage"]
    total_usage = sum(usage.values())

    print(f"   Total primitives: {total_prims}")
    active_prims = sum(1 for c in usage.values() if c > 0)
    print(f"   Active primitives: {active_prims} / {total_prims}")

    if total_usage > 0:
        # Entropy of usage distribution
        probs = np.array([c / total_usage for c in usage.values() if c > 0])
        usage_entropy = -np.sum(probs * np.log(probs + 1e-15))
        max_entropy = np.log(active_prims + 1e-15)
        usage_uniformity = usage_entropy / max_entropy if max_entropy > 0 else 0
        print(
            f"   Usage uniformity: {usage_uniformity:.3f} " f"(1.0 = perfectly uniform)"
        )

    for name, count in sorted(usage.items(), key=lambda x: -x[1])[:8]:
        pct = count / total_usage * 100 if total_usage > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"      {name:20s}: {count:4d} ({pct:5.1f}%) {bar}")

    print()

    # Final verdict
    print("4. VERDICT")
    separation = mean_err_anom > mean_err_norm
    enriched = anom_in_top > expected * 1.3
    diverse_usage = active_prims >= 3
    battery_healthy = (
        post_health["binding_health"] > 0.9 and post_health["chain_health"] > 0.9
    )

    checks = [
        ("Prediction error separates normal/anomalous", separation),
        ("Anomalies enriched in high-error ranking", enriched),
        ("Multiple transition primitives active", diverse_usage),
        ("Battery health maintained post-run", battery_healthy),
    ]

    passed = sum(1 for _, v in checks if v)
    for label, ok in checks:
        print(f"   {'✓' if ok else '✗'} {label}")

    print(f"\n   {passed}/{len(checks)} criteria met")

    return {
        "separation_d": d if "d" in dir() else 0,
        "enrichment_ratio": enrichment_ratio,
        "active_primitives": active_prims,
        "battery_healthy": battery_healthy,
        "passed": passed,
    }


if __name__ == "__main__":
    results = run_battery_driven_test(n_warmup=30, n_steps=100, n_skus=50)
