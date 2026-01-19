"""
vsa_core/probabilistic.py - Probabilistic Superposition (P-Sup) for VSA

Implements Bayesian-style hypothesis tracking using VSA superposition.
Multiple competing hypotheses are maintained simultaneously with probability
weights, enabling gradual disambiguation as evidence accumulates.

Key Concepts:
    - Hypothesis Bundle: Weighted superposition of competing explanations
    - Bayesian Update: P(H|E) ∝ P(E|H) · P(H) using VSA similarity as likelihood
    - Collapse: When confidence exceeds threshold, commit to winner

Mathematical Foundation:
    The probabilistic superposition uses sqrt(probability) as weights to
    preserve quantum-like interference patterns that aid disambiguation.
    This is similar to how quantum amplitudes work, where |ψ|² = probability.

Usage:
    hypotheses = [
        ("shrinkage", shrinkage_vec, 0.4),
        ("clerical_error", clerical_vec, 0.3),
        ("vendor_issue", vendor_vec, 0.3),
    ]
    bundle = p_sup(hypotheses)

    # As evidence arrives, update
    bundle = p_sup_update(bundle, evidence_vec)

    # Check for collapse
    winner = p_sup_collapse(bundle, threshold=0.9)
    if winner:
        print(f"High confidence: {winner}")
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .vectors import normalize, similarity


@dataclass
class HypothesisBundle:
    """Probabilistic superposition of hypotheses.

    Attributes:
        vector: Weighted superposition vector
        hypotheses: List of hypothesis labels
        probabilities: Probability per hypothesis (sums to 1)
        basis_vectors: Individual hypothesis vectors (n, d)
    """
    vector: torch.Tensor
    hypotheses: list[str]
    probabilities: torch.Tensor
    basis_vectors: torch.Tensor

    def __repr__(self) -> str:
        probs = [f"{h}: {p:.2%}" for h, p in zip(self.hypotheses, self.probabilities.tolist())]
        return f"HypothesisBundle({', '.join(probs)})"

    def top_hypothesis(self) -> tuple[str, float]:
        """Get the most probable hypothesis."""
        max_idx = int(torch.argmax(self.probabilities))
        return self.hypotheses[max_idx], float(self.probabilities[max_idx])

    def entropy(self) -> float:
        """Compute entropy of probability distribution (uncertainty measure).

        Returns:
            Entropy in bits. Lower = more certain.
        """
        # Avoid log(0)
        p = torch.clamp(self.probabilities, min=1e-10)
        return float(-torch.sum(p * torch.log2(p)))


def p_sup(
    hypotheses: list[tuple[str, torch.Tensor, float]],
    normalize_probs: bool = True
) -> HypothesisBundle:
    """Create probabilistic superposition of hypotheses.

    Args:
        hypotheses: List of (label, vector, prior_probability) tuples
        normalize_probs: Whether to normalize probabilities to sum to 1

    Returns:
        HypothesisBundle with weighted superposition

    Example:
        bundle = p_sup([
            ("margin_leak", margin_leak_vec, 0.5),
            ("pricing_error", pricing_error_vec, 0.3),
            ("vendor_markup", vendor_markup_vec, 0.2),
        ])
    """
    if len(hypotheses) == 0:
        raise ValueError("At least one hypothesis required")

    labels = [h[0] for h in hypotheses]
    vectors = torch.stack([h[1] for h in hypotheses])
    probs = torch.tensor([h[2] for h in hypotheses], dtype=torch.float32)

    # Normalize probabilities if requested
    if normalize_probs:
        probs = probs / probs.sum()

    # Create weighted superposition using sqrt(prob) as weights
    # This preserves interference patterns for disambiguation
    weights = torch.sqrt(probs)

    # Handle complex vectors
    if vectors.is_complex():
        weights = weights.to(vectors.dtype)

    superposition = torch.einsum('h,hd->d', weights, vectors)
    superposition = normalize(superposition)

    return HypothesisBundle(
        vector=superposition,
        hypotheses=labels,
        probabilities=probs,
        basis_vectors=vectors
    )


def p_sup_update(
    bundle: HypothesisBundle,
    evidence: torch.Tensor,
    likelihood_fn: Callable[[torch.Tensor, torch.Tensor], float] | None = None,
    temperature: float = 1.0
) -> HypothesisBundle:
    """Bayesian update of hypothesis probabilities given evidence.

    Implements: P(H|E) ∝ P(E|H) · P(H)

    Where P(E|H) = similarity(evidence, hypothesis_vector)

    Args:
        bundle: Current hypothesis bundle
        evidence: Evidence vector to condition on
        likelihood_fn: Custom likelihood function (default: VSA similarity)
        temperature: Sharpness of update (lower = sharper)

    Returns:
        Updated HypothesisBundle with new probabilities

    Example:
        # Receive evidence that points toward margin_leak
        new_bundle = p_sup_update(bundle, margin_evidence_vec)
    """
    if likelihood_fn is None:
        def likelihood_fn(e, h):
            return float(similarity(e, h).real) if e.is_complex() else float(similarity(e, h))

    # Compute likelihoods for each hypothesis
    likelihoods = torch.tensor([
        likelihood_fn(evidence, h_vec)
        for h_vec in bundle.basis_vectors
    ], dtype=torch.float32)

    # Apply temperature
    likelihoods = likelihoods / temperature

    # Shift to positive (similarity can be negative)
    # Use softmax-style normalization for numerical stability
    likelihoods = likelihoods - likelihoods.max()
    likelihoods = torch.exp(likelihoods)

    # Bayesian update: posterior ∝ likelihood × prior
    posterior = likelihoods * bundle.probabilities
    posterior = posterior / (posterior.sum() + 1e-10)

    # Rebuild superposition with updated weights
    weights = torch.sqrt(posterior)
    if bundle.basis_vectors.is_complex():
        weights = weights.to(bundle.basis_vectors.dtype)

    new_superposition = torch.einsum('h,hd->d', weights, bundle.basis_vectors)

    return HypothesisBundle(
        vector=normalize(new_superposition),
        hypotheses=bundle.hypotheses,
        probabilities=posterior,
        basis_vectors=bundle.basis_vectors
    )


def p_sup_collapse(
    bundle: HypothesisBundle,
    threshold: float = 0.9
) -> str | None:
    """Collapse superposition if one hypothesis exceeds threshold.

    Once collapsed, the bundle has effectively "decided" on a hypothesis.

    Args:
        bundle: Hypothesis bundle to check
        threshold: Probability threshold for collapse

    Returns:
        Winning hypothesis label, or None if still uncertain

    Example:
        winner = p_sup_collapse(bundle, threshold=0.85)
        if winner:
            trigger_alert(winner)
    """
    max_prob, max_idx = torch.max(bundle.probabilities, dim=0)
    if float(max_prob) >= threshold:
        return bundle.hypotheses[int(max_idx)]
    return None


def p_sup_add_hypothesis(
    bundle: HypothesisBundle,
    label: str,
    vector: torch.Tensor,
    prior: float = 0.1
) -> HypothesisBundle:
    """Add a new hypothesis to existing bundle.

    Redistributes probability mass to accommodate new hypothesis.

    Args:
        bundle: Existing hypothesis bundle
        label: Label for new hypothesis
        vector: Vector for new hypothesis
        prior: Prior probability for new hypothesis (others scaled down)

    Returns:
        New bundle with additional hypothesis
    """
    # Scale down existing probabilities
    scale = 1.0 - prior
    new_probs = bundle.probabilities * scale

    # Add new hypothesis
    new_probs = torch.cat([new_probs, torch.tensor([prior])])
    new_vectors = torch.cat([bundle.basis_vectors, vector.unsqueeze(0)], dim=0)
    new_labels = bundle.hypotheses + [label]

    # Rebuild superposition
    weights = torch.sqrt(new_probs)
    if new_vectors.is_complex():
        weights = weights.to(new_vectors.dtype)

    superposition = torch.einsum('h,hd->d', weights, new_vectors)

    return HypothesisBundle(
        vector=normalize(superposition),
        hypotheses=new_labels,
        probabilities=new_probs,
        basis_vectors=new_vectors
    )


def p_sup_remove_hypothesis(
    bundle: HypothesisBundle,
    label: str
) -> HypothesisBundle:
    """Remove a hypothesis from bundle (e.g., if ruled out).

    Redistributes removed hypothesis's probability to remaining ones.

    Args:
        bundle: Existing hypothesis bundle
        label: Label of hypothesis to remove

    Returns:
        New bundle without the specified hypothesis
    """
    if label not in bundle.hypotheses:
        raise ValueError(f"Hypothesis '{label}' not in bundle")

    idx = bundle.hypotheses.index(label)

    # Remove from all components
    new_labels = bundle.hypotheses[:idx] + bundle.hypotheses[idx+1:]
    new_probs = torch.cat([bundle.probabilities[:idx], bundle.probabilities[idx+1:]])
    new_vectors = torch.cat([bundle.basis_vectors[:idx], bundle.basis_vectors[idx+1:]], dim=0)

    # Renormalize probabilities
    new_probs = new_probs / new_probs.sum()

    # Rebuild superposition
    weights = torch.sqrt(new_probs)
    if new_vectors.is_complex():
        weights = weights.to(new_vectors.dtype)

    superposition = torch.einsum('h,hd->d', weights, new_vectors)

    return HypothesisBundle(
        vector=normalize(superposition),
        hypotheses=new_labels,
        probabilities=new_probs,
        basis_vectors=new_vectors
    )


def p_sup_merge(
    bundles: list[HypothesisBundle],
    weights: list[float] | None = None
) -> HypothesisBundle:
    """Merge multiple hypothesis bundles (e.g., from different data sources).

    Args:
        bundles: List of hypothesis bundles to merge
        weights: Optional weights for each bundle (default: equal)

    Returns:
        Merged bundle with combined hypotheses

    Note:
        Hypotheses with same label across bundles are combined by averaging
        their probability-weighted vectors.
    """
    if len(bundles) == 0:
        raise ValueError("At least one bundle required")

    if weights is None:
        weights = [1.0 / len(bundles)] * len(bundles)

    # Collect all unique hypotheses
    all_labels = {}  # label -> list of (vector, prob, weight)

    for bundle, w in zip(bundles, weights):
        for i, label in enumerate(bundle.hypotheses):
            if label not in all_labels:
                all_labels[label] = []
            all_labels[label].append((
                bundle.basis_vectors[i],
                float(bundle.probabilities[i]),
                w
            ))

    # Merge hypotheses
    merged_labels = []
    merged_vectors = []
    merged_probs = []

    for label, entries in all_labels.items():
        merged_labels.append(label)

        # Weighted average of vectors
        total_weight = sum(prob * w for _, prob, w in entries)
        if total_weight > 0:
            merged_vec = sum(
                (prob * w / total_weight) * vec
                for vec, prob, w in entries
            )
            merged_vectors.append(normalize(merged_vec))
        else:
            merged_vectors.append(entries[0][0])

        # Average probability
        merged_probs.append(sum(prob * w for _, prob, w in entries))

    merged_vectors = torch.stack(merged_vectors)
    merged_probs = torch.tensor(merged_probs)
    merged_probs = merged_probs / merged_probs.sum()

    # Build superposition
    weights_tensor = torch.sqrt(merged_probs)
    if merged_vectors.is_complex():
        weights_tensor = weights_tensor.to(merged_vectors.dtype)

    superposition = torch.einsum('h,hd->d', weights_tensor, merged_vectors)

    return HypothesisBundle(
        vector=normalize(superposition),
        hypotheses=merged_labels,
        probabilities=merged_probs,
        basis_vectors=merged_vectors
    )
