# Long-Term Optimization Research Report

**Date:** 2026-01-16
**Version:** 1.0
**Scope:** Sentinel Engine Long-Term Optimization Roadmap

---

## Executive Summary

This document synthesizes research findings for the four long-term optimization tasks identified in the Hybrid Validation Report. Each section provides specific algorithms, implementation patterns, and recommendations.

### Priority Matrix

| Task | Impact | Complexity | Recommended Priority |
|------|--------|------------|---------------------|
| GPU Acceleration | High (50-100x speedup) | Medium | P1 - Immediate |
| Hierarchical Resonator | High (100K+ rows) | High | P2 - Next Quarter |
| Feedback Loop | Medium (adaptive thresholds) | Medium | P3 - Ongoing |
| Ensemble Resonators | Medium (confidence) | Medium | P4 - After Feedback |

---

## 1. GPU Acceleration for Resonator

### Current State
- **Performance:** 294 seconds for 5K rows on CPU
- **Bottleneck:** Sequential similarity computation in resonator loop
- **Target:** <10 seconds for 5K rows

### Research Findings

#### 1.1 Quick Wins (5-10x speedup)

**Move all operations to GPU:**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
codebook = codebook.to(device).contiguous()
queries = queries.to(device).contiguous()
```

**Batch similarity computation:**
```python
def batch_cosine_similarity_matrix(queries, codebook):
    """All pairwise similarities in one operation."""
    queries_norm = F.normalize(queries, p=2, dim=1)
    codebook_norm = F.normalize(codebook, p=2, dim=1)
    return queries_norm @ codebook_norm.T  # (B, K)
```

**Use torch.compile for JIT optimization:**
```python
@torch.compile(mode="max-autotune")
def gpu_sparse_resonance_step(query, codebook, top_k=64, power=0.64):
    sims = batch_similarity(query, codebook)
    values, indices = torch.topk(sims, top_k)
    weights = torch.pow(torch.clamp(values, min=0), power)
    weights = weights / (weights.sum() + 1e-10)
    return torch.einsum('n,nd->d', weights, codebook[indices])
```

#### 1.2 Major Optimizations (20-50x speedup)

**True batch resonation (process all queries in parallel):**
```python
class GPUBatchResonator:
    def batch_resonate(self, queries, max_iterations=50):
        batch_size = queries.shape[0]
        x = self._normalize_batch(queries.clone())
        converged = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for iteration in range(max_iterations):
            # Process ALL queries at once
            x_new = self._batch_sparse_resonance(x, ~converged)
            deltas = torch.norm(x - x_new, dim=-1)
            converged = converged | (deltas < self.threshold)
            if converged.all():
                break
            x = x_new
        return x
```

**CUDA streams for overlapping transfer and compute:**
```python
compute_stream = torch.cuda.Stream()
transfer_stream = torch.cuda.Stream()

for chunk in chunks:
    with torch.cuda.stream(transfer_stream):
        next_chunk = next_data.to(device, non_blocking=True)
    with torch.cuda.stream(compute_stream):
        result = process(chunk)
```

#### 1.3 Existing Libraries

| Library | Speedup | Features |
|---------|---------|----------|
| **Torchhd** | 24x CPU, 54x GPU | FHRR, BSC, MAP vectors, batch ops |
| **HDTorch** | 87-111x | Custom CUDA kernels, bit packing |
| **OpenHD** | 10.5x | Memory optimization, PARTRAIN |

### Recommended Implementation

**Phase 1 (Week 1-2):** Move to GPU, batch similarity computation
**Phase 2 (Week 3-4):** torch.compile, chunked processing
**Phase 3 (Week 5-6):** Evaluate Torchhd integration

---

## 2. Hierarchical Resonator for 100K+ Rows

### Current State
- **Codebook limit:** ~50K entries (memory bound)
- **Target:** Support 100K-1M row datasets

### Research Findings

#### 2.1 Chunked Codebook Processing

```python
class ChunkedCodebook:
    def __init__(self, vectors, chunk_size=10000, device='cuda'):
        self.chunks = [
            vectors[i:i+chunk_size].to(device)
            for i in range(0, len(vectors), chunk_size)
        ]

    def cleanup(self, query, temperature=1.0):
        all_sims = [
            F.cosine_similarity(query.unsqueeze(0), chunk, dim=1)
            for chunk in self.chunks
        ]
        similarities = torch.cat(all_sims, dim=0)
        weights = F.softmax(similarities * temperature, dim=0)
        return self._weighted_reconstruction(weights)
```

#### 2.2 Hierarchical Codebook (Coarse-to-Fine)

```python
class HierarchicalCodebook:
    """O(sqrt(K)) search instead of O(K)."""
    def __init__(self, dim, n_clusters=100, items_per_cluster=1000):
        self.cluster_centroids = torch.randn(n_clusters, dim)
        self.cluster_items = [torch.randn(items_per_cluster, dim)
                             for _ in range(n_clusters)]

    def cleanup(self, query, top_k_clusters=5):
        # Stage 1: Find best clusters
        cluster_sims = F.cosine_similarity(
            query.unsqueeze(0), self.cluster_centroids, dim=1
        )
        top_clusters = cluster_sims.topk(top_k_clusters).indices

        # Stage 2: Search only within top clusters
        candidates = torch.cat([self.cluster_items[c] for c in top_clusters])
        return self._cleanup_within(query, candidates)
```

#### 2.3 Linear Codes Alternative

Recent research shows linear codes over Boolean fields achieve O(n^3) exact factorization vs exponential for resonator:

```python
class LinearCodeVSA:
    """O(n^3) factorization with 100% success rate."""
    def binding_recovery(self, composite, subcodes):
        combined_basis = torch.cat([s.G for s in subcodes], dim=0)
        solution = self._solve_f2_system(combined_basis, composite)
        return self._extract_factors(solution, subcodes)
```

### Recommended Implementation

**Phase 1:** GPU-based k-means clustering for hierarchy building
**Phase 2:** Coarse-to-fine search implementation
**Phase 3:** Linear code fallback for difficult cases

---

## 3. Feedback Loop for Threshold Calibration

### Current State
- **Thresholds:** Static, manually calibrated
- **Target:** Adaptive thresholds from user feedback

### Research Findings

#### 3.1 Database Schema

```sql
-- User feedback storage
CREATE TABLE detection_feedback (
    id UUID PRIMARY KEY,
    analysis_id UUID NOT NULL,
    sku VARCHAR(255) NOT NULL,
    primitive_key VARCHAR(50) NOT NULL,
    detection_score FLOAT NOT NULL,
    threshold_used FLOAT NOT NULL,
    is_true_positive BOOLEAN,
    reviewer_id UUID,
    reviewed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Bayesian state for each primitive
CREATE TABLE bayesian_threshold_state (
    primitive_key VARCHAR(50) PRIMARY KEY,
    alpha FLOAT DEFAULT 2.0,
    beta FLOAT DEFAULT 2.0,
    min_threshold FLOAT NOT NULL,
    max_threshold FLOAT NOT NULL,
    map_estimate FLOAT NOT NULL,
    feedback_count INTEGER DEFAULT 0
);
```

#### 3.2 Adaptive Threshold Algorithm

```python
class AdaptiveThreshold:
    def __init__(self, primitive_key, base_threshold):
        self.primitive_key = primitive_key
        self.base_threshold = base_threshold
        self.tp_scores = deque(maxlen=500)
        self.fp_scores = deque(maxlen=500)
        self.ewma_alpha = 0.1
        self.current_ewma = None

    def update_with_feedback(self, score, is_true_positive):
        if is_true_positive:
            self.tp_scores.append(score)
        else:
            self.fp_scores.append(score)

    def compute_optimal_threshold(self, target_fpr=0.05):
        if len(self.fp_scores) >= 10 and len(self.tp_scores) >= 10:
            # FP threshold at (1 - target_fpr) percentile
            fp_threshold = np.percentile(list(self.fp_scores), (1 - target_fpr) * 100)
            # Ensure at least 50% recall
            tp_50th = np.percentile(list(self.tp_scores), 50)
            return max(fp_threshold, tp_50th)
        return self.base_threshold
```

#### 3.3 Active Learning for Review Prioritization

```python
class ActiveLearningSelector:
    def select_for_review(self, flagged_items, thresholds, max_items=10):
        # Uncertainty sampling: prioritize items near threshold
        scored_items = []
        for item in flagged_items:
            threshold = thresholds.get(item.primitive_key, 0.5)
            distance = abs(item.score - threshold)
            uncertainty = 1.0 / (distance + 0.01)
            scored_items.append((item, uncertainty))

        scored_items.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in scored_items[:max_items]]
```

#### 3.4 A/B Testing Framework

```python
class ThresholdExperiment:
    def __init__(self, control_thresholds, treatment_thresholds):
        self.control = ThresholdVariant("control", control_thresholds)
        self.treatment = ThresholdVariant("treatment", treatment_thresholds)

    def assign_variant(self, user_id):
        hash_value = int(hashlib.md5(f"{self.id}:{user_id}".encode()).hexdigest(), 16)
        return self.treatment if hash_value % 100 < 50 else self.control

    def compute_significance(self):
        # Two-proportion z-test
        n1, p1 = self.control.total_flags, self.control.precision
        n2, p2 = self.treatment.total_flags, self.treatment.precision
        p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = (p_pool * (1 - p_pool) * (1/n1 + 1/n2)) ** 0.5
        z = (p2 - p1) / se
        return 2 * (1 - stats.norm.cdf(abs(z)))
```

### Recommended Implementation

**Phase 1 (Week 1-2):** Database schema, feedback API endpoint
**Phase 2 (Week 3-4):** EWMA adaptive thresholds, threshold versioning
**Phase 3 (Week 5-6):** Active learning review queue
**Phase 4 (Week 7-8):** A/B testing framework

---

## 4. Ensemble Resonators for High-Stakes Decisions

### Current State
- **Architecture:** Baseline detector + single VSA resonator sanity check
- **Target:** Multiple resonators with uncertainty quantification

### Research Findings

#### 4.1 Multi-Resonator Ensemble

```python
class HighStakesEnsembleResonator:
    def __init__(self, configs):
        # Diversity through different configurations
        self.resonators = [
            Resonator(ResonatorConfig(iters=100, alpha=0.7, top_k=5)),   # Fast
            Resonator(ResonatorConfig(iters=300, alpha=0.85, top_k=15)), # Balanced
            Resonator(ResonatorConfig(iters=600, alpha=0.95, top_k=50)), # Deep
        ]

    def resonate_with_uncertainty(self, query, baseline_score=None):
        results = [r.resonate(query) for r in self.resonators]
        scores = [r.top_matches[0][1] if r.top_matches else 0 for r in results]

        # Epistemic uncertainty = variance between resonators
        epistemic_uncertainty = np.var(scores)

        # Agreement = how many agree on top label
        labels = [r.top_matches[0][0] if r.top_matches else None for r in results]
        valid_labels = [l for l in labels if l]
        most_common = max(set(valid_labels), key=valid_labels.count) if valid_labels else None
        agreement = valid_labels.count(most_common) / len(valid_labels) if valid_labels else 0

        # Require human review if uncertain
        requires_review = (
            epistemic_uncertainty > 0.3 or
            agreement < 0.7 or
            (baseline_score and abs(baseline_score - np.mean(scores)) > 0.3)
        )

        return EnsembleResult(
            prediction=most_common,
            confidence=np.mean(scores),
            uncertainty=epistemic_uncertainty,
            agreement=agreement,
            requires_human_review=requires_review
        )
```

#### 4.2 Temperature Scaling for Calibration

```python
def temperature_scaled_confidence(raw_scores, T=1.5):
    """Soften confidence scores for better calibration."""
    return torch.softmax(torch.tensor(raw_scores) / T, dim=-1)
```

#### 4.3 Conformal Prediction for Coverage Guarantees

```python
def conformal_anomaly_detection(query_score, calibration_scores, alpha=0.05):
    """Anomaly detection with statistical guarantee on FPR."""
    n = len(calibration_scores)
    rank = sum(1 for s in calibration_scores if s >= query_score)
    p_value = (rank + 1) / (n + 1)
    is_anomaly = p_value < alpha  # Controls FPR at alpha
    return is_anomaly, p_value
```

#### 4.4 Cascading Pipeline with Reject Option

```
Stage 0: Data Quality Check
    └─ Contradictions → IMMEDIATE ALERT

Stage 1: Statistical Prefilter (O(n))
    └─ Pass ~10% to Stage 2

Stage 2: Baseline Detector (Fast)
    ├─ Confidence > 0.9 → ACCEPT
    ├─ Confidence < 0.3 → REJECT
    └─ Uncertain → Stage 3

Stage 3: Ensemble Resonator (Sanity Check)
    ├─ Agreement with baseline → CONFIRM
    └─ Disagreement → HUMAN REVIEW

Stage 4: Confidence Calibration
    └─ Temperature scaling + conformal prediction
```

### Recommended Implementation

**Phase 1:** Multi-config resonator ensemble
**Phase 2:** Disagreement-based uncertainty
**Phase 3:** Temperature scaling calibration
**Phase 4:** Conformal prediction integration

---

## Implementation Roadmap

### Quarter 1: Foundation

| Week | Task | Deliverable |
|------|------|-------------|
| 1-2 | GPU Quick Wins | 5-10x speedup |
| 3-4 | Feedback Database Schema | Tables + API |
| 5-6 | torch.compile Integration | 20-30x speedup |
| 7-8 | Adaptive Threshold v1 | EWMA-based |

### Quarter 2: Scale

| Week | Task | Deliverable |
|------|------|-------------|
| 1-2 | Hierarchical Codebook | K-means clustering |
| 3-4 | Active Learning Queue | Review prioritization |
| 5-6 | Batch GPU Resonator | True parallelism |
| 7-8 | A/B Testing Framework | Experiment management |

### Quarter 3: Confidence

| Week | Task | Deliverable |
|------|------|-------------|
| 1-2 | Multi-Resonator Ensemble | 3-resonator config |
| 3-4 | Uncertainty Quantification | Epistemic variance |
| 5-6 | Temperature Calibration | Platt/temperature scaling |
| 7-8 | Conformal Prediction | Statistical guarantees |

---

## Success Metrics

| Metric | Current | Q1 Target | Q2 Target | Q3 Target |
|--------|---------|-----------|-----------|-----------|
| Resonator Time (5K rows) | 294s | <30s | <10s | <5s |
| Max Dataset Size | 50K | 100K | 500K | 1M |
| Threshold Adaptation | Manual | EWMA | Bayesian | A/B Tested |
| Confidence Calibration | None | Basic | Temperature | Conformal |
| Human Review Rate | N/A | N/A | <5% | <2% |

---

## References

### GPU Acceleration
- [Torchhd Library](https://github.com/hyperdimensional-computing/torchhd)
- [HDTorch CUDA Kernels](https://pypi.org/project/hdtorch/)
- [PyTorch CUDA Best Practices](https://docs.pytorch.org/docs/stable/notes/cuda.html)

### Hierarchical Resonators
- [Resonator Networks Paper](https://rctn.org/bruno/papers/resonator1.pdf)
- [Linear Codes for HDC](https://arxiv.org/abs/2403.03278)
- [Nature Machine Intelligence - HRNs](https://www.nature.com/articles/s42256-024-00848-0)

### Feedback Loops
- [Active Learning for Anomaly Detection](https://www.sciencedirect.com/science/article/abs/pii/S0020025524009265)
- [Bayesian Optimization in ML](https://www.geeksforgeeks.org/artificial-intelligence/bayesian-optimization-in-machine-learning/)
- [EWMA Control Charts](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc324.htm)

### Ensemble Methods
- [Deep Ensembles for Uncertainty](https://arxiv.org/abs/1612.01474)
- [Neural Network Calibration](https://arxiv.org/abs/1706.04599)
- [Conformal Prediction Tutorial](https://people.eecs.berkeley.edu/~angelopoulos/publications/downloads/gentle_intro_conformal_dfuq.pdf)

---

**Document Generated:** 2026-01-16
**Contact:** engineering@profit-sentinel.io
