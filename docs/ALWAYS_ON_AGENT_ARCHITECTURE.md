# Always-On Agent Architecture: VSA-Augmented Retail Intelligence

**Date:** January 18, 2026
**Version:** 0.1.0-design
**Status:** Technical Specification

---

## Executive Summary

This document specifies an always-on agentic system for Profit Sentinel that combines:
1. **VSA/HDC verification layer** - Symbolic consistency and hallucination prevention
2. **Novel primitives** - N-Bind, CW-Bundle, T-Bind, P-Sup, SE-Bind for advanced reasoning
3. **Persistent memory** - Letta-style long-term memory with VSA-indexed retrieval
4. **Event-driven processing** - Continuous monitoring without polling overhead

---

## Architecture Overview

```
                    ┌─────────────────────────────────────┐
                    │         Always-On Agent Core        │
                    │                                     │
                    │  ┌───────────────────────────────┐  │
                    │  │     Event Loop (asyncio)      │  │
                    │  │  • File watch (inventory)     │  │
                    │  │  • POS webhook receiver       │  │
                    │  │  • Scheduled scans            │  │
                    │  └───────────┬───────────────────┘  │
                    │              │                      │
                    │              ▼                      │
                    │  ┌───────────────────────────────┐  │
                    │  │   VSA Working Memory (WM)     │  │
                    │  │  • Recent facts bundle        │  │
                    │  │  • Temporal context (T-Bind)  │  │
                    │  │  • Active hypothesis (P-Sup)  │  │
                    │  └───────────┬───────────────────┘  │
                    │              │                      │
                    │              ▼                      │
                    │  ┌───────────────────────────────┐  │
                    │  │    Reasoning Engine           │  │
                    │  │  • Anomaly detection          │  │
                    │  │  • Causal inference           │  │
                    │  │  • Pattern matching           │  │
                    │  └───────────┬───────────────────┘  │
                    │              │                      │
                    │              ▼                      │
                    │  ┌───────────────────────────────┐  │
                    │  │    VSA Long-Term Memory       │  │
                    │  │  • Episodic (past findings)   │  │
                    │  │  • Semantic (domain rules)    │  │
                    │  │  • Procedural (learned acts)  │  │
                    │  └───────────────────────────────┘  │
                    │                                     │
                    └─────────────────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────────┐
                    │         Action Dispatcher           │
                    │  • Alert generation                 │
                    │  • Report synthesis                 │
                    │  • Reorder recommendations          │
                    │  • Fraud flag escalation            │
                    └─────────────────────────────────────┘
```

---

## Novel Primitives Specification

### 1. N-Bind (Negation Binding)

**Purpose:** Represent "not X" in VSA space for exclusion queries and contradiction detection.

**Mathematical Definition:**
```python
def n_bind(v: torch.Tensor) -> torch.Tensor:
    """
    Negation binding: Create anti-vector that is maximally dissimilar.

    For FHRR (complex phasors): π phase shift
        n_bind(v)[i] = e^(i(θ_v[i] + π)) = -v[i]

    Properties:
        - sim(v, n_bind(v)) = -1 (maximally dissimilar)
        - n_bind(n_bind(v)) = v (involutory)
        - bundle(v, n_bind(v)) ≈ 0 (cancellation)
    """
    return -v  # For complex phasors, negation = π phase shift
```

**Use Cases:**
- "Find SKUs that are NOT dead_item AND NOT overstock"
- Contradiction detection: If `sim(fact, n_bind(expected)) > threshold`, flag inconsistency
- Exclusion queries in resonator

**Integration with Existing Code:**
```python
# In operators.py
def query_excluding(bundle: Tensor, include: Tensor, exclude: Tensor) -> Tensor:
    """Query bundle for items matching 'include' but not 'exclude'."""
    query = bind(include, n_bind(exclude))
    return unbind(bundle, query)
```

---

### 2. CW-Bundle (Confidence-Weighted Bundling)

**Purpose:** Bundle facts with confidence scores that decay appropriately and enable probabilistic retrieval.

**Mathematical Definition:**
```python
def cw_bundle(
    vectors: List[torch.Tensor],
    confidences: List[float],
    temperature: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Confidence-weighted bundle with learned magnitude encoding.

    Instead of just weighting the sum, encode confidence in magnitude:
        result[i] = Σ (c_j^τ · v_j[i])

    Where τ (temperature) controls confidence sharpness:
        - τ < 1: Amplify high-confidence items
        - τ = 1: Linear weighting
        - τ > 1: Smooth out confidence differences

    Returns:
        (bundle_vector, confidence_vector) - confidence per dimension
    """
    # Normalize confidences to [0, 1]
    c = torch.tensor(confidences)
    c = torch.clamp(c, 0, 1)

    # Apply temperature
    c_scaled = torch.pow(c, 1.0 / temperature)

    # Weight each vector
    result = torch.zeros_like(vectors[0])
    conf_accumulator = torch.zeros(vectors[0].shape[0], dtype=torch.float)

    for v, conf in zip(vectors, c_scaled):
        result = result + conf * v
        conf_accumulator = conf_accumulator + conf

    # Normalize but preserve confidence signal
    bundle_vec = normalize(result)
    conf_vec = conf_accumulator / len(vectors)  # Average confidence per dim

    return bundle_vec, conf_vec
```

**Use Cases:**
- Accumulate evidence with varying certainty
- "Low confidence" detections don't pollute high-confidence findings
- Enables "soft" voting in ensemble systems

**Integration:**
```python
# In core.py - replace fixed-weight bundling
def bundle_anomaly_evidence(detections: List[Detection]) -> Tuple[Tensor, Tensor]:
    """Bundle detected anomalies with their confidence scores."""
    vectors = [d.vector for d in detections]
    confidences = [d.confidence for d in detections]
    return cw_bundle(vectors, confidences, temperature=0.5)
```

---

### 3. T-Bind (Temporal Binding)

**Purpose:** Encode temporal relationships with decay, enabling causal reasoning and trend detection.

**Mathematical Definition:**
```python
def t_bind(
    v: torch.Tensor,
    timestamp: float,
    reference_time: float,
    decay_rate: float = 0.1,
    max_shift: int = 1000
) -> torch.Tensor:
    """
    Temporal binding with exponential decay and position encoding.

    Combines:
    1. Temporal decay: Recent events have higher magnitude
    2. Position encoding: Events at different times are distinguishable

    Formula:
        t_bind(v, t) = decay(t) · permute(v, pos(t))

    Where:
        - decay(t) = exp(-λ · (t_ref - t))
        - pos(t) = hash(t) mod max_shift
    """
    # Compute decay factor
    time_delta = reference_time - timestamp
    decay_factor = math.exp(-decay_rate * time_delta)

    # Compute temporal position shift (deterministic hash)
    import hashlib
    t_hash = int(hashlib.sha256(str(timestamp).encode()).hexdigest()[:8], 16)
    shift = t_hash % max_shift

    # Apply decay and shift
    decayed = decay_factor * v
    shifted = permute(decayed, shift)

    return shifted


def t_unbind(
    bundle: torch.Tensor,
    timestamp: float,
    reference_time: float,
    decay_rate: float = 0.1,
    max_shift: int = 1000
) -> torch.Tensor:
    """Inverse of t_bind for temporal queries."""
    time_delta = reference_time - timestamp
    decay_factor = math.exp(-decay_rate * time_delta)

    import hashlib
    t_hash = int(hashlib.sha256(str(timestamp).encode()).hexdigest()[:8], 16)
    shift = t_hash % max_shift

    # Reverse operations
    unshifted = inverse_permute(bundle, shift)
    undecayed = unshifted / (decay_factor + 1e-10)

    return undecayed
```

**Use Cases:**
- "What anomalies appeared in the last 7 days?"
- "Is margin_erosion trending upward?" (query across time slices)
- Causal inference: "Did low_stock precede stockout?"

**Integration:**
```python
# In streaming.py - temporal context for agent
class TemporalWorkingMemory:
    """Rolling temporal window of VSA facts."""

    def __init__(self, window_days: int = 30, decay_rate: float = 0.1):
        self.window_days = window_days
        self.decay_rate = decay_rate
        self.temporal_bundle = None
        self.reference_time = time.time()

    def add_event(self, fact_vector: Tensor, timestamp: float):
        """Add timestamped fact to temporal memory."""
        t_encoded = t_bind(fact_vector, timestamp, self.reference_time, self.decay_rate)
        if self.temporal_bundle is None:
            self.temporal_bundle = t_encoded
        else:
            self.temporal_bundle = bundle(self.temporal_bundle, t_encoded)

    def query_time_range(self, query: Tensor, start: float, end: float) -> List[float]:
        """Query for matching facts within time range."""
        scores = []
        # Sample time points within range
        for t in np.linspace(start, end, 100):
            probe = t_unbind(self.temporal_bundle, t, self.reference_time)
            score = similarity(probe, query)
            scores.append((t, score))
        return scores
```

---

### 4. P-Sup (Probabilistic Superposition)

**Purpose:** Maintain multiple hypotheses simultaneously with probability weights, enabling Bayesian-style reasoning.

**Mathematical Definition:**
```python
@dataclass
class HypothesisBundle:
    """Probabilistic superposition of hypotheses."""
    vector: torch.Tensor          # Weighted superposition
    hypotheses: List[str]         # Hypothesis labels
    probabilities: torch.Tensor   # Probability per hypothesis
    basis_vectors: torch.Tensor   # Individual hypothesis vectors


def p_sup(
    hypotheses: List[Tuple[str, torch.Tensor, float]]
) -> HypothesisBundle:
    """
    Create probabilistic superposition of hypotheses.

    Args:
        hypotheses: List of (label, vector, prior_probability)

    Returns:
        HypothesisBundle with weighted superposition

    Mathematical Properties:
        - Probabilities sum to 1
        - Querying with evidence updates probabilities (Bayesian update)
        - Collapse to winner when confidence exceeds threshold
    """
    labels = [h[0] for h in hypotheses]
    vectors = torch.stack([h[1] for h in hypotheses])
    probs = torch.tensor([h[2] for h in hypotheses])

    # Normalize probabilities
    probs = probs / probs.sum()

    # Create weighted superposition
    # Use sqrt(prob) as weight to preserve interference patterns
    weights = torch.sqrt(probs)
    superposition = torch.einsum('h,hd->d', weights.to(vectors.dtype), vectors)
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
    likelihood_fn: Callable[[Tensor, Tensor], float] = similarity
) -> HypothesisBundle:
    """
    Bayesian update of hypothesis probabilities given evidence.

    P(H|E) ∝ P(E|H) · P(H)

    Where P(E|H) = similarity(evidence, hypothesis_vector)
    """
    # Compute likelihoods
    likelihoods = torch.tensor([
        likelihood_fn(evidence, h_vec)
        for h_vec in bundle.basis_vectors
    ])

    # Clamp to positive (similarity can be negative)
    likelihoods = torch.clamp(likelihoods, min=0.01)

    # Bayesian update
    posterior = likelihoods * bundle.probabilities
    posterior = posterior / posterior.sum()

    # Rebuild superposition with updated weights
    weights = torch.sqrt(posterior)
    new_superposition = torch.einsum(
        'h,hd->d',
        weights.to(bundle.basis_vectors.dtype),
        bundle.basis_vectors
    )

    return HypothesisBundle(
        vector=normalize(new_superposition),
        hypotheses=bundle.hypotheses,
        probabilities=posterior,
        basis_vectors=bundle.basis_vectors
    )


def p_sup_collapse(bundle: HypothesisBundle, threshold: float = 0.9) -> Optional[str]:
    """
    Collapse superposition if one hypothesis exceeds threshold.

    Returns winning hypothesis label or None if still uncertain.
    """
    max_prob, max_idx = torch.max(bundle.probabilities, dim=0)
    if max_prob >= threshold:
        return bundle.hypotheses[max_idx]
    return None
```

**Use Cases:**
- "Is this SKU dead_item OR seasonal_low?" - Maintain both until evidence resolves
- Root cause analysis with multiple competing explanations
- Gradual disambiguation as data accumulates

---

### 5. SE-Bind (Schema Evolution Binding)

**Purpose:** Handle evolving schemas (new columns, renamed fields) without breaking existing vectors.

**Mathematical Definition:**
```python
@dataclass
class SchemaRegistry:
    """Registry of field mappings across schema versions."""
    version: str
    field_vectors: Dict[str, torch.Tensor]
    aliases: Dict[str, str]  # old_name -> canonical_name
    transformations: Dict[str, Callable]  # field -> transform_fn


def se_bind(
    value_vector: torch.Tensor,
    field_name: str,
    schema: SchemaRegistry,
    version: Optional[str] = None
) -> torch.Tensor:
    """
    Schema-evolution-aware binding.

    Ensures backward compatibility:
    1. Resolves field aliases to canonical names
    2. Applies version-specific transformations
    3. Binds with canonical field vector

    This allows:
    - 'qty' and 'quantity' and 'on_hand' to map to same semantic slot
    - Schema v1 data to remain queryable after v2 migration
    """
    # Resolve alias
    canonical = schema.aliases.get(field_name, field_name)

    # Get field vector (or create if new field)
    if canonical not in schema.field_vectors:
        # New field - create random orthogonal vector
        schema.field_vectors[canonical] = random_vector(value_vector.shape[0])

    field_vec = schema.field_vectors[canonical]

    # Apply transformation if exists
    if canonical in schema.transformations:
        value_vector = schema.transformations[canonical](value_vector)

    return bind(field_vec, value_vector)


def migrate_bundle(
    old_bundle: torch.Tensor,
    old_schema: SchemaRegistry,
    new_schema: SchemaRegistry
) -> torch.Tensor:
    """
    Migrate a bundle from old schema to new schema.

    For each field that changed:
    1. Unbind with old field vector
    2. Rebind with new field vector
    """
    result = old_bundle.clone()

    for old_field, new_field in new_schema.aliases.items():
        if old_field in old_schema.field_vectors:
            # Unbind old
            old_vec = old_schema.field_vectors[old_field]
            unbound = unbind(result, old_vec)

            # Rebind new
            new_vec = new_schema.field_vectors.get(new_field, old_vec)
            rebound = bind(unbound, new_vec)

            result = bundle(result, rebound)

    return normalize(result)
```

**Use Cases:**
- POS system upgrade changes column names
- Multi-store integration with different schemas
- Backward-compatible queries across historical data

---

## Agent Event Loop

### Core Loop Structure

```python
import asyncio
from dataclasses import dataclass
from typing import AsyncIterator
from watchfiles import awatch


@dataclass
class AgentConfig:
    """Configuration for always-on agent."""
    watch_directories: List[Path]
    webhook_port: int = 8080
    scan_interval_hours: float = 24
    working_memory_window_days: int = 30
    alert_threshold: float = 0.7
    dimensions: int = 8192


class AlwaysOnAgent:
    """
    Event-driven agent with VSA working memory.

    Processes events from multiple sources:
    1. File system changes (new inventory exports)
    2. Webhooks (real-time POS transactions)
    3. Scheduled scans (periodic full analysis)
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.working_memory = TemporalWorkingMemory(
            window_days=config.working_memory_window_days
        )
        self.long_term_memory = VSALongTermMemory(dimensions=config.dimensions)
        self.hypothesis_state: Optional[HypothesisBundle] = None
        self.running = False

    async def start(self):
        """Start the agent event loop."""
        self.running = True

        # Create task group for concurrent event sources
        async with asyncio.TaskGroup() as tg:
            # File watcher
            tg.create_task(self._watch_files())

            # Webhook server
            tg.create_task(self._run_webhook_server())

            # Scheduled scanner
            tg.create_task(self._scheduled_scan())

            # Hypothesis updater (processes working memory)
            tg.create_task(self._hypothesis_loop())

    async def _watch_files(self):
        """Watch for new inventory files."""
        async for changes in awatch(*self.config.watch_directories):
            for change_type, path in changes:
                if path.endswith(('.csv', '.tsv')):
                    await self._process_file(Path(path))

    async def _process_file(self, filepath: Path):
        """Process new inventory file."""
        logger.info(f"Processing new file: {filepath}")

        # Run streaming analysis
        result = process_large_file(filepath, dimensions=self.config.dimensions)

        # Update working memory with findings
        timestamp = time.time()
        for primitive, leaks in result.top_leaks_by_primitive.items():
            for sku, score in leaks:
                if score > self.config.alert_threshold:
                    # Create fact vector
                    fact = self._create_fact_vector(primitive, sku, score)
                    self.working_memory.add_event(fact, timestamp)

        # Store in long-term memory
        await self.long_term_memory.store_episode(
            event_type="file_analysis",
            bundle=result,
            timestamp=timestamp
        )

    async def _hypothesis_loop(self):
        """Continuously update hypotheses based on working memory."""
        while self.running:
            if self.working_memory.temporal_bundle is not None:
                # Check for emerging patterns
                patterns = self._detect_patterns()

                if patterns:
                    # Update or create hypothesis bundle
                    self.hypothesis_state = self._update_hypotheses(patterns)

                    # Check for collapse (high-confidence conclusion)
                    winner = p_sup_collapse(self.hypothesis_state)
                    if winner:
                        await self._trigger_alert(winner)

            await asyncio.sleep(60)  # Check every minute

    def _detect_patterns(self) -> List[Tuple[str, torch.Tensor, float]]:
        """Detect emerging patterns in working memory."""
        patterns = []

        # Query for each primitive
        primitives = ['margin_erosion', 'shrinkage_pattern', 'dead_item']
        for prim in primitives:
            prim_vec = self.ctx.get_primitive(prim)
            score = similarity(self.working_memory.temporal_bundle, prim_vec)
            if score > 0.3:
                patterns.append((prim, prim_vec, score))

        return patterns

    def _update_hypotheses(
        self,
        patterns: List[Tuple[str, torch.Tensor, float]]
    ) -> HypothesisBundle:
        """Update hypothesis bundle with new evidence."""
        if self.hypothesis_state is None:
            # Initialize with uniform priors
            return p_sup([(p[0], p[1], 1.0/len(patterns)) for p in patterns])

        # Bayesian update with pattern evidence
        for pattern_name, pattern_vec, score in patterns:
            self.hypothesis_state = p_sup_update(
                self.hypothesis_state,
                pattern_vec
            )

        return self.hypothesis_state
```

---

## Long-Term Memory Architecture

### VSA-Indexed Persistent Memory

```python
class VSALongTermMemory:
    """
    Persistent memory with VSA-indexed retrieval.

    Three memory types:
    1. Episodic: Past analysis results, timestamped
    2. Semantic: Domain rules, thresholds, relationships
    3. Procedural: Learned response patterns
    """

    def __init__(self, dimensions: int, storage_path: Path):
        self.dimensions = dimensions
        self.storage = SQLiteStorage(storage_path)

        # In-memory indices (VSA bundles)
        self.episodic_index = None
        self.semantic_index = None
        self.procedural_index = None

    async def store_episode(
        self,
        event_type: str,
        bundle: StreamingResult,
        timestamp: float
    ):
        """Store episodic memory with VSA index."""
        # Create episode vector
        episode_vec = self._create_episode_vector(event_type, bundle)

        # Store in database
        episode_id = await self.storage.insert_episode(
            timestamp=timestamp,
            event_type=event_type,
            data=bundle.to_audit_json(),
            vector=episode_vec.numpy()
        )

        # Update VSA index
        t_encoded = t_bind(episode_vec, timestamp, time.time())
        if self.episodic_index is None:
            self.episodic_index = t_encoded
        else:
            self.episodic_index = bundle(self.episodic_index, t_encoded)

    async def recall_similar(
        self,
        query: torch.Tensor,
        memory_type: str = "episodic",
        top_k: int = 5
    ) -> List[Dict]:
        """Recall memories similar to query."""
        index = getattr(self, f"{memory_type}_index")
        if index is None:
            return []

        # Use resonator for cleanup
        resonator = Resonator()
        # ... setup codebook from stored vectors
        result = resonator.resonate(unbind(index, query))

        # Fetch full records
        records = await self.storage.fetch_by_ids(
            [match[0] for match in result.top_matches[:top_k]]
        )
        return records
```

---

## Integration Points

### 1. Existing Sentinel Engine

```python
# In sentinel_engine/agent.py (new file)
from sentinel_engine.streaming import process_large_file
from sentinel_engine.context import create_analysis_context
from vsa_core.operators import bind, unbind, bundle
from vsa_core.advanced_primitives import (  # New module
    n_bind, cw_bundle, t_bind, p_sup, se_bind
)

class SentinelAgent(AlwaysOnAgent):
    """Profit Sentinel specific agent implementation."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.ctx = create_analysis_context(dimensions=config.dimensions)

    async def _trigger_alert(self, finding: str):
        """Generate alert for high-confidence finding."""
        # Query long-term memory for similar past events
        similar = await self.long_term_memory.recall_similar(
            self.ctx.get_primitive(finding)
        )

        # Generate alert with context
        alert = {
            "type": finding,
            "confidence": float(self.hypothesis_state.probabilities.max()),
            "similar_past_events": len(similar),
            "recommendation": self._generate_recommendation(finding, similar)
        }

        # Dispatch via configured channels
        await self.alert_dispatcher.send(alert)
```

### 2. API Integration

```python
# In apps/api/routes/agent.py
from fastapi import APIRouter
from sentinel_engine.agent import SentinelAgent

router = APIRouter(prefix="/agent")

@router.post("/start")
async def start_agent(config: AgentConfigRequest):
    """Start the always-on agent."""
    agent = SentinelAgent(AgentConfig(**config.dict()))
    asyncio.create_task(agent.start())
    return {"status": "started"}

@router.get("/status")
async def get_agent_status():
    """Get current agent status and hypothesis state."""
    return {
        "running": agent.running,
        "working_memory_size": agent.working_memory.size(),
        "active_hypotheses": agent.hypothesis_state.hypotheses if agent.hypothesis_state else [],
        "top_hypothesis": p_sup_collapse(agent.hypothesis_state, 0.5) if agent.hypothesis_state else None
    }

@router.post("/query")
async def query_memory(query: MemoryQueryRequest):
    """Query agent memory."""
    results = await agent.long_term_memory.recall_similar(
        query.vector,
        memory_type=query.memory_type,
        top_k=query.top_k
    )
    return {"results": results}
```

---

## Implementation Roadmap

### Phase 1: Novel Primitives (Week 1)
- [ ] Implement N-Bind in `vsa_core/operators.py`
- [ ] Implement CW-Bundle in `vsa_core/operators.py`
- [ ] Implement T-Bind in `vsa_core/temporal.py` (new)
- [ ] Implement P-Sup in `vsa_core/probabilistic.py` (new)
- [ ] Implement SE-Bind in `vsa_core/schema.py` (new)
- [ ] Unit tests for each primitive

### Phase 2: Agent Core (Week 2)
- [ ] Create `sentinel_engine/agent.py`
- [ ] Implement TemporalWorkingMemory
- [ ] Implement event loop with file watching
- [ ] Add webhook receiver

### Phase 3: Memory Layer (Week 3)
- [ ] Create `sentinel_engine/memory.py`
- [ ] Implement SQLite storage backend
- [ ] Implement VSA-indexed retrieval
- [ ] Add memory consolidation (episodic → semantic)

### Phase 4: Integration (Week 4)
- [ ] API endpoints for agent control
- [ ] Dashboard for agent monitoring
- [ ] Alert dispatcher integration
- [ ] Production deployment configuration

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Event processing latency | <100ms | Single transaction webhook |
| File processing | <30s/10k rows | Streaming mode |
| Memory recall | <50ms | VSA-indexed query |
| Hypothesis update | <10ms | Bayesian update |
| Working memory size | <500MB | 30-day window |

---

## Open Questions

1. **Memory consolidation strategy**: How aggressively should episodic memories be compressed into semantic rules?
2. **Hypothesis decay**: Should inactive hypotheses decay over time?
3. **Multi-store coordination**: How to share learnings across store instances?
4. **GPU acceleration**: Is the agent loop GPU-bound or I/O-bound?

---

**Document Status:** Ready for implementation feedback
**Next Action:** Begin Phase 1 implementation of novel primitives
