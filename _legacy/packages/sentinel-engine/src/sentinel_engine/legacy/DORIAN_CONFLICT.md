# DORIAN CONFLICT - Legacy Sentinel Engine Files

These files have been moved to legacy because they conflict with the new Dorian system.

## Files Moved

| File | Size | Reason |
|------|------|--------|
| `agent.py` | 27KB | Old always-on agent using vsa_core P-Sup operations - replaced by `dorian/agent.py` |
| `context.py` | 68KB | Old AnalysisContext with SparseVSA/DiracVSA - replaced by `dorian/core.py` |

## Conflict Details

### agent.py Conflicts

The old agent.py implements:
- TemporalWorkingMemory with T-Bind decay
- HypothesisEngine with P-Sup probabilistic updates
- AlertDispatcher for notifications

This is replaced by:
- `dorian/agent.py` - Knowledge graph agent with semantic queries
- `diagnostic/agent.py` - Profit Sentinel diagnostic agent
- `diagnostic/engine.py` - Conversational diagnostic flow

### context.py Conflicts

The old context.py implements:
- SparseVector/SparseVSA (sparse optimization)
- DiracVector/DiracVSA (temporal asymmetry with 4 components)
- AnalysisContext (request-scoped state)
- 32 primitives (domain, logical, temporal)

This is replaced by:
- `dorian/core.py` - Production VSA with FAISS indexing, 10M+ fact capacity
- `dorian/ontology.py` - Formal category structure
- `dorian/pipeline.py` - Knowledge loading from multiple sources

## What Remains in sentinel-engine

The following non-conflicting files remain active:
- `core.py` - Column mappings and detection logic (uses Dorian)
- `streaming.py` - Large file processing (adapts to Dorian)
- `batch.py` - Batch utilities (generic)
- `bridge.py` - Bridge patterns (generic)
- `codebook.py` - Codebook management (may need adaptation)
- `flagging.py` - Flagging logic (adapts to Dorian)
- `pipeline.py` - Pipeline utilities (generic)
- `repair_engine.py` - Repair logic (adapts to Dorian)
- `repair_models.py` - Repair models (adapts to Dorian)
- `contradiction_detector.py` - May need adaptation
- `routing/` - Routing logic (generic)
- `vsa_evidence/` - Evidence encoding (may need adaptation)

## Migration Notes

If you need functionality from these legacy files:
1. Check `dorian/agent.py` for agent functionality
2. Check `dorian/core.py` for VSA operations
3. Check `diagnostic/engine.py` for diagnostic flow
4. Do NOT import from legacy - use Dorian API

---
Generated during Dorian integration: 2025-01-25
