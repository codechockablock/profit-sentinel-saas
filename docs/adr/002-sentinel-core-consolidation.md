# ADR 002: Sentinel-Core Consolidation

## Status

Accepted

## Date

2026-02-01

## Context

Following ADR 001 (Package Consolidation), the repository still contained a git submodule `packages/sentinel-core` pointing to `git@github.com:codechockablock/profit-sentinel-core.git`.

Investigation revealed:
- **~24K lines** of code in the submodule
- **Only ONE module actively used**: `semantic_flagging.py` (761 lines)
- **No direct imports** from `sentinel_core` anywhere in the main codebase
- The `semantic_flagging.py` module was **already migrated** to `sentinel_engine.flagging`
- The deploy workflow fetches `core.py` **separately** (shallow clone, not via submodule)

The submodule added complexity without providing value to the production system:
- Required special handling during clone (`git submodule update --init`)
- Created confusion about which package was canonical
- Contained mostly archived research code marked as "legacy"

## Decision

### 1. Remove the Git Submodule

- Deinitialize and remove `packages/sentinel-core`
- Remove `.gitmodules` file (now empty)
- Keep the `profit-sentinel-core` GitHub repository archived for historical reference

### 2. Verify Migration Completeness

Before removal, verify:
- `sentinel_engine.flagging` exports all necessary classes (`SemanticFlagDetector`, `FlagCategory`, `FlagSeverity`, etc.)
- No imports from `sentinel_core` exist in production code
- Deploy workflow continues to work (fetches `core.py` independently)

### 3. Preserve Deploy Workflow

The deploy workflow fetches `core.py` from the private repo:
```yaml
git clone --depth 1 https://${CORE_REPO_PAT}@github.com/codechockablock/profit-sentinel-core.git
cp /tmp/core-repo/core.py packages/sentinel-engine/src/sentinel_engine/core.py
```

This is **independent of the submodule** and should continue unchanged. The private repo remains the source of truth for proprietary VSA detection logic.

## Consequences

### Positive

- **Single source of truth**: `packages/sentinel-engine` is THE VSA package
- **Simplified cloning**: No submodule initialization required
- **Clearer architecture**: One package for all VSA functionality
- **Reduced confusion**: No ambiguity about which flagging module to use

### Negative

- **Research code less accessible**: Must clone archived repo separately to view experiments
- **Historical context lost in main repo**: VSA evolution not visible in git history

### Neutral

- **Deploy workflow unchanged**: Still fetches `core.py` from private repo
- **No code changes required**: Migration was already complete

## Verification

After removal:
```bash
# Verify no submodules
git submodule status  # Should return nothing

# Verify imports work
python -c "from sentinel_engine import SemanticFlagDetector, _FLAGGING_AVAILABLE; print(f'OK: {_FLAGGING_AVAILABLE}')"

# Verify packages structure
ls packages/  # Should show: reasoning, sentinel-engine, vsa-core
```

## Post-Merge Actions

1. Archive `profit-sentinel-core` on GitHub:
   - Settings → General → Danger Zone → Archive this repository
2. Update any external documentation referencing the submodule
3. Remove `CORE_REPO_PAT` secret only if `core.py` fetching is deprecated

## References

- ADR 001: Package Consolidation (initial cleanup)
- Git submodule documentation
- Deploy workflow: `.github/workflows/deploy.yml`
