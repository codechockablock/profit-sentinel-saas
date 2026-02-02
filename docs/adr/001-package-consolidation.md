# ADR 001: Package Consolidation

## Status

Accepted

## Date

2026-02-01

## Context

The codebase had accumulated duplicate code paths and stale artifacts from rapid iteration:

1. **`profit_sentinel_package/`** - A stale fork of the sentinel-engine package that diverged from `packages/sentinel-engine/src/sentinel_engine/`. Files existed in both locations with different content, creating confusion about the source of truth.

2. **`frontend/`** - An orphaned folder containing only `index.html` and `env-config.js.template`. The actual frontend code lives in `apps/web/`.

3. **`public/`** - Default Next.js assets (`vercel.svg`, `next.svg`, etc.) that were never customized or used. The web app in `apps/web/` has its own `public/` directory.

4. **`inventory_report.html`** - A 513KB generated test artifact that was accidentally committed.

5. **`apps/api/src/config.py`** - A 1,118-line file where ~900 lines were static column mapping data (`STANDARD_FIELDS` dict). This data belonged in configuration files, not code.

6. **`packages/sentinel-engine/src/sentinel_engine/__init__.py`** - Complex availability checking logic scattered throughout 450+ lines, making it difficult to understand which components were available.

### Note: `packages/sentinel-core/` was initially preserved

The original review flagged `packages/sentinel-core/` as "empty", but investigation revealed it's a **git submodule** pointing to `profit-sentinel-core.git` containing:
- 40K+ lines of VSA validation experiments
- Referenced in `.github/workflows/deploy.yml`

This was preserved in ADR 001, but later removed in ADR 002 (sentinel-core consolidation) after verifying:
- No direct imports from the submodule in production code
- The only active module (`semantic_flagging.py`) was already migrated to `sentinel-engine.flagging`
- The deploy workflow fetches `core.py` separately (not via the submodule)

## Decision

### 1. Delete Stale/Duplicate Code

Remove the following:
- `profit_sentinel_package/` - Use `packages/sentinel-engine/` as single source of truth
- `frontend/` - Real frontend is `apps/web/`
- `public/` - Unused default assets
- `inventory_report.html` - Generated test artifact

### 2. Extract Column Mappings to YAML

Move `STANDARD_FIELDS`, `FIELD_IMPORTANCE`, and `SUPPORTED_POS_SYSTEMS` from `config.py` to:
- `config/pos_mappings/standard_fields.yaml`
- `config/pos_mappings/field_importance.yaml`
- `config/pos_mappings/supported_systems.yaml`

Create a loader utility at `apps/api/src/utils/column_mappings.py` with:
- `load_standard_fields()` - Cached YAML loader
- `load_field_importance()` - Cached importance rankings
- `load_supported_systems()` - Cached POS system list
- `get_field_aliases(field_name)` - Convenience function
- `get_field_importance(field_name)` - Convenience function

### 3. Centralize Availability Checking

Create `packages/sentinel-engine/src/sentinel_engine/_availability.py` to centralize all component availability flags:
- `_DORIAN_AVAILABLE`
- `_DIAGNOSTIC_AVAILABLE`
- `_CORE_AVAILABLE`
- `_VSA_EVIDENCE_AVAILABLE`
- etc.

The main `__init__.py` imports from this module for cleaner organization.

## Consequences

### Positive

- **Single source of truth**: `packages/sentinel-engine/` is the only sentinel package
- **Simpler mental model**: Contributors know where code lives
- **Configuration changes don't require code deploys**: YAML files can be updated independently
- **Enables future enhancements**:
  - A/B testing different column mappings
  - Customer-specific mapping overrides
  - Analytics on mapping effectiveness
- **Reduced `config.py` from 1,118 lines to 238 lines**: Easier to maintain and review

### Negative

- **External documentation may reference old paths**: Need to update any docs pointing to `profit_sentinel_package/`
- **Slight startup overhead**: YAML parsing on first access (mitigated by `lru_cache`)
- **New dependency on PyYAML**: Already present in requirements, but now critical path

### Neutral

- **Backward compatibility preserved**: All existing imports continue to work
- **Test coverage unchanged**: Existing tests pass without modification

## Implementation Notes

The cleanup was performed in logical commits:

1. `chore: remove duplicate profit_sentinel_package` - Deleted stale code
2. `refactor: extract column mappings to YAML` - Data externalization
3. `refactor: centralize sentinel-engine availability checks` - Code organization

## References

- Original architectural review that identified these issues
- ~~Git submodule configuration in `.gitmodules`~~ (removed in ADR 002)
- Deployment workflow in `.github/workflows/deploy.yml`
- See also: ADR 002 - Sentinel-Core Consolidation
