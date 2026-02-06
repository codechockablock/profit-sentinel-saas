"""
vsa_core - Hyperdimensional Computing / Vector Symbolic Architecture Library

A modular, geometrically-aware VSA implementation for symbolic reasoning.

Quick Start:
    from vsa_core import seed_hash, bind, bundle, Resonator
    from vsa_core.loader import PrimitiveLoader

    # Generate vectors
    sku = seed_hash("SKU123")
    anomaly = seed_hash("primitive:low_stock")

    # Bind fact
    fact = bind(sku, anomaly)

    # Load primitives from YAML
    loader = PrimitiveLoader()
    loader.load_file("primitives/retail_inventory.yaml")

    # Query
    resonator = Resonator()
    resonator.set_codebook(labels, vectors)
    result = resonator.resonate(fact)
    print(result.top_matches)

Modules:
    vsa_core.vectors           - Vector generation and normalization
    vsa_core.operators         - Algebraic operations (bind, bundle, permute)
    vsa_core.resonator         - Iterative query cleanup
    vsa_core.loader            - YAML primitive/rule loading
    vsa_core.types             - Pydantic type definitions
    vsa_core.probabilistic     - Probabilistic superposition (P-Sup)
    vsa_core.schema            - Schema evolution binding (SE-Bind)

Novel Primitives (v3.1.0):
    - n_bind: Negation binding for exclusion queries
    - cw_bundle: Confidence-weighted bundling
    - t_bind/t_unbind: Temporal binding with decay
    - p_sup: Probabilistic superposition of hypotheses
    - se_bind: Schema-evolution-aware binding

Note: geodesic_resonator and manifolds modules were removed in v3.0.0 (unused complexity).
"""

__version__ = "3.1.0"

# Core vector operations
# Algebraic operators
from .operators import (
    bind,
    bind_many,
    bundle,
    bundle_many,
    create_record,
    cw_bundle,
    # Novel primitives (v3.1.0)
    n_bind,
    permute,
    query_excluding,
    query_record,
    role_filler_bind,
    sequence_encode,
    solve_analogy,
    t_bind,
    t_unbind,
    unbind,
    unbind_from_bundle,
    weighted_bundle,
)

# Probabilistic superposition
from .probabilistic import (
    HypothesisBundle,
    p_sup,
    p_sup_add_hypothesis,
    p_sup_collapse,
    p_sup_merge,
    p_sup_remove_hypothesis,
    p_sup_update,
)

# Resonator
from .resonator import Resonator

# Schema evolution
from .schema import (
    FieldSpec,
    SchemaRegistry,
    create_retail_schema,
    create_schema_record,
    migrate_bundle,
    schema_compatibility_check,
    se_bind,
    se_unbind,
)

# Types
from .types import (
    MagnitudeBucket,
    MagnitudeConfig,
    MagnitudeField,
    Primitive,
    PrimitiveSet,
    ResonatorConfig,
    Rule,
    RuleCondition,
    RuleDetection,
    RuleSet,
    VectorConfig,
)
from .vectors import (
    batch_similarity,
    configure,
    get_config,
    get_device,
    identity_vector,
    normalize,
    random_vector,
    seed_hash,
    similarity,
    vector_info,
    zero_vector,
)

__all__ = [
    # Version
    "__version__",
    # Vectors
    "normalize",
    "seed_hash",
    "random_vector",
    "identity_vector",
    "zero_vector",
    "similarity",
    "batch_similarity",
    "vector_info",
    "get_device",
    "get_config",
    "configure",
    # Operators
    "bind",
    "bind_many",
    "bundle",
    "bundle_many",
    "weighted_bundle",
    "unbind",
    "unbind_from_bundle",
    "permute",
    "sequence_encode",
    "solve_analogy",
    "role_filler_bind",
    "create_record",
    "query_record",
    # Novel primitives (v3.1.0)
    "n_bind",
    "query_excluding",
    "cw_bundle",
    "t_bind",
    "t_unbind",
    # Probabilistic superposition
    "HypothesisBundle",
    "p_sup",
    "p_sup_update",
    "p_sup_collapse",
    "p_sup_add_hypothesis",
    "p_sup_remove_hypothesis",
    "p_sup_merge",
    # Schema evolution
    "SchemaRegistry",
    "FieldSpec",
    "se_bind",
    "se_unbind",
    "create_schema_record",
    "migrate_bundle",
    "schema_compatibility_check",
    "create_retail_schema",
    # Resonator
    "Resonator",
    # Types
    "VectorConfig",
    "ResonatorConfig",
    "Primitive",
    "PrimitiveSet",
    "MagnitudeConfig",
    "MagnitudeBucket",
    "MagnitudeField",
    "Rule",
    "RuleSet",
    "RuleCondition",
    "RuleDetection",
]
