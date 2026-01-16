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
    vsa_core.geodesic_resonator- Riemannian geometry-aware resonator
    vsa_core.manifolds         - Riemannian geometry operations
    vsa_core.loader            - YAML primitive/rule loading
    vsa_core.types             - Pydantic type definitions
"""

__version__ = "2.0.0"

# Core vector operations
from .vectors import (
    normalize,
    seed_hash,
    random_vector,
    identity_vector,
    zero_vector,
    similarity,
    batch_similarity,
    vector_info,
    get_device,
    get_config,
    configure,
)

# Algebraic operators
from .operators import (
    bind,
    bind_many,
    bundle,
    bundle_many,
    weighted_bundle,
    unbind,
    unbind_from_bundle,
    permute,
    sequence_encode,
    solve_analogy,
    role_filler_bind,
    create_record,
    query_record,
)

# Resonator
from .resonator import Resonator

# Types
from .types import (
    VectorConfig,
    ResonatorConfig,
    Primitive,
    PrimitiveSet,
    MagnitudeConfig,
    MagnitudeBucket,
    MagnitudeField,
    Rule,
    RuleSet,
    RuleCondition,
    RuleDetection,
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
