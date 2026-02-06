"""
DORIAN ONTOLOGY
===============

The categorical structure of reality.
What kinds of things exist and how they relate.

This is the foundation that all domain knowledge builds upon.

Structure:
1. UPPER ONTOLOGY - The most abstract categories
   - Entity, Process, Property, Relation, Abstract, Concrete

2. MID-LEVEL ONTOLOGY - Domain-spanning categories
   - Physical Object, Mental State, Event, Quantity, Structure

3. DOMAIN ONTOLOGIES - Specific to each field
   - Mathematical Objects, Physical Entities, Mental Phenomena, etc.

4. RELATION TYPES - How things can relate
   - is_a, part_of, causes, enables, requires, etc.

5. AXIOMS - Fundamental truths about the categories
   - Properties of relations (transitive, symmetric, etc.)
   - Constraints (mutual exclusion, exhaustive partitions)

Philosophy:
- Inspired by DOLCE, BFO, SUMO, Cyc
- Designed for machine reasoning, not human readability
- Every fact in the system has a place in this hierarchy

Author: Joseph + Claude
Date: 2026-01-25
"""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

# =============================================================================
# PART 1: RELATION PROPERTIES
# =============================================================================


class RelationProperty(Enum):
    """Properties that relations can have."""

    # Basic properties
    REFLEXIVE = auto()  # xRx for all x
    IRREFLEXIVE = auto()  # not xRx for any x
    SYMMETRIC = auto()  # xRy implies yRx
    ASYMMETRIC = auto()  # xRy implies not yRx
    ANTISYMMETRIC = auto()  # xRy and yRx implies x=y
    TRANSITIVE = auto()  # xRy and yRz implies xRz

    # Special properties
    FUNCTIONAL = auto()  # Each x has at most one y such that xRy
    INVERSE_FUNCTIONAL = auto()  # Each y has at most one x such that xRy

    # Domain properties
    INHERITABLE = auto()  # If A is_a B and B R C, then A R C
    EXCLUSIVE = auto()  # If A R B, then A cannot R C (for C ≠ B)


@dataclass
class RelationType:
    """Definition of a relation type."""

    name: str
    description: str

    # What can be related
    domain: str = "entity"  # Type of subject
    range: str = "entity"  # Type of object

    # Properties
    properties: set[RelationProperty] = field(default_factory=set)

    # Inverse relation (if any)
    inverse: str | None = None

    # Implies other relations
    implies: list[str] = field(default_factory=list)

    # Mutually exclusive with
    excludes: list[str] = field(default_factory=list)

    def is_transitive(self) -> bool:
        return RelationProperty.TRANSITIVE in self.properties

    def is_symmetric(self) -> bool:
        return RelationProperty.SYMMETRIC in self.properties

    def is_inheritable(self) -> bool:
        return RelationProperty.INHERITABLE in self.properties


# =============================================================================
# PART 2: CATEGORY DEFINITION
# =============================================================================


@dataclass
class Category:
    """A category in the ontology."""

    name: str
    description: str

    # Hierarchy
    parent: str | None = None
    children: list[str] = field(default_factory=list)

    # Level in hierarchy
    level: int = 0

    # Domain this belongs to
    domain: str = "upper"

    # Defining properties (what makes something this category)
    defining_properties: list[tuple[str, str]] = field(default_factory=list)

    # Typical properties (usually true but not defining)
    typical_properties: list[tuple[str, str]] = field(default_factory=list)

    # Disjoint with (cannot be both)
    disjoint_with: list[str] = field(default_factory=list)

    # Is this an abstract category or can it have instances?
    is_abstract: bool = False

    def __hash__(self):
        return hash(self.name)


# =============================================================================
# PART 3: THE ONTOLOGY
# =============================================================================


class DorianOntology:
    """
    The complete ontological structure.

    Provides:
    - Category hierarchy
    - Relation definitions
    - Inference rules
    - Consistency checking
    """

    def __init__(self):
        # Categories
        self.categories: dict[str, Category] = {}
        self.category_children: dict[str, set[str]] = defaultdict(set)

        # Relations
        self.relations: dict[str, RelationType] = {}

        # Domain registrations
        self.domains: set[str] = set()

        # Build the ontology
        self._build_upper_ontology()
        self._build_relation_types()
        self._build_mid_level_ontology()
        self._build_domain_ontologies()

    # =========================================================================
    # UPPER ONTOLOGY - Most abstract categories
    # =========================================================================

    def _build_upper_ontology(self):
        """Build the uppermost categories."""

        # ROOT
        self._add_category(
            Category(
                name="thing",
                description="The most general category. Everything is a thing.",
                level=0,
                domain="upper",
                is_abstract=True,
            )
        )

        # LEVEL 1: Primary division
        self._add_category(
            Category(
                name="entity",
                description="Something that exists independently.",
                parent="thing",
                level=1,
                domain="upper",
                is_abstract=True,
            )
        )

        self._add_category(
            Category(
                name="abstract",
                description="Something that exists outside of space and time.",
                parent="thing",
                level=1,
                domain="upper",
                disjoint_with=["concrete"],
                is_abstract=True,
            )
        )

        self._add_category(
            Category(
                name="concrete",
                description="Something that exists in space and time.",
                parent="thing",
                level=1,
                domain="upper",
                disjoint_with=["abstract"],
                is_abstract=True,
            )
        )

        # LEVEL 2: Under Entity
        self._add_category(
            Category(
                name="particular",
                description="A specific, individual entity.",
                parent="entity",
                level=2,
                domain="upper",
                disjoint_with=["universal"],
            )
        )

        self._add_category(
            Category(
                name="universal",
                description="A type, kind, or category of entities.",
                parent="entity",
                level=2,
                domain="upper",
                disjoint_with=["particular"],
            )
        )

        # LEVEL 2: Under Abstract
        self._add_category(
            Category(
                name="mathematical_object",
                description="Abstract objects studied by mathematics.",
                parent="abstract",
                level=2,
                domain="upper",
            )
        )

        self._add_category(
            Category(
                name="proposition",
                description="Something that can be true or false.",
                parent="abstract",
                level=2,
                domain="upper",
            )
        )

        self._add_category(
            Category(
                name="property",
                description="A characteristic that entities can have.",
                parent="abstract",
                level=2,
                domain="upper",
            )
        )

        self._add_category(
            Category(
                name="relation",
                description="A connection between entities.",
                parent="abstract",
                level=2,
                domain="upper",
            )
        )

        self._add_category(
            Category(
                name="structure",
                description="An organized arrangement of elements.",
                parent="abstract",
                level=2,
                domain="upper",
            )
        )

        # LEVEL 2: Under Concrete
        self._add_category(
            Category(
                name="physical_object",
                description="A concrete object with mass and extension.",
                parent="concrete",
                level=2,
                domain="upper",
            )
        )

        self._add_category(
            Category(
                name="process",
                description="Something that unfolds over time.",
                parent="concrete",
                level=2,
                domain="upper",
            )
        )

        self._add_category(
            Category(
                name="event",
                description="A discrete occurrence in space-time.",
                parent="concrete",
                level=2,
                domain="upper",
            )
        )

        self._add_category(
            Category(
                name="state",
                description="A condition that persists over time.",
                parent="concrete",
                level=2,
                domain="upper",
            )
        )

        self._add_category(
            Category(
                name="region",
                description="A portion of space or time.",
                parent="concrete",
                level=2,
                domain="upper",
            )
        )

    # =========================================================================
    # RELATION TYPES
    # =========================================================================

    def _build_relation_types(self):
        """Define the fundamental relation types."""

        # TAXONOMIC RELATIONS
        self._add_relation(
            RelationType(
                name="is_a",
                description="Category membership. X is_a Y means X is an instance of Y.",
                properties={RelationProperty.TRANSITIVE, RelationProperty.ASYMMETRIC},
                implies=["related_to"],
            )
        )

        self._add_relation(
            RelationType(
                name="subtype_of",
                description="Category subsumption. X subtype_of Y means every X is a Y.",
                properties={
                    RelationProperty.TRANSITIVE,
                    RelationProperty.ASYMMETRIC,
                    RelationProperty.INHERITABLE,
                },
                inverse="supertype_of",
                implies=["related_to"],
            )
        )

        self._add_relation(
            RelationType(
                name="supertype_of",
                description="Inverse of subtype_of.",
                properties={RelationProperty.TRANSITIVE, RelationProperty.ASYMMETRIC},
                inverse="subtype_of",
            )
        )

        self._add_relation(
            RelationType(
                name="instance_of",
                description="X instance_of Y means X is a particular of type Y.",
                properties={RelationProperty.ASYMMETRIC},
                domain="particular",
                range="universal",
            )
        )

        # MEREOLOGICAL RELATIONS (Part-whole)
        self._add_relation(
            RelationType(
                name="part_of",
                description="X part_of Y means X is a component of Y.",
                properties={
                    RelationProperty.TRANSITIVE,
                    RelationProperty.ASYMMETRIC,
                    RelationProperty.IRREFLEXIVE,
                },
                inverse="has_part",
                implies=["related_to"],
            )
        )

        self._add_relation(
            RelationType(
                name="has_part",
                description="X has_part Y means Y is a component of X.",
                properties={RelationProperty.TRANSITIVE, RelationProperty.ASYMMETRIC},
                inverse="part_of",
            )
        )

        self._add_relation(
            RelationType(
                name="proper_part_of",
                description="X proper_part_of Y means X is part of Y and X ≠ Y.",
                properties={
                    RelationProperty.TRANSITIVE,
                    RelationProperty.ASYMMETRIC,
                    RelationProperty.IRREFLEXIVE,
                },
                implies=["part_of"],
            )
        )

        self._add_relation(
            RelationType(
                name="component_of",
                description="Functional part. X component_of Y means X is a part that contributes to Y's function.",
                properties={RelationProperty.ASYMMETRIC},
                implies=["part_of"],
            )
        )

        self._add_relation(
            RelationType(
                name="member_of",
                description="X member_of Y means X is an element of collection Y.",
                properties={RelationProperty.ASYMMETRIC},
                implies=["part_of"],
            )
        )

        # CAUSAL RELATIONS
        self._add_relation(
            RelationType(
                name="causes",
                description="X causes Y means X brings about Y.",
                properties={
                    RelationProperty.TRANSITIVE,
                    RelationProperty.ASYMMETRIC,
                    RelationProperty.IRREFLEXIVE,
                },
                inverse="caused_by",
                domain="event",
                range="event",
                implies=["precedes", "related_to"],
            )
        )

        self._add_relation(
            RelationType(
                name="caused_by",
                description="Inverse of causes.",
                properties={RelationProperty.TRANSITIVE, RelationProperty.ASYMMETRIC},
                inverse="causes",
            )
        )

        self._add_relation(
            RelationType(
                name="enables",
                description="X enables Y means X makes Y possible (but doesn't guarantee it).",
                properties={RelationProperty.ASYMMETRIC},
                implies=["related_to"],
            )
        )

        self._add_relation(
            RelationType(
                name="prevents",
                description="X prevents Y means X makes Y impossible or unlikely.",
                properties={RelationProperty.ASYMMETRIC},
                excludes=["enables", "causes"],
            )
        )

        self._add_relation(
            RelationType(
                name="influences",
                description="X influences Y means X affects Y in some way.",
                properties={RelationProperty.ASYMMETRIC},
                implies=["related_to"],
            )
        )

        # TEMPORAL RELATIONS
        self._add_relation(
            RelationType(
                name="precedes",
                description="X precedes Y means X occurs before Y.",
                properties={
                    RelationProperty.TRANSITIVE,
                    RelationProperty.ASYMMETRIC,
                    RelationProperty.IRREFLEXIVE,
                },
                inverse="follows",
                domain="event",
                range="event",
            )
        )

        self._add_relation(
            RelationType(
                name="follows",
                description="X follows Y means X occurs after Y.",
                properties={RelationProperty.TRANSITIVE, RelationProperty.ASYMMETRIC},
                inverse="precedes",
            )
        )

        self._add_relation(
            RelationType(
                name="simultaneous_with",
                description="X simultaneous_with Y means X and Y occur at the same time.",
                properties={
                    RelationProperty.SYMMETRIC,
                    RelationProperty.REFLEXIVE,
                    RelationProperty.TRANSITIVE,
                },
            )
        )

        self._add_relation(
            RelationType(
                name="during",
                description="X during Y means X occurs within the time span of Y.",
                properties={RelationProperty.TRANSITIVE, RelationProperty.ASYMMETRIC},
            )
        )

        # SPATIAL RELATIONS
        self._add_relation(
            RelationType(
                name="located_at",
                description="X located_at Y means X is at location Y.",
                properties={RelationProperty.ASYMMETRIC},
                domain="physical_object",
                range="region",
            )
        )

        self._add_relation(
            RelationType(
                name="adjacent_to",
                description="X adjacent_to Y means X is next to Y.",
                properties={RelationProperty.SYMMETRIC},
            )
        )

        self._add_relation(
            RelationType(
                name="contains",
                description="X contains Y means Y is inside X.",
                properties={RelationProperty.TRANSITIVE, RelationProperty.ASYMMETRIC},
                inverse="contained_in",
            )
        )

        self._add_relation(
            RelationType(
                name="contained_in",
                description="X contained_in Y means X is inside Y.",
                properties={RelationProperty.TRANSITIVE, RelationProperty.ASYMMETRIC},
                inverse="contains",
            )
        )

        # PROPERTY RELATIONS
        self._add_relation(
            RelationType(
                name="has_property",
                description="X has_property Y means X possesses characteristic Y.",
                properties={RelationProperty.ASYMMETRIC, RelationProperty.INHERITABLE},
                range="property",
            )
        )

        self._add_relation(
            RelationType(
                name="has_value",
                description="X has_value Y means X has quantity/measurement Y.",
                properties={RelationProperty.ASYMMETRIC},
            )
        )

        self._add_relation(
            RelationType(
                name="has_quality",
                description="X has_quality Y means X exhibits quality Y.",
                properties={RelationProperty.ASYMMETRIC, RelationProperty.INHERITABLE},
                implies=["has_property"],
            )
        )

        # FUNCTIONAL RELATIONS
        self._add_relation(
            RelationType(
                name="has_function",
                description="X has_function Y means X is designed/used to do Y.",
                properties={RelationProperty.ASYMMETRIC},
            )
        )

        self._add_relation(
            RelationType(
                name="used_for",
                description="X used_for Y means X is employed to achieve Y.",
                properties={RelationProperty.ASYMMETRIC},
                implies=["related_to"],
            )
        )

        self._add_relation(
            RelationType(
                name="capable_of",
                description="X capable_of Y means X can perform or undergo Y.",
                properties={RelationProperty.ASYMMETRIC, RelationProperty.INHERITABLE},
            )
        )

        self._add_relation(
            RelationType(
                name="requires",
                description="X requires Y means X needs Y to exist or function.",
                properties={RelationProperty.ASYMMETRIC},
                implies=["related_to"],
            )
        )

        # IDENTITY AND SIMILARITY
        self._add_relation(
            RelationType(
                name="same_as",
                description="X same_as Y means X and Y are identical.",
                properties={
                    RelationProperty.SYMMETRIC,
                    RelationProperty.REFLEXIVE,
                    RelationProperty.TRANSITIVE,
                },
            )
        )

        self._add_relation(
            RelationType(
                name="similar_to",
                description="X similar_to Y means X resembles Y in some respect.",
                properties={RelationProperty.SYMMETRIC},
            )
        )

        self._add_relation(
            RelationType(
                name="different_from",
                description="X different_from Y means X is not identical to Y.",
                properties={RelationProperty.SYMMETRIC, RelationProperty.IRREFLEXIVE},
                excludes=["same_as"],
            )
        )

        self._add_relation(
            RelationType(
                name="opposite_of",
                description="X opposite_of Y means X is the antithesis of Y.",
                properties={RelationProperty.SYMMETRIC},
                implies=["different_from"],
            )
        )

        # GENERIC
        self._add_relation(
            RelationType(
                name="related_to",
                description="X related_to Y means there is some connection between X and Y.",
                properties={RelationProperty.SYMMETRIC},
            )
        )

        # EPISTEMIC RELATIONS
        self._add_relation(
            RelationType(
                name="implies",
                description="X implies Y means if X is true, Y must be true.",
                properties={RelationProperty.TRANSITIVE, RelationProperty.ASYMMETRIC},
                domain="proposition",
                range="proposition",
            )
        )

        self._add_relation(
            RelationType(
                name="contradicts",
                description="X contradicts Y means X and Y cannot both be true.",
                properties={RelationProperty.SYMMETRIC},
                domain="proposition",
                range="proposition",
            )
        )

        self._add_relation(
            RelationType(
                name="supports",
                description="X supports Y means X provides evidence for Y.",
                properties={RelationProperty.ASYMMETRIC},
                domain="proposition",
                range="proposition",
            )
        )

        self._add_relation(
            RelationType(
                name="equivalent_to",
                description="X equivalent_to Y means X and Y have the same truth conditions.",
                properties={
                    RelationProperty.SYMMETRIC,
                    RelationProperty.TRANSITIVE,
                    RelationProperty.REFLEXIVE,
                },
                domain="proposition",
                range="proposition",
            )
        )

    # =========================================================================
    # MID-LEVEL ONTOLOGY
    # =========================================================================

    def _build_mid_level_ontology(self):
        """Build domain-spanning categories below upper ontology."""

        # PHYSICAL OBJECTS
        self._add_category(
            Category(
                name="living_thing",
                description="A physical object that is alive.",
                parent="physical_object",
                level=3,
                domain="mid",
            )
        )

        self._add_category(
            Category(
                name="non_living_thing",
                description="A physical object that is not alive.",
                parent="physical_object",
                level=3,
                domain="mid",
                disjoint_with=["living_thing"],
            )
        )

        self._add_category(
            Category(
                name="artifact",
                description="An object created by intentional design.",
                parent="physical_object",
                level=3,
                domain="mid",
            )
        )

        self._add_category(
            Category(
                name="natural_object",
                description="An object not created by intentional design.",
                parent="physical_object",
                level=3,
                domain="mid",
            )
        )

        self._add_category(
            Category(
                name="substance",
                description="A kind of matter (water, iron, etc.).",
                parent="physical_object",
                level=3,
                domain="mid",
            )
        )

        # LIVING THINGS
        self._add_category(
            Category(
                name="organism",
                description="A living individual.",
                parent="living_thing",
                level=4,
                domain="mid",
            )
        )

        self._add_category(
            Category(
                name="animal",
                description="A living organism that can move and sense.",
                parent="organism",
                level=5,
                domain="mid",
            )
        )

        self._add_category(
            Category(
                name="plant",
                description="A living organism that photosynthesizes.",
                parent="organism",
                level=5,
                domain="mid",
                disjoint_with=["animal"],
            )
        )

        self._add_category(
            Category(
                name="human",
                description="A human being.",
                parent="animal",
                level=6,
                domain="mid",
            )
        )

        # MATHEMATICAL OBJECTS (deeper)
        self._add_category(
            Category(
                name="number",
                description="A mathematical quantity.",
                parent="mathematical_object",
                level=3,
                domain="mathematics",
            )
        )

        self._add_category(
            Category(
                name="set",
                description="A collection of distinct objects.",
                parent="mathematical_object",
                level=3,
                domain="mathematics",
            )
        )

        self._add_category(
            Category(
                name="function",
                description="A mapping from one set to another.",
                parent="mathematical_object",
                level=3,
                domain="mathematics",
            )
        )

        self._add_category(
            Category(
                name="mathematical_structure",
                description="A set with operations and relations.",
                parent="mathematical_object",
                level=3,
                domain="mathematics",
            )
        )

        self._add_category(
            Category(
                name="proof",
                description="A logical argument establishing truth.",
                parent="mathematical_object",
                level=3,
                domain="mathematics",
            )
        )

        # PROCESSES
        self._add_category(
            Category(
                name="physical_process",
                description="A process involving physical change.",
                parent="process",
                level=3,
                domain="mid",
            )
        )

        self._add_category(
            Category(
                name="mental_process",
                description="A process occurring in the mind.",
                parent="process",
                level=3,
                domain="mid",
            )
        )

        self._add_category(
            Category(
                name="social_process",
                description="A process involving multiple agents.",
                parent="process",
                level=3,
                domain="mid",
            )
        )

        self._add_category(
            Category(
                name="computation",
                description="An information processing process.",
                parent="process",
                level=3,
                domain="mid",
            )
        )

        # MENTAL
        self._add_category(
            Category(
                name="mental_state",
                description="A state of mind.",
                parent="state",
                level=3,
                domain="psychology",
            )
        )

        self._add_category(
            Category(
                name="belief",
                description="A mental state of holding something true.",
                parent="mental_state",
                level=4,
                domain="psychology",
            )
        )

        self._add_category(
            Category(
                name="desire",
                description="A mental state of wanting something.",
                parent="mental_state",
                level=4,
                domain="psychology",
            )
        )

        self._add_category(
            Category(
                name="emotion",
                description="A mental state involving feeling.",
                parent="mental_state",
                level=4,
                domain="psychology",
            )
        )

        self._add_category(
            Category(
                name="intention",
                description="A mental state of planning to do something.",
                parent="mental_state",
                level=4,
                domain="psychology",
            )
        )

        # PROPOSITIONS
        self._add_category(
            Category(
                name="fact",
                description="A true proposition about the world.",
                parent="proposition",
                level=3,
                domain="upper",
            )
        )

        self._add_category(
            Category(
                name="law",
                description="A proposition that holds universally.",
                parent="proposition",
                level=3,
                domain="upper",
            )
        )

        self._add_category(
            Category(
                name="theory",
                description="A system of propositions explaining phenomena.",
                parent="proposition",
                level=3,
                domain="upper",
            )
        )

        self._add_category(
            Category(
                name="hypothesis",
                description="A proposition proposed for testing.",
                parent="proposition",
                level=3,
                domain="upper",
            )
        )

        # STRUCTURES
        self._add_category(
            Category(
                name="system",
                description="An organized collection of interacting components.",
                parent="structure",
                level=3,
                domain="mid",
            )
        )

        self._add_category(
            Category(
                name="pattern",
                description="A regular arrangement or sequence.",
                parent="structure",
                level=3,
                domain="mid",
            )
        )

        self._add_category(
            Category(
                name="hierarchy",
                description="A structure with levels of subordination.",
                parent="structure",
                level=3,
                domain="mid",
            )
        )

        self._add_category(
            Category(
                name="network",
                description="A structure of interconnected nodes.",
                parent="structure",
                level=3,
                domain="mid",
            )
        )

    # =========================================================================
    # DOMAIN ONTOLOGIES
    # =========================================================================

    def _build_domain_ontologies(self):
        """Build domain-specific categories."""

        # =====================================================================
        # MATHEMATICS
        # =====================================================================
        self.domains.add("mathematics")

        # Numbers
        self._add_category(
            Category(
                name="natural_number",
                description="Positive integers: 1, 2, 3, ...",
                parent="number",
                level=4,
                domain="mathematics",
            )
        )

        self._add_category(
            Category(
                name="integer",
                description="Whole numbers: ..., -2, -1, 0, 1, 2, ...",
                parent="number",
                level=4,
                domain="mathematics",
            )
        )

        self._add_category(
            Category(
                name="rational_number",
                description="Numbers expressible as a/b where a,b are integers.",
                parent="number",
                level=4,
                domain="mathematics",
            )
        )

        self._add_category(
            Category(
                name="real_number",
                description="Numbers on the continuous number line.",
                parent="number",
                level=4,
                domain="mathematics",
            )
        )

        self._add_category(
            Category(
                name="complex_number",
                description="Numbers of the form a + bi.",
                parent="number",
                level=4,
                domain="mathematics",
            )
        )

        # Structures
        self._add_category(
            Category(
                name="group",
                description="A set with an associative binary operation, identity, and inverses.",
                parent="mathematical_structure",
                level=4,
                domain="mathematics",
            )
        )

        self._add_category(
            Category(
                name="ring",
                description="A set with two binary operations (addition and multiplication).",
                parent="mathematical_structure",
                level=4,
                domain="mathematics",
            )
        )

        self._add_category(
            Category(
                name="field",
                description="A ring where every non-zero element has a multiplicative inverse.",
                parent="mathematical_structure",
                level=4,
                domain="mathematics",
            )
        )

        self._add_category(
            Category(
                name="vector_space",
                description="A set of vectors with scalar multiplication and addition.",
                parent="mathematical_structure",
                level=4,
                domain="mathematics",
            )
        )

        self._add_category(
            Category(
                name="topological_space",
                description="A set with a topology defining open sets.",
                parent="mathematical_structure",
                level=4,
                domain="mathematics",
            )
        )

        # Operations
        self._add_category(
            Category(
                name="operation",
                description="A function that takes inputs and produces an output.",
                parent="function",
                level=4,
                domain="mathematics",
            )
        )

        self._add_category(
            Category(
                name="binary_operation",
                description="An operation taking two inputs.",
                parent="operation",
                level=5,
                domain="mathematics",
            )
        )

        self._add_category(
            Category(
                name="unary_operation",
                description="An operation taking one input.",
                parent="operation",
                level=5,
                domain="mathematics",
            )
        )

        # =====================================================================
        # PHYSICS
        # =====================================================================
        self.domains.add("physics")

        self._add_category(
            Category(
                name="physical_quantity",
                description="A measurable property of a physical system.",
                parent="property",
                level=3,
                domain="physics",
            )
        )

        self._add_category(
            Category(
                name="force",
                description="An interaction that causes acceleration.",
                parent="physical_quantity",
                level=4,
                domain="physics",
            )
        )

        self._add_category(
            Category(
                name="energy",
                description="The capacity to do work.",
                parent="physical_quantity",
                level=4,
                domain="physics",
            )
        )

        self._add_category(
            Category(
                name="mass",
                description="A measure of matter in an object.",
                parent="physical_quantity",
                level=4,
                domain="physics",
            )
        )

        self._add_category(
            Category(
                name="momentum",
                description="Mass times velocity.",
                parent="physical_quantity",
                level=4,
                domain="physics",
            )
        )

        self._add_category(
            Category(
                name="charge",
                description="A property causing electromagnetic interaction.",
                parent="physical_quantity",
                level=4,
                domain="physics",
            )
        )

        self._add_category(
            Category(
                name="particle",
                description="A fundamental unit of matter.",
                parent="physical_object",
                level=3,
                domain="physics",
            )
        )

        self._add_category(
            Category(
                name="field",
                description="A physical quantity defined at every point in space.",
                parent="physical_object",
                level=3,
                domain="physics",
            )
        )

        self._add_category(
            Category(
                name="wave",
                description="A propagating disturbance.",
                parent="physical_process",
                level=4,
                domain="physics",
            )
        )

        self._add_category(
            Category(
                name="physical_law",
                description="A universal regularity in physical phenomena.",
                parent="law",
                level=4,
                domain="physics",
            )
        )

        # =====================================================================
        # LOGIC
        # =====================================================================
        self.domains.add("logic")

        self._add_category(
            Category(
                name="logical_connective",
                description="An operator that combines propositions.",
                parent="relation",
                level=3,
                domain="logic",
            )
        )

        self._add_category(
            Category(
                name="quantifier",
                description="An operator specifying quantity (all, some, none).",
                parent="relation",
                level=3,
                domain="logic",
            )
        )

        self._add_category(
            Category(
                name="inference_rule",
                description="A rule for deriving conclusions from premises.",
                parent="relation",
                level=3,
                domain="logic",
            )
        )

        self._add_category(
            Category(
                name="valid_argument",
                description="An argument where conclusion follows from premises.",
                parent="structure",
                level=3,
                domain="logic",
            )
        )

        self._add_category(
            Category(
                name="fallacy",
                description="An invalid form of argument.",
                parent="structure",
                level=3,
                domain="logic",
                disjoint_with=["valid_argument"],
            )
        )

        # =====================================================================
        # LANGUAGE
        # =====================================================================
        self.domains.add("language")

        self._add_category(
            Category(
                name="linguistic_expression",
                description="A unit of language.",
                parent="abstract",
                level=2,
                domain="language",
            )
        )

        self._add_category(
            Category(
                name="word",
                description="A minimal free form in language.",
                parent="linguistic_expression",
                level=3,
                domain="language",
            )
        )

        self._add_category(
            Category(
                name="sentence",
                description="A grammatically complete expression.",
                parent="linguistic_expression",
                level=3,
                domain="language",
            )
        )

        self._add_category(
            Category(
                name="meaning",
                description="The semantic content of an expression.",
                parent="abstract",
                level=2,
                domain="language",
            )
        )

        self._add_category(
            Category(
                name="grammatical_category",
                description="A class of words with similar syntactic behavior.",
                parent="universal",
                level=3,
                domain="language",
            )
        )

        # =====================================================================
        # COMPUTER SCIENCE
        # =====================================================================
        self.domains.add("computer_science")

        self._add_category(
            Category(
                name="algorithm",
                description="A finite sequence of well-defined instructions.",
                parent="computation",
                level=4,
                domain="computer_science",
            )
        )

        self._add_category(
            Category(
                name="data_structure",
                description="A way of organizing data for efficient operations.",
                parent="structure",
                level=4,
                domain="computer_science",
            )
        )

        self._add_category(
            Category(
                name="programming_language",
                description="A formal language for expressing computations.",
                parent="linguistic_expression",
                level=4,
                domain="computer_science",
            )
        )

        self._add_category(
            Category(
                name="software",
                description="Programs and data that run on computers.",
                parent="artifact",
                level=4,
                domain="computer_science",
            )
        )

        self._add_category(
            Category(
                name="complexity_class",
                description="A class of problems with similar computational difficulty.",
                parent="universal",
                level=4,
                domain="computer_science",
            )
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _add_category(self, category: Category):
        """Add a category to the ontology."""
        self.categories[category.name] = category

        if category.parent:
            self.category_children[category.parent].add(category.name)
            if category.parent in self.categories:
                self.categories[category.parent].children.append(category.name)

    def _add_relation(self, relation: RelationType):
        """Add a relation type to the ontology."""
        self.relations[relation.name] = relation

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    def get_category(self, name: str) -> Category | None:
        """Get a category by name."""
        return self.categories.get(name.lower())

    def get_relation(self, name: str) -> RelationType | None:
        """Get a relation type by name."""
        return self.relations.get(name.lower())

    def get_ancestors(self, category_name: str) -> list[str]:
        """Get all ancestors of a category (parent, grandparent, etc.)."""
        ancestors = []
        current = self.categories.get(category_name.lower())

        while current and current.parent:
            ancestors.append(current.parent)
            current = self.categories.get(current.parent)

        return ancestors

    def get_descendants(self, category_name: str) -> list[str]:
        """Get all descendants of a category."""
        descendants = []
        to_visit = list(self.category_children.get(category_name.lower(), []))

        while to_visit:
            child = to_visit.pop(0)
            descendants.append(child)
            to_visit.extend(self.category_children.get(child, []))

        return descendants

    def is_subtype(self, child: str, parent: str) -> bool:
        """Check if child is a subtype of parent."""
        return parent.lower() in self.get_ancestors(child)

    def get_common_ancestor(self, cat1: str, cat2: str) -> str | None:
        """Find the lowest common ancestor of two categories."""
        ancestors1 = set(self.get_ancestors(cat1))
        ancestors1.add(cat1.lower())

        current = cat2.lower()
        while current:
            if current in ancestors1:
                return current
            cat = self.categories.get(current)
            current = cat.parent if cat else None

        return None

    def get_transitive_relations(self) -> list[str]:
        """Get all transitive relations."""
        return [name for name, rel in self.relations.items() if rel.is_transitive()]

    def get_symmetric_relations(self) -> list[str]:
        """Get all symmetric relations."""
        return [name for name, rel in self.relations.items() if rel.is_symmetric()]

    def get_inheritable_relations(self) -> list[str]:
        """Get all inheritable relations."""
        return [name for name, rel in self.relations.items() if rel.is_inheritable()]

    def get_disjoint_categories(self, category_name: str) -> list[str]:
        """Get categories that are disjoint with the given category."""
        cat = self.categories.get(category_name.lower())
        return cat.disjoint_with if cat else []

    # =========================================================================
    # INFERENCE SUPPORT
    # =========================================================================

    def can_infer_transitive(self, relation: str) -> bool:
        """Check if transitive inference applies to a relation."""
        rel = self.relations.get(relation.lower())
        return rel and rel.is_transitive()

    def get_implied_relations(self, relation: str) -> list[str]:
        """Get relations implied by the given relation."""
        rel = self.relations.get(relation.lower())
        return rel.implies if rel else []

    def get_inverse_relation(self, relation: str) -> str | None:
        """Get the inverse of a relation."""
        rel = self.relations.get(relation.lower())
        return rel.inverse if rel else None

    def are_disjoint(self, cat1: str, cat2: str) -> bool:
        """Check if two categories are disjoint."""
        c1 = self.categories.get(cat1.lower())
        c2 = self.categories.get(cat2.lower())

        if not c1 or not c2:
            return False

        return cat2.lower() in c1.disjoint_with or cat1.lower() in c2.disjoint_with

    # =========================================================================
    # EXPORT
    # =========================================================================

    def to_facts(self) -> list[tuple[str, str, str]]:
        """Export ontology as facts that can be loaded into Dorian Core."""
        facts = []

        # Category hierarchy
        for name, cat in self.categories.items():
            if cat.parent:
                facts.append((name, "subtype_of", cat.parent))

            facts.append((name, "is_a", "category"))
            facts.append((name, "in_domain", cat.domain))

            for disj in cat.disjoint_with:
                facts.append((name, "disjoint_with", disj))

        # Relation definitions
        for name, rel in self.relations.items():
            facts.append((name, "is_a", "relation"))

            if rel.domain != "entity":
                facts.append((name, "has_domain", rel.domain))
            if rel.range != "entity":
                facts.append((name, "has_range", rel.range))
            if rel.inverse:
                facts.append((name, "inverse_of", rel.inverse))

            for prop in rel.properties:
                facts.append((name, "has_property", prop.name.lower()))

            for implied in rel.implies:
                facts.append((name, "implies", implied))

        return facts

    def stats(self) -> dict:
        """Get ontology statistics."""
        return {
            "categories": len(self.categories),
            "relations": len(self.relations),
            "domains": len(self.domains),
            "max_depth": max((c.level for c in self.categories.values()), default=0),
            "transitive_relations": len(self.get_transitive_relations()),
            "symmetric_relations": len(self.get_symmetric_relations()),
        }

    def __repr__(self) -> str:
        stats = self.stats()
        return f"DorianOntology(categories={stats['categories']}, relations={stats['relations']}, domains={stats['domains']})"


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

# Create a singleton ontology
_ontology = None


def get_ontology() -> DorianOntology:
    """Get the global ontology instance."""
    global _ontology
    if _ontology is None:
        _ontology = DorianOntology()
    return _ontology


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("═" * 60)
    print("DORIAN ONTOLOGY")
    print("═" * 60)

    ont = get_ontology()

    stats = ont.stats()
    print("\nStatistics:")
    print(f"  Categories: {stats['categories']}")
    print(f"  Relations: {stats['relations']}")
    print(f"  Domains: {stats['domains']}")
    print(f"  Max depth: {stats['max_depth']}")

    print(f"\nDomains: {sorted(ont.domains)}")

    print("\nUpper ontology categories:")
    for name, cat in ont.categories.items():
        if cat.domain == "upper" and cat.level <= 2:
            indent = "  " * cat.level
            print(f"{indent}{name}")

    print("\nTransitive relations:")
    for rel in ont.get_transitive_relations()[:10]:
        print(f"  {rel}")

    print("\nSymmetric relations:")
    for rel in ont.get_symmetric_relations()[:10]:
        print(f"  {rel}")

    print("\nAncestors of 'human':")
    print(f"  {ont.get_ancestors('human')}")

    print("\nDescendants of 'mathematical_object':")
    print(f"  {ont.get_descendants('mathematical_object')}")

    print("\nCommon ancestor of 'human' and 'plant':")
    print(f"  {ont.get_common_ancestor('human', 'plant')}")

    print("\nExporting to facts...")
    facts = ont.to_facts()
    print(f"  Generated {len(facts)} facts")

    print("\nSample facts:")
    import random

    for fact in random.sample(facts, 10):
        print(f"  {fact[0]} {fact[1]} {fact[2]}")

    print("\nDone!")
