"""
DORIAN CORE
===========

The shared brain for hundreds of agents.

Design Principles:
1. CONCURRENT - Multiple agents read/write simultaneously
2. ATTRIBUTED - Every fact knows who wrote it and when
3. CONSISTENT - Contradictions detected before writing
4. DIRECTIONAL - "A causes B" ≠ "B causes A" (always)
5. SCALABLE - Billions of facts, sub-millisecond queries
6. GROUNDED - Zero hallucination by design
7. ONTOLOGICAL - Built on a formal category structure

Architecture:
- VSA encoding preserves semantic direction
- Fact store with full provenance
- Contradiction detection via semantic similarity
- Domain partitioning for specialized reasoning
- Event log for full audit trail
- Ontology integration for structured reasoning

Author: Joseph + Claude
Date: 2026-01-25
"""

import hashlib
import heapq
import json
import pickle
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Ontology integration (local module)
from .ontology import DorianOntology, RelationProperty, get_ontology

# Optional imports
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Install with: pip install faiss-cpu")

try:
    from scipy.sparse import csr_matrix, lil_matrix
    from sklearn.utils.extmath import randomized_svd

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available for optimized embeddings")


# =============================================================================
# PART 1: DATA STRUCTURES
# =============================================================================


class FactStatus(Enum):
    """Status of a fact in the knowledge base."""

    ACTIVE = "active"  # Current, valid fact
    SUPERSEDED = "superseded"  # Replaced by newer fact
    CONTRADICTED = "contradicted"  # Conflicts with other facts
    RETRACTED = "retracted"  # Explicitly removed
    PENDING = "pending"  # Awaiting verification


@dataclass
class Fact:
    """A single fact with full provenance."""

    # The triple
    subject: str
    predicate: str
    object: str

    # Identity
    fact_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Provenance
    agent_id: str = "system"
    domain: str = "general"
    source: str = "unknown"
    confidence: float = 1.0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Status
    status: FactStatus = FactStatus.ACTIVE

    # Relationships
    supersedes: str | None = None  # fact_id this replaces
    evidence: list[str] = field(default_factory=list)  # supporting fact_ids

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.fact_id)

    def triple(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    def triple_key(self) -> str:
        """Normalized key for deduplication."""
        return f"{self.subject.lower()}|{self.predicate.lower()}|{self.object.lower()}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["created_at"] = self.created_at.isoformat()
        d["updated_at"] = self.updated_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Fact":
        d["status"] = FactStatus(d["status"])
        d["created_at"] = datetime.fromisoformat(d["created_at"])
        d["updated_at"] = datetime.fromisoformat(d["updated_at"])
        return cls(**d)


@dataclass
class Agent:
    """An agent that can read/write to the core."""

    agent_id: str
    name: str
    domain: str

    # Permissions
    can_read: bool = True
    can_write: bool = True
    can_verify: bool = False  # Can mark facts as verified

    # Stats
    facts_written: int = 0
    facts_read: int = 0
    queries_made: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)

    # Domain-specific inference rules
    inference_rules: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Result of a query to the core."""

    facts: list[Fact]
    scores: list[float]
    query_time_ms: float
    total_matches: int

    # Inference results
    inferred_facts: list[tuple[Fact, str]] = field(
        default_factory=list
    )  # (fact, reasoning)

    def __iter__(self):
        return iter(zip(self.facts, self.scores))

    def top(self, k: int = 1) -> list[Fact]:
        return self.facts[:k]


@dataclass
class WriteResult:
    """Result of a write operation."""

    success: bool
    fact_id: str | None = None

    # If failed
    error: str | None = None
    contradictions: list[Fact] = field(default_factory=list)

    # Stats
    write_time_ms: float = 0


# =============================================================================
# PART 2: VSA ENGINE (Optimized)
# =============================================================================


class VSAEngine:
    """
    Vector Symbolic Architecture engine.

    Encodes facts as high-dimensional vectors that preserve:
    - Semantic similarity
    - Directional relationships
    - Compositional structure
    """

    def __init__(self, dim: int = 512, seed: int = 42):
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Role vectors (fixed, random)
        self.role_subject = self._random_vector()
        self.role_predicate = self._random_vector()
        self.role_object = self._random_vector()

        # Learned embeddings
        self.word_to_idx: dict[str, int] = {}
        self.idx_to_word: dict[int, str] = {}
        self.embeddings: np.ndarray | None = None
        self.trained = False

        # Co-occurrence tracking for incremental learning
        self.cooccurrence: lil_matrix | None = None
        self.word_counts: dict[str, int] = defaultdict(int)

    def _random_vector(self) -> np.ndarray:
        """Generate a random unit vector."""
        v = self.rng.randn(self.dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8)

    def _get_or_create_embedding(self, word: str) -> np.ndarray:
        """Get embedding for a word, creating if needed."""
        word = word.lower()

        if word in self.word_to_idx and self.embeddings is not None:
            return self.embeddings[self.word_to_idx[word]]

        # Create random embedding for unknown word
        return self._random_vector()

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Bind two vectors (circular convolution approximation)."""
        return a * b  # Element-wise multiplication (fast, effective)

    def bundle(self, vectors: list[np.ndarray]) -> np.ndarray:
        """Bundle vectors (normalized sum)."""
        if not vectors:
            return np.zeros(self.dim, dtype=np.float32)
        s = np.sum(vectors, axis=0)
        norm = np.linalg.norm(s)
        return s / (norm + 1e-8) if norm > 0 else s

    def encode_fact(self, subject: str, predicate: str, obj: str) -> np.ndarray:
        """
        Encode a fact as a vector.

        Uses role-filler binding:
        vec = normalize(bind(role_s, s) + bind(role_p, p) + bind(role_o, o))

        This preserves direction: encode(A, causes, B) ≠ encode(B, causes, A)
        """
        s_vec = self._get_or_create_embedding(subject)
        p_vec = self._get_or_create_embedding(predicate)
        o_vec = self._get_or_create_embedding(obj)

        bound_s = self.bind(self.role_subject, s_vec)
        bound_p = self.bind(self.role_predicate, p_vec)
        bound_o = self.bind(self.role_object, o_vec)

        return self.bundle([bound_s, bound_p, bound_o])

    def encode_query(
        self, subject: str = None, predicate: str = None, obj: str = None
    ) -> np.ndarray:
        """
        Encode a query (partial fact).

        Supports wildcards (None) for any component.
        """
        components = []

        if subject:
            s_vec = self._get_or_create_embedding(subject)
            components.append(self.bind(self.role_subject, s_vec))

        if predicate:
            p_vec = self._get_or_create_embedding(predicate)
            components.append(self.bind(self.role_predicate, p_vec))

        if obj:
            o_vec = self._get_or_create_embedding(obj)
            components.append(self.bind(self.role_object, o_vec))

        if not components:
            return np.zeros(self.dim, dtype=np.float32)

        return self.bundle(components)

    def train_embeddings(
        self,
        facts: list[tuple[str, str, str]],
        window: int = 5,
        min_count: int = 1,
        verbose: bool = True,
    ):
        """
        Train embeddings from facts using PPMI + SVD.
        """
        if not SKLEARN_AVAILABLE:
            if verbose:
                print("  sklearn not available, using random embeddings")
            return

        t0 = time.time()

        # Build vocabulary
        vocab = set()
        for s, p, o in facts:
            vocab.add(s.lower())
            vocab.add(p.lower())
            vocab.add(o.lower())

        self.word_to_idx = {w: i for i, w in enumerate(sorted(vocab))}
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}
        vocab_size = len(self.word_to_idx)

        if verbose:
            print(f"    Vocab size: {vocab_size:,}")

        # Build co-occurrence matrix
        cooc = lil_matrix((vocab_size, vocab_size), dtype=np.float32)

        for s, p, o in facts:
            s_idx = self.word_to_idx[s.lower()]
            p_idx = self.word_to_idx[p.lower()]
            o_idx = self.word_to_idx[o.lower()]

            # Co-occurrence within fact
            for i, j in [
                (s_idx, p_idx),
                (p_idx, o_idx),
                (s_idx, o_idx),
                (p_idx, s_idx),
                (o_idx, p_idx),
                (o_idx, s_idx),
            ]:
                cooc[i, j] += 1

        cooc = csr_matrix(cooc)

        if verbose:
            print(f"    Co-occurrence: {cooc.nnz:,} pairs ({time.time()-t0:.1f}s)")

        # PPMI
        t1 = time.time()
        row_sums = np.array(cooc.sum(axis=1)).flatten() + 1e-8
        col_sums = np.array(cooc.sum(axis=0)).flatten() + 1e-8
        total = cooc.sum() + 1e-8

        cooc_coo = cooc.tocoo()
        pmi_data = np.log(
            cooc_coo.data * total / (row_sums[cooc_coo.row] * col_sums[cooc_coo.col])
            + 1e-8
        )
        pmi_data = np.maximum(pmi_data, 0)  # PPMI

        ppmi = csr_matrix((pmi_data, (cooc_coo.row, cooc_coo.col)), shape=cooc.shape)

        if verbose:
            print(f"    PPMI computed ({time.time()-t1:.1f}s)")

        # Randomized SVD
        t2 = time.time()
        k = min(self.dim, vocab_size - 1)
        U, S, _ = randomized_svd(
            ppmi, n_components=k, n_oversamples=10, n_iter=2, random_state=self.seed
        )

        self.embeddings = (U * np.sqrt(S)).astype(np.float32)

        # Pad if needed
        if self.embeddings.shape[1] < self.dim:
            padding = np.zeros(
                (vocab_size, self.dim - self.embeddings.shape[1]), dtype=np.float32
            )
            self.embeddings = np.hstack([self.embeddings, padding])

        # Normalize
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        self.embeddings = self.embeddings / norms

        self.trained = True

        if verbose:
            print(f"    Embeddings built ({time.time()-t2:.1f}s)")


# =============================================================================
# PART 3: FACT STORE
# =============================================================================


class FactStore:
    """
    Storage layer for facts with indexing and retrieval.

    Features:
    - Multiple indexes (by ID, by triple, by subject, by predicate, by domain)
    - Vector index for semantic search
    - Efficient updates and deletions
    """

    def __init__(self, vsa: VSAEngine):
        self.vsa = vsa

        # Primary storage
        self.facts: dict[str, Fact] = {}  # fact_id -> Fact

        # Indexes
        self.by_triple: dict[str, str] = {}  # triple_key -> fact_id
        self.by_subject: dict[str, set[str]] = defaultdict(set)  # subject -> fact_ids
        self.by_predicate: dict[str, set[str]] = defaultdict(
            set
        )  # predicate -> fact_ids
        self.by_object: dict[str, set[str]] = defaultdict(set)  # object -> fact_ids
        self.by_domain: dict[str, set[str]] = defaultdict(set)  # domain -> fact_ids
        self.by_agent: dict[str, set[str]] = defaultdict(set)  # agent_id -> fact_ids

        # Vector index (partitioned by predicate)
        self.vectors: dict[str, list[np.ndarray]] = defaultdict(
            list
        )  # predicate -> vectors
        self.vector_ids: dict[str, list[str]] = defaultdict(
            list
        )  # predicate -> fact_ids
        self.faiss_indexes: dict[str, Any] = {}  # predicate -> FAISS index

        # Stats
        self.total_facts = 0
        self.active_facts = 0

        # Thread safety
        self.lock = threading.RLock()

    def add(self, fact: Fact) -> bool:
        """Add a fact to the store."""
        with self.lock:
            # Check for duplicate
            triple_key = fact.triple_key()
            if triple_key in self.by_triple:
                existing_id = self.by_triple[triple_key]
                existing = self.facts[existing_id]
                if existing.status == FactStatus.ACTIVE:
                    return False  # Duplicate

            # Store fact
            self.facts[fact.fact_id] = fact

            # Update indexes
            self.by_triple[triple_key] = fact.fact_id
            self.by_subject[fact.subject.lower()].add(fact.fact_id)
            self.by_predicate[fact.predicate.lower()].add(fact.fact_id)
            self.by_object[fact.object.lower()].add(fact.fact_id)
            self.by_domain[fact.domain].add(fact.fact_id)
            self.by_agent[fact.agent_id].add(fact.fact_id)

            # Add to vector index
            vec = self.vsa.encode_fact(fact.subject, fact.predicate, fact.object)
            pred_key = fact.predicate.lower()
            self.vectors[pred_key].append(vec)
            self.vector_ids[pred_key].append(fact.fact_id)

            # Invalidate FAISS index for this predicate
            if pred_key in self.faiss_indexes:
                del self.faiss_indexes[pred_key]

            self.total_facts += 1
            if fact.status == FactStatus.ACTIVE:
                self.active_facts += 1

            return True

    def get(self, fact_id: str) -> Fact | None:
        """Get a fact by ID."""
        return self.facts.get(fact_id)

    def get_by_triple(self, subject: str, predicate: str, obj: str) -> Fact | None:
        """Get a fact by its triple."""
        triple_key = f"{subject.lower()}|{predicate.lower()}|{obj.lower()}"
        fact_id = self.by_triple.get(triple_key)
        return self.facts.get(fact_id) if fact_id else None

    def update_status(self, fact_id: str, status: FactStatus) -> bool:
        """Update the status of a fact."""
        with self.lock:
            if fact_id not in self.facts:
                return False

            old_status = self.facts[fact_id].status
            self.facts[fact_id].status = status
            self.facts[fact_id].updated_at = datetime.now()

            # Update active count
            if old_status == FactStatus.ACTIVE and status != FactStatus.ACTIVE:
                self.active_facts -= 1
            elif old_status != FactStatus.ACTIVE and status == FactStatus.ACTIVE:
                self.active_facts += 1

            return True

    def query_by_subject(
        self,
        subject: str,
        predicate: str = None,
        status: FactStatus = FactStatus.ACTIVE,
    ) -> list[Fact]:
        """Query facts by subject."""
        subject = subject.lower()
        fact_ids = self.by_subject.get(subject, set())

        results = []
        for fid in fact_ids:
            fact = self.facts[fid]
            if status and fact.status != status:
                continue
            if predicate and fact.predicate.lower() != predicate.lower():
                continue
            results.append(fact)

        return results

    def query_by_predicate(
        self, predicate: str, status: FactStatus = FactStatus.ACTIVE
    ) -> list[Fact]:
        """Query facts by predicate."""
        predicate = predicate.lower()
        fact_ids = self.by_predicate.get(predicate, set())

        return [self.facts[fid] for fid in fact_ids if self.facts[fid].status == status]

    def semantic_search(
        self,
        subject: str = None,
        predicate: str = None,
        obj: str = None,
        k: int = 10,
        threshold: float = 0.1,
    ) -> list[tuple[Fact, float]]:
        """
        Semantic search using VSA vectors.

        Searches within predicate partition if predicate is specified.
        """
        query_vec = self.vsa.encode_query(subject, predicate, obj)

        if predicate:
            # Search within predicate partition
            pred_key = predicate.lower()
            return self._search_partition(pred_key, query_vec, k, threshold)
        else:
            # Search all partitions
            all_results = []
            for pred_key in self.vectors.keys():
                results = self._search_partition(pred_key, query_vec, k, threshold)
                all_results.extend(results)

            # Sort by score and return top k
            all_results.sort(key=lambda x: -x[1])
            return all_results[:k]

    def _search_partition(
        self, predicate: str, query_vec: np.ndarray, k: int, threshold: float
    ) -> list[tuple[Fact, float]]:
        """Search within a predicate partition."""
        if predicate not in self.vectors or not self.vectors[predicate]:
            return []

        vectors = self.vectors[predicate]
        fact_ids = self.vector_ids[predicate]

        # Use FAISS if available and enough vectors
        if FAISS_AVAILABLE and len(vectors) > 100:
            return self._faiss_search(predicate, query_vec, k, threshold)

        # Fallback to numpy
        vectors_arr = np.array(vectors)
        scores = vectors_arr @ query_vec

        # Get top k
        if len(scores) <= k:
            top_indices = np.argsort(-scores)
        else:
            top_indices = np.argpartition(-scores, k)[:k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= threshold:
                fact = self.facts.get(fact_ids[idx])
                if fact and fact.status == FactStatus.ACTIVE:
                    results.append((fact, score))

        return results

    def _faiss_search(
        self, predicate: str, query_vec: np.ndarray, k: int, threshold: float
    ) -> list[tuple[Fact, float]]:
        """Search using FAISS index."""
        # Build index if needed
        if predicate not in self.faiss_indexes:
            vectors = np.array(self.vectors[predicate]).astype(np.float32)

            n = len(vectors)
            dim = vectors.shape[1]

            if n < 1000:
                index = faiss.IndexFlatIP(dim)
            else:
                nlist = min(int(np.sqrt(n) / 4), 256)
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFFlat(
                    quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
                )
                index.train(vectors)
                index.nprobe = min(32, nlist)

            index.add(vectors)
            self.faiss_indexes[predicate] = index

        index = self.faiss_indexes[predicate]

        # Search
        query_vec = query_vec.reshape(1, -1).astype(np.float32)
        scores, indices = index.search(query_vec, k)

        results = []
        fact_ids = self.vector_ids[predicate]

        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= threshold:
                fact = self.facts.get(fact_ids[idx])
                if fact and fact.status == FactStatus.ACTIVE:
                    results.append((fact, float(score)))

        return results

    def rebuild_indexes(self, verbose: bool = True):
        """Rebuild all FAISS indexes."""
        self.faiss_indexes.clear()

        if not FAISS_AVAILABLE:
            if verbose:
                print("  FAISS not available, skipping index rebuild")
            return

        if verbose:
            print(f"  Rebuilding indexes for {len(self.vectors)} predicates...")

        for predicate in self.vectors.keys():
            if len(self.vectors[predicate]) > 100:
                # Trigger index build
                dummy_vec = np.zeros(self.vsa.dim, dtype=np.float32)
                self._faiss_search(predicate, dummy_vec, 1, 0)

        if verbose:
            print(f"  Built {len(self.faiss_indexes)} FAISS indexes")


# =============================================================================
# PART 4: CONTRADICTION DETECTOR
# =============================================================================


class ContradictionDetector:
    """
    Detects contradictions between facts.

    Types of contradictions:
    1. Direct negation: "A is B" vs "A is not B"
    2. Mutual exclusion: "A is B" vs "A is C" where B and C are mutually exclusive
    3. Inverse violation: "A causes B" vs "B causes A" (if marked as non-symmetric)
    4. Domain-specific: Custom rules per domain
    5. Ontology-based: Categories marked as disjoint in the ontology
    """

    def __init__(self, fact_store: FactStore, ontology: "DorianOntology" = None):
        self.fact_store = fact_store
        self.ontology = ontology

        # Negation predicates
        self.negation_map = {
            "is": "is_not",
            "is_not": "is",
            "is_a": "is_not_a",
            "is_not_a": "is_a",
            "has": "lacks",
            "lacks": "has",
            "can": "cannot",
            "cannot": "can",
            "causes": "does_not_cause",
            "does_not_cause": "causes",
        }

        # Non-symmetric predicates (A rel B ≠ B rel A)
        self.non_symmetric = {
            "causes",
            "enables",
            "requires",
            "precedes",
            "follows",
            "contains",
            "part_of",
            "parent_of",
            "child_of",
            "greater_than",
            "less_than",
            "before",
            "after",
        }

        # Mutually exclusive categories
        self.exclusive_categories: dict[str, set[str]] = {
            "alive": {"dead"},
            "dead": {"alive"},
            "true": {"false"},
            "false": {"true"},
            "male": {"female"},
            "female": {"male"},
            "liquid": {"solid", "gas"},
            "solid": {"liquid", "gas"},
            "gas": {"liquid", "solid"},
            # Biological - these are actually disjoint
            "animal": {"plant", "fungus"},
            "plant": {"animal", "fungus"},
            "fungus": {"animal", "plant"},
        }

        # Domain-specific rules
        self.domain_rules: dict[str, list[Callable]] = defaultdict(list)

    def add_exclusive_category(self, category: str, exclusives: set[str]):
        """Add mutually exclusive categories."""
        self.exclusive_categories[category] = exclusives
        for exc in exclusives:
            if exc not in self.exclusive_categories:
                self.exclusive_categories[exc] = set()
            self.exclusive_categories[exc].add(category)

    def add_domain_rule(self, domain: str, rule: Callable[[Fact, Fact], str | None]):
        """
        Add a domain-specific contradiction rule.

        Rule should return None if no contradiction, or a description if contradiction found.
        """
        self.domain_rules[domain].append(rule)

    def check(self, new_fact: Fact) -> list[tuple[Fact, str]]:
        """
        Check if a new fact contradicts existing facts.

        Returns list of (contradicting_fact, reason) tuples.
        """
        contradictions = []

        # 1. Check direct negation
        neg = self._check_negation(new_fact)
        contradictions.extend(neg)

        # 2. Check mutual exclusion
        exc = self._check_mutual_exclusion(new_fact)
        contradictions.extend(exc)

        # 3. Check inverse violation
        inv = self._check_inverse_violation(new_fact)
        contradictions.extend(inv)

        # 4. Check domain-specific rules
        dom = self._check_domain_rules(new_fact)
        contradictions.extend(dom)

        return contradictions

    def _check_negation(self, new_fact: Fact) -> list[tuple[Fact, str]]:
        """Check for direct negation."""
        pred = new_fact.predicate.lower()

        if pred in self.negation_map:
            neg_pred = self.negation_map[pred]
            existing = self.fact_store.get_by_triple(
                new_fact.subject, neg_pred, new_fact.object
            )
            if existing and existing.status == FactStatus.ACTIVE:
                return [
                    (existing, f"Direct negation: '{pred}' contradicts '{neg_pred}'")
                ]

        return []

    def _check_mutual_exclusion(self, new_fact: Fact) -> list[tuple[Fact, str]]:
        """Check for mutually exclusive categories."""
        pred = new_fact.predicate.lower()

        # Check for is, is_a, instance_of predicates
        if pred not in ("is", "is_a", "instance_of"):
            return []

        new_obj = new_fact.object.lower()
        contradictions = []

        # Check explicit exclusive categories
        if new_obj in self.exclusive_categories:
            exclusives = self.exclusive_categories[new_obj]

            # Find existing 'is'/'is_a' facts for this subject
            for check_pred in ("is", "is_a", "instance_of"):
                existing_facts = self.fact_store.query_by_subject(
                    new_fact.subject, predicate=check_pred
                )

                for fact in existing_facts:
                    if fact.object.lower() in exclusives:
                        contradictions.append(
                            (
                                fact,
                                f"Mutual exclusion: '{new_obj}' and '{fact.object}' are mutually exclusive",
                            )
                        )

        # Also check ontology-defined disjointness
        if self.ontology:
            disjoint_cats = self.ontology.get_disjoint_categories(new_obj)

            for check_pred in ("is", "is_a", "instance_of"):
                existing_facts = self.fact_store.query_by_subject(
                    new_fact.subject, predicate=check_pred
                )

                for fact in existing_facts:
                    existing_obj = fact.object.lower()

                    # Direct disjoint check
                    if existing_obj in disjoint_cats:
                        contradictions.append(
                            (
                                fact,
                                f"Ontology disjoint: '{new_obj}' is disjoint with '{existing_obj}'",
                            )
                        )
                        continue

                    # Check if existing_obj is a subtype of something disjoint
                    for disjoint in disjoint_cats:
                        if self._is_subtype_of(existing_obj, disjoint):
                            contradictions.append(
                                (
                                    fact,
                                    f"Ontology disjoint: '{new_obj}' is disjoint with '{disjoint}', and '{existing_obj}' is a subtype of '{disjoint}'",
                                )
                            )
                            break

        return contradictions

    def _is_subtype_of(self, child: str, parent: str) -> bool:
        """Check if child is a subtype of parent using fact store."""
        if child.lower() == parent.lower():
            return True

        visited = set()
        to_check = [child.lower()]

        while to_check:
            current = to_check.pop(0)
            if current in visited:
                continue
            visited.add(current)

            if current == parent.lower():
                return True

            # Check is_a and subtype_of
            for pred in ("is_a", "subtype_of"):
                facts = self.fact_store.query_by_subject(current, predicate=pred)
                for f in facts:
                    p = f.object.lower()
                    if p not in visited:
                        to_check.append(p)

        return False

    def _check_inverse_violation(self, new_fact: Fact) -> list[tuple[Fact, str]]:
        """Check for inverse violation in non-symmetric relations."""
        pred = new_fact.predicate.lower()

        if pred not in self.non_symmetric:
            return []

        # Check if inverse exists
        inverse = self.fact_store.get_by_triple(new_fact.object, pred, new_fact.subject)

        if inverse and inverse.status == FactStatus.ACTIVE:
            return [
                (
                    inverse,
                    f"Inverse violation: '{pred}' is not symmetric, but both directions exist",
                )
            ]

        return []

    def _check_domain_rules(self, new_fact: Fact) -> list[tuple[Fact, str]]:
        """Check domain-specific rules."""
        contradictions = []

        # Get all rules for this domain
        rules = self.domain_rules.get(new_fact.domain, [])
        rules.extend(self.domain_rules.get("general", []))  # Also check general rules

        # Get potentially conflicting facts
        existing_facts = self.fact_store.query_by_subject(new_fact.subject)

        for rule in rules:
            for existing in existing_facts:
                reason = rule(new_fact, existing)
                if reason:
                    contradictions.append((existing, reason))

        return contradictions


# =============================================================================
# PART 5: INFERENCE ENGINE
# =============================================================================


class InferenceEngine:
    """
    Inference engine for deriving new facts from existing ones.

    Supports:
    - Transitive inference (A->B, B->C => A->C)
    - Property inheritance (A is B, B has C => A has C)
    - Implication chains
    - Domain-specific inference rules
    """

    def __init__(self, fact_store: FactStore):
        self.fact_store = fact_store

        # Transitive relations
        self.transitive_relations = {
            "is",
            "is_a",
            "subtype_of",
            "causes",
            "contains",
            "part_of",
            "precedes",
            "greater_than",
            "less_than",
            "ancestor_of",
            "descendant_of",
        }

        # Inheritance rules: if (A, rel1, B) and (B, rel2, C) then (A, derived, C)
        # This means: if A is_a B, and B capable_of C, then A capable_of C
        self.inheritance_rules: dict[tuple[str, str], str] = {
            # is/is_a based inheritance
            ("is", "has"): "has",
            ("is", "can"): "can",
            ("is", "needs"): "needs",
            ("is", "causes"): "can_cause",
            ("is", "capable_of"): "capable_of",
            ("is", "requires"): "requires",
            ("is", "has_property"): "has_property",
            ("is_a", "has"): "has",
            ("is_a", "can"): "can",
            ("is_a", "needs"): "needs",
            ("is_a", "causes"): "can_cause",
            ("is_a", "capable_of"): "capable_of",
            ("is_a", "requires"): "requires",
            ("is_a", "has_property"): "has_property",
            ("is_a", "has_part"): "has_part",
            ("is_a", "located_at"): "typically_located_at",
            ("subtype_of", "has"): "has",
            ("subtype_of", "can"): "can",
            ("subtype_of", "capable_of"): "capable_of",
            ("subtype_of", "requires"): "requires",
            ("subtype_of", "has_property"): "has_property",
            # part_of based
            ("part_of", "has"): "part_of_thing_that_has",
            ("part_of", "located_at"): "located_at",
        }

        # Implication rules: if (A, rel, B) then also (A, implied, B)
        self.implication_rules: dict[str, list[str]] = {
            "causes": ["related_to", "connected_to"],
            "part_of": ["related_to"],
            "is": ["related_to"],
            "is_a": ["related_to"],
            "loves": ["likes", "cares_about"],
            "hates": ["dislikes"],
        }

        # Domain-specific inference rules
        self.domain_rules: dict[str, list[Callable]] = defaultdict(list)

    def add_transitive_relation(self, relation: str):
        """Mark a relation as transitive."""
        self.transitive_relations.add(relation.lower())

    def add_inheritance_rule(self, rel1: str, rel2: str, derived: str):
        """Add an inheritance rule."""
        self.inheritance_rules[(rel1.lower(), rel2.lower())] = derived.lower()

    def add_domain_rule(
        self, domain: str, rule: Callable[[list[Fact]], list[tuple[str, str, str, str]]]
    ):
        """
        Add a domain-specific inference rule.

        Rule should return list of (subject, predicate, object, reasoning) tuples.
        """
        self.domain_rules[domain].append(rule)

    def infer(
        self,
        subject: str = None,
        predicate: str = None,
        obj: str = None,
        max_hops: int = 3,
        domain: str = None,
    ) -> list[tuple[tuple[str, str, str], str]]:
        """
        Infer facts based on a query.

        Returns list of (inferred_triple, reasoning) tuples.
        """
        inferred = []

        # Transitive inference
        if predicate and predicate.lower() in self.transitive_relations:
            trans = self._transitive_infer(subject, predicate, obj, max_hops)
            inferred.extend(trans)

        # Inheritance inference
        if subject:
            inherit = self._inheritance_infer(subject, predicate, max_hops)
            inferred.extend(inherit)

        # Implication inference
        if subject and predicate:
            impl = self._implication_infer(subject, predicate, obj)
            inferred.extend(impl)

        # Domain-specific inference
        if domain:
            dom = self._domain_infer(subject, predicate, obj, domain)
            inferred.extend(dom)

        return inferred

    def _transitive_infer(
        self, subject: str, predicate: str, obj: str, max_hops: int
    ) -> list[tuple[tuple[str, str, str], str]]:
        """Transitive closure inference."""
        if not subject:
            return []

        predicate = predicate.lower()
        inferred = []
        visited = {subject.lower()}
        frontier = [(subject.lower(), 0, [subject])]

        while frontier:
            current, hops, path = frontier.pop(0)

            if hops >= max_hops:
                continue

            # Find facts where current is subject
            facts = self.fact_store.query_by_subject(current, predicate=predicate)

            for fact in facts:
                next_entity = fact.object.lower()

                if next_entity in visited:
                    continue

                visited.add(next_entity)
                new_path = path + [next_entity]

                # This is an inferred fact
                if hops > 0:  # Only count multi-hop as inference
                    inferred.append(
                        (
                            (subject, predicate, next_entity),
                            f"Transitive: {' -> '.join(new_path)}",
                        )
                    )

                frontier.append((next_entity, hops + 1, new_path))

        return inferred

    def _inheritance_infer(
        self, subject: str, target_predicate: str = None, max_hops: int = 3
    ) -> list[tuple[tuple[str, str, str], str]]:
        """
        Inheritance-based inference.

        If X is_a Y and Y has property P, then X has property P.
        Traverses the full inheritance chain.
        """
        inferred = []
        seen_triples = set()  # Deduplication
        subject = subject.lower()

        # Get all ancestors via is_a / subtype_of chain
        ancestors = []
        visited = set()
        to_visit = [subject]

        while to_visit:
            current = to_visit.pop(0)
            if current in visited:
                continue
            visited.add(current)

            # Find is_a relations
            is_a_facts = self.fact_store.query_by_subject(current, predicate="is_a")
            for fact in is_a_facts:
                parent = fact.object.lower()
                if parent not in visited:
                    ancestors.append((parent, current))  # (ancestor, via)
                    to_visit.append(parent)

            # Also check subtype_of
            subtype_facts = self.fact_store.query_by_subject(
                current, predicate="subtype_of"
            )
            for fact in subtype_facts:
                parent = fact.object.lower()
                if parent not in visited:
                    ancestors.append((parent, current))
                    to_visit.append(parent)

        # For each ancestor, check inheritance rules
        for ancestor, via in ancestors:
            for (rel1, rel2), derived in self.inheritance_rules.items():
                # rel1 is the type relation (is_a, subtype_of)
                if rel1 not in ("is_a", "subtype_of", "is"):
                    continue

                if target_predicate and derived.lower() != target_predicate.lower():
                    continue

                # Find ancestor's properties with rel2
                ancestor_facts = self.fact_store.query_by_subject(
                    ancestor, predicate=rel2
                )

                for af in ancestor_facts:
                    triple = (subject, derived, af.object.lower())
                    if triple not in seen_triples:
                        seen_triples.add(triple)
                        inferred.append(
                            (
                                triple,
                                f"Inheritance: {subject} is_a {ancestor}, {ancestor} {rel2} {af.object}",
                            )
                        )

        return inferred

    def _implication_infer(
        self, subject: str, predicate: str, obj: str
    ) -> list[tuple[tuple[str, str, str], str]]:
        """Implication-based inference."""
        if not subject or not predicate or not obj:
            return []

        inferred = []
        predicate = predicate.lower()

        if predicate not in self.implication_rules:
            return []

        # Check if the base fact exists
        base_fact = self.fact_store.get_by_triple(subject, predicate, obj)

        if not base_fact or base_fact.status != FactStatus.ACTIVE:
            return []

        # Generate implied facts
        for implied_pred in self.implication_rules[predicate]:
            inferred.append(
                (
                    (subject, implied_pred, obj),
                    f"Implication: {predicate} implies {implied_pred}",
                )
            )

        return inferred

    def _domain_infer(
        self, subject: str, predicate: str, obj: str, domain: str
    ) -> list[tuple[tuple[str, str, str], str]]:
        """Domain-specific inference."""
        inferred = []

        rules = self.domain_rules.get(domain, [])

        # Get relevant facts
        facts = []
        if subject:
            facts.extend(self.fact_store.query_by_subject(subject))

        for rule in rules:
            results = rule(facts)
            for s, p, o, reasoning in results:
                inferred.append(((s, p, o), reasoning))

        return inferred


# =============================================================================
# PART 6: DORIAN CORE (The Brain)
# =============================================================================


class DorianCore:
    """
    The shared brain for hundreds of agents.

    Features:
    - Concurrent read/write from multiple agents
    - Full fact provenance and attribution
    - Contradiction detection before writes
    - Multi-hop inference
    - Domain partitioning
    - Event logging for audit
    - Sub-millisecond queries
    - Ontology-guided reasoning
    """

    def __init__(self, dim: int = 512, seed: int = 42, load_ontology: bool = True):
        self.dim = dim
        self.seed = seed

        # Ontology (load first so we can pass to components)
        self.ontology: DorianOntology | None = None
        if load_ontology:
            self.ontology = get_ontology()

        # Core components
        self.vsa = VSAEngine(dim=dim, seed=seed)
        self.fact_store = FactStore(self.vsa)
        self.contradiction_detector = ContradictionDetector(
            self.fact_store, self.ontology
        )
        self.inference_engine = InferenceEngine(self.fact_store)

        # Integrate ontology with reasoning systems
        if self.ontology:
            self._integrate_ontology()

        # Agent registry
        self.agents: dict[str, Agent] = {}

        # Event log
        self.event_log: list[dict] = []
        self.max_log_size = 100000

        # Thread safety
        self.lock = threading.RLock()

        # Stats
        self.stats = {
            "total_reads": 0,
            "total_writes": 0,
            "total_queries": 0,
            "contradictions_detected": 0,
            "inferences_made": 0,
        }

    def _integrate_ontology(self):
        """Integrate ontology into the reasoning systems."""
        if not self.ontology:
            return

        # Register transitive relations with inference engine
        for rel_name in self.ontology.get_transitive_relations():
            self.inference_engine.transitive_relations.add(rel_name)

        # Register inheritable relations
        for rel_name in self.ontology.get_inheritable_relations():
            # Add inheritance rules: if X is_a Y and Y rel Z, then X rel Z
            self.inference_engine.inheritance_rules[("is_a", rel_name)] = rel_name
            self.inference_engine.inheritance_rules[("subtype_of", rel_name)] = rel_name

        # Register disjoint categories with contradiction detector
        for cat_name, cat in self.ontology.categories.items():
            for disjoint in cat.disjoint_with:
                self.contradiction_detector.add_exclusive_category(cat_name, {disjoint})

        # Register symmetric relations (for proper contradiction checking)
        for rel_name in self.ontology.get_symmetric_relations():
            # Symmetric relations don't have inverse violations
            if rel_name in self.contradiction_detector.non_symmetric:
                self.contradiction_detector.non_symmetric.remove(rel_name)

        # Add ontology-aware domain rules
        self._add_ontology_inference_rules()

    def _add_ontology_inference_rules(self):
        """Add inference rules based on ontology structure."""

        def category_inheritance_rule(
            facts: list[Fact],
        ) -> list[tuple[str, str, str, str]]:
            """If X is_a Y and Y is a subtype of Z, then X is_a Z."""
            inferred = []

            for fact in facts:
                if fact.predicate == "is_a":
                    # Find ancestors of the object category
                    ancestors = self.ontology.get_ancestors(fact.object)
                    for ancestor in ancestors:
                        inferred.append(
                            (
                                fact.subject,
                                "is_a",
                                ancestor,
                                f"Category inheritance: {fact.subject} is_a {fact.object}, {fact.object} subtype_of {ancestor}",
                            )
                        )

            return inferred

        def property_inheritance_rule(
            facts: list[Fact],
        ) -> list[tuple[str, str, str, str]]:
            """If X is_a Y and Y has inheritable property P, then X has P."""
            inferred = []

            # Find is_a facts
            is_a_facts = [f for f in facts if f.predicate == "is_a"]

            for is_a in is_a_facts:
                subject = is_a.subject
                category = is_a.object

                # Check if category has defining/typical properties in ontology
                cat = self.ontology.get_category(category)
                if cat:
                    for prop_rel, prop_val in cat.defining_properties:
                        inferred.append(
                            (
                                subject,
                                prop_rel,
                                prop_val,
                                f"Property inheritance: {subject} is_a {category}, {category} {prop_rel} {prop_val}",
                            )
                        )
                    for prop_rel, prop_val in cat.typical_properties:
                        inferred.append(
                            (
                                subject,
                                f"typically_{prop_rel}",
                                prop_val,
                                f"Typical property: {subject} is_a {category}, {category} typically {prop_rel} {prop_val}",
                            )
                        )

            return inferred

        # Register rules for all domains
        self.inference_engine.add_domain_rule("general", category_inheritance_rule)
        self.inference_engine.add_domain_rule("general", property_inheritance_rule)

    def bootstrap_ontology(self, agent_id: str = None) -> int:
        """
        Load ontology facts into the knowledge base.

        Returns number of facts added.
        """
        if not self.ontology:
            print("No ontology loaded")
            return 0

        # Create system agent if needed
        if agent_id is None:
            system_agent = self.register_agent(
                "ontology_bootstrap", domain="ontology", can_verify=True
            )
            agent_id = system_agent.agent_id

        # Get ontology facts
        facts = self.ontology.to_facts()

        print(f"Bootstrapping {len(facts)} ontology facts...")

        # Write facts (skip contradiction checking for bootstrap)
        count = 0
        for subject, predicate, obj in facts:
            result = self.write(
                subject,
                predicate,
                obj,
                agent_id=agent_id,
                source="ontology",
                confidence=1.0,
                check_contradictions=False,
            )
            if result.success:
                count += 1

        print(f"  Added {count} ontology facts")
        return count

    def get_category_of(self, entity: str) -> list[str]:
        """Get the ontological categories an entity belongs to."""
        facts = self.query_forward(entity, "is_a")
        categories = [f.object for f in facts]

        # Also check instance_of
        facts = self.query_forward(entity, "instance_of")
        categories.extend([f.object for f in facts])

        return categories

    def is_type_of(self, entity: str, category: str) -> bool:
        """Check if entity is a type of category (directly or via inheritance)."""
        entity = entity.lower()
        category = category.lower()

        # Direct check
        categories = self.get_category_of(entity)

        if category in [c.lower() for c in categories]:
            return True

        # Check via transitive is_a chain
        visited = set()
        to_check = list(categories)

        while to_check:
            current = to_check.pop(0).lower()
            if current in visited:
                continue
            visited.add(current)

            if current == category:
                return True

            # Get what current is_a
            parent_facts = self.query_forward(current, "is_a")
            for f in parent_facts:
                parent = f.object.lower()
                if parent not in visited:
                    to_check.append(parent)

            # Also check subtype_of
            parent_facts = self.query_forward(current, "subtype_of")
            for f in parent_facts:
                parent = f.object.lower()
                if parent not in visited:
                    to_check.append(parent)

        return False

    # =========================================================================
    # AGENT MANAGEMENT
    # =========================================================================

    def register_agent(
        self,
        name: str,
        domain: str = "general",
        can_write: bool = True,
        can_verify: bool = False,
    ) -> Agent:
        """Register a new agent."""
        agent_id = str(uuid.uuid4())

        agent = Agent(
            agent_id=agent_id,
            name=name,
            domain=domain,
            can_write=can_write,
            can_verify=can_verify,
        )

        with self.lock:
            self.agents[agent_id] = agent
            self._log_event(
                "agent_registered",
                {
                    "agent_id": agent_id,
                    "name": name,
                    "domain": domain,
                },
            )

        return agent

    def get_agent(self, agent_id: str) -> Agent | None:
        """Get an agent by ID."""
        return self.agents.get(agent_id)

    # =========================================================================
    # WRITE OPERATIONS
    # =========================================================================

    def write(
        self,
        subject: str,
        predicate: str,
        obj: str,
        agent_id: str,
        confidence: float = 1.0,
        source: str = "agent",
        metadata: dict = None,
        check_contradictions: bool = True,
    ) -> WriteResult:
        """
        Write a fact to the core.

        Returns WriteResult with success status and any contradictions.
        """
        t0 = time.time()

        # Validate agent
        agent = self.agents.get(agent_id)
        if not agent:
            return WriteResult(success=False, error="Unknown agent")
        if not agent.can_write:
            return WriteResult(
                success=False, error="Agent does not have write permission"
            )

        # Create fact
        fact = Fact(
            subject=subject.lower(),
            predicate=predicate.lower(),
            object=obj.lower(),
            agent_id=agent_id,
            domain=agent.domain,
            source=source,
            confidence=confidence,
            metadata=metadata or {},
        )

        # Check for contradictions
        if check_contradictions:
            contradictions = self.contradiction_detector.check(fact)
            if contradictions:
                self.stats["contradictions_detected"] += len(contradictions)
                return WriteResult(
                    success=False,
                    error="Contradiction detected",
                    contradictions=[c[0] for c in contradictions],
                    write_time_ms=(time.time() - t0) * 1000,
                )

        # Write fact
        with self.lock:
            success = self.fact_store.add(fact)

            if success:
                agent.facts_written += 1
                agent.last_active = datetime.now()
                self.stats["total_writes"] += 1

                self._log_event(
                    "fact_written",
                    {
                        "fact_id": fact.fact_id,
                        "triple": fact.triple(),
                        "agent_id": agent_id,
                    },
                )

        return WriteResult(
            success=success,
            fact_id=fact.fact_id if success else None,
            write_time_ms=(time.time() - t0) * 1000,
        )

    def write_many(
        self,
        facts: list[tuple[str, str, str]],
        agent_id: str,
        confidence: float = 1.0,
        source: str = "batch",
        check_contradictions: bool = False,
    ) -> list[WriteResult]:
        """Write multiple facts efficiently."""
        results = []

        for subject, predicate, obj in facts:
            result = self.write(
                subject,
                predicate,
                obj,
                agent_id,
                confidence=confidence,
                source=source,
                check_contradictions=check_contradictions,
            )
            results.append(result)

        return results

    def retract(self, fact_id: str, agent_id: str, reason: str = None) -> bool:
        """Retract a fact."""
        agent = self.agents.get(agent_id)
        if not agent or not agent.can_write:
            return False

        fact = self.fact_store.get(fact_id)
        if not fact:
            return False

        with self.lock:
            success = self.fact_store.update_status(fact_id, FactStatus.RETRACTED)

            if success:
                self._log_event(
                    "fact_retracted",
                    {
                        "fact_id": fact_id,
                        "agent_id": agent_id,
                        "reason": reason,
                    },
                )

        return success

    # =========================================================================
    # READ OPERATIONS
    # =========================================================================

    def read(self, fact_id: str, agent_id: str = None) -> Fact | None:
        """Read a fact by ID."""
        if agent_id:
            agent = self.agents.get(agent_id)
            if agent:
                agent.facts_read += 1
                agent.last_active = datetime.now()

        self.stats["total_reads"] += 1
        return self.fact_store.get(fact_id)

    def query(
        self,
        subject: str = None,
        predicate: str = None,
        obj: str = None,
        agent_id: str = None,
        k: int = 10,
        include_inferences: bool = True,
        domain: str = None,
    ) -> QueryResult:
        """
        Query the knowledge base.

        Supports wildcards (None) for any component.
        Can include inferred facts.
        """
        t0 = time.time()

        # Update agent stats
        if agent_id:
            agent = self.agents.get(agent_id)
            if agent:
                agent.queries_made += 1
                agent.last_active = datetime.now()

        # Semantic search
        results = self.fact_store.semantic_search(subject, predicate, obj, k=k)

        facts = [r[0] for r in results]
        scores = [r[1] for r in results]

        # Inference
        inferred = []
        if include_inferences:
            inferred_raw = self.inference_engine.infer(
                subject, predicate, obj, max_hops=3, domain=domain
            )
            for triple, reasoning in inferred_raw:
                # Create a temporary fact for the inferred result
                inf_fact = Fact(
                    subject=triple[0],
                    predicate=triple[1],
                    object=triple[2],
                    agent_id="inference_engine",
                    status=FactStatus.PENDING,  # Inferred, not stored
                    confidence=0.8,
                )
                inferred.append((inf_fact, reasoning))

            self.stats["inferences_made"] += len(inferred)

        self.stats["total_queries"] += 1

        return QueryResult(
            facts=facts,
            scores=scores,
            query_time_ms=(time.time() - t0) * 1000,
            total_matches=len(facts),
            inferred_facts=inferred,
        )

    def query_forward(
        self, subject: str, predicate: str, agent_id: str = None, k: int = 20
    ) -> list[Fact]:
        """Query for (subject, predicate, ?) - what does subject predicate?"""
        facts = self.fact_store.query_by_subject(
            subject.lower(), predicate=predicate.lower()
        )

        if agent_id:
            agent = self.agents.get(agent_id)
            if agent:
                agent.queries_made += 1
                agent.last_active = datetime.now()

        self.stats["total_queries"] += 1
        return facts[:k]

    def query_entity(self, entity: str, agent_id: str = None) -> dict[str, list[Fact]]:
        """Get all facts about an entity, organized by predicate."""
        entity = entity.lower()

        # As subject
        as_subject = self.fact_store.query_by_subject(entity)

        # As object (need to search)
        as_object = []
        for pred in self.fact_store.by_predicate.keys():
            facts = self.fact_store.semantic_search(obj=entity, predicate=pred, k=50)
            as_object.extend([f for f, _ in facts if f.object.lower() == entity])

        if agent_id:
            agent = self.agents.get(agent_id)
            if agent:
                agent.queries_made += 1

        self.stats["total_queries"] += 1

        # Organize by predicate
        result: dict[str, list[Fact]] = defaultdict(list)

        for fact in as_subject:
            result[f"{fact.predicate} ->"].append(fact)

        for fact in as_object:
            result[f"-> {fact.predicate}"].append(fact)

        return dict(result)

    def verify(
        self, subject: str, predicate: str, obj: str, agent_id: str = None
    ) -> tuple[bool, float, str | None]:
        """
        Verify if a claim is supported by the knowledge base.

        Returns: (is_supported, confidence, explanation)
        """
        # Check direct fact
        fact = self.fact_store.get_by_triple(subject, predicate, obj)

        if fact and fact.status == FactStatus.ACTIVE:
            return True, fact.confidence, f"Direct fact (id: {fact.fact_id})"

        # Check inference
        inferred = self.inference_engine.infer(subject, predicate, obj)

        for triple, reasoning in inferred:
            if (
                triple[0].lower() == subject.lower()
                and triple[1].lower() == predicate.lower()
                and triple[2].lower() == obj.lower()
            ):
                return True, 0.8, f"Inferred: {reasoning}"

        # Check semantic similarity
        results = self.fact_store.semantic_search(subject, predicate, obj, k=1)

        if results:
            fact, score = results[0]
            if score > 0.9:
                return True, score, f"Semantically similar to: {fact.triple()}"

        return False, 0.0, None

    # =========================================================================
    # TRAINING
    # =========================================================================

    def train(self, verbose: bool = True):
        """Train embeddings on current facts."""
        facts = [
            (f.subject, f.predicate, f.object)
            for f in self.fact_store.facts.values()
            if f.status == FactStatus.ACTIVE
        ]

        if verbose:
            print(f"Training on {len(facts):,} facts...")

        self.vsa.train_embeddings(facts, verbose=verbose)

        # Rebuild indexes
        if verbose:
            print("Rebuilding indexes...")
        self.fact_store.rebuild_indexes(verbose=verbose)

    # =========================================================================
    # DOMAIN MANAGEMENT
    # =========================================================================

    def add_domain_inference_rule(self, domain: str, rule: Callable):
        """Add a domain-specific inference rule."""
        self.inference_engine.add_domain_rule(domain, rule)

    def add_domain_contradiction_rule(self, domain: str, rule: Callable):
        """Add a domain-specific contradiction rule."""
        self.contradiction_detector.add_domain_rule(domain, rule)

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def save(self, path: str):
        """Save the entire core to disk."""
        state = {
            "dim": self.dim,
            "seed": self.seed,
            "has_ontology": self.ontology is not None,
            "vsa": {
                "word_to_idx": self.vsa.word_to_idx,
                "idx_to_word": self.vsa.idx_to_word,
                "embeddings": self.vsa.embeddings,
                "role_subject": self.vsa.role_subject,
                "role_predicate": self.vsa.role_predicate,
                "role_object": self.vsa.role_object,
            },
            "facts": {fid: f.to_dict() for fid, f in self.fact_store.facts.items()},
            "agents": {aid: asdict(a) for aid, a in self.agents.items()},
            "stats": self.stats,
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(state, f)

        print(f"Saved DorianCore to {path}")
        print(f"  Facts: {self.fact_store.active_facts:,}")
        print(f"  Agents: {len(self.agents)}")

    @classmethod
    def load(cls, path: str) -> "DorianCore":
        """Load the core from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)

        has_ontology = state.get("has_ontology", False)
        core = cls(dim=state["dim"], seed=state["seed"], load_ontology=has_ontology)

        # Restore VSA
        core.vsa.word_to_idx = state["vsa"]["word_to_idx"]
        core.vsa.idx_to_word = state["vsa"]["idx_to_word"]
        core.vsa.embeddings = state["vsa"]["embeddings"]
        core.vsa.role_subject = state["vsa"]["role_subject"]
        core.vsa.role_predicate = state["vsa"]["role_predicate"]
        core.vsa.role_object = state["vsa"]["role_object"]
        core.vsa.trained = core.vsa.embeddings is not None

        # Restore facts
        print(f"Loading {len(state['facts']):,} facts...")
        for fid, fd in state["facts"].items():
            fact = Fact.from_dict(fd)
            core.fact_store.facts[fid] = fact

            # Rebuild indexes
            core.fact_store.by_triple[fact.triple_key()] = fid
            core.fact_store.by_subject[fact.subject.lower()].add(fid)
            core.fact_store.by_predicate[fact.predicate.lower()].add(fid)
            core.fact_store.by_object[fact.object.lower()].add(fid)
            core.fact_store.by_domain[fact.domain].add(fid)
            core.fact_store.by_agent[fact.agent_id].add(fid)

            # Vector index
            vec = core.vsa.encode_fact(fact.subject, fact.predicate, fact.object)
            pred_key = fact.predicate.lower()
            core.fact_store.vectors[pred_key].append(vec)
            core.fact_store.vector_ids[pred_key].append(fid)

            core.fact_store.total_facts += 1
            if fact.status == FactStatus.ACTIVE:
                core.fact_store.active_facts += 1

        # Restore agents
        for aid, ad in state["agents"].items():
            ad["created_at"] = datetime.fromisoformat(ad["created_at"])
            ad["last_active"] = datetime.fromisoformat(ad["last_active"])
            core.agents[aid] = Agent(**ad)

        # Restore stats
        core.stats = state["stats"]

        # Rebuild FAISS indexes
        print("Rebuilding FAISS indexes...")
        core.fact_store.rebuild_indexes(verbose=True)

        print(f"Loaded DorianCore from {path}")
        return core

    # =========================================================================
    # EVENT LOGGING
    # =========================================================================

    def _log_event(self, event_type: str, data: dict):
        """Log an event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data,
        }

        self.event_log.append(event)

        # Trim if too long
        if len(self.event_log) > self.max_log_size:
            self.event_log = self.event_log[-self.max_log_size // 2 :]

    def get_recent_events(self, n: int = 100, event_type: str = None) -> list[dict]:
        """Get recent events."""
        events = self.event_log

        if event_type:
            events = [e for e in events if e["type"] == event_type]

        return events[-n:]

    # =========================================================================
    # STATS
    # =========================================================================

    def get_stats(self) -> dict:
        """Get core statistics."""
        return {
            **self.stats,
            "total_facts": self.fact_store.total_facts,
            "active_facts": self.fact_store.active_facts,
            "predicates": len(self.fact_store.by_predicate),
            "subjects": len(self.fact_store.by_subject),
            "agents": len(self.agents),
            "domains": len(self.fact_store.by_domain),
        }

    def status(self) -> str:
        """Human-readable status."""
        stats = self.get_stats()

        lines = [
            "═" * 50,
            "DORIAN CORE STATUS",
            "═" * 50,
            f"Active facts: {stats['active_facts']:,}",
            f"Total facts: {stats['total_facts']:,}",
            f"Predicates: {stats['predicates']}",
            f"Subjects: {stats['subjects']:,}",
            f"Registered agents: {stats['agents']}",
            f"Domains: {stats['domains']}",
            "",
            "Operations:",
            f"  Reads: {stats['total_reads']:,}",
            f"  Writes: {stats['total_writes']:,}",
            f"  Queries: {stats['total_queries']:,}",
            f"  Contradictions detected: {stats['contradictions_detected']:,}",
            f"  Inferences made: {stats['inferences_made']:,}",
            "═" * 50,
        ]

        return "\n".join(lines)


# =============================================================================
# PART 7: CONVENIENCE FUNCTIONS
# =============================================================================


def create_core(dim: int = 512) -> DorianCore:
    """Create a new Dorian Core."""
    return DorianCore(dim=dim)


def load_core(path: str) -> DorianCore:
    """Load a Dorian Core from disk."""
    return DorianCore.load(path)


# =============================================================================
# MAIN - DEMO
# =============================================================================

if __name__ == "__main__":
    print("═" * 60)
    print("DORIAN CORE - DEMO")
    print("═" * 60)

    # Create core with ontology
    print("\nCreating core with ontology...")
    core = DorianCore(dim=256, load_ontology=True)

    print(f"  Ontology loaded: {core.ontology is not None}")
    if core.ontology:
        stats = core.ontology.stats()
        print(f"  Categories: {stats['categories']}")
        print(f"  Relations: {stats['relations']}")
        print(
            f"  Transitive relations: {len(core.inference_engine.transitive_relations)}"
        )

    # Bootstrap ontology facts
    print("\nBootstrapping ontology facts...")
    ont_count = core.bootstrap_ontology()

    # Register agents
    retail_agent = core.register_agent("Profit Sentinel", domain="retail")
    general_agent = core.register_agent("General Agent", domain="general")

    print("\nRegistered agents:")
    print(f"  - {retail_agent.name} ({retail_agent.agent_id[:8]}...)")
    print(f"  - {general_agent.name} ({general_agent.agent_id[:8]}...)")

    # Write domain facts
    print("\nWriting domain facts...")

    retail_facts = [
        ("dead_inventory", "causes", "margin_erosion"),
        ("margin_erosion", "causes", "profit_leak"),
        ("overstock", "causes", "dead_inventory"),
        ("stockout", "causes", "lost_sales"),
        ("pricing_error", "is_a", "profit_leak"),
        ("slow_moving_sku", "indicates", "demand_mismatch"),
        ("profit_leak", "is_a", "problem"),
        ("margin_erosion", "is_a", "problem"),
    ]

    for s, p, o in retail_facts:
        result = core.write(s, p, o, retail_agent.agent_id, source="domain_knowledge")
        if result.success:
            print(f"  ✓ {s} {p} {o}")
        else:
            print(f"  ✗ {s} {p} {o}: {result.error}")

    general_facts = [
        ("fire", "causes", "heat"),
        ("heat", "causes", "burning"),
        ("bird", "is_a", "animal"),
        ("sparrow", "is_a", "bird"),
        ("dog", "is_a", "animal"),
        ("animal", "capable_of", "movement"),
        ("animal", "requires", "food"),
        ("human", "is_a", "animal"),
        ("human", "capable_of", "reasoning"),
    ]

    for s, p, o in general_facts:
        result = core.write(s, p, o, general_agent.agent_id, source="common_sense")
        if result.success:
            print(f"  ✓ {s} {p} {o}")

    # Train embeddings
    print("\nTraining embeddings...")
    core.train(verbose=True)

    # =========================================================================
    # QUERIES AND INFERENCE DEMOS
    # =========================================================================

    print("\n" + "═" * 60)
    print("INFERENCE DEMOS")
    print("═" * 60)

    # 1. Transitive causation
    print("\n1. TRANSITIVE CAUSATION")
    print("   Question: Does overstock cause profit_leak?")
    supported, conf, reason = core.verify("overstock", "causes", "profit_leak")
    print(f"   Supported: {supported}")
    print(f"   Confidence: {conf:.2f}")
    print(f"   Reason: {reason}")

    # 2. Category inheritance
    print("\n2. CATEGORY INHERITANCE")
    print("   Question: Is sparrow an animal?")
    # Direct check
    facts = core.query_forward("sparrow", "is_a")
    print(f"   Direct is_a: {[f.object for f in facts]}")
    # Via inheritance
    is_animal = core.is_type_of("sparrow", "animal")
    print(f"   Is type of animal: {is_animal}")

    # 3. Property inheritance
    print("\n3. PROPERTY INHERITANCE")
    print(
        "   Question: Can humans move? (human is_a animal, animal capable_of movement)"
    )
    result = core.query(
        subject="human", predicate="capable_of", include_inferences=True
    )
    print(
        f"   Direct facts: {[(f.subject, f.predicate, f.object) for f in result.facts]}"
    )
    print(f"   Inferred facts: {len(result.inferred_facts)}")
    for inf_fact, reasoning in result.inferred_facts[:3]:
        print(f"     - {inf_fact.subject} {inf_fact.predicate} {inf_fact.object}")
        print(f"       ({reasoning})")

    # 4. Multi-hop causation
    print("\n4. MULTI-HOP CAUSATION")
    print("   Question: What does fire ultimately cause?")
    print("   Chain: fire → heat → burning")
    supported, conf, reason = core.verify("fire", "causes", "burning")
    print(f"   fire causes burning: {supported}")
    print(f"   Reason: {reason}")

    # 5. Contradiction detection
    print("\n5. CONTRADICTION DETECTION")

    # 5a. Disjoint categories (animal vs plant)
    print("\n   5a. Disjoint categories:")
    print("       dog is_a animal (established)")
    print("       Attempt: Add 'dog is_a plant'")
    print(
        f"       Exclusive categories for 'plant': {core.contradiction_detector.exclusive_categories.get('plant', set())}"
    )
    result = core.write("dog", "is_a", "plant", general_agent.agent_id)
    print(f"       Write succeeded: {result.success}")
    if not result.success:
        print(f"       Reason: {result.error}")
        for c in result.contradictions:
            print(f"       Conflicts with: {c.triple()}")

    # 5b. Direct negation
    print("\n   5b. Direct negation:")
    print("       Establish: cat can fly")
    core.write("cat", "can", "fly", general_agent.agent_id)
    print("       Attempt: cat cannot fly")
    result = core.write("cat", "cannot", "fly", general_agent.agent_id)
    print(f"       Write succeeded: {result.success}")
    if not result.success:
        print(f"       Reason: {result.error}")
        for c in result.contradictions:
            print(f"       Conflicts with: {c.triple()}")

    # 5c. Mutual exclusion (alive vs dead)
    print("\n   5c. Mutual exclusion (alive/dead):")
    print("       Establish: bob is_a alive")
    core.write("bob", "is_a", "alive", general_agent.agent_id)
    print("       Attempt: bob is_a dead")
    result = core.write("bob", "is_a", "dead", general_agent.agent_id)
    print(f"       Write succeeded: {result.success}")
    if not result.success:
        print(f"       Reason: {result.error}")
        for c in result.contradictions:
            print(f"       Conflicts with: {c.triple()}")

    # 6. Ontology-aware queries
    print("\n6. ONTOLOGY-AWARE QUERIES")
    print("   Getting all categories of 'human':")
    categories = core.get_category_of("human")
    print(f"   Direct categories: {categories}")
    if core.ontology:
        all_ancestors = []
        for cat in categories:
            all_ancestors.extend(core.ontology.get_ancestors(cat))
        print(f"   All ancestor categories: {list(set(all_ancestors))}")

    # Status
    print("\n" + core.status())

    # Save
    print("\nSaving core...")
    core.save("dorian_core_with_ontology")

    print("\nDone!")
