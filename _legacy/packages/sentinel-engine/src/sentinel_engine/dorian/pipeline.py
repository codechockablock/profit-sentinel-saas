"""
DORIAN UNIFIED KNOWLEDGE PIPELINE
=================================

Single interface to load knowledge from all sources into Dorian.

Sources:
- Wikidata: Structured entity facts (100M+ entities)
- arXiv: Scientific paper metadata (2.5M papers)
- ConceptNet: Common sense assertions (34M facts)

Features:
- Unified loader interface
- Source provenance tracking
- Deduplication across sources
- Domain filtering
- Progress reporting
- Batch loading with checkpoints

Usage:
    pipeline = KnowledgePipeline(core)
    pipeline.load_conceptnet(max_facts=100000)
    pipeline.load_wikidata_domain('materials')
    pipeline.load_arxiv_category('cs.AI')

    # Or load everything
    pipeline.load_all(max_facts_per_source=100000)

Author: Joseph + Claude
Date: 2026-01-25
"""

import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

# Add path for imports
sys.path.insert(0, "/home/claude")


# =============================================================================
# KNOWLEDGE SOURCE ENUM
# =============================================================================


class KnowledgeSource(Enum):
    """Available knowledge sources."""

    WIKIDATA = "wikidata"
    ARXIV = "arxiv"
    CONCEPTNET = "conceptnet"
    # Domain-specific
    MATH = "math"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    COMPUTER_SCIENCE = "cs"
    ECONOMICS = "economics"
    PHILOSOPHY = "philosophy"
    ONTOLOGY = "ontology"
    # Custom
    CUSTOM = "custom"


# =============================================================================
# TRIPLE WITH PROVENANCE
# =============================================================================


@dataclass
class KnowledgeTriple:
    """A fact with full provenance."""

    subject: str
    predicate: str
    obj: str
    source: KnowledgeSource
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Unique key for deduplication."""
        return f"{self.subject.lower()}|{self.predicate.lower()}|{self.obj.lower()}"

    @property
    def hash(self) -> str:
        """Hash for fast lookup."""
        return hashlib.md5(self.key.encode()).hexdigest()[:16]

    def to_tuple(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.obj)


# =============================================================================
# PIPELINE STATISTICS
# =============================================================================


@dataclass
class LoadStats:
    """Statistics for a load operation."""

    source: KnowledgeSource
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    triples_processed: int = 0
    triples_loaded: int = 0
    triples_skipped_duplicate: int = 0
    triples_skipped_filter: int = 0
    triples_failed: int = 0
    predicates: Counter = field(default_factory=Counter)

    @property
    def duration_seconds(self) -> float:
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

    @property
    def rate(self) -> float:
        d = self.duration_seconds
        return self.triples_loaded / d if d > 0 else 0

    def summary(self) -> str:
        return (
            f"{self.source.value}: {self.triples_loaded:,} loaded, "
            f"{self.triples_skipped_duplicate:,} dupes, "
            f"{self.triples_failed:,} failed "
            f"({self.duration_seconds:.1f}s, {self.rate:.0f}/sec)"
        )


# =============================================================================
# UNIFIED KNOWLEDGE PIPELINE
# =============================================================================


class KnowledgePipeline:
    """
    Unified pipeline for loading knowledge from all sources into Dorian.

    Handles:
    - Multiple source types
    - Deduplication across sources
    - Provenance tracking
    - Domain filtering
    - Progress reporting
    """

    def __init__(self, core, agent_name: str = "knowledge_pipeline"):
        """
        Initialize pipeline.

        Args:
            core: DorianCore instance
            agent_name: Name for the loader agent
        """
        self.core = core
        self.agent = core.register_agent(agent_name, domain="knowledge")
        self.agent_id = self.agent.agent_id

        # Deduplication
        self.seen_hashes: set[str] = set()

        # Statistics
        self.stats: dict[KnowledgeSource, LoadStats] = {}
        self.total_loaded = 0

        # Filters
        self.predicate_whitelist: set[str] | None = None
        self.predicate_blacklist: set[str] = set()
        self.concept_filter: Callable[[str], bool] | None = None

        # Source-specific agents for provenance
        self.source_agents: dict[KnowledgeSource, str] = {}

    def _get_source_agent(self, source: KnowledgeSource) -> str:
        """Get or create agent for a specific source."""
        if source not in self.source_agents:
            agent = self.core.register_agent(
                f"{source.value}_loader", domain="knowledge"
            )
            self.source_agents[source] = agent.agent_id
        return self.source_agents[source]

    def _should_load(self, triple: KnowledgeTriple) -> tuple[bool, str]:
        """
        Check if a triple should be loaded.

        Returns:
            (should_load, reason_if_not)
        """
        # Deduplication
        if triple.hash in self.seen_hashes:
            return False, "duplicate"

        # Predicate whitelist
        if self.predicate_whitelist:
            if triple.predicate.lower() not in self.predicate_whitelist:
                return False, "predicate_filtered"

        # Predicate blacklist
        if triple.predicate.lower() in self.predicate_blacklist:
            return False, "predicate_blacklisted"

        # Concept filter
        if self.concept_filter:
            if not self.concept_filter(triple.subject) and not self.concept_filter(
                triple.obj
            ):
                return False, "concept_filtered"

        return True, ""

    def _load_triple(self, triple: KnowledgeTriple, stats: LoadStats) -> bool:
        """Load a single triple into Dorian."""
        stats.triples_processed += 1

        # Check filters
        should_load, reason = self._should_load(triple)
        if not should_load:
            if reason == "duplicate":
                stats.triples_skipped_duplicate += 1
            else:
                stats.triples_skipped_filter += 1
            return False

        # Load into Dorian
        try:
            agent_id = self._get_source_agent(triple.source)
            result = self.core.write(
                triple.subject,
                triple.predicate,
                triple.obj,
                agent_id,
                confidence=triple.confidence,
            )

            if result.success:
                self.seen_hashes.add(triple.hash)
                stats.triples_loaded += 1
                stats.predicates[triple.predicate] += 1
                self.total_loaded += 1
                return True
            else:
                stats.triples_failed += 1
                return False

        except Exception:
            stats.triples_failed += 1
            return False

    def _load_triples(
        self,
        triples: list[KnowledgeTriple],
        source: KnowledgeSource,
        show_progress: bool = True,
        progress_interval: int = 10000,
    ) -> LoadStats:
        """Load a batch of triples."""
        stats = LoadStats(source=source)
        self.stats[source] = stats

        for i, triple in enumerate(triples):
            self._load_triple(triple, stats)

            if show_progress and (i + 1) % progress_interval == 0:
                print(f"  [{source.value}] {stats.triples_loaded:,}/{i+1:,} loaded...")

        stats.completed_at = datetime.now()

        if show_progress:
            print(f"  {stats.summary()}")

        return stats

    # =========================================================================
    # CONCEPTNET LOADER
    # =========================================================================

    def load_conceptnet(
        self,
        filepath: str = None,
        max_facts: int = None,
        min_weight: float = 1.0,
        use_sample: bool = False,
        show_progress: bool = True,
    ) -> LoadStats:
        """
        Load ConceptNet common sense facts.

        Args:
            filepath: Path to ConceptNet dump (None for default)
            max_facts: Maximum facts to load
            min_weight: Minimum weight threshold
            use_sample: Use built-in sample data (no file needed)
            show_progress: Print progress

        Returns:
            LoadStats for this operation
        """
        print(f"\n{'='*60}")
        print("LOADING CONCEPTNET")
        print(f"{'='*60}")

        triples = []

        if use_sample:
            # Use sample data - try both naming conventions
            try:
                from dorian_conceptnet_loader import SAMPLE_FACTS

                sample_data = SAMPLE_FACTS
            except ImportError:
                from conceptnet_loader_updated import SAMPLE_FACTS

                sample_data = SAMPLE_FACTS

            print(f"Using sample data ({len(sample_data)} facts)")

            for s, r, o in sample_data:
                triples.append(
                    KnowledgeTriple(
                        subject=s,
                        predicate=r,
                        obj=o,
                        source=KnowledgeSource.CONCEPTNET,
                        confidence=1.0,
                    )
                )
        else:
            # Load from file
            try:
                from dorian_conceptnet_loader import load_conceptnet as load_cn

                if filepath is None:
                    filepath = "conceptnet-assertions-5.7.0.csv.gz"

                if not os.path.exists(filepath):
                    print(f"ConceptNet file not found: {filepath}")
                    print("Use use_sample=True for sample data, or download the file.")
                    return LoadStats(source=KnowledgeSource.CONCEPTNET)

                raw_facts = load_cn(
                    filepath, max_facts, min_weight, verbose=show_progress
                )

                for s, r, o in raw_facts:
                    triples.append(
                        KnowledgeTriple(
                            subject=s,
                            predicate=r,
                            obj=o,
                            source=KnowledgeSource.CONCEPTNET,
                            confidence=1.0,
                        )
                    )

            except ImportError as e:
                print(f"ConceptNet loader not available: {e}")
                return LoadStats(source=KnowledgeSource.CONCEPTNET)

        if max_facts and len(triples) > max_facts:
            triples = triples[:max_facts]

        return self._load_triples(triples, KnowledgeSource.CONCEPTNET, show_progress)

    # =========================================================================
    # WIKIDATA LOADER
    # =========================================================================

    def load_wikidata_sample(self, show_progress: bool = True) -> LoadStats:
        """Load Wikidata sample data (no network needed)."""
        print(f"\n{'='*60}")
        print("LOADING WIKIDATA (SAMPLE)")
        print(f"{'='*60}")

        try:
            from dorian_wikidata_loader import resolve_property
            from wikidata_sample_test import load_sample_data

            entities, raw_triples = load_sample_data()
            print(f"Sample: {len(entities)} entities, {len(raw_triples)} triples")

            triples = []
            for s, p, o in raw_triples:
                p_resolved = resolve_property(p)
                triples.append(
                    KnowledgeTriple(
                        subject=str(s),
                        predicate=str(p_resolved),
                        obj=str(o),
                        source=KnowledgeSource.WIKIDATA,
                        confidence=1.0,
                    )
                )

            return self._load_triples(triples, KnowledgeSource.WIKIDATA, show_progress)

        except ImportError as e:
            print(f"Wikidata loader not available: {e}")
            return LoadStats(source=KnowledgeSource.WIKIDATA)

    def load_wikidata_domain(
        self, domain: str, show_progress: bool = True
    ) -> LoadStats:
        """
        Load Wikidata domain via SPARQL (requires network).

        Domains: 'materials', 'hardware', 'products', 'companies'
        """
        print(f"\n{'='*60}")
        print(f"LOADING WIKIDATA DOMAIN: {domain}")
        print(f"{'='*60}")

        try:
            from dorian_wikidata_loader import (
                IMPORTANT_PROPERTIES,
                get_class_hierarchy,
                resolve_property,
            )

            domain_roots = {
                "materials": "Q987767",  # building material
                "hardware": "Q3966",  # hardware
                "products": "Q28877",  # goods
                "companies": "Q783794",  # company
            }

            if domain not in domain_roots:
                print(f"Unknown domain: {domain}")
                print(f"Available: {list(domain_roots.keys())}")
                return LoadStats(source=KnowledgeSource.WIKIDATA)

            root_id = domain_roots[domain]
            print(f"Fetching hierarchy for {root_id}...")

            raw_triples = get_class_hierarchy(root_id)
            print(f"Retrieved {len(raw_triples)} triples")

            triples = []
            for s, p, o in raw_triples:
                triples.append(
                    KnowledgeTriple(
                        subject=str(s),
                        predicate=str(p),
                        obj=str(o),
                        source=KnowledgeSource.WIKIDATA,
                        confidence=1.0,
                        metadata={"domain": domain},
                    )
                )

            return self._load_triples(triples, KnowledgeSource.WIKIDATA, show_progress)

        except Exception as e:
            print(f"Wikidata domain load failed: {e}")
            return LoadStats(source=KnowledgeSource.WIKIDATA)

    # =========================================================================
    # ARXIV LOADER
    # =========================================================================

    def load_arxiv_sample(self, show_progress: bool = True) -> LoadStats:
        """Load arXiv sample papers (no network needed)."""
        print(f"\n{'='*60}")
        print("LOADING ARXIV (SAMPLE)")
        print(f"{'='*60}")

        try:
            from dorian_arxiv_loader import get_category_triples, load_sample_papers

            # Category taxonomy
            cat_triples = get_category_triples()

            # Sample papers
            papers = load_sample_papers()
            paper_triples = []
            for paper in papers:
                paper_triples.extend(paper.to_triples())

            print(f"Sample: {len(papers)} papers, {len(cat_triples)} categories")

            triples = []

            # Categories
            for s, p, o in cat_triples:
                triples.append(
                    KnowledgeTriple(
                        subject=s,
                        predicate=p,
                        obj=o,
                        source=KnowledgeSource.ARXIV,
                        confidence=1.0,
                        metadata={"type": "category"},
                    )
                )

            # Papers
            for s, p, o in paper_triples:
                triples.append(
                    KnowledgeTriple(
                        subject=s,
                        predicate=p,
                        obj=o,
                        source=KnowledgeSource.ARXIV,
                        confidence=1.0,
                        metadata={"type": "paper"},
                    )
                )

            return self._load_triples(triples, KnowledgeSource.ARXIV, show_progress)

        except ImportError as e:
            print(f"arXiv loader not available: {e}")
            return LoadStats(source=KnowledgeSource.ARXIV)

    def load_arxiv_category(
        self, category: str, max_papers: int = 100, show_progress: bool = True
    ) -> LoadStats:
        """
        Load arXiv papers from a category (requires network).

        Categories: 'cs.AI', 'cs.LG', 'cs.CL', 'stat.ML', etc.
        """
        print(f"\n{'='*60}")
        print(f"LOADING ARXIV CATEGORY: {category}")
        print(f"{'='*60}")

        try:
            from dorian_arxiv_loader import arxiv_api_search

            print(f"Searching for papers in {category}...")
            papers = arxiv_api_search(f"cat:{category}", max_results=max_papers)
            print(f"Found {len(papers)} papers")

            triples = []
            for paper in papers:
                for s, p, o in paper.to_triples():
                    triples.append(
                        KnowledgeTriple(
                            subject=s,
                            predicate=p,
                            obj=o,
                            source=KnowledgeSource.ARXIV,
                            confidence=1.0,
                            metadata={"category": category},
                        )
                    )

            return self._load_triples(triples, KnowledgeSource.ARXIV, show_progress)

        except Exception as e:
            print(f"arXiv category load failed: {e}")
            return LoadStats(source=KnowledgeSource.ARXIV)

    # =========================================================================
    # DOMAIN KNOWLEDGE LOADERS
    # =========================================================================

    def load_domain(self, domain: str, show_progress: bool = True) -> LoadStats:
        """
        Load a specific domain's knowledge.

        Domains: 'math', 'physics', 'chemistry', 'biology', 'cs', 'economics', 'philosophy'
        """
        domain_map = {
            "math": (KnowledgeSource.MATH, "dorian_math", "MATH_FACTS"),
            "physics": (KnowledgeSource.PHYSICS, "dorian_physics", "PHYSICS_FACTS"),
            "chemistry": (KnowledgeSource.CHEMISTRY, "dorian_chemistry", "CHEM_FACTS"),
            "biology": (KnowledgeSource.BIOLOGY, "dorian_biology", "BIO_FACTS"),
            "cs": (KnowledgeSource.COMPUTER_SCIENCE, "dorian_cs", "CS_FACTS"),
            "economics": (KnowledgeSource.ECONOMICS, "dorian_economics", "ECON_FACTS"),
            "philosophy": (
                KnowledgeSource.PHILOSOPHY,
                "dorian_philosophy",
                "PHIL_FACTS",
            ),
        }

        if domain not in domain_map:
            print(f"Unknown domain: {domain}")
            print(f"Available: {list(domain_map.keys())}")
            return LoadStats(source=KnowledgeSource.CUSTOM)

        source, module_name, facts_name = domain_map[domain]

        print(f"\n{'='*60}")
        print(f"LOADING DOMAIN: {domain.upper()}")
        print(f"{'='*60}")

        try:
            module = __import__(module_name)
            facts = getattr(module, facts_name, [])

            print(f"Found {len(facts)} {domain} facts")

            triples = []
            for s, p, o in facts:
                triples.append(
                    KnowledgeTriple(
                        subject=str(s),
                        predicate=str(p),
                        obj=str(o),
                        source=source,
                        confidence=1.0,
                        metadata={"domain": domain},
                    )
                )

            return self._load_triples(triples, source, show_progress)

        except ImportError as e:
            print(f"Domain module not available: {e}")
            return LoadStats(source=source)
        except AttributeError as e:
            print(f"Facts not found in module: {e}")
            return LoadStats(source=source)

    def load_all_domains(self, show_progress: bool = True) -> dict[str, LoadStats]:
        """Load all domain knowledge."""
        domains = [
            "math",
            "physics",
            "chemistry",
            "biology",
            "cs",
            "economics",
            "philosophy",
        ]

        print("\n" + "=" * 70)
        print("LOADING ALL DOMAIN KNOWLEDGE")
        print("=" * 70)

        stats = {}
        for domain in domains:
            stats[domain] = self.load_domain(domain, show_progress)

        total = sum(s.triples_loaded for s in stats.values())
        print(f"\nTotal domain facts loaded: {total:,}")

        return stats

    def load_ontology(self, show_progress: bool = True) -> LoadStats:
        """Load foundational ontology (categories, relations, axioms)."""
        print(f"\n{'='*60}")
        print("LOADING ONTOLOGY")
        print(f"{'='*60}")

        try:
            from dorian_ontology import (
                CATEGORY_RELATIONS,
                RELATION_TYPES,
                TOP_LEVEL_CATEGORIES,
                build_category_facts,
                build_relation_facts,
            )

            triples = []

            # Category hierarchy
            cat_facts = build_category_facts()
            for s, p, o in cat_facts:
                triples.append(
                    KnowledgeTriple(
                        subject=str(s),
                        predicate=str(p),
                        obj=str(o),
                        source=KnowledgeSource.ONTOLOGY,
                        confidence=1.0,
                        metadata={"type": "category"},
                    )
                )

            # Relation definitions
            rel_facts = build_relation_facts()
            for s, p, o in rel_facts:
                triples.append(
                    KnowledgeTriple(
                        subject=str(s),
                        predicate=str(p),
                        obj=str(o),
                        source=KnowledgeSource.ONTOLOGY,
                        confidence=1.0,
                        metadata={"type": "relation"},
                    )
                )

            print(f"Found {len(triples)} ontology facts")

            return self._load_triples(triples, KnowledgeSource.ONTOLOGY, show_progress)

        except ImportError as e:
            print(f"Ontology module not available: {e}")
            return LoadStats(source=KnowledgeSource.ONTOLOGY)
        except Exception as e:
            print(f"Ontology load error: {e}")
            return LoadStats(source=KnowledgeSource.ONTOLOGY)

    # =========================================================================
    # CUSTOM FACTS
    # =========================================================================

    def load_custom(
        self,
        facts: list[tuple[str, str, str]],
        confidence: float = 1.0,
        show_progress: bool = True,
    ) -> LoadStats:
        """
        Load custom facts.

        Args:
            facts: List of (subject, predicate, object) tuples
            confidence: Confidence score for all facts
        """
        print(f"\n{'='*60}")
        print(f"LOADING CUSTOM FACTS ({len(facts)})")
        print(f"{'='*60}")

        triples = []
        for s, p, o in facts:
            triples.append(
                KnowledgeTriple(
                    subject=s,
                    predicate=p,
                    obj=o,
                    source=KnowledgeSource.CUSTOM,
                    confidence=confidence,
                )
            )

        return self._load_triples(triples, KnowledgeSource.CUSTOM, show_progress)

    # =========================================================================
    # LOAD ALL
    # =========================================================================

    def load_all_samples(
        self, show_progress: bool = True
    ) -> dict[KnowledgeSource, LoadStats]:
        """Load sample data from all sources (no network needed)."""
        print("\n" + "=" * 70)
        print("LOADING ALL KNOWLEDGE SOURCES (SAMPLES)")
        print("=" * 70)

        t0 = time.time()

        stats = {}
        stats[KnowledgeSource.CONCEPTNET] = self.load_conceptnet(
            use_sample=True, show_progress=show_progress
        )
        stats[KnowledgeSource.WIKIDATA] = self.load_wikidata_sample(
            show_progress=show_progress
        )
        stats[KnowledgeSource.ARXIV] = self.load_arxiv_sample(
            show_progress=show_progress
        )

        elapsed = time.time() - t0

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        total = 0
        for source, s in stats.items():
            print(f"  {s.summary()}")
            total += s.triples_loaded

        print(f"\nTotal: {total:,} facts loaded in {elapsed:.1f}s")
        print(f"Unique predicates: {len(self.get_all_predicates())}")
        print(f"Facts in Dorian: {len(self.core.fact_store.facts):,}")

        return stats

    def load_everything(
        self,
        include_domains: bool = True,
        include_ontology: bool = True,
        show_progress: bool = True,
    ) -> dict[str, LoadStats]:
        """
        Load ALL knowledge sources - the full pipeline.

        This loads:
        - ConceptNet (common sense)
        - Wikidata (structured entities)
        - arXiv (research)
        - Domain knowledge (math, physics, chemistry, biology, cs, economics, philosophy)
        - Ontology (foundational categories and relations)
        """
        print("\n" + "=" * 70)
        print("LOADING COMPLETE KNOWLEDGE BASE")
        print("=" * 70)

        t0 = time.time()
        all_stats = {}

        # External sources (samples for now)
        print("\n--- External Sources ---")
        all_stats["conceptnet"] = self.load_conceptnet(
            use_sample=True, show_progress=show_progress
        )
        all_stats["wikidata"] = self.load_wikidata_sample(show_progress=show_progress)
        all_stats["arxiv"] = self.load_arxiv_sample(show_progress=show_progress)

        # Ontology (load before domains)
        if include_ontology:
            print("\n--- Ontology ---")
            all_stats["ontology"] = self.load_ontology(show_progress=show_progress)

        # Domain knowledge
        if include_domains:
            print("\n--- Domain Knowledge ---")
            domain_stats = self.load_all_domains(show_progress=show_progress)
            all_stats.update(domain_stats)

        elapsed = time.time() - t0

        # Summary
        print("\n" + "=" * 70)
        print("COMPLETE KNOWLEDGE BASE SUMMARY")
        print("=" * 70)

        total_loaded = 0
        total_dupes = 0

        for name, stats in all_stats.items():
            if hasattr(stats, "triples_loaded"):
                print(
                    f"  {name}: {stats.triples_loaded:,} loaded, {stats.triples_skipped_duplicate:,} dupes"
                )
                total_loaded += stats.triples_loaded
                total_dupes += stats.triples_skipped_duplicate

        print(f"\n  TOTAL LOADED: {total_loaded:,}")
        print(f"  DUPLICATES SKIPPED: {total_dupes:,}")
        print(f"  FACTS IN DORIAN: {len(self.core.fact_store.facts):,}")
        print(f"  UNIQUE PREDICATES: {len(self.get_all_predicates())}")
        print(f"  TIME: {elapsed:.1f}s")

        return all_stats

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def get_all_predicates(self) -> set[str]:
        """Get all unique predicates loaded."""
        predicates = set()
        for stats in self.stats.values():
            predicates.update(stats.predicates.keys())
        return predicates

    def get_stats_summary(self) -> str:
        """Get summary of all load operations."""
        lines = ["Knowledge Pipeline Stats:", "-" * 40]

        total_loaded = 0
        total_dupes = 0

        for source, stats in self.stats.items():
            lines.append(f"  {stats.summary()}")
            total_loaded += stats.triples_loaded
            total_dupes += stats.triples_skipped_duplicate

        lines.append("-" * 40)
        lines.append(f"  Total loaded: {total_loaded:,}")
        lines.append(f"  Duplicates skipped: {total_dupes:,}")
        lines.append(f"  Facts in Dorian: {len(self.core.fact_store.facts):,}")

        return "\n".join(lines)

    def set_predicate_filter(
        self, whitelist: set[str] = None, blacklist: set[str] = None
    ):
        """Set predicate filters."""
        if whitelist:
            self.predicate_whitelist = {p.lower() for p in whitelist}
        if blacklist:
            self.predicate_blacklist = {p.lower() for p in blacklist}

    def set_concept_filter(self, filter_fn: Callable[[str], bool]):
        """Set concept filter function."""
        self.concept_filter = filter_fn

    def clear_filters(self):
        """Clear all filters."""
        self.predicate_whitelist = None
        self.predicate_blacklist = set()
        self.concept_filter = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def build_knowledge_base(
    dim: int = 10000, use_samples: bool = True, show_progress: bool = True
):
    """
    Build a complete knowledge base from all sources.

    Args:
        dim: Vector dimension for Dorian
        use_samples: Use sample data (True) or full data (False)
        show_progress: Print progress

    Returns:
        (DorianCore, KnowledgePipeline)
    """
    from dorian_core import DorianCore

    print("=" * 70)
    print("BUILDING DORIAN KNOWLEDGE BASE")
    print("=" * 70)

    # Initialize Dorian
    print("\nInitializing DorianCore...")
    core = DorianCore(dim=dim, load_ontology=False)

    # Create pipeline
    pipeline = KnowledgePipeline(core)

    # Load all sources
    if use_samples:
        pipeline.load_all_samples(show_progress=show_progress)
    else:
        # Full data would go here
        print("Full data loading not yet implemented")
        print("Use use_samples=True for now")

    return core, pipeline


# =============================================================================
# MAIN / DEMO
# =============================================================================

if __name__ == "__main__":
    print("DORIAN UNIFIED KNOWLEDGE PIPELINE")
    print("=" * 70)

    from dorian_core import DorianCore

    # Initialize
    print("\nInitializing DorianCore...")
    core = DorianCore(dim=10000, load_ontology=False)

    # Create pipeline and load EVERYTHING
    pipeline = KnowledgePipeline(core)
    stats = pipeline.load_everything(
        include_domains=True, include_ontology=True, show_progress=True
    )

    # Test queries across all domains
    print("\n" + "=" * 70)
    print("TEST QUERIES ACROSS ALL DOMAINS")
    print("=" * 70)

    queries = [
        # ConceptNet
        ("dog", "is", "Common sense"),
        ("hammer", "used_for", "Common sense"),
        # Wikidata
        ("lumber", "subclass_of", "Wikidata"),
        # arXiv
        ("cs.LG", "has_name", "arXiv"),
        # Math
        ("integer", "subset_of", "Math"),
        ("addition", "has_property", "Math"),
        # Physics
        ("electron", "is_a", "Physics"),
        ("force", "has_unit", "Physics"),
        # Chemistry
        ("hydrogen", "is_a", "Chemistry"),
        ("water", "composed_of", "Chemistry"),
        # Biology
        ("cell", "is_a", "Biology"),
        ("dna", "encodes", "Biology"),
        # CS
        ("algorithm", "is_a", "CS"),
        ("python", "is_a", "CS"),
        # Economics
        ("supply", "related_to", "Economics"),
        # Philosophy
        ("epistemology", "studies", "Philosophy"),
    ]

    for subj, pred, domain in queries:
        results = core.fact_store.query_by_subject(subj, pred)
        if results:
            print(f"\n  [{domain}] {subj} {pred} ?")
            for f in results[:3]:
                print(f"    -> {f.object}")
        else:
            # Try without predicate
            results = core.fact_store.query_by_subject(subj)
            if results:
                print(f"\n  [{domain}] {subj} * ?")
                for f in results[:2]:
                    print(f"    -> {f.predicate}: {f.object}")

    print("\n" + "=" * 70)
    print("✅ Complete knowledge base ready!")
    print("=" * 70)
