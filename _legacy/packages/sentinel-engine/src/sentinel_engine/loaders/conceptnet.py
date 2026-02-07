"""
DORIAN CONCEPTNET LOADER
========================

Load common sense knowledge from ConceptNet into Dorian's VSA knowledge base.

ConceptNet structure:
- Nodes: Concepts in various languages (/c/en/dog, /c/fr/chien)
- Edges: Relations between concepts
- Relations: IsA, PartOf, HasA, UsedFor, CapableOf, AtLocation, etc.
- Weights: Confidence scores

Access methods:
1. REST API - Query specific concepts (JSON-LD)
2. Bulk download - Tab-separated assertions file (~2GB compressed)

Knowledge types:
- Taxonomic: dog IsA animal
- Meronymy: car HasA wheel
- Properties: fire HasProperty hot
- Actions: knife UsedFor cutting
- Locations: bed AtLocation bedroom
- Causation: rain Causes wet
"""

import gzip
import json
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Optional

# =============================================================================
# CONCEPTNET RELATIONS
# =============================================================================

# Core ConceptNet relation types
CONCEPTNET_RELATIONS = {
    # Taxonomic
    "/r/IsA": "is_a",
    "/r/InstanceOf": "instance_of",
    # Part-whole
    "/r/PartOf": "part_of",
    "/r/HasA": "has_part",
    "/r/MadeOf": "made_of",
    # Properties
    "/r/HasProperty": "has_property",
    "/r/HasContext": "has_context",
    # Spatial
    "/r/AtLocation": "located_at",
    "/r/LocatedNear": "near",
    # Actions/Usage
    "/r/UsedFor": "used_for",
    "/r/CapableOf": "capable_of",
    "/r/ReceivesAction": "receives_action",
    # Causation
    "/r/Causes": "causes",
    "/r/CausesDesire": "causes_desire",
    "/r/HasSubevent": "has_subevent",
    "/r/HasFirstSubevent": "has_first_subevent",
    "/r/HasLastSubevent": "has_last_subevent",
    "/r/HasPrerequisite": "has_prerequisite",
    "/r/MotivatedByGoal": "motivated_by",
    "/r/ObstructedBy": "obstructed_by",
    # Desires/Mental states
    "/r/Desires": "desires",
    "/r/CreatedBy": "created_by",
    # Lexical
    "/r/Synonym": "synonym",
    "/r/Antonym": "antonym",
    "/r/SimilarTo": "similar_to",
    "/r/DistinctFrom": "distinct_from",
    "/r/DerivedFrom": "derived_from",
    "/r/SymbolOf": "symbol_of",
    "/r/DefinedAs": "defined_as",
    "/r/MannerOf": "manner_of",
    "/r/RelatedTo": "related_to",
    "/r/FormOf": "form_of",
    "/r/EtymologicallyRelatedTo": "etymologically_related",
    "/r/EtymologicallyDerivedFrom": "etymologically_derived",
    # External links
    "/r/ExternalURL": "external_url",
    "/r/dbpedia": "dbpedia_link",
}


def resolve_relation(rel_uri: str) -> str:
    """Convert ConceptNet relation URI to readable name."""
    return CONCEPTNET_RELATIONS.get(rel_uri, rel_uri.split("/")[-1].lower())


# =============================================================================
# CONCEPTNET API CLIENT
# =============================================================================

CONCEPTNET_API_BASE = "http://api.conceptnet.io"


@dataclass
class ConceptNetEdge:
    """A ConceptNet edge (assertion)."""

    start: str  # Start concept
    end: str  # End concept
    relation: str  # Relation type
    weight: float  # Confidence weight
    surface_text: str  # Human-readable form
    language: str  # Language code
    dataset: str  # Source dataset

    def to_triple(self) -> tuple[str, str, str]:
        """Convert to (subject, predicate, object) triple."""
        return (self.start, resolve_relation(self.relation), self.end)


def parse_concept_uri(uri: str) -> tuple[str, str]:
    """
    Parse ConceptNet concept URI.
    /c/en/dog -> ('en', 'dog')
    /c/en/dog/n -> ('en', 'dog')
    """
    parts = uri.split("/")
    if len(parts) >= 4:
        lang = parts[2]
        concept = parts[3]
        return lang, concept
    return "en", uri


def conceptnet_lookup(
    concept: str, language: str = "en", limit: int = 100
) -> list[ConceptNetEdge]:
    """
    Look up a concept in ConceptNet API.

    Args:
        concept: The concept to look up (e.g., 'dog', 'computer')
        language: Language code (default 'en')
        limit: Maximum edges to return

    Returns:
        List of ConceptNetEdge objects
    """
    # Build URI
    concept_uri = f"/c/{language}/{concept.lower().replace(' ', '_')}"
    url = f"{CONCEPTNET_API_BASE}{concept_uri}?limit={limit}"

    headers = {"User-Agent": "DorianConceptNetLoader/1.0 (research project)"}

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"API error: {e}")
        return []

    return parse_api_response(data)


def conceptnet_query(
    start: str = None, end: str = None, rel: str = None, limit: int = 100
) -> list[ConceptNetEdge]:
    """
    Query ConceptNet for edges matching criteria.

    Args:
        start: Start concept (e.g., '/c/en/dog')
        end: End concept
        rel: Relation (e.g., '/r/IsA')
        limit: Maximum results

    Returns:
        List of matching edges
    """
    params = {"limit": limit}
    if start:
        params["start"] = start if start.startswith("/c/") else f"/c/en/{start}"
    if end:
        params["end"] = end if end.startswith("/c/") else f"/c/en/{end}"
    if rel:
        params["rel"] = rel if rel.startswith("/r/") else f"/r/{rel}"

    url = f"{CONCEPTNET_API_BASE}/query?{urllib.parse.urlencode(params)}"

    headers = {"User-Agent": "DorianConceptNetLoader/1.0 (research project)"}

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"API error: {e}")
        return []

    return parse_api_response(data)


def conceptnet_related(
    concept: str, language: str = "en", limit: int = 50
) -> list[tuple[str, float]]:
    """
    Find concepts related to given concept using embeddings.

    Returns:
        List of (concept, similarity_score) tuples
    """
    concept_uri = f"/c/{language}/{concept.lower().replace(' ', '_')}"
    url = f"{CONCEPTNET_API_BASE}/related{concept_uri}?limit={limit}"

    headers = {"User-Agent": "DorianConceptNetLoader/1.0 (research project)"}

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"API error: {e}")
        return []

    results = []
    for item in data.get("related", []):
        concept_id = item.get("@id", "")
        _, concept_name = parse_concept_uri(concept_id)
        score = item.get("weight", 0)
        results.append((concept_name, score))

    return results


def parse_api_response(data: dict) -> list[ConceptNetEdge]:
    """Parse ConceptNet API JSON response."""
    edges = []

    for edge_data in data.get("edges", []):
        try:
            # Start node
            start_data = edge_data.get("start", {})
            start_lang, start_concept = parse_concept_uri(start_data.get("@id", ""))

            # End node
            end_data = edge_data.get("end", {})
            end_lang, end_concept = parse_concept_uri(end_data.get("@id", ""))

            # Relation
            rel_data = edge_data.get("rel", {})
            relation = rel_data.get("@id", "")

            # Skip non-English or external links
            if start_lang != "en" or end_lang != "en":
                continue
            if relation == "/r/ExternalURL":
                continue

            edge = ConceptNetEdge(
                start=start_concept,
                end=end_concept,
                relation=relation,
                weight=edge_data.get("weight", 1.0),
                surface_text=edge_data.get("surfaceText", ""),
                language="en",
                dataset=edge_data.get("dataset", ""),
            )
            edges.append(edge)

        except Exception:
            continue

    return edges


# =============================================================================
# BULK DUMP PROCESSOR
# =============================================================================


def parse_dump_line(line: str) -> ConceptNetEdge | None:
    """
    Parse a line from ConceptNet bulk dump.

    Format: edge_uri \t relation \t start \t end \t metadata_json
    """
    parts = line.strip().split("\t")
    if len(parts) != 5:
        return None

    edge_uri, relation, start, end, metadata_str = parts

    # Parse start/end concepts
    start_lang, start_concept = parse_concept_uri(start)
    end_lang, end_concept = parse_concept_uri(end)

    # Filter to English only
    if start_lang != "en" or end_lang != "en":
        return None

    # Skip external links
    if relation == "/r/ExternalURL":
        return None

    # Parse metadata
    try:
        metadata = json.loads(metadata_str)
    except Exception:
        metadata = {}

    return ConceptNetEdge(
        start=start_concept,
        end=end_concept,
        relation=relation,
        weight=metadata.get("weight", 1.0),
        surface_text=metadata.get("surfaceText", ""),
        language="en",
        dataset=metadata.get("dataset", ""),
    )


def stream_conceptnet_dump(
    filepath: str,
    max_edges: int = None,
    min_weight: float = 1.0,
    relations: set[str] = None,
) -> Iterator[ConceptNetEdge]:
    """
    Stream edges from ConceptNet bulk dump.

    Args:
        filepath: Path to assertions file (.csv or .csv.gz)
        max_edges: Maximum edges to yield
        min_weight: Minimum weight threshold
        relations: Set of relations to include (None = all)

    Yields:
        ConceptNetEdge objects
    """
    # Determine if gzipped
    if filepath.endswith(".gz"):
        opener = gzip.open
        mode = "rt"
    else:
        opener = open
        mode = "r"

    count = 0

    with opener(filepath, mode, encoding="utf-8", errors="replace") as f:
        for line in f:
            edge = parse_dump_line(line)

            if edge is None:
                continue

            # Apply filters
            if edge.weight < min_weight:
                continue

            if relations and edge.relation not in relations:
                continue

            yield edge
            count += 1

            if max_edges and count >= max_edges:
                break


# =============================================================================
# SAMPLE DATA FOR OFFLINE TESTING
# =============================================================================

# Common sense facts for testing
SAMPLE_EDGES = [
    # Taxonomic (IsA)
    ConceptNetEdge(
        "dog", "animal", "/r/IsA", 4.0, "a dog is an animal", "en", "/d/conceptnet"
    ),
    ConceptNetEdge(
        "cat", "animal", "/r/IsA", 4.0, "a cat is an animal", "en", "/d/conceptnet"
    ),
    ConceptNetEdge(
        "animal",
        "living_thing",
        "/r/IsA",
        4.0,
        "an animal is a living thing",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "car", "vehicle", "/r/IsA", 4.0, "a car is a vehicle", "en", "/d/conceptnet"
    ),
    ConceptNetEdge(
        "bicycle",
        "vehicle",
        "/r/IsA",
        4.0,
        "a bicycle is a vehicle",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "hammer", "tool", "/r/IsA", 4.0, "a hammer is a tool", "en", "/d/conceptnet"
    ),
    ConceptNetEdge(
        "screwdriver",
        "tool",
        "/r/IsA",
        4.0,
        "a screwdriver is a tool",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "apple", "fruit", "/r/IsA", 4.0, "an apple is a fruit", "en", "/d/conceptnet"
    ),
    ConceptNetEdge(
        "banana", "fruit", "/r/IsA", 4.0, "a banana is a fruit", "en", "/d/conceptnet"
    ),
    ConceptNetEdge(
        "fruit", "food", "/r/IsA", 4.0, "fruit is food", "en", "/d/conceptnet"
    ),
    # Part-whole (HasA, PartOf)
    ConceptNetEdge(
        "car", "wheel", "/r/HasA", 4.0, "a car has wheels", "en", "/d/conceptnet"
    ),
    ConceptNetEdge(
        "car", "engine", "/r/HasA", 4.0, "a car has an engine", "en", "/d/conceptnet"
    ),
    ConceptNetEdge(
        "bird", "wing", "/r/HasA", 4.0, "a bird has wings", "en", "/d/conceptnet"
    ),
    ConceptNetEdge(
        "tree", "leaf", "/r/HasA", 4.0, "a tree has leaves", "en", "/d/conceptnet"
    ),
    ConceptNetEdge(
        "house", "room", "/r/HasA", 4.0, "a house has rooms", "en", "/d/conceptnet"
    ),
    ConceptNetEdge(
        "wheel",
        "car",
        "/r/PartOf",
        4.0,
        "a wheel is part of a car",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "engine",
        "car",
        "/r/PartOf",
        4.0,
        "an engine is part of a car",
        "en",
        "/d/conceptnet",
    ),
    # Properties (HasProperty)
    ConceptNetEdge(
        "fire", "hot", "/r/HasProperty", 4.0, "fire is hot", "en", "/d/conceptnet"
    ),
    ConceptNetEdge(
        "ice", "cold", "/r/HasProperty", 4.0, "ice is cold", "en", "/d/conceptnet"
    ),
    ConceptNetEdge(
        "sugar", "sweet", "/r/HasProperty", 4.0, "sugar is sweet", "en", "/d/conceptnet"
    ),
    ConceptNetEdge(
        "lemon", "sour", "/r/HasProperty", 4.0, "a lemon is sour", "en", "/d/conceptnet"
    ),
    ConceptNetEdge(
        "sky", "blue", "/r/HasProperty", 2.0, "the sky is blue", "en", "/d/conceptnet"
    ),
    ConceptNetEdge(
        "grass", "green", "/r/HasProperty", 2.0, "grass is green", "en", "/d/conceptnet"
    ),
    # Materials (MadeOf)
    ConceptNetEdge(
        "table",
        "wood",
        "/r/MadeOf",
        3.0,
        "a table is made of wood",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "window",
        "glass",
        "/r/MadeOf",
        3.0,
        "a window is made of glass",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "shirt",
        "cotton",
        "/r/MadeOf",
        2.0,
        "a shirt is made of cotton",
        "en",
        "/d/conceptnet",
    ),
    # Locations (AtLocation)
    ConceptNetEdge(
        "bed",
        "bedroom",
        "/r/AtLocation",
        4.0,
        "a bed is in a bedroom",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "stove",
        "kitchen",
        "/r/AtLocation",
        4.0,
        "a stove is in a kitchen",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "toilet",
        "bathroom",
        "/r/AtLocation",
        4.0,
        "a toilet is in a bathroom",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "desk",
        "office",
        "/r/AtLocation",
        3.0,
        "a desk is in an office",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "book",
        "library",
        "/r/AtLocation",
        3.0,
        "books are in a library",
        "en",
        "/d/conceptnet",
    ),
    # Usage (UsedFor)
    ConceptNetEdge(
        "knife",
        "cutting",
        "/r/UsedFor",
        4.0,
        "a knife is used for cutting",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "pen",
        "writing",
        "/r/UsedFor",
        4.0,
        "a pen is used for writing",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "hammer",
        "driving_nails",
        "/r/UsedFor",
        4.0,
        "a hammer is used for driving nails",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "car",
        "transportation",
        "/r/UsedFor",
        4.0,
        "a car is used for transportation",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "phone",
        "communication",
        "/r/UsedFor",
        4.0,
        "a phone is used for communication",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "oven",
        "cooking",
        "/r/UsedFor",
        4.0,
        "an oven is used for cooking",
        "en",
        "/d/conceptnet",
    ),
    # Capabilities (CapableOf)
    ConceptNetEdge(
        "bird", "fly", "/r/CapableOf", 4.0, "a bird can fly", "en", "/d/conceptnet"
    ),
    ConceptNetEdge(
        "fish", "swim", "/r/CapableOf", 4.0, "a fish can swim", "en", "/d/conceptnet"
    ),
    ConceptNetEdge(
        "dog", "bark", "/r/CapableOf", 4.0, "a dog can bark", "en", "/d/conceptnet"
    ),
    ConceptNetEdge(
        "human", "think", "/r/CapableOf", 4.0, "humans can think", "en", "/d/conceptnet"
    ),
    ConceptNetEdge(
        "human", "speak", "/r/CapableOf", 4.0, "humans can speak", "en", "/d/conceptnet"
    ),
    # Causation (Causes)
    ConceptNetEdge(
        "rain",
        "wet",
        "/r/Causes",
        3.0,
        "rain causes things to be wet",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "fire", "burn", "/r/Causes", 3.0, "fire causes burning", "en", "/d/conceptnet"
    ),
    ConceptNetEdge(
        "exercise",
        "sweat",
        "/r/Causes",
        2.0,
        "exercise causes sweating",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "eating",
        "full",
        "/r/Causes",
        2.0,
        "eating causes feeling full",
        "en",
        "/d/conceptnet",
    ),
    # Desires
    ConceptNetEdge(
        "person",
        "happiness",
        "/r/Desires",
        3.0,
        "people desire happiness",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "dog", "food", "/r/Desires", 3.0, "dogs desire food", "en", "/d/conceptnet"
    ),
    # Antonyms
    ConceptNetEdge(
        "hot",
        "cold",
        "/r/Antonym",
        4.0,
        "hot is the opposite of cold",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "big",
        "small",
        "/r/Antonym",
        4.0,
        "big is the opposite of small",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "good",
        "bad",
        "/r/Antonym",
        4.0,
        "good is the opposite of bad",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "light",
        "dark",
        "/r/Antonym",
        4.0,
        "light is the opposite of dark",
        "en",
        "/d/conceptnet",
    ),
    # Synonyms
    ConceptNetEdge(
        "big",
        "large",
        "/r/Synonym",
        3.0,
        "big is similar to large",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "happy",
        "joyful",
        "/r/Synonym",
        3.0,
        "happy is similar to joyful",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "fast",
        "quick",
        "/r/Synonym",
        3.0,
        "fast is similar to quick",
        "en",
        "/d/conceptnet",
    ),
    # Prerequisites
    ConceptNetEdge(
        "drive",
        "have_license",
        "/r/HasPrerequisite",
        3.0,
        "driving requires a license",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "cook",
        "have_ingredients",
        "/r/HasPrerequisite",
        3.0,
        "cooking requires ingredients",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "read",
        "open_book",
        "/r/HasPrerequisite",
        2.0,
        "reading requires opening a book",
        "en",
        "/d/conceptnet",
    ),
    # Hardware store specific
    ConceptNetEdge(
        "nail", "fastener", "/r/IsA", 3.0, "a nail is a fastener", "en", "/d/conceptnet"
    ),
    ConceptNetEdge(
        "screw",
        "fastener",
        "/r/IsA",
        3.0,
        "a screw is a fastener",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "bolt", "fastener", "/r/IsA", 3.0, "a bolt is a fastener", "en", "/d/conceptnet"
    ),
    ConceptNetEdge(
        "lumber",
        "building_material",
        "/r/IsA",
        3.0,
        "lumber is a building material",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "plywood",
        "building_material",
        "/r/IsA",
        3.0,
        "plywood is a building material",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "drywall",
        "building_material",
        "/r/IsA",
        3.0,
        "drywall is a building material",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "paint", "coating", "/r/IsA", 2.0, "paint is a coating", "en", "/d/conceptnet"
    ),
    ConceptNetEdge(
        "drill",
        "power_tool",
        "/r/IsA",
        3.0,
        "a drill is a power tool",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "saw", "tool", "/r/IsA", 3.0, "a saw is a tool", "en", "/d/conceptnet"
    ),
    ConceptNetEdge(
        "screwdriver",
        "driving_screws",
        "/r/UsedFor",
        4.0,
        "a screwdriver is for driving screws",
        "en",
        "/d/conceptnet",
    ),
    ConceptNetEdge(
        "hammer",
        "tool_box",
        "/r/AtLocation",
        2.0,
        "a hammer is in a tool box",
        "en",
        "/d/conceptnet",
    ),
]


def load_sample_edges() -> list[ConceptNetEdge]:
    """Return sample edges for offline testing."""
    return SAMPLE_EDGES


def get_sample_triples() -> list[tuple[str, str, str]]:
    """Get all triples from sample edges."""
    triples = []

    for edge in SAMPLE_EDGES:
        triples.append(edge.to_triple())

        # Add type information for concepts
        triples.append((edge.start, "is_a", "concept"))
        triples.append((edge.end, "is_a", "concept"))

    # Add relation type information
    for rel_uri, rel_name in CONCEPTNET_RELATIONS.items():
        triples.append((rel_name, "is_a", "relation"))

    return triples


# =============================================================================
# DORIAN INTEGRATION
# =============================================================================


def load_conceptnet_to_dorian(
    core, agent_id: str, edges: list[ConceptNetEdge], show_progress: bool = True
) -> int:
    """
    Load ConceptNet edges into Dorian knowledge base.

    Args:
        core: DorianCore instance
        agent_id: Registered agent ID
        edges: List of ConceptNetEdge objects
        show_progress: Print progress

    Returns:
        Number of facts loaded
    """
    loaded = 0

    # Load relation types first
    for rel_uri, rel_name in CONCEPTNET_RELATIONS.items():
        try:
            core.write(rel_name, "is_a", "relation", agent_id, 1.0)
            loaded += 1
        except Exception:
            pass

    if show_progress:
        print(f"  Loaded {loaded} relation definitions")

    # Load edges
    for i, edge in enumerate(edges):
        # Main triple
        subj, pred, obj = edge.to_triple()

        try:
            result = core.write(
                subj, pred, obj, agent_id, confidence=min(edge.weight / 4.0, 1.0)
            )
            if result.success:
                loaded += 1
        except Exception:
            pass

        # Mark concepts
        try:
            core.write(subj, "is_a", "concept", agent_id, 1.0)
            core.write(obj, "is_a", "concept", agent_id, 1.0)
            loaded += 2
        except Exception:
            pass

        if show_progress and (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(edges)} edges, {loaded} facts")

    if show_progress:
        print(f"  Complete: {loaded} facts from {len(edges)} edges")

    return loaded


def harvest_concept_to_dorian(
    core, agent_id: str, concept: str, depth: int = 2, show_progress: bool = True
) -> int:
    """
    Harvest a concept and its neighbors from ConceptNet API.

    Requires network access.

    Args:
        core: DorianCore instance
        agent_id: Registered agent ID
        concept: Starting concept
        depth: How many hops to explore
        show_progress: Print progress

    Returns:
        Number of facts loaded
    """
    if show_progress:
        print(f"Harvesting concept: {concept}")

    visited = set()
    to_visit = [(concept, 0)]
    all_edges = []

    while to_visit:
        current, current_depth = to_visit.pop(0)

        if current in visited or current_depth > depth:
            continue

        visited.add(current)

        if show_progress:
            print(f"  Fetching: {current} (depth {current_depth})")

        edges = conceptnet_lookup(current, limit=50)
        all_edges.extend(edges)

        # Add neighbors to visit
        if current_depth < depth:
            for edge in edges:
                if edge.end not in visited:
                    to_visit.append((edge.end, current_depth + 1))
                if edge.start not in visited and edge.start != current:
                    to_visit.append((edge.start, current_depth + 1))

        # Rate limiting
        time.sleep(0.5)

    if show_progress:
        print(f"  Total edges: {len(all_edges)}")

    return load_conceptnet_to_dorian(core, agent_id, all_edges, show_progress)


# =============================================================================
# KNOWLEDGE EXTRACTION HELPERS
# =============================================================================


def get_is_a_hierarchy(edges: list[ConceptNetEdge]) -> dict[str, list[str]]:
    """Build IsA hierarchy from edges."""
    hierarchy = defaultdict(list)

    for edge in edges:
        if edge.relation == "/r/IsA":
            hierarchy[edge.end].append(edge.start)

    return dict(hierarchy)


def get_part_whole_relations(edges: list[ConceptNetEdge]) -> dict[str, list[str]]:
    """Extract part-whole relations."""
    parts = defaultdict(list)

    for edge in edges:
        if edge.relation in ["/r/HasA", "/r/PartOf"]:
            if edge.relation == "/r/HasA":
                parts[edge.start].append(edge.end)
            else:
                parts[edge.end].append(edge.start)

    return dict(parts)


def get_properties(edges: list[ConceptNetEdge], concept: str) -> list[str]:
    """Get properties of a concept."""
    props = []

    for edge in edges:
        if edge.start == concept and edge.relation == "/r/HasProperty":
            props.append(edge.end)

    return props


def get_locations(edges: list[ConceptNetEdge], concept: str) -> list[str]:
    """Get typical locations of a concept."""
    locs = []

    for edge in edges:
        if edge.start == concept and edge.relation == "/r/AtLocation":
            locs.append(edge.end)

    return locs


def get_uses(edges: list[ConceptNetEdge], concept: str) -> list[str]:
    """Get uses of a concept."""
    uses = []

    for edge in edges:
        if edge.start == concept and edge.relation == "/r/UsedFor":
            uses.append(edge.end)

    return uses


# =============================================================================
# DEMO / TESTING
# =============================================================================


def demo_sample_data():
    """Demo with sample edges (no network)."""
    print("=" * 70)
    print("CONCEPTNET SAMPLE DATA DEMO")
    print("=" * 70)

    edges = load_sample_edges()
    print(f"\nSample edges: {len(edges)}")

    # Show by relation type
    by_relation = defaultdict(list)
    for edge in edges:
        rel_name = resolve_relation(edge.relation)
        by_relation[rel_name].append(edge)

    print("\nEdges by relation type:")
    for rel, rel_edges in sorted(by_relation.items()):
        print(f"\n  {rel}: {len(rel_edges)} edges")
        for edge in rel_edges[:3]:
            print(f"    {edge.start} -> {edge.end}")

    # Build hierarchy
    print("\n" + "-" * 70)
    print("IsA Hierarchy:")
    hierarchy = get_is_a_hierarchy(edges)

    for parent, children in sorted(hierarchy.items()):
        print(f"  {parent}:")
        for child in children[:5]:
            print(f"    └─ {child}")

    # Show part-whole
    print("\n" + "-" * 70)
    print("Part-Whole Relations:")
    parts = get_part_whole_relations(edges)

    for whole, part_list in sorted(parts.items()):
        print(f"  {whole} has: {', '.join(part_list[:5])}")

    print("\n" + "-" * 70)
    print("Triples extracted:")
    triples = get_sample_triples()
    print(f"Total: {len(triples)}")


def demo_api():
    """Demo ConceptNet API (requires network)."""
    print("=" * 70)
    print("CONCEPTNET API DEMO")
    print("=" * 70)

    print("\nLooking up: 'hammer'")

    try:
        edges = conceptnet_lookup("hammer", limit=20)

        print(f"\nFound {len(edges)} edges:")
        for edge in edges[:10]:
            rel = resolve_relation(edge.relation)
            print(f"  {edge.start} --[{rel}]--> {edge.end} (weight: {edge.weight:.1f})")

        print("\n\nRelated concepts:")
        related = conceptnet_related("hammer", limit=10)
        for concept, score in related[:10]:
            print(f"  {concept}: {score:.3f}")

    except Exception as e:
        print(f"\nAPI demo skipped (network error): {e}")


if __name__ == "__main__":
    print("DORIAN CONCEPTNET LOADER")
    print("=" * 70)

    demo_sample_data()
    print()

    try:
        demo_api()
    except Exception:
        print("\nAPI demo skipped (no network)")
