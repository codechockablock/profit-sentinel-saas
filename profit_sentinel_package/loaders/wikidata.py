"""
DORIAN WIKIDATA LOADER
======================

Load knowledge from Wikidata into Dorian's VSA knowledge base.

Two modes:
1. SPARQL queries - targeted extraction of specific domains
2. Dump streaming - process full dump line by line (for production)

Wikidata structure:
- Q-items: Entities (Q5 = human, Q42 = Douglas Adams)
- P-properties: Relations (P31 = instance of, P279 = subclass of)
- Claims: Subject-Property-Value triples with qualifiers

Key properties for knowledge graphs:
- P31: instance of (Q42 P31 Q5 = "Douglas Adams is a human")
- P279: subclass of (Q5 P279 Q154 = "human is subclass of mammal")
- P361: part of
- P527: has part
- P1542: has effect
- P1552: has quality
"""

import bz2
import gzip
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

# =============================================================================
# WIKIDATA SPARQL CLIENT
# =============================================================================

WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"


def sparql_query(query: str, max_retries: int = 3) -> list[dict]:
    """
    Execute a SPARQL query against Wikidata.
    Returns list of result bindings.
    """
    url = WIKIDATA_SPARQL_ENDPOINT
    params = urllib.parse.urlencode({"query": query, "format": "json"})

    headers = {
        "User-Agent": "DorianKnowledgeLoader/1.0 (https://github.com/dorian; contact@example.com)",
        "Accept": "application/sparql-results+json",
    }

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(f"{url}?{params}", headers=headers)
            with urllib.request.urlopen(req, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
                return data.get("results", {}).get("bindings", [])
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
            else:
                raise e

    return []


def extract_value(binding: dict, var: str) -> str | None:
    """Extract value from SPARQL binding."""
    if var not in binding:
        return None

    val = binding[var]
    if val["type"] == "uri":
        # Extract QID from URI
        uri = val["value"]
        if "/entity/" in uri:
            return uri.split("/entity/")[-1]
        return uri
    elif val["type"] == "literal":
        return val["value"]

    return val.get("value")


# =============================================================================
# DOMAIN-SPECIFIC QUERIES
# =============================================================================


def get_class_hierarchy(root_class: str, depth: int = 3) -> list[tuple[str, str, str]]:
    """
    Get subclass hierarchy starting from a root class.
    Returns list of (subclass, 'subclass_of', parent) triples.

    Example: get_class_hierarchy('Q7397') gets subclasses of "software"
    """
    query = f"""
    SELECT DISTINCT ?item ?itemLabel ?parent ?parentLabel WHERE {{
      ?item wdt:P279* wd:{root_class} .
      ?item wdt:P279 ?parent .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 10000
    """

    results = sparql_query(query)

    triples = []
    for r in results:
        item = extract_value(r, "item")
        item_label = extract_value(r, "itemLabel")
        parent = extract_value(r, "parent")
        parent_label = extract_value(r, "parentLabel")

        if item and parent:
            # Store both QID and label versions
            triples.append((item, "subclass_of", parent))
            if item_label and parent_label:
                triples.append((item_label, "subclass_of", parent_label))

    return triples


def get_instances_of_class(
    class_id: str, limit: int = 1000
) -> list[tuple[str, str, str]]:
    """
    Get instances of a class.
    Returns list of (instance, 'instance_of', class) triples.

    Example: get_instances_of_class('Q5') gets humans (limited)
    """
    query = f"""
    SELECT DISTINCT ?item ?itemLabel WHERE {{
      ?item wdt:P31 wd:{class_id} .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT {limit}
    """

    results = sparql_query(query)

    triples = []
    for r in results:
        item = extract_value(r, "item")
        item_label = extract_value(r, "itemLabel")

        if item:
            triples.append((item, "instance_of", class_id))
            if item_label:
                triples.append((item_label, "instance_of", class_id))

    return triples


def get_entity_properties(entity_id: str) -> list[tuple[str, str, str]]:
    """
    Get all properties/claims for an entity.
    Returns list of (entity, property, value) triples.
    """
    query = f"""
    SELECT ?prop ?propLabel ?value ?valueLabel WHERE {{
      wd:{entity_id} ?p ?statement .
      ?statement ?ps ?value .
      ?prop wikibase:claim ?p .
      ?prop wikibase:statementProperty ?ps .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 100
    """

    results = sparql_query(query)

    triples = []
    for r in results:
        prop = extract_value(r, "propLabel") or extract_value(r, "prop")
        value = extract_value(r, "valueLabel") or extract_value(r, "value")

        if prop and value:
            triples.append((entity_id, prop, value))

    return triples


def get_retail_product_taxonomy() -> list[tuple[str, str, str]]:
    """
    Get product/goods taxonomy relevant to retail.
    """
    # Key classes for retail:
    # Q28877 - goods
    # Q2424752 - product
    # Q3966 - hardware
    # Q987767 - building material

    all_triples = []

    # Get product hierarchy
    product_classes = ["Q28877", "Q2424752", "Q3966", "Q987767"]

    for cls in product_classes:
        triples = get_class_hierarchy(cls, depth=3)
        all_triples.extend(triples)
        time.sleep(1)  # Rate limiting

    return all_triples


def get_common_relations() -> list[tuple[str, str, str]]:
    """
    Get common relation types (properties) from Wikidata.
    Returns (property_id, 'is_a', 'relation') triples.
    """
    query = """
    SELECT ?prop ?propLabel ?propDescription WHERE {
      ?prop a wikibase:Property .
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    LIMIT 5000
    """

    results = sparql_query(query)

    triples = []
    for r in results:
        prop = extract_value(r, "prop")
        label = extract_value(r, "propLabel")
        desc = extract_value(r, "propDescription")

        if prop and label:
            triples.append((prop, "has_label", label))
            triples.append((label, "is_a", "relation"))
            if desc:
                triples.append((prop, "has_description", desc[:200]))

    return triples


# =============================================================================
# DUMP STREAMING PROCESSOR
# =============================================================================


@dataclass
class WikidataEntity:
    """Parsed Wikidata entity."""

    id: str
    type: str  # 'item' or 'property'
    labels: dict[str, str]  # language -> label
    descriptions: dict[str, str]
    aliases: dict[str, list[str]]
    claims: list[dict]  # Raw claim data

    @property
    def label(self) -> str | None:
        """Get English label or first available."""
        return self.labels.get("en") or next(iter(self.labels.values()), None)

    @property
    def description(self) -> str | None:
        """Get English description or first available."""
        return self.descriptions.get("en") or next(
            iter(self.descriptions.values()), None
        )


def parse_entity_line(line: str) -> WikidataEntity | None:
    """
    Parse a single line from Wikidata JSON dump.
    Lines are JSON objects, array elements separated by commas.
    """
    line = line.strip()

    # Skip array brackets
    if line in ["[", "]"]:
        return None

    # Remove trailing comma
    if line.endswith(","):
        line = line[:-1]

    if not line:
        return None

    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None

    # Extract fields
    entity_id = data.get("id")
    entity_type = data.get("type")

    if not entity_id or not entity_type:
        return None

    # Parse labels
    labels = {}
    for lang, label_data in data.get("labels", {}).items():
        labels[lang] = label_data.get("value", "")

    # Parse descriptions
    descriptions = {}
    for lang, desc_data in data.get("descriptions", {}).items():
        descriptions[lang] = desc_data.get("value", "")

    # Parse aliases
    aliases = {}
    for lang, alias_list in data.get("aliases", {}).items():
        aliases[lang] = [a.get("value", "") for a in alias_list]

    # Keep claims raw for now
    claims = data.get("claims", {})

    return WikidataEntity(
        id=entity_id,
        type=entity_type,
        labels=labels,
        descriptions=descriptions,
        aliases=aliases,
        claims=claims,
    )


def extract_triples_from_entity(entity: WikidataEntity) -> list[tuple[str, str, str]]:
    """
    Extract (subject, predicate, object) triples from a Wikidata entity.
    """
    triples = []

    # Label triple
    if entity.label:
        triples.append((entity.id, "has_label", entity.label))

    # Description triple
    if entity.description:
        triples.append((entity.id, "has_description", entity.description[:200]))

    # Type triple
    if entity.type == "item":
        triples.append((entity.id, "is_a", "entity"))
    elif entity.type == "property":
        triples.append((entity.id, "is_a", "property"))

    # Process claims
    for prop_id, claim_list in entity.claims.items():
        for claim in claim_list:
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})

            if not datavalue:
                continue

            value_type = datavalue.get("type")
            value = datavalue.get("value")

            if value_type == "wikibase-entityid":
                # Link to another entity
                target_id = value.get("id")
                if target_id:
                    triples.append((entity.id, prop_id, target_id))

            elif value_type == "string":
                # String value
                triples.append((entity.id, prop_id, value[:200]))

            elif value_type == "quantity":
                # Numeric value
                amount = value.get("amount", "")
                unit = value.get("unit", "").split("/")[-1]  # Extract unit QID
                triples.append((entity.id, prop_id, f"{amount} {unit}".strip()))

            elif value_type == "time":
                # Time value
                time_str = value.get("time", "")
                triples.append((entity.id, prop_id, time_str))

            elif value_type == "monolingualtext":
                # Text in specific language
                text = value.get("text", "")
                triples.append((entity.id, prop_id, text[:200]))

    return triples


def stream_wikidata_dump(
    filepath: str, max_entities: int = None, filter_fn: callable = None
) -> Iterator[WikidataEntity]:
    """
    Stream entities from a Wikidata JSON dump file.

    Args:
        filepath: Path to .json, .json.gz, or .json.bz2 file
        max_entities: Maximum number of entities to yield
        filter_fn: Optional function to filter entities (return True to include)

    Yields:
        WikidataEntity objects
    """
    # Determine compression
    if filepath.endswith(".bz2"):
        opener = bz2.open
    elif filepath.endswith(".gz"):
        opener = gzip.open
    else:
        opener = open

    count = 0

    with opener(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            entity = parse_entity_line(line)

            if entity is None:
                continue

            if filter_fn and not filter_fn(entity):
                continue

            yield entity
            count += 1

            if max_entities and count >= max_entities:
                break


# =============================================================================
# DORIAN INTEGRATION
# =============================================================================


def load_triples_to_dorian(
    core,
    triples: list[tuple[str, str, str]],
    batch_size: int = 1000,
    show_progress: bool = True,
):
    """
    Load triples into Dorian knowledge base.

    Args:
        core: DorianCore instance
        triples: List of (subject, predicate, object) tuples
        batch_size: Number of triples per batch
        show_progress: Whether to print progress
    """
    total = len(triples)
    loaded = 0

    for i in range(0, total, batch_size):
        batch = triples[i : i + batch_size]

        for subj, pred, obj in batch:
            try:
                core.add_fact(str(subj), str(pred), str(obj))
                loaded += 1
            except Exception:
                # Skip problematic triples
                pass

        if show_progress and (i + batch_size) % 10000 == 0:
            print(f"  Loaded {loaded}/{total} triples...")

    if show_progress:
        print(f"  Done: {loaded}/{total} triples loaded")

    return loaded


def load_wikidata_domain(core, domain: str, limit: int = 10000):
    """
    Load a specific domain from Wikidata into Dorian.

    Domains:
    - 'products': Product/goods taxonomy
    - 'materials': Building materials
    - 'companies': Companies and organizations
    - 'relations': Common relation types
    """
    print(f"Loading Wikidata domain: {domain}")

    if domain == "products":
        # Q28877 = goods, Q2424752 = product
        triples = []
        for cls in ["Q28877", "Q2424752"]:
            triples.extend(get_class_hierarchy(cls))
            time.sleep(1)

    elif domain == "materials":
        # Q987767 = building material
        triples = get_class_hierarchy("Q987767")

    elif domain == "hardware":
        # Q3966 = hardware (tools)
        triples = get_class_hierarchy("Q3966")

    elif domain == "companies":
        # Q4830453 = business, Q783794 = company
        triples = []
        for cls in ["Q4830453", "Q783794"]:
            triples.extend(get_class_hierarchy(cls))
            triples.extend(get_instances_of_class(cls, limit=limit))
            time.sleep(1)

    elif domain == "relations":
        triples = get_common_relations()

    else:
        print(f"Unknown domain: {domain}")
        return 0

    print(f"  Retrieved {len(triples)} triples")

    loaded = load_triples_to_dorian(core, triples)

    return loaded


# =============================================================================
# PROPERTY MAPPINGS
# =============================================================================

# Important Wikidata properties for knowledge extraction
IMPORTANT_PROPERTIES = {
    # Classification
    "P31": "instance_of",
    "P279": "subclass_of",
    "P361": "part_of",
    "P527": "has_part",
    # Relations
    "P1542": "has_effect",
    "P1552": "has_quality",
    "P2283": "uses",
    "P366": "use",
    # Products/Business
    "P176": "manufacturer",
    "P127": "owned_by",
    "P1056": "product",
    "P452": "industry",
    "P2770": "source_of_income",
    # Materials
    "P186": "material_used",
    "P516": "powered_by",
    # Location
    "P17": "country",
    "P131": "located_in",
    "P159": "headquarters_location",
    # Time
    "P571": "inception",
    "P576": "dissolved",
    # Identification
    "P373": "commons_category",
    "P910": "topic_main_category",
}


def resolve_property(prop_id: str) -> str:
    """Convert Wikidata property ID to human-readable relation name."""
    return IMPORTANT_PROPERTIES.get(prop_id, prop_id)


# =============================================================================
# DORIAN INTEGRATION
# =============================================================================


def load_wikidata_to_dorian(
    core,
    agent_id: str,
    triples: list[tuple[str, str, str]],
    resolve_properties: bool = True,
    show_progress: bool = True,
) -> int:
    """
    Load Wikidata triples into Dorian knowledge base.

    Args:
        core: DorianCore instance
        agent_id: Registered agent ID for attribution
        triples: List of (subject, predicate, object) tuples
        resolve_properties: Whether to convert P-codes to readable names
        show_progress: Whether to print progress

    Returns:
        Number of facts successfully loaded
    """
    loaded = 0
    failed = 0

    for i, (subj, pred, obj) in enumerate(triples):
        # Resolve property code to name
        if resolve_properties:
            pred = resolve_property(pred)

        try:
            result = core.write(
                subject=str(subj),
                predicate=str(pred),
                obj=str(obj),
                agent_id=agent_id,
                confidence=1.0,
            )
            if result.success:
                loaded += 1
            else:
                failed += 1
        except Exception:
            failed += 1

        if show_progress and (i + 1) % 1000 == 0:
            print(f"  Loaded {loaded}/{i+1} facts...")

    if show_progress:
        print(f"  Complete: {loaded} loaded, {failed} failed")

    return loaded


def load_wikidata_domain_to_dorian(
    core, agent_id: str, domain: str, limit: int = 10000
) -> int:
    """
    Load a specific Wikidata domain into Dorian.

    Requires network access for SPARQL queries.

    Domains:
    - 'products': Product/goods taxonomy
    - 'materials': Building materials
    - 'hardware': Tools and hardware
    - 'companies': Companies and organizations
    - 'relations': Common relation types

    Returns:
        Number of facts loaded
    """
    print(f"Loading Wikidata domain: {domain}")

    triples = []

    if domain == "products":
        for cls in ["Q28877", "Q2424752"]:  # goods, product
            triples.extend(get_class_hierarchy(cls))
            time.sleep(1)

    elif domain == "materials":
        triples = get_class_hierarchy("Q987767")  # building material

    elif domain == "hardware":
        triples = get_class_hierarchy("Q3966")  # hardware

    elif domain == "companies":
        for cls in ["Q4830453", "Q783794"]:  # business, company
            triples.extend(get_class_hierarchy(cls))
            triples.extend(get_instances_of_class(cls, limit=limit))
            time.sleep(1)

    elif domain == "relations":
        triples = get_common_relations()

    else:
        print(f"Unknown domain: {domain}")
        return 0

    print(f"  Retrieved {len(triples)} triples from Wikidata")

    return load_wikidata_to_dorian(core, agent_id, triples)


def stream_wikidata_dump_to_dorian(
    core,
    agent_id: str,
    filepath: str,
    max_entities: int = None,
    filter_fn: callable = None,
    show_progress: bool = True,
) -> int:
    """
    Stream a Wikidata JSON dump directly into Dorian.

    Memory efficient - processes one entity at a time.

    Args:
        core: DorianCore instance
        agent_id: Registered agent ID
        filepath: Path to Wikidata JSON dump (.json, .json.gz, or .json.bz2)
        max_entities: Maximum entities to process
        filter_fn: Optional filter function(entity) -> bool
        show_progress: Print progress

    Returns:
        Number of facts loaded
    """
    loaded = 0
    entities_processed = 0

    for entity in stream_wikidata_dump(filepath, max_entities, filter_fn):
        triples = extract_triples_from_entity(entity)

        for subj, pred, obj in triples:
            pred_name = resolve_property(pred)
            try:
                result = core.write(
                    subject=str(subj),
                    predicate=str(pred_name),
                    obj=str(obj),
                    agent_id=agent_id,
                    confidence=1.0,
                )
                if result.success:
                    loaded += 1
            except:
                pass

        entities_processed += 1

        if show_progress and entities_processed % 10000 == 0:
            print(f"  Processed {entities_processed} entities, {loaded} facts loaded")

    if show_progress:
        print(f"  Complete: {entities_processed} entities, {loaded} facts")

    return loaded


# =============================================================================
# DEMO / TESTING
# =============================================================================


def demo_sparql():
    """Demo SPARQL queries."""
    print("=" * 60)
    print("WIKIDATA SPARQL DEMO")
    print("=" * 60)

    # Get some building materials
    print("\n1. Building Materials Hierarchy:")
    triples = get_class_hierarchy("Q987767")  # building material
    print(f"   Found {len(triples)} triples")
    for t in triples[:10]:
        print(f"   {t}")

    # Get Douglas Adams properties
    print("\n2. Douglas Adams (Q42) Properties:")
    triples = get_entity_properties("Q42")
    print(f"   Found {len(triples)} triples")
    for t in triples[:10]:
        print(f"   {t}")

    # Get some companies
    print("\n3. Company Instances:")
    triples = get_instances_of_class("Q783794", limit=20)  # company
    print(f"   Found {len(triples)} triples")
    for t in triples[:10]:
        print(f"   {t}")


def demo_entity_parsing():
    """Demo parsing a Wikidata entity JSON."""
    print("=" * 60)
    print("ENTITY PARSING DEMO")
    print("=" * 60)

    # Sample entity JSON (simplified)
    sample_json = """
    {
        "id": "Q42",
        "type": "item",
        "labels": {
            "en": {"value": "Douglas Adams"},
            "de": {"value": "Douglas Adams"}
        },
        "descriptions": {
            "en": {"value": "British author and humorist"}
        },
        "aliases": {
            "en": [{"value": "Douglas Noel Adams"}]
        },
        "claims": {
            "P31": [{"mainsnak": {"datavalue": {"type": "wikibase-entityid", "value": {"id": "Q5"}}}}],
            "P569": [{"mainsnak": {"datavalue": {"type": "time", "value": {"time": "+1952-03-11T00:00:00Z"}}}}]
        }
    }
    """

    entity = parse_entity_line(sample_json)

    print("\nParsed entity:")
    print(f"  ID: {entity.id}")
    print(f"  Type: {entity.type}")
    print(f"  Label: {entity.label}")
    print(f"  Description: {entity.description}")

    print("\nExtracted triples:")
    triples = extract_triples_from_entity(entity)
    for t in triples:
        print(f"  {t}")


if __name__ == "__main__":
    print("DORIAN WIKIDATA LOADER")
    print("=" * 60)

    # Run demos
    demo_entity_parsing()
    print()

    # Only run SPARQL demo if network available
    try:
        demo_sparql()
    except Exception as e:
        print(f"\nSPARQL demo skipped (network error): {e}")
