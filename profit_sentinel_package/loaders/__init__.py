"""
Knowledge Loaders for Dorian

Load facts from various external sources into the Dorian knowledge engine.

Available Loaders:
- WikidataLoader: SPARQL queries + dump streaming from Wikidata
- ArxivLoader: Paper metadata and abstracts from arXiv
- ConceptNetLoader: Common sense knowledge from ConceptNet (34M assertions)

Example:
    from loaders.wikidata import WikidataLoader
    from dorian.core import DorianCore

    dorian = DorianCore()
    loader = WikidataLoader(dorian)
    loader.load_entities(['Q2', 'Q405'])  # Earth, Moon
"""

from .arxiv import ArxivLoader
from .conceptnet import ConceptNetLoader
from .wikidata import WikidataLoader

__all__ = [
    "WikidataLoader",
    "ArxivLoader",
    "ConceptNetLoader",
]
