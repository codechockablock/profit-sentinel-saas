"""
DORIAN ARXIV LOADER
===================

Load scientific paper metadata from arXiv into Dorian's VSA knowledge base.

Three access methods:
1. arXiv API - Search/query interface (Atom XML)
2. OAI-PMH - Bulk metadata harvesting
3. Pre-harvested dumps - Kaggle/Archive.org datasets

arXiv structure:
- Papers have: id, title, authors, abstract, categories, dates
- Categories: cs.AI, math.CO, physics.hep-th, etc.
- ~2.5M papers total

Knowledge we extract:
- Paper -> has_author -> Author
- Paper -> has_category -> Category
- Paper -> published_in -> Year
- Paper -> has_concept -> Concept (from title/abstract)
- Author -> collaborates_with -> Author
- Category -> related_to -> Category
"""

import gzip
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree  # noqa: N817
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# =============================================================================
# ARXIV API CLIENT
# =============================================================================

ARXIV_API_BASE = "http://export.arxiv.org/api/query"
ARXIV_OAI_BASE = "https://oaipmh.arxiv.org/oai"

# Namespaces for parsing
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
OAI_NS = "{http://www.openarchives.org/OAI/2.0/}"
DC_NS = "{http://purl.org/dc/elements/1.1/}"
OAIDC_NS = "{http://www.openarchives.org/OAI/2.0/oai_dc/}"


@dataclass
class ArxivPaper:
    """Parsed arXiv paper metadata."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    primary_category: str
    published: str  # ISO date
    updated: str
    doi: str | None = None
    journal_ref: str | None = None
    comment: str | None = None

    def to_triples(self) -> list[tuple[str, str, str]]:
        """Convert paper to knowledge triples."""
        triples = []
        paper_id = f"arxiv:{self.arxiv_id}"

        # Basic metadata
        triples.append((paper_id, "is_a", "paper"))
        triples.append((paper_id, "has_title", self.title[:200]))

        # Authors
        for author in self.authors:
            author.lower().replace(" ", "_")
            triples.append((paper_id, "has_author", author))
            triples.append((author, "is_a", "researcher"))

        # Author collaborations
        for i, a1 in enumerate(self.authors):
            for a2 in self.authors[i + 1 :]:
                triples.append((a1, "collaborates_with", a2))

        # Categories
        triples.append((paper_id, "has_category", self.primary_category))
        for cat in self.categories:
            if cat != self.primary_category:
                triples.append((paper_id, "has_category", cat))
            triples.append((cat, "is_a", "arxiv_category"))

        # Date
        year = self.published[:4] if self.published else "unknown"
        triples.append((paper_id, "published_year", year))

        # DOI if available
        if self.doi:
            triples.append((paper_id, "has_doi", self.doi))

        # Extract concepts from title
        concepts = extract_concepts(self.title)
        for concept in concepts[:10]:  # Limit to top 10
            triples.append((paper_id, "about", concept))
            triples.append((concept, "is_a", "concept"))

        return triples


def extract_concepts(text: str) -> list[str]:
    """
    Extract key concepts from paper title/abstract.
    Simple extraction - could be enhanced with NLP.
    """
    # Remove common words
    stopwords = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "been",
        "be",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "we",
        "our",
        "their",
        "using",
        "via",
        "based",
        "new",
        "novel",
        "approach",
        "method",
        "methods",
        "results",
        "show",
        "paper",
        "study",
        "analysis",
        "model",
        "models",
    }

    # Extract words
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*\b", text.lower())

    # Filter and dedupe
    concepts = []
    seen = set()
    for word in words:
        if word not in stopwords and len(word) > 2 and word not in seen:
            seen.add(word)
            concepts.append(word)

    return concepts


def arxiv_api_search(
    query: str,
    start: int = 0,
    max_results: int = 100,
    sort_by: str = "submittedDate",
    sort_order: str = "descending",
) -> list[ArxivPaper]:
    """
    Search arXiv via API.

    Query syntax:
    - ti:keyword - title
    - au:author - author
    - abs:keyword - abstract
    - cat:cs.AI - category
    - all:keyword - all fields

    Examples:
    - "ti:neural network"
    - "au:hinton AND cat:cs.LG"
    - "cat:cs.AI AND submittedDate:[2023 TO 2024]"
    """
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }

    url = f"{ARXIV_API_BASE}?{urllib.parse.urlencode(params)}"

    headers = {"User-Agent": "DorianArxivLoader/1.0 (research project)"}

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as response:
            xml_data = response.read().decode("utf-8")
    except Exception as e:
        print(f"API error: {e}")
        return []

    return parse_arxiv_api_response(xml_data)


def parse_arxiv_api_response(xml_data: str) -> list[ArxivPaper]:
    """Parse arXiv API Atom response."""
    papers = []

    try:
        root = ElementTree.fromstring(xml_data)
    except ElementTree.ParseError as e:
        print(f"XML parse error: {e}")
        return []

    for entry in root.findall(f"{ATOM_NS}entry"):
        try:
            # ID (extract arxiv ID from URL)
            id_elem = entry.find(f"{ATOM_NS}id")
            arxiv_id = id_elem.text.split("/abs/")[-1] if id_elem is not None else ""

            # Title
            title_elem = entry.find(f"{ATOM_NS}title")
            title = (
                title_elem.text.strip().replace("\n", " ")
                if title_elem is not None
                else ""
            )

            # Abstract
            summary_elem = entry.find(f"{ATOM_NS}summary")
            abstract = (
                summary_elem.text.strip().replace("\n", " ")
                if summary_elem is not None
                else ""
            )

            # Authors
            authors = []
            for author_elem in entry.findall(f"{ATOM_NS}author"):
                name_elem = author_elem.find(f"{ATOM_NS}name")
                if name_elem is not None:
                    authors.append(name_elem.text)

            # Categories
            categories = []
            primary_category = ""

            # Primary category
            prim_cat = entry.find(f"{ARXIV_NS}primary_category")
            if prim_cat is not None:
                primary_category = prim_cat.get("term", "")
                categories.append(primary_category)

            # All categories
            for cat in entry.findall(f"{ATOM_NS}category"):
                term = cat.get("term", "")
                if term and term not in categories:
                    categories.append(term)

            # Dates
            published_elem = entry.find(f"{ATOM_NS}published")
            published = published_elem.text if published_elem is not None else ""

            updated_elem = entry.find(f"{ATOM_NS}updated")
            updated = updated_elem.text if updated_elem is not None else ""

            # Optional fields
            doi_elem = entry.find(f"{ARXIV_NS}doi")
            doi = doi_elem.text if doi_elem is not None else None

            journal_elem = entry.find(f"{ARXIV_NS}journal_ref")
            journal_ref = journal_elem.text if journal_elem is not None else None

            comment_elem = entry.find(f"{ARXIV_NS}comment")
            comment = comment_elem.text if comment_elem is not None else None

            paper = ArxivPaper(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract[:1000],  # Truncate
                categories=categories,
                primary_category=primary_category,
                published=published,
                updated=updated,
                doi=doi,
                journal_ref=journal_ref,
                comment=comment,
            )
            papers.append(paper)

        except Exception as e:
            print(f"Error parsing entry: {e}")
            continue

    return papers


# =============================================================================
# OAI-PMH HARVESTER
# =============================================================================


def oai_harvest(
    set_spec: str = None,
    from_date: str = None,
    until_date: str = None,
    resumption_token: str = None,
    max_records: int = 1000,
) -> tuple[list[ArxivPaper], str | None]:
    """
    Harvest metadata via OAI-PMH.

    Args:
        set_spec: Category set (e.g., "cs" for computer science)
        from_date: Start date (YYYY-MM-DD)
        until_date: End date (YYYY-MM-DD)
        resumption_token: Token for pagination
        max_records: Maximum records to fetch

    Returns:
        (papers, next_resumption_token)
    """
    params = {
        "verb": "ListRecords",
        "metadataPrefix": "oai_dc",  # Dublin Core format
    }

    if resumption_token:
        params = {"verb": "ListRecords", "resumptionToken": resumption_token}
    else:
        if set_spec:
            params["set"] = set_spec
        if from_date:
            params["from"] = from_date
        if until_date:
            params["until"] = until_date

    url = f"{ARXIV_OAI_BASE}?{urllib.parse.urlencode(params)}"

    headers = {"User-Agent": "DorianArxivLoader/1.0 (research project)"}

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as response:
            xml_data = response.read().decode("utf-8")
    except Exception as e:
        print(f"OAI-PMH error: {e}")
        return [], None

    return parse_oai_response(xml_data)


def parse_oai_response(xml_data: str) -> tuple[list[ArxivPaper], str | None]:
    """Parse OAI-PMH response."""
    papers = []
    resumption_token = None

    try:
        root = ElementTree.fromstring(xml_data)
    except ElementTree.ParseError as e:
        print(f"XML parse error: {e}")
        return [], None

    # Check for errors
    error = root.find(f".//{OAI_NS}error")
    if error is not None:
        print(f"OAI error: {error.text}")
        return [], None

    # Find ListRecords
    list_records = root.find(f".//{OAI_NS}ListRecords")
    if list_records is None:
        return [], None

    # Get resumption token
    token_elem = list_records.find(f"{OAI_NS}resumptionToken")
    if token_elem is not None and token_elem.text:
        resumption_token = token_elem.text

    # Parse records
    for record in list_records.findall(f"{OAI_NS}record"):
        try:
            header = record.find(f"{OAI_NS}header")
            metadata = record.find(f"{OAI_NS}metadata")

            if header is None or metadata is None:
                continue

            # Check if deleted
            if header.get("status") == "deleted":
                continue

            # Get identifier (arxiv ID)
            identifier = header.find(f"{OAI_NS}identifier")
            if identifier is None:
                continue
            arxiv_id = identifier.text.replace("oai:arXiv.org:", "")

            # Get datestamp
            datestamp = header.find(f"{OAI_NS}datestamp")
            published = datestamp.text if datestamp is not None else ""

            # Get setSpec (categories)
            categories = []
            for set_spec in header.findall(f"{OAI_NS}setSpec"):
                if set_spec.text:
                    # Convert "cs:cs:AI" to "cs.AI"
                    parts = set_spec.text.split(":")
                    if len(parts) >= 2:
                        cat = f"{parts[0]}.{parts[-1]}" if len(parts) > 2 else parts[-1]
                        categories.append(cat)

            # Get Dublin Core metadata
            dc = metadata.find(f".//{OAIDC_NS}dc")
            if dc is None:
                dc = metadata  # Try direct access

            title = ""
            abstract = ""
            authors = []

            for child in dc:
                tag = child.tag.replace(DC_NS, "").replace(OAIDC_NS, "")

                if "title" in tag and child.text:
                    title = child.text.strip()
                elif "description" in tag and child.text:
                    abstract = child.text.strip()[:1000]
                elif "creator" in tag and child.text:
                    authors.append(child.text.strip())

            paper = ArxivPaper(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                categories=categories,
                primary_category=categories[0] if categories else "",
                published=published,
                updated=published,
            )
            papers.append(paper)

        except Exception:
            continue

    return papers, resumption_token


# =============================================================================
# CATEGORY DEFINITIONS
# =============================================================================

ARXIV_CATEGORIES = {
    # Computer Science
    "cs.AI": "Artificial Intelligence",
    "cs.CL": "Computation and Language",
    "cs.CC": "Computational Complexity",
    "cs.CE": "Computational Engineering",
    "cs.CG": "Computational Geometry",
    "cs.GT": "Computer Science and Game Theory",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.CY": "Computers and Society",
    "cs.CR": "Cryptography and Security",
    "cs.DS": "Data Structures and Algorithms",
    "cs.DB": "Databases",
    "cs.DL": "Digital Libraries",
    "cs.DM": "Discrete Mathematics",
    "cs.DC": "Distributed Computing",
    "cs.ET": "Emerging Technologies",
    "cs.FL": "Formal Languages",
    "cs.GL": "General Literature",
    "cs.GR": "Graphics",
    "cs.AR": "Hardware Architecture",
    "cs.HC": "Human-Computer Interaction",
    "cs.IR": "Information Retrieval",
    "cs.IT": "Information Theory",
    "cs.LO": "Logic in Computer Science",
    "cs.LG": "Machine Learning",
    "cs.MS": "Mathematical Software",
    "cs.MA": "Multiagent Systems",
    "cs.MM": "Multimedia",
    "cs.NI": "Networking",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.NA": "Numerical Analysis",
    "cs.OS": "Operating Systems",
    "cs.OH": "Other Computer Science",
    "cs.PF": "Performance",
    "cs.PL": "Programming Languages",
    "cs.RO": "Robotics",
    "cs.SI": "Social and Information Networks",
    "cs.SE": "Software Engineering",
    "cs.SD": "Sound",
    "cs.SC": "Symbolic Computation",
    "cs.SY": "Systems and Control",
    # Mathematics
    "math.AG": "Algebraic Geometry",
    "math.AT": "Algebraic Topology",
    "math.AP": "Analysis of PDEs",
    "math.CT": "Category Theory",
    "math.CA": "Classical Analysis",
    "math.CO": "Combinatorics",
    "math.AC": "Commutative Algebra",
    "math.CV": "Complex Variables",
    "math.DG": "Differential Geometry",
    "math.DS": "Dynamical Systems",
    "math.FA": "Functional Analysis",
    "math.GM": "General Mathematics",
    "math.GN": "General Topology",
    "math.GT": "Geometric Topology",
    "math.GR": "Group Theory",
    "math.HO": "History and Overview",
    "math.IT": "Information Theory",
    "math.KT": "K-Theory",
    "math.LO": "Logic",
    "math.MP": "Mathematical Physics",
    "math.MG": "Metric Geometry",
    "math.NT": "Number Theory",
    "math.NA": "Numerical Analysis",
    "math.OA": "Operator Algebras",
    "math.OC": "Optimization",
    "math.PR": "Probability",
    "math.QA": "Quantum Algebra",
    "math.RT": "Representation Theory",
    "math.RA": "Rings and Algebras",
    "math.SP": "Spectral Theory",
    "math.ST": "Statistics Theory",
    "math.SG": "Symplectic Geometry",
    # Physics
    "physics.acc-ph": "Accelerator Physics",
    "physics.ao-ph": "Atmospheric Physics",
    "physics.atom-ph": "Atomic Physics",
    "physics.atm-clus": "Atomic Clusters",
    "physics.bio-ph": "Biological Physics",
    "physics.chem-ph": "Chemical Physics",
    "physics.class-ph": "Classical Physics",
    "physics.comp-ph": "Computational Physics",
    "physics.data-an": "Data Analysis",
    "physics.flu-dyn": "Fluid Dynamics",
    "physics.gen-ph": "General Physics",
    "physics.geo-ph": "Geophysics",
    "physics.hist-ph": "History of Physics",
    "physics.ins-det": "Instrumentation",
    "physics.med-ph": "Medical Physics",
    "physics.optics": "Optics",
    "physics.ed-ph": "Physics Education",
    "physics.soc-ph": "Physics and Society",
    "physics.plasm-ph": "Plasma Physics",
    "physics.pop-ph": "Popular Physics",
    "physics.space-ph": "Space Physics",
    # Statistics
    "stat.AP": "Applications",
    "stat.CO": "Computation",
    "stat.ML": "Machine Learning",
    "stat.ME": "Methodology",
    "stat.OT": "Other Statistics",
    "stat.TH": "Theory",
    # Quantitative Biology
    "q-bio.BM": "Biomolecules",
    "q-bio.CB": "Cell Behavior",
    "q-bio.GN": "Genomics",
    "q-bio.MN": "Molecular Networks",
    "q-bio.NC": "Neurons and Cognition",
    "q-bio.OT": "Other Quantitative Biology",
    "q-bio.PE": "Populations and Evolution",
    "q-bio.QM": "Quantitative Methods",
    "q-bio.SC": "Subcellular Processes",
    "q-bio.TO": "Tissues and Organs",
    # Quantitative Finance
    "q-fin.CP": "Computational Finance",
    "q-fin.EC": "Economics",
    "q-fin.GN": "General Finance",
    "q-fin.MF": "Mathematical Finance",
    "q-fin.PM": "Portfolio Management",
    "q-fin.PR": "Pricing",
    "q-fin.RM": "Risk Management",
    "q-fin.ST": "Statistical Finance",
    "q-fin.TR": "Trading and Microstructure",
    # Electrical Engineering
    "eess.AS": "Audio and Speech Processing",
    "eess.IV": "Image and Video Processing",
    "eess.SP": "Signal Processing",
    "eess.SY": "Systems and Control",
}


def get_category_triples() -> list[tuple[str, str, str]]:
    """Generate triples for arXiv category taxonomy."""
    triples = []

    for cat_id, cat_name in ARXIV_CATEGORIES.items():
        triples.append((cat_id, "has_name", cat_name))
        triples.append((cat_id, "is_a", "arxiv_category"))

        # Extract domain
        domain = cat_id.split(".")[0]
        triples.append((cat_id, "in_domain", domain))

        domain_names = {
            "cs": "Computer Science",
            "math": "Mathematics",
            "physics": "Physics",
            "stat": "Statistics",
            "q-bio": "Quantitative Biology",
            "q-fin": "Quantitative Finance",
            "eess": "Electrical Engineering",
        }

        if domain in domain_names:
            triples.append((domain, "has_name", domain_names[domain]))
            triples.append((domain, "is_a", "research_domain"))

    return triples


# =============================================================================
# SAMPLE DATA FOR OFFLINE TESTING
# =============================================================================

SAMPLE_PAPERS = [
    ArxivPaper(
        arxiv_id="1706.03762",
        title="Attention Is All You Need",
        authors=[
            "Ashish Vaswani",
            "Noam Shazeer",
            "Niki Parmar",
            "Jakob Uszkoreit",
            "Llion Jones",
            "Aidan N. Gomez",
            "Lukasz Kaiser",
            "Illia Polosukhin",
        ],
        abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
        categories=["cs.CL", "cs.LG"],
        primary_category="cs.CL",
        published="2017-06-12",
        updated="2017-12-06",
    ),
    ArxivPaper(
        arxiv_id="1810.04805",
        title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        authors=["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
        abstract="We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers.",
        categories=["cs.CL"],
        primary_category="cs.CL",
        published="2018-10-11",
        updated="2019-05-24",
    ),
    ArxivPaper(
        arxiv_id="2005.14165",
        title="Language Models are Few-Shot Learners",
        authors=["Tom B. Brown", "Benjamin Mann", "Nick Ryder", "Melanie Subbiah"],
        abstract="Recent work has demonstrated substantial gains on many NLP tasks from scaling up language models. We train GPT-3, a 175 billion parameter autoregressive language model.",
        categories=["cs.CL", "cs.LG"],
        primary_category="cs.CL",
        published="2020-05-28",
        updated="2020-07-22",
    ),
    ArxivPaper(
        arxiv_id="1409.0473",
        title="Neural Machine Translation by Jointly Learning to Align and Translate",
        authors=["Dzmitry Bahdanau", "Kyunghyun Cho", "Yoshua Bengio"],
        abstract="Neural machine translation attempts to build a single neural network that can be jointly tuned for translation. We conjecture that the use of a fixed-length vector is a bottleneck.",
        categories=["cs.CL", "cs.LG", "cs.NE", "stat.ML"],
        primary_category="cs.CL",
        published="2014-09-01",
        updated="2016-05-19",
    ),
    ArxivPaper(
        arxiv_id="1412.6980",
        title="Adam: A Method for Stochastic Optimization",
        authors=["Diederik P. Kingma", "Jimmy Ba"],
        abstract="We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions.",
        categories=["cs.LG"],
        primary_category="cs.LG",
        published="2014-12-22",
        updated="2017-01-30",
    ),
    ArxivPaper(
        arxiv_id="1312.6114",
        title="Auto-Encoding Variational Bayes",
        authors=["Diederik P. Kingma", "Max Welling"],
        abstract="We introduce a stochastic variational inference and learning algorithm that scales to large datasets.",
        categories=["stat.ML", "cs.LG"],
        primary_category="stat.ML",
        published="2013-12-20",
        updated="2022-12-11",
    ),
    ArxivPaper(
        arxiv_id="1406.2661",
        title="Generative Adversarial Networks",
        authors=[
            "Ian J. Goodfellow",
            "Jean Pouget-Abadie",
            "Mehdi Mirza",
            "Bing Xu",
            "David Warde-Farley",
            "Sherjil Ozair",
            "Aaron Courville",
            "Yoshua Bengio",
        ],
        abstract="We propose a new framework for estimating generative models via an adversarial process.",
        categories=["stat.ML", "cs.LG"],
        primary_category="stat.ML",
        published="2014-06-10",
        updated="2014-06-10",
    ),
    ArxivPaper(
        arxiv_id="1512.03385",
        title="Deep Residual Learning for Image Recognition",
        authors=["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren", "Jian Sun"],
        abstract="Deeper neural networks are more difficult to train. We present a residual learning framework.",
        categories=["cs.CV"],
        primary_category="cs.CV",
        published="2015-12-10",
        updated="2015-12-10",
    ),
    ArxivPaper(
        arxiv_id="1301.3781",
        title="Efficient Estimation of Word Representations in Vector Space",
        authors=["Tomas Mikolov", "Kai Chen", "Greg Corrado", "Jeffrey Dean"],
        abstract="We propose two novel model architectures for computing continuous vector representations of words from very large data sets.",
        categories=["cs.CL"],
        primary_category="cs.CL",
        published="2013-01-16",
        updated="2013-09-07",
    ),
    ArxivPaper(
        arxiv_id="1502.03167",
        title="Batch Normalization: Accelerating Deep Network Training",
        authors=["Sergey Ioffe", "Christian Szegedy"],
        abstract="Training Deep Neural Networks is complicated by the fact that the distribution of each layer's inputs changes during training.",
        categories=["cs.LG"],
        primary_category="cs.LG",
        published="2015-02-11",
        updated="2015-03-02",
    ),
]


def load_sample_papers() -> list[ArxivPaper]:
    """Return sample papers for offline testing."""
    return SAMPLE_PAPERS


def get_sample_triples() -> list[tuple[str, str, str]]:
    """Get all triples from sample papers."""
    triples = []

    # Category taxonomy
    triples.extend(get_category_triples())

    # Paper triples
    for paper in SAMPLE_PAPERS:
        triples.extend(paper.to_triples())

    return triples


# =============================================================================
# DORIAN INTEGRATION
# =============================================================================


def load_arxiv_to_dorian(
    core, agent_id: str, papers: list[ArxivPaper], show_progress: bool = True
) -> int:
    """
    Load arXiv papers into Dorian knowledge base.

    Args:
        core: DorianCore instance
        agent_id: Registered agent ID
        papers: List of ArxivPaper objects
        show_progress: Print progress

    Returns:
        Number of facts loaded
    """
    loaded = 0

    # Load category taxonomy first
    cat_triples = get_category_triples()
    for s, p, o in cat_triples:
        try:
            result = core.write(s, p, o, agent_id, confidence=1.0)
            if result.success:
                loaded += 1
        except Exception:
            pass

    if show_progress:
        print(f"  Loaded {loaded} category facts")

    # Load paper triples
    for i, paper in enumerate(papers):
        triples = paper.to_triples()
        for s, p, o in triples:
            try:
                result = core.write(s, p, o, agent_id, confidence=1.0)
                if result.success:
                    loaded += 1
            except Exception:
                pass

        if show_progress and (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(papers)} papers, {loaded} facts")

    if show_progress:
        print(f"  Complete: {loaded} facts from {len(papers)} papers")

    return loaded


def harvest_arxiv_category_to_dorian(
    core,
    agent_id: str,
    category: str,
    max_papers: int = 1000,
    show_progress: bool = True,
) -> int:
    """
    Harvest an arXiv category via API and load into Dorian.

    Requires network access.

    Args:
        core: DorianCore instance
        agent_id: Registered agent ID
        category: arXiv category (e.g., "cs.AI", "cs.LG")
        max_papers: Maximum papers to fetch
        show_progress: Print progress

    Returns:
        Number of facts loaded
    """
    if show_progress:
        print(f"Harvesting arXiv category: {category}")

    papers = []
    batch_size = 100

    for start in range(0, max_papers, batch_size):
        query = f"cat:{category}"
        batch = arxiv_api_search(query, start=start, max_results=batch_size)

        if not batch:
            break

        papers.extend(batch)

        if show_progress:
            print(f"  Fetched {len(papers)} papers...")

        # Rate limiting
        time.sleep(3)

        if len(batch) < batch_size:
            break

    if show_progress:
        print(f"  Total papers: {len(papers)}")

    return load_arxiv_to_dorian(core, agent_id, papers, show_progress)


# =============================================================================
# DEMO / TESTING
# =============================================================================


def demo_sample_data():
    """Demo with sample papers (no network)."""
    print("=" * 70)
    print("ARXIV SAMPLE DATA DEMO")
    print("=" * 70)

    papers = load_sample_papers()
    print(f"\nSample papers: {len(papers)}")

    for paper in papers[:5]:
        print(f"\n  [{paper.arxiv_id}] {paper.title[:60]}...")
        print(f"    Authors: {', '.join(paper.authors[:3])}")
        print(f"    Categories: {', '.join(paper.categories)}")

    print("\n" + "-" * 70)
    print("Extracted triples:")

    triples = get_sample_triples()
    print(f"\nTotal triples: {len(triples)}")

    # Show some triples
    shown = set()
    for s, p, o in triples[:30]:
        if p not in shown:
            print(f"  ({s[:30]}, {p}, {o[:30]})")
            shown.add(p)


def demo_api():
    """Demo arXiv API (requires network)."""
    print("=" * 70)
    print("ARXIV API DEMO")
    print("=" * 70)

    print("\nSearching: 'ti:transformer attention'")

    try:
        papers = arxiv_api_search("ti:transformer attention", max_results=5)

        for paper in papers:
            print(f"\n  [{paper.arxiv_id}] {paper.title[:60]}...")
            print(f"    Authors: {', '.join(paper.authors[:3])}")
            print(f"    Published: {paper.published[:10]}")
            print(f"    Categories: {', '.join(paper.categories)}")

    except Exception as e:
        print(f"\nAPI demo skipped (network error): {e}")


if __name__ == "__main__":
    print("DORIAN ARXIV LOADER")
    print("=" * 70)

    demo_sample_data()
    print()

    try:
        demo_api()
    except Exception:
        print("\nAPI demo skipped (no network)")
