"""
Visual AI Repair Diagnosis Engine

Uses VSA primitives (P-Sup, CW-Bundle, T-Bind) for intelligent repair diagnosis.

Key Capabilities:
1. Multi-hypothesis diagnosis using P-Sup (probabilistic superposition)
2. Bayesian updates as evidence accumulates
3. Store context via T-Bind temporal memory
4. Learning from corrections via CW-Bundle

Usage:
    engine = RepairDiagnosisEngine()
    result = engine.diagnose(
        text="my faucet is dripping",
        image_features=vision_features,
        store_id="store-123"
    )
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import time
from dataclasses import dataclass
from typing import Any

import torch
from vsa_core.operators import (
    bind,
    bundle,
    cw_bundle,
    t_bind,
    t_unbind,
    weighted_bundle,
)
from vsa_core.probabilistic import (
    HypothesisBundle,
    p_sup,
    p_sup_collapse,
    p_sup_update,
)

# VSA imports
from vsa_core.vectors import (
    batch_similarity,
    random_vector,
    similarity,
)

from .repair_models import (
    DiagnoseResponse,
    Hypothesis,
    KnowledgeBaseState,
    ProblemStatus,
    VSAHypothesisState,
    expertise_multiplier,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RepairEngineConfig:
    """Configuration for the repair diagnosis engine."""

    # VSA dimensions
    dimensions: int = 4096

    # P-Sup settings
    collapse_threshold: float = 0.85
    min_hypothesis_prob: float = 0.05
    max_hypotheses: int = 5
    update_temperature: float = 0.8

    # T-Bind settings (store memory)
    temporal_decay_rate: float = 0.1  # Decay over days
    temporal_max_shift: int = 1000

    # Confidence thresholds
    high_confidence_threshold: float = 0.75
    needs_more_info_threshold: float = 0.5

    # Device
    device: str = "cpu"


# =============================================================================
# CATEGORY CODEBOOK
# =============================================================================

class CategoryCodebook:
    """
    VSA codebook for problem categories.

    Maintains deterministic vectors for each category based on slug hash.
    """

    def __init__(self, config: RepairEngineConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.dimensions = config.dimensions

        # Category vectors (lazy loaded)
        self._vectors: dict[str, torch.Tensor] = {}
        self._category_info: dict[str, dict[str, Any]] = {}

    def _generate_vector(self, seed: str) -> torch.Tensor:
        """Generate deterministic vector from seed string."""
        # Use hash for deterministic PRNG seed
        hash_bytes = hashlib.sha256(seed.encode()).digest()
        seed_int = int.from_bytes(hash_bytes[:8], "big")

        # Generate random phasor vector
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed_int)

        phases = torch.rand(self.dimensions, generator=gen, device=self.device) * 2 * torch.pi
        return torch.exp(1j * phases).to(torch.complex64)

    def register_category(
        self,
        slug: str,
        name: str,
        description: str | None = None,
        icon: str | None = None,
        parent_slug: str | None = None
    ) -> torch.Tensor:
        """Register a category and return its vector."""
        if slug not in self._vectors:
            self._vectors[slug] = self._generate_vector(f"category:{slug}")
            self._category_info[slug] = {
                "name": name,
                "description": description,
                "icon": icon,
                "parent_slug": parent_slug,
            }
        return self._vectors[slug]

    def get_vector(self, slug: str) -> torch.Tensor:
        """Get vector for a category."""
        if slug not in self._vectors:
            raise ValueError(f"Category '{slug}' not registered")
        return self._vectors[slug]

    def get_info(self, slug: str) -> dict[str, Any]:
        """Get category info."""
        return self._category_info.get(slug, {})

    def get_all_vectors(self) -> torch.Tensor:
        """Get all category vectors as matrix."""
        slugs = list(self._vectors.keys())
        vectors = [self._vectors[s] for s in slugs]
        return torch.stack(vectors)

    def get_all_slugs(self) -> list[str]:
        """Get all registered category slugs."""
        return list(self._vectors.keys())

    def find_nearest(
        self,
        query: torch.Tensor,
        top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Find nearest categories to query vector."""
        all_vectors = self.get_all_vectors()
        slugs = self.get_all_slugs()

        sims = batch_similarity(query, all_vectors)
        sims_real = torch.real(sims) if sims.is_complex() else sims

        # Get top-k
        k = min(top_k, len(slugs))
        values, indices = torch.topk(sims_real, k)

        return [
            (slugs[int(idx)], float(values[i]))
            for i, idx in enumerate(indices)
        ]


# =============================================================================
# TEXT ENCODER
# =============================================================================

class TextEncoder:
    """
    Encode text descriptions into VSA vectors.

    Uses bag-of-words with pre-computed word vectors.
    """

    def __init__(self, config: RepairEngineConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.dimensions = config.dimensions

        # Word vectors (lazy loaded)
        self._word_vectors: dict[str, torch.Tensor] = {}

        # Common repair keywords with boosted weights
        self._keyword_weights = {
            # Plumbing
            "leak": 2.0, "drip": 2.0, "faucet": 2.0, "pipe": 2.0,
            "drain": 2.0, "clog": 2.0, "toilet": 2.0, "water": 1.5,
            "pressure": 1.5, "hot": 1.5, "cold": 1.5,

            # Electrical
            "outlet": 2.0, "switch": 2.0, "light": 2.0, "wire": 2.0,
            "circuit": 2.0, "breaker": 2.0, "spark": 2.0, "power": 1.5,
            "electric": 1.5, "voltage": 1.5,

            # General repair
            "broken": 1.5, "fix": 1.5, "repair": 1.5, "replace": 1.5,
            "install": 1.5, "damage": 1.5, "crack": 1.5, "loose": 1.5,
        }

    def _get_word_vector(self, word: str) -> torch.Tensor:
        """Get or generate vector for a word."""
        word_lower = word.lower()
        if word_lower not in self._word_vectors:
            # Generate deterministic vector from word
            hash_bytes = hashlib.sha256(f"word:{word_lower}".encode()).digest()
            seed_int = int.from_bytes(hash_bytes[:8], "big")

            gen = torch.Generator(device=self.device)
            gen.manual_seed(seed_int)

            phases = torch.rand(self.dimensions, generator=gen, device=self.device) * 2 * torch.pi
            self._word_vectors[word_lower] = torch.exp(1j * phases).to(torch.complex64)

        return self._word_vectors[word_lower]

    def encode(self, text: str) -> torch.Tensor:
        """Encode text into VSA vector."""
        if not text or not text.strip():
            # Return random vector for empty text
            return random_vector(self.dimensions, device=self.device)

        # Tokenize (simple whitespace + punctuation split)
        import re
        words = re.findall(r'\b\w+\b', text.lower())

        if not words:
            return random_vector(self.dimensions, device=self.device)

        # Get vectors and weights
        vectors = []
        weights = []
        for word in words:
            vec = self._get_word_vector(word)
            weight = self._keyword_weights.get(word, 1.0)
            vectors.append(vec)
            weights.append(weight)

        # Weighted bundle
        return weighted_bundle(vectors, weights)


# =============================================================================
# REPAIR DIAGNOSIS ENGINE
# =============================================================================

class RepairDiagnosisEngine:
    """
    Main engine for repair problem diagnosis.

    Uses VSA primitives for:
    - P-Sup: Multi-hypothesis tracking
    - CW-Bundle: Confidence-weighted evidence
    - T-Bind: Store temporal context
    """

    def __init__(self, config: RepairEngineConfig | None = None):
        self.config = config or RepairEngineConfig()
        self.device = torch.device(self.config.device)

        # Components
        self.codebook = CategoryCodebook(self.config)
        self.text_encoder = TextEncoder(self.config)

        # Store memories (T-Bind)
        self._store_memories: dict[str, torch.Tensor] = {}
        self._store_reference_times: dict[str, float] = {}

        # Initialize default categories
        self._init_default_categories()

    def _init_default_categories(self):
        """Initialize default problem categories."""
        categories = [
            ("plumbing", "Plumbing", "Pipes, faucets, toilets, drains", "plumbing"),
            ("plumbing-faucet", "Leaky Faucet", "Dripping or running faucets", "plumbing"),
            ("plumbing-drain", "Clogged Drain", "Slow or blocked drains", "plumbing"),
            ("plumbing-toilet", "Running Toilet", "Toilet won't stop running", "plumbing"),
            ("plumbing-pipe", "Pipe Leak", "Leaking pipes under sink or wall", "plumbing"),
            ("plumbing-waterheater", "Water Heater", "Hot water issues", "plumbing"),
            ("electrical", "Electrical", "Wiring, outlets, switches, lights", "electrical"),
            ("electrical-outlet", "Outlet Issue", "Dead or sparking outlets", "electrical"),
            ("electrical-switch", "Switch Problem", "Light switch not working", "electrical"),
            ("electrical-lighting", "Lighting Issue", "Flickering or dead lights", "electrical"),
            ("hvac", "HVAC", "Heating, cooling, ventilation", "hvac"),
            ("carpentry", "Carpentry", "Wood, framing, trim, doors", "carpentry"),
            ("painting", "Painting", "Interior, exterior, staining", "painting"),
            ("flooring", "Flooring", "Tile, hardwood, carpet, vinyl", "flooring"),
            ("roofing", "Roofing", "Shingles, gutters, flashing", "roofing"),
            ("appliances", "Appliances", "Major and small appliances", "appliances"),
            ("outdoor", "Outdoor", "Lawn, garden, landscaping", "outdoor"),
            ("automotive", "Automotive", "Car maintenance and repair", "automotive"),
        ]

        for slug, name, desc, icon in categories:
            parent = slug.split("-")[0] if "-" in slug else None
            self.codebook.register_category(
                slug=slug,
                name=name,
                description=desc,
                icon=icon,
                parent_slug=parent if parent != slug else None
            )

    def diagnose(
        self,
        text: str | None = None,
        voice_transcript: str | None = None,
        image_features: torch.Tensor | None = None,
        store_id: str | None = None,
        employee_id: str | None = None,
    ) -> tuple[HypothesisBundle, dict[str, Any]]:
        """
        Diagnose a repair problem from multimodal input.

        Args:
            text: Text description of problem
            voice_transcript: Transcript from voice input
            image_features: VSA vector from image analysis
            store_id: Store identifier for context
            employee_id: Employee ID (for context)

        Returns:
            (HypothesisBundle, metadata_dict)
        """
        start_time = time.time()

        # Combine text inputs
        combined_text = " ".join(filter(None, [text, voice_transcript]))

        # Encode inputs
        input_vectors = []

        if combined_text:
            text_vec = self.text_encoder.encode(combined_text)
            input_vectors.append(("text", text_vec, 1.0))

        if image_features is not None:
            input_vectors.append(("image", image_features, 1.2))  # Slightly higher weight

        if not input_vectors:
            raise ValueError("At least one input (text, voice, or image) required")

        # Bundle inputs with confidence
        vectors = [v for _, v, _ in input_vectors]
        weights = [w for _, _, w in input_vectors]
        query_vec = weighted_bundle(vectors, weights)

        # Add store context if available
        store_context_similarity = None
        if store_id and store_id in self._store_memories:
            store_memory = self._store_memories[store_id]
            ref_time = self._store_reference_times.get(store_id, time.time())

            # Query recent context
            recent_context = t_unbind(
                store_memory,
                time.time(),
                ref_time,
                decay_rate=self.config.temporal_decay_rate,
                max_shift=self.config.temporal_max_shift
            )

            # Blend with query (light influence)
            query_vec = weighted_bundle([query_vec, recent_context], [0.9, 0.1])
            sim_val = similarity(query_vec, recent_context)
            if torch.is_tensor(sim_val):
                sim_val = torch.real(sim_val) if sim_val.is_complex() else sim_val
            store_context_similarity = float(sim_val)

        # Find nearest categories
        matches = self.codebook.find_nearest(query_vec, top_k=self.config.max_hypotheses)

        # Build hypotheses with normalized probabilities
        hypotheses = []
        total_sim = sum(max(0, sim) for _, sim in matches)

        for slug, sim in matches:
            if total_sim > 0:
                prob = max(0, sim) / total_sim
            else:
                prob = 1.0 / len(matches)

            # Apply minimum probability threshold
            if prob >= self.config.min_hypothesis_prob:
                cat_vec = self.codebook.get_vector(slug)
                hypotheses.append((slug, cat_vec, prob))

        # Create P-Sup bundle
        if not hypotheses:
            # Fallback: use all matches with equal probability
            for slug, _ in matches:
                cat_vec = self.codebook.get_vector(slug)
                hypotheses.append((slug, cat_vec, 1.0 / len(matches)))

        bundle = p_sup(hypotheses, normalize_probs=True)

        # Compute metadata
        elapsed_ms = (time.time() - start_time) * 1000

        metadata = {
            "elapsed_ms": elapsed_ms,
            "input_types": [name for name, _, _ in input_vectors],
            "query_norm": float(torch.norm(query_vec)),
            "store_context_similarity": store_context_similarity,
            "entropy": bundle.entropy(),
        }

        # Update store memory with this problem
        if store_id:
            self._update_store_memory(store_id, query_vec)

        return bundle, metadata

    def _update_store_memory(self, store_id: str, problem_vec: torch.Tensor):
        """Update store memory with T-Bind encoded problem."""
        now = time.time()

        # Initialize store memory if needed
        if store_id not in self._store_memories:
            self._store_memories[store_id] = torch.zeros(
                self.config.dimensions,
                dtype=torch.complex64,
                device=self.device
            )
            self._store_reference_times[store_id] = now

        ref_time = self._store_reference_times[store_id]

        # T-Bind encode the problem
        t_encoded = t_bind(
            problem_vec,
            now,
            ref_time,
            decay_rate=self.config.temporal_decay_rate,
            max_shift=self.config.temporal_max_shift
        )

        # Bundle into memory
        self._store_memories[store_id] = bundle(
            self._store_memories[store_id],
            t_encoded
        )

    def refine_diagnosis(
        self,
        bundle: HypothesisBundle,
        evidence: torch.Tensor,
    ) -> HypothesisBundle:
        """
        Refine diagnosis with additional evidence.

        Args:
            bundle: Current hypothesis bundle
            evidence: New evidence vector

        Returns:
            Updated HypothesisBundle
        """
        return p_sup_update(
            bundle,
            evidence,
            temperature=self.config.update_temperature
        )

    def check_collapse(
        self,
        bundle: HypothesisBundle
    ) -> str | None:
        """
        Check if hypothesis should collapse to single answer.

        Returns category slug if confident enough, None otherwise.
        """
        return p_sup_collapse(bundle, threshold=self.config.collapse_threshold)

    def bundle_to_response(
        self,
        bundle: HypothesisBundle,
        problem_id: str,
        metadata: dict[str, Any]
    ) -> DiagnoseResponse:
        """Convert HypothesisBundle to API response."""
        hypotheses = []
        for i, (label, prob) in enumerate(
            zip(bundle.hypotheses, bundle.probabilities.tolist())
        ):
            info = self.codebook.get_info(label)
            hypotheses.append(Hypothesis(
                category_slug=label,
                category_name=info.get("name", label),
                probability=prob,
                explanation=info.get("description"),
                icon=info.get("icon"),
            ))

        # Sort by probability
        hypotheses.sort(key=lambda h: h.probability, reverse=True)
        top = hypotheses[0]

        # Determine if more info needed
        needs_more_info = top.probability < self.config.needs_more_info_threshold

        # Generate follow-up questions based on top hypotheses
        follow_up_questions = []
        if needs_more_info and len(hypotheses) >= 2:
            # Questions to disambiguate top 2
            h1, h2 = hypotheses[0], hypotheses[1]
            follow_up_questions = self._generate_follow_up_questions(h1, h2)

        return DiagnoseResponse(
            problem_id=problem_id,
            status=ProblemStatus.DIAGNOSED,
            hypotheses=hypotheses,
            top_hypothesis=top,
            confidence=top.probability,
            entropy=metadata.get("entropy", 0.0),
            needs_more_info=needs_more_info,
            follow_up_questions=follow_up_questions,
            similar_recent_problems=None,  # TODO: implement
        )

    def _generate_follow_up_questions(
        self,
        h1: Hypothesis,
        h2: Hypothesis
    ) -> list[str]:
        """Generate questions to disambiguate between top hypotheses."""
        questions = []

        # Category-specific questions
        if "plumbing" in h1.category_slug or "plumbing" in h2.category_slug:
            questions.append("Is there visible water damage or wetness?")
            questions.append("Can you hear water running when nothing is on?")

        if "electrical" in h1.category_slug or "electrical" in h2.category_slug:
            questions.append("Have you noticed any flickering lights or sparks?")
            questions.append("Is this affecting multiple rooms or just one area?")

        if "faucet" in h1.category_slug or "faucet" in h2.category_slug:
            questions.append("Is the drip from the spout or the base of the faucet?")
            questions.append("Does it drip constantly or only sometimes?")

        if "drain" in h1.category_slug or "drain" in h2.category_slug:
            questions.append("Is the water draining slowly or not at all?")
            questions.append("Is there a bad smell coming from the drain?")

        # Generic disambiguation
        if not questions:
            questions.append("Can you describe what you see in more detail?")
            questions.append("When did you first notice this problem?")

        return questions[:3]  # Return max 3 questions

    def incorporate_correction(
        self,
        problem_vec: torch.Tensor,
        original_category: str,
        corrected_category: str,
        employee_level: int,
        knowledge_state: KnowledgeBaseState | None = None
    ) -> KnowledgeBaseState:
        """
        Incorporate employee correction into knowledge base.

        Uses CW-Bundle to weight corrections by employee expertise.

        Args:
            problem_vec: The problem's VSA vector
            original_category: What the AI thought
            corrected_category: What the employee corrected to
            employee_level: Employee's expertise level (1-10)
            knowledge_state: Existing knowledge state (or None)

        Returns:
            Updated KnowledgeBaseState
        """
        # Get correction vector (bound problem + category)
        correct_cat_vec = self.codebook.get_vector(corrected_category)
        correction_vec = bind(problem_vec, correct_cat_vec)

        # Calculate expertise weight
        weight = expertise_multiplier(employee_level)

        if knowledge_state is None:
            # Initialize new knowledge state
            return KnowledgeBaseState(
                category_slug=corrected_category,
                knowledge_vector=self._tensor_to_base64(correction_vec),
                aggregate_confidence=weight / 2.5,  # Normalize to ~0-1 range
                total_corrections=1,
                correction_weights=[weight],
                dimensions=self.config.dimensions,
            )

        # Decode existing knowledge
        existing_vec = self._base64_to_tensor(knowledge_state.knowledge_vector)

        # CW-Bundle: combine existing knowledge with new correction
        combined, conf = cw_bundle(
            [existing_vec, correction_vec],
            [knowledge_state.aggregate_confidence, weight],
            temperature=0.8
        )

        # Update state
        return KnowledgeBaseState(
            category_slug=corrected_category,
            knowledge_vector=self._tensor_to_base64(combined),
            aggregate_confidence=float(conf.mean()),
            total_corrections=knowledge_state.total_corrections + 1,
            correction_weights=knowledge_state.correction_weights + [weight],
            dimensions=self.config.dimensions,
        )

    def _tensor_to_base64(self, tensor: torch.Tensor) -> str:
        """Serialize tensor to base64 string."""
        buffer = io.BytesIO()
        torch.save(tensor.cpu(), buffer)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _base64_to_tensor(self, b64: str) -> torch.Tensor:
        """Deserialize tensor from base64 string."""
        buffer = io.BytesIO(base64.b64decode(b64))
        return torch.load(buffer, map_location=self.device)

    def serialize_hypothesis_state(
        self,
        bundle: HypothesisBundle
    ) -> VSAHypothesisState:
        """Serialize HypothesisBundle for database storage."""
        return VSAHypothesisState(
            hypotheses=bundle.hypotheses,
            probabilities=bundle.probabilities.tolist(),
            superposition_vector=self._tensor_to_base64(bundle.vector),
            basis_vectors=self._tensor_to_base64(bundle.basis_vectors),
            dimensions=self.config.dimensions,
            update_count=0,
        )

    def deserialize_hypothesis_state(
        self,
        state: VSAHypothesisState
    ) -> HypothesisBundle:
        """Deserialize HypothesisBundle from database storage."""
        return HypothesisBundle(
            vector=self._base64_to_tensor(state.superposition_vector),
            hypotheses=state.hypotheses,
            probabilities=torch.tensor(state.probabilities),
            basis_vectors=self._base64_to_tensor(state.basis_vectors),
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_engine(
    dimensions: int = 4096,
    device: str = "cpu"
) -> RepairDiagnosisEngine:
    """Create a repair diagnosis engine with specified config."""
    config = RepairEngineConfig(
        dimensions=dimensions,
        device=device,
    )
    return RepairDiagnosisEngine(config)
