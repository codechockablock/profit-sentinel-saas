"""
engine/bridge.py - VSA-Symbolic Bridge

Connects VSA subsymbolic inference with symbolic reasoning.

The bridge translates:
- VSA resonator results → Symbolic facts
- Symbolic proofs → Confidence scores
- Rules → VSA query patterns

This enables:
1. VSA detects anomaly patterns (fast, fuzzy)
2. Symbolic reasoning explains root causes (precise, explainable)
3. Combined confidence from both systems
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from reasoning.inference import ProofTree, backward_chain, forward_chain
from reasoning.knowledge_base import KnowledgeBase
from reasoning.terms import Atom, Term, Var

logger = logging.getLogger(__name__)


@dataclass
class BridgeResult:
    """Combined result from VSA + symbolic reasoning."""
    entity_id: str
    vsa_anomalies: list[tuple[str, float]]  # (anomaly, similarity)
    symbolic_conclusions: list[tuple[str, float]]  # (conclusion, confidence)
    root_causes: list[dict[str, Any]]
    recommended_actions: list[str]
    proof_tree: ProofTree | None = None
    combined_confidence: float = 0.0
    explanation: str = ""


class VSASymbolicBridge:
    """Bridge between VSA and symbolic reasoning systems.

    Example:
        bridge = VSASymbolicBridge()
        bridge.set_resonator(resonator)
        bridge.set_knowledge_base(kb)
        bridge.set_primitive_loader(loader)

        # Analyze entity
        result = bridge.analyze("SKU123", entity_vector)
        print(result.explanation)
        print(result.root_causes)
    """

    def __init__(
        self,
        vsa_weight: float = 0.6,
        symbolic_weight: float = 0.4
    ):
        """Initialize bridge.

        Args:
            vsa_weight: Weight for VSA confidence in combined score
            symbolic_weight: Weight for symbolic confidence
        """
        self.vsa_weight = vsa_weight
        self.symbolic_weight = symbolic_weight

        self._resonator = None
        self._kb: KnowledgeBase | None = None
        self._primitive_loader = None
        self._rule_mappings: dict[str, str] = {}  # VSA primitive → symbolic predicate

    def set_resonator(self, resonator) -> None:
        """Set VSA resonator."""
        self._resonator = resonator

    def set_knowledge_base(self, kb: KnowledgeBase) -> None:
        """Set symbolic knowledge base."""
        self._kb = kb

    def set_primitive_loader(self, loader) -> None:
        """Set primitive loader."""
        self._primitive_loader = loader

    def register_mapping(self, vsa_primitive: str, symbolic_predicate: str) -> None:
        """Register VSA primitive to symbolic predicate mapping.

        Args:
            vsa_primitive: VSA primitive path (e.g., "inventory.low_stock")
            symbolic_predicate: Symbolic predicate name (e.g., "has_anomaly")
        """
        self._rule_mappings[vsa_primitive] = symbolic_predicate

    def vsa_to_facts(
        self,
        entity_id: str,
        resonator_result,
        threshold: float = 0.4
    ) -> list[Term]:
        """Convert VSA resonator results to symbolic facts.

        Args:
            entity_id: Entity identifier
            resonator_result: Result from resonator
            threshold: Minimum similarity for fact creation

        Returns:
            List of symbolic facts
        """
        facts = []

        for anomaly_name, similarity in resonator_result.top_matches:
            if similarity < threshold:
                continue

            # Create fact based on mapping
            predicate = self._rule_mappings.get(anomaly_name, "detected")

            # has_anomaly(entity_id, anomaly_type)
            fact = Term(predicate, Atom(entity_id), Atom(anomaly_name))
            facts.append(fact)

            # Also add confidence fact
            conf_fact = Term("confidence", Atom(entity_id), Atom(anomaly_name), Atom(similarity))
            facts.append(conf_fact)

        return facts

    def analyze(
        self,
        entity_id: str,
        entity_vector: torch.Tensor,
        context: dict[str, Any] | None = None
    ) -> BridgeResult:
        """Perform combined VSA + symbolic analysis.

        Args:
            entity_id: Entity identifier
            entity_vector: Entity's VSA vector
            context: Additional context data

        Returns:
            Combined analysis result
        """
        context = context or {}

        # Step 1: VSA resonance
        vsa_anomalies = []
        if self._resonator:
            res_result = self._resonator.resonate(entity_vector)
            vsa_anomalies = res_result.top_matches

        # Step 2: Convert to symbolic facts
        facts = self.vsa_to_facts(entity_id, res_result)

        # Add context facts
        for key, value in context.items():
            if isinstance(value, (int, float, str)):
                facts.append(Term(key, Atom(entity_id), Atom(value)))

        # Step 3: Assert facts to KB (temporarily)
        if self._kb:
            for fact in facts:
                self._kb.add_fact(fact, source="vsa_bridge")

            # Step 4: Run forward chaining
            derived, _ = forward_chain(self._kb, max_iterations=50)

            # Step 5: Query for conclusions
            symbolic_conclusions = []

            # Query for alerts
            for theta in self._kb.query(Term("alert", Atom(entity_id), Var("Type"))):
                alert_type = theta.get("Type")
                if isinstance(alert_type, Atom):
                    symbolic_conclusions.append((str(alert_type.value), 1.0))

            # Query for root causes
            root_causes = []
            for theta in self._kb.query(Term("root_cause", Atom(entity_id), Var("Cause"))):
                cause = theta.get("Cause")
                if isinstance(cause, Atom):
                    root_causes.append({
                        "code": str(cause.value),
                        "confidence": 0.8,  # Could be from proof
                    })

            # Query for actions
            recommended_actions = []
            for theta in self._kb.query(Term("recommended_action", Atom(entity_id), Var("Action"))):
                action = theta.get("Action")
                if isinstance(action, Atom):
                    recommended_actions.append(str(action.value))

            # Get proof tree for main anomaly
            proof_tree = None
            if vsa_anomalies:
                main_anomaly = vsa_anomalies[0][0]
                goal = Term("has_anomaly", Atom(entity_id), Atom(main_anomaly))
                proof_tree = backward_chain(self._kb, goal)

            # Retract temporary facts
            for fact in facts:
                self._kb.retract_fact(fact)

        else:
            symbolic_conclusions = []
            root_causes = []
            recommended_actions = []
            proof_tree = None

        # Step 6: Compute combined confidence
        vsa_conf = vsa_anomalies[0][1] if vsa_anomalies else 0.0
        sym_conf = 1.0 if symbolic_conclusions else 0.0
        combined = self.vsa_weight * vsa_conf + self.symbolic_weight * sym_conf

        # Step 7: Generate explanation
        explanation = self._generate_explanation(
            entity_id, vsa_anomalies, symbolic_conclusions, root_causes, proof_tree
        )

        return BridgeResult(
            entity_id=entity_id,
            vsa_anomalies=vsa_anomalies,
            symbolic_conclusions=symbolic_conclusions,
            root_causes=root_causes,
            recommended_actions=recommended_actions,
            proof_tree=proof_tree,
            combined_confidence=combined,
            explanation=explanation
        )

    def _generate_explanation(
        self,
        entity_id: str,
        vsa_anomalies: list[tuple[str, float]],
        symbolic_conclusions: list[tuple[str, float]],
        root_causes: list[dict[str, Any]],
        proof_tree: ProofTree | None
    ) -> str:
        """Generate human-readable explanation."""
        lines = [f"Analysis for {entity_id}:"]

        if vsa_anomalies:
            lines.append("\nDetected Patterns (VSA):")
            for anomaly, conf in vsa_anomalies[:5]:
                lines.append(f"  • {anomaly}: {conf:.1%} confidence")

        if symbolic_conclusions:
            lines.append("\nSymbolic Conclusions:")
            for conclusion, conf in symbolic_conclusions:
                lines.append(f"  • {conclusion}")

        if root_causes:
            lines.append("\nProbable Root Causes:")
            for cause in root_causes[:3]:
                lines.append(f"  • {cause['code']}: {cause.get('confidence', 0):.1%}")

        if proof_tree and proof_tree.is_valid:
            lines.append("\nProof:")
            lines.append(proof_tree.explain())

        return "\n".join(lines)

    def batch_analyze(
        self,
        entities: dict[str, torch.Tensor],
        contexts: dict[str, dict] | None = None
    ) -> list[BridgeResult]:
        """Analyze multiple entities.

        Args:
            entities: Dict mapping entity_id to vector
            contexts: Optional contexts per entity

        Returns:
            List of results
        """
        contexts = contexts or {}
        results = []

        for entity_id, vector in entities.items():
            context = contexts.get(entity_id, {})
            result = self.analyze(entity_id, vector, context)
            results.append(result)

        return results


class PlaybookGenerator:
    """Generate investigation playbooks from analysis results.

    Creates structured, actionable playbooks for detected anomalies.
    """

    def __init__(self):
        self._templates: dict[str, dict[str, Any]] = {}

    def register_template(
        self,
        anomaly_type: str,
        template: dict[str, Any]
    ) -> None:
        """Register playbook template for anomaly type.

        Args:
            anomaly_type: Type of anomaly
            template: Template with steps, owners, etc.
        """
        self._templates[anomaly_type] = template

    def generate(self, result: BridgeResult) -> dict[str, Any]:
        """Generate playbook from bridge result.

        Args:
            result: Analysis result

        Returns:
            Playbook dictionary
        """
        playbook = {
            "entity_id": result.entity_id,
            "generated_at": None,  # Would be timestamp
            "severity": self._determine_severity(result),
            "summary": self._generate_summary(result),
            "investigation_steps": [],
            "recommended_actions": [],
            "escalation_path": [],
            "documentation_links": [],
        }

        # Add steps from templates
        for anomaly, _ in result.vsa_anomalies:
            if anomaly in self._templates:
                template = self._templates[anomaly]
                playbook["investigation_steps"].extend(
                    template.get("steps", [])
                )
                playbook["documentation_links"].extend(
                    template.get("docs", [])
                )

        # Add recommended actions
        playbook["recommended_actions"] = result.recommended_actions

        # Add root cause specific steps
        for cause in result.root_causes:
            code = cause.get("code", "")
            if code in self._templates:
                template = self._templates[code]
                playbook["investigation_steps"].extend(
                    template.get("steps", [])
                )

        # Deduplicate
        playbook["investigation_steps"] = list(dict.fromkeys(
            playbook["investigation_steps"]
        ))
        playbook["documentation_links"] = list(dict.fromkeys(
            playbook["documentation_links"]
        ))

        return playbook

    def _determine_severity(self, result: BridgeResult) -> str:
        """Determine severity from result."""
        if result.combined_confidence > 0.8:
            return "critical"
        elif result.combined_confidence > 0.6:
            return "high"
        elif result.combined_confidence > 0.4:
            return "medium"
        else:
            return "low"

    def _generate_summary(self, result: BridgeResult) -> str:
        """Generate summary paragraph."""
        if not result.vsa_anomalies:
            return f"No anomalies detected for {result.entity_id}."

        main_anomaly = result.vsa_anomalies[0][0]
        confidence = result.combined_confidence

        summary = f"Entity {result.entity_id} shows {main_anomaly} pattern "
        summary += f"with {confidence:.0%} confidence. "

        if result.root_causes:
            summary += f"Probable cause: {result.root_causes[0]['code']}."

        return summary

    def to_markdown(self, playbook: dict[str, Any]) -> str:
        """Export playbook to Markdown."""
        lines = [
            f"# Investigation Playbook: {playbook['entity_id']}",
            "",
            f"**Severity:** {playbook['severity'].upper()}",
            "",
            "## Summary",
            playbook['summary'],
            "",
            "## Investigation Steps",
        ]

        for i, step in enumerate(playbook['investigation_steps'], 1):
            lines.append(f"{i}. {step}")

        lines.extend([
            "",
            "## Recommended Actions",
        ])

        for action in playbook['recommended_actions']:
            lines.append(f"- {action}")

        if playbook['documentation_links']:
            lines.extend([
                "",
                "## Documentation",
            ])
            for link in playbook['documentation_links']:
                lines.append(f"- {link}")

        return "\n".join(lines)

    def to_json(self, playbook: dict[str, Any]) -> str:
        """Export playbook to JSON."""
        import json
        return json.dumps(playbook, indent=2)
