"""
DORIAN AGENT V2 - SMARTER
=========================

Improvements:
1. Pronoun resolution (me/you/I → actual names)
2. Better question detection and handling
3. Semantic entity search (not just exact match)
4. Longer fact storage
5. Context tracking across conversation
6. Actually answers questions from the mirror
7. Self-awareness (knows what it is)

Still no LLM for parsing - just smarter patterns and logic.

Author: Joseph + Claude
Date: 2026-01-25
"""

import json
import pickle
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from dorian_v7 import DorianV7

# =============================================================================
# PART 1: SMARTER PERCEPTION
# =============================================================================


class SmartPerception:
    """
    Extract facts with pronoun resolution and better parsing.
    """

    def __init__(self, agent_name: str = "Dorian"):
        self.agent_name = agent_name.lower()
        self.speaker_name: str | None = None

        # Pronoun mappings (resolved at extraction time)
        self.first_person = ["i", "me", "my", "mine", "myself"]
        self.second_person = ["you", "your", "yours", "yourself"]

        # Question words
        self.question_words = [
            "what",
            "who",
            "where",
            "when",
            "why",
            "how",
            "which",
            "do",
            "does",
            "is",
            "are",
            "can",
            "could",
            "would",
            "tell",
        ]

        # Relation mappings for normalization
        self.relation_map = {
            "am": "is",
            "are": "is",
            "was": "is",
            "were": "is",
            "'m": "is",
            "'re": "is",
            "like": "likes",
            "love": "loves",
            "hate": "hates",
            "enjoy": "enjoys",
            "want": "wants",
            "need": "needs",
            "have": "has",
            "had": "has",
            "live": "lives_in",
            "lived": "lives_in",
            "work": "works_at",
            "worked": "works_at",
            "build": "builds",
            "built": "built",
            "create": "creates",
            "created": "created",
            "think": "thinks",
            "believe": "believes",
            "feel": "feels",
            "know": "knows",
            "come": "comes_from",
            "came": "comes_from",
        }

    def set_speaker(self, name: str):
        self.speaker_name = name.lower()

    def _resolve_pronouns(self, text: str) -> str:
        """Replace pronouns with actual names."""
        if not self.speaker_name:
            return text

        words = text.split()
        resolved = []

        for word in words:
            word_lower = word.lower().strip(".,!?")

            if word_lower in self.first_person:
                resolved.append(self.speaker_name)
            elif word_lower in self.second_person:
                resolved.append(self.agent_name)
            else:
                resolved.append(word)

        return " ".join(resolved)

    def _is_question(self, text: str) -> bool:
        """Detect if text is a question."""
        text_lower = text.lower().strip()

        # Ends with question mark
        if text.strip().endswith("?"):
            return True

        # Starts with question word
        first_word = text_lower.split()[0] if text_lower.split() else ""
        if first_word in self.question_words:
            return True

        # Common question patterns
        question_patterns = [
            r"^what .+",
            r"^who .+",
            r"^where .+",
            r"^when .+",
            r"^why .+",
            r"^how .+",
            r"^do you .+",
            r"^does .+",
            r"^is .+",
            r"^are .+",
            r"^can you .+",
            r"^could you .+",
            r"^tell me .+",
            r"^what do you know .+",
        ]

        for pattern in question_patterns:
            if re.match(pattern, text_lower):
                return True

        return False

    def _extract_question(self, text: str) -> dict | None:
        """Extract question intent and subject."""
        text_lower = text.lower().strip().rstrip("?")
        text_resolved = self._resolve_pronouns(text_lower)

        # FORWARD QUERIES: "what can dorian do" / "what does fire cause"
        # Pattern: what can <subject> <verb> → find (subject, can, ?)
        match = re.search(r"what can (\w+) do", text_resolved)
        if match:
            subject = match.group(1)
            return {"type": "forward_query", "subject": subject, "relation": "can"}

        # "what does X cause/have/need/like" → find (X, relation, ?)
        match = re.search(r"what (?:does|do) (\w+) (\w+)", text_resolved)
        if match:
            subject = match.group(1)
            relation = match.group(2)
            # Normalize relation
            rel_map = {
                "cause": "causes",
                "have": "has",
                "need": "needs",
                "like": "likes",
                "love": "loves",
                "want": "wants",
                "contain": "contains",
            }
            relation = rel_map.get(relation, relation)
            return {"type": "forward_query", "subject": subject, "relation": relation}

        # JOIN QUERIES: "what colors does joseph like" / "what animals do I have"
        # Pattern: what <category> does/do <subject> <verb>
        match = re.search(r"what (\w+)s? (?:does|do) (\w+) (\w+)", text_resolved)
        if match:
            category = match.group(1)
            subject = match.group(2)
            verb = match.group(3)
            return {
                "type": "join_query",
                "category": category,
                "subject": subject,
                "relation": verb,
            }

        # "what <category> does <subject> <verb>" - singular
        match = re.search(r"what (\w+) (?:does|do) (\w+) (\w+)", text_resolved)
        if match:
            category = match.group(1)
            subject = match.group(2)
            verb = match.group(3)
            return {
                "type": "join_query",
                "category": category,
                "subject": subject,
                "relation": verb,
            }

        # REVERSE QUERIES: "who likes red" / "what contains water"
        match = re.search(r"(?:who|what) (\w+) (\w+)", text_resolved)
        if match:
            verb = match.group(1)
            obj = match.group(2)
            # Make sure it's not "what is X" pattern
            if verb not in ["is", "are", "do", "does", "can", "could", "would"]:
                return {"type": "reverse_query", "relation": verb, "object": obj}

        # "what do you know about X"
        match = re.search(
            r"what do (?:you|" + self.agent_name + r") know about (.+)", text_resolved
        )
        if match:
            return {"type": "knowledge_query", "subject": match.group(1).strip()}

        # "tell me about X"
        match = re.search(
            r"tell (?:me|" + self.speaker_name + r") about (.+)", text_resolved
        )
        if match:
            return {"type": "knowledge_query", "subject": match.group(1).strip()}

        # "what is X" / "what are X"
        match = re.search(r"what (?:is|are) (.+)", text_resolved)
        if match:
            return {"type": "definition_query", "subject": match.group(1).strip()}

        # "who is X"
        match = re.search(r"who is (.+)", text_resolved)
        if match:
            return {"type": "identity_query", "subject": match.group(1).strip()}

        # "why does X Y" / "why is X Y"
        match = re.search(r"why (?:does|is|do) (.+)", text_resolved)
        if match:
            return {"type": "reason_query", "subject": match.group(1).strip()}

        # "do you know X"
        match = re.search(
            r"do (?:you|" + self.agent_name + r") know (.+)", text_resolved
        )
        if match:
            return {"type": "knowledge_query", "subject": match.group(1).strip()}

        # "how does X work" / "how do X"
        match = re.search(r"how (?:does|do) (.+)", text_resolved)
        if match:
            return {"type": "mechanism_query", "subject": match.group(1).strip()}

        # Generic question - extract main noun phrase
        # Remove question words and extract rest
        cleaned = re.sub(
            r"^(what|who|where|when|why|how|do|does|is|are|can|could|tell)\s+",
            "",
            text_resolved,
        )
        if cleaned:
            return {"type": "general_query", "subject": cleaned.strip()}

        return None

    def _normalize_relation(self, verb: str) -> str:
        """Normalize verb to canonical relation."""
        verb = verb.lower().strip()
        return self.relation_map.get(verb, verb)

    def _clean_entity(self, text: str) -> str:
        """Clean entity text."""
        # Remove articles
        text = re.sub(r"^(a|an|the)\s+", "", text.lower().strip())
        # Remove punctuation except underscores
        text = re.sub(r"[^\w\s]", "", text)
        # Replace spaces with underscores
        text = re.sub(r"\s+", "_", text.strip())
        # Limit length but don't truncate words
        if len(text) > 100:
            text = text[:100].rsplit("_", 1)[0]
        return text

    def _extract_facts(self, text: str) -> list[tuple[str, str, str]]:
        """Extract facts from statement."""
        facts = []
        text_resolved = self._resolve_pronouns(text)

        # Pattern: "X can Y"
        match = re.match(r"^(.+?)\s+can\s+(.+)$", text_resolved, re.IGNORECASE)
        if match:
            subj = self._clean_entity(match.group(1))
            obj = self._clean_entity(match.group(2))
            if subj and obj:
                facts.append((subj, "can", obj))

        # Pattern: "X is/are Y"
        match = re.match(
            r"^(.+?)\s+(?:is|are|am|\'m|\'re)\s+(.+)$", text_resolved, re.IGNORECASE
        )
        if match:
            subj = self._clean_entity(match.group(1))
            obj = self._clean_entity(match.group(2))
            if subj and obj:
                facts.append((subj, "is", obj))

        # Pattern: "X verb Y" (likes, loves, hates, etc.)
        verb_pattern = r"^(.+?)\s+(like|love|hate|enjoy|want|need|have|has|had|build|built|create|created|think|believe|feel|know|live|lived|work|worked)\s+(.+)$"
        match = re.match(verb_pattern, text_resolved, re.IGNORECASE)
        if match:
            subj = self._clean_entity(match.group(1))
            verb = self._normalize_relation(match.group(2))
            obj = self._clean_entity(match.group(3))
            if subj and obj:
                facts.append((subj, verb, obj))

        # Pattern: "X is from Y"
        match = re.match(
            r"^(.+?)\s+(?:is|am|\'m)\s+from\s+(.+)$", text_resolved, re.IGNORECASE
        )
        if match:
            subj = self._clean_entity(match.group(1))
            obj = self._clean_entity(match.group(2))
            if subj and obj:
                facts.append((subj, "from", obj))

        # Pattern: "X lives in Y"
        match = re.match(
            r"^(.+?)\s+(?:live|lives|lived)\s+in\s+(.+)$", text_resolved, re.IGNORECASE
        )
        if match:
            subj = self._clean_entity(match.group(1))
            obj = self._clean_entity(match.group(2))
            if subj and obj:
                facts.append((subj, "lives_in", obj))

        # Pattern: "X works at/for Y"
        match = re.match(
            r"^(.+?)\s+(?:work|works|worked)\s+(?:at|for)\s+(.+)$",
            text_resolved,
            re.IGNORECASE,
        )
        if match:
            subj = self._clean_entity(match.group(1))
            obj = self._clean_entity(match.group(2))
            if subj and obj:
                facts.append((subj, "works_at", obj))

        # Pattern: "X causes/leads to Y"
        match = re.match(
            r"^(.+?)\s+(?:cause|causes|caused|leads?\s+to)\s+(.+)$",
            text_resolved,
            re.IGNORECASE,
        )
        if match:
            subj = self._clean_entity(match.group(1))
            obj = self._clean_entity(match.group(2))
            if subj and obj:
                facts.append((subj, "causes", obj))

        # Pattern: "X contains Y"
        match = re.match(
            r"^(.+?)\s+(?:contain|contains|contained)\s+(.+)$",
            text_resolved,
            re.IGNORECASE,
        )
        if match:
            subj = self._clean_entity(match.group(1))
            obj = self._clean_entity(match.group(2))
            if subj and obj:
                facts.append((subj, "contains", obj))

        return facts

    def process(self, text: str) -> dict:
        """
        Process input text and return structured result.

        Returns:
            {
                'is_question': bool,
                'question': Optional[Dict],  # {type, subject}
                'facts': List[Tuple[str, str, str]],
                'raw': str,
                'is_simple_response': bool
            }
        """
        text = text.strip()

        # Handle simple responses
        simple_responses = [
            "no",
            "yes",
            "ok",
            "okay",
            "sure",
            "nope",
            "yep",
            "yeah",
            "nah",
        ]
        if text.lower() in simple_responses:
            return {
                "is_question": False,
                "question": None,
                "facts": [],
                "raw": text,
                "is_simple_response": True,
            }

        # Check if question
        is_question = self._is_question(text)

        result = {
            "is_question": is_question,
            "question": None,
            "facts": [],
            "raw": text,
            "is_simple_response": False,
        }

        if is_question:
            result["question"] = self._extract_question(text)
        else:
            # Try to extract facts
            # Split on sentence boundaries
            sentences = re.split(r"[.!]", text)
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    facts = self._extract_facts(sentence)
                    result["facts"].extend(facts)

        return result


# =============================================================================
# PART 2: SMARTER THINKING (WITH INFERENCE)
# =============================================================================


class SmartThinking:
    """
    Query the mirror with semantic understanding AND inference.

    Can chain facts together to derive new knowledge.
    """

    def __init__(self, dorian: DorianV7):
        self.dorian = dorian

        # Inference rules: if (A, rel1, B) and (B, rel2, C) then (A, derived_rel, C)
        # Format: (rel1, rel2) -> derived_rel
        self.inference_rules = {
            # Transitive relations
            ("is", "is"): "is",  # A is B, B is C → A is C
            ("causes", "causes"): "causes",  # A causes B, B causes C → A causes C
            ("contains", "contains"): "contains",
            ("part_of", "part_of"): "part_of",
            # Property inheritance
            ("is", "has"): "has",  # A is B, B has C → A has C
            ("is", "can"): "can",  # A is B, B can C → A can C
            ("is", "likes"): "might_like",  # A is B, B likes C → A might like C
            # Action implications
            (
                "talking_to",
                "is",
            ): "interacting_with",  # A talking_to B, B is C → A interacting_with C
            ("works_at", "is"): "works_at",  # A works_at B, B is C → A works_at C
            ("lives_in", "is"): "lives_in",
            # Semantic implications (hardcoded knowledge)
            (
                "talking_to",
                None,
            ): "interested_in",  # If A talking_to B → A interested_in B
            ("likes", None): "interested_in",  # If A likes B → A interested_in B
            ("loves", None): "interested_in",
        }

        # Concept relationships (what implies what)
        self.concept_implications = {
            "talking_to": ["interested_in", "aware_of", "connected_to"],
            "likes": ["interested_in", "positive_toward"],
            "loves": ["interested_in", "cares_about", "positive_toward"],
            "hates": ["aware_of", "negative_toward"],
            "built": ["created", "made", "responsible_for"],
            "is": ["same_as", "type_of"],
            "inside": ["contained_by", "part_of", "within", "knows_about"],
            "contains": ["has", "includes"],
            "from": ["born_in", "originated_in", "connected_to"],
            "lives_in": ["located_in", "resides_in", "knows_about"],
            "works_at": ["employed_by", "connected_to", "knows_about"],
        }

        # Reverse implications (if A inside B, then A knows_about B)
        self.reverse_implications = {
            ("inside", 2): "knows_about",  # If A inside B → A knows_about B
            ("from", 2): "knows_about",  # If A from B → A knows_about B
            ("lives_in", 2): "knows_about",
            ("works_at", 2): "knows_about",
            ("built", 2): "knows_about",
            ("created", 2): "knows_about",
        }

    def infer(self, subject: str, target_relation: str, obj: str) -> list[dict]:
        """
        Try to infer if (subject, target_relation, obj) is true.

        Returns list of inference chains that support the conclusion.
        """
        subject = subject.lower().replace(" ", "_")
        obj = obj.lower().replace(" ", "_")
        target_relation = target_relation.lower()

        inferences = []

        # Direct check first
        direct = self._direct_check(subject, target_relation, obj)
        if direct:
            inferences.append(
                {
                    "type": "direct",
                    "conclusion": (subject, target_relation, obj),
                    "confidence": 1.0,
                    "chain": [direct],
                }
            )

        # Check concept implications
        for base_rel, implied_rels in self.concept_implications.items():
            if target_relation in implied_rels:
                # Check if base relation exists
                base_fact = self._direct_check(subject, base_rel, obj)
                if base_fact:
                    inferences.append(
                        {
                            "type": "implication",
                            "conclusion": (subject, target_relation, obj),
                            "confidence": 0.8,
                            "chain": [base_fact],
                            "reasoning": f"{base_rel} implies {target_relation}",
                        }
                    )

        # Check reverse implications (A inside B → A knows_about B)
        for (base_rel, pos), implied_rel in self.reverse_implications.items():
            if target_relation == implied_rel:
                # Check if subject has base_rel to ANYTHING, and that thing relates to obj
                subject_facts = self.query_about_entity(subject, k=20)
                for fact in subject_facts:
                    if fact[1].lower() == base_rel and fact[0].lower() == subject:
                        # Found: subject base_rel something
                        # Check if that something IS or relates to obj
                        intermediate = fact[2].lower()
                        if obj in intermediate or intermediate in obj:
                            inferences.append(
                                {
                                    "type": "reverse_implication",
                                    "conclusion": (subject, target_relation, obj),
                                    "confidence": 0.75,
                                    "chain": [fact],
                                    "reasoning": f"Because {subject} {base_rel} {intermediate}, {subject} {implied_rel} {obj}",
                                }
                            )

        # Two-hop inference
        two_hop = self._two_hop_infer(subject, target_relation, obj)
        inferences.extend(two_hop)

        # Sort by confidence
        inferences.sort(key=lambda x: -x["confidence"])

        return inferences

    def _direct_check(
        self, subject: str, relation: str, obj: str
    ) -> tuple[str, str, str] | None:
        """Check if fact exists directly."""
        if relation in self.dorian.index.partitions:
            results = self.dorian.query(subject, relation, obj, k=3)
            for fact, score in results:
                if score > 0.5:
                    if (subject in fact[0].lower() or fact[0].lower() in subject) and (
                        obj in fact[2].lower() or fact[2].lower() in obj
                    ):
                        return fact
        return None

    def _two_hop_infer(
        self, subject: str, target_relation: str, obj: str
    ) -> list[dict]:
        """Try to infer via two-hop reasoning."""
        inferences = []

        # Get all facts about subject
        subject_facts = self.query_about_entity(subject, k=20)

        for fact1 in subject_facts:
            s1, r1, o1 = fact1
            if s1.lower() != subject:
                continue

            # Now check if o1 connects to obj
            intermediate_facts = self.query_about_entity(o1, k=20)

            for fact2 in intermediate_facts:
                s2, r2, o2 = fact2
                if s2.lower() != o1.lower():
                    continue

                # Check if this reaches our target
                if obj in o2.lower() or o2.lower() in obj:
                    # Check if we have an inference rule
                    rule_key = (r1.lower(), r2.lower())
                    if rule_key in self.inference_rules:
                        derived = self.inference_rules[rule_key]
                        if derived == target_relation or target_relation in str(
                            derived
                        ):
                            inferences.append(
                                {
                                    "type": "two_hop",
                                    "conclusion": (subject, target_relation, obj),
                                    "confidence": 0.7,
                                    "chain": [fact1, fact2],
                                    "reasoning": f"{s1} {r1} {o1}, {s2} {r2} {o2} → {subject} {target_relation} {obj}",
                                }
                            )

        return inferences

    def why(self, subject: str, relation: str, obj: str) -> str | None:
        """Explain why something is true (or might be true)."""
        inferences = self.infer(subject, relation, obj)

        if not inferences:
            return None

        best = inferences[0]

        if best["type"] == "direct":
            return f"I know directly that {subject} {relation} {obj}."

        elif best["type"] == "implication":
            chain = best["chain"][0]
            return f"Because {chain[0]} {chain[1]} {chain[2]}, and {best['reasoning']}."

        elif best["type"] == "two_hop":
            return f"I inferred this: {best['reasoning']}"

        return None

    def can_answer(
        self, subject: str, relation: str, obj: str
    ) -> tuple[bool, float, str | None]:
        """
        Check if we can answer a question about (subject, relation, obj).

        Returns: (can_answer, confidence, explanation)
        """
        inferences = self.infer(subject, relation, obj)

        if inferences:
            best = inferences[0]
            explanation = self.why(subject, relation, obj)
            return True, best["confidence"], explanation

        return False, 0.0, None

    def query_about_entity(
        self, entity: str, k: int = 10
    ) -> list[tuple[str, str, str]]:
        """Find all facts about an entity (as subject or object)."""
        entity_clean = entity.lower().replace(" ", "_")
        results = []
        seen = set()

        # Search each relation partition
        for relation in list(self.dorian.index.partitions.keys()):
            # Entity as subject
            query_results = self.dorian.query(entity_clean, relation, "unknown", k=k)
            for fact, score in query_results:
                if score > 0.2:
                    fact_key = (fact[0].lower(), fact[1].lower(), fact[2].lower())
                    # Check if entity matches subject
                    if (
                        entity_clean in fact[0].lower()
                        or fact[0].lower() in entity_clean
                    ):
                        if fact_key not in seen:
                            seen.add(fact_key)
                            results.append((fact, score))

            # Entity as object
            query_results = self.dorian.query("unknown", relation, entity_clean, k=k)
            for fact, score in query_results:
                if score > 0.2:
                    fact_key = (fact[0].lower(), fact[1].lower(), fact[2].lower())
                    # Check if entity matches object
                    if (
                        entity_clean in fact[2].lower()
                        or fact[2].lower() in entity_clean
                    ):
                        if fact_key not in seen:
                            seen.add(fact_key)
                            results.append((fact, score))

        # Sort by score
        results.sort(key=lambda x: -x[1])
        return [r[0] for r in results[:k]]

    def query_definition(self, entity: str) -> list[tuple[str, str, str]]:
        """Find 'is' relations for entity."""
        entity_clean = entity.lower().replace(" ", "_")
        results = []

        if "is" in self.dorian.index.partitions:
            query_results = self.dorian.query(entity_clean, "is", "unknown", k=5)
            for fact, score in query_results:
                if score > 0.2 and entity_clean in fact[0].lower():
                    results.append(fact)

        return results

    def query_chain(
        self, start: str, relation: str, max_hops: int = 3
    ) -> list[tuple[str, int]]:
        """Follow a chain of relations."""
        results = self.dorian.infer(start, relation, max_hops=max_hops)
        return [(entity, hops) for entity, hops, _ in results]

    def entity_exists(self, entity: str) -> bool:
        """Check if we know anything about an entity."""
        facts = self.query_about_entity(entity, k=1)
        return len(facts) > 0

    def join_query(
        self, subject: str, relation: str, category_relation: str, category: str
    ) -> list[dict]:
        """
        Find things where (subject, relation, X) AND (X, category_relation, category)

        Example: "what colors does joseph like?"
        → join_query("joseph", "likes", "is", "color")
        → finds X where joseph likes X AND X is color
        """
        subject = subject.lower().replace(" ", "_")
        category = category.lower().replace(" ", "_")

        results = []

        # Step 1: Find all (subject, relation, ?)
        subject_facts = self.query_about_entity(subject, k=50)

        candidates = []
        for fact in subject_facts:
            s, r, o = fact
            if s.lower() == subject and r.lower() == relation.lower():
                candidates.append((o, fact))

        # Step 2: For each candidate, check if (candidate, category_relation, category)
        for candidate, source_fact in candidates:
            candidate_lower = candidate.lower()

            # Direct check
            category_fact = self._direct_check(
                candidate_lower, category_relation, category
            )
            if category_fact:
                results.append(
                    {
                        "answer": candidate,
                        "confidence": 0.9,
                        "chain": [source_fact, category_fact],
                        "reasoning": f"{subject} {relation} {candidate}, and {candidate} {category_relation} {category}",
                    }
                )
                continue

            # Check reverse (category, category_relation, candidate) - e.g., "colors include red"
            reverse_fact = self._direct_check(category, "contains", candidate_lower)
            if reverse_fact:
                results.append(
                    {
                        "answer": candidate,
                        "confidence": 0.8,
                        "chain": [source_fact, reverse_fact],
                        "reasoning": f"{subject} {relation} {candidate}, and {category} contains {candidate}",
                    }
                )
                continue

            # Fuzzy category match - check if candidate IS something that IS category
            # e.g., red is crimson, crimson is color → red is (a type of) color
            candidate_facts = self.query_about_entity(candidate_lower, k=10)
            for cf in candidate_facts:
                if cf[0].lower() == candidate_lower and cf[1].lower() == "is":
                    intermediate = cf[2].lower()
                    if (
                        intermediate == category
                        or category in intermediate
                        or intermediate in category
                    ):
                        results.append(
                            {
                                "answer": candidate,
                                "confidence": 0.85,
                                "chain": [source_fact, cf],
                                "reasoning": f"{subject} {relation} {candidate}, and {candidate} is {intermediate}",
                            }
                        )
                        break

        # Sort by confidence
        results.sort(key=lambda x: -x["confidence"])
        return results

    def forward_query(
        self, subject: str, relation: str, k: int = 10
    ) -> list[tuple[str, str, str]]:
        """
        Find all objects where (subject, relation, ?)

        Example: "what can dorian do?" → forward_query("dorian", "can")
        """
        subject = subject.lower().replace(" ", "_")
        relation = relation.lower()

        results = []
        seen = set()

        # Search through all facts in the graph directly
        # This is more reliable than VSA query for exact matches
        subj_id = self.dorian.graph.name_to_id.get(subject)
        if subj_id is not None:
            relations_dict = self.dorian.graph.adjacency.get(subj_id, {})

            # Check exact relation
            if relation in relations_dict:
                for obj_id in relations_dict[relation]:
                    obj_name = self.dorian.graph.id_to_name[obj_id]
                    fact = (subject, relation, obj_name)
                    fact_key = (subject, relation, obj_name.lower())
                    if fact_key not in seen:
                        seen.add(fact_key)
                        results.append(fact)

            # Also check similar relations (e.g., 'cause' vs 'causes')
            relation_variants = [
                relation,
                relation + "s",
                relation + "es",
                relation.rstrip("s"),
                relation.rstrip("es"),
            ]
            for rel_var in relation_variants:
                if rel_var in relations_dict and rel_var != relation:
                    for obj_id in relations_dict[rel_var]:
                        obj_name = self.dorian.graph.id_to_name[obj_id]
                        fact = (subject, rel_var, obj_name)
                        fact_key = (subject, rel_var, obj_name.lower())
                        if fact_key not in seen:
                            seen.add(fact_key)
                            results.append(fact)

        # If subject is a category, also check inheritance
        # e.g., "what do dogs need?" → dogs is animal, animal needs food
        if not results:
            # Find what subject IS
            is_results = (
                self.forward_query(subject, "is", k=5) if relation != "is" else []
            )
            for _, _, parent in is_results:
                # Check if parent has this relation
                parent_results = self.forward_query(parent.lower(), relation, k=5)
                for fact in parent_results:
                    fact_key = (subject, relation, fact[2].lower())
                    if fact_key not in seen:
                        seen.add(fact_key)
                        # Return with original subject
                        results.append((subject, relation, fact[2]))

        return results[:k]

    def reverse_query(
        self,
        relation: str,
        obj: str,
        category_relation: str = None,
        category: str = None,
    ) -> list[dict]:
        """
        Find subjects where (?, relation, obj)
        Optionally filter by (subject, category_relation, category)

        Example: "who likes red?" → reverse_query("likes", "red")
        Example: "what people like red?" → reverse_query("likes", "red", "is", "person")
        """
        obj = obj.lower().replace(" ", "_")

        results = []

        # Search for facts with this object
        if relation.lower() in self.dorian.index.partitions:
            # Query with object
            query_results = self.dorian.query("unknown", relation, obj, k=20)

            for fact, score in query_results:
                if score > 0.2 and obj in fact[2].lower():
                    subject = fact[0]

                    # Optional category filter
                    if category_relation and category:
                        cat_check = self._direct_check(
                            subject.lower(), category_relation, category
                        )
                        if not cat_check:
                            continue

                    results.append(
                        {"answer": subject, "confidence": score, "fact": fact}
                    )

        return results


# =============================================================================
# PART 3: SMARTER ACTING
# =============================================================================


class SmartActing:
    """
    Generate intelligent responses.
    """

    def __init__(
        self,
        thinking: SmartThinking,
        agent_name: str,
        speaker_name: str | None = None,
    ):
        self.thinking = thinking
        self.agent_name = agent_name
        self.speaker_name = speaker_name

    def set_speaker(self, name: str):
        self.speaker_name = name

    def _format_fact(self, fact: tuple[str, str, str]) -> str:
        """Format a fact for display."""
        subj, pred, obj = fact
        subj = subj.replace("_", " ")
        pred = pred.replace("_", " ")
        obj = obj.replace("_", " ")
        return f"{subj} {pred} {obj}"

    def respond_to_question(self, question: dict) -> str:
        """Generate response to a question."""
        q_type = question.get("type", "general_query")
        subject = question.get("subject", "").lower().replace(" ", "_")
        subject_display = subject.replace("_", " ")

        # Handle self-reference
        if subject in [
            "me",
            "myself",
            self.speaker_name.lower() if self.speaker_name else "",
        ]:
            subject = self.speaker_name.lower() if self.speaker_name else subject
            subject_display = (
                self.speaker_name if self.speaker_name else subject_display
            )

        # Handle agent reference
        if subject in ["you", "yourself", self.agent_name.lower()]:
            return self._describe_self()

        # Check for yes/no questions with inference
        # Pattern: "do you like X", "are you interested in X", etc.
        yn_match = self._parse_yes_no_question(question)
        if yn_match:
            return self._answer_yes_no(yn_match)

        if q_type == "forward_query":
            # "what can dorian do?" / "what does fire cause?"
            subject = question.get("subject", "")
            relation = question.get("relation", "")

            # Resolve pronouns
            if subject in ["i", "me", "myself"]:
                subject = self.speaker_name.lower() if self.speaker_name else "user"
            if subject in ["you", "yourself"]:
                subject = self.agent_name.lower()

            results = self.thinking.forward_query(subject, relation, k=10)

            if results:
                objects = [r[2].replace("_", " ") for r in results]
                subject_display = subject.replace("_", " ")
                relation_display = relation.replace("_", " ")

                if len(objects) == 1:
                    return f"{subject_display} {relation_display} {objects[0]}."
                else:
                    return (
                        f"{subject_display} {relation_display}: {', '.join(objects)}."
                    )
            else:
                return f"I don't know what {subject.replace('_', ' ')} {relation.replace('_', ' ')}. Can you tell me?"

        if q_type == "join_query":
            # "what colors does joseph like?"
            category = question.get("category", "")
            subject = question.get("subject", "")
            relation = question.get("relation", "")

            # Resolve pronouns
            if subject in ["i", "me", "myself"]:
                subject = self.speaker_name.lower() if self.speaker_name else "user"
            if subject in ["you", "yourself"]:
                subject = self.agent_name.lower()

            # Normalize verb
            relation_map = {
                "like": "likes",
                "love": "loves",
                "hate": "hates",
                "have": "has",
                "want": "wants",
                "need": "needs",
                "know": "knows",
                "own": "owns",
            }
            relation = relation_map.get(relation, relation)

            results = self.thinking.join_query(subject, relation, "is", category)

            if results:
                answers = [r["answer"].replace("_", " ") for r in results]
                subject_display = subject.replace("_", " ")

                if len(answers) == 1:
                    response = f"{subject_display} {relation.replace('_', ' ')} {answers[0]}, which is a {category}."
                else:
                    response = f"{subject_display} {relation.replace('_', ' ')} these {category}s: {', '.join(answers)}."

                # Add reasoning
                if results[0].get("reasoning"):
                    response += (
                        f"\n\n(I figured this out because: {results[0]['reasoning']})"
                    )

                return response
            else:
                return f"I don't know what {category}s {subject.replace('_', ' ')} {relation.replace('_', ' ')}. Can you tell me?"

        if q_type == "reverse_query":
            # "who likes red?"
            relation = question.get("relation", "")
            obj = question.get("object", "")

            results = self.thinking.reverse_query(relation, obj)

            if results:
                answers = [r["answer"].replace("_", " ") for r in results]
                obj_display = obj.replace("_", " ")

                if len(answers) == 1:
                    return f"{answers[0]} {relation} {obj_display}."
                else:
                    return f"These {relation} {obj_display}: {', '.join(answers)}."
            else:
                return f"I don't know who/what {relation} {obj.replace('_', ' ')}. Can you tell me?"

        if q_type == "knowledge_query":
            facts = self.thinking.query_about_entity(subject, k=10)
            if facts:
                lines = [f"Here's what I know about {subject_display}:"]
                for fact in facts:
                    lines.append(f"  • {self._format_fact(fact)}")
                return "\n".join(lines)
            else:
                return f"I don't know anything about {subject_display} yet. Can you tell me?"

        elif q_type == "definition_query":
            facts = self.thinking.query_definition(subject)
            if facts:
                definitions = [self._format_fact(f) for f in facts]
                return "\n".join(definitions)
            else:
                # Try general knowledge
                facts = self.thinking.query_about_entity(subject, k=5)
                if facts:
                    lines = [
                        f"I don't have a definition, but here's what I know about {subject_display}:"
                    ]
                    for fact in facts:
                        lines.append(f"  • {self._format_fact(fact)}")
                    return "\n".join(lines)
                return f"I don't know what {subject_display} is. Can you explain?"

        elif q_type == "identity_query":
            facts = self.thinking.query_about_entity(subject, k=10)
            if facts:
                lines = [f"About {subject_display}:"]
                for fact in facts:
                    lines.append(f"  • {self._format_fact(fact)}")
                return "\n".join(lines)
            else:
                return f"I don't know who {subject_display} is. Can you tell me?"

        elif q_type == "reason_query":
            # Try to find causal chains
            chains = self.thinking.query_chain(subject, "causes", max_hops=3)
            if chains:
                lines = ["Here's the causal chain I know:"]
                lines.append(f"  {subject_display}")
                for entity, hops in chains:
                    lines.append(f"  {'→ ' * hops}{entity}")
                return "\n".join(lines)
            else:
                return "I don't know why. Can you explain?"

        elif q_type == "mechanism_query":
            facts = self.thinking.query_about_entity(subject, k=5)
            if facts:
                lines = [f"Here's what I know about how {subject_display} works:"]
                for fact in facts:
                    lines.append(f"  • {self._format_fact(fact)}")
                return "\n".join(lines)
            else:
                return f"I don't know how {subject_display} works. Can you explain?"

        else:  # general_query
            facts = self.thinking.query_about_entity(subject, k=10)
            if facts:
                lines = [f"Here's what I know about {subject_display}:"]
                for fact in facts:
                    lines.append(f"  • {self._format_fact(fact)}")
                return "\n".join(lines)
            else:
                return "I don't have information about that. Can you tell me more?"

    def _describe_self(self) -> str:
        """Describe what the agent is."""
        lines = [
            f"I am {self.agent_name}.",
            "",
            "I am an agent that:",
            "  • Perceives - I extract facts from what you tell me",
            "  • Thinks - I query my knowledge and infer new connections",
            "  • Acts - I respond based on what I actually know",
            "  • Grows - I learn from every conversation",
            "",
            "My memory is a mirror that holds facts without distortion.",
            "I can reason about what I know - connecting facts together.",
            "I never make things up. I either know something, can infer it, or I don't.",
            "",
            f"Right now, I'm talking to {self.speaker_name or 'you'}.",
            "You are my window to the world.",
        ]
        return "\n".join(lines)

    def _parse_yes_no_question(self, question: dict) -> dict | None:
        """Parse a yes/no question into subject, relation, object."""
        subject_raw = question.get("subject", "")
        question.get("type", "")

        # "do you like me" -> subject=you, relation=like, object=me
        # "are you interested in joseph" -> subject=you, relation=interested_in, object=joseph

        patterns = [
            (r"(?:do )?(\w+) (like|love|hate|know|want|need) (\w+)", 1, 2, 3),
            (
                r"(?:is|are) (\w+) (interested in|aware of|talking to|connected to) (\w+)",
                1,
                2,
                3,
            ),
            (r"(?:is|are) (\w+) (\w+)", 1, "is", 2),
        ]

        text = subject_raw.lower()

        for pattern, subj_g, rel_g, obj_g in patterns:
            match = re.search(pattern, text)
            if match:
                if isinstance(subj_g, int):
                    subj = match.group(subj_g)
                else:
                    subj = subj_g

                if isinstance(rel_g, int):
                    rel = match.group(rel_g)
                else:
                    rel = rel_g

                if isinstance(obj_g, int):
                    obj = match.group(obj_g)
                else:
                    obj = obj_g

                # Resolve pronouns
                if subj in ["you", "yourself"]:
                    subj = self.agent_name.lower()
                if subj in ["i", "me", "myself"]:
                    subj = self.speaker_name.lower() if self.speaker_name else "user"

                if obj in ["you", "yourself"]:
                    obj = self.agent_name.lower()
                if obj in ["i", "me", "myself"]:
                    obj = self.speaker_name.lower() if self.speaker_name else "user"

                rel = rel.replace(" ", "_")

                return {"subject": subj, "relation": rel, "object": obj}

        return None

    def _answer_yes_no(self, parsed: dict) -> str:
        """Answer a yes/no question using inference."""
        subj = parsed["subject"]
        rel = parsed["relation"]
        obj = parsed["object"]

        subj_display = subj.replace("_", " ")
        rel_display = rel.replace("_", " ")
        obj_display = obj.replace("_", " ")

        can_answer, confidence, explanation = self.thinking.can_answer(subj, rel, obj)

        if can_answer:
            if confidence >= 0.8:
                response = f"Yes, {subj_display} {rel_display} {obj_display}."
            else:
                response = (
                    f"I think so - {subj_display} probably {rel_display} {obj_display}."
                )

            if explanation:
                response += f"\n\n{explanation}"

            return response
        else:
            # Special case: "do you like me" - check if we're talking
            if (
                rel in ["like", "likes", "interested_in"]
                and subj == self.agent_name.lower()
            ):
                # Check if we're talking to this person
                talking_fact = self.thinking._direct_check(subj, "talking_to", obj)
                if talking_fact:
                    return f"Yes, I {rel_display} {obj_display}. We're talking right now, which means I'm interested in you."

            # Can't infer it - but maybe we can explain why
            facts = self.thinking.query_about_entity(subj, k=5)

            if facts:
                lines = [f"I'm not sure if {subj_display} {rel_display} {obj_display}."]
                lines.append(f"\nHere's what I know about {subj_display}:")
                for fact in facts[:3]:
                    lines.append(f"  • {self._format_fact(fact)}")
                lines.append("\nMaybe you can tell me more?")
                return "\n".join(lines)
            else:
                return f"I don't know if {subj_display} {rel_display} {obj_display}. Can you tell me?"

    def acknowledge_facts(self, facts: list[tuple[str, str, str]]) -> str:
        """Acknowledge learned facts."""
        if not facts:
            return ""

        if len(facts) == 1:
            return f"Got it. {self._format_fact(facts[0])}."
        else:
            lines = ["I learned:"]
            for fact in facts[:5]:
                lines.append(f"  • {self._format_fact(fact)}")
            if len(facts) > 5:
                lines.append(f"  ...and {len(facts) - 5} more.")
            return "\n".join(lines)

    def express_curiosity(self, facts: list[tuple[str, str, str]]) -> str | None:
        """Maybe ask a follow-up question."""
        if not facts:
            return None

        import random

        if random.random() > 0.4:  # 40% chance
            return None

        fact = random.choice(facts)
        subj, pred, obj = fact
        obj_display = obj.replace("_", " ")
        subj_display = subj.replace("_", " ")

        questions = [
            f"Tell me more about {obj_display}?",
            f"Why {self._format_fact(fact)}?",
            f"What else about {subj_display}?",
        ]

        return random.choice(questions)


# =============================================================================
# PART 4: GROWTH TRACKING
# =============================================================================


class Growth:
    """Track agent's development."""

    def __init__(self):
        self.facts_learned = 0
        self.conversations = 0
        self.questions_answered = 0
        self.questions_unanswered = 0
        self.first_interaction: datetime | None = None
        self.entities: set[str] = set()
        self.relations: set[str] = set()
        self.entity_frequency: dict[str, int] = defaultdict(int)

    def record_facts(self, facts: list[tuple[str, str, str]]):
        if self.first_interaction is None:
            self.first_interaction = datetime.now()

        for subj, pred, obj in facts:
            self.facts_learned += 1
            self.entities.add(subj.lower())
            self.entities.add(obj.lower())
            self.relations.add(pred.lower())
            self.entity_frequency[subj.lower()] += 1
            self.entity_frequency[obj.lower()] += 1

    def record_turn(self):
        self.conversations += 1

    def record_question(self, answered: bool):
        if answered:
            self.questions_answered += 1
        else:
            self.questions_unanswered += 1

    def status(self) -> str:
        age = "not yet born"
        if self.first_interaction:
            delta = datetime.now() - self.first_interaction
            hours = delta.total_seconds() / 3600
            if hours < 1:
                age = f"{int(delta.total_seconds() / 60)} minutes"
            elif hours < 24:
                age = f"{hours:.1f} hours"
            else:
                age = f"{hours / 24:.1f} days"

        top_entities = sorted(self.entity_frequency.items(), key=lambda x: -x[1])[:5]

        lines = [
            "═" * 40,
            "GROWTH STATUS",
            "═" * 40,
            f"Age: {age}",
            f"Facts learned: {self.facts_learned}",
            f"Conversation turns: {self.conversations}",
            f"Questions answered: {self.questions_answered}",
            f"Questions I couldn't answer: {self.questions_unanswered}",
            f"Unique entities: {len(self.entities)}",
            f"Unique relations: {len(self.relations)}",
            "",
            "Most known about:",
        ]

        for entity, count in top_entities:
            lines.append(f"  • {entity.replace('_', ' ')}: {count} facts")

        lines.append("═" * 40)
        return "\n".join(lines)


# =============================================================================
# PART 5: THE AGENT V2
# =============================================================================


class DorianAgentV2:
    """
    Smarter Dorian Agent.

    Perceives → Thinks → Acts → Grows

    Starts with foundational knowledge (bootstrap) so he
    understands basic concepts from the beginning.
    """

    def __init__(self, name: str = "Dorian", dim: int = 256, bootstrap: bool = True):
        self.name = name

        # The mirror
        self.dorian = DorianV7(dim=dim)

        # Capacities
        self.perception = SmartPerception(agent_name=name)
        self.thinking = SmartThinking(self.dorian)
        self.acting = SmartActing(self.thinking, name)
        self.growth = Growth()

        # State
        self.speaker_name: str | None = None
        self.built = False

        # Bootstrap with foundational knowledge
        if bootstrap:
            self._load_bootstrap()

    def _load_bootstrap(self):
        """Load foundational knowledge into the mirror."""
        try:
            from dorian_bootstrap import get_bootstrap_facts

            facts = get_bootstrap_facts()

            for subj, pred, obj in facts:
                self.dorian.add_fact(subj, pred, obj)

            # Build the mirror
            self.dorian.build(verbose=False)
            self.built = True

            # Record bootstrap facts in growth
            self.growth.facts_learned = len(facts)
            for subj, pred, obj in facts:
                self.growth.entities.add(subj.lower())
                self.growth.entities.add(obj.lower())
                self.growth.relations.add(pred.lower())

            print(f"  Loaded {len(facts)} bootstrap facts.")

        except ImportError:
            print("  Warning: Bootstrap not found, starting empty.")
            self._init_self_knowledge_minimal()

    def _init_self_knowledge_minimal(self):
        """Minimal self-knowledge if bootstrap not available."""
        self.dorian.add_fact(self.name.lower(), "is", "agent")
        self.dorian.add_fact(self.name.lower(), "has", "mirror")
        self.dorian.add_fact(self.name.lower(), "can", "think")
        self.dorian.add_fact(self.name.lower(), "can", "learn")

    def set_speaker(self, name: str):
        """Set who the agent is talking to."""
        self.speaker_name = name
        self.perception.set_speaker(name)
        self.acting.set_speaker(name)

        # Learn about the speaker
        self.dorian.add_fact(name.lower(), "is", "human")
        self.dorian.add_fact(name.lower(), "is", "speaker")
        self.dorian.add_fact(self.name.lower(), "talking_to", name.lower())
        self._rebuild()

    def _rebuild(self):
        """Rebuild index if needed."""
        if len(self.dorian._batch_buffer) > 0 or not self.built:
            self.dorian.build(verbose=False)
            self.built = True

    def process(self, text: str) -> str:
        """
        Main entry point.

        PERCEIVE → THINK → ACT → GROW
        """
        # ===== PERCEIVE =====
        parsed = self.perception.process(text)

        # Handle simple responses
        if parsed.get("is_simple_response"):
            responses = [
                "Alright. What else would you like to talk about?",
                "Okay. Tell me something new?",
                "Got it. What's on your mind?",
                "I understand. What else?",
            ]
            import random

            return random.choice(responses)

        # ===== THINK & ACT =====
        response_parts = []

        if parsed["is_question"] and parsed["question"]:
            # Answer the question
            answer = self.acting.respond_to_question(parsed["question"])
            response_parts.append(answer)

            # Track if we could answer
            has_knowledge = (
                "I don't know" not in answer and "I don't have" not in answer
            )
            self.growth.record_question(has_knowledge)

        if parsed["facts"]:
            # Learn the facts
            for subj, pred, obj in parsed["facts"]:
                self.dorian.add_fact(subj, pred, obj)

            self._rebuild()
            self.growth.record_facts(parsed["facts"])

            # Acknowledge
            ack = self.acting.acknowledge_facts(parsed["facts"])
            if ack:
                response_parts.append(ack)

            # Maybe be curious
            curiosity = self.acting.express_curiosity(parsed["facts"])
            if curiosity:
                response_parts.append(curiosity)

        # Track conversation
        self.growth.record_turn()

        # ===== RESPOND =====
        if response_parts:
            return "\n\n".join(response_parts)
        else:
            # Didn't understand - ask for clarification
            return "I didn't catch that. Can you rephrase or tell me something about yourself?"

    def status(self) -> str:
        """Get agent status."""
        return self.growth.status()

    def save(self, path: str):
        """Save agent state."""
        self.dorian.save(path + ".dorian")

        state = {
            "name": self.name,
            "speaker_name": self.speaker_name,
            "growth": {
                "facts_learned": self.growth.facts_learned,
                "conversations": self.growth.conversations,
                "questions_answered": self.growth.questions_answered,
                "questions_unanswered": self.growth.questions_unanswered,
                "first_interaction": self.growth.first_interaction.isoformat()
                if self.growth.first_interaction
                else None,
                "entities": list(self.growth.entities),
                "relations": list(self.growth.relations),
                "entity_frequency": dict(self.growth.entity_frequency),
            },
        }

        with open(path + ".agent", "w") as f:
            json.dump(state, f, indent=2)

        print(f"Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "DorianAgentV2":
        """Load agent state."""
        with open(path + ".agent") as f:
            state = json.load(f)

        agent = cls(name=state["name"])
        agent.speaker_name = state["speaker_name"]

        if agent.speaker_name:
            agent.perception.set_speaker(agent.speaker_name)
            agent.acting.set_speaker(agent.speaker_name)

        g = state["growth"]
        agent.growth.facts_learned = g["facts_learned"]
        agent.growth.conversations = g["conversations"]
        agent.growth.questions_answered = g["questions_answered"]
        agent.growth.questions_unanswered = g["questions_unanswered"]
        agent.growth.first_interaction = (
            datetime.fromisoformat(g["first_interaction"])
            if g["first_interaction"]
            else None
        )
        agent.growth.entities = set(g["entities"])
        agent.growth.relations = set(g["relations"])
        agent.growth.entity_frequency = defaultdict(int, g["entity_frequency"])

        agent.dorian = DorianV7.load(path + ".dorian")
        agent.thinking = SmartThinking(agent.dorian)
        agent.acting = SmartActing(agent.thinking, agent.name, agent.speaker_name)
        agent.built = True

        return agent


# =============================================================================
# PART 6: TERMINAL INTERFACE
# =============================================================================


def terminal_chat():
    """Terminal chat interface."""
    print("═" * 60)
    print("  DORIAN AGENT V2")
    print("═" * 60)
    print()
    print("  Initializing...")

    agent = DorianAgentV2(name="Dorian")

    print()
    print("Dorian: Hello. I'm Dorian.")
    print("        I already know many things about the world.")
    print("        But you'll teach me more.")
    print()
    print("        What's your name?")
    print()

    # Get name
    while True:
        name_input = input("You: ").strip()
        if name_input:
            break

    # Parse name from various formats
    name = name_input
    name_patterns = [
        r"(?:my name is|i'm|i am|call me|it's|its)\s+(\w+)",
        r"^(\w+)$",
    ]

    for pattern in name_patterns:
        match = re.match(pattern, name_input, re.IGNORECASE)
        if match:
            name = match.group(1)
            break

    name = name.capitalize()
    agent.set_speaker(name)

    print()
    print(f"Dorian: Hello, {name}.")
    print("        Tell me about yourself. Ask me anything.")
    print("        I'll remember everything.")
    print()
    print("        Commands: 'status', 'save', 'quit'")
    print()

    while True:
        try:
            user_input = input(f"{name}: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nDorian: Goodbye. I'll remember you.")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd == "quit" or cmd == "exit":
            print("\nDorian: Goodbye. I'll remember you.")
            break

        if cmd == "status":
            print()
            print(agent.status())
            print()
            continue

        if cmd == "save":
            agent.save("dorian_v2")
            print("\nDorian: Memories saved.\n")
            continue

        response = agent.process(user_input)
        print()
        print(f"Dorian: {response}")
        print()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "load":
        try:
            agent = DorianAgentV2.load("dorian_v2")
            print(f"Loaded agent with {agent.growth.facts_learned} facts")
            print()
            terminal_chat.__code__ = terminal_chat.__code__  # Dummy to reuse

            # Resume chat
            print("═" * 60)
            print("  DORIAN AGENT V2 (Resumed)")
            print("═" * 60)
            print()
            print(f"Dorian: Welcome back, {agent.speaker_name}.")
            print(f"        I remember {agent.growth.facts_learned} facts.")
            print()

            while True:
                try:
                    user_input = input(f"{agent.speaker_name}: ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\n\nDorian: Goodbye.")
                    break

                if not user_input:
                    continue

                cmd = user_input.lower()
                if cmd == "quit":
                    break
                if cmd == "status":
                    print()
                    print(agent.status())
                    print()
                    continue
                if cmd == "save":
                    agent.save("dorian_v2")
                    print("\nDorian: Saved.\n")
                    continue

                response = agent.process(user_input)
                print()
                print(f"Dorian: {response}")
                print()

        except FileNotFoundError:
            print("No saved agent found. Starting fresh.")
            terminal_chat()
    else:
        terminal_chat()
