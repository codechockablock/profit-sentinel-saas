"""
DORIAN PHILOSOPHY DOMAIN
========================

Comprehensive philosophy knowledge for Dorian Core.

Author: Joseph + Claude
Date: 2026-01-25
"""

PHIL_CATEGORIES = [
    # Branches
    ("philosophy", "field", "Love of wisdom"),
    ("epistemology", "philosophy", "Study of knowledge"),
    ("metaphysics", "philosophy", "Study of reality"),
    ("ethics", "philosophy", "Study of morality"),
    ("logic", "philosophy", "Study of valid reasoning"),
    ("aesthetics", "philosophy", "Study of beauty and art"),
    ("political_philosophy", "philosophy", "Study of government and justice"),
    ("philosophy_of_mind", "philosophy", "Study of consciousness"),
    ("philosophy_of_science", "philosophy", "Study of scientific method"),
    ("philosophy_of_language", "philosophy", "Study of meaning"),
    # Epistemology
    ("knowledge", "abstract", "Justified true belief"),
    ("belief", "abstract", "Mental state of acceptance"),
    ("truth", "abstract", "Correspondence with reality"),
    ("justification", "abstract", "Reason for belief"),
    ("rationalism", "theory", "Knowledge from reason"),
    ("empiricism", "theory", "Knowledge from experience"),
    ("skepticism", "theory", "Doubt about knowledge"),
    ("foundationalism", "theory", "Basic beliefs support others"),
    ("coherentism", "theory", "Beliefs support each other"),
    ("pragmatism", "theory", "Truth is what works"),
    ("a_priori", "knowledge", "Knowledge independent of experience"),
    ("a_posteriori", "knowledge", "Knowledge from experience"),
    # Metaphysics
    ("being", "abstract", "Existence itself"),
    ("substance", "abstract", "Fundamental stuff of reality"),
    ("essence", "abstract", "What makes a thing what it is"),
    ("existence", "abstract", "The state of being"),
    ("causation", "abstract", "Cause and effect"),
    ("materialism", "theory", "Only matter exists"),
    ("physicalism", "materialism", "Only physical things exist"),
    ("idealism", "theory", "Mind/ideas are fundamental"),
    ("dualism", "theory", "Mind and matter both exist"),
    ("monism", "theory", "Only one substance exists"),
    ("determinism", "theory", "Everything is caused"),
    ("free_will", "abstract", "Ability to choose"),
    ("compatibilism", "theory", "Free will compatible with determinism"),
    # Ethics
    ("good", "abstract", "Moral value"),
    ("evil", "abstract", "Moral disvalue"),
    ("virtue", "abstract", "Good character trait"),
    ("vice", "abstract", "Bad character trait"),
    ("duty", "abstract", "Moral obligation"),
    ("rights", "abstract", "Moral entitlements"),
    ("justice", "abstract", "Fair distribution"),
    ("consequentialism", "theory", "Actions judged by outcomes"),
    ("utilitarianism", "consequentialism", "Maximize happiness"),
    ("deontology", "theory", "Actions judged by rules"),
    ("kantian_ethics", "deontology", "Categorical imperative"),
    ("virtue_ethics", "theory", "Character-based ethics"),
    ("care_ethics", "theory", "Relationships and care"),
    ("moral_relativism", "theory", "Morality is relative"),
    ("moral_realism", "theory", "Objective moral facts exist"),
    # Logic
    ("argument", "abstract", "Premises supporting conclusion"),
    ("premise", "abstract", "Starting assumption"),
    ("conclusion", "abstract", "What follows from premises"),
    ("validity", "abstract", "Conclusion follows from premises"),
    ("soundness", "abstract", "Valid with true premises"),
    ("fallacy", "abstract", "Error in reasoning"),
    ("propositional_logic", "logic", "Logic of propositions"),
    ("predicate_logic", "logic", "Logic with quantifiers"),
    ("modal_logic", "logic", "Logic of possibility/necessity"),
    ("ad_hominem", "fallacy", "Attacking the person"),
    ("straw_man", "fallacy", "Misrepresenting argument"),
    ("false_dilemma", "fallacy", "False either/or"),
    ("circular_reasoning", "fallacy", "Conclusion in premises"),
    # Political philosophy
    ("state", "abstract", "Political organization"),
    ("liberty", "abstract", "Freedom from constraint"),
    ("equality", "abstract", "Same status/treatment"),
    ("authority", "abstract", "Right to command"),
    ("liberalism", "theory", "Individual rights focus"),
    ("conservatism", "theory", "Traditional institutions"),
    ("socialism", "theory", "Social ownership"),
    ("marxism", "socialism", "Class struggle, communism"),
    ("libertarianism", "theory", "Minimal state"),
    ("anarchism", "theory", "No state"),
    ("social_contract", "theory", "Government from agreement"),
    # Philosophy of mind
    ("consciousness", "abstract", "Subjective experience"),
    ("qualia", "abstract", "Subjective quality of experience"),
    ("intentionality", "abstract", "Aboutness of mental states"),
    ("substance_dualism", "dualism", "Mind and body separate"),
    ("identity_theory", "physicalism", "Mental states are brain states"),
    ("functionalism", "theory", "Mind as functional states"),
    ("panpsychism", "theory", "Consciousness is fundamental"),
    # Schools
    ("ancient_philosophy", "philosophy", "Greek and Roman"),
    ("platonism", "ancient_philosophy", "Plato's philosophy"),
    ("aristotelianism", "ancient_philosophy", "Aristotle's philosophy"),
    ("stoicism", "ancient_philosophy", "Virtue and acceptance"),
    ("epicureanism", "ancient_philosophy", "Pleasure as highest good"),
    ("medieval_philosophy", "philosophy", "Middle Ages"),
    ("scholasticism", "medieval_philosophy", "Systematic theology"),
    ("modern_philosophy", "philosophy", "17th-19th century"),
    ("cartesianism", "modern_philosophy", "Descartes's philosophy"),
    ("british_empiricism", "modern_philosophy", "Locke, Berkeley, Hume"),
    ("german_idealism", "modern_philosophy", "Kant, Hegel"),
    ("contemporary_philosophy", "philosophy", "20th-21st century"),
    ("analytic_philosophy", "contemporary_philosophy", "Logic and language"),
    ("continental_philosophy", "contemporary_philosophy", "European tradition"),
    ("phenomenology", "continental_philosophy", "Study of experience"),
    ("existentialism", "continental_philosophy", "Existence precedes essence"),
    ("postmodernism", "contemporary_philosophy", "Skepticism of grand narratives"),
    ("logical_positivism", "analytic_philosophy", "Verification principle"),
    # Philosophers
    ("philosopher", "person", "One who philosophizes"),
    ("socrates", "philosopher", "Greek, dialectic method"),
    ("plato", "philosopher", "Forms, Republic"),
    ("aristotle", "philosopher", "Logic, virtue ethics"),
    ("epicurus", "philosopher", "Pleasure and atoms"),
    ("marcus_aurelius", "philosopher", "Stoic emperor"),
    ("augustine", "philosopher", "Christian Platonist"),
    ("aquinas", "philosopher", "Scholastic theologian"),
    ("descartes", "philosopher", "Cogito ergo sum"),
    ("spinoza", "philosopher", "Monist rationalist"),
    ("leibniz", "philosopher", "Monads"),
    ("locke", "philosopher", "Empiricist, tabula rasa"),
    ("berkeley", "philosopher", "Idealist"),
    ("hume", "philosopher", "Skeptical empiricist"),
    ("kant", "philosopher", "Transcendental idealism"),
    ("hegel", "philosopher", "Absolute idealism"),
    ("marx", "philosopher", "Historical materialism"),
    ("nietzsche", "philosopher", "Will to power"),
    ("kierkegaard", "philosopher", "Existentialist"),
    ("mill", "philosopher", "Utilitarianism"),
    ("bentham", "philosopher", "Utilitarianism founder"),
    ("husserl", "philosopher", "Phenomenology"),
    ("heidegger", "philosopher", "Being and Time"),
    ("sartre", "philosopher", "Existentialism"),
    ("camus", "philosopher", "Absurdism"),
    ("beauvoir", "philosopher", "Feminist existentialist"),
    ("wittgenstein", "philosopher", "Language games"),
    ("russell", "philosopher", "Logic"),
    ("rawls", "philosopher", "Justice as fairness"),
    ("foucault", "philosopher", "Power/knowledge"),
    ("derrida", "philosopher", "Deconstruction"),
]

PHIL_FACTS = [
    # Epistemology
    ("knowledge", "defined_as", "justified_true_belief"),
    ("rationalism", "claims", "reason_source_of_knowledge"),
    ("empiricism", "claims", "experience_source_of_knowledge"),
    ("descartes", "advocated", "rationalism"),
    ("locke", "advocated", "empiricism"),
    ("hume", "advocated", "empiricism"),
    ("kant", "synthesized", "rationalism_and_empiricism"),
    # Metaphysics
    ("materialism", "claims", "only_matter_exists"),
    ("idealism", "claims", "mind_is_fundamental"),
    ("dualism", "claims", "mind_and_matter_both_exist"),
    ("descartes", "advocated", "dualism"),
    ("spinoza", "advocated", "monism"),
    # Ethics
    ("utilitarianism", "maximizes", "happiness"),
    ("utilitarianism", "judges_by", "consequences"),
    ("deontology", "judges_by", "rules"),
    ("virtue_ethics", "focuses_on", "character"),
    ("bentham", "founded", "utilitarianism"),
    ("mill", "developed", "utilitarianism"),
    ("kant", "founded", "deontology"),
    ("aristotle", "founded", "virtue_ethics"),
    ("categorical_imperative", "proposed_by", "kant"),
    # Logic
    ("aristotle", "founded", "formal_logic"),
    ("valid_argument", "has", "true_conclusion_if_premises_true"),
    ("sound_argument", "is", "valid_with_true_premises"),
    # Political
    ("social_contract", "proposed_by", "hobbes"),
    ("social_contract", "proposed_by", "locke"),
    ("social_contract", "proposed_by", "rousseau"),
    ("rawls", "proposed", "veil_of_ignorance"),
    ("marx", "advocated", "communism"),
    # Mind
    ("descartes", "proposed", "mind_body_dualism"),
    ("functionalism", "defines_mind_by", "functional_role"),
    ("hard_problem", "concerns", "consciousness"),
    # Schools
    ("platonism", "proposed", "theory_of_forms"),
    ("existentialism", "claims", "existence_precedes_essence"),
    ("sartre", "advocated", "existentialism"),
    ("phenomenology", "founded_by", "husserl"),
    # Key arguments
    ("cogito", "proposed_by", "descartes"),
    ("cogito", "states", "i_think_therefore_i_am"),
    ("ockhams_razor", "states", "simplest_explanation_best"),
    ("allegory_of_cave", "proposed_by", "plato"),
    # Works
    ("republic", "written_by", "plato"),
    ("nicomachean_ethics", "written_by", "aristotle"),
    ("meditations", "written_by", "descartes"),
    ("leviathan", "written_by", "hobbes"),
    ("critique_of_pure_reason", "written_by", "kant"),
    ("being_and_time", "written_by", "heidegger"),
    ("being_and_nothingness", "written_by", "sartre"),
]


def load_philosophy_into_core(core, agent_id: str = None) -> int:
    if agent_id is None:
        phil_agent = core.register_agent(
            "philosophy_loader", domain="philosophy", can_verify=True
        )
        agent_id = phil_agent.agent_id

    count = 0

    print(f"  Loading {len(PHIL_CATEGORIES)} philosophy categories...")
    for name, parent, description in PHIL_CATEGORIES:
        if core.ontology:
            from dorian_ontology import Category

            parent_cat = core.ontology.categories.get(parent)
            parent_level = parent_cat.level if parent_cat else 3
            core.ontology._add_category(
                Category(
                    name=name,
                    description=description,
                    parent=parent,
                    domain="philosophy",
                    level=parent_level + 1,
                )
            )
        result = core.write(
            name,
            "subtype_of",
            parent,
            agent_id,
            source="philosophy_ontology",
            check_contradictions=False,
        )
        if result.success:
            count += 1

    print(f"  Loading {len(PHIL_FACTS)} philosophy facts...")
    for s, p, o in PHIL_FACTS:
        result = core.write(
            s, p, o, agent_id, source="philosophy_knowledge", check_contradictions=False
        )
        if result.success:
            count += 1

    print(f"  Total: {count} philosophy facts loaded")
    return count


if __name__ == "__main__":
    print(f"Philosophy: {len(PHIL_CATEGORIES)} categories, {len(PHIL_FACTS)} facts")
