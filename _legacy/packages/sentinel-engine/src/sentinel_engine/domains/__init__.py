"""
Domain Knowledge for Dorian

Pre-built knowledge bases for various academic and technical domains.

Available Domains:
- math: Mathematical concepts, theorems, operations (299 facts)
- physics: Physical laws, particles, forces (378 facts)
- chemistry: Elements, reactions, compounds (442 facts)
- biology: Organisms, genetics, ecology (366 facts)
- cs: Computer science, algorithms, data structures (777 facts)
- economics: Markets, finance, monetary policy (286 facts)
- philosophy: Ethics, logic, epistemology (177 facts)
- web: Web technologies, programming, protocols

Example:
    from domains.math import load_math_knowledge
    from dorian.core import DorianCore

    dorian = DorianCore()
    count = load_math_knowledge(dorian)
    print(f"Loaded {count} math facts")
"""

from .biology import load_biology_knowledge
from .chemistry import load_chemistry_knowledge
from .cs import load_cs_knowledge
from .economics import load_economics_knowledge
from .math import load_math_knowledge
from .philosophy import load_philosophy_knowledge
from .physics import load_physics_knowledge
from .web import load_web_knowledge

__all__ = [
    "load_math_knowledge",
    "load_physics_knowledge",
    "load_chemistry_knowledge",
    "load_biology_knowledge",
    "load_cs_knowledge",
    "load_economics_knowledge",
    "load_philosophy_knowledge",
    "load_web_knowledge",
]


def load_all_domains(dorian) -> int:
    """Load all domain knowledge into Dorian."""
    total = 0
    total += load_math_knowledge(dorian)
    total += load_physics_knowledge(dorian)
    total += load_chemistry_knowledge(dorian)
    total += load_biology_knowledge(dorian)
    total += load_cs_knowledge(dorian)
    total += load_economics_knowledge(dorian)
    total += load_philosophy_knowledge(dorian)
    total += load_web_knowledge(dorian)
    return total
