"""
Routing Module - Smart hot/cold path routing for hybrid VSA-LLM architecture.

Hot path (VSA): Sub-millisecond inference for known patterns
Cold path (LLM): Deep reasoning for ambiguous/novel cases

Benchmark results from validated research:
- Hot path: 0.003ms (target <50ms)
- Cold path: ~500ms
- Speedup: 5,059x
- 80%+ queries resolved on hot path

Reference: RESEARCH_SUMMARY.md - Hot/Cold Path Architecture
"""

from .smart_router import (
    AnalysisResult,
    ColdPathRequest,
    HotPathResult,
    RoutingDecision,
    SmartRouter,
    create_smart_router,
)

__all__ = [
    "SmartRouter",
    "RoutingDecision",
    "HotPathResult",
    "ColdPathRequest",
    "AnalysisResult",
    "create_smart_router",
]
