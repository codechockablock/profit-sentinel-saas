"""
Metrics endpoints - Resonator Validation Dashboard.

Exposes metrics for monitoring the VSA/HDC resonator and baseline detector
performance in production. Used for observability and debugging.

Metrics include:
- Engine availability and configuration
- Resonator calibration parameters
- Detection primitive status
- Recent analysis statistics
"""

import logging
from datetime import datetime

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory metrics store (reset on restart - use Redis for persistence)
_metrics_store: dict = {
    "analyses_total": 0,
    "analyses_success": 0,
    "analyses_failed": 0,
    "total_rows_processed": 0,
    "total_leaks_detected": 0,
    "last_analysis_time": None,
    "avg_analysis_duration_ms": 0,
    "primitive_detection_counts": {},
    "resonator_validations": {
        "total": 0,
        "converged": 0,
        "flagged": 0,
    },
}


def record_analysis_metrics(
    success: bool,
    rows_processed: int,
    leaks_detected: int,
    duration_ms: float,
    primitive_counts: dict[str, int],
    resonator_converged: int = 0,
    resonator_flagged: int = 0,
):
    """
    Record metrics from an analysis run.

    Called by AnalysisService after each analysis completes.
    """
    global _metrics_store

    _metrics_store["analyses_total"] += 1
    if success:
        _metrics_store["analyses_success"] += 1
    else:
        _metrics_store["analyses_failed"] += 1

    _metrics_store["total_rows_processed"] += rows_processed
    _metrics_store["total_leaks_detected"] += leaks_detected
    _metrics_store["last_analysis_time"] = datetime.utcnow().isoformat()

    # Update rolling average duration
    n = _metrics_store["analyses_total"]
    old_avg = _metrics_store["avg_analysis_duration_ms"]
    _metrics_store["avg_analysis_duration_ms"] = old_avg + (duration_ms - old_avg) / n

    # Update primitive counts
    for prim, count in primitive_counts.items():
        if prim not in _metrics_store["primitive_detection_counts"]:
            _metrics_store["primitive_detection_counts"][prim] = 0
        _metrics_store["primitive_detection_counts"][prim] += count

    # Update resonator stats
    _metrics_store["resonator_validations"]["total"] += (
        resonator_converged + resonator_flagged
    )
    _metrics_store["resonator_validations"]["converged"] += resonator_converged
    _metrics_store["resonator_validations"]["flagged"] += resonator_flagged


@router.get("/engine")
async def engine_status() -> dict:
    """
    Get sentinel engine status and configuration.

    Returns:
        Engine availability, version, and calibration parameters.
    """
    try:
        from sentinel_engine import (
            _DIAGNOSTIC_AVAILABLE,
            _DORIAN_AVAILABLE,
            DEFAULT_DIMENSIONS,
            DEFAULT_MAX_CODEBOOK_SIZE,
            __version__,
        )

        return {
            "status": "available",
            "version": __version__,
            "dorian_available": _DORIAN_AVAILABLE,
            "diagnostic_available": _DIAGNOSTIC_AVAILABLE,
            "configuration": {
                "dimensions": DEFAULT_DIMENSIONS,
                "max_codebook_size": DEFAULT_MAX_CODEBOOK_SIZE,
            },
            "calibration_version": "v5.0.0",
        }
    except ImportError as e:
        logger.warning(f"Sentinel engine not available: {e}")
        return {
            "status": "unavailable",
            "error": str(e),
        }


@router.get("/primitives")
async def primitive_metrics() -> dict:
    """
    Get detection primitive performance metrics.

    Returns:
        Per-primitive detection counts and validation benchmarks.
    """
    try:
        from sentinel_engine import LEAK_METADATA, get_all_primitives

        primitives = get_all_primitives()

        # Benchmark F1 scores from validation (v2.1.3)
        benchmark_f1 = {
            "negative_inventory": 1.0,
            "overstock": 1.0,
            "shrinkage_pattern": 0.969,
            "dead_item": 0.835,
            "high_margin_leak": 0.716,
            "low_stock": 0.71,
            "margin_erosion": 0.705,
            "price_discrepancy": 0.646,
            "zero_cost_anomaly": None,
            "negative_profit": None,
            "severe_inventory_deficit": None,
        }

        primitive_data = {}
        for p in primitives:
            metadata = LEAK_METADATA.get(p, {})
            primitive_data[p] = {
                "severity": metadata.get("severity", "unknown"),
                "category": metadata.get("category", "Unknown"),
                "detection_count": _metrics_store["primitive_detection_counts"].get(
                    p, 0
                ),
                "benchmark_f1": benchmark_f1.get(p, None),
            }

        return {
            "primitives": primitive_data,
            "total_primitives": len(primitives),
            "benchmark_avg_f1": 0.823,  # v2.1.3 calibrated
        }
    except ImportError:
        return {
            "primitives": {},
            "total_primitives": 0,
            "error": "Engine not available",
        }


@router.get("/resonator")
async def resonator_metrics() -> dict:
    """
    Get VSA/HDC resonator validation metrics.

    Returns:
        Resonator configuration, validation stats, and convergence rates.

    Note: Resonator is legacy (v2.x-v4.x). Dorian (v5.0) uses FAISS indexing.
    """
    total = _metrics_store["resonator_validations"]["total"]
    converged = _metrics_store["resonator_validations"]["converged"]
    flagged = _metrics_store["resonator_validations"]["flagged"]

    convergence_rate = converged / total if total > 0 else None

    return {
        "status": "legacy",
        "role": "sanity_checker",
        "note": "Resonator is legacy (v2.x-v4.x). Dorian v5.0 uses FAISS indexing.",
        "validation_stats": {
            "total_validations": total,
            "converged": converged,
            "flagged": flagged,
            "convergence_rate": convergence_rate,
        },
    }


@router.get("/resonator-legacy")
async def resonator_metrics_legacy() -> dict:
    """Legacy resonator metrics (deprecated)."""
    return {
        "status": "deprecated",
        "note": "Resonator is legacy. Dorian v5.0 uses FAISS indexing.",
    }


@router.get("/analysis")
async def analysis_metrics() -> dict:
    """
    Get analysis throughput and performance metrics.

    Returns:
        Analysis counts, success rates, and timing statistics.
    """
    total = _metrics_store["analyses_total"]
    success = _metrics_store["analyses_success"]
    failed = _metrics_store["analyses_failed"]

    success_rate = success / total if total > 0 else None

    return {
        "throughput": {
            "total_analyses": total,
            "successful": success,
            "failed": failed,
            "success_rate": success_rate,
        },
        "volume": {
            "total_rows_processed": _metrics_store["total_rows_processed"],
            "total_leaks_detected": _metrics_store["total_leaks_detected"],
        },
        "performance": {
            "avg_analysis_duration_ms": round(
                _metrics_store["avg_analysis_duration_ms"], 2
            ),
            "last_analysis": _metrics_store["last_analysis_time"],
        },
    }


@router.get("/dashboard")
async def dashboard() -> dict:
    """
    Get comprehensive metrics dashboard.

    Aggregates all metrics into a single response for dashboard rendering.
    """
    engine = await engine_status()
    primitives = await primitive_metrics()
    resonator = await resonator_metrics()
    analysis = await analysis_metrics()

    # Calculate health score (0-100)
    health_factors = []

    # Engine availability (40 points)
    if engine.get("status") == "available":
        health_factors.append(40)

    # Success rate (30 points)
    success_rate = analysis["throughput"].get("success_rate")
    if success_rate is not None:
        health_factors.append(success_rate * 30)
    elif analysis["throughput"]["total_analyses"] == 0:
        health_factors.append(30)  # No failures yet

    # Resonator convergence (30 points)
    convergence_rate = resonator.get("validation_stats", {}).get("convergence_rate")
    if convergence_rate is not None:
        health_factors.append(convergence_rate * 30)
    elif resonator.get("status") == "active":
        health_factors.append(30)  # No validations yet

    health_score = sum(health_factors)

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "health_score": round(health_score, 1),
        "health_status": (
            "healthy"
            if health_score >= 80
            else "degraded" if health_score >= 50 else "unhealthy"
        ),
        "engine": engine,
        "primitives": primitives,
        "resonator": resonator,
        "analysis": analysis,
    }


@router.post("/reset")
async def reset_metrics() -> dict:
    """
    Reset all metrics counters.

    Use for testing or when restarting metrics collection.
    """
    global _metrics_store

    _metrics_store = {
        "analyses_total": 0,
        "analyses_success": 0,
        "analyses_failed": 0,
        "total_rows_processed": 0,
        "total_leaks_detected": 0,
        "last_analysis_time": None,
        "avg_analysis_duration_ms": 0,
        "primitive_detection_counts": {},
        "resonator_validations": {
            "total": 0,
            "converged": 0,
            "flagged": 0,
        },
    }

    logger.info("Metrics store reset")

    return {
        "status": "reset",
        "timestamp": datetime.utcnow().isoformat(),
    }
