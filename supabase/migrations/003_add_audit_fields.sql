-- Migration: 003_add_audit_fields
-- Description: Add v3.3 audit trail fields for reproducibility
-- Created: 2026-01-17

-- Add seeding summary for reproducibility audit
ALTER TABLE analysis_synopses
    ADD COLUMN IF NOT EXISTS seeding_summary JSONB DEFAULT '{}';
-- Example: {"master_seed": "eb19443f7da570d1", "total_entities": 10057, ...}

-- Add top leaks per primitive (SKU + score for evidence chain)
ALTER TABLE analysis_synopses
    ADD COLUMN IF NOT EXISTS top_leaks_by_primitive JSONB DEFAULT '{}';
-- Example: {"low_stock": [{"sku": "ABC123", "score": 0.92}, ...], ...}

-- Add engine version for backward compatibility tracking
ALTER TABLE analysis_synopses
    ADD COLUMN IF NOT EXISTS engine_version TEXT DEFAULT '3.3';

-- Add codebook size used (affects reproducibility)
ALTER TABLE analysis_synopses
    ADD COLUMN IF NOT EXISTS codebook_size INTEGER;

-- Comments for new columns
COMMENT ON COLUMN analysis_synopses.seeding_summary IS 'v3.3: Deterministic seeding summary - master_seed hash, entity counts';
COMMENT ON COLUMN analysis_synopses.top_leaks_by_primitive IS 'v3.3: Top 10 leaked SKUs per primitive with similarity scores';
COMMENT ON COLUMN analysis_synopses.engine_version IS 'Sentinel engine version used for analysis';
COMMENT ON COLUMN analysis_synopses.codebook_size IS 'VSA codebook size used (affects vector assignments)';

-- Index for engine version (useful for migration/debugging)
CREATE INDEX IF NOT EXISTS idx_analysis_synopses_engine_version
    ON analysis_synopses(engine_version);

-- Index on seeding master_seed for reproducibility lookups
CREATE INDEX IF NOT EXISTS idx_analysis_synopses_master_seed
    ON analysis_synopses((seeding_summary->>'master_seed'))
    WHERE seeding_summary->>'master_seed' IS NOT NULL;

-- Update materialized view to include new audit data
DROP MATERIALIZED VIEW IF EXISTS daily_analysis_stats;

CREATE MATERIALIZED VIEW daily_analysis_stats AS
SELECT
    DATE(created_at) AS analysis_date,
    COUNT(*) AS analysis_count,
    AVG(file_row_count) AS avg_rows,
    AVG(processing_time_seconds) AS avg_processing_time,
    SUM((detection_counts->>'low_stock')::integer) AS total_low_stock,
    SUM((detection_counts->>'high_margin_leak')::integer) AS total_margin_leaks,
    SUM((detection_counts->>'negative_inventory')::integer) AS total_negative_inv,
    SUM(total_impact_estimate_low) AS total_impact_low,
    SUM(total_impact_estimate_high) AS total_impact_high,
    -- v3.3: Engine version distribution
    jsonb_object_agg(
        COALESCE(engine_version, 'unknown'),
        COUNT(*) FILTER (WHERE engine_version IS NOT NULL)
    ) AS engine_version_counts,
    -- v3.3: Average entities seeded
    AVG((seeding_summary->>'total_entities')::integer) AS avg_entities_seeded
FROM analysis_synopses
WHERE created_at >= NOW() - INTERVAL '90 days'
GROUP BY DATE(created_at)
ORDER BY analysis_date DESC;

-- Recreate unique index for REFRESH CONCURRENTLY
CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_analysis_stats_date
    ON daily_analysis_stats(analysis_date);

COMMENT ON MATERIALIZED VIEW daily_analysis_stats IS 'Daily aggregated analysis statistics for dashboards (v3.3 with audit fields)';
