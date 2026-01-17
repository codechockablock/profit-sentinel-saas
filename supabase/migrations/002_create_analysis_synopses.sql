-- Migration: 002_create_analysis_synopses
-- Description: Stores aggregate statistics from analyses (no PII/SKUs)
-- Created: 2026-01-17

-- Analysis synopses - aggregate stats only, no PII
CREATE TABLE IF NOT EXISTS analysis_synopses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Link to email signup (optional - for unlocked analyses)
    email_signup_id UUID REFERENCES email_signups(id) ON DELETE SET NULL,

    -- File metadata (no filename stored - privacy)
    file_hash TEXT NOT NULL,  -- SHA256 hash of file for dedup
    file_row_count INTEGER NOT NULL,
    file_column_count INTEGER,

    -- Aggregate detection counts (no SKUs)
    detection_counts JSONB NOT NULL DEFAULT '{}',
    -- Example: {"low_stock": 42, "high_margin_leak": 15, ...}

    -- Impact estimates (aggregated)
    total_impact_estimate_low DECIMAL(12,2),
    total_impact_estimate_high DECIMAL(12,2),
    currency TEXT DEFAULT 'USD',

    -- Dataset statistics (aggregated)
    dataset_stats JSONB NOT NULL DEFAULT '{}',
    -- Example: {"avg_margin": 0.32, "avg_qty": 45.2, "categories": 12}

    -- Processing metadata
    dimensions_used INTEGER NOT NULL DEFAULT 8192,
    processing_time_seconds DECIMAL(6,2),
    peak_memory_mb INTEGER,

    -- Client info
    client_timezone TEXT,
    client_locale TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT analysis_synopses_file_hash_check CHECK (LENGTH(file_hash) = 64)
);

-- Index for file deduplication
CREATE INDEX IF NOT EXISTS idx_analysis_synopses_file_hash ON analysis_synopses(file_hash);

-- Index for email lookup
CREATE INDEX IF NOT EXISTS idx_analysis_synopses_email ON analysis_synopses(email_signup_id)
    WHERE email_signup_id IS NOT NULL;

-- Index for created_at (analytics)
CREATE INDEX IF NOT EXISTS idx_analysis_synopses_created_at ON analysis_synopses(created_at DESC);

-- Enable Row Level Security
ALTER TABLE analysis_synopses ENABLE ROW LEVEL SECURITY;

-- Policy: Service role full access
CREATE POLICY "Service role full access" ON analysis_synopses
    FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

-- Policy: Anon can insert (for tracking)
CREATE POLICY "Anon can insert synopses" ON analysis_synopses
    FOR INSERT
    TO anon
    WITH CHECK (TRUE);

-- Comments
COMMENT ON TABLE analysis_synopses IS 'Aggregate statistics from file analyses - NO PII or SKUs stored';
COMMENT ON COLUMN analysis_synopses.file_hash IS 'SHA256 hash of uploaded file for deduplication';
COMMENT ON COLUMN analysis_synopses.detection_counts IS 'JSON object with primitive -> count mapping';
COMMENT ON COLUMN analysis_synopses.dataset_stats IS 'Aggregated statistics: avg_margin, avg_qty, category_count';

-- Materialized view for daily analytics (refreshed hourly)
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_analysis_stats AS
SELECT
    DATE(created_at) AS analysis_date,
    COUNT(*) AS analysis_count,
    AVG(file_row_count) AS avg_rows,
    AVG(processing_time_seconds) AS avg_processing_time,
    SUM((detection_counts->>'low_stock')::integer) AS total_low_stock,
    SUM((detection_counts->>'high_margin_leak')::integer) AS total_margin_leaks,
    SUM((detection_counts->>'negative_inventory')::integer) AS total_negative_inv,
    SUM(total_impact_estimate_low) AS total_impact_low,
    SUM(total_impact_estimate_high) AS total_impact_high
FROM analysis_synopses
WHERE created_at >= NOW() - INTERVAL '90 days'
GROUP BY DATE(created_at)
ORDER BY analysis_date DESC;

-- Unique index required for REFRESH CONCURRENTLY
CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_analysis_stats_date
    ON daily_analysis_stats(analysis_date);

-- Refresh function
CREATE OR REPLACE FUNCTION refresh_daily_analysis_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- Refresh every 100 inserts or every hour (handled by pg_cron)
    REFRESH MATERIALIZED VIEW CONCURRENTLY daily_analysis_stats;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

COMMENT ON MATERIALIZED VIEW daily_analysis_stats IS 'Daily aggregated analysis statistics for dashboards';
