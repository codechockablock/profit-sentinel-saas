-- Migration: 009_add_analysis_stats_refresh
-- Description: Schedule periodic refresh for daily_analysis_stats materialized view.
--              The refresh function was defined in 002 but never triggered.
-- Created: 2026-02-06

-- Enable pg_cron extension (available on Supabase by default)
CREATE EXTENSION IF NOT EXISTS pg_cron;

-- Schedule concurrent refresh every 15 minutes.
-- The unique index idx_daily_analysis_stats_date (from 002) is required
-- for REFRESH CONCURRENTLY.
SELECT cron.schedule(
    'refresh-daily-analysis-stats',
    '*/15 * * * *',
    $$REFRESH MATERIALIZED VIEW CONCURRENTLY daily_analysis_stats$$
);
