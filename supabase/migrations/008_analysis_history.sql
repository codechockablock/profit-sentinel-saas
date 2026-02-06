-- Migration: 008_analysis_history
-- Description: Add user_id, analysis label, and full result storage
--              to analysis_synopses for cross-report pattern detection.
-- Created: 2026-02-06

-- =========================================================================
-- 1. Add new columns to analysis_synopses
-- =========================================================================

-- Link analyses to authenticated users (nullable for anonymous analyses)
ALTER TABLE analysis_synopses
    ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL;

-- Human-readable label (auto-generated or user-editable)
ALTER TABLE analysis_synopses
    ADD COLUMN IF NOT EXISTS analysis_label TEXT;

-- Full analysis result (the complete JSON returned by /analysis/analyze)
-- Stored ONLY for authenticated users; anonymous results are NOT persisted.
ALTER TABLE analysis_synopses
    ADD COLUMN IF NOT EXISTS full_result JSONB;

-- Row count from the original upload (for quick display without parsing full_result)
-- file_row_count already exists, but let's add column_count if missing
-- (already added in 002 as file_column_count)

-- Track the original filename (user-facing label, not a path)
ALTER TABLE analysis_synopses
    ADD COLUMN IF NOT EXISTS original_filename TEXT;

-- =========================================================================
-- 2. Indexes for user-scoped queries
-- =========================================================================

-- Primary lookup: "show me my recent analyses"
CREATE INDEX IF NOT EXISTS idx_analysis_synopses_user_created
    ON analysis_synopses (user_id, created_at DESC)
    WHERE user_id IS NOT NULL;

-- Secondary: lookup by user + file hash (detect re-uploads)
CREATE INDEX IF NOT EXISTS idx_analysis_synopses_user_hash
    ON analysis_synopses (user_id, file_hash)
    WHERE user_id IS NOT NULL;

-- =========================================================================
-- 3. RLS policies for user access
-- =========================================================================

-- Users can read their own analyses
CREATE POLICY "Users can read own analyses" ON analysis_synopses
    FOR SELECT
    USING (auth.uid() = user_id);

-- Users can update their own analyses (e.g., rename label)
CREATE POLICY "Users can update own analyses" ON analysis_synopses
    FOR UPDATE
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- Users can delete their own analyses
CREATE POLICY "Users can delete own analyses" ON analysis_synopses
    FOR DELETE
    USING (auth.uid() = user_id);

-- =========================================================================
-- 4. Helper function: auto-generate analysis label
-- =========================================================================

CREATE OR REPLACE FUNCTION generate_analysis_label(
    p_filename TEXT,
    p_row_count INTEGER,
    p_total_issues INTEGER
)
RETURNS TEXT AS $$
BEGIN
    RETURN COALESCE(
        regexp_replace(p_filename, '\.(csv|xlsx|xls|tsv)$', '', 'i'),
        'Analysis'
    ) || ' — ' || p_row_count || ' rows, ' || p_total_issues || ' issues';
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION generate_analysis_label IS 'Auto-generate a human-readable label for an analysis';

-- =========================================================================
-- 5. Comments
-- =========================================================================

COMMENT ON COLUMN analysis_synopses.user_id IS 'FK to auth.users — NULL for anonymous analyses';
COMMENT ON COLUMN analysis_synopses.analysis_label IS 'Human-readable label, auto-generated or user-edited';
COMMENT ON COLUMN analysis_synopses.full_result IS 'Complete JSON result from /analysis/analyze (authenticated only)';
COMMENT ON COLUMN analysis_synopses.original_filename IS 'Original upload filename (display only, not a file path)';
