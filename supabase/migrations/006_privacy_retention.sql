-- Migration: 006_privacy_retention
-- Description: Add data retention policies, auto-cleanup functions, and pg_cron schedules
-- Created: 2026-01-25
-- Security Audit Items: H3-H4, H8, M11

-- =============================================================================
-- ENABLE PG_CRON EXTENSION
-- =============================================================================
-- Note: pg_cron must be enabled in Supabase Dashboard → Database → Extensions
-- This creates the extension if it exists, otherwise it will fail gracefully

CREATE EXTENSION IF NOT EXISTS pg_cron;

-- =============================================================================
-- EMAIL SIGNUP RETENTION POLICY
-- =============================================================================

-- Add last_activity_at column to track user engagement
ALTER TABLE email_signups
ADD COLUMN IF NOT EXISTS last_activity_at TIMESTAMPTZ DEFAULT NOW();

-- Add unsubscribed_at column for opt-out tracking
ALTER TABLE email_signups
ADD COLUMN IF NOT EXISTS unsubscribed_at TIMESTAMPTZ;

-- Add deletion_requested_at for GDPR/CCPA tracking
ALTER TABLE email_signups
ADD COLUMN IF NOT EXISTS deletion_requested_at TIMESTAMPTZ;

-- Index for retention cleanup queries
CREATE INDEX IF NOT EXISTS idx_email_signups_last_activity
    ON email_signups(last_activity_at)
    WHERE unsubscribed_at IS NULL AND deletion_requested_at IS NULL;

-- Function to clean up stale email signups (2 years of inactivity)
CREATE OR REPLACE FUNCTION public.cleanup_stale_email_signups()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    WITH deleted AS (
        DELETE FROM email_signups
        WHERE (
            -- Delete if unsubscribed more than 30 days ago
            (unsubscribed_at IS NOT NULL AND unsubscribed_at < NOW() - INTERVAL '30 days')
            OR
            -- Delete if deletion was requested
            (deletion_requested_at IS NOT NULL)
            OR
            -- Delete if no activity for 2 years
            (last_activity_at < NOW() - INTERVAL '2 years' AND unsubscribed_at IS NULL)
        )
        RETURNING id
    )
    SELECT COUNT(*) INTO deleted_count FROM deleted;

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

COMMENT ON FUNCTION cleanup_stale_email_signups IS
    'Delete email signups that are unsubscribed (30d), deletion-requested, or inactive (2y). '
    'Schedule via pg_cron: SELECT cron.schedule(''cleanup-emails'', ''0 2 * * *'', $$SELECT cleanup_stale_email_signups()$$);';

-- =============================================================================
-- TECHNICAL METADATA CLEANUP (IP, User Agent - 7 days)
-- =============================================================================

-- Function to anonymize technical metadata older than 7 days
CREATE OR REPLACE FUNCTION public.anonymize_technical_metadata()
RETURNS INTEGER AS $$
DECLARE
    anonymized_count INTEGER;
BEGIN
    WITH anonymized AS (
        UPDATE email_signups
        SET
            ip_address = NULL,
            user_agent = NULL,
            updated_at = NOW()
        WHERE created_at < NOW() - INTERVAL '7 days'
          AND (ip_address IS NOT NULL OR user_agent IS NOT NULL)
        RETURNING id
    )
    SELECT COUNT(*) INTO anonymized_count FROM anonymized;

    RETURN anonymized_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

COMMENT ON FUNCTION anonymize_technical_metadata IS
    'Anonymize IP and User Agent after 7 days per privacy policy. '
    'Schedule via pg_cron: SELECT cron.schedule(''anonymize-metadata'', ''0 3 * * *'', $$SELECT anonymize_technical_metadata()$$);';

-- =============================================================================
-- UNSUBSCRIBE FUNCTION
-- =============================================================================

-- Function to handle unsubscribe requests
CREATE OR REPLACE FUNCTION public.unsubscribe_email(p_email TEXT)
RETURNS BOOLEAN AS $$
DECLARE
    updated_count INTEGER;
BEGIN
    UPDATE email_signups
    SET
        unsubscribed_at = NOW(),
        marketing_consent = FALSE,
        updated_at = NOW()
    WHERE LOWER(email) = LOWER(p_email)
      AND unsubscribed_at IS NULL;

    GET DIAGNOSTICS updated_count = ROW_COUNT;

    RETURN updated_count > 0;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

COMMENT ON FUNCTION unsubscribe_email IS 'Mark email as unsubscribed. Called from unsubscribe endpoint.';

-- =============================================================================
-- DELETION REQUEST FUNCTION (GDPR/CCPA)
-- =============================================================================

-- Function to process data deletion request
CREATE OR REPLACE FUNCTION public.process_deletion_request(p_email TEXT)
RETURNS TABLE (
    email_signups_deleted INTEGER,
    user_profiles_deleted INTEGER,
    total_deleted INTEGER
) AS $$
DECLARE
    v_email_count INTEGER := 0;
    v_profile_count INTEGER := 0;
BEGIN
    -- Delete from email_signups
    WITH deleted AS (
        DELETE FROM email_signups
        WHERE LOWER(email) = LOWER(p_email)
        RETURNING id
    )
    SELECT COUNT(*) INTO v_email_count FROM deleted;

    -- Delete from user_profiles (will cascade to auth.users if FK exists)
    WITH deleted AS (
        DELETE FROM user_profiles
        WHERE LOWER(email) = LOWER(p_email)
        RETURNING id
    )
    SELECT COUNT(*) INTO v_profile_count FROM deleted;

    RETURN QUERY SELECT
        v_email_count,
        v_profile_count,
        v_email_count + v_profile_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

COMMENT ON FUNCTION process_deletion_request IS
    'Process GDPR/CCPA data deletion request. Deletes all data for an email address.';

-- =============================================================================
-- UPDATE ACTIVITY FUNCTION
-- =============================================================================

-- Function to update last activity (call when user interacts)
CREATE OR REPLACE FUNCTION public.update_email_activity(p_email TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    UPDATE email_signups
    SET last_activity_at = NOW(),
        updated_at = NOW()
    WHERE LOWER(email) = LOWER(p_email);

    RETURN FOUND;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- =============================================================================
-- SCHEDULE PG_CRON JOBS
-- =============================================================================

-- Note: These need to be run manually in Supabase SQL Editor after enabling pg_cron
-- The DO block will attempt to schedule them if pg_cron is available

DO $$
BEGIN
    -- Only schedule if pg_cron is available
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_cron') THEN
        -- Schedule trial expiration (hourly)
        PERFORM cron.schedule(
            'expire-trials',
            '0 * * * *',  -- Every hour
            $$SELECT expire_stale_trials()$$
        );

        -- Schedule monthly analysis count reset (1st of month at midnight UTC)
        PERFORM cron.schedule(
            'reset-analysis-counts',
            '0 0 1 * *',  -- 1st of each month at 00:00 UTC
            $$SELECT reset_monthly_analysis_counts()$$
        );

        -- Schedule email signup cleanup (daily at 2am UTC)
        PERFORM cron.schedule(
            'cleanup-stale-emails',
            '0 2 * * *',  -- Daily at 02:00 UTC
            $$SELECT cleanup_stale_email_signups()$$
        );

        -- Schedule technical metadata anonymization (daily at 3am UTC)
        PERFORM cron.schedule(
            'anonymize-metadata',
            '0 3 * * *',  -- Daily at 03:00 UTC
            $$SELECT anonymize_technical_metadata()$$
        );

        RAISE NOTICE 'pg_cron jobs scheduled successfully';
    ELSE
        RAISE NOTICE 'pg_cron extension not available. Please enable it in Supabase Dashboard and run the scheduling commands manually.';
    END IF;
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Could not schedule pg_cron jobs: %. Please run scheduling commands manually.', SQLERRM;
END $$;

-- =============================================================================
-- GRANT PERMISSIONS
-- =============================================================================

GRANT EXECUTE ON FUNCTION cleanup_stale_email_signups() TO service_role;
GRANT EXECUTE ON FUNCTION anonymize_technical_metadata() TO service_role;
GRANT EXECUTE ON FUNCTION unsubscribe_email(TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION process_deletion_request(TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION update_email_activity(TEXT) TO service_role;

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON COLUMN email_signups.last_activity_at IS 'Last time user interacted (for 2-year retention)';
COMMENT ON COLUMN email_signups.unsubscribed_at IS 'When user unsubscribed (delete after 30 days)';
COMMENT ON COLUMN email_signups.deletion_requested_at IS 'When GDPR/CCPA deletion was requested';

-- =============================================================================
-- MANUAL PG_CRON SCHEDULING (Run if auto-scheduling failed)
-- =============================================================================

-- If the above DO block failed, run these manually in Supabase SQL Editor:
--
-- SELECT cron.schedule('expire-trials', '0 * * * *', $$SELECT expire_stale_trials()$$);
-- SELECT cron.schedule('reset-analysis-counts', '0 0 1 * *', $$SELECT reset_monthly_analysis_counts()$$);
-- SELECT cron.schedule('cleanup-stale-emails', '0 2 * * *', $$SELECT cleanup_stale_email_signups()$$);
-- SELECT cron.schedule('anonymize-metadata', '0 3 * * *', $$SELECT anonymize_technical_metadata()$$);
--
-- To verify scheduled jobs:
-- SELECT * FROM cron.job;
