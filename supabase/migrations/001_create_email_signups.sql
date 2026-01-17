-- Migration: 001_create_email_signups
-- Description: Email capture table for trust hook unlocks
-- Created: 2026-01-17

-- Email signups for report unlocking
CREATE TABLE IF NOT EXISTS email_signups (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Core fields
    email TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'web_unlock',  -- web_unlock, landing_page, demo_request

    -- Context
    company_name TEXT,
    role TEXT,  -- owner, manager, analyst, developer
    store_count INTEGER,  -- Number of stores
    pos_system TEXT,  -- Paladin, Square, Lightspeed, etc.

    -- Marketing consent
    marketing_consent BOOLEAN DEFAULT FALSE,

    -- Metadata
    ip_address INET,
    user_agent TEXT,
    referrer TEXT,
    utm_source TEXT,
    utm_medium TEXT,
    utm_campaign TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT email_signups_email_unique UNIQUE (email, source)
);

-- Index for email lookups
CREATE INDEX IF NOT EXISTS idx_email_signups_email ON email_signups(email);

-- Index for source filtering
CREATE INDEX IF NOT EXISTS idx_email_signups_source ON email_signups(source);

-- Index for created_at (for analytics)
CREATE INDEX IF NOT EXISTS idx_email_signups_created_at ON email_signups(created_at DESC);

-- Enable Row Level Security
ALTER TABLE email_signups ENABLE ROW LEVEL SECURITY;

-- Policy: Service role can do everything
CREATE POLICY "Service role full access" ON email_signups
    FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

-- Policy: Anon can insert (for public signup forms)
CREATE POLICY "Anon can insert signups" ON email_signups
    FOR INSERT
    TO anon
    WITH CHECK (TRUE);

-- Updated at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_email_signups_updated_at
    BEFORE UPDATE ON email_signups
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Comments
COMMENT ON TABLE email_signups IS 'Captures email addresses from trust hook unlocks and landing page signups';
COMMENT ON COLUMN email_signups.source IS 'Where the signup came from: web_unlock, landing_page, demo_request';
COMMENT ON COLUMN email_signups.pos_system IS 'POS system used: Paladin, Square, Lightspeed, Clover, Shopify, etc.';
