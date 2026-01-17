-- Migration: 003_create_user_profiles
-- Description: User profiles for authenticated users (post-MVP)
-- Created: 2026-01-17

-- User profiles - extends Supabase auth.users
CREATE TABLE IF NOT EXISTS user_profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,

    -- Basic info
    email TEXT NOT NULL,
    full_name TEXT,
    company_name TEXT,
    role TEXT,  -- owner, manager, analyst

    -- Business info
    store_count INTEGER DEFAULT 1,
    pos_system TEXT,
    industry TEXT,  -- hardware, grocery, apparel, etc.
    annual_revenue_range TEXT,  -- <1M, 1-5M, 5-20M, 20M+

    -- Subscription (placeholder for Phase 2)
    subscription_tier TEXT DEFAULT 'free',  -- free, pro, enterprise
    subscription_status TEXT DEFAULT 'active',
    trial_ends_at TIMESTAMPTZ,

    -- Usage tracking
    analyses_this_month INTEGER DEFAULT 0,
    total_analyses INTEGER DEFAULT 0,
    last_analysis_at TIMESTAMPTZ,

    -- Preferences
    preferred_currency TEXT DEFAULT 'USD',
    timezone TEXT DEFAULT 'America/New_York',
    email_notifications BOOLEAN DEFAULT TRUE,
    weekly_digest BOOLEAN DEFAULT FALSE,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT user_profiles_email_unique UNIQUE (email)
);

-- Index for email lookups
CREATE INDEX IF NOT EXISTS idx_user_profiles_email ON user_profiles(email);

-- Index for subscription filtering
CREATE INDEX IF NOT EXISTS idx_user_profiles_subscription ON user_profiles(subscription_tier, subscription_status);

-- Enable Row Level Security
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;

-- Policy: Users can read their own profile
CREATE POLICY "Users can read own profile" ON user_profiles
    FOR SELECT
    USING (auth.uid() = id);

-- Policy: Users can update their own profile
CREATE POLICY "Users can update own profile" ON user_profiles
    FOR UPDATE
    USING (auth.uid() = id)
    WITH CHECK (auth.uid() = id);

-- Policy: Service role full access
CREATE POLICY "Service role full access" ON user_profiles
    FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

-- Trigger to create profile on user signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.user_profiles (id, email)
    VALUES (NEW.id, NEW.email)
    ON CONFLICT (id) DO NOTHING;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create trigger if not exists
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_new_user();

-- Updated at trigger
CREATE TRIGGER update_user_profiles_updated_at
    BEFORE UPDATE ON user_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Comments
COMMENT ON TABLE user_profiles IS 'Extended user profiles for authenticated users';
COMMENT ON COLUMN user_profiles.subscription_tier IS 'free, pro, enterprise - for Phase 2 billing';
COMMENT ON COLUMN user_profiles.analyses_this_month IS 'Rate limiting: reset monthly';

-- Function to increment analysis count
CREATE OR REPLACE FUNCTION increment_analysis_count(user_id UUID)
RETURNS VOID AS $$
BEGIN
    UPDATE user_profiles
    SET
        analyses_this_month = analyses_this_month + 1,
        total_analyses = total_analyses + 1,
        last_analysis_at = NOW(),
        updated_at = NOW()
    WHERE id = user_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to reset monthly counts (call via pg_cron)
CREATE OR REPLACE FUNCTION reset_monthly_analysis_counts()
RETURNS VOID AS $$
BEGIN
    UPDATE user_profiles
    SET analyses_this_month = 0, updated_at = NOW();
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

COMMENT ON FUNCTION increment_analysis_count IS 'Increment user analysis count - call after each analysis';
COMMENT ON FUNCTION reset_monthly_analysis_counts IS 'Reset monthly counts - schedule via pg_cron on 1st of month';
