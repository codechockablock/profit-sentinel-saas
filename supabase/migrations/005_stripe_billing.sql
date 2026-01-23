-- Migration: 005_stripe_billing
-- Description: Add Stripe billing fields for subscription management
-- Created: 2026-01-21

-- =============================================================================
-- ADD STRIPE BILLING COLUMNS TO USER_PROFILES
-- =============================================================================

-- Stripe customer and subscription tracking
ALTER TABLE user_profiles
ADD COLUMN IF NOT EXISTS stripe_customer_id TEXT,
ADD COLUMN IF NOT EXISTS stripe_subscription_id TEXT,
ADD COLUMN IF NOT EXISTS current_period_end TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS trial_starts_at TIMESTAMPTZ;

-- Rename trial_ends_at if it exists, otherwise create trial_expires_at
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'user_profiles' AND column_name = 'trial_ends_at'
    ) THEN
        ALTER TABLE user_profiles RENAME COLUMN trial_ends_at TO trial_expires_at;
    ELSE
        ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS trial_expires_at TIMESTAMPTZ;
    END IF;
END $$;

-- Update subscription_status to support all Stripe states
-- Values: trialing, active, past_due, canceled, expired, incomplete
COMMENT ON COLUMN user_profiles.subscription_status IS
    'Subscription status: trialing, active, past_due, canceled, expired, incomplete';

-- Create index for Stripe customer lookups
CREATE INDEX IF NOT EXISTS idx_user_profiles_stripe_customer
    ON user_profiles(stripe_customer_id) WHERE stripe_customer_id IS NOT NULL;

-- Create index for subscription status queries
CREATE INDEX IF NOT EXISTS idx_user_profiles_subscription_status
    ON user_profiles(subscription_status);

-- =============================================================================
-- UPDATE NEW USER TRIGGER TO SET TRIAL
-- =============================================================================

-- Drop and recreate the trigger function with trial setup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.user_profiles (
        id,
        email,
        subscription_tier,
        subscription_status,
        trial_starts_at,
        trial_expires_at
    )
    VALUES (
        NEW.id,
        NEW.email,
        'pro',  -- Full access during trial
        'trialing',
        NOW(),
        NOW() + INTERVAL '14 days'
    )
    ON CONFLICT (id) DO UPDATE SET
        -- If profile exists, still set up trial if not already set
        subscription_tier = CASE
            WHEN user_profiles.trial_starts_at IS NULL THEN 'pro'
            ELSE user_profiles.subscription_tier
        END,
        subscription_status = CASE
            WHEN user_profiles.trial_starts_at IS NULL THEN 'trialing'
            ELSE user_profiles.subscription_status
        END,
        trial_starts_at = CASE
            WHEN user_profiles.trial_starts_at IS NULL THEN NOW()
            ELSE user_profiles.trial_starts_at
        END,
        trial_expires_at = CASE
            WHEN user_profiles.trial_expires_at IS NULL THEN NOW() + INTERVAL '14 days'
            ELSE user_profiles.trial_expires_at
        END;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- =============================================================================
-- ACCESS CHECK FUNCTION
-- =============================================================================

-- Function to check if user has active access (trialing or subscribed)
CREATE OR REPLACE FUNCTION public.check_user_access(p_user_id UUID)
RETURNS TABLE (
    has_access BOOLEAN,
    access_reason TEXT,
    subscription_status TEXT,
    trial_days_left INTEGER,
    current_period_end TIMESTAMPTZ
) AS $$
DECLARE
    v_status TEXT;
    v_trial_expires TIMESTAMPTZ;
    v_period_end TIMESTAMPTZ;
    v_tier TEXT;
BEGIN
    -- Get user subscription data
    SELECT
        up.subscription_status,
        up.subscription_tier,
        up.trial_expires_at,
        up.current_period_end
    INTO v_status, v_tier, v_trial_expires, v_period_end
    FROM user_profiles up
    WHERE up.id = p_user_id;

    -- User not found
    IF v_status IS NULL THEN
        RETURN QUERY SELECT
            FALSE,
            'user_not_found'::TEXT,
            'none'::TEXT,
            NULL::INTEGER,
            NULL::TIMESTAMPTZ;
        RETURN;
    END IF;

    -- Active subscription
    IF v_status = 'active' THEN
        RETURN QUERY SELECT
            TRUE,
            'active_subscription'::TEXT,
            v_status,
            NULL::INTEGER,
            v_period_end;
        RETURN;
    END IF;

    -- Trialing - check if still within trial period
    IF v_status = 'trialing' THEN
        IF v_trial_expires > NOW() THEN
            RETURN QUERY SELECT
                TRUE,
                'active_trial'::TEXT,
                v_status,
                EXTRACT(DAY FROM (v_trial_expires - NOW()))::INTEGER,
                v_trial_expires;
            RETURN;
        ELSE
            -- Trial expired - update status
            UPDATE user_profiles
            SET subscription_status = 'expired',
                subscription_tier = 'free',
                updated_at = NOW()
            WHERE id = p_user_id;

            RETURN QUERY SELECT
                FALSE,
                'trial_expired'::TEXT,
                'expired'::TEXT,
                0::INTEGER,
                v_trial_expires;
            RETURN;
        END IF;
    END IF;

    -- Past due - still allow access but flag it
    IF v_status = 'past_due' THEN
        RETURN QUERY SELECT
            TRUE,  -- Allow access but payment needed
            'payment_past_due'::TEXT,
            v_status,
            NULL::INTEGER,
            v_period_end;
        RETURN;
    END IF;

    -- Canceled/Expired/Other - no access
    RETURN QUERY SELECT
        FALSE,
        'subscription_' || COALESCE(v_status, 'none'),
        COALESCE(v_status, 'none'),
        NULL::INTEGER,
        v_period_end;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- =============================================================================
-- TRIAL EXPIRATION CRON FUNCTION
-- =============================================================================

-- Function to expire old trials (call via pg_cron daily)
CREATE OR REPLACE FUNCTION public.expire_stale_trials()
RETURNS INTEGER AS $$
DECLARE
    expired_count INTEGER;
BEGIN
    WITH expired AS (
        UPDATE user_profiles
        SET subscription_status = 'expired',
            subscription_tier = 'free',
            updated_at = NOW()
        WHERE subscription_status = 'trialing'
          AND trial_expires_at < NOW()
        RETURNING id
    )
    SELECT COUNT(*) INTO expired_count FROM expired;

    RETURN expired_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

COMMENT ON FUNCTION expire_stale_trials IS
    'Expire trials past their end date. Schedule via pg_cron: SELECT cron.schedule(''expire-trials'', ''0 * * * *'', $$SELECT expire_stale_trials()$$);';

-- =============================================================================
-- SUBSCRIPTION UPDATE FUNCTION (for webhook handler)
-- =============================================================================

-- Function to update subscription from Stripe webhook data
CREATE OR REPLACE FUNCTION public.update_subscription(
    p_stripe_customer_id TEXT,
    p_stripe_subscription_id TEXT,
    p_status TEXT,
    p_current_period_end TIMESTAMPTZ
)
RETURNS UUID AS $$
DECLARE
    v_user_id UUID;
    v_tier TEXT;
BEGIN
    -- Determine tier based on status
    v_tier := CASE
        WHEN p_status IN ('active', 'trialing', 'past_due') THEN 'pro'
        ELSE 'free'
    END;

    -- Update user profile
    UPDATE user_profiles
    SET stripe_subscription_id = p_stripe_subscription_id,
        subscription_status = p_status,
        subscription_tier = v_tier,
        current_period_end = p_current_period_end,
        updated_at = NOW()
    WHERE stripe_customer_id = p_stripe_customer_id
    RETURNING id INTO v_user_id;

    RETURN v_user_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- =============================================================================
-- GRANT PERMISSIONS
-- =============================================================================

-- Allow service role to call these functions
GRANT EXECUTE ON FUNCTION check_user_access(UUID) TO service_role;
GRANT EXECUTE ON FUNCTION expire_stale_trials() TO service_role;
GRANT EXECUTE ON FUNCTION update_subscription(TEXT, TEXT, TEXT, TIMESTAMPTZ) TO service_role;

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON COLUMN user_profiles.stripe_customer_id IS 'Stripe customer ID (cus_xxx)';
COMMENT ON COLUMN user_profiles.stripe_subscription_id IS 'Active Stripe subscription ID (sub_xxx)';
COMMENT ON COLUMN user_profiles.current_period_end IS 'End of current billing period';
COMMENT ON COLUMN user_profiles.trial_starts_at IS 'When 14-day trial started';
COMMENT ON COLUMN user_profiles.trial_expires_at IS 'When trial expires (trial_starts_at + 14 days)';

-- =============================================================================
-- BACKFILL EXISTING USERS WITH TRIAL (if needed)
-- =============================================================================

-- Set up trial for existing users who don't have one
UPDATE user_profiles
SET trial_starts_at = created_at,
    trial_expires_at = created_at + INTERVAL '14 days',
    subscription_status = CASE
        WHEN created_at + INTERVAL '14 days' > NOW() THEN 'trialing'
        ELSE 'expired'
    END,
    subscription_tier = CASE
        WHEN created_at + INTERVAL '14 days' > NOW() THEN 'pro'
        ELSE 'free'
    END
WHERE trial_starts_at IS NULL
  AND subscription_status IS NULL;
