-- ===========================================================================
-- Migration 011: Eagle's Eye Executive View Schema
--
-- Adds organizational hierarchy (orgs → regions → stores), store snapshots,
-- agent briefings, and agent actions for the executive dashboard.
--
-- The old `stores` table (migration 010) remains untouched.
-- New `org_stores` table is the org-scoped replacement.
--
-- Run this in the Supabase SQL Editor.
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- 1. Organizations (a business entity like "Acme Hardware Group")
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    owner_user_id UUID NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE organizations ENABLE ROW LEVEL SECURITY;

-- RLS requires user_roles, so we create user_roles first (below),
-- then add policies after both tables exist.

-- ---------------------------------------------------------------------------
-- 2. User roles within an organization
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS user_roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('owner', 'regional_manager', 'store_manager', 'viewer')),
    scope_type TEXT NOT NULL CHECK (scope_type IN ('business', 'region', 'store')),
    scope_id UUID, -- NULL for business scope, region_id or store_id for narrower scope
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, org_id)
);

ALTER TABLE user_roles ENABLE ROW LEVEL SECURITY;

CREATE INDEX idx_user_roles_user_id ON user_roles(user_id);
CREATE INDEX idx_user_roles_org_id ON user_roles(org_id);

-- RLS: users can see their own roles
CREATE POLICY "Users can view their own roles"
    ON user_roles FOR SELECT
    USING (user_id = auth.uid());

-- RLS: owners can manage all roles in their org
CREATE POLICY "Owners can manage roles"
    ON user_roles FOR ALL
    USING (org_id IN (
        SELECT org_id FROM user_roles
        WHERE user_id = auth.uid() AND role = 'owner'
    ));

-- Now add org RLS policies (user_roles table exists)
CREATE POLICY "Org members can view their org"
    ON organizations FOR SELECT
    USING (id IN (
        SELECT org_id FROM user_roles WHERE user_id = auth.uid()
    ));

CREATE POLICY "Org owners can update"
    ON organizations FOR UPDATE
    USING (owner_user_id = auth.uid());

CREATE POLICY "Users can create orgs"
    ON organizations FOR INSERT
    WITH CHECK (owner_user_id = auth.uid());

-- ---------------------------------------------------------------------------
-- 3. Regions (groupings like "North Region", "South Region")
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS regions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(org_id, name)
);

ALTER TABLE regions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Org members can view regions"
    ON regions FOR SELECT
    USING (org_id IN (
        SELECT org_id FROM user_roles WHERE user_id = auth.uid()
    ));

CREATE POLICY "Org members can manage regions"
    ON regions FOR ALL
    USING (org_id IN (
        SELECT org_id FROM user_roles
        WHERE user_id = auth.uid()
        AND role IN ('owner', 'regional_manager')
    ));

CREATE INDEX idx_regions_org_id ON regions(org_id);

-- ---------------------------------------------------------------------------
-- 4. Org-scoped stores (replaces user-scoped `stores` table for eagle-eye)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS org_stores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    region_id UUID REFERENCES regions(id) ON DELETE SET NULL,
    name TEXT NOT NULL,
    address TEXT DEFAULT '',
    store_type TEXT DEFAULT 'hardware_store',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_upload_at TIMESTAMPTZ,
    item_count INTEGER DEFAULT 0,
    total_impact NUMERIC DEFAULT 0,
    exposure_trend NUMERIC DEFAULT 0,
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'inactive')),
    UNIQUE(org_id, name)
);

ALTER TABLE org_stores ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Org members can view stores"
    ON org_stores FOR SELECT
    USING (org_id IN (
        SELECT org_id FROM user_roles WHERE user_id = auth.uid()
    ));

CREATE POLICY "Owners and regional managers can manage stores"
    ON org_stores FOR ALL
    USING (org_id IN (
        SELECT org_id FROM user_roles
        WHERE user_id = auth.uid()
        AND role IN ('owner', 'regional_manager')
    ));

CREATE INDEX idx_org_stores_org_id ON org_stores(org_id);
CREATE INDEX idx_org_stores_region_id ON org_stores(region_id);

-- ---------------------------------------------------------------------------
-- 5. Store snapshots (updated after each analysis, used for eagle's eye view)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS store_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    store_id UUID NOT NULL REFERENCES org_stores(id) ON DELETE CASCADE,
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    snapshot_at TIMESTAMPTZ DEFAULT NOW(),
    item_count INTEGER DEFAULT 0,
    flagged_count INTEGER DEFAULT 0,
    total_impact_low NUMERIC DEFAULT 0,
    total_impact_high NUMERIC DEFAULT 0,
    dead_stock_count INTEGER DEFAULT 0,
    dead_stock_capital NUMERIC DEFAULT 0,
    margin_erosion_count INTEGER DEFAULT 0,
    margin_erosion_impact NUMERIC DEFAULT 0,
    shrinkage_count INTEGER DEFAULT 0,
    shrinkage_impact NUMERIC DEFAULT 0,
    stockout_risk_count INTEGER DEFAULT 0,
    overstock_count INTEGER DEFAULT 0,
    prediction_count INTEGER DEFAULT 0,
    critical_prediction_count INTEGER DEFAULT 0,
    pending_actions INTEGER DEFAULT 0,
    completed_actions INTEGER DEFAULT 0
);

ALTER TABLE store_snapshots ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Org members can view snapshots"
    ON store_snapshots FOR SELECT
    USING (org_id IN (
        SELECT org_id FROM user_roles WHERE user_id = auth.uid()
    ));

CREATE POLICY "System can insert snapshots"
    ON store_snapshots FOR INSERT
    WITH CHECK (org_id IN (
        SELECT org_id FROM user_roles WHERE user_id = auth.uid()
    ));

CREATE INDEX idx_store_snapshots_store_id ON store_snapshots(store_id);
CREATE INDEX idx_store_snapshots_org_id_time ON store_snapshots(org_id, snapshot_at DESC);

-- ---------------------------------------------------------------------------
-- 6. Agent briefings (persisted so the agent doesn't regenerate on every load)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS agent_briefings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id UUID NOT NULL,
    scope_type TEXT NOT NULL,
    scope_id UUID,
    briefing_text TEXT NOT NULL,
    action_items JSONB DEFAULT '[]',
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ DEFAULT NOW() + INTERVAL '4 hours',
    data_hash TEXT -- hash of input data, regenerate if data changes
);

ALTER TABLE agent_briefings ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view their own briefings"
    ON agent_briefings FOR SELECT
    USING (user_id = auth.uid());

CREATE POLICY "Users can insert their own briefings"
    ON agent_briefings FOR INSERT
    WITH CHECK (user_id = auth.uid());

CREATE INDEX idx_agent_briefings_user ON agent_briefings(user_id, generated_at DESC);
CREATE INDEX idx_agent_briefings_org ON agent_briefings(org_id);

-- ---------------------------------------------------------------------------
-- 7. Agent actions (approve, defer, reject — full audit trail)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS agent_actions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    store_id UUID REFERENCES org_stores(id),
    user_id UUID NOT NULL,
    action_type TEXT NOT NULL CHECK (action_type IN (
        'transfer', 'clearance', 'reorder', 'price_adjustment',
        'vendor_contact', 'threshold_change', 'custom'
    )),
    description TEXT NOT NULL,
    reasoning TEXT,
    financial_impact NUMERIC DEFAULT 0,
    confidence NUMERIC DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending', 'approved', 'deferred', 'rejected', 'completed', 'auto_approved'
    )),
    source TEXT DEFAULT 'agent' CHECK (source IN ('agent', 'manual')),
    linked_finding_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    decided_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    deferred_until TIMESTAMPTZ,
    outcome_notes TEXT
);

ALTER TABLE agent_actions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Org members can view actions"
    ON agent_actions FOR SELECT
    USING (org_id IN (
        SELECT org_id FROM user_roles WHERE user_id = auth.uid()
    ));

CREATE POLICY "Org members can manage actions"
    ON agent_actions FOR ALL
    USING (org_id IN (
        SELECT org_id FROM user_roles WHERE user_id = auth.uid()
    ));

CREATE INDEX idx_agent_actions_org_status ON agent_actions(org_id, status);
CREATE INDEX idx_agent_actions_store ON agent_actions(store_id, status);
