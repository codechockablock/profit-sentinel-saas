-- Store management table
-- Each user can create named stores (locations) for multi-store inventory analysis.
-- Single-store users get one auto-created "Main Store" on first login.

CREATE TABLE stores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id),
    name TEXT NOT NULL,
    address TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_upload_at TIMESTAMPTZ,
    item_count INTEGER DEFAULT 0,
    total_impact NUMERIC DEFAULT 0,
    UNIQUE(user_id, name)
);

-- RLS: users can only see/manage their own stores
ALTER TABLE stores ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can manage own stores"
    ON stores FOR ALL
    USING (user_id = auth.uid())
    WITH CHECK (user_id = auth.uid());

CREATE INDEX idx_stores_user_id ON stores(user_id);
