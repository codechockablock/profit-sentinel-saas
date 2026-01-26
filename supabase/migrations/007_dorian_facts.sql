-- =============================================================================
-- 007_dorian_facts.sql - Dorian Knowledge Moat Persistence
-- =============================================================================
--
-- Purpose: Store anonymized patterns confirmed by users to build the data moat.
-- Every confirmed pattern makes the system smarter for future customers.
--
-- CRITICAL PRIVACY RULES:
-- - NO PII (no emails, customer names, store names)
-- - NO raw SKUs (anonymize to category: "lumber", "electrical", "plumbing")
-- - NO exact quantities (just pattern: "negative", "overstock", "low")
-- - ONLY save confirmed patterns (user verified "yes this is correct")
--
-- =============================================================================

-- Enable pgvector extension for high-dimensional vector storage
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- CORE FACTS TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS dorian_facts (
    -- Identity
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- The Triple (core knowledge representation)
    subject TEXT NOT NULL,           -- Anonymized entity (e.g., "hardware_store")
    predicate TEXT NOT NULL,         -- Relationship (e.g., "has_pattern")
    object TEXT NOT NULL,            -- Value (e.g., "seasonal_receiving_gap")

    -- Vector Embedding (for semantic similarity search)
    -- Using 512 dimensions to match Dorian's default VSA encoding
    vector vector(512),

    -- Confidence & Verification
    confidence FLOAT DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    confirmations INT DEFAULT 1 CHECK (confirmations >= 0),

    -- Categorization (for filtering and segmentation)
    industry TEXT,                   -- e.g., "hardware", "grocery", "pharmacy"
    pattern_type TEXT,               -- e.g., "receiving_gap", "shrinkage", "overstock"
    sku_category TEXT,               -- e.g., "lumber", "electrical", "plumbing"

    -- Provenance (who/what created this)
    agent_id TEXT DEFAULT 'diagnostic',
    domain TEXT DEFAULT 'retail',
    source TEXT DEFAULT 'user_confirmation',

    -- Status
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'superseded', 'contradicted', 'retracted', 'pending')),

    -- Extensible metadata (JSON for future-proofing)
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add comment for documentation
COMMENT ON TABLE dorian_facts IS 'Anonymized patterns confirmed by users - the data moat. NO PII stored.';
COMMENT ON COLUMN dorian_facts.vector IS 'VSA-encoded 512-dim vector for semantic similarity search';
COMMENT ON COLUMN dorian_facts.confirmations IS 'Number of times users have confirmed this pattern';

-- =============================================================================
-- INDEXES FOR PERFORMANCE
-- =============================================================================

-- Vector similarity search using IVFFlat (good balance of speed/accuracy)
-- lists = 100 is appropriate for up to ~1M vectors
CREATE INDEX IF NOT EXISTS dorian_facts_vector_idx
    ON dorian_facts
    USING ivfflat (vector vector_cosine_ops)
    WITH (lists = 100);

-- Lookup indexes for common queries
CREATE INDEX IF NOT EXISTS dorian_facts_subject_idx ON dorian_facts(subject);
CREATE INDEX IF NOT EXISTS dorian_facts_predicate_idx ON dorian_facts(predicate);
CREATE INDEX IF NOT EXISTS dorian_facts_pattern_type_idx ON dorian_facts(pattern_type);
CREATE INDEX IF NOT EXISTS dorian_facts_industry_idx ON dorian_facts(industry);
CREATE INDEX IF NOT EXISTS dorian_facts_sku_category_idx ON dorian_facts(sku_category);
CREATE INDEX IF NOT EXISTS dorian_facts_status_idx ON dorian_facts(status) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS dorian_facts_created_idx ON dorian_facts(created_at DESC);

-- Composite index for industry + pattern_type queries
CREATE INDEX IF NOT EXISTS dorian_facts_industry_pattern_idx
    ON dorian_facts(industry, pattern_type)
    WHERE status = 'active';

-- =============================================================================
-- AUTOMATIC UPDATED_AT TRIGGER
-- =============================================================================

CREATE OR REPLACE FUNCTION update_dorian_facts_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS dorian_facts_updated_at ON dorian_facts;
CREATE TRIGGER dorian_facts_updated_at
    BEFORE UPDATE ON dorian_facts
    FOR EACH ROW
    EXECUTE FUNCTION update_dorian_facts_updated_at();

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- Function to find similar facts by vector similarity
CREATE OR REPLACE FUNCTION find_similar_facts(
    query_vector vector(512),
    similarity_threshold FLOAT DEFAULT 0.8,
    max_results INT DEFAULT 10,
    filter_industry TEXT DEFAULT NULL,
    filter_pattern_type TEXT DEFAULT NULL
)
RETURNS TABLE (
    fact_id UUID,
    subject TEXT,
    predicate TEXT,
    object TEXT,
    similarity FLOAT,
    confirmations INT,
    industry TEXT,
    pattern_type TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        df.id,
        df.subject,
        df.predicate,
        df.object,
        1 - (df.vector <=> query_vector) AS similarity,
        df.confirmations,
        df.industry,
        df.pattern_type
    FROM dorian_facts df
    WHERE df.status = 'active'
        AND df.vector IS NOT NULL
        AND (filter_industry IS NULL OR df.industry = filter_industry)
        AND (filter_pattern_type IS NULL OR df.pattern_type = filter_pattern_type)
        AND 1 - (df.vector <=> query_vector) >= similarity_threshold
    ORDER BY df.vector <=> query_vector
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Function to increment confirmation count
CREATE OR REPLACE FUNCTION increment_fact_confirmation(fact_uuid UUID)
RETURNS void AS $$
BEGIN
    UPDATE dorian_facts
    SET confirmations = confirmations + 1,
        updated_at = NOW()
    WHERE id = fact_uuid;
END;
$$ LANGUAGE plpgsql;

-- Function to upsert a fact (insert or increment confirmation if similar exists)
CREATE OR REPLACE FUNCTION upsert_dorian_fact(
    p_subject TEXT,
    p_predicate TEXT,
    p_object TEXT,
    p_vector vector(512),
    p_confidence FLOAT DEFAULT 1.0,
    p_industry TEXT DEFAULT NULL,
    p_pattern_type TEXT DEFAULT NULL,
    p_sku_category TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'
)
RETURNS UUID AS $$
DECLARE
    existing_id UUID;
    new_id UUID;
    similarity_threshold FLOAT := 0.95;  -- Very high similarity = same fact
BEGIN
    -- Check for highly similar existing fact
    SELECT df.id INTO existing_id
    FROM dorian_facts df
    WHERE df.status = 'active'
        AND df.vector IS NOT NULL
        AND 1 - (df.vector <=> p_vector) >= similarity_threshold
    ORDER BY df.vector <=> p_vector
    LIMIT 1;

    IF existing_id IS NOT NULL THEN
        -- Increment confirmation on existing fact
        UPDATE dorian_facts
        SET confirmations = confirmations + 1,
            confidence = GREATEST(confidence, p_confidence),
            updated_at = NOW()
        WHERE id = existing_id;

        RETURN existing_id;
    ELSE
        -- Insert new fact
        INSERT INTO dorian_facts (
            subject, predicate, object, vector, confidence,
            industry, pattern_type, sku_category, metadata
        ) VALUES (
            p_subject, p_predicate, p_object, p_vector, p_confidence,
            p_industry, p_pattern_type, p_sku_category, p_metadata
        )
        RETURNING id INTO new_id;

        RETURN new_id;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- ANALYTICS VIEWS
-- =============================================================================

-- View for pattern frequency analysis
CREATE OR REPLACE VIEW dorian_pattern_stats AS
SELECT
    pattern_type,
    industry,
    COUNT(*) as fact_count,
    SUM(confirmations) as total_confirmations,
    AVG(confidence) as avg_confidence,
    MAX(updated_at) as last_updated
FROM dorian_facts
WHERE status = 'active'
GROUP BY pattern_type, industry
ORDER BY total_confirmations DESC;

-- View for industry breakdown
CREATE OR REPLACE VIEW dorian_industry_stats AS
SELECT
    industry,
    COUNT(*) as fact_count,
    SUM(confirmations) as total_confirmations,
    COUNT(DISTINCT pattern_type) as unique_patterns,
    MAX(created_at) as last_fact_added
FROM dorian_facts
WHERE status = 'active'
GROUP BY industry
ORDER BY total_confirmations DESC;

-- =============================================================================
-- ROW LEVEL SECURITY (optional, for multi-tenant future)
-- =============================================================================

-- Enable RLS
ALTER TABLE dorian_facts ENABLE ROW LEVEL SECURITY;

-- Policy: Service role can do everything
CREATE POLICY dorian_facts_service_all ON dorian_facts
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- Policy: Authenticated users can read active facts
CREATE POLICY dorian_facts_read_active ON dorian_facts
    FOR SELECT
    TO authenticated
    USING (status = 'active');

-- =============================================================================
-- SEED DATA (optional - common retail patterns)
-- =============================================================================

-- Note: Vectors would be populated by the application
-- These are placeholder entries showing the expected data format

-- COMMENT OUT in production, uncomment for dev/testing:
/*
INSERT INTO dorian_facts (subject, predicate, object, confidence, industry, pattern_type, sku_category, metadata)
VALUES
    ('hardware_store', 'has_pattern', 'q4_receiving_gap', 0.95, 'hardware', 'receiving_gap', 'lumber', '{"season": "fall", "typical_months": [10,11,12]}'),
    ('hardware_store', 'has_pattern', 'seasonal_overstock', 0.90, 'hardware', 'overstock', 'outdoor', '{"season": "spring", "typical_months": [3,4,5]}'),
    ('pharmacy', 'has_pattern', 'expiration_shrinkage', 0.85, 'pharmacy', 'shrinkage', 'otc_meds', '{"cause": "expiration", "typical_loss_rate": 0.02}'),
    ('grocery', 'has_pattern', 'vendor_short_ship', 0.92, 'grocery', 'receiving_gap', 'produce', '{"cause": "vendor", "frequency": "weekly"}');
*/

-- =============================================================================
-- GRANT PERMISSIONS
-- =============================================================================

-- Grant usage to authenticated and service roles
GRANT USAGE ON SCHEMA public TO authenticated, service_role;
GRANT ALL ON dorian_facts TO service_role;
GRANT SELECT ON dorian_facts TO authenticated;
GRANT SELECT ON dorian_pattern_stats TO authenticated, service_role;
GRANT SELECT ON dorian_industry_stats TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION find_similar_facts TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION increment_fact_confirmation TO service_role;
GRANT EXECUTE ON FUNCTION upsert_dorian_fact TO service_role;
