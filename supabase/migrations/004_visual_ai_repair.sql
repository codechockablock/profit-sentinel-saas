-- Migration: 004_visual_ai_repair
-- Description: Visual AI Repair Assistant - employee empowerment + customer assistance
-- Created: 2026-01-18

-- ============================================================================
-- EMPLOYEES & EXPERTISE
-- ============================================================================

-- Employees with gamification and expertise tracking
CREATE TABLE IF NOT EXISTS employees (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    store_id UUID NOT NULL,  -- Link to stores table (future)

    -- Identity
    name TEXT NOT NULL,
    email TEXT,
    pin_hash TEXT,  -- For quick login at terminal

    -- Gamification
    xp INTEGER NOT NULL DEFAULT 0,
    level INTEGER NOT NULL DEFAULT 1,
    current_streak INTEGER NOT NULL DEFAULT 0,
    longest_streak INTEGER NOT NULL DEFAULT 0,
    last_active_date DATE,

    -- Stats
    total_assists INTEGER NOT NULL DEFAULT 0,
    total_corrections INTEGER NOT NULL DEFAULT 0,
    corrections_accepted INTEGER NOT NULL DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT employees_xp_positive CHECK (xp >= 0),
    CONSTRAINT employees_level_valid CHECK (level >= 1 AND level <= 10)
);

CREATE INDEX idx_employees_store ON employees(store_id);
CREATE INDEX idx_employees_xp ON employees(xp DESC);

-- Employee skill vectors per category
CREATE TABLE IF NOT EXISTS employee_skills (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    employee_id UUID NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    category_id UUID NOT NULL,  -- References problem_categories

    -- Mastery score (0-100)
    mastery_score DECIMAL(5,2) NOT NULL DEFAULT 0,

    -- Stats for this category
    assists_count INTEGER NOT NULL DEFAULT 0,
    corrections_count INTEGER NOT NULL DEFAULT 0,

    -- VSA skill vector (serialized)
    skill_vector_blob BYTEA,

    -- Timestamps
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT employee_skills_unique UNIQUE (employee_id, category_id),
    CONSTRAINT employee_skills_mastery_range CHECK (mastery_score >= 0 AND mastery_score <= 100)
);

CREATE INDEX idx_employee_skills_employee ON employee_skills(employee_id);
CREATE INDEX idx_employee_skills_mastery ON employee_skills(mastery_score DESC);

-- Employee badges
CREATE TABLE IF NOT EXISTS employee_badges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    employee_id UUID NOT NULL REFERENCES employees(id) ON DELETE CASCADE,

    badge_type TEXT NOT NULL,
    badge_name TEXT NOT NULL,
    badge_description TEXT,

    -- For category-specific badges
    category_id UUID,

    -- When earned
    earned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT employee_badges_unique UNIQUE (employee_id, badge_type, category_id)
);

CREATE INDEX idx_employee_badges_employee ON employee_badges(employee_id);

-- ============================================================================
-- PROBLEM CATEGORIES
-- ============================================================================

-- Hierarchical problem categories for diagnosis
CREATE TABLE IF NOT EXISTS problem_categories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Hierarchy
    parent_id UUID REFERENCES problem_categories(id) ON DELETE SET NULL,

    -- Identity
    name TEXT NOT NULL UNIQUE,
    slug TEXT NOT NULL UNIQUE,
    description TEXT,

    -- Display
    icon TEXT,  -- Icon name or emoji
    sort_order INTEGER NOT NULL DEFAULT 0,

    -- VSA encoding
    category_vector_blob BYTEA,

    -- Metadata
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_problem_categories_parent ON problem_categories(parent_id);
CREATE INDEX idx_problem_categories_slug ON problem_categories(slug);

-- Seed common hardware store problem categories
INSERT INTO problem_categories (name, slug, description, icon, sort_order) VALUES
    ('Plumbing', 'plumbing', 'Pipes, faucets, toilets, drains', '🔧', 1),
    ('Electrical', 'electrical', 'Wiring, outlets, switches, lights', '⚡', 2),
    ('HVAC', 'hvac', 'Heating, cooling, ventilation', '🌡️', 3),
    ('Carpentry', 'carpentry', 'Wood, framing, trim, doors', '🪵', 4),
    ('Painting', 'painting', 'Interior, exterior, staining', '🎨', 5),
    ('Flooring', 'flooring', 'Tile, hardwood, carpet, vinyl', '🏠', 6),
    ('Roofing', 'roofing', 'Shingles, gutters, flashing', '🏗️', 7),
    ('Appliances', 'appliances', 'Major and small appliances', '🔌', 8),
    ('Outdoor', 'outdoor', 'Lawn, garden, landscaping', '🌿', 9),
    ('Automotive', 'automotive', 'Car maintenance and repair', '🚗', 10)
ON CONFLICT (slug) DO NOTHING;

-- Add subcategories for plumbing (example)
INSERT INTO problem_categories (name, slug, description, parent_id, sort_order)
SELECT
    name, slug, description,
    (SELECT id FROM problem_categories WHERE slug = 'plumbing'),
    sort_order
FROM (VALUES
    ('Leaky Faucet', 'plumbing-faucet', 'Dripping or running faucets', 1),
    ('Clogged Drain', 'plumbing-drain', 'Slow or blocked drains', 2),
    ('Running Toilet', 'plumbing-toilet', 'Toilet won''t stop running', 3),
    ('Pipe Leak', 'plumbing-pipe', 'Leaking pipes under sink or wall', 4),
    ('Water Heater', 'plumbing-waterheater', 'Hot water issues', 5)
) AS t(name, slug, description, sort_order)
ON CONFLICT (slug) DO NOTHING;

-- ============================================================================
-- PROBLEMS & DIAGNOSIS
-- ============================================================================

-- Submitted problems (customer or employee)
CREATE TABLE IF NOT EXISTS problems (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    store_id UUID NOT NULL,

    -- Who submitted
    employee_id UUID REFERENCES employees(id) ON DELETE SET NULL,
    session_id TEXT,  -- Anonymous customer session

    -- Input
    text_description TEXT,
    voice_transcript TEXT,
    image_hash TEXT,  -- For dedup only, NOT stored

    -- Diagnosis state
    status TEXT NOT NULL DEFAULT 'pending',
    -- 'pending', 'diagnosed', 'refined', 'solved', 'corrected'

    -- VSA hypothesis state (serialized HypothesisBundle)
    hypothesis_blob BYTEA,

    -- Current best diagnosis
    top_category_id UUID REFERENCES problem_categories(id),
    confidence DECIMAL(4,3),  -- 0.000 to 1.000

    -- Follow-up questions (JSON array)
    follow_up_questions JSONB DEFAULT '[]',

    -- Metadata
    difficulty_level INTEGER,  -- 1-5, set after solution

    -- Privacy: customer sessions auto-delete
    is_anonymous BOOLEAN NOT NULL DEFAULT FALSE,
    expires_at TIMESTAMPTZ,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT problems_status_valid CHECK (
        status IN ('pending', 'diagnosed', 'refined', 'solved', 'corrected')
    ),
    CONSTRAINT problems_confidence_range CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1))
);

CREATE INDEX idx_problems_store ON problems(store_id);
CREATE INDEX idx_problems_employee ON problems(employee_id) WHERE employee_id IS NOT NULL;
CREATE INDEX idx_problems_status ON problems(status);
CREATE INDEX idx_problems_created ON problems(created_at DESC);
CREATE INDEX idx_problems_expires ON problems(expires_at) WHERE expires_at IS NOT NULL;

-- Problem hypotheses (denormalized for API response)
CREATE TABLE IF NOT EXISTS problem_hypotheses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    problem_id UUID NOT NULL REFERENCES problems(id) ON DELETE CASCADE,
    category_id UUID NOT NULL REFERENCES problem_categories(id),

    probability DECIMAL(4,3) NOT NULL,  -- 0.000 to 1.000
    explanation TEXT,

    -- Order (1 = top hypothesis)
    rank INTEGER NOT NULL,

    CONSTRAINT problem_hypotheses_unique UNIQUE (problem_id, category_id),
    CONSTRAINT problem_hypotheses_prob_range CHECK (probability >= 0 AND probability <= 1)
);

CREATE INDEX idx_problem_hypotheses_problem ON problem_hypotheses(problem_id);

-- ============================================================================
-- SOLUTIONS
-- ============================================================================

-- Generated solutions
CREATE TABLE IF NOT EXISTS solutions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    problem_id UUID NOT NULL REFERENCES problems(id) ON DELETE CASCADE,
    category_id UUID NOT NULL REFERENCES problem_categories(id),

    -- Content
    title TEXT NOT NULL,
    summary TEXT NOT NULL,

    -- Steps (JSON array)
    steps JSONB NOT NULL DEFAULT '[]',
    -- [{"order": 1, "instruction": "...", "tip": "...", "caution": "..."}]

    -- Requirements
    tools_required JSONB DEFAULT '[]',
    estimated_minutes INTEGER,
    difficulty_level INTEGER NOT NULL DEFAULT 3,  -- 1-5

    -- Video references
    video_urls JSONB DEFAULT '[]',

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT solutions_difficulty_range CHECK (difficulty_level >= 1 AND difficulty_level <= 5)
);

CREATE INDEX idx_solutions_problem ON solutions(problem_id);
CREATE INDEX idx_solutions_category ON solutions(category_id);

-- Solution parts with inventory status
CREATE TABLE IF NOT EXISTS solution_parts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    solution_id UUID NOT NULL REFERENCES solutions(id) ON DELETE CASCADE,

    -- Part info
    part_name TEXT NOT NULL,
    part_description TEXT,
    quantity INTEGER NOT NULL DEFAULT 1,

    -- Link to inventory (optional)
    sku TEXT,

    -- Inventory status at time of solution
    in_stock BOOLEAN,
    stock_quantity INTEGER,
    unit_price DECIMAL(10,2),

    -- Substitute info
    has_substitute BOOLEAN DEFAULT FALSE,
    substitute_sku TEXT,
    substitute_name TEXT,

    -- Order
    sort_order INTEGER NOT NULL DEFAULT 0,

    -- Is this required or optional?
    is_required BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE INDEX idx_solution_parts_solution ON solution_parts(solution_id);
CREATE INDEX idx_solution_parts_sku ON solution_parts(sku) WHERE sku IS NOT NULL;

-- ============================================================================
-- LEARNING & CORRECTIONS
-- ============================================================================

-- Employee corrections to AI diagnoses
CREATE TABLE IF NOT EXISTS diagnosis_corrections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    problem_id UUID NOT NULL REFERENCES problems(id) ON DELETE CASCADE,
    employee_id UUID NOT NULL REFERENCES employees(id) ON DELETE CASCADE,

    -- What was wrong
    original_category_id UUID REFERENCES problem_categories(id),
    original_confidence DECIMAL(4,3),

    -- What's correct
    corrected_category_id UUID NOT NULL REFERENCES problem_categories(id),
    correction_notes TEXT,

    -- Was this accepted into the knowledge base?
    is_accepted BOOLEAN,
    accepted_at TIMESTAMPTZ,

    -- XP awarded for this correction
    xp_awarded INTEGER DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_corrections_problem ON diagnosis_corrections(problem_id);
CREATE INDEX idx_corrections_employee ON diagnosis_corrections(employee_id);
CREATE INDEX idx_corrections_accepted ON diagnosis_corrections(is_accepted) WHERE is_accepted = TRUE;

-- Learning events (XP gains, badge unlocks, etc.)
CREATE TABLE IF NOT EXISTS learning_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    employee_id UUID NOT NULL REFERENCES employees(id) ON DELETE CASCADE,

    -- Event type
    event_type TEXT NOT NULL,
    -- 'assist', 'correction', 'badge_earned', 'level_up', 'streak_extended'

    -- XP change
    xp_delta INTEGER NOT NULL DEFAULT 0,
    xp_after INTEGER NOT NULL,

    -- Context
    problem_id UUID REFERENCES problems(id) ON DELETE SET NULL,
    category_id UUID REFERENCES problem_categories(id),
    badge_type TEXT,

    -- Metadata
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_learning_events_employee ON learning_events(employee_id);
CREATE INDEX idx_learning_events_type ON learning_events(event_type);
CREATE INDEX idx_learning_events_created ON learning_events(created_at DESC);

-- ============================================================================
-- REPAIR GUIDES (CURATED KNOWLEDGE)
-- ============================================================================

-- Curated repair guides (can be seeded or created by experts)
CREATE TABLE IF NOT EXISTS repair_guides (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    category_id UUID NOT NULL REFERENCES problem_categories(id),

    -- Content
    title TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    summary TEXT NOT NULL,

    -- Full guide content (Markdown)
    content TEXT NOT NULL,

    -- Structured data
    steps JSONB NOT NULL DEFAULT '[]',
    tools_required JSONB DEFAULT '[]',
    parts_typically_needed JSONB DEFAULT '[]',

    -- Difficulty and time
    difficulty_level INTEGER NOT NULL DEFAULT 3,
    estimated_minutes INTEGER,

    -- Videos
    video_urls JSONB DEFAULT '[]',

    -- Search
    keywords TEXT[],

    -- Metadata
    is_published BOOLEAN NOT NULL DEFAULT FALSE,
    view_count INTEGER NOT NULL DEFAULT 0,
    helpful_count INTEGER NOT NULL DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_repair_guides_category ON repair_guides(category_id);
CREATE INDEX idx_repair_guides_slug ON repair_guides(slug);
CREATE INDEX idx_repair_guides_published ON repair_guides(is_published) WHERE is_published = TRUE;
CREATE INDEX idx_repair_guides_keywords ON repair_guides USING GIN(keywords);

-- ============================================================================
-- STORE MEMORY (VSA T-Bind)
-- ============================================================================

-- Store-level memory for "we've seen this before" context
CREATE TABLE IF NOT EXISTS store_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    store_id UUID NOT NULL UNIQUE,

    -- VSA memory vector (serialized)
    memory_vector_blob BYTEA,

    -- Stats
    total_problems INTEGER NOT NULL DEFAULT 0,

    -- Category distribution (JSON)
    category_distribution JSONB DEFAULT '{}',

    -- Last updated
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_store_memory_store ON store_memory(store_id);

-- ============================================================================
-- KNOWLEDGE BASE (VSA CW-Bundle accumulation)
-- ============================================================================

-- System-wide learned knowledge from employee corrections
CREATE TABLE IF NOT EXISTS knowledge_base (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    category_id UUID NOT NULL REFERENCES problem_categories(id) UNIQUE,

    -- VSA knowledge vector (CW-Bundle of corrections)
    knowledge_vector_blob BYTEA,

    -- Confidence (weighted average from corrections)
    aggregate_confidence DECIMAL(4,3) NOT NULL DEFAULT 0.5,

    -- Stats
    total_corrections INTEGER NOT NULL DEFAULT 0,

    -- Last update
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_knowledge_base_category ON knowledge_base(category_id);

-- ============================================================================
-- ROW LEVEL SECURITY
-- ============================================================================

-- Enable RLS on all tables
ALTER TABLE employees ENABLE ROW LEVEL SECURITY;
ALTER TABLE employee_skills ENABLE ROW LEVEL SECURITY;
ALTER TABLE employee_badges ENABLE ROW LEVEL SECURITY;
ALTER TABLE problems ENABLE ROW LEVEL SECURITY;
ALTER TABLE problem_hypotheses ENABLE ROW LEVEL SECURITY;
ALTER TABLE solutions ENABLE ROW LEVEL SECURITY;
ALTER TABLE solution_parts ENABLE ROW LEVEL SECURITY;
ALTER TABLE diagnosis_corrections ENABLE ROW LEVEL SECURITY;
ALTER TABLE learning_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE repair_guides ENABLE ROW LEVEL SECURITY;
ALTER TABLE store_memory ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge_base ENABLE ROW LEVEL SECURITY;
ALTER TABLE problem_categories ENABLE ROW LEVEL SECURITY;

-- Service role has full access
CREATE POLICY "Service role full access" ON employees
    FOR ALL USING (auth.role() = 'service_role') WITH CHECK (auth.role() = 'service_role');
CREATE POLICY "Service role full access" ON employee_skills
    FOR ALL USING (auth.role() = 'service_role') WITH CHECK (auth.role() = 'service_role');
CREATE POLICY "Service role full access" ON employee_badges
    FOR ALL USING (auth.role() = 'service_role') WITH CHECK (auth.role() = 'service_role');
CREATE POLICY "Service role full access" ON problems
    FOR ALL USING (auth.role() = 'service_role') WITH CHECK (auth.role() = 'service_role');
CREATE POLICY "Service role full access" ON problem_hypotheses
    FOR ALL USING (auth.role() = 'service_role') WITH CHECK (auth.role() = 'service_role');
CREATE POLICY "Service role full access" ON solutions
    FOR ALL USING (auth.role() = 'service_role') WITH CHECK (auth.role() = 'service_role');
CREATE POLICY "Service role full access" ON solution_parts
    FOR ALL USING (auth.role() = 'service_role') WITH CHECK (auth.role() = 'service_role');
CREATE POLICY "Service role full access" ON diagnosis_corrections
    FOR ALL USING (auth.role() = 'service_role') WITH CHECK (auth.role() = 'service_role');
CREATE POLICY "Service role full access" ON learning_events
    FOR ALL USING (auth.role() = 'service_role') WITH CHECK (auth.role() = 'service_role');
CREATE POLICY "Service role full access" ON repair_guides
    FOR ALL USING (auth.role() = 'service_role') WITH CHECK (auth.role() = 'service_role');
CREATE POLICY "Service role full access" ON store_memory
    FOR ALL USING (auth.role() = 'service_role') WITH CHECK (auth.role() = 'service_role');
CREATE POLICY "Service role full access" ON knowledge_base
    FOR ALL USING (auth.role() = 'service_role') WITH CHECK (auth.role() = 'service_role');
CREATE POLICY "Service role full access" ON problem_categories
    FOR ALL USING (auth.role() = 'service_role') WITH CHECK (auth.role() = 'service_role');

-- Public read for categories
CREATE POLICY "Public read categories" ON problem_categories
    FOR SELECT TO anon USING (is_active = TRUE);

-- Public read for published guides
CREATE POLICY "Public read published guides" ON repair_guides
    FOR SELECT TO anon USING (is_published = TRUE);

-- ============================================================================
-- CLEANUP FUNCTIONS
-- ============================================================================

-- Auto-delete expired anonymous problems
CREATE OR REPLACE FUNCTION cleanup_expired_problems()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM problems
    WHERE is_anonymous = TRUE
      AND expires_at IS NOT NULL
      AND expires_at < NOW();

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Update employee streak
CREATE OR REPLACE FUNCTION update_employee_streak(p_employee_id UUID)
RETURNS void AS $$
DECLARE
    last_date DATE;
    today DATE := CURRENT_DATE;
BEGIN
    SELECT last_active_date INTO last_date
    FROM employees
    WHERE id = p_employee_id;

    IF last_date IS NULL OR last_date < today - INTERVAL '1 day' THEN
        -- Reset streak
        UPDATE employees
        SET current_streak = 1,
            last_active_date = today,
            updated_at = NOW()
        WHERE id = p_employee_id;
    ELSIF last_date = today - INTERVAL '1 day' THEN
        -- Extend streak
        UPDATE employees
        SET current_streak = current_streak + 1,
            longest_streak = GREATEST(longest_streak, current_streak + 1),
            last_active_date = today,
            updated_at = NOW()
        WHERE id = p_employee_id;
    ELSE
        -- Same day, just update timestamp
        UPDATE employees
        SET last_active_date = today,
            updated_at = NOW()
        WHERE id = p_employee_id;
    END IF;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE employees IS 'Employee profiles with gamification for repair assistant';
COMMENT ON TABLE employee_skills IS 'Per-category skill mastery for each employee';
COMMENT ON TABLE employee_badges IS 'Earned badges and achievements';
COMMENT ON TABLE problem_categories IS 'Hierarchical taxonomy of repair problem types';
COMMENT ON TABLE problems IS 'Customer/employee submitted repair problems';
COMMENT ON TABLE problem_hypotheses IS 'P-Sup hypothesis probabilities for each problem';
COMMENT ON TABLE solutions IS 'Generated repair solutions with steps';
COMMENT ON TABLE solution_parts IS 'Parts needed for solutions with inventory status';
COMMENT ON TABLE diagnosis_corrections IS 'Employee corrections to AI diagnoses';
COMMENT ON TABLE learning_events IS 'XP gains, badges, level ups - gamification log';
COMMENT ON TABLE repair_guides IS 'Curated expert repair guides';
COMMENT ON TABLE store_memory IS 'VSA T-Bind store history context';
COMMENT ON TABLE knowledge_base IS 'VSA CW-Bundle learned knowledge per category';
