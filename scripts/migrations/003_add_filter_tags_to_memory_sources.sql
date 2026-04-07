-- Migration: Add filter_tags column to memory_sources table
-- Story: VEPAGE-769 (S9 Retrieval Endpoints)
-- Purpose: Enable scope-based access control for memory sources, matching
--          the pattern used by memory tables (episodic, semantic, etc.)
--
-- filter_tags is a JSON column that stores custom tags including "scope"
-- which is auto-injected from the client's write_scope on save.
-- Retrieval endpoints filter by filter_tags->>'scope' IN (client.read_scopes).

-- Add the column (nullable, no default — existing rows get NULL)
ALTER TABLE memory_sources ADD COLUMN IF NOT EXISTS filter_tags JSON;

-- GIN index for flexible tag queries
CREATE INDEX IF NOT EXISTS ix_memory_sources_filter_tags_gin
    ON memory_sources USING gin ((filter_tags::jsonb));

-- Btree index for scope-based access control queries
CREATE INDEX IF NOT EXISTS ix_memory_sources_org_filter_scope
    ON memory_sources USING btree (organization_id, ((filter_tags->>'scope')::text));
