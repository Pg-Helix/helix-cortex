-- Workflow 2 (ADR-0007, helix-atlas): cortex's own schema. Run once, by
-- helix_cortex_api (which owns the `cortex` schema outright -- see
-- helix-foundry compose/initdb/01-install.sh's `CREATE SCHEMA cortex
-- AUTHORIZATION helix_cortex_api`). No helix-kernel SQL is touched here;
-- cortex.chunk is registered with helix_ai.register_source exactly like
-- any other application table (see internal/api's startup code), not a
-- kernel schema change.
--
-- FORCE ROW LEVEL SECURITY on every table below, same lesson as ADR-0006
-- decision 1: without it, helix_cortex_api -- the OWNER of these tables
-- -- would bypass its own RLS policies entirely, making the tenant/ACL
-- checks below decorative rather than binding.

CREATE TABLE IF NOT EXISTS cortex.collection (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant     text NOT NULL,
    name       text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant, name)
);

CREATE TABLE IF NOT EXISTS cortex.collection_permission (
    collection_id uuid NOT NULL REFERENCES cortex.collection(id) ON DELETE CASCADE,
    tenant        text NOT NULL,
    principal     text NOT NULL,
    role          text NOT NULL CHECK (role IN ('owner', 'reader')),
    granted_by    text NOT NULL,
    granted_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (collection_id, principal)
);

CREATE TABLE IF NOT EXISTS cortex.document (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    collection_id   uuid NOT NULL REFERENCES cortex.collection(id) ON DELETE CASCADE,
    tenant          text NOT NULL,
    name            text NOT NULL,
    content_type    text NOT NULL CHECK (content_type IN ('text/plain', 'text/markdown')),
    chunker_version text NOT NULL,
    -- Denormalized pointer to the currently-active version, for cheap
    -- status/metadata reads without a join; document_version remains the
    -- source of truth for history (superseded_at).
    active_version_id uuid,
    created_at      timestamptz NOT NULL DEFAULT now(),
    -- A document's identity is (collection_id, name): re-POSTing the
    -- same name is a re-ingestion of the same logical document (new
    -- version if content/chunker changed, idempotent no-op if not),
    -- never a second, independent document that happens to share a name.
    UNIQUE (collection_id, name)
);

CREATE TABLE IF NOT EXISTS cortex.document_version (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     uuid NOT NULL REFERENCES cortex.document(id) ON DELETE CASCADE,
    tenant          text NOT NULL,
    digest          text NOT NULL,
    chunker_version text NOT NULL,
    byte_size       integer NOT NULL,
    chunk_count     integer NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),
    superseded_at   timestamptz,
    UNIQUE (document_id, digest, chunker_version)
);

ALTER TABLE cortex.document
    ADD CONSTRAINT document_active_version_fk
    FOREIGN KEY (active_version_id) REFERENCES cortex.document_version(id);

-- Primary key (document_id, content_digest, chunker_version, chunk_index)
-- is the ADR-0007 decision-3 row key: register_source auto-detects this
-- as the kernel-visible source key. Changed content or chunker version
-- means a DIFFERENT primary key -- old rows are explicitly DELETEd (firing
-- kernel's existing capture trigger, tombstoning their derived_value) and
-- new rows INSERTed, never UPDATEd in place. Unchanged re-ingestion is
-- `INSERT ... ON CONFLICT DO NOTHING` on this same key -- idempotent by
-- construction, no capture-trigger activity, no new job.
CREATE TABLE IF NOT EXISTS cortex.chunk (
    document_id      uuid NOT NULL REFERENCES cortex.document(id) ON DELETE CASCADE,
    content_digest    text NOT NULL,
    chunker_version   text NOT NULL,
    chunk_index       integer NOT NULL,
    document_version_id uuid NOT NULL REFERENCES cortex.document_version(id) ON DELETE CASCADE,
    tenant            text NOT NULL,
    collection_id     uuid NOT NULL REFERENCES cortex.collection(id) ON DELETE CASCADE,
    text              text NOT NULL,
    char_start        integer NOT NULL,
    char_end          integer NOT NULL,
    created_at        timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (document_id, content_digest, chunker_version, chunk_index)
);

-- A functional index, not a stored generated tsvector column: this
-- table is register_source'd with helix-kernel, whose get_job_input
-- does `SELECT to_jsonb(t) FROM cortex.chunk AS t ...` over every
-- column -- a persisted tsvector column would ride along into that
-- jsonb (and the worker's embedding input) for no benefit, and tsvector
-- has no clean JSON representation to begin with. Computed at query
-- time instead; the GIN index below still makes that computation
-- indexable.
CREATE INDEX IF NOT EXISTS chunk_fts_idx ON cortex.chunk USING GIN (to_tsvector('english', text));
CREATE INDEX IF NOT EXISTS chunk_collection_idx ON cortex.chunk (collection_id, tenant);

-- Row Level Security: tenant is the non-bypassable floor on every table;
-- document/chunk/document_version additionally require an explicit
-- collection_permission grant for current_setting('helix.principal') --
-- this is what makes "filter inside the query before LIMIT" true by
-- construction (PostgreSQL pushes FORCE RLS predicates ahead of
-- ORDER BY/LIMIT), not merely by careful query authorship in Go.

ALTER TABLE cortex.collection ENABLE ROW LEVEL SECURITY;
ALTER TABLE cortex.collection FORCE ROW LEVEL SECURITY;
CREATE POLICY collection_tenant_acl ON cortex.collection
    USING (
        tenant = current_setting('helix.tenant', true)
        AND EXISTS (
            SELECT 1 FROM cortex.collection_permission cp
            WHERE cp.collection_id = collection.id
              AND cp.principal = current_setting('helix.principal', true)
        )
    )
    WITH CHECK (tenant = current_setting('helix.tenant', true));

ALTER TABLE cortex.collection_permission ENABLE ROW LEVEL SECURITY;
ALTER TABLE cortex.collection_permission FORCE ROW LEVEL SECURITY;
-- Tenant is the RLS floor here; "only an owner may grant/revoke/list
-- grants" is enforced in the application handler (a mutating-privilege
-- decision, not a row-visibility one -- see internal/api).
CREATE POLICY collection_permission_tenant ON cortex.collection_permission
    USING (tenant = current_setting('helix.tenant', true))
    WITH CHECK (tenant = current_setting('helix.tenant', true));

ALTER TABLE cortex.document ENABLE ROW LEVEL SECURITY;
ALTER TABLE cortex.document FORCE ROW LEVEL SECURITY;
CREATE POLICY document_tenant_acl ON cortex.document
    USING (
        tenant = current_setting('helix.tenant', true)
        AND EXISTS (
            SELECT 1 FROM cortex.collection_permission cp
            WHERE cp.collection_id = document.collection_id
              AND cp.principal = current_setting('helix.principal', true)
        )
    )
    WITH CHECK (tenant = current_setting('helix.tenant', true));

ALTER TABLE cortex.document_version ENABLE ROW LEVEL SECURITY;
ALTER TABLE cortex.document_version FORCE ROW LEVEL SECURITY;
CREATE POLICY document_version_tenant_acl ON cortex.document_version
    USING (
        tenant = current_setting('helix.tenant', true)
        AND EXISTS (
            SELECT 1 FROM cortex.document d
            JOIN cortex.collection_permission cp ON cp.collection_id = d.collection_id
            WHERE d.id = document_version.document_id
              AND cp.principal = current_setting('helix.principal', true)
        )
    )
    WITH CHECK (tenant = current_setting('helix.tenant', true));

-- Tenant-only, NOT principal-ACL'd like the three tables above: this
-- table is register_source'd with helix-kernel, and get_job_input (a
-- function in helix-kernel we never modify) re-reads this exact table
-- by key while running as helix_owner, having set only helix.tenant --
-- it has no concept of "principal" at all. Reproduced directly against
-- the real stack: with a principal-EXISTS clause here too,
-- get_job_input's own re-read returns zero rows (RLS hides them) and
-- it reports every single job stale_source, forever, which eventually
-- dead-letters the job after exhausting its retry budget -- not a
-- security feature, a correctness bug that silently breaks every
-- embedding job. Principal-level ACL for chunk reads is instead
-- enforced explicitly inside cortex's own retrieval/status queries
-- (internal/store), which already join collection_permission for other
-- reasons -- "filtered inside the query before LIMIT" remains true,
-- just not via a blanket policy on this one kernel-touched table.
ALTER TABLE cortex.chunk ENABLE ROW LEVEL SECURITY;
ALTER TABLE cortex.chunk FORCE ROW LEVEL SECURITY;
CREATE POLICY chunk_tenant_acl ON cortex.chunk
    USING (tenant = current_setting('helix.tenant', true))
    WITH CHECK (tenant = current_setting('helix.tenant', true));
