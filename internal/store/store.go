// Package store is Workflow 2's (ADR-0007) data access layer: cortex's
// own collection/document/chunk/permission tables, plus the read-only
// joins into helix_ai.derived_value that dense retrieval needs. Every
// method opens its own transaction and sets helix.tenant/helix.principal
// via SET LOCAL (never session-level SET) so a pooled connection handed
// to a different principal's next request can never retain this
// request's context.
package store

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/Pg-Helix/helix-cortex/internal/auth"
	protocol "github.com/Pg-Helix/helix-protocol/gen/go"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

// ErrForbidden, ErrNotFound and ErrTenantMismatch map directly to the
// request-level error codes in helix-protocol's errors/codes.md; the
// HTTP layer translates them, never leaking which case applied beyond
// what that code already implies.
var (
	ErrForbidden      = errors.New("store: forbidden")
	ErrNotFound       = errors.New("store: not found")
	ErrTenantMismatch = errors.New("store: tenant mismatch")
)

// Store is Workflow 2's data access layer, holding the helix_cortex_api
// pool (never helix_worker's -- claim_job/publish_result/get_job_input
// are never called from here, only from the separate embedding worker).
type Store struct {
	Pool               *pgxpool.Pool
	ChunkModelID       string
	ChunkModelRevision string
	ChunkDims          int
}

func (s *Store) targetModel() string {
	return s.ChunkModelID + ":" + s.ChunkModelRevision
}

// withTenant runs fn inside a transaction with helix.tenant/helix.principal
// set via SET LOCAL for exactly that transaction's lifetime.
func (s *Store) withTenant(ctx context.Context, p auth.Principal, fn func(tx pgx.Tx) error) error {
	tx, err := s.Pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("store: begin: %w", err)
	}
	defer tx.Rollback(ctx)

	if _, err := tx.Exec(ctx, `SELECT set_config('helix.tenant', $1, true), set_config('helix.principal', $2, true)`, p.Tenant, p.Name); err != nil {
		return fmt.Errorf("store: set session context: %w", err)
	}
	if err := fn(tx); err != nil {
		return err
	}
	return tx.Commit(ctx)
}

// ensureSourceRegistered registers cortex.chunk with helix-kernel for
// the current tenant (set_config'd on tx already) if it isn't already --
// register_source's own UNIQUE constraint is (tenant, schema, table,
// model, pipeline), so a brand-new tenant needs its own one-time
// registration against the same physical table (ADR-0007 decision 3).
// Idempotent under a concurrent duplicate call: tolerates the unique-
// violation a race would produce and re-selects.
func (s *Store) ensureSourceRegistered(ctx context.Context, tx pgx.Tx) (int64, error) {
	var id int64
	err := tx.QueryRow(ctx, `
		SELECT id FROM helix_ai.source
		WHERE schema_name = 'cortex' AND table_name = 'chunk'
		  AND model_id = $1 AND pipeline_version = $2
		  AND tenant = current_setting('helix.tenant', true)
	`, s.ChunkModelID, chunkPipelineVersion).Scan(&id)
	if err == nil {
		return id, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return 0, fmt.Errorf("store: look up chunk source registration: %w", err)
	}

	err = tx.QueryRow(ctx, `SELECT helix_ai.register_source('cortex', 'chunk', $1, $2, $3, $4, 'cosine')`,
		s.ChunkModelID, s.ChunkModelRevision, chunkPipelineVersion, s.ChunkDims).Scan(&id)
	if err == nil {
		return id, nil
	}
	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) && pgErr.Code == "23505" { // unique_violation: a concurrent registration won the race
		if selErr := tx.QueryRow(ctx, `
			SELECT id FROM helix_ai.source
			WHERE schema_name = 'cortex' AND table_name = 'chunk'
			  AND model_id = $1 AND pipeline_version = $2
			  AND tenant = current_setting('helix.tenant', true)
		`, s.ChunkModelID, chunkPipelineVersion).Scan(&id); selErr == nil {
			return id, nil
		}
	}
	return 0, fmt.Errorf("store: register chunk source: %w", err)
}

// chunkPipelineVersion is frozen here (not ingest.ChunkerVersion) so a
// chunker-boundary change and an embedding-pipeline change are tracked
// independently -- ADR-0007 decision 3 treats them as two different
// triggers for regeneration, not one.
const chunkPipelineVersion = "cortex-chunk-pipeline-v1"

func (s *Store) CreateCollection(ctx context.Context, p auth.Principal, name string) (protocol.Collection, error) {
	var out protocol.Collection
	err := s.withTenant(ctx, p, func(tx pgx.Tx) error {
		// Every value is generated client-side and the INSERT below has
		// no RETURNING at all: RETURNING's visibility check uses the
		// policy's USING clause regardless of which columns are
		// returned, and this table's USING clause requires an existing
		// collection_permission row for the caller -- which cannot exist
		// yet for a collection that doesn't exist yet. Reproduced
		// directly against the real stack ("new row violates row-level
		// security policy") before this fix; confirmed RETURNING any
		// column (not just id) hits the same check.
		id := uuid.NewString()
		createdAt := time.Now().UTC()
		if _, err := tx.Exec(ctx, `
			INSERT INTO cortex.collection (id, tenant, name, created_at) VALUES ($1, $2, $3, $4)
		`, id, p.Tenant, name, createdAt); err != nil {
			return fmt.Errorf("store: create collection: %w", err)
		}
		_, err := tx.Exec(ctx, `
			INSERT INTO cortex.collection_permission (collection_id, tenant, principal, role, granted_by)
			VALUES ($1, $2, $3, 'owner', $3)
		`, id, p.Tenant, p.Name)
		if err != nil {
			return fmt.Errorf("store: grant creator ownership: %w", err)
		}
		out = protocol.Collection{Id: id, Tenant: p.Tenant, Name: name, CreatedAt: createdAt}
		return nil
	})
	return out, err
}

// requireRole checks the caller holds one of allowedRoles on collectionID
// -- authorizing ingestion/deletion/ACL-changes separately from read
// access is the point (ADR-0007 decision 1), so callers pass different
// allowedRoles sets for mutating vs read-only operations.
func requireRole(ctx context.Context, tx pgx.Tx, collectionID string, p auth.Principal, allowedRoles ...string) error {
	var role string
	err := tx.QueryRow(ctx, `
		SELECT role FROM cortex.collection_permission
		WHERE collection_id = $1 AND principal = $2 AND tenant = $3
	`, collectionID, p.Name, p.Tenant).Scan(&role)
	if errors.Is(err, pgx.ErrNoRows) {
		return ErrForbidden
	}
	if err != nil {
		return fmt.Errorf("store: check permission: %w", err)
	}
	for _, r := range allowedRoles {
		if role == r {
			return nil
		}
	}
	return ErrForbidden
}

func (s *Store) GrantPermission(ctx context.Context, p auth.Principal, collectionID string, grant protocol.Permission) (protocol.Permission, error) {
	var out protocol.Permission
	err := s.withTenant(ctx, p, func(tx pgx.Tx) error {
		if err := requireRole(ctx, tx, collectionID, p, "owner"); err != nil {
			return err
		}
		_, err := tx.Exec(ctx, `
			INSERT INTO cortex.collection_permission (collection_id, tenant, principal, role, granted_by)
			VALUES ($1, $2, $3, $4, $5)
			ON CONFLICT (collection_id, principal) DO UPDATE SET role = EXCLUDED.role, granted_by = EXCLUDED.granted_by, granted_at = now()
		`, collectionID, p.Tenant, grant.Principal, string(grant.Role), p.Name)
		if err != nil {
			return fmt.Errorf("store: grant permission: %w", err)
		}
		out = grant
		return nil
	})
	return out, err
}

func (s *Store) RevokePermission(ctx context.Context, p auth.Principal, collectionID, targetPrincipal string) error {
	return s.withTenant(ctx, p, func(tx pgx.Tx) error {
		if err := requireRole(ctx, tx, collectionID, p, "owner"); err != nil {
			return err
		}
		_, err := tx.Exec(ctx, `
			DELETE FROM cortex.collection_permission WHERE collection_id = $1 AND principal = $2
		`, collectionID, targetPrincipal)
		if err != nil {
			return fmt.Errorf("store: revoke permission: %w", err)
		}
		return nil
	})
}

func (s *Store) ListPermissions(ctx context.Context, p auth.Principal, collectionID string) ([]protocol.Permission, error) {
	var out []protocol.Permission
	err := s.withTenant(ctx, p, func(tx pgx.Tx) error {
		if err := requireRole(ctx, tx, collectionID, p, "owner"); err != nil {
			return err
		}
		rows, err := tx.Query(ctx, `SELECT principal, role FROM cortex.collection_permission WHERE collection_id = $1 ORDER BY principal`, collectionID)
		if err != nil {
			return fmt.Errorf("store: list permissions: %w", err)
		}
		defer rows.Close()
		for rows.Next() {
			var perm protocol.Permission
			var role string
			if err := rows.Scan(&perm.Principal, &role); err != nil {
				return fmt.Errorf("store: scan permission: %w", err)
			}
			perm.Role = protocol.PermissionRole(role)
			out = append(out, perm)
		}
		return rows.Err()
	})
	return out, err
}
