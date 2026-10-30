package store

import (
	"context"
	"fmt"
	"strconv"

	"github.com/Pg-Helix/helix-cortex/internal/auth"
	"github.com/jackc/pgx/v5"
)

// backfillBatchLimit bounds each backfill scan/schedule pass. Callers
// (BackfillTenant, the per-document check in IngestDocument) loop until
// a pass schedules nothing, so a large backlog still completes -- just
// across several bounded passes rather than one unbounded one.
const backfillBatchLimit = 50

// chunkSourceKey builds the jsonb source_key for one chunk row, in the
// exact shape helix-kernel's own capture trigger produces (every value,
// including the integer chunk_index, cast to text) -- request_refresh
// and every ->> extraction elsewhere in this package depend on that
// convention matching exactly, never a hand-varied encoding.
func chunkSourceKey(documentID, contentDigest, chunkerVersion string, chunkIndex int) map[string]string {
	return map[string]string{
		"document_id":     documentID,
		"content_digest":  contentDigest,
		"chunker_version": chunkerVersion,
		"chunk_index":     strconv.Itoa(chunkIndex),
	}
}

// backfillChunks calls request_refresh for each given chunk key against
// sourceID. Idempotent per key: request_refresh's own ON CONFLICT upsert
// on derived_value makes a repeated call for a key already enqueued at
// this source safe, but callers here only ever pass keys already
// confirmed (via scanBackfillCandidates' NOT EXISTS) to have no row yet
// at sourceID, so this never actually re-triggers an existing one.
func backfillChunks(ctx context.Context, tx pgx.Tx, sourceID int64, keys []chunkKey) error {
	for _, k := range keys {
		sourceKey := chunkSourceKey(k.documentID, k.contentDigest, k.chunkerVersion, k.chunkIndex)
		if _, err := tx.Exec(ctx, `SELECT helix_ai.request_refresh($1, $2)`, sourceID, sourceKey); err != nil {
			return fmt.Errorf("store: backfill request_refresh for %v: %w", k, err)
		}
	}
	return nil
}

type chunkKey struct {
	documentID, contentDigest, chunkerVersion string
	chunkIndex                                int
}

// scanBackfillCandidates finds up to backfillBatchLimit chunk keys
// (optionally restricted to one document) that have no derived_value
// row at all under sourceID yet -- the backfill-needed signal. "No row
// at any status" (not "no ready row") is deliberate: a key already
// enqueued pending/failed/stale at this source must not be re-enqueued,
// only one genuinely never scheduled for this binding.
func scanBackfillCandidates(ctx context.Context, tx pgx.Tx, sourceID int64, documentID *string) ([]chunkKey, error) {
	query := `
		SELECT c.document_id, c.content_digest, c.chunker_version, c.chunk_index
		FROM cortex.chunk c
		WHERE NOT EXISTS (
			SELECT 1 FROM helix_ai.derived_value dv
			WHERE dv.source_id = $1
			  AND (dv.source_key ->> 'document_id') = c.document_id::text
			  AND (dv.source_key ->> 'content_digest') = c.content_digest
			  AND (dv.source_key ->> 'chunker_version') = c.chunker_version
			  AND (dv.source_key ->> 'chunk_index') = c.chunk_index::text
		)
	`
	args := []any{sourceID}
	if documentID != nil {
		query += " AND c.document_id = $2"
		args = append(args, *documentID)
	}
	query += " LIMIT $" + strconv.Itoa(len(args)+1)
	args = append(args, backfillBatchLimit)

	rows, err := tx.Query(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("store: scan backfill candidates: %w", err)
	}
	defer rows.Close()
	var keys []chunkKey
	for rows.Next() {
		var k chunkKey
		if err := rows.Scan(&k.documentID, &k.contentDigest, &k.chunkerVersion, &k.chunkIndex); err != nil {
			return nil, fmt.Errorf("store: scan backfill candidate: %w", err)
		}
		keys = append(keys, k)
	}
	return keys, rows.Err()
}

// backfillDocument schedules backfill for exactly one document's chunks
// under sourceID. Called from IngestDocument's idempotent-no-op path:
// unchanged bytes under the same chunker version never touch
// cortex.chunk (ON CONFLICT DO NOTHING -- no row changes, no capture-
// trigger activity), so switching the active embedding binding and then
// re-POSTing the same unchanged document must not just return the old
// status without ever scheduling the new binding's embedding -- this is
// the explicit, bounded (one document) fix for that gap.
func (s *Store) backfillDocument(ctx context.Context, tx pgx.Tx, sourceID int64, documentID string) error {
	keys, err := scanBackfillCandidates(ctx, tx, sourceID, &documentID)
	if err != nil {
		return err
	}
	return backfillChunks(ctx, tx, sourceID, keys)
}

// EnsureActiveSourceID resolves (registering if necessary) the current
// active source_id for tenant, for callers outside an existing
// transaction -- cmd/cortex-api's --backfill-tenant mode, which has no
// HTTP request/collection context to derive one from.
func (s *Store) EnsureActiveSourceID(ctx context.Context, tenant string) (int64, error) {
	p := auth.Principal{Name: "backfill", Tenant: tenant}
	var sourceID int64
	err := s.withTenant(ctx, p, func(tx pgx.Tx) error {
		var err error
		sourceID, err = s.ensureSourceRegistered(ctx, tx)
		return err
	})
	return sourceID, err
}

// BackfillTenant schedules up to one batch of backfill work for the
// given tenant's chunks under sourceID and reports how many keys it
// scheduled. Callers loop until it returns 0 to drain a full backlog
// (e.g. a mock -> real binding cutover across every existing document)
// across several bounded passes -- safe to stop and resume at any
// point: a crash mid-loop leaves exactly the not-yet-scheduled keys
// still missing a derived_value row, which the next call finds again.
// Distinct from backfillDocument: this is an explicit, administrator-
// invoked operation (cmd/cortex-api's --backfill-tenant mode), never
// wired into a request's hot path, so a binding cutover's backlog scan
// cost is never paid by ordinary search/answer/ingest traffic.
func (s *Store) BackfillTenant(ctx context.Context, tenant string, sourceID int64) (int, error) {
	// request_refresh itself doesn't read helix.tenant (it trusts the
	// source row's own stored tenant); this is set only so RLS on
	// cortex.chunk -- tenant-only by design, see migrations/001 -- shows
	// the right tenant's rows to scanBackfillCandidates. The principal
	// name is irrelevant here: chunk's RLS policy never checks it.
	p := auth.Principal{Name: "backfill", Tenant: tenant}
	var scheduled int
	err := s.withTenant(ctx, p, func(tx pgx.Tx) error {
		keys, err := scanBackfillCandidates(ctx, tx, sourceID, nil)
		if err != nil {
			return err
		}
		if err := backfillChunks(ctx, tx, sourceID, keys); err != nil {
			return err
		}
		scheduled = len(keys)
		return nil
	})
	return scheduled, err
}
