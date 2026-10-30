package store

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"

	"github.com/Pg-Helix/helix-cortex/internal/auth"
	"github.com/Pg-Helix/helix-cortex/internal/ingest"
	protocol "github.com/Pg-Helix/helix-protocol/gen/go"
	"github.com/jackc/pgx/v5"
)

// ErrUnsupportedContentType, ErrTooLarge mirror the request-level codes
// errors/codes.md adds for Workflow 2 (unsupported_content_type /
// content_too_large / chunk_limit_exceeded) -- the HTTP layer maps them.
var ErrUnsupportedContentType = errors.New("store: unsupported content type")

// IngestDocument chunks content deterministically and atomically
// replaces or idempotently no-ops the named document's chunks (ADR-0007
// decision 3). Requires the owner grant on collectionID.
func (s *Store) IngestDocument(ctx context.Context, p auth.Principal, collectionID, name, contentType, content string) (protocol.Document, error) {
	if contentType != "text/plain" && contentType != "text/markdown" {
		return protocol.Document{}, ErrUnsupportedContentType
	}
	chunks, err := ingest.Split(content)
	if err != nil {
		return protocol.Document{}, err // *ingest.ErrTooLarge or a plain validation error; HTTP layer inspects it
	}
	digest := sha256.Sum256([]byte(content))
	digestHex := hex.EncodeToString(digest[:])

	var out protocol.Document
	err = s.withTenant(ctx, p, func(tx pgx.Tx) error {
		if err := requireRole(ctx, tx, collectionID, p, "owner"); err != nil {
			return err
		}
		sourceID, err := s.ensureSourceRegistered(ctx, tx)
		if err != nil {
			return err
		}

		documentID, existingChunkerVersion, existingVersionID, existingDigest, isNew, err := s.resolveDocumentForIngest(ctx, tx, collectionID, p.Tenant, name, contentType)
		if err != nil {
			return err
		}
		switch {
		case isNew:
			// Nothing to compare against or supersede -- fall through to
			// create the first version below.
		case existingDigest != nil && *existingDigest == digestHex && existingChunkerVersion != nil && *existingChunkerVersion == ingest.ChunkerVersion:
			// Idempotent no-op for CONTENT: identical bytes under the same
			// chunker version already active, so no row changes, no
			// capture-trigger activity, no new job for that reason. But an
			// unchanged document can still be missing embedding work under
			// the CURRENTLY active binding (sourceID), e.g. right after a
			// mock -> real binding cutover: unchanged bytes never touch
			// cortex.chunk, so nothing re-fires the capture trigger under
			// the new binding on their own. Bounded to this one document's
			// chunks (not a tenant-wide scan) -- cheap, and resumable
			// across repeated re-ingests if a batch limit is ever hit.
			if err := s.backfillDocument(ctx, tx, sourceID, documentID); err != nil {
				return err
			}
			out, err = s.fetchDocument(ctx, tx, documentID, sourceID)
			return err
		default:
			if err := s.supersedeDocument(ctx, tx, documentID, *existingVersionID, *existingChunkerVersion, *existingDigest); err != nil {
				return err
			}
		}

		versionID, err := s.createVersion(ctx, tx, documentID, p.Tenant, digestHex, len(content), len(chunks))
		if err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `UPDATE cortex.document SET chunker_version = $1, active_version_id = $2 WHERE id = $3`,
			ingest.ChunkerVersion, versionID, documentID); err != nil {
			return fmt.Errorf("store: activate version: %w", err)
		}

		for _, c := range chunks {
			_, err := tx.Exec(ctx, `
				INSERT INTO cortex.chunk (document_id, content_digest, chunker_version, chunk_index, document_version_id, tenant, collection_id, text, char_start, char_end)
				VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
				ON CONFLICT (document_id, content_digest, chunker_version, chunk_index) DO NOTHING
			`, documentID, digestHex, ingest.ChunkerVersion, c.Index, versionID, p.Tenant, collectionID, c.Text, c.CharStart, c.CharEnd)
			if err != nil {
				return fmt.Errorf("store: insert chunk %d: %w", c.Index, err)
			}
		}
		// registration is the side effect that matters for a freshly
		// (re)activated version; kernel's own capture trigger (fired by
		// the INSERTs above) does the enqueueing under sourceID.
		out, err = s.fetchDocument(ctx, tx, documentID, sourceID)
		return err
	})
	return out, err
}

// resolveDocumentForIngest finds-or-creates the named document, safe
// under concurrent ingests of the SAME name (first-creation races and
// competing content updates both land here, since a re-POST to an
// existing name is just an update per ADR-0007 decision 3's document
// identity). FOR UPDATE on the existing-row path serializes concurrent
// updaters against each other -- a second request blocks until the
// first commits, then sees its result and proceeds correctly (an
// idempotent no-op or a genuine supersede against the WINNER's content,
// never racing past supersedeDocument's tombstoning of the other
// version's chunks). The not-found path races INSERT...ON CONFLICT DO
// NOTHING for first creation; Postgres itself serializes two concurrent
// inserters of the same (collection_id, name) at the unique index, so
// at most one iteration of this loop ever actually creates a row -- the
// loser re-SELECTs FOR UPDATE and resumes as the normal existing-row
// case instead of silently treating itself as "created".
func (s *Store) resolveDocumentForIngest(ctx context.Context, tx pgx.Tx, collectionID, tenant, name, contentType string) (documentID string, existingChunkerVersion, existingVersionID, existingDigest *string, isNew bool, err error) {
	for attempt := 0; attempt < 2; attempt++ {
		selErr := tx.QueryRow(ctx, `
			SELECT d.id, d.chunker_version, d.active_version_id, dv.digest
			FROM cortex.document d
			LEFT JOIN cortex.document_version dv ON dv.id = d.active_version_id
			WHERE d.collection_id = $1 AND d.name = $2
			FOR UPDATE OF d
		`, collectionID, name).Scan(&documentID, &existingChunkerVersion, &existingVersionID, &existingDigest)
		if selErr == nil {
			return documentID, existingChunkerVersion, existingVersionID, existingDigest, false, nil
		}
		if !errors.Is(selErr, pgx.ErrNoRows) {
			return "", nil, nil, nil, false, fmt.Errorf("store: look up document: %w", selErr)
		}

		var id string
		insErr := tx.QueryRow(ctx, `
			INSERT INTO cortex.document (collection_id, tenant, name, content_type, chunker_version)
			VALUES ($1, $2, $3, $4, '')
			ON CONFLICT (collection_id, name) DO NOTHING
			RETURNING id
		`, collectionID, tenant, name, contentType).Scan(&id)
		if insErr == nil {
			return id, nil, nil, nil, true, nil
		}
		if !errors.Is(insErr, pgx.ErrNoRows) {
			return "", nil, nil, nil, false, fmt.Errorf("store: create document: %w", insErr)
		}
		// Lost the race to a concurrent first creation: loop back and
		// re-SELECT ... FOR UPDATE, which now finds (and locks) the
		// winner's committed row.
	}
	return "", nil, nil, nil, false, fmt.Errorf("store: could not resolve document %q in collection %q after retrying the creation race", name, collectionID)
}

func (s *Store) createVersion(ctx context.Context, tx pgx.Tx, documentID, tenant, digest string, byteSize, chunkCount int) (string, error) {
	var id string
	err := tx.QueryRow(ctx, `
		INSERT INTO cortex.document_version (document_id, tenant, digest, chunker_version, byte_size, chunk_count)
		VALUES ($1, $2, $3, $4, $5, $6)
		ON CONFLICT (document_id, digest, chunker_version) DO UPDATE SET chunk_count = EXCLUDED.chunk_count
		RETURNING id
	`, documentID, tenant, digest, ingest.ChunkerVersion, byteSize, chunkCount).Scan(&id)
	if err != nil {
		return "", fmt.Errorf("store: create document version: %w", err)
	}
	return id, nil
}

// supersedeDocument deletes the old active version's chunk rows (firing
// helix-kernel's existing capture trigger, which tombstones their
// derived_value rows) and marks the version superseded. Called only when
// content or chunker version genuinely changed -- the idempotent-no-op
// path above never reaches here.
func (s *Store) supersedeDocument(ctx context.Context, tx pgx.Tx, documentID, oldVersionID, oldChunkerVersion, oldDigest string) error {
	if _, err := tx.Exec(ctx, `
		DELETE FROM cortex.chunk WHERE document_id = $1 AND content_digest = $2 AND chunker_version = $3
	`, documentID, oldDigest, oldChunkerVersion); err != nil {
		return fmt.Errorf("store: delete superseded chunks: %w", err)
	}
	if _, err := tx.Exec(ctx, `UPDATE cortex.document_version SET superseded_at = now() WHERE id = $1`, oldVersionID); err != nil {
		return fmt.Errorf("store: mark version superseded: %w", err)
	}
	return nil
}

// DeleteDocument removes a document and all its chunks (tombstoning
// their derived_value rows via the same capture-trigger path) --
// immediate, not a soft-delete flag, so a subsequent re-ingest at the
// same key is a genuine reactivation (generation bump), never silently
// matching a job issued before this delete (ADR-0007 decision 3).
func (s *Store) DeleteDocument(ctx context.Context, p auth.Principal, collectionID, documentID string) error {
	return s.withTenant(ctx, p, func(tx pgx.Tx) error {
		if err := requireRole(ctx, tx, collectionID, p, "owner"); err != nil {
			return err
		}
		tag, err := tx.Exec(ctx, `DELETE FROM cortex.document WHERE id = $1 AND collection_id = $2`, documentID, collectionID)
		if err != nil {
			return fmt.Errorf("store: delete document: %w", err)
		}
		if tag.RowsAffected() == 0 {
			return ErrNotFound
		}
		return nil
	})
}

func (s *Store) GetDocument(ctx context.Context, p auth.Principal, collectionID, documentID string) (protocol.Document, error) {
	var out protocol.Document
	err := s.withTenant(ctx, p, func(tx pgx.Tx) error {
		if err := requireRole(ctx, tx, collectionID, p, "owner", "reader"); err != nil {
			return err
		}
		sourceID, err := s.ensureSourceRegistered(ctx, tx)
		if err != nil {
			return err
		}
		out, err = s.fetchDocument(ctx, tx, documentID, sourceID)
		return err
	})
	return out, err
}

// fetchDocument reads the document's metadata plus its aggregate chunk
// readiness, joining into helix_ai.derived_value using the same ->>
// extraction convention get_job_input/request_refresh use kernel-side
// (never hand-reconstructing the jsonb source_key, which risks drifting
// from however _row_key_jsonb actually serializes it). sourceID scopes
// the readiness join to the CURRENTLY active embedding binding: without
// it, a document with sidecars under more than one binding (e.g. mid
// mock -> real cutover) double-counts "total" and can report Ready from
// an old binding's already-published value while the active binding's
// own chunks are still pending.
func (s *Store) fetchDocument(ctx context.Context, tx pgx.Tx, documentID string, sourceID int64) (protocol.Document, error) {
	var out protocol.Document
	var activeVersionID *string
	err := tx.QueryRow(ctx, `
		SELECT id, collection_id, tenant, name, content_type, chunker_version, active_version_id, created_at
		FROM cortex.document WHERE id = $1
	`, documentID).Scan(&out.Id, &out.CollectionId, &out.Tenant, &out.Name, &out.ContentType, &out.ChunkerVersion, &activeVersionID, &out.CreatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return out, ErrNotFound
	}
	if err != nil {
		return out, fmt.Errorf("store: fetch document: %w", err)
	}
	if activeVersionID == nil {
		out.Status = protocol.Pending
		return out, nil
	}
	err = tx.QueryRow(ctx, `
		SELECT dv.digest, dv.byte_size, dv.chunk_count
		FROM cortex.document_version dv WHERE dv.id = $1
	`, *activeVersionID).Scan(&out.Digest, &out.ByteSize, &out.ChunkCount)
	if err != nil {
		return out, fmt.Errorf("store: fetch active version: %w", err)
	}

	var ready, total int
	err = tx.QueryRow(ctx, `
		SELECT count(*) FILTER (WHERE dv.status = 'ready'), count(*)
		FROM cortex.chunk c
		JOIN helix_ai.derived_value dv
		  ON (dv.source_key ->> 'document_id') = c.document_id::text
		 AND (dv.source_key ->> 'content_digest') = c.content_digest
		 AND (dv.source_key ->> 'chunker_version') = c.chunker_version
		 AND (dv.source_key ->> 'chunk_index') = c.chunk_index::text
		WHERE c.document_id = $1 AND c.content_digest = $2 AND c.chunker_version = $3 AND dv.source_id = $4
	`, documentID, out.Digest, out.ChunkerVersion, sourceID).Scan(&ready, &total)
	if err != nil {
		return out, fmt.Errorf("store: aggregate chunk status: %w", err)
	}
	switch {
	case total == 0:
		out.Status = protocol.Pending
	case ready == total:
		out.Status = protocol.Ready
	default:
		out.Status = protocol.Pending
	}
	return out, nil
}
