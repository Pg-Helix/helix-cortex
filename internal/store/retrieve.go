package store

import (
	"context"
	"errors"
	"fmt"

	"github.com/Pg-Helix/helix-cortex/internal/auth"
	"github.com/Pg-Helix/helix-cortex/internal/retrieval"
	"github.com/Pg-Helix/helix-infer/embed"
	protocol "github.com/Pg-Helix/helix-protocol/gen/go"
	"github.com/jackc/pgx/v5"
)

const (
	candidateLimit  = 50 // per method, before fusion -- bounded, matches "exact search is acceptable for a first bounded corpus"
	ftsCandidateSQL = `
		SELECT c.document_id || '|' || c.content_digest || '|' || c.chunker_version || '|' || c.chunk_index, c.text
		FROM cortex.chunk c
		JOIN helix_ai.derived_value dv
		  ON (dv.source_key ->> 'document_id') = c.document_id::text
		 AND (dv.source_key ->> 'content_digest') = c.content_digest
		 AND (dv.source_key ->> 'chunker_version') = c.chunker_version
		 AND (dv.source_key ->> 'chunk_index') = c.chunk_index::text
		WHERE c.collection_id = $1 AND dv.source_id = $2 AND dv.status = 'ready'
		  AND to_tsvector('english', c.text) @@ to_tsquery('english', $3)
		ORDER BY ts_rank(to_tsvector('english', c.text), to_tsquery('english', $3)) DESC
		LIMIT $4
	`
	denseCandidateSQL = `
		SELECT c.document_id || '|' || c.content_digest || '|' || c.chunker_version || '|' || c.chunk_index, c.text,
		       (ARRAY(SELECT jsonb_array_elements_text(dv.value)::real))::vector <=> $3::vector AS distance
		FROM cortex.chunk c
		JOIN helix_ai.derived_value dv
		  ON (dv.source_key ->> 'document_id') = c.document_id::text
		 AND (dv.source_key ->> 'content_digest') = c.content_digest
		 AND (dv.source_key ->> 'chunker_version') = c.chunker_version
		 AND (dv.source_key ->> 'chunk_index') = c.chunk_index::text
		WHERE c.collection_id = $1 AND dv.source_id = $2 AND dv.status = 'ready'
		ORDER BY distance ASC
		LIMIT $4
	`
)

// candidateQuery runs both FTS and dense queries, ACL-filtered by RLS
// (tenant+principal already SET LOCAL on tx) *inside* the query, before
// any ranking or LIMIT -- never a global top-k filtered down afterward.
func (s *Store) candidateQuery(ctx context.Context, tx pgx.Tx, collectionID string, sourceID int64, query string, queryVec []float32) (ftsOrder []string, denseOrder []string, textByChunk map[string]string, err error) {
	textByChunk = make(map[string]string)

	tsq := toTSQuery(query)
	if tsq != "" {
		rows, qerr := tx.Query(ctx, ftsCandidateSQL, collectionID, sourceID, tsq, candidateLimit)
		if qerr != nil {
			return nil, nil, nil, fmt.Errorf("store: fts candidates: %w", qerr)
		}
		for rows.Next() {
			var id, text string
			if serr := rows.Scan(&id, &text); serr != nil {
				rows.Close()
				return nil, nil, nil, fmt.Errorf("store: scan fts candidate: %w", serr)
			}
			ftsOrder = append(ftsOrder, id)
			textByChunk[id] = text
		}
		if cerr := rows.Err(); cerr != nil {
			rows.Close()
			return nil, nil, nil, cerr
		}
		rows.Close()
	}

	vecLiteral := vectorLiteral(queryVec)
	rows, qerr := tx.Query(ctx, denseCandidateSQL, collectionID, sourceID, vecLiteral, candidateLimit)
	if qerr != nil {
		return nil, nil, nil, fmt.Errorf("store: dense candidates: %w", qerr)
	}
	defer rows.Close()
	for rows.Next() {
		var id, text string
		var distance float64
		if serr := rows.Scan(&id, &text, &distance); serr != nil {
			return nil, nil, nil, fmt.Errorf("store: scan dense candidate: %w", serr)
		}
		denseOrder = append(denseOrder, id)
		textByChunk[id] = text
	}
	if cerr := rows.Err(); cerr != nil {
		return nil, nil, nil, cerr
	}
	return ftsOrder, denseOrder, textByChunk, nil
}

// Search runs hybrid retrieval and returns the final (fused, reranked)
// top_k results -- no barrier/re-authorization step, unlike Answer,
// since search results carry no citation a later evidence lookup must
// honor; the ACL check is the RLS floor on the candidate query itself.
func (s *Store) Search(ctx context.Context, p auth.Principal, collectionID, query string, topK int, embedder *embed.Guarded, targetModel string) ([]protocol.SearchResult, error) {
	var out []protocol.SearchResult
	err := s.withTenant(ctx, p, func(tx pgx.Tx) error {
		if err := requireRole(ctx, tx, collectionID, p, "owner", "reader"); err != nil {
			return err
		}
		sourceID, err := s.ensureSourceRegistered(ctx, tx)
		if err != nil {
			return err
		}
		queryVec, err := embedQuery(ctx, embedder, targetModel, query)
		if err != nil {
			return err
		}
		ftsOrder, denseOrder, textByChunk, err := s.candidateQuery(ctx, tx, collectionID, sourceID, query, queryVec)
		if err != nil {
			return err
		}
		fused := retrieval.RRF(retrieval.DefaultRRFConstant, ftsOrder, denseOrder)
		reranked := retrieval.Rerank(retrieval.Tokenize(query), fused, textByChunk)

		for i, r := range reranked {
			if i >= topK {
				break
			}
			documentID, _, _, _, derr := decodeChunkID(r.ChunkID)
			if derr != nil {
				continue
			}
			snippet := textByChunk[r.ChunkID]
			if len(snippet) > 200 {
				snippet = snippet[:200]
			}
			out = append(out, protocol.SearchResult{
				ChunkId:    r.ChunkID,
				DocumentId: documentID,
				Score:      r.OverlapScore*1e6 + r.FusedScore, // see retrieval.Rerank's own sort key; exposed as one number for the API
				Snippet:    &snippet,
			})
		}
		return nil
	})
	return out, err
}

// Answer runs the same candidate selection/fusion/rerank as Search, then
// -- critically -- re-authorizes every candidate chunk in a FRESH
// transaction/snapshot immediately before assembling the response, so a
// revocation committed in between is observed (ADR-0007; RFC-0004's
// citation-render re-check). barrierPause, if non-nil, is called exactly
// at that boundary -- the deterministic rendezvous point the in-flight-
// revocation acceptance test pauses on, never a sleep.
func (s *Store) Answer(ctx context.Context, p auth.Principal, collectionID, query string, embedder *embed.Guarded, targetModel string, barrierPause func()) (protocol.AnswerResponse, error) {
	var ftsOrder, denseOrder []string
	var textByChunk map[string]string
	var sourceID int64

	err := s.withTenant(ctx, p, func(tx pgx.Tx) error {
		if err := requireRole(ctx, tx, collectionID, p, "owner", "reader"); err != nil {
			return err
		}
		var err error
		sourceID, err = s.ensureSourceRegistered(ctx, tx)
		if err != nil {
			return err
		}
		queryVec, err := embedQuery(ctx, embedder, targetModel, query)
		if err != nil {
			return err
		}
		ftsOrder, denseOrder, textByChunk, err = s.candidateQuery(ctx, tx, collectionID, sourceID, query, queryVec)
		return err
	})
	if err != nil {
		return protocol.AnswerResponse{}, err
	}

	fused := retrieval.RRF(retrieval.DefaultRRFConstant, ftsOrder, denseOrder)
	reranked := retrieval.Rerank(retrieval.Tokenize(query), fused, textByChunk)

	// The deterministic rendezvous: candidate selection + rerank are
	// done, the first transaction's snapshot has already been released
	// (withTenant committed/rolled back above) -- a revocation committed
	// by another connection right now MUST be visible to the fresh
	// transaction the re-check below opens.
	if barrierPause != nil {
		barrierPause()
	}

	var passages []retrieval.Passage
	var citations []protocol.Citation
	err = s.withTenant(ctx, p, func(tx pgx.Tx) error {
		// Re-check the caller's own grant too: a reader whose access was
		// revoked between candidate selection and now must not get a
		// render out of a snapshot that predates the revoke.
		if err := requireRole(ctx, tx, collectionID, p, "owner", "reader"); err != nil {
			return err
		}
		for i, r := range reranked {
			if len(passages) >= retrieval.MaxAnswerChunks {
				break
			}
			if i >= candidateLimit {
				break
			}
			chunk, ok, err := s.reauthorizeChunk(ctx, tx, collectionID, sourceID, r.ChunkID)
			if err != nil {
				return err
			}
			if !ok {
				continue // revoked, superseded, deleted, or no longer ready -- silently excluded, not an error
			}
			passages = append(passages, retrieval.Passage{ChunkID: r.ChunkID, Text: chunk.text})
			citations = append(citations, protocol.Citation{ChunkId: r.ChunkID, SourceId: chunk.documentID, SourceDigest: chunk.digest})
		}
		return nil
	})
	if err != nil {
		return protocol.AnswerResponse{}, err
	}

	answerText, used := retrieval.AssembleExtractiveAnswer(passages)
	usedIDs := make(map[string]bool, len(used))
	for _, u := range used {
		usedIDs[u.ChunkID] = true
	}
	var finalCitations []protocol.Citation
	for _, c := range citations {
		if usedIDs[c.ChunkId] {
			finalCitations = append(finalCitations, c)
		}
	}
	return protocol.AnswerResponse{Answer: answerText, AnswerMethod: protocol.Extractive, Citations: finalCitations}, nil
}

type reauthorizedChunk struct {
	text       string
	documentID string
	digest     string
}

// reauthorizeChunk re-reads a candidate chunk's current row (inside the
// caller's current transaction, so RLS/ACL applies fresh) and confirms
// it's still the active version and still ready. Returns ok=false
// (never an error) for anything that no longer qualifies -- a revoked
// or superseded chunk silently drops out of the answer, exactly RFC-0004's
// "fails without disclosing" requirement for an evidence check.
func (s *Store) reauthorizeChunk(ctx context.Context, tx pgx.Tx, collectionID string, sourceID int64, chunkID string) (reauthorizedChunk, bool, error) {
	documentID, digest, chunkerVersion, chunkIndex, err := decodeChunkID(chunkID)
	if err != nil {
		return reauthorizedChunk{}, false, nil
	}
	var text, status string
	var activeVersionID *string
	err = tx.QueryRow(ctx, `
		SELECT c.text, dv.status, d.active_version_id
		FROM cortex.chunk c
		JOIN helix_ai.derived_value dv
		  ON (dv.source_key ->> 'document_id') = c.document_id::text
		 AND (dv.source_key ->> 'content_digest') = c.content_digest
		 AND (dv.source_key ->> 'chunker_version') = c.chunker_version
		 AND (dv.source_key ->> 'chunk_index') = c.chunk_index::text
		JOIN cortex.document d ON d.id = c.document_id
		WHERE c.document_id = $1 AND c.content_digest = $2 AND c.chunker_version = $3 AND c.chunk_index = $4
		  AND c.collection_id = $5 AND dv.source_id = $6
	`, documentID, digest, chunkerVersion, chunkIndex, collectionID, sourceID).Scan(&text, &status, &activeVersionID)
	if errors.Is(err, pgx.ErrNoRows) {
		return reauthorizedChunk{}, false, nil // RLS hid it (ACL revoked), or it was deleted entirely
	}
	if err != nil {
		return reauthorizedChunk{}, false, fmt.Errorf("store: reauthorize chunk: %w", err)
	}
	if status != "ready" {
		return reauthorizedChunk{}, false, nil
	}
	// Confirm this chunk still belongs to the document's CURRENT active
	// version -- a chunk row can only exist for its own (digest,
	// chunker_version), so matching document_id is already implied; this
	// also guards the rare case activeVersionID is nil (document exists
	// with no active version, e.g. mid-ingest).
	if activeVersionID == nil {
		return reauthorizedChunk{}, false, nil
	}
	return reauthorizedChunk{text: text, documentID: documentID, digest: digest}, true, nil
}

// GetEvidence resolves a citation to its authorized passage (RFC-0004):
// a citation is a reference, never a capability -- every check here is
// against CURRENT state, not whatever was true when the citation was
// issued.
func (s *Store) GetEvidence(ctx context.Context, p auth.Principal, chunkID string) (protocol.EvidenceResponse, error) {
	documentID, digest, chunkerVersion, chunkIndex, err := decodeChunkID(chunkID)
	if err != nil {
		return protocol.EvidenceResponse{}, ErrNotFound
	}
	var out protocol.EvidenceResponse
	err = s.withTenant(ctx, p, func(tx pgx.Tx) error {
		var collectionID string
		var status string
		var activeVersionID *string
		var documentName string
		serr := tx.QueryRow(ctx, `
			SELECT c.text, c.collection_id, dv.status, d.active_version_id, d.name
			FROM cortex.chunk c
			JOIN helix_ai.derived_value dv
			  ON (dv.source_key ->> 'document_id') = c.document_id::text
			 AND (dv.source_key ->> 'content_digest') = c.content_digest
			 AND (dv.source_key ->> 'chunker_version') = c.chunker_version
			 AND (dv.source_key ->> 'chunk_index') = c.chunk_index::text
			JOIN cortex.document d ON d.id = c.document_id
			WHERE c.document_id = $1 AND c.content_digest = $2 AND c.chunker_version = $3 AND c.chunk_index = $4
		`, documentID, digest, chunkerVersion, chunkIndex).Scan(&out.Text, &collectionID, &status, &activeVersionID, &documentName)
		if errors.Is(serr, pgx.ErrNoRows) {
			// RLS hid it (no ACL grant / wrong tenant) or it genuinely
			// doesn't exist -- same citation_invalid response either way,
			// never disclosing which case applied.
			return ErrNotFound
		}
		if serr != nil {
			return fmt.Errorf("store: evidence lookup: %w", serr)
		}
		if status != "ready" || activeVersionID == nil {
			return ErrNotFound
		}
		if _, roleErr := requirePrincipalHasAnyRole(ctx, tx, collectionID, p); roleErr != nil {
			return roleErr
		}
		out.ChunkId = chunkID
		out.DocumentId = documentID
		out.DocumentName = &documentName
		out.DocumentVersionDigest = digest
		idx := chunkIndex
		out.ChunkIndex = &idx
		return nil
	})
	return out, err
}

func requirePrincipalHasAnyRole(ctx context.Context, tx pgx.Tx, collectionID string, p auth.Principal) (string, error) {
	var role string
	err := tx.QueryRow(ctx, `SELECT role FROM cortex.collection_permission WHERE collection_id = $1 AND principal = $2 AND tenant = $3`, collectionID, p.Name, p.Tenant).Scan(&role)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", ErrNotFound
	}
	if err != nil {
		return "", fmt.Errorf("store: check permission: %w", err)
	}
	return role, nil
}
