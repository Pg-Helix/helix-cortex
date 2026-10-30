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
	candidateLimit = 50 // per method, before fusion -- bounded, matches "exact search is acceptable for a first bounded corpus"
	// Both candidate queries carry their OWN explicit collection_permission
	// join and active-version equality check, not just a `collection_id =
	// $1` filter trusting an earlier one-time role check in the same
	// transaction: under READ COMMITTED, a revoke committed between that
	// earlier check and this statement -- or, now that query embedding
	// runs outside any transaction, during the real network call before
	// this statement's own fresh transaction even begins -- must still be
	// enforced INSIDE this query, before ranking/LIMIT, never after.
	ftsCandidateSQL = `
		SELECT c.document_id || '|' || c.content_digest || '|' || c.chunker_version || '|' || c.chunk_index, c.text
		FROM cortex.chunk c
		JOIN cortex.document d ON d.id = c.document_id AND d.active_version_id = c.document_version_id
		JOIN cortex.collection_permission cp
		  ON cp.collection_id = c.collection_id
		 AND cp.tenant = current_setting('helix.tenant', true)
		 AND cp.principal = current_setting('helix.principal', true)
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
		JOIN cortex.document d ON d.id = c.document_id AND d.active_version_id = c.document_version_id
		JOIN cortex.collection_permission cp
		  ON cp.collection_id = c.collection_id
		 AND cp.tenant = current_setting('helix.tenant', true)
		 AND cp.principal = current_setting('helix.principal', true)
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

// candidateQuery runs both FTS and dense queries, each independently
// ACL- and active-version-filtered inside the query itself (see the SQL
// above), before any ranking or LIMIT -- never a global top-k filtered
// down afterward. A thin wrapper over the two single-method halves
// below, which the evaluation instrumentation (SearchStage) also calls
// independently -- never a second, separately-maintained implementation
// of either query.
func (s *Store) candidateQuery(ctx context.Context, tx pgx.Tx, collectionID string, sourceID int64, query string, queryVec []float32) (ftsOrder []string, denseOrder []string, textByChunk map[string]string, err error) {
	textByChunk = make(map[string]string)
	ftsOrder, err = ftsCandidates(ctx, tx, collectionID, sourceID, query, textByChunk)
	if err != nil {
		return nil, nil, nil, err
	}
	denseOrder, err = denseCandidates(ctx, tx, collectionID, sourceID, queryVec, textByChunk)
	if err != nil {
		return nil, nil, nil, err
	}
	return ftsOrder, denseOrder, textByChunk, nil
}

func ftsCandidates(ctx context.Context, tx pgx.Tx, collectionID string, sourceID int64, query string, textByChunk map[string]string) ([]string, error) {
	tsq := toTSQuery(query)
	if tsq == "" {
		return nil, nil
	}
	rows, qerr := tx.Query(ctx, ftsCandidateSQL, collectionID, sourceID, tsq, candidateLimit)
	if qerr != nil {
		return nil, fmt.Errorf("store: fts candidates: %w", qerr)
	}
	defer rows.Close()
	var order []string
	for rows.Next() {
		var id, text string
		if serr := rows.Scan(&id, &text); serr != nil {
			return nil, fmt.Errorf("store: scan fts candidate: %w", serr)
		}
		order = append(order, id)
		textByChunk[id] = text
	}
	return order, rows.Err()
}

func denseCandidates(ctx context.Context, tx pgx.Tx, collectionID string, sourceID int64, queryVec []float32, textByChunk map[string]string) ([]string, error) {
	vecLiteral := vectorLiteral(queryVec)
	rows, qerr := tx.Query(ctx, denseCandidateSQL, collectionID, sourceID, vecLiteral, candidateLimit)
	if qerr != nil {
		return nil, fmt.Errorf("store: dense candidates: %w", qerr)
	}
	defer rows.Close()
	var order []string
	for rows.Next() {
		var id, text string
		var distance float64
		if serr := rows.Scan(&id, &text, &distance); serr != nil {
			return nil, fmt.Errorf("store: scan dense candidate: %w", serr)
		}
		order = append(order, id)
		textByChunk[id] = text
	}
	return order, rows.Err()
}

// retrievalCandidates runs the shared prefix of Search and Answer: a
// role check + source registration (one short transaction), THEN query
// embedding -- the real network/model call -- OUTSIDE any transaction,
// so a pooled connection is never held while waiting on it, THEN
// candidate selection/fusion/rerank in a fresh transaction (its own
// role re-check, since real time -- and a real revoke -- may have
// passed during the network call). The ACL/active-version enforcement
// candidateQuery's own SQL performs is what makes this fresh re-check
// matter: it is not just a courtesy error-shape check.
func (s *Store) retrievalCandidates(ctx context.Context, p auth.Principal, collectionID, query string, embedder *embed.Guarded, targetModel string) (sourceID int64, reranked []retrieval.RerankedChunk, textByChunk map[string]string, err error) {
	err = s.withTenant(ctx, p, func(tx pgx.Tx) error {
		if err := requireRole(ctx, tx, collectionID, p, "owner", "reader"); err != nil {
			return err
		}
		var rerr error
		sourceID, rerr = s.ensureSourceRegistered(ctx, tx)
		return rerr
	})
	if err != nil {
		return 0, nil, nil, err
	}

	queryVec, err := embedQuery(ctx, embedder, targetModel, s.ChunkModelID, query)
	if err != nil {
		return 0, nil, nil, err
	}

	var ftsOrder, denseOrder []string
	err = s.withTenant(ctx, p, func(tx pgx.Tx) error {
		if err := requireRole(ctx, tx, collectionID, p, "owner", "reader"); err != nil {
			return err
		}
		var qerr error
		ftsOrder, denseOrder, textByChunk, qerr = s.candidateQuery(ctx, tx, collectionID, sourceID, query, queryVec)
		return qerr
	})
	if err != nil {
		return 0, nil, nil, err
	}

	fused := retrieval.RRF(retrieval.DefaultRRFConstant, ftsOrder, denseOrder)
	reranked = s.rankFused(query, fused, textByChunk)
	return sourceID, reranked, textByChunk, nil
}

// rankFused applies the deterministic lexical-overlap reranker to
// fused's RRF order, unless s.DisableRerank -- see Store.DisableRerank's
// own doc comment for why this opt-out exists and the measured
// evidence behind it.
func (s *Store) rankFused(query string, fused []retrieval.ScoredChunk, textByChunk map[string]string) []retrieval.RerankedChunk {
	if s.DisableRerank {
		out := make([]retrieval.RerankedChunk, len(fused))
		for i, f := range fused {
			out[i] = retrieval.RerankedChunk{ChunkID: f.ChunkID, FusedScore: f.Score}
		}
		return out
	}
	return retrieval.Rerank(retrieval.Tokenize(query), fused, textByChunk)
}

// Search runs hybrid retrieval and returns the final (fused, reranked)
// top_k results, re-authorizing each candidate in a FRESH transaction
// immediately before rendering any snippet -- the same two-phase shape
// Answer uses, for the same reason: a revocation committed after
// candidate selection (including during the query-embedding network
// call) must be observed before any text is returned, not just relied
// on as a one-time check at the top.
func (s *Store) Search(ctx context.Context, p auth.Principal, collectionID, query string, topK int, embedder *embed.Guarded, targetModel string) ([]protocol.SearchResult, error) {
	sourceID, reranked, _, err := s.retrievalCandidates(ctx, p, collectionID, query, embedder, targetModel)
	if err != nil {
		return nil, err
	}

	var out []protocol.SearchResult
	err = s.withTenant(ctx, p, func(tx pgx.Tx) error {
		if err := requireRole(ctx, tx, collectionID, p, "owner", "reader"); err != nil {
			return err
		}
		for i, r := range reranked {
			if len(out) >= topK {
				break
			}
			if i >= candidateLimit {
				break
			}
			chunk, ok, rerr := s.reauthorizeChunk(ctx, tx, collectionID, sourceID, r.ChunkID)
			if rerr != nil {
				return rerr
			}
			if !ok {
				continue // revoked, superseded, deleted, or no longer ready -- silently excluded, not an error
			}
			snippet := chunk.text
			if len(snippet) > 200 {
				snippet = snippet[:200]
			}
			out = append(out, protocol.SearchResult{
				ChunkId:    r.ChunkID,
				DocumentId: chunk.documentID,
				Score:      r.OverlapScore*1e6 + r.FusedScore, // see retrieval.Rerank's own sort key; exposed as one number for the API
				Snippet:    &snippet,
			})
		}
		return nil
	})
	return out, err
}

// SearchStage is test-only evaluation instrumentation (helix-labs/eval),
// never reachable unless the caller is gated the same way the in-flight-
// revocation barrier is (CORTEX_TEST_HOOKS=1 -- see internal/api). It
// returns ONE intermediate stage of hybrid retrieval instead of the
// final served ranking, by calling the SAME ftsCandidates/
// denseCandidates/retrieval.RRF/retrieval.Rerank Search and Answer use
// -- never a second, separately-maintained implementation that could
// silently drift from what callers actually receive. stage is one of
// "fts-only", "dense-only", "hybrid-prererank", "final".
func (s *Store) SearchStage(ctx context.Context, p auth.Principal, collectionID, query string, topK int, embedder *embed.Guarded, targetModel, stage string) ([]protocol.SearchResult, error) {
	var sourceID int64
	err := s.withTenant(ctx, p, func(tx pgx.Tx) error {
		if err := requireRole(ctx, tx, collectionID, p, "owner", "reader"); err != nil {
			return err
		}
		var rerr error
		sourceID, rerr = s.ensureSourceRegistered(ctx, tx)
		return rerr
	})
	if err != nil {
		return nil, err
	}

	var queryVec []float32
	if stage != "fts-only" {
		queryVec, err = embedQuery(ctx, embedder, targetModel, s.ChunkModelID, query)
		if err != nil {
			return nil, err
		}
	}

	textByChunk := make(map[string]string)
	var ftsOrder, denseOrder []string
	err = s.withTenant(ctx, p, func(tx pgx.Tx) error {
		if err := requireRole(ctx, tx, collectionID, p, "owner", "reader"); err != nil {
			return err
		}
		var rerr error
		switch stage {
		case "fts-only":
			ftsOrder, rerr = ftsCandidates(ctx, tx, collectionID, sourceID, query, textByChunk)
		case "dense-only":
			denseOrder, rerr = denseCandidates(ctx, tx, collectionID, sourceID, queryVec, textByChunk)
		case "hybrid-prererank", "final":
			ftsOrder, rerr = ftsCandidates(ctx, tx, collectionID, sourceID, query, textByChunk)
			if rerr == nil {
				denseOrder, rerr = denseCandidates(ctx, tx, collectionID, sourceID, queryVec, textByChunk)
			}
		default:
			return fmt.Errorf("store: unknown eval stage %q", stage)
		}
		return rerr
	})
	if err != nil {
		return nil, err
	}

	var ordered []retrieval.RerankedChunk
	switch stage {
	case "fts-only":
		for _, id := range ftsOrder {
			ordered = append(ordered, retrieval.RerankedChunk{ChunkID: id})
		}
	case "dense-only":
		for _, id := range denseOrder {
			ordered = append(ordered, retrieval.RerankedChunk{ChunkID: id})
		}
	case "hybrid-prererank":
		for _, f := range retrieval.RRF(retrieval.DefaultRRFConstant, ftsOrder, denseOrder) {
			ordered = append(ordered, retrieval.RerankedChunk{ChunkID: f.ChunkID, FusedScore: f.Score})
		}
	case "final":
		fused := retrieval.RRF(retrieval.DefaultRRFConstant, ftsOrder, denseOrder)
		ordered = s.rankFused(query, fused, textByChunk)
	}

	var out []protocol.SearchResult
	err = s.withTenant(ctx, p, func(tx pgx.Tx) error {
		if err := requireRole(ctx, tx, collectionID, p, "owner", "reader"); err != nil {
			return err
		}
		for i, r := range ordered {
			if len(out) >= topK {
				break
			}
			if i >= candidateLimit {
				break
			}
			chunk, ok, rerr := s.reauthorizeChunk(ctx, tx, collectionID, sourceID, r.ChunkID)
			if rerr != nil {
				return rerr
			}
			if !ok {
				continue
			}
			snippet := chunk.text
			if len(snippet) > 200 {
				snippet = snippet[:200]
			}
			out = append(out, protocol.SearchResult{ChunkId: r.ChunkID, DocumentId: chunk.documentID, Snippet: &snippet})
		}
		return nil
	})
	return out, err
}

// Answer runs the same candidate selection/fusion/rerank as Search (via
// retrievalCandidates: role check + register, THEN query embedding
// outside any transaction, THEN a fresh candidate-select transaction),
// then -- critically -- re-authorizes every candidate chunk in ANOTHER
// fresh transaction/snapshot immediately before assembling the
// response, so a revocation committed in between is observed (ADR-0007;
// RFC-0004's citation-render re-check). barrierPause, if non-nil, is
// called exactly at that boundary -- the deterministic rendezvous point
// the in-flight-revocation acceptance test pauses on, never a sleep.
func (s *Store) Answer(ctx context.Context, p auth.Principal, collectionID, query string, embedder *embed.Guarded, targetModel string, barrierPause func()) (protocol.AnswerResponse, error) {
	sourceID, reranked, _, err := s.retrievalCandidates(ctx, p, collectionID, query, embedder, targetModel)
	if err != nil {
		return protocol.AnswerResponse{}, err
	}

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
// the caller still holds a grant, the chunk is still ready, and it still
// belongs to the document's CURRENT active version. Returns ok=false
// (never an error) for anything that no longer qualifies -- a revoked
// or superseded chunk silently drops out of the answer, exactly RFC-0004's
// "fails without disclosing" requirement for an evidence check. The
// explicit collection_permission join (not just a check elsewhere in the
// caller's control flow) matters once query embedding runs outside any
// transaction: a revoke committed during that real network call must be
// observed by THIS per-candidate re-check, not only by a role check that
// ran before the network call started.
func (s *Store) reauthorizeChunk(ctx context.Context, tx pgx.Tx, collectionID string, sourceID int64, chunkID string) (reauthorizedChunk, bool, error) {
	documentID, digest, chunkerVersion, chunkIndex, err := decodeChunkID(chunkID)
	if err != nil {
		return reauthorizedChunk{}, false, nil
	}
	var text, status string
	var activeVersionID, chunkVersionID *string
	err = tx.QueryRow(ctx, `
		SELECT c.text, dv.status, d.active_version_id, c.document_version_id
		FROM cortex.chunk c
		JOIN helix_ai.derived_value dv
		  ON (dv.source_key ->> 'document_id') = c.document_id::text
		 AND (dv.source_key ->> 'content_digest') = c.content_digest
		 AND (dv.source_key ->> 'chunker_version') = c.chunker_version
		 AND (dv.source_key ->> 'chunk_index') = c.chunk_index::text
		JOIN cortex.document d ON d.id = c.document_id
		JOIN cortex.collection_permission cp
		  ON cp.collection_id = c.collection_id
		 AND cp.tenant = current_setting('helix.tenant', true)
		 AND cp.principal = current_setting('helix.principal', true)
		WHERE c.document_id = $1 AND c.content_digest = $2 AND c.chunker_version = $3 AND c.chunk_index = $4
		  AND c.collection_id = $5 AND dv.source_id = $6
	`, documentID, digest, chunkerVersion, chunkIndex, collectionID, sourceID).Scan(&text, &status, &activeVersionID, &chunkVersionID)
	if errors.Is(err, pgx.ErrNoRows) {
		return reauthorizedChunk{}, false, nil // ACL revoked, deleted, or RLS hid it
	}
	if err != nil {
		return reauthorizedChunk{}, false, fmt.Errorf("store: reauthorize chunk: %w", err)
	}
	if status != "ready" {
		return reauthorizedChunk{}, false, nil
	}
	// Equality, not just non-null: a chunk row only ever exists for its
	// own (digest, chunker_version), so a non-null active_version_id that
	// belongs to a DIFFERENT (superseded) version than this chunk's own
	// would otherwise still pass.
	if activeVersionID == nil || chunkVersionID == nil || *activeVersionID != *chunkVersionID {
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
		sourceID, serr := s.ensureSourceRegistered(ctx, tx)
		if serr != nil {
			return serr
		}
		var collectionID string
		var status string
		var activeVersionID, chunkVersionID *string
		var documentName string
		serr = tx.QueryRow(ctx, `
			SELECT c.text, c.collection_id, dv.status, d.active_version_id, c.document_version_id, d.name
			FROM cortex.chunk c
			JOIN helix_ai.derived_value dv
			  ON (dv.source_key ->> 'document_id') = c.document_id::text
			 AND (dv.source_key ->> 'content_digest') = c.content_digest
			 AND (dv.source_key ->> 'chunker_version') = c.chunker_version
			 AND (dv.source_key ->> 'chunk_index') = c.chunk_index::text
			JOIN cortex.document d ON d.id = c.document_id
			WHERE c.document_id = $1 AND c.content_digest = $2 AND c.chunker_version = $3 AND c.chunk_index = $4
			  AND dv.source_id = $5
		`, documentID, digest, chunkerVersion, chunkIndex, sourceID).Scan(&out.Text, &collectionID, &status, &activeVersionID, &chunkVersionID, &documentName)
		if errors.Is(serr, pgx.ErrNoRows) {
			// RLS hid it (no ACL grant / wrong tenant) or it genuinely
			// doesn't exist -- same citation_invalid response either way,
			// never disclosing which case applied.
			return ErrNotFound
		}
		if serr != nil {
			return fmt.Errorf("store: evidence lookup: %w", serr)
		}
		// Equality, not just non-null: a chunk row only ever exists for
		// its own (digest, chunker_version), so a non-null active_version_id
		// that belongs to a DIFFERENT version than this chunk's own would
		// otherwise pass -- the chunk must belong to the document's
		// CURRENT active version, not merely "some version is active".
		if status != "ready" || activeVersionID == nil || chunkVersionID == nil || *activeVersionID != *chunkVersionID {
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
