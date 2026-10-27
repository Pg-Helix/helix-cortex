// Package retrieval implements RFC-0004's hybrid fusion, a small
// deterministic reranker, and deterministic extractive answer assembly
// (ADR-0007). All pure functions over already-ACL-filtered candidates --
// the ACL filtering itself happens in the SQL that produces each input
// list (RLS, decision 4), not here.
package retrieval

import "sort"

// DefaultRRFConstant is RFC-0004's chosen fusion method's smoothing
// constant k. A common default (no corpus-specific tuning data exists
// yet for Phase 1, per RFC-0004's own "Alternatives considered").
const DefaultRRFConstant = 60.0

// ScoredChunk is one chunk's fused score after RRF.
type ScoredChunk struct {
	ChunkID string
	Score   float64
}

// RRF fuses one or more ranked candidate lists (each ordered best-first
// by chunk id) via Reciprocal Rank Fusion: score(c) = sum over lists
// containing c of 1/(k + rank), rank 1-based. A chunk absent from a list
// contributes 0 from it. Deterministic: ties break by chunk id ascending
// -- never by map iteration order, which Go does not guarantee stable.
func RRF(k float64, lists ...[]string) []ScoredChunk {
	scores := make(map[string]float64)
	for _, list := range lists {
		for rank, id := range list {
			scores[id] += 1.0 / (k + float64(rank+1))
		}
	}
	out := make([]ScoredChunk, 0, len(scores))
	for id, s := range scores {
		out = append(out, ScoredChunk{ChunkID: id, Score: s})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].ChunkID < out[j].ChunkID
	})
	return out
}
