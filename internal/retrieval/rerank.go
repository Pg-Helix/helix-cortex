package retrieval

import (
	"sort"
	"strings"
	"unicode"
)

// RerankedChunk is one chunk after the deterministic lexical-overlap
// reranker, ordered by its combined sort key -- never a no-op pass-
// through of the fused order (RFC-0004 calls for reranking; this is a
// small, real implementation of it, explicitly not deferred).
type RerankedChunk struct {
	ChunkID      string
	OverlapScore float64 // fraction of query terms present in the chunk's text, in [0,1]
	FusedScore   float64 // the RRF score this chunk entered reranking with
}

// Tokenize lowercases and splits on anything that isn't a letter or
// digit -- deliberately simple (no stemming, no stopword removal) so
// the overlap score stays easy to hand-verify in a test fixture.
func Tokenize(s string) []string {
	var tokens []string
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			tokens = append(tokens, cur.String())
			cur.Reset()
		}
	}
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			cur.WriteRune(unicode.ToLower(r))
		} else {
			flush()
		}
	}
	flush()
	return tokens
}

func overlapFraction(queryTerms []string, chunkTerms []string) float64 {
	if len(queryTerms) == 0 {
		return 0
	}
	set := make(map[string]bool, len(chunkTerms))
	for _, t := range chunkTerms {
		set[t] = true
	}
	matched := 0
	for _, t := range queryTerms {
		if set[t] {
			matched++
		}
	}
	return float64(matched) / float64(len(queryTerms))
}

// Rerank re-scores fused's chunks by lexical overlap between queryTerms
// and each chunk's text (textByChunk), then sorts by (overlap desc,
// fused score desc, chunk id asc) -- the last two only ever matter as
// deterministic tie-breaks when overlap scores are equal.
func Rerank(queryTerms []string, fused []ScoredChunk, textByChunk map[string]string) []RerankedChunk {
	out := make([]RerankedChunk, 0, len(fused))
	for _, f := range fused {
		terms := Tokenize(textByChunk[f.ChunkID])
		out = append(out, RerankedChunk{
			ChunkID:      f.ChunkID,
			OverlapScore: overlapFraction(queryTerms, terms),
			FusedScore:   f.Score,
		})
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].OverlapScore != out[j].OverlapScore {
			return out[i].OverlapScore > out[j].OverlapScore
		}
		if out[i].FusedScore != out[j].FusedScore {
			return out[i].FusedScore > out[j].FusedScore
		}
		return out[i].ChunkID < out[j].ChunkID
	})
	return out
}
