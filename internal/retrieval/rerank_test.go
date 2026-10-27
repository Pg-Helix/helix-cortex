package retrieval

import "testing"

// TestRerank_HandCalculatedOverlap: query "apple banana" (2 terms).
// chunkA text contains both terms (overlap = 2/2 = 1.0); chunkB
// contains only "apple" (overlap = 1/2 = 0.5); chunkC contains neither
// (overlap = 0/2 = 0.0). Fused (RRF) order deliberately starts as
// [C, B, A] -- the OPPOSITE of the expected reranked order -- so this
// test actually exercises reranking changing the order, not merely
// preserving whatever order it was handed.
func TestRerank_HandCalculatedOverlap(t *testing.T) {
	fused := []ScoredChunk{
		{ChunkID: "C", Score: 0.03},
		{ChunkID: "B", Score: 0.02},
		{ChunkID: "A", Score: 0.01},
	}
	text := map[string]string{
		"A": "apple pie and banana split for dessert",
		"B": "apple pie, nothing else relevant",
		"C": "nothing relevant in here at all",
	}
	queryTerms := Tokenize("apple banana")
	if len(queryTerms) != 2 {
		t.Fatalf("tokenize: got %v, want 2 terms", queryTerms)
	}

	got := Rerank(queryTerms, fused, text)

	wantOrder := []string{"A", "B", "C"}
	wantOverlap := map[string]float64{"A": 1.0, "B": 0.5, "C": 0.0}
	if len(got) != 3 {
		t.Fatalf("got %d results, want 3", len(got))
	}
	for i, id := range wantOrder {
		if got[i].ChunkID != id {
			t.Fatalf("result %d: chunk_id = %q, want %q (full: %+v)", i, got[i].ChunkID, id, got)
		}
		if got[i].OverlapScore != wantOverlap[id] {
			t.Fatalf("chunk %s: overlap = %v, want %v", id, got[i].OverlapScore, wantOverlap[id])
		}
	}
}

// TestRerank_TiesBrokenByFusedScoreThenChunkID: equal overlap falls back
// to the original fused score, then chunk id.
func TestRerank_TiesBrokenByFusedScoreThenChunkID(t *testing.T) {
	fused := []ScoredChunk{
		{ChunkID: "Z", Score: 0.05},
		{ChunkID: "Y", Score: 0.09},
	}
	text := map[string]string{
		"Z": "same words here",
		"Y": "same words here",
	}
	got := Rerank(Tokenize("same words"), fused, text)
	if got[0].ChunkID != "Y" || got[1].ChunkID != "Z" {
		t.Fatalf("expected Y (higher fused score) before Z on an overlap tie, got %+v", got)
	}
}

func TestTokenize_LowercasesAndSplitsOnPunctuation(t *testing.T) {
	got := Tokenize("Apple, BANANA! (cherry?)")
	want := []string{"apple", "banana", "cherry"}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token %d: got %q, want %q", i, got[i], want[i])
		}
	}
}
