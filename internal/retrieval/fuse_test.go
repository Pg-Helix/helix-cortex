package retrieval

import "testing"

// TestRRF_HandCalculated uses expected scores computed by hand (not
// copied from running the implementation): k=60, fts=[A,B,C] (A rank1,
// B rank2, C rank3), dense=[B,A,D] (B rank1, A rank2, D rank3).
//
//	RRF(A) = 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.032522474881015337
//	RRF(B) = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.032522474881015337 (tied with A)
//	RRF(C) = 1/(60+3)            = 1/63        = 0.015873015873015872
//	RRF(D) = 1/(60+3)            = 1/63        = 0.015873015873015872 (tied with C)
//
// A and B are an exact tie (they just swap rank 1/2 between the two
// lists), as are C and D -- so the deterministic chunk-id tie-break is
// genuinely exercised, not incidental.
func TestRRF_HandCalculated(t *testing.T) {
	fts := []string{"A", "B", "C"}
	dense := []string{"B", "A", "D"}

	got := RRF(60.0, fts, dense)

	want := []ScoredChunk{
		{ChunkID: "A", Score: 0.032522474881015337},
		{ChunkID: "B", Score: 0.032522474881015337},
		{ChunkID: "C", Score: 0.015873015873015872},
		{ChunkID: "D", Score: 0.015873015873015872},
	}
	if len(got) != len(want) {
		t.Fatalf("got %d results, want %d: %+v", len(got), len(want), got)
	}
	for i := range want {
		if got[i].ChunkID != want[i].ChunkID {
			t.Fatalf("result %d: chunk_id = %q, want %q (full: %+v)", i, got[i].ChunkID, want[i].ChunkID, got)
		}
		diff := got[i].Score - want[i].Score
		if diff < -1e-12 || diff > 1e-12 {
			t.Fatalf("result %d (%s): score = %.18f, want %.18f", i, got[i].ChunkID, got[i].Score, want[i].Score)
		}
	}
}

// TestRRF_OnlyInOneList: a chunk appearing in just one list still scores
// (not excluded), lower than one appearing in both at a comparable rank.
//
//	RRF(E, dense-only rank1) = 1/61 = 0.016393442622950821
func TestRRF_OnlyInOneList(t *testing.T) {
	fts := []string{"X"}
	dense := []string{"E"}
	got := RRF(60.0, fts, dense)
	var eScore float64
	found := false
	for _, r := range got {
		if r.ChunkID == "E" {
			eScore = r.Score
			found = true
		}
	}
	if !found {
		t.Fatalf("expected chunk E present in fused results, got %+v", got)
	}
	want := 1.0 / 61.0
	if diff := eScore - want; diff < -1e-12 || diff > 1e-12 {
		t.Fatalf("E score = %.18f, want %.18f", eScore, want)
	}
}

func TestRRF_DeterministicAcrossRuns(t *testing.T) {
	fts := []string{"B", "A", "C"}
	dense := []string{"A", "C", "B"}
	first := RRF(60.0, fts, dense)
	for i := 0; i < 20; i++ {
		again := RRF(60.0, fts, dense)
		if len(again) != len(first) {
			t.Fatalf("run %d: length changed", i)
		}
		for j := range first {
			if again[j] != first[j] {
				t.Fatalf("run %d: result %d changed from %+v to %+v", i, j, first[j], again[j])
			}
		}
	}
}
