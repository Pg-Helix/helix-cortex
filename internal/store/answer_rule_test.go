package store

import (
	"encoding/json"
	"testing"

	"github.com/Pg-Helix/helix-cortex/internal/retrieval"
	protocol "github.com/Pg-Helix/helix-protocol/gen/go"
)

func TestSupportsAnswer(t *testing.T) {
	tau := 0.707
	gated := &Store{AnswerMinSimilarity: &tau}
	sims := map[string]float64{"above": 0.71, "at": 0.707, "below": 0.70}
	for id, want := range map[string]bool{"above": true, "at": true, "below": false, "fts-only-no-dense-similarity": false} {
		if got := gated.supportsAnswer(sims, id); got != want {
			t.Errorf("supportsAnswer(%q) = %v, want %v", id, got, want)
		}
	}
	// nil threshold (mock embeddings only) uses every reauthorized passage.
	if !(&Store{}).supportsAnswer(nil, "anything") {
		t.Error("nil AnswerMinSimilarity must not filter")
	}
}

// The decline is the exact documented wire body -- an authorized
// collection with no supporting passage (or no passages at all, e.g. an
// empty collection) must serialize citations as [], never null.
func TestBuildAnswerResponse_DeclineWireBody(t *testing.T) {
	for name, passages := range map[string][]retrieval.Passage{"nil": nil, "empty": {}} {
		b, err := json.Marshal(buildAnswerResponse(passages, map[string]protocol.Citation{}))
		if err != nil {
			t.Fatal(err)
		}
		if want := `{"answer":"","answer_method":"extractive","citations":[]}`; string(b) != want {
			t.Fatalf("%s passages: got %s, want %s", name, b, want)
		}
	}
}

// Cites exactly the passages the answer text used, in order -- every
// citation after the first included, none that were not used.
func TestBuildAnswerResponse_CitesUsedPassages(t *testing.T) {
	passages := []retrieval.Passage{{ChunkID: "a", Text: "first"}, {ChunkID: "b", Text: "second"}, {ChunkID: "c", Text: "  "}}
	cites := map[string]protocol.Citation{
		"a": {ChunkId: "a", SourceId: "doc-a", SourceDigest: "da"},
		"b": {ChunkId: "b", SourceId: "doc-b", SourceDigest: "db"},
		"c": {ChunkId: "c", SourceId: "doc-c", SourceDigest: "dc"},
	}
	got := buildAnswerResponse(passages, cites)
	if got.Answer != "first ... second" || got.AnswerMethod != protocol.Extractive {
		t.Fatalf("answer = %q (%s)", got.Answer, got.AnswerMethod)
	}
	if len(got.Citations) != 2 || got.Citations[0] != cites["a"] || got.Citations[1] != cites["b"] {
		t.Fatalf("citations = %+v, want a then b (c contributed no text)", got.Citations)
	}
}
