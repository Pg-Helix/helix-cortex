package store

import "testing"

func TestEstimatedTokens_OverestimatesConservatively(t *testing.T) {
	// A 1500-rune string must estimate to MORE than nomic-embed-text's
	// confirmed-safe 2048-token ceiling's reasonable fraction, proving
	// the heuristic errs toward rejecting, never toward silently
	// accepting something that could get truncated server-side.
	s := make([]rune, 1500)
	for i := range s {
		s[i] = 'a'
	}
	got := estimatedTokens(string(s))
	if got < 500 {
		t.Fatalf("estimatedTokens(1500 runes) = %d, expected a conservative (large) estimate", got)
	}
}

func TestToTSQuery_EmptyForNoTerms(t *testing.T) {
	if got := toTSQuery("   "); got != "" {
		t.Fatalf("got %q, want empty string for a query with no usable terms", got)
	}
}

func TestVectorLiteral_Format(t *testing.T) {
	got := vectorLiteral([]float32{0.1, 0.2, -0.3})
	want := "[0.1,0.2,-0.3]"
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}
