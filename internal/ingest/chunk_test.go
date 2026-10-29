package ingest

import (
	"errors"
	"strings"
	"testing"
)

func TestSplit_Deterministic(t *testing.T) {
	content := strings.Repeat("ab", 1000) // 2000 chars
	a, err := Split(content)
	if err != nil {
		t.Fatalf("Split: %v", err)
	}
	b, err := Split(content)
	if err != nil {
		t.Fatalf("Split: %v", err)
	}
	if len(a) != len(b) {
		t.Fatalf("non-deterministic chunk count: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("non-deterministic chunk %d: %+v vs %+v", i, a[i], b[i])
		}
	}
}

func TestSplit_OverlapAndCoverage(t *testing.T) {
	content := strings.Repeat("x", 2500)
	chunks, err := Split(content)
	if err != nil {
		t.Fatalf("Split: %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks for 2500 chars, got %d", len(chunks))
	}
	for i := 1; i < len(chunks); i++ {
		overlap := chunks[i-1].CharEnd - chunks[i].CharStart
		if overlap != ChunkOverlapChars {
			t.Fatalf("chunk %d: overlap = %d, want %d", i, overlap, ChunkOverlapChars)
		}
	}
	last := chunks[len(chunks)-1]
	if last.CharEnd != len(content) {
		t.Fatalf("last chunk ends at %d, want %d (full coverage)", last.CharEnd, len(content))
	}
	if chunks[0].CharStart != 0 {
		t.Fatalf("first chunk starts at %d, want 0", chunks[0].CharStart)
	}
}

func TestSplit_RuneBoundarySafe(t *testing.T) {
	// Multi-byte runes (emoji, 4 bytes each in UTF-8) placed right where a
	// naive byte-offset split would land mid-sequence.
	content := strings.Repeat("a", MaxChunkChars-1) + strings.Repeat("😀", 50)
	chunks, err := Split(content)
	if err != nil {
		t.Fatalf("Split: %v", err)
	}
	for _, c := range chunks {
		if !isValidUTF8(c.Text) {
			t.Fatalf("chunk %d text is not valid UTF-8: %q", c.Index, c.Text)
		}
	}
}

func isValidUTF8(s string) bool {
	for _, r := range s {
		if r == '�' {
			return false
		}
	}
	return true
}

func TestSplit_RejectsOversizedDocument(t *testing.T) {
	content := strings.Repeat("a", MaxDocumentBytes+1)
	_, err := Split(content)
	var tooLarge *ErrTooLarge
	if !errors.As(err, &tooLarge) {
		t.Fatalf("expected ErrTooLarge, got %v", err)
	}
}

func TestSplit_RejectsEmpty(t *testing.T) {
	if _, err := Split(""); err == nil {
		t.Fatal("expected an error for empty content, got nil")
	}
}

func TestSplit_RejectsInvalidUTF8(t *testing.T) {
	if _, err := Split(string([]byte{0xff, 0xfe, 0xfd})); err == nil {
		t.Fatal("expected an error for invalid UTF-8, got nil")
	}
}
