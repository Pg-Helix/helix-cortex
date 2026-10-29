package ingest

import "testing"

func TestExtractChunkText_PullsOnlyTextField(t *testing.T) {
	raw := []byte(`{"document_id":"doc-1","tenant":"acme","content_digest":"abc","text":"the actual passage"}`)
	got, err := ExtractChunkText(raw)
	if err != nil {
		t.Fatalf("ExtractChunkText: %v", err)
	}
	if got != "the actual passage" {
		t.Fatalf("got %q, want %q", got, "the actual passage")
	}
}

func TestExtractChunkText_RejectsMissingText(t *testing.T) {
	raw := []byte(`{"document_id":"doc-1"}`)
	if _, err := ExtractChunkText(raw); err == nil {
		t.Fatal("expected an error for a row with no text field, got nil")
	}
}
