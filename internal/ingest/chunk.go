// Package ingest implements deterministic chunking for C2 (ADR-0007).
// UTF-8 text and Markdown only for this slice -- Markdown is chunked as
// plain text (no structural/header-aware splitting), which is a
// deliberate simplification, not an oversight: RFC-0004's ingestion
// scope is only partially covered by this milestone (no remote URL,
// HTML extraction, PDF or OCR).
package ingest

import (
	"fmt"
	"unicode/utf8"
)

// ChunkerVersion identifies this exact chunking policy. Bumped whenever
// the boundary/overlap algorithm changes, so existing chunks are
// replaced (ADR-0007 decision 3) rather than silently reinterpreted.
const ChunkerVersion = "chunker-v1"

const (
	// MaxChunkChars and ChunkOverlapChars define this version's fixed,
	// documented policy: split on rune boundaries only (never inside a
	// multi-byte UTF-8 sequence), fixed-size windows with a fixed
	// overlap. No sentence/paragraph awareness in this slice.
	MaxChunkChars     = 1000
	ChunkOverlapChars = 100

	// MaxDocumentBytes and MaxChunkCount are the bounded-input limits
	// ADR-0007/the task require: oversized or runaway input is rejected
	// outright, never silently truncated.
	MaxDocumentBytes = 1 << 20 // 1 MiB, matching get_job_input's own bound
	MaxChunkCount    = 500
)

// Chunk is one deterministic slice of a document's text, in rune
// (character) offsets -- not byte offsets, so char_start/char_end are
// meaningful across multi-byte UTF-8 content.
type Chunk struct {
	Index     int
	Text      string
	CharStart int
	CharEnd   int
}

// ErrTooLarge and ErrUnsupportedContentType are returned by Chunk/the
// ingest handler for bounded-input violations -- distinguishable from a
// generic error so the HTTP layer can map them to the right error code
// (content_too_large / chunk_limit_exceeded / unsupported_content_type).
type ErrTooLarge struct{ Reason string }

func (e *ErrTooLarge) Error() string { return "ingest: " + e.Reason }

// Split deterministically chunks content into fixed-size, overlapping,
// rune-boundary-safe windows. Returns ErrTooLarge if content exceeds
// MaxDocumentBytes or would produce more than MaxChunkCount chunks --
// rejected outright, never silently truncated.
func Split(content string) ([]Chunk, error) {
	if len(content) > MaxDocumentBytes {
		return nil, &ErrTooLarge{Reason: fmt.Sprintf("document is %d bytes, exceeds the %d byte limit", len(content), MaxDocumentBytes)}
	}
	if !utf8.ValidString(content) {
		return nil, fmt.Errorf("ingest: content is not valid UTF-8")
	}
	runes := []rune(content)
	n := len(runes)
	if n == 0 {
		return nil, fmt.Errorf("ingest: content is empty")
	}

	step := MaxChunkChars - ChunkOverlapChars
	if step <= 0 {
		// Defensive: MaxChunkChars/ChunkOverlapChars are compile-time
		// constants here, so this can only happen if someone breaks that
		// invariant later -- fail loudly rather than looping forever.
		return nil, fmt.Errorf("ingest: chunk overlap must be smaller than chunk size")
	}

	var chunks []Chunk
	for start := 0; start < n; start += step {
		end := start + MaxChunkChars
		if end > n {
			end = n
		}
		chunks = append(chunks, Chunk{
			Index:     len(chunks),
			Text:      string(runes[start:end]),
			CharStart: start,
			CharEnd:   end,
		})
		if len(chunks) > MaxChunkCount {
			return nil, &ErrTooLarge{Reason: fmt.Sprintf("document would produce more than %d chunks", MaxChunkCount)}
		}
		if end == n {
			break
		}
	}
	return chunks, nil
}
