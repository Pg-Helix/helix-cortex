package ingest

import (
	"encoding/json"
	"fmt"
)

// ExtractChunkText is the versioned input-extraction strategy for
// embedding a cortex.chunk row (ADR-0007): get_job_input returns the
// chunk's ENTIRE row as jsonb (document_id, tenant, collection_id,
// digests, timestamps, text, ...) -- this reads only the `text` field,
// so a chunk's own identifiers and metadata are never embedded
// alongside its actual content as prose.
func ExtractChunkText(raw json.RawMessage) (string, error) {
	var row struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &row); err != nil {
		return "", fmt.Errorf("ingest: extract chunk text: %w", err)
	}
	if row.Text == "" {
		return "", fmt.Errorf("ingest: chunk row has no text field")
	}
	return row.Text, nil
}
