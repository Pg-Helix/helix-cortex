package store

import (
	"fmt"
	"strconv"
	"strings"
)

// chunkID encodes cortex.chunk's composite primary key (document_id,
// content_digest, chunker_version, chunk_index) as one opaque string for
// the HTTP API -- none of the four parts can contain "|" (uuid, hex
// digest, a version tag like "chunker-v1", and a decimal integer).
func chunkID(documentID, contentDigest, chunkerVersion string, chunkIndex int) string {
	return documentID + "|" + contentDigest + "|" + chunkerVersion + "|" + strconv.Itoa(chunkIndex)
}

func decodeChunkID(id string) (documentID, contentDigest, chunkerVersion string, chunkIndex int, err error) {
	parts := strings.Split(id, "|")
	if len(parts) != 4 {
		return "", "", "", 0, fmt.Errorf("store: malformed chunk id %q", id)
	}
	idx, convErr := strconv.Atoi(parts[3])
	if convErr != nil {
		return "", "", "", 0, fmt.Errorf("store: malformed chunk id %q: %w", id, convErr)
	}
	return parts[0], parts[1], parts[2], idx, nil
}
