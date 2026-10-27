package retrieval

import "strings"

// MaxAnswerChars and MaxAnswerChunks bound the extractive answer this
// milestone assembles -- deterministic concatenation of authorized
// passage text, never a generative model call (RFC-0004's generative
// RAG quality is explicitly out of scope here).
const (
	MaxAnswerChars  = 2000
	MaxAnswerChunks = 3
)

// Passage is one authorized chunk's text, already past every
// authorization re-check (ACL, tenant, generation, digest, readiness) --
// AssembleExtractiveAnswer trusts its callers to have done that, it does
// no authorization of its own.
type Passage struct {
	ChunkID string
	Text    string
}

// AssembleExtractiveAnswer deterministically concatenates up to
// MaxAnswerChunks passages (already in final rank order), truncated to
// MaxAnswerChars, and reports exactly which passages ended up included
// -- the caller cites only those, never a passage that was selected but
// then dropped by the character bound.
func AssembleExtractiveAnswer(passages []Passage) (answer string, used []Passage) {
	var b strings.Builder
	for i, p := range passages {
		if i >= MaxAnswerChunks {
			break
		}
		text := strings.TrimSpace(p.Text)
		if text == "" {
			continue
		}
		sep := ""
		if b.Len() > 0 {
			sep = " ... "
		}
		if b.Len()+len(sep)+len(text) > MaxAnswerChars {
			remaining := MaxAnswerChars - b.Len() - len(sep)
			if remaining <= 0 {
				break
			}
			text = text[:remaining]
		}
		b.WriteString(sep)
		b.WriteString(text)
		used = append(used, Passage{ChunkID: p.ChunkID, Text: text})
		if b.Len() >= MaxAnswerChars {
			break
		}
	}
	return b.String(), used
}
