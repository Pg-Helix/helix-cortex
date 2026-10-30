package store

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/Pg-Helix/helix-cortex/internal/embedconfig"
	"github.com/Pg-Helix/helix-cortex/internal/retrieval"
	"github.com/Pg-Helix/helix-infer/embed"
)

// ErrQueryTooLong is returned when a query's ESTIMATED token count
// exceeds maxQueryTokens. "Estimated" because bounding in the model's
// actual units requires its real tokenizer, which this codebase
// deliberately does not vendor for a one-line safety check; runes/3 is
// a conservative OVERestimate for English/Markdown-ish text (never
// under-estimates, so it never lets a genuinely-too-long query through
// -- it can reject some borderline-fine queries, which is the safe
// direction for a bounded-input check, not the dangerous one). Found
// necessary by testing a real provider directly: Ollama's
// /v1/embeddings silently truncated a 20k-word input to 2048 tokens
// (HTTP 200, no error) despite the model's Modelfile declaring
// num_ctx=8192 -- a bound enforced here, before the request leaves this
// process, is the only way to get an honest rejection instead of a
// silently-wrong embedding.
var ErrQueryTooLong = errors.New("store: query is too long")

const maxQueryTokensEstimate = 500

func estimatedTokens(s string) int {
	return utf8.RuneCountInString(s)/3 + 1
}

// toTSQuery turns free text into a safe to_tsquery argument: tokenized
// (same tokenizer retrieval.Tokenize uses, so FTS and the reranker agree
// on what a "term" is) and joined with & (AND) -- never string-
// concatenated into the query itself. Returns "" for a query with no
// usable terms, meaning the FTS half of candidate selection is skipped
// for this query (dense retrieval still runs).
func toTSQuery(query string) string {
	terms := retrieval.Tokenize(query)
	if len(terms) == 0 {
		return ""
	}
	return strings.Join(terms, " & ")
}

// vectorLiteral formats a float32 slice as a pgvector input literal,
// e.g. "[0.1,0.2,0.3]" -- passed as a query parameter and cast with
// ::vector in SQL, never string-concatenated into the query text.
func vectorLiteral(vec []float32) string {
	parts := make([]string, len(vec))
	for i, v := range vec {
		parts[i] = strconv.FormatFloat(float64(v), 'g', -1, 32)
	}
	return "[" + strings.Join(parts, ",") + "]"
}

// embedQuery embeds the query text with the same provider/model
// ingestion used, so query and document vectors share preprocessing,
// model revision, dimensions and distance metric (ADR-0007) -- never a
// second, differently-configured embedder. modelID selects the exact
// same versioned query-prefix convention ingestion's chunk-text
// extraction uses for the document side (embedconfig.QueryPrefix) --
// nomic-embed-text's "search_query: " task instruction is part of this
// pipeline's preprocessing contract, not a string a handler happens to
// prepend.
func embedQuery(ctx context.Context, embedder *embed.Guarded, targetModel, modelID, query string) ([]float32, error) {
	if estimatedTokens(query) > maxQueryTokensEstimate {
		return nil, ErrQueryTooLong
	}
	prefixed := embedconfig.QueryPrefix(modelID) + query
	res, err := embedder.Embed(ctx, embed.EmbedRequest{Model: targetModel, Input: []string{prefixed}})
	if err != nil {
		return nil, fmt.Errorf("store: embed query: %w", err)
	}
	if len(res.Vectors) != 1 {
		return nil, fmt.Errorf("store: embedder returned %d vectors for 1 query input", len(res.Vectors))
	}
	return res.Vectors[0], nil
}
