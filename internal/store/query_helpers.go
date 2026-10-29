package store

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/Pg-Helix/helix-cortex/internal/retrieval"
	"github.com/Pg-Helix/helix-infer/embed"
)

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
// second, differently-configured embedder.
func embedQuery(ctx context.Context, embedder *embed.Guarded, targetModel, query string) ([]float32, error) {
	res, err := embedder.Embed(ctx, embed.EmbedRequest{Model: targetModel, Input: []string{query}})
	if err != nil {
		return nil, fmt.Errorf("store: embed query: %w", err)
	}
	if len(res.Vectors) == 0 {
		return nil, fmt.Errorf("store: embedder returned no vectors for query")
	}
	return res.Vectors[0], nil
}
