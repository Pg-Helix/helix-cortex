// Package embedconfig resolves the ONE effective embedding profile that
// both cmd/cortex (the chunk/raw worker) and cmd/cortex-api (query
// embedding, source registration, backfill and the /answers rule) use:
// provider, internal model id + revision, wire model, dimensions and
// query/document prefixes. Resolve validates the whole profile before
// anything is built from it; NewProvider builds the provider from that
// same Profile, and the /answers threshold comes from that same Profile
// -- so the provider that embeds, the identity vectors are written under
// and the rule that interprets similarity can never come from different
// configuration. Each caller reads model id/revision/dims under its own
// existing env names (CORTEX_MODEL_* vs CORTEX_CHUNK_MODEL_*).
//
// What is configurable: CORTEX_EMBED_PROVIDER, CORTEX_EMBED_BASE_URL,
// CORTEX_EMBED_WIRE_MODEL, CORTEX_EMBED_API_KEY, CORTEX_EMBED_MAX_RETRIES
// and the caller's model id/revision/dims. Fixed by code: the prefixes
// (QueryPrefix/DocumentPrefix, keyed by model id) and the calibrations
// table below. Fixed by SQL: the cosine metric (cortex registers every
// chunk source with metric 'cosine' and ranks by pgvector's <=>).
package embedconfig

import (
	"fmt"
	"os"
	"strconv"

	"github.com/Pg-Helix/helix-infer/embed"
)

// Profile is one resolved, validated embedding configuration.
type Profile struct {
	Provider       string // "mock" or "openai-compat"
	ModelID        string // internal binding identity, never sent on the wire
	Revision       string
	WireModel      string // what an openai-compat endpoint is asked for; "" for mock
	Dims           int
	QueryPrefix    string
	DocumentPrefix string
	// AnswerMinSimilarity is set iff this profile exactly matches a
	// calibrated one (calibrations below); nil otherwise.
	AnswerMinSimilarity *float64

	baseURL    string
	apiKey     string
	maxRetries int
}

// calibration is one embedding profile the /answers evidence rule was
// measured on (helix-atlas ADR-0008). Every field is part of the claim:
// a similarity threshold means nothing for different weights, a
// different dimensionality or different preprocessing.
type calibration struct {
	provider, modelID, revision, wireModel string
	dims                                   int
	queryPrefix, documentPrefix            string
	answerMinSimilarity                    float64
}

// calibrations: nomic-embed-text v1.5 served by Ollama 0.40.0 from the
// weights helix-foundry's images/ollama pins (sha256:970aa74c...), 768
// dims, cosine, with the model card's task prefixes. 0.707 is the
// rounded midpoint (0.70704569) of the best interval on the 27
// helix-labs/eval development cases; helix-labs/eval/calibrate.py
// replays the selection from the repository. Clearing it means the
// passage is close enough to the question to be used as evidence; it
// does not establish that the assembled answer is complete.
//
// What Cortex can check here is the CONFIGURATION: provider, wire tag,
// dims, prefixes. It cannot check which weights an endpoint actually
// serves -- an OpenAI-compatible server can answer to
// "nomic-embed-text:v1.5" with anything. That is helix-foundry's job:
// its Ollama image bakes in the pinned weights and refuses to build on
// a digest mismatch, and helix-labs verifies the running manifest's
// digest before every evaluation.
var calibrations = []calibration{{
	provider: "openai-compat", modelID: "nomic-embed-text", revision: "v1.5",
	wireModel: "nomic-embed-text:v1.5", dims: 768,
	queryPrefix: "search_query: ", documentPrefix: "search_document: ",
	answerMinSimilarity: 0.707,
}}

// Resolve reads the provider half of the configuration from the
// environment, combines it with the caller's model id/revision/dims and
// validates the result:
//   - an unknown provider, or openai-compat without a base URL or wire
//     model, is an error;
//   - a configuration whose model id + revision names a calibrated
//     profile must match that profile in EVERY field (provider, wire
//     model, dims, prefixes) -- otherwise it is an error, so neither a
//     mock provider, a different wire tag nor a different dimensionality
//     can embed, register vectors or serve answers under the calibrated
//     identity;
//   - anything else (mock bindings, a new real revision not yet
//     calibrated) resolves with no answer threshold. Whether that may
//     SERVE /answers is the caller's decision: see RequireAnswerRule.
func Resolve(modelID, revision string, dims int) (Profile, error) {
	p := Profile{
		Provider: getenv("CORTEX_EMBED_PROVIDER", "mock"), ModelID: modelID, Revision: revision, Dims: dims,
		QueryPrefix: QueryPrefix(modelID), DocumentPrefix: DocumentPrefix(modelID),
	}
	switch p.Provider {
	case "mock":
	case "openai-compat":
		p.baseURL = os.Getenv("CORTEX_EMBED_BASE_URL")
		if p.baseURL == "" {
			return Profile{}, fmt.Errorf("embedconfig: CORTEX_EMBED_BASE_URL is required when CORTEX_EMBED_PROVIDER=openai-compat")
		}
		p.WireModel = os.Getenv("CORTEX_EMBED_WIRE_MODEL")
		if p.WireModel == "" {
			return Profile{}, fmt.Errorf("embedconfig: CORTEX_EMBED_WIRE_MODEL is required when CORTEX_EMBED_PROVIDER=openai-compat -- the internal model_id:revision binding is never sent to the server as-is")
		}
		// Not required: Ollama ignores this header's value entirely.
		p.apiKey = os.Getenv("CORTEX_EMBED_API_KEY")
		p.maxRetries = getenvInt("CORTEX_EMBED_MAX_RETRIES", 2)
	default:
		return Profile{}, fmt.Errorf("embedconfig: unknown CORTEX_EMBED_PROVIDER %q (want \"mock\" or \"openai-compat\")", p.Provider)
	}
	for _, c := range calibrations {
		if c.modelID != modelID || c.revision != revision {
			continue
		}
		if p.Provider != c.provider || p.WireModel != c.wireModel || p.Dims != c.dims ||
			p.QueryPrefix != c.queryPrefix || p.DocumentPrefix != c.documentPrefix {
			return Profile{}, fmt.Errorf("embedconfig: %s:%s is a calibrated embedding profile (provider=%s wire_model=%s dims=%d); "+
				"this configuration has provider=%s wire_model=%q dims=%d -- refusing to embed, write vectors or serve answers under that identity",
				modelID, revision, c.provider, c.wireModel, c.dims, p.Provider, p.WireModel, p.Dims)
		}
		t := c.answerMinSimilarity
		p.AnswerMinSimilarity = &t
	}
	return p, nil
}

// RequireAnswerRule is the /answers serving check: a real provider must
// have a calibrated threshold (it would otherwise answer from any ranked
// passage, or decline everything, silently). Mock embeddings carry no
// similarity signal and serve without the rule.
func (p Profile) RequireAnswerRule() error {
	if p.AnswerMinSimilarity == nil && p.Provider != "mock" {
		return fmt.Errorf("embedconfig: no calibrated /answers similarity threshold for %s:%s (provider=%s wire_model=%s dims=%d)",
			p.ModelID, p.Revision, p.Provider, p.WireModel, p.Dims)
	}
	return nil
}

// NewProvider builds the embedding provider from this same profile.
func (p Profile) NewProvider() (embed.EmbeddingProvider, error) {
	switch p.Provider {
	case "mock":
		return embed.NewMock(p.Dims), nil
	case "openai-compat":
		return embed.NewOpenAICompat(embed.OpenAICompatConfig{
			BaseURL: p.baseURL, APIKey: p.apiKey, WireModel: p.WireModel, MaxRetries: p.maxRetries,
		}), nil
	}
	return nil, fmt.Errorf("embedconfig: unresolved provider %q", p.Provider)
}

// String is the profile both binaries log at startup (no endpoint, no
// key), so an operator -- and helix-labs -- can compare them.
func (p Profile) String() string {
	rule := "none"
	if p.AnswerMinSimilarity != nil {
		rule = fmt.Sprintf("cosine>=%.3f", *p.AnswerMinSimilarity)
	}
	return fmt.Sprintf("provider=%s model=%s:%s wire_model=%q dims=%d metric=cosine query_prefix=%q document_prefix=%q answer_rule=%s",
		p.Provider, p.ModelID, p.Revision, p.WireModel, p.Dims, p.QueryPrefix, p.DocumentPrefix, rule)
}

// QueryPrefix and DocumentPrefix return the task-instruction prefix a
// given model_id requires for retrieval, per that model's own
// documentation -- nomic-embed-text-v1.5's model card: "search_query: "
// for queries, "search_document: " for documents/passages being
// indexed. This is versioned preprocessing keyed by model_id, which is
// already part of the binding identity that triggers regeneration on
// change (helix-kernel's source uniqueness constraint) -- never an
// arbitrary string a handler happens to prepend, and never applied to a
// model that doesn't call for it (Mock, or any future model without a
// documented prefix requirement, both fall through to "").
func QueryPrefix(modelID string) string {
	switch modelID {
	case "nomic-embed-text":
		return "search_query: "
	default:
		return ""
	}
}

func DocumentPrefix(modelID string) string {
	switch modelID {
	case "nomic-embed-text":
		return "search_document: "
	default:
		return ""
	}
}

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func getenvInt(key string, fallback int) int {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return fallback
	}
	return n
}
