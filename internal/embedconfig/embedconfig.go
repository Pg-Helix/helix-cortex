// Package embedconfig is the ONE validated configuration path for
// building an embedding provider -- used by both cmd/cortex (the
// chunk/raw worker) and cmd/cortex-api (the query embedder), so the
// provider type, endpoint, wire-model name and retry policy can never
// drift between the two processes that must agree on them. Each caller
// still reads its own model-id/revision/dims env vars under its own
// existing names (CORTEX_MODEL_* vs CORTEX_CHUNK_MODEL_*) and passes
// dims in; this package owns only the PROVIDER half of the config.
package embedconfig

import (
	"fmt"
	"os"
	"strconv"

	"github.com/Pg-Helix/helix-infer/embed"
)

// Build reads CORTEX_EMBED_PROVIDER ("mock", the default -- fast,
// network-free, never silently substituted for "openai-compat" even if
// that provider's own env vars are also set -- or "openai-compat") and
// constructs the matching provider. dims is the caller's own already-
// parsed model dimensionality, used to size Mock's deterministic
// vectors; it plays no role for openai-compat, whose real provider
// reports its own dims per response (validated by Guarded against the
// registry's declared value regardless of which provider is active).
func Build(dims int) (embed.EmbeddingProvider, error) {
	switch provider := getenv("CORTEX_EMBED_PROVIDER", "mock"); provider {
	case "mock":
		return embed.NewMock(dims), nil
	case "openai-compat":
		baseURL := os.Getenv("CORTEX_EMBED_BASE_URL")
		if baseURL == "" {
			return nil, fmt.Errorf("embedconfig: CORTEX_EMBED_BASE_URL is required when CORTEX_EMBED_PROVIDER=openai-compat")
		}
		wireModel := os.Getenv("CORTEX_EMBED_WIRE_MODEL")
		if wireModel == "" {
			return nil, fmt.Errorf("embedconfig: CORTEX_EMBED_WIRE_MODEL is required when CORTEX_EMBED_PROVIDER=openai-compat -- the internal model_id:revision binding is never sent to the server as-is")
		}
		// Not required: Ollama ignores this header's value entirely, and
		// a real OpenAI-compatible provider that DOES require a key is
		// still supported -- just set this env var in that profile.
		apiKey := os.Getenv("CORTEX_EMBED_API_KEY")
		maxRetries := getenvInt("CORTEX_EMBED_MAX_RETRIES", 2)
		return embed.NewOpenAICompat(embed.OpenAICompatConfig{
			BaseURL:    baseURL,
			APIKey:     apiKey,
			WireModel:  wireModel,
			MaxRetries: maxRetries,
		}), nil
	default:
		return nil, fmt.Errorf("embedconfig: unknown CORTEX_EMBED_PROVIDER %q (want \"mock\" or \"openai-compat\")", provider)
	}
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
