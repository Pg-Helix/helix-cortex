package embedconfig

import (
	"os"
	"testing"
)

func withEnv(t *testing.T, kv map[string]string, fn func()) {
	t.Helper()
	for k, v := range kv {
		old, had := os.LookupEnv(k)
		os.Setenv(k, v)
		t.Cleanup(func() {
			if had {
				os.Setenv(k, old)
			} else {
				os.Unsetenv(k)
			}
		})
	}
	fn()
}

func TestBuild_DefaultsToMock(t *testing.T) {
	withEnv(t, map[string]string{"CORTEX_EMBED_PROVIDER": ""}, func() {
		p, err := Build(8)
		if err != nil {
			t.Fatal(err)
		}
		if p == nil {
			t.Fatal("expected a non-nil provider")
		}
	})
}

func TestBuild_OpenAICompatRequiresBaseURL(t *testing.T) {
	withEnv(t, map[string]string{"CORTEX_EMBED_PROVIDER": "openai-compat", "CORTEX_EMBED_WIRE_MODEL": "m:1"}, func() {
		_, err := Build(768)
		if err == nil {
			t.Fatal("expected an error when CORTEX_EMBED_BASE_URL is unset")
		}
	})
}

func TestBuild_OpenAICompatRequiresWireModel(t *testing.T) {
	withEnv(t, map[string]string{"CORTEX_EMBED_PROVIDER": "openai-compat", "CORTEX_EMBED_BASE_URL": "http://unused.invalid"}, func() {
		_, err := Build(768)
		if err == nil {
			t.Fatal("expected an error when CORTEX_EMBED_WIRE_MODEL is unset")
		}
	})
}

func TestBuild_OpenAICompatSucceedsWithRequiredFields(t *testing.T) {
	withEnv(t, map[string]string{
		"CORTEX_EMBED_PROVIDER":   "openai-compat",
		"CORTEX_EMBED_BASE_URL":   "http://unused.invalid",
		"CORTEX_EMBED_WIRE_MODEL": "nomic-embed-text:v1.5",
	}, func() {
		p, err := Build(768)
		if err != nil {
			t.Fatal(err)
		}
		if p == nil {
			t.Fatal("expected a non-nil provider")
		}
	})
}

func TestBuild_RejectsUnknownProvider(t *testing.T) {
	withEnv(t, map[string]string{"CORTEX_EMBED_PROVIDER": "not-a-real-provider"}, func() {
		_, err := Build(8)
		if err == nil {
			t.Fatal("expected an error for an unknown provider")
		}
	})
}

func TestPrefixes(t *testing.T) {
	if got := QueryPrefix("nomic-embed-text"); got != "search_query: " {
		t.Fatalf("got %q, want %q", got, "search_query: ")
	}
	if got := DocumentPrefix("nomic-embed-text"); got != "search_document: " {
		t.Fatalf("got %q, want %q", got, "search_document: ")
	}
	if got := QueryPrefix("local-mock-embed"); got != "" {
		t.Fatalf("mock model should have no prefix, got %q", got)
	}
	if got := DocumentPrefix("local-mock-embed"); got != "" {
		t.Fatalf("mock model should have no prefix, got %q", got)
	}
}
