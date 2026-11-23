package embedconfig

import (
	"os"
	"strings"
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

// realEnv is the supported Foundry real-embeddings provider config.
var realEnv = map[string]string{
	"CORTEX_EMBED_PROVIDER":   "openai-compat",
	"CORTEX_EMBED_BASE_URL":   "http://unused.invalid",
	"CORTEX_EMBED_WIRE_MODEL": "nomic-embed-text:v1.5",
}

func with(kv map[string]string, extra ...string) map[string]string {
	out := map[string]string{}
	for k, v := range kv {
		out[k] = v
	}
	for i := 0; i+1 < len(extra); i += 2 {
		out[extra[i]] = extra[i+1]
	}
	return out
}

func TestResolve_DefaultsToMockWithoutRule(t *testing.T) {
	withEnv(t, map[string]string{"CORTEX_EMBED_PROVIDER": ""}, func() {
		p, err := Resolve("local-mock-embed", "chunk-v1", 8)
		if err != nil {
			t.Fatal(err)
		}
		if p.Provider != "mock" || p.AnswerMinSimilarity != nil || p.RequireAnswerRule() != nil {
			t.Fatalf("ordinary mock: %s", p)
		}
		if prov, err := p.NewProvider(); err != nil || prov == nil {
			t.Fatalf("NewProvider: %v %v", prov, err)
		}
	})
}

func TestResolve_OpenAICompatRequiredFields(t *testing.T) {
	for name, env := range map[string]map[string]string{
		"no base url":   with(realEnv, "CORTEX_EMBED_BASE_URL", ""),
		"no wire model": with(realEnv, "CORTEX_EMBED_WIRE_MODEL", ""),
		"unknown":       {"CORTEX_EMBED_PROVIDER": "not-a-real-provider"},
	} {
		withEnv(t, env, func() {
			if _, err := Resolve("other-model", "v1", 768); err == nil {
				t.Errorf("%s: expected an error", name)
			}
		})
	}
}

func TestResolve_SupportedProfileActivatesTheCalibratedRule(t *testing.T) {
	withEnv(t, realEnv, func() {
		p, err := Resolve("nomic-embed-text", "v1.5", 768)
		if err != nil {
			t.Fatal(err)
		}
		if p.AnswerMinSimilarity == nil || *p.AnswerMinSimilarity != 0.707 || p.RequireAnswerRule() != nil {
			t.Fatalf("supported profile: %s", p)
		}
		if p.QueryPrefix != "search_query: " || p.DocumentPrefix != "search_document: " || p.WireModel != "nomic-embed-text:v1.5" {
			t.Fatalf("supported profile: %s", p)
		}
		if prov, err := p.NewProvider(); err != nil || prov == nil {
			t.Fatalf("NewProvider: %v %v", prov, err)
		}
	})
}

// The defect: the calibrated identity (nomic-embed-text:v1.5) used to
// activate 0.707 whatever actually produced the vectors.
func TestResolve_CalibratedIdentityMustMatchEveryField(t *testing.T) {
	for name, c := range map[string]struct {
		env  map[string]string
		dims int
	}{
		"mock provider":  {map[string]string{"CORTEX_EMBED_PROVIDER": "mock"}, 768},
		"wrong wire tag": {with(realEnv, "CORTEX_EMBED_WIRE_MODEL", "nomic-embed-text:latest"), 768},
		"wrong dims":     {realEnv, 96},
	} {
		withEnv(t, c.env, func() {
			p, err := Resolve("nomic-embed-text", "v1.5", c.dims)
			if err == nil || !strings.Contains(err.Error(), "calibrated embedding profile") {
				t.Errorf("%s: want a calibrated-profile mismatch error, got %v (profile %s)", name, err, p)
			}
		})
	}
}

func TestResolve_UncalibratedRealRevisionResolvesButCannotServe(t *testing.T) {
	withEnv(t, with(realEnv, "CORTEX_EMBED_WIRE_MODEL", "nomic-embed-text:v2"), func() {
		p, err := Resolve("nomic-embed-text", "v2", 768)
		if err != nil {
			t.Fatal(err) // registering/backfilling a new real revision stays possible
		}
		if p.AnswerMinSimilarity != nil || p.RequireAnswerRule() == nil {
			t.Fatalf("uncalibrated real revision must not serve /answers: %s", p)
		}
	})
}

func TestProfileString_NoSecrets(t *testing.T) {
	withEnv(t, with(realEnv, "CORTEX_EMBED_API_KEY", "sk-should-not-appear"), func() {
		p, err := Resolve("nomic-embed-text", "v1.5", 768)
		if err != nil {
			t.Fatal(err)
		}
		if s := p.String(); strings.Contains(s, "sk-should-not-appear") || strings.Contains(s, "unused.invalid") {
			t.Fatalf("profile string leaks endpoint/key: %s", s)
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
