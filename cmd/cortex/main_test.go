package main

import (
	"strings"
	"testing"
)

func setenv(t *testing.T, kv string) {
	for _, k := range []string{"CORTEX_EMBED_PROVIDER", "CORTEX_EMBED_BASE_URL", "CORTEX_EMBED_WIRE_MODEL",
		"CORTEX_MODEL_ID", "CORTEX_MODEL_REVISION", "CORTEX_MODEL_DIMS"} {
		t.Setenv(k, "")
	}
	for _, f := range strings.Fields(kv) {
		k, v, _ := strings.Cut(f, "=")
		t.Setenv(k, v)
	}
}

// The worker writes vectors under the identity it claims: it must refuse
// to claim a calibrated identity it does not actually match.
func TestWorkerProfile_Matrix(t *testing.T) {
	real := "CORTEX_EMBED_PROVIDER=openai-compat CORTEX_EMBED_BASE_URL=http://unused.invalid CORTEX_EMBED_WIRE_MODEL=nomic-embed-text:v1.5"
	nomic := "CORTEX_MODEL_ID=nomic-embed-text CORTEX_MODEL_REVISION=v1.5"
	for _, c := range []struct{ name, env, wantErr, wantDocPrefix string }{
		{"chunk-worker supported Nomic v1.5", real + " " + nomic + " CORTEX_MODEL_DIMS=768", "", "search_document: "},
		{"chunk-worker ordinary mock chunk-v1", "CORTEX_MODEL_REVISION=chunk-v1", "", ""},
		{"Workflow 1 raw worker (defaults)", "", "", ""},
		{"mock version change chunk-v2", "CORTEX_MODEL_REVISION=chunk-v2", "", ""},
		{"mock + Nomic v1.5 identity", nomic + " CORTEX_MODEL_DIMS=768", "calibrated embedding profile", ""},
		{"Nomic v1.5, 96 dims", real + " " + nomic + " CORTEX_MODEL_DIMS=96", "calibrated embedding profile", ""},
		{"Nomic v1.5, wrong wire tag", real + " " + nomic + " CORTEX_MODEL_DIMS=768 CORTEX_EMBED_WIRE_MODEL=other:1", "calibrated embedding profile", ""},
	} {
		t.Run(c.name, func(t *testing.T) {
			setenv(t, c.env)
			p, err := workerProfile()
			if c.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), c.wantErr) {
					t.Fatalf("want error containing %q, got %v (%s)", c.wantErr, err, p)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if p.DocumentPrefix != c.wantDocPrefix {
				t.Fatalf("document prefix %q, want %q", p.DocumentPrefix, c.wantDocPrefix)
			}
		})
	}
}
