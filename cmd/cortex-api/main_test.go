package main

import (
	"strings"
	"testing"
)

const realProvider = "CORTEX_EMBED_PROVIDER=openai-compat CORTEX_EMBED_BASE_URL=http://unused.invalid CORTEX_EMBED_WIRE_MODEL=nomic-embed-text:v1.5"
const nomic = "CORTEX_CHUNK_MODEL_ID=nomic-embed-text CORTEX_CHUNK_MODEL_REVISION=v1.5"

func setenv(t *testing.T, kv string) {
	for _, k := range []string{"CORTEX_EMBED_PROVIDER", "CORTEX_EMBED_BASE_URL", "CORTEX_EMBED_WIRE_MODEL", "CORTEX_BACKFILL_TENANT",
		"CORTEX_CHUNK_MODEL_ID", "CORTEX_CHUNK_MODEL_REVISION", "CORTEX_CHUNK_MODEL_DIMS"} {
		t.Setenv(k, "")
	}
	for _, f := range strings.Fields(kv) {
		k, v, _ := strings.Cut(f, "=")
		t.Setenv(k, v)
	}
}

// The startup matrix through the API's own env reading (startupProfile
// runs before any connection or listener exists).
func TestStartupProfile_Matrix(t *testing.T) {
	for _, c := range []struct {
		name, env string
		wantErr   string  // "" = starts
		wantRule  float64 // 0 = no answer rule
	}{
		{"supported Nomic v1.5 / 768 / nomic tag", realProvider + " " + nomic + " CORTEX_CHUNK_MODEL_DIMS=768", "", 0.707},
		{"ordinary mock (defaults)", "", "", 0},
		{"mock provider + Nomic v1.5 identity", nomic + " CORTEX_CHUNK_MODEL_DIMS=768", "calibrated embedding profile", 0},
		{"wrong wire tag", realProvider + " " + nomic + " CORTEX_CHUNK_MODEL_DIMS=768 CORTEX_EMBED_WIRE_MODEL=nomic-embed-text:latest", "calibrated embedding profile", 0},
		{"96 dims", realProvider + " " + nomic + " CORTEX_CHUNK_MODEL_DIMS=96", "calibrated embedding profile", 0},
		{"unknown real revision", realProvider + " CORTEX_CHUNK_MODEL_ID=nomic-embed-text CORTEX_CHUNK_MODEL_REVISION=v2 CORTEX_CHUNK_MODEL_DIMS=768", "no calibrated /answers similarity threshold", 0},
		{"backfill: mock + Nomic v1.5 identity", "CORTEX_BACKFILL_TENANT=acme " + nomic + " CORTEX_CHUNK_MODEL_DIMS=768", "calibrated embedding profile", 0},
		{"backfill: mock version change chunk-v3/16", "CORTEX_BACKFILL_TENANT=acme CORTEX_CHUNK_MODEL_REVISION=chunk-v3 CORTEX_CHUNK_MODEL_DIMS=16", "", 0},
		{"backfill: new real revision before calibration", "CORTEX_BACKFILL_TENANT=acme " + realProvider + " CORTEX_CHUNK_MODEL_ID=nomic-embed-text CORTEX_CHUNK_MODEL_REVISION=v2 CORTEX_CHUNK_MODEL_DIMS=768", "", 0},
	} {
		t.Run(c.name, func(t *testing.T) {
			setenv(t, c.env)
			p, err := startupProfile()
			if c.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), c.wantErr) {
					t.Fatalf("want error containing %q, got %v (%s)", c.wantErr, err, p)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			got := 0.0
			if p.AnswerMinSimilarity != nil {
				got = *p.AnswerMinSimilarity
			}
			if got != c.wantRule {
				t.Fatalf("answer rule = %v, want %v (%s)", got, c.wantRule, p)
			}
		})
	}
}
