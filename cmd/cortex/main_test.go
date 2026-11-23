package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/Pg-Helix/helix-cortex/internal/worker"
	"github.com/Pg-Helix/helix-infer/embed"
)

func setenv(t *testing.T, kv string) {
	for _, k := range []string{"CORTEX_EMBED_PROVIDER", "CORTEX_EMBED_BASE_URL", "CORTEX_EMBED_WIRE_MODEL", "CORTEX_INPUT_EXTRACTOR",
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

const supportedEnv = "CORTEX_EMBED_PROVIDER=openai-compat CORTEX_EMBED_BASE_URL=http://unused.invalid CORTEX_EMBED_WIRE_MODEL=nomic-embed-text:v1.5 " +
	"CORTEX_MODEL_ID=nomic-embed-text CORTEX_MODEL_REVISION=v1.5 CORTEX_MODEL_DIMS=768"

// setExtractor: "<absent>" unsets the variable entirely.
func setExtractor(t *testing.T, v string) {
	t.Setenv("CORTEX_INPUT_EXTRACTOR", "")
	if v == "<absent>" {
		os.Unsetenv("CORTEX_INPUT_EXTRACTOR")
		return
	}
	t.Setenv("CORTEX_INPUT_EXTRACTOR", v)
}

// The calibrated identity's vectors were produced from
// "search_document: " + chunk text: a worker claiming it must use exactly
// chunk-text -- explicit raw, empty or absent is refused before any claim.
func TestResolveWorker_ExtractorBoundToCalibratedIdentity(t *testing.T) {
	for _, c := range []struct{ name, env, extractor, want, wantErr string }{
		{"supported + chunk-text", supportedEnv, "chunk-text", "chunk-text", ""},
		{"supported + raw", supportedEnv, "raw", "", "require CORTEX_INPUT_EXTRACTOR=chunk-text"},
		{"supported + empty", supportedEnv, "", "", "require CORTEX_INPUT_EXTRACTOR=chunk-text"},
		{"supported + absent", supportedEnv, "<absent>", "", "require CORTEX_INPUT_EXTRACTOR=chunk-text"},
		{"Workflow 1 raw mock worker (defaults)", "", "<absent>", "raw", ""},
		{"Workflow 2 mock chunk worker", "CORTEX_MODEL_REVISION=chunk-v1", "chunk-text", "chunk-text", ""},
		{"mock version change chunk-v2", "CORTEX_MODEL_REVISION=chunk-v2", "chunk-text", "chunk-text", ""},
		{"mock explicit raw", "", "raw", "raw", ""},
		{"unknown extractor", "", "html", "", "unknown CORTEX_INPUT_EXTRACTOR"},
	} {
		t.Run(c.name, func(t *testing.T) {
			setenv(t, c.env)
			setExtractor(t, c.extractor)
			cfg, err := resolveWorker()
			if c.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), c.wantErr) {
					t.Fatalf("want error containing %q, got %v (extractor %q)", c.wantErr, err, cfg.extractor)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if cfg.extractor != c.want {
				t.Fatalf("effective extractor %q, want %q", cfg.extractor, c.want)
			}
		})
	}
}

// A representative cortex.chunk row as get_job_input returns it: text
// plus the identity/tenancy metadata that must never reach the model.
const chunkRow = `{"tenant":"acme","document_id":"7d0f3c1e-2b8a-4f61-9a55-0c1d2e3f4a5b","document_version_id":"v-1",` +
	`"collection_id":"c-1","content_digest":"9ce4386bcbe681879ca014b446ac47f366627d0a0b753f1ddb74a4e968c9429f",` +
	`"chunker_version":"chunker-v1","chunk_index":0,"text":"Employees accrue fifteen days of paid leave per year.",` +
	`"created_at":"2026-11-23T10:00:00Z"}`

type oneJobSource struct {
	job       *worker.Job
	input     json.RawMessage
	published json.RawMessage
}

func (s *oneJobSource) ClaimJob(ctx context.Context, model string, lease, attempts int) (*worker.Job, error) {
	if s.job == nil || s.job.TargetModel != model {
		return nil, nil
	}
	return s.job, nil
}
func (s *oneJobSource) GetJobInput(ctx context.Context, j *worker.Job) (json.RawMessage, error) {
	return s.input, nil
}
func (s *oneJobSource) PublishResult(ctx context.Context, j *worker.Job, v json.RawMessage) (worker.Outcome, error) {
	s.published = v
	return worker.OutcomeOK, nil
}

// The production wiring (resolveWorker -> build -> RunOnce) for the
// supported profile, against a local OpenAI-compatible endpoint that
// records exactly what it was asked to embed.
func TestSupportedWorker_SendsPrefixedChunkTextToTheProvider(t *testing.T) {
	var got struct {
		Model string   `json:"model"`
		Input []string `json:"input"`
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		vec := make([]float32, 768)
		vec[0] = 1
		json.NewEncoder(w).Encode(map[string]any{"model": got.Model, "data": []map[string]any{{"index": 0, "embedding": vec}}})
	}))
	defer srv.Close()
	setenv(t, supportedEnv+" CORTEX_EMBED_BASE_URL="+srv.URL+"/v1")
	setExtractor(t, "chunk-text")
	cfg, err := resolveWorker()
	if err != nil {
		t.Fatal(err)
	}
	src := &oneJobSource{job: &worker.Job{JobID: "j1", LeaseEpoch: 1, TargetModel: "nomic-embed-text:v1.5"}, input: json.RawMessage(chunkRow)}
	w, err := cfg.build(src)
	if err != nil {
		t.Fatal(err)
	}
	if handled, err := w.RunOnce(context.Background()); !handled || err != nil {
		t.Fatalf("RunOnce: handled=%v err=%v", handled, err)
	}
	want := "search_document: Employees accrue fifteen days of paid leave per year."
	if got.Model != "nomic-embed-text:v1.5" || len(got.Input) != 1 || got.Input[0] != want {
		t.Fatalf("provider received model=%q input=%q, want %q", got.Model, got.Input, want)
	}
	if strings.Count(got.Input[0], "search_document: ") != 1 {
		t.Fatalf("prefix applied more than once: %q", got.Input[0])
	}
	if src.published == nil {
		t.Fatal("the embedded vector was not published")
	}
}

// Workflow 1's default worker: mock provider, raw extractor -- the whole
// row reaches the provider. The mock vector is a pure function of its
// input text, so the published vector proves exactly what was embedded.
func TestWorkflow1Worker_SendsTheRawRow(t *testing.T) {
	setenv(t, "")
	setExtractor(t, "<absent>")
	cfg, err := resolveWorker()
	if err != nil {
		t.Fatal(err)
	}
	row := `{"id":1,"body":"hello world"}`
	src := &oneJobSource{job: &worker.Job{JobID: "j1", LeaseEpoch: 1, TargetModel: "local-mock-embed:v1"}, input: json.RawMessage(row)}
	w, err := cfg.build(src)
	if err != nil {
		t.Fatal(err)
	}
	if handled, err := w.RunOnce(context.Background()); !handled || err != nil {
		t.Fatalf("RunOnce: handled=%v err=%v", handled, err)
	}
	res, err := embed.NewMock(8).Embed(context.Background(), embed.EmbedRequest{Input: []string{row}})
	if err != nil {
		t.Fatal(err)
	}
	want, _ := json.Marshal(res.Vectors[0])
	if string(src.published) != string(want) {
		t.Fatalf("published %s, want the mock vector of the raw row %s", src.published, want)
	}
}
