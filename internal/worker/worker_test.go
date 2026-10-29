package worker

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/Pg-Helix/helix-infer/embed"
	protocol "github.com/Pg-Helix/helix-protocol/gen/go"
)

// fakeSource is an in-memory JobSource: lets RunOnce's claim/embed/publish
// sequencing be tested without a live PostgreSQL. Call counts catch the
// "must not call PublishResult on an inference failure" invariant.
type fakeSource struct {
	job          *Job
	input        json.RawMessage
	claimErr     error
	inputErr     error
	publishErr   error
	outcome      Outcome
	claimCalls   int
	inputCalls   int
	publishCalls int
	publishedJob *Job
}

func (f *fakeSource) ClaimJob(ctx context.Context, targetModel string, leaseSeconds, maxAttempts int) (*Job, error) {
	f.claimCalls++
	if f.claimErr != nil {
		return nil, f.claimErr
	}
	return f.job, nil
}

func (f *fakeSource) GetJobInput(ctx context.Context, job *Job) (json.RawMessage, error) {
	f.inputCalls++
	if f.inputErr != nil {
		return nil, f.inputErr
	}
	return f.input, nil
}

func (f *fakeSource) PublishResult(ctx context.Context, job *Job, value json.RawMessage) (Outcome, error) {
	f.publishCalls++
	f.publishedJob = job
	if f.publishErr != nil {
		return "", f.publishErr
	}
	if f.outcome == "" {
		return OutcomeOK, nil
	}
	return f.outcome, nil
}

// mustEmbedder builds a Guarded expecting dims-dimensional vectors (the
// registry's declared dims, which NewEmbedder uses as the expected-dims
// check) wrapping provider, which may legitimately report a different
// dims itself -- that mismatch is exactly what some tests below exercise.
func mustEmbedder(t *testing.T, dims int, provider embed.EmbeddingProvider) *embed.Guarded {
	t.Helper()
	reg := embed.NewRegistry()
	d := dims
	if err := reg.Register(protocol.Model{Id: "mock", Revision: "v1", Capability: protocol.Embed, Dims: &d}); err != nil {
		t.Fatalf("register: %v", err)
	}
	g, err := embed.NewEmbedder(reg, embed.RegistryKey("mock", "v1"), provider, embed.Options{MaxConcurrency: 1})
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}
	return g
}

func TestRunOnce_NoClaimableJob(t *testing.T) {
	src := &fakeSource{job: nil}
	w := &Worker{Source: src, Embedder: mustEmbedder(t, 8, embed.NewMock(8)), TargetModel: "mock:v1"}

	handled, err := w.RunOnce(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if handled {
		t.Fatal("expected handled=false when nothing is claimable")
	}
	if src.inputCalls != 0 || src.publishCalls != 0 {
		t.Fatalf("must not fetch input or publish when nothing was claimed: input=%d publish=%d", src.inputCalls, src.publishCalls)
	}
}

func TestRunOnce_HappyPath(t *testing.T) {
	job := &Job{JobID: "job-1", LeaseEpoch: 1, SourceDigest: "d1", TargetModel: "mock:v1", PipelineVersion: "p1"}
	src := &fakeSource{job: job, input: json.RawMessage(`{"id":1,"body":"hello"}`)}
	w := &Worker{Source: src, Embedder: mustEmbedder(t, 8, embed.NewMock(8)), TargetModel: "mock:v1"}

	handled, err := w.RunOnce(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !handled {
		t.Fatal("expected handled=true")
	}
	if src.publishCalls != 1 {
		t.Fatalf("expected exactly one publish, got %d", src.publishCalls)
	}
	if src.publishedJob != job {
		t.Fatal("published the wrong job")
	}
}

func TestRunOnce_DimensionMismatchNeverPublishes(t *testing.T) {
	job := &Job{JobID: "job-1", LeaseEpoch: 1, SourceDigest: "d1", TargetModel: "mock:v1", PipelineVersion: "p1"}
	src := &fakeSource{job: job, input: json.RawMessage(`{"id":1}`)}
	// Provider reports 4 dims; registry/Guarded expects 8 -- Guarded itself
	// must catch this (embed.ErrDimMismatch) before RunOnce ever reaches
	// PublishResult.
	w := &Worker{Source: src, Embedder: mustEmbedder(t, 8, embed.NewMock(4)), TargetModel: "mock:v1"}

	handled, err := w.RunOnce(context.Background())
	if err == nil {
		t.Fatal("expected a dimension-mismatch error")
	}
	if !handled {
		t.Fatal("a claimed job that failed inference still counts as handled (it was attempted)")
	}
	if !errors.Is(err, embed.ErrDimMismatch) {
		t.Fatalf("expected ErrDimMismatch, got %v", err)
	}
	if src.publishCalls != 0 {
		t.Fatalf("must never publish after a dimension mismatch, got %d calls", src.publishCalls)
	}
}

func TestRunOnce_CancellationNeverPublishes(t *testing.T) {
	job := &Job{JobID: "job-1", LeaseEpoch: 1, SourceDigest: "d1", TargetModel: "mock:v1", PipelineVersion: "p1"}
	src := &fakeSource{job: job, input: json.RawMessage(`{"id":1}`)}
	w := &Worker{Source: src, Embedder: mustEmbedder(t, 8, embed.NewMock(8)), TargetModel: "mock:v1"}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	handled, err := w.RunOnce(ctx)
	if err == nil {
		t.Fatal("expected an error from a canceled context")
	}
	if !handled {
		t.Fatal("a claimed job that was canceled mid-flight still counts as handled")
	}
	if src.publishCalls != 0 {
		t.Fatalf("must never publish once the context was canceled, got %d calls", src.publishCalls)
	}
}

// recordingProvider captures the Input it was actually asked to embed --
// used to prove InputExtractor controls what reaches the embedder, not
// just that RunOnce still succeeds.
type recordingProvider struct {
	dims     int
	lastReq  embed.EmbedRequest
	provider embed.EmbeddingProvider
}

func (r *recordingProvider) Embed(ctx context.Context, req embed.EmbedRequest) (*embed.EmbedResult, error) {
	r.lastReq = req
	return r.provider.Embed(ctx, req)
}

func TestRunOnce_InputExtractor_Nil_UsesWholeRowStringified(t *testing.T) {
	job := &Job{JobID: "job-1", LeaseEpoch: 1, SourceDigest: "d1", TargetModel: "mock:v1", PipelineVersion: "p1"}
	raw := json.RawMessage(`{"id":1,"body":"hello"}`)
	src := &fakeSource{job: job, input: raw}
	rec := &recordingProvider{dims: 8, provider: embed.NewMock(8)}
	w := &Worker{Source: src, Embedder: mustEmbedder(t, 8, rec), TargetModel: "mock:v1"}

	if _, err := w.RunOnce(context.Background()); err != nil {
		t.Fatalf("RunOnce: %v", err)
	}
	if len(rec.lastReq.Input) != 1 || rec.lastReq.Input[0] != string(raw) {
		t.Fatalf("expected the whole raw row as input, got %q", rec.lastReq.Input)
	}
}

func TestRunOnce_InputExtractor_Chunk_ExtractsOnlyText(t *testing.T) {
	job := &Job{JobID: "job-1", LeaseEpoch: 1, SourceDigest: "d1", TargetModel: "mock:v1", PipelineVersion: "p1"}
	raw := json.RawMessage(`{"document_id":"doc-1","tenant":"acme","text":"the real passage"}`)
	src := &fakeSource{job: job, input: raw}
	rec := &recordingProvider{dims: 8, provider: embed.NewMock(8)}
	w := &Worker{
		Source: src, Embedder: mustEmbedder(t, 8, rec), TargetModel: "mock:v1",
		InputExtractor: func(raw json.RawMessage) (string, error) {
			var row struct {
				Text string `json:"text"`
			}
			if err := json.Unmarshal(raw, &row); err != nil {
				return "", err
			}
			return row.Text, nil
		},
	}

	if _, err := w.RunOnce(context.Background()); err != nil {
		t.Fatalf("RunOnce: %v", err)
	}
	if len(rec.lastReq.Input) != 1 || rec.lastReq.Input[0] != "the real passage" {
		t.Fatalf("expected only the text field as input, got %q", rec.lastReq.Input)
	}
}

func TestRunOnce_RejectedOutcomeIsAnError(t *testing.T) {
	job := &Job{JobID: "job-1", LeaseEpoch: 1, SourceDigest: "d1", TargetModel: "mock:v1", PipelineVersion: "p1"}
	src := &fakeSource{job: job, input: json.RawMessage(`{"id":1}`), outcome: OutcomeStaleLease}
	w := &Worker{Source: src, Embedder: mustEmbedder(t, 8, embed.NewMock(8)), TargetModel: "mock:v1"}

	handled, err := w.RunOnce(context.Background())
	if err == nil {
		t.Fatal("expected an error when publish_result returns a non-ok outcome")
	}
	if !handled {
		t.Fatal("expected handled=true")
	}
}
