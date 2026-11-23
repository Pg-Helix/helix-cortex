// Package worker is the C1 integration slice: claim a job, fetch its
// bounded authorized input (helix_ai.get_job_input, ADR-0006), run
// inference OUTSIDE any database transaction, then a short CAS publish.
// JobSource is the only boundary to the database -- RunOnce itself never
// touches *sql.DB/pgx directly, so the claim/embed/publish sequencing is
// unit-testable with a fake JobSource and helix-infer's real Mock provider,
// with no live PostgreSQL required for that part.
package worker

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/Pg-Helix/helix-infer/embed"
)

// Job is the claimed job's identity, exactly as returned by
// helix_ai.claim_job -- the fields a worker needs to fetch input and later
// publish, nothing it would need application-table privilege to read.
type Job struct {
	JobID           string
	LeaseEpoch      int
	SourceDigest    string
	TargetModel     string
	PipelineVersion string
}

// Outcome mirrors helix_ai.publish_outcome (kernel/sql/helix--0.1.0.sql).
type Outcome string

const (
	OutcomeOK               Outcome = "ok"
	OutcomeStaleLease       Outcome = "stale_lease"
	OutcomeStaleSource      Outcome = "stale_source"
	OutcomeModelMismatch    Outcome = "model_mismatch"
	OutcomePipelineMismatch Outcome = "pipeline_mismatch"
	OutcomeTombstoned       Outcome = "tombstoned"
	OutcomeDeadLettered     Outcome = "dead_lettered"
)

// JobSource is the database boundary: claim/fetch-input/publish, each one
// exactly the SQL function of the same name in helix-kernel. A real
// implementation (see dbsource.go) calls these through a restricted
// helix_worker connection; tests can fake this interface entirely.
type JobSource interface {
	ClaimJob(ctx context.Context, targetModel string, leaseSeconds, maxAttempts int) (*Job, error)
	GetJobInput(ctx context.Context, job *Job) (json.RawMessage, error)
	PublishResult(ctx context.Context, job *Job, value json.RawMessage) (Outcome, error)
}

// Worker runs one model's claim/embed/publish loop.
type Worker struct {
	Source       JobSource
	Embedder     *embed.Guarded
	TargetModel  string
	LeaseSeconds int
	MaxAttempts  int

	// InputExtractor, if set, extracts the text actually handed to the
	// embedder from get_job_input's full jsonb row -- added for ADR-0007
	// (Workflow 2): a chunk row carries document_id/tenant/digests
	// alongside its own text, and embedding the whole row verbatim would
	// bury the real content in its own metadata. nil preserves this
	// worker's original, unmodified behavior (the whole row, stringified)
	// -- a backward-compatible addition to this contract, not new
	// behavior for any existing caller.
	InputExtractor func(json.RawMessage) (string, error)

	// EmbedTimeout bounds a single embed call -- with the mock provider
	// this was moot (no network, effectively instant), but a real
	// provider call has no deadline otherwise beyond this worker's
	// entire process lifetime (RunOnce is called with the process's
	// root context, cancelled only on shutdown). 0 disables the bound
	// (ctx is used as-is), which previously was the ONLY behavior.
	EmbedTimeout time.Duration
}

func (w *Worker) extractInput(raw json.RawMessage) (string, error) {
	if w.InputExtractor == nil {
		return string(raw), nil
	}
	return w.InputExtractor(raw)
}

// RunOnce claims at most one job and, if one was claimable, fetches its
// input, embeds it, and publishes the result. Returns handled=false (no
// error) when nothing was claimable -- the caller's loop should back off,
// not treat that as a failure.
//
// Inference happens entirely between GetJobInput and PublishResult, as a
// plain Go call with no database statement open around it -- there is
// nothing here for a long model call to block inside.
func (w *Worker) RunOnce(ctx context.Context) (handled bool, err error) {
	job, err := w.Source.ClaimJob(ctx, w.TargetModel, w.LeaseSeconds, w.MaxAttempts)
	if err != nil {
		return false, fmt.Errorf("worker: claim: %w", err)
	}
	if job == nil {
		return false, nil
	}

	input, err := w.Source.GetJobInput(ctx, job)
	if err != nil {
		return true, fmt.Errorf("worker: get input for job %s: %w", job.JobID, err)
	}

	text, err := w.extractInput(input)
	if err != nil {
		return true, fmt.Errorf("worker: extract input for job %s: %w", job.JobID, err)
	}

	embedCtx := ctx
	if w.EmbedTimeout > 0 {
		var cancel context.CancelFunc
		embedCtx, cancel = context.WithTimeout(ctx, w.EmbedTimeout)
		defer cancel()
	}
	res, err := w.Embedder.Embed(embedCtx, embed.EmbedRequest{Model: job.TargetModel, Input: []string{text}})
	if err != nil {
		// Deliberately does not call PublishResult: an inference failure
		// (including a dimension mismatch Guarded itself caught) is not one
		// of publish_result's CAS outcomes, and isn't a signal this worker
		// can hand back distinctly from "nothing happened yet" -- the job
		// stays claimable until its lease expires and claim_job's own
		// retry/dead-letter policy (RFC-0003) takes over.
		return true, fmt.Errorf("worker: embed job %s: %w", job.JobID, err)
	}

	// Defensive even though the adapter's own boundary validation (see
	// helix-infer/internal/providers) already guarantees len(Vectors) ==
	// len(Input) == 1 whenever Embed returns no error: a worker that
	// indexes Vectors[0] unconditionally would panic on any future
	// provider whose validation has a gap, rather than failing this one
	// job the way every other embed error above already does.
	if len(res.Vectors) != 1 {
		return true, fmt.Errorf("worker: embed job %s: provider returned %d vectors for 1 input", job.JobID, len(res.Vectors))
	}
	value, err := json.Marshal(res.Vectors[0])
	if err != nil {
		return true, fmt.Errorf("worker: marshal result for job %s: %w", job.JobID, err)
	}

	outcome, err := w.Source.PublishResult(ctx, job, value)
	if err != nil {
		return true, fmt.Errorf("worker: publish job %s: %w", job.JobID, err)
	}
	if outcome != OutcomeOK {
		return true, fmt.Errorf("worker: job %s rejected: %s", job.JobID, outcome)
	}
	return true, nil
}
