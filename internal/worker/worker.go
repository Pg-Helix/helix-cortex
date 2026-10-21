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

	res, err := w.Embedder.Embed(ctx, embed.EmbedRequest{Model: job.TargetModel, Input: []string{string(input)}})
	if err != nil {
		// Deliberately does not call PublishResult: an inference failure
		// (including a dimension mismatch Guarded itself caught) is not one
		// of publish_result's CAS outcomes, and isn't a signal this worker
		// can hand back distinctly from "nothing happened yet" -- the job
		// stays claimable until its lease expires and claim_job's own
		// retry/dead-letter policy (RFC-0003) takes over.
		return true, fmt.Errorf("worker: embed job %s: %w", job.JobID, err)
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
