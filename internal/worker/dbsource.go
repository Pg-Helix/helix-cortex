package worker

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/jackc/pgx/v5/pgxpool"
)

// DBSource is the real JobSource: each method is exactly one call to the
// helix_ai SQL function of the same name, over a pool that must already be
// connected as the restricted helix_worker role (or a role granted the
// same three EXECUTE privileges and nothing else -- see helix-kernel
// test/regress/sql/setup.sql).
type DBSource struct {
	Pool *pgxpool.Pool
}

func (s *DBSource) ClaimJob(ctx context.Context, targetModel string, leaseSeconds, maxAttempts int) (*Job, error) {
	// claim_job returns helix_ai.job, a single composite value that is
	// NULL when nothing is claimable -- used in a FROM clause, that still
	// produces exactly one row with every column NULL (verified against a
	// real PostgreSQL; it does NOT produce zero rows), so every column
	// must be scanned as nullable and checked explicitly.
	var jobID, sourceDigest, targetModel2, pipelineVersion *string
	var leaseEpoch *int
	err := s.Pool.QueryRow(ctx,
		`SELECT job_id::text, lease_epoch, source_digest, target_model, pipeline_version
		 FROM helix_ai.claim_job($1, $2, $3)`,
		targetModel, leaseSeconds, maxAttempts,
	).Scan(&jobID, &leaseEpoch, &sourceDigest, &targetModel2, &pipelineVersion)
	if err != nil {
		return nil, fmt.Errorf("claim_job: %w", err)
	}
	if jobID == nil {
		return nil, nil
	}
	return &Job{
		JobID:           *jobID,
		LeaseEpoch:      *leaseEpoch,
		SourceDigest:    *sourceDigest,
		TargetModel:     *targetModel2,
		PipelineVersion: *pipelineVersion,
	}, nil
}

func (s *DBSource) GetJobInput(ctx context.Context, job *Job) (json.RawMessage, error) {
	var raw []byte
	if err := s.Pool.QueryRow(ctx,
		`SELECT helix_ai.get_job_input($1, $2)`, job.JobID, job.LeaseEpoch,
	).Scan(&raw); err != nil {
		return nil, fmt.Errorf("get_job_input: %w", err)
	}
	return json.RawMessage(raw), nil
}

func (s *DBSource) PublishResult(ctx context.Context, job *Job, value json.RawMessage) (Outcome, error) {
	var outcome string
	if err := s.Pool.QueryRow(ctx,
		`SELECT helix_ai.publish_result($1, $2, $3, $4, $5, $6)`,
		job.JobID, job.LeaseEpoch, job.SourceDigest, job.TargetModel, job.PipelineVersion, []byte(value),
	).Scan(&outcome); err != nil {
		return "", fmt.Errorf("publish_result: %w", err)
	}
	return Outcome(outcome), nil
}
