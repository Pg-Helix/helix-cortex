package worker

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/Pg-Helix/helix-infer/embed"
	protocol "github.com/Pg-Helix/helix-protocol/gen/go"
	"github.com/jackc/pgx/v5/pgxpool"
)

// TestIntegration_CrashReleaseAndDuplicateDelivery requires a real
// PostgreSQL with helix-kernel installed and a helix_worker role already
// granted claim_job/publish_result/get_job_input (test/regress/sql/setup.sql
// in helix-kernel). Skips (does not fail) when either env var below is
// unset, so `go test ./...` stays fast and DB-free everywhere else.
//
// Two separate connections, deliberately: CORTEX_TEST_APP_DATABASE_URL
// (an app-owner/superuser connection) does the setup only an application
// role should do (CREATE TABLE, register_source, INSERT) --
// CORTEX_TEST_WORKER_DATABASE_URL must authenticate as the real,
// restricted helix_worker role for every Worker/DBSource call, which is
// the one guarantee the review asked to prove against a real database and
// a real restricted role, not a superuser standing in for one.
func TestIntegration_CrashReleaseAndDuplicateDelivery(t *testing.T) {
	appURL := os.Getenv("CORTEX_TEST_APP_DATABASE_URL")
	workerURL := os.Getenv("CORTEX_TEST_WORKER_DATABASE_URL")
	if appURL == "" || workerURL == "" {
		t.Skip("CORTEX_TEST_APP_DATABASE_URL / CORTEX_TEST_WORKER_DATABASE_URL not set; skipping integration test (needs a real PostgreSQL with helix-kernel installed, an app-owner connection, and a helix_worker-authenticated connection)")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	appPool, err := pgxpool.New(ctx, appURL)
	if err != nil {
		t.Fatalf("connect (app): %v", err)
	}
	defer appPool.Close()

	workerPool, err := pgxpool.New(ctx, workerURL)
	if err != nil {
		t.Fatalf("connect (worker): %v", err)
	}
	defer workerPool.Close()

	const targetModel = "embed-v1:cortex-integration-rev"
	setupSQL := []string{
		`CREATE TABLE IF NOT EXISTS public.cortex_it (id bigint PRIMARY KEY, body text)`,
		`SET helix.tenant = 'cortex-it-tenant'`,
		`SELECT helix_ai.register_source('public', 'cortex_it', 'embed-v1', 'cortex-integration-rev', 'pipeline-v1')`,
	}
	for _, stmt := range setupSQL {
		if _, err := appPool.Exec(ctx, stmt); err != nil {
			t.Fatalf("setup %q: %v", stmt, err)
		}
	}
	t.Cleanup(func() {
		appPool.Exec(context.Background(), `DROP TABLE IF EXISTS public.cortex_it CASCADE`)
	})

	if _, err := appPool.Exec(ctx, `INSERT INTO public.cortex_it VALUES (1, 'alpha') ON CONFLICT (id) DO NOTHING`); err != nil {
		t.Fatalf("insert: %v", err)
	}

	reg := embed.NewRegistry()
	w := &Worker{
		Source:       &DBSource{Pool: workerPool},
		Embedder:     mustEmbedderForIntegration(t, reg, 8),
		TargetModel:  targetModel,
		LeaseSeconds: 1, // short lease: expires fast for the crash/re-lease half below
		MaxAttempts:  5,
	}

	// First worker claims the job but "crashes" before publishing: just
	// don't call RunOnce's publish step -- claim directly instead so we
	// keep the job_id/lease_epoch to attempt a late publish afterward.
	crashed, err := w.Source.ClaimJob(ctx, targetModel, 1, 5)
	if err != nil {
		t.Fatalf("claim (worker A): %v", err)
	}
	if crashed == nil {
		t.Fatal("expected a claimable job for worker A")
	}

	// Wait out the 1s lease, then run the real worker loop: it re-claims
	// at a higher epoch and completes normally -- the duplicate-delivery
	// half (the SAME job_id now has two workers who each believe they
	// hold a live lease).
	time.Sleep(1200 * time.Millisecond)
	handled, err := w.RunOnce(ctx)
	if err != nil {
		t.Fatalf("worker B RunOnce: %v", err)
	}
	if !handled {
		t.Fatal("expected worker B to claim and complete the re-leased job")
	}

	// Worker A's "crashed" lease is now stale; its publish must be
	// rejected, not silently accepted alongside worker B's.
	outcome, err := w.Source.PublishResult(ctx, crashed, []byte(`[0,0,0,0,0,0,0,0]`))
	if err != nil {
		t.Fatalf("worker A late publish: %v", err)
	}
	if outcome != OutcomeStaleLease {
		t.Fatalf("expected worker A's late publish to be rejected as stale_lease, got %q", outcome)
	}
}

func mustEmbedderForIntegration(t *testing.T, reg *embed.Registry, dims int) *embed.Guarded {
	t.Helper()
	d := dims
	if err := reg.Register(protocol.Model{Id: "embed-v1", Revision: "cortex-integration-rev", Capability: protocol.Embed, Dims: &d}); err != nil {
		t.Fatalf("register: %v", err)
	}
	g, err := embed.NewEmbedder(reg, embed.RegistryKey("embed-v1", "cortex-integration-rev"), embed.NewMock(dims), embed.Options{MaxConcurrency: 1})
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}
	return g
}
