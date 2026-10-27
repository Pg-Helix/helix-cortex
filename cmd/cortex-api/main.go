// Command cortex-api is the Workflow 2 (ADR-0007) ingestion/retrieval
// HTTP service. A separate binary and a separate credential from
// cmd/cortex's embedding worker -- this process never calls
// claim_job/publish_result/get_job_input, mirroring RFC-0003's own
// privilege boundary for the embedding worker itself. Chunk embedding
// jobs this process enqueues (via the capture trigger, automatically,
// on every chunk INSERT) are picked up by cmd/cortex's unmodified
// worker loop, running as helix_worker.
package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/Pg-Helix/helix-cortex/internal/api"
	"github.com/Pg-Helix/helix-cortex/internal/auth"
	"github.com/Pg-Helix/helix-cortex/internal/store"
	"github.com/Pg-Helix/helix-cortex/internal/testhooks"
	"github.com/Pg-Helix/helix-infer/embed"
	protocol "github.com/Pg-Helix/helix-protocol/gen/go"
	"github.com/jackc/pgx/v5/pgxpool"
)

func main() {
	databaseURL := os.Getenv("DATABASE_URL")
	if databaseURL == "" {
		log.Fatal("cortex-api: DATABASE_URL is required (a connection authenticated as helix_cortex_api)")
	}

	authenticator := mustAuthenticator()

	// Deliberately a distinct default revision from cmd/cortex's own
	// "v1" default: this target_model is claimed by a worker instance
	// running with CORTEX_INPUT_EXTRACTOR=chunk-text, which must never
	// be the same target_model a "raw"-mode instance (Workflow 1's
	// generic demo) claims -- they'd apply the wrong extraction strategy
	// to each other's jobs.
	modelID := getenv("CORTEX_CHUNK_MODEL_ID", "local-mock-embed")
	modelRevision := getenv("CORTEX_CHUNK_MODEL_REVISION", "chunk-v1")
	dims := getenvInt("CORTEX_CHUNK_MODEL_DIMS", 8)
	targetModel := modelID + ":" + modelRevision

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	pool, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		log.Fatalf("cortex-api: connect: %v", err)
	}
	defer pool.Close()

	reg := embed.NewRegistry()
	if err := reg.Register(protocol.Model{Id: modelID, Revision: modelRevision, Capability: protocol.Embed, Dims: &dims}); err != nil {
		log.Fatalf("cortex-api: register model: %v", err)
	}
	embedder, err := embed.NewEmbedder(reg, embed.RegistryKey(modelID, modelRevision), embed.NewMock(dims), embed.Options{MaxConcurrency: 4})
	if err != nil {
		log.Fatalf("cortex-api: build embedder: %v", err)
	}

	var hooks *testhooks.Registry
	if os.Getenv("CORTEX_TEST_HOOKS") == "1" {
		hooks = testhooks.NewRegistry()
		log.Print("cortex-api: CORTEX_TEST_HOOKS=1 -- in-flight-revocation test barrier endpoints are enabled; never set this in a real deployment")
	}

	srv := &api.Server{
		Store: &store.Store{
			Pool:               pool,
			ChunkModelID:       modelID,
			ChunkModelRevision: modelRevision,
			ChunkDims:          dims,
		},
		Authenticator: authenticator,
		Embedder:      embedder,
		TargetModel:   targetModel,
		TestHooks:     hooks,
	}

	addr := getenv("CORTEX_API_ADDR", ":8080")
	httpServer := &http.Server{Addr: addr, Handler: srv.Handler()}
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = httpServer.Shutdown(shutdownCtx)
	}()

	log.Printf("cortex-api: listening on %s (target_model=%s)", addr, targetModel)
	if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("cortex-api: %v", err)
	}
}

// mustAuthenticator fails closed (ADR-0007 decision 1): there is no "no
// auth configured" mode. CORTEX_AUTH_PROFILE must be exactly
// "local-fixture" today (the only implementation that exists; a real
// OIDC profile is a deferred, separate Authenticator behind the same
// interface), and the fixture file it names must parse and name at
// least one principal.
func mustAuthenticator() auth.Authenticator {
	profile := os.Getenv("CORTEX_AUTH_PROFILE")
	if profile != "local-fixture" {
		log.Fatalf("cortex-api: CORTEX_AUTH_PROFILE must be \"local-fixture\" (got %q) -- refusing to start with no configured authentication", profile)
	}
	path := os.Getenv("CORTEX_AUTH_FIXTURE_FILE")
	if path == "" {
		log.Fatal("cortex-api: CORTEX_AUTH_FIXTURE_FILE is required when CORTEX_AUTH_PROFILE=local-fixture")
	}
	a, err := auth.NewLocalFixtureAuthenticator(path)
	if err != nil {
		log.Fatalf("cortex-api: %v", err)
	}
	return a
}

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func getenvInt(key string, fallback int) int {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return fallback
	}
	return n
}
