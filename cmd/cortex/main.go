// Command cortex is the C1 worker: the smallest deterministic
// infer/cortex integration proving claim -> authorized input -> inference
// outside a database transaction -> short CAS publish (RFC-0003,
// ADR-0006). No paid provider or external credential -- embeds with
// helix-infer's deterministic Mock provider, per RFC-0005's local stack.
package main

import (
	"context"
	"encoding/json"
	"log"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/Pg-Helix/helix-cortex/internal/embedconfig"
	"github.com/Pg-Helix/helix-cortex/internal/ingest"
	"github.com/Pg-Helix/helix-cortex/internal/worker"
	"github.com/Pg-Helix/helix-infer/embed"
	protocol "github.com/Pg-Helix/helix-protocol/gen/go"
	"github.com/jackc/pgx/v5/pgxpool"
)

func main() {
	databaseURL := os.Getenv("DATABASE_URL")
	if databaseURL == "" {
		log.Fatal("cortex: DATABASE_URL is required (a connection authenticated as helix_worker or an equivalently restricted role)")
	}
	modelID := getenv("CORTEX_MODEL_ID", "local-mock-embed")
	modelRevision := getenv("CORTEX_MODEL_REVISION", "v1")
	dims := getenvInt("CORTEX_MODEL_DIMS", 8)
	leaseSeconds := getenvInt("CORTEX_LEASE_SECONDS", 30)
	maxAttempts := getenvInt("CORTEX_MAX_ATTEMPTS", 5)
	pollInterval := time.Duration(getenvInt("CORTEX_POLL_INTERVAL_MS", 500)) * time.Millisecond

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	pool, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		log.Fatalf("cortex: connect: %v", err)
	}
	defer pool.Close()

	reg := embed.NewRegistry()
	if err := reg.Register(protocol.Model{
		Id:         modelID,
		Revision:   modelRevision,
		Capability: protocol.Embed,
		Dims:       &dims,
	}); err != nil {
		log.Fatalf("cortex: register model: %v", err)
	}

	modelKey := embed.RegistryKey(modelID, modelRevision)
	targetModel := modelID + ":" + modelRevision
	provider, err := embedconfig.Build(dims)
	if err != nil {
		log.Fatalf("cortex: %v", err)
	}
	embedder, err := embed.NewEmbedder(reg, modelKey, provider, embed.Options{MaxConcurrency: 4})
	if err != nil {
		log.Fatalf("cortex: build embedder: %v", err)
	}

	embedTimeout := time.Duration(getenvInt("CORTEX_EMBED_TIMEOUT_SECONDS", leaseSeconds)) * time.Second
	w := &worker.Worker{
		Source:       &worker.DBSource{Pool: pool},
		Embedder:     embedder,
		TargetModel:  targetModel,
		LeaseSeconds: leaseSeconds,
		MaxAttempts:  maxAttempts,
		EmbedTimeout: embedTimeout,
	}
	// CORTEX_INPUT_EXTRACTOR selects a versioned input-extraction
	// strategy (ADR-0007): "raw" (default, unmodified from this
	// worker's original behavior -- the whole get_job_input row,
	// stringified) or "chunk-text" (extracts only cortex.chunk's own
	// `text` field, for a worker instance dedicated to Workflow 2's
	// chunk-embedding target_model -- never the same target_model as a
	// "raw" instance, or jobs meant for one extractor would be claimed
	// and misinterpreted by the other).
	switch extractor := getenv("CORTEX_INPUT_EXTRACTOR", "raw"); extractor {
	case "raw":
	case "chunk-text":
		// The model's own required task-instruction prefix (e.g.
		// nomic-embed-text's "search_document: ") is versioned
		// preprocessing keyed by modelID -- embedconfig.DocumentPrefix,
		// the same lookup cortex-api's query-side embedding uses for
		// "search_query: " -- composed here, not a string ingest.Split
		// or ExtractChunkText itself would need to know about.
		docPrefix := embedconfig.DocumentPrefix(modelID)
		w.InputExtractor = func(raw json.RawMessage) (string, error) {
			text, err := ingest.ExtractChunkText(raw)
			if err != nil {
				return "", err
			}
			return docPrefix + text, nil
		}
	default:
		log.Fatalf("cortex: unknown CORTEX_INPUT_EXTRACTOR %q (want \"raw\" or \"chunk-text\")", extractor)
	}

	log.Printf("cortex: worker loop started for target_model=%s poll_interval=%s", targetModel, pollInterval)
	for {
		select {
		case <-ctx.Done():
			log.Print("cortex: shutting down")
			return
		default:
		}

		handled, err := w.RunOnce(ctx)
		if err != nil {
			log.Printf("cortex: %v", err)
		}
		if !handled {
			select {
			case <-ctx.Done():
				return
			case <-time.After(pollInterval):
			}
		}
	}
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
