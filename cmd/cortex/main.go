// Command cortex is the C1 worker: the smallest deterministic
// infer/cortex integration proving claim -> authorized input -> inference
// outside a database transaction -> short CAS publish (RFC-0003,
// ADR-0006). No paid provider or external credential -- embeds with
// helix-infer's deterministic Mock provider, per RFC-0005's local stack.
package main

import (
	"context"
	"encoding/json"
	"fmt"
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
	// Everything that decides what this worker embeds, under which
	// identity, is resolved and validated once, before any connection:
	// the embedding profile (provider, identity, wire tag, dims,
	// prefixes) and the input extractor bound to it.
	cfg, err := resolveWorker()
	if err != nil {
		log.Fatalf("cortex: %v", err)
	}
	log.Printf("cortex: embedding profile: %s", cfg.profile)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	pool, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		log.Fatalf("cortex: connect: %v", err)
	}
	defer pool.Close()

	w, err := cfg.build(&worker.DBSource{Pool: pool})
	if err != nil {
		log.Fatalf("cortex: %v", err)
	}
	log.Printf("cortex: worker loop started for target_model=%s input_extractor=%s document_prefix=%q poll_interval=%s",
		w.TargetModel, cfg.extractor, cfg.documentPrefix(), cfg.pollInterval)
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
			case <-time.After(cfg.pollInterval):
			}
		}
	}
}

// workerConfig is this worker's one resolved configuration: build
// constructs the worker from it and main reports it, so what runs and
// what is logged cannot diverge.
type workerConfig struct {
	profile embedconfig.Profile
	// extractor is the effective input extractor (ADR-0007):
	//   "raw"        -- the whole get_job_input row, stringified (Workflow
	//                   1's original behavior, the default for profiles
	//                   that impose no document input);
	//   "chunk-text" -- profile.DocumentPrefix + the chunk row's text
	//                   field only (Workflow 2's chunk worker).
	// A worker for one target_model must never mix the two.
	extractor    string
	leaseSeconds int
	maxAttempts  int
	pollInterval time.Duration
	embedTimeout time.Duration
}

// resolveWorker reads and validates the worker's environment. A profile
// with a required document input (a calibrated identity, e.g.
// nomic-embed-text:v1.5 -> chunk-text) must set CORTEX_INPUT_EXTRACTOR to
// exactly that: the calibration's vectors were produced from
// "search_document: " + chunk text, so a worker embedding the raw row --
// explicitly, or by an absent/empty setting falling back to the default --
// would publish incompatible vectors under the calibrated identity.
func resolveWorker() (workerConfig, error) {
	profile, err := workerProfile()
	if err != nil {
		return workerConfig{}, err
	}
	extractor := os.Getenv("CORTEX_INPUT_EXTRACTOR")
	switch {
	case profile.DocumentInput != "" && extractor != profile.DocumentInput:
		return workerConfig{}, fmt.Errorf("%s:%s is a calibrated profile whose vectors require CORTEX_INPUT_EXTRACTOR=%s, got %q -- refusing to claim its jobs",
			profile.ModelID, profile.Revision, profile.DocumentInput, extractor)
	case extractor == "":
		extractor = "raw"
	case extractor != "raw" && extractor != "chunk-text":
		return workerConfig{}, fmt.Errorf("unknown CORTEX_INPUT_EXTRACTOR %q (want \"raw\" or \"chunk-text\")", extractor)
	}
	lease := getenvInt("CORTEX_LEASE_SECONDS", 30)
	return workerConfig{
		profile: profile, extractor: extractor,
		leaseSeconds: lease, maxAttempts: getenvInt("CORTEX_MAX_ATTEMPTS", 5),
		pollInterval: time.Duration(getenvInt("CORTEX_POLL_INTERVAL_MS", 500)) * time.Millisecond,
		embedTimeout: time.Duration(getenvInt("CORTEX_EMBED_TIMEOUT_SECONDS", lease)) * time.Second,
	}, nil
}

// documentPrefix is what chunk-text prepends; raw prepends nothing.
func (c workerConfig) documentPrefix() string {
	if c.extractor == "chunk-text" {
		return c.profile.DocumentPrefix
	}
	return ""
}

// build constructs the worker exactly as it will run against src: the
// registry entry and provider from the profile, and the input extractor.
func (c workerConfig) build(src worker.JobSource) (*worker.Worker, error) {
	dims := c.profile.Dims
	reg := embed.NewRegistry()
	if err := reg.Register(protocol.Model{Id: c.profile.ModelID, Revision: c.profile.Revision, Capability: protocol.Embed, Dims: &dims}); err != nil {
		return nil, fmt.Errorf("register model: %w", err)
	}
	provider, err := c.profile.NewProvider()
	if err != nil {
		return nil, err
	}
	key := embed.RegistryKey(c.profile.ModelID, c.profile.Revision)
	embedder, err := embed.NewEmbedder(reg, key, provider, embed.Options{MaxConcurrency: 4})
	if err != nil {
		return nil, fmt.Errorf("build embedder: %w", err)
	}
	w := &worker.Worker{
		Source: src, Embedder: embedder, TargetModel: c.profile.ModelID + ":" + c.profile.Revision,
		LeaseSeconds: c.leaseSeconds, MaxAttempts: c.maxAttempts, EmbedTimeout: c.embedTimeout,
	}
	if c.extractor == "chunk-text" {
		prefix := c.documentPrefix()
		w.InputExtractor = func(raw json.RawMessage) (string, error) {
			text, err := ingest.ExtractChunkText(raw)
			if err != nil {
				return "", err
			}
			return prefix + text, nil
		}
	}
	return w, nil
}

// workerProfile resolves this worker's embedding profile from its own
// env names (CORTEX_MODEL_*); defaults are Workflow 1's raw mock worker.
func workerProfile() (embedconfig.Profile, error) {
	return embedconfig.Resolve(getenv("CORTEX_MODEL_ID", "local-mock-embed"),
		getenv("CORTEX_MODEL_REVISION", "v1"), getenvInt("CORTEX_MODEL_DIMS", 8))
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
