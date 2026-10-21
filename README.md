# helix-cortex

Connectors, ingestion, embeddings, semantic catalog, retrieval, RAG and evaluation.

| | |
|---|---|
| Ownership group | AI and Retrieval |
| Primary form | Go / SQL |
| Visibility | PUBLIC |
| Status | **C1: worker only.** `cmd/cortex` claims a job, fetches its bounded input via `helix_ai.get_job_input`, embeds it with helix-infer's deterministic Mock provider, and publishes through `helix_ai.publish_result` — the smallest real workflow-1 slice. Connectors/ingestion/retrieval/catalog/evaluation below are still design only. |

Design source of truth: the private `helix-atlas` repository (`architecture/repos/helix-cortex.md`).
Organization master prompt: PgHelix Organization Master Prompt, 16 Oct 2026.

## The worker (`cmd/cortex`)

```
DATABASE_URL=postgres://helix_worker:...@host/db go run ./cmd/cortex
```

Connects as the restricted `helix_worker` role (helix-kernel's
`test/regress/sql/setup.sql`/ADR-0006) — no privilege on application
tables, only `EXECUTE` on `claim_job`/`get_job_input`/`publish_result`.
Registers and embeds with a deterministic, no-network mock model
(`CORTEX_MODEL_ID`/`CORTEX_MODEL_REVISION`/`CORTEX_MODEL_DIMS`, defaults
`local-mock-embed:v1`, 8 dims) — no paid provider or external credential.
See `internal/worker` for the claim/fetch-input/embed/publish logic.

## Build and test

```
go build ./...
go vet ./...
go test ./... -race
```

`internal/worker/integration_test.go`'s
`TestIntegration_CrashReleaseAndDuplicateDelivery` additionally needs a
real PostgreSQL with helix-kernel installed; set
`CORTEX_TEST_APP_DATABASE_URL` (an app-owner connection, for setup) and
`CORTEX_TEST_WORKER_DATABASE_URL` (authenticated as `helix_worker`) to run
it — it skips otherwise, so the rest of the suite stays DB-free.

## License

Not yet decided. The license is an ADR in `helix-atlas`; until it is accepted, no license is granted and all rights are reserved.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).
