# helix-cortex

Connectors, ingestion, embeddings, semantic catalog, retrieval, RAG and evaluation.

| | |
|---|---|
| Ownership group | AI and Retrieval |
| Primary form | Go / SQL |
| Visibility | PUBLIC |
| Status | **Workflow 1 worker + Workflow 2 ingestion/retrieval API (unmerged feature branches).** `cmd/cortex` is the claim/embed/publish worker (also run as the chunk-worker). `cmd/cortex-api` serves authorized ingestion, hybrid FTS+pgvector search with a deterministic reranker, extractive answers with citations, and evidence lookup (ADR-0007). Embeddings come from helix-infer: Mock by default, or an OpenAI-compatible endpoint (`CORTEX_EMBED_PROVIDER=openai-compat`). Connectors, semantic catalog and generative RAG are still design only. See helix-labs `MANIFEST.md` for the tested commits. |

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

## Schema migrations (`migrations/`)

`migrations/migrate.sh` is the only supported way to apply `cortex` schema
changes. It applies `*.sql` in order, one transaction per file, and records
each file's checksum in `cortex.schema_migrations`. A rerun applies nothing,
and an edited, already-applied file is refused. Connect it as a trusted setup
role through the standard `PG*` variables (for example the Foundry bootstrap
admin). It then switches to `SET ROLE helix_cortex_owner` for the DDL.
Running the SQL as `helix_cortex_api` fails by design (`permission denied
for schema cortex`): the runtime credential cannot create or alter schema
objects. See helix-foundry's README for the Compose invocation.

## Embedding profile and the `/answers` rule

Both binaries resolve one embedding profile at startup (`internal/embedconfig.Resolve`) and log it as `embedding profile: ...`. The profile covers the provider, internal model id and revision, wire model, dimensions and query/document prefixes. The metric is always cosine, fixed in the SQL. The provider, the identity that vectors are written under and the `/answers` similarity threshold all come from that one profile.

- The calibrated profile is `openai-compat`, `nomic-embed-text:v1.5`, wire tag `nomic-embed-text:v1.5`, 768 dimensions, with Nomic's `search_query: ` / `search_document: ` prefixes. Only that exact profile activates the 0.707 threshold (helix-atlas ADR-0008).
- A configuration that claims that identity with any other provider (including mock), wire tag or dimensionality is refused by the API, the backfill mode and the worker before they connect.
- A real provider without a calibrated profile can be registered and backfilled, but `cortex-api` will not serve with it.
- Mock bindings run without the rule.
- `sh ci/startup-matrix.sh` prints every case.

Cortex can only check the configuration. It cannot see which weights an OpenAI-compatible endpoint actually serves, because any server can answer to a familiar tag. That guarantee comes from helix-foundry: its Ollama image bakes in the pinned weights digest and fails the build on a mismatch. helix-labs also checks the running manifest's digest against the pin before every evaluation.

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
