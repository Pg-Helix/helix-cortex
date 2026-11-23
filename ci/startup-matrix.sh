#!/bin/sh
# Startup matrix: builds cmd/cortex-api and cmd/cortex and starts each
# under one embedding configuration per row, with no database (the pool
# connects lazily) and a synthetic auth fixture. Prints, per row, whether
# the process reached its listener / worker loop and which /answers rule
# it selected, or the startup error. Never contacts a model endpoint.
# Usage: sh ci/startup-matrix.sh   (from the helix-cortex root)
set -eu
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
go build -o "$T/cortex-api" ./cmd/cortex-api
go build -o "$T/cortex" ./cmd/cortex
printf '{"%s": {"principal": "p", "tenant": "t"}}\n' "$(od -An -tx1 -N24 /dev/urandom | tr -d ' \n')" > "$T/fixture.json"
REAL="CORTEX_EMBED_PROVIDER=openai-compat CORTEX_EMBED_BASE_URL=http://127.0.0.1:9/v1 CORTEX_EMBED_WIRE_MODEL=nomic-embed-text:v1.5"

# start BINARY ENV...: run for up to 3s, report the outcome line.
start() {
    bin=$1; shift
    env -i PATH="$PATH" DATABASE_URL=postgres://u@127.0.0.1:9/db CORTEX_AUTH_PROFILE=local-fixture \
        CORTEX_AUTH_FIXTURE_FILE="$T/fixture.json" CORTEX_API_ADDR=127.0.0.1:0 "$@" "$T/$bin" > "$T/out" 2>&1 &
    pid=$!; i=0
    while [ $i -lt 30 ] && kill -0 $pid 2>/dev/null && ! grep -q "listening on\|worker loop started" "$T/out"; do sleep 0.1; i=$((i + 1)); done
    if kill -0 $pid 2>/dev/null; then kill $pid; wait $pid 2>/dev/null || true; s=STARTED; else wait $pid 2>/dev/null || true; s=REFUSED; fi
    if [ $s = REFUSED ]; then
        detail=$(tail -1 "$T/out" | sed 's/^[0-9/: ]*//' | cut -c1-150)
        # Backfill exits at its first database call: getting that far
        # means the configuration itself was accepted.
        case "$detail" in *dial*|*connect*) s="ACCEPTED (stopped at database)"; detail=$(grep -o "embedding profile: .*" "$T/out" | cut -c1-150) ;; esac
    else
        detail=$(grep -o "evidence rule.*\|embedding profile: .*" "$T/out" | tr '\n' ' ' | cut -c1-260)
    fi
    grep -o "embedding profile: .*" "$T/out" | head -1 > "$T/last-profile" || true
    echo "$s | $detail"
}
row() { name=$1; shift; printf '%-46s ' "$name"; start "$@"; }

echo "== cortex-api (CORTEX_CHUNK_MODEL_*) =="
row "supported Nomic v1.5 / 768 / nomic tag"   cortex-api $REAL CORTEX_CHUNK_MODEL_ID=nomic-embed-text CORTEX_CHUNK_MODEL_REVISION=v1.5 CORTEX_CHUNK_MODEL_DIMS=768
cp "$T/last-profile" "$T/api-supported"
row "ordinary mock (defaults)"                  cortex-api
row "mock provider + Nomic v1.5 identity"       cortex-api CORTEX_CHUNK_MODEL_ID=nomic-embed-text CORTEX_CHUNK_MODEL_REVISION=v1.5 CORTEX_CHUNK_MODEL_DIMS=768
row "Nomic v1.5, wrong wire tag"                cortex-api $REAL CORTEX_EMBED_WIRE_MODEL=nomic-embed-text:latest CORTEX_CHUNK_MODEL_ID=nomic-embed-text CORTEX_CHUNK_MODEL_REVISION=v1.5 CORTEX_CHUNK_MODEL_DIMS=768
row "Nomic v1.5, 96 dims"                       cortex-api $REAL CORTEX_CHUNK_MODEL_ID=nomic-embed-text CORTEX_CHUNK_MODEL_REVISION=v1.5 CORTEX_CHUNK_MODEL_DIMS=96
row "Nomic unknown revision v2"                 cortex-api $REAL CORTEX_CHUNK_MODEL_ID=nomic-embed-text CORTEX_CHUNK_MODEL_REVISION=v2 CORTEX_CHUNK_MODEL_DIMS=768
row "backfill: mock provider + Nomic v1.5"      cortex-api CORTEX_BACKFILL_TENANT=t CORTEX_CHUNK_MODEL_ID=nomic-embed-text CORTEX_CHUNK_MODEL_REVISION=v1.5 CORTEX_CHUNK_MODEL_DIMS=768
row "backfill: mock version change chunk-v3/16" cortex-api CORTEX_BACKFILL_TENANT=t CORTEX_CHUNK_MODEL_REVISION=chunk-v3 CORTEX_CHUNK_MODEL_DIMS=16
echo "== cortex worker (CORTEX_MODEL_*) =="
row "chunk-worker supported Nomic v1.5"         cortex $REAL CORTEX_INPUT_EXTRACTOR=chunk-text CORTEX_MODEL_ID=nomic-embed-text CORTEX_MODEL_REVISION=v1.5 CORTEX_MODEL_DIMS=768
cp "$T/last-profile" "$T/worker-supported"
row "chunk-worker ordinary mock chunk-v1"       cortex CORTEX_INPUT_EXTRACTOR=chunk-text CORTEX_MODEL_REVISION=chunk-v1
row "chunk-worker mock + Nomic v1.5 identity"   cortex CORTEX_INPUT_EXTRACTOR=chunk-text CORTEX_MODEL_ID=nomic-embed-text CORTEX_MODEL_REVISION=v1.5 CORTEX_MODEL_DIMS=768
row "chunk-worker Nomic v1.5, 96 dims"          cortex $REAL CORTEX_INPUT_EXTRACTOR=chunk-text CORTEX_MODEL_ID=nomic-embed-text CORTEX_MODEL_REVISION=v1.5 CORTEX_MODEL_DIMS=96
row "Workflow 1 raw worker (defaults)"          cortex
if [ -s "$T/api-supported" ] && cmp -s "$T/api-supported" "$T/worker-supported"; then
    echo "supported cortex-api and chunk-worker resolve the same profile: yes"
else
    echo "supported cortex-api and chunk-worker resolve the same profile: NO"; exit 1
fi
