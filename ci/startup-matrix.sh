#!/bin/sh
# Startup matrix: starts cmd/cortex-api and cmd/cortex under one embedding
# configuration per row -- no database (the pool connects lazily; backfill
# stops at its first query), no model endpoint, a synthetic auth fixture --
# and checks each row against its EXPECTED outcome:
#
#   STARTED[:text]  reached its startup marker ("listening on" / "worker
#                   loop started") within the deadline, and logged text
#   REFUSED:text    exited before logging a resolved profile, with text in
#                   its fatal message (a configuration rejection)
#   DB              backfill: logged its resolved profile, then exited at
#                   its first database call (configuration accepted)
#
# Any other outcome -- wrong outcome, timeout without the marker, missing
# text -- fails the row; the script exits nonzero if any row failed or if
# the supported cortex-api and chunk-worker resolve different profiles.
# Usage: sh ci/startup-matrix.sh            (from the helix-cortex root)
#        BIN_DIR=dir sh ci/startup-matrix.sh  (prebuilt cortex-api + cortex)
set -eu
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
if [ -n "${BIN_DIR:-}" ]; then
    cp "$BIN_DIR/cortex-api" "$BIN_DIR/cortex" "$T/"
else
    go build -o "$T/cortex-api" ./cmd/cortex-api
    go build -o "$T/cortex" ./cmd/cortex
fi
printf '{"%s": {"principal": "p", "tenant": "t"}}\n' "$(od -An -tx1 -N24 /dev/urandom | tr -d ' \n')" > "$T/fixture.json"
REAL="CORTEX_EMBED_PROVIDER=openai-compat CORTEX_EMBED_BASE_URL=http://127.0.0.1:9/v1 CORTEX_EMBED_WIRE_MODEL=nomic-embed-text:v1.5"
API_NOMIC="CORTEX_CHUNK_MODEL_ID=nomic-embed-text CORTEX_CHUNK_MODEL_REVISION=v1.5"
W_NOMIC="CORTEX_MODEL_ID=nomic-embed-text CORTEX_MODEL_REVISION=v1.5"
CALIBRATED="is a calibrated embedding profile"
NEEDS_CHUNK_TEXT="require CORTEX_INPUT_EXTRACTOR=chunk-text"
failures=0

# row NAME EXPECTED BINARY ENV...
row() {
    name=$1 expected=$2 bin=$3; shift 3
    env -i PATH="$PATH" DATABASE_URL=postgres://u@127.0.0.1:9/db CORTEX_AUTH_PROFILE=local-fixture \
        CORTEX_AUTH_FIXTURE_FILE="$T/fixture.json" CORTEX_API_ADDR=127.0.0.1:0 CORTEX_POLL_INTERVAL_MS=200 \
        "$@" "$T/$bin" > "$T/out" 2>&1 &
    pid=$! i=0
    while [ $i -lt 100 ] && kill -0 $pid 2>/dev/null && ! grep -q "listening on\|worker loop started" "$T/out"; do
        sleep 0.1; i=$((i + 1))
    done
    if grep -q "listening on\|worker loop started" "$T/out"; then
        kill $pid; wait $pid 2>/dev/null || true; got=STARTED
    elif kill -0 $pid 2>/dev/null; then
        kill $pid; wait $pid 2>/dev/null || true; got=TIMEOUT
    else
        code=0; wait $pid || code=$?
        if [ $code -eq 0 ]; then got="EXITED-0"
        elif ! grep -q "embedding profile:" "$T/out"; then got=REFUSED
        elif grep -q "backfill: .*\(dial\|connect\)" "$T/out"; then got=DB
        else got="EXITED-$code"
        fi
    fi
    want=${expected%%:*}; text=""
    [ "$want" = "$expected" ] || text=${expected#*:}
    if [ "$got" = "$want" ] && { [ -z "$text" ] || grep -qF -- "$text" "$T/out"; }; then
        printf 'ok   %-48s %s\n' "$name" "$expected"
    else
        failures=$((failures + 1))
        printf 'FAIL %-48s expected %s, got %s: %s\n' "$name" "$expected" "$got" "$(tail -1 "$T/out" | cut -c1-160)"
    fi
    grep -o "embedding profile: .*" "$T/out" | head -1 > "$T/profile" || true
}

echo "== cortex-api (CORTEX_CHUNK_MODEL_*) =="
row "supported Nomic v1.5 / 768 / nomic tag"   "STARTED:answer_rule=cosine>=0.707" cortex-api $REAL $API_NOMIC CORTEX_CHUNK_MODEL_DIMS=768
cp "$T/profile" "$T/api-supported"
row "ordinary mock (defaults)"                  "STARTED:answer_rule=none"          cortex-api
row "mock provider + Nomic v1.5 identity"       "REFUSED:$CALIBRATED"              cortex-api $API_NOMIC CORTEX_CHUNK_MODEL_DIMS=768
row "Nomic v1.5, wrong wire tag"                "REFUSED:$CALIBRATED"              cortex-api $REAL CORTEX_EMBED_WIRE_MODEL=nomic-embed-text:latest $API_NOMIC CORTEX_CHUNK_MODEL_DIMS=768
row "Nomic v1.5, 96 dims"                       "REFUSED:$CALIBRATED"              cortex-api $REAL $API_NOMIC CORTEX_CHUNK_MODEL_DIMS=96
row "Nomic unknown revision v2"                 "REFUSED:no calibrated /answers similarity threshold" cortex-api $REAL CORTEX_CHUNK_MODEL_ID=nomic-embed-text CORTEX_CHUNK_MODEL_REVISION=v2 CORTEX_CHUNK_MODEL_DIMS=768
row "backfill: mock provider + Nomic v1.5"      "REFUSED:$CALIBRATED"              cortex-api CORTEX_BACKFILL_TENANT=t $API_NOMIC CORTEX_CHUNK_MODEL_DIMS=768
row "backfill: mock version change chunk-v3/16" "DB"                               cortex-api CORTEX_BACKFILL_TENANT=t CORTEX_CHUNK_MODEL_REVISION=chunk-v3 CORTEX_CHUNK_MODEL_DIMS=16
row "backfill: real revision v2, uncalibrated"  "DB"                               cortex-api CORTEX_BACKFILL_TENANT=t $REAL CORTEX_CHUNK_MODEL_ID=nomic-embed-text CORTEX_CHUNK_MODEL_REVISION=v2 CORTEX_CHUNK_MODEL_DIMS=768
echo "== cortex worker (CORTEX_MODEL_*) =="
row "chunk-worker supported Nomic v1.5"         'STARTED:input_extractor=chunk-text document_prefix="search_document: "' cortex $REAL CORTEX_INPUT_EXTRACTOR=chunk-text $W_NOMIC CORTEX_MODEL_DIMS=768
cp "$T/profile" "$T/worker-supported"
row "supported Nomic v1.5, extractor raw"       "REFUSED:$NEEDS_CHUNK_TEXT"        cortex $REAL CORTEX_INPUT_EXTRACTOR=raw $W_NOMIC CORTEX_MODEL_DIMS=768
row "supported Nomic v1.5, extractor empty"     "REFUSED:$NEEDS_CHUNK_TEXT"        cortex $REAL CORTEX_INPUT_EXTRACTOR= $W_NOMIC CORTEX_MODEL_DIMS=768
row "supported Nomic v1.5, extractor absent"    "REFUSED:$NEEDS_CHUNK_TEXT"        cortex $REAL $W_NOMIC CORTEX_MODEL_DIMS=768
row "Workflow 2 mock chunk worker chunk-v1"     'STARTED:input_extractor=chunk-text document_prefix=""' cortex CORTEX_INPUT_EXTRACTOR=chunk-text CORTEX_MODEL_REVISION=chunk-v1
row "mock version change chunk-worker chunk-v2" "STARTED:input_extractor=chunk-text" cortex CORTEX_INPUT_EXTRACTOR=chunk-text CORTEX_MODEL_REVISION=chunk-v2
row "chunk-worker mock + Nomic v1.5 identity"   "REFUSED:$CALIBRATED"              cortex CORTEX_INPUT_EXTRACTOR=chunk-text $W_NOMIC CORTEX_MODEL_DIMS=768
row "chunk-worker Nomic v1.5, 96 dims"          "REFUSED:$CALIBRATED"              cortex $REAL CORTEX_INPUT_EXTRACTOR=chunk-text $W_NOMIC CORTEX_MODEL_DIMS=96
row "Workflow 1 raw worker (defaults)"          "STARTED:input_extractor=raw"      cortex

if [ -s "$T/api-supported" ] && cmp -s "$T/api-supported" "$T/worker-supported"; then
    echo "ok   supported cortex-api and chunk-worker resolve the same profile"
else
    failures=$((failures + 1)); echo "FAIL supported cortex-api and chunk-worker resolve different (or no) profiles"
fi
echo "startup matrix: $failures failure(s)"
[ "$failures" -eq 0 ]
