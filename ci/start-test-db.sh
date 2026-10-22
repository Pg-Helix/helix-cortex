#!/bin/sh
# Starts a disposable PostgreSQL with the helix-kernel extension
# installed, for the real-database integration test
# (TestIntegration_CrashReleaseAndDuplicateDelivery). Prints the two
# DATABASE_URL env vars that test needs -- eval its output, or source it
# via `$(sh ci/start-test-db.sh)`.
#
# Usage: KERNEL_DIR=/path/to/helix-kernel sh ci/start-test-db.sh
set -eu
KERNEL_DIR=${KERNEL_DIR:?set KERNEL_DIR to a helix-kernel checkout}
PG_MAJOR=${PG_MAJOR:-17}
PORT=${PORT:-55433}
NAME=${NAME:-cortex-ci-postgres}

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" -e POSTGRES_HOST_AUTH_METHOD=trust -p "127.0.0.1:$PORT:5432" -v "$KERNEL_DIR":/src:ro "postgres:$PG_MAJOR" postgres >/dev/null
docker exec "$NAME" sh -c "apt-get update -qq && apt-get install -y -qq build-essential postgresql-server-dev-$PG_MAJOR >/dev/null" >&2
docker exec "$NAME" sh -c "mkdir /build && cp -r /src/. /build/ && cd /build && make >/dev/null && make install >/dev/null" >&2
docker restart "$NAME" >/dev/null
i=0
until docker exec "$NAME" pg_isready -U postgres >/dev/null 2>&1; do
    i=$((i + 1))
    [ "$i" -le 60 ] || { echo "start-test-db.sh: postgres never became ready" >&2; exit 1; }
    sleep 1
done

docker exec -u postgres "$NAME" psql -v ON_ERROR_STOP=1 -c "
CREATE ROLE helix_owner NOLOGIN NOSUPERUSER NOBYPASSRLS;
CREATE ROLE helix_worker LOGIN PASSWORD 'cortex-ci-worker';
CREATE EXTENSION helix;
" >&2
docker exec -u postgres "$NAME" psql -v ON_ERROR_STOP=1 -f /src/sql/post_install_ownership.sql >&2
docker exec -u postgres "$NAME" psql -v ON_ERROR_STOP=1 -c "
GRANT CREATE ON SCHEMA public TO helix_owner;
ALTER DEFAULT PRIVILEGES GRANT SELECT, TRIGGER ON TABLES TO helix_owner;
GRANT USAGE ON SCHEMA helix_ai TO helix_worker;
GRANT EXECUTE ON FUNCTION helix_ai.claim_job(text, integer, integer) TO helix_worker;
GRANT EXECUTE ON FUNCTION helix_ai.publish_result(uuid, integer, text, text, text, jsonb) TO helix_worker;
GRANT EXECUTE ON FUNCTION helix_ai.get_job_input(uuid, integer) TO helix_worker;
" >&2

echo "export CORTEX_TEST_APP_DATABASE_URL='postgres://postgres@127.0.0.1:$PORT/postgres?sslmode=disable'"
echo "export CORTEX_TEST_WORKER_DATABASE_URL='postgres://helix_worker:cortex-ci-worker@127.0.0.1:$PORT/postgres?sslmode=disable'"
