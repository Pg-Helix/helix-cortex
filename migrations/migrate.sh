#!/bin/sh
# Applies this directory's *.sql migrations to cortex's schema, in
# filename order, tracked by checksum in cortex.schema_migrations --
# the smallest mechanism that gives: migration identity (filename),
# drift detection (a previously-applied file whose content changed is a
# hard failure, never silently re-applied or silently ignored),
# serialized application (one file = one transaction: BEGIN, apply,
# record, COMMIT -- a mid-file SQL error rolls back that whole file,
# leaving schema_migrations exactly as it was before this run touched
# it), and a real fresh-install/upgrade story (run once against an
# empty schema, or run again against an already-migrated one -- the
# second run applies nothing and exits 0).
#
# Connects as whatever role/credentials the caller's PG* environment
# names (PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE, psql's own
# convention) and switches into helix_cortex_owner via SET ROLE for the
# DDL itself -- the schema's actual owner, distinct from the
# helix_cortex_api credential the running service uses for ordinary
# request handling. helix_cortex_api is never a member of
# helix_cortex_owner (see compose/initdb/01-install.sh), so it is
# incapable of SET ROLE helix_cortex_owner even if it tried: the
# separation is enforced by Postgres's own role-membership model, not
# merely by this script's convention of connecting as a different user.
set -eu
MIGRATIONS_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

checksum() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

psql -v ON_ERROR_STOP=1 -c "
    SET ROLE helix_cortex_owner;
    CREATE TABLE IF NOT EXISTS cortex.schema_migrations (
        filename   text PRIMARY KEY,
        checksum   text NOT NULL,
        applied_at timestamptz NOT NULL DEFAULT now()
    );
" >/dev/null

for f in "$MIGRATIONS_DIR"/*.sql; do
    name=$(basename "$f")
    sum=$(checksum "$f")

    already=$(psql -v ON_ERROR_STOP=1 -tAq -c "
        SELECT checksum FROM cortex.schema_migrations WHERE filename = '$name'
    ")
    if [ -n "$already" ]; then
        if [ "$already" != "$sum" ]; then
            echo "migrate: $name has already been applied with a DIFFERENT checksum ($already != $sum) -- refusing to continue. Migration files are immutable once applied; add a new file instead of editing this one." >&2
            exit 1
        fi
        echo "migrate: $name already applied, skipping"
        continue
    fi

    echo "migrate: applying $name"
    {
        echo "BEGIN;"
        echo "SET ROLE helix_cortex_owner;"
        cat "$f"
        echo "INSERT INTO cortex.schema_migrations (filename, checksum) VALUES ('$name', '$sum');"
        echo "COMMIT;"
    } | psql -v ON_ERROR_STOP=1 >/dev/null
done

echo "migrate: done"
