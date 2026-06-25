#!/usr/bin/env bash
set -euo pipefail

source .env

mkdir -p "database-dumps"
SQL_OUTPUT_FILE="database-dumps/$(date '+%Y-%m-%d-%H-%M').sql"

export PGPASSWORD="$POSTGRES_PASSWORD"

/usr/lib/postgresql/17/bin/pg_dump \
  --dbname="$POSTGRES_DB" \
  --host="$POSTGRES_HOST" \
  --port="$POSTGRES_PORT" \
  --username="$POSTGRES_USER" \
  --no-owner \
  --no-acl \
  --format=p \
  > "$SQL_OUTPUT_FILE"

echo "Database dump created at: $SQL_OUTPUT_FILE"