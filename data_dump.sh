#!/usr/bin/env bash
set -euo pipefail

cd /home/idi-dev/institutional-books-1-ve-pipeline

source .env

mkdir -p "${DATA_DIR_PATH}/database-dumps"
SQL_OUTPUT_FILE="${DATA_DIR_PATH}/database-dumps/$(date '+%Y-%m-%d-%H-%M').sql"

export PGPASSWORD="$POSTGRES_PASSWORD"

/usr/bin/pg_dump \
  --dbname="$POSTGRES_DB" \
  --host="$POSTGRES_HOST" \
  --port="$POSTGRES_PORT" \
  --username="$POSTGRES_USER" \
  --no-owner \
  --no-acl \
  --format=p \
  --table caption \
  > "$SQL_OUTPUT_FILE"

echo "Database dump created at: $SQL_OUTPUT_FILE"