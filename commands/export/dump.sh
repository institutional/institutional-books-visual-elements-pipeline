source .env

# Create local backup of the database
mkdir -p "${DATA_DIR_PATH}database-dumps"
SQL_OUTPUT_FILE="${DATA_DIR_PATH}database-dumps/$(date '+%Y-%m-%d-%H-%M').sql"

export PGPASSWORD="$POSTGRES_PASSWORD"

pg_dump \
    --dbname="$POSTGRES_DB" \
    --host="$POSTGRES_HOST" \
    --port="$POSTGRES_PORT" \
    --username="$POSTGRES_USER" \
    --no-owner \
    --no-acl \
    --format=p \
    > "$SQL_OUTPUT_FILE"