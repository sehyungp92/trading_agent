#!/usr/bin/env bash
# Database initialization script for crypto_trader.
# Runs inside the postgres container via /docker-entrypoint-initdb.d/.
set -euo pipefail

echo "=== crypto_trader: initializing database ==="

# ── Timezone ──────────────────────────────────────────────────────
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-SQL
  ALTER DATABASE $POSTGRES_DB SET timezone TO 'UTC';
SQL

# ── Roles ─────────────────────────────────────────────────────────
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-SQL
  -- Writer role (engine INSERT/UPDATE/DELETE)
  DO \$\$
  BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'trading_writer') THEN
      CREATE ROLE trading_writer LOGIN PASSWORD '${POSTGRES_WRITER_PASSWORD}';
    END IF;
  END
  \$\$;

  -- Reader role (dashboard SELECT only)
  DO \$\$
  BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'trading_reader') THEN
      CREATE ROLE trading_reader LOGIN PASSWORD '${POSTGRES_READER_PASSWORD}';
    END IF;
  END
  \$\$;
SQL

# ── Run numbered migrations ──────────────────────────────────────
MIGRATION_DIR="/docker-entrypoint-initdb.d/migrations"
if [ -d "$MIGRATION_DIR" ]; then
  for f in "$MIGRATION_DIR"/*.sql; do
    echo "  applying migration: $(basename "$f")"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -f "$f"
  done
fi

# ── Grant privileges ─────────────────────────────────────────────
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-SQL
  -- Writer: full DML on all tables
  GRANT USAGE ON SCHEMA public TO trading_writer;
  GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO trading_writer;
  GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO trading_writer;
  ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO trading_writer;
  ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE, SELECT ON SEQUENCES TO trading_writer;

  -- Reader: SELECT only
  GRANT USAGE ON SCHEMA public TO trading_reader;
  GRANT SELECT ON ALL TABLES IN SCHEMA public TO trading_reader;
  ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO trading_reader;
SQL

echo "=== crypto_trader: database ready ==="
