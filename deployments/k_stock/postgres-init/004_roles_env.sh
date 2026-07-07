#!/usr/bin/env bash
set -euo pipefail

: "${POSTGRES_WRITER_PASSWORD:?POSTGRES_WRITER_PASSWORD must be set}"
: "${POSTGRES_READER_PASSWORD:?POSTGRES_READER_PASSWORD must be set}"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  -v writer_password="$POSTGRES_WRITER_PASSWORD" \
  -v reader_password="$POSTGRES_READER_PASSWORD" <<'SQL'
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'trading_writer') THEN
        CREATE ROLE trading_writer WITH LOGIN;
    END IF;

    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'trading_reader') THEN
        CREATE ROLE trading_reader WITH LOGIN;
    END IF;
END
$$;

ALTER ROLE trading_writer WITH PASSWORD :'writer_password';
ALTER ROLE trading_reader WITH PASSWORD :'reader_password';

GRANT USAGE ON SCHEMA public TO trading_writer;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading_writer;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO trading_writer;

GRANT USAGE ON SCHEMA public TO trading_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO trading_reader;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT ALL ON TABLES TO trading_writer;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE, SELECT ON SEQUENCES TO trading_writer;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO trading_reader;
SQL
