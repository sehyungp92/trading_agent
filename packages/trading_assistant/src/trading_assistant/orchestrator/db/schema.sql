-- Event queue with idempotent deduplication and dead-letter support
CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,
    bot_id          TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    payload         TEXT NOT NULL,
    exchange_timestamp TEXT NOT NULL,
    received_at     TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending | processing | acked | failed | dead_letter
    processed_at    TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    retry_count     INTEGER NOT NULL DEFAULT 0,
    max_retries     INTEGER NOT NULL DEFAULT 3,
    last_error      TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_status ON events(status);
CREATE INDEX IF NOT EXISTS idx_events_bot_id ON events(bot_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_status_created_at ON events(status, created_at);

CREATE TABLE IF NOT EXISTS relay_quarantine (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT NOT NULL,
    raw_event_id    TEXT NOT NULL DEFAULT '',
    reason          TEXT NOT NULL,
    payload         TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_relay_quarantine_source_created_at
    ON relay_quarantine(source, created_at);

CREATE TABLE IF NOT EXISTS relay_ingest_classifications (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT NOT NULL,
    raw_event_id    TEXT NOT NULL DEFAULT '',
    event_id        TEXT NOT NULL DEFAULT '',
    classification  TEXT NOT NULL,
    reason          TEXT NOT NULL DEFAULT '',
    payload         TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_relay_ingest_source_created_at
    ON relay_ingest_classifications(source, created_at);
CREATE INDEX IF NOT EXISTS idx_relay_ingest_event_id
    ON relay_ingest_classifications(event_id);

-- Watermark tracking for relay pull protocol. The relay returns a single
-- globally-ordered stream and the orchestrator stores one row keyed
-- "relay". The column is named bot_id for legacy reasons; in practice it
-- is a free-form stream identifier (typically the literal string "relay").
-- See orchestrator/adapters/vps_receiver.py and docs/plans/implementation.md.
CREATE TABLE IF NOT EXISTS watermarks (
    bot_id      TEXT PRIMARY KEY,
    last_event_id TEXT NOT NULL,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
