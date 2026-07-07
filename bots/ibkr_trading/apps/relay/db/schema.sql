-- Relay event buffer schema

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id    TEXT    NOT NULL UNIQUE,
    bot_id      TEXT    NOT NULL,
    event_type  TEXT    NOT NULL,
    payload     TEXT    NOT NULL,
    exchange_timestamp TEXT,
    received_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    acked       INTEGER NOT NULL DEFAULT 0,
    priority    INTEGER NOT NULL DEFAULT 3
);

CREATE INDEX IF NOT EXISTS idx_events_acked     ON events(acked);
CREATE INDEX IF NOT EXISTS idx_events_priority  ON events(priority);
CREATE INDEX IF NOT EXISTS idx_events_bot_id    ON events(bot_id);
CREATE INDEX IF NOT EXISTS idx_events_event_id  ON events(event_id);
CREATE INDEX IF NOT EXISTS idx_events_received  ON events(received_at);
