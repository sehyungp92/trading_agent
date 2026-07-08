CREATE TABLE IF NOT EXISTS instrumentation_events (
    event_id TEXT PRIMARY KEY,
    logical_event_id TEXT,
    event_type TEXT NOT NULL,
    bot_id TEXT NOT NULL,
    family_id TEXT,
    portfolio_id TEXT,
    account_alias TEXT,
    strategy_id TEXT,
    symbol TEXT,
    exchange_timestamp TIMESTAMPTZ,
    local_timestamp TIMESTAMPTZ,
    payload JSONB NOT NULL,
    lineage JSONB,
    received_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_instrumentation_events_type_ts
    ON instrumentation_events (event_type, exchange_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_instrumentation_events_strategy_ts
    ON instrumentation_events (strategy_id, exchange_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_instrumentation_events_portfolio_ts
    ON instrumentation_events (portfolio_id, exchange_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_instrumentation_events_logical
    ON instrumentation_events (logical_event_id);

CREATE INDEX IF NOT EXISTS idx_instrumentation_events_decision
    ON instrumentation_events ((COALESCE(payload->>'decision_id', payload->'payload'->>'decision_id')));

CREATE INDEX IF NOT EXISTS idx_instrumentation_events_bar
    ON instrumentation_events ((COALESCE(payload->>'bar_id', payload->'payload'->>'bar_id')));
