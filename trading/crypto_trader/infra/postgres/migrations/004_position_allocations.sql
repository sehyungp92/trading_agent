-- 004_position_allocations.sql: exchange truth and strategy ownership read models.

CREATE TABLE IF NOT EXISTS strategy_position_allocations (
    position_instance_id TEXT PRIMARY KEY,
    strategy_id          TEXT NOT NULL,
    symbol               TEXT NOT NULL,
    direction            TEXT NOT NULL,
    allocated_qty        DOUBLE PRECISION NOT NULL,
    avg_entry            DOUBLE PRECISION NOT NULL DEFAULT 0,
    risk_r               DOUBLE PRECISION NOT NULL DEFAULT 0,
    entry_time           TIMESTAMPTZ,
    status               TEXT NOT NULL DEFAULT 'OPEN',
    confidence           TEXT NOT NULL DEFAULT 'unknown',
    source               TEXT NOT NULL DEFAULT 'unknown',
    entry_order_ids      JSONB NOT NULL DEFAULT '[]',
    entry_fill_ids       JSONB NOT NULL DEFAULT '[]',
    exit_order_ids       JSONB NOT NULL DEFAULT '[]',
    exit_fill_ids        JSONB NOT NULL DEFAULT '[]',
    metadata             JSONB NOT NULL DEFAULT '{}',
    last_update_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_strategy_position_allocations_symbol
    ON strategy_position_allocations(symbol, direction, strategy_id);

CREATE TABLE IF NOT EXISTS exchange_positions (
    symbol            TEXT PRIMARY KEY,
    direction         TEXT NOT NULL,
    qty               DOUBLE PRECISION NOT NULL,
    avg_entry         DOUBLE PRECISION NOT NULL DEFAULT 0,
    unrealized_pnl    DOUBLE PRECISION NOT NULL DEFAULT 0,
    liquidation_price DOUBLE PRECISION,
    observed_at       TIMESTAMPTZ NOT NULL,
    metadata          JSONB NOT NULL DEFAULT '{}',
    last_update_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
