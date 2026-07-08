-- 001_tables.sql: Core tables for crypto_trader instrumentation.

-- 1. trades: completed round-trips (append-only, idempotent via trade_id PK)
CREATE TABLE IF NOT EXISTS trades (
    trade_id          TEXT PRIMARY KEY,
    strategy_id       TEXT NOT NULL,
    symbol            TEXT NOT NULL,
    direction         TEXT NOT NULL,
    entry_time        TIMESTAMPTZ NOT NULL,
    exit_time         TIMESTAMPTZ NOT NULL,
    entry_price       DOUBLE PRECISION NOT NULL,
    exit_price        DOUBLE PRECISION NOT NULL,
    position_size     DOUBLE PRECISION NOT NULL,
    pnl               DOUBLE PRECISION NOT NULL,
    net_pnl           DOUBLE PRECISION NOT NULL,
    r_multiple        DOUBLE PRECISION,
    commission        DOUBLE PRECISION DEFAULT 0,
    funding_paid      DOUBLE PRECISION DEFAULT 0,
    setup_grade       TEXT,
    exit_reason       TEXT,
    confirmation_type TEXT,
    entry_method      TEXT,
    confluences       JSONB DEFAULT '[]',
    mae_r             DOUBLE PRECISION,
    mfe_r             DOUBLE PRECISION,
    exit_efficiency   DOUBLE PRECISION,
    market_context    JSONB,
    created_at        TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_trades_strategy_exit ON trades(strategy_id, exit_time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time DESC);

-- 2. equity_snapshots: periodic equity readings (every 5 min = 288/day)
CREATE TABLE IF NOT EXISTS equity_snapshots (
    id        BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    equity    DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_snapshots(timestamp DESC);

-- 3. daily_snapshots: end-of-day aggregates (from DailySnapshot events)
CREATE TABLE IF NOT EXISTS daily_snapshots (
    trade_date           DATE PRIMARY KEY,
    total_trades         INTEGER DEFAULT 0,
    win_count            INTEGER DEFAULT 0,
    loss_count           INTEGER DEFAULT 0,
    gross_pnl            DOUBLE PRECISION DEFAULT 0,
    net_pnl              DOUBLE PRECISION DEFAULT 0,
    max_drawdown_pct     DOUBLE PRECISION DEFAULT 0,
    sharpe_rolling_30d   DOUBLE PRECISION,
    sortino_rolling_30d  DOUBLE PRECISION,
    per_strategy_summary JSONB DEFAULT '{}',
    created_at           TIMESTAMPTZ DEFAULT now()
);

-- 4. positions: current open positions (full-table upsert every 5 min)
CREATE TABLE IF NOT EXISTS positions (
    strategy_id    TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    direction      TEXT NOT NULL,
    qty            DOUBLE PRECISION NOT NULL,
    avg_entry      DOUBLE PRECISION NOT NULL,
    unrealized_pnl DOUBLE PRECISION DEFAULT 0,
    risk_r         DOUBLE PRECISION DEFAULT 0,
    stop_price     DOUBLE PRECISION,
    entry_time     TIMESTAMPTZ,
    last_update_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (strategy_id, symbol)
);

-- 5. health_snapshots: periodic health reports (every 60 min via EventEmitter)
CREATE TABLE IF NOT EXISTS health_snapshots (
    id         BIGSERIAL PRIMARY KEY,
    timestamp  TIMESTAMPTZ NOT NULL,
    assessment TEXT NOT NULL,
    uptime_sec DOUBLE PRECISION,
    alerts     JSONB DEFAULT '[]',
    report     JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_health_ts ON health_snapshots(timestamp DESC);
