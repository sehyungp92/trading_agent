"""PostgreSQL persistence layer with extended schema for trading infrastructure.

Unified across all trading families (swing, momentum, stock).
Includes the UNION of all DDL tables and PgStore methods.
"""
from __future__ import annotations

import asyncpg
import json
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Optional

from .schema import (
    RiskDailyStrategyRow,
    RiskDailyPortfolioRow,
    TradeRow,
    TradeMarksRow,
    StrategyStateRow,
    AdapterStateRow,
)

DDL = """\
-- ============================================================
-- EXISTING TABLES (preserved, minor enhancements)
-- ============================================================

CREATE TABLE IF NOT EXISTS orders (
    oms_order_id TEXT PRIMARY KEY,
    client_order_id TEXT,
    strategy_id TEXT NOT NULL,
    account_id TEXT NOT NULL DEFAULT 'default',
    instrument_symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty INT NOT NULL,
    order_type TEXT NOT NULL,
    limit_price NUMERIC,
    stop_price NUMERIC,
    tif TEXT DEFAULT 'DAY',
    role TEXT NOT NULL,
    status TEXT NOT NULL,
    broker TEXT DEFAULT 'IBKR',
    broker_order_id TEXT,
    perm_id TEXT,
    oca_group TEXT DEFAULT '',
    filled_qty INT NOT NULL DEFAULT 0,
    remaining_qty NUMERIC DEFAULT 0,
    avg_fill_price NUMERIC,
    reprice_count INT DEFAULT 0,
    entry_policy JSONB,
    risk_context JSONB,
    reject_reason TEXT DEFAULT '',
    retry_count INT DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    queued_at TIMESTAMPTZ,
    queue_priority INT,
    queue_reason TEXT DEFAULT '',
    queue_attempt INT NOT NULL DEFAULT 0,
    queue_expires_at TIMESTAMPTZ,
    queue_claimed_by TEXT,
    queue_claimed_at TIMESTAMPTZ,
    queue_claim_expires_at TIMESTAMPTZ,
    dequeued_at TIMESTAMPTZ,
    queue_denial_reason TEXT DEFAULT '',
    submitted_at TIMESTAMPTZ,
    acked_at TIMESTAMPTZ,
    last_update_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_orders_client
    ON orders(strategy_id, client_order_id)
    WHERE client_order_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_orders_status_update
    ON orders(status, last_update_at);
CREATE INDEX IF NOT EXISTS idx_orders_instrument_status
    ON orders(instrument_symbol, status);
CREATE INDEX IF NOT EXISTS idx_orders_strategy_created
    ON orders(strategy_id, created_at);
CREATE INDEX IF NOT EXISTS idx_orders_queued_ready
    ON orders(queue_priority, queued_at)
    WHERE status = 'QUEUED';
CREATE INDEX IF NOT EXISTS idx_orders_queue_expiry
    ON orders(queue_expires_at)
    WHERE status = 'QUEUED';
CREATE INDEX IF NOT EXISTS idx_orders_queue_claim_expiry
    ON orders(queue_claim_expires_at)
    WHERE status = 'QUEUED' AND queue_claimed_by IS NOT NULL;


CREATE TABLE IF NOT EXISTS order_events (
    event_id BIGSERIAL PRIMARY KEY,
    oms_order_id TEXT REFERENCES orders(oms_order_id),
    strategy_id TEXT,
    account_id TEXT DEFAULT 'default',
    event_type TEXT NOT NULL,
    from_state TEXT,
    to_state TEXT,
    reason TEXT DEFAULT '',
    payload JSONB NOT NULL DEFAULT '{}',
    event_ts TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_oe_ts ON order_events(event_ts);
CREATE INDEX IF NOT EXISTS idx_oe_oms_ts ON order_events(oms_order_id, event_ts);
CREATE INDEX IF NOT EXISTS idx_oe_type_ts ON order_events(event_type, event_ts);


CREATE TABLE IF NOT EXISTS fills (
    fill_id TEXT PRIMARY KEY,
    oms_order_id TEXT NOT NULL REFERENCES orders(oms_order_id),
    broker_fill_id TEXT NOT NULL UNIQUE,
    price NUMERIC NOT NULL,
    qty NUMERIC NOT NULL,
    fill_ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    fees NUMERIC DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_fills_order
    ON fills(oms_order_id);
CREATE INDEX IF NOT EXISTS idx_fills_ts
    ON fills(fill_ts);


CREATE TABLE IF NOT EXISTS positions (
    account_id TEXT NOT NULL DEFAULT 'default',
    instrument_symbol TEXT NOT NULL,
    strategy_id TEXT NOT NULL DEFAULT '',
    net_qty NUMERIC DEFAULT 0,
    avg_price NUMERIC DEFAULT 0,
    realized_pnl NUMERIC DEFAULT 0,
    unrealized_pnl NUMERIC DEFAULT 0,
    open_risk_dollars NUMERIC DEFAULT 0,
    open_risk_R NUMERIC DEFAULT 0,
    last_update_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (account_id, instrument_symbol, strategy_id)
);

CREATE INDEX IF NOT EXISTS idx_positions_instrument
    ON positions(instrument_symbol);
CREATE INDEX IF NOT EXISTS idx_positions_update
    ON positions(last_update_at);
CREATE INDEX IF NOT EXISTS idx_positions_strategy
    ON positions(strategy_id);


-- ============================================================
-- NEW TABLES: Daily Risk
-- ============================================================

CREATE TABLE IF NOT EXISTS risk_daily_strategy (
    trade_date DATE NOT NULL,
    strategy_id TEXT NOT NULL,
    family_id TEXT NOT NULL DEFAULT 'unknown',
    daily_realized_r NUMERIC NOT NULL DEFAULT 0,
    daily_realized_usd NUMERIC DEFAULT 0,
    open_risk_r NUMERIC DEFAULT 0,
    filled_entries INT NOT NULL DEFAULT 0,
    halted BOOLEAN NOT NULL DEFAULT FALSE,
    halt_reason TEXT,
    last_update_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (trade_date, strategy_id)
);
CREATE INDEX IF NOT EXISTS idx_risk_daily_strategy_family
    ON risk_daily_strategy(family_id);

CREATE TABLE IF NOT EXISTS risk_daily_portfolio (
    trade_date DATE NOT NULL,
    family_id TEXT NOT NULL DEFAULT 'unknown',
    daily_realized_r NUMERIC NOT NULL DEFAULT 0,
    daily_realized_usd NUMERIC DEFAULT 0,
    portfolio_open_risk_r NUMERIC DEFAULT 0,
    halted BOOLEAN NOT NULL DEFAULT FALSE,
    halt_reason TEXT,
    last_update_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (trade_date, family_id)
);


-- ============================================================
-- NEW TABLES: Trade Telemetry
-- ============================================================

CREATE TABLE IF NOT EXISTS trades (
    trade_id TEXT PRIMARY KEY,
    strategy_id TEXT NOT NULL,
    account_id TEXT NOT NULL DEFAULT 'default',
    instrument_symbol TEXT NOT NULL,
    direction TEXT NOT NULL,  -- LONG/SHORT
    quantity INT NOT NULL,
    entry_ts TIMESTAMPTZ NOT NULL,
    entry_price NUMERIC NOT NULL,
    exit_ts TIMESTAMPTZ,
    exit_price NUMERIC,
    realized_r NUMERIC,
    realized_usd NUMERIC,
    exit_reason TEXT,  -- STOP, TP1, TP2, TRAIL, STALE, EOD, EVENT, MANUAL, RECON_FLAT
    setup_tag TEXT,    -- strategy-specific (e.g., Predator, ABEC_Cont, AKC_Probe)
    entry_type TEXT,   -- PROBE, BOS, HTF, ADD, FULL (strategy-specific entry class)
    notes TEXT,
    meta JSONB DEFAULT '{}',  -- regime_score, zscore, div_magnitude, etc.
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_trades_strategy_entry
    ON trades(strategy_id, entry_ts);
CREATE INDEX IF NOT EXISTS idx_trades_instrument_entry
    ON trades(instrument_symbol, entry_ts);
CREATE INDEX IF NOT EXISTS idx_trades_exit
    ON trades(exit_ts);


CREATE TABLE IF NOT EXISTS trade_marks (
    trade_id TEXT PRIMARY KEY REFERENCES trades(trade_id),
    duration_seconds INT,
    duration_bars INT,
    mae_r NUMERIC,  -- Max Adverse Excursion in R
    mfe_r NUMERIC,  -- Max Favorable Excursion in R
    mae_usd NUMERIC,
    mfe_usd NUMERIC,
    max_adverse_price NUMERIC,
    max_favorable_price NUMERIC
);


-- ============================================================
-- Cross-Strategy Signals (momentum / stock families)
-- ============================================================

CREATE TABLE IF NOT EXISTS strategy_signals (
    strategy_id TEXT PRIMARY KEY,
    last_entry_ts TIMESTAMPTZ,
    last_direction TEXT,  -- 'LONG' or 'SHORT'
    daily_entry_count INT NOT NULL DEFAULT 0,
    signal_date DATE NOT NULL DEFAULT CURRENT_DATE,
    chop_score INT NOT NULL DEFAULT 0,  -- NQDTC chop score (0-4), used by Helix throttle
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_strategy_signals_date
    ON strategy_signals(signal_date);


-- ============================================================
-- NEW TABLES: Health Monitoring
-- ============================================================

CREATE TABLE IF NOT EXISTS strategy_state (
    strategy_id TEXT PRIMARY KEY,
    instance_id TEXT DEFAULT 'primary',
    last_heartbeat_ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    mode TEXT NOT NULL DEFAULT 'RUNNING',  -- RUNNING, STAND_DOWN, HALTED
    stand_down_reason TEXT,
    last_decision_code TEXT,
    last_decision_details JSONB,
    last_error_ts TIMESTAMPTZ,
    last_error TEXT,
    last_seen_bar_ts TIMESTAMPTZ,
    heat_r NUMERIC DEFAULT 0,
    daily_pnl_r NUMERIC DEFAULT 0
);


CREATE TABLE IF NOT EXISTS adapter_state (
    adapter_id TEXT PRIMARY KEY,  -- e.g., 'ibkr_vps1'
    broker TEXT NOT NULL DEFAULT 'IBKR',
    last_heartbeat_ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    connected BOOLEAN NOT NULL DEFAULT FALSE,
    last_disconnect_ts TIMESTAMPTZ,
    disconnect_count_24h INT DEFAULT 0,
    last_error_code TEXT,
    last_error_message TEXT
);


-- 5-min snapshots of strategy_state captured by apps/watchdog/snapshot.py.
-- Used by v_daily_strategy_activity to compute bars/denials deltas across
-- a session for the quiet-day classifier.
CREATE TABLE IF NOT EXISTS strategy_heartbeat_history (
    captured_at         TIMESTAMPTZ NOT NULL,
    strategy_id         TEXT        NOT NULL,
    mode                TEXT,
    last_decision_code  TEXT,
    bars_processed      BIGINT,
    last_seen_bar_ts    TIMESTAMPTZ,
    consecutive_denials INT,
    denials_today       INT,
    PRIMARY KEY (captured_at, strategy_id)
);
CREATE INDEX IF NOT EXISTS idx_heartbeat_history_captured
    ON strategy_heartbeat_history(captured_at);
CREATE INDEX IF NOT EXISTS idx_heartbeat_history_strategy
    ON strategy_heartbeat_history(strategy_id, captured_at);


-- ============================================================
-- NEW TABLES: Reconciliation
-- ============================================================

CREATE TABLE IF NOT EXISTS recon_watermarks (
    broker TEXT NOT NULL,
    account_id TEXT NOT NULL,
    last_exec_ts TIMESTAMPTZ,
    last_recon_ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    status TEXT NOT NULL DEFAULT 'OK',  -- OK, WARN, FAIL
    details JSONB,
    PRIMARY KEY (broker, account_id)
);

CREATE TABLE IF NOT EXISTS reconciliation_authority_leases (
    broker TEXT NOT NULL,
    account_id TEXT NOT NULL,
    client_id INT NOT NULL,
    family_id TEXT NOT NULL,
    recon_kind TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    acquired_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL,
    last_snapshot_id TEXT,
    PRIMARY KEY (broker, account_id, client_id, family_id, recon_kind)
);

CREATE INDEX IF NOT EXISTS idx_recon_authority_expiry
    ON reconciliation_authority_leases(expires_at);


-- ============================================================
-- Overlay Position Tracking (swing family)
-- ============================================================

CREATE TABLE IF NOT EXISTS overlay_positions (
    symbol TEXT NOT NULL,
    shares INT NOT NULL DEFAULT 0,
    notional NUMERIC NOT NULL DEFAULT 0,
    pct_of_nav NUMERIC NOT NULL DEFAULT 0,
    rebalance_ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol)
);


-- ============================================================
-- Paper Trading Equity Tracker (paper mode only)
-- ============================================================

CREATE TABLE IF NOT EXISTS paper_equity (
    account_scope TEXT PRIMARY KEY,
    equity DOUBLE PRECISION NOT NULL DEFAULT 10000.0,
    initial_equity DOUBLE PRECISION NOT NULL DEFAULT 10000.0,
    total_pnl DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    total_commission DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    trade_count INT NOT NULL DEFAULT 0,
    last_update_at TIMESTAMPTZ NOT NULL DEFAULT now()
);


-- ============================================================
-- Active Runtime Config
-- ============================================================

CREATE TABLE IF NOT EXISTS active_runtime_config (
    account_id TEXT NOT NULL DEFAULT '',
    config_scope TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    runtime_env TEXT NOT NULL,
    config_version TEXT NOT NULL,
    deployment_id TEXT,
    source_hash TEXT,
    payload JSONB NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ,
    PRIMARY KEY (account_id, config_scope, scope_id, runtime_env)
);

ALTER TABLE active_runtime_config
    ADD COLUMN IF NOT EXISTS account_id TEXT NOT NULL DEFAULT '';

UPDATE active_runtime_config
SET account_id = COALESCE(
    NULLIF(payload->>'account_id', ''),
    CASE WHEN config_scope = 'account' THEN scope_id ELSE '' END
)
WHERE account_id = '';

ALTER TABLE active_runtime_config
    DROP CONSTRAINT IF EXISTS active_runtime_config_pkey;

ALTER TABLE active_runtime_config
    ADD CONSTRAINT active_runtime_config_pkey
    PRIMARY KEY (account_id, config_scope, scope_id, runtime_env);

CREATE INDEX IF NOT EXISTS idx_active_runtime_config_applied
    ON active_runtime_config(account_id, runtime_env, applied_at DESC);

CREATE INDEX IF NOT EXISTS idx_active_runtime_config_scope
    ON active_runtime_config(account_id, config_scope, runtime_env);

CREATE OR REPLACE VIEW v_active_runtime_config AS
SELECT *
FROM active_runtime_config
WHERE expires_at IS NULL OR expires_at > now();


-- ============================================================
-- VIEWS: Dashboard Support
-- ============================================================

CREATE OR REPLACE VIEW v_portfolio_daily_summary AS
SELECT
    trade_date,
    SUM(daily_realized_r) AS daily_realized_r,
    SUM(daily_realized_usd) AS daily_realized_usd,
    SUM(portfolio_open_risk_r) AS portfolio_open_risk_r,
    BOOL_OR(halted) AS halted,
    string_agg(DISTINCT halt_reason, '; ')
        FILTER (WHERE halt_reason IS NOT NULL AND halt_reason != '') AS halt_reason,
    MAX(last_update_at) AS last_update_at
FROM risk_daily_portfolio
GROUP BY trade_date;


CREATE OR REPLACE VIEW v_live_positions AS
SELECT
    account_id,
    instrument_symbol,
    strategy_id,
    net_qty,
    avg_price,
    unrealized_pnl,
    realized_pnl,
    last_update_at,
    EXTRACT(EPOCH FROM (now() - last_update_at)) / 60 AS stale_minutes
FROM positions
WHERE net_qty != 0;


CREATE OR REPLACE VIEW v_working_orders AS
SELECT
    oms_order_id,
    strategy_id,
    instrument_symbol,
    role,
    side,
    qty,
    filled_qty,
    stop_price,
    limit_price,
    status,
    broker_order_id,
    created_at,
    queued_at,
    queue_priority,
    queue_reason,
    queue_attempt,
    queue_expires_at,
    dequeued_at,
    queue_denial_reason,
    EXTRACT(EPOCH FROM (now() - created_at)) / 60 AS age_minutes
FROM orders
WHERE status IN ('CREATED', 'RISK_APPROVED', 'QUEUED', 'ROUTED', 'ACKED', 'WORKING', 'PARTIALLY_FILLED');


CREATE OR REPLACE VIEW v_today_risk AS
SELECT
    s.strategy_id,
    s.daily_realized_r AS strategy_realized_r,
    s.open_risk_r AS strategy_open_risk_r,
    s.filled_entries,
    s.halted AS strategy_halted,
    s.halt_reason AS strategy_halt_reason,
    p.daily_realized_r AS portfolio_realized_r,
    p.halted AS portfolio_halted,
    p.halt_reason AS portfolio_halt_reason
FROM risk_daily_strategy s
LEFT JOIN risk_daily_portfolio p
    ON s.trade_date = p.trade_date AND s.family_id = p.family_id
WHERE s.trade_date = CURRENT_DATE;


CREATE OR REPLACE VIEW v_active_halts AS
SELECT
    'STRATEGY' AS halt_level,
    strategy_id AS entity,
    halt_reason,
    last_update_at
FROM risk_daily_strategy
WHERE halted = TRUE AND trade_date = CURRENT_DATE
UNION ALL
SELECT
    'PORTFOLIO' AS halt_level,
    'ALL' AS entity,
    halt_reason,
    last_update_at
FROM risk_daily_portfolio
WHERE halted = TRUE AND trade_date = CURRENT_DATE;


CREATE OR REPLACE VIEW v_strategy_health AS
SELECT
    strategy_id,
    mode,
    last_heartbeat_ts,
    EXTRACT(EPOCH FROM (now() - last_heartbeat_ts)) AS heartbeat_age_sec,
    last_decision_code,
    last_decision_details,
    last_seen_bar_ts,
    EXTRACT(EPOCH FROM (now() - last_seen_bar_ts)) AS bar_age_sec,
    CASE
        -- 180s matches apps/watchdog/checks.py:check_heartbeats default
        -- (stale_threshold_sec). Engines emit every 15s; this gives ~12 missed
        -- heartbeats before the dashboard flips to STALE, which is the same
        -- window the watchdog uses to alert.
        WHEN EXTRACT(EPOCH FROM (now() - last_heartbeat_ts)) > 180 THEN 'STALE'
        WHEN mode != 'RUNNING' THEN mode
        ELSE 'OK'
    END AS health_status,
    heat_r,
    daily_pnl_r,
    last_error,
    last_error_ts
FROM strategy_state;


CREATE OR REPLACE VIEW v_adapter_health AS
SELECT
    adapter_id,
    broker,
    connected,
    last_heartbeat_ts,
    EXTRACT(EPOCH FROM (now() - last_heartbeat_ts)) AS heartbeat_age_sec,
    CASE
        WHEN NOT connected THEN 'DISCONNECTED'
        WHEN EXTRACT(EPOCH FROM (now() - last_heartbeat_ts)) > 60 THEN 'STALE'
        ELSE 'OK'
    END AS health_status,
    disconnect_count_24h,
    last_error_code,
    last_error_message
FROM adapter_state;


CREATE OR REPLACE VIEW v_recent_fills AS
SELECT
    f.fill_id,
    o.strategy_id,
    o.instrument_symbol,
    o.side,
    f.qty,
    f.price,
    f.fill_ts,
    o.role,
    COALESCE(o.limit_price, o.stop_price) AS intended_price,
    CASE
        WHEN o.side = 'BUY' THEN f.price - COALESCE(o.limit_price, o.stop_price)
        ELSE COALESCE(o.limit_price, o.stop_price) - f.price
    END AS slippage
FROM fills f
JOIN orders o ON f.oms_order_id = o.oms_order_id
WHERE f.fill_ts >= now() - INTERVAL '24 hours';


CREATE OR REPLACE VIEW v_today_trades AS
SELECT
    t.trade_id,
    t.strategy_id,
    t.instrument_symbol,
    t.direction,
    t.quantity,
    t.entry_ts,
    t.entry_price,
    t.exit_ts,
    t.exit_price,
    t.realized_r,
    t.exit_reason,
    t.entry_type,
    m.mae_r,
    m.mfe_r,
    m.duration_seconds / 60 AS duration_minutes
FROM trades t
LEFT JOIN trade_marks m ON t.trade_id = m.trade_id
WHERE t.entry_ts >= CURRENT_DATE
ORDER BY t.entry_ts DESC;


-- Flatten the JSONB last_decision_details once so the dashboard does not
-- have to encode the schema. Keys here mirror the contract in
-- libs/services/decision_codes.py.
CREATE OR REPLACE VIEW v_strategy_diagnostics AS
SELECT
    sh.strategy_id,
    sh.mode,
    sh.health_status,
    sh.last_heartbeat_ts,
    sh.heartbeat_age_sec,
    sh.last_decision_code,
    sh.last_seen_bar_ts,
    sh.bar_age_sec,
    NULLIF(sh.last_decision_details->'liveness'->>'bars_processed', '')::bigint
        AS bars_processed,
    sh.last_decision_details->'liveness'->'symbol_freshness'
        AS symbol_freshness,
    NULLIF(sh.last_decision_details->'oms_health'->>'submitted', '')::int
        AS intents_submitted,
    NULLIF(sh.last_decision_details->'oms_health'->>'denied', '')::int
        AS intents_denied,
    NULLIF(sh.last_decision_details->'oms_health'->>'consecutive_denials', '')::int
        AS consecutive_denials,
    sh.last_decision_details->'ib_farm_status'
        AS ib_farm_status,
    sh.last_error,
    sh.last_error_ts
FROM v_strategy_health sh;


CREATE OR REPLACE VIEW v_daily_strategy_activity AS
WITH hb_agg AS (
    SELECT
        (captured_at AT TIME ZONE 'UTC')::date AS day,
        strategy_id,
        MAX(bars_processed) - MIN(bars_processed) AS bars,
        MAX(denials_today)                       AS denials,
        MAX(last_seen_bar_ts)                    AS last_bar_ts,
        (array_agg(last_decision_code ORDER BY captured_at DESC))[1] AS last_decision_code,
        (array_agg(mode ORDER BY captured_at DESC))[1]               AS last_mode
    FROM strategy_heartbeat_history
    GROUP BY (captured_at AT TIME ZONE 'UTC')::date, strategy_id
),
trade_agg AS (
    SELECT
        (entry_ts AT TIME ZONE 'UTC')::date AS day,
        strategy_id,
        count(*) AS trades
    FROM trades
    GROUP BY (entry_ts AT TIME ZONE 'UTC')::date, strategy_id
)
SELECT
    h.day,
    h.strategy_id,
    rds.family_id,
    h.bars,
    h.denials,
    h.last_bar_ts,
    h.last_decision_code,
    h.last_mode,
    COALESCE(t.trades, 0)            AS trades,
    rds.daily_realized_r,
    rds.daily_realized_usd
FROM hb_agg h
LEFT JOIN trade_agg t USING (day, strategy_id)
LEFT JOIN risk_daily_strategy rds
    ON rds.trade_date = h.day AND rds.strategy_id = h.strategy_id;


-- ============================================================
-- MIGRATION: Rename old risk_daily to risk_daily_strategy
-- ============================================================

DO $$
BEGIN
    -- If old table exists, migrate data
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'risk_daily') THEN
        INSERT INTO risk_daily_strategy (trade_date, strategy_id, daily_realized_r, open_risk_r, filled_entries, halted)
        SELECT trade_date, strategy_id, realized_r, max_open_risk_r, filled_entries, is_halted
        FROM risk_daily
        ON CONFLICT (trade_date, strategy_id) DO NOTHING;

        DROP TABLE IF EXISTS risk_daily;
    END IF;
END $$;


-- ============================================================
-- MIGRATION: orders table column alignment
-- ============================================================

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name='orders' AND column_name='rejection_reason') THEN
        ALTER TABLE orders RENAME COLUMN rejection_reason TO reject_reason;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='orders' AND column_name='retry_count') THEN
        ALTER TABLE orders ADD COLUMN retry_count INT DEFAULT 0;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='orders' AND column_name='queued_at') THEN
        ALTER TABLE orders ADD COLUMN queued_at TIMESTAMPTZ;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='orders' AND column_name='queue_priority') THEN
        ALTER TABLE orders ADD COLUMN queue_priority INT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='orders' AND column_name='queue_reason') THEN
        ALTER TABLE orders ADD COLUMN queue_reason TEXT DEFAULT '';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='orders' AND column_name='queue_attempt') THEN
        ALTER TABLE orders ADD COLUMN queue_attempt INT NOT NULL DEFAULT 0;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='orders' AND column_name='queue_expires_at') THEN
        ALTER TABLE orders ADD COLUMN queue_expires_at TIMESTAMPTZ;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='orders' AND column_name='queue_claimed_by') THEN
        ALTER TABLE orders ADD COLUMN queue_claimed_by TEXT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='orders' AND column_name='queue_claimed_at') THEN
        ALTER TABLE orders ADD COLUMN queue_claimed_at TIMESTAMPTZ;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='orders' AND column_name='queue_claim_expires_at') THEN
        ALTER TABLE orders ADD COLUMN queue_claim_expires_at TIMESTAMPTZ;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='orders' AND column_name='dequeued_at') THEN
        ALTER TABLE orders ADD COLUMN dequeued_at TIMESTAMPTZ;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='orders' AND column_name='queue_denial_reason') THEN
        ALTER TABLE orders ADD COLUMN queue_denial_reason TEXT DEFAULT '';
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_orders_queued_ready
    ON orders(queue_priority, queued_at)
    WHERE status = 'QUEUED';
CREATE INDEX IF NOT EXISTS idx_orders_queue_expiry
    ON orders(queue_expires_at)
    WHERE status = 'QUEUED';
CREATE INDEX IF NOT EXISTS idx_orders_queue_claim_expiry
    ON orders(queue_claim_expires_at)
    WHERE status = 'QUEUED' AND queue_claimed_by IS NOT NULL;
"""


class PgStore:
    """PostgreSQL store with extended tables for trading infrastructure.

    Unified across all trading families -- includes the union of methods
    from swing_trader, momentum_trader, and stock_trader PgStore classes.
    """

    REQUIRED_VIEWS = (
        "v_active_runtime_config",
        "v_portfolio_daily_summary",
        "v_strategy_diagnostics",
        "v_daily_strategy_activity",
    )

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def init_schema(self) -> None:
        """Initialize database schema."""
        async with self._pool.acquire() as conn:
            await conn.execute(DDL)
            await self._verify_required_views(conn)

    async def _verify_required_views(self, conn) -> None:
        """Fail fast when canonical dashboard/watchdog views are missing."""
        missing: list[str] = []
        for view_name in self.REQUIRED_VIEWS:
            exists = await conn.fetchval("SELECT to_regclass($1)", f"public.{view_name}")
            if exists is None:
                missing.append(view_name)
        if missing:
            raise RuntimeError(
                "Database schema bootstrap missing required view(s): "
                + ", ".join(missing)
            )

    # ------------------------------------------------------------------
    # Strategy Signals (momentum / stock -- cross-strategy coordination)
    # ------------------------------------------------------------------

    async def upsert_strategy_signal(
        self,
        strategy_id: str,
        direction: str,
        entry_ts: datetime,
    ) -> None:
        """Record a strategy's latest entry direction and time."""
        today = entry_ts.date()
        await self._pool.execute(
            """
            INSERT INTO strategy_signals
                (strategy_id, last_entry_ts, last_direction, daily_entry_count, signal_date, updated_at)
            VALUES ($1, $2, $3, 1, $4, now())
            ON CONFLICT (strategy_id) DO UPDATE SET
                last_entry_ts = EXCLUDED.last_entry_ts,
                last_direction = EXCLUDED.last_direction,
                daily_entry_count = CASE
                    WHEN strategy_signals.signal_date = EXCLUDED.signal_date
                    THEN strategy_signals.daily_entry_count + 1
                    ELSE 1
                END,
                signal_date = EXCLUDED.signal_date,
                updated_at = now()
            """,
            strategy_id,
            entry_ts,
            direction,
            today,
        )

    async def update_chop_score(
        self, strategy_id: str, chop_score: int,
    ) -> None:
        """Update NQDTC chop score for cross-strategy throttling."""
        await self._pool.execute(
            """
            UPDATE strategy_signals
            SET chop_score = $2, updated_at = now()
            WHERE strategy_id = $1
            """,
            strategy_id, chop_score,
        )

    async def get_strategy_signal(
        self, strategy_id: str
    ) -> Optional[dict]:
        """Get a strategy's latest signal (entry time, direction, chop_score)."""
        r = await self._pool.fetchrow(
            "SELECT strategy_id, last_entry_ts, last_direction, daily_entry_count, "
            "signal_date, chop_score "
            "FROM strategy_signals WHERE strategy_id = $1",
            strategy_id,
        )
        if r is None:
            return None
        return {
            "strategy_id": r["strategy_id"],
            "last_entry_ts": r["last_entry_ts"],
            "last_direction": r["last_direction"],
            "daily_entry_count": r["daily_entry_count"],
            "signal_date": r["signal_date"],
            "chop_score": r.get("chop_score", 0),
        }

    async def get_all_strategy_signals(self) -> list[dict]:
        """Get all strategy signals for cross-strategy checks."""
        rows = await self._pool.fetch(
            "SELECT strategy_id, last_entry_ts, last_direction, daily_entry_count, "
            "signal_date, chop_score "
            "FROM strategy_signals"
        )
        return [
            {
                "strategy_id": r["strategy_id"],
                "last_entry_ts": r["last_entry_ts"],
                "last_direction": r["last_direction"],
                "daily_entry_count": r["daily_entry_count"],
                "signal_date": r["signal_date"],
                "chop_score": r.get("chop_score", 0),
            }
            for r in rows
        ]

    async def get_directional_risk_R(self, direction: str) -> float:
        """Sum open risk R for all positions in a given direction."""
        r = await self._pool.fetchrow(
            """
            SELECT COALESCE(SUM(open_risk_R), 0) AS total_risk
            FROM positions
            WHERE net_qty != 0
            AND CASE WHEN $1 = 'LONG' THEN net_qty > 0 ELSE net_qty < 0 END
            """,
            direction,
        )
        return float(r["total_risk"]) if r else 0.0

    async def get_directional_risk_R_for_strategies(
        self, direction: str, strategy_ids: list[str],
    ) -> float:
        """Sum open risk R for positions in a given direction, filtered by strategy IDs."""
        r = await self._pool.fetchrow(
            """
            SELECT COALESCE(SUM(open_risk_R), 0) AS total_risk
            FROM positions
            WHERE strategy_id = ANY($2)
            AND net_qty != 0
            AND CASE WHEN $1 = 'LONG' THEN net_qty > 0 ELSE net_qty < 0 END
            """,
            direction,
            strategy_ids,
        )
        return float(r["total_risk"]) if r else 0.0

    async def get_directional_risk_dollars_for_strategies(
        self, direction: str, strategy_ids: list[str],
    ) -> float:
        """Sum active risk dollars in a direction, including pending entries."""
        if not strategy_ids:
            return 0.0
        working_statuses = [
            "RISK_APPROVED",
            "ROUTED",
            "ACKED",
            "WORKING",
            "PARTIALLY_FILLED",
        ]
        r = await self._pool.fetchrow(
            """
            SELECT
                (
                    SELECT COALESCE(SUM(open_risk_dollars), 0)
                    FROM positions
                    WHERE strategy_id = ANY($2)
                    AND net_qty != 0
                    AND CASE WHEN $1 = 'LONG' THEN net_qty > 0 ELSE net_qty < 0 END
                ) AS open_risk,
                (
                    SELECT COALESCE(SUM(
                        COALESCE((risk_context->>'risk_dollars')::double precision, 0.0)
                        * CASE
                            WHEN qty > 0 AND COALESCE(remaining_qty, 0) > 0
                            THEN remaining_qty::double precision / qty::double precision
                            ELSE 1.0
                        END
                    ), 0.0)
                    FROM orders
                    WHERE strategy_id = ANY($2)
                    AND role = 'ENTRY'
                    AND status = ANY($3)
                    AND CASE WHEN $1 = 'LONG' THEN side = 'BUY' ELSE side = 'SELL' END
                ) AS pending_risk
            """,
            direction,
            strategy_ids,
            working_statuses,
        )
        if r is None:
            return 0.0
        return float(r["open_risk"] or 0.0) + float(r["pending_risk"] or 0.0)

    async def get_sibling_positions_for_symbol(
        self, strategy_ids: list[str], symbol: str,
    ) -> bool:
        """Check if any sibling strategy holds an open position in the given symbol."""
        r = await self._pool.fetchval(
            """
            SELECT EXISTS(
                SELECT 1 FROM positions
                WHERE strategy_id = ANY($1)
                AND instrument_symbol = $2
                AND net_qty != 0
            )
            """,
            strategy_ids,
            symbol,
        )
        return bool(r)

    async def get_open_position_count_for_strategies(
        self, strategy_ids: list[str],
    ) -> int:
        """Count family open positions plus pending entry orders."""
        if not strategy_ids:
            return 0
        working_statuses = [
            "RISK_APPROVED",
            "ROUTED",
            "ACKED",
            "WORKING",
            "PARTIALLY_FILLED",
        ]
        row = await self._pool.fetchrow(
            """
            SELECT
                (
                    SELECT COUNT(*)
                    FROM positions
                    WHERE strategy_id = ANY($1)
                    AND net_qty != 0
                ) AS open_positions,
                (
                    SELECT COUNT(*)
                    FROM orders
                    WHERE strategy_id = ANY($1)
                    AND role = 'ENTRY'
                    AND status = ANY($2)
                ) AS pending_entries
            """,
            strategy_ids,
            working_statuses,
        )
        if row is None:
            return 0
        return int(row["open_positions"] or 0) + int(row["pending_entries"] or 0)

    async def get_symbol_open_risk_dollars_for_strategies(
        self, strategy_ids: list[str], symbol: str,
    ) -> float:
        """Sum open risk dollars for one symbol within a strategy family."""
        if not strategy_ids or not symbol:
            return 0.0
        r = await self._pool.fetchval(
            """
            SELECT COALESCE(SUM(open_risk_dollars), 0)
            FROM positions
            WHERE strategy_id = ANY($1)
            AND instrument_symbol = $2
            AND net_qty != 0
            """,
            strategy_ids,
            symbol,
        )
        return float(r or 0.0)

    async def get_symbols_open_risk_dollars_for_strategies(
        self, strategy_ids: list[str], symbols: list[str],
    ) -> float:
        """Sum open risk dollars for a set of symbols within a strategy family."""
        if not strategy_ids or not symbols:
            return 0.0
        r = await self._pool.fetchval(
            """
            SELECT COALESCE(SUM(open_risk_dollars), 0)
            FROM positions
            WHERE strategy_id = ANY($1)
            AND instrument_symbol = ANY($2)
            AND net_qty != 0
            """,
            strategy_ids,
            symbols,
        )
        return float(r or 0.0)

    async def get_active_risk_dollars_for_strategies(
        self, strategy_ids: list[str],
    ) -> float:
        """Sum open position risk plus pending entry risk for a family."""
        if not strategy_ids:
            return 0.0
        working_statuses = [
            "RISK_APPROVED",
            "ROUTED",
            "ACKED",
            "WORKING",
            "PARTIALLY_FILLED",
        ]
        row = await self._pool.fetchrow(
            """
            SELECT
                (
                    SELECT COALESCE(SUM(open_risk_dollars), 0)
                    FROM positions
                    WHERE strategy_id = ANY($1)
                    AND net_qty != 0
                ) AS open_risk,
                (
                    SELECT COALESCE(SUM(
                        COALESCE((risk_context->>'risk_dollars')::double precision, 0.0)
                        * CASE
                            WHEN qty > 0 AND COALESCE(remaining_qty, 0) > 0
                            THEN remaining_qty::double precision / qty::double precision
                            ELSE 1.0
                        END
                    ), 0.0)
                    FROM orders
                    WHERE strategy_id = ANY($1)
                    AND role = 'ENTRY'
                    AND status = ANY($2)
                ) AS pending_risk
            """,
            strategy_ids,
            working_statuses,
        )
        if row is None:
            return 0.0
        return float(row["open_risk"] or 0.0) + float(row["pending_risk"] or 0.0)

    async def get_completed_trade_counts_for_strategies(
        self, strategy_ids: list[str],
    ) -> dict[str, int]:
        """Count completed trades per strategy for family balance rules."""
        if not strategy_ids:
            return {}
        rows = await self._pool.fetch(
            """
            SELECT strategy_id, COUNT(*) AS trade_count
            FROM trades
            WHERE strategy_id = ANY($1)
            AND exit_ts IS NOT NULL
            GROUP BY strategy_id
            """,
            strategy_ids,
        )
        return {row["strategy_id"]: int(row["trade_count"] or 0) for row in rows}

    async def get_recent_strategy_r_multiples(
        self, strategy_id: str, limit: int,
    ) -> list[float]:
        """Most recent completed trade R values for live dynamic allocation."""
        if not strategy_id or limit <= 0:
            return []
        rows = await self._pool.fetch(
            """
            SELECT realized_r
            FROM trades
            WHERE strategy_id = $1
            AND realized_r IS NOT NULL
            AND exit_ts IS NOT NULL
            ORDER BY exit_ts DESC
            LIMIT $2
            """,
            strategy_id,
            limit,
        )
        return [float(row["realized_r"]) for row in rows]

    async def get_family_aggregate_mnq_eq(self, strategy_ids: list[str]) -> int:
        """Sum open and pending contracts, converting NQ to 10x MNQ-eq."""
        if not strategy_ids:
            return 0
        working_statuses = [
            "RISK_APPROVED",
            "ROUTED",
            "ACKED",
            "WORKING",
            "PARTIALLY_FILLED",
        ]
        rows = await self._pool.fetch(
            "SELECT instrument_symbol, COALESCE(SUM(ABS(net_qty)), 0)::int AS total_qty "
            "FROM positions WHERE strategy_id = ANY($1) AND net_qty != 0 "
            "GROUP BY instrument_symbol",
            strategy_ids,
        )
        total = 0
        for row in rows:
            qty = int(row["total_qty"])
            total += qty * 10 if row["instrument_symbol"] == "NQ" else qty
        pending_rows = await self._pool.fetch(
            """
            SELECT instrument_symbol, COALESCE(SUM(
                CASE
                    WHEN COALESCE(remaining_qty, 0) > 0 THEN remaining_qty
                    ELSE qty
                END
            ), 0)::int AS total_qty
            FROM orders
            WHERE strategy_id = ANY($1)
            AND role = 'ENTRY'
            AND status = ANY($2)
            GROUP BY instrument_symbol
            """,
            strategy_ids,
            working_statuses,
        )
        for row in pending_rows:
            qty = int(row["total_qty"])
            total += qty * 10 if row["instrument_symbol"] == "NQ" else qty
        return total

    # ------------------------------------------------------------------
    # Risk Daily Strategy
    # ------------------------------------------------------------------

    async def upsert_risk_daily_strategy(self, row: RiskDailyStrategyRow) -> None:
        await self._pool.execute(
            """
            INSERT INTO risk_daily_strategy
                (trade_date, strategy_id, family_id, daily_realized_r, daily_realized_usd,
                 open_risk_r, filled_entries, halted, halt_reason, last_update_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (trade_date, strategy_id) DO UPDATE SET
                family_id = EXCLUDED.family_id,
                daily_realized_r = EXCLUDED.daily_realized_r,
                daily_realized_usd = EXCLUDED.daily_realized_usd,
                open_risk_r = EXCLUDED.open_risk_r,
                filled_entries = EXCLUDED.filled_entries,
                halted = EXCLUDED.halted,
                halt_reason = EXCLUDED.halt_reason,
                last_update_at = EXCLUDED.last_update_at
            """,
            row.trade_date,
            row.strategy_id,
            row.family_id,
            row.daily_realized_r,
            row.daily_realized_usd,
            row.open_risk_r,
            row.filled_entries,
            row.halted,
            row.halt_reason,
            row.last_update_at,
        )

    async def get_risk_daily_strategy(
        self,
        strategy_id: str,
        trade_date: date,
    ) -> Optional[RiskDailyStrategyRow]:
        r = await self._pool.fetchrow(
            "SELECT * FROM risk_daily_strategy WHERE strategy_id=$1 AND trade_date=$2",
            strategy_id,
            trade_date,
        )
        if r is None:
            return None
        return RiskDailyStrategyRow(
            trade_date=r["trade_date"],
            strategy_id=r["strategy_id"],
            family_id=r.get("family_id", "unknown"),
            daily_realized_r=r["daily_realized_r"],
            daily_realized_usd=r["daily_realized_usd"],
            open_risk_r=r["open_risk_r"],
            filled_entries=r["filled_entries"],
            halted=r["halted"],
            halt_reason=r["halt_reason"],
            last_update_at=r["last_update_at"],
        )

    async def get_risk_daily_strategies_for_date(
        self,
        trade_date: date,
        strategy_ids: list[str] | None = None,
    ) -> list[RiskDailyStrategyRow]:
        """Get strategy risk rows for a given date, optionally filtered by strategy IDs."""
        if strategy_ids:
            placeholders = ", ".join(f"${i+2}" for i in range(len(strategy_ids)))
            rows = await self._pool.fetch(
                f"SELECT * FROM risk_daily_strategy WHERE trade_date = $1 AND strategy_id IN ({placeholders}) ORDER BY strategy_id",
                trade_date, *strategy_ids,
            )
        else:
            rows = await self._pool.fetch(
                """
                SELECT * FROM risk_daily_strategy
                WHERE trade_date = $1
                ORDER BY strategy_id
                """,
                trade_date,
            )
        return [
            RiskDailyStrategyRow(
                trade_date=r["trade_date"],
                strategy_id=r["strategy_id"],
                family_id=r.get("family_id", "unknown"),
                daily_realized_r=r["daily_realized_r"],
                daily_realized_usd=r["daily_realized_usd"],
                open_risk_r=r["open_risk_r"],
                filled_entries=r["filled_entries"],
                halted=r["halted"],
                halt_reason=r["halt_reason"],
                last_update_at=r["last_update_at"],
            )
            for r in rows
        ]

    async def get_risk_daily_strategy_totals(
        self,
        start_date: date,
        end_date: date,
        strategy_ids: list[str] | None = None,
    ) -> dict[str, Decimal]:
        """Aggregate realized R and USD over a date range, optionally filtered by strategy IDs."""
        if strategy_ids:
            placeholders = ", ".join(f"${i+3}" for i in range(len(strategy_ids)))
            row = await self._pool.fetchrow(
                f"""
                SELECT
                    COALESCE(SUM(daily_realized_r), 0) AS total_r,
                    COALESCE(SUM(daily_realized_usd), 0) AS total_usd
                FROM risk_daily_strategy
                WHERE trade_date BETWEEN $1 AND $2
                    AND strategy_id IN ({placeholders})
                """,
                start_date, end_date, *strategy_ids,
            )
        else:
            row = await self._pool.fetchrow(
                """
                SELECT
                    COALESCE(SUM(daily_realized_r), 0) AS total_r,
                    COALESCE(SUM(daily_realized_usd), 0) AS total_usd
                FROM risk_daily_strategy
                WHERE trade_date BETWEEN $1 AND $2
                """,
                start_date,
                end_date,
            )
        return {
            "total_r": row["total_r"] if row else Decimal("0"),
            "total_usd": row["total_usd"] if row else Decimal("0"),
        }

    async def halt_strategy(self, strategy_id: str, reason: str, trade_date: date) -> None:
        await self._pool.execute(
            """
            UPDATE risk_daily_strategy
            SET halted = TRUE, halt_reason = $3, last_update_at = now()
            WHERE strategy_id = $1 AND trade_date = $2
            """,
            strategy_id,
            trade_date,
            reason,
        )

    # ------------------------------------------------------------------
    # Risk Daily Portfolio
    # ------------------------------------------------------------------

    async def upsert_risk_daily_portfolio(self, row: RiskDailyPortfolioRow) -> None:
        await self._pool.execute(
            """
            INSERT INTO risk_daily_portfolio
                (trade_date, family_id, daily_realized_r, daily_realized_usd,
                 portfolio_open_risk_r, halted, halt_reason, last_update_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (trade_date, family_id) DO UPDATE SET
                daily_realized_r = EXCLUDED.daily_realized_r,
                daily_realized_usd = EXCLUDED.daily_realized_usd,
                portfolio_open_risk_r = EXCLUDED.portfolio_open_risk_r,
                halted = EXCLUDED.halted,
                halt_reason = EXCLUDED.halt_reason,
                last_update_at = EXCLUDED.last_update_at
            """,
            row.trade_date,
            row.family_id,
            row.daily_realized_r,
            row.daily_realized_usd,
            row.portfolio_open_risk_r,
            row.halted,
            row.halt_reason,
            row.last_update_at,
        )

    async def get_risk_daily_portfolio(
        self,
        trade_date: date,
        family_id: str = "unknown",
    ) -> Optional[RiskDailyPortfolioRow]:
        """Get portfolio risk for a given date and family."""
        r = await self._pool.fetchrow(
            "SELECT * FROM risk_daily_portfolio WHERE trade_date = $1 AND family_id = $2",
            trade_date,
            family_id,
        )
        if r is None:
            return None
        return RiskDailyPortfolioRow(
            trade_date=r["trade_date"],
            family_id=r["family_id"],
            daily_realized_r=r["daily_realized_r"],
            daily_realized_usd=r["daily_realized_usd"],
            portfolio_open_risk_r=r["portfolio_open_risk_r"],
            halted=r["halted"],
            halt_reason=r["halt_reason"],
            last_update_at=r["last_update_at"],
        )

    async def halt_portfolio(self, reason: str, trade_date: date, family_id: str = "unknown") -> None:
        await self._pool.execute(
            """
            UPDATE risk_daily_portfolio
            SET halted = TRUE, halt_reason = $2, last_update_at = now()
            WHERE trade_date = $1 AND family_id = $3
            """,
            trade_date,
            reason,
            family_id,
        )

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------

    async def save_trade(self, row: TradeRow) -> None:
        await self._pool.execute(
            """
            INSERT INTO trades
                (trade_id, strategy_id, account_id, instrument_symbol, direction,
                 quantity, entry_ts, entry_price, exit_ts, exit_price,
                 realized_r, realized_usd, exit_reason, setup_tag, entry_type, notes, meta)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17::jsonb)
            ON CONFLICT (trade_id) DO UPDATE SET
                exit_ts = EXCLUDED.exit_ts,
                exit_price = EXCLUDED.exit_price,
                realized_r = EXCLUDED.realized_r,
                realized_usd = EXCLUDED.realized_usd,
                exit_reason = EXCLUDED.exit_reason,
                notes = EXCLUDED.notes,
                meta = EXCLUDED.meta
            """,
            row.trade_id,
            row.strategy_id,
            row.account_id,
            row.instrument_symbol,
            row.direction,
            row.quantity,
            row.entry_ts,
            row.entry_price,
            row.exit_ts,
            row.exit_price,
            row.realized_r,
            row.realized_usd,
            row.exit_reason,
            row.setup_tag,
            row.entry_type,
            row.notes,
            row.meta_json,
        )

    async def get_trades_since(self, since: datetime) -> list[TradeRow]:
        rows = await self._pool.fetch(
            "SELECT * FROM trades WHERE entry_ts >= $1 ORDER BY entry_ts",
            since,
        )
        return [self._to_trade_row(r) for r in rows]

    async def get_open_trades(self) -> list[TradeRow]:
        rows = await self._pool.fetch(
            "SELECT * FROM trades WHERE exit_ts IS NULL ORDER BY entry_ts",
        )
        return [self._to_trade_row(r) for r in rows]

    def _to_trade_row(self, r: asyncpg.Record) -> TradeRow:
        return TradeRow(
            trade_id=r["trade_id"],
            strategy_id=r["strategy_id"],
            account_id=r["account_id"],
            instrument_symbol=r["instrument_symbol"],
            direction=r["direction"],
            quantity=r["quantity"],
            entry_ts=r["entry_ts"],
            entry_price=r["entry_price"],
            exit_ts=r["exit_ts"],
            exit_price=r["exit_price"],
            realized_r=r["realized_r"],
            realized_usd=r["realized_usd"],
            exit_reason=r["exit_reason"],
            setup_tag=r["setup_tag"],
            entry_type=r["entry_type"],
            notes=r["notes"],
            meta_json=json.dumps(r["meta"]) if r["meta"] else "{}",
        )

    # ------------------------------------------------------------------
    # Trade Marks
    # ------------------------------------------------------------------

    async def save_trade_marks(self, row: TradeMarksRow) -> None:
        await self._pool.execute(
            """
            INSERT INTO trade_marks
                (trade_id, duration_seconds, duration_bars, mae_r, mfe_r,
                 mae_usd, mfe_usd, max_adverse_price, max_favorable_price)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (trade_id) DO UPDATE SET
                duration_seconds = EXCLUDED.duration_seconds,
                duration_bars = EXCLUDED.duration_bars,
                mae_r = EXCLUDED.mae_r,
                mfe_r = EXCLUDED.mfe_r,
                mae_usd = EXCLUDED.mae_usd,
                mfe_usd = EXCLUDED.mfe_usd,
                max_adverse_price = EXCLUDED.max_adverse_price,
                max_favorable_price = EXCLUDED.max_favorable_price
            """,
            row.trade_id,
            row.duration_seconds,
            row.duration_bars,
            row.mae_r,
            row.mfe_r,
            row.mae_usd,
            row.mfe_usd,
            row.max_adverse_price,
            row.max_favorable_price,
        )

    # ------------------------------------------------------------------
    # Strategy State
    # ------------------------------------------------------------------

    async def upsert_strategy_state(self, row: StrategyStateRow) -> None:
        await self._pool.execute(
            """
            INSERT INTO strategy_state
                (strategy_id, instance_id, last_heartbeat_ts, mode, stand_down_reason,
                 last_decision_code, last_decision_details, last_error_ts, last_error,
                 last_seen_bar_ts, heat_r, daily_pnl_r)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11, $12)
            ON CONFLICT (strategy_id) DO UPDATE SET
                last_heartbeat_ts = EXCLUDED.last_heartbeat_ts,
                mode = EXCLUDED.mode,
                stand_down_reason = EXCLUDED.stand_down_reason,
                last_decision_code = COALESCE(
                    NULLIF(EXCLUDED.last_decision_code, ''),
                    strategy_state.last_decision_code
                ),
                last_decision_details = CASE
                    WHEN NULLIF(EXCLUDED.last_decision_code, '') IS NOT NULL
                         OR EXCLUDED.last_decision_details IS DISTINCT FROM '{}'::jsonb
                    THEN EXCLUDED.last_decision_details
                    ELSE strategy_state.last_decision_details
                END,
                last_error_ts = EXCLUDED.last_error_ts,
                last_error = EXCLUDED.last_error,
                last_seen_bar_ts = EXCLUDED.last_seen_bar_ts,
                heat_r = EXCLUDED.heat_r,
                daily_pnl_r = EXCLUDED.daily_pnl_r
            """,
            row.strategy_id,
            row.instance_id,
            row.last_heartbeat_ts,
            row.mode,
            row.stand_down_reason,
            row.last_decision_code,
            row.last_decision_details_json,
            row.last_error_ts,
            row.last_error,
            row.last_seen_bar_ts,
            row.heat_r,
            row.daily_pnl_r,
        )

    async def record_strategy_decision(
        self,
        strategy_id: str,
        decision_code: str,
        details: Optional[dict] = None,
        last_seen_bar_ts: Optional[datetime] = None,
    ) -> None:
        if not decision_code:
            return
        await self._pool.execute(
            """
            INSERT INTO strategy_state
                (strategy_id, last_decision_code, last_decision_details, last_seen_bar_ts)
            VALUES ($1, $2, $3::jsonb, $4)
            ON CONFLICT (strategy_id) DO UPDATE SET
                last_decision_code = EXCLUDED.last_decision_code,
                last_decision_details = EXCLUDED.last_decision_details,
                last_seen_bar_ts = COALESCE(
                    EXCLUDED.last_seen_bar_ts,
                    strategy_state.last_seen_bar_ts
                )
            """,
            strategy_id,
            decision_code,
            json.dumps(details or {}, default=str),
            last_seen_bar_ts,
        )

    async def get_strategy_states(self) -> list[StrategyStateRow]:
        rows = await self._pool.fetch("SELECT * FROM strategy_state")
        return [
            StrategyStateRow(
                strategy_id=r["strategy_id"],
                instance_id=r["instance_id"],
                last_heartbeat_ts=r["last_heartbeat_ts"],
                mode=r["mode"],
                stand_down_reason=r["stand_down_reason"],
                last_decision_code=r["last_decision_code"],
                last_decision_details_json=json.dumps(r["last_decision_details"])
                if r["last_decision_details"]
                else "{}",
                last_error_ts=r["last_error_ts"],
                last_error=r["last_error"],
                last_seen_bar_ts=r["last_seen_bar_ts"],
                heat_r=r["heat_r"],
                daily_pnl_r=r["daily_pnl_r"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Adapter State
    # ------------------------------------------------------------------

    async def upsert_adapter_state(self, row: AdapterStateRow) -> None:
        await self._pool.execute(
            """
            INSERT INTO adapter_state
                (adapter_id, broker, last_heartbeat_ts, connected, last_disconnect_ts,
                 disconnect_count_24h, last_error_code, last_error_message)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (adapter_id) DO UPDATE SET
                last_heartbeat_ts = EXCLUDED.last_heartbeat_ts,
                connected = EXCLUDED.connected,
                last_disconnect_ts = EXCLUDED.last_disconnect_ts,
                disconnect_count_24h = EXCLUDED.disconnect_count_24h,
                last_error_code = EXCLUDED.last_error_code,
                last_error_message = EXCLUDED.last_error_message
            """,
            row.adapter_id,
            row.broker,
            row.last_heartbeat_ts,
            row.connected,
            row.last_disconnect_ts,
            row.disconnect_count_24h,
            row.last_error_code,
            row.last_error_message,
        )

    async def record_adapter_disconnect(
        self, adapter_id: str, error_code: str = None, error_msg: str = None
    ) -> None:
        await self._pool.execute(
            """
            UPDATE adapter_state SET
                connected = FALSE,
                last_disconnect_ts = now(),
                disconnect_count_24h = disconnect_count_24h + 1,
                last_error_code = COALESCE($2, last_error_code),
                last_error_message = COALESCE($3, last_error_message)
            WHERE adapter_id = $1
            """,
            adapter_id,
            error_code,
            error_msg,
        )

    async def record_adapter_connect(self, adapter_id: str) -> None:
        await self._pool.execute(
            """
            UPDATE adapter_state SET
                connected = TRUE,
                last_heartbeat_ts = now()
            WHERE adapter_id = $1
            """,
            adapter_id,
        )

    # ------------------------------------------------------------------
    # Overlay Positions (swing family)
    # ------------------------------------------------------------------

    async def upsert_overlay_positions(
        self,
        positions: list[dict],
    ) -> None:
        """Upsert overlay position rows after each rebalance.

        Each dict: {symbol, shares, notional, pct_of_nav, rebalance_ts}.
        Rows with shares=0 are deleted (position closed).
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Remove closed positions
                closed = [p["symbol"] for p in positions if p["shares"] == 0]
                if closed:
                    await conn.execute(
                        "DELETE FROM overlay_positions WHERE symbol = ANY($1::text[])",
                        closed,
                    )
                # Batch upsert active positions via UNNEST
                active = [p for p in positions if p["shares"] != 0]
                if active:
                    await conn.execute(
                        """
                        INSERT INTO overlay_positions
                            (symbol, shares, notional, pct_of_nav, rebalance_ts)
                        SELECT * FROM UNNEST($1::text[], $2::int[], $3::numeric[],
                                             $4::numeric[], $5::timestamptz[])
                        ON CONFLICT (symbol) DO UPDATE SET
                            shares = EXCLUDED.shares,
                            notional = EXCLUDED.notional,
                            pct_of_nav = EXCLUDED.pct_of_nav,
                            rebalance_ts = EXCLUDED.rebalance_ts
                        """,
                        [p["symbol"] for p in active],
                        [p["shares"] for p in active],
                        [p["notional"] for p in active],
                        [p["pct_of_nav"] for p in active],
                        [p["rebalance_ts"] for p in active],
                    )
