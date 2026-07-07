-- 002_views.sql: Convenience views for dashboard queries.

-- Today's trades
CREATE OR REPLACE VIEW v_today_trades AS
SELECT *,
       EXTRACT(EPOCH FROM (exit_time - entry_time)) / 60 AS duration_minutes
FROM trades
WHERE exit_time >= (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')::date
ORDER BY exit_time DESC;

-- Per-strategy daily stats (derived from trades — no engine writes needed)
CREATE OR REPLACE VIEW v_strategy_today AS
SELECT strategy_id,
       COUNT(*)                                     AS trades_today,
       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)    AS wins_today,
       SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END)   AS losses_today,
       COALESCE(SUM(r_multiple), 0)                 AS daily_pnl_r,
       COALESCE(SUM(net_pnl), 0)                    AS daily_pnl_usd
FROM trades
WHERE exit_time >= (CURRENT_TIMESTAMP AT TIME ZONE 'UTC')::date
GROUP BY strategy_id;

-- Equity curve (hourly samples, last 90 days)
CREATE OR REPLACE VIEW v_equity_curve_90d AS
SELECT DISTINCT ON (date_trunc('hour', timestamp))
       date_trunc('hour', timestamp) AS ts,
       equity
FROM equity_snapshots
WHERE timestamp >= now() - INTERVAL '90 days'
ORDER BY date_trunc('hour', timestamp), timestamp DESC;

-- Daily P&L (last 30 days)
CREATE OR REPLACE VIEW v_daily_pnl_30d AS
SELECT trade_date, net_pnl, total_trades, win_count, loss_count
FROM daily_snapshots
WHERE trade_date >= CURRENT_DATE - 30
ORDER BY trade_date;

-- Portfolio summary (from current positions)
CREATE OR REPLACE VIEW v_portfolio_summary AS
SELECT COUNT(*)                          AS open_positions,
       COALESCE(SUM(unrealized_pnl), 0)  AS total_unrealized_pnl,
       COALESCE(SUM(risk_r), 0)          AS total_heat_r
FROM positions;
