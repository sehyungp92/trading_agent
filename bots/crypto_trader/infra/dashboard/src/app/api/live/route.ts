import { NextResponse } from "next/server";
import pool from "@/lib/db";
import type {
  AllocationResidualRow,
  ExchangePositionRow,
  LiveBatchResponse,
  SafetyEventRow,
  StrategyPositionAllocationRow,
} from "@/lib/types";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const [
      equityRes,
      strategyRes,
      positionsRes,
      tradesRes,
      healthRes,
      summaryRes,
      safetyRes,
      exchangePositions,
      strategyAllocations,
    ] =
      await Promise.all([
        pool.query(
          "SELECT equity FROM equity_snapshots ORDER BY timestamp DESC LIMIT 1"
        ),
        pool.query("SELECT * FROM v_strategy_today"),
        pool.query(
          `SELECT *,
                  EXTRACT(EPOCH FROM (now() - last_update_at)) / 60 AS stale_minutes,
                  COALESCE(EXTRACT(EPOCH FROM (now() - entry_time)) / 60, 0) AS age_minutes
           FROM positions`
        ),
        pool.query("SELECT * FROM v_today_trades LIMIT 50"),
        pool.query(
          "SELECT timestamp, assessment, uptime_sec, alerts, report FROM health_snapshots ORDER BY timestamp DESC LIMIT 1"
        ),
        pool.query("SELECT * FROM v_portfolio_summary"),
        querySafetyEvents(),
        queryExchangePositions(),
        queryStrategyPositionAllocations(),
      ]);

    const equity = equityRes.rows[0]?.equity ?? 0;
    const summary = summaryRes.rows[0] ?? {
      open_positions: 0,
      total_unrealized_pnl: 0,
      total_heat_r: 0,
    };

    // Sum daily PnL across strategies
    const dailyPnlUsd = strategyRes.rows.reduce(
      (sum: number, s: { daily_pnl_usd: number }) => sum + (s.daily_pnl_usd ?? 0),
      0
    );

    const healthRow = healthRes.rows[0] ?? null;
    const health = healthRow ? {
      timestamp: healthRow.timestamp,
      assessment: healthRow.assessment,
      uptime_sec: healthRow.uptime_sec,
      alerts: healthRow.alerts ?? [],
      postgres_sink: healthRow.report?.postgres_sink ?? null,
    } : null;
    const allocationResiduals = buildAllocationResiduals(exchangePositions, strategyAllocations);

    const response: LiveBatchResponse = {
      portfolio: {
        equity,
        daily_pnl_usd: dailyPnlUsd,
        unrealized_pnl: summary.total_unrealized_pnl,
        heat_r: summary.total_heat_r,
        open_positions: summary.open_positions,
      },
      strategies: strategyRes.rows,
      positions: positionsRes.rows,
      exchange_positions: exchangePositions,
      strategy_position_allocations: strategyAllocations,
      allocation_residuals: allocationResiduals,
      trades: tradesRes.rows,
      health,
      safety_events: safetyRes,
    };

    return NextResponse.json(response);
  } catch (err) {
    console.error("API /live error:", err);
    return NextResponse.json({ error: "Database query failed" }, { status: 500 });
  }
}

async function queryExchangePositions(): Promise<ExchangePositionRow[]> {
  try {
    const result = await pool.query<ExchangePositionRow>(
      `SELECT symbol, direction, qty, avg_entry, unrealized_pnl,
              liquidation_price, observed_at
       FROM exchange_positions
       ORDER BY symbol`
    );
    return result.rows;
  } catch (err) {
    if (isMissingRelation(err)) return [];
    throw err;
  }
}

async function queryStrategyPositionAllocations(): Promise<StrategyPositionAllocationRow[]> {
  try {
    const result = await pool.query<StrategyPositionAllocationRow>(
      `SELECT position_instance_id, strategy_id, symbol, direction,
              allocated_qty, avg_entry, risk_r, entry_time, status,
              confidence, source, entry_order_ids, entry_fill_ids,
              exit_order_ids, exit_fill_ids
       FROM strategy_position_allocations
       ORDER BY symbol, strategy_id, position_instance_id`
    );
    return result.rows;
  } catch (err) {
    if (isMissingRelation(err)) return [];
    throw err;
  }
}

function buildAllocationResiduals(
  exchangePositions: ExchangePositionRow[],
  strategyAllocations: StrategyPositionAllocationRow[]
): AllocationResidualRow[] {
  const residuals: AllocationResidualRow[] = [];
  const seen = new Set<string>();
  const keyOf = (symbol: string, direction: string) => `${symbol}|${direction}`;

  for (const net of exchangePositions) {
    const key = keyOf(net.symbol, net.direction);
    seen.add(key);
    const allocatedQty = strategyAllocations
      .filter((row) => row.symbol === net.symbol && row.direction === net.direction)
      .reduce((sum, row) => sum + Number(row.allocated_qty ?? 0), 0);
    const unallocatedQty = Number(net.qty ?? 0) - allocatedQty;
    if (Math.abs(unallocatedQty) > 1e-8) {
      residuals.push({
        symbol: net.symbol,
        direction: net.direction,
        net_exchange_qty: Number(net.qty ?? 0),
        allocated_qty: allocatedQty,
        unallocated_qty: unallocatedQty,
        unknown_allocation: unallocatedQty > 1e-8,
        status: "DRIFT",
      });
    }
  }

  const allocatedByKey = new Map<string, { symbol: string; direction: string; qty: number }>();
  for (const row of strategyAllocations) {
    const key = keyOf(row.symbol, row.direction);
    const current = allocatedByKey.get(key) ?? { symbol: row.symbol, direction: row.direction, qty: 0 };
    current.qty += Number(row.allocated_qty ?? 0);
    allocatedByKey.set(key, current);
  }
  allocatedByKey.forEach((row, key) => {
    if (seen.has(key) || row.qty <= 1e-8) return;
    residuals.push({
      symbol: row.symbol,
      direction: row.direction,
      net_exchange_qty: 0,
      allocated_qty: row.qty,
      unallocated_qty: -row.qty,
      unknown_allocation: false,
      status: "DRIFT",
    });
  });

  return residuals;
}

function isMissingRelation(err: unknown): boolean {
  const code = typeof err === "object" && err !== null && "code" in err
    ? String((err as { code?: unknown }).code)
    : "";
  return code === "42P01" || code === "42703";
}

async function querySafetyEvents(): Promise<SafetyEventRow[]> {
  try {
    const result = await pool.query<SafetyEventRow>(
      `WITH safety_events AS (
         SELECT
           event_id,
           event_type,
           COALESCE(exchange_timestamp, local_timestamp, received_at) AS timestamp,
           strategy_id,
           symbol,
           payload,
           COALESCE(payload->'payload', '{}'::jsonb) AS body,
           COALESCE(payload->'metadata', '{}'::jsonb) AS envelope_metadata,
           COALESCE(payload->'payload'->'metadata', '{}'::jsonb) AS body_metadata
         FROM instrumentation_events
       )
       SELECT
         event_id,
         event_type,
         timestamp,
         COALESCE(
           payload->>'severity',
           body->>'severity',
           envelope_metadata->>'severity',
           body_metadata->>'severity'
         ) AS severity,
         COALESCE(strategy_id, payload->>'strategy_id', body->>'strategy_id') AS strategy_id,
         COALESCE(symbol, payload->>'symbol', body->>'symbol', payload->>'pair', body->>'pair') AS symbol,
         COALESCE(
           payload->>'status',
           body->>'status',
           envelope_metadata->>'status',
           body_metadata->>'status'
         ) AS status,
         COALESCE(
           payload->>'description',
           body->>'description',
           payload->>'reject_reason',
           body->>'reject_reason',
           payload->>'denial_reason',
           body->>'denial_reason',
           payload->>'reason',
           body->>'reason',
           payload->>'message',
           body->>'message',
           envelope_metadata->>'description',
           body_metadata->>'description'
         ) AS description
       FROM safety_events
       WHERE event_type = 'reconciliation_event'
          OR event_type = 'error'
          OR LOWER(COALESCE(payload->>'severity', body->>'severity', envelope_metadata->>'severity', body_metadata->>'severity', '')) IN ('warning', 'error', 'critical')
          OR LOWER(COALESCE(payload->>'status', body->>'status', envelope_metadata->>'status', body_metadata->>'status', '')) IN ('open', 'failed', 'rejected', 'blocked')
          OR LOWER(COALESCE(payload->>'event_kind', body->>'event_kind', envelope_metadata->>'event_kind', body_metadata->>'event_kind', '')) = 'rejected'
          OR LOWER(COALESCE(payload->>'approved', body->>'approved', '')) = 'false'
          OR LOWER(COALESCE(payload->>'action', body->>'action', envelope_metadata->>'action', body_metadata->>'action', '')) = 'block'
          OR COALESCE(payload->>'reject_reason', body->>'reject_reason') IS NOT NULL
          OR COALESCE(payload->>'denial_reason', body->>'denial_reason') IS NOT NULL
          OR COALESCE(envelope_metadata->>'discrepancy_kind', body_metadata->>'discrepancy_kind', body->>'discrepancy_kind') IS NOT NULL
       ORDER BY timestamp DESC
       LIMIT 25`
    );
    return result.rows;
  } catch (err) {
    const code = typeof err === "object" && err !== null && "code" in err
      ? String((err as { code?: unknown }).code)
      : "";
    if (isMissingRelation(err)) {
      console.warn("Safety event query skipped:", code);
      return [];
    }
    throw err;
  }
}
