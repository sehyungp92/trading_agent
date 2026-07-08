import { NextResponse } from 'next/server';
import { query } from '@/lib/db';
import type { StrategyData } from '@/lib/types';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    // v_strategy_health handles heartbeat_age_sec and health_status computation;
    // LEFT JOIN risk_daily_strategy for today's realized P&L and risk metrics
    const rows = await query<StrategyData>(
      `SELECT
         sh.strategy_id,
         sh.mode,
         sh.last_heartbeat_ts,
         sh.heartbeat_age_sec,
         sh.health_status,
         sh.heat_r,
         sh.daily_pnl_r,
         sh.last_error,
         sh.last_error_ts,
         COALESCE(rds.daily_realized_r, 0)   AS daily_realized_r,
         COALESCE(rds.daily_realized_usd, 0)  AS daily_realized_usd,
         COALESCE(rds.open_risk_r, 0)         AS open_risk_r,
         COALESCE(rds.filled_entries, 0)      AS filled_entries,
         COALESCE(rds.halted, FALSE)          AS halted,
         rds.halt_reason
       FROM v_strategy_health sh
       LEFT JOIN risk_daily_strategy rds
         ON sh.strategy_id = rds.strategy_id AND rds.trade_date = (now() AT TIME ZONE 'America/New_York')::date
       ORDER BY sh.strategy_id`
    );

    return NextResponse.json(rows, {
      headers: { 'Cache-Control': 'no-store' },
    });
  } catch (err) {
    console.error('[api/strategies] SQL error:', err instanceof Error ? err.message : err);
    return NextResponse.json(
      { error: 'database_query_failed', detail: err instanceof Error ? err.message : 'unknown' },
      { status: 500 },
    );
  }
}
