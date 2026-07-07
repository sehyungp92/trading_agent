import { NextResponse } from 'next/server';
import { query } from '@/lib/db';
import type { PositionRow } from '@/lib/types';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const rows = await query<PositionRow>(
      `SELECT
         account_id,
         instrument_symbol,
         strategy_id,
         net_qty,
         avg_price,
         unrealized_pnl,
         realized_pnl,
         open_risk_dollars,
         open_risk_r,
         last_update_at,
         EXTRACT(EPOCH FROM (now() - last_update_at)) / 60 AS stale_minutes
       FROM positions
       WHERE net_qty != 0
       ORDER BY last_update_at DESC`
    );

    return NextResponse.json(rows, {
      headers: { 'Cache-Control': 'no-store' },
    });
  } catch (err) {
    console.error('[api/positions] SQL error:', err instanceof Error ? err.message : err);
    return NextResponse.json(
      { error: 'database_query_failed', detail: err instanceof Error ? err.message : 'unknown' },
      { status: 500 },
    );
  }
}
