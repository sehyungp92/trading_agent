import { NextResponse } from 'next/server';
import { query } from '@/lib/db';
import type { EquityCurvePoint } from '@/lib/types';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const rows = await query<EquityCurvePoint>(
      `SELECT
         trade_date,
         daily_realized_r,
         SUM(daily_realized_r) OVER (ORDER BY trade_date) AS cumulative_r
       FROM v_portfolio_daily_summary
       WHERE trade_date >= CURRENT_DATE - INTERVAL '90 days'
       ORDER BY trade_date`
    );

    return NextResponse.json(rows, {
      headers: { 'Cache-Control': 'no-store' },
    });
  } catch (err) {
    console.error('[api/equity-curve] SQL error:', err instanceof Error ? err.message : err);
    return NextResponse.json(
      { error: 'database_query_failed', detail: err instanceof Error ? err.message : 'unknown' },
      { status: 500 },
    );
  }
}
