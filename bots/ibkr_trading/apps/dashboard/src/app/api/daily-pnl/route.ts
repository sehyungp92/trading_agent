import { NextResponse } from 'next/server';
import { query } from '@/lib/db';
import type { DailyPnlPoint } from '@/lib/types';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const rows = await query<DailyPnlPoint>(
      `SELECT trade_date, daily_realized_r
       FROM v_portfolio_daily_summary
       WHERE trade_date >= CURRENT_DATE - INTERVAL '30 days'
       ORDER BY trade_date`
    );

    return NextResponse.json(rows, {
      headers: { 'Cache-Control': 'no-store' },
    });
  } catch (err) {
    console.error('[api/daily-pnl] SQL error:', err instanceof Error ? err.message : err);
    return NextResponse.json(
      { error: 'database_query_failed', detail: err instanceof Error ? err.message : 'unknown' },
      { status: 500 },
    );
  }
}
