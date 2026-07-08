import { NextResponse } from 'next/server';
import { query } from '@/lib/db';
import type { TradeRow } from '@/lib/types';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    // v_today_trades joins trades + trade_marks, filters entry_ts >= CURRENT_DATE,
    // orders by entry_ts DESC, and exposes duration_minutes
    const rows = await query<TradeRow>(
      `SELECT * FROM v_today_trades LIMIT 50`
    );

    return NextResponse.json(rows, {
      headers: { 'Cache-Control': 'no-store' },
    });
  } catch (err) {
    console.error('[api/trades] SQL error:', err instanceof Error ? err.message : err);
    return NextResponse.json(
      { error: 'database_query_failed', detail: err instanceof Error ? err.message : 'unknown' },
      { status: 500 },
    );
  }
}
