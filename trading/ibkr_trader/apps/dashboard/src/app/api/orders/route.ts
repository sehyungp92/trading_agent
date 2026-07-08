import { NextResponse } from 'next/server';
import { query } from '@/lib/db';
import type { OrderRow } from '@/lib/types';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    // v_working_orders filters to active statuses and computes age_minutes
    const rows = await query<OrderRow>(
      `SELECT * FROM v_working_orders ORDER BY created_at DESC`
    );

    return NextResponse.json(rows, {
      headers: { 'Cache-Control': 'no-store' },
    });
  } catch (err) {
    console.error('[api/orders] SQL error:', err instanceof Error ? err.message : err);
    return NextResponse.json(
      { error: 'database_query_failed', detail: err instanceof Error ? err.message : 'unknown' },
      { status: 500 },
    );
  }
}
