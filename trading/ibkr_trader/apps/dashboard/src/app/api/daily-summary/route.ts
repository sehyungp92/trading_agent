import { NextResponse } from 'next/server';
import { query } from '@/lib/db';
import type { DailyClassificationRow } from '@/lib/types';

export const dynamic = 'force-dynamic';

interface RawRow {
  strategy_id: string;
  family_id: string | null;
  bars: number | null;
  denials: number | null;
  trades: number | null;
  last_bar_ts: string | null;
  last_decision_code: string | null;
  daily_realized_r: number | null;
  family_disconnect_count_24h: number;
}

const SESSION_STALE_HOURS = 8;

function classify(row: RawRow): DailyClassificationRow['classification'] {
  const bars = row.bars ?? 0;
  const trades = row.trades ?? 0;
  const denials = row.denials ?? 0;
  if (trades > 0) return 'ACTIVE';

  let stale = false;
  if (row.last_bar_ts) {
    const ageHours = (Date.now() - Date.parse(row.last_bar_ts)) / 3600_000;
    stale = ageHours > SESSION_STALE_HOURS;
  }

  if (bars === 0 || row.last_bar_ts === null || stale) {
    return row.family_disconnect_count_24h > 0 ? 'BROKER_DOWN' : 'DEAD';
  }
  if (denials > 0) return 'BLOCKED';
  return 'NORMAL_QUIET';
}

export async function GET() {
  try {
    // Mirrors apps/watchdog/checks.py::classify_daily_activity. Pulls
    // v_daily_strategy_activity and adapter disconnect counts for the
    // BROKER_DOWN signal.
    const rows = await query<RawRow>(
      `SELECT
         a.strategy_id,
         a.family_id,
         a.bars,
         a.denials,
         a.trades,
         a.last_bar_ts,
         a.last_decision_code,
         a.daily_realized_r,
         COALESCE(ad.disconnect_count_24h, 0) AS family_disconnect_count_24h
       FROM v_daily_strategy_activity a
       LEFT JOIN adapter_state ad
         ON ad.adapter_id = a.family_id
       WHERE a.day = (now() AT TIME ZONE 'UTC')::date
       ORDER BY a.strategy_id`
    );

    const out: DailyClassificationRow[] = rows.map(r => ({
      strategy_id: r.strategy_id,
      family_id: r.family_id,
      bars: r.bars,
      trades: r.trades,
      denials: r.denials,
      last_bar_ts: r.last_bar_ts,
      last_decision_code: r.last_decision_code,
      classification: classify(r),
      daily_realized_r: r.daily_realized_r,
    }));

    return NextResponse.json(out, {
      headers: { 'Cache-Control': 'no-store' },
    });
  } catch (err) {
    console.error('[api/daily-summary] SQL error:', err instanceof Error ? err.message : err);
    return NextResponse.json(
      { error: 'database_query_failed', detail: err instanceof Error ? err.message : 'unknown' },
      { status: 500 },
    );
  }
}
