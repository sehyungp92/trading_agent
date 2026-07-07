import { NextResponse } from 'next/server';
import { query } from '@/lib/db';
import type { StrategyDiagnosticsRow } from '@/lib/types';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    // v_strategy_diagnostics flattens the last_decision_details JSONB so the
    // dashboard does not have to encode the key contract. The schema is
    // owned by libs/services/decision_codes.py.
    const rows = await query<StrategyDiagnosticsRow>(
      `SELECT
         strategy_id,
         mode,
         health_status,
         last_heartbeat_ts,
         heartbeat_age_sec,
         last_decision_code,
         last_seen_bar_ts,
         bar_age_sec,
         bars_processed,
         symbol_freshness,
         intents_submitted,
         intents_denied,
         consecutive_denials,
         ib_farm_status,
         last_error,
         last_error_ts
       FROM v_strategy_diagnostics
       ORDER BY strategy_id`
    );

    return NextResponse.json(rows, {
      headers: { 'Cache-Control': 'no-store' },
    });
  } catch (err) {
    console.error('[api/strategies/diagnostics] SQL error:', err instanceof Error ? err.message : err);
    return NextResponse.json(
      { error: 'database_query_failed', detail: err instanceof Error ? err.message : 'unknown' },
      { status: 500 },
    );
  }
}
