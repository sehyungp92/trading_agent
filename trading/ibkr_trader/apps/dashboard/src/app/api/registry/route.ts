import { NextResponse } from 'next/server';
import { query } from '@/lib/db';

export const dynamic = 'force-dynamic';

interface RegistryRow {
  strategy_id: string;
  family_id: string;
}

/**
 * Serves strategy→family mappings from the database.
 * Replaces the hardcoded STRATEGY_CONFIG as the source of truth for which
 * system a strategy belongs to.
 */
export async function GET() {
  try {
    const rows = await query<RegistryRow>(
      `SELECT DISTINCT ON (strategy_id) strategy_id, family_id
       FROM risk_daily_strategy
       WHERE family_id IS NOT NULL
         AND family_id != 'unknown'
       ORDER BY strategy_id, trade_date DESC`
    );

    // Build a strategy_id → family_id mapping
    const registry: Record<string, string> = {};
    for (const row of rows) {
      registry[row.strategy_id] = row.family_id;
    }

    return NextResponse.json(registry, {
      headers: { 'Cache-Control': 'public, max-age=300' },
    });
  } catch (err) {
    console.error('[api/registry] SQL error:', err instanceof Error ? err.message : err);
    return NextResponse.json(
      { error: 'database_query_failed', detail: err instanceof Error ? err.message : 'unknown' },
      { status: 500 },
    );
  }
}
