import { NextResponse } from 'next/server';
import { query } from '@/lib/db';
import { getDashboardAccountId, getDashboardRuntimeEnv } from '@/lib/active-config';

export const dynamic = 'force-dynamic';

type RuntimeConfigRow = {
  account_id: string;
  config_scope: string;
  scope_id: string;
  runtime_env: string;
  config_version: string;
  deployment_id: string | null;
  source_hash: string | null;
  payload: Record<string, unknown>;
  applied_at: string;
  expires_at: string | null;
  freshness_status: 'fresh' | 'stale';
};

function hasNumber(payload: Record<string, unknown>, key: string): boolean {
  const value = payload[key];
  const numberValue = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(numberValue);
}

function hasBoolean(payload: Record<string, unknown>, key: string): boolean {
  return typeof payload[key] === 'boolean';
}

function missingRequiredPayloadWarnings(row: RuntimeConfigRow): string[] {
  if (row.config_scope === 'account') {
    return [
      ...(!hasNumber(row.payload, 'heat_cap_R') ? [`${row.scope_id} missing account heat cap`] : []),
      ...(!hasNumber(row.payload, 'portfolio_daily_stop_R') ? [`${row.scope_id} missing account daily stop`] : []),
      ...(!hasNumber(row.payload, 'portfolio_weekly_stop_R') ? [`${row.scope_id} missing account weekly stop`] : []),
      ...(!hasBoolean(row.payload, 'global_standdown') ? [`${row.scope_id} missing account global stand-down`] : []),
    ];
  }
  if (row.config_scope === 'family') {
    return [
      ...(!hasNumber(row.payload, 'family_heat_cap_R') ? [`${row.scope_id} missing family heat cap`] : []),
      ...(!hasNumber(row.payload, 'family_daily_stop_R') ? [`${row.scope_id} missing family daily stop`] : []),
      ...(!hasNumber(row.payload, 'family_weekly_stop_R') ? [`${row.scope_id} missing family weekly stop`] : []),
    ];
  }
  if (row.config_scope === 'strategy') {
    return [
      ...(!hasNumber(row.payload, 'risk_per_trade') ? [`${row.scope_id} missing strategy risk per trade`] : []),
      ...(!hasNumber(row.payload, 'max_heat_R') && !hasNumber(row.payload, 'strategy_heat_cap_R')
        ? [`${row.scope_id} missing strategy heat cap`]
        : []),
      ...(!hasNumber(row.payload, 'max_daily_loss_R') ? [`${row.scope_id} missing strategy daily loss limit`] : []),
      ...(!hasNumber(row.payload, 'max_weekly_loss_R') ? [`${row.scope_id} missing strategy weekly loss limit`] : []),
    ];
  }
  return [];
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const runtimeEnv = searchParams.get('runtime_env')?.trim() || getDashboardRuntimeEnv();
    const accountId = searchParams.get('account_id')?.trim() || getDashboardAccountId();
    const rows = await query<RuntimeConfigRow>(
      `SELECT
         account_id,
         config_scope,
         scope_id,
         runtime_env,
         config_version,
         deployment_id,
         source_hash,
         payload,
         applied_at,
         expires_at,
         CASE
           WHEN expires_at IS NOT NULL AND expires_at <= now() THEN 'stale'
           WHEN applied_at < now() - INTERVAL '24 hours' THEN 'stale'
           ELSE 'fresh'
         END AS freshness_status
       FROM active_runtime_config
       WHERE runtime_env = $1
         AND account_id = $2
       ORDER BY config_scope, scope_id`,
      [runtimeEnv, accountId],
    );

    const hasAccount = rows.some(row =>
      row.config_scope === 'account' && row.scope_id === accountId && row.account_id === accountId
    );
    const hasFamily = rows.some(row => row.config_scope === 'family');
    const hasStrategy = rows.some(row => row.config_scope === 'strategy');
    const stale = rows.some(row => row.freshness_status === 'stale');
    const requiredPayloadWarnings = rows.flatMap(missingRequiredPayloadWarnings);

    return NextResponse.json(
      {
        status: rows.length > 0 && hasAccount && hasFamily && hasStrategy && !stale && requiredPayloadWarnings.length === 0
          ? 'ok'
          : 'degraded',
        target: {
          runtime_env: runtimeEnv,
          account_id: accountId || null,
        },
        warnings: [
          ...(!accountId ? ['dashboard account id is not configured'] : []),
          ...(!hasAccount
            ? [`missing account active config for ${runtimeEnv}/${accountId || 'unconfigured account'}`]
            : []),
          ...(!hasFamily ? ['missing family active config'] : []),
          ...(!hasStrategy ? ['missing strategy active config'] : []),
          ...(stale ? ['active config older than 24 hours or expired'] : []),
          ...requiredPayloadWarnings,
        ],
        records: rows,
        serverTime: new Date().toISOString(),
      },
      { headers: { 'Cache-Control': 'no-store' } },
    );
  } catch (err) {
    console.error('[api/runtime-config] SQL error:', err instanceof Error ? err.message : err);
    return NextResponse.json(
      { status: 'degraded', error: 'database_query_failed', detail: err instanceof Error ? err.message : 'unknown' },
      { status: 500 },
    );
  }
}
