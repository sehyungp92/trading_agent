import { NextResponse } from 'next/server';
import { query } from '@/lib/db';
import { getDashboardAccountId, getDashboardRuntimeEnv } from '@/lib/active-config';
import type { PortfolioData } from '@/lib/types';

export const dynamic = 'force-dynamic';

type ActiveAccountConfigRow = {
  account_id: string;
  payload: Record<string, unknown>;
  freshness_status: 'fresh' | 'stale';
  scope_id: string;
  runtime_env: string;
  applied_at: string;
  expires_at: string | null;
};

function configNumber(
  payload: Record<string, unknown> | undefined,
  key: string,
): number | null {
  const value = payload?.[key];
  const numberValue = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(numberValue) ? numberValue : null;
}

function configBoolean(
  payload: Record<string, unknown> | undefined,
  key: string,
): boolean | null {
  const value = payload?.[key];
  return typeof value === 'boolean' ? value : null;
}

export async function GET() {
  try {
    const runtimeEnv = getDashboardRuntimeEnv();
    const accountId = getDashboardAccountId();
    const [portfolioRows, unrealizedRows, activeConfigRows] = await Promise.all([
      query<{
        daily_realized_r: number;
        daily_realized_usd: number;
        portfolio_open_risk_r: number;
        halted: boolean;
        halt_reason: string | null;
      }>(
        `SELECT daily_realized_r, daily_realized_usd, portfolio_open_risk_r, halted, halt_reason
         FROM v_portfolio_daily_summary
         WHERE trade_date = (now() AT TIME ZONE 'America/New_York')::date`
      ),
      query<{ unrealized_pnl: number; heat_r: number }>(
        `SELECT
           COALESCE(SUM(unrealized_pnl), 0) AS unrealized_pnl,
           COALESCE(SUM(open_risk_r), 0) AS heat_r
         FROM positions
         WHERE net_qty != 0`
      ),
      query<ActiveAccountConfigRow>(
        `SELECT
           account_id,
           payload,
           scope_id,
           runtime_env,
           applied_at,
           expires_at,
           CASE
             WHEN expires_at IS NOT NULL AND expires_at <= now() THEN 'stale'
             WHEN applied_at < now() - INTERVAL '24 hours' THEN 'stale'
             ELSE 'fresh'
           END AS freshness_status
         FROM active_runtime_config
         WHERE config_scope = 'account'
           AND account_id = $2
           AND runtime_env = $1
           AND scope_id = $2
         ORDER BY applied_at DESC
         LIMIT 1`,
        [runtimeEnv, accountId],
      ),
    ]);

    const portfolio = portfolioRows[0] ?? {
      daily_realized_r: 0,
      daily_realized_usd: 0,
      portfolio_open_risk_r: 0,
      halted: false,
      halt_reason: null,
    };

    const unrealized = unrealizedRows[0] ?? { unrealized_pnl: 0, heat_r: 0 };
    const activeAccountConfig = activeConfigRows[0] ?? null;
    const activePayload = activeAccountConfig?.payload;
    const heatCapR = configNumber(activePayload, 'heat_cap_R');
    const dailyStopR = configNumber(activePayload, 'portfolio_daily_stop_R');
    const weeklyStopR = configNumber(activePayload, 'portfolio_weekly_stop_R');
    const globalStanddown = configBoolean(activePayload, 'global_standdown');
    const accountKeyWarnings = activeAccountConfig == null ? [] : [
      ...(heatCapR == null ? ['missing account heat cap'] : []),
      ...(dailyStopR == null ? ['missing account daily stop'] : []),
      ...(weeklyStopR == null ? ['missing account weekly stop'] : []),
      ...(globalStanddown == null ? ['missing account global stand-down'] : []),
    ];

    const data: PortfolioData = {
      daily_realized_r: portfolio.daily_realized_r,
      daily_realized_usd: portfolio.daily_realized_usd,
      portfolio_open_risk_r: portfolio.portfolio_open_risk_r,
      unrealized_pnl: unrealized.unrealized_pnl,
      halted: portfolio.halted,
      halt_reason: portfolio.halt_reason,
      heat_r: unrealized.heat_r,
      heat_cap_R: heatCapR,
      portfolio_daily_stop_R: dailyStopR,
      portfolio_weekly_stop_R: weeklyStopR,
      global_standdown: globalStanddown,
      active_config_status: activeAccountConfig == null || accountKeyWarnings.length > 0
        ? 'missing'
        : activeAccountConfig.freshness_status,
      active_config_warnings: [
        ...(!accountId ? ['dashboard account id is not configured'] : []),
        ...(activeAccountConfig == null
          ? [`missing account active config for ${runtimeEnv}/${accountId || 'unconfigured account'}`]
          : []),
        ...(activeAccountConfig?.freshness_status === 'stale'
          ? ['account active config older than 24 hours or expired']
          : []),
        ...accountKeyWarnings,
        ...(globalStanddown === true ? ['account global stand-down active'] : []),
      ],
    };

    return NextResponse.json(data, {
      headers: { 'Cache-Control': 'no-store' },
    });
  } catch (err) {
    console.error('[api/portfolio] SQL error:', err instanceof Error ? err.message : err);
    return NextResponse.json(
      { error: 'database_query_failed', detail: err instanceof Error ? err.message : 'unknown' },
      { status: 500 },
    );
  }
}
