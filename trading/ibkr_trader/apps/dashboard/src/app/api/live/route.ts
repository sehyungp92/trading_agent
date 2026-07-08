import { NextResponse } from 'next/server';
import { query } from '@/lib/db';
import { getDashboardAccountId, getDashboardRuntimeEnv } from '@/lib/active-config';
import { getEvidencePipelineHealth } from '@/lib/evidence-health';
import {
  SYSTEM_ORDER,
  getSystem,
  setRegistryCache,
  type PortfolioData,
  type StrategyData,
  type PositionRow,
  type TradeRow,
  type OrderRow,
  type StrategyHealthRow,
  type AdapterHealthRow,
  type HaltRow,
  type HealthData,
  type SystemPnlSummary,
  type QueueSummary,
  type LiveBatchResponse,
} from '@/lib/types';

export const dynamic = 'force-dynamic';

type ActiveAccountConfigRow = {
  account_id: string;
  config_scope: string;
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

function configNumberFromAny(
  payload: Record<string, unknown> | undefined,
  keys: string[],
): number | null {
  for (const key of keys) {
    const value = configNumber(payload, key);
    if (value !== null) {
      return value;
    }
  }
  return null;
}

function configString(
  payload: Record<string, unknown> | undefined,
  key: string,
): string | null {
  const value = payload?.[key];
  return typeof value === 'string' && value.trim() ? value.trim() : null;
}

function configBoolean(
  payload: Record<string, unknown> | undefined,
  key: string,
): boolean | null {
  const value = payload?.[key];
  return typeof value === 'boolean' ? value : null;
}

function isNonEmptyString(value: string | null | undefined): value is string {
  return typeof value === 'string' && value.length > 0;
}

export async function GET() {
  try {
    const runtimeEnv = getDashboardRuntimeEnv();
    const accountId = getDashboardAccountId();
    // Parallel DB queries (down from separate endpoint fan-out)
    // Note: strategies and health used to both query v_strategy_health; now we query it once.
    const [
      portfolioRows,
      unrealizedRows,
      rawStrategyRows,
      adapterRows,
      haltRows,
      positionRows,
      tradeRows,
      orderRows,
      registryRows,
      queueRows,
      activeConfigRows,
      activeRiskConfigRows,
      evidence,
    ] = await Promise.all([
      // Portfolio realized (aggregated view handles multi-family)
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
      // Portfolio unrealized + heat
      query<{ unrealized_pnl: number; heat_r: number }>(
        `SELECT
           COALESCE(SUM(unrealized_pnl), 0) AS unrealized_pnl,
           COALESCE(SUM(open_risk_r), 0) AS heat_r
         FROM positions
         WHERE net_qty != 0`
      ),
      // Strategies (v_strategy_health + risk_daily_strategy; one query serves both strategies and health)
      query<StrategyData>(
        `SELECT
           sh.strategy_id,
           sh.mode,
           sh.last_heartbeat_ts,
           sh.heartbeat_age_sec,
           sh.health_status,
           sh.heat_r,
           sh.daily_pnl_r,
           sh.last_error,
           sh.last_error_ts,
           COALESCE(rds.daily_realized_r, 0)   AS daily_realized_r,
           COALESCE(rds.daily_realized_usd, 0)  AS daily_realized_usd,
           COALESCE(rds.open_risk_r, 0)         AS open_risk_r,
           COALESCE(rds.filled_entries, 0)      AS filled_entries,
           COALESCE(rds.halted, FALSE)          AS halted,
           rds.halt_reason
         FROM v_strategy_health sh
         LEFT JOIN risk_daily_strategy rds
           ON sh.strategy_id = rds.strategy_id AND rds.trade_date = (now() AT TIME ZONE 'America/New_York')::date
         ORDER BY sh.strategy_id`
      ),
      // Adapter health
      query<AdapterHealthRow>(
        `SELECT * FROM v_adapter_health ORDER BY adapter_id`
      ),
      // Active halts
      query<HaltRow>(
        `SELECT halt_level, entity, halt_reason, last_update_at
         FROM v_active_halts
         ORDER BY halt_level, entity`
      ),
      // Positions
      query<PositionRow>(
        `SELECT
           account_id, instrument_symbol, strategy_id, net_qty, avg_price,
           unrealized_pnl, realized_pnl, open_risk_dollars, open_risk_r,
           last_update_at,
           EXTRACT(EPOCH FROM (now() - last_update_at)) / 60 AS stale_minutes
         FROM positions
         WHERE net_qty != 0
         ORDER BY last_update_at DESC`
      ),
      // Trades
      query<TradeRow>(
        `SELECT * FROM v_today_trades LIMIT 50`
      ),
      // Orders
      query<OrderRow>(
        `SELECT * FROM v_working_orders ORDER BY created_at DESC`
      ),
      // Registry: strategy-family mapping from DB (drives getSystem())
      query<{ strategy_id: string; family_id: string }>(
        `SELECT DISTINCT ON (strategy_id) strategy_id, family_id
         FROM risk_daily_strategy
         WHERE family_id IS NOT NULL
           AND family_id != 'unknown'
         ORDER BY strategy_id, trade_date DESC`
      ),
      query<QueueSummary>(
        `SELECT
           COUNT(*)::int AS queued_count,
           MIN(queued_at) AS oldest_queued_at,
           EXTRACT(EPOCH FROM (now() - MIN(queued_at))) AS oldest_queued_age_seconds
         FROM orders
         WHERE status = 'QUEUED'`
      ),
      query<ActiveAccountConfigRow>(
        `SELECT
           account_id,
           config_scope,
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
      query<ActiveAccountConfigRow>(
        `SELECT
           account_id,
           config_scope,
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
         WHERE runtime_env = $1
           AND account_id = $2
           AND config_scope IN ('family', 'strategy')
         ORDER BY config_scope, scope_id`,
        [runtimeEnv, accountId],
      ),
      getEvidencePipelineHealth(),
    ]);

    // Assemble portfolio
    const portfolioRaw = portfolioRows[0] ?? {
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
    const activeConfigStatus = activeAccountConfig == null || accountKeyWarnings.length > 0
      ? 'missing'
      : activeAccountConfig.freshness_status;
    const activeConfigWarnings = [
      ...(!accountId ? ['dashboard account id is not configured'] : []),
      ...(activeAccountConfig == null
        ? [`missing account active config for ${runtimeEnv}/${accountId || 'unconfigured account'}`]
        : []),
      ...(activeAccountConfig?.freshness_status === 'stale'
        ? ['account active config older than 24 hours or expired']
        : []),
      ...accountKeyWarnings,
      ...(globalStanddown === true ? ['account global stand-down active'] : []),
    ];
    const portfolio: PortfolioData = {
      daily_realized_r: portfolioRaw.daily_realized_r,
      daily_realized_usd: portfolioRaw.daily_realized_usd,
      portfolio_open_risk_r: portfolioRaw.portfolio_open_risk_r,
      unrealized_pnl: unrealized.unrealized_pnl,
      halted: portfolioRaw.halted,
      halt_reason: portfolioRaw.halt_reason,
      heat_r: unrealized.heat_r,
      heat_cap_R: heatCapR,
      portfolio_daily_stop_R: dailyStopR,
      portfolio_weekly_stop_R: weeklyStopR,
      global_standdown: globalStanddown,
      active_config_status: activeConfigStatus,
      active_config_warnings: activeConfigWarnings,
    };

    const strategyConfigById = new Map(
      activeRiskConfigRows
        .filter(row => row.config_scope === 'strategy')
        .map(row => [row.scope_id, row]),
    );
    const familyConfigById = new Map(
      activeRiskConfigRows
        .filter(row => row.config_scope === 'family')
        .map(row => [row.scope_id, row]),
    );
    const familyMap: Record<string, string> = {};
    for (const r of registryRows) {
      familyMap[r.strategy_id] = r.family_id;
    }
    for (const row of Array.from(strategyConfigById.values())) {
      const familyId = configString(row.payload, 'family_id');
      if (familyId && !(row.scope_id in familyMap)) {
        familyMap[row.scope_id] = familyId;
      }
    }
    setRegistryCache(familyMap);

    const strategyRows: StrategyData[] = rawStrategyRows.map(s => {
      const activeStrategyConfig = strategyConfigById.get(s.strategy_id) ?? null;
      const strategyPayload = activeStrategyConfig?.payload;
      const familyId = configString(strategyPayload, 'family_id') ?? familyMap[s.strategy_id] ?? null;
      const activeAllocatedNav = configNumber(strategyPayload, 'allocated_nav');
      const activeUnitRiskDollars = configNumber(strategyPayload, 'unit_risk_dollars');
      const activeRiskPerTrade = configNumber(strategyPayload, 'risk_per_trade');
      const activeMaxHeatR = configNumberFromAny(strategyPayload, [
        'max_heat_R',
        'strategy_heat_cap_R',
      ]);
      const activeMaxDailyLossR = configNumber(strategyPayload, 'max_daily_loss_R');
      const activeMaxWeeklyLossR = configNumber(strategyPayload, 'max_weekly_loss_R');
      const missingWarnings = [
        ...(activeStrategyConfig == null ? [`missing strategy active config for ${s.strategy_id}`] : []),
        ...(activeRiskPerTrade == null ? ['missing strategy risk per trade'] : []),
        ...(activeMaxHeatR == null ? ['missing strategy heat cap'] : []),
        ...(activeMaxDailyLossR == null ? ['missing strategy daily loss limit'] : []),
        ...(activeMaxWeeklyLossR == null ? ['missing strategy weekly loss limit'] : []),
      ];

      return {
        ...s,
        family_id: familyId,
        active_config_status: activeStrategyConfig == null || missingWarnings.length > 0
          ? 'missing'
          : activeStrategyConfig.freshness_status,
        active_config_warnings: [
          ...missingWarnings,
          ...(activeStrategyConfig?.freshness_status === 'stale'
            ? ['strategy active config older than 24 hours or expired']
            : []),
        ],
        active_allocated_nav: activeAllocatedNav,
        active_unit_risk_dollars: activeUnitRiskDollars,
        active_risk_per_trade: activeRiskPerTrade,
        active_max_heat_R: activeMaxHeatR,
        active_max_daily_loss_R: activeMaxDailyLossR,
        active_max_weekly_loss_R: activeMaxWeeklyLossR,
      };
    });

    // Derive health from strategy rows (no redundant v_strategy_health query)
    const healthStrategies: StrategyHealthRow[] = strategyRows.map(s => ({
      strategy_id: s.strategy_id,
      mode: s.mode,
      last_heartbeat_ts: s.last_heartbeat_ts,
      heartbeat_age_sec: s.heartbeat_age_sec,
      health_status: s.health_status,
      heat_r: s.heat_r,
      daily_pnl_r: s.daily_pnl_r,
      last_error: s.last_error,
      last_error_ts: s.last_error_ts,
    }));

    const health: HealthData = {
      strategies: healthStrategies,
      adapters: adapterRows,
      halts: haltRows,
      evidence,
    };

    // Compute per-system P&L summaries
    const systemPnl: SystemPnlSummary[] = SYSTEM_ORDER.map(sys => {
      const strats = strategyRows.filter(s => getSystem(s.strategy_id) === sys);
      const familyIds = Array.from(new Set(strats.map(s => s.family_id).filter(isNonEmptyString)));
      const familyConfigs = familyIds.map(familyId => familyConfigById.get(familyId) ?? null);
      const familyHeatCaps = familyConfigs
        .map(row => configNumber(row?.payload, 'family_heat_cap_R'))
        .filter((value): value is number => value !== null);
      const activeHeatCapR = familyIds.length > 0 && familyHeatCaps.length === familyIds.length
        ? familyHeatCaps.reduce((sum, value) => sum + value, 0)
        : null;
      const familyMissingWarnings = [
        ...(familyIds.length === 0 && strats.length > 0 ? ['missing strategy family mapping'] : []),
        ...familyIds
          .filter(familyId => !familyConfigById.has(familyId))
          .map(familyId => `missing family active config for ${familyId}`),
        ...familyIds
          .filter(familyId => {
            const familyConfig = familyConfigById.get(familyId);
            return familyConfig != null && configNumber(familyConfig.payload, 'family_heat_cap_R') == null;
          })
          .map(familyId => `missing family heat cap for ${familyId}`),
        ...familyIds
          .filter(familyId => {
            const familyConfig = familyConfigById.get(familyId);
            return familyConfig != null && configNumber(familyConfig.payload, 'family_daily_stop_R') == null;
          })
          .map(familyId => `missing family daily stop for ${familyId}`),
        ...familyIds
          .filter(familyId => {
            const familyConfig = familyConfigById.get(familyId);
            return familyConfig != null && configNumber(familyConfig.payload, 'family_weekly_stop_R') == null;
          })
          .map(familyId => `missing family weekly stop for ${familyId}`),
      ];
      const familyStaleWarnings = familyConfigs.flatMap(row =>
        row?.freshness_status === 'stale'
          ? [`${row.scope_id} family active config older than 24 hours or expired`]
          : [],
      );
      const familyWarnings = [...familyMissingWarnings, ...familyStaleWarnings];
      const familyStatus: 'fresh' | 'stale' | 'missing' = familyMissingWarnings.length > 0
        ? 'missing'
        : familyStaleWarnings.length > 0
        ? 'stale'
        : 'fresh';
      return {
        system: sys,
        daily_realized_r: strats.reduce((sum, s) => sum + s.daily_realized_r, 0),
        daily_realized_usd: strats.reduce((sum, s) => sum + s.daily_realized_usd, 0),
        heat_r: strats.reduce((sum, s) => sum + s.heat_r, 0),
        active_heat_cap_R: activeHeatCapR,
        active_config_status: familyStatus,
        active_config_warnings: familyWarnings,
        filled_entries: strats.reduce((sum, s) => sum + s.filled_entries, 0),
        strategy_count: strats.length,
        healthy_count: strats.filter(s => s.health_status === 'OK').length,
      };
    }).filter(sp => sp.strategy_count > 0);

    const data: LiveBatchResponse = {
      portfolio,
      strategies: strategyRows,
      positions: positionRows,
      trades: tradeRows,
      orders: orderRows,
      health,
      systemPnl,
      queue: queueRows[0] ?? {
        queued_count: 0,
        oldest_queued_at: null,
        oldest_queued_age_seconds: null,
      },
      serverTime: new Date().toISOString(),
    };

    return NextResponse.json(data, {
      headers: { 'Cache-Control': 'no-store' },
    });
  } catch (err) {
    console.error('[api/live] SQL error:', err instanceof Error ? err.message : err);
    return NextResponse.json(
      { error: 'database_query_failed', detail: err instanceof Error ? err.message : 'unknown' },
      { status: 500 },
    );
  }
}
