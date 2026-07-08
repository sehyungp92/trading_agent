// System and strategy constants

export type SystemId = 'swing_trader' | 'momentum_trader' | 'stock_trader' | 'unknown';

export interface SystemConfig {
  label: string;
  shortLabel: string;
  color: string;       // Tailwind border/accent class
  dotColor: string;    // Tailwind bg class for small dots
  textColor: string;   // Tailwind text class
  bgColor: string;     // Tailwind bg class for badges/headers
}

export const SYSTEM_CONFIG: Record<SystemId, SystemConfig> = {
  swing_trader: {
    label: 'Swing Trader',
    shortLabel: 'SWING',
    color: 'border-blue-500',
    dotColor: 'bg-blue-500',
    textColor: 'text-blue-400',
    bgColor: 'bg-blue-950/30',
  },
  momentum_trader: {
    label: 'Momentum Trader',
    shortLabel: 'MOMO',
    color: 'border-purple-500',
    dotColor: 'bg-purple-500',
    textColor: 'text-purple-400',
    bgColor: 'bg-purple-950/30',
  },
  stock_trader: {
    label: 'Stock Trader',
    shortLabel: 'STOCK',
    color: 'border-emerald-500',
    dotColor: 'bg-emerald-500',
    textColor: 'text-emerald-400',
    bgColor: 'bg-emerald-950/30',
  },
  unknown: {
    label: 'Unknown Strategy',
    shortLabel: 'UNKNOWN',
    color: 'border-amber-500',
    dotColor: 'bg-amber-500',
    textColor: 'text-amber-400',
    bgColor: 'bg-amber-950/30',
  },
};

export const SYSTEM_ORDER: SystemId[] = ['swing_trader', 'momentum_trader', 'stock_trader', 'unknown'];

export interface StrategyConfig {
  system: SystemId;
  priority: number;
}

export const STRATEGY_CONFIG: Record<string, StrategyConfig> = {
  ATRSS:                { system: 'swing_trader',    priority: 0 },
  TPC:                  { system: 'swing_trader',    priority: 1 },
  AKC_HELIX:            { system: 'swing_trader',    priority: 2 },
  'NQDTC_v2.1':         { system: 'momentum_trader', priority: 1 },
  NQ_REGIME:            { system: 'momentum_trader', priority: 0 },
  VdubusNQ_v4:          { system: 'momentum_trader', priority: 0 },
  DownturnDominator_v1: { system: 'momentum_trader', priority: 1 },
  IARIC_v1:             { system: 'stock_trader',    priority: 0 },
  ALCB_v1:              { system: 'stock_trader',    priority: 1 },
};

/** Map family_id from DB to SystemId */
const FAMILY_TO_SYSTEM: Record<string, SystemId> = {
  swing: 'swing_trader',
  momentum: 'momentum_trader',
  stock: 'stock_trader',
};

/** Runtime registry cache populated by dashboard API queries. */
let _registryCache: Record<string, SystemId> | null = null;

/** Update the registry cache with strategy-to-family mappings from the DB. */
export function setRegistryCache(familyMap: Record<string, string>): void {
  _registryCache = {};
  for (const [strategyId, familyId] of Object.entries(familyMap)) {
    _registryCache[strategyId] = FAMILY_TO_SYSTEM[familyId] ?? 'unknown';
  }
}

/** Get the system a strategy belongs to.
 *  Priority: 1) DB registry cache, 2) hardcoded STRATEGY_CONFIG, 3) explicit unknown bucket */
export function getSystem(strategyId: string): SystemId {
  if (_registryCache && strategyId in _registryCache) {
    return _registryCache[strategyId];
  }
  return STRATEGY_CONFIG[strategyId]?.system ?? 'unknown';
}

/** Get the SystemConfig for a strategy */
export function getSystemConfig(strategyId: string): SystemConfig {
  return SYSTEM_CONFIG[getSystem(strategyId)];
}

/** Group strategies by system, sorted by SYSTEM_ORDER then priority */
export function groupBySystem<T extends { strategy_id: string }>(items: T[]): Map<SystemId, T[]> {
  const grouped = new Map<SystemId, T[]>();
  for (const sys of SYSTEM_ORDER) grouped.set(sys, []);
  for (const item of items) {
    const sys = getSystem(item.strategy_id);
    grouped.get(sys)!.push(item);
  }
  // Sort each group by priority
  SYSTEM_ORDER.forEach(sys => {
    const list = grouped.get(sys);
    if (list) {
      list.sort((a, b) =>
        (STRATEGY_CONFIG[a.strategy_id]?.priority ?? 99) -
        (STRATEGY_CONFIG[b.strategy_id]?.priority ?? 99)
      );
    }
  });
  return grouped;
}

// API response types

export interface PortfolioData {
  daily_realized_r: number;
  daily_realized_usd: number;
  portfolio_open_risk_r: number;
  unrealized_pnl: number;
  halted: boolean;
  halt_reason: string | null;
  heat_r: number; // sum of strategy heat_r from positions
  heat_cap_R: number | null;
  portfolio_daily_stop_R: number | null;
  portfolio_weekly_stop_R: number | null;
  global_standdown: boolean | null;
  active_config_status: 'fresh' | 'stale' | 'missing';
  active_config_warnings: string[];
}

export interface StrategyData {
  strategy_id: string;
  family_id: string | null;
  mode: string;
  last_heartbeat_ts: string | null;
  heartbeat_age_sec: number;
  health_status: string;
  heat_r: number;
  daily_pnl_r: number;
  last_error: string | null;
  last_error_ts: string | null;
  // from risk_daily_strategy
  daily_realized_r: number;
  daily_realized_usd: number;
  open_risk_r: number;
  filled_entries: number;
  halted: boolean;
  halt_reason: string | null;
  active_config_status: 'fresh' | 'stale' | 'missing';
  active_config_warnings: string[];
  active_allocated_nav: number | null;
  active_unit_risk_dollars: number | null;
  active_risk_per_trade: number | null;
  active_max_heat_R: number | null;
  active_max_daily_loss_R: number | null;
  active_max_weekly_loss_R: number | null;
}

export interface PositionRow {
  account_id: string;
  instrument_symbol: string;
  strategy_id: string;
  net_qty: number;
  avg_price: number;
  unrealized_pnl: number;
  realized_pnl: number;
  open_risk_dollars: number;
  open_risk_r: number;
  last_update_at: string;
  stale_minutes: number;
}

export interface TradeRow {
  trade_id: string;
  strategy_id: string;
  instrument_symbol: string;
  direction: string;
  quantity: number;
  entry_ts: string;
  entry_price: number;
  exit_ts: string | null;
  exit_price: number | null;
  realized_r: number | null;
  exit_reason: string | null;
  entry_type: string | null;
  mae_r: number | null;
  mfe_r: number | null;
  duration_minutes: number | null;
}

export interface OrderRow {
  oms_order_id: string;
  strategy_id: string;
  instrument_symbol: string;
  role: string;
  side: string;
  qty: number;
  filled_qty: number;
  stop_price: number | null;
  limit_price: number | null;
  status: string;
  broker_order_id: string | null;
  created_at: string;
  queued_at: string | null;
  queue_priority: number | null;
  queue_reason: string | null;
  queue_attempt: number;
  queue_expires_at: string | null;
  dequeued_at: string | null;
  queue_denial_reason: string | null;
  age_minutes: number;
}

export interface QueueSummary {
  queued_count: number;
  oldest_queued_at: string | null;
  oldest_queued_age_seconds: number | null;
}

export type EvidenceHealthStatus = 'OK' | 'WARNING' | 'ERROR' | 'UNKNOWN';

export interface EvidencePipelineHealth {
  status: EvidenceHealthStatus;
  checked_at: string;
  warnings: string[];
  relay: {
    status: EvidenceHealthStatus;
    url: string | null;
    reachable: boolean;
    pending_events: number | null;
    oldest_pending_age_seconds: number | null;
    per_bot_pending: Record<string, number>;
    warnings: string[];
  };
  assistant: {
    status: EvidenceHealthStatus;
    required_bot_ids: string[];
    missing_bot_ids: string[];
    stale_bot_ids: string[];
    last_event_per_bot: Record<string, string>;
    warnings: string[];
  };
}

export interface HealthData {
  strategies: StrategyHealthRow[];
  adapters: AdapterHealthRow[];
  halts: HaltRow[];
  evidence: EvidencePipelineHealth | null;
}

export interface StrategyHealthRow {
  strategy_id: string;
  mode: string;
  last_heartbeat_ts: string | null;
  heartbeat_age_sec: number;
  health_status: string;
  heat_r: number;
  daily_pnl_r: number;
  last_error: string | null;
  last_error_ts: string | null;
}

export interface AdapterHealthRow {
  adapter_id: string;
  broker: string;
  connected: boolean;
  last_heartbeat_ts: string | null;
  heartbeat_age_sec: number;
  health_status: string;
  disconnect_count_24h: number;
  last_error_code: string | null;
  last_error_message: string | null;
}

export interface HaltRow {
  halt_level: string;
  entity: string;
  halt_reason: string | null;
  last_update_at: string;
}

export interface StrategyDiagnosticsRow {
  strategy_id: string;
  mode: string;
  health_status: string;
  last_heartbeat_ts: string | null;
  heartbeat_age_sec: number | null;
  last_decision_code: string | null;
  last_seen_bar_ts: string | null;
  bar_age_sec: number | null;
  bars_processed: number | null;
  symbol_freshness: Record<string, string> | null;
  intents_submitted: number | null;
  intents_denied: number | null;
  consecutive_denials: number | null;
  ib_farm_status: Record<string, string> | null;
  last_error: string | null;
  last_error_ts: string | null;
}

export interface DailyClassificationRow {
  strategy_id: string;
  family_id: string | null;
  bars: number | null;
  trades: number | null;
  denials: number | null;
  last_bar_ts: string | null;
  last_decision_code: string | null;
  classification: 'ACTIVE' | 'NORMAL_QUIET' | 'BLOCKED' | 'DEAD' | 'BROKER_DOWN';
  daily_realized_r: number | null;
}

export interface EquityCurvePoint {
  trade_date: string;
  daily_realized_r: number;
  cumulative_r: number;
}

export interface DailyPnlPoint {
  trade_date: string;
  daily_realized_r: number;
}

export interface EnvData {
  mode: string;
  account_id: string;
  ib_port: number;
}

// API response types

export interface SystemPnlSummary {
  system: SystemId;
  daily_realized_r: number;
  daily_realized_usd: number;
  heat_r: number;
  active_heat_cap_R: number | null;
  active_config_status: 'fresh' | 'stale' | 'missing';
  active_config_warnings: string[];
  filled_entries: number;
  strategy_count: number;
  healthy_count: number;
}

export interface LiveBatchResponse {
  portfolio: PortfolioData;
  strategies: StrategyData[];
  positions: PositionRow[];
  trades: TradeRow[];
  orders: OrderRow[];
  health: HealthData;
  systemPnl: SystemPnlSummary[];
  queue: QueueSummary;
  serverTime: string;
}

export interface ChartBatchResponse {
  equityCurve: EquityCurvePoint[];
  dailyPnl: DailyPnlPoint[];
  serverTime: string;
}
