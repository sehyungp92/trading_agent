export interface PortfolioData {
  equity: number;
  daily_pnl_usd: number;
  unrealized_pnl: number;
  heat_r: number;
  open_positions: number;
}

export interface StrategyData {
  strategy_id: string;
  trades_today: number;
  wins_today: number;
  losses_today: number;
  daily_pnl_r: number;
  daily_pnl_usd: number;
}

export interface PositionRow {
  strategy_id: string;
  symbol: string;
  direction: string;
  qty: number;
  avg_entry: number;
  unrealized_pnl: number;
  risk_r: number;
  stop_price: number | null;
  entry_time: string | null;
  stale_minutes: number;
  age_minutes: number;
}

export interface ExchangePositionRow {
  symbol: string;
  direction: string;
  qty: number;
  avg_entry: number;
  unrealized_pnl: number;
  liquidation_price: number | null;
  observed_at: string;
}

export interface StrategyPositionAllocationRow {
  position_instance_id: string;
  strategy_id: string;
  symbol: string;
  direction: string;
  allocated_qty: number;
  avg_entry: number;
  risk_r: number;
  entry_time: string | null;
  status: string;
  confidence: string;
  source: string;
  entry_order_ids: string[];
  entry_fill_ids: string[];
  exit_order_ids: string[];
  exit_fill_ids: string[];
}

export interface AllocationResidualRow {
  symbol: string;
  direction: string;
  net_exchange_qty: number;
  allocated_qty: number;
  unallocated_qty: number;
  unknown_allocation: boolean;
  status: string;
}

export interface TradeRow {
  trade_id: string;
  strategy_id: string;
  symbol: string;
  direction: string;
  entry_price: number;
  exit_price: number;
  entry_time: string;
  exit_time: string;
  pnl: number;
  net_pnl: number;
  r_multiple: number | null;
  exit_reason: string | null;
  setup_grade: string | null;
  duration_minutes: number;
}

export interface HealthData {
  assessment: string;
  uptime_sec: number | null;
  alerts: string[];
  timestamp: string;
  postgres_sink?: {
    enabled?: boolean;
    worker_alive?: boolean;
    queue_depth?: number;
    queue_capacity?: number;
    jobs_dropped?: number;
    write_failures?: number;
    last_error?: string;
  } | null;
}

export interface SafetyEventRow {
  event_id: string;
  event_type: string;
  timestamp: string | null;
  severity: string | null;
  strategy_id: string | null;
  symbol: string | null;
  status: string | null;
  description: string | null;
}

export interface EquityCurvePoint {
  ts: string;
  equity: number;
}

export interface DailyPnlPoint {
  trade_date: string;
  net_pnl: number;
  total_trades: number;
}

export interface LiveBatchResponse {
  portfolio: PortfolioData;
  strategies: StrategyData[];
  positions: PositionRow[];
  exchange_positions: ExchangePositionRow[];
  strategy_position_allocations: StrategyPositionAllocationRow[];
  allocation_residuals: AllocationResidualRow[];
  trades: TradeRow[];
  health: HealthData | null;
  safety_events: SafetyEventRow[];
}

export interface ChartBatchResponse {
  equity_curve: EquityCurvePoint[];
  daily_pnl: DailyPnlPoint[];
}
