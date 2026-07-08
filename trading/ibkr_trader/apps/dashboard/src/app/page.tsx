'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  PortfolioData, StrategyData, PositionRow, TradeRow, OrderRow,
  HealthData, EquityCurvePoint, DailyPnlPoint, EnvData,
  LiveBatchResponse, ChartBatchResponse, SystemPnlSummary,
} from '@/lib/types';
import { cn } from '@/lib/utils';
import { PortfolioHeader } from '@/components/PortfolioHeader';
import { StrategyGrid } from '@/components/StrategyGrid';
import { PositionsTable } from '@/components/PositionsTable';
import { TradesTable } from '@/components/TradesTable';
import { OrdersTable } from '@/components/OrdersTable';
import { SystemHealth } from '@/components/SystemHealth';
import { EquityCurve } from '@/components/EquityCurve';
import { DailyPnlBars } from '@/components/DailyPnlBars';
import { RefreshIndicator } from '@/components/RefreshIndicator';
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts';

const LIVE_INTERVAL_MS = 30_000;
const CHART_INTERVAL_MS = 5 * 60_000;

async function fetchJson<T>(url: string): Promise<T | null> {
  try {
    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) return null;
    return res.json() as Promise<T>;
  } catch {
    return null;
  }
}

export default function Dashboard() {
  const [portfolio, setPortfolio] = useState<PortfolioData | null>(null);
  const [strategies, setStrategies] = useState<StrategyData[] | null>(null);
  const [positions, setPositions] = useState<PositionRow[] | null>(null);
  const [trades, setTrades] = useState<TradeRow[] | null>(null);
  const [orders, setOrders] = useState<OrderRow[] | null>(null);
  const [health, setHealth] = useState<HealthData | null>(null);
  const [equityCurve, setEquityCurve] = useState<EquityCurvePoint[] | null>(null);
  const [dailyPnl, setDailyPnl] = useState<DailyPnlPoint[] | null>(null);
  const [envData, setEnvData] = useState<EnvData | null>(null);
  const [systemPnl, setSystemPnl] = useState<SystemPnlSummary[] | null>(null);
  const [serverTime, setServerTime] = useState<string | null>(null);

  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [nextRefreshIn, setNextRefreshIn] = useState(LIVE_INTERVAL_MS / 1000);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const liveTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const chartTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const countdownRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const nextLiveRef = useRef<number>(Date.now() + LIVE_INTERVAL_MS);

  const fetchLive = useCallback(async () => {
    setIsRefreshing(true);
    const data = await fetchJson<LiveBatchResponse>('/api/live');

    if (data) {
      setPortfolio(data.portfolio);
      setStrategies(data.strategies);
      setPositions(data.positions);
      setTrades(data.trades);
      setOrders(data.orders);
      setHealth(data.health);
      setSystemPnl(data.systemPnl);
      setServerTime(data.serverTime);
    } else {
      // Fallback to defaults on error
      setPortfolio(p => p ?? {
        daily_realized_r: 0,
        daily_realized_usd: 0,
        portfolio_open_risk_r: 0,
        unrealized_pnl: 0,
        halted: false,
        halt_reason: null,
        heat_r: 0,
        heat_cap_R: null,
        portfolio_daily_stop_R: null,
        portfolio_weekly_stop_R: null,
        global_standdown: null,
        active_config_status: 'missing',
        active_config_warnings: ['missing account active config'],
      });
      setStrategies(s => s ?? []);
      setPositions(p => p ?? []);
      setTrades(t => t ?? []);
      setOrders(o => o ?? []);
      setHealth(h => h ?? { strategies: [], adapters: [], halts: [], evidence: null });
      setSystemPnl(sp => sp ?? []);
    }

    setLastUpdate(new Date());
    setIsRefreshing(false);
    nextLiveRef.current = Date.now() + LIVE_INTERVAL_MS;
  }, []);

  const fetchCharts = useCallback(async () => {
    const data = await fetchJson<ChartBatchResponse>('/api/charts');
    if (data) {
      setEquityCurve(data.equityCurve);
      setDailyPnl(data.dailyPnl);
    } else {
      setEquityCurve(ec => ec ?? []);
      setDailyPnl(dp => dp ?? []);
    }
  }, []);

  // Force refresh for keyboard shortcut
  const handleForceRefresh = useCallback(() => {
    fetchLive();
    fetchCharts();
  }, [fetchLive, fetchCharts]);

  useKeyboardShortcuts({ onRefresh: handleForceRefresh });

  // Dynamic browser tab title
  useEffect(() => {
    const totalR = portfolio?.daily_realized_r ?? 0;
    const sign = totalR >= 0 ? '+' : '';
    document.title = `${sign}${totalR.toFixed(2)}R | Trading Monitor`;
  }, [portfolio?.daily_realized_r]);

  useEffect(() => {
    fetchJson<EnvData>('/api/env').then(d => { if (d) setEnvData(d); });
  }, []);

  useEffect(() => {
    // Initial fetch
    fetchLive();
    fetchCharts();

    // Live polling
    liveTimerRef.current = setInterval(fetchLive, LIVE_INTERVAL_MS);
    // Chart polling
    chartTimerRef.current = setInterval(fetchCharts, CHART_INTERVAL_MS);

    // Countdown ticker
    countdownRef.current = setInterval(() => {
      const remaining = Math.max(0, Math.round((nextLiveRef.current - Date.now()) / 1000));
      setNextRefreshIn(remaining);
    }, 1000);

    return () => {
      if (liveTimerRef.current) clearInterval(liveTimerRef.current);
      if (chartTimerRef.current) clearInterval(chartTimerRef.current);
      if (countdownRef.current) clearInterval(countdownRef.current);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <main className="p-4 space-y-4 max-w-[1800px] mx-auto pb-12">
      {/* Header row */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-bold font-mono text-gray-200 tracking-wider">
            TRADING MONITOR
          </h1>
          {envData && (
            <span className={cn(
              'text-xs font-mono font-bold px-2 py-0.5 rounded border tracking-wider',
              envData.mode === 'live'
                ? 'text-green-300 border-green-700 bg-green-950/50 animate-pulse'
                : envData.mode === 'paper'
                ? 'text-amber-300 border-amber-700 bg-amber-950/40'
                : envData.mode === 'backtest'
                ? 'text-blue-300 border-blue-700 bg-blue-950/40'
                : 'text-gray-500 border-gray-700 bg-gray-900/40'
            )}>
              {envData.mode === 'live'
                ? '\u25cf LIVE'
                : envData.mode === 'paper'
                ? '\u25c6 PAPER'
                : envData.mode.toUpperCase()}
            </span>
          )}
          {envData?.account_id && (
            <span className="text-xs font-mono text-gray-600">{envData.account_id}</span>
          )}
        </div>
        <span className="text-xs text-gray-600 font-mono">
          {serverTime
            ? new Date(serverTime).toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' })
            : new Date().toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' })
          }
        </span>
      </div>

      {/* Portfolio P&L + System Health side by side */}
      <div className="grid xl:grid-cols-2 gap-4">
        <PortfolioHeader portfolio={portfolio} health={health} systemPnl={systemPnl} />
        <SystemHealth health={health} />
      </div>

      <StrategyGrid strategies={strategies} systemPnl={systemPnl} />

      {/* Charts left | Tables right */}
      <div className="grid xl:grid-cols-2 gap-4">
        <div className="flex flex-col gap-4">
          <EquityCurve data={equityCurve} />
          <DailyPnlBars data={dailyPnl} />
        </div>
        <div className="flex flex-col gap-4 h-full">
          <PositionsTable positions={positions} />
          <OrdersTable orders={orders} />
          <TradesTable trades={trades} />
        </div>
      </div>

      <RefreshIndicator
        lastUpdate={lastUpdate}
        nextRefreshIn={nextRefreshIn}
        isRefreshing={isRefreshing}
        serverTime={serverTime}
      />
    </main>
  );
}
