"use client";

import { useCallback, useEffect, useState } from "react";
import type { LiveBatchResponse, ChartBatchResponse } from "@/lib/types";
import { PortfolioHeader } from "@/components/PortfolioHeader";
import { StrategyGrid } from "@/components/StrategyGrid";
import { PositionsTable } from "@/components/PositionsTable";
import { TradesTable } from "@/components/TradesTable";
import { EquityCurve } from "@/components/EquityCurve";
import { DailyPnlBars } from "@/components/DailyPnlBars";
import { SystemHealth } from "@/components/SystemHealth";
import { SafetyEvents } from "@/components/SafetyEvents";
import { RefreshIndicator } from "@/components/RefreshIndicator";
import { Skeleton } from "@/components/ui/skeleton";

const LIVE_INTERVAL = 30_000; // 30s
const CHART_INTERVAL = 300_000; // 5 min

export default function DashboardPage() {
  const [live, setLive] = useState<LiveBatchResponse | null>(null);
  const [charts, setCharts] = useState<ChartBatchResponse | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchLive = useCallback(async () => {
    try {
      const res = await fetch("/api/live");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: LiveBatchResponse = await res.json();
      setLive(data);
      setLastUpdate(new Date());
      setError(null);

      // Dynamic tab title
      const pnl = data.portfolio.daily_pnl_usd;
      const sign = pnl >= 0 ? "+" : "";
      document.title = `${sign}$${Math.abs(pnl).toFixed(0)} | Crypto Trader`;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Fetch failed");
    }
  }, []);

  const fetchCharts = useCallback(async () => {
    try {
      const res = await fetch("/api/charts");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: ChartBatchResponse = await res.json();
      setCharts(data);
    } catch {
      // Charts are non-critical — silently retry
    }
  }, []);

  // Initial load + polling
  useEffect(() => {
    fetchLive();
    fetchCharts();

    const liveTimer = setInterval(fetchLive, LIVE_INTERVAL);
    const chartTimer = setInterval(fetchCharts, CHART_INTERVAL);

    return () => {
      clearInterval(liveTimer);
      clearInterval(chartTimer);
    };
  }, [fetchLive, fetchCharts]);

  // Keyboard refresh (R key)
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "r" && !e.ctrlKey && !e.metaKey) {
        fetchLive();
        fetchCharts();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [fetchLive, fetchCharts]);

  if (error && !live) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="text-accent-red text-lg mb-2">Connection Error</p>
          <p className="text-zinc-500 text-sm">{error}</p>
          <button
            onClick={fetchLive}
            className="mt-4 px-4 py-2 bg-surface-2 rounded border border-surface-3 text-sm text-zinc-300 hover:bg-surface-3"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <main className="min-h-screen p-6 max-w-7xl mx-auto space-y-5">
      {/* Header bar */}
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-semibold text-zinc-200">Crypto Trader</h1>
        <RefreshIndicator intervalSec={30} lastUpdate={lastUpdate} />
      </div>

      {/* Portfolio overview */}
      {live ? (
        <PortfolioHeader data={live.portfolio} />
      ) : (
        <Skeleton className="h-24 w-full" />
      )}

      {/* Strategy cards */}
      {live ? (
        <StrategyGrid strategies={live.strategies} />
      ) : (
        <div className="flex gap-4">
          <Skeleton className="h-32 flex-1" />
          <Skeleton className="h-32 flex-1" />
          <Skeleton className="h-32 flex-1" />
        </div>
      )}

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {charts ? (
          <>
            <EquityCurve data={charts.equity_curve} />
            <DailyPnlBars data={charts.daily_pnl} />
          </>
        ) : (
          <>
            <Skeleton className="h-72" />
            <Skeleton className="h-72" />
          </>
        )}
      </div>

      {/* Positions */}
      {live ? (
        <PositionsTable
          positions={live.positions}
          exchangePositions={live.exchange_positions}
          strategyAllocations={live.strategy_position_allocations}
          allocationResiduals={live.allocation_residuals}
        />
      ) : (
        <Skeleton className="h-48 w-full" />
      )}

      {/* Trades + Health side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          {live ? (
            <TradesTable trades={live.trades} />
          ) : (
            <Skeleton className="h-64 w-full" />
          )}
        </div>
        <div className="space-y-4">
          {live ? (
            <>
              <SystemHealth data={live.health} safetyEvents={live.safety_events} />
              <SafetyEvents events={live.safety_events} />
            </>
          ) : (
            <Skeleton className="h-64 w-full" />
          )}
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div className="fixed bottom-4 right-4 bg-accent-red/20 border border-accent-red/40 rounded-lg px-4 py-2 text-sm text-accent-red">
          {error}
        </div>
      )}
    </main>
  );
}
