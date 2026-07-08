'use client';
import { PortfolioData, HealthData, SystemPnlSummary, SYSTEM_CONFIG } from '@/lib/types';
import { fmtR, fmtUSD, toPercent } from '@/lib/formatters';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { AlertTriangle, Wifi, WifiOff } from 'lucide-react';
import { cn } from '@/lib/utils';

interface Props {
  portfolio: PortfolioData | null;
  health: HealthData | null;
  systemPnl?: SystemPnlSummary[] | null;
}

function formatGateLimitR(value: number | null): string {
  return value != null && Number.isFinite(value) ? `${value.toFixed(2)}R` : '--';
}

export function PortfolioHeader({ portfolio, health, systemPnl }: Props) {
  const r = portfolio?.daily_realized_r ?? 0;
  const usd = portfolio?.daily_realized_usd ?? 0;
  const heatR = portfolio?.heat_r ?? 0;
  const heatCapR = portfolio?.heat_cap_R ?? null;
  const hasHeatCap = heatCapR != null && Number.isFinite(heatCapR) && heatCapR > 0;
  const heatPct = hasHeatCap ? toPercent(heatR, heatCapR) : 0;
  const heatCapLabel = hasHeatCap ? `${Number(heatCapR).toFixed(2)}R` : '--';
  const dailyStopR = portfolio?.portfolio_daily_stop_R ?? null;
  const weeklyStopR = portfolio?.portfolio_weekly_stop_R ?? null;
  const globalStanddown = portfolio?.global_standdown ?? null;
  const dailyStopLabel = formatGateLimitR(dailyStopR);
  const weeklyStopLabel = formatGateLimitR(weeklyStopR);
  const standdownLabel = globalStanddown == null ? '--' : globalStanddown ? 'ACTIVE' : 'CLEAR';
  const standdownBadge =
    globalStanddown == null ? 'warning' : globalStanddown ? 'danger' : 'success';
  const halted = portfolio?.halted ?? false;
  const activeConfigWarnings = portfolio?.active_config_warnings ?? [];
  const halts = health?.halts ?? [];
  const adapters = health?.adapters ?? [];

  const heatColor =
    heatPct > 90 ? 'bg-red-500' : heatPct > 60 ? 'bg-amber-500' : 'bg-green-500';

  return (
    <div className="rounded-lg border border-gray-800 bg-[#111318] p-4 space-y-3">
      {/* Halt banner */}
      {(halted || halts.length > 0) && (
        <div className="flex items-center gap-2 rounded-md bg-red-950 border border-red-800 px-3 py-2">
          <AlertTriangle className="h-4 w-4 text-red-400 flex-shrink-0" />
          <span className="text-red-200 font-mono text-xs">
            {halts.length > 0
              ? halts.map(h => `${h.halt_level} [${h.entity}]: ${h.halt_reason}`).join(' | ')
              : `PORTFOLIO HALTED: ${portfolio?.halt_reason ?? 'unknown'}`}
          </span>
        </div>
      )}

      {activeConfigWarnings.length > 0 && (
        <div className="flex items-center gap-2 rounded-md bg-amber-950/60 border border-amber-800 px-3 py-2">
          <AlertTriangle className="h-4 w-4 text-amber-300 flex-shrink-0" />
          <span className="text-amber-100 font-mono text-xs">
            {activeConfigWarnings.join(' | ')}
          </span>
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {/* Zone 1: Daily P&L */}
        <div>
          <p className="text-xs text-gray-500 font-mono uppercase tracking-wider mb-1">Today P&amp;L</p>
          <p className={cn('text-3xl font-bold font-mono', r >= 0 ? 'text-green-400' : 'text-red-400')}>
            {fmtR(r)}
          </p>
          <p className={cn('text-sm font-mono mt-0.5', r >= 0 ? 'text-green-600' : 'text-red-600')}>
            {fmtUSD(usd)}
          </p>
        </div>

        {/* Zone 2: Heat gauge */}
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <p className="text-xs text-gray-500 font-mono uppercase tracking-wider">Portfolio Heat</p>
            <span className={cn('text-sm font-mono font-semibold',
              !hasHeatCap ? 'text-amber-300' :
              heatPct > 90 ? 'text-red-400' : heatPct > 60 ? 'text-amber-400' : 'text-green-400'
            )}>
              {heatR.toFixed(2)}R / {heatCapLabel}
            </span>
          </div>
          <Progress value={heatPct} indicatorClassName={heatColor} className="h-3" />
          <p className="text-xs text-gray-600 font-mono">{heatPct.toFixed(0)}% capacity</p>
        </div>

        {/* Zone 3: Broker adapter status */}
        <div>
          <p className="text-xs text-gray-500 font-mono uppercase tracking-wider mb-2">Broker Connections</p>
          {adapters.length === 0 ? (
            <span className="text-xs text-gray-600 font-mono">No adapters registered</span>
          ) : (
            <div className="flex flex-wrap gap-2">
              {adapters.map(a => (
                <div key={a.adapter_id} className="flex items-center gap-1.5">
                  {a.connected ? (
                    <Wifi className="h-3.5 w-3.5 text-green-400" />
                  ) : (
                    <WifiOff className="h-3.5 w-3.5 text-red-400" />
                  )}
                  <Badge variant={a.connected ? 'success' : 'danger'} className="text-xs">
                    {a.broker} {a.health_status}
                  </Badge>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Unrealized sub-line */}
      <div className="flex items-center gap-4 pt-1 border-t border-gray-800">
        <span className="text-xs text-gray-500 font-mono">
          Unrealized: <span className={(portfolio?.unrealized_pnl ?? 0) >= 0 ? 'text-green-500' : 'text-red-500'}>
            {fmtUSD(portfolio?.unrealized_pnl)}
          </span>
        </span>
        <span className="text-xs text-gray-500 font-mono">
          Open Risk: <span className="text-amber-400">{(portfolio?.portfolio_open_risk_r ?? 0).toFixed(2)}R</span>
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1 border-t border-gray-800">
        <div className="flex items-center justify-between gap-3">
          <span className="text-xs text-gray-500 font-mono uppercase tracking-wider">Daily Stop</span>
          <span className={cn('text-sm font-mono font-semibold', dailyStopR == null ? 'text-amber-300' : 'text-red-300')}>
            {dailyStopLabel}
          </span>
        </div>
        <div className="flex items-center justify-between gap-3">
          <span className="text-xs text-gray-500 font-mono uppercase tracking-wider">Weekly Stop</span>
          <span className={cn('text-sm font-mono font-semibold', weeklyStopR == null ? 'text-amber-300' : 'text-red-300')}>
            {weeklyStopLabel}
          </span>
        </div>
        <div className="flex items-center justify-between gap-3">
          <span className="text-xs text-gray-500 font-mono uppercase tracking-wider">Global Stand-down</span>
          <Badge variant={standdownBadge} className="text-xs">
            {standdownLabel}
          </Badge>
        </div>
      </div>

      {/* Per-system P&L badges */}
      {systemPnl && systemPnl.length > 0 && (
        <div className="flex items-center gap-3 pt-1">
          {systemPnl.map(sp => {
            const sysCfg = SYSTEM_CONFIG[sp.system];
            return (
              <span
                key={sp.system}
                className={cn(
                  'inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-mono font-semibold border border-gray-700',
                  sysCfg.bgColor
                )}
              >
                <span className={cn('w-2 h-2 rounded-full', sysCfg.dotColor)} />
                <span className={sysCfg.textColor}>{sysCfg.shortLabel}</span>
                <span className={sp.daily_realized_r >= 0 ? 'text-green-400' : 'text-red-400'}>
                  {fmtR(sp.daily_realized_r)}
                </span>
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}
