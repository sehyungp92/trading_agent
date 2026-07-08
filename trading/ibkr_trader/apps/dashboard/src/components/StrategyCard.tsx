'use client';
import { StrategyData } from '@/lib/types';
import { fmtR, fmtAge, toPercent } from '@/lib/formatters';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { StrategyDiagnostics } from '@/components/StrategyDiagnostics';

interface Props {
  strategy: StrategyData;
}

function statusVariant(status: string): 'success' | 'danger' | 'warning' | 'default' {
  if (status === 'OK') return 'success';
  if (status === 'STALE') return 'warning';
  if (status === 'HALTED') return 'danger';
  if (status === 'STAND_DOWN') return 'default';
  return 'default';
}

export function StrategyCard({ strategy }: Props) {
  const maxHeat = strategy.active_max_heat_R;
  const hasHeatCap = maxHeat !== null && maxHeat > 0;
  const heatPct = hasHeatCap ? toPercent(strategy.heat_r, maxHeat) : 0;
  const dailyStop = strategy.active_max_daily_loss_R !== null
    ? strategy.active_max_daily_loss_R - Math.abs(strategy.daily_pnl_r)
    : null;
  const dailyStopLow = dailyStop !== null && dailyStop < 0.5;
  const dailyStopBreached = dailyStop !== null && dailyStop < 0;
  const riskPct = strategy.active_risk_per_trade !== null
    ? `${(strategy.active_risk_per_trade * 100).toFixed(2)}% risk`
    : 'Risk config missing';

  const heatColor =
    heatPct > 95 ? 'bg-red-500' : heatPct > 80 ? 'bg-amber-500' : 'bg-green-500';

  return (
    <div className="rounded-lg border border-gray-800 bg-[#111318] p-4 space-y-3">
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="font-mono font-semibold text-sm text-gray-100 leading-tight">
            {strategy.strategy_id}
          </p>
          <p className="text-xs text-gray-500 font-mono mt-0.5">
            {riskPct}
          </p>
        </div>
        <Badge variant={statusVariant(strategy.health_status)}>
          {strategy.health_status}
        </Badge>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div>
          <p className="text-xs text-gray-500 font-mono">Daily P&amp;L</p>
          <p
            className={cn(
              'text-lg font-bold font-mono',
              strategy.daily_realized_r >= 0 ? 'text-green-400' : 'text-red-400',
            )}
          >
            {fmtR(strategy.daily_realized_r)}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500 font-mono">Entries</p>
          <p className="text-lg font-bold font-mono text-gray-200">{strategy.filled_entries}</p>
        </div>
      </div>

      <div className="space-y-1">
        <div className="flex justify-between text-xs font-mono">
          <span className="text-gray-500">Heat</span>
          <span className={heatPct > 80 ? 'text-amber-400' : 'text-gray-400'}>
            {strategy.heat_r.toFixed(2)}R / {hasHeatCap ? `${maxHeat.toFixed(2)}R` : 'No cap'}
          </span>
        </div>
        <Progress value={heatPct} indicatorClassName={heatColor} className="h-1.5" />
      </div>

      <div className="flex justify-between items-center text-xs font-mono">
        <span className="text-gray-500">Stop remaining</span>
        <span
          className={cn(
            dailyStopLow ? 'text-amber-400' : 'text-gray-400',
            dailyStopBreached ? 'text-red-400' : '',
          )}
        >
          {dailyStop !== null ? `${dailyStop.toFixed(2)}R` : '\u2014'}
        </span>
      </div>

      {strategy.active_config_warnings.length > 0 && (
        <p className="text-xs text-amber-400 font-mono truncate" title={strategy.active_config_warnings.join('; ')}>
          {strategy.active_config_warnings[0]}
        </p>
      )}

      <div className="text-xs text-gray-600 font-mono border-t border-gray-800 pt-2">
        Heartbeat: {fmtAge(strategy.heartbeat_age_sec)} ago
        {strategy.last_error && (
          <p className="text-amber-500 truncate mt-0.5" title={strategy.last_error}>
            Error: {strategy.last_error}
          </p>
        )}
      </div>

      <StrategyDiagnostics strategyId={strategy.strategy_id} />
    </div>
  );
}
