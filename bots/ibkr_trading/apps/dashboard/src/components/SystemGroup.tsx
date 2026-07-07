'use client';
import { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import {
  StrategyData,
  SystemPnlSummary,
  SYSTEM_CONFIG,
  type SystemId,
} from '@/lib/types';
import { fmtR } from '@/lib/formatters';
import { StrategyCard } from './StrategyCard';
import { cn } from '@/lib/utils';

interface Props {
  system: SystemId;
  strategies: StrategyData[];
  systemPnl?: SystemPnlSummary;
}

export function SystemGroup({ system, strategies, systemPnl }: Props) {
  const [collapsed, setCollapsed] = useState(false);
  const cfg = SYSTEM_CONFIG[system];

  const dailyR = systemPnl?.daily_realized_r ?? strategies.reduce((s, st) => s + st.daily_realized_r, 0);
  const heatR = systemPnl?.heat_r ?? strategies.reduce((s, st) => s + st.heat_r, 0);
  const healthyCount = systemPnl?.healthy_count ?? strategies.filter(s => s.health_status === 'OK').length;
  const totalCount = strategies.length;
  const filledEntries = systemPnl?.filled_entries ?? strategies.reduce((s, st) => s + st.filled_entries, 0);
  const groupHeatCap = systemPnl?.active_heat_cap_R ?? null;
  const hasHeatCap = groupHeatCap !== null && groupHeatCap > 0;
  const heatPct = hasHeatCap ? Math.min(100, (heatR / groupHeatCap) * 100) : 0;
  const configWarnings = systemPnl?.active_config_warnings ?? [];

  return (
    <div
      id={`system-${system}`}
      className={cn('rounded-lg border-l-4 border border-gray-800 bg-[#0d0f13]', cfg.color)}
    >
      {/* Header */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="w-full flex items-center justify-between gap-4 px-4 py-3 hover:bg-gray-900/40 transition-colors"
      >
        <div className="flex items-center gap-3">
          {collapsed ? (
            <ChevronRight className="h-4 w-4 text-gray-500 flex-shrink-0" />
          ) : (
            <ChevronDown className="h-4 w-4 text-gray-500 flex-shrink-0" />
          )}
          <span className={cn('font-mono font-bold text-sm tracking-wider', cfg.textColor)}>
            {cfg.shortLabel}
          </span>
          <span className="text-xs text-gray-600 font-mono">
            {totalCount} strat{totalCount !== 1 ? 's' : ''}
          </span>
        </div>

        <div className="flex items-center gap-6 text-xs font-mono">
          {/* Daily P&L */}
          <span className={cn('font-semibold', dailyR >= 0 ? 'text-green-400' : 'text-red-400')}>
            {fmtR(dailyR)}
          </span>

          {/* Heat mini-bar */}
          <div className="flex items-center gap-2 w-32">
            <span
              className={cn(configWarnings.length > 0 ? 'text-amber-400' : 'text-gray-500')}
              title={configWarnings.join('; ') || undefined}
            >
              {heatR.toFixed(1)}R{hasHeatCap ? `/${groupHeatCap.toFixed(1)}R` : ''}
            </span>
            <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
              <div
                className={cn('h-full rounded-full transition-all', cfg.dotColor)}
                style={{ width: `${heatPct}%` }}
              />
            </div>
          </div>

          {/* Entries */}
          <span className="text-gray-400">{filledEntries} entries</span>

          {/* Health */}
          <span className={cn(
            healthyCount === totalCount ? 'text-green-500' : 'text-amber-400'
          )}>
            {healthyCount}/{totalCount} OK
          </span>

          {configWarnings.length > 0 && (
            <span className="text-amber-400" title={configWarnings.join('; ')}>
              CONFIG
            </span>
          )}
        </div>
      </button>

      {/* Strategy cards */}
      {!collapsed && (
        <div className="px-4 pb-4 pt-1 grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-3">
          {strategies.map(s => (
            <StrategyCard key={s.strategy_id} strategy={s} />
          ))}
        </div>
      )}
    </div>
  );
}
