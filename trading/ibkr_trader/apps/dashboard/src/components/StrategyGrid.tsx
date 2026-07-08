'use client';
import { StrategyData, SystemPnlSummary, SYSTEM_ORDER, groupBySystem } from '@/lib/types';
import { SystemGroup } from './SystemGroup';
import { Skeleton } from '@/components/ui/skeleton';

interface Props {
  strategies: StrategyData[] | null;
  systemPnl?: SystemPnlSummary[] | null;
}

export function StrategyGrid({ strategies, systemPnl }: Props) {
  if (strategies == null) {
    // 3 group-shaped skeletons
    return (
      <div className="space-y-3">
        {Array.from({ length: 3 }).map((_, i) => (
          <Skeleton key={i} className="h-32 rounded-lg" />
        ))}
      </div>
    );
  }

  const grouped = groupBySystem(strategies);
  const pnlMap = new Map((systemPnl ?? []).map(sp => [sp.system, sp]));

  return (
    <div className="space-y-3">
      {SYSTEM_ORDER.map(sys => {
        const strats = grouped.get(sys) ?? [];
        if (strats.length === 0) return null;
        return (
          <SystemGroup
            key={sys}
            system={sys}
            strategies={strats}
            systemPnl={pnlMap.get(sys)}
          />
        );
      })}
    </div>
  );
}
