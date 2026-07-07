"use client";

import type { StrategyData } from "@/lib/types";
import { StrategyCard } from "@/components/StrategyCard";

const ALL_STRATEGIES = ["momentum", "trend", "breakout"];

export function StrategyGrid({ strategies }: { strategies: StrategyData[] }) {
  // Ensure all 3 strategies are shown even if no trades today
  const byId = new Map(strategies.map((s) => [s.strategy_id, s]));
  const rows = ALL_STRATEGIES.map(
    (id) =>
      byId.get(id) ?? {
        strategy_id: id,
        trades_today: 0,
        wins_today: 0,
        losses_today: 0,
        daily_pnl_r: 0,
        daily_pnl_usd: 0,
      }
  );

  return (
    <div className="flex gap-4">
      {rows.map((s) => (
        <StrategyCard key={s.strategy_id} data={s} />
      ))}
    </div>
  );
}
