"use client";

import type { StrategyData } from "@/lib/types";
import { fmtR, fmtUSD, colorClass } from "@/lib/formatters";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const STRATEGY_LABELS: Record<string, string> = {
  momentum: "Momentum Pullback",
  trend: "Institutional Anchor",
  breakout: "Volume Breakout",
};

export function StrategyCard({ data }: { data: StrategyData }) {
  const label = STRATEGY_LABELS[data.strategy_id] ?? data.strategy_id;
  const winRate =
    data.trades_today > 0
      ? ((data.wins_today / data.trades_today) * 100).toFixed(0) + "%"
      : "—";

  return (
    <Card className="flex-1 min-w-[220px]">
      <CardHeader>
        <CardTitle>{label}</CardTitle>
        <Badge variant={data.daily_pnl_r >= 0 ? "green" : "red"}>
          {fmtR(data.daily_pnl_r)}
        </Badge>
      </CardHeader>

      <div className="grid grid-cols-3 gap-3 text-center">
        <div>
          <p className="text-xs text-zinc-500">Trades</p>
          <p className="font-mono text-sm text-zinc-200">{data.trades_today}</p>
        </div>
        <div>
          <p className="text-xs text-zinc-500">W / L</p>
          <p className="font-mono text-sm text-zinc-200">
            {data.wins_today} / {data.losses_today}
          </p>
        </div>
        <div>
          <p className="text-xs text-zinc-500">Win Rate</p>
          <p className="font-mono text-sm text-zinc-200">{winRate}</p>
        </div>
      </div>

      <div className="mt-3 pt-3 border-t border-surface-3 text-center">
        <p className="text-xs text-zinc-500">Daily P&L</p>
        <p className={`font-mono text-sm font-medium ${colorClass(data.daily_pnl_usd)}`}>
          {fmtUSD(data.daily_pnl_usd)}
        </p>
      </div>
    </Card>
  );
}
