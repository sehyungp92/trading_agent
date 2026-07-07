"use client";

import type { TradeRow } from "@/lib/types";
import { fmtR, fmtUSD, fmtDate, colorClass } from "@/lib/formatters";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export function TradesTable({ trades }: { trades: TradeRow[] }) {
  if (trades.length === 0) {
    return (
      <Card>
        <CardHeader><CardTitle>Today&apos;s Trades</CardTitle></CardHeader>
        <p className="text-sm text-zinc-500 text-center py-6">No trades today</p>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Today&apos;s Trades</CardTitle>
        <span className="text-xs text-zinc-500">{trades.length} trades</span>
      </CardHeader>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs text-zinc-500 uppercase border-b border-surface-3">
              <th className="text-left py-2 pr-3">Time</th>
              <th className="text-left py-2 pr-3">Strategy</th>
              <th className="text-left py-2 pr-3">Symbol</th>
              <th className="text-left py-2 pr-3">Dir</th>
              <th className="text-right py-2 pr-3">R</th>
              <th className="text-right py-2 pr-3">P&L</th>
              <th className="text-left py-2 pr-3">Grade</th>
              <th className="text-left py-2 pr-3">Exit</th>
              <th className="text-right py-2">Duration</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((t) => (
              <tr
                key={t.trade_id}
                className="border-b border-surface-3/50 hover:bg-surface-2/50"
              >
                <td className="py-2 pr-3 text-zinc-400 whitespace-nowrap">
                  {fmtDate(t.exit_time)}
                </td>
                <td className="py-2 pr-3 text-zinc-400">{t.strategy_id}</td>
                <td className="py-2 pr-3 font-mono font-medium text-zinc-200">
                  {t.symbol}
                </td>
                <td className="py-2 pr-3">
                  <Badge variant={t.direction === "long" ? "green" : "red"}>
                    {t.direction}
                  </Badge>
                </td>
                <td className={`py-2 pr-3 text-right font-mono ${colorClass(t.r_multiple ?? 0)}`}>
                  {fmtR(t.r_multiple)}
                </td>
                <td className={`py-2 pr-3 text-right font-mono ${colorClass(t.net_pnl)}`}>
                  {fmtUSD(t.net_pnl)}
                </td>
                <td className="py-2 pr-3">
                  {t.setup_grade ? (
                    <Badge variant={t.setup_grade === "A" ? "blue" : "neutral"}>
                      {t.setup_grade}
                    </Badge>
                  ) : (
                    "—"
                  )}
                </td>
                <td className="py-2 pr-3 text-zinc-400 text-xs">
                  {t.exit_reason ?? "—"}
                </td>
                <td className="py-2 text-right text-zinc-400">
                  {t.duration_minutes > 0
                    ? `${Math.round(t.duration_minutes)}m`
                    : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
