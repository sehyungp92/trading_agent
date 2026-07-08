"use client";

import type { ReactNode } from "react";
import type {
  AllocationResidualRow,
  ExchangePositionRow,
  PositionRow,
  StrategyPositionAllocationRow,
} from "@/lib/types";
import { fmtUSD, colorClass } from "@/lib/formatters";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export function PositionsTable({
  positions,
  exchangePositions = [],
  strategyAllocations = [],
  allocationResiduals = [],
}: {
  positions: PositionRow[];
  exchangePositions?: ExchangePositionRow[];
  strategyAllocations?: StrategyPositionAllocationRow[];
  allocationResiduals?: AllocationResidualRow[];
}) {
  if (positions.length === 0 && exchangePositions.length === 0 && strategyAllocations.length === 0) {
    return (
      <Card>
        <CardHeader><CardTitle>Position Exposure</CardTitle></CardHeader>
        <p className="text-sm text-zinc-500 text-center py-6">No open positions</p>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader><CardTitle>Position Exposure</CardTitle></CardHeader>
      <div className="space-y-5">
        <ExposureSection title="Net Exchange Exposure">
          {(exchangePositions.length > 0 ? exchangePositions : positions).map((p) => (
            <tr key={`net-${p.symbol}`} className="border-b border-surface-3/50">
              <td className="py-2 pr-4 font-mono font-medium text-zinc-200">{p.symbol}</td>
              <td className="py-2 pr-4"><DirectionBadge direction={p.direction} /></td>
              <td className="py-2 pr-4 text-right font-mono text-zinc-300">{Number(p.qty).toFixed(4)}</td>
              <td className="py-2 pr-4 text-right font-mono text-zinc-300">${Number(p.avg_entry).toLocaleString()}</td>
              <td className={`py-2 text-right font-mono ${colorClass(Number(p.unrealized_pnl ?? 0))}`}>
                {fmtUSD(Number(p.unrealized_pnl ?? 0))}
              </td>
            </tr>
          ))}
        </ExposureSection>

        <AllocationSection allocations={strategyAllocations} positions={positions} />

        {allocationResiduals.length > 0 && (
          <DriftSection residuals={allocationResiduals} />
        )}
      </div>
    </Card>
  );
}

function ExposureSection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="overflow-x-auto">
      <p className="text-xs uppercase text-zinc-500 mb-2">{title}</p>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-xs text-zinc-500 uppercase border-b border-surface-3">
            <th className="text-left py-2 pr-4">Symbol</th>
            <th className="text-left py-2 pr-4">Dir</th>
            <th className="text-right py-2 pr-4">Qty</th>
            <th className="text-right py-2 pr-4">Entry</th>
            <th className="text-right py-2">P/L</th>
          </tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

function AllocationSection({
  allocations,
  positions,
}: {
  allocations: StrategyPositionAllocationRow[];
  positions: PositionRow[];
}) {
  const rows = allocations.length > 0
    ? allocations
    : positions.map((p) => ({
        position_instance_id: `${p.strategy_id}-${p.symbol}`,
        strategy_id: p.strategy_id,
        symbol: p.symbol,
        direction: p.direction,
        allocated_qty: p.qty,
        avg_entry: p.avg_entry,
        risk_r: p.risk_r,
        entry_time: p.entry_time,
        status: "OPEN",
        confidence: "legacy",
        source: "positions",
        entry_order_ids: [],
        entry_fill_ids: [],
        exit_order_ids: [],
        exit_fill_ids: [],
      }));

  return (
    <div className="overflow-x-auto">
      <p className="text-xs uppercase text-zinc-500 mb-2">Strategy Allocations</p>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-xs text-zinc-500 uppercase border-b border-surface-3">
            <th className="text-left py-2 pr-4">Symbol</th>
            <th className="text-left py-2 pr-4">Strategy</th>
            <th className="text-left py-2 pr-4">Dir</th>
            <th className="text-right py-2 pr-4">Qty</th>
            <th className="text-right py-2 pr-4">Risk R</th>
            <th className="text-right py-2">Source</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((p) => (
            <tr key={p.position_instance_id} className="border-b border-surface-3/50">
              <td className="py-2 pr-4 font-mono font-medium text-zinc-200">{p.symbol}</td>
              <td className="py-2 pr-4 text-zinc-400">{p.strategy_id}</td>
              <td className="py-2 pr-4"><DirectionBadge direction={p.direction} /></td>
              <td className="py-2 pr-4 text-right font-mono text-zinc-300">{Number(p.allocated_qty).toFixed(4)}</td>
              <td className="py-2 pr-4 text-right font-mono text-zinc-300">{Number(p.risk_r).toFixed(2)}R</td>
              <td className="py-2 text-right text-zinc-400">{p.source} / {p.confidence}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DriftSection({ residuals }: { residuals: AllocationResidualRow[] }) {
  return (
    <div className="overflow-x-auto">
      <p className="text-xs uppercase text-zinc-500 mb-2">Unallocated / Drift</p>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-xs text-zinc-500 uppercase border-b border-surface-3">
            <th className="text-left py-2 pr-4">Symbol</th>
            <th className="text-left py-2 pr-4">Dir</th>
            <th className="text-right py-2 pr-4">Net</th>
            <th className="text-right py-2 pr-4">Allocated</th>
            <th className="text-right py-2">Residual</th>
          </tr>
        </thead>
        <tbody>
          {residuals.map((row) => (
            <tr key={`drift-${row.symbol}-${row.direction}`} className="border-b border-surface-3/50">
              <td className="py-2 pr-4 font-mono font-medium text-zinc-200">{row.symbol}</td>
              <td className="py-2 pr-4"><DirectionBadge direction={row.direction} /></td>
              <td className="py-2 pr-4 text-right font-mono text-zinc-300">{row.net_exchange_qty.toFixed(4)}</td>
              <td className="py-2 pr-4 text-right font-mono text-zinc-300">{row.allocated_qty.toFixed(4)}</td>
              <td className="py-2 text-right font-mono text-accent-red">{row.unallocated_qty.toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DirectionBadge({ direction }: { direction: string }) {
  const normalized = direction.toLowerCase();
  return (
    <Badge variant={normalized === "long" ? "green" : "red"}>
      {direction}
    </Badge>
  );
}
