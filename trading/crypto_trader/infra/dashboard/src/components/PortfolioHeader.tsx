"use client";

import type { PortfolioData } from "@/lib/types";
import { fmtUSD, colorClass } from "@/lib/formatters";
import { Card } from "@/components/ui/card";

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div>
      <p className="text-xs text-zinc-500 uppercase tracking-wide">{label}</p>
      <p className={`text-xl font-mono font-semibold ${color ?? "text-zinc-100"}`}>
        {value}
      </p>
    </div>
  );
}

export function PortfolioHeader({ data }: { data: PortfolioData }) {
  return (
    <Card className="flex items-center gap-10 px-6 py-5">
      <Stat
        label="Portfolio Equity"
        value={`$${data.equity.toLocaleString("en-US", { minimumFractionDigits: 2 })}`}
      />
      <Stat
        label="Daily P&L"
        value={fmtUSD(data.daily_pnl_usd)}
        color={colorClass(data.daily_pnl_usd)}
      />
      <Stat
        label="Unrealized"
        value={fmtUSD(data.unrealized_pnl)}
        color={colorClass(data.unrealized_pnl)}
      />
      <Stat
        label="Heat"
        value={`${data.heat_r.toFixed(2)}R`}
        color={data.heat_r > 3 ? "text-accent-red" : "text-zinc-100"}
      />
      <Stat label="Open" value={String(data.open_positions)} />
    </Card>
  );
}
