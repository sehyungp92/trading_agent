"use client";

import type { DailyPnlPoint } from "@/lib/types";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
} from "recharts";

export function DailyPnlBars({ data }: { data: DailyPnlPoint[] }) {
  if (data.length === 0) {
    return (
      <Card>
        <CardHeader><CardTitle>Daily P&L (30d)</CardTitle></CardHeader>
        <p className="text-sm text-zinc-500 text-center py-12">No daily data yet</p>
      </Card>
    );
  }

  const formatted = data.map((d) => ({
    date: new Date(d.trade_date).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
    }),
    pnl: d.net_pnl,
    trades: d.total_trades,
  }));

  return (
    <Card>
      <CardHeader><CardTitle>Daily P&L (30d)</CardTitle></CardHeader>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={formatted}>
            <CartesianGrid stroke="#24242f" strokeDasharray="3 3" />
            <XAxis
              dataKey="date"
              tick={{ fill: "#71717a", fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: "#24242f" }}
            />
            <YAxis
              tick={{ fill: "#71717a", fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: "#24242f" }}
              tickFormatter={(v: number) => `$${v.toFixed(0)}`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#111118",
                border: "1px solid #24242f",
                borderRadius: 8,
                fontSize: 12,
              }}
              formatter={(value, _name, item) => {
                const pnl = Number(value);
                const trades = Number(item.payload?.trades ?? 0);
                return [`$${pnl.toFixed(2)} (${trades} trades)`, "Net P&L"];
              }}
            />
            <Bar dataKey="pnl" radius={[3, 3, 0, 0]}>
              {formatted.map((entry, idx) => (
                <Cell
                  key={idx}
                  fill={entry.pnl >= 0 ? "#22c55e" : "#ef4444"}
                  fillOpacity={0.8}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}
