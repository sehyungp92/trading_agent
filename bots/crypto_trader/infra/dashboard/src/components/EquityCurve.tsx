"use client";

import type { EquityCurvePoint } from "@/lib/types";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

export function EquityCurve({ data }: { data: EquityCurvePoint[] }) {
  if (data.length === 0) {
    return (
      <Card>
        <CardHeader><CardTitle>Equity Curve (90d)</CardTitle></CardHeader>
        <p className="text-sm text-zinc-500 text-center py-12">No equity data yet</p>
      </Card>
    );
  }

  const formatted = data.map((d) => ({
    ts: new Date(d.ts).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
    equity: d.equity,
  }));

  return (
    <Card>
      <CardHeader><CardTitle>Equity Curve (90d)</CardTitle></CardHeader>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={formatted}>
            <CartesianGrid stroke="#24242f" strokeDasharray="3 3" />
            <XAxis
              dataKey="ts"
              tick={{ fill: "#71717a", fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: "#24242f" }}
            />
            <YAxis
              tick={{ fill: "#71717a", fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: "#24242f" }}
              tickFormatter={(v: number) => `$${(v / 1000).toFixed(1)}k`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#111118",
                border: "1px solid #24242f",
                borderRadius: 8,
                fontSize: 12,
              }}
              formatter={(v: number) => [
                `$${v.toLocaleString("en-US", { minimumFractionDigits: 2 })}`,
                "Equity",
              ]}
            />
            <Line
              type="monotone"
              dataKey="equity"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}
