'use client';
import { EquityCurvePoint } from '@/lib/types';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  ReferenceLine,
  Tooltip,
  CartesianGrid,
} from 'recharts';
import { fmtDate, fmtR } from '@/lib/formatters';

interface Props {
  data: EquityCurvePoint[] | null;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload as EquityCurvePoint;
  return (
    <div className="rounded border border-gray-700 bg-[#111318] px-3 py-2 text-xs font-mono shadow-lg">
      <p className="text-gray-400">{fmtDate(label)}</p>
      <p className="text-gray-200">Cumul: <span className={d.cumulative_r >= 0 ? 'text-green-400' : 'text-red-400'}>{fmtR(d.cumulative_r)}</span></p>
      <p className="text-gray-400">Daily: {fmtR(d.daily_realized_r)}</p>
    </div>
  );
}

export function EquityCurve({ data }: Props) {
  const finalR = data && data.length > 0 ? data[data.length - 1].cumulative_r : 0;
  const lineColor = finalR >= 0 ? '#22c55e' : '#ef4444';

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>90-Day Equity Curve</CardTitle>
          {data && data.length > 0 && (
            <span className={`text-sm font-mono font-bold ${finalR >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {fmtR(finalR)}
            </span>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {data == null ? (
          <div className="h-[280px] flex items-center justify-center text-gray-600 font-mono text-sm">
            Loading…
          </div>
        ) : data.length === 0 ? (
          <div className="h-[280px] flex items-center justify-center text-gray-600 font-mono text-sm">
            No historical data
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis
                dataKey="trade_date"
                tickFormatter={v => fmtDate(v)}
                tick={{ fill: '#6b7280', fontSize: 10, fontFamily: 'monospace' }}
                axisLine={{ stroke: '#374151' }}
                tickLine={false}
                interval="preserveStartEnd"
              />
              <YAxis
                tickFormatter={v => `${v}R`}
                tick={{ fill: '#6b7280', fontSize: 10, fontFamily: 'monospace' }}
                axisLine={{ stroke: '#374151' }}
                tickLine={false}
                width={45}
              />
              <ReferenceLine y={0} stroke="#374151" strokeDasharray="4 4" />
              <Tooltip content={<CustomTooltip />} />
              <Line
                type="monotone"
                dataKey="cumulative_r"
                stroke={lineColor}
                strokeWidth={1.5}
                dot={false}
                activeDot={{ r: 3, fill: lineColor }}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}
