'use client';
import { DailyPnlPoint } from '@/lib/types';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  ReferenceLine,
  Tooltip,
  CartesianGrid,
} from 'recharts';
import { fmtDate, fmtR } from '@/lib/formatters';

interface Props {
  data: DailyPnlPoint[] | null;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const r = payload[0].value as number;
  return (
    <div className="rounded border border-gray-700 bg-[#111318] px-3 py-2 text-xs font-mono shadow-lg">
      <p className="text-gray-400">{fmtDate(label)}</p>
      <p className={r >= 0 ? 'text-green-400' : 'text-red-400'}>{fmtR(r)}</p>
    </div>
  );
}

export function DailyPnlBars({ data }: Props) {
  const totalR = data?.reduce((sum, d) => sum + d.daily_realized_r, 0) ?? 0;
  const wins = data?.filter(d => d.daily_realized_r > 0).length ?? 0;
  const total = data?.length ?? 0;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>30-Day Daily P&amp;L</CardTitle>
          {data && data.length > 0 && (
            <div className="flex items-center gap-3 text-xs font-mono">
              <span className="text-gray-500">{wins}/{total} wins</span>
              <span className={totalR >= 0 ? 'text-green-400' : 'text-red-400'}>{fmtR(totalR)}</span>
            </div>
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
            <BarChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
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
              <ReferenceLine y={0} stroke="#374151" />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
              <Bar dataKey="daily_realized_r" radius={[2, 2, 0, 0]}>
                {data.map((entry, idx) => (
                  <Cell
                    key={idx}
                    fill={entry.daily_realized_r >= 0 ? '#22c55e' : '#ef4444'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
}
