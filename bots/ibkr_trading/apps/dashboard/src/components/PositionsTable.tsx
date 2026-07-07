'use client';
import { PositionRow, getSystemConfig } from '@/lib/types';
import { fmtUSD, fmtAge } from '@/lib/formatters';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { cn } from '@/lib/utils';

interface Props {
  positions: PositionRow[] | null;
}

export function PositionsTable({ positions }: Props) {
  return (
    <Card className="flex flex-col flex-1 min-h-0">
      <CardHeader>
        <CardTitle>Open Positions ({positions?.length ?? 0})</CardTitle>
      </CardHeader>
      <CardContent className="p-0 flex-1 flex flex-col min-h-0">
        <div className="overflow-x-auto overflow-y-auto flex-1 min-h-0">
          <table className="w-full text-xs font-mono">
            <thead className="sticky top-0 bg-[#111318]">
              <tr className="border-b border-gray-800 text-gray-500">
                <th className="px-4 py-2 text-left">Symbol</th>
                <th className="px-4 py-2 text-left">Strategy</th>
                <th className="px-4 py-2 text-right">Qty</th>
                <th className="px-4 py-2 text-right">Avg</th>
                <th className="px-4 py-2 text-right">Unreal P&amp;L</th>
                <th className="px-4 py-2 text-right">Risk$</th>
                <th className="px-4 py-2 text-right">Risk R</th>
                <th className="px-4 py-2 text-right">Age</th>
              </tr>
            </thead>
            <tbody>
              {positions == null ? (
                <tr>
                  <td colSpan={8} className="px-4 py-6 text-center text-gray-600">Loading…</td>
                </tr>
              ) : positions.length === 0 ? (
                <tr>
                  <td colSpan={8} className="px-4 py-6 text-center text-gray-600">No open positions</td>
                </tr>
              ) : (
                positions.map(p => (
                  <tr
                    key={`${p.account_id}-${p.instrument_symbol}-${p.strategy_id}`}
                    className={cn(
                      'border-b border-gray-800/50 hover:bg-[#1a1d24]',
                      p.unrealized_pnl > 0 ? 'bg-green-950/10' : p.unrealized_pnl < 0 ? 'bg-red-950/10' : ''
                    )}
                  >
                    <td className="px-4 py-2 font-semibold text-gray-100">{p.instrument_symbol}</td>
                    <td className="px-4 py-2 text-gray-400">
                      <span className="inline-flex items-center gap-1.5">
                        <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${getSystemConfig(p.strategy_id).dotColor}`} />
                        {p.strategy_id}
                      </span>
                    </td>
                    <td className="px-4 py-2 text-right text-gray-200">{p.net_qty}</td>
                    <td className="px-4 py-2 text-right text-gray-300">${p.avg_price.toFixed(2)}</td>
                    <td className={cn('px-4 py-2 text-right font-semibold',
                      p.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'
                    )}>
                      {fmtUSD(p.unrealized_pnl)}
                    </td>
                    <td className="px-4 py-2 text-right text-amber-400">{fmtUSD(p.open_risk_dollars)}</td>
                    <td className="px-4 py-2 text-right text-amber-400">{p.open_risk_r.toFixed(2)}R</td>
                    <td className="px-4 py-2 text-right text-gray-500">{fmtAge(p.stale_minutes * 60)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
