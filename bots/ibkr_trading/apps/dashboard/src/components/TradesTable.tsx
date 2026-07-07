'use client';
import { TradeRow, getSystemConfig } from '@/lib/types';
import { fmtR, fmtTime, fmtHoldTime } from '@/lib/formatters';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

interface Props {
  trades: TradeRow[] | null;
}

function exitReasonStyle(reason: string | null): string {
  switch (reason) {
    case 'STOP':   return 'bg-red-900 text-red-200 border-red-800';
    case 'TP1':    return 'bg-green-900 text-green-200 border-green-800';
    case 'TP2':    return 'bg-emerald-900 text-emerald-200 border-emerald-800';
    case 'TRAIL':  return 'bg-teal-900 text-teal-200 border-teal-800';
    case 'STALE':
    case 'EOD':    return 'bg-gray-800 text-gray-300 border-gray-700';
    case 'EVENT':  return 'bg-amber-900 text-amber-200 border-amber-800';
    case 'MANUAL': return 'bg-purple-900 text-purple-200 border-purple-800';
    default:       return 'bg-gray-800 text-gray-300 border-gray-700';
  }
}

export function TradesTable({ trades }: Props) {
  return (
    <Card className="flex flex-col flex-1 min-h-0">
      <CardHeader>
        <CardTitle>Today&apos;s Trades ({trades?.length ?? 0})</CardTitle>
      </CardHeader>
      <CardContent className="p-0 flex-1 flex flex-col min-h-0">
        <div className="overflow-x-auto overflow-y-auto flex-1 min-h-0">
          <table className="w-full text-xs font-mono">
            <thead className="sticky top-0 bg-[#111318]">
              <tr className="border-b border-gray-800 text-gray-500">
                <th className="px-4 py-2 text-left">Symbol</th>
                <th className="px-4 py-2 text-left">Strat</th>
                <th className="px-4 py-2 text-left">Dir</th>
                <th className="px-4 py-2 text-right">Entry</th>
                <th className="px-4 py-2 text-right">Exit</th>
                <th className="px-4 py-2 text-right">R</th>
                <th className="px-4 py-2 text-right">Hold</th>
                <th className="px-4 py-2 text-left">Exit</th>
              </tr>
            </thead>
            <tbody>
              {trades == null ? (
                <tr>
                  <td colSpan={8} className="px-4 py-6 text-center text-gray-600">Loading…</td>
                </tr>
              ) : trades.length === 0 ? (
                <tr>
                  <td colSpan={8} className="px-4 py-6 text-center text-gray-600">No trades today</td>
                </tr>
              ) : (
                trades.map(t => {
                  const isWin = (t.realized_r ?? 0) > 0;
                  const isOpen = t.exit_ts == null;
                  return (
                    <tr
                      key={t.trade_id}
                      className={cn(
                        'border-b border-gray-800/50 hover:bg-[#1a1d24]',
                        isOpen ? '' : isWin ? 'bg-green-950/20' : 'bg-red-950/20'
                      )}
                    >
                      <td className="px-4 py-2 font-semibold text-gray-100">{t.instrument_symbol}</td>
                      <td className="px-4 py-2 text-gray-400 truncate max-w-[100px]">
                        <span className="inline-flex items-center gap-1.5">
                          <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${getSystemConfig(t.strategy_id).dotColor}`} />
                          {t.strategy_id}
                        </span>
                      </td>
                      <td className={cn('px-4 py-2', t.direction === 'LONG' ? 'text-green-400' : 'text-red-400')}>
                        {t.direction}
                      </td>
                      <td className="px-4 py-2 text-right text-gray-300">{fmtTime(t.entry_ts)}</td>
                      <td className="px-4 py-2 text-right text-gray-300">{t.exit_ts ? fmtTime(t.exit_ts) : '\u2014'}</td>
                      <td className={cn('px-4 py-2 text-right font-semibold',
                        isOpen ? 'text-gray-400' : isWin ? 'text-green-400' : 'text-red-400'
                      )}>
                        {isOpen ? 'OPEN' : fmtR(t.realized_r)}
                      </td>
                      <td className="px-4 py-2 text-right text-gray-500">{fmtHoldTime(t.duration_minutes)}</td>
                      <td className="px-4 py-2">
                        <span className={cn(
                          'inline-flex items-center rounded px-1.5 py-0.5 text-xs border',
                          exitReasonStyle(t.exit_reason)
                        )}>
                          {t.exit_reason ?? 'OPEN'}
                        </span>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
