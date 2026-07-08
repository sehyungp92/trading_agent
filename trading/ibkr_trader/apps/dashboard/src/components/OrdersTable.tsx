'use client';
import { OrderRow, getSystemConfig } from '@/lib/types';
import { fmtAge } from '@/lib/formatters';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { cn } from '@/lib/utils';

interface Props {
  orders: OrderRow[] | null;
}

const statusColor: Record<string, string> = {
  WORKING: 'text-green-400',
  PARTIALLY_FILLED: 'text-amber-400',
  ACKED: 'text-blue-400',
  ROUTED: 'text-blue-400',
  QUEUED: 'text-amber-300',
  RISK_APPROVED: 'text-gray-400',
  CREATED: 'text-gray-500',
};

export function OrdersTable({ orders }: Props) {
  return (
    <Card className="flex flex-col flex-1 min-h-0">
      <CardHeader>
        <CardTitle>Working Orders ({orders?.length ?? 0})</CardTitle>
      </CardHeader>
      <CardContent className="p-0 flex-1 flex flex-col min-h-0">
        <div className="overflow-x-auto overflow-y-auto flex-1 min-h-0">
          <table className="w-full text-xs font-mono">
            <thead className="sticky top-0 bg-[#111318]">
              <tr className="border-b border-gray-800 text-gray-500">
                <th className="px-4 py-2 text-left">Symbol</th>
                <th className="px-4 py-2 text-left">Strategy</th>
                <th className="px-4 py-2 text-left">Role</th>
                <th className="px-4 py-2 text-left">Side</th>
                <th className="px-4 py-2 text-right">Qty</th>
                <th className="px-4 py-2 text-right">Filled</th>
                <th className="px-4 py-2 text-right">Price</th>
                <th className="px-4 py-2 text-left">Status</th>
                <th className="px-4 py-2 text-right">Age</th>
              </tr>
            </thead>
            <tbody>
              {orders == null ? (
                <tr>
                  <td colSpan={9} className="px-4 py-6 text-center text-gray-600">Loading...</td>
                </tr>
              ) : orders.length === 0 ? (
                <tr>
                  <td colSpan={9} className="px-4 py-6 text-center text-gray-600">No working orders</td>
                </tr>
              ) : (
                orders.map(o => (
                  <tr key={o.oms_order_id} className="border-b border-gray-800/50 hover:bg-[#1a1d24]">
                    <td className="px-4 py-2 font-semibold text-gray-100">{o.instrument_symbol}</td>
                    <td className="px-4 py-2 text-gray-400 truncate max-w-[100px]">
                      <span className="inline-flex items-center gap-1.5">
                        <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${getSystemConfig(o.strategy_id).dotColor}`} />
                        {o.strategy_id}
                      </span>
                    </td>
                    <td className="px-4 py-2 text-gray-300">{o.role}</td>
                    <td className={cn('px-4 py-2', o.side === 'BUY' ? 'text-green-400' : 'text-red-400')}>
                      {o.side}
                    </td>
                    <td className="px-4 py-2 text-right text-gray-200">{o.qty}</td>
                    <td className="px-4 py-2 text-right text-gray-400">{o.filled_qty}</td>
                    <td className="px-4 py-2 text-right text-gray-300">
                      ${(o.limit_price ?? o.stop_price ?? 0).toFixed(2)}
                    </td>
                    <td className={cn('px-4 py-2', statusColor[o.status] ?? 'text-gray-400')}>
                      <div className="flex flex-col">
                        <span>{o.status}</span>
                        {o.status === 'QUEUED' && o.queue_reason ? (
                          <span className="max-w-[140px] truncate text-[10px] text-gray-500">
                            {o.queue_reason}
                          </span>
                        ) : null}
                      </div>
                    </td>
                    <td
                      className={cn(
                        'px-4 py-2 text-right',
                        o.status === 'QUEUED' &&
                          o.queued_at &&
                          Date.now() - new Date(o.queued_at).getTime() > 150_000
                          ? 'text-amber-300'
                          : 'text-gray-500',
                      )}
                    >
                      {fmtAge(
                        o.status === 'QUEUED' && o.queued_at
                          ? (Date.now() - new Date(o.queued_at).getTime()) / 1000
                          : o.age_minutes * 60,
                      )}
                    </td>
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
