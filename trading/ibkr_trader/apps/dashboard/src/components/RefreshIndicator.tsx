'use client';
import { fmtTime } from '@/lib/formatters';
import { RefreshCw } from 'lucide-react';
import { cn } from '@/lib/utils';

interface Props {
  lastUpdate: Date | null;
  nextRefreshIn: number; // seconds
  isRefreshing: boolean;
  serverTime?: string | null;
}

export function RefreshIndicator({ lastUpdate, nextRefreshIn, isRefreshing, serverTime }: Props) {
  const displayTime = serverTime ? fmtTime(serverTime) : lastUpdate ? fmtTime(lastUpdate.toISOString()) : null;

  return (
    <div className="fixed bottom-4 right-4 flex items-center gap-2 rounded-full border border-gray-800 bg-[#111318] px-3 py-1.5 text-xs font-mono text-gray-500 shadow-lg">
      <RefreshCw
        className={cn('h-3 w-3', isRefreshing ? 'animate-spin text-green-400' : 'text-gray-600')}
      />
      <span>
        {displayTime ? `Updated ${displayTime}` : 'Connecting\u2026'}
      </span>
      {!isRefreshing && (
        <span className="text-gray-700">&middot; {nextRefreshIn}s</span>
      )}
      <span className="text-gray-700 hidden sm:inline">&middot; R:refresh 1-3:jump</span>
    </div>
  );
}
