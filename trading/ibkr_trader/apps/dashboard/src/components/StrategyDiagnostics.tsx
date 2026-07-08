'use client';
import { useEffect, useState } from 'react';
import type { StrategyDiagnosticsRow } from '@/lib/types';
import { fmtAge } from '@/lib/formatters';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

interface Props {
  strategyId: string;
}

// Tone mapping mirrors the canonical taxonomy in libs/services/decision_codes.py.
// 'ACTIVE' is intentionally absent -- it is a daily-classification label, not a
// decision_code, so it never reaches this column.
function decisionTone(code: string | null): 'success' | 'warning' | 'danger' | 'default' {
  if (!code) return 'default';
  if (code === 'ENTRY_FILLED' || code === 'MANAGING_POSITION') return 'success';
  if (code === 'EVALUATED_NO_SIGNAL' || code === 'IDLE' || code === 'SIGNAL_EMITTED') return 'default';
  if (code.startsWith('BLOCKED_')) return 'warning';
  if (code === 'HALT_GUARDED' || code === 'STAND_DOWN') return 'danger';
  return 'default';
}

function farmsFromStatus(status: Record<string, string> | null): { ok: number; broken: string[] } {
  if (!status) return { ok: 0, broken: [] };
  let ok = 0;
  const broken: string[] = [];
  for (const [farm, state] of Object.entries(status)) {
    if (state === 'OK') ok += 1;
    else broken.push(`${farm}=${state}`);
  }
  return { ok, broken };
}

/** Collapsible diagnostics panel rendered under StrategyCard. Pulls from
 * v_strategy_diagnostics (libs/oms/persistence/postgres.py) so the watchdog's
 * view of the world is operator-readable without a DB shell. */
export function StrategyDiagnostics({ strategyId }: Props) {
  const [open, setOpen] = useState(false);
  const [data, setData] = useState<StrategyDiagnosticsRow | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;

    async function load() {
      try {
        const res = await fetch('/api/strategies/diagnostics', { cache: 'no-store' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const rows: StrategyDiagnosticsRow[] = await res.json();
        const row = rows.find(r => r.strategy_id === strategyId) ?? null;
        if (!cancelled) {
          setData(row);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : 'fetch failed');
      }
    }

    load();
    const t = setInterval(load, 15_000); // matches engine heartbeat cadence
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, [open, strategyId]);

  if (!open) {
    return (
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="text-xs text-gray-500 hover:text-gray-300 font-mono underline-offset-2 hover:underline"
      >
        diagnostics ▾
      </button>
    );
  }

  return (
    <div className="space-y-2 border-t border-gray-800 pt-2 mt-2 text-xs font-mono">
      <div className="flex items-center justify-between">
        <span className="text-gray-500 uppercase tracking-wide">Diagnostics</span>
        <button
          type="button"
          onClick={() => setOpen(false)}
          className="text-gray-500 hover:text-gray-300"
        >
          hide ▴
        </button>
      </div>

      {error && <p className="text-red-400">{error}</p>}
      {!error && !data && <p className="text-gray-500">Loading…</p>}

      {data && (
        <div className="space-y-1.5">
          <Row label="Decision">
            <Badge variant={decisionTone(data.last_decision_code)}>
              {data.last_decision_code ?? 'UNKNOWN'}
            </Badge>
          </Row>
          <Row label="Bar age">
            <span className={cn(
              'text-gray-300',
              (data.bar_age_sec ?? 0) > 600 && 'text-amber-400',
            )}>
              {data.bar_age_sec === null ? '—' : `${fmtAge(data.bar_age_sec)} ago`}
            </span>
          </Row>
          <Row label="Bars today">
            <span className="text-gray-300">{data.bars_processed ?? 0}</span>
          </Row>
          <Row label="Denials">
            <span className={cn(
              (data.consecutive_denials ?? 0) > 5 && 'text-red-400',
              (data.consecutive_denials ?? 0) > 0 && (data.consecutive_denials ?? 0) <= 5 && 'text-amber-400',
              (data.consecutive_denials ?? 0) === 0 && 'text-gray-300',
            )}>
              {data.intents_denied ?? 0}{' '}
              <span className="text-gray-500">
                ({data.consecutive_denials ?? 0} consec)
              </span>
            </span>
          </Row>
          <SymbolFreshness freshness={data.symbol_freshness} />
          <FarmStatus status={data.ib_farm_status} />
        </div>
      )}
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-gray-500">{label}</span>
      <span className="text-right">{children}</span>
    </div>
  );
}

function SymbolFreshness({ freshness }: { freshness: Record<string, string> | null }) {
  if (!freshness || Object.keys(freshness).length === 0) {
    return (
      <Row label="Symbols">
        <span className="text-gray-500">—</span>
      </Row>
    );
  }
  const now = Date.now();
  const entries = Object.entries(freshness)
    .map(([sym, ts]) => ({ sym, ageSec: Math.max(0, (now - Date.parse(ts)) / 1000) }))
    .sort((a, b) => b.ageSec - a.ageSec);
  return (
    <div>
      <span className="text-gray-500">Symbols</span>
      <div className="mt-0.5 flex flex-wrap gap-1">
        {entries.map(e => (
          <span
            key={e.sym}
            className={cn(
              'rounded bg-gray-900 px-1.5 py-0.5 text-[10px]',
              e.ageSec > 1800 ? 'text-amber-400' : 'text-gray-400',
            )}
            title={`${fmtAge(e.ageSec)} ago`}
          >
            {e.sym}: {fmtAge(e.ageSec)}
          </span>
        ))}
      </div>
    </div>
  );
}

function FarmStatus({ status }: { status: Record<string, string> | null }) {
  const { ok, broken } = farmsFromStatus(status);
  if (ok === 0 && broken.length === 0) return null;
  return (
    <Row label="IB farms">
      {broken.length === 0 ? (
        <span className="text-green-400">{ok} OK</span>
      ) : (
        <span className="text-red-400" title={broken.join(', ')}>
          {broken.length} broken
        </span>
      )}
    </Row>
  );
}
