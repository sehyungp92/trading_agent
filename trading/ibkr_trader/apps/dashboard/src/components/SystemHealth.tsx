'use client';
import { HealthData } from '@/lib/types';
import { fmtAge, fmtDateTime } from '@/lib/formatters';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

interface Props {
  health: HealthData | null;
}

function healthVariant(status: string): 'success' | 'danger' | 'warning' | 'default' {
  if (status === 'OK') return 'success';
  if (status === 'DISCONNECTED' || status === 'ERROR') return 'danger';
  if (status === 'STALE' || status === 'WARNING' || status === 'UNKNOWN') return 'warning';
  return 'default';
}

export function SystemHealth({ health }: Props) {
  const strategies = health?.strategies ?? [];
  const adapters = health?.adapters ?? [];
  const evidence = health?.evidence ?? null;
  const evidenceWarnings = evidence?.warnings ?? [];

  return (
    <Card>
      <CardHeader>
        <CardTitle>System Health</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Evidence pipeline */}
        <div>
          <div className="flex items-baseline justify-between mb-2">
            <p className="text-xs text-gray-500 font-mono uppercase tracking-wider">Evidence Pipeline</p>
            {evidence ? (
              <Badge variant={healthVariant(evidence.status)}>{evidence.status}</Badge>
            ) : (
              <Badge variant="warning">UNKNOWN</Badge>
            )}
          </div>
          {evidence ? (
            <div className="space-y-2 text-xs font-mono">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2 min-w-0">
                  <Badge variant={healthVariant(evidence.relay.status)}>Relay</Badge>
                  <span className="text-gray-300 truncate">
                    {evidence.relay.reachable ? 'reachable' : 'unreachable'}
                  </span>
                </div>
                <div className="flex items-center gap-3 flex-shrink-0 text-gray-500">
                  <span>{evidence.relay.pending_events ?? 0} pending</span>
                  <span>oldest {fmtAge(evidence.relay.oldest_pending_age_seconds)}</span>
                </div>
              </div>
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2 min-w-0">
                  <Badge variant={healthVariant(evidence.assistant.status)}>Assistant</Badge>
                  <span className="text-gray-300 truncate">
                    {evidence.assistant.required_bot_ids.length > 0
                      ? `${evidence.assistant.required_bot_ids.length} required`
                      : 'no required bot list'}
                  </span>
                </div>
                <div className="flex items-center gap-3 flex-shrink-0 text-gray-500">
                  <span>{evidence.assistant.missing_bot_ids.length} missing</span>
                  <span>{evidence.assistant.stale_bot_ids.length} stale</span>
                </div>
              </div>
              {evidenceWarnings.length > 0 && (
                <div className="space-y-1">
                  {evidenceWarnings.slice(0, 3).map((warning, idx) => (
                    <p
                      key={`${warning}-${idx}`}
                      className={cn(
                        'truncate',
                        evidence.status === 'ERROR' ? 'text-red-300' : 'text-amber-300',
                      )}
                    >
                      {warning}
                    </p>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <p className="text-xs text-gray-600 font-mono">No evidence health data</p>
          )}
        </div>

        {/* Adapters */}
        <div>
          <div className="flex items-baseline justify-between mb-2">
            <p className="text-xs text-gray-500 font-mono uppercase tracking-wider">Broker Adapters</p>
            {adapters.length > 0 && (() => {
              const bad = adapters.filter(a => a.health_status !== 'OK');
              const totalDiscon = adapters.reduce((s, a) => s + (a.disconnect_count_24h || 0), 0);
              if (bad.length === 0 && totalDiscon === 0) {
                return <Badge variant="success">All OK</Badge>;
              }
              if (bad.length > 0) {
                return <Badge variant="danger">{bad.length} down</Badge>;
              }
              return <Badge variant="warning">{totalDiscon} discon/24h</Badge>;
            })()}
          </div>
          {adapters.length === 0 ? (
            <p className="text-xs text-gray-600 font-mono">No adapters registered</p>
          ) : (
            <div className="space-y-2">
              {adapters.map(a => (
                <div key={a.adapter_id} className="flex items-center justify-between gap-2 text-xs font-mono">
                  <div className="flex items-center gap-2 min-w-0">
                    <Badge variant={healthVariant(a.health_status)}>{a.health_status}</Badge>
                    <span className="text-gray-300 truncate">{a.adapter_id}</span>
                  </div>
                  <div className="flex items-center gap-3 flex-shrink-0">
                    <span className="text-gray-500">HB: {fmtAge(a.heartbeat_age_sec)}</span>
                    {a.disconnect_count_24h > 0 && (
                      <span className="text-amber-400">{a.disconnect_count_24h} discon</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Strategy heartbeats */}
        <div>
          <p className="text-xs text-gray-500 font-mono uppercase tracking-wider mb-2">Strategy Heartbeats</p>
          {strategies.length === 0 ? (
            <p className="text-xs text-gray-600 font-mono">No strategies registered</p>
          ) : (
            <div className="space-y-2">
              {strategies.map(s => (
                <div key={s.strategy_id} className="flex items-center justify-between gap-2 text-xs font-mono">
                  <div className="flex items-center gap-2 min-w-0">
                    <Badge variant={healthVariant(s.health_status)}>{s.health_status}</Badge>
                    <span className="text-gray-300 truncate">{s.strategy_id}</span>
                  </div>
                  <div className="flex items-center gap-3 flex-shrink-0">
                    <span className="text-gray-500">{fmtAge(s.heartbeat_age_sec)} ago</span>
                    <span className={cn(s.mode === 'RUNNING' ? 'text-green-500' : 'text-amber-400')}>
                      {s.mode}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Last errors */}
        {strategies.some(s => s.last_error) && (
          <div>
            <p className="text-xs text-gray-500 font-mono uppercase tracking-wider mb-2">Recent Errors</p>
            <div className="space-y-1">
              {strategies.filter(s => s.last_error).map(s => (
                <div key={s.strategy_id} className="text-xs font-mono">
                  <span className="text-amber-400">{s.strategy_id}</span>
                  <span className="text-gray-500"> @ {fmtDateTime(s.last_error_ts)}: </span>
                  <span className="text-red-300">{s.last_error}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
