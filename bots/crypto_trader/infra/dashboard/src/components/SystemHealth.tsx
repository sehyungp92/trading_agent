"use client";

import type { HealthData, SafetyEventRow } from "@/lib/types";
import { fmtAge, fmtDate } from "@/lib/formatters";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const ASSESSMENT_VARIANT: Record<string, "green" | "amber" | "red" | "neutral"> = {
  healthy: "green",
  degraded: "amber",
  critical: "red",
};

export function hasBlockingSafetyEvent(events: SafetyEventRow[]): boolean {
  return events.some((event) => {
    const severity = (event.severity ?? "").toLowerCase();
    const status = (event.status ?? "").toLowerCase();
    return (
      severity === "critical" ||
      severity === "error" ||
      status === "open" ||
      status === "failed" ||
      status === "blocked" ||
      status === "rejected"
    );
  });
}

export function SystemHealth({
  data,
  safetyEvents = [],
}: {
  data: HealthData | null;
  safetyEvents?: SafetyEventRow[];
}) {
  if (!data) {
    return (
      <Card>
        <CardHeader><CardTitle>System Health</CardTitle></CardHeader>
        <p className="text-sm text-zinc-500 text-center py-4">No health data</p>
      </Card>
    );
  }

  const uptimeStr = data.uptime_sec ? fmtAge(data.uptime_sec / 60) : "--";
  const blockingSafetyEvent = hasBlockingSafetyEvent(safetyEvents);
  const variant = blockingSafetyEvent ? "red" : ASSESSMENT_VARIANT[data.assessment] ?? "neutral";
  const assessment = blockingSafetyEvent && data.assessment === "healthy" ? "attention" : data.assessment;

  return (
    <Card>
      <CardHeader>
        <CardTitle>System Health</CardTitle>
        <Badge variant={variant}>{assessment}</Badge>
      </CardHeader>

      <div className="flex gap-6 text-sm">
        <div>
          <p className="text-xs text-zinc-500">Uptime</p>
          <p className="font-mono text-zinc-200">{uptimeStr}</p>
        </div>
        <div>
          <p className="text-xs text-zinc-500">Last Report</p>
          <p className="font-mono text-zinc-200">{fmtDate(data.timestamp)}</p>
        </div>
      </div>

      {data.postgres_sink?.enabled && (
        <div className="mt-3 pt-3 border-t border-surface-3 text-sm">
          <p className="text-xs text-zinc-500 mb-2">Postgres Sink</p>
          <div className="grid grid-cols-3 gap-3">
            <Metric label="Queue" value={`${data.postgres_sink.queue_depth ?? 0}/${data.postgres_sink.queue_capacity ?? "--"}`} />
            <Metric label="Dropped" value={String(data.postgres_sink.jobs_dropped ?? 0)} />
            <Metric label="Failures" value={String(data.postgres_sink.write_failures ?? 0)} />
          </div>
          {data.postgres_sink.last_error ? (
            <p className="mt-2 text-xs text-accent-red truncate">{String(data.postgres_sink.last_error)}</p>
          ) : null}
        </div>
      )}

      {data.alerts.length > 0 && (
        <div className="mt-3 pt-3 border-t border-surface-3">
          <p className="text-xs text-zinc-500 mb-1">Alerts</p>
          <div className="flex flex-wrap gap-1">
            {data.alerts.map((alert, i) => (
              <Badge key={i} variant="amber">
                {typeof alert === "string" ? alert : JSON.stringify(alert)}
              </Badge>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs text-zinc-500">{label}</p>
      <p className="font-mono text-zinc-200">{value}</p>
    </div>
  );
}
