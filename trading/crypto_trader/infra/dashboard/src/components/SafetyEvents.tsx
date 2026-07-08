"use client";

import type { SafetyEventRow } from "@/lib/types";
import { fmtDate } from "@/lib/formatters";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

function variantFor(event: SafetyEventRow): "green" | "amber" | "red" | "neutral" {
  const severity = (event.severity ?? "").toLowerCase();
  const status = (event.status ?? "").toLowerCase();
  if (
    severity === "error" ||
    severity === "critical" ||
    status === "open" ||
    status === "failed" ||
    status === "blocked" ||
    status === "rejected"
  ) return "red";
  if (severity === "warning" || event.event_type === "portfolio_rule") return "amber";
  return "neutral";
}

function labelFor(event: SafetyEventRow): string {
  if (event.description) return event.description;
  return event.event_type.replaceAll("_", " ");
}

export function SafetyEvents({ events }: { events: SafetyEventRow[] }) {
  if (events.length === 0) {
    return (
      <Card>
        <CardHeader><CardTitle>Safety Events</CardTitle></CardHeader>
        <p className="text-sm text-zinc-500 text-center py-6">No recent safety events</p>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader><CardTitle>Safety Events</CardTitle></CardHeader>
      <div className="space-y-3">
        {events.slice(0, 8).map((event) => (
          <div key={event.event_id} className="border-b border-surface-3/60 last:border-0 pb-3 last:pb-0">
            <div className="flex items-center justify-between gap-3">
              <div className="min-w-0">
                <p className="text-sm text-zinc-200 truncate">{labelFor(event)}</p>
                <p className="text-xs text-zinc-500 font-mono">
                  {event.strategy_id ?? "portfolio"} {event.symbol ? `/ ${event.symbol}` : ""}
                </p>
              </div>
              <Badge variant={variantFor(event)} className="shrink-0 max-w-32 truncate">
                {event.status ?? event.severity ?? event.event_type}
              </Badge>
            </div>
            <p className="text-xs text-zinc-500 mt-1">{fmtDate(event.timestamp)}</p>
          </div>
        ))}
      </div>
    </Card>
  );
}
