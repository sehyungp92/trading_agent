/** Format R-multiple with sign. */
export function fmtR(r: number | null): string {
  if (r == null) return "--";
  const sign = r >= 0 ? "+" : "";
  return `${sign}${r.toFixed(2)}R`;
}

/** Format USD with sign. */
export function fmtUSD(v: number): string {
  const sign = v >= 0 ? "+" : "";
  return `${sign}$${Math.abs(v).toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

/** Format date as short UTC string. */
export function fmtDate(iso: string | null): string {
  if (!iso) return "--";
  const d = new Date(iso);
  return d.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "UTC",
  });
}

/** Format age in minutes to human-readable. */
export function fmtAge(minutes: number): string {
  if (minutes < 60) return `${Math.round(minutes)}m`;
  if (minutes < 1440) return `${(minutes / 60).toFixed(1)}h`;
  return `${(minutes / 1440).toFixed(1)}d`;
}

/** Format percentage with sign. */
export function fmtPct(v: number): string {
  const sign = v >= 0 ? "+" : "";
  return `${sign}${v.toFixed(2)}%`;
}

/** Color class for positive/negative values. */
export function colorClass(v: number): string {
  if (v > 0) return "text-accent-green";
  if (v < 0) return "text-accent-red";
  return "text-zinc-400";
}
