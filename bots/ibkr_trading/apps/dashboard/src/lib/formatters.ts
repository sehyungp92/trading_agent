/** Format R-value: "+1.23R" or "-0.50R" */
export function fmtR(r: number | null | undefined): string {
  if (r == null || isNaN(r)) return '\u2014';
  const sign = r >= 0 ? '+' : '';
  return `${sign}${r.toFixed(2)}R`;
}

/** Format USD: "+$1,234.56" or "-$500.00" */
export function fmtUSD(usd: number | null | undefined): string {
  if (usd == null || isNaN(usd)) return '\u2014';
  const abs = Math.abs(usd);
  const sign = usd >= 0 ? '+' : '-';
  return `${sign}$${abs.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

/** Format age in seconds: "2m 30s" or "5h 10m" */
export function fmtAge(seconds: number | null | undefined): string {
  if (seconds == null || isNaN(seconds)) return '\u2014';
  const s = Math.floor(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

/** Format hold time in minutes: "5m" or "2h 15m" */
export function fmtHoldTime(minutes: number | null | undefined): string {
  if (minutes == null || isNaN(minutes)) return '\u2014';
  const m = Math.floor(minutes);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

/** Format date: "Feb 20" */
export function fmtDate(ts: string | null | undefined): string {
  if (!ts) return '\u2014';
  return new Date(ts).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

/** Format time: "14:32:05" */
export function fmtTime(ts: string | null | undefined): string {
  if (!ts) return '\u2014';
  return new Date(ts).toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
}

/** Format datetime: "Feb 20 14:32" */
export function fmtDateTime(ts: string | null | undefined): string {
  if (!ts) return '\u2014';
  const d = new Date(ts);
  return `${fmtDate(ts)} ${d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false })}`;
}

/** Clamp a value between min and max, returns 0-100 percentage */
export function toPercent(value: number, max: number): number {
  return Math.min(100, Math.max(0, (value / max) * 100));
}
