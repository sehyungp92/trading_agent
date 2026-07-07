"use client";

import { useEffect, useState } from "react";

export function RefreshIndicator({
  intervalSec,
  lastUpdate,
}: {
  intervalSec: number;
  lastUpdate: Date | null;
}) {
  const [countdown, setCountdown] = useState(intervalSec);

  useEffect(() => {
    setCountdown(intervalSec);
    const timer = setInterval(() => {
      setCountdown((prev) => (prev <= 1 ? intervalSec : prev - 1));
    }, 1000);
    return () => clearInterval(timer);
  }, [intervalSec, lastUpdate]);

  const lastStr = lastUpdate
    ? lastUpdate.toLocaleTimeString("en-US", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
        timeZone: "UTC",
      })
    : "—";

  return (
    <div className="flex items-center gap-3 text-xs text-zinc-500">
      <span className="inline-block w-1.5 h-1.5 rounded-full bg-accent-green animate-pulse" />
      <span>Next refresh in {countdown}s</span>
      <span className="text-zinc-600">|</span>
      <span>Last: {lastStr} UTC</span>
    </div>
  );
}
