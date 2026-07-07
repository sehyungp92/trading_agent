"""SignalHealthAnalyzer — aggregates signal_evolution data into component health metrics.

Follows the same pattern as SlippageAnalyzer: pure computation, no side effects.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict

from trading_assistant.schemas.events import TradeEvent
from trading_assistant.schemas.signal_health import ComponentHealth, SignalHealthReport


class SignalHealthAnalyzer:
    def __init__(self, bot_id: str, date: str) -> None:
        self._bot_id = bot_id
        self._date = date

    def compute(self, trades: list[TradeEvent]) -> SignalHealthReport:
        """Compute signal health from trades with signal_evolution data."""
        trades_with_data = [t for t in trades if t.signal_evolution and len(t.signal_evolution) >= 2]

        if not trades_with_data:
            return SignalHealthReport(
                bot_id=self._bot_id,
                date=self._date,
                total_trades_with_data=0,
                coverage_pct=0.0,
            )

        # Collect per-component data across all trades
        comp_data: dict[str, list[dict]] = defaultdict(list)

        for t in trades_with_data:
            bars = t.signal_evolution
            # Extract component names (all keys except "bar")
            component_names = set()
            for bar in bars:
                component_names.update(k for k in bar if k != "bar")

            for comp in component_names:
                values = [bar.get(comp) for bar in bars if bar.get(comp) is not None]
                if len(values) < 2:
                    continue
                entry_val = values[0]
                exit_val = values[-1]
                val_range = max(values) - min(values)
                comp_data[comp].append({
                    "entry_value": entry_val,
                    "exit_value": exit_val,
                    "range": val_range,
                    "values": values,
                    "pnl": t.pnl,
                })

        # Build component health metrics
        components: list[ComponentHealth] = []
        for comp_name, entries in sorted(comp_data.items()):
            if not entries:
                continue

            trade_count = len(entries)
            avg_entry = statistics.mean(e["entry_value"] for e in entries)
            avg_exit = statistics.mean(e["exit_value"] for e in entries)
            avg_range = statistics.mean(e["range"] for e in entries)

            # Stability: 1 - (avg_std / avg_range), clamped to [0, 1]
            stds = []
            for e in entries:
                if len(e["values"]) >= 2:
                    stds.append(statistics.stdev(e["values"]))
            avg_std = statistics.mean(stds) if stds else 0.0
            stability = max(0.0, min(1.0, 1.0 - (avg_std / avg_range))) if avg_range > 0 else 1.0

            # Win correlation: Pearson between entry value and PnL (requires ≥3 trades)
            win_corr = 0.0
            if trade_count >= 3:
                entry_vals = [e["entry_value"] for e in entries]
                pnls = [e["pnl"] for e in entries]
                win_corr = self._pearson(entry_vals, pnls)

            # Trend during trade: avg (exit - entry)
            trend = statistics.mean(e["exit_value"] - e["entry_value"] for e in entries)

            components.append(ComponentHealth(
                component_name=comp_name,
                trade_count=trade_count,
                avg_entry_value=round(avg_entry, 4),
                avg_exit_value=round(avg_exit, 4),
                avg_range=round(avg_range, 4),
                stability=round(stability, 4),
                win_correlation=round(win_corr, 4),
                trend_during_trade=round(trend, 4),
            ))

        coverage_pct = len(trades_with_data) / len(trades) * 100 if trades else 0.0

        return SignalHealthReport(
            bot_id=self._bot_id,
            date=self._date,
            components=components,
            total_trades_with_data=len(trades_with_data),
            coverage_pct=round(coverage_pct, 1),
        )

    @staticmethod
    def _pearson(x: list[float], y: list[float]) -> float:
        """Compute Pearson correlation coefficient."""
        n = len(x)
        if n < 3:
            return 0.0
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)
        dx = [xi - mean_x for xi in x]
        dy = [yi - mean_y for yi in y]
        num = sum(a * b for a, b in zip(dx, dy, strict=False))
        den_x = math.sqrt(sum(a * a for a in dx))
        den_y = math.sqrt(sum(b * b for b in dy))
        if den_x == 0 or den_y == 0:
            return 0.0
        return num / (den_x * den_y)
