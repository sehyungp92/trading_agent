"""FillQualityAnalyzer — aggregates order fill details into quality metrics.

Follows the same pattern as SlippageAnalyzer: pure computation, no side effects.
"""
from __future__ import annotations

import statistics
from collections import defaultdict

from trading_assistant.schemas.events import TradeEvent
from trading_assistant.schemas.fill_quality import FillQualityReport, FillStats, SymbolFillQuality


class FillQualityAnalyzer:
    def __init__(self, bot_id: str, date: str) -> None:
        self._bot_id = bot_id
        self._date = date

    def compute(self, trades: list[TradeEvent]) -> FillQualityReport:
        """Compute fill quality from trades with fill detail data."""
        trades_with_data = [
            t for t in trades
            if t.entry_fill_details or t.exit_fill_details
        ]

        if not trades_with_data:
            return FillQualityReport(
                bot_id=self._bot_id,
                date=self._date,
            )

        # Collect per-symbol and overall stats
        entry_slippages: list[float] = []
        exit_slippages: list[float] = []
        entry_latencies: list[float] = []
        exit_latencies: list[float] = []
        entry_adverse_count = 0
        exit_adverse_count = 0
        entry_fill_types: dict[str, int] = defaultdict(int)
        exit_fill_types: dict[str, int] = defaultdict(int)
        entry_total = 0
        exit_total = 0

        symbol_entries: dict[str, list[dict]] = defaultdict(list)
        symbol_exits: dict[str, list[dict]] = defaultdict(list)

        for t in trades_with_data:
            if t.entry_fill_details:
                entry_total += 1
                details = t.entry_fill_details
                slip = details.get("slippage_bps", 0.0)
                entry_slippages.append(slip)
                lat = details.get("fill_latency_ms", 0.0)
                entry_latencies.append(lat)
                if slip > 0:
                    entry_adverse_count += 1
                ft = details.get("fill_type", "unknown")
                entry_fill_types[ft] += 1
                symbol_entries[t.pair].append(details)

            if t.exit_fill_details:
                exit_total += 1
                details = t.exit_fill_details
                slip = details.get("slippage_bps", 0.0)
                exit_slippages.append(slip)
                lat = details.get("fill_latency_ms", 0.0)
                exit_latencies.append(lat)
                if slip > 0:
                    exit_adverse_count += 1
                ft = details.get("fill_type", "unknown")
                exit_fill_types[ft] += 1
                symbol_exits[t.pair].append(details)

        overall_entry = self._build_stats(
            entry_slippages, entry_latencies, entry_adverse_count,
            entry_total, dict(entry_fill_types),
        )
        overall_exit = self._build_stats(
            exit_slippages, exit_latencies, exit_adverse_count,
            exit_total, dict(exit_fill_types),
        )

        # Per-symbol breakdown
        all_symbols = set(symbol_entries.keys()) | set(symbol_exits.keys())
        by_symbol: dict[str, SymbolFillQuality] = {}
        for sym in sorted(all_symbols):
            e_dets = symbol_entries.get(sym, [])
            x_dets = symbol_exits.get(sym, [])
            e_stats = self._build_stats_from_details(e_dets)
            x_stats = self._build_stats_from_details(x_dets)
            net_impact = e_stats.avg_slippage_bps + x_stats.avg_slippage_bps
            by_symbol[sym] = SymbolFillQuality(
                symbol=sym,
                entry_stats=e_stats,
                exit_stats=x_stats,
                net_adverse_impact_bps=round(net_impact, 2),
            )

        coverage_pct = len(trades_with_data) / len(trades) * 100 if trades else 0.0

        # Adverse selection: >60% fills adverse AND avg slippage > 1 bps
        all_slippages = entry_slippages + exit_slippages
        total_fills = entry_total + exit_total
        total_adverse = entry_adverse_count + exit_adverse_count
        adverse_pct = total_adverse / total_fills if total_fills > 0 else 0.0
        avg_slip = statistics.mean(all_slippages) if all_slippages else 0.0
        adverse_detected = adverse_pct > 0.6 and avg_slip > 1.0

        return FillQualityReport(
            bot_id=self._bot_id,
            date=self._date,
            overall_entry=overall_entry,
            overall_exit=overall_exit,
            by_symbol=by_symbol,
            coverage_pct=round(coverage_pct, 1),
            adverse_selection_detected=adverse_detected,
        )

    @staticmethod
    def _build_stats(
        slippages: list[float],
        latencies: list[float],
        adverse_count: int,
        total: int,
        fill_types: dict[str, int],
    ) -> FillStats:
        if not slippages:
            return FillStats()
        sorted_slip = sorted(slippages)
        n = len(sorted_slip)
        return FillStats(
            sample_count=n,
            avg_slippage_bps=round(statistics.mean(sorted_slip), 2),
            median_slippage_bps=round(statistics.median(sorted_slip), 2),
            p95_slippage_bps=round(sorted_slip[int(n * 0.95)] if n >= 20 else sorted_slip[-1], 2),
            avg_fill_latency_ms=round(statistics.mean(latencies), 2) if latencies else 0.0,
            adverse_fill_pct=round(adverse_count / total * 100, 1) if total > 0 else 0.0,
            by_fill_type=fill_types,
        )

    @classmethod
    def _build_stats_from_details(cls, details: list[dict]) -> FillStats:
        if not details:
            return FillStats()
        slippages = [d.get("slippage_bps", 0.0) for d in details]
        latencies = [d.get("fill_latency_ms", 0.0) for d in details]
        adverse = sum(1 for s in slippages if s > 0)
        fill_types: dict[str, int] = defaultdict(int)
        for d in details:
            fill_types[d.get("fill_type", "unknown")] += 1
        return cls._build_stats(slippages, latencies, adverse, len(details), dict(fill_types))
