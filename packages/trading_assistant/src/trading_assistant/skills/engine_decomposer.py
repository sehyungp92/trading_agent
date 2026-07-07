"""Engine decomposer — splits per-bot metrics into per-engine metrics.

Uses ``entry_signal`` from TradeEvent + ``sub_engines`` from StrategyRegistry
to classify each trade into its originating engine, then computes per-engine
performance metrics with regime breakdowns.
"""
from __future__ import annotations

import re
import statistics
from collections import defaultdict

from trading_assistant.schemas.engine_metrics import (
    EngineDecomposition,
    EngineMetrics,
    RegimeEngineStats,
)
from trading_assistant.schemas.events import TradeEvent
from trading_assistant.schemas.strategy_profile import StrategyRegistry


class EngineDecomposer:
    """Decomposes trade lists into per-engine metrics."""

    def __init__(self, strategy_registry: StrategyRegistry | None = None) -> None:
        self._registry = strategy_registry
        self._engine_patterns: dict[str, list[tuple[re.Pattern, str]]] = {}
        if strategy_registry:
            self._build_engine_patterns()

    def _build_engine_patterns(self) -> None:
        """Build case-insensitive prefix patterns from strategy sub_engines."""
        if not self._registry:
            return
        for sid, profile in self._registry.strategies.items():
            if not profile.sub_engines:
                continue
            patterns: list[tuple[re.Pattern, str]] = []
            for engine_name in profile.sub_engines:
                # Match engine name as prefix in entry_signal (case-insensitive)
                # e.g. "REVERSAL" matches "reversal_short", "reversal_long_aggressive"
                pattern = re.compile(rf"^{re.escape(engine_name)}[_\-]?", re.IGNORECASE)
                patterns.append((pattern, engine_name.upper()))
            self._engine_patterns[sid] = patterns

    def parse_engine(self, entry_signal: str, strategy_id: str) -> str:
        """Extract engine tag from entry_signal; fallback to full signal string."""
        if not entry_signal:
            return "unknown"
        patterns = self._engine_patterns.get(strategy_id, [])
        for pattern, engine_name in patterns:
            if pattern.match(entry_signal):
                return engine_name
        # No pattern matched — use full signal as engine tag
        return entry_signal

    def decompose(
        self,
        trades: list[TradeEvent],
        bot_id: str,
        period: str = "",
    ) -> EngineDecomposition:
        """Group trades by engine, compute per-engine metrics + regime breakdown."""
        if not trades:
            return EngineDecomposition(bot_id=bot_id, period=period)

        # Determine strategy_id for this bot
        strategy_id = self._resolve_strategy_id(trades, bot_id)

        # Group trades by engine
        engine_trades: dict[str, list[TradeEvent]] = defaultdict(list)
        unmapped = 0
        for trade in trades:
            engine = self.parse_engine(trade.entry_signal, strategy_id)
            if engine == "unknown":
                unmapped += 1
            engine_trades[engine].append(trade)

        engines: list[EngineMetrics] = []
        for engine_name, etrades in sorted(engine_trades.items()):
            if engine_name == "unknown":
                continue
            metrics = self._compute_engine_metrics(
                engine_name, etrades, strategy_id, bot_id,
            )
            engines.append(metrics)

        return EngineDecomposition(
            bot_id=bot_id,
            period=period,
            engines=engines,
            unmapped_trades=unmapped,
        )

    def _resolve_strategy_id(self, trades: list[TradeEvent], bot_id: str) -> str:
        """Find strategy_id for this bot from registry."""
        if not self._registry:
            return ""
        bot_strategies = self._registry.strategies_for_bot(bot_id)
        if len(bot_strategies) == 1:
            return next(iter(bot_strategies))
        # Try to infer from trade strategy_id field
        for trade in trades:
            if trade.strategy_id and trade.strategy_id in self._registry.strategies:
                return trade.strategy_id
        # Return first match if any
        if bot_strategies:
            return next(iter(bot_strategies))
        return ""

    @staticmethod
    def _compute_trade_stats(
        trades: list[TradeEvent],
    ) -> tuple[int, float, float, float]:
        """Compute (trade_count, win_rate, avg_pnl, profit_factor) for a trade set."""
        n = len(trades)
        if n == 0:
            return 0, 0.0, 0.0, 0.0
        wins = sum(1 for t in trades if t.pnl > 0)
        pnls = [t.pnl for t in trades]
        win_rate = wins / n
        avg_pnl = statistics.mean(pnls)
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else (
            float("inf") if gross_profit > 0 else 0.0
        )
        return n, round(win_rate, 4), round(avg_pnl, 4), round(min(pf, 99.99), 4)

    def _compute_engine_metrics(
        self,
        engine_name: str,
        trades: list[TradeEvent],
        strategy_id: str,
        bot_id: str,
    ) -> EngineMetrics:
        """Compute aggregate metrics for a single engine."""
        n, win_rate, avg_pnl, profit_factor = self._compute_trade_stats(trades)

        exit_effs = [t.exit_efficiency for t in trades if t.exit_efficiency is not None]
        avg_exit_eff = statistics.mean(exit_effs) if exit_effs else 0.0

        signal_strengths = [t.entry_signal_strength for t in trades if t.entry_signal_strength > 0]
        avg_signal = statistics.mean(signal_strengths) if signal_strengths else 0.0

        regime_breakdown = self._compute_regime_breakdown(trades)

        return EngineMetrics(
            engine=engine_name,
            strategy_id=strategy_id,
            bot_id=bot_id,
            trade_count=n,
            win_rate=win_rate,
            avg_pnl=avg_pnl,
            profit_factor=profit_factor,
            avg_exit_efficiency=round(avg_exit_eff, 4),
            avg_signal_strength=round(avg_signal, 4),
            regime_breakdown=regime_breakdown,
        )

    def _compute_regime_breakdown(self, trades: list[TradeEvent]) -> list[RegimeEngineStats]:
        """Compute per-regime stats for a set of trades."""
        regime_groups: dict[str, list[TradeEvent]] = defaultdict(list)
        for t in trades:
            regime = t.market_regime or "unknown"
            regime_groups[regime].append(t)

        breakdown: list[RegimeEngineStats] = []
        for regime, rtrades in sorted(regime_groups.items()):
            n, win_rate, avg_pnl, pf = self._compute_trade_stats(rtrades)
            breakdown.append(RegimeEngineStats(
                regime=regime,
                trade_count=n,
                win_rate=win_rate,
                avg_pnl=avg_pnl,
                profit_factor=pf,
            ))
        return breakdown
