"""Ablation analyzer — statistical analysis of boolean flag on/off performance.

Scans ``strategy_params_at_entry`` for boolean parameters, splits trades by
flag state, and computes per-state win rates, average PnL, and statistical
significance via Mann-Whitney U test.
"""
from __future__ import annotations

import statistics
from collections import defaultdict

from trading_assistant.schemas.ablation_analysis import (
    AblationAnalysis,
    AblationFlagStats,
    AblationRegimeStats,
)
from trading_assistant.schemas.events import TradeEvent


class AblationAnalyzer:
    """Analyzes boolean ablation flags from strategy_params_at_entry."""

    def _extract_boolean_params(
        self, trades: list[TradeEvent],
    ) -> dict[str, list[tuple[bool, TradeEvent]]]:
        """Scan trades for params where all observed values are boolean.

        Returns mapping of param_name -> [(flag_value, trade), ...].
        Must check isinstance(val, bool) BEFORE float(val) since float(True)==1.0.
        """
        # First pass: collect all values per param to identify booleans
        param_values: dict[str, list[tuple[object, TradeEvent]]] = defaultdict(list)
        for t in trades:
            if not t.strategy_params_at_entry:
                continue
            for key, val in t.strategy_params_at_entry.items():
                param_values[key].append((val, t))

        # Second pass: filter to params where ALL values are boolean
        boolean_params: dict[str, list[tuple[bool, TradeEvent]]] = {}
        for key, pairs in param_values.items():
            all_bool = all(isinstance(v, bool) for v, _ in pairs)
            if all_bool and len(pairs) >= 2:
                boolean_params[key] = [(v, t) for v, t in pairs]  # type: ignore[misc]
        return boolean_params

    def _compute_flag_stats(
        self,
        flag_name: str,
        trades_by_state: dict[bool, list[TradeEvent]],
        bot_id: str,
        strategy_id: str = "",
    ) -> AblationFlagStats:
        """Compute win rate, avg PnL, and significance for a single flag."""
        enabled_trades = trades_by_state.get(True, [])
        disabled_trades = trades_by_state.get(False, [])

        enabled_pnls = [t.pnl for t in enabled_trades]
        disabled_pnls = [t.pnl for t in disabled_trades]

        enabled_wins = sum(1 for p in enabled_pnls if p > 0)
        disabled_wins = sum(1 for p in disabled_pnls if p > 0)

        enabled_wr = enabled_wins / len(enabled_pnls) if enabled_pnls else 0.0
        disabled_wr = disabled_wins / len(disabled_pnls) if disabled_pnls else 0.0

        enabled_avg = statistics.mean(enabled_pnls) if enabled_pnls else 0.0
        disabled_avg = statistics.mean(disabled_pnls) if disabled_pnls else 0.0

        # Mann-Whitney U test for significance
        p_value = self._mann_whitney_u(enabled_pnls, disabled_pnls)

        # Regime breakdown
        regime_breakdown = self._compute_regime_breakdown(enabled_trades, disabled_trades)

        return AblationFlagStats(
            flag_name=flag_name,
            strategy_id=strategy_id,
            bot_id=bot_id,
            enabled_count=len(enabled_trades),
            disabled_count=len(disabled_trades),
            enabled_win_rate=round(enabled_wr, 4),
            disabled_win_rate=round(disabled_wr, 4),
            enabled_avg_pnl=round(enabled_avg, 4),
            disabled_avg_pnl=round(disabled_avg, 4),
            pnl_delta=round(disabled_avg - enabled_avg, 4),
            statistical_significance=round(p_value, 6),
            regime_breakdown=regime_breakdown,
        )

    @staticmethod
    def _mann_whitney_u(x: list[float], y: list[float]) -> float:
        """Simple Mann-Whitney U test returning approximate p-value.

        Uses normal approximation for n >= 8 per group.
        Returns 1.0 if either group is empty or too small.
        """
        nx, ny = len(x), len(y)
        if nx < 2 or ny < 2:
            return 1.0

        # Rank all observations
        combined = [(v, 0) for v in x] + [(v, 1) for v in y]
        combined.sort(key=lambda p: p[0])

        # Assign ranks (handle ties by averaging)
        ranks: list[float] = [0.0] * len(combined)
        i = 0
        while i < len(combined):
            j = i
            while j < len(combined) and combined[j][0] == combined[i][0]:
                j += 1
            avg_rank = (i + j + 1) / 2  # 1-based average rank
            for k in range(i, j):
                ranks[k] = avg_rank
            i = j

        # Sum ranks for group x
        r1 = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 0)
        u1 = r1 - nx * (nx + 1) / 2

        # Normal approximation
        mu = nx * ny / 2
        sigma = (nx * ny * (nx + ny + 1) / 12) ** 0.5
        if sigma == 0:
            return 1.0
        z = abs(u1 - mu) / sigma

        # Approximate two-tailed p-value using logistic approximation of normal CDF
        # P(Z > z) ≈ 1 / (1 + exp(1.7 * z))
        import math
        try:
            p_one_tail = 1.0 / (1.0 + math.exp(1.7 * z))
        except OverflowError:
            p_one_tail = 0.0
        return min(2.0 * p_one_tail, 1.0)

    @staticmethod
    def _compute_regime_breakdown(
        enabled_trades: list[TradeEvent],
        disabled_trades: list[TradeEvent],
    ) -> list[AblationRegimeStats]:
        """Compute per-regime stats comparing enabled vs disabled."""
        regime_enabled: dict[str, list[TradeEvent]] = defaultdict(list)
        regime_disabled: dict[str, list[TradeEvent]] = defaultdict(list)

        for t in enabled_trades:
            regime_enabled[t.market_regime or "unknown"].append(t)
        for t in disabled_trades:
            regime_disabled[t.market_regime or "unknown"].append(t)

        all_regimes = sorted(set(regime_enabled) | set(regime_disabled))
        breakdown: list[AblationRegimeStats] = []

        for regime in all_regimes:
            en = regime_enabled.get(regime, [])
            dis = regime_disabled.get(regime, [])
            en_pnls = [t.pnl for t in en]
            dis_pnls = [t.pnl for t in dis]
            en_wr = sum(1 for p in en_pnls if p > 0) / len(en_pnls) if en_pnls else 0.0
            dis_wr = sum(1 for p in dis_pnls if p > 0) / len(dis_pnls) if dis_pnls else 0.0
            en_avg = statistics.mean(en_pnls) if en_pnls else 0.0
            dis_avg = statistics.mean(dis_pnls) if dis_pnls else 0.0
            breakdown.append(AblationRegimeStats(
                regime=regime,
                enabled_count=len(en),
                disabled_count=len(dis),
                enabled_win_rate=round(en_wr, 4),
                disabled_win_rate=round(dis_wr, 4),
                pnl_delta=round(dis_avg - en_avg, 4),
            ))
        return breakdown

    def analyze(
        self,
        trades: list[TradeEvent],
        bot_id: str,
        min_per_state: int = 10,
        period: str = "",
    ) -> AblationAnalysis:
        """Full ablation analysis. Only includes flags with >= min_per_state per state."""
        if not trades:
            return AblationAnalysis(bot_id=bot_id, period=period)

        boolean_params = self._extract_boolean_params(trades)
        if not boolean_params:
            return AblationAnalysis(bot_id=bot_id, period=period)

        flags: list[AblationFlagStats] = []
        flags_with_signal: list[str] = []

        for flag_name, pairs in sorted(boolean_params.items()):
            # Group by state
            by_state: dict[bool, list[TradeEvent]] = defaultdict(list)
            for val, trade in pairs:
                by_state[val].append(trade)

            # Check minimum per state
            if len(by_state.get(True, [])) < min_per_state:
                continue
            if len(by_state.get(False, [])) < min_per_state:
                continue

            stats = self._compute_flag_stats(flag_name, by_state, bot_id)
            flags.append(stats)

            if stats.statistical_significance < 0.10:
                flags_with_signal.append(flag_name)

        return AblationAnalysis(
            bot_id=bot_id,
            period=period,
            flags=flags,
            flags_with_signal=flags_with_signal,
        )
