"""Counterfactual simulator — what-if trade replay with modified strategy rules.

Replays historical trades under modified conditions:
  - Remove filter: include missed opportunities blocked by a specific filter
  - Add regime gate: exclude trades in a specific market regime
  - Exclude trades: remove trades matching arbitrary criteria

Does NOT require tick-level data — works with existing TradeEvent and
MissedOpportunityEvent records.
"""
from __future__ import annotations

from collections.abc import Callable

from trading_assistant.schemas.counterfactual import (
    CounterfactualResult,
    CounterfactualScenario,
    ScenarioType,
)
from trading_assistant.schemas.events import MissedOpportunityEvent, TradeEvent


class CounterfactualSimulator:
    """Lightweight counterfactual trade replay engine."""

    def simulate_remove_filter(
        self,
        trades: list[TradeEvent],
        missed: list[MissedOpportunityEvent],
        filter_name: str,
    ) -> CounterfactualResult:
        """Simulate removing a filter — add back blocked missed opportunities."""
        baseline_pnl, baseline_count, baseline_wr = self._compute_baseline(trades)

        # Include missed ops that were blocked by this filter
        unfiltered = [m for m in missed if m.blocked_by == filter_name]
        additional_pnl = sum(m.outcome_24h or 0.0 for m in unfiltered)
        additional_wins = sum(1 for m in unfiltered if (m.outcome_24h or 0.0) > 0)

        modified_pnl = baseline_pnl + additional_pnl
        modified_count = baseline_count + len(unfiltered)
        modified_wins = sum(1 for t in trades if t.pnl > 0) + additional_wins
        modified_wr = modified_wins / modified_count if modified_count > 0 else 0.0

        return CounterfactualResult(
            scenario=CounterfactualScenario(
                scenario_type=ScenarioType.REMOVE_FILTER,
                description=f"Remove {filter_name} filter",
                parameters={"filter_name": filter_name},
            ),
            baseline_pnl=baseline_pnl,
            modified_pnl=modified_pnl,
            baseline_trade_count=baseline_count,
            modified_trade_count=modified_count,
            baseline_win_rate=baseline_wr,
            modified_win_rate=modified_wr,
        )

    def simulate_regime_gate(
        self,
        trades: list[TradeEvent],
        missed: list[MissedOpportunityEvent],
        regime_to_exclude: str,
    ) -> CounterfactualResult:
        """Simulate adding a regime gate — exclude trades in specified regime."""
        baseline_pnl, baseline_count, baseline_wr = self._compute_baseline(trades)

        kept = [t for t in trades if (t.market_regime or "") != regime_to_exclude]
        modified_pnl = sum(t.pnl for t in kept)
        modified_wins = sum(1 for t in kept if t.pnl > 0)
        modified_count = len(kept)
        modified_wr = modified_wins / modified_count if modified_count > 0 else 0.0

        return CounterfactualResult(
            scenario=CounterfactualScenario(
                scenario_type=ScenarioType.ADD_REGIME_GATE,
                description=f"Add regime gate for {regime_to_exclude}",
                parameters={"regime": regime_to_exclude},
            ),
            baseline_pnl=baseline_pnl,
            modified_pnl=modified_pnl,
            baseline_trade_count=baseline_count,
            modified_trade_count=modified_count,
            baseline_win_rate=baseline_wr,
            modified_win_rate=modified_wr,
        )

    def simulate_exclude(
        self,
        trades: list[TradeEvent],
        missed: list[MissedOpportunityEvent],
        exclude_fn: Callable[[TradeEvent], bool],
    ) -> CounterfactualResult:
        """Simulate excluding trades that match a criteria function."""
        baseline_pnl, baseline_count, baseline_wr = self._compute_baseline(trades)

        kept = [t for t in trades if not exclude_fn(t)]
        modified_pnl = sum(t.pnl for t in kept)
        modified_wins = sum(1 for t in kept if t.pnl > 0)
        modified_count = len(kept)
        modified_wr = modified_wins / modified_count if modified_count > 0 else 0.0

        return CounterfactualResult(
            scenario=CounterfactualScenario(
                scenario_type=ScenarioType.EXCLUDE_TRADES,
                description="Exclude trades by criteria",
                parameters={},
            ),
            baseline_pnl=baseline_pnl,
            modified_pnl=modified_pnl,
            baseline_trade_count=baseline_count,
            modified_trade_count=modified_count,
            baseline_win_rate=baseline_wr,
            modified_win_rate=modified_wr,
        )

    @staticmethod
    def _compute_baseline(trades: list[TradeEvent]) -> tuple[float, int, float]:
        pnl = sum(t.pnl for t in trades)
        count = len(trades)
        wins = sum(1 for t in trades if t.pnl > 0)
        wr = wins / count if count > 0 else 0.0
        return pnl, count, wr
