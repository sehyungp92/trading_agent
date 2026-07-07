"""Interaction analyzer — coordinator event analysis for swing_multi_01.

Deterministic pipeline. No LLM calls. Analyzes coordinator actions (stop-tightening,
size-boosting, overlay signals) and estimates their impact on trade outcomes.
"""
from __future__ import annotations

from collections import defaultdict

from trading_assistant.schemas.interaction_analysis import (
    CoordinatorAction,
    InteractionEffect,
    InteractionReport,
)
from trading_assistant.schemas.events import TradeEvent


class InteractionAnalyzer:
    """Analyzes coordinator interaction effects on swing_multi_01 trades."""

    def __init__(
        self,
        week_start: str,
        week_end: str,
        bot_id: str = "swing_multi_01",
    ) -> None:
        self.week_start = week_start
        self.week_end = week_end
        self.bot_id = bot_id

    def compute(
        self,
        coordinator_events: list[CoordinatorAction],
        trades: list[TradeEvent],
    ) -> InteractionReport:
        """Compute interaction analysis from coordinator events and trades.

        Args:
            coordinator_events: parsed coordinator action events
            trades: swing_multi_01 trades for the period
        """
        if not coordinator_events:
            return InteractionReport(
                week_start=self.week_start,
                week_end=self.week_end,
                bot_id=self.bot_id,
            )

        # Group events by rule + action_type
        grouped: dict[tuple[str, str], list[CoordinatorAction]] = defaultdict(list)
        for evt in coordinator_events:
            grouped[(evt.rule, evt.action)].append(evt)

        # Build trade index by symbol for matching
        trades_by_symbol: dict[str, list[TradeEvent]] = defaultdict(list)
        for t in trades:
            trades_by_symbol[t.pair].append(t)

        effects: list[InteractionEffect] = []
        total_benefit = 0.0

        for (rule, action_type), events in grouped.items():
            if action_type == "tighten_stop_be":
                effect = self._analyze_stop_tightening(
                    rule, action_type, events, trades_by_symbol,
                )
            elif action_type == "size_boost":
                effect = self._analyze_size_boost(
                    rule, action_type, events, trades_by_symbol,
                )
            else:
                effect = self._analyze_generic(
                    rule, action_type, events,
                )

            effects.append(effect)
            total_benefit += effect.net_benefit

        # Overlay regime summary
        overlay_summary = self._compute_overlay_summary(trades)

        # Recommendation
        if total_benefit > 0:
            recommendation = (
                f"Coordinator is net positive (+${total_benefit:.0f}). "
                f"Keep current rules active."
            )
        elif total_benefit < 0:
            recommendation = (
                f"Coordinator is net negative (${total_benefit:.0f}). "
                f"Review rule parameters — tightening may be too aggressive."
            )
        else:
            recommendation = "Coordinator had no measurable impact this period."

        return InteractionReport(
            week_start=self.week_start,
            week_end=self.week_end,
            bot_id=self.bot_id,
            total_coordination_events=len(coordinator_events),
            effects=effects,
            overlay_regime_summary=overlay_summary,
            net_coordinator_benefit=round(total_benefit, 2),
            recommendation=recommendation,
        )

    def _analyze_stop_tightening(
        self,
        rule: str,
        action_type: str,
        events: list[CoordinatorAction],
        trades_by_symbol: dict[str, list[TradeEvent]],
    ) -> InteractionEffect:
        """Analyze stop-tightening actions and estimate counterfactual impact."""
        affected = 0
        total_impact = 0.0
        total_without = 0.0
        trigger = events[0].trigger_strategy if events else ""
        target = events[0].target_strategy if events else ""

        for evt in events:
            symbol = evt.symbol
            matched_trades = trades_by_symbol.get(symbol, [])

            for trade in matched_trades:
                # Match trade active during the event
                if not self._trade_overlaps_event(trade, evt):
                    continue

                affected += 1
                actual_pnl = trade.pnl

                # Counterfactual: if exit was STOP_LOSS, the tightened stop triggered it.
                # Without tightening, trade might have continued.
                if trade.exit_reason == "STOP_LOSS":
                    # Estimate what would have happened without tightening
                    # Use post-exit prices if available
                    if trade.post_exit_4h_price is not None:
                        if trade.side == "LONG":
                            hypothetical = (trade.post_exit_4h_price - trade.entry_price) * trade.position_size
                        else:
                            hypothetical = (trade.entry_price - trade.post_exit_4h_price) * trade.position_size
                        total_without += hypothetical
                        total_impact += actual_pnl
                    else:
                        total_without += actual_pnl
                        total_impact += actual_pnl
                else:
                    # Trade wasn't stopped out by tightened stop — survived
                    # Benefit = avoided drawdown (no direct PnL change)
                    total_impact += actual_pnl
                    total_without += actual_pnl

        net = total_impact - total_without

        return InteractionEffect(
            rule=rule,
            action_type=action_type,
            trigger_strategy=trigger,
            target_strategy=target,
            action_count=len(events),
            affected_trades=affected,
            estimated_pnl_impact=round(total_impact, 2),
            estimated_pnl_without=round(total_without, 2),
            net_benefit=round(net, 2),
            confidence=min(0.8, 0.3 + affected * 0.1) if affected > 0 else 0.0,
        )

    def _analyze_size_boost(
        self,
        rule: str,
        action_type: str,
        events: list[CoordinatorAction],
        trades_by_symbol: dict[str, list[TradeEvent]],
    ) -> InteractionEffect:
        """Analyze size-boost actions and compute PnL uplift from extra allocation."""
        affected = 0
        total_impact = 0.0
        total_without = 0.0
        boost_factor = 1.25  # default boost is 1.25x
        trigger = events[0].trigger_strategy if events else ""
        target = events[0].target_strategy if events else ""

        for evt in events:
            factor = evt.details.get("boost_factor", boost_factor)
            symbol = evt.symbol
            matched_trades = trades_by_symbol.get(symbol, [])

            for trade in matched_trades:
                if not self._trade_overlaps_event(trade, evt):
                    continue

                affected += 1
                actual_pnl = trade.pnl
                # Without boost, PnL would be actual / boost_factor
                pnl_without = actual_pnl / factor if factor > 0 else actual_pnl
                total_impact += actual_pnl
                total_without += pnl_without

        net = total_impact - total_without

        return InteractionEffect(
            rule=rule,
            action_type=action_type,
            trigger_strategy=trigger,
            target_strategy=target,
            action_count=len(events),
            affected_trades=affected,
            estimated_pnl_impact=round(total_impact, 2),
            estimated_pnl_without=round(total_without, 2),
            net_benefit=round(net, 2),
            confidence=min(0.8, 0.3 + affected * 0.1) if affected > 0 else 0.0,
        )

    def _analyze_generic(
        self,
        rule: str,
        action_type: str,
        events: list[CoordinatorAction],
    ) -> InteractionEffect:
        """Generic analysis for overlay signal changes and other event types."""
        trigger = events[0].trigger_strategy if events else ""
        target = events[0].target_strategy if events else ""
        return InteractionEffect(
            rule=rule,
            action_type=action_type,
            trigger_strategy=trigger,
            target_strategy=target,
            action_count=len(events),
            affected_trades=0,
            confidence=0.2,
        )

    def _compute_overlay_summary(
        self, trades: list[TradeEvent],
    ) -> dict:
        """Group trade performance by overlay regime (market_regime field)."""
        regime_data: dict[str, dict] = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "wins": 0})

        for t in trades:
            regime = t.market_regime or "unknown"
            regime_data[regime]["trades"] += 1
            regime_data[regime]["pnl"] += t.pnl
            if t.pnl > 0:
                regime_data[regime]["wins"] += 1

        result: dict = {}
        for regime, data in regime_data.items():
            result[regime] = {
                "trades": data["trades"],
                "pnl": round(data["pnl"], 2),
                "win_rate": round(data["wins"] / data["trades"], 4) if data["trades"] > 0 else 0.0,
            }
        return result

    @staticmethod
    def _trade_overlaps_event(trade: TradeEvent, event: CoordinatorAction) -> bool:
        """Check if a trade was active when the coordinator event occurred."""
        if not event.timestamp:
            # No timestamp on event — match by symbol only
            return True

        try:
            from datetime import datetime
            evt_time = datetime.fromisoformat(event.timestamp)
            return trade.entry_time <= evt_time <= trade.exit_time
        except (ValueError, TypeError):
            # Can't parse timestamp — fall back to symbol-only match
            return True
