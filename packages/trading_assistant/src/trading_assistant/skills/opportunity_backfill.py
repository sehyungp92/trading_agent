# skills/opportunity_backfill.py
"""Opportunity backfill service — computes hypothetical outcomes for missed opportunities.

Applies simulation policy (fill model, fees, slippage, TP/SL) to determine
what would have happened if a missed signal had been traded.
"""
from __future__ import annotations

from dataclasses import dataclass

from trading_assistant.schemas.events import MissedOpportunityEvent, MarketSnapshot
from trading_assistant.schemas.simulation_policy import SimulationPolicy, FillModel, TPSLMethod


@dataclass
class BackfillResult:
    """Result of backfilling a single missed opportunity."""
    adjusted_entry: float
    fees: float
    slippage: float
    would_have_hit_tp: bool | None = None
    would_have_hit_sl: bool | None = None
    net_outcome_1h: float | None = None
    net_outcome_4h: float | None = None
    net_outcome_24h: float | None = None
    confidence: float = 0.0
    assumption_tags: list[str] | None = None

    def __post_init__(self):
        if self.assumption_tags is None:
            self.assumption_tags = []


class OpportunityBackfill:
    """Computes hypothetical outcomes for missed opportunities given a simulation policy."""

    def __init__(self, policy: SimulationPolicy) -> None:
        self._policy = policy

    def compute(
        self,
        event: MissedOpportunityEvent,
        snapshot: MarketSnapshot | None = None,
    ) -> BackfillResult:
        """Compute backfill for a single missed opportunity."""
        snapshot = snapshot or event.market_snapshot
        entry = self._determine_entry_price(event, snapshot)
        slippage = self._apply_slippage(entry)
        adjusted_entry = entry + slippage
        fees = self._compute_fees(entry)

        # Net outcomes — subtract round-trip fees from raw outcomes
        net_1h = self._net_outcome(event.outcome_1h, fees)
        net_4h = self._net_outcome(event.outcome_4h, fees)
        net_24h = self._net_outcome(event.outcome_24h, fees)

        # TP/SL evaluation
        tp_hit, sl_hit = self._evaluate_tpsl(
            adjusted_entry, event, snapshot
        )

        assumption_tags = list(event.assumption_tags) if event.assumption_tags else []
        assumption_tags.append(f"fill_model:{self._policy.fill_model.value}")

        return BackfillResult(
            adjusted_entry=adjusted_entry,
            fees=fees,
            slippage=slippage,
            would_have_hit_tp=tp_hit,
            would_have_hit_sl=sl_hit,
            net_outcome_1h=net_1h,
            net_outcome_4h=net_4h,
            net_outcome_24h=net_24h,
            confidence=event.confidence,
            assumption_tags=assumption_tags,
        )

    def compute_batch(
        self,
        events: list[MissedOpportunityEvent],
        snapshots: list[MarketSnapshot | None] | None = None,
    ) -> list[BackfillResult]:
        """Compute backfill for a batch of missed opportunities."""
        if snapshots is None:
            snapshots = [None] * len(events)
        return [self.compute(e, s) for e, s in zip(events, snapshots, strict=False)]

    def _determine_entry_price(
        self,
        event: MissedOpportunityEvent,
        snapshot: MarketSnapshot | None,
    ) -> float:
        """Determine the entry price based on the fill model."""
        if self._policy.fill_model == FillModel.HYPOTHETICAL:
            return event.hypothetical_entry

        if snapshot is None:
            return event.hypothetical_entry

        if self._policy.fill_model == FillModel.MID_PRICE:
            return snapshot.mid if snapshot.mid > 0 else event.hypothetical_entry

        if self._policy.fill_model == FillModel.ASK_FOR_LONG:
            return snapshot.ask if snapshot.ask > 0 else event.hypothetical_entry

        if self._policy.fill_model == FillModel.WORST_CASE:
            # Worst case: ask for long, bid for short — use ask since we don't know side
            return snapshot.ask if snapshot.ask > 0 else event.hypothetical_entry

        return event.hypothetical_entry

    def _apply_slippage(self, price: float) -> float:
        """Compute slippage adjustment (always increases effective cost)."""
        return price * self._policy.slippage_bps / 10_000

    def _compute_fees(self, entry_price: float) -> float:
        """Compute round-trip fees based on policy."""
        notional = entry_price * self._policy.default_position_size
        return notional * self._policy.fees_bps / 10_000 * 2

    def _evaluate_tpsl(
        self,
        entry: float,
        event: MissedOpportunityEvent,
        snapshot: MarketSnapshot | None,
    ) -> tuple[bool | None, bool | None]:
        """Evaluate TP/SL hit detection based on policy method."""
        tpsl = self._policy.tpsl

        if tpsl.method == TPSLMethod.NONE:
            return None, None

        outcomes = [
            v for v in [event.outcome_1h, event.outcome_4h, event.outcome_24h]
            if v is not None
        ]
        if not outcomes:
            return None, None

        if tpsl.method == TPSLMethod.FIXED_PCT:
            tp_target = entry * tpsl.tp_pct / 100 if tpsl.tp_pct > 0 else None
            sl_target = entry * tpsl.sl_pct / 100 if tpsl.sl_pct > 0 else None
            max_gain = max(outcomes) if outcomes else 0
            max_loss = min(outcomes) if outcomes else 0
            tp_hit = max_gain >= tp_target if tp_target is not None else None
            sl_hit = max_loss <= -sl_target if sl_target is not None else None
            return tp_hit, sl_hit

        if tpsl.method == TPSLMethod.ATR_MULTIPLE:
            atr = snapshot.atr_14 if snapshot and snapshot.atr_14 > 0 else 0
            if atr == 0:
                return None, None
            tp_target = atr * tpsl.atr_tp_multiple if tpsl.atr_tp_multiple > 0 else None
            sl_target = atr * tpsl.atr_sl_multiple if tpsl.atr_sl_multiple > 0 else None
            max_gain = max(outcomes) if outcomes else 0
            max_loss = min(outcomes) if outcomes else 0
            tp_hit = max_gain >= tp_target if tp_target is not None else None
            sl_hit = max_loss <= -sl_target if sl_target is not None else None
            return tp_hit, sl_hit

        return None, None

    @staticmethod
    def _net_outcome(raw: float | None, fees: float) -> float | None:
        """Subtract fees from raw outcome; propagate None."""
        if raw is None:
            return None
        return raw - fees
