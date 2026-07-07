"""PostgreSQL-Instrumentation bridge using the decorator pattern.

InstrumentedTradeRecorder wraps a TradeRecorder to emit instrumentation events
while guaranteeing that PostgreSQL writes always succeed, even if instrumentation fails.

Architecture:
- On record_entry: calls inner.record_entry FIRST (PG is primary), then kit.log_entry
- On record_exit: calls inner.record_exit FIRST, then kit.log_exit
- On record: pass-through to inner.record
- Failures in kit methods are swallowed; PG writes must always succeed
"""
from __future__ import annotations

import logging
from decimal import Decimal
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger("instrumentation.pg_bridge")


class InstrumentedTradeRecorder:
    """Decorator wrapping TradeRecorder with instrumentation via InstrumentationKit.

    This class implements the decorator pattern, ensuring:
    1. PG writes (TradeRecorder) are always attempted first and must succeed
    2. Instrumentation (kit) is called as best-effort, failures are swallowed
    3. trade_id is always returned, even if instrumentation fails

    Usage::

        recorder = TradeRecorder(store)
        kit = InstrumentationKit(ctx, strategy_id="ATRSS")
        instrumented = InstrumentedTradeRecorder(recorder, kit)

        # record_entry calls recorder FIRST, then kit (both best-effort)
        trade_id = await instrumented.record_entry(...)
        # record_exit calls recorder FIRST, then kit (both best-effort)
        await instrumented.record_exit(...)
    """

    def __init__(self, inner_recorder, kit):
        """Initialize with a wrapped TradeRecorder and InstrumentationKit.

        Args:
            inner_recorder: TradeRecorder instance (the primary recorder)
            kit: InstrumentationKit instance (best-effort instrumentation)
        """
        self._inner = inner_recorder
        self._kit = kit

    async def record_entry(
        self,
        strategy_id: str,
        instrument: str,
        direction: str,
        quantity: int,
        entry_price: Decimal,
        entry_ts: datetime,
        setup_tag: str = None,
        entry_type: str = None,
        meta: dict = None,
        account_id: str = "default",
    ) -> str:
        """Record trade entry: call PG first (primary), then kit (best-effort).

        This method guarantees that the PostgreSQL write succeeds and returns trade_id,
        even if instrumentation fails.

        Args:
            strategy_id: Strategy identifier
            instrument: Trading instrument/symbol
            direction: "LONG" or "SHORT"
            quantity: Position size
            entry_price: Entry fill price
            entry_ts: Entry timestamp
            setup_tag: Optional setup tag
            entry_type: Optional entry type
            meta: Optional metadata dict
            account_id: Account identifier (default: "default")

        Returns:
            trade_id: Unique trade identifier from PostgreSQL write

        Raises:
            Any exception from inner_recorder.record_entry (PG write failures)
        """
        # CRITICAL: Call inner recorder FIRST. This must succeed.
        trade_id = await self._inner.record_entry(
            strategy_id=strategy_id,
            instrument=instrument,
            direction=direction,
            quantity=quantity,
            entry_price=entry_price,
            entry_ts=entry_ts,
            setup_tag=setup_tag,
            entry_type=entry_type,
            meta=meta,
            account_id=account_id,
        )

        # Best-effort instrumentation: swallow exceptions
        if self._kit is not None:
            try:
                self._kit.log_entry(
                    trade_id=trade_id,
                    pair=instrument,
                    side=direction,
                    entry_price=float(entry_price),
                    position_size=float(quantity),
                    position_size_quote=float(entry_price * quantity),
                    entry_signal="trade_recorded",
                    entry_signal_id=trade_id,
                    entry_signal_strength=1.0,
                    active_filters=[],
                    passed_filters=[],
                    strategy_params={},
                    signal_factors=[],
                    filter_decisions=[],
                    sizing_inputs={"quantity": quantity},
                    portfolio_state_at_entry={},
                    exchange_timestamp=entry_ts,
                )
            except Exception as e:
                # Log but don't raise: instrumentation is best-effort
                logger.warning(
                    f"Instrumentation failed on record_entry for trade_id={trade_id}: {e}",
                    exc_info=True,
                )

        return trade_id

    async def record_exit(
        self,
        trade_id: str,
        exit_price: Decimal,
        exit_ts: datetime,
        exit_reason: str,
        realized_r: Decimal,
        realized_usd: Decimal = None,
        notes: str = None,
        mae_r: Decimal = None,
        mfe_r: Decimal = None,
        duration_seconds: int = None,
        duration_bars: int = None,
        max_adverse_price: Decimal = None,
        max_favorable_price: Decimal = None,
    ) -> None:
        """Record trade exit: call PG first (primary), then kit (best-effort).

        This method guarantees that the PostgreSQL write succeeds,
        even if instrumentation fails.

        Args:
            trade_id: Trade identifier to update
            exit_price: Exit fill price
            exit_ts: Exit timestamp
            exit_reason: Exit reason category
            realized_r: Realized R multiple
            realized_usd: Realized PnL in USD
            notes: Optional notes
            mae_r: Max adverse excursion in R
            mfe_r: Max favorable excursion in R
            duration_seconds: Trade duration in seconds
            duration_bars: Trade duration in bars
            max_adverse_price: Max adverse price reached
            max_favorable_price: Max favorable price reached

        Raises:
            Any exception from inner_recorder.record_exit (PG write failures)
        """
        # CRITICAL: Call inner recorder FIRST. This must succeed.
        await self._inner.record_exit(
            trade_id=trade_id,
            exit_price=exit_price,
            exit_ts=exit_ts,
            exit_reason=exit_reason,
            realized_r=realized_r,
            realized_usd=realized_usd,
            notes=notes,
            mae_r=mae_r,
            mfe_r=mfe_r,
            duration_seconds=duration_seconds,
            duration_bars=duration_bars,
            max_adverse_price=max_adverse_price,
            max_favorable_price=max_favorable_price,
        )

        # Best-effort instrumentation: swallow exceptions
        if self._kit is not None:
            try:
                self._kit.log_exit(
                    trade_id=trade_id,
                    exit_price=float(exit_price),
                    exit_reason=exit_reason,
                    fees_paid=0.0,
                    exchange_timestamp=exit_ts,
                )
            except Exception as e:
                # Log but don't raise: instrumentation is best-effort
                logger.warning(
                    f"Instrumentation failed on record_exit for trade_id={trade_id}: {e}",
                    exc_info=True,
                )

    async def record(self, data: dict) -> str:
        """Record a complete trade: pass-through to inner recorder.

        Args:
            data: Complete trade data dict

        Returns:
            trade_id: Unique trade identifier

        Raises:
            Any exception from inner_recorder.record (PG write failures)
        """
        return await self._inner.record(data)
