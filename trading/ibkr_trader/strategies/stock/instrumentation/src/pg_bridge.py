"""PostgreSQL-first trade recorder bridge for instrumentation."""
from __future__ import annotations

import logging
from hashlib import sha256
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

logger = logging.getLogger("instrumentation.pg_bridge")


class InstrumentedTradeRecorder:
    """Decorate a TradeRecorder with best-effort instrumentation hooks."""

    def __init__(
        self,
        inner_recorder,
        kit,
        *,
        strategy_id: str,
        strategy_type: str,
    ) -> None:
        self._inner = inner_recorder
        self._kit = kit
        self._strategy_id = strategy_id
        self._strategy_type = strategy_type

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
        meta: Optional[dict[str, Any]] = None,
        account_id: str = "default",
    ) -> str:
        if self._inner is not None:
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
        else:
            trade_id = self._fallback_trade_id(
                instrument=instrument,
                direction=direction,
                entry_ts=entry_ts,
                entry_price=entry_price,
            )

        if self._kit is None:
            return trade_id

        context = dict(meta or {})
        try:
            self._kit.log_entry(
                trade_id=trade_id,
                pair=instrument,
                side=direction,
                entry_price=float(entry_price),
                position_size=float(quantity),
                position_size_quote=float(entry_price) * float(quantity),
                entry_signal=str(
                    context.get("entry_signal")
                    or setup_tag
                    or entry_type
                    or "entry"
                ),
                entry_signal_id=str(context.get("entry_signal_id") or trade_id),
                entry_signal_strength=float(context.get("entry_signal_strength", 1.0)),
                strategy_params=context.get("strategy_params") or {},
                signal_factors=context.get("signal_factors"),
                filter_decisions=context.get("filter_decisions"),
                sizing_inputs=context.get("sizing_inputs"),
                portfolio_state=context.get("portfolio_state")
                or context.get("portfolio_state_at_entry"),
                session_type=str(context.get("session_type", "")),
                contract_month=str(context.get("contract_month", "")),
                margin_used_pct=_float_or_none(context.get("margin_used_pct")),
                concurrent_positions=_int_or_none(
                    context.get("concurrent_positions")
                    or context.get("concurrent_positions_at_entry")
                ),
                drawdown_pct=_float_or_none(context.get("drawdown_pct")),
                drawdown_tier=str(context.get("drawdown_tier", "")),
                drawdown_size_mult=_float_or_none(context.get("drawdown_size_mult")),
                bar_id=context.get("bar_id"),
                exchange_timestamp=context.get("exchange_timestamp") or entry_ts,
                expected_entry_price=_float_or_none(context.get("expected_entry_price")),
                entry_latency_ms=_int_or_none(context.get("entry_latency_ms")),
                signal_evolution=context.get("signal_evolution"),
                execution_timestamps=context.get("execution_timestamps"),
            )
        except Exception as exc:
            logger.warning(
                "Instrumentation failed on record_entry for %s/%s: %s",
                self._strategy_type,
                trade_id,
                exc,
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
        meta: Optional[dict[str, Any]] = None,
    ) -> None:
        if self._inner is not None:
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

        if self._kit is None:
            return

        context = dict(meta or {})
        try:
            self._kit.log_exit(
                trade_id=trade_id,
                exit_price=float(exit_price),
                exit_reason=exit_reason,
                fees_paid=_float_or_default(context.get("fees_paid"), 0.0),
                exchange_timestamp=context.get("exchange_timestamp") or exit_ts,
                expected_exit_price=_float_or_none(context.get("expected_exit_price")),
                exit_latency_ms=_int_or_none(context.get("exit_latency_ms")),
                mfe_r=_float_or_none(mfe_r),
                mae_r=_float_or_none(mae_r),
                mfe_price=_float_or_none(max_favorable_price),
                mae_price=_float_or_none(max_adverse_price),
                session_transitions=context.get("session_transitions"),
            )
        except Exception as exc:
            logger.warning(
                "Instrumentation failed on record_exit for %s/%s: %s",
                self._strategy_type,
                trade_id,
                exc,
            )

    async def record(self, data: dict) -> str:
        if self._inner is None:
            raise RuntimeError("Raw record() is unavailable without an inner recorder")
        return await self._inner.record(data)

    def _fallback_trade_id(
        self,
        *,
        instrument: str,
        direction: str,
        entry_ts: datetime,
        entry_price: Decimal,
    ) -> str:
        raw = "|".join(
            [
                self._strategy_id,
                instrument,
                direction,
                entry_ts.isoformat(),
                str(entry_price),
            ]
        )
        return sha256(raw.encode()).hexdigest()[:16]


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_or_default(value: Any, default: float) -> float:
    converted = _float_or_none(value)
    return converted if converted is not None else default


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
