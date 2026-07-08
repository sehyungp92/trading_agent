"""Trade recording service for completed trade telemetry.

Call on trade entry and exit to record trade details and MAE/MFE.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from ..oms.persistence.postgres import PgStore
from ..oms.persistence.schema import TradeRow, TradeMarksRow


def _serialize(v):
    """Convert non-JSON-native types for meta_json."""
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, datetime):
        return v.isoformat()
    return v


class TradeRecorder:
    """Records completed trades with telemetry."""

    def __init__(self, store: PgStore):
        self._store = store

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
        """Record trade entry. Returns trade_id for later exit recording."""
        trade_id = uuid.uuid4().hex[:16]

        row = TradeRow(
            trade_id=trade_id,
            strategy_id=strategy_id,
            account_id=account_id,
            instrument_symbol=instrument,
            direction=direction,
            quantity=quantity,
            entry_ts=entry_ts,
            entry_price=entry_price,
            setup_tag=setup_tag,
            entry_type=entry_type,
            meta_json=json.dumps(meta) if meta else "{}",
        )
        await self._store.save_trade(row)
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
        # MAE/MFE data
        mae_r: Decimal = None,
        mfe_r: Decimal = None,
        duration_seconds: int = None,
        duration_bars: int = None,
        max_adverse_price: Decimal = None,
        max_favorable_price: Decimal = None,
    ) -> None:
        """Record trade exit with optional MAE/MFE metrics."""
        # Update trade record
        await self._store._pool.execute(
            """
            UPDATE trades SET
                exit_ts = $2,
                exit_price = $3,
                exit_reason = $4,
                realized_r = $5,
                realized_usd = $6,
                notes = $7
            WHERE trade_id = $1
            """,
            trade_id,
            exit_ts,
            exit_price,
            exit_reason,
            realized_r,
            realized_usd,
            notes,
        )

        # Record trade marks if provided
        if mae_r is not None or mfe_r is not None:
            marks = TradeMarksRow(
                trade_id=trade_id,
                duration_seconds=duration_seconds,
                duration_bars=duration_bars,
                mae_r=mae_r,
                mfe_r=mfe_r,
                max_adverse_price=max_adverse_price,
                max_favorable_price=max_favorable_price,
            )
            await self._store.save_trade_marks(marks)

    async def record(self, data: dict) -> str:
        """Record a complete trade (entry + exit) in one call.

        Used by strategies that build a full TradeRecord at exit time.
        """
        _KNOWN = {
            "trade_id", "symbol", "direction", "shares", "entry_ts",
            "exit_ts", "entry_price", "exit_price", "r_multiple",
            "realized_pnl", "entry_type", "strategy_id", "account_id",
            "exit_reason", "stop_price",
        }

        trade_id = data.get("trade_id") or uuid.uuid4().hex[:16]
        row = TradeRow(
            trade_id=trade_id,
            strategy_id=data.get("strategy_id", ""),
            account_id=data.get("account_id", "default"),
            instrument_symbol=data.get("symbol", ""),
            direction=str(data.get("direction", "")),
            quantity=data.get("shares", 0),
            entry_ts=data.get("entry_ts"),
            entry_price=Decimal(str(data.get("entry_price", 0))),
            exit_ts=data.get("exit_ts"),
            exit_price=Decimal(str(data.get("exit_price", 0))),
            exit_reason=data.get("exit_reason", ""),
            realized_r=Decimal(str(data.get("r_multiple", 0))),
            realized_usd=Decimal(str(data.get("realized_pnl", 0))),
            setup_tag=str(data.get("entry_type", "")),
            entry_type=str(data.get("entry_type", "")),
            meta_json=json.dumps(
                {k: _serialize(v) for k, v in data.items() if k not in _KNOWN},
                default=str,
            ),
        )
        await self._store.save_trade(row)
        return trade_id
