from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from backtests.shared.parity.trade_outcomes import normalize_trade_outcome, normalize_trade_outcome_stream

UTC = timezone.utc


@dataclass
class _Trade:
    symbol: str = "MNQ"
    direction: str = "SHORT"
    qty: int = 2
    entry_time: datetime = datetime(2026, 4, 25, 10, 0, tzinfo=UTC)
    exit_time: datetime = datetime(2026, 4, 25, 11, 0, tzinfo=UTC)
    pnl: float = 125.5
    commission: float = 2.5
    exit_type: str = "stop"
    extra: dict[str, object] = field(default_factory=lambda: {"nested": {"b": 2, "a": 1}})


def test_normalize_trade_outcome_handles_dataclass_records() -> None:
    outcome = normalize_trade_outcome(_Trade())

    assert outcome.symbol == "MNQ"
    assert outcome.side == "SELL"
    assert outcome.qty == 2
    assert outcome.net_pnl == 125.5
    assert outcome.gross_pnl == 128.0
    assert outcome.exit_reason == "stop"
    assert outcome.metadata["extra"] == {"nested": {"a": 1, "b": 2}}


def test_normalize_trade_outcome_stream_serializes_timestamps_and_metadata() -> None:
    stream = normalize_trade_outcome_stream([
        {
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 100,
            "entry_time": datetime(2026, 4, 25, 13, 0, tzinfo=UTC),
            "exit_time": datetime(2026, 4, 25, 14, 0, tzinfo=UTC),
            "net_pnl": 87.25,
            "commission": 1.25,
            "exit_reason": "tp1",
            "context": {"route": "SMART", "session": "RTH"},
        }
    ])

    assert stream == [
        {
            "commission": 1.25,
            "decision_ts": None,
            "entry_ts": "2026-04-25T13:00:00+00:00",
            "exit_reason": "tp1",
            "exit_ts": "2026-04-25T14:00:00+00:00",
            "fill_ts": "2026-04-25T13:00:00+00:00",
            "gross_pnl": 88.5,
            "metadata": {"context": {"route": "SMART", "session": "RTH"}},
            "net_pnl": 87.25,
            "qty": 100,
            "side": "BUY",
            "symbol": "AAPL",
        }
    ]


def test_normalize_trade_outcome_coerces_iso_timestamps_and_keeps_canonical_fields_out_of_metadata() -> None:
    outcome = normalize_trade_outcome(
        {
            "symbol": "MSFT",
            "direction": "LONG",
            "quantity": 50,
            "entry_time": "2026-04-25T14:00:00+00:00",
            "filled_at": "2026-04-25T14:01:00+00:00",
            "exit_time": "2026-04-25T15:00:00+00:00",
            "gross_pnl": 120.0,
            "commission": 2.0,
            "exit_reason": "target",
            "context": {"route": "SMART"},
        }
    )

    assert outcome.side == "BUY"
    assert outcome.entry_ts == datetime(2026, 4, 25, 14, 0, tzinfo=UTC)
    assert outcome.fill_ts == datetime(2026, 4, 25, 14, 1, tzinfo=UTC)
    assert outcome.exit_ts == datetime(2026, 4, 25, 15, 0, tzinfo=UTC)
    assert outcome.metadata == {"context": {"route": "SMART"}}
