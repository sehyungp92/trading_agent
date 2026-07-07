from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

from backtests.shared.parity.legacy_result_outputs import (
    decision_stream_from_records,
    decision_stream_from_trades,
    merge_decision_streams,
    trade_outcomes_from_records,
)

UTC = timezone.utc


def test_decision_stream_from_records_normalizes_strategy_specific_records() -> None:
    stream = decision_stream_from_records(
        [
            {
                "timestamp": datetime(2026, 4, 25, 13, 0, tzinfo=UTC),
                "symbol": "MNQ",
                "decision": "placed",
                "session_block": "RTH",
            },
            {
                "timestamp": datetime(2026, 4, 25, 13, 5, tzinfo=UTC),
                "symbol": "NQ",
                "passed_all": False,
                "first_block_reason": "slope",
            },
            {
                "timestamp": datetime(2026, 4, 25, 13, 10, tzinfo=UTC),
                "symbol": "MSFT",
                "to_state": "READY",
                "reason": "delayed_confirm",
                "date": date(2026, 4, 25),
            },
            {
                "timestamp": datetime(2026, 4, 25, 13, 15, tzinfo=UTC),
                "symbol": "MNQ",
                "setup_class": "M",
                "entry_stop": 20260.0,
                "stop0": 20200.0,
            },
        ],
        timeframe="5m",
    )

    assert [event["code"] for event in stream] == [
        "ENTRY_REQUESTED",
        "SIGNAL_FILTERED",
        "FSM_READY",
        "SETUP_DETECTED",
    ]
    assert stream[2]["details"]["date"] == "2026-04-25"


def test_decision_stream_and_trade_outcomes_handle_alternate_trade_shapes() -> None:
    trade = SimpleNamespace(
        symbol="MNQ",
        direction=1,
        setup_time=datetime(2026, 4, 25, 13, 0, tzinfo=UTC),
        entry_time=datetime(2026, 4, 25, 13, 5, tzinfo=UTC),
        exit_time=datetime(2026, 4, 25, 14, 0, tzinfo=UTC),
        avg_entry=20250.0,
        exit_price=20310.0,
        entry_contracts=2,
        adjusted_pnl=480.0,
        commission=8.0,
        exit_reason="TRAIL",
    )

    stream = decision_stream_from_trades([trade], timeframe="5m")
    outcomes = trade_outcomes_from_records([trade])

    assert [event["code"] for event in stream] == ["ENTRY_FILLED", "EXIT_FILLED"]
    assert stream[0]["details"]["entry_price"] == 20250.0
    assert stream[0]["details"]["qty"] == 2
    assert outcomes[0]["qty"] == 2
    assert outcomes[0]["net_pnl"] == 480.0
    assert outcomes[0]["gross_pnl"] == 488.0


def test_merge_decision_streams_sorts_by_timestamp_code_and_symbol() -> None:
    merged = merge_decision_streams(
        [{"code": "EXIT_FILLED", "ts": "2026-04-25T13:05:00+00:00", "symbol": "B"}],
        [{"code": "ENTRY_FILLED", "ts": "2026-04-25T13:05:00+00:00", "symbol": "A"}],
    )

    assert [(event["code"], event["symbol"]) for event in merged] == [
        ("ENTRY_FILLED", "A"),
        ("EXIT_FILLED", "B"),
    ]
