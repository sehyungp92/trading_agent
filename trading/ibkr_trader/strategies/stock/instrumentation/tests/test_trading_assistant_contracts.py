import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

from strategies.stock.instrumentation.src.error_logger import ErrorLogger
from strategies.stock.instrumentation.src.market_snapshot import MarketSnapshot, MarketSnapshotService
from strategies.stock.instrumentation.src.missed_opportunity import MissedOpportunityLogger
from strategies.stock.instrumentation.src.trade_logger import TradeLogger


def _config(tmpdir: str) -> dict:
    return {
        "bot_id": "contract_bot",
        "strategy_type": "strategy_alcb",
        "data_dir": tmpdir,
        "data_source_id": "ibkr_us_equities",
        "market_snapshots": {"interval_seconds": 60, "symbols": ["AAPL"]},
    }


def _snapshot_service() -> MagicMock:
    svc = MagicMock(spec=MarketSnapshotService)
    svc.capture_now.return_value = MarketSnapshot(
        snapshot_id="snap1",
        symbol="AAPL",
        timestamp=datetime.now(timezone.utc).isoformat(),
        bid=100.0,
        ask=100.1,
        mid=100.05,
        spread_bps=10.0,
        last_trade_price=100.05,
        volume_24h=1_500_000,
        atr_14=2.5,
        funding_rate=0.0,
        open_interest=0.0,
    )
    return svc


def test_trade_exit_event_contains_trading_assistant_aliases():
    tmpdir = tempfile.mkdtemp()
    logger = TradeLogger(_config(tmpdir), _snapshot_service())

    logger.log_entry(
        trade_id="trade_1",
        pair="AAPL",
        side="LONG",
        entry_price=100.0,
        position_size=10,
        position_size_quote=1000.0,
        entry_signal="orb_breakout",
        entry_signal_id="sig_1",
        entry_signal_strength=0.9,
        active_filters=["spread_gate"],
        passed_filters=["spread_gate"],
        strategy_params={"stop0": 99.0},
        exchange_timestamp=datetime.now(timezone.utc),
    )
    logger.log_exit(
        trade_id="trade_1",
        exit_price=101.0,
        exit_reason="TAKE_PROFIT",
        exchange_timestamp=datetime.now(timezone.utc),
    )

    trade_file = next((Path(tmpdir) / "trades").glob("*.jsonl"))
    exit_event = json.loads(trade_file.read_text(encoding="utf-8").splitlines()[-1])

    assert exit_event["bot_id"] == "contract_bot"
    assert exit_event["market_snapshot"]["symbol"] == "AAPL"
    assert exit_event["volume_24h"] == exit_event["volume_24h_at_entry"]
    assert exit_event["spread_at_entry"] == exit_event["spread_at_entry_bps"]
    assert exit_event["exit_time"]


def test_missed_event_contains_assistant_aliases_and_margin():
    tmpdir = tempfile.mkdtemp()
    logger = MissedOpportunityLogger(_config(tmpdir), _snapshot_service())

    logger.log_missed(
        pair="AAPL",
        side="LONG",
        signal="orb_breakout",
        signal_id="sig_1",
        signal_strength=0.7,
        blocked_by="spread_gate",
        filter_decisions=[
            {
                "filter_name": "spread_gate",
                "threshold": 0.35,
                "actual_value": 0.42,
                "passed": False,
                "margin_pct": 20.0,
            }
        ],
        exchange_timestamp=datetime.now(timezone.utc),
    )

    missed_file = next((Path(tmpdir) / "missed").glob("*.jsonl"))
    event = json.loads(missed_file.read_text(encoding="utf-8").splitlines()[0])

    assert event["bot_id"] == "contract_bot"
    assert event["hypothetical_entry"] == event["hypothetical_entry_price"]
    assert event["confidence"] == 0.0
    assert event["margin_pct"] == 20.0


def test_error_logger_writes_triage_compatible_payload_and_counts_recent():
    tmpdir = tempfile.mkdtemp()
    logger = ErrorLogger(_config(tmpdir))

    try:
        raise ValueError("feed disconnected")
    except ValueError as exc:
        logger.log_exception(
            "market_data_disconnect",
            exc,
            severity="high",
            category="connection_lost",
            context={"component": "scanner_loop"},
        )

    error_file = next((Path(tmpdir) / "errors").glob("*.jsonl"))
    event = json.loads(error_file.read_text(encoding="utf-8").splitlines()[0])

    assert event["bot_id"] == "contract_bot"
    assert event["error_type"] == "market_data_disconnect"
    assert event["severity"] == "high"
    assert event["category"] == "connection_lost"
    assert "feed disconnected" in event["message"]
    assert event["stack_trace"]
    assert logger.count_recent() == 1
