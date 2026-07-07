from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from libs.instrumentation.lineage import LineageContext
from libs.oms.services.factory import (
    _make_portfolio_rule_logger,
    _make_reconciliation_lifecycle_writer,
)
from strategies.stock.instrumentation.src.market_snapshot import MarketSnapshot
from strategies.stock.instrumentation.src.missed_opportunity import MissedOpportunityLogger
from strategies.stock.instrumentation.src.order_logger import OrderLogger
from strategies.stock.instrumentation.src.trade_logger import TradeLogger


class _SnapshotService:
    def capture_now(self, symbol: str) -> MarketSnapshot:
        return MarketSnapshot(
            snapshot_id=f"snapshot-{symbol}",
            symbol=symbol,
            timestamp="2026-06-21T14:30:00+00:00",
            bid=99.95,
            ask=100.05,
            mid=100.0,
            spread_bps=10.0,
            last_trade_price=100.0,
            volume_24h=1_000_000,
            atr_14=1.5,
        )


def test_ibkr_runtime_emitters_persist_assistant_lineage(tmp_path: Path) -> None:
    lineage = _lineage()
    config = {
        "bot_id": lineage.bot_id,
        "strategy_id": lineage.strategy_id,
        "data_dir": str(tmp_path),
        "data_source_id": "ptg4_emitter_test",
        "lineage": lineage,
    }
    snapshot_service = _SnapshotService()

    trade_logger = TradeLogger(config, snapshot_service, strategy_type="stock")
    trade_logger.log_entry(
        trade_id="trade-1",
        pair="AAPL",
        side="LONG",
        entry_price=100.0,
        position_size=10,
        position_size_quote=1_000,
        entry_signal="breakout",
        entry_signal_id="signal-1",
        entry_signal_strength=0.8,
        active_filters=["quality"],
        passed_filters=["quality"],
        strategy_params={"stop0": 98.0},
        exchange_timestamp=_ts(),
    )
    trade_logger.log_exit("trade-1", exit_price=101.0, exit_reason="TARGET", exchange_timestamp=_ts())

    missed_logger = MissedOpportunityLogger(config, snapshot_service)
    missed_logger.log_missed(
        pair="AAPL",
        side="LONG",
        signal="breakout",
        signal_id="missed-signal-1",
        signal_strength=0.75,
        blocked_by="portfolio_rule",
        exchange_timestamp=_ts(),
    )

    order_logger = OrderLogger(config, strategy_type="stock")
    order_logger.log_order(
        order_id="order-1",
        pair="AAPL",
        side="LONG",
        order_type="LIMIT",
        status="FILLED",
        requested_qty=10,
        filled_qty=10,
        requested_price=100.0,
        fill_price=100.05,
        exchange_timestamp=_ts(),
    )

    fill_writer = _make_reconciliation_lifecycle_writer(str(tmp_path), lineage=lambda: lineage)
    fill_writer({
        "lifecycle_action": "inferred_fill",
        "status": "observed",
        "phase": "fill_reconciliation",
        "source": "broker_fill",
        "details": {"strategy_id": lineage.strategy_id, "fill_id": "fill-1", "order_id": "order-1"},
        "timestamp": _ts(),
    })

    portfolio_rule_logger = _make_portfolio_rule_logger(
        data_dir=str(tmp_path),
        family_id="stock",
        lineage=lambda: lineage,
    )
    portfolio_rule_logger({
        "rule": "directional_cap",
        "approved": True,
        "strategy_id": lineage.strategy_id,
        "direction": "LONG",
        "symbol": "AAPL",
    })

    paths = {
        "trade": _only(tmp_path / "trades", "trades_*.jsonl"),
        "missed_opportunity": _only(tmp_path / "missed", "missed_*.jsonl"),
        "order": _only(tmp_path / "orders", "orders_*.jsonl"),
        "fill": _only(tmp_path / "inferred_fills", "inferred_fills_*.jsonl"),
        "portfolio_rule": _only(tmp_path / "portfolio_rules", "rules_*.jsonl"),
    }
    for event_class, path in paths.items():
        records = _records(path)
        assert records, event_class
        for record in records:
            _assert_assistant_trace(record, lineage)


def _lineage() -> LineageContext:
    return LineageContext(
        bot_id="bot1",
        strategy_id="strat1",
        family_id="stock",
        portfolio_id="paper",
        account_alias="paper_ibkr",
        strategy_version="strat1.v1",
        config_version="cfg-v1",
        portfolio_config_version="pcfg-v1",
        risk_config_version="risk-v1",
        allocation_version="alloc-v1",
        strategy_registry_version="registry-v1",
        deployment_id="deploy-acceptance-1",
        parameter_set_id="param-v1",
        code_sha="abc123",
        trace_id="trace-acceptance-1",
        proposal_ids=("proposal-acceptance-1",),
        source_weekly_signal_ids=("weekly-signal-breakout",),
        strategy_change_record_ids=("change-acceptance-1",),
        candidate_ids=("acceptance-candidate",),
    )


def _assert_assistant_trace(record: dict[str, Any], lineage: LineageContext) -> None:
    nested = record.get("lineage", {})
    assert record["bot_id"] == lineage.bot_id
    assert record["strategy_id"] == lineage.strategy_id
    assert record["proposal_ids"] == list(lineage.proposal_ids)
    assert record["source_weekly_signal_ids"] == list(lineage.source_weekly_signal_ids)
    assert record["strategy_change_record_ids"] == list(lineage.strategy_change_record_ids)
    assert record["candidate_ids"] == list(lineage.candidate_ids)
    assert nested["proposal_ids"] == list(lineage.proposal_ids)
    assert nested["source_weekly_signal_ids"] == list(lineage.source_weekly_signal_ids)
    assert nested["strategy_change_record_ids"] == list(lineage.strategy_change_record_ids)
    assert nested["candidate_ids"] == list(lineage.candidate_ids)


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _only(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    assert len(matches) == 1
    return matches[0]


def _ts() -> datetime:
    return datetime(2026, 6, 21, 14, 30, tzinfo=timezone.utc)
