from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

from libs.instrumentation.lineage import LineageContext
from strategies.momentum.instrumentation.src.facade import InstrumentationKit as MomentumKit
from strategies.stock.instrumentation.src.facade import InstrumentationKit as StockKit
from strategies.swing.instrumentation.src.kit import InstrumentationKit as SwingKit


def _lineage(strategy_id: str, family_id: str) -> LineageContext:
    return LineageContext(
        bot_id=f"{family_id}_bot",
        strategy_id=strategy_id,
        family_id=family_id,
        strategy_version=f"{strategy_id}.1",
        config_version="cfg_1",
        deployment_id="dep_1",
        parameter_set_id="param_1",
        code_sha="abc123",
        trace_id="trace_1",
    )


def _read_decision(tmp_path):
    [path] = (tmp_path / "decisions").glob("*.jsonl")
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


def test_stock_facade_writes_live_decision_event_without_pg_store(tmp_path) -> None:
    manager = SimpleNamespace(
        _config={"data_dir": str(tmp_path), "bot_id": "stock_bot"},
        _strategy_id="IARIC_v1",
        _pg_store=None,
        lineage=_lineage("IARIC_v1", "stock"),
    )
    kit = StockKit(manager, strategy_type="strategy_iaric")

    kit._record_strategy_decision(
        "ENTRY:pullback",
        {"pair": "AAPL", "bar_id": "bar_1", "emitted_actions": [{"type": "intent"}]},
        datetime(2026, 6, 3, 14, 30, tzinfo=timezone.utc),
    )

    event = _read_decision(tmp_path)
    assert event["event_type"] == "decision_event"
    assert event["strategy_id"] == "IARIC_v1"
    assert event["code"] == "ENTRY:pullback"
    assert event["bar_id"] == "bar_1"
    assert event["emitted_actions"] == [{"type": "intent"}]
    assert event["deployment_id"] == "dep_1"


def test_momentum_facade_writes_live_decision_event_without_pg_store(tmp_path) -> None:
    manager = SimpleNamespace(
        _config={"data_dir": str(tmp_path), "bot_id": "momentum_bot"},
        _strategy_id="NQ_REGIME",
        _pg_store=None,
        lineage=_lineage("NQ_REGIME", "momentum"),
    )
    kit = MomentumKit(manager, strategy_type="nq_regime")

    kit._record_strategy_decision("FILTER:pass", {"pair": "MNQ", "bar_id": "bar_2"})

    event = _read_decision(tmp_path)
    assert event["event_type"] == "decision_event"
    assert event["strategy_id"] == "NQ_REGIME"
    assert event["code"] == "FILTER:pass"
    assert event["bar_id"] == "bar_2"


def test_swing_kit_writes_live_decision_event_without_pg_store(tmp_path) -> None:
    ctx = SimpleNamespace(
        data_dir=str(tmp_path),
        bot_id="swing_bot",
        pg_store=None,
        lineage=_lineage("ATRSS", "swing"),
    )
    kit = SwingKit(ctx, strategy_id="ATRSS")

    kit._record_strategy_decision("NO_SIGNAL", {"pair": "GLD", "bar_id": "bar_3"})

    event = _read_decision(tmp_path)
    assert event["event_type"] == "decision_event"
    assert event["strategy_id"] == "ATRSS"
    assert event["code"] == "NO_SIGNAL"
    assert event["bar_id"] == "bar_3"
