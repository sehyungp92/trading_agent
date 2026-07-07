from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from libs.oms.models.events import OMSEvent, OMSEventType
from strategies.momentum.instrumentation.src.bootstrap import (
    InstrumentationManager as MomentumInstrumentationManager,
)
from strategies.stock.instrumentation.src.bootstrap import (
    InstrumentationManager as StockInstrumentationManager,
)
from strategies.swing.instrumentation.src.context import InstrumentationContext
from tests.unit.test_risk_decision_contract import (
    _lineage,
    test_momentum_instrumentation_persists_risk_halt_for_sidecar as _momentum_risk_halt,
    test_stock_instrumentation_persists_risk_halt_for_sidecar as _stock_risk_halt,
    test_swing_context_persists_risk_halt_for_sidecar as _swing_risk_halt,
)


def test_stock_risk_halt_jsonl_schema_lineage_and_sidecar(tmp_path) -> None:
    _stock_risk_halt(tmp_path)


def test_momentum_risk_halt_jsonl_schema_lineage_and_sidecar(tmp_path) -> None:
    _momentum_risk_halt(tmp_path)


@pytest.mark.asyncio
async def test_swing_risk_halt_jsonl_schema_lineage_and_sidecar(tmp_path) -> None:
    await _swing_risk_halt(tmp_path)


def _portfolio_halt_event() -> OMSEvent:
    return OMSEvent(
        event_type=OMSEventType.RISK_HALT,
        timestamp=datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc),
        strategy_id="",
        payload={"reason": "portfolio_halted", "source": "reconciliation"},
    )


def _assert_portfolio_halt(path, *, family_id: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["event_type"] == "risk_halt"
    assert payload["schema_version"] == "risk_halt_v1"
    assert payload["halt_scope"] == "portfolio"
    assert payload["strategy_id"] == ""
    assert payload["lineage"]["strategy_id"] == ""
    assert payload["family_id"] == family_id
    assert payload["reason"] == "portfolio_halted"


def test_stock_portfolio_scoped_risk_halt_does_not_inherit_strategy_id(tmp_path) -> None:
    manager = object.__new__(StockInstrumentationManager)
    manager._config = {"data_dir": str(tmp_path)}
    manager.lineage = _lineage()

    manager._handle_risk_halt(_portfolio_halt_event())

    _assert_portfolio_halt(next((tmp_path / "risk_halts").glob("risk_halts_*.jsonl")), family_id="stock")


def test_momentum_portfolio_scoped_risk_halt_does_not_inherit_strategy_id(tmp_path) -> None:
    manager = object.__new__(MomentumInstrumentationManager)
    manager._config = {"data_dir": str(tmp_path)}
    manager.lineage = replace(_lineage(), family_id="momentum", strategy_id="NQ_REGIME")

    manager._handle_risk_halt(_portfolio_halt_event())

    _assert_portfolio_halt(next((tmp_path / "risk_halts").glob("risk_halts_*.jsonl")), family_id="momentum")


def test_swing_portfolio_scoped_risk_halt_does_not_inherit_strategy_id(tmp_path) -> None:
    ctx = object.__new__(InstrumentationContext)
    ctx.data_dir = str(tmp_path)
    ctx.lineage = replace(_lineage(), family_id="swing", strategy_id="TPC")

    ctx._handle_risk_halt(_portfolio_halt_event())

    _assert_portfolio_halt(next((tmp_path / "risk_halts").glob("risk_halts_*.jsonl")), family_id="swing")
