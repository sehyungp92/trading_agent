from __future__ import annotations

import json
from pathlib import Path

from libs.instrumentation.lineage import LineageContext
from strategies.momentum.instrumentation.src.trade_logger import TradeLogger as MomentumTradeLogger
from strategies.stock.instrumentation.src.missed_opportunity import (
    MissedOpportunityLogger as StockMissedOpportunityLogger,
)
from strategies.stock.instrumentation.src.trade_logger import TradeLogger as StockTradeLogger
from strategies.swing.instrumentation.src.missed_opportunity import (
    MissedOpportunityLogger as SwingMissedOpportunityLogger,
)
from strategies.swing.instrumentation.src.trade_logger import TradeLogger as SwingTradeLogger


def _lineage() -> LineageContext:
    return LineageContext(
        bot_id="test_bot",
        strategy_id="TPC",
        family_id="swing",
        portfolio_id="paper_default",
        strategy_version="TPC.1",
        config_version="cfg_1",
        portfolio_config_version="pcfg_1",
        risk_config_version="risk_1",
        allocation_version="alloc_1",
        strategy_registry_version="registry_1",
        deployment_id="dep_1",
        parameter_set_id="param_1",
        code_sha="abc123",
        trace_id="trace_1",
        account_alias="paper_ibkr_1",
    )


def _error_payload(root: Path) -> dict:
    path = next((root / "errors").glob("instrumentation_errors_*.jsonl"))
    return json.loads(path.read_text(encoding="utf-8").strip())


def _assert_enriched_error(payload: dict, *, component: str, context_key: str) -> None:
    assert payload["event_type"] == "error"
    assert payload["schema_version"] == "error_event_v2"
    assert payload["scope"] == "strategy"
    assert payload["component"] == component
    assert payload["method"] == "fallback"
    assert payload["message"] == "boom"
    assert payload["error_type"] == "ValueError"
    assert payload["context"][context_key] == "ctx_1"
    assert payload["deployment_id"] == "dep_1"
    assert payload["parameter_set_id"] == "param_1"
    assert payload["param_set_id"] == "param_1"
    assert payload["lineage"]["trace_id"] == "trace_1"
    assert "lineage_gaps" not in payload


def test_momentum_trade_fallback_error_uses_enriched_contract(tmp_path) -> None:
    logger = object.__new__(MomentumTradeLogger)
    logger.data_dir = tmp_path / "trades"
    logger.data_dir.mkdir(parents=True)
    logger._lineage = _lineage()

    logger._write_error("fallback", "ctx_1", ValueError("boom"))

    _assert_enriched_error(
        _error_payload(tmp_path),
        component="trade_logger",
        context_key="trade_id",
    )


def test_stock_trade_without_error_logger_uses_enriched_contract(tmp_path) -> None:
    logger = object.__new__(StockTradeLogger)
    logger.data_dir = tmp_path / "trades"
    logger.data_dir.mkdir(parents=True)
    logger._lineage = _lineage()
    logger._error_logger = None

    logger._write_error("fallback", "ctx_1", ValueError("boom"))

    _assert_enriched_error(
        _error_payload(tmp_path),
        component="trade_logger",
        context_key="trade_id",
    )


def test_swing_trade_fallback_error_uses_enriched_contract(tmp_path) -> None:
    logger = object.__new__(SwingTradeLogger)
    logger.data_dir = tmp_path / "trades"
    logger.data_dir.mkdir(parents=True)
    logger._lineage = _lineage()

    logger._write_error("fallback", "ctx_1", ValueError("boom"))

    _assert_enriched_error(
        _error_payload(tmp_path),
        component="trade_logger",
        context_key="trade_id",
    )


def test_stock_missed_without_error_logger_uses_enriched_contract(tmp_path) -> None:
    logger = object.__new__(StockMissedOpportunityLogger)
    logger.data_dir = tmp_path / "missed"
    logger.data_dir.mkdir(parents=True)
    logger._lineage = _lineage()
    logger._error_logger = None

    logger._write_error("fallback", "ctx_1", ValueError("boom"))

    _assert_enriched_error(
        _error_payload(tmp_path),
        component="missed_opportunity",
        context_key="context",
    )


def test_swing_missed_fallback_error_uses_enriched_contract(tmp_path) -> None:
    logger = object.__new__(SwingMissedOpportunityLogger)
    logger.data_dir = tmp_path / "missed"
    logger.data_dir.mkdir(parents=True)
    logger._lineage = _lineage()

    logger._write_error("fallback", "ctx_1", ValueError("boom"))

    _assert_enriched_error(
        _error_payload(tmp_path),
        component="missed_opportunity",
        context_key="context",
    )
