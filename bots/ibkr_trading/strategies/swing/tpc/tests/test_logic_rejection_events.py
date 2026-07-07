"""Verify TPC core/logic.py emits SETUP_REJECTED records via rejection_collector."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from strategies.swing._shared.etf_core import ETFCoreState
from strategies.swing.tpc.config import TPCSymbolConfig
from strategies.swing.tpc.core import logic
from strategies.swing.tpc.models import Direction, RegimeGrade


@dataclass
class _Bar:
    timestamp: datetime
    open: float = 100.0
    high: float = 100.5
    low: float = 99.5
    close: float = 100.0
    volume: float = 1.0


@dataclass
class _BarInput:
    symbol: str = "QQQ"
    bar_15m: _Bar | None = None
    bars_15m: Any = None
    bars_30m: Any = None
    bars_1h: Any = None
    bars_4h: Any = None
    bars_daily: Any = None
    indicators: dict[str, float] = field(default_factory=dict)
    equity: float = 10_000.0
    timestamp: datetime | None = None
    decision_code: str = ""
    decision_details: dict[str, Any] = field(default_factory=dict)


def _cfg() -> TPCSymbolConfig:
    return TPCSymbolConfig(symbol="QQQ", longs_enabled=True, shorts_enabled=True)


def _now() -> datetime:
    return datetime(2026, 5, 9, 14, 0, tzinfo=timezone.utc)


def test_no_collector_keeps_existing_behaviour():
    """Without a collector, _evaluate_setup returns None and side-effects nothing."""
    bar_input = _BarInput(bar_15m=_Bar(timestamp=_now()))
    cfg = _cfg()
    state = ETFCoreState()
    with patch("strategies.swing.tpc.core.logic.gates.session_filter", return_value=False):
        result = logic._evaluate_setup(state, bar_input, cfg)
    assert result is None


def test_session_filter_blocked_emits_rejection():
    bar_input = _BarInput(bar_15m=_Bar(timestamp=_now()))
    cfg = _cfg()
    state = ETFCoreState()
    rejections: list[dict] = []
    with patch("strategies.swing.tpc.core.logic.gates.session_filter", return_value=False):
        logic._evaluate_setup(state, bar_input, cfg, rejection_collector=rejections)
    assert len(rejections) == 1
    assert rejections[0]["blocked_by"] == "session_filter_blocked"
    assert rejections[0]["lane"] == "prelude"
    assert rejections[0]["symbol"] == "QQQ"


def test_news_filter_blocked_emits_rejection():
    bar_input = _BarInput(bar_15m=_Bar(timestamp=_now()))
    cfg = _cfg()
    state = ETFCoreState()
    rejections: list[dict] = []
    with patch("strategies.swing.tpc.core.logic.gates.session_filter", return_value=True), \
         patch("strategies.swing.tpc.core.logic.gates.news_filter", return_value=False):
        logic._evaluate_setup(state, bar_input, cfg, rejection_collector=rejections)
    assert any(r["blocked_by"] == "news_filter_blocked" for r in rejections)


def test_regime_flat_emits_rejection():
    bar_input = _BarInput(bar_15m=_Bar(timestamp=_now()))
    cfg = _cfg()
    state = ETFCoreState()
    rejections: list[dict] = []
    with patch("strategies.swing.tpc.core.logic.gates.session_filter", return_value=True), \
         patch("strategies.swing.tpc.core.logic.gates.news_filter", return_value=True), \
         patch("strategies.swing.tpc.core.logic.gates.regime_direction",
               return_value=(Direction.FLAT, RegimeGrade.VALID, "neutral")):
        logic._evaluate_setup(state, bar_input, cfg, rejection_collector=rejections)
    assert any(r["blocked_by"] == "regime_flat" for r in rejections)


def test_longs_disabled_emits_rejection():
    bar_input = _BarInput(bar_15m=_Bar(timestamp=_now()))
    cfg = TPCSymbolConfig(symbol="QQQ", longs_enabled=False, shorts_enabled=True)
    state = ETFCoreState()
    rejections: list[dict] = []
    with patch("strategies.swing.tpc.core.logic.gates.session_filter", return_value=True), \
         patch("strategies.swing.tpc.core.logic.gates.news_filter", return_value=True), \
         patch("strategies.swing.tpc.core.logic.gates.regime_direction",
               return_value=(Direction.LONG, RegimeGrade.VALID, "ok")):
        logic._evaluate_setup(state, bar_input, cfg, rejection_collector=rejections)
    assert any(r["blocked_by"] == "longs_disabled" for r in rejections)


def test_shorts_disabled_emits_rejection():
    bar_input = _BarInput(bar_15m=_Bar(timestamp=_now()))
    cfg = TPCSymbolConfig(symbol="QQQ", longs_enabled=True, shorts_enabled=False)
    state = ETFCoreState()
    rejections: list[dict] = []
    with patch("strategies.swing.tpc.core.logic.gates.session_filter", return_value=True), \
         patch("strategies.swing.tpc.core.logic.gates.news_filter", return_value=True), \
         patch("strategies.swing.tpc.core.logic.gates.regime_direction",
               return_value=(Direction.SHORT, RegimeGrade.VALID, "ok")):
        logic._evaluate_setup(state, bar_input, cfg, rejection_collector=rejections)
    assert any(r["blocked_by"] == "shorts_disabled" for r in rejections)


def test_shorts_require_a_plus_emits_rejection():
    bar_input = _BarInput(bar_15m=_Bar(timestamp=_now()))
    cfg = TPCSymbolConfig(symbol="QQQ", shorts_enabled=True, shorts_require_a_plus=True)
    state = ETFCoreState()
    rejections: list[dict] = []
    with patch("strategies.swing.tpc.core.logic.gates.session_filter", return_value=True), \
         patch("strategies.swing.tpc.core.logic.gates.news_filter", return_value=True), \
         patch("strategies.swing.tpc.core.logic.gates.regime_direction",
               return_value=(Direction.SHORT, RegimeGrade.VALID, "ok")):
        logic._evaluate_setup(state, bar_input, cfg, rejection_collector=rejections)
    assert any(r["blocked_by"] == "shorts_require_a_plus" for r in rejections)


def test_pullback_not_detected_emits_lane_rejection():
    bar_input = _BarInput(bar_15m=_Bar(timestamp=_now()))
    cfg = _cfg()
    state = ETFCoreState()
    rejections: list[dict] = []
    with patch("strategies.swing.tpc.core.logic.gates.session_filter", return_value=True), \
         patch("strategies.swing.tpc.core.logic.gates.news_filter", return_value=True), \
         patch("strategies.swing.tpc.core.logic.gates.regime_direction",
               return_value=(Direction.LONG, RegimeGrade.VALID, "ok")), \
         patch("strategies.swing.tpc.core.logic.signals.detect_pullback", return_value=None):
        logic._evaluate_setup(state, bar_input, cfg, rejection_collector=rejections)
    assert any(r["blocked_by"] == "pullback_not_detected" for r in rejections)


def test_atr_4h_unavailable_emits_rejection():
    bar_input = _BarInput(bar_15m=_Bar(timestamp=_now()), indicators={"atr_4h": float("nan")})
    cfg = _cfg()
    state = ETFCoreState()
    rejections: list[dict] = []
    pullback = SimpleNamespace(
        pullback_type=SimpleNamespace(value="TYPE_A"),
        depth=0.4, low=99.0, high=100.5, value_hits=2, orderly=True,
    )
    with patch("strategies.swing.tpc.core.logic.gates.session_filter", return_value=True), \
         patch("strategies.swing.tpc.core.logic.gates.news_filter", return_value=True), \
         patch("strategies.swing.tpc.core.logic.gates.regime_direction",
               return_value=(Direction.LONG, RegimeGrade.VALID, "ok")), \
         patch("strategies.swing.tpc.core.logic.signals.detect_pullback", return_value=pullback), \
         patch("strategies.swing.tpc.core.logic.signals.check_confirmation",
               return_value=(True, ["vwap", "higher_low"])), \
         patch("strategies.swing.tpc.core.logic._confirmation_combo_allowed", return_value=True):
        logic._evaluate_setup(state, bar_input, cfg, rejection_collector=rejections)
    assert any(r["blocked_by"] == "atr_4h_unavailable" for r in rejections)


def test_setup_snapshot_meta_contains_new_keys_when_setup_succeeds():
    """If a setup is constructed end-to-end, meta now includes score, daily_has_room, orderly_pullback."""
    from strategies.swing.tpc.models import PullbackType
    bar_input = _BarInput(bar_15m=_Bar(timestamp=_now()), indicators={"atr_4h": 1.0})
    cfg = TPCSymbolConfig(
        symbol="QQQ", longs_enabled=True, shorts_enabled=True, score_a_min=10, score_b_min=8,
        asset_context_min_score=-1.0,
    )
    state = ETFCoreState()
    pullback = SimpleNamespace(
        pullback_type=PullbackType.TYPE_A,
        depth=0.4, low=99.0, high=100.5, value_hits=2, orderly=True,
    )

    with patch("strategies.swing.tpc.core.logic.gates.session_filter", return_value=True), \
         patch("strategies.swing.tpc.core.logic.gates.news_filter", return_value=True), \
         patch("strategies.swing.tpc.core.logic.gates.regime_direction",
               return_value=(Direction.LONG, RegimeGrade.A_PLUS, "ok")), \
         patch("strategies.swing.tpc.core.logic.gates.daily_room_filter", return_value=True), \
         patch("strategies.swing.tpc.core.logic.signals.detect_pullback", return_value=pullback), \
         patch("strategies.swing.tpc.core.logic.signals.check_confirmation",
               return_value=(True, ["vwap", "higher_low"])), \
         patch("strategies.swing.tpc.core.logic._confirmation_combo_allowed", return_value=True), \
         patch("strategies.swing.tpc.core.logic._entry_plan",
               return_value=(100.0, "MARKET", 0.0, 0.0, "market_next_bar")), \
         patch("strategies.swing.tpc.core.logic._initial_stop", return_value=99.0), \
         patch("strategies.swing.tpc.core.logic.stops.validate_stop", return_value=True), \
         patch("strategies.swing.tpc.core.logic.context.score_asset_context",
               return_value=(2.0, {"detail": 1})), \
         patch("strategies.swing.tpc.core.logic.allocator.score_setup", return_value=18.0), \
         patch("strategies.swing.tpc.core.logic.allocator.compute_risk_pct", return_value=0.0085), \
         patch("strategies.swing.tpc.core.logic.allocator.compute_position_size", return_value=10), \
         patch("strategies.swing.tpc.core.logic._daily_levels", return_value=[101.0, 99.0]):
        setup = logic._evaluate_setup(state, bar_input, cfg)

    assert setup is not None
    assert "score" in setup.meta
    assert "daily_has_room" in setup.meta
    assert "orderly_pullback" in setup.meta
    assert setup.meta["score"] == pytest.approx(18.0)
    assert setup.meta["daily_has_room"] is True
    assert setup.meta["orderly_pullback"] is True


def test_on_bar_common_wraps_rejections_into_events():
    """on_bar_common should append SETUP_REJECTED events when collector accumulates."""
    from strategies.swing._shared import etf_core
    from strategies.swing.tpc.core.state import TPCCoreState

    bar_input = _BarInput(bar_15m=_Bar(timestamp=_now()))
    cfg = _cfg()
    state = TPCCoreState()

    with patch("strategies.swing.tpc.core.logic.gates.session_filter", return_value=False):
        next_state, actions, events = etf_core.on_bar_common(
            state, bar_input, cfg,
            strategy_id="TPC",
            evaluate_setup=logic._evaluate_setup,
        )
    rejected = [e for e in events if e.code == "SETUP_REJECTED"]
    assert len(rejected) == 1
    assert rejected[0].details["blocked_by"] == "session_filter_blocked"
    # NO_SIGNAL still emitted (existing behavior)
    assert any(e.code == "NO_SIGNAL" for e in events)
