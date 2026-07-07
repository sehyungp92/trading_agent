"""Tests for the TPC instrumentation adapter."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pytest

from strategies.swing._shared.etf_core import (
    ETFCoreState,
    ETFFill,
    ETFOrderUpdate,
    ETFPosition,
    SetupSnapshot,
)
from strategies.swing.tpc import instrumentation_adapter as adapter
from strategies.swing.tpc.config import TPCSymbolConfig
from strategies.swing.tpc.models import Direction


class _RecordingKit:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def _record(self, name: str, **kwargs: Any) -> None:
        self.calls.append((name, kwargs))

    def log_entry(self, **kwargs: Any) -> None:
        self._record("log_entry", **kwargs)

    def log_exit(self, **kwargs: Any) -> None:
        self._record("log_exit", **kwargs)

    def log_stop_adjustment(self, **kwargs: Any) -> None:
        self._record("log_stop_adjustment", **kwargs)

    def log_missed(self, **kwargs: Any) -> None:
        self._record("log_missed", **kwargs)

    def on_indicator_snapshot(self, **kwargs: Any) -> None:
        self._record("on_indicator_snapshot", **kwargs)

    def on_filter_decision(self, **kwargs: Any) -> None:
        self._record("on_filter_decision", **kwargs)

    def on_order_event(self, **kwargs: Any) -> None:
        self._record("on_order_event", **kwargs)


@dataclass
class _StubBar:
    timestamp: datetime
    open: float = 100.0
    high: float = 101.0
    low: float = 99.0
    close: float = 100.5
    volume: float = 1.0


@dataclass
class _StubBarInput:
    symbol: str
    bar_15m: _StubBar
    indicators: dict[str, float] = field(default_factory=dict)


def _make_cfg(symbol: str = "QQQ") -> TPCSymbolConfig:
    return TPCSymbolConfig(symbol=symbol, score_a_min=14, score_b_min=11, confirmation_required=2)


def _make_setup(symbol: str = "QQQ", score: float = 18.0) -> SetupSnapshot:
    ts = datetime(2026, 5, 9, 14, 0, tzinfo=timezone.utc)
    return SetupSnapshot(
        setup_id=f"TPC-{symbol}-{int(ts.timestamp())}",
        strategy_id="TPC",
        symbol=symbol,
        direction=Direction.LONG,
        grade="a_plus",
        setup_type="TYPE_A",
        entry_model="market_next_bar",
        state="entry_ready",
        created_ts=ts,
        entry_price=100.0,
        stop_price=99.0,
        qty=10,
        score=score,
        risk_pct=0.0085,
        t1_r=1.5,
        t1_partial_pct=0.45,
        t2_r=2.75,
        t2_partial_pct=0.275,
        target_price=101.5,
        meta={
            "confirmations": ["vwap", "higher_low"],
            "depth": 0.40,
            "rr": 3.0,
            "atr_4h": 1.2,
            "asset_context_score": 1.5,
            "setup_lane": "primary",
            "pullback_timeframe": "1h",
            "score": score,
            "daily_has_room": True,
            "orderly_pullback": True,
        },
    )


def _make_position(symbol: str = "QQQ", direction: Direction = Direction.LONG) -> ETFPosition:
    return ETFPosition(
        setup_id=f"TPC-{symbol}-1",
        symbol=symbol,
        direction=direction,
        qty_open=10,
        qty_initial=10,
        entry_price=100.0,
        current_stop=99.0,
        initial_stop=99.0,
        entry_ts=datetime(2026, 5, 9, 14, 0, tzinfo=timezone.utc),
        risk_per_share=1.0,
        setup_type="TYPE_A",
        grade="a_plus",
        entry_model="market_next_bar",
        score=18.0,
        stop_order_id=f"{symbol}-stop-1",
        mfe_price=102.5,
        mae_price=99.5,
        bars_held_15m=8,
        meta={"setup_lane": "primary"},
    )


def _make_fill(symbol: str = "QQQ", role: str = "entry", price: float = 100.0, qty: int = 10) -> ETFFill:
    return ETFFill(
        oms_order_id="oms-1",
        fill_price=price,
        fill_qty=qty,
        symbol=symbol,
        fill_time=datetime(2026, 5, 9, 14, 15, tzinfo=timezone.utc),
        commission=0.5,
        order_role=role,
    )


def test_param_set_id_is_8_char_md5_and_stable():
    cfg = _make_cfg()
    pid_a = adapter.param_set_id(cfg)
    pid_b = adapter.param_set_id(cfg)
    assert pid_a == pid_b
    assert len(pid_a) == 8
    assert all(c in "0123456789abcdef" for c in pid_a)


def test_route_entry_emits_log_entry_with_full_payload():
    kit = _RecordingKit()
    cfg = _make_cfg()
    setup = _make_setup()
    fill = _make_fill(role="entry")
    state = ETFCoreState()

    adapter.route_entry(kit, setup, fill, cfg, state)

    assert len(kit.calls) == 1
    name, kwargs = kit.calls[0]
    assert name == "log_entry"
    assert kwargs["trade_id"] == setup.setup_id
    assert kwargs["pair"] == "QQQ"
    assert kwargs["side"] == "LONG"
    assert kwargs["entry_price"] == 100.0
    assert kwargs["entry_signal"] == "TYPE_A"
    assert kwargs["entry_signal_id"] == setup.setup_id
    assert 0.0 <= kwargs["entry_signal_strength"] <= 1.0
    assert kwargs["expected_entry_price"] == setup.entry_price
    factor_names = {f["factor_name"] for f in kwargs["signal_factors"]}
    assert factor_names == {
        "grade", "setup_type", "score", "confirmations_count", "asset_context_score",
        "depth_atr", "rr_planned", "daily_has_room", "orderly_pullback",
    }
    assert kwargs["strategy_params"]["param_set_id"] == adapter.param_set_id(cfg)
    assert kwargs["strategy_params"]["lane"] == "primary"
    assert kwargs["sizing_inputs"]["sizing_model"] == "tpc_score_band"


@pytest.mark.parametrize(
    "event_code,reason,expected",
    [
        ("STOP_FILLED", "STOP", "STOP_LOSS"),
        ("EXIT_FILLED", "T1", "TAKE_PROFIT_T1"),
        ("EXIT_FILLED", "T2", "TAKE_PROFIT_T2"),
        ("EXIT_FILLED", "TIME_STOP", "TIME_STOP_NO_PROGRESS"),
        ("EXIT_FILLED", "STALL_EXIT", "TIME_STOP_STALL"),
        ("EXIT_FILLED", "RUNNER_TIME_STOP", "TIME_STOP_RUNNER"),
        ("EXIT_FILLED", "MFE_GIVEBACK", "TRAILING_GIVEBACK"),
    ],
)
def test_exit_reason_mapping(event_code: str, reason: str, expected: str):
    assert adapter.exit_reason_for(event_code, reason) == expected


def test_route_exit_passes_mfe_mae_directly_from_position():
    kit = _RecordingKit()
    pos = _make_position()
    fill = _make_fill(role="exit", price=101.5)
    fill.fill_time = datetime(2026, 5, 9, 14, 30, tzinfo=timezone.utc)

    adapter.route_exit(kit, pos, fill, event_code="EXIT_FILLED", event_reason="T1")

    assert len(kit.calls) == 1
    name, kwargs = kit.calls[0]
    assert name == "log_exit"
    assert kwargs["exit_reason"] == "TAKE_PROFIT_T1"
    assert kwargs["mfe_price"] == pos.mfe_price
    assert kwargs["mae_price"] == pos.mae_price
    assert kwargs["mfe_r"] == pos.mfe_r
    assert kwargs["mae_r"] == pos.mae_r
    assert kwargs["fees_paid"] == 0.5
    # Long: pnl_pct = (101.5 - 100.0) / 100.0 = 0.015
    assert abs(kwargs["pnl_pct"] - 0.015) < 1e-9


def test_route_exit_short_pnl_sign_flips():
    kit = _RecordingKit()
    pos = _make_position(direction=Direction.SHORT)
    pos.entry_price = 100.0
    pos.current_stop = 101.0
    fill = _make_fill(role="stop", price=98.5)

    adapter.route_exit(kit, pos, fill, event_code="STOP_FILLED", event_reason="STOP")

    name, kwargs = kit.calls[0]
    # Short: pnl_per_share = entry - exit = 100.0 - 98.5 = 1.5; pnl_pct = 0.015
    assert abs(kwargs["pnl_pct"] - 0.015) < 1e-9
    assert kwargs["exit_reason"] == "STOP_LOSS"


def test_route_stop_adjustment_skipped_when_unchanged():
    kit = _RecordingKit()
    pos = _make_position()
    adapter.route_stop_adjustment(
        kit, setup_id="s1", symbol="QQQ", old_stop=99.0, new_stop=99.0,
        action_reason="profit_floor", position=pos,
    )
    assert kit.calls == []


def test_route_stop_adjustment_taxonomy():
    kit = _RecordingKit()
    pos = _make_position()
    adapter.route_stop_adjustment(
        kit, setup_id="s1", symbol="QQQ", old_stop=99.0, new_stop=100.0,
        action_reason="t1_profit_lock", position=pos,
    )
    name, kwargs = kit.calls[0]
    assert name == "log_stop_adjustment"
    assert kwargs["adjustment_type"] == "BREAKEVEN"
    assert kwargs["trigger"] == "T1_HIT"
    assert kwargs["old_stop"] == 99.0
    assert kwargs["new_stop"] == 100.0
    assert "mfe_r" in kwargs["metadata"]


def test_route_missed_payload_shape():
    kit = _RecordingKit()
    cfg = _make_cfg()
    rejection = {
        "symbol": "QQQ",
        "lane": "primary",
        "blocked_by": "asset_context_score_low",
        "block_reason": "asset_context_score=0.5 < min=1.0",
        "direction": "LONG",
        "grade": "a",
        "details": {"asset_context_score": 0.5, "asset_context_min_score": 1.0},
    }

    bar_ts = datetime(2026, 5, 9, 14, 0, tzinfo=timezone.utc)
    adapter.route_missed(kit, rejection, cfg, bar_ts=bar_ts)

    name, kwargs = kit.calls[0]
    assert name == "log_missed"
    assert kwargs["pair"] == "QQQ"
    assert kwargs["side"] == "LONG"
    assert kwargs["signal"] == "TPC_primary"
    assert kwargs["blocked_by"] == "asset_context_score_low"
    assert kwargs["market_regime"] == "a"
    assert 0.0 <= kwargs["signal_strength"] <= 1.0
    assert "primary" in kwargs["signal_id"]
    # signal_id uses epoch seconds, not ISO string
    assert str(int(bar_ts.timestamp())) in kwargs["signal_id"]


_BLOCK_REASONS: tuple[str, ...] = (
    "bar_none_or_session_filter_blocked",
    "news_filter_blocked",
    "regime_flat",
    "longs_disabled",
    "shorts_disabled",
    "shorts_require_a_plus",
    "pullback_not_detected",
    "pullback_30m_disabled",
    "confirmation_combo_failed",
    "confirmation_max_count_exceeded",
    "entry_plan_invalid",
    "stop_validation_failed",
    "daily_room_insufficient",
    "asset_context_score_low",
    "type_c_requires_a_plus",
    "min_short_score",
    "score_below_grade_threshold",
    "risk_pct_zero",
    "qty_zero",
    "second_entry_score_min",
)


@pytest.mark.parametrize("blocked_by", _BLOCK_REASONS)
def test_route_missed_payload_for_each_block_reason(blocked_by: str):
    kit = _RecordingKit()
    cfg = _make_cfg()
    rejection = {
        "symbol": "QQQ",
        "lane": "primary",
        "direction": "LONG",
        "grade": "a",
        "blocked_by": blocked_by,
        "block_reason": f"{blocked_by} fired",
        "details": {"score": 12.0},
    }
    bar_ts = datetime(2026, 5, 9, 14, 0, tzinfo=timezone.utc)
    adapter.route_missed(kit, rejection, cfg, bar_ts=bar_ts)
    assert len(kit.calls) == 1
    name, kwargs = kit.calls[0]
    assert name == "log_missed"
    assert kwargs["blocked_by"] == blocked_by
    assert kwargs["pair"] == "QQQ"
    assert kwargs["side"] == "LONG"
    assert 0.0 <= kwargs["signal_strength"] <= 1.0


def test_route_filter_decisions_emits_per_rejection():
    kit = _RecordingKit()
    cfg = _make_cfg()
    bar = _StubBar(timestamp=datetime(2026, 5, 9, 14, 0, tzinfo=timezone.utc))
    bar_input = _StubBarInput(symbol="QQQ", bar_15m=bar)
    rejections = [
        {"blocked_by": "asset_context_score_low", "lane": "primary", "grade": "a",
         "details": {"score": 10.0, "threshold": 14.0, "actual": 10.0}},
        {"blocked_by": "regime_flat", "lane": "prelude", "grade": "flat",
         "details": {}},
    ]
    adapter.route_filter_decisions(kit, bar_input, cfg, rejections=rejections)

    assert len(kit.calls) == 2
    names = {c[0] for c in kit.calls}
    assert names == {"on_filter_decision"}
    assert kit.calls[0][1]["filter_name"] == "asset_context_score_low"
    assert kit.calls[0][1]["passed"] is False
    assert kit.calls[1][1]["filter_name"] == "regime_flat"


def test_route_filter_decisions_emits_passed_per_gate_on_entry():
    kit = _RecordingKit()
    cfg = _make_cfg()
    bar = _StubBar(timestamp=datetime(2026, 5, 9, 14, 0, tzinfo=timezone.utc))
    bar_input = _StubBarInput(symbol="QQQ", bar_15m=bar)
    setup = _make_setup()
    adapter.route_filter_decisions(kit, bar_input, cfg, rejections=None, entry_setup=setup)
    # one passed=True per active gate
    assert len(kit.calls) == len(adapter._ENTRY_GATES)
    for _, kwargs in kit.calls:
        assert kwargs["passed"] is True
        assert kwargs["strategy_id"] == "TPC"
        assert kwargs["signal_name"] == "primary_TYPE_A"


def test_route_exit_mfe_mae_pct_is_signed():
    kit = _RecordingKit()
    pos = _make_position()  # LONG, entry=100, mfe=102.5, mae=99.5
    fill = _make_fill(role="exit", price=101.5)
    adapter.route_exit(kit, pos, fill, event_code="EXIT_FILLED", event_reason="T1")
    name, kwargs = kit.calls[0]
    assert kwargs["mfe_pct"] == pytest.approx(0.025)
    # MAE for LONG is below entry → mae_price < entry_price → adverse but signed negative
    assert kwargs["mae_pct"] == pytest.approx(-0.005)


def test_route_exit_t1_expected_uses_target_price_from_meta():
    kit = _RecordingKit()
    pos = _make_position()
    pos.meta = {"setup_lane": "primary", "target_price": 101.5}
    fill = _make_fill(role="exit", price=101.5)
    adapter.route_exit(kit, pos, fill, event_code="EXIT_FILLED", event_reason="T1")
    _, kwargs = kit.calls[0]
    assert kwargs["expected_exit_price"] == 101.5


def test_route_exit_stop_expected_uses_current_stop():
    kit = _RecordingKit()
    pos = _make_position()
    fill = _make_fill(role="stop", price=98.5)
    adapter.route_exit(kit, pos, fill, event_code="STOP_FILLED", event_reason="STOP")
    _, kwargs = kit.calls[0]
    assert kwargs["expected_exit_price"] == pos.current_stop


def test_route_order_event_translates_terminal_event():
    kit = _RecordingKit()
    update = ETFOrderUpdate(
        oms_order_id="oms-99",
        status="rejected",
        symbol="QQQ",
        timestamp=datetime(2026, 5, 9, 14, 30, tzinfo=timezone.utc),
        order_role="stop_replace",
    )

    class _Event:
        code = "ORDER_TERMINAL"
        details = {"setup_id": "TPC-QQQ-1", "status": "rejected"}

    adapter.route_order_event(kit, update, _Event())
    assert len(kit.calls) == 1
    name, kwargs = kit.calls[0]
    assert name == "on_order_event"
    assert kwargs["order_id"] == "oms-99"
    assert kwargs["pair"] == "QQQ"
    assert kwargs["status"] == "rejected"
    assert kwargs["related_trade_id"] == "TPC-QQQ-1"
    assert kwargs["strategy_id"] == "TPC"


def test_signal_strength_proxy_uses_score_when_available():
    cfg = _make_cfg()
    assert adapter.signal_strength_proxy(14.0, "a", cfg) == pytest.approx(1.0)
    assert adapter.signal_strength_proxy(7.0, "a", cfg) == pytest.approx(0.5)
    assert adapter.signal_strength_proxy(28.0, "a_plus", cfg) == pytest.approx(1.0)  # capped


def test_signal_strength_proxy_falls_back_to_grade():
    cfg = _make_cfg()
    assert adapter.signal_strength_proxy(None, "a_plus", cfg) == 0.9
    assert adapter.signal_strength_proxy(None, "flat", cfg) == 0.0
    assert adapter.signal_strength_proxy(None, None, cfg) == 0.0


def test_route_indicator_snapshot_no_setup_uses_no_signal():
    kit = _RecordingKit()
    cfg = _make_cfg()
    bar = _StubBar(timestamp=datetime(2026, 5, 9, 14, 0, tzinfo=timezone.utc))
    bar_input = _StubBarInput(symbol="QQQ", bar_15m=bar, indicators={"atr_4h": 1.2})
    state = ETFCoreState()

    adapter.route_indicator_snapshot(kit, bar_input, state, cfg, decision="NO_SIGNAL", setup=None)

    name, kwargs = kit.calls[0]
    assert name == "on_indicator_snapshot"
    assert kwargs["pair"] == "QQQ"
    assert kwargs["signal_name"] == "NO_SIGNAL"
    assert kwargs["signal_strength"] == 0.0
    assert kwargs["decision"] == "NO_SIGNAL"
    assert kwargs["strategy_id"] == "TPC"


def test_route_indicator_snapshot_with_setup_uses_lane_setup_type():
    kit = _RecordingKit()
    cfg = _make_cfg()
    bar = _StubBar(timestamp=datetime(2026, 5, 9, 14, 0, tzinfo=timezone.utc))
    bar_input = _StubBarInput(symbol="QQQ", bar_15m=bar, indicators={"atr_4h": 1.2, "vwap_15m": 100.5})
    state = ETFCoreState()
    setup = _make_setup(score=18.0)

    adapter.route_indicator_snapshot(kit, bar_input, state, cfg, decision="ENTRY_REQUESTED", setup=setup)

    name, kwargs = kit.calls[0]
    assert kwargs["signal_name"] == "primary_TYPE_A"
    assert kwargs["signal_strength"] > 0.0
    assert kwargs["decision"] == "ENTRY_REQUESTED"


def test_kit_exceptions_are_swallowed():
    class _BoomKit:
        def log_entry(self, **_: Any) -> None:
            raise RuntimeError("boom")

        def log_exit(self, **_: Any) -> None:
            raise RuntimeError("boom")

        def log_stop_adjustment(self, **_: Any) -> None:
            raise RuntimeError("boom")

        def log_missed(self, **_: Any) -> None:
            raise RuntimeError("boom")

        def on_indicator_snapshot(self, **_: Any) -> None:
            raise RuntimeError("boom")

        def on_filter_decision(self, **_: Any) -> None:
            raise RuntimeError("boom")

        def on_order_event(self, **_: Any) -> None:
            raise RuntimeError("boom")

    kit = _BoomKit()
    cfg = _make_cfg()
    setup = _make_setup()
    pos = _make_position()
    fill = _make_fill()
    state = ETFCoreState()
    bar_input = _StubBarInput(symbol="QQQ", bar_15m=_StubBar(timestamp=datetime.now(timezone.utc)))

    # Each call must not raise — instrumentation never blocks trading
    adapter.route_entry(kit, setup, fill, cfg, state)
    adapter.route_exit(kit, pos, fill, "EXIT_FILLED", "T1")
    adapter.route_stop_adjustment(kit, "s1", "QQQ", 99.0, 100.0, "t1_profit_lock", pos)
    adapter.route_missed(kit, {"symbol": "QQQ", "lane": "primary", "blocked_by": "x"}, cfg)
    adapter.route_indicator_snapshot(kit, bar_input, state, cfg, "NO_SIGNAL", None)
    adapter.route_filter_decisions(kit, bar_input, cfg, rejections=[
        {"blocked_by": "x", "lane": "primary", "details": {}},
    ])
    adapter.route_order_event(
        kit,
        ETFOrderUpdate(oms_order_id="o1", status="rejected", symbol="QQQ"),
        type("E", (), {"code": "ORDER_TERMINAL", "details": {}})(),
    )
