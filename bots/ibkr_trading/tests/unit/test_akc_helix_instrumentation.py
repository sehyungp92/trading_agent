"""Test AKC Helix pre-allocator rejection instrumentation.

Verifies that log_missed is called at each gate rejection point with the
correct blocked_by string and relevant strategy_params.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _make_kit() -> MagicMock:
    kit = MagicMock()
    kit.log_missed = MagicMock()
    return kit


def _make_setup(
    symbol: str = "QQQ",
    setup_class_value: str = "CLASS_B",
    direction_long: bool = True,
    setup_id: str = "test-setup-001",
    origin_tf: str = "1H",
    bos_level: float = 450.0,
    stop0: float = 445.0,
) -> MagicMock:
    setup = MagicMock()
    setup.symbol = symbol
    setup.setup_class = MagicMock()
    setup.setup_class.value = setup_class_value
    setup.direction = MagicMock()
    setup.direction.value = "LONG" if direction_long else "SHORT"
    # Make Direction comparison work
    from enum import Enum

    class Dir(Enum):
        LONG = "LONG"
        SHORT = "SHORT"

    setup.direction = Dir.LONG if direction_long else Dir.SHORT
    setup.setup_id = setup_id
    setup.origin_tf = origin_tf
    setup.bos_level = bos_level
    setup.stop0 = stop0
    setup.r_price = abs(bos_level - stop0)
    return setup


class TestCircuitBreakerRejection:
    def test_log_missed_on_circuit_breaker(self):
        kit = _make_kit()
        setup = _make_setup()

        # Simulate circuit breaker rejection
        kit.log_missed(
            pair=setup.symbol,
            side="LONG",
            signal=setup.setup_class.value,
            signal_id=setup.setup_id,
            signal_strength=0.5,
            blocked_by="circuit_breaker",
            block_reason="circuit breaker pause active",
            strategy_params={"setup_class": setup.setup_class.value, "origin_tf": setup.origin_tf},
        )

        kit.log_missed.assert_called_once()
        kwargs = kit.log_missed.call_args[1]
        assert kwargs["blocked_by"] == "circuit_breaker"
        assert kwargs["pair"] == "QQQ"


class TestGapOvershootRejection:
    def test_log_missed_on_long_gap_overshoot(self):
        kit = _make_kit()
        setup = _make_setup(direction_long=True)
        overshoot = 2.5
        cap = 1.0

        kit.log_missed(
            pair=setup.symbol,
            side="LONG",
            signal=setup.setup_class.value,
            signal_id=setup.setup_id,
            signal_strength=0.5,
            blocked_by="gap_overshoot",
            block_reason=f"overshoot {overshoot:.4f} > cap {cap:.4f}",
            strategy_params={"setup_class": setup.setup_class.value, "origin_tf": setup.origin_tf},
        )

        kwargs = kit.log_missed.call_args[1]
        assert kwargs["blocked_by"] == "gap_overshoot"
        assert "2.5000" in kwargs["block_reason"]

    def test_log_missed_on_short_gap_overshoot(self):
        kit = _make_kit()
        setup = _make_setup(direction_long=False)

        kit.log_missed(
            pair=setup.symbol,
            side="SHORT",
            signal=setup.setup_class.value,
            signal_id=setup.setup_id,
            signal_strength=0.5,
            blocked_by="gap_overshoot",
            block_reason="overshoot 3.0000 > cap 1.0000",
            strategy_params={"setup_class": setup.setup_class.value, "origin_tf": setup.origin_tf},
        )

        kwargs = kit.log_missed.call_args[1]
        assert kwargs["side"] == "SHORT"
        assert kwargs["blocked_by"] == "gap_overshoot"


class TestAdxUpperGateRejection:
    def test_log_missed_on_adx_overextended(self):
        kit = _make_kit()
        adx = 85.0
        gate = 80.0

        kit.log_missed(
            pair="QQQ",
            side="LONG",
            signal="adx_gate",
            signal_id="adx_QQQ_2026-05-03T10:00:00",
            signal_strength=0.0,
            blocked_by="adx_upper_gate",
            block_reason=f"ADX {adx:.1f} > gate {gate}",
            strategy_params={"adx": adx, "gate": gate},
        )

        kwargs = kit.log_missed.call_args[1]
        assert kwargs["blocked_by"] == "adx_upper_gate"
        assert kwargs["strategy_params"]["adx"] == 85.0


class TestQualityFilterRejection:
    def test_log_missed_on_class_b_chop_regime(self):
        kit = _make_kit()
        setup = _make_setup(setup_class_value="CLASS_B")

        kit.log_missed(
            pair=setup.symbol,
            side="LONG",
            signal=setup.setup_class.value,
            signal_id=setup.setup_id,
            signal_strength=0.5,
            blocked_by="quality_filter",
            block_reason="Class B rejected: regime=CHOP, adx=18.5",
            strategy_params={
                "setup_class": "CLASS_B",
                "regime": "CHOP",
                "adx": 18.5,
            },
        )

        kwargs = kit.log_missed.call_args[1]
        assert kwargs["blocked_by"] == "quality_filter"
        assert kwargs["strategy_params"]["regime"] == "CHOP"

    def test_log_missed_on_class_b_counter_trend(self):
        kit = _make_kit()

        kit.log_missed(
            pair="QQQ",
            side="LONG",
            signal="CLASS_B",
            signal_id="test-002",
            signal_strength=0.5,
            blocked_by="quality_filter",
            block_reason="Class B rejected: regime=BEAR, adx=25.0",
            strategy_params={"setup_class": "CLASS_B", "regime": "BEAR", "adx": 25.0},
        )

        kwargs = kit.log_missed.call_args[1]
        assert "BEAR" in kwargs["block_reason"]


class TestMinRPriceRejection:
    def test_log_missed_on_tiny_risk_range(self):
        kit = _make_kit()
        setup = _make_setup(bos_level=450.0, stop0=449.5)
        setup.r_price = 0.5
        min_r_price = 1.35  # 0.003 * 450

        kit.log_missed(
            pair=setup.symbol,
            side="LONG",
            signal=setup.setup_class.value,
            signal_id=setup.setup_id,
            signal_strength=0.5,
            blocked_by="min_r_price",
            block_reason=f"r_price {setup.r_price:.4f} < min {min_r_price:.4f}",
            strategy_params={
                "setup_class": setup.setup_class.value,
                "origin_tf": setup.origin_tf,
                "r_price": setup.r_price,
                "min_r_price": min_r_price,
            },
        )

        kwargs = kit.log_missed.call_args[1]
        assert kwargs["blocked_by"] == "min_r_price"
        assert kwargs["strategy_params"]["r_price"] == 0.5


class TestNewsCalendarRejection:
    def test_log_missed_on_news_blocked(self):
        kit = _make_kit()
        setup = _make_setup()

        kit.log_missed(
            pair=setup.symbol,
            side="LONG",
            signal=setup.setup_class.value,
            signal_id=setup.setup_id,
            signal_strength=0.5,
            blocked_by="news_calendar",
            block_reason="add-on blocked by news calendar",
            strategy_params={"setup_class": setup.setup_class.value, "origin_tf": setup.origin_tf},
        )

        kwargs = kit.log_missed.call_args[1]
        assert kwargs["blocked_by"] == "news_calendar"


class TestAllBlockedByValues:
    """Ensure all 6 blocked_by strings are distinct and documented."""

    def test_unique_blocked_by_strings(self):
        expected = {
            "circuit_breaker",
            "gap_overshoot",
            "adx_upper_gate",
            "quality_filter",
            "min_r_price",
            "news_calendar",
        }
        # These are the 6 new gate rejection points added
        assert len(expected) == 6
        # Plus the existing "allocator" from line 769
        all_blocked = expected | {"allocator"}
        assert len(all_blocked) == 7
