"""Tests for canonical backtest economic profiles."""

from __future__ import annotations

from datetime import date

import pytest

from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.backtest.profiles import (
    LIVE_PARITY_PROFILE,
    assert_backtest_config_matches_profile,
    build_backtest_config_from_profile,
    profile_hash,
)


def test_live_parity_profile_canonical_values() -> None:
    profile = LIVE_PARITY_PROFILE
    assert profile.profile_id == "live_parity_v1"
    assert profile.symbols == ("BTC", "ETH", "SOL")
    assert profile.initial_equity == 10_000.0
    assert profile.taker_fee_bps == 4.5
    assert profile.maker_fee_bps == 1.0
    assert profile.slippage_bps == 2.0
    assert profile.spread_bps == 1.0
    assert profile.apply_funding is True
    assert profile.warmup_days == 60
    assert profile.terminal_accounting_mode == "terminal_mark"


def test_build_backtest_config_from_profile() -> None:
    config = build_backtest_config_from_profile(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 3, 1),
    )
    assert config.symbols == ["BTC", "ETH", "SOL"]
    assert config.initial_equity == 10_000.0
    assert config.taker_fee_bps == 4.5
    assert config.spread_bps == 1.0
    assert config.apply_funding is True
    assert config.warmup_days == 60
    assert_backtest_config_matches_profile(config)


def test_profile_mismatch_raises() -> None:
    config = BacktestConfig(symbols=["BTC"], taker_fee_bps=3.5, spread_bps=2.0)
    with pytest.raises(ValueError, match="live_parity_v1"):
        assert_backtest_config_matches_profile(config)


def test_profile_hash_is_stable_and_sensitive() -> None:
    first = profile_hash()
    second = profile_hash()
    assert first == second

    changed = LIVE_PARITY_PROFILE.__class__(
        profile_id=LIVE_PARITY_PROFILE.profile_id,
        taker_fee_bps=5.0,
    )
    assert profile_hash(changed) != first
