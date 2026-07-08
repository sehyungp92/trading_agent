"""Tests for BreakoutConfig — construction, serialization, round-trip."""

from __future__ import annotations

import pytest

from crypto_trader.strategy.breakout.config import (
    BreakoutConfig,
    BreakoutConfirmParams,
    BreakoutExitParams,
    BreakoutStopParams,
    BreakoutSymbolFilterParams,
    BreakoutTrailParams,
    ProfileParams,
    BalanceParams,
    BreakoutSetupParams,
    IndicatorParams,
)


class TestBreakoutConfigDefaults:
    """Test default construction of BreakoutConfig."""

    def test_default_construction(self):
        """BreakoutConfig() creates a valid config with all defaults."""
        cfg = BreakoutConfig()
        assert cfg.m30_indicators is not None
        assert cfg.h4_indicators is not None
        assert cfg.profile is not None
        assert cfg.balance is not None
        assert cfg.context is not None
        assert cfg.setup is not None
        assert cfg.confirmation is not None
        assert cfg.entry is not None
        assert cfg.stops is not None
        assert cfg.exits is not None
        assert cfg.trail is not None
        assert cfg.risk is not None
        assert cfg.limits is not None
        assert cfg.filters is not None
        assert cfg.reentry is not None
        assert cfg.symbol_filter is not None

    def test_symbols_default(self):
        """Default symbols are BTC, ETH, SOL."""
        cfg = BreakoutConfig()
        assert cfg.symbols == ["BTC", "ETH", "SOL"]

    def test_m30_indicators_ema_slow(self):
        """M30 indicators default ema_slow is 100."""
        cfg = BreakoutConfig()
        assert cfg.m30_indicators.ema_slow == 100

    def test_h4_indicators_ema_slow(self):
        """H4 indicators default ema_slow is 50."""
        cfg = BreakoutConfig()
        assert cfg.h4_indicators.ema_slow == 50

    def test_profile_params_defaults(self):
        """ProfileParams defaults: lookback_bars=36, num_bins=50, hvn_threshold=1.2."""
        cfg = BreakoutConfig()
        assert cfg.profile.lookback_bars == 36
        assert cfg.profile.num_bins == 50
        assert cfg.profile.hvn_threshold_pct == 1.2

    def test_reentry_max_wait_bars_default(self):
        """Re-entry state expires after 12 M30 bars by default."""
        cfg = BreakoutConfig()
        assert cfg.reentry.max_wait_bars == 12
        assert cfg.reentry.risk_scale == 1.0


class TestBreakoutConfigSerialization:
    """Test to_dict / from_dict round-trip and partial deserialization."""

    def test_to_dict_roundtrip(self):
        """to_dict() then from_dict() produces identical config."""
        original = BreakoutConfig()
        d = original.to_dict()
        restored = BreakoutConfig.from_dict(d)
        assert restored.symbols == original.symbols
        assert restored.m30_indicators.ema_slow == original.m30_indicators.ema_slow
        assert restored.h4_indicators.ema_slow == original.h4_indicators.ema_slow
        assert restored.profile.lookback_bars == original.profile.lookback_bars
        assert restored.profile.num_bins == original.profile.num_bins
        assert restored.risk.risk_pct_a == original.risk.risk_pct_a
        assert restored.trail.trail_r_adaptive == original.trail.trail_r_adaptive

    def test_from_dict_partial(self):
        """from_dict with only some keys uses defaults for the rest."""
        d = {"symbols": ["BTC"]}
        cfg = BreakoutConfig.from_dict(d)
        assert cfg.symbols == ["BTC"]
        # Everything else should be default
        assert cfg.m30_indicators.ema_slow == 100
        assert cfg.profile.lookback_bars == 36

    def test_from_dict_nested(self):
        """from_dict correctly routes nested dicts to sub-dataclasses."""
        d = {
            "profile": {"lookback_bars": 96, "num_bins": 100},
            "risk": {"risk_pct_b": 0.01},
        }
        cfg = BreakoutConfig.from_dict(d)
        assert isinstance(cfg.profile, ProfileParams)
        assert cfg.profile.lookback_bars == 96
        assert cfg.profile.num_bins == 100
        assert cfg.risk.risk_pct_b == 0.01

    def test_from_dict_old_reentry_config_uses_max_wait_default(self):
        """Older configs without max_wait_bars keep loading with the new default."""
        cfg = BreakoutConfig.from_dict({
            "symbols": ["BTC"],
            "reentry": {
                "enabled": True,
                "cooldown_bars": 4,
                "max_loss_r": 1.0,
                "max_reentries": 2,
                "min_confluences_override": 0,
            },
        })
        assert cfg.reentry.cooldown_bars == 4
        assert cfg.reentry.max_wait_bars == 12
        assert cfg.reentry.risk_scale == 1.0

    def test_tuple_fields_serialize(self):
        """Tuple fields (major_symbols) serialize as lists in to_dict."""
        cfg = BreakoutConfig()
        d = cfg.to_dict()
        # major_symbols is in the risk sub-dict
        assert isinstance(d["risk"]["major_symbols"], list)
        assert d["risk"]["major_symbols"] == ["BTC", "ETH"]

    def test_from_dict_unknown_keys(self):
        """Extra keys in dict don't cause errors (passed through to cls())."""
        # Verify known keys still work alongside unknown ones being absent.
        d = {"symbols": ["SOL"]}
        cfg = BreakoutConfig.from_dict(d)
        assert cfg.symbols == ["SOL"]


class TestBakedDefaults:
    """Verify baked round 3/4 defaults are correct."""

    def test_body_ratio_min(self):
        cfg = BreakoutConfig()
        assert cfg.setup.body_ratio_min == 0.4675

    def test_be_buffer_r(self):
        cfg = BreakoutConfig()
        assert cfg.exits.be_buffer_r == 0.525

    def test_stops_atr_mult(self):
        cfg = BreakoutConfig()
        assert cfg.stops.atr_mult == 1.0

    def test_structure_trail_disabled(self):
        cfg = BreakoutConfig()
        assert cfg.trail.structure_trail_enabled is False

    def test_min_bars_in_zone(self):
        cfg = BreakoutConfig()
        assert cfg.balance.min_bars_in_zone == 4

    def test_tp2_r_above_tp1(self):
        """tp2_r (2.0) > tp1_r (0.8) — no cascade."""
        cfg = BreakoutConfig()
        assert cfg.exits.tp2_r == 2.0
        assert cfg.exits.tp2_r > cfg.exits.tp1_r

    def test_eth_long_only(self):
        cfg = BreakoutConfig()
        assert cfg.symbol_filter.eth_direction == "long_only"

    def test_tp1_r_raised(self):
        """tp1_r raised to 0.8 (match trend pattern)."""
        cfg = BreakoutConfig()
        assert cfg.exits.tp1_r == 0.8

    def test_tp1_frac_lowered(self):
        """tp1_frac lowered to 0.3 (keep 70% runner)."""
        cfg = BreakoutConfig()
        assert cfg.exits.tp1_frac == 0.3

    def test_invalidation_depth_deeper(self):
        """invalidation_depth_atr raised to 1.2 (require deeper penetration)."""
        cfg = BreakoutConfig()
        assert cfg.exits.invalidation_depth_atr == 1.2

    def test_invalidation_min_bars(self):
        """invalidation_min_bars default is 3."""
        cfg = BreakoutConfig()
        assert cfg.exits.invalidation_min_bars == 3

    def test_trail_buffer_tight_baked(self):
        """trail_buffer_tight baked to 0.1575 from round 4."""
        cfg = BreakoutConfig()
        assert cfg.trail.trail_buffer_tight == 0.1575

    def test_trail_activation_r_earlier(self):
        """trail_activation_r lowered to 0.3 (match trend)."""
        cfg = BreakoutConfig()
        assert cfg.trail.trail_activation_r == 0.3

    def test_trail_activation_bars_faster(self):
        """trail_activation_bars lowered to 4 (breakout momentum fades faster)."""
        cfg = BreakoutConfig()
        assert cfg.trail.trail_activation_bars == 4

    def test_hvn_threshold_lowered(self):
        """hvn_threshold_pct baked to 1.2 (was 1.5)."""
        cfg = BreakoutConfig()
        assert cfg.profile.hvn_threshold_pct == 1.2

    def test_lookback_bars_baked(self):
        """lookback_bars baked to 36 (R1 mutation)."""
        cfg = BreakoutConfig()
        assert cfg.profile.lookback_bars == 36

    def test_model1_require_volume(self):
        """model1_require_volume default is True."""
        cfg = BreakoutConfig()
        assert cfg.confirmation.model1_require_volume is True

    def test_model1_min_volume_mult(self):
        """model1_min_volume_mult default is 1.0."""
        cfg = BreakoutConfig()
        assert cfg.confirmation.model1_min_volume_mult == 1.0

    def test_model1_direction_close_disabled(self):
        """model1_require_direction_close baked to False (R1 mutation)."""
        cfg = BreakoutConfig()
        assert cfg.confirmation.model1_require_direction_close is False

    def test_model2_disabled(self):
        """Model 2 disabled (25% WR, -1.80R on 4 trades)."""
        cfg = BreakoutConfig()
        assert cfg.confirmation.enable_model2 is False

    def test_use_farther_disabled(self):
        """use_farther disabled (tighter stops for lower avg loser R)."""
        cfg = BreakoutConfig()
        assert cfg.stops.use_farther is False

    def test_time_stop_reduce(self):
        """time_stop_action is reduce (match trend; keep 50% alive)."""
        cfg = BreakoutConfig()
        assert cfg.exits.time_stop_action == "reduce"

    def test_quick_exit_enabled(self):
        """Quick exit enabled by default (cut stagnant trades)."""
        cfg = BreakoutConfig()
        assert cfg.exits.quick_exit_enabled is True

    def test_runner_frac(self):
        """runner_frac is 0.3 (1.0 - 0.3 - 0.4 = 0.3)."""
        cfg = BreakoutConfig()
        assert cfg.exits.runner_frac == 0.3
