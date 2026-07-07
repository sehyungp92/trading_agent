"""Tests for trend strategy configuration."""

import pytest

from crypto_trader.strategy.trend.config import (
    IndicatorParams,
    RegimeParams,
    TrendConfig,
    TrendConfirmationParams,
    TrendEntryParams,
    TrendExitParams,
    TrendLimitParams,
    TrendPerpFilterParams,
    TrendReentryParams,
    TrendRiskParams,
    TrendSetupParams,
    TrendStopParams,
    TrendTrailParams,
)


class TestTrendConfigDefaults:
    def test_default_symbols(self):
        cfg = TrendConfig()
        assert cfg.symbols == ["BTC", "ETH", "SOL"]

    def test_default_h1_indicators(self):
        cfg = TrendConfig()
        assert cfg.h1_indicators.ema_fast == 20
        assert cfg.h1_indicators.ema_mid == 50

    def test_default_d1_indicators(self):
        cfg = TrendConfig()
        assert cfg.d1_indicators.ema_fast == 21
        assert cfg.d1_indicators.ema_mid == 50

    def test_default_regime_loose(self):
        cfg = TrendConfig()
        assert cfg.regime.a_min_adx == 12.0  # Baked from round 1
        assert cfg.regime.require_structure is False
        assert cfg.regime.require_ema_cross is False

    def test_default_impulse_min_atr_move(self):
        """impulse_min_atr_move baked from round 2 (was 1.5)."""
        cfg = TrendConfig()
        assert cfg.setup.impulse_min_atr_move == 0.8

    def test_default_enable_hammer(self):
        """Hammer disabled — 3 trades, WR 33%, avg R -0.12."""
        cfg = TrendConfig()
        assert cfg.confirmation.enable_hammer is False

    def test_default_atr_mult(self):
        """atr_mult baked from round 2 (was 1.3)."""
        cfg = TrendConfig()
        assert cfg.stops.atr_mult == 2.0

    def test_default_time_stop_bars(self):
        """time_stop_bars baked from round 2 (was 12)."""
        cfg = TrendConfig()
        assert cfg.exits.time_stop_bars == 20

    def test_default_tp1_r(self):
        """tp1_r lowered to 0.8 so TP1 fires within avg MFE range."""
        cfg = TrendConfig()
        assert cfg.exits.tp1_r == 0.8

    def test_default_setup_loose(self):
        cfg = TrendConfig()
        assert cfg.setup.min_confluences == 0
        assert cfg.setup.pullback_max_retrace == 0.75
        assert cfg.setup.strict_orderly_pullback is False
        assert cfg.setup.use_weighted_confluence is False
        assert cfg.setup.weekly_room_filter_enabled is False

    def test_default_new_confirmation_flags(self):
        cfg = TrendConfig()
        assert cfg.confirmation.require_confirmation_for_b is False
        assert cfg.confirmation.enforce_volume_on_trigger is False

    def test_default_entry_mode_legacy(self):
        cfg = TrendConfig()
        assert cfg.entry.mode == "legacy"

    def test_default_scratch_exit_disabled(self):
        cfg = TrendConfig()
        assert cfg.exits.scratch_exit_enabled is False

    def test_default_relative_strength_filter_disabled(self):
        cfg = TrendConfig()
        assert cfg.filters.relative_strength_filter_enabled is False

    def test_default_reentry_recovery_flags(self):
        cfg = TrendConfig()
        assert cfg.reentry.max_wait_bars == 0
        assert cfg.reentry.require_same_direction is False
        assert cfg.reentry.only_after_scratch_exit is False
        assert cfg.reentry.risk_scale == 1.0

    def test_indicator_params_fields(self):
        """IndicatorParams has same field names as momentum's for duck-typing."""
        params = IndicatorParams()
        assert hasattr(params, "ema_fast")
        assert hasattr(params, "ema_mid")
        assert hasattr(params, "ema_slow")
        assert hasattr(params, "adx_period")
        assert hasattr(params, "atr_period")
        assert hasattr(params, "rsi_period")
        assert hasattr(params, "volume_ma_period")
        assert hasattr(params, "atr_avg_period")


class TestTrendConfigRoundtrip:
    def test_to_dict_from_dict_roundtrip(self):
        cfg = TrendConfig()
        d = cfg.to_dict()
        cfg2 = TrendConfig.from_dict(d)
        assert cfg2.regime.a_min_adx == cfg.regime.a_min_adx
        assert cfg2.symbols == cfg.symbols
        assert cfg2.trail.trail_r_ceiling == cfg.trail.trail_r_ceiling

    def test_from_dict_partial(self):
        cfg = TrendConfig.from_dict({"regime": {"a_min_adx": 25.0}})
        assert cfg.regime.a_min_adx == 25.0
        # Other fields keep defaults
        assert cfg.setup.min_confluences == 0

    def test_to_dict_tuples_become_lists(self):
        cfg = TrendConfig()
        d = cfg.to_dict()
        assert isinstance(d["risk"]["major_symbols"], list)

    def test_h1_regime_fields_roundtrip(self):
        """New h1_regime fields survive to_dict/from_dict roundtrip."""
        cfg = TrendConfig()
        d = cfg.to_dict()
        assert d["regime"]["h1_regime_enabled"] is True
        assert d["regime"]["h1_min_adx"] == 22.0
        cfg2 = TrendConfig.from_dict(d)
        assert cfg2.regime.h1_regime_enabled is True
        assert cfg2.regime.h1_min_adx == 22.0

    def test_roundtrip_new_round6_fields(self):
        cfg = TrendConfig()
        d = cfg.to_dict()
        assert d["setup"]["strict_orderly_pullback"] is False
        assert d["setup"]["use_weighted_confluence"] is False
        assert d["confirmation"]["require_confirmation_for_b"] is False
        assert d["entry"]["mode"] == "legacy"
        assert d["exits"]["scratch_exit_enabled"] is False
        assert d["filters"]["relative_strength_filter_enabled"] is False

        cfg2 = TrendConfig.from_dict(d)
        assert cfg2.setup.strict_orderly_pullback is False
        assert cfg2.setup.use_weighted_confluence is False
        assert cfg2.confirmation.require_confirmation_for_b is False
        assert cfg2.entry.mode == "legacy"
        assert cfg2.exits.scratch_exit_enabled is False
        assert cfg2.filters.relative_strength_filter_enabled is False

    def test_roundtrip_new_round7_reentry_fields(self):
        cfg = TrendConfig()
        d = cfg.to_dict()
        assert d["reentry"]["max_wait_bars"] == 0
        assert d["reentry"]["require_same_direction"] is False
        assert d["reentry"]["only_after_scratch_exit"] is False
        assert d["reentry"]["risk_scale"] == 1.0

        cfg2 = TrendConfig.from_dict(d)
        assert cfg2.reentry.max_wait_bars == 0
        assert cfg2.reentry.require_same_direction is False
        assert cfg2.reentry.only_after_scratch_exit is False
        assert cfg2.reentry.risk_scale == 1.0

    def test_default_min_room_r(self):
        """min_room_r default changed to 1.0."""
        cfg = TrendConfig()
        assert cfg.setup.min_room_r == 1.0

    def test_nested_mutation_compatible(self):
        """Verify dot-notation mutation works with TrendConfig."""
        from crypto_trader.optimize.config_mutator import apply_mutations
        cfg = TrendConfig()
        mutated = apply_mutations(cfg, {"regime.a_min_adx": 25.0})
        assert mutated.regime.a_min_adx == 25.0
        assert cfg.regime.a_min_adx == 12.0  # Original unchanged (baked from round 1)
