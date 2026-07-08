"""Tests for config_mutator — dot-notation mutation of MomentumConfig."""

import pytest

from crypto_trader.optimize.config_mutator import apply_mutations, merge_mutations
from crypto_trader.strategy.momentum.config import MomentumConfig


class TestApplyMutations:
    def test_single_mutation(self):
        cfg = MomentumConfig()
        result = apply_mutations(cfg, {"entry.entry_on_break": True})
        assert result.entry.entry_on_break is True
        # Original unchanged
        assert cfg.entry.entry_on_break is False

    def test_multiple_mutations_same_section(self):
        cfg = MomentumConfig()
        result = apply_mutations(cfg, {
            "exits.tp1_r": 1.5,
            "exits.tp2_r": 3.0,
        })
        assert result.exits.tp1_r == 1.5
        assert result.exits.tp2_r == 3.0

    def test_mutations_across_sections(self):
        cfg = MomentumConfig()
        result = apply_mutations(cfg, {
            "entry.entry_on_break": True,
            "stops.atr_buffer_mult": 0.4,
            "bias.h1_adx_threshold": 25.0,
        })
        assert result.entry.entry_on_break is True
        assert result.stops.atr_buffer_mult == 0.4
        assert result.bias.h1_adx_threshold == 25.0

    def test_unknown_section_raises(self):
        cfg = MomentumConfig()
        with pytest.raises(ValueError, match="Unknown config section"):
            apply_mutations(cfg, {"nonexistent.field": 42})

    def test_unknown_field_raises(self):
        cfg = MomentumConfig()
        with pytest.raises(ValueError, match="Unknown field"):
            apply_mutations(cfg, {"entry.nonexistent_field": True})

    def test_bad_key_format_raises(self):
        cfg = MomentumConfig()
        with pytest.raises(ValueError, match="section.field"):
            apply_mutations(cfg, {"noperiod": True})

    def test_empty_mutations_returns_copy(self):
        cfg = MomentumConfig()
        result = apply_mutations(cfg, {})
        assert result is not cfg
        assert result.entry.entry_on_break == cfg.entry.entry_on_break

    def test_tuple_fields_preserved(self):
        cfg = MomentumConfig()
        result = apply_mutations(cfg, {"risk.risk_pct_a": 0.01})
        assert result.risk.major_symbols == ("BTC", "ETH")
        assert isinstance(result.risk.major_symbols, tuple)

    def test_returns_new_config_instance(self):
        cfg = MomentumConfig()
        result = apply_mutations(cfg, {"indicators.ema_fast": 15})
        assert result is not cfg
        assert result.indicators is not cfg.indicators


class TestMergeMutations:
    def test_merge_empty(self):
        assert merge_mutations({}, {}) == {}

    def test_merge_overlay_wins(self):
        base = {"a": 1, "b": 2}
        overlay = {"b": 99, "c": 3}
        result = merge_mutations(base, overlay)
        assert result == {"a": 1, "b": 99, "c": 3}

    def test_merge_preserves_base(self):
        base = {"a": 1}
        overlay = {"b": 2}
        merge_mutations(base, overlay)
        assert base == {"a": 1}
