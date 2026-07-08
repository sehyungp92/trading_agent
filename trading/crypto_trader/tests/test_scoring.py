"""Tests for scoring — normalizers, composite score, hard rejects."""

import math

import pytest

from crypto_trader.optimize.scoring import (
    check_hard_rejects,
    composite_score,
    normalize_calmar,
    normalize_capture,
    normalize_coverage,
    normalize_edge,
    normalize_entry_quality,
    normalize_expectancy,
    normalize_exit_efficiency,
    normalize_hold,
    normalize_risk,
    normalize_sharpe,
    normalize_stability,
)


class TestNormalizers:
    def test_coverage_zero(self):
        assert normalize_coverage({"total_trades": 0}) == 0.0

    def test_coverage_mid(self):
        assert normalize_coverage({"total_trades": 15}) == 0.5

    def test_coverage_max(self):
        assert normalize_coverage({"total_trades": 30}) == 1.0

    def test_coverage_over_max(self):
        assert normalize_coverage({"total_trades": 200}) == 1.0

    def test_risk_zero_dd(self):
        assert normalize_risk({"max_drawdown_pct": 0.0}) == 1.0

    def test_risk_max_dd(self):
        assert normalize_risk({"max_drawdown_pct": 50.0}) == 0.0

    def test_risk_mid(self):
        assert normalize_risk({"max_drawdown_pct": 25.0}) == 0.5

    def test_edge_pf_1(self):
        assert normalize_edge({"profit_factor": 1.0}) == 0.0

    def test_edge_pf_7(self):
        assert normalize_edge({"profit_factor": 7.0}) == 1.0

    def test_edge_pf_4(self):
        assert normalize_edge({"profit_factor": 4.0}) == pytest.approx(0.5)

    def test_edge_pf_below_1(self):
        assert normalize_edge({"profit_factor": 0.5}) == 0.0

    def test_capture_clipped(self):
        assert normalize_capture({"exit_efficiency": 1.5}) == 1.0
        assert normalize_capture({"exit_efficiency": -0.5}) == 0.0

    def test_expectancy(self):
        assert normalize_expectancy({"expectancy_r": 0.3}) == pytest.approx(0.5)
        assert normalize_expectancy({"expectancy_r": -0.1}) == 0.0

    def test_hold_optimal(self):
        result = normalize_hold({"avg_bars_held": 12})
        assert result == pytest.approx(1.0)

    def test_hold_far(self):
        result = normalize_hold({"avg_bars_held": 100})
        assert result < 0.01

    def test_entry_quality_perfect(self):
        assert normalize_entry_quality({"avg_mae_r": 0.0}) == 1.0

    def test_entry_quality_worst(self):
        assert normalize_entry_quality({"avg_mae_r": 1.0}) == 0.0

    def test_entry_quality_negative_mae(self):
        """Negative MAE (typical) should use absolute value."""
        result = normalize_entry_quality({"avg_mae_r": -0.59})
        assert result == pytest.approx(0.41)

    def test_entry_quality_negative_mae_capped(self):
        """Large negative MAE should be capped at 1.0, not exceed it."""
        result = normalize_entry_quality({"avg_mae_r": -1.5})
        assert result == 0.0  # |1.5| > 1.0 → max(1.0 - 1.5, 0) = 0.0

    def test_entry_quality_small_negative(self):
        """Small negative MAE should score well."""
        result = normalize_entry_quality({"avg_mae_r": -0.1})
        assert result == pytest.approx(0.9)

    def test_exit_efficiency(self):
        assert normalize_exit_efficiency({"exit_efficiency": 0.5}) == 0.5

    def test_calmar_zero(self):
        assert normalize_calmar({"calmar_ratio": 0.0}) == 0.0

    def test_calmar_max(self):
        assert normalize_calmar({"calmar_ratio": 10.0}) == 1.0

    def test_calmar_at_ceiling(self):
        assert normalize_calmar({"calmar_ratio": 5.0}) == 1.0

    def test_calmar_mid(self):
        assert normalize_calmar({"calmar_ratio": 2.5}) == pytest.approx(0.5)

    def test_sharpe_max(self):
        assert normalize_sharpe({"sharpe_ratio": 3.0}) == 1.0

    def test_sharpe_at_ceiling(self):
        assert normalize_sharpe({"sharpe_ratio": 2.5}) == 1.0

    def test_stability_zero_duration(self):
        assert normalize_stability({"max_drawdown_duration": 0}) == 1.0

    def test_stability_max_duration(self):
        assert normalize_stability({"max_drawdown_duration": 2000}) == 0.0


class TestHardRejects:
    def test_passes_all(self):
        metrics = {"max_drawdown_pct": 20.0, "total_trades": 50.0}
        rejected, reason = check_hard_rejects(metrics, {
            "max_drawdown_pct": ("<=", 45.0),
            "total_trades": (">=", 30.0),
        })
        assert rejected is False

    def test_rejects_drawdown(self):
        metrics = {"max_drawdown_pct": 50.0, "total_trades": 50.0}
        rejected, reason = check_hard_rejects(metrics, {
            "max_drawdown_pct": ("<=", 45.0),
        })
        assert rejected is True
        assert "max_drawdown_pct" in reason

    def test_rejects_insufficient_trades(self):
        metrics = {"total_trades": 10.0}
        rejected, reason = check_hard_rejects(metrics, {
            "total_trades": (">=", 30.0),
        })
        assert rejected is True


class TestCompositeScore:
    def test_weighted_sum(self):
        metrics = {"total_trades": 100.0, "max_drawdown_pct": 0.0}
        score, rejected, reason = composite_score(
            metrics, {"coverage": 0.5, "risk": 0.5}
        )
        assert score == pytest.approx(1.0)
        assert rejected is False

    def test_hard_reject_short_circuits(self):
        metrics = {"total_trades": 5.0, "max_drawdown_pct": 0.0}
        score, rejected, reason = composite_score(
            metrics,
            {"coverage": 1.0},
            hard_rejects={"total_trades": (">=", 30.0)},
        )
        assert score == 0.0
        assert rejected is True
