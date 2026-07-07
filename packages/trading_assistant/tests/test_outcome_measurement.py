"""Tests for Phase 1: Rigorous Outcome Measurement.

Covers:
- schemas/outcome_measurement.py (Verdict, MeasurementQuality, OutcomeMeasurement, helpers)
- skills/auto_outcome_measurer.py (AutoOutcomeMeasurer)
- skills/suggestion_scorer.py (SuggestionScorer quality/verdict filtering)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from trading_assistant.schemas.outcome_measurement import (
    MeasurementQuality,
    OutcomeMeasurement,
    Verdict,
    compute_measurement_quality,
    compute_significance,
)
from trading_assistant.skills.auto_outcome_measurer import AutoOutcomeMeasurer
from trading_assistant.skills.suggestion_scorer import SuggestionScorer


# ---------------------------------------------------------------------------
# Helpers for creating fake curated data on disk
# ---------------------------------------------------------------------------

def _write_summary(
    curated_dir: Path,
    date: str,
    bot_id: str,
    net_pnl: float = 100,
    total_trades: int = 10,
    win_count: int = 6,
    max_drawdown_pct: float = 5,
) -> None:
    bot_dir = curated_dir / date / bot_id
    bot_dir.mkdir(parents=True, exist_ok=True)
    (bot_dir / "summary.json").write_text(json.dumps({
        "net_pnl": net_pnl,
        "total_trades": total_trades,
        "win_count": win_count,
        "max_drawdown_pct": max_drawdown_pct,
    }))


def _write_regime(
    curated_dir: Path,
    date: str,
    bot_id: str,
    regime: str = "trending",
) -> None:
    bot_dir = curated_dir / date / bot_id
    bot_dir.mkdir(parents=True, exist_ok=True)
    (bot_dir / "regime_analysis.json").write_text(json.dumps({
        "dominant_regime": regime,
    }))


def _write_suggestion(
    findings_dir: Path,
    suggestion_id: str,
    bot_id: str,
    status: str = "deployed",
    deployed_at: str | None = None,
) -> None:
    """Append a suggestion record to suggestions.jsonl."""
    path = findings_dir / "suggestions.jsonl"
    rec = {
        "suggestion_id": suggestion_id,
        "bot_id": bot_id,
        "title": f"Suggestion {suggestion_id}",
        "tier": "parameter",
        "source_report_id": "rpt-1",
        "status": status,
        "deployed_at": deployed_at,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def _write_outcome(
    findings_dir: Path,
    suggestion_id: str,
    *,
    measurement_quality: str = "high",
    verdict: str = "",
    pnl_delta: float | None = None,
    pnl_delta_7d: float | None = None,
) -> None:
    """Append an outcome record to outcomes.jsonl."""
    path = findings_dir / "outcomes.jsonl"
    rec: dict = {
        "suggestion_id": suggestion_id,
        "measurement_quality": measurement_quality,
    }
    if verdict:
        rec["verdict"] = verdict
    if pnl_delta is not None:
        rec["pnl_delta"] = pnl_delta
    if pnl_delta_7d is not None:
        rec["pnl_delta_7d"] = pnl_delta_7d
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def _base_measurement(**overrides) -> OutcomeMeasurement:
    """Create an OutcomeMeasurement with sensible defaults, overriding any field."""
    defaults = dict(
        suggestion_id="sug-1",
        implemented_date="2026-03-01",
        measurement_date="2026-03-08",
        window_days=7,
        pnl_before=100.0,
        pnl_after=120.0,
        win_rate_before=0.6,
        win_rate_after=0.65,
        before_trade_count=20,
        after_trade_count=20,
        measurement_quality=MeasurementQuality.HIGH,
    )
    defaults.update(overrides)
    return OutcomeMeasurement(**defaults)


# ===================================================================
# 1. schemas/outcome_measurement.py
# ===================================================================


class TestVerdictEnum:
    def test_verdict_has_five_values(self):
        assert len(Verdict) == 5
        expected = {"positive", "neutral", "negative", "inconclusive", "insufficient_data"}
        assert {v.value for v in Verdict} == expected


class TestMeasurementQualityEnum:
    def test_measurement_quality_has_four_values(self):
        assert len(MeasurementQuality) == 4
        expected = {"high", "medium", "low", "insufficient"}
        assert {v.value for v in MeasurementQuality} == expected


class TestOutcomeMeasurementVerdict:
    """Tests for the computed verdict property."""

    def test_insufficient_data_before_too_few(self):
        m = _base_measurement(before_trade_count=2, after_trade_count=20)
        assert m.verdict == Verdict.INSUFFICIENT_DATA

    def test_insufficient_data_after_too_few(self):
        m = _base_measurement(before_trade_count=20, after_trade_count=1)
        assert m.verdict == Verdict.INSUFFICIENT_DATA

    def test_inconclusive_low_quality(self):
        m = _base_measurement(measurement_quality=MeasurementQuality.LOW)
        assert m.verdict == Verdict.INCONCLUSIVE

    def test_inconclusive_insufficient_quality(self):
        m = _base_measurement(measurement_quality=MeasurementQuality.INSUFFICIENT)
        assert m.verdict == Verdict.INCONCLUSIVE

    def test_positive_when_pnl_up_and_wr_stable(self):
        # pnl_change = 20/100 = 0.2 > 0.1, wr_change = 0.65 - 0.6 = 0.05 >= -0.05
        m = _base_measurement(pnl_before=100, pnl_after=120, win_rate_before=0.6, win_rate_after=0.65)
        assert m.verdict == Verdict.POSITIVE

    def test_negative_when_pnl_drops_significantly(self):
        # pnl_change = -20/100 = -0.2 < -0.1
        m = _base_measurement(pnl_before=100, pnl_after=80)
        assert m.verdict == Verdict.NEGATIVE

    def test_negative_when_wr_drops_significantly(self):
        # pnl_change = 5/100 = 0.05 (not < -0.1), wr_change = 0.4 - 0.6 = -0.2 < -0.1
        m = _base_measurement(pnl_before=100, pnl_after=105, win_rate_before=0.6, win_rate_after=0.4)
        assert m.verdict == Verdict.NEGATIVE

    def test_neutral_borderline_case(self):
        # pnl_change = 5/100 = 0.05 (not > 0.1, not < -0.1), wr_change = 0.02 (not < -0.1)
        m = _base_measurement(pnl_before=100, pnl_after=105, win_rate_before=0.6, win_rate_after=0.62)
        assert m.verdict == Verdict.NEUTRAL

    def test_inconclusive_when_pnl_before_zero(self):
        m = _base_measurement(pnl_before=0, pnl_after=50)
        assert m.verdict == Verdict.INCONCLUSIVE


class TestOutcomeMeasurementPnlDelta:
    def test_pnl_delta_computed(self):
        m = _base_measurement(pnl_before=100, pnl_after=150)
        assert m.pnl_delta == pytest.approx(50.0)

    def test_pnl_delta_negative(self):
        m = _base_measurement(pnl_before=200, pnl_after=150)
        assert m.pnl_delta == pytest.approx(-50.0)


class TestComputeMeasurementQuality:
    def test_insufficient_when_few_before_trades(self):
        result = compute_measurement_quality(
            regime_matched=True, before_trade_count=2, after_trade_count=20,
            volatility_ratio=1.0, concurrent_changes=[],
        )
        assert result == MeasurementQuality.INSUFFICIENT

    def test_insufficient_when_few_after_trades(self):
        result = compute_measurement_quality(
            regime_matched=True, before_trade_count=20, after_trade_count=1,
            volatility_ratio=1.0, concurrent_changes=[],
        )
        assert result == MeasurementQuality.INSUFFICIENT

    def test_low_when_regime_not_matched(self):
        result = compute_measurement_quality(
            regime_matched=False, before_trade_count=20, after_trade_count=20,
            volatility_ratio=1.0, concurrent_changes=[],
        )
        assert result == MeasurementQuality.LOW

    def test_low_when_three_or_more_concurrent_changes(self):
        result = compute_measurement_quality(
            regime_matched=True, before_trade_count=20, after_trade_count=20,
            volatility_ratio=1.0, concurrent_changes=["a", "b", "c"],
        )
        assert result == MeasurementQuality.LOW

    def test_high_when_no_issues(self):
        result = compute_measurement_quality(
            regime_matched=True, before_trade_count=20, after_trade_count=20,
            volatility_ratio=1.0, concurrent_changes=[],
        )
        assert result == MeasurementQuality.HIGH

    def test_medium_when_one_issue_high_volatility(self):
        # volatility_ratio > 2.0 is 1 issue, all else clean
        result = compute_measurement_quality(
            regime_matched=True, before_trade_count=20, after_trade_count=20,
            volatility_ratio=2.5, concurrent_changes=[],
        )
        assert result == MeasurementQuality.MEDIUM

    def test_low_when_two_issues(self):
        # volatility_ratio > 2.0 (1 issue) + 1 concurrent change (1 issue) = 2 issues
        result = compute_measurement_quality(
            regime_matched=True, before_trade_count=20, after_trade_count=20,
            volatility_ratio=3.0, concurrent_changes=["x"],
        )
        assert result == MeasurementQuality.LOW


class TestComputeSignificance:
    def test_returns_zero_when_noise_zero(self):
        assert compute_significance(10.0, 0.0, 10, 10) == 0.0

    def test_returns_zero_when_n_before_zero(self):
        assert compute_significance(10.0, 5.0, 0, 10) == 0.0

    def test_returns_zero_when_n_after_zero(self):
        assert compute_significance(10.0, 5.0, 10, 0) == 0.0

    def test_positive_for_valid_inputs(self):
        result = compute_significance(10.0, 5.0, 10, 10)
        assert result > 0.0

    def test_scales_with_effect_size(self):
        small = compute_significance(5.0, 5.0, 10, 10)
        large = compute_significance(20.0, 5.0, 10, 10)
        assert large > small

    def test_manual_calculation(self):
        # effect=10, noise=5, n_before=10, n_after=10
        # denominator = 5 * sqrt(1/10 + 1/10) = 5 * sqrt(0.2) = 5 * 0.4472...
        expected = 10.0 / (5.0 * math.sqrt(0.2))
        result = compute_significance(10.0, 5.0, 10, 10)
        assert result == pytest.approx(expected, rel=1e-6)


# ===================================================================
# 2. skills/auto_outcome_measurer.py
# ===================================================================


class TestAutoOutcomeMeasurerMeasure:
    """Tests for AutoOutcomeMeasurer.measure() using real filesystem."""

    def test_returns_none_when_no_summaries(self, tmp_path):
        measurer = AutoOutcomeMeasurer(curated_dir=tmp_path)
        result = measurer.measure("sug-1", "bot_a", "2026-03-07")
        assert result is None

    def test_returns_none_when_only_before_summaries(self, tmp_path):
        # Only write data before the implementation date, none after
        _write_summary(tmp_path, "2026-03-05", "bot_a")
        _write_summary(tmp_path, "2026-03-06", "bot_a")
        measurer = AutoOutcomeMeasurer(curated_dir=tmp_path)
        result = measurer.measure("sug-1", "bot_a", "2026-03-07", before_days=3, after_days=3)
        assert result is None

    def test_measure_computes_basic_metrics(self, tmp_path):
        # Before: 2 days of data, After: 2 days of data
        _write_summary(tmp_path, "2026-03-05", "bot_a", net_pnl=50, total_trades=5, win_count=3)
        _write_summary(tmp_path, "2026-03-06", "bot_a", net_pnl=60, total_trades=5, win_count=3)
        _write_summary(tmp_path, "2026-03-07", "bot_a", net_pnl=80, total_trades=8, win_count=5)
        _write_summary(tmp_path, "2026-03-08", "bot_a", net_pnl=90, total_trades=7, win_count=4)
        measurer = AutoOutcomeMeasurer(curated_dir=tmp_path)
        result = measurer.measure("sug-1", "bot_a", "2026-03-07", before_days=3, after_days=3)
        assert result is not None
        assert result.pnl_before == pytest.approx(110.0)  # 50 + 60
        assert result.pnl_after == pytest.approx(170.0)   # 80 + 90
        assert result.before_trade_count == 10             # 5 + 5
        assert result.after_trade_count == 15              # 8 + 7

    def test_measure_computes_regime_from_files(self, tmp_path):
        for d in range(1, 8):
            date = f"2026-03-{d:02d}"
            _write_summary(tmp_path, date, "bot_a", net_pnl=50, total_trades=5, win_count=3)
        _write_regime(tmp_path, "2026-03-01", "bot_a", "trending")
        _write_regime(tmp_path, "2026-03-02", "bot_a", "trending")
        _write_regime(tmp_path, "2026-03-04", "bot_a", "trending")
        _write_regime(tmp_path, "2026-03-05", "bot_a", "trending")

        measurer = AutoOutcomeMeasurer(curated_dir=tmp_path)
        result = measurer.measure("sug-1", "bot_a", "2026-03-04", before_days=4, after_days=4)
        assert result is not None
        assert result.before_regime == "trending"
        assert result.after_regime == "trending"
        assert result.regime_matched is True

    def test_measure_detects_regime_mismatch(self, tmp_path):
        for d in range(1, 8):
            date = f"2026-03-{d:02d}"
            _write_summary(tmp_path, date, "bot_a", net_pnl=50, total_trades=5, win_count=3)
        # Before period: trending
        _write_regime(tmp_path, "2026-03-01", "bot_a", "trending")
        _write_regime(tmp_path, "2026-03-02", "bot_a", "trending")
        _write_regime(tmp_path, "2026-03-03", "bot_a", "trending")
        # After period: ranging
        _write_regime(tmp_path, "2026-03-04", "bot_a", "ranging")
        _write_regime(tmp_path, "2026-03-05", "bot_a", "ranging")
        _write_regime(tmp_path, "2026-03-06", "bot_a", "ranging")

        measurer = AutoOutcomeMeasurer(curated_dir=tmp_path)
        result = measurer.measure("sug-1", "bot_a", "2026-03-04", before_days=4, after_days=4)
        assert result is not None
        assert result.before_regime == "trending"
        assert result.after_regime == "ranging"
        assert result.regime_matched is False

    def test_measure_computes_volatility(self, tmp_path):
        # Write enough days with varied PnL for stdev > 0
        _write_summary(tmp_path, "2026-03-01", "bot_a", net_pnl=10, total_trades=5, win_count=3)
        _write_summary(tmp_path, "2026-03-02", "bot_a", net_pnl=50, total_trades=5, win_count=3)
        _write_summary(tmp_path, "2026-03-03", "bot_a", net_pnl=20, total_trades=5, win_count=3)
        _write_summary(tmp_path, "2026-03-04", "bot_a", net_pnl=60, total_trades=5, win_count=3)

        measurer = AutoOutcomeMeasurer(curated_dir=tmp_path)
        result = measurer.measure("sug-1", "bot_a", "2026-03-03", before_days=3, after_days=3)
        assert result is not None
        assert result.before_volatility > 0
        assert result.after_volatility > 0

    def test_measure_finds_concurrent_changes(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()
        # Write the target suggestion
        _write_suggestion(findings, "sug-target", "bot_a", "deployed", "2026-03-04T00:00:00")
        # Write a concurrent deployed suggestion within the window
        _write_suggestion(findings, "sug-other", "bot_a", "deployed", "2026-03-05T00:00:00")

        for d in range(1, 12):
            _write_summary(tmp_path, f"2026-03-{d:02d}", "bot_a", net_pnl=50, total_trades=5, win_count=3)

        measurer = AutoOutcomeMeasurer(curated_dir=tmp_path, findings_dir=findings)
        result = measurer.measure("sug-target", "bot_a", "2026-03-04", before_days=7, after_days=7)
        assert result is not None
        assert "sug-other" in result.concurrent_changes

    def test_measure_computes_measurement_quality(self, tmp_path):
        for d in range(1, 15):
            _write_summary(tmp_path, f"2026-03-{d:02d}", "bot_a", net_pnl=50, total_trades=10, win_count=6)
        measurer = AutoOutcomeMeasurer(curated_dir=tmp_path)
        result = measurer.measure("sug-1", "bot_a", "2026-03-07", before_days=7, after_days=7)
        assert result is not None
        assert result.measurement_quality in list(MeasurementQuality)

    def test_measure_computes_significance_score(self, tmp_path):
        # Use varied PnL so noise > 0 and significance is non-trivial
        _write_summary(tmp_path, "2026-03-01", "bot_a", net_pnl=10, total_trades=5, win_count=3)
        _write_summary(tmp_path, "2026-03-02", "bot_a", net_pnl=50, total_trades=5, win_count=3)
        _write_summary(tmp_path, "2026-03-03", "bot_a", net_pnl=200, total_trades=8, win_count=6)
        _write_summary(tmp_path, "2026-03-04", "bot_a", net_pnl=250, total_trades=7, win_count=5)

        measurer = AutoOutcomeMeasurer(curated_dir=tmp_path)
        result = measurer.measure("sug-1", "bot_a", "2026-03-03", before_days=3, after_days=3)
        assert result is not None
        assert isinstance(result.significance_score, float)

    def test_measure_returns_all_enhanced_fields_populated(self, tmp_path):
        for d in range(1, 15):
            _write_summary(tmp_path, f"2026-03-{d:02d}", "bot_a", net_pnl=50 + d, total_trades=10, win_count=6)
            _write_regime(tmp_path, f"2026-03-{d:02d}", "bot_a", "trending")

        measurer = AutoOutcomeMeasurer(curated_dir=tmp_path)
        result = measurer.measure("sug-1", "bot_a", "2026-03-07", before_days=7, after_days=7)
        assert result is not None
        # All enhanced fields should be populated
        assert result.before_regime != ""
        assert result.after_regime != ""
        assert isinstance(result.regime_matched, bool)
        assert isinstance(result.before_volatility, float)
        assert isinstance(result.after_volatility, float)
        assert isinstance(result.volatility_ratio, float)
        assert isinstance(result.measurement_quality, MeasurementQuality)
        assert isinstance(result.effect_size, float)
        assert isinstance(result.noise_estimate, float)
        assert isinstance(result.significance_score, float)


class TestDominantRegime:
    def test_empty_input(self):
        assert AutoOutcomeMeasurer._dominant_regime([]) == ""

    def test_picks_most_frequent(self):
        data = [
            {"dominant_regime": "trending"},
            {"dominant_regime": "trending"},
            {"dominant_regime": "ranging"},
        ]
        assert AutoOutcomeMeasurer._dominant_regime(data) == "trending"

    def test_handles_missing_key(self):
        data = [{"other": "value"}, {"other": "value"}]
        assert AutoOutcomeMeasurer._dominant_regime(data) == ""


class TestComputeVolatility:
    def test_returns_zero_for_single_data_point(self):
        assert AutoOutcomeMeasurer._compute_volatility([{"net_pnl": 100}]) == 0.0

    def test_returns_zero_for_empty_list(self):
        assert AutoOutcomeMeasurer._compute_volatility([]) == 0.0

    def test_positive_stdev_for_varied_data(self):
        summaries = [{"net_pnl": 10}, {"net_pnl": 50}, {"net_pnl": 30}]
        result = AutoOutcomeMeasurer._compute_volatility(summaries)
        assert result > 0.0


class TestFindConcurrentChanges:
    def test_filters_by_bot_id_and_window(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()
        # Same bot, deployed within window
        _write_suggestion(findings, "sug-nearby", "bot_a", "deployed", "2026-03-06T00:00:00")
        # Different bot, should be excluded
        _write_suggestion(findings, "sug-other-bot", "bot_b", "deployed", "2026-03-06T00:00:00")
        # Same bot, outside window (too far in the future)
        _write_suggestion(findings, "sug-far", "bot_a", "deployed", "2026-04-01T00:00:00")

        measurer = AutoOutcomeMeasurer(curated_dir=tmp_path, findings_dir=findings)
        result = measurer._find_concurrent_changes("sug-target", "bot_a", "2026-03-05", 7)
        assert "sug-nearby" in result
        assert "sug-other-bot" not in result
        assert "sug-far" not in result

    def test_returns_empty_when_no_findings_dir(self, tmp_path):
        measurer = AutoOutcomeMeasurer(curated_dir=tmp_path, findings_dir=None)
        result = measurer._find_concurrent_changes("sug-1", "bot_a", "2026-03-05", 7)
        assert result == []


class TestEstimateNoise:
    def test_returns_zero_for_single_value(self):
        assert AutoOutcomeMeasurer._estimate_noise([42.0]) == 0.0

    def test_returns_zero_for_empty(self):
        assert AutoOutcomeMeasurer._estimate_noise([]) == 0.0

    def test_positive_stdev_for_varied_data(self):
        result = AutoOutcomeMeasurer._estimate_noise([10.0, 50.0, 30.0])
        assert result > 0.0


# ===================================================================
# 3. skills/suggestion_scorer.py — quality/verdict filtering
# ===================================================================


class TestSuggestionScorerQualityFiltering:
    """Scorer filters out low-quality outcomes (only high/medium counted)."""

    def test_filters_out_low_quality_outcomes(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()

        # Three suggestions, each with one outcome at a different quality level
        _write_suggestion(findings, "sug-1", "bot_a", "deployed")
        _write_suggestion(findings, "sug-2", "bot_a", "deployed")
        _write_suggestion(findings, "sug-3", "bot_a", "deployed")

        # High quality outcome (positive) — should be counted
        _write_outcome(findings, "sug-1", measurement_quality="high", verdict="positive")
        # Low quality outcome (positive) — should be excluded
        _write_outcome(findings, "sug-2", measurement_quality="low", verdict="positive")
        # Insufficient quality outcome — should be excluded
        _write_outcome(findings, "sug-3", measurement_quality="insufficient", verdict="positive")

        scorer = SuggestionScorer(findings)
        scorecard = scorer.compute_scorecard()
        # Only 1 high-quality outcome should be counted
        assert len(scorecard.scores) == 1
        assert scorecard.scores[0].sample_size == 1

    def test_includes_medium_quality(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()

        _write_suggestion(findings, "sug-1", "bot_a", "deployed")
        _write_suggestion(findings, "sug-2", "bot_a", "deployed")
        _write_outcome(findings, "sug-1", measurement_quality="medium", verdict="positive")
        _write_outcome(findings, "sug-2", measurement_quality="medium", verdict="negative")

        scorer = SuggestionScorer(findings)
        scorecard = scorer.compute_scorecard()
        assert len(scorecard.scores) == 1
        assert scorecard.scores[0].sample_size == 2


class TestSuggestionScorerVerdictField:
    """Scorer uses verdict field when available for win detection."""

    def test_uses_verdict_for_positive_detection(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()

        _write_suggestion(findings, "sug-1", "bot_a", "deployed")
        _write_suggestion(findings, "sug-2", "bot_a", "deployed")
        _write_outcome(findings, "sug-1", measurement_quality="high", verdict="positive")
        _write_outcome(findings, "sug-2", measurement_quality="high", verdict="negative")

        scorer = SuggestionScorer(findings)
        scorecard = scorer.compute_scorecard()
        assert len(scorecard.scores) == 1
        assert scorecard.scores[0].win_rate == pytest.approx(0.5)

    def test_legacy_pnl_delta_7d_fallback(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()

        _write_suggestion(findings, "sug-1", "bot_a", "deployed")
        _write_suggestion(findings, "sug-2", "bot_a", "deployed")
        # No verdict field, falls back to pnl_delta_7d
        _write_outcome(findings, "sug-1", measurement_quality="high", pnl_delta_7d=100.0)
        _write_outcome(findings, "sug-2", measurement_quality="high", pnl_delta_7d=-50.0)

        scorer = SuggestionScorer(findings)
        scorecard = scorer.compute_scorecard()
        assert len(scorecard.scores) == 1
        assert scorecard.scores[0].win_rate == pytest.approx(0.5)

    def test_pnl_delta_field_fallback(self, tmp_path):
        """pnl_delta from OutcomeMeasurement schema used as fallback."""
        findings = tmp_path / "findings"
        findings.mkdir()

        _write_suggestion(findings, "sug-1", "bot_a", "deployed")
        # No verdict, no pnl_delta_7d, uses pnl_delta
        _write_outcome(findings, "sug-1", measurement_quality="high", pnl_delta=75.0)

        scorer = SuggestionScorer(findings)
        scorecard = scorer.compute_scorecard()
        assert len(scorecard.scores) == 1
        assert scorecard.scores[0].win_rate == pytest.approx(1.0)  # positive

    def test_empty_outcomes_returns_empty_scorecard(self, tmp_path):
        findings = tmp_path / "findings"
        findings.mkdir()
        _write_suggestion(findings, "sug-1", "bot_a", "deployed")
        # No outcomes file at all
        scorer = SuggestionScorer(findings)
        scorecard = scorer.compute_scorecard()
        assert scorecard.scores == []


# ===================================================================
# 4. _estimate_composite_delta formula alignment
# ===================================================================


class TestEstimateCompositeDelta:
    """Verify _estimate_composite_delta uses window_days (calendar days)
    not trade_count, and matches the soul.md-aligned formula:
    0.4 * calmar_imp + 0.3 * pf_imp - 0.3 * max(0, dd_inc).
    """

    def test_improvement_returns_positive(self):
        m = _base_measurement(
            pnl_before=70.0, pnl_after=140.0,
            win_rate_before=0.5, win_rate_after=0.6,
            drawdown_before=0.05, drawdown_after=0.04,
            before_volatility=10.0, after_volatility=10.0,
            window_days=7,
        )
        delta = AutoOutcomeMeasurer._estimate_composite_delta(m)
        assert delta > 0

    def test_decline_returns_negative(self):
        m = _base_measurement(
            pnl_before=140.0, pnl_after=70.0,
            win_rate_before=0.6, win_rate_after=0.4,
            drawdown_before=0.03, drawdown_after=0.10,
            before_volatility=10.0, after_volatility=10.0,
            window_days=7,
        )
        delta = AutoOutcomeMeasurer._estimate_composite_delta(m)
        assert delta < 0

    def test_uses_window_days_not_trade_count(self):
        """Calmar proxy must use calendar days, not trade count."""
        m = _base_measurement(
            pnl_before=70.0, pnl_after=70.0,
            win_rate_before=0.5, win_rate_after=0.5,
            drawdown_before=0.05, drawdown_after=0.05,
            before_volatility=10.0, after_volatility=10.0,
            before_trade_count=50, after_trade_count=5,
            window_days=7,
        )
        # With identical PnL/vol/wr/dd, delta should be ~0 regardless of trade count
        delta = AutoOutcomeMeasurer._estimate_composite_delta(m)
        assert delta == pytest.approx(0.0, abs=1e-9)

    def test_drawdown_increase_penalized(self):
        m = _base_measurement(
            pnl_before=70.0, pnl_after=70.0,
            win_rate_before=0.5, win_rate_after=0.5,
            drawdown_before=0.02, drawdown_after=0.12,
            before_volatility=10.0, after_volatility=10.0,
            window_days=7,
        )
        delta = AutoOutcomeMeasurer._estimate_composite_delta(m)
        # Drawdown increased: calmar worsened, PF efficiency worsened, dd penalty
        assert delta < 0

    def test_drawdown_decrease_improves_delta(self):
        """Drawdown decrease improves calmar and PF efficiency → positive delta."""
        m = _base_measurement(
            pnl_before=70.0, pnl_after=70.0,
            win_rate_before=0.5, win_rate_after=0.5,
            drawdown_before=0.12, drawdown_after=0.02,
            before_volatility=10.0, after_volatility=10.0,
            window_days=7,
        )
        delta = AutoOutcomeMeasurer._estimate_composite_delta(m)
        # dd_inc = -0.10, max(0, -0.10) = 0 → no dd penalty
        # But calmar and PF efficiency improve with lower drawdown
        assert delta > 0
