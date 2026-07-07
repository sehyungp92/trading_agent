# tests/test_adaptive_integration.py
"""Integration tests — ThresholdLearner → StrategyEngine adaptive threshold wiring.

ThresholdLearner learns optimal thresholds from suggestion+outcome data.
StrategyEngine accepts an optional threshold_learner parameter to use learned values.
These tests verify the full loop: learn → inject → detect with learned thresholds.
"""
from __future__ import annotations

import json
from pathlib import Path


from trading_assistant.analysis.strategy_engine import StrategyEngine
from trading_assistant.schemas.detection_context import DetectionContext
from trading_assistant.schemas.weekly_metrics import BotWeeklySummary, FilterWeeklySummary
from trading_assistant.skills.threshold_learner import ThresholdLearner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_suggestions(path: Path, suggestions: list[dict]) -> None:
    """Write suggestion dicts to suggestions.jsonl."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in suggestions:
            f.write(json.dumps(s, default=str) + "\n")


def _write_outcomes(path: Path, outcomes: list[dict]) -> None:
    """Write outcome dicts to outcomes.jsonl."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for o in outcomes:
            f.write(json.dumps(o, default=str) + "\n")


def _make_suggestion_dict(
    suggestion_id: str,
    bot_id: str,
    detector_name: str,
    threshold_name: str,
    threshold_value: float,
    observed_value: float,
) -> dict:
    """Create a suggestion dict with detection_context for JSONL writing."""
    ctx = DetectionContext(
        detector_name=detector_name,
        bot_id=bot_id,
        threshold_name=threshold_name,
        threshold_value=threshold_value,
        observed_value=observed_value,
    )
    return {
        "suggestion_id": suggestion_id,
        "bot_id": bot_id,
        "title": f"Suggestion {suggestion_id}",
        "tier": "parameter",
        "category": "entry_signal",
        "source_report_id": "weekly-test",
        "detection_context": ctx.model_dump(),
    }


def _make_outcome_dict(suggestion_id: str, pnl_delta_7d: float) -> dict:
    """Create an outcome dict for JSONL writing."""
    return {
        "suggestion_id": suggestion_id,
        "implemented_date": "2026-03-01",
        "pnl_delta_7d": pnl_delta_7d,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAdaptiveIntegration:
    """Tests for ThresholdLearner standalone and StrategyEngine with defaults."""

    def test_strategy_engine_detects_with_defaults(self):
        """StrategyEngine generates suggestions using default thresholds."""
        engine = StrategyEngine(
            week_start="2026-02-24",
            week_end="2026-03-02",
            tight_stop_ratio=0.3,
        )

        # Create a bot summary that triggers tight_stop detection (ratio < 0.3)
        summary = BotWeeklySummary(
            week_start="2026-02-24",
            week_end="2026-03-02",
            bot_id="bot_a",
            total_trades=50,
            win_count=30,
            loss_count=20,
            avg_win=200.0,
            avg_loss=-40.0,  # ratio = 40/200 = 0.2, < 0.3 threshold
        )
        suggestions = engine.analyze_parameters(summary)
        assert len(suggestions) == 1
        assert suggestions[0].detection_context is not None
        assert suggestions[0].detection_context.threshold_value == 0.3  # default

    def test_threshold_learner_learns_from_data(self, tmp_path: Path):
        """ThresholdLearner learns optimal thresholds from suggestion+outcome pairs."""
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        suggestion_dicts = []
        outcome_dicts = []
        for i in range(15):
            sid = f"s{i:03d}"
            observed = 0.05 + i * 0.016
            suggestion_dicts.append(_make_suggestion_dict(
                suggestion_id=sid,
                bot_id="bot_a",
                detector_name="tight_stop",
                threshold_name="tight_stop_ratio",
                threshold_value=0.3,
                observed_value=round(observed, 4),
            ))
            pnl = 500.0 if observed >= 0.15 else -200.0
            outcome_dicts.append(_make_outcome_dict(sid, pnl))

        _write_suggestions(findings_dir / "suggestions.jsonl", suggestion_dicts)
        _write_outcomes(findings_dir / "outcomes.jsonl", outcome_dicts)

        learner = ThresholdLearner(findings_dir, min_samples=10)
        profiles = learner.learn_thresholds()

        assert "bot_a" in profiles
        profile = profiles["bot_a"]
        key = "tight_stop:tight_stop_ratio"
        assert key in profile.thresholds
        record = profile.thresholds[key]
        assert record.sample_count == 15
        assert record.learned_value is not None

    def test_defaults_used_without_learner(self):
        """StrategyEngine uses default thresholds across all detector tiers."""
        engine = StrategyEngine(
            week_start="2026-02-24",
            week_end="2026-03-02",
            tight_stop_ratio=0.3,
            filter_cost_threshold=0.0,
            regime_min_weeks=3,
        )

        # --- Tier 1: analyze_parameters uses default tight_stop_ratio ---
        summary = BotWeeklySummary(
            week_start="2026-02-24",
            week_end="2026-03-02",
            bot_id="bot_x",
            total_trades=40,
            win_count=25,
            loss_count=15,
            avg_win=100.0,
            avg_loss=-20.0,  # ratio = 0.2 < 0.3
        )
        param_suggestions = engine.analyze_parameters(summary)
        assert len(param_suggestions) == 1
        ctx = param_suggestions[0].detection_context
        assert ctx is not None
        assert ctx.threshold_value == 0.3  # default tight_stop_ratio

        # --- Tier 2: analyze_filters uses default filter_cost_threshold ---
        filter_summaries = [FilterWeeklySummary(
            bot_id="bot_x",
            filter_name="regime_filter",
            total_blocks=10,
            blocks_that_would_have_won=7,
            net_impact_pnl=-500.0,
            confidence=0.8,
        )]
        filter_suggestions = engine.analyze_filters("bot_x", filter_summaries)
        assert len(filter_suggestions) == 1
        ctx2 = filter_suggestions[0].detection_context
        assert ctx2 is not None
        assert ctx2.threshold_value == 0.0  # default filter_cost_threshold

        # --- Tier 4: alpha_decay uses its default ---
        decay_suggestions = engine.detect_alpha_decay(
            bot_id="bot_x",
            rolling_sharpe_30d=0.5,
            rolling_sharpe_60d=0.8,
            rolling_sharpe_90d=1.0,
            decay_threshold=0.3,
        )
        assert len(decay_suggestions) == 1
        ctx3 = decay_suggestions[0].detection_context
        assert ctx3 is not None
        assert ctx3.threshold_value == 0.3  # default decay_threshold

        # All detection_context values use defaults
        for s in param_suggestions + filter_suggestions + decay_suggestions:
            assert s.detection_context is not None
            assert s.detection_context.detector_name in (
                "tight_stop", "filter_cost", "alpha_decay",
            )

    def test_old_suggestions_without_context_skipped(self, tmp_path: Path):
        """Legacy suggestions without detection_context are gracefully skipped."""
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        suggestion_dicts = []
        outcome_dicts = []

        # 8 suggestions WITH detection_context
        for i in range(8):
            sid = f"new{i:03d}"
            suggestion_dicts.append(_make_suggestion_dict(
                suggestion_id=sid,
                bot_id="bot_a",
                detector_name="tight_stop",
                threshold_name="tight_stop_ratio",
                threshold_value=0.3,
                observed_value=0.1 + i * 0.02,
            ))
            outcome_dicts.append(_make_outcome_dict(sid, 200.0))

        # 7 legacy suggestions WITHOUT detection_context
        for i in range(7):
            sid = f"old{i:03d}"
            suggestion_dicts.append({
                "suggestion_id": sid,
                "bot_id": "bot_a",
                "title": f"Legacy suggestion {sid}",
                "tier": "parameter",
                # No detection_context field
            })
            outcome_dicts.append(_make_outcome_dict(sid, 100.0))

        _write_suggestions(findings_dir / "suggestions.jsonl", suggestion_dicts)
        _write_outcomes(findings_dir / "outcomes.jsonl", outcome_dicts)

        # With min_samples=5, the 8 context-bearing suggestions are enough
        learner = ThresholdLearner(findings_dir, min_samples=5)
        profiles = learner.learn_thresholds()

        # Should succeed — only the 8 with context are used
        assert "bot_a" in profiles
        key = "tight_stop:tight_stop_ratio"
        assert key in profiles["bot_a"].thresholds
        assert profiles["bot_a"].thresholds[key].sample_count == 8

        # With min_samples=10, the 8 are not enough
        learner2 = ThresholdLearner(findings_dir, min_samples=10)
        profiles2 = learner2.learn_thresholds()
        assert profiles2 == {}

    def test_threshold_drift_is_detectable(self, tmp_path: Path):
        """Learned threshold clearly differs from the default when data warrants it."""
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        suggestion_dicts = []
        outcome_dicts = []

        # Create data where optimal threshold is clearly NOT the default (0.3).
        # All positive outcomes have observed_value >= 0.10, all negative < 0.10.
        # So optimal threshold should converge near 0.10, far from 0.3 default.
        for i in range(20):
            sid = f"s{i:03d}"
            if i < 10:
                # Low observed values -> negative outcomes
                observed = 0.02 + i * 0.008  # 0.02 to 0.092
                pnl = -300.0
            else:
                # High observed values -> positive outcomes
                observed = 0.10 + (i - 10) * 0.02  # 0.10 to 0.28
                pnl = 500.0

            suggestion_dicts.append(_make_suggestion_dict(
                suggestion_id=sid,
                bot_id="bot_drift",
                detector_name="tight_stop",
                threshold_name="tight_stop_ratio",
                threshold_value=0.3,
                observed_value=round(observed, 4),
            ))
            outcome_dicts.append(_make_outcome_dict(sid, pnl))

        _write_suggestions(findings_dir / "suggestions.jsonl", suggestion_dicts)
        _write_outcomes(findings_dir / "outcomes.jsonl", outcome_dicts)

        learner = ThresholdLearner(findings_dir, min_samples=10)
        profiles = learner.learn_thresholds()

        assert "bot_drift" in profiles
        key = "tight_stop:tight_stop_ratio"
        record = profiles["bot_drift"].thresholds[key]

        # The learned value should differ from the default (0.3)
        assert record.default_value == 0.3
        assert record.learned_value is not None
        assert record.learned_value != record.default_value

        # The drift should be substantial (learned should be near 0.10, not 0.3)
        drift = abs(record.learned_value - record.default_value)
        assert drift > 0.05, (
            f"Expected substantial drift from default 0.3, "
            f"got learned_value={record.learned_value} (drift={drift})"
        )

        # Confidence should be positive (20 samples / 30 cap)
        assert record.confidence > 0
        assert record.sample_count == 20

    def test_per_bot_learner_specialization(self, tmp_path: Path):
        """Two bots with different outcome distributions get different learned thresholds."""
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        suggestion_dicts = []
        outcome_dicts = []

        # bot_a: positive outcomes at HIGH observed values (>= 0.20)
        for i in range(15):
            sid = f"a{i:03d}"
            observed = 0.05 + i * 0.02  # 0.05 to 0.33
            suggestion_dicts.append(_make_suggestion_dict(
                suggestion_id=sid,
                bot_id="bot_a",
                detector_name="tight_stop",
                threshold_name="tight_stop_ratio",
                threshold_value=0.3,
                observed_value=round(observed, 4),
            ))
            pnl = 300.0 if observed >= 0.20 else -150.0
            outcome_dicts.append(_make_outcome_dict(sid, pnl))

        # bot_b: positive outcomes at LOW observed values (< 0.15)
        for i in range(15):
            sid = f"b{i:03d}"
            observed = 0.05 + i * 0.02  # 0.05 to 0.33
            suggestion_dicts.append(_make_suggestion_dict(
                suggestion_id=sid,
                bot_id="bot_b",
                detector_name="tight_stop",
                threshold_name="tight_stop_ratio",
                threshold_value=0.3,
                observed_value=round(observed, 4),
            ))
            pnl = 300.0 if observed < 0.15 else -150.0
            outcome_dicts.append(_make_outcome_dict(sid, pnl))

        _write_suggestions(findings_dir / "suggestions.jsonl", suggestion_dicts)
        _write_outcomes(findings_dir / "outcomes.jsonl", outcome_dicts)

        learner = ThresholdLearner(findings_dir, min_samples=10)
        profiles = learner.learn_thresholds()

        assert "bot_a" in profiles
        assert "bot_b" in profiles

        key = "tight_stop:tight_stop_ratio"
        rec_a = profiles["bot_a"].thresholds[key]
        rec_b = profiles["bot_b"].thresholds[key]

        # The learned thresholds should differ since the outcome distributions differ
        assert rec_a.learned_value != rec_b.learned_value

    def test_insufficient_data_returns_empty(self, tmp_path: Path):
        """Only 5 suggestion+outcome pairs -- below min_samples, no profiles computed."""
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        suggestion_dicts = []
        outcome_dicts = []
        for i in range(5):
            sid = f"s{i:03d}"
            suggestion_dicts.append(_make_suggestion_dict(
                suggestion_id=sid,
                bot_id="bot_a",
                detector_name="tight_stop",
                threshold_name="tight_stop_ratio",
                threshold_value=0.3,
                observed_value=0.1 + i * 0.02,
            ))
            outcome_dicts.append(_make_outcome_dict(sid, 100.0))

        _write_suggestions(findings_dir / "suggestions.jsonl", suggestion_dicts)
        _write_outcomes(findings_dir / "outcomes.jsonl", outcome_dicts)

        learner = ThresholdLearner(findings_dir, min_samples=10)
        profiles = learner.learn_thresholds()

        # Below min_samples=10 so no profiles computed
        assert profiles == {}

    def test_full_loop_learner_to_engine(self, tmp_path: Path):
        """Full loop: write data → learn → inject into engine → verify engine uses learned threshold."""
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        suggestion_dicts = []
        outcome_dicts = []
        # Create data where optimal threshold differs from default 0.3:
        # positive outcomes at observed >= 0.10, negative below
        for i in range(20):
            sid = f"s{i:03d}"
            if i < 10:
                observed = 0.02 + i * 0.008
                pnl = -300.0
            else:
                observed = 0.10 + (i - 10) * 0.02
                pnl = 500.0
            suggestion_dicts.append(_make_suggestion_dict(
                suggestion_id=sid,
                bot_id="bot_a",
                detector_name="tight_stop",
                threshold_name="tight_stop_ratio",
                threshold_value=0.3,
                observed_value=round(observed, 4),
            ))
            outcome_dicts.append(_make_outcome_dict(sid, pnl))

        _write_suggestions(findings_dir / "suggestions.jsonl", suggestion_dicts)
        _write_outcomes(findings_dir / "outcomes.jsonl", outcome_dicts)

        # Learn thresholds
        learner = ThresholdLearner(findings_dir, min_samples=10)
        learner.learn_thresholds()

        # Create a new learner instance (simulates reload) and inject into engine
        learner2 = ThresholdLearner(findings_dir, min_samples=10)
        engine = StrategyEngine(
            week_start="2026-02-24", week_end="2026-03-02",
            threshold_learner=learner2,
        )

        # loss/win ratio = 20/100 = 0.2
        # Default threshold = 0.3 → would trigger (0.2 < 0.3)
        # Learned threshold should be near 0.10 → should NOT trigger (0.2 >= 0.10)
        summary = BotWeeklySummary(
            week_start="2026-02-24", week_end="2026-03-02",
            bot_id="bot_a", total_trades=50,
            win_count=30, loss_count=20,
            avg_win=100.0, avg_loss=-20.0,
        )
        result = engine.analyze_parameters(summary)
        # With learned threshold near 0.10, ratio 0.2 should NOT trigger
        assert len(result) == 0

    def test_engine_with_learner_mixed_coverage(self, tmp_path: Path):
        """Learner has data for some detectors but not others."""
        findings_dir = tmp_path / "findings"
        findings_dir.mkdir()

        suggestion_dicts = []
        outcome_dicts = []
        # Only provide tight_stop data — no filter_cost data
        for i in range(15):
            sid = f"s{i:03d}"
            observed = 0.05 + i * 0.016
            suggestion_dicts.append(_make_suggestion_dict(
                suggestion_id=sid,
                bot_id="bot_a",
                detector_name="tight_stop",
                threshold_name="tight_stop_ratio",
                threshold_value=0.3,
                observed_value=round(observed, 4),
            ))
            pnl = 500.0 if observed >= 0.15 else -200.0
            outcome_dicts.append(_make_outcome_dict(sid, pnl))

        _write_suggestions(findings_dir / "suggestions.jsonl", suggestion_dicts)
        _write_outcomes(findings_dir / "outcomes.jsonl", outcome_dicts)

        learner = ThresholdLearner(findings_dir, min_samples=10)
        learner.learn_thresholds()

        learner2 = ThresholdLearner(findings_dir, min_samples=10)
        engine = StrategyEngine(
            week_start="2026-02-24", week_end="2026-03-02",
            threshold_learner=learner2,
        )

        # tight_stop: uses learned value (not default 0.3)
        summary = BotWeeklySummary(
            week_start="2026-02-24", week_end="2026-03-02",
            bot_id="bot_a", total_trades=50,
            win_count=30, loss_count=20,
            avg_win=100.0, avg_loss=-20.0,
        )
        engine.analyze_parameters(summary)
        # The learner should have adjusted the threshold from 0.3

        # filter_cost: no learned data → uses default 0.0
        f = FilterWeeklySummary(
            bot_id="bot_a", filter_name="test_filter",
            total_blocks=10, blocks_that_would_have_won=7,
            net_impact_pnl=-50.0, confidence=0.8,
        )
        filter_result = engine.analyze_filters("bot_a", [f])
        assert len(filter_result) == 1
        # Filter threshold should still be the default 0.0 (no learned data)
        assert filter_result[0].detection_context.threshold_value == 0.0
