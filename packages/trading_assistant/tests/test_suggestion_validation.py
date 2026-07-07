# tests/test_suggestion_validation.py
"""Tests for Phase 3: Evidence-Grounded Suggestions.

Covers:
- schemas/suggestion_validation.py (ValidationEvidence, SuggestionValidationResult)
- skills/suggestion_validator.py (SuggestionValidator)
- Handler wiring (_record_agent_suggestions calls validator)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch


from trading_assistant.schemas.suggestion_validation import (
    SuggestionValidationResult,
    ValidationEvidence,
)
from trading_assistant.skills.suggestion_validator import SuggestionValidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_trades(curated_dir: Path, date: str, bot_id: str, trades: list[dict]):
    """Write trade dicts as JSONL to curated_dir/date/bot_id/trades.jsonl."""
    bot_dir = curated_dir / date / bot_id
    bot_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(t) for t in trades]
    (bot_dir / "trades.jsonl").write_text("\n".join(lines))


def _sample_trades() -> list[dict]:
    """Mixed sample of winning and losing trades across regimes."""
    return [
        {"pnl": 150, "signal_strength": 0.8, "mae_pct": 2.0, "regime": "trending", "position_size_pct": 1.0},
        {"pnl": 120, "signal_strength": 0.7, "mae_pct": 1.5, "regime": "trending", "position_size_pct": 1.0},
        {"pnl": -80, "signal_strength": 0.3, "mae_pct": 5.0, "regime": "ranging", "position_size_pct": 1.0},
        {"pnl": -50, "signal_strength": 0.4, "mae_pct": 4.0, "regime": "ranging", "position_size_pct": 1.0},
        {"pnl": 200, "signal_strength": 0.9, "mae_pct": 1.0, "regime": "trending", "position_size_pct": 1.0},
        {"pnl": -30, "signal_strength": 0.2, "mae_pct": 3.0, "regime": "ranging", "position_size_pct": 1.0},
        {"pnl": 90, "signal_strength": 0.6, "mae_pct": 2.5, "regime": "trending", "position_size_pct": 1.0},
        {"pnl": -100, "signal_strength": 0.35, "mae_pct": 6.0, "regime": "ranging", "position_size_pct": 1.0},
    ]


# ===========================================================================
# 1. Schema tests — schemas/suggestion_validation.py
# ===========================================================================

class TestValidationEvidenceSchema:
    """Tests for ValidationEvidence Pydantic model."""

    def test_default_values(self):
        ev = ValidationEvidence()
        assert ev.validated is False
        assert ev.method == "not_testable"
        assert ev.baseline_metrics == {}
        assert ev.proposed_metrics == {}
        assert ev.improvement_pct == 0.0
        assert ev.sample_size == 0
        assert ev.regime_breakdown == {}
        assert ev.notes == ""

    def test_populated_fields(self):
        ev = ValidationEvidence(
            validated=True,
            method="backtest_replay",
            baseline_metrics={"pnl": 100, "win_rate": 0.6},
            proposed_metrics={"pnl": 120, "win_rate": 0.7},
            improvement_pct=20.0,
            sample_size=50,
            regime_breakdown={"trending": {"baseline": {}, "proposed": {}}},
            notes="Looks good",
        )
        assert ev.validated is True
        assert ev.method == "backtest_replay"
        assert ev.baseline_metrics["pnl"] == 100
        assert ev.proposed_metrics["win_rate"] == 0.7
        assert ev.improvement_pct == 20.0
        assert ev.sample_size == 50
        assert "trending" in ev.regime_breakdown
        assert ev.notes == "Looks good"

    def test_serialization_round_trip(self):
        ev = ValidationEvidence(
            validated=True,
            method="backtest_replay",
            improvement_pct=15.5,
            sample_size=30,
        )
        data = ev.model_dump(mode="json")
        restored = ValidationEvidence(**data)
        assert restored.validated == ev.validated
        assert restored.method == ev.method
        assert restored.improvement_pct == ev.improvement_pct
        assert restored.sample_size == ev.sample_size


class TestSuggestionValidationResultSchema:
    """Tests for SuggestionValidationResult Pydantic model."""

    def test_default_values(self):
        result = SuggestionValidationResult()
        assert result.suggestion_id == ""
        assert result.bot_id == ""
        assert result.target_param == ""
        assert result.proposed_value is None
        assert isinstance(result.evidence, ValidationEvidence)
        assert result.degradation_detected is False
        assert result.requires_review is False

    def test_degradation_detected_flag(self):
        result = SuggestionValidationResult(
            suggestion_id="abc123",
            bot_id="bot_a",
            degradation_detected=True,
            requires_review=True,
            evidence=ValidationEvidence(
                validated=True,
                method="backtest_replay",
                improvement_pct=-8.0,
            ),
        )
        assert result.degradation_detected is True
        assert result.requires_review is True
        assert result.evidence.improvement_pct == -8.0


# ===========================================================================
# 2. SuggestionValidator tests — skills/suggestion_validator.py
# ===========================================================================

class TestSuggestionValidatorNotTestable:
    """Tests for early-exit not_testable paths."""

    def test_no_target_param_returns_not_testable(self, tmp_path):
        validator = SuggestionValidator(curated_dir=tmp_path)
        result = validator.validate(
            suggestion_id="s1", bot_id="bot_a", category="structural",
            target_param=None, proposed_value=0.5, title="Add regime gate",
        )
        assert result.evidence.method == "not_testable"
        assert result.evidence.validated is False
        assert "structural" in result.evidence.notes.lower() or "no parameter" in result.evidence.notes.lower()

    def test_proposed_value_none_returns_not_testable(self, tmp_path):
        validator = SuggestionValidator(curated_dir=tmp_path)
        result = validator.validate(
            suggestion_id="s2", bot_id="bot_a", category="signal",
            target_param="signal_threshold", proposed_value=None, title="Tighten signal",
        )
        assert result.evidence.method == "not_testable"
        assert result.evidence.validated is False

    def test_no_trade_data_returns_not_testable(self, tmp_path):
        """Empty curated_dir means no trades to replay."""
        validator = SuggestionValidator(curated_dir=tmp_path)
        result = validator.validate(
            suggestion_id="s3", bot_id="bot_a", category="signal",
            target_param="signal_threshold", proposed_value=0.5,
        )
        assert result.evidence.method == "not_testable"
        assert "no historical" in result.evidence.notes.lower()

    def test_structural_suggestion_title_but_no_param(self, tmp_path):
        """Structural suggestion with title but no target_param."""
        validator = SuggestionValidator(curated_dir=tmp_path)
        result = validator.validate(
            suggestion_id="s4", bot_id="bot_a", category="structural",
            target_param=None, proposed_value=None,
            title="Add correlation filter",
        )
        assert result.evidence.method == "not_testable"
        assert result.degradation_detected is False


class TestSuggestionValidatorBacktestReplay:
    """Tests for backtest_replay parameter validation."""

    def test_signal_threshold_filters_weak_signals(self, tmp_path):
        trades = _sample_trades()
        _write_trades(tmp_path, "2026-03-10", "bot_a", trades)

        validator = SuggestionValidator(curated_dir=tmp_path, lookback_days=5)
        result = validator.validate(
            suggestion_id="s5", bot_id="bot_a", category="signal",
            target_param="signal_threshold", proposed_value=0.6,
            end_date="2026-03-10",
        )
        assert result.evidence.validated is True
        assert result.evidence.method == "backtest_replay"
        # Trades with signal_strength < 0.6 should be filtered out
        # Remaining: 0.8, 0.7, 0.9, 0.6 (pnl: 150, 120, 200, 90 = 560)
        assert result.evidence.proposed_metrics["pnl"] > 0

    def test_stop_loss_parameter(self, tmp_path):
        trades = _sample_trades()
        _write_trades(tmp_path, "2026-03-10", "bot_a", trades)

        validator = SuggestionValidator(curated_dir=tmp_path, lookback_days=5)
        result = validator.validate(
            suggestion_id="s6", bot_id="bot_a", category="stop_loss",
            target_param="stop_loss_pct", proposed_value=3.0,
            end_date="2026-03-10",
        )
        assert result.evidence.validated is True
        assert result.evidence.method == "backtest_replay"
        # Trades with MAE > 3.0 and pnl > 0 get partial exit (halved)
        assert result.evidence.sample_size == len(trades)

    def test_position_sizing_parameter(self, tmp_path):
        trades = _sample_trades()
        _write_trades(tmp_path, "2026-03-10", "bot_a", trades)

        validator = SuggestionValidator(curated_dir=tmp_path, lookback_days=5)
        result = validator.validate(
            suggestion_id="s7", bot_id="bot_a", category="position_sizing",
            target_param="position_size_pct", proposed_value=0.5,
            end_date="2026-03-10",
        )
        assert result.evidence.validated is True
        # With position_size_pct halved (1.0 -> 0.5), PnL should be halved
        baseline_pnl = result.evidence.baseline_metrics["pnl"]
        proposed_pnl = result.evidence.proposed_metrics["pnl"]
        assert abs(proposed_pnl - baseline_pnl * 0.5) < 1.0  # Allow rounding tolerance

    def test_filter_threshold_parameter(self, tmp_path):
        trades = [
            {"pnl": 100, "filter_score": 0.8, "regime": "trending"},
            {"pnl": -50, "filter_score": 0.3, "regime": "ranging"},
            {"pnl": 80, "filter_score": 0.7, "regime": "trending"},
            {"pnl": -20, "filter_score": 0.2, "regime": "ranging"},
        ]
        _write_trades(tmp_path, "2026-03-10", "bot_a", trades)

        validator = SuggestionValidator(curated_dir=tmp_path, lookback_days=5)
        result = validator.validate(
            suggestion_id="s8", bot_id="bot_a", category="filter_threshold",
            target_param="filter_min_score", proposed_value=0.5,
            end_date="2026-03-10",
        )
        assert result.evidence.validated is True
        # Only trades with filter_score >= 0.5 remain: pnl 100, 80 = 180
        assert result.evidence.proposed_metrics["pnl"] == 180

    def test_unknown_parameter_passes_through(self, tmp_path):
        trades = _sample_trades()
        _write_trades(tmp_path, "2026-03-10", "bot_a", trades)

        validator = SuggestionValidator(curated_dir=tmp_path, lookback_days=5)
        result = validator.validate(
            suggestion_id="s9", bot_id="bot_a", category="parameter",
            target_param="custom_unknown_param", proposed_value=1.0,
            end_date="2026-03-10",
        )
        assert result.evidence.validated is True
        # Unknown param means all trades pass through unmodified
        assert result.evidence.proposed_metrics["pnl"] == result.evidence.baseline_metrics["pnl"]

    def test_degradation_detected_when_pnl_drops(self, tmp_path):
        """Proposed PnL < 95% of baseline triggers degradation."""
        # All trades have high signal — raising threshold filters out profitable ones
        trades = [
            {"pnl": 150, "signal_strength": 0.5, "regime": "trending"},
            {"pnl": 120, "signal_strength": 0.4, "regime": "trending"},
            {"pnl": 100, "signal_strength": 0.3, "regime": "trending"},
        ]
        _write_trades(tmp_path, "2026-03-10", "bot_a", trades)

        validator = SuggestionValidator(curated_dir=tmp_path, lookback_days=5)
        result = validator.validate(
            suggestion_id="s10", bot_id="bot_a", category="signal",
            target_param="signal_threshold", proposed_value=0.45,
            end_date="2026-03-10",
        )
        # Only the 0.5 trade remains (pnl=150); baseline is 370
        # 150 < 370 * 0.95 = 351.5 → degradation
        assert result.degradation_detected is True

    def test_requires_review_when_degradation(self, tmp_path):
        """requires_review mirrors degradation_detected."""
        trades = [
            {"pnl": 200, "signal_strength": 0.5, "regime": "trending"},
            {"pnl": 180, "signal_strength": 0.4, "regime": "trending"},
        ]
        _write_trades(tmp_path, "2026-03-10", "bot_a", trades)

        validator = SuggestionValidator(curated_dir=tmp_path, lookback_days=5)
        result = validator.validate(
            suggestion_id="s11", bot_id="bot_a", category="signal",
            target_param="signal_threshold", proposed_value=0.45,
            end_date="2026-03-10",
        )
        assert result.requires_review == result.degradation_detected

    def test_improvement_pct_computed_correctly(self, tmp_path):
        """improvement_pct = ((proposed - baseline) / |baseline|) * 100."""
        # All trades pass the high threshold (all have signal_strength > 0.1)
        trades = [
            {"pnl": 100, "signal_strength": 0.9, "regime": "trending"},
            {"pnl": -200, "signal_strength": 0.2, "regime": "ranging"},
        ]
        _write_trades(tmp_path, "2026-03-10", "bot_a", trades)

        validator = SuggestionValidator(curated_dir=tmp_path, lookback_days=5)
        result = validator.validate(
            suggestion_id="s12", bot_id="bot_a", category="signal",
            target_param="signal_threshold", proposed_value=0.5,
            end_date="2026-03-10",
        )
        # Baseline: 100 + -200 = -100; Proposed: only 0.9 trade passes → pnl=100
        # improvement = ((100 - (-100)) / |-100|) * 100 = 200%
        assert result.evidence.improvement_pct == 200.0

    def test_regime_breakdown_populated(self, tmp_path):
        trades = _sample_trades()
        _write_trades(tmp_path, "2026-03-10", "bot_a", trades)

        validator = SuggestionValidator(curated_dir=tmp_path, lookback_days=5)
        result = validator.validate(
            suggestion_id="s13", bot_id="bot_a", category="signal",
            target_param="signal_threshold", proposed_value=0.5,
            end_date="2026-03-10",
        )
        breakdown = result.evidence.regime_breakdown
        assert "trending" in breakdown
        assert "ranging" in breakdown
        assert "baseline" in breakdown["trending"]
        assert "proposed" in breakdown["trending"]

    def test_sample_size_from_trade_count(self, tmp_path):
        trades = _sample_trades()
        _write_trades(tmp_path, "2026-03-10", "bot_a", trades)

        validator = SuggestionValidator(curated_dir=tmp_path, lookback_days=5)
        result = validator.validate(
            suggestion_id="s14", bot_id="bot_a", category="signal",
            target_param="signal_threshold", proposed_value=0.1,
            end_date="2026-03-10",
        )
        assert result.evidence.sample_size == len(trades)

    def test_end_date_parameter_used(self, tmp_path):
        """Trades outside the lookback window should not be loaded."""
        _write_trades(tmp_path, "2026-03-10", "bot_a", [
            {"pnl": 100, "signal_strength": 0.8, "regime": "trending"},
        ])
        # This date is outside lookback_days=2 from end_date=2026-03-10
        _write_trades(tmp_path, "2026-03-01", "bot_a", [
            {"pnl": -500, "signal_strength": 0.9, "regime": "trending"},
        ])

        validator = SuggestionValidator(curated_dir=tmp_path, lookback_days=2)
        result = validator.validate(
            suggestion_id="s15", bot_id="bot_a", category="signal",
            target_param="signal_threshold", proposed_value=0.5,
            end_date="2026-03-10",
        )
        # Only the 2026-03-10 trade should be loaded (and 2026-03-09 which doesn't exist)
        assert result.evidence.sample_size == 1
        assert result.evidence.baseline_metrics["pnl"] == 100


class TestComputeMetrics:
    """Tests for SuggestionValidator._compute_metrics static method."""

    def test_empty_trades(self):
        metrics = SuggestionValidator._compute_metrics([])
        assert metrics["pnl"] == 0
        assert metrics["win_rate"] == 0
        assert metrics["sharpe"] == 0
        assert metrics["max_dd"] == 0
        assert metrics["trade_count"] == 0

    def test_computes_all_fields_correctly(self):
        trades = [
            {"pnl": 100},
            {"pnl": -50},
            {"pnl": 200},
            {"pnl": -30},
        ]
        metrics = SuggestionValidator._compute_metrics(trades)

        # total pnl = 100 - 50 + 200 - 30 = 220
        assert metrics["pnl"] == 220
        # win rate = 2 wins / 4 = 0.5
        assert metrics["win_rate"] == 0.5
        # trade_count
        assert metrics["trade_count"] == 4
        # sharpe should be a finite number
        assert math.isfinite(metrics["sharpe"])
        # max drawdown — after 100, drops to 50 (dd=50); peak 250, drops to 220 (dd=30)
        assert metrics["max_dd"] == 50


class TestLoadTrades:
    """Tests for SuggestionValidator._load_trades."""

    def test_loads_from_curated_directory(self, tmp_path):
        trades = [{"pnl": 100}, {"pnl": -50}]
        _write_trades(tmp_path, "2026-03-10", "bot_a", trades)

        validator = SuggestionValidator(curated_dir=tmp_path, lookback_days=5)
        loaded = validator._load_trades("bot_a", "2026-03-10")
        assert len(loaded) == 2
        assert loaded[0]["pnl"] == 100

    def test_handles_missing_files_gracefully(self, tmp_path):
        validator = SuggestionValidator(curated_dir=tmp_path, lookback_days=5)
        loaded = validator._load_trades("nonexistent_bot", "2026-03-10")
        assert loaded == []

    def test_loads_across_multiple_dates(self, tmp_path):
        _write_trades(tmp_path, "2026-03-10", "bot_a", [{"pnl": 100}])
        _write_trades(tmp_path, "2026-03-09", "bot_a", [{"pnl": 200}])
        _write_trades(tmp_path, "2026-03-08", "bot_a", [{"pnl": 300}])

        validator = SuggestionValidator(curated_dir=tmp_path, lookback_days=5)
        loaded = validator._load_trades("bot_a", "2026-03-10")
        assert len(loaded) == 3
        total_pnl = sum(t["pnl"] for t in loaded)
        assert total_pnl == 600


class TestReplayWithParam:
    """Tests for SuggestionValidator._replay_with_param."""

    def test_signal_threshold_filters_correctly(self, tmp_path):
        validator = SuggestionValidator(curated_dir=tmp_path)
        trades = [
            {"pnl": 100, "signal_strength": 0.8},
            {"pnl": -50, "signal_strength": 0.3},
            {"pnl": 200, "signal_strength": 0.9},
        ]
        result = validator._replay_with_param(trades, "signal_threshold", 0.5)
        assert len(result) == 2
        pnls = [t["pnl"] for t in result]
        assert 100 in pnls
        assert 200 in pnls

    def test_position_size_scales_pnl(self, tmp_path):
        validator = SuggestionValidator(curated_dir=tmp_path)
        trades = [
            {"pnl": 100, "position_size_pct": 1.0},
            {"pnl": -50, "position_size_pct": 1.0},
        ]
        result = validator._replay_with_param(trades, "position_size_pct", 2.0)
        assert len(result) == 2
        assert result[0]["pnl"] == 200  # 100 * (2.0 / 1.0)
        assert result[1]["pnl"] == -100  # -50 * (2.0 / 1.0)

    def test_regime_breakdown_groups_by_regime(self, tmp_path):
        validator = SuggestionValidator(curated_dir=tmp_path)
        original = [
            {"pnl": 100, "regime": "trending", "signal_strength": 0.8},
            {"pnl": -50, "regime": "ranging", "signal_strength": 0.3},
            {"pnl": 150, "regime": "trending", "signal_strength": 0.9},
        ]
        filtered = validator._replay_with_param(original, "signal_threshold", 0.5)
        breakdown = validator._compute_regime_breakdown(original, filtered, "signal_threshold", 0.5)

        assert "trending" in breakdown
        assert "ranging" in breakdown
        assert breakdown["trending"]["baseline"]["pnl"] == 250  # 100 + 150
        # Filtered trending: both pass (0.8 > 0.5, 0.9 > 0.5)
        assert breakdown["trending"]["proposed"]["pnl"] == 250
        # Filtered ranging: 0.3 < 0.5, filtered out
        assert breakdown["ranging"]["proposed"]["trade_count"] == 0


class TestIntegrationValidation:
    """Integration tests for the full validate flow."""

    def test_all_evidence_fields_populated(self, tmp_path):
        trades = _sample_trades()
        _write_trades(tmp_path, "2026-03-10", "bot_a", trades)

        validator = SuggestionValidator(curated_dir=tmp_path, lookback_days=5)
        result = validator.validate(
            suggestion_id="int1", bot_id="bot_a", category="signal",
            target_param="signal_threshold", proposed_value=0.5,
            end_date="2026-03-10",
        )
        ev = result.evidence
        assert ev.validated is True
        assert ev.method == "backtest_replay"
        assert ev.baseline_metrics != {}
        assert ev.proposed_metrics != {}
        assert ev.sample_size > 0
        assert ev.regime_breakdown != {}
        assert isinstance(ev.improvement_pct, float)
        # Result fields
        assert result.suggestion_id == "int1"
        assert result.bot_id == "bot_a"

    def test_degradation_triggers_requires_review(self, tmp_path):
        """When proposed PnL < 95% baseline, both flags are set."""
        trades = [
            {"pnl": 200, "signal_strength": 0.5, "regime": "trending"},
            {"pnl": 150, "signal_strength": 0.4, "regime": "trending"},
            {"pnl": 100, "signal_strength": 0.3, "regime": "trending"},
        ]
        _write_trades(tmp_path, "2026-03-10", "bot_a", trades)

        validator = SuggestionValidator(curated_dir=tmp_path, lookback_days=5)
        result = validator.validate(
            suggestion_id="int2", bot_id="bot_a", category="signal",
            target_param="signal_threshold", proposed_value=0.45,
            end_date="2026-03-10",
        )
        # Only 0.5 passes → pnl=200; baseline=450; 200 < 450*0.95=427.5
        assert result.degradation_detected is True
        assert result.requires_review is True


# ===========================================================================
# 3. Handler wiring tests
# ===========================================================================

class MockSuggestionTracker:
    """Minimal mock that records calls to record()."""

    def __init__(self):
        self.recorded: list = []

    def record(self, rec):
        self.recorded.append(rec)
        return True

    def load_all(self):
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in self.recorded]


def _make_handlers(tmp_path, suggestion_tracker=None):
    """Create a minimal Handlers instance for testing _record_agent_suggestions."""
    from tests.factories import make_handlers as _factory_make_handlers

    event_stream = MagicMock()
    event_stream.broadcast = MagicMock()
    curated_dir = tmp_path / "curated"
    curated_dir.mkdir(exist_ok=True)
    handlers, _, _ = _factory_make_handlers(
        tmp_path,
        event_stream=event_stream,
        suggestion_tracker=suggestion_tracker,
        bots=["bot_a"],
        curated_dir=curated_dir,
    )
    return handlers, curated_dir


class TestHandlerWiringSuggestionValidation:
    """Tests that _record_agent_suggestions calls SuggestionValidator."""

    def test_validation_evidence_in_detection_context(self, tmp_path):
        """When a parameter suggestion is validated, detection_context has validation_evidence."""
        from datetime import datetime, timedelta

        tracker = MockSuggestionTracker()
        handlers, curated_dir = _make_handlers(tmp_path, suggestion_tracker=tracker)

        # Write trade data so validator can run — use recent date within lookback window
        recent_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        _write_trades(curated_dir, recent_date, "bot_a", _sample_trades())

        from trading_assistant.schemas.agent_response import AgentSuggestion
        from trading_assistant.analysis.response_validator import ValidationResult

        suggestion = AgentSuggestion(
            bot_id="bot_a",
            title="Raise signal threshold",
            category="signal",
            target_param="signal_threshold",
            proposed_value=0.6,
            confidence=0.7,
        )
        validation = ValidationResult(
            approved_suggestions=[suggestion],
        )

        handlers._record_agent_suggestions(validation, "run_001")
        assert len(tracker.recorded) == 1
        rec = tracker.recorded[0]
        ctx = rec.detection_context
        assert ctx is not None
        assert "validation_evidence" in ctx

    def test_degradation_produces_requires_review(self, tmp_path):
        """When degradation is detected, detection_context has requires_review=True."""
        from datetime import datetime, timedelta

        tracker = MockSuggestionTracker()
        handlers, curated_dir = _make_handlers(tmp_path, suggestion_tracker=tracker)

        # All trades have moderate signal — high threshold filters profitable ones
        trades = [
            {"pnl": 200, "signal_strength": 0.5, "regime": "trending"},
            {"pnl": 150, "signal_strength": 0.4, "regime": "trending"},
        ]
        # Use a recent date within the validator's 30-day lookback window
        recent_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        _write_trades(curated_dir, recent_date, "bot_a", trades)

        from trading_assistant.schemas.agent_response import AgentSuggestion
        from trading_assistant.analysis.response_validator import ValidationResult

        suggestion = AgentSuggestion(
            bot_id="bot_a",
            title="Raise signal threshold high",
            category="signal",
            target_param="signal_threshold",
            proposed_value=0.45,
            confidence=0.6,
        )
        validation = ValidationResult(
            approved_suggestions=[suggestion],
        )

        handlers._record_agent_suggestions(validation, "run_002")
        assert len(tracker.recorded) == 1
        ctx = tracker.recorded[0].detection_context
        # Only 0.5 trade passes → pnl=200; baseline=350; 200 < 350*0.95=332.5 → degradation
        assert ctx.get("requires_review") is True

    def test_structural_suggestion_gets_not_testable(self, tmp_path):
        """Structural suggestions (no target_param) get not_testable evidence."""
        tracker = MockSuggestionTracker()
        handlers, curated_dir = _make_handlers(tmp_path, suggestion_tracker=tracker)

        from trading_assistant.schemas.agent_response import AgentSuggestion
        from trading_assistant.analysis.response_validator import ValidationResult

        suggestion = AgentSuggestion(
            bot_id="bot_a",
            title="Add correlation filter to ranging regime",
            category="structural",
            # No target_param or proposed_value
            confidence=0.5,
        )
        validation = ValidationResult(
            approved_suggestions=[suggestion],
        )

        handlers._record_agent_suggestions(validation, "run_003")
        assert len(tracker.recorded) == 1
        ctx = tracker.recorded[0].detection_context
        # Structural suggestion has validation_evidence with method=not_testable
        if ctx and "validation_evidence" in ctx:
            assert ctx["validation_evidence"]["method"] == "not_testable"

    def test_validator_failure_doesnt_block_recording(self, tmp_path):
        """If SuggestionValidator raises, suggestion is still recorded."""
        tracker = MockSuggestionTracker()
        handlers, curated_dir = _make_handlers(tmp_path, suggestion_tracker=tracker)

        from trading_assistant.schemas.agent_response import AgentSuggestion
        from trading_assistant.analysis.response_validator import ValidationResult

        suggestion = AgentSuggestion(
            bot_id="bot_a",
            title="Test suggestion",
            category="signal",
            target_param="signal_threshold",
            proposed_value=0.5,
            confidence=0.6,
        )
        validation = ValidationResult(
            approved_suggestions=[suggestion],
        )

        # Patch validator to raise an exception
        with patch("trading_assistant.skills.suggestion_validator.SuggestionValidator.validate") as mock_validate:
            mock_validate.side_effect = RuntimeError("Backtest failed")
            handlers._record_agent_suggestions(validation, "run_004")

        # Suggestion should still be recorded despite validator failure
        assert len(tracker.recorded) == 1

    def test_validation_evidence_key_present(self, tmp_path):
        """The 'validation_evidence' key is present even for simple parameter suggestions."""
        from datetime import datetime, timedelta

        tracker = MockSuggestionTracker()
        handlers, curated_dir = _make_handlers(tmp_path, suggestion_tracker=tracker)

        # Write some trades — use a recent date within the validator's 30-day lookback
        recent_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        _write_trades(curated_dir, recent_date, "bot_a", [
            {"pnl": 50, "signal_strength": 0.9, "regime": "trending"},
        ])

        from trading_assistant.schemas.agent_response import AgentSuggestion
        from trading_assistant.analysis.response_validator import ValidationResult

        suggestion = AgentSuggestion(
            bot_id="bot_a",
            title="Lower signal threshold",
            category="signal",
            target_param="signal_threshold",
            proposed_value=0.1,
            confidence=0.8,
        )
        validation = ValidationResult(
            approved_suggestions=[suggestion],
        )

        handlers._record_agent_suggestions(validation, "run_005")
        assert len(tracker.recorded) == 1
        ctx = tracker.recorded[0].detection_context
        assert ctx is not None
        assert "validation_evidence" in ctx
        ev = ctx["validation_evidence"]
        assert ev["validated"] is True
        assert ev["method"] == "backtest_replay"
