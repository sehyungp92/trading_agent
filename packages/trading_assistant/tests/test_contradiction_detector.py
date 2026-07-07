"""Tests for contradiction detection — schemas, detector, and integration."""
from __future__ import annotations

import json
from pathlib import Path


from trading_assistant.schemas.contradiction import (
    ContradictionItem,
    ContradictionReport,
    ContradictionType,
)
from trading_assistant.skills.contradiction_detector import ContradictionDetector


# ---------------------------------------------------------------------------
# Schema round-trips
# ---------------------------------------------------------------------------

class TestContradictionSchemas:
    def test_contradiction_item_round_trip(self):
        item = ContradictionItem(
            type=ContradictionType.REGIME_DIRECTION_CONFLICT,
            bot_id="bot_a",
            description="test contradiction",
            day_a="2026-03-01",
            day_b="2026-03-02",
            severity="high",
            evidence={"regime": "trending", "pnl_a": 100, "pnl_b": -80},
        )
        data = item.model_dump(mode="json")
        restored = ContradictionItem(**data)
        assert restored.type == ContradictionType.REGIME_DIRECTION_CONFLICT
        assert restored.evidence["pnl_a"] == 100

    def test_contradiction_report_round_trip(self):
        report = ContradictionReport(
            date="2026-03-03",
            lookback_days=3,
            items=[
                ContradictionItem(
                    type=ContradictionType.FACTOR_QUALITY_DIVERGENCE,
                    bot_id="bot_a",
                    description="factor diverged",
                    day_a="2026-03-01",
                    day_b="2026-03-02",
                ),
            ],
            bots_analyzed=["bot_a"],
        )
        data = report.model_dump(mode="json")
        restored = ContradictionReport(**data)
        assert len(restored.items) == 1
        assert restored.bots_analyzed == ["bot_a"]

    def test_contradiction_type_enum_values(self):
        assert ContradictionType.REGIME_DIRECTION_CONFLICT.value == "regime_direction_conflict"
        assert ContradictionType.RISK_EXPOSURE_CONFLICT.value == "risk_exposure_conflict"


# ---------------------------------------------------------------------------
# Detector — insufficient data
# ---------------------------------------------------------------------------

class TestDetectorInsufficientData:
    def test_empty_curated_dir(self, tmp_path: Path):
        detector = ContradictionDetector(
            date="2026-03-03", bots=["bot_a"],
            curated_dir=tmp_path, lookback_days=3,
        )
        report = detector.detect()
        assert report.items == []
        assert report.bots_analyzed == ["bot_a"]

    def test_single_day_only(self, tmp_path: Path):
        """One day of data — need at least 2 to compare."""
        _write_curated_day(tmp_path, "2026-03-03", "bot_a", {
            "summary": {"market_regime": "trending", "net_pnl": 100},
            "regime_analysis": {"regimes": [{"regime": "trending", "total_pnl": 100}]},
        })
        detector = ContradictionDetector(
            date="2026-03-03", bots=["bot_a"],
            curated_dir=tmp_path, lookback_days=3,
        )
        report = detector.detect()
        assert report.items == []


# ---------------------------------------------------------------------------
# Regime direction conflict
# ---------------------------------------------------------------------------

class TestRegimeDirectionConflict:
    def test_regime_conflict_detected(self, tmp_path: Path):
        """Same regime, profitable day A, unprofitable day B."""
        _write_curated_day(tmp_path, "2026-03-02", "bot_a", {
            "summary": {"market_regime": "trending"},
            "regime_analysis": {"regimes": [{"regime": "trending", "total_pnl": 200}]},
        })
        _write_curated_day(tmp_path, "2026-03-03", "bot_a", {
            "summary": {"market_regime": "trending"},
            "regime_analysis": {"regimes": [{"regime": "trending", "total_pnl": -150}]},
        })
        detector = ContradictionDetector(
            date="2026-03-03", bots=["bot_a"],
            curated_dir=tmp_path, lookback_days=3,
        )
        report = detector.detect()
        regime_items = [i for i in report.items if i.type == ContradictionType.REGIME_DIRECTION_CONFLICT]
        assert len(regime_items) == 1
        assert "trending" in regime_items[0].description

    def test_no_conflict_when_regime_changed(self, tmp_path: Path):
        """Different regime between days — not a contradiction."""
        _write_curated_day(tmp_path, "2026-03-02", "bot_a", {
            "summary": {"market_regime": "trending"},
            "regime_analysis": {"regimes": [{"regime": "trending", "total_pnl": 200}]},
        })
        _write_curated_day(tmp_path, "2026-03-03", "bot_a", {
            "summary": {"market_regime": "ranging"},
            "regime_analysis": {"regimes": [{"regime": "trending", "total_pnl": -150}]},
        })
        detector = ContradictionDetector(
            date="2026-03-03", bots=["bot_a"],
            curated_dir=tmp_path, lookback_days=3,
        )
        report = detector.detect()
        regime_items = [i for i in report.items if i.type == ContradictionType.REGIME_DIRECTION_CONFLICT]
        assert len(regime_items) == 0


# ---------------------------------------------------------------------------
# Factor quality divergence
# ---------------------------------------------------------------------------

class TestFactorQualityDivergence:
    def test_factor_divergence_detected(self, tmp_path: Path):
        """Win rate drops >20pp while contribution stays high."""
        _write_curated_day(tmp_path, "2026-03-02", "bot_a", {
            "summary": {},
            "factor_attribution": {
                "factors": [
                    {"factor_name": "rsi", "win_rate": 0.70, "avg_contribution": 0.5, "trade_count": 20},
                ],
            },
        })
        _write_curated_day(tmp_path, "2026-03-03", "bot_a", {
            "summary": {},
            "factor_attribution": {
                "factors": [
                    {"factor_name": "rsi", "win_rate": 0.40, "avg_contribution": 0.45, "trade_count": 20},
                ],
            },
        })
        detector = ContradictionDetector(
            date="2026-03-03", bots=["bot_a"],
            curated_dir=tmp_path, lookback_days=3,
        )
        report = detector.detect()
        factor_items = [i for i in report.items if i.type == ContradictionType.FACTOR_QUALITY_DIVERGENCE]
        assert len(factor_items) == 1
        assert "rsi" in factor_items[0].description

    def test_no_divergence_when_contribution_also_drops(self, tmp_path: Path):
        """Win rate and contribution both drop — not a divergence."""
        _write_curated_day(tmp_path, "2026-03-02", "bot_a", {
            "summary": {},
            "factor_attribution": {
                "factors": [
                    {"factor_name": "rsi", "win_rate": 0.70, "avg_contribution": 0.5, "trade_count": 20},
                ],
            },
        })
        _write_curated_day(tmp_path, "2026-03-03", "bot_a", {
            "summary": {},
            "factor_attribution": {
                "factors": [
                    {"factor_name": "rsi", "win_rate": 0.40, "avg_contribution": 0.1, "trade_count": 20},
                ],
            },
        })
        detector = ContradictionDetector(
            date="2026-03-03", bots=["bot_a"],
            curated_dir=tmp_path, lookback_days=3,
        )
        report = detector.detect()
        factor_items = [i for i in report.items if i.type == ContradictionType.FACTOR_QUALITY_DIVERGENCE]
        assert len(factor_items) == 0


# ---------------------------------------------------------------------------
# Exit process conflict
# ---------------------------------------------------------------------------

class TestExitProcessConflict:
    def test_exit_process_conflict_detected(self, tmp_path: Path):
        """Exit efficiency improving but failures increasing."""
        _write_curated_day(tmp_path, "2026-03-02", "bot_a", {
            "summary": {},
            "exit_efficiency": {"avg_efficiency": 0.40},
            "process_failures": [{"id": 1}, {"id": 2}],
        })
        _write_curated_day(tmp_path, "2026-03-03", "bot_a", {
            "summary": {},
            "exit_efficiency": {"avg_efficiency": 0.55},
            "process_failures": [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}],
        })
        detector = ContradictionDetector(
            date="2026-03-03", bots=["bot_a"],
            curated_dir=tmp_path, lookback_days=3,
        )
        report = detector.detect()
        exit_items = [i for i in report.items if i.type == ContradictionType.EXIT_PROCESS_CONFLICT]
        assert len(exit_items) == 1

    def test_no_conflict_when_both_improving(self, tmp_path: Path):
        _write_curated_day(tmp_path, "2026-03-02", "bot_a", {
            "summary": {},
            "exit_efficiency": {"avg_efficiency": 0.40},
            "process_failures": [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}],
        })
        _write_curated_day(tmp_path, "2026-03-03", "bot_a", {
            "summary": {},
            "exit_efficiency": {"avg_efficiency": 0.55},
            "process_failures": [{"id": 1}],
        })
        detector = ContradictionDetector(
            date="2026-03-03", bots=["bot_a"],
            curated_dir=tmp_path, lookback_days=3,
        )
        report = detector.detect()
        exit_items = [i for i in report.items if i.type == ContradictionType.EXIT_PROCESS_CONFLICT]
        assert len(exit_items) == 0


# ---------------------------------------------------------------------------
# Risk exposure conflict
# ---------------------------------------------------------------------------

class TestRiskExposureConflict:
    def test_risk_exposure_conflict_detected(self, tmp_path: Path):
        """Exposure rising while drawdown deepening."""
        _write_curated_day(tmp_path, "2026-03-02", "bot_a", {
            "summary": {"exposure_pct": 30.0, "max_drawdown_pct": -3.0},
        })
        _write_curated_day(tmp_path, "2026-03-03", "bot_a", {
            "summary": {"exposure_pct": 42.0, "max_drawdown_pct": -6.0},
        })
        detector = ContradictionDetector(
            date="2026-03-03", bots=["bot_a"],
            curated_dir=tmp_path, lookback_days=3,
        )
        report = detector.detect()
        risk_items = [i for i in report.items if i.type == ContradictionType.RISK_EXPOSURE_CONFLICT]
        assert len(risk_items) == 1
        assert risk_items[0].severity == "high"

    def test_no_conflict_small_changes(self, tmp_path: Path):
        """Small changes should not trigger."""
        _write_curated_day(tmp_path, "2026-03-02", "bot_a", {
            "summary": {"exposure_pct": 30.0, "max_drawdown_pct": -3.0},
        })
        _write_curated_day(tmp_path, "2026-03-03", "bot_a", {
            "summary": {"exposure_pct": 32.0, "max_drawdown_pct": -3.5},
        })
        detector = ContradictionDetector(
            date="2026-03-03", bots=["bot_a"],
            curated_dir=tmp_path, lookback_days=3,
        )
        report = detector.detect()
        risk_items = [i for i in report.items if i.type == ContradictionType.RISK_EXPOSURE_CONFLICT]
        assert len(risk_items) == 0


# ---------------------------------------------------------------------------
# Multi-bot detection
# ---------------------------------------------------------------------------

class TestMultiBot:
    def test_multi_bot_detection(self, tmp_path: Path):
        """Contradictions detected independently per bot."""
        for bot in ["bot_a", "bot_b"]:
            _write_curated_day(tmp_path, "2026-03-02", bot, {
                "summary": {"market_regime": "trending"},
                "regime_analysis": {"regimes": [{"regime": "trending", "total_pnl": 200}]},
            })
            _write_curated_day(tmp_path, "2026-03-03", bot, {
                "summary": {"market_regime": "trending"},
                "regime_analysis": {"regimes": [{"regime": "trending", "total_pnl": -150}]},
            })
        detector = ContradictionDetector(
            date="2026-03-03", bots=["bot_a", "bot_b"],
            curated_dir=tmp_path, lookback_days=3,
        )
        report = detector.detect()
        assert len(report.items) == 2
        bot_ids = {item.bot_id for item in report.items}
        assert bot_ids == {"bot_a", "bot_b"}


# ---------------------------------------------------------------------------
# Assembler integration
# ---------------------------------------------------------------------------

class TestAssemblerIntegration:
    def test_contradictions_in_prompt_package(self, tmp_path: Path):
        """ContextBuilder.load_contradictions returns items for assembler injection."""
        from trading_assistant.analysis.context_builder import ContextBuilder

        _write_curated_day(tmp_path, "2026-03-02", "bot_a", {
            "summary": {"exposure_pct": 30.0, "max_drawdown_pct": -3.0},
        })
        _write_curated_day(tmp_path, "2026-03-03", "bot_a", {
            "summary": {"exposure_pct": 42.0, "max_drawdown_pct": -6.0},
        })
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        ctx = ContextBuilder(memory_dir)
        contradictions = ctx.load_contradictions("2026-03-03", ["bot_a"], tmp_path)
        assert len(contradictions) >= 1
        assert contradictions[0]["type"] == "risk_exposure_conflict"

    def test_no_contradictions_returns_empty(self, tmp_path: Path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        ctx = ContextBuilder(memory_dir)
        contradictions = ctx.load_contradictions("2026-03-03", ["bot_a"], tmp_path)
        assert contradictions == []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_curated_day(
    base_dir: Path, date: str, bot: str, files: dict[str, dict | list],
) -> None:
    """Write curated JSON files for a bot on a date."""
    bot_dir = base_dir / date / bot
    bot_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in files.items():
        path = bot_dir / f"{filename}.json" if not filename.endswith(".json") else bot_dir / filename
        path.write_text(json.dumps(data, indent=2))
