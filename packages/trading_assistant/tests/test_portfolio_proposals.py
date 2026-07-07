# tests/test_portfolio_proposals.py
"""Tests for the portfolio-level improvement capability.

Covers: schemas, response parser, response validator guardrails,
suggestion tracker extensions, ground truth portfolio snapshot,
context builder injection, and strategy engine portfolio detectors.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestPortfolioProposalSchema:
    def test_create_allocation_rebalance(self):
        from trading_assistant.schemas.portfolio_proposal import PortfolioProposal, PortfolioProposalType

        p = PortfolioProposal(
            proposal_type=PortfolioProposalType.ALLOCATION_REBALANCE,
            current_config={"swing": 0.33, "stock": 0.33, "momentum": 0.34},
            proposed_config={"swing": 0.40, "stock": 0.30, "momentum": 0.30},
            evidence_summary="Swing Calmar 2.1 vs portfolio 1.4",
            expected_portfolio_calmar_delta=0.15,
            confidence=0.6,
            observation_window_days=60,
        )
        assert p.proposal_type == PortfolioProposalType.ALLOCATION_REBALANCE
        assert p.confidence == 0.6

    def test_all_proposal_types_exist(self):
        from trading_assistant.schemas.portfolio_proposal import PortfolioProposalType

        assert hasattr(PortfolioProposalType, "ALLOCATION_REBALANCE")
        assert hasattr(PortfolioProposalType, "RISK_CAP_CHANGE")
        assert hasattr(PortfolioProposalType, "COORDINATION_CHANGE")
        assert hasattr(PortfolioProposalType, "DRAWDOWN_TIER_CHANGE")

    def test_confidence_clamped(self):
        from trading_assistant.schemas.portfolio_proposal import PortfolioProposal, PortfolioProposalType

        with pytest.raises(ValidationError):
            PortfolioProposal(
                proposal_type=PortfolioProposalType.ALLOCATION_REBALANCE,
                confidence=1.5,  # out of range
            )


class TestPortfolioMetricsSchema:
    def test_family_daily_snapshot(self):
        from trading_assistant.schemas.portfolio_metrics import FamilyDailySnapshot

        snap = FamilyDailySnapshot(
            family="swing", date="2026-03-01",
            strategy_ids=["ATRSS", "AKC_HELIX"],
            total_net_pnl=150.0, trade_count=10, win_count=6,
        )
        assert snap.family == "swing"
        assert snap.total_net_pnl == 150.0

    def test_portfolio_rolling_metrics(self):
        from trading_assistant.schemas.portfolio_metrics import PortfolioRollingMetrics

        m = PortfolioRollingMetrics(
            date="2026-03-01",
            sharpe_7d=1.2, sharpe_30d=0.9,
            calmar_30d=1.5,
        )
        assert m.sharpe_7d == 1.2


class TestSuggestionTierPortfolio:
    def test_portfolio_tier_exists(self):
        from trading_assistant.schemas.strategy_suggestions import SuggestionTier

        assert SuggestionTier.PORTFOLIO.value == "portfolio"

    def test_category_to_tier_portfolio_entries(self):
        from trading_assistant.schemas.agent_response import CATEGORY_TO_TIER

        assert CATEGORY_TO_TIER["portfolio_allocation"] == "portfolio"
        assert CATEGORY_TO_TIER["portfolio_risk_cap"] == "portfolio"
        assert CATEGORY_TO_TIER["portfolio_coordination"] == "portfolio"
        assert CATEGORY_TO_TIER["portfolio_drawdown_tier"] == "portfolio"


class TestParsedAnalysisPortfolioField:
    def test_portfolio_proposals_default_empty(self):
        from trading_assistant.schemas.agent_response import ParsedAnalysis

        pa = ParsedAnalysis(raw_report="test")
        assert pa.portfolio_proposals == []


# ---------------------------------------------------------------------------
# Response parser tests
# ---------------------------------------------------------------------------

class TestResponseParserPortfolio:
    def test_parse_portfolio_proposals(self):
        from trading_assistant.analysis.response_parser import parse_response

        response = """Some analysis text.
<!-- STRUCTURED_OUTPUT
{
  "predictions": [],
  "suggestions": [],
  "structural_proposals": [],
  "portfolio_proposals": [
    {
      "proposal_type": "allocation_rebalance",
      "current_config": {"swing": 0.33},
      "proposed_config": {"swing": 0.40},
      "evidence_summary": "Swing outperforms",
      "expected_portfolio_calmar_delta": 0.1,
      "confidence": 0.6,
      "observation_window_days": 60
    }
  ]
}
-->"""
        parsed = parse_response(response)
        assert parsed.parse_success
        assert len(parsed.portfolio_proposals) == 1
        assert parsed.portfolio_proposals[0].proposal_type.value == "allocation_rebalance"

    def test_parse_empty_portfolio_proposals(self):
        from trading_assistant.analysis.response_parser import parse_response

        response = """Analysis.
<!-- STRUCTURED_OUTPUT
{"predictions": [], "suggestions": [], "structural_proposals": []}
-->"""
        parsed = parse_response(response)
        assert parsed.parse_success
        assert parsed.portfolio_proposals == []


# ---------------------------------------------------------------------------
# Response validator guardrail tests
# ---------------------------------------------------------------------------

class TestPortfolioValidatorGuardrails:
    def _make_proposal(self, **kwargs):
        from trading_assistant.schemas.portfolio_proposal import PortfolioProposal, PortfolioProposalType

        defaults = {
            "proposal_type": PortfolioProposalType.ALLOCATION_REBALANCE,
            "current_config": {"swing": 0.33, "stock": 0.33, "momentum": 0.34},
            "proposed_config": {"swing": 0.40, "stock": 0.30, "momentum": 0.30},
            "evidence_summary": "Test evidence",
            "confidence": 0.6,
            "observation_window_days": 60,
        }
        defaults.update(kwargs)
        return PortfolioProposal(**defaults)

    def _make_validator(self, prediction_accuracy=None):
        from trading_assistant.analysis.response_validator import ResponseValidator

        return ResponseValidator(prediction_accuracy=prediction_accuracy or {})

    def test_approve_valid_allocation(self):
        v = self._make_validator()
        proposals = [self._make_proposal()]
        approved, blocked = v._validate_portfolio_proposals(proposals)
        assert len(approved) == 1
        assert len(blocked) == 0

    def test_block_excessive_allocation_change(self):

        v = self._make_validator()
        proposals = [self._make_proposal(
            proposed_config={"swing": 0.60, "stock": 0.20, "momentum": 0.20},  # 27% change
        )]
        approved, blocked = v._validate_portfolio_proposals(proposals)
        assert len(blocked) == 1
        assert "exceeds" in blocked[0].reason

    def test_block_below_floor(self):
        v = self._make_validator()
        proposals = [self._make_proposal(
            proposed_config={"swing": 0.36, "stock": 0.03, "momentum": 0.34},  # stock < 5%
            current_config={"swing": 0.33, "stock": 0.06, "momentum": 0.34},
        )]
        approved, blocked = v._validate_portfolio_proposals(proposals)
        assert len(blocked) == 1
        assert "floor" in blocked[0].reason

    def test_block_insufficient_evidence_allocation(self):
        v = self._make_validator()
        proposals = [self._make_proposal(observation_window_days=30)]  # needs 60
        approved, blocked = v._validate_portfolio_proposals(proposals)
        assert len(blocked) == 1
        assert "60" in blocked[0].reason

    def test_block_risk_cap_excessive_change(self):
        from trading_assistant.schemas.portfolio_proposal import PortfolioProposalType

        v = self._make_validator()
        proposals = [self._make_proposal(
            proposal_type=PortfolioProposalType.RISK_CAP_CHANGE,
            current_config={"heat_cap_R": 10.0},
            proposed_config={"heat_cap_R": 15.0},  # 50% change
            observation_window_days=90,
        )]
        approved, blocked = v._validate_portfolio_proposals(proposals)
        assert len(blocked) == 1
        assert "20%" in blocked[0].reason

    def test_block_risk_cap_insufficient_evidence(self):
        from trading_assistant.schemas.portfolio_proposal import PortfolioProposalType

        v = self._make_validator()
        proposals = [self._make_proposal(
            proposal_type=PortfolioProposalType.RISK_CAP_CHANGE,
            current_config={"heat_cap_R": 10.0},
            proposed_config={"heat_cap_R": 10.5},
            observation_window_days=60,  # needs 90
        )]
        approved, blocked = v._validate_portfolio_proposals(proposals)
        assert len(blocked) == 1
        assert "90" in blocked[0].reason

    def test_block_drawdown_tier_removal(self):
        from trading_assistant.schemas.portfolio_proposal import PortfolioProposalType

        v = self._make_validator()
        proposals = [self._make_proposal(
            proposal_type=PortfolioProposalType.DRAWDOWN_TIER_CHANGE,
            current_config={"drawdown_tiers": [[0.05, 0.5], [0.10, 0.25]]},
            proposed_config={"drawdown_tiers": [[0.05, 0.5]]},  # removed a tier
            observation_window_days=90,
        )]
        approved, blocked = v._validate_portfolio_proposals(proposals)
        assert len(blocked) == 1
        assert "removal" in blocked[0].reason.lower()

    def test_block_drawdown_tier_loosening(self):
        from trading_assistant.schemas.portfolio_proposal import PortfolioProposalType

        v = self._make_validator()
        proposals = [self._make_proposal(
            proposal_type=PortfolioProposalType.DRAWDOWN_TIER_CHANGE,
            current_config={"drawdown_tiers": [[0.05, 0.5], [0.10, 0.25]]},
            proposed_config={"drawdown_tiers": [[0.08, 0.5], [0.10, 0.25]]},  # loosened first tier
            observation_window_days=90,
        )]
        approved, blocked = v._validate_portfolio_proposals(proposals)
        assert len(blocked) == 1
        assert "loosened" in blocked[0].reason.lower()

    def test_approve_drawdown_tier_tightening(self):
        from trading_assistant.schemas.portfolio_proposal import PortfolioProposalType

        v = self._make_validator()
        proposals = [self._make_proposal(
            proposal_type=PortfolioProposalType.DRAWDOWN_TIER_CHANGE,
            current_config={"drawdown_tiers": [[0.05, 0.5], [0.10, 0.25]]},
            proposed_config={"drawdown_tiers": [[0.04, 0.4], [0.08, 0.20]]},  # tightened
            observation_window_days=90,
        )]
        approved, blocked = v._validate_portfolio_proposals(proposals)
        assert len(approved) == 1

    def test_block_all_when_low_prediction_accuracy(self):
        v = self._make_validator(prediction_accuracy={"overall_accuracy": 0.35})
        proposals = [self._make_proposal()]
        approved, blocked = v._validate_portfolio_proposals(proposals)
        assert len(blocked) == 1
        assert "prediction accuracy" in blocked[0].reason.lower()

    def test_allow_when_good_prediction_accuracy(self):
        v = self._make_validator(prediction_accuracy={"overall_accuracy": 0.65})
        proposals = [self._make_proposal()]
        approved, blocked = v._validate_portfolio_proposals(proposals)
        assert len(approved) == 1

    def test_validate_method_includes_portfolio(self):
        """Full validate() flow includes portfolio proposals in result."""
        from trading_assistant.analysis.response_validator import ResponseValidator
        from trading_assistant.schemas.agent_response import ParsedAnalysis

        v = ResponseValidator()
        parsed = ParsedAnalysis(raw_report="test", parse_success=True)
        result = v.validate(parsed)
        assert hasattr(result, "approved_portfolio_proposals")
        assert hasattr(result, "blocked_portfolio_proposals")
        assert result.approved_portfolio_proposals == []
        assert result.blocked_portfolio_proposals == []


# ---------------------------------------------------------------------------
# Suggestion tracker extensions
# ---------------------------------------------------------------------------

class TestSuggestionTrackerPortfolio:
    def test_deployed_portfolio_count_empty(self, tmp_path: Path):
        from trading_assistant.skills.suggestion_tracker import SuggestionTracker

        tracker = SuggestionTracker(tmp_path)
        assert tracker.get_deployed_portfolio_count() == 0

    def test_deployed_portfolio_count_with_deployed(self, tmp_path: Path):
        from trading_assistant.skills.suggestion_tracker import SuggestionTracker
        from trading_assistant.schemas.suggestion_tracking import SuggestionRecord

        tracker = SuggestionTracker(tmp_path)
        record = SuggestionRecord(
            suggestion_id="pf-001", bot_id="PORTFOLIO",
            title="Test", tier="portfolio", category="portfolio_allocation",
            source_report_id="weekly-001",
        )
        tracker.record(record)
        tracker.accept("pf-001")
        tracker.mark_deployed("pf-001")
        assert tracker.get_deployed_portfolio_count() == 1

    def test_last_portfolio_proposal_date_none(self, tmp_path: Path):
        from trading_assistant.skills.suggestion_tracker import SuggestionTracker

        tracker = SuggestionTracker(tmp_path)
        assert tracker.get_last_portfolio_proposal_date() is None

    def test_last_portfolio_proposal_date(self, tmp_path: Path):
        from trading_assistant.skills.suggestion_tracker import SuggestionTracker
        from trading_assistant.schemas.suggestion_tracking import SuggestionRecord

        tracker = SuggestionTracker(tmp_path)
        record = SuggestionRecord(
            suggestion_id="pf-002", bot_id="PORTFOLIO",
            title="Test", tier="portfolio", category="portfolio_allocation",
            source_report_id="weekly-001",
        )
        tracker.record(record)
        date = tracker.get_last_portfolio_proposal_date()
        assert date is not None

    def test_last_portfolio_proposal_date_filtered(self, tmp_path: Path):
        from trading_assistant.skills.suggestion_tracker import SuggestionTracker
        from trading_assistant.schemas.suggestion_tracking import SuggestionRecord

        tracker = SuggestionTracker(tmp_path)
        tracker.record(SuggestionRecord(
            suggestion_id="pf-003", bot_id="PORTFOLIO",
            title="Alloc", tier="portfolio", category="portfolio_allocation",
            source_report_id="weekly-001",
        ))
        tracker.record(SuggestionRecord(
            suggestion_id="pf-004", bot_id="PORTFOLIO",
            title="Risk", tier="portfolio", category="portfolio_risk_cap",
            source_report_id="weekly-001",
        ))
        assert tracker.get_last_portfolio_proposal_date("portfolio_allocation") is not None
        assert tracker.get_last_portfolio_proposal_date("portfolio_drawdown_tier") is None


# ---------------------------------------------------------------------------
# Ground truth portfolio snapshot
# ---------------------------------------------------------------------------

class TestGroundTruthPortfolioSnapshot:
    def _setup_curated(self, tmp_path: Path, bots: list[str], days: int = 30):
        end = datetime(2026, 3, 15)
        for bot in bots:
            for d in range(days):
                date_str = (end - timedelta(days=d)).strftime("%Y-%m-%d")
                bot_dir = tmp_path / date_str / bot
                bot_dir.mkdir(parents=True, exist_ok=True)
                summary = {
                    "date": date_str,
                    "net_pnl": 10.0,
                    "total_trades": 5,
                    "winning_trades": 3,
                    "avg_win": 5.0,
                    "avg_loss": -3.0,
                    "win_count": 3,
                    "loss_count": 2,
                    "max_drawdown_pct": 0.02,
                    "avg_process_quality": 70.0,
                }
                (bot_dir / "summary.json").write_text(json.dumps(summary))

    def test_portfolio_snapshot_equal_weights(self, tmp_path: Path):
        from trading_assistant.skills.ground_truth_computer import GroundTruthComputer

        bots = ["bot1", "bot2"]
        self._setup_curated(tmp_path, bots)
        computer = GroundTruthComputer(tmp_path)
        result = computer.compute_portfolio_snapshot(bots, "2026-03-15")
        assert "portfolio_composite" in result
        assert 0 <= result["portfolio_composite"] <= 1
        assert len(result["per_bot"]) == 2

    def test_portfolio_snapshot_custom_weights(self, tmp_path: Path):
        from trading_assistant.skills.ground_truth_computer import GroundTruthComputer

        bots = ["bot1", "bot2"]
        self._setup_curated(tmp_path, bots)
        computer = GroundTruthComputer(tmp_path)
        result = computer.compute_portfolio_snapshot(
            bots, "2026-03-15", allocations={"bot1": 0.7, "bot2": 0.3},
        )
        assert result["per_bot"]["bot1"]["allocation"] == 0.7

    def test_portfolio_snapshot_empty_bots(self, tmp_path: Path):
        from trading_assistant.skills.ground_truth_computer import GroundTruthComputer

        computer = GroundTruthComputer(tmp_path)
        result = computer.compute_portfolio_snapshot([], "2026-03-15")
        assert result["portfolio_composite"] == 0.5


# ---------------------------------------------------------------------------
# Context builder injection
# ---------------------------------------------------------------------------

class TestContextBuilderPortfolio:
    def test_load_portfolio_outcomes_empty(self, tmp_path: Path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        ctx = ContextBuilder(memory_dir)
        assert ctx.load_portfolio_outcomes() == []

    def test_load_portfolio_outcomes_with_data(self, tmp_path: Path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        memory_dir = tmp_path / "memory"
        findings_dir = memory_dir / "findings"
        findings_dir.mkdir(parents=True)
        (findings_dir / "portfolio_outcomes.jsonl").write_text(
            json.dumps({
                "suggestion_id": "pf-001",
                "verdict": "positive",
                "composite_delta": 0.05,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }) + "\n"
        )
        ctx = ContextBuilder(memory_dir)
        outcomes = ctx.load_portfolio_outcomes()
        assert len(outcomes) == 1
        assert outcomes[0]["verdict"] == "positive"

    def test_load_portfolio_metrics_empty(self, tmp_path: Path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        ctx = ContextBuilder(memory_dir, curated_dir=tmp_path / "curated")
        assert ctx.load_portfolio_metrics() == {}

    def test_load_portfolio_metrics_with_data(self, tmp_path: Path):
        from trading_assistant.analysis.context_builder import ContextBuilder

        curated_dir = tmp_path / "curated"
        portfolio_dir = curated_dir / "2026-03-15" / "portfolio"
        portfolio_dir.mkdir(parents=True)
        metrics = {"sharpe_7d": 1.2, "calmar_30d": 1.5}
        (portfolio_dir / "portfolio_rolling_metrics.json").write_text(json.dumps(metrics))

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        ctx = ContextBuilder(memory_dir, curated_dir=curated_dir)
        loaded = ctx.load_portfolio_metrics()
        assert loaded["sharpe_7d"] == 1.2

    def test_portfolio_in_context_priority(self):
        from trading_assistant.analysis.context_builder import ContextBuilder

        assert "portfolio_outcomes" in ContextBuilder._CONTEXT_PRIORITY
        assert "portfolio_rolling_metrics" in ContextBuilder._CONTEXT_PRIORITY


# ---------------------------------------------------------------------------
# Strategy engine portfolio detectors
# ---------------------------------------------------------------------------

class TestStrategyEnginePortfolioDetectors:
    def _make_engine(self):
        from trading_assistant.analysis.strategy_engine import StrategyEngine

        return StrategyEngine(week_start="2026-03-08", week_end="2026-03-14")

    def test_detect_family_imbalance_no_data(self):
        engine = self._make_engine()
        result = engine.detect_family_imbalance({}, {})
        assert result == []

    def test_detect_family_imbalance_insufficient_days(self):
        engine = self._make_engine()
        # Only 5 days of data, needs 30
        summaries = {"swing": {"total_net_pnl": -10, "days": 5}}
        allocations = {"swing": 0.5}
        result = engine.detect_family_imbalance(summaries, allocations, min_days=30)
        assert result == []

    def test_detect_correlation_concentration_no_alert(self):
        engine = self._make_engine()
        # Low correlation, no issue
        matrix = {"bot1_bot2": 0.3}
        allocations = {"bot1": 0.5, "bot2": 0.5}
        result = engine.detect_correlation_concentration(matrix, allocations)
        assert result == []

    def test_detect_correlation_concentration_alert(self):
        engine = self._make_engine()
        matrix = {"bot1_bot2": 0.85}
        allocations = {"bot1": 0.3, "bot2": 0.3}
        result = engine.detect_correlation_concentration(matrix, allocations)
        assert len(result) >= 1
        assert result[0].bot_id == "PORTFOLIO"

    def test_detect_heat_cap_underutilized(self):
        engine = self._make_engine()
        daily_heat = [2.0] * 30  # 20% of 10.0 cap
        result = engine.detect_heat_cap_utilization(daily_heat, 10.0)
        assert len(result) >= 1

    def test_detect_heat_cap_normal(self):
        engine = self._make_engine()
        daily_heat = [6.0] * 30  # 60% of 10.0 cap — normal
        result = engine.detect_heat_cap_utilization(daily_heat, 10.0)
        assert result == []

    def test_detect_drawdown_tier_miscalibration_insufficient_data(self):
        engine = self._make_engine()
        result = engine.detect_drawdown_tier_miscalibration(
            [0.01, 0.02], [[0.05, 0.5]], min_days=90,
        )
        assert result == []


# ---------------------------------------------------------------------------
# Weekly prompt assembler portfolio output spec
# ---------------------------------------------------------------------------

class TestWeeklyPromptPortfolioOutput:
    def test_portfolio_proposals_in_structured_output(self):
        from trading_assistant.analysis.weekly_prompt_assembler import _WEEKLY_INSTRUCTIONS

        assert "portfolio_proposals" in _WEEKLY_INSTRUCTIONS
        assert "allocation_rebalance" in _WEEKLY_INSTRUCTIONS
        assert "risk_cap_change" in _WEEKLY_INSTRUCTIONS

    def test_portfolio_improvement_section(self):
        from trading_assistant.analysis.weekly_prompt_assembler import _WEEKLY_INSTRUCTIONS

        assert "PORTFOLIO IMPROVEMENT ASSESSMENT" in _WEEKLY_INSTRUCTIONS
        assert "family_snapshots" in _WEEKLY_INSTRUCTIONS
        assert "drawdown_correlation" in _WEEKLY_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Portfolio what-if
# ---------------------------------------------------------------------------

class TestPortfolioWhatIf:
    def test_evaluate_returns_metrics(self):
        from trading_assistant.skills.portfolio_what_if import PortfolioWhatIf

        family_daily_pnl = {
            "swing": [10.0] * 60,
            "stock": [-5.0] * 60,
        }
        current_weights = {"swing": 0.5, "stock": 0.5}
        wif = PortfolioWhatIf(family_daily_pnl=family_daily_pnl, current_weights=current_weights)
        result = wif.evaluate({"swing": 0.6, "stock": 0.4})
        assert "sharpe_new" in result or "calmar_delta" in result or "error" in result


# ---------------------------------------------------------------------------
# Trade-level what-if tests
# ---------------------------------------------------------------------------

def _make_trade(
    pnl: float,
    exit_time: datetime | None = None,
    entry_time: datetime | None = None,
    market_regime: str = "",
    bot_id: str = "bot1",
):
    """Helper to build a minimal TradeEvent for testing."""
    from trading_assistant.schemas.events import TradeEvent

    now = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    return TradeEvent(
        trade_id=f"t-{id(pnl)}-{pnl}",
        bot_id=bot_id,
        pair="BTC/USDT",
        side="LONG",
        entry_time=entry_time or now - timedelta(hours=1),
        exit_time=exit_time or now,
        entry_price=100.0,
        exit_price=100.0 + pnl,
        position_size=1.0,
        pnl=pnl,
        pnl_pct=pnl,
        market_regime=market_regime,
    )


class TestPortfolioWhatIfTradeLevelEnriched:
    """Tests for trade-level enriched what-if analysis."""

    def test_evaluate_with_trades_returns_enriched_metrics(self):
        """When family_trades provided, all enriched fields are present."""
        from trading_assistant.skills.portfolio_what_if import PortfolioWhatIf

        base = datetime(2026, 3, 1, tzinfo=timezone.utc)
        trades_swing = [
            _make_trade(10.0, exit_time=base + timedelta(days=i), market_regime="trending_up")
            for i in range(30)
        ]
        trades_stock = [
            _make_trade(-5.0, exit_time=base + timedelta(days=i), market_regime="ranging")
            for i in range(30)
        ]

        wif = PortfolioWhatIf(
            family_daily_pnl={},
            current_weights={"swing": 0.5, "stock": 0.5},
            family_trades={"swing": trades_swing, "stock": trades_stock},
        )
        result = wif.evaluate({"swing": 0.6, "stock": 0.4})

        # Enriched fields
        assert "current_sortino" in result
        assert "proposed_sortino" in result
        assert "sortino_delta" in result
        assert "current_profit_factor" in result
        assert "proposed_profit_factor" in result
        assert "total_trades" in result
        assert result["total_trades"] == 60
        assert "trades_by_family" in result
        assert result["trades_by_family"]["swing"] == 30
        assert result["trades_by_family"]["stock"] == 30
        assert "current_pnl_by_regime" in result
        assert "proposed_pnl_by_regime" in result
        assert "trending_up" in result["current_pnl_by_regime"]
        assert "ranging" in result["current_pnl_by_regime"]
        assert result["method"] == "trade_level_rescaling"

    def test_evaluate_with_trades_max_drawdown_more_granular(self):
        """Intra-day drawdown exceeds daily drawdown — trade-level catches it."""
        from trading_assistant.skills.portfolio_what_if import PortfolioWhatIf

        base = datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc)
        # Family A: loses in the morning, gains in the afternoon (same day)
        # Family B: opposite pattern
        # Daily sum looks flat, but intra-day equity dips
        trades_a = [
            _make_trade(-100.0, exit_time=base + timedelta(hours=2)),   # morning loss
            _make_trade(110.0, exit_time=base + timedelta(hours=6)),    # afternoon gain
        ]
        trades_b = [
            _make_trade(50.0, exit_time=base + timedelta(hours=1)),    # morning gain
            _make_trade(-40.0, exit_time=base + timedelta(hours=5)),    # afternoon loss
        ]

        wif_trades = PortfolioWhatIf(
            family_daily_pnl={},
            current_weights={"a": 0.5, "b": 0.5},
            family_trades={"a": trades_a, "b": trades_b},
        )
        result_trades = wif_trades.evaluate({"a": 0.5, "b": 0.5})

        # Daily aggregate: a = -100+110=10, b = 50-40=10. Daily PnL = [20]. DD from daily = 0.
        wif_daily = PortfolioWhatIf(
            family_daily_pnl={"a": [10.0], "b": [10.0]},
            current_weights={"a": 0.5, "b": 0.5},
        )
        result_daily = wif_daily.evaluate({"a": 0.5, "b": 0.5})

        # Trade-level should capture intra-day drawdown that daily misses
        assert result_trades["current_max_drawdown"] >= result_daily["current_max_drawdown"]

    def test_evaluate_without_trades_backward_compatible(self):
        """Without family_trades, output is identical to original behavior."""
        from trading_assistant.skills.portfolio_what_if import PortfolioWhatIf

        family_pnl = {"swing": [10.0, -5.0, 8.0, -2.0] * 15, "stock": [5.0, 3.0, -1.0, 7.0] * 15}
        weights = {"swing": 0.5, "stock": 0.5}

        wif = PortfolioWhatIf(family_daily_pnl=family_pnl, current_weights=weights)
        result = wif.evaluate({"swing": 0.6, "stock": 0.4})

        assert result["method"] == "linear_rescaling"
        assert "current_sortino" not in result
        assert "proposed_profit_factor" not in result
        assert "total_trades" not in result
        assert "trades_by_family" not in result
        # Standard fields present
        assert "current_sharpe" in result
        assert "calmar_delta" in result
        assert "days_analyzed" in result

    def test_evaluate_with_trades_method_field(self):
        """Method is 'trade_level_rescaling' when trades provided."""
        from trading_assistant.skills.portfolio_what_if import PortfolioWhatIf

        trades = {"swing": [_make_trade(10.0)]}
        wif = PortfolioWhatIf(
            family_daily_pnl={},
            current_weights={"swing": 0.5},
            family_trades=trades,
        )
        result = wif.evaluate({"swing": 0.6})
        assert result["method"] == "trade_level_rescaling"

    def test_compute_daily_pnl_from_trades(self):
        """Verify date grouping and PnL summation."""
        from trading_assistant.skills.portfolio_what_if import PortfolioWhatIf

        base = datetime(2026, 3, 1, tzinfo=timezone.utc)
        trades = {
            "swing": [
                _make_trade(10.0, exit_time=base),
                _make_trade(5.0, exit_time=base),  # same day
                _make_trade(-3.0, exit_time=base + timedelta(days=1)),
            ],
        }
        result = PortfolioWhatIf._compute_daily_pnl_from_trades(trades)
        assert "swing" in result
        assert len(result["swing"]) == 2  # 2 unique dates
        assert result["swing"][0] == 15.0  # 10 + 5 on day 1
        assert result["swing"][1] == -3.0  # -3 on day 2

    def test_trade_date_extraction(self):
        """Exit_time preferred, entry_time fallback."""
        from trading_assistant.skills.portfolio_what_if import PortfolioWhatIf

        base_exit = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
        base_entry = datetime(2026, 3, 14, 10, 0, 0, tzinfo=timezone.utc)

        # Trade with both times — should use exit_time
        trade = _make_trade(10.0, exit_time=base_exit, entry_time=base_entry)
        assert PortfolioWhatIf._trade_date(trade) == "2026-03-15"

        # Trade with only entry_time
        from types import SimpleNamespace
        trade_no_exit = SimpleNamespace(exit_time=None, entry_time=base_entry, pnl=5.0)
        assert PortfolioWhatIf._trade_date(trade_no_exit) == "2026-03-14"

        # Trade with neither
        trade_no_ts = SimpleNamespace(exit_time=None, entry_time=None, pnl=5.0)
        assert PortfolioWhatIf._trade_date(trade_no_ts) == ""

    def test_empty_family_trades_fallback(self):
        """Empty dict for family_trades falls back to daily PnL."""
        from trading_assistant.skills.portfolio_what_if import PortfolioWhatIf

        family_pnl = {"swing": [10.0] * 30}
        wif = PortfolioWhatIf(
            family_daily_pnl=family_pnl,
            current_weights={"swing": 0.5},
            family_trades={},
        )
        result = wif.evaluate({"swing": 0.6})
        # Empty dict is falsy → falls back to daily PnL path
        assert result["method"] == "linear_rescaling"
        assert result["days_analyzed"] == 30

    def test_sortino_ratio(self):
        """Verify downside-only deviation computation."""
        from trading_assistant.skills.portfolio_what_if import PortfolioWhatIf

        # All positive → inf (no downside)
        assert PortfolioWhatIf._sortino([10.0, 20.0, 15.0]) == float("inf")

        # All negative
        result = PortfolioWhatIf._sortino([-10.0, -20.0, -15.0])
        assert result < 0  # Negative mean, positive downside_dev → negative Sortino

        # Mixed — should compute finite value
        result = PortfolioWhatIf._sortino([10.0, -5.0, 8.0, -3.0, 12.0, -1.0])
        assert result > 0
        assert result != float("inf")

        # Too few returns
        assert PortfolioWhatIf._sortino([5.0]) == 0.0

    def test_inf_values_capped_for_json_safety(self):
        """Sortino/profit_factor inf values must be capped for JSON serialization."""
        import json as _json
        from trading_assistant.skills.portfolio_what_if import PortfolioWhatIf

        base = datetime(2026, 3, 1, tzinfo=timezone.utc)
        # All winning trades → profit_factor = inf, all-positive daily → sortino = inf
        trades = {"swing": [
            _make_trade(10.0, exit_time=base + timedelta(days=i))
            for i in range(30)
        ]}
        wif = PortfolioWhatIf(
            family_daily_pnl={},
            current_weights={"swing": 0.5},
            family_trades=trades,
        )
        result = wif.evaluate({"swing": 0.6})

        # Must be JSON-serializable (no inf)
        serialized = _json.dumps(result)  # would raise ValueError if inf present
        assert "Infinity" not in serialized
        # Capped values should be finite
        assert result["current_profit_factor"] < 10000
        assert result["current_sortino"] < 10000

    def test_profit_factor_with_rescaling(self):
        """Profit factor changes when family weights change (mixed win/loss families)."""
        from trading_assistant.skills.portfolio_what_if import PortfolioWhatIf

        base = datetime(2026, 3, 1, tzinfo=timezone.utc)
        # Swing: mostly winners
        trades_swing = [
            _make_trade(10.0, exit_time=base + timedelta(hours=i))
            for i in range(8)
        ] + [_make_trade(-5.0, exit_time=base + timedelta(hours=8))]

        # Stock: mostly losers
        trades_stock = [
            _make_trade(-10.0, exit_time=base + timedelta(hours=i))
            for i in range(8)
        ] + [_make_trade(5.0, exit_time=base + timedelta(hours=8))]

        wif = PortfolioWhatIf(
            family_daily_pnl={},
            current_weights={"swing": 0.5, "stock": 0.5},
            family_trades={"swing": trades_swing, "stock": trades_stock},
        )

        # Increase swing (winners), decrease stock (losers) → PF should improve
        result = wif.evaluate({"swing": 0.7, "stock": 0.3})
        assert result["proposed_profit_factor"] > result["current_profit_factor"]
        assert "profit_factor_delta" in result
        assert result["profit_factor_delta"] > 0

    def test_handler_load_family_trades(self, tmp_path: Path):
        """Mock curated directory with JSONL files, verify correct family grouping."""
        from trading_assistant.schemas.strategy_profile import StrategyProfile, StrategyRegistry

        # Setup strategy registry
        registry = StrategyRegistry(strategies={
            "strat1": StrategyProfile(bot_id="bot_a", family="swing"),
            "strat2": StrategyProfile(bot_id="bot_b", family="stock"),
        })

        # Create curated data
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        bot_a_dir = tmp_path / today / "bot_a"
        bot_a_dir.mkdir(parents=True)
        bot_b_dir = tmp_path / today / "bot_b"
        bot_b_dir.mkdir(parents=True)

        trade_a = {
            "trade_id": "t1", "bot_id": "bot_a", "pair": "BTC/USDT",
            "side": "LONG", "entry_time": "2026-03-15T10:00:00+00:00",
            "exit_time": "2026-03-15T12:00:00+00:00",
            "entry_price": 100.0, "exit_price": 110.0,
            "position_size": 1.0, "pnl": 10.0, "pnl_pct": 10.0,
        }
        trade_b = {
            "trade_id": "t2", "bot_id": "bot_b", "pair": "ETH/USDT",
            "side": "SHORT", "entry_time": "2026-03-15T10:00:00+00:00",
            "exit_time": "2026-03-15T14:00:00+00:00",
            "entry_price": 50.0, "exit_price": 45.0,
            "position_size": 2.0, "pnl": 10.0, "pnl_pct": 10.0,
        }
        (bot_a_dir / "trades.jsonl").write_text(json.dumps(trade_a) + "\n")
        (bot_b_dir / "trades.jsonl").write_text(json.dumps(trade_b) + "\n")

        # Build a minimal handler-like object with the method
        from trading_assistant.orchestrator.handlers import Handlers

        handler = Handlers.__new__(Handlers)
        handler._curated_dir = tmp_path
        handler._strategy_registry = registry

        result = handler._load_family_trades_for_what_if(lookback_days=1)
        assert "swing" in result
        assert "stock" in result
        assert len(result["swing"]) == 1
        assert len(result["stock"]) == 1
        assert result["swing"][0].pnl == 10.0
        assert result["stock"][0].pnl == 10.0


# ---------------------------------------------------------------------------
# Portfolio outcome measurer
# ---------------------------------------------------------------------------

class TestPortfolioOutcomeMeasurer:
    def test_instantiation(self, tmp_path: Path):
        from trading_assistant.skills.portfolio_outcome_measurer import PortfolioOutcomeMeasurer

        measurer = PortfolioOutcomeMeasurer(
            findings_dir=tmp_path / "findings",
            curated_dir=tmp_path / "curated",
        )
        assert measurer is not None

    def test_check_emergency_no_data(self, tmp_path: Path):
        from trading_assistant.skills.portfolio_outcome_measurer import PortfolioOutcomeMeasurer

        measurer = PortfolioOutcomeMeasurer(
            findings_dir=tmp_path / "findings",
            curated_dir=tmp_path / "curated",
        )
        result = measurer.check_emergency()
        assert result is None  # No deployed suggestions → no emergency

    def test_what_if_accuracy_no_detection_context(self):
        """No detection_context → returns None."""
        from trading_assistant.skills.portfolio_outcome_measurer import PortfolioOutcomeMeasurer

        result = PortfolioOutcomeMeasurer._check_what_if_accuracy({}, 0.05)
        assert result is None

    def test_what_if_accuracy_no_what_if_result(self):
        """detection_context without what_if_result → returns None."""
        from trading_assistant.skills.portfolio_outcome_measurer import PortfolioOutcomeMeasurer

        suggestion = {"detection_context": {"current_config": {"a": 0.5}}}
        result = PortfolioOutcomeMeasurer._check_what_if_accuracy(suggestion, 0.05)
        assert result is None

    def test_what_if_accuracy_daily_aggregate_correct(self):
        """Daily-aggregate mode with matching direction → high accuracy."""
        from trading_assistant.skills.portfolio_outcome_measurer import PortfolioOutcomeMeasurer

        suggestion = {
            "detection_context": {
                "what_if_result": {
                    "calmar_delta": 0.15,
                    "sharpe_delta": 0.08,
                    "drawdown_delta": -0.02,
                },
            },
        }
        # Actual outcome is positive (composite improved)
        result = PortfolioOutcomeMeasurer._check_what_if_accuracy(suggestion, 0.05)
        assert result is not None
        assert result["mode"] == "daily_aggregate"
        assert result["calmar_direction_match"] is True
        assert result["sharpe_direction_match"] is True
        assert result["direction_accuracy"] >= 0.5

    def test_what_if_accuracy_trade_level_mode(self):
        """Trade-level mode detected when sortino present, includes sortino+pf check."""
        from trading_assistant.skills.portfolio_outcome_measurer import PortfolioOutcomeMeasurer

        suggestion = {
            "detection_context": {
                "what_if_result": {
                    "calmar_delta": 0.10,
                    "sharpe_delta": 0.05,
                    "drawdown_delta": -0.01,
                    "current_sortino": 1.2,
                    "proposed_sortino": 1.5,
                    "sortino_delta": 0.3,
                    "current_profit_factor": 1.5,
                    "proposed_profit_factor": 1.8,
                    "profit_factor_delta": 0.3,
                },
            },
        }
        result = PortfolioOutcomeMeasurer._check_what_if_accuracy(suggestion, 0.08)
        assert result is not None
        assert result["mode"] == "trade_level"
        assert "sortino_predicted" in result
        assert "sortino_direction_match" in result
        assert "profit_factor_predicted" in result
        assert "profit_factor_direction_match" in result
        # All predicted positive, actual positive → all match
        assert result["direction_accuracy"] == 1.0

    def test_what_if_accuracy_wrong_direction(self):
        """Predicted improvement but actual decline → low accuracy."""
        from trading_assistant.skills.portfolio_outcome_measurer import PortfolioOutcomeMeasurer

        suggestion = {
            "detection_context": {
                "what_if_result": {
                    "calmar_delta": 0.20,
                    "sharpe_delta": 0.10,
                    "drawdown_delta": -0.05,
                },
            },
        }
        # Actual outcome is negative (composite declined)
        result = PortfolioOutcomeMeasurer._check_what_if_accuracy(suggestion, -0.08)
        assert result is not None
        assert result["calmar_direction_match"] is False
        assert result["sharpe_direction_match"] is False
        assert result["direction_accuracy"] < 0.5

    def test_what_if_accuracy_inconclusive_actual(self):
        """Near-zero actual delta → direction treated as neutral (matches anything)."""
        from trading_assistant.skills.portfolio_outcome_measurer import PortfolioOutcomeMeasurer

        suggestion = {
            "detection_context": {
                "what_if_result": {
                    "calmar_delta": 0.15,
                    "sharpe_delta": -0.05,
                    "drawdown_delta": 0.0,
                },
            },
        }
        # Actual delta within ±0.02 → neutral
        result = PortfolioOutcomeMeasurer._check_what_if_accuracy(suggestion, 0.01)
        assert result is not None
        # Neutral actual matches everything
        assert result["direction_accuracy"] == 1.0

    def test_measure_deployed_includes_what_if_accuracy(self, tmp_path: Path):
        """Full integration: measure_deployed attaches what_if_accuracy to outcomes."""
        import json
        from datetime import datetime, timedelta, timezone
        from trading_assistant.skills.portfolio_outcome_measurer import PortfolioOutcomeMeasurer

        findings_dir = tmp_path / "findings"
        findings_dir.mkdir(parents=True)

        # Write a deployed suggestion with what-if predictions
        deploy_date = datetime.now(timezone.utc) - timedelta(days=35)
        suggestion = {
            "suggestion_id": "abc123",
            "bot_id": "PORTFOLIO",
            "status": "deployed",
            "category": "portfolio_allocation_rebalance",
            "deployed_at": deploy_date.isoformat(),
            "detection_context": {
                "what_if_result": {
                    "calmar_delta": 0.12,
                    "sharpe_delta": 0.05,
                    "drawdown_delta": -0.03,
                },
            },
        }
        (findings_dir / "suggestions.jsonl").write_text(
            json.dumps(suggestion) + "\n",
        )

        # Write ground truth before and after
        before_date = (deploy_date - timedelta(days=1)).strftime("%Y-%m-%d")
        after_date = (deploy_date + timedelta(days=31)).strftime("%Y-%m-%d")
        gt_lines = [
            json.dumps({"snapshot_date": before_date, "composite_score": 0.50}),
            json.dumps({"snapshot_date": after_date, "composite_score": 0.58}),
        ]
        (findings_dir / "portfolio_ground_truth.jsonl").write_text(
            "\n".join(gt_lines) + "\n",
        )

        measurer = PortfolioOutcomeMeasurer(
            findings_dir=findings_dir,
            curated_dir=tmp_path / "curated",
        )
        outcomes = measurer.measure_deployed()
        assert len(outcomes) == 1
        assert "what_if_accuracy" in outcomes[0]
        acc = outcomes[0]["what_if_accuracy"]
        assert acc["mode"] == "daily_aggregate"
        assert acc["calmar_direction_match"] is True
        assert acc["direction_accuracy"] >= 0.5

        # Running again should produce no new outcomes (dedup guard)
        outcomes_2 = measurer.measure_deployed()
        assert len(outcomes_2) == 0


# ---------------------------------------------------------------------------
# SuggestionScorer portfolio integration tests
# ---------------------------------------------------------------------------


class TestScorerPortfolioIntegration:
    """Verify SuggestionScorer reads portfolio outcomes and handles composite_delta."""

    def test_scorer_reads_portfolio_outcomes(self, tmp_path: Path):
        """SuggestionScorer._load_outcomes includes portfolio_outcomes.jsonl."""
        from trading_assistant.skills.suggestion_scorer import SuggestionScorer

        findings = tmp_path / "findings"
        findings.mkdir()

        # Bot-level outcome
        (findings / "outcomes.jsonl").write_text(
            json.dumps({
                "suggestion_id": "bot-001",
                "verdict": "positive",
                "pnl_delta": 50.0,
                "measurement_quality": "high",
            }) + "\n"
        )
        # Portfolio outcome
        (findings / "portfolio_outcomes.jsonl").write_text(
            json.dumps({
                "suggestion_id": "port-001",
                "verdict": "positive",
                "composite_delta": 0.08,
                "measurement_quality": "high",
            }) + "\n"
        )
        # Suggestions for both
        (findings / "suggestions.jsonl").write_text(
            json.dumps({"suggestion_id": "bot-001", "bot_id": "bot_a", "category": "signal"}) + "\n"
            + json.dumps({"suggestion_id": "port-001", "bot_id": "PORTFOLIO", "category": "allocation"}) + "\n"
        )

        scorer = SuggestionScorer(findings)
        scorecard = scorer.compute_scorecard()
        categories = {s.category for s in scorecard.scores}
        assert "signal" in categories
        assert "allocation" in categories
        assert len(scorecard.scores) == 2

    def test_composite_delta_used_as_pnl_fallback(self, tmp_path: Path):
        """When pnl_delta and pnl_delta_7d are absent, composite_delta is used."""
        from trading_assistant.skills.suggestion_scorer import SuggestionScorer

        findings = tmp_path / "findings"
        findings.mkdir()

        (findings / "outcomes.jsonl").write_text("")
        (findings / "portfolio_outcomes.jsonl").write_text(
            json.dumps({
                "suggestion_id": "port-002",
                "verdict": "positive",
                "composite_delta": 0.12,
                "measurement_quality": "high",
            }) + "\n"
        )
        (findings / "suggestions.jsonl").write_text(
            json.dumps({"suggestion_id": "port-002", "bot_id": "PORTFOLIO", "category": "risk_cap"}) + "\n"
        )

        scorer = SuggestionScorer(findings)
        scorecard = scorer.compute_scorecard()
        assert len(scorecard.scores) == 1
        score = scorecard.scores[0]
        assert score.avg_pnl_delta == 0.12
        assert score.win_rate == 1.0

    def test_lowercase_verdicts_detected_correctly(self, tmp_path: Path):
        """Lowercase verdicts from PortfolioOutcomeMeasurer are correctly identified."""
        from trading_assistant.skills.suggestion_scorer import SuggestionScorer

        findings = tmp_path / "findings"
        findings.mkdir()

        (findings / "outcomes.jsonl").write_text("")
        outcomes = [
            {"suggestion_id": "p1", "verdict": "positive", "composite_delta": 0.05, "measurement_quality": "high"},
            {"suggestion_id": "p2", "verdict": "negative", "composite_delta": -0.03, "measurement_quality": "high"},
            {"suggestion_id": "p3", "verdict": "inconclusive", "composite_delta": 0.01, "measurement_quality": "high"},
        ]
        (findings / "portfolio_outcomes.jsonl").write_text(
            "\n".join(json.dumps(o) for o in outcomes) + "\n"
        )
        (findings / "suggestions.jsonl").write_text(
            "\n".join(
                json.dumps({"suggestion_id": f"p{i}", "bot_id": "PORTFOLIO", "category": "allocation"})
                for i in range(1, 4)
            ) + "\n"
        )

        scorer = SuggestionScorer(findings)
        scorecard = scorer.compute_scorecard()
        assert len(scorecard.scores) == 1
        score = scorecard.scores[0]
        # 1 positive out of 2 conclusive (inconclusive excluded from both num and denom)
        assert score.sample_size == 2
        assert abs(score.win_rate - 0.5) < 0.01

    def test_portfolio_measurer_emits_lowercase_verdicts(self, tmp_path: Path):
        """PortfolioOutcomeMeasurer produces lowercase verdicts."""
        from trading_assistant.skills.portfolio_outcome_measurer import PortfolioOutcomeMeasurer

        findings = tmp_path / "findings"
        findings.mkdir()

        deploy_date = datetime.now(timezone.utc) - timedelta(days=35)
        suggestion = {
            "suggestion_id": "pv-001",
            "bot_id": "PORTFOLIO",
            "status": "deployed",
            "deployed_at": deploy_date.isoformat(),
            "category": "allocation",
        }
        (findings / "suggestions.jsonl").write_text(json.dumps(suggestion) + "\n")

        before_date = (deploy_date - timedelta(days=1)).strftime("%Y-%m-%d")
        after_date = (deploy_date + timedelta(days=31)).strftime("%Y-%m-%d")
        gt_lines = [
            json.dumps({"snapshot_date": before_date, "composite_score": 0.50}),
            json.dumps({"snapshot_date": after_date, "composite_score": 0.60}),
        ]
        (findings / "portfolio_ground_truth.jsonl").write_text("\n".join(gt_lines) + "\n")

        measurer = PortfolioOutcomeMeasurer(findings_dir=findings, curated_dir=tmp_path / "curated")
        outcomes = measurer.measure_deployed()
        assert len(outcomes) == 1
        assert outcomes[0]["verdict"] == "positive"  # lowercase, not "POSITIVE"
        assert outcomes[0]["outcome_source"] == "early_warning"

        # Negative case
        (findings / "portfolio_outcomes.jsonl").write_text("")  # reset
        gt_lines_neg = [
            json.dumps({"snapshot_date": before_date, "composite_score": 0.60}),
            json.dumps({"snapshot_date": after_date, "composite_score": 0.45}),
        ]
        (findings / "portfolio_ground_truth.jsonl").write_text("\n".join(gt_lines_neg) + "\n")
        (findings / "suggestions.jsonl").write_text(
            json.dumps({**suggestion, "suggestion_id": "pv-002"}) + "\n"
        )
        outcomes2 = measurer.measure_deployed()
        assert len(outcomes2) == 1
        assert outcomes2[0]["verdict"] == "negative"  # lowercase


# ---------------------------------------------------------------------------
# Portfolio curated pipeline tests (A1–A3)
# ---------------------------------------------------------------------------


def _make_handlers(tmp_path):
    """Create a minimal Handlers instance for testing curated pipeline."""
    from unittest.mock import AsyncMock, MagicMock

    from trading_assistant.orchestrator.handlers import Handlers

    curated_dir = tmp_path / "curated"
    curated_dir.mkdir()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "findings").mkdir()
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    h = Handlers(
        agent_runner=AsyncMock(),
        event_stream=MagicMock(),
        dispatcher=MagicMock(),
        notification_prefs=MagicMock(),
        curated_dir=curated_dir,
        memory_dir=memory_dir,
        runs_dir=runs_dir,
        source_root=tmp_path,
        bots=["bot_a", "bot_b"],
        raw_data_dir=raw_dir,
    )
    return h, curated_dir, raw_dir


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )


def _make_trade_event(bot_id, symbol, sector="", pnl=10.0, side="LONG", exposure=100.0):
    return {
        "event_type": "trade",
        "payload": {
            "bot_id": bot_id,
            "pair": symbol,
            "sector": sector,
            "side": side,
            "position_size_quote": exposure,
            "exposure_pct": 5.0,
            "strategy_id": "strat_a",
            "net_pnl": pnl,
            "entry_price": 100.0,
            "exit_price": 101.0,
        },
    }


def _make_summary_json(bot_id, net_pnl=10.0):
    return {
        "bot_id": bot_id,
        "date": "2026-03-27",
        "trade_count": 5,
        "win_count": 3,
        "loss_count": 2,
        "net_pnl": net_pnl,
        "gross_pnl": net_pnl + 2.0,
        "win_rate": 0.6,
        "avg_win": 5.0,
        "avg_loss": -2.5,
        "max_win": 10.0,
        "max_loss": -5.0,
        "total_volume": 1000.0,
    }


class TestPortfolioCuratedPipeline:
    def test_rebuild_curated_writes_portfolio_files(self, tmp_path):
        h, curated_dir, raw_dir = _make_handlers(tmp_path)
        date = "2026-03-27"

        # Write raw trade events for two bots
        for bot_id in ["bot_a", "bot_b"]:
            _write_jsonl(
                raw_dir / date / bot_id / "trade.jsonl",
                [_make_trade_event(bot_id, "AAPL", sector="Technology", pnl=10.0)],
            )
            _write_jsonl(
                raw_dir / date / bot_id / "portfolio_rule.jsonl",
                [{"event_type": "portfolio_rule_check", "payload": {
                    "rule_name": "max_exposure", "result": "pass",
                }}],
            )
        _write_jsonl(
            raw_dir / date / "bot_a" / "risk_decision.jsonl",
            [{
                "event_type": "risk_decision",
                "strategy_id": "strat_a",
                "risk_decision_id": "risk-deny-1",
                "decision": "deny",
                "deny_reason": "gross_exposure_limit",
            }],
        )

        h._rebuild_daily_curated_from_raw(date, ["bot_a", "bot_b"])

        portfolio_dir = curated_dir / date / "portfolio"
        assert portfolio_dir.exists()
        assert (portfolio_dir / "rule_blocks_summary.json").exists()
        assert (portfolio_dir / "family_snapshots.json").exists()
        assert (portfolio_dir / "concurrent_position_analysis.json").exists()
        assert (portfolio_dir / "sector_exposure.json").exists()

        # sector_exposure should have Technology sector
        sector_data = json.loads((portfolio_dir / "sector_exposure.json").read_text())
        assert sector_data["sector_count"] >= 1
        assert "Technology" in sector_data["sectors"]
        rules_data = json.loads((portfolio_dir / "rule_blocks_summary.json").read_text())
        assert any(record.get("risk_decision_id") == "risk-deny-1" for record in rules_data["records"])

    def test_sector_exposure_aggregation(self):
        from trading_assistant.skills.build_daily_metrics import build_sector_exposure

        events = [
            _make_trade_event("b1", "AAPL", sector="Technology", exposure=100.0),
            _make_trade_event("b1", "MSFT", sector="Technology", exposure=200.0),
            _make_trade_event("b1", "JPM", sector="Financials", exposure=150.0, side="SHORT"),
        ]
        result = build_sector_exposure(events)
        assert result["sector_count"] == 2
        tech = result["sectors"]["Technology"]
        assert tech["position_count"] == 2
        assert tech["total_exposure"] == 300.0
        assert tech["long_exposure"] == 300.0
        assert tech["short_exposure"] == 0.0
        assert sorted(tech["symbols"]) == ["AAPL", "MSFT"]

        fin = result["sectors"]["Financials"]
        assert fin["position_count"] == 1
        assert fin["short_exposure"] == 150.0

    def test_sector_exposure_empty_when_no_sectors(self):
        from trading_assistant.skills.build_daily_metrics import build_sector_exposure

        events = [_make_trade_event("b1", "AAPL", sector="", exposure=100.0)]
        result = build_sector_exposure(events)
        assert result["sector_count"] == 0
        assert result["sectors"] == {}

    def test_risk_card_has_correlation_matrix(self, tmp_path):
        h, curated_dir, raw_dir = _make_handlers(tmp_path)
        date = "2026-03-27"

        # Create 20 days of summary.json history for two bots
        for bot_id in ["bot_a", "bot_b"]:
            for d in range(20):
                past = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=d)).strftime("%Y-%m-%d")
                bot_dir = curated_dir / past / bot_id
                bot_dir.mkdir(parents=True, exist_ok=True)
                (bot_dir / "summary.json").write_text(
                    json.dumps(_make_summary_json(bot_id, net_pnl=10.0 + d)),
                    encoding="utf-8",
                )

        # Write trade data for today
        for bot_id in ["bot_a", "bot_b"]:
            _write_jsonl(
                raw_dir / date / bot_id / "trade.jsonl",
                [_make_trade_event(bot_id, "AAPL", sector="Tech", pnl=10.0)],
            )

        h._rebuild_daily_curated_from_raw(date, ["bot_a", "bot_b"])

        risk_card_path = curated_dir / date / "portfolio_risk_card.json"
        assert risk_card_path.exists()
        risk_data = json.loads(risk_card_path.read_text())
        assert "correlation_matrix" in risk_data

    def test_drawdown_correlation_written_by_portfolio_detectors(self, tmp_path):
        from unittest.mock import MagicMock

        h, curated_dir, raw_dir = _make_handlers(tmp_path)
        week_start = "2026-03-20"

        # Set up family_snapshots.json across 7 days
        for d in range(7):
            date_str = (datetime.strptime(week_start, "%Y-%m-%d") + timedelta(days=d)).strftime("%Y-%m-%d")
            pf_dir = curated_dir / date_str / "portfolio"
            pf_dir.mkdir(parents=True, exist_ok=True)
            (pf_dir / "family_snapshots.json").write_text(
                json.dumps([
                    {"family": "momentum", "total_net_pnl": 10.0 + d, "trade_count": 3, "win_count": 2},
                    {"family": "swing", "total_net_pnl": 5.0 - d, "trade_count": 2, "win_count": 1},
                ]),
                encoding="utf-8",
            )

        engine = MagicMock()
        engine.detect_family_imbalance.return_value = []
        engine.detect_correlation_concentration.return_value = []
        engine.detect_drawdown_tier_miscalibration.return_value = []
        engine.detect_coordination_gaps.return_value = []
        engine.detect_heat_cap_utilization.return_value = []

        h._run_portfolio_detectors(engine, week_start, "2026-03-26", MagicMock())

        dd_path = curated_dir / "weekly" / week_start / "drawdown_correlation.json"
        assert dd_path.exists()
        dd_data = json.loads(dd_path.read_text())
        assert "systemic_risk_score" in dd_data

    def test_portfolio_rebuild_tolerates_missing_raw_dir(self, tmp_path):
        h, curated_dir, raw_dir = _make_handlers(tmp_path)
        # raw dir exists but has no date subdirectory — should not crash
        h._rebuild_daily_curated_from_raw("2026-03-27", ["bot_a"])
        # No portfolio dir created since no data
        assert not (curated_dir / "2026-03-27" / "portfolio").exists()

    def test_family_snapshots_without_strategy_registry(self, tmp_path):
        from trading_assistant.skills.build_daily_metrics import build_family_snapshots

        summaries_data = [
            _make_summary_json("bot_a", net_pnl=10.0),
            _make_summary_json("bot_b", net_pnl=20.0),
        ]
        from trading_assistant.schemas.daily_metrics import BotDailySummary

        bot_summaries = [BotDailySummary.model_validate(s) for s in summaries_data]
        # strategy_registry=None should not crash
        result = build_family_snapshots(bot_summaries, None)
        assert isinstance(result, list)
