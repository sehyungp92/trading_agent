# tests/test_production_e2e.py
"""Production-shaped end-to-end test.

Chains the complete feedback loop from raw trade events through outcome
measurement and verifies that the next analysis prompt contains all
learning artifacts.

Steps:
1.  Create TradeEvent fixtures → raw data
2.  DailyMetricsBuilder.write_curated() → curated files
3.  QualityGate.run() → can_proceed
4.  DailyPromptAssembler → PromptPackage
5.  Simulate Claude response with STRUCTURED_OUTPUT block
6.  ResponseParser extracts structured data
7.  ResponseValidator strips/annotates
8.  SuggestionTracker records suggestions
9.  accept() + mark_deployed() (production lifecycle)
10. Record outcome via SuggestionOutcome
11. mark_measured() completes lifecycle
12. ForecastTracker records week
13. Assert: next ContextBuilder.base_package() contains learning artifacts
14. Assert: measured suggestion is NOT in active_suggestions
"""
from __future__ import annotations

from pathlib import Path


from trading_assistant.analysis.context_builder import ContextBuilder
from trading_assistant.analysis.prompt_assembler import DailyPromptAssembler
from trading_assistant.analysis.quality_gate import QualityGate
from trading_assistant.analysis.response_parser import parse_response
from trading_assistant.analysis.response_validator import ResponseValidator
from trading_assistant.schemas.events import TradeEvent, MissedOpportunityEvent
from trading_assistant.schemas.forecast_tracking import ForecastRecord
from trading_assistant.schemas.suggestion_tracking import (
    SuggestionOutcome,
    SuggestionRecord,
    SuggestionStatus,
)
from trading_assistant.skills.build_daily_metrics import DailyMetricsBuilder
from trading_assistant.skills.forecast_tracker import ForecastTracker
from trading_assistant.skills.suggestion_scorer import SuggestionScorer
from trading_assistant.skills.suggestion_tracker import SuggestionTracker
from tests.factories import make_trade, make_missed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATE = "2026-03-10"
BOT_ID = "bot1"


def _make_trades(n: int = 10) -> list[TradeEvent]:
    """Generate realistic trade events."""
    trades: list[TradeEvent] = []
    for i in range(n):
        win = i % 3 != 0
        pnl = 120.0 if win else -60.0
        trades.append(make_trade(
            trade_id=f"{BOT_ID}_t{i}",
            bot_id=BOT_ID,
            pair="BTCUSDT",
            entry_price=50000.0,
            position_size=0.1,
            pnl=pnl,
            pnl_pct=pnl / 50000.0 * 100,
            entry_signal="EMA_cross",
            exit_reason="TAKE_PROFIT" if win else "STOP_LOSS",
            market_regime="trending_up",
            process_quality_score=90 if win else 55,
            root_causes=["normal_win"] if win else ["regime_mismatch"],
        ))
    return trades


def _make_missed() -> list[MissedOpportunityEvent]:
    return [make_missed(
        bot_id=BOT_ID,
        pair="ETHUSDT",
        signal="RSI_oversold",
        signal_strength=0.8,
        blocked_by="volume_gate",
        hypothetical_entry=3000.0,
        outcome_1h=50.0,
        confidence=0.6,
    )]


MOCK_CLAUDE_RESPONSE = """\
# Daily Analysis Report for 2026-03-10

## Portfolio Overview
Total PnL: +600.00 USDT across 10 trades. Win rate 70%. No crowding alerts.

## bot1
### What worked
EMA cross signals in trending_up regime produced consistent winners.

### What failed
3 losing trades with regime_mismatch root cause.

### Missed opportunities
1 missed ETHUSDT RSI signal blocked by volume_gate — hypothetical +50 USDT.

## Actionable Items
1. Widen stop loss by 0.3 ATR to reduce premature exits.
   Expected impact: +0.2% to +0.5% daily PnL, drawdown reduction ~10%.
   Evidence: 3 trades with regime_mismatch, 30-day sample.

<!-- STRUCTURED_OUTPUT
{
  "predictions": [
    {
      "bot_id": "bot1",
      "metric": "pnl",
      "direction": "improve",
      "confidence": 0.7,
      "timeframe_days": 7,
      "reasoning": "Trending regime continues, EMA signal strong"
    },
    {
      "bot_id": "bot1",
      "metric": "win_rate",
      "direction": "stable",
      "confidence": 0.6,
      "timeframe_days": 7,
      "reasoning": "No regime change expected"
    }
  ],
  "suggestions": [
    {
      "suggestion_id": "#prod_s001",
      "bot_id": "bot1",
      "category": "exit_timing",
      "title": "Widen stop loss by 0.3 ATR",
      "expected_impact": "+0.2% to +0.5% daily PnL",
      "confidence": 0.75,
      "evidence_summary": "3 regime_mismatch trades over 30 days",
      "proposed_value": 0.3,
      "target_param": "stop_atr_multiplier"
    }
  ],
  "structural_proposals": []
}
-->
"""


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestProductionE2E:
    """Single test proving the full closed-loop from raw events to learning context."""

    def test_full_production_loop(self, tmp_path: Path):
        # --- Directory layout ---
        curated_dir = tmp_path / "data" / "curated"
        memory_dir = tmp_path / "memory"
        findings_dir = memory_dir / "findings"
        findings_dir.mkdir(parents=True)
        (memory_dir / "policies" / "v1").mkdir(parents=True)

        # --- Step 1: Create raw trade data ---
        trades = _make_trades(10)
        missed = _make_missed()

        # --- Step 2: DailyMetricsBuilder → curated files ---
        builder = DailyMetricsBuilder(date=DATE, bot_id=BOT_ID)
        output_dir = builder.write_curated(
            trades=trades,
            missed=missed,
            base_dir=curated_dir,
        )
        assert output_dir.exists()
        assert (output_dir / "summary.json").exists()
        assert (output_dir / "trades.jsonl").exists()

        # --- Step 3: QualityGate check ---
        gate = QualityGate(
            report_id=f"daily-{DATE}",
            date=DATE,
            expected_bots=[BOT_ID],
            curated_dir=curated_dir,
        )
        checklist = gate.run()
        assert checklist.can_proceed is True

        # --- Step 4: DailyPromptAssembler → PromptPackage ---
        assembler = DailyPromptAssembler(
            date=DATE,
            bots=[BOT_ID],
            curated_dir=curated_dir,
            memory_dir=memory_dir,
        )
        pkg = assembler.assemble()
        assert pkg.system_prompt is not None
        assert pkg.task_prompt is not None
        assert BOT_ID in pkg.task_prompt
        assert pkg.instructions

        # --- Step 5: Simulate Claude response ---
        response_text = MOCK_CLAUDE_RESPONSE

        # --- Step 6: ResponseParser extracts structured data ---
        parsed = parse_response(response_text)
        assert parsed.parse_success is True
        assert len(parsed.suggestions) == 1
        assert len(parsed.predictions) == 2
        assert parsed.suggestions[0].suggestion_id == "#prod_s001"
        assert parsed.suggestions[0].category == "exit_timing"

        # --- Step 7: ResponseValidator strips/annotates ---
        validator = ResponseValidator()
        result = validator.validate(parsed)
        assert len(result.approved_suggestions) == 1
        assert result.approved_suggestions[0].suggestion_id == "#prod_s001"

        # --- Step 8: SuggestionTracker records suggestions ---
        tracker = SuggestionTracker(store_dir=findings_dir)
        for s in result.approved_suggestions:
            rec = SuggestionRecord(
                suggestion_id=s.suggestion_id,
                bot_id=s.bot_id,
                title=s.title,
                tier="parameter",
                category=s.category,
                source_report_id=f"daily-{DATE}",
                confidence=s.confidence,
            )
            tracker.record(rec)

        all_suggestions = tracker.load_all()
        assert len(all_suggestions) == 1
        assert all_suggestions[0]["status"] == SuggestionStatus.PROPOSED.value

        # --- Step 9: accept() + mark_deployed() (production lifecycle) ---
        tracker.accept("#prod_s001")
        tracker.mark_deployed("#prod_s001")

        all_suggestions = tracker.load_all()
        s = [r for r in all_suggestions if r["suggestion_id"] == "#prod_s001"][0]
        assert s["status"] == SuggestionStatus.DEPLOYED.value
        assert s["accepted_at"] is not None
        assert s["deployed_at"] is not None

        # --- Step 10: Record outcome ---
        outcome = SuggestionOutcome(
            suggestion_id="#prod_s001",
            implemented_date=DATE,
            pnl_delta_7d=180.0,
            win_rate_delta_7d=0.04,
            drawdown_delta_7d=-0.02,
        )
        tracker.record_outcome(outcome)

        outcomes = tracker.load_outcomes()
        assert len(outcomes) == 1
        assert outcomes[0]["pnl_delta_7d"] == 180.0

        # --- Step 11: mark_measured() completes lifecycle ---
        tracker.mark_measured("#prod_s001")

        all_suggestions = tracker.load_all()
        s = [r for r in all_suggestions if r["suggestion_id"] == "#prod_s001"][0]
        assert s["status"] == SuggestionStatus.MEASURED.value
        assert s["measured_at"] is not None
        assert s["resolved_at"] is not None  # MEASURED sets resolved_at

        # --- Step 12: ForecastTracker records week ---
        forecast_tracker = ForecastTracker(findings_dir)
        forecast_record = ForecastRecord(
            week_start="2026-03-04",
            week_end="2026-03-10",
            predictions_reviewed=2,
            correct_predictions=1,
            accuracy=0.5,
            by_bot={BOT_ID: 0.5},
        )
        forecast_tracker.record_week(forecast_record)

        loaded_records = forecast_tracker.load_all()
        assert len(loaded_records) == 1

        # --- Step 13: Assert next ContextBuilder.base_package() has learning artifacts ---
        ctx = ContextBuilder(memory_dir, curated_dir=curated_dir)
        next_pkg = ctx.base_package()

        # Outcome measurements present
        assert "outcome_measurements" in next_pkg.data
        assert len(next_pkg.data["outcome_measurements"]) == 1
        assert next_pkg.data["outcome_measurements"][0]["suggestion_id"] == "#prod_s001"

        # Forecast meta-analysis present
        assert "forecast_meta_analysis" in next_pkg.data

        # Category scorecard present (1 positive outcome in exit_timing)
        scorer = SuggestionScorer(findings_dir)
        scorecard = scorer.compute_scorecard()
        assert len(scorecard.scores) >= 1
        exit_score = scorecard.get_score(BOT_ID, "exit_timing")
        assert exit_score is not None
        assert exit_score.win_rate == 1.0
        assert exit_score.sample_size == 1

        assert "category_scorecard" in next_pkg.data

        # --- Step 14: Assert measured suggestion is NOT in active_suggestions ---
        active = ctx.load_active_suggestions()
        active_ids = [s["suggestion_id"] for s in active]
        assert "#prod_s001" not in active_ids, (
            "Measured suggestion should not appear in active_suggestions"
        )
