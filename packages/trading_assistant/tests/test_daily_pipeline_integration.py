"""Integration test -- full daily analysis pipeline end-to-end.

Creates trade events -> runs data reduction -> computes portfolio risk ->
quality gate -> prompt assembly. Validates every stage produces correct output.
"""
import json
from pathlib import Path


from trading_assistant.schemas.events import TradeEvent, MissedOpportunityEvent
from trading_assistant.skills.build_daily_metrics import DailyMetricsBuilder
from trading_assistant.skills.compute_portfolio_risk import PortfolioRiskComputer
from trading_assistant.analysis.quality_gate import QualityGate
from trading_assistant.analysis.prompt_assembler import DailyPromptAssembler
from tests.factories import make_trade, make_missed


def _make_trades(bot_id: str, n: int = 10) -> list[TradeEvent]:
    trades = []
    for i in range(n):
        pnl = 100.0 if i % 3 != 0 else -50.0
        trades.append(make_trade(
            trade_id=f"{bot_id}_t{i}",
            bot_id=bot_id,
            pair="BTCUSDT",
            entry_price=50000.0,
            pnl=pnl,
            pnl_pct=pnl / 500.0,
            entry_signal="EMA cross",
            exit_reason="TAKE_PROFIT" if pnl > 0 else "STOP_LOSS",
            market_regime="trending_up",
            process_quality_score=85 if pnl > 0 else 55,
            root_causes=["normal_win"] if pnl > 0 else ["regime_mismatch"],
        ))
    return trades


def _make_missed(bot_id: str) -> list[MissedOpportunityEvent]:
    return [
        make_missed(
            bot_id=bot_id,
            pair="ETHUSDT",
            signal="RSI divergence",
            blocked_by="volatility_filter",
            hypothetical_entry=3000.0,
            outcome_24h=500.0,
            confidence=0.7,
            assumption_tags=["mid_fill"],
        ),
    ]


class TestDailyPipelineIntegration:
    def test_full_pipeline(self, tmp_path: Path):
        date = "2026-03-01"
        bots = ["bot1", "bot2"]
        curated_dir = tmp_path / "curated"
        memory_dir = tmp_path / "memory"

        # --- Step 1: Data reduction ---
        for bot_id in bots:
            trades = _make_trades(bot_id)
            missed = _make_missed(bot_id)
            builder = DailyMetricsBuilder(date=date, bot_id=bot_id)
            builder.write_curated(trades, missed, base_dir=curated_dir)

        # Verify curated files exist
        for bot_id in bots:
            bot_dir = curated_dir / date / bot_id
            assert (bot_dir / "summary.json").exists()
            assert (bot_dir / "winners.json").exists()

        # --- Step 2: Portfolio risk ---
        summaries = []
        for bot_id in bots:
            summary_data = json.loads((curated_dir / date / bot_id / "summary.json").read_text())
            from trading_assistant.schemas.daily_metrics import BotDailySummary
            summaries.append(BotDailySummary(**summary_data))

        computer = PortfolioRiskComputer(
            date=date,
            bot_summaries=summaries,
            position_details={
                "bot1": [{"symbol": "BTC", "direction": "LONG", "exposure_pct": 20.0}],
                "bot2": [{"symbol": "ETH", "direction": "LONG", "exposure_pct": 15.0}],
            },
        )
        risk_card = computer.compute()
        risk_path = curated_dir / date / "portfolio_risk_card.json"
        risk_path.write_text(json.dumps(risk_card.model_dump(mode="json"), indent=2))

        # --- Step 3: Quality gate ---
        gate = QualityGate(
            report_id=f"daily-{date}",
            date=date,
            expected_bots=bots,
            curated_dir=curated_dir,
        )
        checklist = gate.run()
        assert checklist.overall == "PASS", f"Quality gate failed: {checklist.blocking_issues}"

        # --- Step 4: Prompt assembly ---
        # Set up memory dir
        policies_dir = memory_dir / "policies" / "v1"
        policies_dir.mkdir(parents=True)
        (policies_dir / "agent.md").write_text("You are a trading analyst.")
        (policies_dir / "trading_rules.md").write_text("Max 3 suggestions.")
        (policies_dir / "soul.md").write_text("Be helpful.")
        findings_dir = memory_dir / "findings"
        findings_dir.mkdir(parents=True)
        (findings_dir / "corrections.jsonl").write_text("")
        (findings_dir / "prompt_patterns.jsonl").write_text("")

        assembler = DailyPromptAssembler(
            date=date,
            bots=bots,
            curated_dir=curated_dir,
            memory_dir=memory_dir,
        )
        prompt = assembler.assemble()

        assert prompt.system_prompt
        assert prompt.task_prompt
        assert prompt.data
        assert "bot1" in prompt.data
        assert "bot2" in prompt.data
        assert "portfolio_risk_card" in prompt.data
        assert prompt.instructions
        assert len(prompt.context_files) > 0
