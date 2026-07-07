# tests/test_auto_outcome_measurer.py
"""Tests for automated suggestion outcome measurement."""
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

from trading_assistant.schemas.outcome_measurement import OutcomeMeasurement, Verdict


class TestOutcomeMeasurementSchema:
    def test_verdict_positive(self):
        m = OutcomeMeasurement(
            suggestion_id="s1",
            implemented_date="2026-02-20",
            measurement_date="2026-03-01",
            window_days=7,
            pnl_before=100.0, pnl_after=200.0,
            win_rate_before=0.5, win_rate_after=0.6,
            drawdown_before=5.0, drawdown_after=4.0,
            before_trade_count=10, after_trade_count=10,
        )
        assert m.verdict == Verdict.POSITIVE
        assert m.pnl_delta == 100.0

    def test_verdict_negative(self):
        m = OutcomeMeasurement(
            suggestion_id="s2",
            implemented_date="2026-02-20",
            measurement_date="2026-03-01",
            window_days=7,
            pnl_before=200.0, pnl_after=50.0,
            win_rate_before=0.6, win_rate_after=0.4,
            before_trade_count=10, after_trade_count=10,
        )
        assert m.verdict == Verdict.NEGATIVE

    def test_verdict_neutral(self):
        m = OutcomeMeasurement(
            suggestion_id="s3",
            implemented_date="2026-02-20",
            measurement_date="2026-03-01",
            window_days=7,
            pnl_before=100.0, pnl_after=102.0,
            win_rate_before=0.5, win_rate_after=0.51,
            before_trade_count=10, after_trade_count=10,
        )
        assert m.verdict == Verdict.NEUTRAL


class TestAutoOutcomeMeasurer:
    def test_crypto_categories_have_target_metrics(self):
        from trading_assistant.skills.auto_outcome_measurer import AutoOutcomeMeasurer

        mapping = AutoOutcomeMeasurer.CATEGORY_TO_TARGET_METRIC

        assert mapping["funding_threshold"] == "pnl"
        assert mapping["leverage_cap"] == "drawdown"
        assert mapping["confluence_count"] == "win_rate"
        assert mapping["setup_grade_filter"] == "win_rate"

    def test_measure_suggestion_outcome(self, tmp_path):
        from trading_assistant.skills.auto_outcome_measurer import AutoOutcomeMeasurer

        # Create curated summaries for before/after periods
        self._write_summaries(tmp_path, "2026-02-15", "bot1", pnl=50, wins=5, total=10)
        self._write_summaries(tmp_path, "2026-02-16", "bot1", pnl=60, wins=6, total=10)
        self._write_summaries(tmp_path, "2026-02-25", "bot1", pnl=100, wins=8, total=10)
        self._write_summaries(tmp_path, "2026-02-26", "bot1", pnl=90, wins=7, total=10)

        measurer = AutoOutcomeMeasurer(curated_dir=tmp_path)
        result = measurer.measure(
            suggestion_id="s1",
            bot_id="bot1",
            implemented_date="2026-02-20",
            before_days=7,
            after_days=7,
        )
        assert isinstance(result, OutcomeMeasurement)
        assert result.pnl_after > result.pnl_before

    def test_insufficient_data_returns_none(self, tmp_path):
        from trading_assistant.skills.auto_outcome_measurer import AutoOutcomeMeasurer

        measurer = AutoOutcomeMeasurer(curated_dir=tmp_path)
        result = measurer.measure(
            suggestion_id="s1",
            bot_id="bot1",
            implemented_date="2026-02-20",
            before_days=7,
            after_days=7,
        )
        assert result is None

    def test_measure_prefers_net_pnl(self, tmp_path):
        from trading_assistant.skills.auto_outcome_measurer import AutoOutcomeMeasurer

        self._write_summaries(tmp_path, "2026-02-15", "bot1", pnl=100, wins=5, total=10, net_pnl=80)
        self._write_summaries(tmp_path, "2026-02-25", "bot1", pnl=150, wins=8, total=10, net_pnl=140)

        measurer = AutoOutcomeMeasurer(curated_dir=tmp_path)
        result = measurer.measure(
            suggestion_id="s1",
            bot_id="bot1",
            implemented_date="2026-02-20",
            before_days=7,
            after_days=7,
        )

        assert result is not None
        assert result.pnl_before == 80
        assert result.pnl_after == 140

    def test_progressive_measurement_records_feedback_once_when_selected(self, tmp_path):
        from trading_assistant.schemas.proposal_ledger import ProposalCandidate, ProposalKind, ProposalSource
        from trading_assistant.skills.auto_outcome_measurer import AutoOutcomeMeasurer
        from trading_assistant.skills.proposal_ledger import ProposalLedger

        impl = datetime(2026, 3, 1)
        for offset in range(-30, 0):
            day = (impl + timedelta(days=offset)).strftime("%Y-%m-%d")
            self._write_summaries(tmp_path, day, "bot1", pnl=10, wins=5, total=10)
        for offset in range(0, 30):
            day = (impl + timedelta(days=offset)).strftime("%Y-%m-%d")
            self._write_summaries(tmp_path, day, "bot1", pnl=20, wins=7, total=10)

        findings = tmp_path / "findings"
        findings.mkdir()
        (findings / "suggestions.jsonl").write_text(json.dumps({
            "suggestion_id": "s-feedback",
            "bot_id": "bot1",
            "category": "exit_timing",
            "status": "deployed",
            "hypothesis_id": "hyp-feedback",
            "proposal_id": "proposal-feedback",
        }) + "\n", encoding="utf-8")

        ledger = ProposalLedger(findings)
        ledger.record_candidate(ProposalCandidate(
            proposal_id="proposal-feedback",
            source=ProposalSource.LLM_DAILY,
            kind=ProposalKind.PARAMETER_CHANGE,
            bot_id="bot1",
            title="Improve exits",
        ))
        calibration_tracker = MagicMock()
        hypothesis_library = MagicMock()

        measurer = AutoOutcomeMeasurer(
            curated_dir=tmp_path,
            findings_dir=findings,
            calibration_tracker=calibration_tracker,
            hypothesis_library=hypothesis_library,
            proposal_ledger=ledger,
        )

        selected = measurer.measure_progressive(
            suggestion_id="s-feedback",
            bot_id="bot1",
            implemented_date="2026-03-01",
        )

        assert selected is not None
        assert selected.window_days == 30
        assert calibration_tracker.record_outcome.call_count == 0
        assert hypothesis_library.record_outcome.call_count == 0
        assert ledger.get_by_id("proposal-feedback").outcomes == []

        measurer.record_measurement_feedback(selected)

        assert calibration_tracker.record_outcome.call_count == 0
        hypothesis_library.record_outcome.assert_called_once_with(
            "hyp-feedback", positive=True,
        )
        record = ledger.get_by_id("proposal-feedback")
        assert record is not None
        assert len(record.outcomes) == 1
        assert record.outcomes[0].verdict == "positive"

    def _write_summaries(
        self,
        base: Path,
        date: str,
        bot_id: str,
        pnl: float,
        wins: int,
        total: int,
        net_pnl: float | None = None,
    ):
        bot_dir = base / date / bot_id
        bot_dir.mkdir(parents=True, exist_ok=True)
        (bot_dir / "summary.json").write_text(json.dumps({
            "date": date, "bot_id": bot_id,
            "gross_pnl": pnl,
            "net_pnl": pnl if net_pnl is None else net_pnl,
            "win_count": wins,
            "total_trades": total,
            "max_drawdown_pct": 3.0,
        }))
