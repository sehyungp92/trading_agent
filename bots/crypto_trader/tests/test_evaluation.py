"""Tests for evaluation — 5-dimension report builder and text formatter."""

from unittest.mock import MagicMock

from crypto_trader.backtest.metrics import PerformanceMetrics
from crypto_trader.optimize.evaluation import (
    build_end_of_round_report,
    build_evaluation_report,
    format_dimension_text,
)
from crypto_trader.optimize.phase_state import PhaseState
from crypto_trader.optimize.types import EndOfRoundArtifacts


class TestBuildEvaluationReport:
    def test_report_with_no_journal(self):
        metrics = PerformanceMetrics(
            total_trades=50,
            win_rate=55.0,
            profit_factor=2.0,
            max_drawdown_pct=15.0,
            avg_mae_r=0.3,
            avg_mfe_r=1.5,
            avg_bars_held=10.0,
            exit_efficiency=0.6,
            a_setup_win_rate=60.0,
            b_setup_win_rate=45.0,
        )
        report = build_evaluation_report(metrics)

        assert "Signal Extraction" in report
        assert report["Signal Extraction"]["total_trades"] == 50
        assert report["Signal Discrimination"]["win_rate"] == 55.0
        assert report["Signal Discrimination"]["a_b_gap"] == 15.0
        assert report["Entry Mechanism"]["avg_mae_r"] == 0.3
        assert report["Trade Management"]["avg_bars_held"] == 10.0
        assert report["Exit Mechanism"]["exit_efficiency"] == 0.6

    def test_report_with_journal_entries(self):
        metrics = PerformanceMetrics(total_trades=3, exit_efficiency=0.5)

        class FakeEntry:
            def __init__(self, grade, method, reason):
                self.setup_grade = grade
                self.entry_method = method
                self.exit_reason = reason

        entries = [
            FakeEntry("A", "close", "tp1"),
            FakeEntry("A", "break", "trailing_stop"),
            FakeEntry("B", "close", "hard_time_stop"),
        ]
        report = build_evaluation_report(metrics, entries)

        assert report["Signal Extraction"]["a_setups"] == 2
        assert report["Signal Extraction"]["b_setups"] == 1
        assert report["Entry Mechanism"]["entry_method_breakdown"]["close"] == 2
        assert report["Exit Mechanism"]["trail_exits"] == 1
        assert report["Exit Mechanism"]["tp_exits"] == 1
        assert report["Trade Management"]["time_stop_rate"] > 0

    def test_report_empty_metrics(self):
        metrics = PerformanceMetrics()
        report = build_evaluation_report(metrics)

        assert report["Signal Extraction"]["total_trades"] == 0
        assert len(report) == 5


class TestFormatDimensionText:
    def test_formats_floats(self):
        data = {"win_rate": 55.0, "profit_factor": 2.1}
        text = format_dimension_text("Signal", data)
        assert "55.0000" in text
        assert "profit_factor" in text

    def test_formats_non_floats(self):
        data = {"total_trades": 50, "breakdown": {"A": 10}}
        text = format_dimension_text("Signal", data)
        assert "50" in text


class TestBuildEndOfRoundReport:
    def test_basic_report(self):
        state = PhaseState()
        state.phase_results = {
            1: {"base_score": 0.2, "final_score": 0.5, "focus": "Signal", "accepted_count": 3},
        }
        state.cumulative_mutations = {"entry.on_break": True}

        artifacts = EndOfRoundArtifacts(
            final_diagnostics_text="All metrics look good.",
            dimension_reports={
                "Signal Extraction": "total_trades: 50\nwin_rate: 55.0",
                "Entry Mechanism": "avg_mae_r: 0.3",
            },
            overall_verdict="Strategy ready for live testing.",
        )

        report = build_end_of_round_report("momentum", state, artifacts)

        assert "momentum" in report.upper() or "momentum" in report
        assert "Signal Extraction" in report
        assert "Entry Mechanism" in report
        assert "Strategy ready for live testing" in report
        assert "entry.on_break" in report

    def test_empty_artifacts(self):
        state = PhaseState()
        artifacts = EndOfRoundArtifacts()
        report = build_end_of_round_report("test", state, artifacts)
        assert "END-OF-ROUND" in report

    def test_extra_sections(self):
        state = PhaseState()
        artifacts = EndOfRoundArtifacts(
            extra_sections={"Custom Section": "custom content here"},
        )
        report = build_end_of_round_report("test", state, artifacts)
        assert "Custom Section" in report
        assert "custom content here" in report


# ══════════════════════════════════════════════════════════════════════
# Deeper evaluation dimension tests (Step 6)
# ══════════════════════════════════════════════════════════════════════


def _make_insights(**overrides):
    """Create a mock DiagnosticInsights with sensible defaults."""
    defaults = {
        "n_trades": 10,
        "win_rate": 55.0,
        "mean_r": 0.3,
        "profit_factor": 1.8,
        "per_confirmation": {
            "engulfing": {"n": 5, "wr": 60, "avg_r": 0.5, "total_r": 2.5, "pnl": 50},
            "hammer": {"n": 3, "wr": 50, "avg_r": 0.2, "total_r": 0.6, "pnl": 15},
        },
        "per_asset": {
            "BTC": {"n": 5, "wr": 60, "avg_r": 0.4},
            "ETH": {"n": 3, "wr": 40, "avg_r": -0.1},
        },
        "exit_attribution": {
            "trailing_stop": {"n": 4, "wr": 75, "avg_r": 0.8, "total_r": 3.2, "pnl_share": 0.5},
            "protective_stop": {"n": 3, "wr": 0, "avg_r": -0.9, "total_r": -2.7, "pnl_share": -0.3},
            "tp1": {"n": 2, "wr": 100, "avg_r": 1.0, "total_r": 2.0, "pnl_share": 0.3},
        },
        "mfe_capture": {
            "avg_mfe_r": 1.2, "avg_mae_r": -0.3,
            "avg_capture_pct": 0.6, "avg_giveback_pct": 0.4,
        },
        "direction": {
            "long": {"n": 6, "wr": 60, "avg_r": 0.4},
            "short": {"n": 4, "wr": 45, "avg_r": 0.1},
        },
        "confluence": {
            1: {"n": 4, "wr": 50, "avg_r": 0.1},
            2: {"n": 3, "wr": 60, "avg_r": 0.4},
            3: {"n": 2, "wr": 70, "avg_r": 0.8},
        },
        "grade": {
            "A": {"n": 3, "wr": 70, "avg_r": 0.6},
            "B": {"n": 7, "wr": 50, "avg_r": 0.2},
        },
        "duration": {"avg_bars": 8, "avg_hours": 2.0},
        "concentration": {"top1_pct": 0.3, "top20_pct": 0.7},
        "r_stats": {"mean": 0.3, "median": 0.2, "std": 0.8, "skew": 0.5},
        "worst_trades": [
            {"symbol": "BTC", "r_multiple": -0.9, "mfe_r": 0.3},
            {"symbol": "ETH", "r_multiple": -0.5, "mfe_r": 1.5},  # right-then-stopped
        ],
        "best_trades": [
            {"symbol": "BTC", "r_multiple": 2.0, "mfe_r": 2.5},
        ],
    }
    defaults.update(overrides)
    obj = MagicMock()
    for k, v in defaults.items():
        setattr(obj, k, v)
    return obj


class TestDeeperSignalExtraction:
    def test_r_skew_in_report(self):
        """R-distribution skewness appears in signal extraction."""
        metrics = PerformanceMetrics(total_trades=10, profit_factor=1.8)
        insights = _make_insights(r_stats={"mean": 0.3, "median": 0.2, "std": 0.8, "skew": 1.2})
        report = build_evaluation_report(metrics, insights=insights)
        assert "r_skew" in report["Signal Extraction"]
        assert report["Signal Extraction"]["r_skew"] == 1.2

    def test_concentration_in_report(self):
        """Concentration metrics in signal extraction."""
        metrics = PerformanceMetrics(total_trades=10, profit_factor=2.0)
        insights = _make_insights(concentration={"top1_pct": 0.6, "top20_pct": 0.9})
        report = build_evaluation_report(metrics, insights=insights)
        assert report["Signal Extraction"]["top1_pct"] == 0.6
        assert "concentrated" in report["Signal Extraction"]["assessment"].lower()


class TestDeeperSignalDiscrimination:
    def test_waste_quantification(self):
        """R lost to bad signals quantified."""
        metrics = PerformanceMetrics(total_trades=10, profit_factor=1.5)
        insights = _make_insights(per_confirmation={
            "engulfing": {"n": 5, "wr": 60, "avg_r": 0.5, "total_r": 2.5, "pnl": 50},
            "micro_shift": {"n": 3, "wr": 30, "avg_r": -0.4, "total_r": -1.2, "pnl": -12},
        })
        report = build_evaluation_report(metrics, insights=insights)
        dim = report["Signal Discrimination"]
        assert "r_lost_to_bad_signals" in dim
        assert dim["r_lost_to_bad_signals"] < 0

    def test_confluence_ladder(self):
        """Confluence effectiveness ladder in discrimination."""
        metrics = PerformanceMetrics(total_trades=10)
        insights = _make_insights()
        report = build_evaluation_report(metrics, insights=insights)
        assert "confluence_ladder" in report["Signal Discrimination"]


class TestDeeperExitMechanism:
    def test_right_then_stopped_detection(self):
        """Trades with MFE > 1R but R < 0 detected."""
        metrics = PerformanceMetrics(total_trades=10, exit_efficiency=0.4)
        insights = _make_insights(worst_trades=[
            {"symbol": "BTC", "r_multiple": -0.5, "mfe_r": 1.5},  # right-then-stopped
            {"symbol": "ETH", "r_multiple": -0.3, "mfe_r": 0.2},  # normal loser
        ])
        report = build_evaluation_report(metrics, insights=insights)
        dim = report["Exit Mechanism"]
        assert dim["right_then_stopped_count"] == 1

    def test_exit_attribution_detail(self):
        """Full exit attribution details in report."""
        metrics = PerformanceMetrics(total_trades=10, exit_efficiency=0.5)
        insights = _make_insights()
        report = build_evaluation_report(metrics, insights=insights)
        dim = report["Exit Mechanism"]
        assert "exit_attribution_detail" in dim


class TestDeeperEntryAndManagement:
    def test_direction_analysis_in_entry(self):
        """Per-direction entry quality in entry mechanism."""
        metrics = PerformanceMetrics(total_trades=10, avg_mae_r=0.3)
        insights = _make_insights()
        report = build_evaluation_report(metrics, insights=insights)
        dim = report["Entry Mechanism"]
        assert "long_avg_r" in dim

    def test_stagnation_detection_in_management(self):
        """Stagnation detected when high bars + low MFE."""
        metrics = PerformanceMetrics(total_trades=10, avg_bars_held=20)
        insights = _make_insights(
            duration={"avg_bars": 20, "avg_hours": 5},
            mfe_capture={"avg_mfe_r": 0.3, "avg_mae_r": -0.3,
                         "avg_capture_pct": 0.2, "avg_giveback_pct": 0.8},
        )
        report = build_evaluation_report(metrics, insights=insights)
        dim = report["Trade Management"]
        assert dim["stagnation_detected"] is True
