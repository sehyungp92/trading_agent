"""Tests for FillQualityAnalyzer — fill quality schemas and analyzer logic."""
from __future__ import annotations


from trading_assistant.schemas.fill_quality import FillQualityReport, FillStats, SymbolFillQuality
from trading_assistant.skills.fill_quality_analyzer import FillQualityAnalyzer
from tests.factories import make_trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trade(
    pair: str = "NQ",
    pnl: float = 100.0,
    entry_fill: dict | None = None,
    exit_fill: dict | None = None,
):
    return make_trade(
        trade_id=f"t-{pair}-{pnl}", bot_id="test_bot", pair=pair, pnl=pnl,
        entry_fill_details=entry_fill, exit_fill_details=exit_fill,
    )


# ---------------------------------------------------------------------------
# Schema round-trips
# ---------------------------------------------------------------------------

class TestFillQualitySchemas:
    def test_fill_stats_defaults(self):
        fs = FillStats()
        assert fs.sample_count == 0
        assert fs.avg_slippage_bps == 0.0
        assert fs.by_fill_type == {}

    def test_fill_stats_round_trip(self):
        fs = FillStats(
            sample_count=10,
            avg_slippage_bps=1.5,
            median_slippage_bps=1.2,
            p95_slippage_bps=3.0,
            avg_fill_latency_ms=50.0,
            adverse_fill_pct=60.0,
            by_fill_type={"market": 8, "limit": 2},
        )
        data = fs.model_dump(mode="json")
        restored = FillStats(**data)
        assert restored.sample_count == 10
        assert restored.by_fill_type["market"] == 8

    def test_symbol_fill_quality_round_trip(self):
        sfq = SymbolFillQuality(
            symbol="NQ",
            entry_stats=FillStats(sample_count=5, avg_slippage_bps=1.0),
            exit_stats=FillStats(sample_count=5, avg_slippage_bps=0.5),
            net_adverse_impact_bps=1.5,
        )
        data = sfq.model_dump(mode="json")
        restored = SymbolFillQuality(**data)
        assert restored.symbol == "NQ"
        assert restored.net_adverse_impact_bps == 1.5

    def test_report_round_trip(self):
        report = FillQualityReport(
            bot_id="bot_a",
            date="2026-03-01",
            coverage_pct=80.0,
            adverse_selection_detected=True,
        )
        data = report.model_dump(mode="json")
        restored = FillQualityReport(**data)
        assert restored.adverse_selection_detected is True
        assert restored.coverage_pct == 80.0


# ---------------------------------------------------------------------------
# Analyzer — empty / no-data cases
# ---------------------------------------------------------------------------

class TestFillQualityAnalyzerEmpty:
    def test_empty_trade_list(self):
        analyzer = FillQualityAnalyzer("bot_a", "2026-03-01")
        report = analyzer.compute([])
        assert report.bot_id == "bot_a"
        assert report.coverage_pct == 0.0
        assert report.adverse_selection_detected is False

    def test_trades_without_fill_details(self):
        trades = [_make_trade(pnl=50.0), _make_trade(pnl=-20.0)]
        analyzer = FillQualityAnalyzer("bot_a", "2026-03-01")
        report = analyzer.compute(trades)
        assert report.coverage_pct == 0.0
        assert report.overall_entry.sample_count == 0
        assert report.overall_exit.sample_count == 0


# ---------------------------------------------------------------------------
# Analyzer — basic computations
# ---------------------------------------------------------------------------

class TestFillQualityAnalyzerBasic:
    def test_entry_only_fills(self):
        trades = [
            _make_trade(entry_fill={"slippage_bps": 2.0, "fill_latency_ms": 30.0, "fill_type": "market"}),
            _make_trade(entry_fill={"slippage_bps": -1.0, "fill_latency_ms": 20.0, "fill_type": "limit"}),
        ]
        analyzer = FillQualityAnalyzer("bot_a", "2026-03-01")
        report = analyzer.compute(trades)

        assert report.overall_entry.sample_count == 2
        assert report.overall_entry.avg_slippage_bps == 0.5  # (2.0 + -1.0) / 2
        assert report.overall_entry.avg_fill_latency_ms == 25.0
        # Only first fill (2.0 > 0) is adverse
        assert report.overall_entry.adverse_fill_pct == 50.0
        assert report.overall_exit.sample_count == 0

    def test_exit_only_fills(self):
        trades = [
            _make_trade(exit_fill={"slippage_bps": 3.0, "fill_latency_ms": 10.0, "fill_type": "market"}),
        ]
        analyzer = FillQualityAnalyzer("bot_a", "2026-03-01")
        report = analyzer.compute(trades)

        assert report.overall_exit.sample_count == 1
        assert report.overall_exit.avg_slippage_bps == 3.0
        assert report.overall_entry.sample_count == 0

    def test_both_entry_and_exit_fills(self):
        trades = [
            _make_trade(
                entry_fill={"slippage_bps": 1.0, "fill_latency_ms": 20.0, "fill_type": "market"},
                exit_fill={"slippage_bps": 0.5, "fill_latency_ms": 15.0, "fill_type": "limit"},
            ),
        ]
        analyzer = FillQualityAnalyzer("bot_a", "2026-03-01")
        report = analyzer.compute(trades)

        assert report.overall_entry.sample_count == 1
        assert report.overall_exit.sample_count == 1
        assert report.coverage_pct == 100.0

    def test_coverage_pct_partial(self):
        trades = [
            _make_trade(entry_fill={"slippage_bps": 1.0, "fill_latency_ms": 10.0, "fill_type": "market"}),
            _make_trade(),  # no fill data
            _make_trade(),  # no fill data
            _make_trade(),  # no fill data
        ]
        analyzer = FillQualityAnalyzer("bot_a", "2026-03-01")
        report = analyzer.compute(trades)
        assert report.coverage_pct == 25.0


# ---------------------------------------------------------------------------
# Analyzer — per-symbol breakdown
# ---------------------------------------------------------------------------

class TestFillQualityAnalyzerPerSymbol:
    def test_single_symbol(self):
        trades = [
            _make_trade(
                pair="NQ",
                entry_fill={"slippage_bps": 2.0, "fill_latency_ms": 10.0, "fill_type": "market"},
            ),
        ]
        analyzer = FillQualityAnalyzer("bot_a", "2026-03-01")
        report = analyzer.compute(trades)

        assert "NQ" in report.by_symbol
        nq = report.by_symbol["NQ"]
        assert nq.entry_stats.sample_count == 1
        assert nq.entry_stats.avg_slippage_bps == 2.0

    def test_multiple_symbols(self):
        trades = [
            _make_trade(
                pair="NQ",
                entry_fill={"slippage_bps": 2.0, "fill_latency_ms": 10.0, "fill_type": "market"},
            ),
            _make_trade(
                pair="ES",
                entry_fill={"slippage_bps": 1.0, "fill_latency_ms": 5.0, "fill_type": "limit"},
            ),
            _make_trade(
                pair="NQ",
                entry_fill={"slippage_bps": 4.0, "fill_latency_ms": 15.0, "fill_type": "market"},
            ),
        ]
        analyzer = FillQualityAnalyzer("bot_a", "2026-03-01")
        report = analyzer.compute(trades)

        assert len(report.by_symbol) == 2
        assert report.by_symbol["NQ"].entry_stats.sample_count == 2
        assert report.by_symbol["ES"].entry_stats.sample_count == 1

    def test_net_adverse_impact_bps(self):
        trades = [
            _make_trade(
                pair="NQ",
                entry_fill={"slippage_bps": 2.0, "fill_latency_ms": 10.0, "fill_type": "market"},
                exit_fill={"slippage_bps": 1.5, "fill_latency_ms": 10.0, "fill_type": "market"},
            ),
        ]
        analyzer = FillQualityAnalyzer("bot_a", "2026-03-01")
        report = analyzer.compute(trades)

        nq = report.by_symbol["NQ"]
        assert nq.net_adverse_impact_bps == 3.5  # 2.0 + 1.5


# ---------------------------------------------------------------------------
# Analyzer — adverse selection detection
# ---------------------------------------------------------------------------

class TestFillQualityAnalyzerAdverseSelection:
    def test_adverse_detected_when_high_pct_and_slippage(self):
        """Adverse selection: >60% adverse fills AND avg slippage > 1 bps."""
        trades = [
            _make_trade(entry_fill={"slippage_bps": 3.0, "fill_latency_ms": 10.0, "fill_type": "market"}),
            _make_trade(entry_fill={"slippage_bps": 2.0, "fill_latency_ms": 10.0, "fill_type": "market"}),
            _make_trade(entry_fill={"slippage_bps": 4.0, "fill_latency_ms": 10.0, "fill_type": "market"}),
            _make_trade(entry_fill={"slippage_bps": -0.5, "fill_latency_ms": 10.0, "fill_type": "limit"}),
        ]
        analyzer = FillQualityAnalyzer("bot_a", "2026-03-01")
        report = analyzer.compute(trades)
        # 3 out of 4 adverse (75%) and avg slippage = (3+2+4-0.5)/4 = 2.125 > 1
        assert report.adverse_selection_detected is True

    def test_no_adverse_when_low_pct(self):
        """Below 60% adverse → not detected even if avg slippage high."""
        trades = [
            _make_trade(entry_fill={"slippage_bps": 5.0, "fill_latency_ms": 10.0, "fill_type": "market"}),
            _make_trade(entry_fill={"slippage_bps": -1.0, "fill_latency_ms": 10.0, "fill_type": "limit"}),
            _make_trade(entry_fill={"slippage_bps": -2.0, "fill_latency_ms": 10.0, "fill_type": "limit"}),
        ]
        analyzer = FillQualityAnalyzer("bot_a", "2026-03-01")
        report = analyzer.compute(trades)
        # 1 out of 3 adverse (33%) — below 60% threshold
        assert report.adverse_selection_detected is False

    def test_no_adverse_when_low_slippage(self):
        """All adverse but avg slippage <= 1 bps → not detected."""
        trades = [
            _make_trade(entry_fill={"slippage_bps": 0.3, "fill_latency_ms": 10.0, "fill_type": "market"}),
            _make_trade(entry_fill={"slippage_bps": 0.5, "fill_latency_ms": 10.0, "fill_type": "market"}),
            _make_trade(entry_fill={"slippage_bps": 0.8, "fill_latency_ms": 10.0, "fill_type": "market"}),
        ]
        analyzer = FillQualityAnalyzer("bot_a", "2026-03-01")
        report = analyzer.compute(trades)
        # 100% adverse but avg slippage = 0.53 < 1 bps
        assert report.adverse_selection_detected is False


# ---------------------------------------------------------------------------
# Analyzer — fill type tracking
# ---------------------------------------------------------------------------

class TestFillQualityAnalyzerFillTypes:
    def test_fill_type_frequency(self):
        trades = [
            _make_trade(entry_fill={"slippage_bps": 1.0, "fill_latency_ms": 10.0, "fill_type": "market"}),
            _make_trade(entry_fill={"slippage_bps": 0.5, "fill_latency_ms": 10.0, "fill_type": "market"}),
            _make_trade(entry_fill={"slippage_bps": -0.2, "fill_latency_ms": 10.0, "fill_type": "limit"}),
        ]
        analyzer = FillQualityAnalyzer("bot_a", "2026-03-01")
        report = analyzer.compute(trades)

        assert report.overall_entry.by_fill_type["market"] == 2
        assert report.overall_entry.by_fill_type["limit"] == 1

    def test_unknown_fill_type_default(self):
        trades = [
            _make_trade(entry_fill={"slippage_bps": 1.0, "fill_latency_ms": 10.0}),
        ]
        analyzer = FillQualityAnalyzer("bot_a", "2026-03-01")
        report = analyzer.compute(trades)

        assert report.overall_entry.by_fill_type.get("unknown", 0) == 1
