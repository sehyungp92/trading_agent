"""Tests for the structured health report."""

import json
import time

from click.testing import CliRunner

from crypto_trader.cli import cli
from crypto_trader.live.health_report import HealthAlert, HealthReport, HealthReportBuilder


class TestHealthReport:
    def test_to_text_healthy(self):
        r = HealthReport(
            timestamp="2026-01-01T00:00:00Z",
            uptime_sec=3600.0,
            assessment="healthy",
        )
        text = r.to_text()
        assert "HEALTHY" in text
        assert "3600s" in text

    def test_to_text_with_alerts(self):
        r = HealthReport(
            timestamp="2026-01-01T00:00:00Z",
            uptime_sec=100.0,
            alerts=[{"severity": "error", "name": "no_bars", "message": "BTC/M15 stale"}],
            assessment="degraded",
        )
        text = r.to_text()
        assert "DEGRADED" in text
        assert "no_bars" in text

    def test_to_dict_roundtrip(self):
        r = HealthReport(
            timestamp="2026-01-01T00:00:00Z",
            uptime_sec=100.0,
            assessment="healthy",
        )
        d = r.to_dict()
        assert d["timestamp"] == "2026-01-01T00:00:00Z"
        assert d["assessment"] == "healthy"


class TestHealthReportBuilder:
    def _make_builder(self) -> HealthReportBuilder:
        return HealthReportBuilder()

    def test_healthy_report(self):
        builder = self._make_builder()
        now = time.monotonic()
        report = builder.build(
            uptime_sec=3600.0,
            health_status={"total_errors": 0, "consecutive_errors": 0},
            stale_feeds=[],
            funnels={},
            positions=[],
            portfolio_state={"heat_R": 0, "heat_cap_R": 5},
            tf_last_bar={},
            now_mono=now,
        )
        assert report.assessment == "healthy"
        assert len(report.alerts) == 0

    def test_degraded_with_stale_feeds(self):
        builder = self._make_builder()
        now = time.monotonic()
        report = builder.build(
            uptime_sec=3600.0,
            health_status={"total_errors": 0, "consecutive_errors": 0},
            stale_feeds=[("BTC", "15m", 2000.0)],
            funnels={},
            positions=[],
            portfolio_state={},
            tf_last_bar={("BTC", "15m"): now - 2000},
            now_mono=now,
        )
        assert report.assessment == "degraded"
        assert len(report.alerts) == 1
        assert report.alerts[0]["name"] == "no_bars"

    def test_data_flow_status(self):
        builder = self._make_builder()
        now = time.monotonic()
        report = builder.build(
            uptime_sec=100.0,
            health_status={"total_errors": 0, "consecutive_errors": 0},
            stale_feeds=[],
            funnels={},
            positions=[],
            portfolio_state={},
            tf_last_bar={
                ("BTC", "15m"): now - 30,   # OK (< 900*2)
                ("ETH", "1h"): now - 8000,   # STALE (> 3600*2)
            },
            now_mono=now,
        )
        assert report.data_flow["BTC/15m"]["status"] == "OK"
        assert report.data_flow["ETH/1h"]["status"] == "STALE"

    def test_error_burst_alert(self):
        builder = self._make_builder()
        now = time.monotonic()
        report = builder.build(
            uptime_sec=100.0,
            health_status={"total_errors": 20, "consecutive_errors": 15},
            stale_feeds=[],
            funnels={},
            positions=[],
            portfolio_state={},
            tf_last_bar={},
            now_mono=now,
        )
        alert_names = [a["name"] for a in report.alerts]
        assert "error_burst" in alert_names

    def test_pipeline_stalled_alert(self):
        builder = self._make_builder()
        now = time.monotonic()
        report = builder.build(
            uptime_sec=100.0,
            health_status={"total_errors": 0, "consecutive_errors": 0},
            stale_feeds=[],
            funnels={"momentum": {"bars_received": {}, "indicators_ready": {}}},
            positions=[],
            portfolio_state={},
            tf_last_bar={},
            now_mono=now,
        )
        alert_names = [a["name"] for a in report.alerts]
        assert "pipeline_stalled" in alert_names

    def test_missing_cached_funnel_does_not_alert_as_stalled(self):
        builder = self._make_builder()
        now = time.monotonic()
        report = builder.build(
            uptime_sec=100.0,
            health_status={"total_errors": 0, "consecutive_errors": 0},
            stale_feeds=[],
            funnels={"momentum": {}},
            positions=[],
            portfolio_state={},
            tf_last_bar={},
            now_mono=now,
        )

        assert report.signal_funnels == {}
        assert [a["name"] for a in report.alerts] == []

    def test_funnel_summary(self):
        builder = self._make_builder()
        now = time.monotonic()
        report = builder.build(
            uptime_sec=100.0,
            health_status={"total_errors": 0, "consecutive_errors": 0},
            stale_feeds=[],
            funnels={
                "momentum": {
                    "bars_received": {"BTC": 10, "ETH": 5},
                    "indicators_ready": {"BTC": 10, "ETH": 5},
                    "setups_detected": {"BTC": 2},
                    "confirmations": {"BTC": 1},
                    "entries_attempted": {"BTC": 1},
                    "fills": {"BTC": 1},
                    "gate_rejections": {},
                },
            },
            positions=[],
            portfolio_state={},
            tf_last_bar={},
            now_mono=now,
        )
        assert report.signal_funnels["momentum"]["bars_received"] == 15
        assert report.signal_funnels["momentum"]["entries_attempted"] == 1
        assert (
            report.to_dict()["assistant_strategy_ids"]["momentum"]
            == "MomentumPullback_M15"
        )

    def test_portfolio_info(self):
        builder = self._make_builder()
        now = time.monotonic()
        report = builder.build(
            uptime_sec=100.0,
            health_status={"total_errors": 0, "consecutive_errors": 0},
            stale_feeds=[],
            funnels={},
            positions=[],
            portfolio_state={"heat_R": 1.5, "heat_cap_R": 5.0, "daily_pnl_R": 0.3, "open_risk_count": 2},
            tf_last_bar={},
            now_mono=now,
        )
        assert report.portfolio["heat_R"] == 1.5
        assert report.portfolio["open_risk_count"] == 2


class TestStatusCli:
    def test_status_renders_scalar_daily_pnl_from_jsonl(self, tmp_path):
        health_report = {
            "timestamp": "2026-04-25T12:00:00+00:00",
            "uptime_sec": 3600.0,
            "data_flow": {},
            "signal_funnels": {},
            "gate_breakdown": {},
            "positions": [],
            "portfolio": {
                "heat_R": 1.5,
                "heat_cap_R": 5.0,
                "daily_pnl_R": 0.3,
                "open_risk_count": 2,
            },
            "system": {"total_errors": 0, "stale_feed_count": 0},
            "alerts": [],
            "assessment": "healthy",
        }
        (tmp_path / "health_reports.jsonl").write_text(
            json.dumps({"timestamp": health_report["timestamp"], "report": health_report}) + "\n",
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--state-dir", str(tmp_path)])

        assert result.exit_code == 0
        assert "Daily P&L: +0.3R" in result.output
