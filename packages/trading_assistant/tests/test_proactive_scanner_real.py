# tests/test_proactive_scanner_real.py
"""Tests for real proactive scanning logic."""
from trading_assistant.schemas.notifications import NotificationPriority
from trading_assistant.skills.proactive_scanner import ProactiveScanner
from tests.factories import make_trade


def _make_trade(pnl: float, date_str: str = "2026-03-01"):
    return make_trade(
        trade_id=f"t_{pnl}", pnl=pnl, pair="BTCUSDT",
        entry_price=50000,
        entry_time=f"{date_str}T10:00:00",
        exit_time=f"{date_str}T11:00:00",
    )


class TestUnusualLossDetection:
    def test_detects_unusual_loss(self):
        # Normal losses: ~$50-100
        historical = [_make_trade(-60), _make_trade(-80), _make_trade(-50),
                      _make_trade(-70), _make_trade(-90)]
        # Unusual: $500 loss (>2 sigma from mean)
        scanner = ProactiveScanner()
        result = scanner.detect_unusual_losses(
            bot_id="bot1",
            recent_trade=_make_trade(-500),
            historical_losses=[t.pnl for t in historical],
        )
        assert result is not None
        assert result.priority == NotificationPriority.HIGH

    def test_normal_loss_not_flagged(self):
        historical = [_make_trade(-60), _make_trade(-80), _make_trade(-50)]
        scanner = ProactiveScanner()
        result = scanner.detect_unusual_losses(
            bot_id="bot1",
            recent_trade=_make_trade(-75),
            historical_losses=[t.pnl for t in historical],
        )
        assert result is None


class TestRepeatedErrorDetection:
    def test_detects_repeated_errors(self):
        scanner = ProactiveScanner()
        errors = [
            {"bot_id": "bot1", "error_type": "api_timeout", "timestamp": "2026-03-01T10:00:00"},
            {"bot_id": "bot1", "error_type": "api_timeout", "timestamp": "2026-03-01T10:15:00"},
            {"bot_id": "bot1", "error_type": "api_timeout", "timestamp": "2026-03-01T10:30:00"},
            {"bot_id": "bot1", "error_type": "api_timeout", "timestamp": "2026-03-01T10:45:00"},
        ]
        result = scanner.detect_repeated_errors(errors, threshold=3)
        assert len(result) >= 1
        assert result[0].priority in (NotificationPriority.HIGH, NotificationPriority.CRITICAL)

    def test_sparse_errors_not_flagged(self):
        scanner = ProactiveScanner()
        errors = [
            {"bot_id": "bot1", "error_type": "api_timeout", "timestamp": "2026-03-01T10:00:00"},
            {"bot_id": "bot1", "error_type": "connection_reset", "timestamp": "2026-03-01T12:00:00"},
        ]
        result = scanner.detect_repeated_errors(errors, threshold=3)
        assert len(result) == 0


class TestHeartbeatMonitoring:
    def test_detects_missing_heartbeat(self):
        scanner = ProactiveScanner()
        result = scanner.check_heartbeats(
            bot_heartbeats={"bot1": "2026-03-01T08:00:00", "bot2": "2026-03-01T10:00:00"},
            current_time="2026-03-01T12:00:00",
            max_gap_hours=2,
        )
        # bot1 last seen 4 hours ago
        assert len(result) >= 1
        assert any("bot1" in r.title for r in result)

    def test_recent_heartbeat_ok(self):
        scanner = ProactiveScanner()
        result = scanner.check_heartbeats(
            bot_heartbeats={"bot1": "2026-03-01T11:30:00"},
            current_time="2026-03-01T12:00:00",
            max_gap_hours=2,
        )
        assert len(result) == 0
