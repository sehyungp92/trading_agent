"""Tests for health monitoring."""

from crypto_trader.live.health import HealthMonitor


class TestHealthMonitor:
    def test_initial_state(self):
        h = HealthMonitor()
        assert not h.is_stale()
        assert not h.should_reconnect()

    def test_stale_detection(self):
        h = HealthMonitor(expected_bar_interval_sec=0.01, stale_multiplier=2.0)
        h._last_bar_time -= 0.03
        assert h.is_stale()

    def test_bar_received_resets_stale(self):
        h = HealthMonitor(expected_bar_interval_sec=0.01, stale_multiplier=2.0)
        h._last_bar_time -= 0.03
        assert h.is_stale()
        h.on_bar_received()
        assert not h.is_stale()

    def test_consecutive_errors_reconnect(self):
        h = HealthMonitor()
        assert not h.should_reconnect()
        for _ in range(5):
            h.on_error("test")
        assert h.should_reconnect()

    def test_bar_resets_consecutive_errors(self):
        h = HealthMonitor()
        for _ in range(3):
            h.on_error("test")
        h.on_bar_received()
        assert not h.should_reconnect()

    def test_backoff_delay(self):
        h = HealthMonitor()
        assert h.get_backoff_delay() == 1.0  # base * 2^0
        h.on_error("test")
        assert h.get_backoff_delay() == 2.0  # base * 2^1
        h.on_error("test")
        assert h.get_backoff_delay() == 4.0

    def test_backoff_max(self):
        h = HealthMonitor()
        for _ in range(20):
            h.on_error("test")
        assert h.get_backoff_delay(max_delay=60.0) <= 60.0

    def test_get_status(self):
        h = HealthMonitor()
        h.on_poll()
        h.on_poll()
        status = h.get_status()
        assert status["total_polls"] == 2
        assert status["total_errors"] == 0
        assert "timestamp" in status
        assert "is_stale" in status

    def test_heartbeat_no_crash(self):
        h = HealthMonitor()
        h.heartbeat()  # should just log
