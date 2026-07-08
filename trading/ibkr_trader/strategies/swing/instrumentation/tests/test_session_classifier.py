from datetime import datetime
from strategies.swing.instrumentation.src.session_classifier import SessionClassifier


class TestSessionClassifier:
    def test_rth_during_market_hours(self):
        dt = datetime(2026, 3, 5, 10, 30)
        result = SessionClassifier.classify(dt)
        assert result["market_session"] == "RTH"

    def test_pre_market(self):
        dt = datetime(2026, 3, 5, 8, 0)
        result = SessionClassifier.classify(dt)
        assert result["market_session"] == "PRE"

    def test_post_market(self):
        dt = datetime(2026, 3, 5, 17, 0)
        result = SessionClassifier.classify(dt)
        assert result["market_session"] == "ETH_POST"

    def test_weekend(self):
        dt = datetime(2026, 3, 7, 12, 0)
        result = SessionClassifier.classify(dt)
        assert result["market_session"] == "WEEKEND"

    def test_minutes_into_rth(self):
        dt = datetime(2026, 3, 5, 10, 30)
        result = SessionClassifier.classify(dt)
        assert result["minutes_into_session"] == 60

    def test_minutes_into_pre(self):
        dt = datetime(2026, 3, 5, 5, 30)
        result = SessionClassifier.classify(dt)
        assert result["minutes_into_session"] == 90
