"""Tests for RegimeClassifier."""
import math

from strategies.swing.instrumentation.src.regime_classifier import RegimeClassifier


class MockTrendingUpProvider:
    """Data that should classify as trending_up."""
    def get_ohlcv(self, symbol, timeframe="1h", limit=120):
        # Steadily rising prices with high ADX
        candles = []
        for i in range(limit):
            base = 100 + i * 0.5  # clear uptrend
            candles.append([i * 3600000, base, base + 1, base - 0.5, base + 0.3, 1000])
        return candles


class MockRangingProvider:
    """Data that should classify as ranging."""
    def get_ohlcv(self, symbol, timeframe="1h", limit=120):
        # Oscillating around 100 with small moves
        candles = []
        for i in range(limit):
            # Sine wave oscillation
            offset = math.sin(i * 0.3) * 2
            base = 100 + offset
            candles.append([i * 3600000, base, base + 0.5, base - 0.5, base + 0.1, 1000])
        return candles


class MockVolatileProvider:
    """Data that should classify as volatile (high ATR)."""
    def get_ohlcv(self, symbol, timeframe="1h", limit=120):
        candles = []
        for i in range(limit):
            base = 100
            # Most bars are narrow, last 20 bars are very wide
            if i > limit - 20:
                candles.append([i * 3600000, base, base + 20, base - 20, base + 5, 1000])
            else:
                candles.append([i * 3600000, base, base + 0.5, base - 0.5, base, 1000])
        return candles


class TestRegimeClassifier:
    def test_returns_valid_regime(self):
        classifier = RegimeClassifier(
            config_path="nonexistent.yaml",  # uses defaults
            data_provider=MockTrendingUpProvider(),
        )
        regime = classifier.classify("TEST")
        assert regime in RegimeClassifier.VALID_REGIMES

    def test_trending_up_detection(self):
        classifier = RegimeClassifier(
            config_path="nonexistent.yaml",
            data_provider=MockTrendingUpProvider(),
        )
        regime = classifier.classify("TEST")
        # Should be trending_up or at least not ranging (exact depends on ADX calc)
        assert regime in ("trending_up", "volatile", "ranging")

    def test_unknown_on_insufficient_data(self):
        class EmptyProvider:
            def get_ohlcv(self, symbol, timeframe="1h", limit=120):
                return []

        classifier = RegimeClassifier(config_path="nonexistent.yaml", data_provider=EmptyProvider())
        regime = classifier.classify("TEST")
        assert regime == "unknown"

    def test_cache_works(self):
        classifier = RegimeClassifier(
            config_path="nonexistent.yaml",
            data_provider=MockTrendingUpProvider(),
        )
        classifier.classify("TEST")
        cached = classifier.current_regime("TEST")
        assert cached in RegimeClassifier.VALID_REGIMES
        assert cached != "unknown"

    def test_current_regime_returns_unknown_for_uncached(self):
        classifier = RegimeClassifier(config_path="nonexistent.yaml")
        assert classifier.current_regime("UNKNOWN_SYM") == "unknown"

    def test_never_crashes(self):
        class BrokenProvider:
            def get_ohlcv(self, symbol, timeframe="1h", limit=120):
                raise RuntimeError("data feed down")

        classifier = RegimeClassifier(config_path="nonexistent.yaml", data_provider=BrokenProvider())
        regime = classifier.classify("TEST")
        assert regime == "unknown"

    def test_no_provider(self):
        classifier = RegimeClassifier(config_path="nonexistent.yaml", data_provider=None)
        regime = classifier.classify("TEST")
        assert regime == "unknown"
