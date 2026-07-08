import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from strategies.momentum.instrumentation.src.regime_classifier import RegimeClassifier, RegimeConfig


class MockDataProvider:
    """Provides synthetic candle data for testing regime classification."""

    def __init__(self, trend="up"):
        self.trend = trend

    def get_ohlcv(self, symbol, timeframe="1h", limit=120):
        """Generate synthetic candles.

        For trending_up: steadily rising closes with moderate ranges
        For trending_down: steadily falling closes with moderate ranges
        For ranging: oscillating closes around a fixed level
        For volatile: large high-low ranges
        """
        candles = []
        base = 20000.0

        for i in range(limit):
            ts = i * 3600000

            if self.trend == "up":
                close = base + i * 10
                high = close + 20
                low = close - 15
                open_ = close - 5
            elif self.trend == "down":
                close = base - i * 10
                high = close + 15
                low = close - 20
                open_ = close + 5
            elif self.trend == "volatile":
                close = base + (i % 10) * 50 - 200
                high = close + 300
                low = close - 300
                open_ = close - 50
            else:  # ranging
                close = base + (i % 5 - 2) * 5
                high = close + 8
                low = close - 8
                open_ = close - 2

            candles.append([ts, open_, high, low, close, 1000])

        return candles


class TestRegimeConfig:
    def test_defaults(self):
        config = RegimeConfig()
        assert config.ma_period == 50
        assert config.adx_period == 14
        assert config.adx_trend_threshold == 25.0

    def test_custom_values(self):
        config = RegimeConfig(ma_period=100, adx_trend_threshold=30.0)
        assert config.ma_period == 100
        assert config.adx_trend_threshold == 30.0


class TestRegimeClassifier:
    def test_returns_valid_regime(self):
        classifier = RegimeClassifier(
            config_path="/nonexistent/path.yaml",
            data_provider=MockDataProvider("up"),
        )
        regime = classifier.classify("NQ")
        assert regime in ["trending_up", "trending_down", "ranging", "volatile", "unknown"]

    def test_uptrend_detection(self):
        classifier = RegimeClassifier(
            config_path="/nonexistent/path.yaml",
            data_provider=MockDataProvider("up"),
        )
        regime = classifier.classify("NQ")
        assert regime in ["trending_up", "volatile"]  # strong trend may show as volatile

    def test_downtrend_detection(self):
        classifier = RegimeClassifier(
            config_path="/nonexistent/path.yaml",
            data_provider=MockDataProvider("down"),
        )
        regime = classifier.classify("NQ")
        assert regime in ["trending_down", "volatile"]

    def test_unknown_on_no_data(self):
        """Should return 'unknown' when no data provider is set."""
        classifier = RegimeClassifier(
            config_path="/nonexistent/path.yaml",
            data_provider=None,
        )
        regime = classifier.classify("NQ")
        assert regime == "unknown"

    def test_unknown_on_insufficient_data(self):
        """Should return 'unknown' when not enough candles."""
        provider = MagicMock()
        provider.get_ohlcv.return_value = [[0, 100, 110, 90, 105, 1000]]
        classifier = RegimeClassifier(
            config_path="/nonexistent/path.yaml",
            data_provider=provider,
        )
        regime = classifier.classify("NQ")
        assert regime == "unknown"

    def test_cache_updated_on_classify(self):
        classifier = RegimeClassifier(
            config_path="/nonexistent/path.yaml",
            data_provider=MockDataProvider("up"),
        )
        classifier.classify("NQ")
        cached = classifier.current_regime("NQ")
        assert cached != "unknown"

    def test_cache_fallback_on_error(self):
        """On error, should return cached value, not crash."""
        classifier = RegimeClassifier(
            config_path="/nonexistent/path.yaml",
            data_provider=MockDataProvider("up"),
        )
        # First classify successfully to populate cache
        first_regime = classifier.classify("NQ")

        # Now make data provider fail
        classifier.data_provider = MagicMock()
        classifier.data_provider.get_ohlcv.side_effect = Exception("network error")

        # Should return cached value
        regime = classifier.classify("NQ")
        assert regime == first_regime

    def test_current_regime_unknown_if_not_classified(self):
        classifier = RegimeClassifier(
            config_path="/nonexistent/path.yaml",
            data_provider=None,
        )
        assert classifier.current_regime("ES") == "unknown"

    def test_config_loaded_from_yaml(self):
        tmpdir = tempfile.mkdtemp()
        config_path = Path(tmpdir) / "regime.yaml"
        config_path.write_text(
            "ma_period: 100\n"
            "adx_period: 20\n"
            "adx_trend_threshold: 30.0\n"
            "adx_range_threshold: 15.0\n"
            "atr_period: 14\n"
            "atr_volatile_percentile: 85.0\n"
            "atr_lookback_bars: 120\n"
            "slope_lookback_bars: 7\n"
        )
        classifier = RegimeClassifier(config_path=str(config_path), data_provider=None)
        assert classifier.config.ma_period == 100
        assert classifier.config.adx_trend_threshold == 30.0
        assert classifier.config.atr_volatile_percentile == 85.0
