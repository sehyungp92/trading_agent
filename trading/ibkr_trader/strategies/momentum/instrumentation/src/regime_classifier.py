import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("instrumentation.regime_classifier")


@dataclass
class RegimeConfig:
    """Thresholds for regime classification. Loaded from YAML."""
    ma_period: int = 50
    adx_period: int = 14
    adx_trend_threshold: float = 25.0
    adx_range_threshold: float = 20.0
    atr_period: int = 14
    atr_volatile_percentile: float = 80.0
    atr_lookback_bars: int = 100
    slope_lookback_bars: int = 5


class RegimeClassifier:
    """
    Deterministic market regime classifier.

    Usage:
        classifier = RegimeClassifier(config_path, data_provider)
        regime = classifier.classify("NQ")
        # Returns: "trending_up" | "trending_down" | "ranging" | "volatile"
    """

    def __init__(self, config_path: str = "instrumentation/config/regime_classifier_config.yaml",
                 data_provider=None):
        """
        Args:
            config_path: path to regime classifier config
            data_provider: object with get_ohlcv(symbol, timeframe, limit) method
                returning [[ts, o, h, l, c, v], ...]
        """
        self.data_provider = data_provider
        self.config = self._load_config(config_path)
        self._cache = {}

    def _load_config(self, path: str) -> RegimeConfig:
        config_file = Path(path)
        if config_file.exists():
            try:
                import yaml
                with open(config_file) as f:
                    raw = yaml.safe_load(f) or {}
                return RegimeConfig(**{k: v for k, v in raw.items() if k in RegimeConfig.__dataclass_fields__})
            except Exception as e:
                logger.warning("Failed to load regime config: %s", e)
        return RegimeConfig()

    def classify(self, symbol: str, timeframe: str = "1h") -> str:
        """
        Classify the current market regime for a symbol.

        Returns one of: "trending_up", "trending_down", "ranging", "volatile"
        Falls back to "unknown" if insufficient data.
        """
        try:
            candles = self._fetch_candles(symbol, timeframe)
            if not candles or len(candles) < self.config.atr_lookback_bars:
                return "unknown"

            closes = [c[4] for c in candles]
            highs = [c[2] for c in candles]
            lows = [c[3] for c in candles]

            ma = sum(closes[-self.config.ma_period:]) / self.config.ma_period
            current_price = closes[-1]
            above_ma = current_price > ma

            ma_recent = sum(closes[-self.config.slope_lookback_bars:]) / self.config.slope_lookback_bars
            offset = self.config.slope_lookback_bars + 5
            if len(closes) >= offset:
                ma_prior = sum(closes[-offset:-5]) / self.config.slope_lookback_bars
            else:
                ma_prior = ma_recent
            ma_slope_positive = ma_recent > ma_prior

            adx = self._compute_adx(highs, lows, closes, self.config.adx_period)

            atrs = self._compute_atr_series(highs, lows, closes, self.config.atr_period)
            current_atr = atrs[-1] if atrs else 0
            atr_percentile = self._percentile_rank(atrs, current_atr)

            if atr_percentile >= self.config.atr_volatile_percentile:
                regime = "volatile"
            elif adx >= self.config.adx_trend_threshold:
                if above_ma and ma_slope_positive:
                    regime = "trending_up"
                elif not above_ma and not ma_slope_positive:
                    regime = "trending_down"
                else:
                    regime = "ranging"
            elif adx < self.config.adx_range_threshold:
                regime = "ranging"
            else:
                if above_ma and ma_slope_positive:
                    regime = "trending_up"
                elif not above_ma and not ma_slope_positive:
                    regime = "trending_down"
                else:
                    regime = "ranging"

            self._cache[symbol] = regime
            return regime

        except Exception:
            return self._cache.get(symbol, "unknown")

    def current_regime(self, symbol: str) -> str:
        """Get the most recently computed regime (cached)."""
        return self._cache.get(symbol, "unknown")

    def _fetch_candles(self, symbol: str, timeframe: str) -> list:
        """Fetch candles from data provider."""
        if self.data_provider is None:
            return []
        limit = max(self.config.atr_lookback_bars, self.config.ma_period) + 20
        return self.data_provider.get_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def _compute_adx(self, highs: list, lows: list, closes: list, period: int) -> float:
        """Simplified ADX calculation."""
        if len(highs) < period + 1:
            return 0

        plus_dm = []
        minus_dm = []
        tr = []

        for i in range(1, len(highs)):
            up = highs[i] - highs[i-1]
            down = lows[i-1] - lows[i]
            plus_dm.append(up if up > down and up > 0 else 0)
            minus_dm.append(down if down > up and down > 0 else 0)
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr.append(max(hl, hc, lc))

        def wilder_smooth(data, period):
            if len(data) < period:
                return [0]
            result = [sum(data[:period]) / period]
            for i in range(period, len(data)):
                result.append((result[-1] * (period - 1) + data[i]) / period)
            return result

        smoothed_tr = wilder_smooth(tr, period)
        smoothed_plus = wilder_smooth(plus_dm, period)
        smoothed_minus = wilder_smooth(minus_dm, period)

        if not smoothed_tr or smoothed_tr[-1] == 0:
            return 0

        plus_di = smoothed_plus[-1] / smoothed_tr[-1] * 100
        minus_di = smoothed_minus[-1] / smoothed_tr[-1] * 100

        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 0

        dx = abs(plus_di - minus_di) / di_sum * 100
        return dx

    def _compute_atr_series(self, highs: list, lows: list, closes: list, period: int) -> list:
        if len(highs) < period + 1:
            return []
        trs = []
        for i in range(1, len(highs)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            trs.append(max(hl, hc, lc))

        atrs = []
        if len(trs) >= period:
            atr = sum(trs[:period]) / period
            atrs.append(atr)
            for i in range(period, len(trs)):
                atr = (atr * (period - 1) + trs[i]) / period
                atrs.append(atr)
        return atrs

    def _percentile_rank(self, series: list, value: float) -> float:
        if not series:
            return 50.0
        lookback = series[-self.config.atr_lookback_bars:]
        count_below = sum(1 for v in lookback if v < value)
        return (count_below / len(lookback)) * 100
