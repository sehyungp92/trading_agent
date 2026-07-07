"""Regime Classifier — deterministic market regime tagger.

Simple rules-based classifier using standard technical indicators.
Returns one of: trending_up, trending_down, ranging, volatile, unknown.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("instrumentation.regime_classifier")


@dataclass
class RegimeConfig:
    """Thresholds for regime classification."""
    ma_period: int = 50
    adx_period: int = 14
    adx_trend_threshold: float = 25.0
    adx_range_threshold: float = 20.0
    atr_period: int = 14
    atr_volatile_percentile: float = 80.0
    atr_lookback_bars: int = 100
    slope_lookback_bars: int = 5


class RegimeClassifier:
    """Deterministic market regime classifier.

    Usage::

        classifier = RegimeClassifier(config_path, data_provider)
        regime = classifier.classify("QQQ")
        # Returns: "trending_up" | "trending_down" | "ranging" | "volatile"
    """

    VALID_REGIMES = {"trending_up", "trending_down", "ranging", "volatile", "unknown"}

    def __init__(
        self,
        config_path: str = "instrumentation/config/regime_classifier_config.yaml",
        data_provider=None,
    ):
        self.data_provider = data_provider
        self.config = self._load_config(config_path)
        self._cache: dict[str, str] = {}

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
        """Classify the current market regime for a symbol."""
        try:
            candles = self._fetch_candles(symbol, timeframe)
            if not candles or len(candles) < self.config.atr_lookback_bars:
                return "unknown"

            closes = [self._get_field(c, 4, "close") for c in candles]
            highs = [self._get_field(c, 2, "high") for c in candles]
            lows = [self._get_field(c, 3, "low") for c in candles]

            # Current price vs MA
            ma = sum(closes[-self.config.ma_period:]) / self.config.ma_period
            current_price = closes[-1]
            above_ma = current_price > ma

            # MA slope
            sl = self.config.slope_lookback_bars
            ma_recent = sum(closes[-sl:]) / sl
            ma_prior = sum(closes[-(sl + 5):-5]) / sl
            ma_slope_positive = ma_recent > ma_prior

            # ADX
            adx = self._compute_adx(highs, lows, closes, self.config.adx_period)

            # ATR percentile
            atrs = self._compute_atr_series(highs, lows, closes, self.config.atr_period)
            current_atr = atrs[-1] if atrs else 0
            atr_percentile = self._percentile_rank(atrs, current_atr)

            # Classification logic
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

    @staticmethod
    def _get_field(candle, index: int, attr: str) -> float:
        """Extract field from candle — supports lists, tuples, and objects."""
        if isinstance(candle, (list, tuple)):
            return float(candle[index])
        return float(getattr(candle, attr, 0))

    def _fetch_candles(self, symbol: str, timeframe: str) -> list:
        """Fetch candles from the data provider."""
        if self.data_provider is None:
            return []
        limit = max(self.config.atr_lookback_bars, self.config.ma_period) + 20
        if hasattr(self.data_provider, "get_ohlcv"):
            return self.data_provider.get_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if isinstance(self.data_provider, dict):
            sym_data = self.data_provider.get(symbol, {})
            return sym_data.get("hourly_bars", [])
        return []

    def _compute_adx(self, highs: list, lows: list, closes: list, period: int) -> float:
        """Simplified ADX calculation."""
        if len(highs) < period + 1:
            return 0

        plus_dm = []
        minus_dm = []
        tr = []

        for i in range(1, len(highs)):
            up = highs[i] - highs[i - 1]
            down = lows[i - 1] - lows[i]
            plus_dm.append(up if up > down and up > 0 else 0)
            minus_dm.append(down if down > up and down > 0 else 0)

            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
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
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
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
