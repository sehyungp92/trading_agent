from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, pstdev

from strategies.momentum.nq_regime.core.state import BarData


@dataclass(frozen=True, slots=True)
class IndicatorSnapshot:
    vwap: float = 0.0
    vwap_sd: float = 0.0
    vwap_slope: float = 0.0
    atr_15m: float = 0.0
    atr_5m: float = 0.0
    ema9_15m: float = 0.0
    ema20_15m: float = 0.0
    ema50_15m: float = 0.0
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    kc_upper: float = 0.0
    kc_lower: float = 0.0
    squeeze_on: bool = False
    squeeze_duration: int = 0
    rsi14_15m: float = 50.0
    macd_15m: float = 0.0
    macd_signal_15m: float = 0.0
    volume_multiple_15m: float = 1.0
    volume_multiple_5m: float = 1.0
    am_vwap_control: float = 0.0
    trend_direction: int = 0


def build_indicator_snapshot(
    bars_5m: list[BarData],
    bars_15m: list[BarData],
    prior_snapshot: IndicatorSnapshot | None = None,
) -> IndicatorSnapshot:
    if not bars_5m:
        return IndicatorSnapshot()
    closes_5m = [bar.close for bar in bars_5m]
    volumes_5m = [max(bar.volume, 0.0) for bar in bars_5m]
    typical_values = [(bar.high + bar.low + bar.close) / 3.0 for bar in bars_5m]
    total_volume = sum(volumes_5m)
    vwap = (
        sum(value * volume for value, volume in zip(typical_values, volumes_5m)) / total_volume
        if total_volume > 0
        else fmean(closes_5m)
    )
    recent_vwap = prior_snapshot.vwap if prior_snapshot else vwap
    vwap_slope = vwap - recent_vwap
    residuals = [close - vwap for close in closes_5m[-40:]]
    vwap_sd = pstdev(residuals) if len(residuals) >= 2 else 0.0

    closes_15m = [bar.close for bar in bars_15m] or [bars_5m[-1].close]
    highs_15m = [bar.high for bar in bars_15m] or [bars_5m[-1].high]
    lows_15m = [bar.low for bar in bars_15m] or [bars_5m[-1].low]
    vols_15m = [max(bar.volume, 0.0) for bar in bars_15m] or [bars_5m[-1].volume]

    atr_15m = _atr(highs_15m, lows_15m, closes_15m, 14)
    atr_5m = _atr([b.high for b in bars_5m], [b.low for b in bars_5m], closes_5m, 14)
    ema9 = _ema(closes_15m, 9)
    ema20 = _ema(closes_15m, 20)
    ema50 = _ema(closes_15m, 50)
    bb_mid = fmean(closes_15m[-20:]) if len(closes_15m) >= 1 else closes_15m[-1]
    bb_sd = pstdev(closes_15m[-20:]) if len(closes_15m) >= 20 else 0.0
    bb_upper = bb_mid + 2.0 * bb_sd
    bb_lower = bb_mid - 2.0 * bb_sd
    kc_mid = ema20
    kc_upper = kc_mid + 1.5 * atr_15m
    kc_lower = kc_mid - 1.5 * atr_15m
    squeeze_on = bb_upper < kc_upper and bb_lower > kc_lower if atr_15m > 0 else False
    squeeze_duration = _squeeze_duration(closes_15m, highs_15m, lows_15m)
    rsi = _rsi(closes_15m, 14)
    macd, macd_signal = _macd(closes_15m)
    avg_vol_15m = fmean(vols_15m[-20:]) if vols_15m else 0.0
    volume_multiple_15m = vols_15m[-1] / avg_vol_15m if avg_vol_15m > 0 else 1.0
    avg_vol_5m = fmean(volumes_5m[-20:]) if volumes_5m else 0.0
    volume_multiple_5m = volumes_5m[-1] / avg_vol_5m if avg_vol_5m > 0 else 1.0
    am_sample = bars_5m[:24] if len(bars_5m) > 24 else bars_5m
    am_vwap_control = 0.0
    if am_sample:
        above = sum(1 for bar in am_sample if bar.close >= vwap)
        am_vwap_control = (above / len(am_sample)) - 0.5
    trend_direction = 1 if ema9 > ema20 > ema50 and closes_15m[-1] > vwap else -1 if ema9 < ema20 < ema50 and closes_15m[-1] < vwap else 0
    return IndicatorSnapshot(
        vwap=vwap,
        vwap_sd=vwap_sd,
        vwap_slope=vwap_slope,
        atr_15m=atr_15m,
        atr_5m=atr_5m,
        ema9_15m=ema9,
        ema20_15m=ema20,
        ema50_15m=ema50,
        bb_upper=bb_upper,
        bb_lower=bb_lower,
        kc_upper=kc_upper,
        kc_lower=kc_lower,
        squeeze_on=squeeze_on,
        squeeze_duration=squeeze_duration,
        rsi14_15m=rsi,
        macd_15m=macd,
        macd_signal_15m=macd_signal,
        volume_multiple_15m=volume_multiple_15m,
        volume_multiple_5m=volume_multiple_5m,
        am_vwap_control=am_vwap_control,
        trend_direction=trend_direction,
    )


def _ema(values: list[float], length: int) -> float:
    if not values:
        return 0.0
    alpha = 2.0 / (length + 1.0)
    ema = values[0]
    for value in values[1:]:
        ema = alpha * value + (1.0 - alpha) * ema
    return ema


def _atr(highs: list[float], lows: list[float], closes: list[float], length: int) -> float:
    if not highs or len(highs) != len(lows):
        return 0.0
    trs: list[float] = []
    prior_close = closes[0] if closes else (highs[0] + lows[0]) / 2.0
    for high, low, close in zip(highs, lows, closes):
        trs.append(max(high - low, abs(high - prior_close), abs(low - prior_close)))
        prior_close = close
    sample = trs[-length:]
    return fmean(sample) if sample else 0.0


def _rsi(values: list[float], length: int) -> float:
    if len(values) <= length:
        return 50.0
    gains: list[float] = []
    losses: list[float] = []
    for prev, curr in zip(values[-(length + 1):-1], values[-length:]):
        change = curr - prev
        if change >= 0:
            gains.append(change)
        else:
            losses.append(abs(change))
    avg_gain = fmean(gains) if gains else 0.0
    avg_loss = fmean(losses) if losses else 0.0
    if avg_loss <= 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    macd_line = _ema(values, 12) - _ema(values, 26)
    macd_series: list[float] = []
    for idx in range(1, len(values) + 1):
        sample = values[:idx]
        macd_series.append(_ema(sample, 12) - _ema(sample, 26))
    return macd_line, _ema(macd_series, 9)


def _squeeze_duration(closes: list[float], highs: list[float], lows: list[float]) -> int:
    duration = 0
    for idx in range(len(closes), 0, -1):
        sample = closes[:idx]
        if len(sample) < 20:
            break
        mid = fmean(sample[-20:])
        sd = pstdev(sample[-20:])
        atr = _atr(highs[:idx], lows[:idx], closes[:idx], 14)
        if mid + 2.0 * sd < _ema(sample, 20) + 1.5 * atr and mid - 2.0 * sd > _ema(sample, 20) - 1.5 * atr:
            duration += 1
        else:
            break
    return duration

