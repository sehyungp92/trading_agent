from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from strategies.swing._shared import indicators as ind
from strategies.swing.tpc.config import TPCSymbolConfig

INDICATOR_CONFIG_KEYS = (
    "ma_50_period", "ma_100_period", "rsi_period", "adx_period", "atr_period",
    "ema_20_period", "ema_50_period", "vwap_anchor_hour", "vwap_anchor_minute",
)


@dataclass(slots=True)
class TPCIndicatorArrays:
    ma50_4h: np.ndarray
    ma100_4h: np.ndarray
    rsi_4h: np.ndarray
    atr_4h: np.ndarray
    adx_4h: np.ndarray
    plus_di_4h: np.ndarray
    minus_di_4h: np.ndarray
    ema20_1h: np.ndarray
    ema50_1h: np.ndarray
    atr_1h: np.ndarray
    vwap_1h: np.ndarray
    ema20_15m: np.ndarray
    atr_15m: np.ndarray
    volume_sma_15m: np.ndarray
    vwap_15m: np.ndarray
    ema20_30m: np.ndarray
    ema50_30m: np.ndarray
    atr_30m: np.ndarray
    volume_sma_30m: np.ndarray
    vwap_30m: np.ndarray


def compute_indicators(bars_15m, bars_30m, bars_1h, bars_4h, cfg: TPCSymbolConfig) -> TPCIndicatorArrays:
    if bars_30m is None:
        nan30 = np.full(0, np.nan, dtype=float)
        ema20_30m = ema50_30m = atr_30m = volume_sma_30m = vwap_30m = nan30
    else:
        ema20_30m = ind.ema(bars_30m.closes, cfg.ema_20_period)
        ema50_30m = ind.ema(bars_30m.closes, cfg.ema_50_period)
        atr_30m = ind.atr(bars_30m.highs, bars_30m.lows, bars_30m.closes, cfg.atr_period)
        volume_sma_30m = ind.volume_sma(bars_30m.volumes, 20)
        vwap_30m = ind.vwap_anchored(
            bars_30m.highs,
            bars_30m.lows,
            bars_30m.closes,
            bars_30m.volumes,
            bars_30m.times,
            cfg.vwap_anchor_hour,
            cfg.vwap_anchor_minute,
        )
    adx_4h, plus_di_4h, minus_di_4h = ind.adx(
        bars_4h.highs,
        bars_4h.lows,
        bars_4h.closes,
        cfg.adx_period,
    )
    return TPCIndicatorArrays(
        ma50_4h=ind.sma(bars_4h.closes, cfg.ma_50_period),
        ma100_4h=ind.sma(bars_4h.closes, cfg.ma_100_period),
        rsi_4h=ind.rsi(bars_4h.closes, cfg.rsi_period),
        atr_4h=ind.atr(bars_4h.highs, bars_4h.lows, bars_4h.closes, cfg.atr_period),
        adx_4h=adx_4h,
        plus_di_4h=plus_di_4h,
        minus_di_4h=minus_di_4h,
        ema20_1h=ind.ema(bars_1h.closes, cfg.ema_20_period),
        ema50_1h=ind.ema(bars_1h.closes, cfg.ema_50_period),
        atr_1h=ind.atr(bars_1h.highs, bars_1h.lows, bars_1h.closes, cfg.atr_period),
        vwap_1h=ind.vwap_anchored(
            bars_1h.highs,
            bars_1h.lows,
            bars_1h.closes,
            bars_1h.volumes,
            bars_1h.times,
            cfg.vwap_anchor_hour,
            cfg.vwap_anchor_minute,
        ),
        ema20_15m=ind.ema(bars_15m.closes, cfg.ema_20_period),
        atr_15m=ind.atr(bars_15m.highs, bars_15m.lows, bars_15m.closes, cfg.atr_period),
        volume_sma_15m=ind.volume_sma(bars_15m.volumes, 20),
        vwap_15m=ind.vwap_anchored(
            bars_15m.highs,
            bars_15m.lows,
            bars_15m.closes,
            bars_15m.volumes,
            bars_15m.times,
            cfg.vwap_anchor_hour,
            cfg.vwap_anchor_minute,
        ),
        ema20_30m=ema20_30m,
        ema50_30m=ema50_30m,
        atr_30m=atr_30m,
        volume_sma_30m=volume_sma_30m,
        vwap_30m=vwap_30m,
    )


def snapshot(arrays: TPCIndicatorArrays, i15: int, i30: int | None, j1h: int, j4h: int) -> dict[str, float]:
    data = {
        "ma50_4h": _at(arrays.ma50_4h, j4h),
        "ma100_4h": _at(arrays.ma100_4h, j4h),
        "rsi_4h": _at(arrays.rsi_4h, j4h),
        "atr_4h": _at(arrays.atr_4h, j4h),
        "adx_4h": _at(arrays.adx_4h, j4h),
        "plus_di_4h": _at(arrays.plus_di_4h, j4h),
        "minus_di_4h": _at(arrays.minus_di_4h, j4h),
        "ema20_1h": _at(arrays.ema20_1h, j1h),
        "ema50_1h": _at(arrays.ema50_1h, j1h),
        "atr_1h": _at(arrays.atr_1h, j1h),
        "vwap_1h": _at(arrays.vwap_1h, j1h),
        "ema20_15m": _at(arrays.ema20_15m, i15),
        "atr_15m": _at(arrays.atr_15m, i15),
        "volume_sma_15m": _at(arrays.volume_sma_15m, i15),
        "vwap_15m": _at(arrays.vwap_15m, i15),
    }
    if i30 is not None:
        data.update(
            {
                "ema20_30m": _at(arrays.ema20_30m, i30),
                "ema50_30m": _at(arrays.ema50_30m, i30),
                "atr_30m": _at(arrays.atr_30m, i30),
                "volume_sma_30m": _at(arrays.volume_sma_30m, i30),
                "vwap_30m": _at(arrays.vwap_30m, i30),
            }
        )
    return data


def _at(values: np.ndarray, idx: int | None) -> float:
    if idx is None or idx < 0 or idx >= len(values):
        return float("nan")
    return float(values[idx])
