"""Empirical hidden-alpha tests for TPC structural variants.

This is a diagnostic harness, not an optimizer. It tests four hypotheses:

1. 4h trend + 30m controlled pullback + 15m confirmation.
2. Current 4h/1h/15m stack with daily alignment used as a regime permission.
3. Value-touch entries instead of confirmation-close entries.
4. Confirmation as a filter that permits a retest/limit entry.

The train/holdout split defaults to 2025-11-01, matching recent TPC work.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.swing.analysis.tpc_alpha_audit import summarise_trade_items, trade_to_row
from backtests.swing.auto.tpc.plugin import _extract_tpc_metrics
from backtests.swing.auto.tpc.round5_oos_repair import infer_holdout_warmup
from backtests.swing.config_tpc import TPCBacktestConfig
from backtests.swing.data.cache import load_bars
from backtests.swing.data.multitimeframe import (
    resample_15m_to_30m,
    resample_1h_to_4h,
)
from backtests.swing.data.preprocessing import build_numpy_arrays, normalize_timezone
from backtests.swing.engine.tpc_engine import run_tpc_independent
from strategies.swing._shared import indicators as ind
from strategies.swing._shared.models import Direction
from strategies.swing._shared.session import is_in_session_window
from strategies.swing.tpc import gates
from strategies.swing.tpc.config import TPCSymbolConfig
from strategies.swing.tpc.models import RegimeGrade

DATA_DIR = ROOT / "backtests" / "swing" / "data" / "raw"
ROUND7_CONFIG = ROOT / "backtests" / "output" / "swing" / "tpc" / "round_7" / "optimized_config.json"
DEFAULT_OUTPUT_ROOT = ROOT / "backtests" / "output" / "swing" / "tpc"
SYMBOLS = ("QQQ", "GLD")
CONTEXT_SYMBOLS = {"QQQ": "NQ", "GLD": "GC"}
INITIAL_EQUITY = 100_000.0
ET_TZ = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class FullReplaySpec:
    option: str
    name: str
    data_mode: str
    mutations: dict[str, Any]
    daily_filter: str = ""
    note: str = ""


@dataclass(frozen=True)
class EventSpec:
    option: str
    name: str
    stack: str
    entry_model: str
    params: dict[str, Any]
    daily_filter: str = ""
    note: str = ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROUND7_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-end", default="2025-11-01")
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--skip-events", action="store_true")
    args = parser.parse_args()

    started = time.time()
    output_dir = args.output_dir or (
        DEFAULT_OUTPUT_ROOT
        / f"hidden_alpha_variants_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    mutations = read_json(args.config)
    train_end = coerce_utc(args.train_end)
    print(f"[hidden-alpha] output={output_dir.resolve()}", flush=True)
    print(f"[hidden-alpha] train_end={train_end.isoformat()}", flush=True)

    full_rows: list[dict[str, Any]] = []
    if not args.skip_full:
        full_rows = run_full_replay_suite(mutations, train_end)
        write_csv(output_dir / "full_replay_results.csv", full_rows)
        write_json(output_dir / "full_replay_results.json", full_rows)

    event_rows: list[dict[str, Any]] = []
    if not args.skip_events:
        cfgs = TPCBacktestConfig(initial_equity=INITIAL_EQUITY, data_dir=DATA_DIR).with_overrides(mutations).symbol_configs
        event_rows = run_event_suite(cfgs, train_end)
        write_csv(output_dir / "event_study_results.csv", event_rows)
        write_json(output_dir / "event_study_results.json", event_rows)

    report = format_report(full_rows, event_rows, elapsed_seconds=time.time() - started)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report, flush=True)
    print(f"[hidden-alpha] completed in {(time.time() - started) / 60.0:.1f} min", flush=True)


# ---------------------------------------------------------------------------
# Full replay suite
# ---------------------------------------------------------------------------


def run_full_replay_suite(mutations: dict[str, Any], train_end: pd.Timestamp) -> list[dict[str, Any]]:
    specs = build_full_replay_specs(mutations)
    data_cache: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    indicator_cache_train_by_mode: dict[str, dict[Any, Any]] = {}
    indicator_cache_oos_by_mode: dict[str, dict[Any, Any]] = {}

    for spec in specs:
        print(f"[hidden-alpha] full replay {spec.option}/{spec.name}", flush=True)
        train_data = data_cache.setdefault(
            (spec.data_mode, "train"),
            build_replay_data(spec.data_mode, end_date=train_end),
        )
        full_data = data_cache.setdefault(
            (spec.data_mode, "full"),
            build_replay_data(spec.data_mode, end_date=None),
        )
        warmup = infer_holdout_warmup(full_data, train_end.isoformat())
        with daily_regime_filter(spec.daily_filter):
            train_cfg = make_config(spec.mutations)
            oos_cfg = make_config({**spec.mutations, "warmup_15m": warmup})
            train_result = run_tpc_independent(
                train_data,
                train_cfg,
                indicator_cache=indicator_cache_train_by_mode.setdefault(spec.data_mode, {}),
            )
            oos_result = run_tpc_independent(
                full_data,
                oos_cfg,
                indicator_cache=indicator_cache_oos_by_mode.setdefault(spec.data_mode, {}),
            )
        rows.extend(flatten_full_summary(spec, "train", summarize_backtest(train_result)))
        rows.extend(flatten_full_summary(spec, "oos", summarize_backtest(oos_result)))
    return rows


def build_full_replay_specs(base: dict[str, Any]) -> list[FullReplaySpec]:
    def merge(extra: dict[str, Any]) -> dict[str, Any]:
        out = dict(base)
        out.update(extra)
        return out

    return [
        FullReplaySpec("baseline", "current_round7", "current", dict(base), note="Current 4h/1h/15m round-7 config."),
        FullReplaySpec("option1", "pb30_same_params", "pb30", dict(base), note="4h trend, 30m pullback, original durations."),
        FullReplaySpec(
            "option1",
            "pb30_scaled_duration",
            "pb30",
            merge({"all.pullback_min_bars_1h": 6, "all.pullback_max_bars_1h": 20}),
            note="30m pullback with duration scaled to roughly current 1h span.",
        ),
        FullReplaySpec(
            "option1",
            "pb30_quality",
            "pb30",
            merge(
                {
                    "all.pullback_min_bars_1h": 6,
                    "all.pullback_max_bars_1h": 20,
                    "all.pullback_orderly_required": True,
                    "all.type_a_value_hits_min": 2,
                }
            ),
            note="30m pullback with control/value quality filter.",
        ),
        FullReplaySpec(
            "option1",
            "pb30_loose_confirm",
            "pb30",
            merge(
                {
                    "all.pullback_min_bars_1h": 4,
                    "all.pullback_max_bars_1h": 16,
                    "all.confirmation_required": 1,
                    "QQQ.confirmation_required": 1,
                    "GLD.confirmation_required": 1,
                }
            ),
            note="30m pullback with shorter duration and single confirmation.",
        ),
        FullReplaySpec("option2", "daily_sma20", "current", dict(base), daily_filter="sma20"),
        FullReplaySpec("option2", "daily_sma50", "current", dict(base), daily_filter="sma50"),
        FullReplaySpec("option2", "daily_sma20_50", "current", dict(base), daily_filter="sma20_50"),
        FullReplaySpec("option2", "daily_ret20_sma20", "current", dict(base), daily_filter="ret20_sma20"),
    ]


def build_replay_data(data_mode: str, *, end_date: pd.Timestamp | None) -> dict[str, dict[str, Any]]:
    end_ts = coerce_utc(end_date, end_of_day=True) if end_date is not None else None
    data: dict[str, dict[str, Any]] = {}
    for symbol in SYMBOLS:
        df15 = slice_timestamp_index(normalize_timezone(load_bars(DATA_DIR / f"{symbol}_15m.parquet")), None, end_ts)
        df1h = slice_timestamp_index(normalize_timezone(load_bars(DATA_DIR / f"{symbol}_1h.parquet")), None, end_ts)
        dfd = slice_timestamp_index(normalize_timezone(load_bars(DATA_DIR / f"{symbol}_1d.parquet")), None, end_ts)
        df30 = resample_15m_to_30m(df15)
        df4h = resample_1h_to_4h(df1h)

        if data_mode == "current":
            pullback_df = df1h
            idx_1h = align_completed(df15.index, df1h.index, "15min", "1h")
        elif data_mode == "pb30":
            pullback_df = df30
            idx_1h = align_completed(df15.index, df30.index, "15min", "30min")
        else:
            raise ValueError(f"unknown data mode {data_mode!r}")

        data[symbol] = {
            "bars_15m": build_numpy_arrays(df15),
            "bars_30m": build_numpy_arrays(df30),
            "bars_1h": build_numpy_arrays(pullback_df),
            "bars_4h": build_numpy_arrays(df4h),
            "bars_daily": build_numpy_arrays(dfd),
            "idx_30m": align_completed(df15.index, df30.index, "15min", "30min"),
            "idx_1h": idx_1h,
            "idx_4h": align_completed(df15.index, df4h.index, "15min", "4h"),
            "idx_daily": align_daily_previous_session(df15.index, dfd.index),
            "context_symbol": CONTEXT_SYMBOLS.get(symbol, ""),
            "context_indicators": build_context_arrays(symbol, df15, end_ts=end_ts),
        }
    return data


def make_config(mutations: dict[str, Any]) -> TPCBacktestConfig:
    return TPCBacktestConfig(
        initial_equity=INITIAL_EQUITY,
        data_dir=DATA_DIR,
        symbols=SYMBOLS,
    ).with_overrides(mutations)


@contextmanager
def daily_regime_filter(mode: str):
    if not mode:
        yield
        return
    original = gates.regime_direction

    def wrapped(bar_input: Any, cfg: TPCSymbolConfig) -> tuple[Direction, RegimeGrade, str]:
        direction, grade, reason = original(bar_input, cfg)
        if direction == Direction.FLAT:
            return direction, grade, reason
        if not daily_alignment_ok(bar_input.bars_daily, direction, mode):
            return Direction.FLAT, RegimeGrade.INVALID, f"daily_filter_{mode}"
        return direction, grade, reason

    gates.regime_direction = wrapped
    try:
        yield
    finally:
        gates.regime_direction = original


def daily_alignment_ok(bars_daily: Any, direction: Direction, mode: str) -> bool:
    if bars_daily is None or len(bars_daily.closes) < 60:
        return False
    closes = np.asarray(bars_daily.closes, dtype=float)
    close = float(closes[-1])
    sma20 = float(np.nanmean(closes[-20:]))
    sma50 = float(np.nanmean(closes[-50:]))
    ret20 = close / max(float(closes[-21]), 1e-9) - 1.0 if len(closes) >= 21 else 0.0
    sma50_prev = float(np.nanmean(closes[-55:-5])) if len(closes) >= 55 else sma50
    if not np.isfinite([close, sma20, sma50, ret20, sma50_prev]).all():
        return False
    if mode == "sma20":
        return close >= sma20 if direction == Direction.LONG else close <= sma20
    if mode == "sma50":
        if direction == Direction.LONG:
            return close >= sma50 and sma50 >= sma50_prev
        return close <= sma50 and sma50 <= sma50_prev
    if mode == "sma20_50":
        return close >= sma20 >= sma50 if direction == Direction.LONG else close <= sma20 <= sma50
    if mode == "ret20_sma20":
        if direction == Direction.LONG:
            return close >= sma20 and ret20 >= 0.0
        return close <= sma20 and ret20 <= 0.0
    raise ValueError(f"unknown daily filter {mode!r}")


def summarize_backtest(result: Any) -> dict[str, Any]:
    metrics = dict(_extract_tpc_metrics(result, INITIAL_EQUITY))
    trade_rows = [trade_to_row("all", trade) for trade in result.trades]
    symbols: dict[str, dict[str, Any]] = {}
    for symbol in SYMBOLS:
        items = [row for row in trade_rows if row.get("symbol") == symbol]
        symbols[symbol] = summarise_trade_items(items) if items else empty_trade_summary()
    return {"headline": metrics, "symbols": symbols}


def flatten_full_summary(spec: FullReplaySpec, split: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    head = summary["headline"]
    row = {
        "option": spec.option,
        "name": spec.name,
        "split": split,
        "data_mode": spec.data_mode,
        "daily_filter": spec.daily_filter,
        "note": spec.note,
        "total_trades": head.get("total_trades", 0.0),
        "net_return_pct": head.get("net_return_pct", 0.0),
        "avg_r": head.get("avg_r", 0.0),
        "win_rate": head.get("win_rate", 0.0),
        "dollar_profit_factor": head.get("dollar_profit_factor", head.get("profit_factor", 0.0)),
        "max_dd_pct": head.get("max_dd_pct", 0.0),
        "avg_mfe_r": head.get("avg_mfe_r", 0.0),
        "excellent_trades": head.get("excellent_trades", 0.0),
        "excellent_rate": head.get("excellent_rate", 0.0),
        "low_mfe_loss_rate": head.get("low_mfe_loss_rate", 0.0),
        "right_then_lost_rate": head.get("right_then_lost_rate", 0.0),
        "top5_winner_share": head.get("top5_winner_share", 0.0),
    }
    for symbol, prefix in (("QQQ", "qqq"), ("GLD", "gld")):
        sym = summary["symbols"].get(symbol, empty_trade_summary())
        row[f"{prefix}_trades"] = sym.get("trades", 0)
        row[f"{prefix}_pnl_dollars"] = sym.get("pnl_dollars", 0.0)
        row[f"{prefix}_avg_r"] = sym.get("avg_r", 0.0)
        row[f"{prefix}_win_rate"] = sym.get("win_rate", 0.0)
        row[f"{prefix}_excellent_rate"] = sym.get("excellent_rate", 0.0)
        row[f"{prefix}_avg_mfe_r"] = sym.get("avg_mfe_r", 0.0)
        row[f"{prefix}_low_mfe_loss_rate"] = sym.get("low_mfe_loss_rate", 0.0)
    return [row]


# ---------------------------------------------------------------------------
# Event study suite
# ---------------------------------------------------------------------------


def run_event_suite(cfgs: dict[str, TPCSymbolConfig], train_end: pd.Timestamp) -> list[dict[str, Any]]:
    specs = build_event_specs()
    prepared: dict[tuple[str, str], dict[str, Any]] = {}
    for stack in sorted({spec.stack for spec in specs}):
        for symbol in SYMBOLS:
            prepared[(stack, symbol)] = prepare_event_symbol(symbol, stack)

    rows: list[dict[str, Any]] = []
    for spec in specs:
        print(f"[hidden-alpha] event study {spec.option}/{spec.name}", flush=True)
        events: list[dict[str, Any]] = []
        for symbol in SYMBOLS:
            symbol_data = prepared[(spec.stack, symbol)]
            events.extend(scan_event_signals(symbol_data, cfgs[symbol], spec, train_end))
        rows.extend(flatten_event_summary(spec, events))
    return rows


def build_event_specs() -> list[EventSpec]:
    base = {
        "fib_low": 0.33,
        "fib_high": 0.80,
        "pullback_min": 3,
        "pullback_max": 10,
        "confirm_required": 1,
        "combo": "structure_or_vwap",
        "stop_source": "signal",
        "stop_buffer_atr": 0.12,
        "max_stop_atr": 1.5,
        "min_stop_atr": 0.05,
        "horizon": 36,
        "value_hits": 1,
        "max_extension_atr": 2.25,
        "wait_bars": 4,
    }
    pb30_scaled = {**base, "pullback_min": 6, "pullback_max": 20}
    strict = {**base, "confirm_required": 2, "value_hits": 2, "orderly": True}

    specs = [
        EventSpec("baseline", "current_confirm_market", "current", "confirm_market", dict(base)),
        EventSpec("option1", "pb30_confirm_same", "pb30", "confirm_market", dict(base)),
        EventSpec("option1", "pb30_confirm_scaled", "pb30", "confirm_market", dict(pb30_scaled)),
        EventSpec("option1", "pb30_confirm_quality", "pb30", "confirm_market", dict(strict)),
        EventSpec("option2", "daily_sma20_confirm", "current", "confirm_market", dict(base), daily_filter="sma20"),
        EventSpec("option2", "daily_sma50_confirm", "current", "confirm_market", dict(base), daily_filter="sma50"),
        EventSpec("option2", "daily_sma20_50_confirm", "current", "confirm_market", dict(base), daily_filter="sma20_50"),
        EventSpec("option2", "daily_ret20_sma20_confirm", "current", "confirm_market", dict(base), daily_filter="ret20_sma20"),
        EventSpec("option3", "touch_ema20_market", "current", "value_touch_market", {**base, "target": "ema20"}),
        EventSpec("option3", "touch_vwap_market", "current", "value_touch_market", {**base, "target": "vwap"}),
        EventSpec("option3", "touch_ema20_limit4", "current", "value_touch_limit", {**base, "target": "ema20", "wait_bars": 4}),
        EventSpec("option3", "touch_vwap_limit4", "current", "value_touch_limit", {**base, "target": "vwap", "wait_bars": 4}),
        EventSpec(
            "option3",
            "touch_ema20_limit4_daily",
            "current",
            "value_touch_limit",
            {**base, "target": "ema20", "wait_bars": 4},
            daily_filter="sma20_50",
        ),
        EventSpec("option4", "confirm_vwap_retest4", "current", "confirm_retest", {**base, "target": "vwap", "wait_bars": 4}),
        EventSpec("option4", "confirm_ema20_retest4", "current", "confirm_retest", {**base, "target": "ema20", "wait_bars": 4}),
        EventSpec("option4", "confirm_midpoint_retest4", "current", "confirm_retest", {**base, "target": "midpoint", "wait_bars": 4}),
        EventSpec("option4", "confirm_vwap_retest8", "current", "confirm_retest", {**base, "target": "vwap", "wait_bars": 8}),
        EventSpec(
            "option4",
            "confirm_ema20_retest4_daily",
            "current",
            "confirm_retest",
            {**base, "target": "ema20", "wait_bars": 4},
            daily_filter="sma20_50",
        ),
    ]
    return specs


def prepare_event_symbol(symbol: str, stack: str) -> dict[str, Any]:
    df15 = normalize_timezone(load_bars(DATA_DIR / f"{symbol}_15m.parquet"))
    df1h = normalize_timezone(load_bars(DATA_DIR / f"{symbol}_1h.parquet"))
    dfd = normalize_timezone(load_bars(DATA_DIR / f"{symbol}_1d.parquet"))
    df30 = resample_15m_to_30m(df15)
    df4h = resample_1h_to_4h(df1h)
    if stack == "current":
        primary, pullback = df15, df1h
        idx_pullback = align_completed(primary.index, pullback.index, "15min", "1h")
    elif stack == "pb30":
        primary, pullback = df15, df30
        idx_pullback = align_completed(primary.index, pullback.index, "15min", "30min")
    else:
        raise ValueError(f"unknown event stack {stack!r}")
    return {
        "symbol": symbol,
        "stack": stack,
        "primary": primary,
        "pullback": pullback,
        "trend": df4h,
        "daily": dfd,
        "p": compute_frame_indicators(primary),
        "b": compute_frame_indicators(pullback),
        "t": compute_frame_indicators(df4h),
        "d": compute_frame_indicators(dfd),
        "idx_pullback": idx_pullback,
        "idx_trend": align_completed(primary.index, df4h.index, "15min", "4h"),
        "idx_daily": align_daily_previous_session(primary.index, dfd.index),
    }


def scan_event_signals(
    data: dict[str, Any],
    cfg: TPCSymbolConfig,
    spec: EventSpec,
    train_end: pd.Timestamp,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    i = 2_000
    primary = data["primary"]
    while i < len(primary) - 3:
        timestamp = primary.index[i]
        if not session_ok(timestamp, cfg):
            i += 1
            continue
        direction, grade = event_trend_direction(data, i, cfg)
        if direction == 0:
            i += 1
            continue
        if spec.daily_filter and not event_daily_ok(data, i, direction, spec.daily_filter):
            i += 1
            continue
        if is_extended_from_value(data, i, direction, spec.params):
            i += 1
            continue
        pullback = event_pullback(data, i, direction, spec.params)
        if pullback is None:
            i += 1
            continue

        entry = event_entry_plan(data, i, direction, pullback, spec)
        if entry is None:
            i += 1
            continue
        outcome = evaluate_event_outcome(data, entry, direction, pullback, spec.params)
        if outcome is None:
            i += 1
            continue

        entry_ts = pd.Timestamp(entry["entry_ts"])
        rows.append(
            {
                "symbol": data["symbol"],
                "split": "train" if entry_ts < train_end else "oos",
                "direction": "LONG" if direction > 0 else "SHORT",
                "grade": grade,
                "entry_ts": entry_ts.isoformat(),
                "entry_price": entry["entry_price"],
                "stop_price": outcome["stop_price"],
                "entry_delay_bars": int(entry["entry_i"] - i),
                **outcome,
            }
        )
        i = max(i + 1, int(outcome["exit_i"]) + 1)
    return rows


def event_trend_direction(data: dict[str, Any], i: int, cfg: TPCSymbolConfig) -> tuple[int, str]:
    j = int(data["idx_trend"][i])
    if j < cfg.ma_100_period + 2:
        return 0, "invalid"
    t = data["t"]
    close = t["close"][j]
    ma50 = t["sma50"][j]
    ma100 = t["sma100"][j]
    rsi = t["rsi"][j]
    atr = t["atr"][j]
    plus_di = t["plus_di"][j]
    minus_di = t["minus_di"][j]
    if not np.isfinite([close, ma50, ma100, rsi, atr]).all() or atr <= 0:
        return 0, "nan"
    ma50_slope = ma50 - float(np.nanmean(t["close"][max(0, j - 5):j]))
    ma100_slope = ma100 - float(np.nanmean(t["close"][max(0, j - 12): max(1, j - 1)]))
    if close > ma50 and ma50_slope > 0 and rsi >= cfg.rsi_long_band[0]:
        if not event_trend_quality_ok(1, ma50_slope, ma100_slope, atr, plus_di, minus_di, cfg):
            return 0, "quality"
        grade = "a_plus" if close > ma100 and ma50 > ma100 and ma100_slope > 0 and rsi <= cfg.rsi_a_plus_long_max else "valid"
        return 1, grade
    if close < ma50 and ma50_slope < 0 and rsi <= cfg.rsi_short_band[1]:
        if not event_trend_quality_ok(-1, ma50_slope, ma100_slope, atr, plus_di, minus_di, cfg):
            return 0, "quality"
        grade = "a_plus" if close < ma100 and ma50 < ma100 and ma100_slope < 0 and rsi >= cfg.rsi_a_plus_short_min else "valid"
        return -1, grade
    return 0, "conflict"


def event_trend_quality_ok(
    direction: int,
    ma50_slope: float,
    ma100_slope: float,
    atr: float,
    plus_di: float,
    minus_di: float,
    cfg: TPCSymbolConfig,
) -> bool:
    atr = max(float(atr), 1e-9)
    if cfg.require_di_alignment:
        if direction > 0 and not (plus_di > minus_di):
            return False
        if direction < 0 and not (minus_di > plus_di):
            return False
    if cfg.min_ma50_slope_atr_4h > 0:
        slope_r = ma50_slope / atr if direction > 0 else -ma50_slope / atr
        if slope_r < cfg.min_ma50_slope_atr_4h:
            return False
    if cfg.min_ma100_slope_atr_4h > 0:
        slope_r = ma100_slope / atr if direction > 0 else -ma100_slope / atr
        if slope_r < cfg.min_ma100_slope_atr_4h:
            return False
    return True


def event_daily_ok(data: dict[str, Any], i: int, direction: int, mode: str) -> bool:
    jd = int(data["idx_daily"][i])
    if jd < 60:
        return False
    closes = data["d"]["close"][: jd + 1]
    close = float(closes[-1])
    sma20 = float(np.nanmean(closes[-20:]))
    sma50 = float(np.nanmean(closes[-50:]))
    sma50_prev = float(np.nanmean(closes[-55:-5])) if len(closes) >= 55 else sma50
    ret20 = close / max(float(closes[-21]), 1e-9) - 1.0
    if mode == "sma20":
        return close >= sma20 if direction > 0 else close <= sma20
    if mode == "sma50":
        return close >= sma50 and sma50 >= sma50_prev if direction > 0 else close <= sma50 and sma50 <= sma50_prev
    if mode == "sma20_50":
        return close >= sma20 >= sma50 if direction > 0 else close <= sma20 <= sma50
    if mode == "ret20_sma20":
        return close >= sma20 and ret20 >= 0 if direction > 0 else close <= sma20 and ret20 <= 0
    return True


def is_extended_from_value(data: dict[str, Any], i: int, direction: int, params: dict[str, Any]) -> bool:
    del direction
    jpb = int(data["idx_pullback"][i])
    jtr = int(data["idx_trend"][i])
    ema20 = data["b"]["ema20"][jpb]
    atr = data["t"]["atr"][jtr]
    if not np.isfinite([ema20, atr]).all() or atr <= 0:
        return False
    close = data["p"]["close"][i]
    return abs(close - ema20) > float(params.get("max_extension_atr", 2.25)) * atr


def event_pullback(data: dict[str, Any], i: int, direction: int, params: dict[str, Any]) -> dict[str, Any] | None:
    j = int(data["idx_pullback"][i])
    max_bars = int(params.get("pullback_max", 10))
    min_bars = int(params.get("pullback_min", 3))
    lookback = min(max_bars + 6, j + 1)
    if j < max_bars + 8 or lookback < max_bars + 2:
        return None
    b = data["b"]
    start = j - lookback + 1
    highs = b["high"][start : j + 1]
    lows = b["low"][start : j + 1]
    if direction > 0:
        impulse_low = float(np.nanmin(lows[: max(2, lookback // 2)]))
        impulse_high = float(np.nanmax(highs))
        current_low = float(np.nanmin(lows[-max_bars:]))
        depth = (impulse_high - current_low) / max(impulse_high - impulse_low, 1e-9)
        value_hits = count_value_hits(data, i, direction, current_low)
    else:
        impulse_high = float(np.nanmax(highs[: max(2, lookback // 2)]))
        impulse_low = float(np.nanmin(lows))
        current_high = float(np.nanmax(highs[-max_bars:]))
        depth = (current_high - impulse_low) / max(impulse_high - impulse_low, 1e-9)
        value_hits = count_value_hits(data, i, direction, current_high)
    if value_hits < int(params.get("value_hits", 1)):
        return None
    if not (float(params.get("fib_low", 0.33)) <= depth <= float(params.get("fib_high", 0.80))):
        return None
    recent_ranges = highs[-max_bars:] - lows[-max_bars:]
    impulse_ranges = highs[: max(2, min_bars)] - lows[: max(2, min_bars)]
    orderly = float(np.nanmean(recent_ranges)) <= float(np.nanmean(impulse_ranges)) * 1.25
    if bool(params.get("orderly", False)) and not orderly:
        return None
    return {
        "depth": float(depth),
        "low": float(np.nanmin(lows[-max_bars:])),
        "high": float(np.nanmax(highs[-max_bars:])),
        "value_hits": int(value_hits),
        "orderly": bool(orderly),
    }


def count_value_hits(data: dict[str, Any], i: int, direction: int, extreme: float) -> int:
    return sum(
        np.isfinite(level) and ((extreme <= level) if direction > 0 else (extreme >= level))
        for level in value_levels(data, i).values()
    )


def value_levels(data: dict[str, Any], i: int) -> dict[str, float]:
    j = int(data["idx_pullback"][i])
    return {
        "ema20": float(data["b"]["ema20"][j]),
        "ema50": float(data["b"]["ema50"][j]),
        "vwap": float(data["b"]["vwap"][j]),
    }


def event_entry_plan(
    data: dict[str, Any],
    i: int,
    direction: int,
    pullback: dict[str, Any],
    spec: EventSpec,
) -> dict[str, Any] | None:
    model = spec.entry_model
    if model == "confirm_market":
        ok, _triggers = event_confirmation(data, i, direction, spec.params)
        if not ok:
            return None
        return market_entry_next_bar(data, i)
    if model == "value_touch_market":
        if not current_bar_touches_target(data, i, direction, spec.params):
            return None
        return market_entry_next_bar(data, i)
    if model == "value_touch_limit":
        if not current_bar_touches_target(data, i, direction, spec.params):
            return None
        target = target_price(data, i, direction, spec.params)
        return limit_entry_after_signal(data, i, direction, target, int(spec.params.get("wait_bars", 4)))
    if model == "confirm_retest":
        ok, _triggers = event_confirmation(data, i, direction, spec.params)
        if not ok:
            return None
        target = target_price(data, i, direction, spec.params)
        if target is None:
            return None
        signal_close = float(data["p"]["close"][i])
        if direction > 0 and target >= signal_close:
            return None
        if direction < 0 and target <= signal_close:
            return None
        return limit_entry_after_signal(data, i, direction, target, int(spec.params.get("wait_bars", 4)))
    raise ValueError(f"unknown event entry model {model!r}")


def event_confirmation(data: dict[str, Any], i: int, direction: int, params: dict[str, Any]) -> tuple[bool, list[str]]:
    if i < 6:
        return False, []
    p = data["p"]
    open_, high, low, close = p["open"][i], p["high"][i], p["low"][i], p["close"][i]
    rng = max(high - low, 1e-9)
    prev_highs = p["high"][i - 5 : i]
    prev_lows = p["low"][i - 5 : i]
    triggers: list[str] = []
    vwap = p["vwap"][i]
    ema20 = p["ema20"][i]
    vol_sma = p["volume_sma"][i]
    if direction > 0:
        if close > open_ and (min(open_, close) - low) / rng >= 0.35:
            triggers.append("bullish_reversal")
        if np.isfinite(vwap) and close > vwap:
            triggers.append("vwap_reclaim")
        if low > float(np.nanmin(prev_lows[-3:])):
            triggers.append("higher_low")
        if close > float(np.nanmax(prev_highs[-3:])):
            triggers.append("micro_break")
        if np.isfinite(ema20) and close > ema20:
            triggers.append("trendline_break")
        if (close - low) / rng >= 2.0 / 3.0:
            triggers.append("upper_third_close")
    else:
        if close < open_ and (high - max(open_, close)) / rng >= 0.35:
            triggers.append("bearish_reversal")
        if np.isfinite(vwap) and close < vwap:
            triggers.append("vwap_loss")
        if high < float(np.nanmax(prev_highs[-3:])):
            triggers.append("lower_high")
        if close < float(np.nanmin(prev_lows[-3:])):
            triggers.append("micro_break")
        if np.isfinite(ema20) and close < ema20:
            triggers.append("trendline_break")
        if (close - low) / rng <= 1.0 / 3.0:
            triggers.append("lower_third_close")
    if np.isfinite(vol_sma) and p["volume"][i] >= 1.3 * vol_sma:
        triggers.append("volume_expansion")
    names = set(triggers)
    if len(names) < int(params.get("confirm_required", 1)):
        return False, triggers
    has_vwap = any("vwap" in item for item in names)
    has_structure = any(item in names for item in ("higher_low", "lower_high", "micro_break"))
    has_micro = "micro_break" in names
    mode = str(params.get("combo", "structure_or_vwap"))
    if mode == "structure_or_vwap" and not (has_structure or has_vwap):
        return False, triggers
    if mode == "structure_vwap" and not (has_structure and has_vwap):
        return False, triggers
    if mode == "micro_vwap" and not (has_micro and has_vwap):
        return False, triggers
    return True, triggers


def current_bar_touches_target(data: dict[str, Any], i: int, direction: int, params: dict[str, Any]) -> bool:
    target = target_price(data, i, direction, params)
    if target is None or not np.isfinite(target):
        return False
    high = float(data["p"]["high"][i])
    low = float(data["p"]["low"][i])
    return low <= target <= high


def target_price(data: dict[str, Any], i: int, direction: int, params: dict[str, Any]) -> float | None:
    del direction
    target = str(params.get("target", "ema20"))
    if target == "midpoint":
        return float((data["p"]["high"][i] + data["p"]["low"][i]) / 2.0)
    levels = value_levels(data, i)
    value = levels.get(target)
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def market_entry_next_bar(data: dict[str, Any], i: int) -> dict[str, Any] | None:
    entry_i = i + 1
    if entry_i >= len(data["p"]["open"]):
        return None
    return {
        "entry_i": entry_i,
        "entry_ts": data["primary"].index[entry_i],
        "entry_price": float(data["p"]["open"][entry_i]),
    }


def limit_entry_after_signal(
    data: dict[str, Any],
    i: int,
    direction: int,
    limit_price: float | None,
    wait_bars: int,
) -> dict[str, Any] | None:
    if limit_price is None or not np.isfinite(limit_price):
        return None
    end = min(len(data["p"]["close"]) - 1, i + max(wait_bars, 1))
    for entry_i in range(i + 1, end + 1):
        high = float(data["p"]["high"][entry_i])
        low = float(data["p"]["low"][entry_i])
        if low <= limit_price <= high:
            return {
                "entry_i": entry_i,
                "entry_ts": data["primary"].index[entry_i],
                "entry_price": float(limit_price),
            }
        if direction > 0 and float(data["p"]["close"][entry_i]) < limit_price * 0.98:
            return None
        if direction < 0 and float(data["p"]["close"][entry_i]) > limit_price * 1.02:
            return None
    return None


def evaluate_event_outcome(
    data: dict[str, Any],
    entry: dict[str, Any],
    direction: int,
    pullback: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any] | None:
    entry_i = int(entry["entry_i"])
    entry_price = float(entry["entry_price"])
    jtr = int(data["idx_trend"][entry_i])
    atr = float(data["t"]["atr"][jtr])
    if not np.isfinite(atr) or atr <= 0:
        return None
    signal_i = max(0, entry_i - 1)
    buffer = float(params.get("stop_buffer_atr", 0.12)) * atr
    if params.get("stop_source") == "pullback":
        stop = pullback["low"] - buffer if direction > 0 else pullback["high"] + buffer
    else:
        stop = data["p"]["low"][signal_i] - buffer if direction > 0 else data["p"]["high"][signal_i] + buffer
    risk = abs(entry_price - stop)
    if risk <= 0:
        return None
    stop_atr = risk / max(atr, 1e-9)
    if stop_atr > float(params.get("max_stop_atr", 1.5)) or stop_atr < float(params.get("min_stop_atr", 0.05)):
        return None
    mfe_r = 0.0
    mae_r = 0.0
    hit1 = False
    hit2 = False
    stopped = False
    horizon = int(params.get("horizon", 36))
    exit_i = min(entry_i + horizon, len(data["p"]["close"]) - 1)
    outcome_r: float | None = None
    for k in range(entry_i, min(entry_i + horizon, len(data["p"]["close"]))):
        high = float(data["p"]["high"][k])
        low = float(data["p"]["low"][k])
        if direction > 0:
            mfe_r = max(mfe_r, (high - entry_price) / risk)
            mae_r = max(mae_r, (entry_price - low) / risk)
            stop_hit = low <= stop
            one_hit = high >= entry_price + risk
            two_hit = high >= entry_price + 2.0 * risk
        else:
            mfe_r = max(mfe_r, (entry_price - low) / risk)
            mae_r = max(mae_r, (high - entry_price) / risk)
            stop_hit = high >= stop
            one_hit = low <= entry_price - risk
            two_hit = low <= entry_price - 2.0 * risk
        if stop_hit:
            stopped = True
            outcome_r = -1.0
            exit_i = k
            break
        if two_hit:
            hit2 = True
        if one_hit:
            hit1 = True
            outcome_r = 1.0
            exit_i = k
            break
    if outcome_r is None:
        close = float(data["p"]["close"][exit_i])
        raw_r = (close - entry_price) / risk if direction > 0 else (entry_price - close) / risk
        outcome_r = max(-1.0, min(1.0, raw_r))
    return {
        "r": float(outcome_r),
        "mfe_r": float(mfe_r),
        "mae_r": float(mae_r),
        "hit1": bool(hit1),
        "hit2": bool(hit2),
        "stopped": bool(stopped),
        "exit_i": int(exit_i),
        "stop_price": float(stop),
        "stop_atr": float(stop_atr),
        "depth": pullback["depth"],
        "value_hits": pullback["value_hits"],
        "orderly": pullback["orderly"],
    }


def flatten_event_summary(spec: EventSpec, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "oos"):
        split_events = [event for event in events if event["split"] == split]
        rows.append(event_summary_row(spec, split, "ALL", split_events))
        for symbol in SYMBOLS:
            rows.append(event_summary_row(spec, split, symbol, [event for event in split_events if event["symbol"] == symbol]))
    return rows


def event_summary_row(spec: EventSpec, split: str, cohort: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        base = empty_event_summary()
    else:
        rs = np.asarray([event["r"] for event in events], dtype=float)
        wins = rs[rs > 0]
        losses = rs[rs < 0]
        base = {
            "signals": len(events),
            "avg_r": float(np.mean(rs)),
            "total_r": float(np.sum(rs)),
            "profit_factor_r": float(np.sum(wins) / abs(np.sum(losses))) if losses.size and abs(np.sum(losses)) > 0 else (float(np.sum(wins)) if wins.size else 0.0),
            "hit1_rate": float(np.mean([event["hit1"] for event in events])),
            "hit2_rate": float(np.mean([event["hit2"] for event in events])),
            "stop_rate": float(np.mean([event["stopped"] for event in events])),
            "avg_mfe_r": float(np.mean([event["mfe_r"] for event in events])),
            "avg_mae_r": float(np.mean([event["mae_r"] for event in events])),
            "avg_stop_atr": float(np.mean([event["stop_atr"] for event in events])),
            "avg_entry_delay_bars": float(np.mean([event["entry_delay_bars"] for event in events])),
            "avg_depth": float(np.mean([event["depth"] for event in events])),
            "orderly_rate": float(np.mean([event["orderly"] for event in events])),
        }
    return {
        "option": spec.option,
        "name": spec.name,
        "split": split,
        "cohort": cohort,
        "stack": spec.stack,
        "entry_model": spec.entry_model,
        "daily_filter": spec.daily_filter,
        "note": spec.note,
        **base,
    }


# ---------------------------------------------------------------------------
# Data and formatting helpers
# ---------------------------------------------------------------------------


def compute_frame_indicators(df: pd.DataFrame) -> dict[str, np.ndarray]:
    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    volumes = df["volume"].to_numpy(dtype=float) if "volume" in df.columns else np.zeros(len(df), dtype=float)
    times = tuple(pd.Timestamp(value).to_pydatetime() for value in df.index.values)
    adx, plus_di, minus_di = ind.adx(highs, lows, closes, 14)
    return {
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "sma20": ind.sma(closes, 20),
        "sma50": ind.sma(closes, 50),
        "sma100": ind.sma(closes, 100),
        "ema20": ind.ema(closes, 20),
        "ema50": ind.ema(closes, 50),
        "rsi": ind.rsi(closes, 14),
        "atr": ind.atr(highs, lows, closes, 14),
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "volume_sma": ind.volume_sma(volumes, 20),
        "vwap": ind.vwap_anchored(highs, lows, closes, volumes, times, 9, 30),
    }


def build_context_arrays(symbol: str, df15: pd.DataFrame, *, end_ts: pd.Timestamp | None) -> dict[str, np.ndarray]:
    context_symbol = CONTEXT_SYMBOLS.get(symbol, "")
    if not context_symbol:
        return {}
    path_1h = DATA_DIR / f"{context_symbol}_1h.parquet"
    path_1d = DATA_DIR / f"{context_symbol}_1d.parquet"
    if not path_1h.exists() or not path_1d.exists():
        return {}
    context_1h = slice_timestamp_index(normalize_timezone(load_bars(path_1h)), None, end_ts)
    context_daily = slice_timestamp_index(normalize_timezone(load_bars(path_1d)), None, end_ts)
    if context_1h.empty or context_daily.empty:
        return {}
    close_1h = context_1h["close"].astype(float)
    close_daily = context_daily["close"].astype(float)
    hourly = pd.DataFrame(
        {
            "context_close_1h": close_1h,
            "context_sma20_1h": close_1h.rolling(20, min_periods=20).mean(),
            "context_sma50_1h": close_1h.rolling(50, min_periods=50).mean(),
            "context_ret12_1h": close_1h.pct_change(12),
            "context_ret24_1h": close_1h.pct_change(24),
        },
        index=context_1h.index,
    )
    daily = pd.DataFrame(
        {
            "context_close_daily": close_daily,
            "context_sma20_daily": close_daily.rolling(20, min_periods=20).mean(),
            "context_sma50_daily": close_daily.rolling(50, min_periods=50).mean(),
            "context_ret20_daily": close_daily.pct_change(20),
        },
        index=context_daily.index,
    )
    idx_1h = align_completed(df15.index, context_1h.index, "15min", "1h")
    idx_daily = align_daily_previous_session(df15.index, context_daily.index)
    out: dict[str, np.ndarray] = {}
    for key in hourly:
        out[key] = take_aligned(hourly[key].to_numpy(dtype=float), idx_1h)
    for key in daily:
        out[key] = take_aligned(daily[key].to_numpy(dtype=float), idx_daily)
    return out


def align_completed(
    lower_times: pd.DatetimeIndex,
    higher_times: pd.DatetimeIndex,
    lower_freq: str,
    higher_freq: str,
) -> np.ndarray:
    if len(higher_times) == 0:
        return np.full(len(lower_times), -1, dtype=np.int64)
    lower_close = pd.DatetimeIndex(lower_times) + pd.Timedelta(lower_freq)
    higher_close = pd.DatetimeIndex(higher_times) + pd.Timedelta(higher_freq)
    idx = np.searchsorted(higher_close.values, lower_close.values, side="right").astype(np.int64) - 1
    return np.minimum(idx, len(higher_close) - 1)


def align_daily_previous_session(lower_times: pd.DatetimeIndex, daily_times: pd.DatetimeIndex) -> np.ndarray:
    lower_dates = pd.DatetimeIndex(lower_times).normalize().values.astype("datetime64[D]")
    daily_dates = pd.DatetimeIndex(daily_times).normalize().values.astype("datetime64[D]")
    if len(daily_dates) == 0:
        return np.full(len(lower_dates), -1, dtype=np.int64)
    idx = np.searchsorted(daily_dates, lower_dates, side="left").astype(np.int64) - 1
    return np.minimum(idx, len(daily_dates) - 1)


def take_aligned(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    out = np.full(len(idx), np.nan, dtype=float)
    mask = (idx >= 0) & (idx < len(values))
    out[mask] = values[idx[mask]]
    return out


def session_ok(timestamp: pd.Timestamp, cfg: TPCSymbolConfig) -> bool:
    py = pd.Timestamp(timestamp).to_pydatetime()
    if not is_in_session_window(py, cfg.primary_windows_et):
        return False
    if cfg.avoid_windows_et and is_in_session_window(py, cfg.avoid_windows_et):
        return False
    local = py.astimezone(ET_TZ)
    return not (local.weekday() == 4 and local.hour >= 14)


def empty_trade_summary() -> dict[str, Any]:
    return {
        "trades": 0,
        "pnl_dollars": 0.0,
        "avg_r": 0.0,
        "win_rate": 0.0,
        "excellent_rate": 0.0,
        "avg_mfe_r": 0.0,
        "low_mfe_loss_rate": 0.0,
    }


def empty_event_summary() -> dict[str, Any]:
    return {
        "signals": 0,
        "avg_r": 0.0,
        "total_r": 0.0,
        "profit_factor_r": 0.0,
        "hit1_rate": 0.0,
        "hit2_rate": 0.0,
        "stop_rate": 0.0,
        "avg_mfe_r": 0.0,
        "avg_mae_r": 0.0,
        "avg_stop_atr": 0.0,
        "avg_entry_delay_bars": 0.0,
        "avg_depth": 0.0,
        "orderly_rate": 0.0,
    }


def format_report(full_rows: list[dict[str, Any]], event_rows: list[dict[str, Any]], *, elapsed_seconds: float) -> str:
    lines = [
        "# TPC Hidden Alpha Variant Tests",
        "",
        f"Elapsed minutes: {elapsed_seconds / 60.0:.1f}",
        "",
    ]
    if full_rows:
        lines.extend(["## Full Replay OOS", "", "| Option | Variant | Trades | Net | AvgR | PF | QQQ AvgR | GLD AvgR |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for row in sorted([r for r in full_rows if r["split"] == "oos"], key=lambda r: (r["option"], -float(r["net_return_pct"]))):
            lines.append(
                f"| {row['option']} | {row['name']} | {row['total_trades']:.0f} | "
                f"{row['net_return_pct']:+.2f}% | {row['avg_r']:+.3f} | {row['dollar_profit_factor']:.2f} | "
                f"{row['qqq_avg_r']:+.3f} | {row['gld_avg_r']:+.3f} |"
            )
        lines.append("")
    if event_rows:
        all_oos = [row for row in event_rows if row["split"] == "oos" and row["cohort"] == "ALL"]
        lines.extend(["## Event Study OOS", "", "| Option | Variant | Signals | AvgR | PF | Hit1 | MFE | Stop |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for row in sorted(all_oos, key=lambda r: (r["option"], -float(r["avg_r"]))):
            lines.append(
                f"| {row['option']} | {row['name']} | {row['signals']:.0f} | {row['avg_r']:+.3f} | "
                f"{row['profit_factor_r']:.2f} | {row['hit1_rate']:.1%} | {row['avg_mfe_r']:.2f} | {row['stop_rate']:.1%} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation Guardrails",
            "- Full replay uses the round-7 TPC implementation and includes available QQQ->NQ / GLD->GC context indicators.",
            "- Event-study R is a conservative 1R/stop probe, not the production T1/T2/runner exit model.",
            "- Holdout rows are the post-2025-11-01 sample and remain small; strong-looking low-N rows are research leads, not promotion evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def coerce_utc(value: str | pd.Timestamp | None, *, end_of_day: bool = False) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    if end_of_day and ts == ts.normalize():
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return ts


def slice_timestamp_index(
    df: pd.DataFrame,
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
) -> pd.DataFrame:
    if start_ts is not None:
        df = df.loc[df.index >= start_ts]
    if end_ts is not None:
        df = df.loc[df.index <= end_ts]
    return df


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(normalize_jsonable(payload), indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(normalize_jsonable(rows))


def normalize_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): normalize_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [normalize_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return 0.0
    return value


if __name__ == "__main__":
    main()
