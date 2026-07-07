"""Baseline and diagnostics for the 15m ETF swing strategies."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.shared.data.ibkr.store import detect_large_gaps, ensure_utc_index
from backtests.swing.config_tpc import TPCBacktestConfig
from backtests.swing.data.multitimeframe import (
    align_15m_to_1h,
    align_15m_to_30m,
    align_15m_to_4h,
    align_daily_to_15m,
    resample_15m_to_30m,
    resample_1h_to_4h,
)
from backtests.swing.data.replay_cache import load_tpc_replay_bundle
from backtests.swing.engine.tpc_engine import run_tpc_independent

DATA_DIR = Path("backtests/swing/data/raw")
DEFAULT_OUTPUT_ROOT = Path("backtests/output/swing")
DEFAULT_REPORT = DEFAULT_OUTPUT_ROOT / "etf_baseline" / "etf_baseline_diagnostics.txt"
DEFAULT_JSON = DEFAULT_OUTPUT_ROOT / "etf_baseline" / "etf_baseline_summary.json"
SYMBOLS = ("QQQ", "GLD")


@dataclass(frozen=True)
class AlignmentReport:
    symbol: str
    start: str
    end: str
    bars_15m: int
    bars_30m: int
    bars_1h: int
    bars_4h: int
    bars_daily: int
    duplicate_15m: int
    null_ohlc_15m: int
    large_gaps_15m: int
    hourly_common: int
    hourly_coverage_pct: float
    hourly_p95_abs_diff: float
    hourly_max_abs_diff: float
    hourly_gt_10c: int
    future_30m: int
    future_1h: int
    future_4h: int
    future_daily: int
    first_unaligned_1h: int
    first_unaligned_4h: int
    first_unaligned_daily: int

    @property
    def passed(self) -> bool:
        return (
            self.duplicate_15m == 0
            and self.null_ohlc_15m == 0
            and self.large_gaps_15m == 0
            and self.future_30m == 0
            and self.future_1h == 0
            and self.future_4h == 0
            and self.future_daily == 0
            and self.hourly_coverage_pct >= 0.95
            and self.hourly_p95_abs_diff <= 0.10
            and self.hourly_max_abs_diff <= 1.00
        )


@dataclass(frozen=True)
class StrategyBaseline:
    strategy: str
    elapsed_seconds: float
    equity_points: int
    trades: int
    final_equity: float
    net_pnl: float
    net_return_pct: float
    win_rate: float
    profit_factor: float
    avg_r: float
    total_r: float
    max_dd_pct: float
    max_dd_dollars: float
    sharpe_daily: float
    trades_per_month: float
    avg_hold_bars: float
    top5_winner_share: float
    rolling20_avg_r: float
    rolling20_min_r: float
    rolling20_slope: float


StrategySpec = tuple[
    type,
    Callable[..., Any],
    Callable[[dict[str, dict[str, Any]], Any], Any],
]

STRATEGIES: dict[str, StrategySpec] = {
    "TPC": (TPCBacktestConfig, load_tpc_replay_bundle, run_tpc_independent),
}

ALPHA_NOTES = {
    "TPC": "Trend continuation after a 4h trend regime and a controlled 1h pullback into value.",
}

ALPHA_BASELINE_TARGETS = {
    "TPC": {"trades_per_month": 1.0, "avg_r": 0.0, "mfe_r": 1.0},
}


def infer_training_window(data_dir: Path, holdout_months: int) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    starts: list[pd.Timestamp] = []
    intraday_ends: list[pd.Timestamp] = []
    for symbol in SYMBOLS:
        for timeframe in ("15m", "1h"):
            idx = _read_index(data_dir / f"{symbol}_{timeframe}.parquet")
            starts.append(idx.min())
            intraday_ends.append(idx.max())
        daily_idx = _read_index(data_dir / f"{symbol}_1d.parquet")
        starts.append(daily_idx.min())
    data_end = min(intraday_ends)
    train_end = data_end - pd.DateOffset(months=holdout_months)
    return max(starts), pd.Timestamp(train_end), data_end


def _coerce_report_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _estimate_holdout_months(data_end: pd.Timestamp, end: pd.Timestamp) -> int:
    if end >= data_end:
        return 0
    best_months = 0
    best_delta_seconds = float("inf")
    for months in range(1, 37):
        candidate = pd.Timestamp(data_end - pd.DateOffset(months=months))
        delta_seconds = abs((candidate - end).total_seconds())
        if delta_seconds < best_delta_seconds:
            best_months = months
            best_delta_seconds = delta_seconds
    return best_months if best_delta_seconds <= 7 * 24 * 3600 else 0


def verify_etf_data_alignment(data_dir: Path, symbols: tuple[str, ...] = SYMBOLS) -> list[AlignmentReport]:
    reports: list[AlignmentReport] = []
    for symbol in symbols:
        df15 = _read_bars(data_dir / f"{symbol}_15m.parquet")
        df1h = _read_bars(data_dir / f"{symbol}_1h.parquet")
        dfd = _read_bars(data_dir / f"{symbol}_1d.parquet")
        df30 = resample_15m_to_30m(df15)
        df4h = resample_1h_to_4h(df1h)

        idx30 = align_15m_to_30m(df15, df30)
        idx1h = align_15m_to_1h(df15, df1h)
        idx4h = align_15m_to_4h(df15, df4h)
        idxd = align_daily_to_15m(df15, dfd)

        hourly_recon = _resample_to_hourly_start(df15)
        common = hourly_recon.index.intersection(df1h.index)
        diffs = hourly_recon.loc[common, ["open", "high", "low", "close"]].sub(
            df1h.loc[common, ["open", "high", "low", "close"]]
        ).abs()
        row_max = diffs.max(axis=1) if len(diffs) else pd.Series(dtype=float)

        reports.append(
            AlignmentReport(
                symbol=symbol,
                start=str(df15.index.min()),
                end=str(df15.index.max()),
                bars_15m=len(df15),
                bars_30m=len(df30),
                bars_1h=len(df1h),
                bars_4h=len(df4h),
                bars_daily=len(dfd),
                duplicate_15m=int(df15.index.duplicated().sum()),
                null_ohlc_15m=int(df15[["open", "high", "low", "close"]].isna().any(axis=1).sum()),
                large_gaps_15m=len(detect_large_gaps(df15, "15m")),
                hourly_common=len(common),
                hourly_coverage_pct=len(common) / max(len(df1h), 1),
                hourly_p95_abs_diff=float(row_max.quantile(0.95)) if len(row_max) else 0.0,
                hourly_max_abs_diff=float(row_max.max()) if len(row_max) else 0.0,
                hourly_gt_10c=int((row_max > 0.10).sum()) if len(row_max) else 0,
                future_30m=_future_intraday_count(df15.index, df30.index, idx30, "15min", "30min"),
                future_1h=_future_intraday_count(df15.index, df1h.index, idx1h, "15min", "1h"),
                future_4h=_future_intraday_count(df15.index, df4h.index, idx4h, "15min", "4h"),
                future_daily=_future_daily_count(df15.index, dfd.index, idxd),
                first_unaligned_1h=int((idx1h < 0).sum()),
                first_unaligned_4h=int((idx4h < 0).sum()),
                first_unaligned_daily=int((idxd < 0).sum()),
            )
        )
    return reports


def run_baselines(
    data_dir: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial_equity: float,
) -> tuple[dict[str, StrategyBaseline], dict[str, Any]]:
    summaries: dict[str, StrategyBaseline] = {}
    results: dict[str, Any] = {}
    for name, (config_cls, loader, runner) in STRATEGIES.items():
        print(f"Running {name} baseline {start.isoformat()} -> {end.isoformat()}...", flush=True)
        t0 = time.time()
        cfg = config_cls(initial_equity=initial_equity, data_dir=data_dir)
        bundle = loader(data_dir, start_date=start, end_date=end)
        result = runner(bundle.data, cfg)
        elapsed = time.time() - t0
        summaries[name] = summarize_strategy(name, result, initial_equity, start, end, elapsed)
        results[name] = result
        print(
            f"  {name}: trades={summaries[name].trades} "
            f"PF={summaries[name].profit_factor:.2f} avgR={summaries[name].avg_r:+.3f} "
            f"net={summaries[name].net_return_pct:+.2f}%",
            flush=True,
        )
    return summaries, results


def summarize_strategy(
    name: str,
    result: Any,
    initial_equity: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    elapsed_seconds: float,
) -> StrategyBaseline:
    trades = sorted(list(getattr(result, "trades", [])), key=lambda t: getattr(t, "entry_time", None) or datetime.min)
    equity = np.asarray(getattr(result, "combined_equity", []), dtype=float)
    timestamps = np.asarray(getattr(result, "combined_timestamps", []), dtype=object)
    final_equity = float(equity[-1]) if len(equity) else initial_equity
    pnls = np.asarray([float(getattr(t, "pnl_dollars", 0.0) or 0.0) for t in trades], dtype=float)
    rs = np.asarray([float(getattr(t, "r_multiple", 0.0) or 0.0) for t in trades], dtype=float)
    holds = np.asarray([float(getattr(t, "bars_held", 0.0) or 0.0) for t in trades], dtype=float)
    gross_profit = float(np.sum(pnls[pnls > 0])) if len(pnls) else 0.0
    gross_loss = abs(float(np.sum(pnls[pnls < 0]))) if len(pnls) else 0.0
    max_dd_pct, max_dd_dollars = _max_drawdown(equity)
    rolling = _rolling_stats(rs, 20)
    months = max((end - start).total_seconds() / (30.4375 * 24 * 3600), 1.0)
    winners = sorted([float(x) for x in pnls if x > 0], reverse=True)
    winner_sum = sum(winners)
    return StrategyBaseline(
        strategy=name,
        elapsed_seconds=elapsed_seconds,
        equity_points=len(equity),
        trades=len(trades),
        final_equity=final_equity,
        net_pnl=final_equity - initial_equity,
        net_return_pct=(final_equity - initial_equity) / initial_equity * 100.0,
        win_rate=float(np.mean(pnls > 0)) if len(pnls) else 0.0,
        profit_factor=gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0),
        avg_r=float(np.mean(rs)) if len(rs) else 0.0,
        total_r=float(np.sum(rs)) if len(rs) else 0.0,
        max_dd_pct=max_dd_pct,
        max_dd_dollars=max_dd_dollars,
        sharpe_daily=_daily_sharpe(equity, timestamps),
        trades_per_month=len(trades) / months,
        avg_hold_bars=float(np.mean(holds)) if len(holds) else 0.0,
        top5_winner_share=sum(winners[:5]) / winner_sum if winner_sum > 0 else 0.0,
        rolling20_avg_r=rolling["last"],
        rolling20_min_r=rolling["min"],
        rolling20_slope=rolling["slope"],
    )


def build_report(
    *,
    alignment: list[AlignmentReport],
    baselines: dict[str, StrategyBaseline],
    results: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    data_end: pd.Timestamp,
    holdout_months: int,
    initial_equity: float,
) -> str:
    lines = [
        "ETF SWING BASELINE DIAGNOSTICS",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Training window: {start.isoformat()} -> {end.isoformat()}",
        f"Latest common intraday data: {data_end.isoformat()}",
        f"Holdout: most recent {holdout_months} months",
        f"Initial equity per independent strategy: ${initial_equity:,.0f}",
        "",
        "DATA ALIGNMENT",
        "-" * 80,
    ]
    for item in alignment:
        status = "PASS" if item.passed else "CHECK"
        lines.append(
            f"{item.symbol}: {status} | 15m={item.bars_15m:,} 30m={item.bars_30m:,} "
            f"1h={item.bars_1h:,} 4h={item.bars_4h:,} 1d={item.bars_daily:,}"
        )
        lines.append(
            f"  range={item.start} -> {item.end} | dupes={item.duplicate_15m} "
            f"null_ohlc={item.null_ohlc_15m} large_gaps={item.large_gaps_15m}"
        )
        lines.append(
            f"  15m->1h reconstruction: common={item.hourly_common:,} "
            f"coverage={item.hourly_coverage_pct:.1%} p95_abs=${item.hourly_p95_abs_diff:.4f} "
            f"max_abs=${item.hourly_max_abs_diff:.4f} bars_gt_10c={item.hourly_gt_10c}"
        )
        lines.append(
            f"  completed-bar leaks: 30m={item.future_30m} 1h={item.future_1h} "
            f"4h={item.future_4h} daily={item.future_daily} | initial no-context bars: "
            f"1h={item.first_unaligned_1h} 4h={item.first_unaligned_4h} daily={item.first_unaligned_daily}"
        )
    lines.extend(["", "BASELINE SCOREBOARD", "-" * 80])
    lines.append(
        f"{'Strategy':8s} {'Trades':>7s} {'Ret%':>9s} {'PF':>7s} {'AvgR':>8s} "
        f"{'TotR':>9s} {'DD%':>8s} {'Sharpe':>8s} {'T/Mo':>8s} {'Time':>8s}"
    )
    for name in ("TPC",):
        m = baselines[name]
        pf = "inf" if np.isinf(m.profit_factor) else f"{m.profit_factor:.2f}"
        lines.append(
            f"{name:8s} {m.trades:7d} {m.net_return_pct:+8.2f}% {pf:>7s} "
            f"{m.avg_r:+8.3f} {m.total_r:+9.2f} {m.max_dd_pct:7.2f}% "
            f"{m.sharpe_daily:8.2f} {m.trades_per_month:8.2f} {m.elapsed_seconds:7.1f}s"
        )
    lines.append("")
    for name in ("TPC",):
        lines.extend(_strategy_report(name, results[name], baselines[name]))
        lines.append("")
    lines.extend(_portfolio_style_snapshot(results, baselines))
    lines.extend(_optimization_readiness(baselines, alignment))
    return "\n".join(lines)


def build_strategy_full_diagnostics(
    name: str,
    result: Any,
    metrics: StrategyBaseline,
    *,
    alignment: list[AlignmentReport],
    start: pd.Timestamp,
    end: pd.Timestamp,
    data_end: pd.Timestamp,
    holdout_months: int,
    initial_equity: float,
) -> str:
    lines = [
        f"{name} BASELINE FULL DIAGNOSTICS",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Training window: {start.isoformat()} -> {end.isoformat()}",
        f"Latest common intraday data: {data_end.isoformat()}",
        f"Holdout: most recent {holdout_months} months",
        f"Initial equity: ${initial_equity:,.0f}",
        "Comparable basis: current-code final optimized config replay",
        "",
        "BASELINE RESULT",
        "-" * 80,
        f"Trades={metrics.trades}, net={metrics.net_return_pct:+.2f}%, "
        f"PF={metrics.profit_factor:.2f}, avgR={metrics.avg_r:+.3f}, totalR={metrics.total_r:+.2f}",
        f"MaxDD={metrics.max_dd_pct:.2f}% (${metrics.max_dd_dollars:,.2f}), "
        f"Sharpe={metrics.sharpe_daily:.2f}, trades/month={metrics.trades_per_month:.2f}",
        f"Avg hold={metrics.avg_hold_bars:.1f} 15m bars, top-5 winner share={metrics.top5_winner_share:.1%}",
        "",
        "DATA READINESS",
        "-" * 80,
    ]
    for item in alignment:
        lines.append(
            f"{item.symbol}: {'PASS' if item.passed else 'CHECK'} | "
            f"15m={item.bars_15m:,}, 1h={item.bars_1h:,}, 4h={item.bars_4h:,}, "
            f"leaks 30m/1h/4h/daily={item.future_30m}/{item.future_1h}/{item.future_4h}/{item.future_daily}"
        )
    lines.extend(_bespoke_strategy_diagnostics(name, result, metrics))
    lines.extend(["", "GENERAL DIAGNOSTICS", "-" * 80])
    lines.extend(_strategy_report(name, result, metrics))
    lines.extend(_strategy_optimizer_readiness(name, result, metrics))
    return "\n".join(lines)


def build_tpc_optimized_full_diagnostics(
    mutations: dict[str, Any],
    *,
    data_dir: Path,
    initial_equity: float = 100_000.0,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    holdout_months: int | None = None,
    title: str = "TPC OPTIMISED CONFIG FULL DIAGNOSTICS",
) -> str:
    inferred_start, inferred_end, data_end = infer_training_window(data_dir, holdout_months or 0)
    start = _coerce_report_timestamp(start_date) if start_date is not None else inferred_start
    end = _coerce_report_timestamp(end_date) if end_date is not None else inferred_end
    effective_holdout = holdout_months if holdout_months is not None else _estimate_holdout_months(data_end, end)

    t0 = time.time()
    alignment = verify_etf_data_alignment(data_dir)
    cfg = TPCBacktestConfig(initial_equity=initial_equity, data_dir=data_dir).with_overrides(mutations)
    bundle = load_tpc_replay_bundle(data_dir, start_date=start, end_date=end)
    result = run_tpc_independent(bundle.data, cfg, indicator_cache={})
    metrics = summarize_strategy("TPC", result, initial_equity, start, end, time.time() - t0)
    report = build_strategy_full_diagnostics(
        "TPC",
        result,
        metrics,
        alignment=alignment,
        start=start,
        end=end,
        data_end=data_end,
        holdout_months=effective_holdout,
        initial_equity=initial_equity,
    )
    report = report.replace("TPC BASELINE FULL DIAGNOSTICS", title, 1)
    report = report.replace("BASELINE RESULT", "OPTIMISED CONFIG RESULT", 1)
    report = report.replace("Promotable baseline:", "Promotable optimised config:", 1)
    report = report.replace("ALPHA_CAPTURED_BASELINE", "ALPHA_CAPTURED_CONFIG")
    config_lines = [
        "",
        "FINAL OPTIMISED CONFIG MUTATIONS",
        "-" * 80,
    ]
    if mutations:
        config_lines.extend(f"{key}: {value}" for key, value in sorted(mutations.items()))
    else:
        config_lines.append("No mutations; baseline TPC config.")
    config_lines.extend(["", f"Replay source fingerprint: {bundle.cache_source_fingerprint}"])
    return f"{report}\n{chr(10).join(config_lines)}"


def _strategy_report(name: str, result: Any, metrics: StrategyBaseline) -> list[str]:
    trades = sorted(list(getattr(result, "trades", [])), key=lambda t: getattr(t, "entry_time", None) or datetime.min)
    events = list(getattr(result, "decision_stream", []))
    lines = [
        f"{name} DIAGNOSTICS",
        "-" * 80,
        f"Alpha thesis: {ALPHA_NOTES[name]}",
        f"Net=${metrics.net_pnl:+,.2f}, PF={metrics.profit_factor:.2f}, avgR={metrics.avg_r:+.3f}, "
        f"trades/month={metrics.trades_per_month:.2f}, maxDD={metrics.max_dd_pct:.2f}%",
        f"Rolling 20-trade avgR={metrics.rolling20_avg_r:+.3f}, min={metrics.rolling20_min_r:+.3f}, "
        f"slope={metrics.rolling20_slope:+.5f}",
        f"Winner concentration: top 5 winners = {metrics.top5_winner_share:.1%} of gross wins",
    ]
    lines.extend(_alpha_capture_report(name, trades, metrics))
    if not trades:
        lines.append("No closed trades. Primary weakness: the default gates are too restrictive for this training window.")
        lines.extend(_event_funnel(events))
        return lines
    lines.extend(_cohort_table("By symbol", trades, lambda t: getattr(t, "symbol", "")))
    lines.extend(_cohort_table("By setup type", trades, lambda t: getattr(t, "entry_type", "") or "unknown"))
    lines.extend(_cohort_table("By setup grade", trades, lambda t: getattr(t, "regime_entry", "") or "unknown"))
    lines.extend(_cohort_table("By entry model", trades, lambda t: getattr(t, "leg_type", "") or "unknown"))
    lines.extend(_cohort_table("By exit reason", trades, lambda t: getattr(t, "exit_reason", "") or "unknown"))
    lines.extend(_time_stability(trades))
    lines.extend(_mfe_mae_report(trades))
    lines.extend(_event_funnel(events))
    return lines


def _bespoke_strategy_diagnostics(name: str, result: Any, metrics: StrategyBaseline) -> list[str]:
    trades = sorted(list(getattr(result, "trades", [])), key=lambda t: getattr(t, "entry_time", None) or datetime.min)
    events = list(getattr(result, "decision_stream", []))
    if name == "TPC":
        return _tpc_bespoke_diagnostics(trades, events, metrics)
    return []


def _tpc_bespoke_diagnostics(trades: list[Any], events: list[dict[str, Any]], metrics: StrategyBaseline) -> list[str]:
    alpha = _alpha_capture_payload("TPC", trades, metrics)
    setup_types = Counter(str(getattr(t, "entry_type", "") or "unknown") for t in trades)
    has_classic = any("classic" in key for key in setup_types)
    has_shallow = any("shallow" in key for key in setup_types)
    has_second_entry = any("second" in key for key in setup_types)
    if has_classic and has_shallow:
        cohort_strength = "Classic value pullbacks and shallow continuation retests are both represented."
    elif has_classic and has_second_entry:
        cohort_strength = "Classic value pullbacks dominate, with a second-entry branch now firing but still thin."
    elif has_classic:
        cohort_strength = "Classic value pullbacks are active, but alternative continuation branches are not adding sample yet."
    else:
        cohort_strength = "Continuation sample is not anchored by the intended classic value-pullback cohort yet."
    false_positive_rate = alpha["never_worked"] / max(metrics.trades, 1)
    lines = [
        "",
        "TPC BESPOKE CONTINUATION DIAGNOSTICS",
        "-" * 80,
        "Spec chain: 4h trend regime -> 1h controlled value pullback -> 15m/30m continuation confirmation -> T1/T2/runner.",
        _component_line("Expected-return capture", metrics.avg_r > 0.0 and metrics.net_return_pct > 0.0, f"net={metrics.net_return_pct:+.2f}%, avgR={metrics.avg_r:+.3f}"),
        _component_line("Optimisable frequency", metrics.trades_per_month >= ALPHA_BASELINE_TARGETS["TPC"]["trades_per_month"], f"{metrics.trades_per_month:.2f} trades/month"),
        _component_line("Raw continuation movement", alpha["avg_mfe_r"] >= 1.0, f"avg MFE={alpha['avg_mfe_r']:+.2f}R"),
        _component_line("Realisation quality", metrics.avg_r > 0.0 or alpha["right_then_lost"] == 0, f"right-then-lost={alpha['right_then_lost']}, never-worked={alpha['never_worked']}"),
    ]
    lines.extend(_event_funnel_summary_lines(events))
    lines.extend(_cohort_table("TPC pullback model coverage", trades, lambda t: getattr(t, "entry_type", "") or "unknown"))
    lines.extend(_cohort_table("TPC setup grade coverage", trades, lambda t: getattr(t, "regime_entry", "") or "unknown"))
    lines.extend(_cohort_table("TPC symbol x pullback model", trades, lambda t: f"{getattr(t, 'symbol', '')}:{getattr(t, 'entry_type', '') or 'unknown'}"))
    lines.extend(_mfe_bucket_table("TPC MFE realisation buckets", trades))
    lines.extend(
        [
            "",
            "TPC targeted diagnosis:",
            f"  + Strength to preserve: {cohort_strength}",
            f"  - Main weakness to repair: false positives still matter ({alpha['never_worked']} never-worked losers, {false_positive_rate:.1%} of trades) and {alpha['right_then_lost']} trades still gave back >=1R MFE.",
            "  - Phase-auto priority: broaden Type B/second-entry sampling only where MFE, avgR, and false-positive control hold together, then continue exit capture tuning.",
        ]
    )
    return lines


def _component_line(label: str, passed: bool, detail: str) -> str:
    return f"  {'PASS' if passed else 'CHECK'} | {label}: {detail}"


def _event_funnel_summary_lines(events: list[dict[str, Any]]) -> list[str]:
    counts = Counter(str(event.get("code", "")) for event in events)
    entries = counts.get("ENTRY_REQUESTED", 0)
    partials = counts.get("PARTIAL_EXIT_FILLED", 0)
    stops = counts.get("STOP_FILLED", 0)
    no_signal = counts.get("NO_SIGNAL", 0)
    total = sum(counts.values())
    return [
        "",
        "Signal funnel summary:",
        f"  no_signal={no_signal:,}, entries={entries:,}, partials={partials:,}, stops={stops:,}, events={total:,}",
        f"  entry density={entries / max(total, 1):.4%}, partials per entry={partials / max(entries, 1):.2f}, stops per entry={stops / max(entries, 1):.2f}",
    ]


def _mfe_bucket_table(title: str, trades: list[Any]) -> list[str]:
    buckets: dict[str, list[Any]] = {
        "never_0_0.5R": [],
        "worked_0.5_1R": [],
        "worked_1_2R": [],
        "worked_2R_plus": [],
    }
    for trade in trades:
        mfe = float(getattr(trade, "mfe_r", 0.0) or 0.0)
        if mfe < 0.5:
            buckets["never_0_0.5R"].append(trade)
        elif mfe < 1.0:
            buckets["worked_0.5_1R"].append(trade)
        elif mfe < 2.0:
            buckets["worked_1_2R"].append(trade)
        else:
            buckets["worked_2R_plus"].append(trade)
    lines = ["", f"{title}:"]
    lines.append(f"  {'Bucket':20s} {'N':>5s} {'WR':>6s} {'PF':>7s} {'AvgR':>8s} {'TotR':>9s} {'PnL':>11s}")
    for key, group in buckets.items():
        stats = _trade_stats(group)
        pf = "inf" if np.isinf(stats["pf"]) else f"{stats['pf']:.2f}"
        lines.append(
            f"  {key:20s} {stats['n']:5d} {stats['wr']:5.0%} {pf:>7s} "
            f"{stats['avg_r']:+8.3f} {stats['total_r']:+9.2f} ${stats['pnl']:+10,.2f}"
        )
    return lines


def _dominant_share(trades: list[Any], key_fn: Callable[[Any], str]) -> float:
    if not trades:
        return 0.0
    counts = Counter(str(key_fn(trade)) for trade in trades)
    return counts.most_common(1)[0][1] / len(trades) if counts else 0.0


def _dominant_label(trades: list[Any], key_fn: Callable[[Any], str]) -> str:
    if not trades:
        return "no trades"
    counts = Counter(str(key_fn(trade)) for trade in trades)
    if not counts:
        return "no labelled trades"
    key, count = counts.most_common(1)[0]
    return f"{key}={count}/{len(trades)} ({count / len(trades):.1%})"


def _alpha_capture_report(name: str, trades: list[Any], metrics: StrategyBaseline) -> list[str]:
    payload = _alpha_capture_payload(name, trades, metrics)
    lines = [
        "",
        "Alpha capture assessment:",
        f"  Verdict: {payload['verdict']}",
    ]
    if payload["strengths"]:
        lines.append("  Strengths:")
        lines.extend(f"    + {item}" for item in payload["strengths"])
    if payload["weaknesses"]:
        lines.append("  Weaknesses:")
        lines.extend(f"    - {item}" for item in payload["weaknesses"])
    lines.append(f"  Optimisation implication: {payload['optimisation_implication']}")
    return lines


def _alpha_capture_payload(name: str, trades: list[Any], metrics: StrategyBaseline) -> dict[str, Any]:
    targets = ALPHA_BASELINE_TARGETS[name]
    strengths: list[str] = []
    weaknesses: list[str] = []
    stats = _trade_stats(trades)
    mfe = np.asarray([float(getattr(t, "mfe_r", 0.0) or 0.0) for t in trades], dtype=float)
    rs = np.asarray([float(getattr(t, "r_multiple", 0.0) or 0.0) for t in trades], dtype=float)
    avg_mfe = float(np.mean(mfe)) if len(mfe) else 0.0
    never_worked = int(np.sum((mfe < 0.5) & (rs <= 0.0))) if len(trades) else 0
    right_then_lost = int(np.sum((mfe >= 1.0) & (rs <= 0.0))) if len(trades) else 0
    symbols = Counter(str(getattr(t, "symbol", "")) for t in trades)
    setup_types = Counter(str(getattr(t, "entry_type", "") or "unknown") for t in trades)
    grades = Counter(str(getattr(t, "regime_entry", "") or "unknown") for t in trades)
    exit_reasons = Counter(str(getattr(t, "exit_reason", "") or "unknown") for t in trades)

    if metrics.net_return_pct > 0.0 and metrics.avg_r > 0.0:
        strengths.append("Expected return and per-trade expectancy are positive on the current training window.")
    else:
        weaknesses.append("Expected return is not captured yet: net return and average R are both below the promotion target.")
    if metrics.trades_per_month >= targets["trades_per_month"]:
        strengths.append(f"Trade frequency is usable for optimisation at {metrics.trades_per_month:.2f} trades/month.")
    else:
        weaknesses.append(f"Trade frequency is thin at {metrics.trades_per_month:.2f} trades/month, so parameter changes need sample-size caution.")
    if avg_mfe >= targets["mfe_r"]:
        strengths.append(f"Signals are finding favourable movement: average MFE is {avg_mfe:+.2f}R.")
    elif trades:
        weaknesses.append(f"Average MFE is only {avg_mfe:+.2f}R, suggesting entries are not yet reaching the intended move.")
    if right_then_lost:
        weaknesses.append(f"{right_then_lost} trades moved at least +1R before finishing flat or worse, highlighting exit or stop-management giveback.")
    if trades and never_worked / max(len(trades), 1) >= 0.35:
        weaknesses.append(f"{never_worked} losers never reached +0.5R, pointing to false positives in the entry funnel.")
    if metrics.top5_winner_share >= 0.75 and stats["n"] >= 10:
        weaknesses.append(f"Gross wins are concentrated: the top 5 winners contribute {metrics.top5_winner_share:.1%} of gross wins.")
    if symbols:
        dominant_symbol, dominant_n = symbols.most_common(1)[0]
        if dominant_n / max(len(trades), 1) >= 0.80:
            weaknesses.append(f"Symbol capture is unbalanced: {dominant_symbol} contributes {dominant_n}/{len(trades)} trades.")

    if name == "TPC":
        _extend_tpc_alpha(strengths, weaknesses, setup_types, exit_reasons, avg_mfe, metrics)

    verdict = _alpha_verdict(metrics, strengths, weaknesses, len(trades))
    return {
        "verdict": verdict,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "avg_mfe_r": avg_mfe,
        "right_then_lost": right_then_lost,
        "never_worked": never_worked,
        "optimisation_implication": _alpha_optimisation_implication(name, metrics, avg_mfe, right_then_lost, never_worked),
    }


def _extend_tpc_alpha(
    strengths: list[str],
    weaknesses: list[str],
    setup_types: Counter,
    exit_reasons: Counter,
    avg_mfe: float,
    metrics: StrategyBaseline,
) -> None:
    if any("classic" in key for key in setup_types) and any("shallow" in key for key in setup_types):
        strengths.append("The current config captures both classic value pullbacks and shallow continuation retests, matching the frequency objective in swing_3.md.")
    else:
        weaknesses.append("The pullback mix is narrow; the current config is not yet testing both Type A and Type B continuation edges well.")
    if exit_reasons and exit_reasons.most_common(1)[0][0].upper() == "STOP" and exit_reasons.most_common(1)[0][1] / max(sum(exit_reasons.values()), 1) > 0.75:
        weaknesses.append("Most trades end via stop, so pullback entries are not being converted into the intended T1/T2/runner pathway.")
    if avg_mfe >= 1.0 and metrics.avg_r < 0.0:
        strengths.append("The continuation thesis is partially present: trades often move favourably before final realised R turns negative.")


def _alpha_verdict(metrics: StrategyBaseline, strengths: list[str], weaknesses: list[str], trade_count: int) -> str:
    if trade_count < 10:
        return "LOW_SAMPLE_ALPHA_UNPROVEN"
    if metrics.net_return_pct > 0.0 and metrics.avg_r > 0.0 and metrics.trades_per_month >= 0.75:
        return "ALPHA_CAPTURED_BASELINE"
    if any("favourable movement" in item or "raw movement" in item or "partially present" in item for item in strengths):
        return "RAW_ALPHA_PRESENT_REALISATION_WEAK"
    if weaknesses:
        return "ALPHA_NOT_CAPTURED_YET"
    return "MIXED"


def _alpha_optimisation_implication(
    name: str,
    metrics: StrategyBaseline,
    avg_mfe: float,
    right_then_lost: int,
    never_worked: int,
) -> str:
    if name == "TPC" and avg_mfe >= 1.0 and metrics.avg_r < 0.0:
        return "Prioritise exit sequencing, stop movement, and T1/T2 calibration before aggressively loosening entry gates."
    if never_worked > right_then_lost:
        return "Focus first on entry filters and false-positive rejection."
    return "Rank candidates by net return, average R, and trades/month while treating drawdown as a guardrail."


def _cohort_table(title: str, trades: list[Any], key_fn: Callable[[Any], str], min_count: int = 1) -> list[str]:
    groups: dict[str, list[Any]] = defaultdict(list)
    for trade in trades:
        groups[str(key_fn(trade))].append(trade)
    rows = []
    for key, group in groups.items():
        if len(group) >= min_count:
            rows.append((key, _trade_stats(group)))
    rows.sort(key=lambda item: (-item[1]["n"], item[0]))
    lines = [f"", f"{title}:"]
    lines.append(f"  {'Cohort':24s} {'N':>5s} {'WR':>6s} {'PF':>7s} {'AvgR':>8s} {'TotR':>9s} {'PnL':>11s}")
    for key, stats in rows[:12]:
        pf = "inf" if np.isinf(stats["pf"]) else f"{stats['pf']:.2f}"
        lines.append(
            f"  {key[:24]:24s} {stats['n']:5d} {stats['wr']:5.0%} {pf:>7s} "
            f"{stats['avg_r']:+8.3f} {stats['total_r']:+9.2f} ${stats['pnl']:+10,.2f}"
        )
    return lines


def _time_stability(trades: list[Any]) -> list[str]:
    by_year: dict[str, list[Any]] = defaultdict(list)
    by_month: dict[str, list[Any]] = defaultdict(list)
    for trade in trades:
        ts = pd.Timestamp(getattr(trade, "entry_time", None))
        if pd.isna(ts):
            continue
        by_year[str(ts.year)].append(trade)
        by_month[ts.strftime("%Y-%m")].append(trade)
    lines = ["", "Time stability:"]
    for label, groups in (("Year", by_year), ("Worst months", by_month)):
        rows = [(key, _trade_stats(group)) for key, group in groups.items()]
        if label == "Worst months":
            rows = sorted(rows, key=lambda item: item[1]["pnl"])[:6]
        else:
            rows = sorted(rows, key=lambda item: item[0])
        lines.append(f"  {label}:")
        for key, stats in rows:
            lines.append(
                f"    {key:8s} N={stats['n']:3d} avgR={stats['avg_r']:+.3f} "
                f"totR={stats['total_r']:+.2f} PnL=${stats['pnl']:+,.2f}"
            )
    return lines


def _mfe_mae_report(trades: list[Any]) -> list[str]:
    mfe = np.asarray([float(getattr(t, "mfe_r", 0.0) or 0.0) for t in trades])
    mae = np.asarray([float(getattr(t, "mae_r", 0.0) or 0.0) for t in trades])
    rs = np.asarray([float(getattr(t, "r_multiple", 0.0) or 0.0) for t in trades])
    right_then_lost = int(np.sum((mfe >= 1.0) & (rs <= 0.0)))
    never_worked = int(np.sum((mfe < 0.5) & (rs <= 0.0)))
    lines = [
        "",
        "Excursion quality:",
        f"  Avg MFE={float(np.mean(mfe)):+.3f}R, avg MAE={float(np.mean(mae)):+.3f}R, "
        f"median MFE={float(np.median(mfe)):+.3f}R",
        f"  Right-then-lost trades (MFE >= 1R and final <= 0R): {right_then_lost}",
        f"  Never-worked losers (MFE < 0.5R and final <= 0R): {never_worked}",
    ]
    return lines


def _event_funnel(events: list[dict[str, Any]]) -> list[str]:
    counts = Counter(str(event.get("code", "")) for event in events)
    symbol_counts = Counter(str(event.get("symbol", "")) for event in events if event.get("code") == "ENTRY_REQUESTED")
    lines = ["", "Decision event funnel:"]
    if not counts:
        lines.append("  No decision events captured.")
        return lines
    total = sum(counts.values())
    for code, count in counts.most_common(12):
        lines.append(f"  {code:24s} {count:8d} ({count / total:6.1%})")
    if symbol_counts:
        lines.append("  Entries requested by symbol: " + ", ".join(f"{k}={v}" for k, v in sorted(symbol_counts.items())))
    return lines


def _portfolio_style_snapshot(results: dict[str, Any], baselines: dict[str, StrategyBaseline]) -> list[str]:
    lines = ["PORTFOLIO-STYLE SNAPSHOT", "-" * 80]
    total_trades = sum(m.trades for m in baselines.values())
    total_r = sum(m.total_r for m in baselines.values())
    total_pnl = sum(m.net_pnl for m in baselines.values())
    lines.append(f"Independent-strategy total: trades={total_trades}, totalR={total_r:+.2f}, net=${total_pnl:+,.2f}")
    all_trades = []
    for result in results.values():
        all_trades.extend(getattr(result, "trades", []))
    lines.extend(_cohort_table("All ETF trades by symbol", all_trades, lambda t: getattr(t, "symbol", "")) if all_trades else ["No ETF trades."])
    return lines


def _optimization_readiness(baselines: dict[str, StrategyBaseline], alignment: list[AlignmentReport]) -> list[str]:
    lines = ["", "PHASED AUTO-OPTIMISATION READINESS", "-" * 80]
    lines.append("Use the same training window for phase-auto runs by passing --holdout-months 6, or the explicit --end-date above.")
    if all(item.passed for item in alignment):
        lines.append("Data readiness: PASS. No completed-bar leaks or structural 15m data defects detected.")
    else:
        lines.append("Data readiness: CHECK. Review the alignment section before optimisation.")
    for name, metrics in baselines.items():
        if metrics.trades < 20:
            verdict = "LOW_SAMPLE: loosen/diagnose signal gates before trusting objective scores."
        elif metrics.profit_factor < 1.0:
            verdict = "NEGATIVE_EDGE: start with entry/filter and exit diagnostics."
        elif metrics.max_dd_pct > 20.0:
            verdict = "RISK_HEAVY: prioritise exits, stops, and sizing phases."
        else:
            verdict = "OPTIMISABLE_BASELINE"
        lines.append(f"{name}: {verdict}")
    return lines


def _strategy_optimizer_readiness(name: str, result: Any, metrics: StrategyBaseline) -> list[str]:
    trades = sorted(list(getattr(result, "trades", [])), key=lambda t: getattr(t, "entry_time", None) or datetime.min)
    payload = _strategy_readiness_payload(name, metrics, trades)
    gates = [
        ("Positive net return", metrics.net_return_pct > 0.0, f"{metrics.net_return_pct:+.2f}%"),
        ("Positive avgR", metrics.avg_r > 0.0, f"{metrics.avg_r:+.3f}R"),
        ("PF >= 1.20", metrics.profit_factor >= 1.20, f"{metrics.profit_factor:.2f}"),
        (
            f"Frequency >= {ALPHA_BASELINE_TARGETS[name]['trades_per_month']:.2f}/month",
            metrics.trades_per_month >= ALPHA_BASELINE_TARGETS[name]["trades_per_month"],
            f"{metrics.trades_per_month:.2f}/month",
        ),
        ("Sample >= 30 trades", metrics.trades >= 30, str(metrics.trades)),
        ("MaxDD <= 15%", metrics.max_dd_pct <= 15.0, f"{metrics.max_dd_pct:.2f}%"),
    ]
    lines = ["", "OPTIMISATION READINESS", "-" * 80]
    lines.append(f"Promotable baseline: {'YES' if payload['promotable_baseline'] else 'NO'}")
    lines.append(f"Verdict: {payload['verdict']}")
    lines.append(f"Next optimisation focus: {payload['optimisation_implication']}")
    for label, passed, detail in gates:
        lines.append(_component_line(label, passed, detail))
    return lines


def _trade_stats(trades: list[Any]) -> dict[str, float]:
    pnls = np.asarray([float(getattr(t, "pnl_dollars", 0.0) or 0.0) for t in trades])
    rs = np.asarray([float(getattr(t, "r_multiple", 0.0) or 0.0) for t in trades])
    gross_profit = float(np.sum(pnls[pnls > 0])) if len(pnls) else 0.0
    gross_loss = abs(float(np.sum(pnls[pnls < 0]))) if len(pnls) else 0.0
    return {
        "n": len(trades),
        "wr": float(np.mean(pnls > 0)) if len(pnls) else 0.0,
        "pf": gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0),
        "avg_r": float(np.mean(rs)) if len(rs) else 0.0,
        "total_r": float(np.sum(rs)) if len(rs) else 0.0,
        "pnl": float(np.sum(pnls)) if len(pnls) else 0.0,
    }


def _rolling_stats(values: np.ndarray, window: int) -> dict[str, float]:
    if len(values) < window:
        avg = float(np.mean(values)) if len(values) else 0.0
        return {"last": avg, "min": avg, "slope": 0.0}
    rolling = np.asarray([float(np.mean(values[i : i + window])) for i in range(len(values) - window + 1)])
    x = np.arange(len(rolling))
    slope = float(np.polyfit(x, rolling, 1)[0]) if len(rolling) >= 3 else 0.0
    return {"last": float(rolling[-1]), "min": float(np.min(rolling)), "slope": slope}


def _max_drawdown(equity: np.ndarray) -> tuple[float, float]:
    if len(equity) < 2:
        return 0.0, 0.0
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    dd_pct = np.where(peak > 0, dd / peak, 0.0)
    return float(np.max(dd_pct) * 100.0), float(np.max(dd))


def _daily_sharpe(equity: np.ndarray, timestamps: np.ndarray) -> float:
    if len(equity) < 2 or len(timestamps) != len(equity):
        return 0.0
    series = pd.Series(equity, index=pd.to_datetime(timestamps))
    daily = series.resample("1D").last().dropna()
    returns = daily.pct_change().dropna()
    std = float(returns.std())
    if len(returns) < 2 or std <= 0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(252.0))


def _read_index(path: Path) -> pd.DatetimeIndex:
    df = pd.read_parquet(path)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return idx


def _read_bars(path: Path) -> pd.DataFrame:
    df = ensure_utc_index(pd.read_parquet(path))
    df.columns = df.columns.str.lower()
    return df


def _resample_to_hourly_start(df15: pd.DataFrame) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df15.columns:
        agg["volume"] = "sum"
    return df15.resample("1h", label="left", closed="left").agg(agg).dropna(subset=["open", "close"])


def _future_intraday_count(
    lower_index: pd.DatetimeIndex,
    higher_index: pd.DatetimeIndex,
    idx_map: np.ndarray,
    lower_freq: str,
    higher_freq: str,
) -> int:
    valid = idx_map >= 0
    if not np.any(valid):
        return 0
    lower_close = pd.DatetimeIndex(lower_index[valid]) + pd.Timedelta(lower_freq)
    higher_close = pd.DatetimeIndex(higher_index[idx_map[valid]]) + pd.Timedelta(higher_freq)
    return int(np.sum(higher_close.values > lower_close.values))


def _future_daily_count(
    lower_index: pd.DatetimeIndex,
    daily_index: pd.DatetimeIndex,
    idx_map: np.ndarray,
) -> int:
    valid = idx_map >= 0
    if not np.any(valid):
        return 0
    lower_dates = pd.DatetimeIndex(lower_index[valid]).normalize().values.astype("datetime64[D]")
    daily_dates = pd.DatetimeIndex(daily_index[idx_map[valid]]).normalize().values.astype("datetime64[D]")
    return int(np.sum(daily_dates >= lower_dates))


def write_strategy_artifacts(
    *,
    output_root: Path,
    alignment: list[AlignmentReport],
    baselines: dict[str, StrategyBaseline],
    results: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    data_end: pd.Timestamp,
    holdout_months: int,
    initial_equity: float,
) -> list[Path]:
    written: list[Path] = []
    for name in ("TPC",):
        strategy_dir = output_root / name.lower() / "baseline"
        strategy_dir.mkdir(parents=True, exist_ok=True)
        result = results[name]
        metrics = baselines[name]

        full_report = build_strategy_full_diagnostics(
            name,
            result,
            metrics,
            alignment=alignment,
            start=start,
            end=end,
            data_end=data_end,
            holdout_months=holdout_months,
            initial_equity=initial_equity,
        )
        diagnostics_path = strategy_dir / "full_diagnostics.txt"
        diagnostics_path.write_text(full_report, encoding="utf-8")
        written.append(diagnostics_path)

        baseline_path = strategy_dir / "baseline_results.json"
        baseline_path.write_text(
            json.dumps(
                _strategy_artifact_payload(
                    name,
                    result,
                    metrics,
                    alignment=alignment,
                    start=start,
                    end=end,
                    data_end=data_end,
                    holdout_months=holdout_months,
                    initial_equity=initial_equity,
                ),
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        written.append(baseline_path)

        cohort_path = strategy_dir / "cohort_diagnostics.json"
        cohort_path.write_text(
            json.dumps(_cohort_diagnostics_payload(result), indent=2, default=str),
            encoding="utf-8",
        )
        written.append(cohort_path)

        trades_path = strategy_dir / "trade_records.csv"
        _trade_records_frame(result).to_csv(trades_path, index=False)
        written.append(trades_path)

        manifest_path = strategy_dir / "run_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "strategy": name,
                    "artifact_type": "baseline",
                    "training_start": start.isoformat(),
                    "training_end": end.isoformat(),
                    "holdout_months": holdout_months,
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "files": {
                        "full_diagnostics": str(diagnostics_path),
                        "baseline_results": str(baseline_path),
                        "cohort_diagnostics": str(cohort_path),
                        "trade_records": str(trades_path),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        written.append(manifest_path)
    return written


def _strategy_artifact_payload(
    name: str,
    result: Any,
    metrics: StrategyBaseline,
    *,
    alignment: list[AlignmentReport],
    start: pd.Timestamp,
    end: pd.Timestamp,
    data_end: pd.Timestamp,
    holdout_months: int,
    initial_equity: float,
) -> dict[str, Any]:
    trades = sorted(list(getattr(result, "trades", [])), key=lambda t: getattr(t, "entry_time", None) or datetime.min)
    events = list(getattr(result, "decision_stream", []))
    return {
        "strategy": name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "training_start": start.isoformat(),
        "training_end": end.isoformat(),
        "latest_common_intraday_data": data_end.isoformat(),
        "holdout_months": holdout_months,
        "initial_equity": initial_equity,
        "alpha_thesis": ALPHA_NOTES[name],
        "baseline": asdict(metrics),
        "alpha_capture": _alpha_capture_payload(name, trades, metrics),
        "alignment": [asdict(item) | {"passed": item.passed} for item in alignment],
        "event_funnel": _event_funnel_payload(events),
        "cohorts": _cohort_dimensions_payload(trades),
        "excursion": _excursion_payload(trades),
        "optimizer_readiness": _strategy_readiness_payload(name, metrics, trades),
    }


def _cohort_diagnostics_payload(result: Any) -> dict[str, Any]:
    trades = sorted(list(getattr(result, "trades", [])), key=lambda t: getattr(t, "entry_time", None) or datetime.min)
    return {
        "event_funnel": _event_funnel_payload(list(getattr(result, "decision_stream", []))),
        "cohorts": _cohort_dimensions_payload(trades),
        "excursion": _excursion_payload(trades),
    }


def _cohort_dimensions_payload(trades: list[Any]) -> dict[str, dict[str, dict[str, float]]]:
    dimensions: dict[str, Callable[[Any], str]] = {
        "symbol": lambda t: getattr(t, "symbol", "") or "unknown",
        "direction": lambda t: "LONG" if int(getattr(t, "direction", 0) or 0) > 0 else "SHORT",
        "setup_type": lambda t: getattr(t, "entry_type", "") or "unknown",
        "setup_grade": lambda t: getattr(t, "regime_entry", "") or "unknown",
        "entry_model": lambda t: getattr(t, "leg_type", "") or "unknown",
        "exit_reason": lambda t: getattr(t, "exit_reason", "") or "unknown",
        "year": lambda t: str(pd.Timestamp(getattr(t, "entry_time", None)).year),
    }
    return {name: _cohort_stats_payload(trades, key_fn) for name, key_fn in dimensions.items()}


def _cohort_stats_payload(trades: list[Any], key_fn: Callable[[Any], str]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[Any]] = defaultdict(list)
    for trade in trades:
        groups[str(key_fn(trade))].append(trade)
    return {key: _json_trade_stats(group) for key, group in sorted(groups.items())}


def _json_trade_stats(trades: list[Any]) -> dict[str, float]:
    stats = _trade_stats(trades)
    mfe = np.asarray([float(getattr(t, "mfe_r", 0.0) or 0.0) for t in trades], dtype=float)
    mae = np.asarray([float(getattr(t, "mae_r", 0.0) or 0.0) for t in trades], dtype=float)
    rs = np.asarray([float(getattr(t, "r_multiple", 0.0) or 0.0) for t in trades], dtype=float)
    return {
        **{key: float(value) if isinstance(value, (float, np.floating)) else value for key, value in stats.items()},
        "avg_mfe_r": float(np.mean(mfe)) if len(mfe) else 0.0,
        "avg_mae_r": float(np.mean(mae)) if len(mae) else 0.0,
        "right_then_lost": int(np.sum((mfe >= 1.0) & (rs <= 0.0))) if len(trades) else 0,
        "never_worked": int(np.sum((mfe < 0.5) & (rs <= 0.0))) if len(trades) else 0,
    }


def _event_funnel_payload(events: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(event.get("code", "")) for event in events)
    entries_by_symbol = Counter(str(event.get("symbol", "")) for event in events if event.get("code") == "ENTRY_REQUESTED")
    total = sum(counts.values())
    return {
        "total_events": total,
        "counts": dict(counts.most_common()),
        "entries_by_symbol": dict(sorted(entries_by_symbol.items())),
        "entry_density": counts.get("ENTRY_REQUESTED", 0) / max(total, 1),
        "partials_per_entry": counts.get("PARTIAL_EXIT_FILLED", 0) / max(counts.get("ENTRY_REQUESTED", 0), 1),
        "stops_per_entry": counts.get("STOP_FILLED", 0) / max(counts.get("ENTRY_REQUESTED", 0), 1),
    }


def _excursion_payload(trades: list[Any]) -> dict[str, Any]:
    mfe = np.asarray([float(getattr(t, "mfe_r", 0.0) or 0.0) for t in trades], dtype=float)
    mae = np.asarray([float(getattr(t, "mae_r", 0.0) or 0.0) for t in trades], dtype=float)
    rs = np.asarray([float(getattr(t, "r_multiple", 0.0) or 0.0) for t in trades], dtype=float)
    return {
        "avg_mfe_r": float(np.mean(mfe)) if len(mfe) else 0.0,
        "median_mfe_r": float(np.median(mfe)) if len(mfe) else 0.0,
        "avg_mae_r": float(np.mean(mae)) if len(mae) else 0.0,
        "median_mae_r": float(np.median(mae)) if len(mae) else 0.0,
        "right_then_lost": int(np.sum((mfe >= 1.0) & (rs <= 0.0))) if len(trades) else 0,
        "never_worked": int(np.sum((mfe < 0.5) & (rs <= 0.0))) if len(trades) else 0,
        "mfe_2r_plus": int(np.sum(mfe >= 2.0)) if len(trades) else 0,
    }


def _strategy_readiness_payload(name: str, metrics: StrategyBaseline, trades: list[Any]) -> dict[str, Any]:
    alpha = _alpha_capture_payload(name, trades, metrics)
    return {
        "verdict": alpha["verdict"],
        "promotable_baseline": (
            metrics.net_return_pct > 0.0
            and metrics.avg_r > 0.0
            and metrics.profit_factor >= 1.2
            and metrics.trades_per_month >= ALPHA_BASELINE_TARGETS[name]["trades_per_month"]
        ),
        "optimisation_implication": alpha["optimisation_implication"],
    }


def _trade_records_frame(result: Any) -> pd.DataFrame:
    fields = [
        "symbol",
        "direction",
        "entry_type",
        "leg_type",
        "regime_entry",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "qty",
        "initial_stop",
        "r_multiple",
        "pnl_dollars",
        "pnl_points",
        "commission",
        "mfe_r",
        "mae_r",
        "bars_held",
        "exit_reason",
        "campaign_id",
        "score_entry",
        "quality_score",
        "signal_time",
        "fill_time",
        "signal_bar_index",
        "fill_bar_index",
    ]
    rows = []
    for trade in sorted(list(getattr(result, "trades", [])), key=lambda t: getattr(t, "entry_time", None) or datetime.min):
        rows.append({field: getattr(trade, field, "") for field in fields})
    return pd.DataFrame(rows, columns=fields)


def _json_payload(
    alignment: list[AlignmentReport],
    baselines: dict[str, StrategyBaseline],
    results: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    data_end: pd.Timestamp,
    holdout_months: int,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "training_start": start.isoformat(),
        "training_end": end.isoformat(),
        "latest_common_intraday_data": data_end.isoformat(),
        "holdout_months": holdout_months,
        "alignment": [asdict(item) | {"passed": item.passed} for item in alignment],
        "strategies": {name: asdict(metrics) for name, metrics in baselines.items()},
        "alpha_capture": {
            name: _alpha_capture_payload(
                name,
                sorted(list(getattr(results[name], "trades", [])), key=lambda t: getattr(t, "entry_time", None) or datetime.min),
                baselines[name],
            )
            for name in baselines
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--holdout-months", type=int, default=6)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--equity", type=float, default=100_000.0)
    parser.add_argument("--output", default=str(DEFAULT_REPORT))
    parser.add_argument("--summary-json", default=str(DEFAULT_JSON))
    parser.add_argument(
        "--strategy-output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root folder for per-strategy baseline/full-diagnostics artifacts.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    inferred_start, inferred_end, data_end = infer_training_window(data_dir, args.holdout_months)
    start = pd.Timestamp(args.start_date) if args.start_date else inferred_start
    end = pd.Timestamp(args.end_date) if args.end_date else inferred_end
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")

    output = Path(args.output)
    summary_json = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    alignment = verify_etf_data_alignment(data_dir)
    baselines, results = run_baselines(data_dir, start=start, end=end, initial_equity=args.equity)
    report = build_report(
        alignment=alignment,
        baselines=baselines,
        results=results,
        start=start,
        end=end,
        data_end=data_end,
        holdout_months=args.holdout_months,
        initial_equity=args.equity,
    )
    output.write_text(report, encoding="utf-8")
    summary_json.write_text(
        json.dumps(_json_payload(alignment, baselines, results, start, end, data_end, args.holdout_months), indent=2),
        encoding="utf-8",
    )
    strategy_paths = write_strategy_artifacts(
        output_root=Path(args.strategy_output_root),
        alignment=alignment,
        baselines=baselines,
        results=results,
        start=start,
        end=end,
        data_end=data_end,
        holdout_months=args.holdout_months,
        initial_equity=args.equity,
    )
    print(f"Wrote {output}")
    print(f"Wrote {summary_json}")
    for path in strategy_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
