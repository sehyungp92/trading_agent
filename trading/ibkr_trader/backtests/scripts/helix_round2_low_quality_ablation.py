"""Audit low-quality Helix round_2 acceptances on IS and holdout data.

The optimizer uses fast independent replay for candidate discovery, but the
strategy implementation notes require synchronized replay for headline checks.
This script therefore runs exact mutation variants through the synchronized
portfolio engine, and also reports independent metrics when requested.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "backtests" / "swing" / "data" / "raw"
ROUND_1_CONFIG = PROJECT_ROOT / "backtests" / "output" / "swing" / "helix" / "round_1" / "optimized_config.json"
ROUND_2_DIR = PROJECT_ROOT / "backtests" / "output" / "swing" / "helix" / "round_2"
ROUND_2_CONFIG = ROUND_2_DIR / "optimized_config.json"
IS_END_DATE = "2026-03-20"
HOLDOUT_START = datetime(2026, 3, 21, tzinfo=timezone.utc)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict) and isinstance(payload.get("mutations"), dict):
        return dict(payload["mutations"])
    if isinstance(payload, dict) and isinstance(payload.get("cumulative_mutations"), dict):
        return dict(payload["cumulative_mutations"])
    if isinstance(payload, dict):
        return dict(payload)
    raise TypeError(f"Unexpected JSON payload in {path}")


def _latest_swing_data_end() -> str:
    import pandas as pd

    latest = None
    for symbol in ("QQQ", "GLD"):
        path = DATA_DIR / f"{symbol}_1h.parquet"
        df = pd.read_parquet(path)
        if df.empty:
            continue
        ts = df.index[-1]
        if getattr(ts, "tzinfo", None) is None:
            ts = ts.tz_localize(timezone.utc)
        latest = ts if latest is None or ts > latest else latest
    if latest is None:
        raise RuntimeError(f"Could not detect latest data date under {DATA_DIR}")
    return latest.date().isoformat()


def _variant_map() -> dict[str, dict[str, Any]]:
    """Build exact mutation dictionaries for targeted ablations."""

    final = _load_json(ROUND_2_CONFIG)
    round1 = _load_json(ROUND_1_CONFIG)

    def with_updates(**updates: Any) -> dict[str, Any]:
        out = dict(final)
        out.update(updates)
        return out

    def round1_value(key: str) -> Any:
        if key not in round1:
            raise KeyError(f"{key} is not present in round_1 config")
        return round1[key]

    return {
        "current_final": dict(final),
        "add_bail_d_bars_10": with_updates(**{"param_overrides.CLASS_D_BAIL_BARS": 10}),
        "add_bail_d_bars_8": with_updates(**{"param_overrides.CLASS_D_BAIL_BARS": 8}),
        "add_bail_d_bars_6": with_updates(**{"param_overrides.CLASS_D_BAIL_BARS": 6}),
        "revert_sug_stall_onset_5_to_round1": with_updates(
            **{"param_overrides.TRAIL_STALL_ONSET": round1_value("param_overrides.TRAIL_STALL_ONSET")}
        ),
        "revert_trail_stall_rate_p20_to_phase3": with_updates(
            **{"param_overrides.TRAIL_STALL_RATE": 0.12}
        ),
        "revert_trail_fade_penalty_p10_to_phase3": with_updates(
            **{"param_overrides.TRAIL_FADE_PENALTY": 1.0}
        ),
        "revert_r_partial_5_p10_to_phase3": with_updates(
            **{"param_overrides.R_PARTIAL_5": 6.0}
        ),
        "revert_trail_r_div_p20_to_phase3": with_updates(
            **{"param_overrides.TRAIL_R_DIV": 7.0}
        ),
        "revert_low_quality_cluster": with_updates(
            **{
                "param_overrides.TRAIL_STALL_ONSET": round1_value("param_overrides.TRAIL_STALL_ONSET"),
                "param_overrides.TRAIL_STALL_RATE": 0.12,
                "param_overrides.TRAIL_FADE_PENALTY": 1.0,
                "param_overrides.R_PARTIAL_5": 6.0,
            }
        ),
        "add_bail_d_bars_10_and_revert_stall": with_updates(
            **{
                "param_overrides.CLASS_D_BAIL_BARS": 10,
                "param_overrides.TRAIL_STALL_ONSET": round1_value("param_overrides.TRAIL_STALL_ONSET"),
            }
        ),
    }


def _trade_net_pnl(trade: Any) -> float:
    if hasattr(trade, "net_pnl_dollars"):
        return float(getattr(trade, "net_pnl_dollars", 0.0) or 0.0)
    return float(getattr(trade, "pnl_dollars", 0.0) or 0.0) - float(getattr(trade, "commission", 0.0) or 0.0)


def _trade_net_r(trade: Any) -> float:
    if hasattr(trade, "net_r_multiple"):
        return float(getattr(trade, "net_r_multiple", 0.0) or 0.0)
    return float(getattr(trade, "r_multiple", 0.0) or 0.0)


def _entry_time(trade: Any) -> datetime:
    value = getattr(trade, "entry_time", None) or getattr(trade, "entry_dt", None)
    if value is None:
        raise ValueError(f"Trade has no entry timestamp: {trade!r}")
    if isinstance(value, np.datetime64):
        value = value.astype("datetime64[us]").astype(datetime)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _collect_trades(result: Any) -> list[Any]:
    trades: list[Any] = []
    for symbol, symbol_result in result.symbol_results.items():
        for trade in symbol_result.trades:
            if not getattr(trade, "symbol", ""):
                try:
                    trade.symbol = symbol
                except Exception:
                    pass
            trades.append(trade)
    return sorted(trades, key=_entry_time)


def _subset_metrics(trades: list[Any], initial_equity: float) -> dict[str, Any]:
    if not trades:
        return {
            "total_trades": 0,
            "profit_factor": 0.0,
            "net_return_pct": 0.0,
            "max_r_dd": 0.0,
            "exit_efficiency": 0.0,
            "waste_ratio": 0.0,
            "tail_pct": 0.0,
            "min_regime_pf": 0.0,
            "total_r": 0.0,
            "win_rate": 0.0,
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
            "trade_count_by_symbol": {},
        }

    rs = [_trade_net_r(t) for t in trades]
    pnls = [_trade_net_pnl(t) for t in trades]
    wins = [t for t in trades if _trade_net_r(t) > 0]
    losses = [t for t in trades if _trade_net_r(t) <= 0]
    gross_win_r = sum(_trade_net_r(t) for t in wins)
    gross_loss_r = abs(sum(_trade_net_r(t) for t in losses))
    gross_win_pnl = sum(p for p in pnls if p > 0)
    gross_loss_pnl = abs(sum(p for p in pnls if p < 0))

    def regime_pf(regime: str) -> float:
        regime_pnls = [p for p, t in zip(pnls, trades) if getattr(t, "regime_at_entry", "") == regime]
        if not regime_pnls:
            return 999.0
        w = sum(p for p in regime_pnls if p > 0)
        l = abs(sum(p for p in regime_pnls if p < 0))
        return w / l if l > 0 else 999.0

    cum_r = np.cumsum(rs)
    peak_r = np.maximum.accumulate(cum_r)
    max_r_dd = float(np.max(peak_r - cum_r)) if len(cum_r) else 0.0

    total_r = float(sum(rs))
    sum_mfe_pos = sum(float(getattr(t, "mfe_r", 0.0) or 0.0) for t in trades if float(getattr(t, "mfe_r", 0.0) or 0.0) > 0)
    exit_eff = total_r / sum_mfe_pos if sum_mfe_pos > 0 else 0.0
    stale_r = abs(sum(_trade_net_r(t) for t in trades if getattr(t, "exit_reason", "") == "STALE"))
    short_hold_r = abs(sum(_trade_net_r(t) for t in trades if int(getattr(t, "bars_held", 0) or 0) <= 10 and _trade_net_r(t) < 0))
    big_winner_r = sum(_trade_net_r(t) for t in wins if _trade_net_r(t) >= 3.0)
    symbol_counts: dict[str, int] = {}
    for trade in trades:
        symbol = getattr(trade, "symbol", "") or "UNKNOWN"
        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

    return {
        "total_trades": len(trades),
        "profit_factor": gross_win_pnl / gross_loss_pnl if gross_loss_pnl > 0 else 999.0,
        "net_return_pct": sum(pnls) / initial_equity * 100.0,
        "max_r_dd": max_r_dd,
        "exit_efficiency": max(0.0, min(1.0, exit_eff)),
        "waste_ratio": max(0.0, 1.0 - ((stale_r + short_hold_r) / gross_win_r if gross_win_r > 0 else 0.0)),
        "tail_pct": big_winner_r / gross_win_r if gross_win_r > 0 else 0.0,
        "bull_pf": regime_pf("BULL"),
        "bear_pf": regime_pf("BEAR"),
        "min_regime_pf": min(regime_pf("BULL"), regime_pf("BEAR")),
        "total_r": total_r,
        "gross_win_r": gross_win_r,
        "gross_loss_r": gross_loss_r,
        "stale_r": stale_r,
        "short_hold_r": short_hold_r,
        "big_winner_r": big_winner_r,
        "win_rate": len(wins) / len(trades) * 100.0,
        "avg_win_r": gross_win_r / len(wins) if wins else 0.0,
        "avg_loss_r": -gross_loss_r / len(losses) if losses else 0.0,
        "trade_count_by_symbol": symbol_counts,
    }


def _phase4_score(metrics: dict[str, Any]) -> dict[str, Any]:
    from backtests.swing.auto.helix.plugin import score_phase_metrics
    from backtests.swing.auto.helix.scoring import HelixMetrics, composite_score

    fields = HelixMetrics.__dataclass_fields__
    clean = {key: metrics.get(key, 0.0) for key in fields}
    clean["total_trades"] = int(clean.get("total_trades", 0) or 0)
    metric_obj = HelixMetrics(**clean)
    no_rejects = {
        "min_trades": 0,
        "min_pf": 0.0,
        "max_r_dd": 999.0,
        "min_tail_pct": 0.0,
        "min_regime_pf": 0.0,
    }
    return {
        "default_no_hard_reject": asdict(composite_score(metric_obj, hard_rejects=no_rejects)),
        "phase4_no_hard_reject": asdict(score_phase_metrics(4, metric_obj, hard_rejects=no_rejects)),
    }


def _run_variant(args: tuple[str, dict[str, Any], float, str, list[str]]) -> dict[str, Any]:
    name, mutations, equity, data_end, modes = args

    from backtests.swing.auto.config_mutator import mutate_helix_config
    from backtests.swing.auto.helix.scoring import extract_helix_metrics
    from backtests.swing.config_helix import HelixBacktestConfig
    from backtests.swing.data.replay_cache import load_helix_replay_bundle
    from backtests.swing.engine.helix_portfolio_engine import run_helix_independent, run_helix_synchronized

    base_config_is = HelixBacktestConfig(
        initial_equity=equity,
        data_dir=DATA_DIR,
        end_date=IS_END_DATE,
        track_shadows=False,
    )
    base_config_full = HelixBacktestConfig(
        initial_equity=equity,
        data_dir=DATA_DIR,
        end_date=data_end,
        track_shadows=False,
    )
    config_is = mutate_helix_config(base_config_is, mutations)
    config_full = mutate_helix_config(base_config_full, mutations)
    data_is = load_helix_replay_bundle(
        config_is.symbols,
        config_is.data_dir,
        end_date=config_is.end_date,
    )
    data_full = load_helix_replay_bundle(
        config_full.symbols,
        config_full.data_dir,
        end_date=config_full.end_date,
    )

    runner = {
        "independent": run_helix_independent,
        "synchronized": run_helix_synchronized,
    }
    output: dict[str, Any] = {
        "variant": name,
        "equity": equity,
        "mutations": mutations,
        "modes": {},
    }

    for mode in modes:
        run = runner[mode]
        is_result = run(data_is.data, config_is)
        is_metrics = asdict(extract_helix_metrics(is_result, equity))
        full_result = run(data_full.data, config_full)
        full_trades = _collect_trades(full_result)
        holdout_trades = [t for t in full_trades if _entry_time(t) >= HOLDOUT_START]
        pre_holdout_trades = [t for t in full_trades if _entry_time(t) < HOLDOUT_START]
        mode_payload: dict[str, Any] = {
            "is_official_metrics": is_metrics,
            "is_official_score": _phase4_score(is_metrics),
            "pre_holdout_trade_window_metrics": _subset_metrics(pre_holdout_trades, equity),
            "holdout_trade_window_metrics": _subset_metrics(holdout_trades, equity),
            "full_trade_count": len(full_trades),
            "holdout_trade_count": len(holdout_trades),
        }
        heat_stats = getattr(full_result, "heat_stats", None)
        if heat_stats is not None:
            mode_payload["full_heat_stats"] = asdict(heat_stats)
        output["modes"][mode] = mode_payload

    return output


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out if math.isfinite(out) else 0.0


def _delta_metrics(current: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    keys = [
        "net_return_pct",
        "profit_factor",
        "total_r",
        "max_r_dd",
        "exit_efficiency",
        "waste_ratio",
        "tail_pct",
        "min_regime_pf",
        "total_trades",
        "win_rate",
    ]
    return {
        key: _safe_float(candidate.get(key)) - _safe_float(current.get(key))
        for key in keys
    }


def _summarize(results: list[dict[str, Any]], modes: list[str]) -> dict[str, Any]:
    by_variant = {item["variant"]: item for item in results}
    current = by_variant["current_final"]
    comparisons: dict[str, Any] = {}
    for variant, item in by_variant.items():
        if variant == "current_final":
            continue
        comparisons[variant] = {}
        for mode in modes:
            cur_mode = current["modes"][mode]
            item_mode = item["modes"][mode]
            comparisons[variant][mode] = {
                "is_official_delta": _delta_metrics(
                    cur_mode["is_official_metrics"],
                    item_mode["is_official_metrics"],
                ),
                "holdout_delta": _delta_metrics(
                    cur_mode["holdout_trade_window_metrics"],
                    item_mode["holdout_trade_window_metrics"],
                ),
                "pre_holdout_trade_window_delta": _delta_metrics(
                    cur_mode["pre_holdout_trade_window_metrics"],
                    item_mode["pre_holdout_trade_window_metrics"],
                ),
            }
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equity", type=float, default=25_000.0)
    parser.add_argument("--data-end", default="", help="Holdout data end date; defaults to latest QQQ/GLD 1h data.")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--mode",
        action="append",
        choices=["independent", "synchronized"],
        help="Backtest mode to run. Repeat for both. Defaults to synchronized.",
    )
    parser.add_argument("--variant", action="append", help="Variant name to run. Defaults to all targeted variants.")
    args = parser.parse_args()

    data_end = args.data_end or _latest_swing_data_end()
    modes = args.mode or ["synchronized"]
    variants = _variant_map()
    if args.variant:
        selected = {}
        for name in args.variant:
            if name not in variants:
                raise KeyError(f"Unknown variant {name!r}. Known: {', '.join(sorted(variants))}")
            selected[name] = variants[name]
        variants = selected
        if "current_final" not in variants:
            variants = {"current_final": _variant_map()["current_final"], **variants}

    tasks = [(name, mutations, args.equity, data_end, modes) for name, mutations in variants.items()]
    if args.max_workers <= 1 or len(tasks) == 1:
        results = [_run_variant(task) for task in tasks]
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        results = []
        with ProcessPoolExecutor(max_workers=min(args.max_workers, len(tasks))) as pool:
            futures = {pool.submit(_run_variant, task): task[0] for task in tasks}
            for future in as_completed(futures):
                name = futures[future]
                print(f"completed {name}", flush=True)
                results.append(future.result())
        order = {name: idx for idx, name in enumerate(variants)}
        results.sort(key=lambda item: order[item["variant"]])

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "equity": args.equity,
        "is_end_date": IS_END_DATE,
        "holdout_start": HOLDOUT_START.date().isoformat(),
        "data_end": data_end,
        "modes": modes,
        "round_1_config": str(ROUND_1_CONFIG),
        "round_2_config": str(ROUND_2_CONFIG),
        "results": results,
        "comparisons_vs_current_final": _summarize(results, modes),
        "notes": [
            "IS official metrics are computed from replay ending at 2026-03-20.",
            "Holdout metrics are closed-trade window metrics from entries on or after 2026-03-21.",
            "Synchronized mode is the docs-compliant headline mode; independent mode is the optimizer's fast candidate-discovery mode.",
        ],
    }

    out_path = ROUND_2_DIR / f"low_quality_acceptance_ablation_{int(args.equity)}_{'_'.join(modes)}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
