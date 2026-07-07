"""Compare selected swing portfolio candidates on the shared OOS window.

Runs the live-like unified portfolio replay over the full validation window,
then splits trades at the shared OOS cutoff. This preserves warmup and
pre-cutoff portfolio state while ranking the requested candidate set on OOS.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from backtests.shared.validation.oos_validation import _get_entry_time, _get_r_multiple
from backtests.swing.auto.config_mutator import mutate_unified_config
from backtests.swing.auto.portfolio_synergy.run_latest_two_rounds import _json_default
from backtests.swing.config_unified import UnifiedBacktestConfig
from backtests.swing.engine.unified_portfolio_engine import load_unified_data, run_unified


STARTING_EQUITY = 50_000.0
BASE_CONFIG = ROOT / "backtests" / "output" / "swing" / "portfolio_synergy" / "round_2" / "optimized_config.json"
DEFAULT_DATA_DIR = ROOT / "backtests" / "swing" / "data" / "raw"
BACKTEST_START = "2024-01-01"
OOS_CUTOFF = "2026-03-21"
DATA_END = "2026-05-01"

REQUESTED_CANDIDATES: list[tuple[str, dict[str, Any]]] = [
    ("baseline_round_2", {}),
    (
        "balanced_509_77_atrss_71",
        {
            "atrss.unit_risk_pct": 0.0165,
            "atrss.max_heat_R": 2.15,
            "helix.unit_risk_pct": 0.0130,
            "helix.max_heat_R": 2.10,
            "tpc.unit_risk_pct": 0.0050,
            "tpc.max_heat_R": 4.00,
            "tpc_param.all.max_risk_pct": 0.020,
            "tpc_param.all.risk_a_plus_pct": 0.020,
            "tpc_param.all.risk_a_pct": 0.012,
            "tpc_param.all.risk_b_pct": 0.009,
        },
    ),
    (
        "target60_helix_led",
        {
            "atrss.unit_risk_pct": 0.0145,
            "atrss.max_heat_R": 1.95,
            "helix.unit_risk_pct": 0.0185,
            "helix.max_heat_R": 2.80,
            "tpc.unit_risk_pct": 0.0065,
            "tpc.max_heat_R": 4.25,
            "tpc_param.all.max_risk_pct": 0.026,
            "tpc_param.all.risk_a_plus_pct": 0.026,
            "tpc_param.all.risk_a_pct": 0.016,
            "tpc_param.all.risk_b_pct": 0.012,
        },
    ),
    (
        "refined60_restore_atrss_heat",
        {
            "atrss.unit_risk_pct": 0.0135,
            "atrss.max_heat_R": 2.15,
            "helix.unit_risk_pct": 0.0185,
            "helix.max_heat_R": 2.60,
            "tpc.unit_risk_pct": 0.0065,
            "tpc.max_heat_R": 4.25,
            "tpc_param.all.max_risk_pct": 0.026,
            "tpc_param.all.risk_a_plus_pct": 0.026,
            "tpc_param.all.risk_a_pct": 0.016,
            "tpc_param.all.risk_b_pct": 0.012,
        },
    ),
    (
        "refined60_scaled95",
        {
            "atrss.unit_risk_pct": 0.01375,
            "atrss.max_heat_R": 2.15,
            "helix.unit_risk_pct": 0.0175,
            "helix.max_heat_R": 2.50,
            "tpc.unit_risk_pct": 0.0062,
            "tpc.max_heat_R": 4.10,
            "tpc_param.all.max_risk_pct": 0.025,
            "tpc_param.all.risk_a_plus_pct": 0.025,
            "tpc_param.all.risk_a_pct": 0.015,
            "tpc_param.all.risk_b_pct": 0.011,
        },
    ),
]

_DATA = None


def _dt(value: str) -> datetime:
    return datetime.combine(date.fromisoformat(value), time.min)


def _end_exclusive(value: str) -> datetime:
    return datetime.combine(date.fromisoformat(value) + timedelta(days=1), time.min)


def _init_worker(data_dir: str, equity: float, start: str, end: str) -> None:
    global _DATA
    seed = UnifiedBacktestConfig(
        initial_equity=float(equity),
        data_dir=Path(data_dir),
        start_date=start,
        end_date=end,
    )
    _DATA = load_unified_data(seed)


def _max_dd_pct(equity: np.ndarray) -> float:
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(abs(np.min((equity - peak) / peak)) * 100.0)


def _profit_factor(values: list[float]) -> float:
    wins = sum(item for item in values if item > 0)
    losses = abs(sum(item for item in values if item < 0))
    return wins / losses if losses > 0 else float("inf")


def _strategy_unit_risks(config: UnifiedBacktestConfig, equity: float) -> dict[str, float]:
    return {
        "ATRSS": float(config.atrss.unit_risk_pct) * equity,
        "AKC_HELIX": float(config.helix.unit_risk_pct) * equity,
        "TPC": float(config.tpc.unit_risk_pct) * equity,
    }


def _trade_rows(result: Any, config: UnifiedBacktestConfig, equity: float) -> list[dict[str, Any]]:
    unit_risk = _strategy_unit_risks(config, equity)
    rows: list[dict[str, Any]] = []
    for sid, attr in (
        ("ATRSS", "atrss_trades"),
        ("AKC_HELIX", "helix_trades"),
        ("TPC", "tpc_trades"),
    ):
        for trade in getattr(result, attr, []) or []:
            entry = _get_entry_time(trade)
            r_mult = float(_get_r_multiple(trade) or 0.0)
            rows.append(
                {
                    "strategy": sid,
                    "entry_time": entry,
                    "r": r_mult,
                    "static_pnl": r_mult * unit_risk[sid],
                }
            )
    rows.sort(key=lambda item: item["entry_time"])
    return rows


def _window_trade_metrics(rows: list[dict[str, Any]], start: datetime, end: datetime) -> dict[str, Any]:
    window = [row for row in rows if start <= row["entry_time"] < end]
    pnls = [float(row["static_pnl"]) for row in window]
    rs = [float(row["r"]) for row in window]
    by_strategy: dict[str, dict[str, float]] = {
        sid: {"trades": 0.0, "total_r": 0.0, "static_pnl": 0.0}
        for sid in ("ATRSS", "AKC_HELIX", "TPC")
    }
    for row in window:
        item = by_strategy[row["strategy"]]
        item["trades"] += 1
        item["total_r"] += row["r"]
        item["static_pnl"] += row["static_pnl"]

    positive_total = sum(max(item["static_pnl"], 0.0) for item in by_strategy.values())
    for item in by_strategy.values():
        item["share_pct"] = (
            max(item["static_pnl"], 0.0) / positive_total * 100.0
            if positive_total > 0.0
            else 0.0
        )

    return {
        "total_trades": len(window),
        "winning_trades": sum(1 for pnl in pnls if pnl > 0),
        "win_rate_pct": (sum(1 for pnl in pnls if pnl > 0) / len(window) * 100.0) if window else 0.0,
        "profit_factor": _profit_factor(pnls),
        "net_r": sum(rs),
        "avg_r": (sum(rs) / len(rs)) if rs else 0.0,
        "static_pnl": sum(pnls),
        "static_return_pct": (sum(pnls) / STARTING_EQUITY * 100.0) if STARTING_EQUITY else 0.0,
        "strategy_summary": by_strategy,
        "atrss_share_pct": by_strategy["ATRSS"]["share_pct"],
        "non_atrss_share_pct": by_strategy["AKC_HELIX"]["share_pct"] + by_strategy["TPC"]["share_pct"],
    }


def _window_equity_metrics(result: Any, start: datetime, end: datetime) -> dict[str, Any]:
    raw_timestamps = getattr(result, "combined_timestamps", [])
    raw_equity = getattr(result, "combined_equity", [])
    timestamps = np.asarray(raw_timestamps)
    equity = np.asarray(raw_equity, dtype=float)
    if len(timestamps) == 0 or len(equity) == 0:
        return {"equity_return_pct": 0.0, "equity_pnl": 0.0, "max_drawdown_pct": 0.0}

    norm_ts = []
    for ts in timestamps:
        stamp = ts if isinstance(ts, datetime) else pd.Timestamp(ts).to_pydatetime()
        if stamp.tzinfo is not None:
            stamp = stamp.astimezone(timezone.utc).replace(tzinfo=None)
        norm_ts.append(stamp)
    ts_array = np.asarray(norm_ts, dtype=object)
    mask = (ts_array >= start) & (ts_array < end)
    if not np.any(mask):
        return {"equity_return_pct": 0.0, "equity_pnl": 0.0, "max_drawdown_pct": 0.0}
    window_equity = equity[mask]
    if len(window_equity) < 2:
        return {"equity_return_pct": 0.0, "equity_pnl": 0.0, "max_drawdown_pct": 0.0}
    pnl = float(window_equity[-1] - window_equity[0])
    base = float(window_equity[0])
    return {
        "equity_start": base,
        "equity_end": float(window_equity[-1]),
        "equity_pnl": pnl,
        "equity_return_pct": (pnl / base * 100.0) if base else 0.0,
        "max_drawdown_pct": _max_dd_pct(window_equity),
    }


def _score_oos(row: dict[str, Any]) -> float:
    oos = row["oos"]
    ret = float(oos["static_return_pct"])
    dd = float(oos["equity"]["max_drawdown_pct"])
    trades = float(oos["total_trades"])
    pf = float(oos["profit_factor"])
    pf_score = min(pf, 6.0) / 6.0 if math.isfinite(pf) else 1.0
    balance = 1.0 - min(abs(float(oos["atrss_share_pct"]) - 65.0) / 35.0, 1.0)
    return (
        0.42 * (ret / 25.0)
        + 0.20 * (trades / 25.0)
        + 0.16 * pf_score
        + 0.14 * balance
        - 0.30 * max(dd - 3.0, 0.0) / 10.0
    )


def _evaluate_candidate(task: tuple[str, dict[str, Any], dict[str, Any], float, str, str, str, str]) -> dict[str, Any]:
    if _DATA is None:
        raise RuntimeError("OOS worker was not initialized.")
    name, base, overrides, equity, data_dir, start, cutoff, end = task
    mutations = dict(base)
    mutations.update(overrides)
    mutations["start_date"] = start
    mutations["end_date"] = end
    config = UnifiedBacktestConfig(initial_equity=equity, data_dir=Path(data_dir), start_date=start, end_date=end)
    config = mutate_unified_config(config, mutations)
    result = run_unified(_DATA, config)
    rows = _trade_rows(result, config, equity)
    is_metrics = _window_trade_metrics(rows, _dt(start), _dt(cutoff))
    oos_metrics = _window_trade_metrics(rows, _dt(cutoff), _end_exclusive(end))
    oos_metrics["equity"] = _window_equity_metrics(result, _dt(cutoff), _end_exclusive(end))
    full_equity = np.asarray(getattr(result, "combined_equity", []), dtype=float)
    row = {
        "name": name,
        "mutations": overrides,
        "is": is_metrics,
        "oos": oos_metrics,
        "full": {
            "final_equity": float(full_equity[-1]) if len(full_equity) else equity,
            "max_drawdown_pct": _max_dd_pct(full_equity),
        },
    }
    row["oos_score"] = _score_oos(row)
    return row


def run(args: argparse.Namespace) -> Path:
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    base_path = Path(args.base_config)
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    base = json.loads(base_path.read_text())

    output_dir = ROOT / "backtests" / "output" / "swing" / "portfolio_synergy" / "oos"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"candidate_oos_compare_{stamp}.json"
    report_path = output_dir / f"candidate_oos_compare_{stamp}.txt"

    tasks = [
        (name, base, overrides, float(args.equity), str(data_dir), args.start, args.oos_cutoff, args.data_end)
        for name, overrides in REQUESTED_CANDIDATES
    ]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=max(1, int(args.max_workers)),
        initializer=_init_worker,
        initargs=(str(data_dir), float(args.equity), args.start, args.data_end),
    ) as pool:
        future_map = {pool.submit(_evaluate_candidate, task): task[0] for task in tasks}
        for future in as_completed(future_map):
            row = future.result()
            oos = row["oos"]
            print(
                f"{row['name']}: OOS ret={oos['static_return_pct']:+.2f}% "
                f"dd={oos['equity']['max_drawdown_pct']:.2f}% pf={oos['profit_factor']:.2f} "
                f"trades={oos['total_trades']} atrss={oos['atrss_share_pct']:.2f}%",
                flush=True,
            )
            results.append(row)

    ranked = sorted(results, key=lambda item: item["oos_score"], reverse=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_config": str(base_path),
        "data_dir": str(data_dir),
        "equity": float(args.equity),
        "window": {
            "start": args.start,
            "oos_cutoff": args.oos_cutoff,
            "data_end": args.data_end,
            "split_basis": "entry_time",
        },
        "ranked": ranked,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")

    lines = [
        "SWING PORTFOLIO OOS CANDIDATE COMPARISON",
        "=" * 72,
        f"Base config: {base_path}",
        f"Replay: {args.start} to {args.data_end}; OOS: {args.oos_cutoff} to {args.data_end}",
        "Metrics: static initial-risk trade metrics plus OOS equity max DD.",
        "",
        "Ranked OOS",
    ]
    for row in ranked:
        oos = row["oos"]
        shares = oos["strategy_summary"]
        lines.append(
            f"  {row['name']:<34} score={row['oos_score']:+.4f} "
            f"ret={oos['static_return_pct']:+.2f}% dd={oos['equity']['max_drawdown_pct']:.2f}% "
            f"pf={oos['profit_factor']:.2f} trades={oos['total_trades']} "
            f"atrss={oos['atrss_share_pct']:.2f}%"
        )
        lines.append(
            "    "
            + ", ".join(
                f"{sid}:{item['share_pct']:.1f}%/{int(item['trades'])}tr/{item['total_r']:+.2f}R/${item['static_pnl']:+,.0f}"
                for sid, item in shares.items()
            )
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {output_path}", flush=True)
    print(f"Wrote {report_path}", flush=True)
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default=BASE_CONFIG)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--equity", type=float, default=STARTING_EQUITY)
    parser.add_argument("--start", default=BACKTEST_START)
    parser.add_argument("--oos-cutoff", default=OOS_CUTOFF)
    parser.add_argument("--data-end", default=DATA_END)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
