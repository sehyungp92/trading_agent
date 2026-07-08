"""Round-6 TPC QQQ alpha/excellent-trade search.

The search is intentionally structural rather than OOS-window chasing:

* candidates use the shared TPC core and replay engine;
* optional setup-score refits use the capped ``alpha7`` score model;
* ranking has seven objective components;
* a candidate must improve QQQ excellent trades in both train and OOS before it
  can pass the main gate.

The OOS window remains selection OOS, so any accepted candidate still needs
fresh forward validation.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import multiprocessing as mp
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.swing.auto.tpc.round5_oos_repair import (
    Candidate,
    DATA_DIR,
    DEFAULT_TRAIN_END,
    dedupe_candidates,
    evaluate_train_oos_candidates,
    infer_data_end,
    normalize_jsonable,
    read_json,
    scale,
    write_json,
)

ROUND6_ROOT = ROOT / "backtests" / "output" / "swing" / "tpc" / "round_6"
DEFAULT_CONFIG_PATH = ROUND6_ROOT / "optimized_config.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-end", default=DEFAULT_TRAIN_END)
    parser.add_argument("--max-workers", type=int, default=max(1, min(6, (mp.cpu_count() or 2) - 1)))
    parser.add_argument("--candidate-limit", type=int, default=640)
    args = parser.parse_args()

    started = time.time()
    output_dir = args.output_dir or (
        ROUND6_ROOT / f"qqq_alpha_search_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    incumbent = read_json(args.config)
    candidates = build_qqq_alpha_candidates(incumbent, args.candidate_limit)
    print(f"[tpc-r6-qqq] loaded incumbent with {len(incumbent)} overrides", flush=True)
    print(f"[tpc-r6-qqq] built {len(candidates)} QQQ structural candidates", flush=True)
    validation_rows = evaluate_train_oos_candidates(
        candidates,
        data_dir=DATA_DIR,
        train_end=args.train_end,
        max_workers=args.max_workers,
        output_path=output_dir / "train_oos_validation_progress.jsonl",
    )
    baseline = first_by_name(validation_rows, "BASE_R6")
    if baseline is None:
        raise RuntimeError("BASE_R6 missing from validation rows")
    scored = [score_candidate(row, baseline) for row in validation_rows]
    scored_sorted = sorted(scored, key=lambda row: row["qqq_alpha_objective"], reverse=True)
    recommendation = first_candidate(scored_sorted, "passed_qqq_alpha_gate")

    write_json(
        output_dir / "run_spec.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "train_end": args.train_end,
            "data_end": infer_data_end(DATA_DIR),
            "config": str(args.config.resolve()),
            "candidate_limit": args.candidate_limit,
            "max_workers": args.max_workers,
            "score_model_note": "Structural score-model candidates use alpha7, a seven-component setup score implemented in the shared TPC core.",
            "objective_components": objective_components(),
            "selection_oos_note": "OOS is selection OOS. QQQ OOS sample is small; require fresh holdout/forward validation.",
        },
    )
    write_json(output_dir / "candidate_manifest.json", [candidate_to_dict(c) for c in candidates])
    write_json(output_dir / "train_oos_validation.json", scored_sorted)
    write_csv(output_dir / "train_oos_validation.csv", flatten_validation_rows(scored_sorted))
    if recommendation is not None:
        write_json(output_dir / "recommended_config.json", recommendation["mutations"])
    report = format_report(
        baseline=baseline,
        scored=scored_sorted,
        recommendation=recommendation,
        elapsed_seconds=time.time() - started,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report, flush=True)
    print(f"[tpc-r6-qqq] output: {output_dir.resolve()}", flush=True)


def build_qqq_alpha_candidates(incumbent: dict[str, Any], limit: int) -> list[Candidate]:
    candidates: list[Candidate] = [Candidate("BASE_R6", "baseline", dict(incumbent), intent="Round-6 incumbent")]

    alpha7_thresholds = {
        "alpha7_balanced": {
            "all.score_model": "alpha7",
            "all.score_a_plus_min": 13,
            "all.score_a_min": 10,
            "all.score_b_min": 9,
            "GLD.score_a_min": 11,
            "GLD.score_b_min": 10,
            "QQQ.score_a_min": 10,
            "QQQ.score_b_min": 9,
            "QQQ.addon_min_score": 12,
            "QQQ.min_short_score": 12,
        },
        "alpha7_strict": {
            "all.score_model": "alpha7",
            "all.score_a_plus_min": 14,
            "all.score_a_min": 11,
            "all.score_b_min": 10,
            "GLD.score_a_min": 12,
            "GLD.score_b_min": 11,
            "QQQ.score_a_min": 11,
            "QQQ.score_b_min": 10,
            "QQQ.addon_min_score": 13,
            "QQQ.min_short_score": 13,
        },
        "alpha7_supply": {
            "all.score_model": "alpha7",
            "all.score_a_plus_min": 13,
            "all.score_a_min": 9,
            "all.score_b_min": 8,
            "GLD.score_a_min": 11,
            "GLD.score_b_min": 10,
            "QQQ.score_a_min": 9,
            "QQQ.score_b_min": 8,
            "QQQ.addon_min_score": 12,
            "QQQ.min_short_score": 12,
        },
    }

    def add(name: str, stage: str, extra: dict[str, Any], *, base: dict[str, Any] | None = None, intent: str = "") -> None:
        muts = dict(incumbent)
        if base:
            muts.update(base)
        muts.update(extra)
        candidates.append(Candidate(name=name, stage=stage, mutations=muts, source=name, intent=intent))

    for base_name, base in alpha7_thresholds.items():
        add(base_name, "score_alpha7_base", {}, base=base, intent="Seven-component shared-core score calibration")

    singles = qqq_single_mutations()
    for name, muts in singles.items():
        for base_name, base in alpha7_thresholds.items():
            add(f"{base_name}_{name}", "qqq_single", muts, base=base)

    structural = pick(
        singles,
        "typeb_score14_value1",
        "typeb_score15_value2",
        "typeb_fib18_45_score14",
        "typeb_fib20_42_score14",
        "shorts_score13_aplus",
        "shorts_score12",
        "confirm1_preferred",
        "confirm2_micro_vwap",
        "room15",
        "ma100_030_no_di",
        "session_wide_pm",
        "entry_adaptive",
        "entry_market_next",
    )
    exits = pick(
        singles,
        "t1r100_partial65_stop040",
        "t1r110_partial65_stop040",
        "t1r090_partial70_stop035",
        "profit_floor_light",
        "profit_floor_runner",
        "mfe_giveback_200_045_050",
    )
    source_quality = pick(
        singles,
        "value_hits2",
        "orderly_pullbacks",
        "volume_contract125",
        "score_a11_b10",
        "score_a10_b09",
        "fib_a30_82",
        "fib_a38_72",
    )

    for base_name, base in alpha7_thresholds.items():
        for (a_name, a_muts), (b_name, b_muts) in itertools.product(structural.items(), exits.items()):
            if keys_conflict(a_muts, b_muts):
                continue
            add(f"{base_name}_pair_{a_name}__{b_name}", "qqq_structure_exit_pair", {**a_muts, **b_muts}, base=base)
        for (a_name, a_muts), (b_name, b_muts) in itertools.product(structural.items(), source_quality.items()):
            if keys_conflict(a_muts, b_muts):
                continue
            add(f"{base_name}_pair_{a_name}__{b_name}", "qqq_structure_quality_pair", {**a_muts, **b_muts}, base=base)

    packages = {
        "typeb_supply_capture": {
            **singles["typeb_fib18_45_score14"],
            **singles["t1r100_partial65_stop040"],
            **singles["room15"],
        },
        "typeb_quality_capture": {
            **singles["typeb_score15_value2"],
            **singles["confirm2_micro_vwap"],
            **singles["t1r110_partial65_stop040"],
        },
        "short_alpha_capture": {
            **singles["shorts_score13_aplus"],
            **singles["confirm2_micro_vwap"],
            **singles["profit_floor_light"],
        },
        "adaptive_breakout_capture": {
            **singles["entry_adaptive"],
            **singles["confirm1_preferred"],
            **singles["t1r100_partial65_stop040"],
        },
        "looser_trend_quality_exit": {
            **singles["ma100_030_no_di"],
            **singles["value_hits2"],
            **singles["profit_floor_runner"],
        },
    }
    for base_name, base in alpha7_thresholds.items():
        for name, muts in packages.items():
            add(f"{base_name}_package_{name}", "qqq_targeted_package", muts, base=base)

    return dedupe_candidates(candidates)[:limit]


def qqq_single_mutations() -> dict[str, dict[str, Any]]:
    return {
        "typeb_score14_value1": {"QQQ.type_b_enabled": True, "QQQ.score_b_min": 14, "QQQ.type_b_value_hits_min": 1},
        "typeb_score15_value2": {"QQQ.type_b_enabled": True, "QQQ.score_b_min": 15, "QQQ.type_b_value_hits_min": 2},
        "typeb_fib18_45_score14": {
            "QQQ.type_b_enabled": True,
            "QQQ.type_b_requires_a_plus": False,
            "QQQ.fib_b_low": 0.18,
            "QQQ.fib_b_high": 0.45,
            "QQQ.score_b_min": 14,
        },
        "typeb_fib20_42_score14": {
            "QQQ.type_b_enabled": True,
            "QQQ.type_b_requires_a_plus": False,
            "QQQ.fib_b_low": 0.20,
            "QQQ.fib_b_high": 0.42,
            "QQQ.score_b_min": 14,
        },
        "shorts_score12": {"QQQ.shorts_enabled": True, "QQQ.min_short_score": 12},
        "shorts_score13_aplus": {"QQQ.shorts_enabled": True, "QQQ.min_short_score": 13, "QQQ.shorts_require_a_plus": True},
        "confirm1_preferred": {"QQQ.confirmation_required": 1, "QQQ.confirmation_combo_mode": "preferred"},
        "confirm2_micro_vwap": {"QQQ.confirmation_required": 2, "QQQ.confirmation_combo_mode": "micro_vwap"},
        "confirm2_structure_vwap": {"QQQ.confirmation_required": 2, "QQQ.confirmation_combo_mode": "structure_vwap"},
        "confirm_max3": {"QQQ.confirmation_max_count": 3},
        "room15": {"QQQ.daily_room_min_r": 1.5},
        "room25": {"QQQ.daily_room_min_r": 2.5},
        "extension225": {"QQQ.max_extension_atr_mult": 2.25},
        "ma100_030_no_di": {"QQQ.min_ma100_slope_atr_4h": 0.03, "QQQ.require_di_alignment": False},
        "ma100_060_di": {"QQQ.min_ma100_slope_atr_4h": 0.06, "QQQ.require_di_alignment": True},
        "ma50_020_ma100_030": {"QQQ.min_ma50_slope_atr_4h": 0.02, "QQQ.min_ma100_slope_atr_4h": 0.03},
        "session_wide_pm": {"QQQ.primary_windows_et": ((9, 35, 11, 30), (12, 45, 15, 55))},
        "session_no_lunch": {"QQQ.primary_windows_et": ((9, 35, 11, 0), (13, 30, 15, 45))},
        "session_late_only": {"QQQ.primary_windows_et": ((14, 0, 15, 55),)},
        "entry_market_next": {"QQQ.entry_order_model": "market_next_bar"},
        "entry_adaptive": {
            "QQQ.entry_order_model": "adaptive_structure_stop",
            "QQQ.entry_adaptive_stop_limit_min_atr_mult": 0.08,
            "QQQ.entry_adaptive_stop_limit_max_atr_mult": 0.24,
        },
        "entry_stop_market": {"QQQ.entry_order_model": "structure_stop_market"},
        "entry_ttl2": {"QQQ.entry_order_ttl_hours": 2.0},
        "entry_ttl6": {"QQQ.entry_order_ttl_hours": 6.0},
        "value_hits2": {"QQQ.type_a_value_hits_min": 2},
        "orderly_pullbacks": {"QQQ.pullback_orderly_required": True},
        "volume_contract125": {"QQQ.pullback_volume_contract_max": 1.25},
        "score_a11_b10": {"QQQ.score_a_min": 11, "QQQ.score_b_min": 10},
        "score_a10_b09": {"QQQ.score_a_min": 10, "QQQ.score_b_min": 9},
        "fib_a30_82": {"QQQ.fib_a_low": 0.30, "QQQ.fib_a_high": 0.82},
        "fib_a38_72": {"QQQ.fib_a_low": 0.38, "QQQ.fib_a_high": 0.72},
        "duration_3_12": {"QQQ.pullback_min_bars_1h": 3, "QQQ.pullback_max_bars_1h": 12},
        "duration_4_12": {"QQQ.pullback_min_bars_1h": 4, "QQQ.pullback_max_bars_1h": 12},
        "t1r100_partial65_stop040": {"QQQ.t1_r": 1.00, "QQQ.t1_partial_pct": 0.65, "QQQ.t1_stop_r": 0.40},
        "t1r110_partial65_stop040": {"QQQ.t1_r": 1.10, "QQQ.t1_partial_pct": 0.65, "QQQ.t1_stop_r": 0.40},
        "t1r090_partial70_stop035": {"QQQ.t1_r": 0.90, "QQQ.t1_partial_pct": 0.70, "QQQ.t1_stop_r": 0.35},
        "t1r125_partial65_stop040": {"QQQ.t1_r": 1.25, "QQQ.t1_partial_pct": 0.65, "QQQ.t1_stop_r": 0.40},
        "profit_floor_light": {"QQQ.profit_floor_ladder": ((1.0, 0.10), (1.5, 0.40), (2.25, 0.90))},
        "profit_floor_runner": {"QQQ.profit_floor_ladder": ((1.25, 0.20), (2.00, 0.75), (3.00, 1.50))},
        "mfe_giveback_200_045_050": {
            "QQQ.mfe_giveback_trigger_r": 2.0,
            "QQQ.mfe_giveback_retain_frac": 0.45,
            "QQQ.mfe_giveback_lock_r": 0.50,
            "QQQ.mfe_giveback_after_t1_only": True,
        },
        "runner_72": {"QQQ.runner_max_hold_bars_15m": 72},
        "risk_020": {"QQQ.max_risk_pct": 0.020, "QQQ.risk_a_pct": 0.013, "QQQ.risk_a_plus_pct": 0.020},
        "risk_0275": {"QQQ.max_risk_pct": 0.0275, "QQQ.risk_a_pct": 0.0175, "QQQ.risk_a_plus_pct": 0.0275},
    }


def score_candidate(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    train = row.get("train", {}) or {}
    oos = row.get("oos", {}) or {}
    base_train = baseline.get("train", {}) or {}
    base_oos = baseline.get("oos", {}) or {}

    base_train_qqq_ex = float(base_train.get("qqq_excellent_trades", 0.0))
    base_oos_qqq_ex = float(base_oos.get("qqq_excellent_trades", 0.0))
    base_train_qqq_trades = max(float(base_train.get("qqq_trade_count", 0.0)), 1.0)
    base_oos_qqq_trades = max(float(base_oos.get("qqq_trade_count", 0.0)), 1.0)
    base_train_net = max(float(base_train.get("net_return_pct", 0.0)), 1e-9)
    base_oos_net = float(base_oos.get("net_return_pct", 0.0))

    train_qqq_ex = float(train.get("qqq_excellent_trades", 0.0))
    oos_qqq_ex = float(oos.get("qqq_excellent_trades", 0.0))
    train_qqq_trades = float(train.get("qqq_trade_count", 0.0))
    oos_qqq_trades = float(oos.get("qqq_trade_count", 0.0))
    train_qqq_rate = float(train.get("qqq_excellent_rate", 0.0))
    oos_qqq_rate = float(oos.get("qqq_excellent_rate", 0.0))
    train_qqq_avg_r = float((train.get("cohorts") or {}).get("QQQ", {}).get("avg_r", 0.0))
    oos_qqq_avg_r = float((oos.get("cohorts") or {}).get("QQQ", {}).get("avg_r", 0.0))
    train_net = float(train.get("net_return_pct", 0.0))
    oos_net = float(oos.get("net_return_pct", 0.0))
    train_pf = float(train.get("dollar_profit_factor", 0.0))
    oos_pf = float(oos.get("dollar_profit_factor", 0.0))

    # Seven objective components, matching the user's score-size constraint.
    components = {
        "train_qqq_excellent_count": scale(train_qqq_ex - base_train_qqq_ex, -2.0, 8.0),
        "oos_qqq_excellent_count": scale(oos_qqq_ex - base_oos_qqq_ex, 0.0, 3.0),
        "train_qqq_alpha_quality": 0.55 * scale(train_qqq_rate, 0.55, 0.85) + 0.45 * scale(float(train.get("qqq_trade_count", 0.0)), 14.0, 30.0),
        "oos_qqq_alpha_quality": 0.60 * scale(oos_qqq_rate, 0.20, 0.75) + 0.40 * scale(float((oos.get("cohorts") or {}).get("QQQ", {}).get("avg_r", 0.0)), -0.25, 1.25),
        "qqq_supply": 0.55 * scale(train_qqq_trades / base_train_qqq_trades, 0.80, 1.80) + 0.45 * scale(oos_qqq_trades / base_oos_qqq_trades, 1.0, 3.0),
        "train_book_preservation": 0.55 * scale(train_net / base_train_net, 0.85, 1.15) + 0.45 * scale(train_pf, 1.45, 2.20),
        "oos_book_preservation": 0.55 * scale(oos_net - base_oos_net, -1.0, 5.0) + 0.45 * scale(oos_pf, 1.10, 2.50),
    }
    objective = (
        0.22 * components["train_qqq_excellent_count"]
        + 0.22 * components["oos_qqq_excellent_count"]
        + 0.14 * components["train_qqq_alpha_quality"]
        + 0.14 * components["oos_qqq_alpha_quality"]
        + 0.10 * components["qqq_supply"]
        + 0.10 * components["train_book_preservation"]
        + 0.08 * components["oos_book_preservation"]
    )

    train_retention = train_net / base_train_net
    oos_delta = oos_net - base_oos_net
    passed = (
        row.get("name") != "BASE_R6"
        and train_qqq_ex > base_train_qqq_ex
        and oos_qqq_ex > base_oos_qqq_ex
        and train_qqq_trades >= base_train_qqq_trades
        and oos_qqq_trades >= base_oos_qqq_trades
        and train_qqq_rate >= 0.50
        and train_qqq_avg_r > 0.0
        and oos_qqq_avg_r > 0.0
        and train_retention >= 0.85
        and oos_net >= 0.0
        and float(oos.get("max_dd_pct", 0.0)) <= max(8.0, float(base_oos.get("max_dd_pct", 0.0)) + 3.0)
    )
    real_alpha = (
        train_qqq_ex > base_train_qqq_ex
        and oos_qqq_ex > base_oos_qqq_ex
        and train_qqq_rate >= 0.55
        and oos_qqq_rate > 0.0
        and train_retention >= 0.80
    )
    out = dict(row)
    out.update(
        {
            "qqq_alpha_objective": objective,
            "score_components": components,
            "score_component_count": len(components),
            "passed_qqq_alpha_gate": passed,
            "real_alpha_evidence": real_alpha,
            "delta": {
                "train_qqq_excellent_trades": train_qqq_ex - base_train_qqq_ex,
                "oos_qqq_excellent_trades": oos_qqq_ex - base_oos_qqq_ex,
                "train_qqq_trade_count": train_qqq_trades - float(base_train.get("qqq_trade_count", 0.0)),
                "oos_qqq_trade_count": oos_qqq_trades - float(base_oos.get("qqq_trade_count", 0.0)),
                "train_net_return_pct": train_net - float(base_train.get("net_return_pct", 0.0)),
                "oos_net_return_pct": oos_delta,
            },
            "retention": {
                "train_net": train_retention,
                "train_qqq_trades": train_qqq_trades / base_train_qqq_trades,
                "oos_qqq_trades": oos_qqq_trades / base_oos_qqq_trades,
            },
        }
    )
    return out


def objective_components() -> list[str]:
    return [
        "train_qqq_excellent_count",
        "oos_qqq_excellent_count",
        "train_qqq_alpha_quality",
        "oos_qqq_alpha_quality",
        "qqq_supply",
        "train_book_preservation",
        "oos_book_preservation",
    ]


def format_report(
    *,
    baseline: dict[str, Any],
    scored: list[dict[str, Any]],
    recommendation: dict[str, Any] | None,
    elapsed_seconds: float,
) -> str:
    base_train = baseline.get("train", {}) or {}
    base_oos = baseline.get("oos", {}) or {}
    non_base = [row for row in scored if row.get("name") != "BASE_R6"]
    gate = [row for row in non_base if row.get("passed_qqq_alpha_gate")]
    real = [row for row in non_base if row.get("real_alpha_evidence")]
    lines = [
        "# TPC Round 6 QQQ Alpha Search",
        "",
        f"Elapsed minutes: {elapsed_seconds / 60.0:.1f}",
        "",
        "## Guardrails",
        "- Structural candidates run through the shared TPC core and replay engine.",
        "- Score-model candidates use `alpha7`, a seven-component setup score.",
        "- Objective has exactly seven components.",
        "- Main gate requires QQQ excellent-trade count to rise in both train and OOS.",
        "- OOS is selection OOS; small QQQ OOS samples require fresh validation.",
        "",
        "## Baseline",
        f"Train QQQ: trades {base_train.get('qqq_trade_count', 0.0):.0f}, excellent {base_train.get('qqq_excellent_trades', 0.0):.0f}, excellent rate {base_train.get('qqq_excellent_rate', 0.0):.1%}.",
        f"OOS QQQ: trades {base_oos.get('qqq_trade_count', 0.0):.0f}, excellent {base_oos.get('qqq_excellent_trades', 0.0):.0f}, excellent rate {base_oos.get('qqq_excellent_rate', 0.0):.1%}.",
        f"Whole book: train net {base_train.get('net_return_pct', 0.0):+.2f}%, OOS net {base_oos.get('net_return_pct', 0.0):+.2f}%.",
        "",
        "## Train+OOS Leaders",
        "| Candidate | Gate | Train QQQ Ex | OOS QQQ Ex | Train QQQ Trades | OOS QQQ Trades | Train Net | OOS Net |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in non_base[:18]:
        train = row.get("train", {}) or {}
        oos = row.get("oos", {}) or {}
        lines.append(
            f"| {row.get('name', '')} | {str(row.get('passed_qqq_alpha_gate', False))} | "
            f"{train.get('qqq_excellent_trades', 0.0):.0f} | {oos.get('qqq_excellent_trades', 0.0):.0f} | "
            f"{train.get('qqq_trade_count', 0.0):.0f} | {oos.get('qqq_trade_count', 0.0):.0f} | "
            f"{train.get('net_return_pct', 0.0):+.2f}% | {oos.get('net_return_pct', 0.0):+.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Gate Counts",
            f"- Passed QQQ alpha gate: {len(gate)}",
            f"- Real-alpha evidence before all preservation gates: {len(real)}",
        ]
    )
    if recommendation is not None:
        train = recommendation.get("train", {}) or {}
        oos = recommendation.get("oos", {}) or {}
        lines.extend(
            [
                "",
                "## Recommendation",
                f"Selected candidate: `{recommendation.get('name', '')}`.",
                f"Train QQQ: trades {train.get('qqq_trade_count', 0.0):.0f}, excellent {train.get('qqq_excellent_trades', 0.0):.0f}, excellent rate {train.get('qqq_excellent_rate', 0.0):.1%}.",
                f"OOS QQQ: trades {oos.get('qqq_trade_count', 0.0):.0f}, excellent {oos.get('qqq_excellent_trades', 0.0):.0f}, excellent rate {oos.get('qqq_excellent_rate', 0.0):.1%}.",
                f"Whole book: train net {train.get('net_return_pct', 0.0):+.2f}%, OOS net {oos.get('net_return_pct', 0.0):+.2f}%.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Recommendation",
                "No candidate passed the strict QQQ alpha gate. Do not promote a QQQ mutation from this run.",
            ]
        )
    return "\n".join(lines) + "\n"


def flatten_validation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat = []
    for row in rows:
        train = row.get("train", {}) or {}
        oos = row.get("oos", {}) or {}
        delta = row.get("delta", {}) or {}
        retention = row.get("retention", {}) or {}
        flat.append(
            {
                "name": row.get("name", ""),
                "stage": row.get("stage", ""),
                "objective": row.get("qqq_alpha_objective", 0.0),
                "passed_qqq_alpha_gate": row.get("passed_qqq_alpha_gate", False),
                "real_alpha_evidence": row.get("real_alpha_evidence", False),
                "score_component_count": row.get("score_component_count", ""),
                "train_net_return_pct": train.get("net_return_pct", ""),
                "oos_net_return_pct": oos.get("net_return_pct", ""),
                "train_qqq_trade_count": train.get("qqq_trade_count", ""),
                "train_qqq_excellent_trades": train.get("qqq_excellent_trades", ""),
                "train_qqq_excellent_rate": train.get("qqq_excellent_rate", ""),
                "oos_qqq_trade_count": oos.get("qqq_trade_count", ""),
                "oos_qqq_excellent_trades": oos.get("qqq_excellent_trades", ""),
                "oos_qqq_excellent_rate": oos.get("qqq_excellent_rate", ""),
                "delta_train_qqq_excellent_trades": delta.get("train_qqq_excellent_trades", ""),
                "delta_oos_qqq_excellent_trades": delta.get("oos_qqq_excellent_trades", ""),
                "delta_train_qqq_trade_count": delta.get("train_qqq_trade_count", ""),
                "delta_oos_qqq_trade_count": delta.get("oos_qqq_trade_count", ""),
                "train_net_retention": retention.get("train_net", ""),
                "train_qqq_trade_retention": retention.get("train_qqq_trades", ""),
                "oos_qqq_trade_retention": retention.get("oos_qqq_trades", ""),
                "error": row.get("error", ""),
            }
        )
    return flat


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
        writer.writerows(rows)


def candidate_to_dict(candidate: Candidate) -> dict[str, Any]:
    return {
        "name": candidate.name,
        "stage": candidate.stage,
        "source": candidate.source,
        "intent": candidate.intent,
        "mutations": normalize_jsonable(candidate.mutations),
    }


def first_by_name(rows: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("name") == name), None)


def first_candidate(rows: list[dict[str, Any]], flag: str) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("name") != "BASE_R6" and row.get(flag)), None)


def pick(family: dict[str, dict[str, Any]], *names: str) -> dict[str, dict[str, Any]]:
    return {name: family[name] for name in names if name in family}


def keys_conflict(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return any(key in right and normalize_jsonable(value) != normalize_jsonable(right[key]) for key, value in left.items())


if __name__ == "__main__":
    main()
