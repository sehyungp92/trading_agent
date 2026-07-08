"""Focused follow-up sweep around the robust momentum round-2 OOS repair.

This script starts from the best IS-preserving second-phase family
(`setup.min_room_b=0.0` plus ETH long-only) and tests local combinations that
the generic second-phase sweep did not cover, especially multi-symbol direction
guards and a small set of entry/exit/risk perturbations.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analyze_oos_repair import (
    DATA_DIR,
    DEFAULT_SYMBOLS,
    RepairContext,
    _compact_result,
    _dedupe_tasks,
    _load_config,
    _round_config_path,
    _run_tasks,
    _task,
    _task_from_mutations,
    _write_strategy_config,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SECOND_RESULTS = (
    ROOT
    / "output"
    / "momentum"
    / "round_2_oos_repair_second_phase"
    / "20260526T044553Z_resumable"
    / "all_results.json"
)


def _load_result_payload(results_path: Path, label: str) -> dict[str, Any]:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    for item in results:
        if item.get("label") == label:
            return deepcopy(item["payload"])
    raise KeyError(f"Missing candidate {label!r} in {results_path}")


def _add_mutation_task(
    tasks: list[dict[str, Any]],
    *,
    label: str,
    base_payload: dict[str, Any],
    mutations: dict[str, Any],
    thesis: str,
) -> None:
    task = _task_from_mutations(
        strategy="momentum",
        label=f"followup:{label}",
        stage="followup",
        group="robust_local",
        base_payload=base_payload,
        mutations=mutations,
        thesis=thesis,
    )
    if task is not None:
        tasks.append(task)


def build_tasks(results_path: Path) -> list[dict[str, Any]]:
    h4_2_eth = _load_result_payload(
        results_path,
        "second_phase:bias.h4_ema_slope_lookback_2_symbol_filter_eth_direction_long_only",
    )
    h4_10_eth = _load_result_payload(
        results_path,
        "second_phase:bias.h4_ema_slope_lookback_10_symbol_filter_eth_direction_long_only",
    )
    first_repair = _load_config(
        "momentum",
        ROOT / "output" / "momentum" / "round_2_oos_repair" / "20260526T044553Z_resumable" / "recommended_config.json",
    ).to_dict()

    tasks: list[dict[str, Any]] = [
        _task(
            label="checkpoint:current_round2",
            stage="checkpoint",
            group="checkpoint",
            payload=_load_config("momentum", _round_config_path("momentum", 2)).to_dict(),
        ),
        _task(
            label="checkpoint:first_phase_min_room_b_0",
            stage="checkpoint",
            group="checkpoint",
            payload=first_repair,
        ),
        _task(
            label="checkpoint:h4_2_eth_long",
            stage="checkpoint",
            group="checkpoint",
            payload=h4_2_eth,
        ),
        _task(
            label="checkpoint:h4_10_eth_long",
            stage="checkpoint",
            group="checkpoint",
            payload=h4_10_eth,
        ),
    ]

    h4_values = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
    btc_modes = ["both", "long_only", "disabled"]
    sol_modes = ["both", "long_only", "short_only", "disabled"]
    for h4 in h4_values:
        _add_mutation_task(
            tasks,
            label=f"h4_{h4}_eth_long",
            base_payload=h4_2_eth,
            mutations={"bias.h4_ema_slope_lookback": h4},
            thesis="Local h4 lookback stability around ETH long-only repair.",
        )
    for h4 in [2, 6, 8, 10]:
        for btc_mode in btc_modes:
            for sol_mode in sol_modes:
                if btc_mode == "both" and sol_mode == "both":
                    continue
                _add_mutation_task(
                    tasks,
                    label=f"h4_{h4}_eth_long_btc_{btc_mode}_sol_{sol_mode}",
                    base_payload=h4_2_eth,
                    mutations={
                        "bias.h4_ema_slope_lookback": h4,
                        "symbol_filter.btc_direction": btc_mode,
                        "symbol_filter.sol_direction": sol_mode,
                    },
                    thesis="Test combined direction guards for losing BTC-short/SOL sleeves.",
                )

    single_mutations: list[tuple[str, dict[str, Any], str]] = [
        ("entry_break", {"entry.mode": "break"}, "Test break entries on robust sleeve."),
        ("entry_hybrid", {"entry.mode": "hybrid_grade"}, "Test hybrid entries on robust sleeve."),
        ("entry_confirm_preferred", {"entry.mode": "confirm_preferred"}, "Test confirmation-preferred entries."),
        ("weak_confluence_1", {"confirmation.min_confluences_for_weak": 1}, "Recover frequency with weaker trigger gate."),
        ("weak_confluence_3", {"confirmation.min_confluences_for_weak": 3}, "Tighten weak confirmations."),
        ("volume_threshold_1_0", {"confirmation.volume_threshold_mult": 1.0}, "Normalize volume threshold."),
        ("volume_threshold_1_2", {"confirmation.volume_threshold_mult": 1.2}, "Tighten volume threshold."),
        ("risk_b_0076", {"risk.risk_pct_b": 0.0076}, "Undo B-risk scaling."),
        ("risk_b_0100", {"risk.risk_pct_b": 0.01}, "Moderate B-risk scaling."),
        ("risk_b_0150", {"risk.risk_pct_b": 0.015}, "Increase B-risk if edge survives."),
        ("reentry_off", {"reentry.enabled": False}, "Disable reentry churn."),
        ("stop_min_2_25", {"stops.min_stop_atr_mult": 2.25}, "Widen minimum stop to reduce stop churn."),
        ("stop_min_2_5", {"stops.min_stop_atr_mult": 2.5}, "Widen minimum stop further."),
        ("atr_buffer_0_4", {"stops.atr_buffer_mult": 0.4}, "Increase stop buffer."),
        ("atr_buffer_0_5", {"stops.atr_buffer_mult": 0.5}, "Increase stop buffer further."),
        ("quick_exit_off", {"exits.quick_exit_enabled": False}, "Disable quick exit."),
        ("quick_exit_4", {"exits.quick_exit_bars": 4}, "Faster quick exit."),
        ("quick_exit_8", {"exits.quick_exit_bars": 8}, "Slower quick exit."),
        ("trail_activation_bars_8", {"trail.trail_activation_bars": 8}, "Delay trailing activation."),
        ("trail_activation_r_0_45", {"trail.trail_activation_r": 0.45}, "Require more profit before trailing."),
        ("mfe_giveback_0_6", {"exits.mfe_retrace_giveback_r": 0.6}, "Tighten MFE giveback."),
        ("mfe_giveback_1_0", {"exits.mfe_retrace_giveback_r": 1.0}, "Loosen MFE giveback."),
    ]
    for label, mutations, thesis in single_mutations:
        _add_mutation_task(
            tasks,
            label=label,
            base_payload=h4_2_eth,
            mutations=mutations,
            thesis=thesis,
        )

    combo_mutations: list[tuple[str, dict[str, Any], str]] = [
        (
            "eth_btc_long_entry_break",
            {"symbol_filter.btc_direction": "long_only", "entry.mode": "break"},
            "Pair BTC long-only with break entry.",
        ),
        (
            "eth_sol_disabled_entry_break",
            {"symbol_filter.sol_direction": "disabled", "entry.mode": "break"},
            "Remove SOL sleeve and test break entry.",
        ),
        (
            "eth_btc_long_sol_disabled",
            {"symbol_filter.btc_direction": "long_only", "symbol_filter.sol_direction": "disabled"},
            "Keep only ETH/BTC long sleeves.",
        ),
        (
            "eth_btc_long_sol_long",
            {"symbol_filter.btc_direction": "long_only", "symbol_filter.sol_direction": "long_only"},
            "Remove BTC/SOL short sleeves.",
        ),
        (
            "eth_btc_long_sol_short",
            {"symbol_filter.btc_direction": "long_only", "symbol_filter.sol_direction": "short_only"},
            "Remove BTC shorts and SOL longs.",
        ),
        (
            "eth_btc_long_sol_disabled_risk_015",
            {
                "symbol_filter.btc_direction": "long_only",
                "symbol_filter.sol_direction": "disabled",
                "risk.risk_pct_b": 0.015,
            },
            "Risk-scale the strongest long-only sleeve.",
        ),
        (
            "eth_btc_long_sol_long_risk_015",
            {
                "symbol_filter.btc_direction": "long_only",
                "symbol_filter.sol_direction": "long_only",
                "risk.risk_pct_b": 0.015,
            },
            "Risk-scale long-only BTC/SOL guard.",
        ),
        (
            "eth_btc_long_sol_long_stop_225",
            {
                "symbol_filter.btc_direction": "long_only",
                "symbol_filter.sol_direction": "long_only",
                "stops.min_stop_atr_mult": 2.25,
            },
            "Combine long-only guard with wider stop.",
        ),
    ]
    for label, mutations, thesis in combo_mutations:
        _add_mutation_task(
            tasks,
            label=label,
            base_payload=h4_2_eth,
            mutations=mutations,
            thesis=thesis,
        )

    return _dedupe_tasks(tasks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_SECOND_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--executor", choices=("subprocess", "thread", "process", "sequential"), default="subprocess")
    parser.add_argument("--subprocess-timeout", type=int, default=1800)
    parser.add_argument("--top-full", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        ROOT
        / "output"
        / "momentum"
        / "round_2_oos_repair_followup"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    tasks = build_tasks(args.results_path)
    if args.dry_run:
        print(json.dumps({"task_count": len(tasks), "output_dir": str(output_dir)}, indent=2))
        return
    context = RepairContext(
        strategy="momentum",
        round_num=2,
        symbols=DEFAULT_SYMBOLS,
        data_dir=DATA_DIR,
        is_start="2026-02-25",
        is_end="2026-04-20",
        oos_start="2026-04-21",
        oos_end="2026-05-23",
    )
    results = _run_tasks(
        tasks,
        context=context,
        max_workers=args.max_workers,
        executor_kind=args.executor,
        process_batch_size=4,
        subprocess_timeout=args.subprocess_timeout,
        top_full=args.top_full,
        output_dir=output_dir,
    )
    completed = [
        item
        for item in results
        if not item.get("error")
        and "oos" in item.get("evaluation", {})
        and "is" in item.get("evaluation", {})
    ]
    first_is_floor = 17.815069140834368 * 0.95
    robust = [
        item
        for item in completed
        if item["evaluation"]["is"]["metrics"]["net_return_pct"] >= first_is_floor
        and item["evaluation"]["oos"]["metrics"]["total_trades"] >= 25
    ]
    best = max(
        robust or completed,
        key=lambda item: (
            item["evaluation"]["oos"]["metrics"]["net_return_pct"],
            item["evaluation"]["full"]["metrics"]["net_return_pct"],
            item["evaluation"]["is"]["metrics"]["net_return_pct"],
        ),
    )
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_count": len(results),
        "completed_with_is": len(completed),
        "robust_count": len(robust),
        "selection_rule": "max OOS, then full, then IS; robust filter IS >= 95% first repair and OOS trades >= 25",
        "best": _compact_result(best, include_payload=True),
        "top_robust": [
            _compact_result(item)
            for item in sorted(
                robust,
                key=lambda item: (
                    item["evaluation"]["oos"]["metrics"]["net_return_pct"],
                    item["evaluation"]["full"]["metrics"]["net_return_pct"],
                    item["evaluation"]["is"]["metrics"]["net_return_pct"],
                ),
                reverse=True,
            )[:25]
        ],
        "top_oos": [
            _compact_result(item)
            for item in sorted(
                completed,
                key=lambda item: (
                    item["evaluation"]["oos"]["metrics"]["net_return_pct"],
                    item["evaluation"]["oos"]["metrics"]["total_trades"],
                ),
                reverse=True,
            )[:25]
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    recommended_path = output_dir / "recommended_followup_config.json"
    _write_strategy_config(recommended_path, best["payload"])
    summary["recommended_config_path"] = str(recommended_path)
    (output_dir / "followup_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "best": _compact_result(best)}, indent=2, default=str))


if __name__ == "__main__":
    main()
