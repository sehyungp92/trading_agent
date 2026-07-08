"""Round-6 TPC incremental mutation search.

This runner starts from the promoted round-6 OOS repair config and asks a
different question from the round-5 repair sweep: can any additional mutation
improve OOS without giving back the newly repaired in-sample profile or trading
frequency?

The OOS window is still selection OOS, not a fresh holdout.
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

from backtests.swing.auto.tpc.phase_candidates import get_phase_candidates
from backtests.swing.auto.tpc.round5_oos_repair import (
    Candidate,
    DATA_DIR,
    DEFAULT_TRAIN_END,
    dedupe_candidates,
    evaluate_oos_candidates,
    evaluate_train_oos_candidates,
    infer_data_end,
    normalize_jsonable,
    read_json,
    scale,
    write_csv,
    write_json,
)

ROUND6_ROOT = ROOT / "backtests" / "output" / "swing" / "tpc" / "round_6"
DEFAULT_CONFIG_PATH = ROUND6_ROOT / "optimized_config.json"
DEFAULT_SUMMARY_PATH = ROUND6_ROOT / "run_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-end", default=DEFAULT_TRAIN_END)
    parser.add_argument("--max-workers", type=int, default=max(1, min(6, (mp.cpu_count() or 2) - 1)))
    parser.add_argument("--candidate-limit", type=int, default=900)
    parser.add_argument("--shortlist", type=int, default=120)
    args = parser.parse_args()

    started = time.time()
    output_dir = args.output_dir or (
        ROUND6_ROOT / f"incremental_search_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    incumbent = read_json(args.config)
    summary = read_json(args.summary) if args.summary.exists() else {}
    baseline_oos = baseline_oos_from_summary(summary)
    candidates = build_incremental_candidates(incumbent, args.candidate_limit)

    print(f"[tpc-r6] loaded incumbent with {len(incumbent)} overrides", flush=True)
    print(f"[tpc-r6] built {len(candidates)} candidate configs", flush=True)
    print(
        "[tpc-r6] baseline OOS "
        f"net={float(baseline_oos.get('net_return_pct', 0.0)):+.2f}% "
        f"trades={float(baseline_oos.get('total_trades', 0.0)):.0f}",
        flush=True,
    )

    oos_rows = evaluate_oos_candidates(
        candidates,
        baseline_oos=baseline_oos,
        data_dir=DATA_DIR,
        train_end=args.train_end,
        max_workers=args.max_workers,
        output_path=output_dir / "oos_progress.jsonl",
    )
    oos_scored = [score_oos_stage(row, baseline_oos) for row in oos_rows]
    oos_sorted = sorted(oos_scored, key=lambda row: row["oos_stage_objective"], reverse=True)
    shortlist = select_train_oos_shortlist(candidates, oos_sorted, limit=args.shortlist)

    print(f"[tpc-r6] validating {len(shortlist)} shortlisted configs on train+OOS", flush=True)
    validation_rows = evaluate_train_oos_candidates(
        shortlist,
        data_dir=DATA_DIR,
        train_end=args.train_end,
        max_workers=args.max_workers,
        output_path=output_dir / "train_oos_validation_progress.jsonl",
    )

    baseline_validation = first_by_name(validation_rows, "BASE_R6")
    if baseline_validation is None:
        raise RuntimeError("BASE_R6 was not validated; shortlist construction failed")

    scored = [score_incremental_validation(row, baseline_validation) for row in validation_rows]
    scored_sorted = sorted(scored, key=lambda row: row["objective"], reverse=True)
    best_balanced = first_candidate(scored_sorted, "passed_balanced_gate")
    best_both = first_candidate(scored_sorted, "improves_train_and_oos_net")
    best_frequency = first_candidate(scored_sorted, "passed_frequency_gate")
    best_overall = next((row for row in scored_sorted if row.get("name") != "BASE_R6"), None)
    recommendation = best_balanced or best_both or best_frequency or best_overall

    write_json(
        output_dir / "run_spec.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "train_end": args.train_end,
            "data_end": infer_data_end(DATA_DIR),
            "config": str(args.config.resolve()),
            "summary": str(args.summary.resolve()),
            "candidate_limit": args.candidate_limit,
            "shortlist": args.shortlist,
            "max_workers": args.max_workers,
            "selection_oos_note": "The six-month OOS window was used for mutation selection; use fresh data before promotion.",
        },
    )
    write_json(output_dir / "candidate_manifest.json", [candidate_to_dict(c) for c in candidates])
    write_json(output_dir / "oos_results.json", oos_sorted)
    write_csv(output_dir / "oos_results.csv", flatten_oos_rows(oos_sorted))
    write_json(output_dir / "train_oos_validation.json", scored_sorted)
    write_csv(output_dir / "train_oos_validation.csv", flatten_validation_rows(scored_sorted))
    if recommendation is not None:
        write_json(output_dir / "recommended_config.json", recommendation.get("mutations", incumbent))
    report = format_incremental_report(
        baseline=baseline_validation,
        scored=scored_sorted,
        oos_rows=oos_sorted,
        recommendation=recommendation,
        elapsed_seconds=time.time() - started,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report, flush=True)
    print(f"[tpc-r6] output: {output_dir.resolve()}", flush=True)


def baseline_oos_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    final = summary.get("final_metrics", {}) or {}
    out: dict[str, Any] = {}
    for key, value in final.items():
        if str(key).startswith("oos_"):
            out[str(key)[4:]] = value
    return out


def build_incremental_candidates(incumbent: dict[str, Any], limit: int) -> list[Candidate]:
    candidates: list[Candidate] = [
        Candidate("BASE_R6", "baseline", dict(incumbent), intent="Round-6 incumbent"),
    ]

    def add(name: str, stage: str, extra: dict[str, Any], *, intent: str = "") -> None:
        muts = dict(incumbent)
        muts.update(extra)
        candidates.append(Candidate(name=name, stage=stage, mutations=muts, source=name, intent=intent))

    def remove(name: str, key: str) -> None:
        muts = dict(incumbent)
        muts.pop(key, None)
        candidates.append(Candidate(name=f"remove::{name}", stage="active_key_ablation", mutations=muts, source=key))

    # A small diagnostic ablation layer catches accepted keys that only survived
    # because of in-sample fit or stale interactions with earlier mutations.
    for key in sorted(incumbent):
        remove(key, key)

    for phase in range(1, 7):
        for name, muts in get_phase_candidates(phase):
            add(f"phase{phase}_{name}", f"phase{phase}_single", muts, intent="Prior phase candidate tested on round-6 base")

    sessions = session_mutations()
    trend = trend_mutations()
    signal_quality = signal_quality_mutations()
    second_entries = second_entry_mutations()
    exits = exit_mutations()
    risk = risk_mutations()
    balance = symbol_balance_mutations()

    families: list[tuple[str, dict[str, dict[str, Any]]]] = [
        ("session", sessions),
        ("trend", trend),
        ("signal", signal_quality),
        ("second", second_entries),
        ("exit", exits),
        ("risk", risk),
        ("balance", balance),
    ]
    for stage, family in families:
        for name, muts in family.items():
            add(f"{stage}_{name}", f"{stage}_single", muts)

    # Known near-frontier anchors from the round-5 repair validation. These are
    # tested both alone and as bases for more granular overlays.
    anchors = {
        "t1stop040": {"all.t1_stop_r": 0.40},
        "t1stop035": {"all.t1_stop_r": 0.35},
        "gld_no8": {"GLD.primary_windows_et": ((9, 30, 11, 30), (13, 0, 16, 0))},
        "value_hits2": {"all.type_a_value_hits_min": 2},
        "gld_value_hits2": {"GLD.type_a_value_hits_min": 2},
        "source17": {"all.second_entry_score_min": 15, "all.second_entry_min_source_score": 17.0},
        "addon_off": {"all.addon_enabled": False},
        "qqq_typeb_supply": {"QQQ.type_b_enabled": True, "QQQ.score_b_min": 15},
    }
    for name, muts in anchors.items():
        add(f"anchor_{name}", "anchor_single", muts)

    priority_pair_sets = [
        ("trend_exit", pick(trend, "ma100_070_di", "adx_12_di", "ma50_030_ma100_060_di"), pick(exits, "t1_stop_035", "t1_stop_040", "floor_light", "mfe_giveback_200_045_050")),
        ("trend_session", pick(trend, "ma100_070_di", "adx_12_di", "gld_adx16_di"), pick(sessions, "gld_regular_hours", "gld_no8_full_afternoon", "avoid_1045_1145")),
        ("signal_exit", pick(signal_quality, "gld_value_hits2", "gld_score16_b16", "gld_preferred_confirm", "gld_room30"), pick(exits, "t1_stop_035", "floor_light", "runner_72")),
        ("second_exit", pick(second_entries, "source16", "source17", "source_a_plus", "wait12"), pick(exits, "t1_stop_035", "t1_stop_040", "floor_balanced")),
        ("risk_signal", pick(risk, "dynamic_070_125", "dynamic_065_120", "risk_stack_0225"), pick(signal_quality, "gld_value_hits2", "gld_score16_b16", "all_confirm_max3")),
        ("session_balance", pick(sessions, "gld_regular_hours", "gld_no8_full_afternoon", "gld_compact_quality"), pick(balance, "qqq_typeb_score15", "qqq_typeb_fib20_38", "gld_no_type_c")),
    ]
    for set_name, left, right in priority_pair_sets:
        for (a_name, a_muts), (b_name, b_muts) in itertools.product(left.items(), right.items()):
            if keys_conflict(a_muts, b_muts):
                continue
            add(f"priority_pair_{set_name}_{a_name}_{b_name}", "priority_cross_pair", {**a_muts, **b_muts})

    core_filters = {
        "gld_no8": anchors["gld_no8"],
        "gld_regular": sessions["gld_regular_hours"],
        "source17": anchors["source17"],
        "gld_score16": {"GLD.score_a_min": 16, "GLD.score_b_min": 16},
        "gld_value2": anchors["gld_value_hits2"],
    }
    monetizers = {
        "t1stop035": exits["t1_stop_035"],
        "t1stop040": exits["t1_stop_040"],
        "floor_light": exits["floor_light"],
        "mfe_giveback_200_045_050": exits["mfe_giveback_200_045_050"],
        "dynamic_070_125": risk["dynamic_070_125"],
        "qqq_typeb_supply": anchors["qqq_typeb_supply"],
    }
    for (a_name, a_muts), (b_name, b_muts), (c_name, c_muts) in itertools.product(
        core_filters.items(),
        core_filters.items(),
        monetizers.items(),
    ):
        if a_name >= b_name or keys_conflict(a_muts, b_muts) or keys_conflict({**a_muts, **b_muts}, c_muts):
            continue
        add(f"priority_package_{a_name}_{b_name}_{c_name}", "priority_targeted_package", {**a_muts, **b_muts, **c_muts})

    overlay_groups = [
        ("session", sessions),
        ("exit", exits),
        ("risk", risk),
        ("second", second_entries),
        ("signal", signal_quality),
        ("balance", balance),
    ]
    for anchor_name, anchor_muts in anchors.items():
        for group_name, group in overlay_groups:
            for overlay_name, overlay_muts in group.items():
                if keys_conflict(anchor_muts, overlay_muts):
                    continue
                add(
                    f"combo_{anchor_name}_{group_name}_{overlay_name}",
                    f"anchor_x_{group_name}",
                    {**anchor_muts, **overlay_muts},
                )

    # Focused cross-family pairs. This avoids a blind combinatorial explosion
    # while still testing the main causal hypotheses in both directions.
    pair_plan = [
        ("trend", trend, "exit", exits),
        ("trend", trend, "session", sessions),
        ("signal", signal_quality, "exit", exits),
        ("second", second_entries, "exit", exits),
        ("session", sessions, "balance", balance),
        ("risk", risk, "exit", exits),
        ("risk", risk, "signal", signal_quality),
    ]
    for left_name, left, right_name, right in pair_plan:
        for (a_name, a_muts), (b_name, b_muts) in itertools.product(left.items(), right.items()):
            if keys_conflict(a_muts, b_muts):
                continue
            add(f"pair_{left_name}_{a_name}__{right_name}_{b_name}", "cross_family_pair", {**a_muts, **b_muts})
            if len(candidates) >= limit:
                return dedupe_candidates(candidates)[:limit]

    # A final tier of three-way packages around the most plausible repair shape:
    # trend/session quality plus a less destructive exit or risk change.
    for (a_name, a_muts), (b_name, b_muts), (c_name, c_muts) in itertools.product(
        core_filters.items(),
        core_filters.items(),
        monetizers.items(),
    ):
        if a_name >= b_name or keys_conflict(a_muts, b_muts) or keys_conflict({**a_muts, **b_muts}, c_muts):
            continue
        add(f"package_{a_name}_{b_name}_{c_name}", "targeted_package", {**a_muts, **b_muts, **c_muts})
        if len(candidates) >= limit:
            break

    return dedupe_candidates(candidates)[:limit]


def pick(family: dict[str, dict[str, Any]], *names: str) -> dict[str, dict[str, Any]]:
    return {name: family[name] for name in names if name in family}


def keys_conflict(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return any(key in right and normalize_jsonable(value) != normalize_jsonable(right[key]) for key, value in left.items())


def session_mutations() -> dict[str, dict[str, Any]]:
    return {
        "avoid_1030_1200": {"all.avoid_windows_et": ((10, 30, 12, 0),)},
        "avoid_1045_1145": {"all.avoid_windows_et": ((10, 45, 11, 45),)},
        "avoid_1100_1130": {"all.avoid_windows_et": ((11, 0, 11, 30),)},
        "avoid_1130_1230": {"all.avoid_windows_et": ((11, 30, 12, 30),)},
        "avoid_1200_1300": {"all.avoid_windows_et": ((12, 0, 13, 0),)},
        "avoid_1430_1500": {"all.avoid_windows_et": ((14, 30, 15, 0),)},
        "gld_regular_hours": {"GLD.primary_windows_et": ((9, 30, 11, 30), (13, 0, 15, 45))},
        "gld_no8_full_afternoon": {"GLD.primary_windows_et": ((9, 30, 11, 30), (13, 0, 16, 0))},
        "gld_compact_quality": {"GLD.primary_windows_et": ((8, 30, 11, 0), (13, 30, 15, 15))},
        "gld_no_late": {"GLD.primary_windows_et": ((8, 0, 11, 30), (13, 0, 15, 0))},
        "gld_avoid_lunch_late": {"GLD.avoid_windows_et": ((11, 0, 12, 30), (15, 0, 16, 0))},
        "qqq_morning_only": {"QQQ.primary_windows_et": ((9, 35, 11, 30),)},
        "qqq_afternoon_only": {"QQQ.primary_windows_et": ((13, 30, 15, 45),)},
        "qqq_wider_morning": {"QQQ.primary_windows_et": ((9, 35, 12, 0), (13, 30, 15, 45))},
    }


def trend_mutations() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for slope in (0.04, 0.05, 0.07, 0.08, 0.10):
        out[f"ma100_{int(slope * 1000):03d}"] = {"all.min_ma100_slope_atr_4h": slope}
        out[f"ma100_{int(slope * 1000):03d}_di"] = {
            "all.min_ma100_slope_atr_4h": slope,
            "all.require_di_alignment": True,
        }
    for adx in (10.0, 12.0, 14.0, 15.0, 18.0):
        out[f"adx_{int(adx)}"] = {"all.min_adx_4h": adx}
        out[f"adx_{int(adx)}_di"] = {"all.min_adx_4h": adx, "all.require_di_alignment": True}
    for slope in (0.02, 0.03, 0.04, 0.05):
        out[f"ma50_{int(slope * 1000):03d}_ma100_060_di"] = {
            "all.min_ma50_slope_atr_4h": slope,
            "all.min_ma100_slope_atr_4h": 0.06,
            "all.require_di_alignment": True,
        }
    out.update(
        {
            "max_adx_35": {"all.max_adx_4h": 35.0},
            "max_adx_40": {"all.max_adx_4h": 40.0},
            "gld_adx16_di": {"GLD.min_adx_4h": 16.0, "GLD.require_di_alignment": True},
            "gld_adx20_di": {"GLD.min_adx_4h": 20.0, "GLD.require_di_alignment": True},
            "gld_ma50_040_ma100_060_di": {
                "GLD.min_ma50_slope_atr_4h": 0.04,
                "GLD.min_ma100_slope_atr_4h": 0.06,
                "GLD.require_di_alignment": True,
            },
            "qqq_ma100_060_di": {"QQQ.min_ma100_slope_atr_4h": 0.06, "QQQ.require_di_alignment": True},
        }
    )
    return out


def signal_quality_mutations() -> dict[str, dict[str, Any]]:
    return {
        "all_value_hits2": {"all.type_a_value_hits_min": 2},
        "all_value_hits3": {"all.type_a_value_hits_min": 3},
        "gld_value_hits2": {"GLD.type_a_value_hits_min": 2},
        "qqq_value_hits2": {"QQQ.type_a_value_hits_min": 2},
        "gld_score16_b16": {"GLD.score_a_min": 16, "GLD.score_b_min": 16},
        "gld_score17_b17": {"GLD.score_a_min": 17, "GLD.score_b_min": 17},
        "qqq_score_a15_b15": {"QQQ.score_a_min": 15, "QQQ.score_b_min": 15},
        "all_orderly_pullbacks": {"all.pullback_orderly_required": True},
        "gld_orderly_pullbacks": {"GLD.pullback_orderly_required": True},
        "all_volume_contract_125": {"all.pullback_volume_contract_max": 1.25},
        "all_volume_contract_110": {"all.pullback_volume_contract_max": 1.10},
        "all_confirm2": {"all.confirmation_required": 2},
        "all_confirm_max2": {"all.confirmation_max_count": 2},
        "all_confirm_max3": {"all.confirmation_max_count": 3},
        "gld_preferred_confirm": {"GLD.confirmation_required": 2, "GLD.confirmation_combo_mode": "preferred"},
        "gld_structure_vwap_confirm": {"GLD.confirmation_required": 2, "GLD.confirmation_combo_mode": "structure_vwap"},
        "gld_require_structure": {"GLD.require_structure_confirmation": True},
        "gld_require_vwap": {"GLD.require_vwap_confirmation": True},
        "all_fib_a_038_070": {"all.fib_a_low": 0.38, "all.fib_a_high": 0.70},
        "all_fib_a_042_072": {"all.fib_a_low": 0.42, "all.fib_a_high": 0.72},
        "gld_fib_a_038_070": {"GLD.fib_a_low": 0.38, "GLD.fib_a_high": 0.70},
        "gld_room30": {"GLD.daily_room_min_r": 3.0},
        "gld_room35": {"GLD.daily_room_min_r": 3.5},
        "gld_extension175": {"GLD.max_extension_atr_mult": 1.75},
    }


def second_entry_mutations() -> dict[str, dict[str, Any]]:
    return {
        "type_c_off": {"all.type_c_enabled": False},
        "type_c_aplus": {"all.type_c_requires_a_plus": True},
        "score15": {"all.second_entry_score_min": 15},
        "score16": {"all.second_entry_score_min": 16},
        "source16": {"all.second_entry_score_min": 15, "all.second_entry_min_source_score": 16.0},
        "source17": {"all.second_entry_score_min": 15, "all.second_entry_min_source_score": 17.0},
        "source18": {"all.second_entry_score_min": 16, "all.second_entry_min_source_score": 18.0},
        "source_a_plus": {"all.second_entry_requires_source_a_plus": True},
        "wait8": {"all.second_entry_max_wait_bars_15m": 8},
        "wait12": {"all.second_entry_max_wait_bars_15m": 12},
        "wait20": {"all.second_entry_max_wait_bars_15m": 20},
        "source17_wait12": {
            "all.second_entry_score_min": 15,
            "all.second_entry_min_source_score": 17.0,
            "all.second_entry_max_wait_bars_15m": 12,
        },
        "source_aplus_score15": {
            "all.second_entry_score_min": 15,
            "all.second_entry_requires_source_a_plus": True,
        },
    }


def exit_mutations() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for stop_r in (0.20, 0.25, 0.35, 0.38, 0.40, 0.45, 0.50, 0.60):
        out[f"t1_stop_{int(stop_r * 100):03d}"] = {"all.t1_stop_r": stop_r}
    for stop_r in (0.35, 0.40, 0.45):
        out[f"gld_t1_stop_{int(stop_r * 100):03d}"] = {"GLD.t1_stop_r": stop_r}
        out[f"qqq_t1_stop_{int(stop_r * 100):03d}"] = {"QQQ.t1_stop_r": stop_r}
    for t1 in (0.95, 1.00, 1.05, 1.15, 1.25):
        out[f"gld_t1r_{int(t1 * 100):03d}"] = {"GLD.t1_r": t1}
    for t1 in (1.10, 1.15, 1.20, 1.30, 1.40):
        out[f"qqq_t1r_{int(t1 * 100):03d}"] = {"QQQ.t1_r": t1}
    for pct in (0.35, 0.40, 0.50, 0.55, 0.60, 0.65):
        out[f"all_t1_partial_{int(pct * 100):02d}"] = {"all.t1_partial_pct": pct}
        out[f"gld_t1_partial_{int(pct * 100):02d}"] = {"GLD.t1_partial_pct": pct}
    for t2 in (1.60, 1.75, 2.25, 2.50):
        out[f"t2r_{int(t2 * 100):03d}"] = {"all.t2_r": t2}
    out.update(
        {
            "floor_light": {"all.profit_floor_ladder": ((1.0, 0.10), (1.5, 0.40), (2.25, 0.90))},
            "floor_balanced": {
                "all.profit_floor_ladder": ((0.75, 0.10), (1.50, 0.50), (2.25, 1.10), (3.25, 1.80))
            },
            "floor_fast": {"all.profit_floor_ladder": ((0.60, 0.00), (1.10, 0.30), (1.60, 0.70), (2.25, 1.15))},
            "floor_slow_runner": {
                "all.profit_floor_ladder": ((1.25, 0.20), (2.00, 0.75), (3.00, 1.50), (4.00, 2.25))
            },
            "mfe_giveback_150_045_025": {
                "all.mfe_giveback_trigger_r": 1.50,
                "all.mfe_giveback_retain_frac": 0.45,
                "all.mfe_giveback_lock_r": 0.25,
                "all.mfe_giveback_after_t1_only": True,
            },
            "mfe_giveback_200_045_050": {
                "all.mfe_giveback_trigger_r": 2.00,
                "all.mfe_giveback_retain_frac": 0.45,
                "all.mfe_giveback_lock_r": 0.50,
                "all.mfe_giveback_after_t1_only": True,
            },
            "mfe_giveback_250_055_075": {
                "all.mfe_giveback_trigger_r": 2.50,
                "all.mfe_giveback_retain_frac": 0.55,
                "all.mfe_giveback_lock_r": 0.75,
                "all.mfe_giveback_after_t1_only": True,
            },
            "mfe_giveback_any_150_045_025": {
                "all.mfe_giveback_trigger_r": 1.50,
                "all.mfe_giveback_retain_frac": 0.45,
                "all.mfe_giveback_lock_r": 0.25,
                "all.mfe_giveback_after_t1_only": False,
            },
            "runner_48": {"all.runner_max_hold_bars_15m": 48},
            "runner_72": {"all.runner_max_hold_bars_15m": 72},
            "runner_96": {"all.runner_max_hold_bars_15m": 96},
            "max_hold_32": {"all.max_hold_bars_15m": 32},
            "max_hold_40": {"all.max_hold_bars_15m": 40},
            "max_hold_44": {"all.max_hold_bars_15m": 44},
            "time_stop_mfe075": {"all.time_stop_min_mfe_r": 0.75},
            "time_stop_mfe100": {"all.time_stop_min_mfe_r": 1.00},
            "stall_44_mfe075_cur020": {
                "all.stall_exit_bars_15m": 44,
                "all.stall_exit_min_mfe_r": 0.75,
                "all.stall_exit_max_current_r": 0.20,
            },
            "stall_60_mfe125_cur040": {
                "all.stall_exit_bars_15m": 60,
                "all.stall_exit_min_mfe_r": 1.25,
                "all.stall_exit_max_current_r": 0.40,
            },
            "addon_off": {"all.addon_enabled": False},
            "addon_smaller_score18": {"all.addon_size_mult": 0.15, "all.addon_min_score": 18},
            "addon_later_score18": {"all.addon_trigger_r": 1.75, "all.addon_min_score": 18},
        }
    )
    return out


def risk_mutations() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for max_risk in (0.0175, 0.0200, 0.0225, 0.0275):
        out[f"risk_stack_{int(max_risk * 10000):04d}"] = {
            "all.max_risk_pct": max_risk,
            "all.risk_a_pct": round(max_risk * 0.64, 5),
            "all.risk_a_plus_pct": max_risk,
            "all.risk_b_pct": round(max_risk * 0.40, 5),
        }
    for floor, ceil, min_mult, max_mult in (
        (14.0, 21.0, 0.70, 1.25),
        (15.0, 21.0, 0.65, 1.20),
        (16.0, 22.0, 0.60, 1.30),
        (17.0, 22.0, 0.55, 1.35),
    ):
        out[f"dynamic_{int(min_mult * 100):03d}_{int(max_mult * 100):03d}"] = {
            "all.dynamic_risk_enabled": True,
            "all.dynamic_risk_score_floor": floor,
            "all.dynamic_risk_score_ceiling": ceil,
            "all.dynamic_risk_min_mult": min_mult,
            "all.dynamic_risk_max_mult": max_mult,
            "all.dynamic_risk_curve": 1.0,
        }
    out.update(
        {
            "notional_060": {"all.max_position_notional_pct": 6.0},
            "notional_070": {"all.max_position_notional_pct": 7.0},
            "notional_100": {"all.max_position_notional_pct": 10.0},
            "qqq_risk_015": {"QQQ.max_risk_pct": 0.015, "QQQ.risk_a_pct": 0.010, "QQQ.risk_a_plus_pct": 0.015},
            "qqq_risk_020": {"QQQ.max_risk_pct": 0.020, "QQQ.risk_a_pct": 0.013, "QQQ.risk_a_plus_pct": 0.020},
            "gld_risk_020": {"GLD.max_risk_pct": 0.020, "GLD.risk_a_pct": 0.013, "GLD.risk_a_plus_pct": 0.020},
            "gld_risk_0275": {"GLD.max_risk_pct": 0.0275, "GLD.risk_a_pct": 0.0175, "GLD.risk_a_plus_pct": 0.0275},
        }
    )
    return out


def symbol_balance_mutations() -> dict[str, dict[str, Any]]:
    return {
        "qqq_typeb_score15": {"QQQ.type_b_enabled": True, "QQQ.score_b_min": 15},
        "qqq_typeb_score14": {"QQQ.type_b_enabled": True, "QQQ.score_b_min": 14},
        "qqq_typeb_strict_value2": {
            "QQQ.type_b_enabled": True,
            "QQQ.score_b_min": 15,
            "QQQ.type_b_value_hits_min": 2,
        },
        "qqq_typeb_fib20_38": {
            "QQQ.type_b_enabled": True,
            "QQQ.type_b_requires_a_plus": False,
            "QQQ.fib_b_low": 0.20,
            "QQQ.fib_b_high": 0.38,
            "QQQ.score_b_min": 15,
        },
        "qqq_allow_shorts_score18": {"QQQ.shorts_enabled": True, "QQQ.min_short_score": 18},
        "gld_no_type_c": {"GLD.type_c_enabled": False},
        "gld_type_c_aplus": {"GLD.type_c_requires_a_plus": True},
        "gld_structure_stop": {"GLD.entry_order_model": "structure_stop"},
        "gld_structure_stop_market": {"GLD.entry_order_model": "structure_stop_market"},
        "gld_ttl_2h": {"GLD.entry_order_ttl_hours": 2.0},
        "gld_ttl_6h": {"GLD.entry_order_ttl_hours": 6.0},
    }


def score_oos_stage(row: dict[str, Any], baseline_oos: dict[str, Any]) -> dict[str, Any]:
    oos = row.get("oos", {}) or {}
    base_net = float(baseline_oos.get("net_return_pct", 0.0))
    base_trades = max(float(baseline_oos.get("total_trades", 0.0)), 1.0)
    base_avg_r = float(baseline_oos.get("avg_r", 0.0))
    base_pf = float(baseline_oos.get("dollar_profit_factor", 0.0))
    net = float(oos.get("net_return_pct", 0.0))
    trades = float(oos.get("total_trades", 0.0))
    avg_r = float(oos.get("avg_r", 0.0))
    pf = float(oos.get("dollar_profit_factor", 0.0))
    dd = float(oos.get("max_dd_pct", 0.0))
    objective = (
        0.42 * scale(net - base_net, -2.0, 6.0)
        + 0.18 * scale(avg_r - base_avg_r, -0.20, 0.60)
        + 0.14 * scale(pf - base_pf, -0.30, 1.30)
        + 0.16 * scale(trades / base_trades, 0.60, 1.40)
        - 0.10 * scale(dd, 4.5, 12.0)
    )
    out = dict(row)
    out["oos_stage_objective"] = objective
    out["oos_vs_round6"] = {
        "net_return_pct": net - base_net,
        "total_trades": trades - float(baseline_oos.get("total_trades", 0.0)),
        "avg_r": avg_r - base_avg_r,
        "dollar_profit_factor": pf - base_pf,
        "max_dd_pct": dd - float(baseline_oos.get("max_dd_pct", 0.0)),
    }
    return out


def select_train_oos_shortlist(
    candidates: list[Candidate],
    oos_rows: list[dict[str, Any]],
    *,
    limit: int,
) -> list[Candidate]:
    by_name = {candidate.name: candidate for candidate in candidates}
    selected: list[Candidate] = [by_name["BASE_R6"]]

    def add_name(name: str) -> None:
        candidate = by_name.get(name)
        if candidate is not None:
            selected.append(candidate)

    def best_rows(key: str, n: int, reverse: bool = True) -> list[dict[str, Any]]:
        return sorted(oos_rows, key=lambda row: float(row.get(key, 0.0)), reverse=reverse)[:n]

    for row in best_rows("oos_stage_objective", max(36, limit // 2)):
        add_name(str(row.get("name", "")))
    for row in sorted(oos_rows, key=lambda row: float((row.get("oos") or {}).get("net_return_pct", -999.0)), reverse=True)[:36]:
        add_name(str(row.get("name", "")))
    for row in sorted(oos_rows, key=lambda row: float((row.get("oos") or {}).get("avg_r", -999.0)), reverse=True)[:24]:
        add_name(str(row.get("name", "")))
    for row in sorted(
        [r for r in oos_rows if float((r.get("oos") or {}).get("net_return_pct", 0.0)) > 0.0],
        key=lambda row: float((row.get("oos") or {}).get("total_trades", -999.0)),
        reverse=True,
    )[:24]:
        add_name(str(row.get("name", "")))

    stages = sorted({str(row.get("stage", "")) for row in oos_rows if row.get("stage")})
    for stage in stages:
        rows = [row for row in oos_rows if str(row.get("stage", "")) == stage]
        for row in sorted(rows, key=lambda item: float(item.get("oos_stage_objective", -999.0)), reverse=True)[:4]:
            add_name(str(row.get("name", "")))

    return dedupe_candidates(selected)[:limit]


def score_incremental_validation(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    train = row.get("train", {}) or {}
    oos = row.get("oos", {}) or {}
    base_train = baseline.get("train", {}) or {}
    base_oos = baseline.get("oos", {}) or {}

    train_net = float(train.get("net_return_pct", 0.0))
    train_trades = float(train.get("total_trades", 0.0))
    train_pf = float(train.get("dollar_profit_factor", 0.0))
    train_dd = float(train.get("max_dd_pct", 0.0))
    oos_net = float(oos.get("net_return_pct", 0.0))
    oos_trades = float(oos.get("total_trades", 0.0))
    oos_avg_r = float(oos.get("avg_r", 0.0))
    oos_pf = float(oos.get("dollar_profit_factor", 0.0))
    oos_dd = float(oos.get("max_dd_pct", 0.0))

    base_train_net = max(float(base_train.get("net_return_pct", 0.0)), 1e-9)
    base_train_trades = max(float(base_train.get("total_trades", 0.0)), 1e-9)
    base_train_pf = float(base_train.get("dollar_profit_factor", 0.0))
    base_train_dd = float(base_train.get("max_dd_pct", 0.0))
    base_oos_net = float(base_oos.get("net_return_pct", 0.0))
    base_oos_trades = max(float(base_oos.get("total_trades", 0.0)), 1e-9)
    base_oos_avg_r = float(base_oos.get("avg_r", 0.0))
    base_oos_pf = float(base_oos.get("dollar_profit_factor", 0.0))
    base_oos_dd = float(base_oos.get("max_dd_pct", 0.0))

    train_net_retention = train_net / base_train_net
    train_trade_retention = train_trades / base_train_trades
    oos_trade_retention = oos_trades / base_oos_trades
    oos_net_delta = oos_net - base_oos_net
    train_net_delta = train_net - float(base_train.get("net_return_pct", 0.0))

    frequency_ok = train_trade_retention >= 0.80 and oos_trade_retention >= 0.80
    train_material_ok = (
        train_net_retention >= 0.90
        and train_trade_retention >= 0.80
        and train_pf >= max(1.50, base_train_pf - 0.20)
        and train_dd <= max(18.0, base_train_dd + 5.0)
    )
    significant_oos = (
        oos_net_delta >= 1.00
        and oos_avg_r >= base_oos_avg_r
        and oos_pf >= base_oos_pf
        and oos_dd <= max(7.5, base_oos_dd + 3.0)
        and oos_trade_retention >= 0.80
    )
    improves_both = train_net_delta > 0.25 and oos_net_delta > 0.25
    passed_balanced = train_material_ok and significant_oos
    passed_frequency = frequency_ok and oos_net_delta > 0.25 and train_net_retention >= 0.85

    objective = (
        0.34 * scale(oos_net_delta, -0.50, 5.50)
        + 0.14 * scale(oos_avg_r - base_oos_avg_r, -0.10, 0.55)
        + 0.12 * scale(oos_pf - base_oos_pf, -0.20, 1.20)
        + 0.12 * scale(oos_trade_retention, 0.70, 1.40)
        + 0.16 * scale(train_net_retention, 0.85, 1.12)
        + 0.08 * scale(train_trade_retention, 0.75, 1.15)
        + 0.06 * scale(train_pf - base_train_pf, -0.25, 0.75)
        - 0.08 * scale(oos_dd - base_oos_dd, 0.0, 5.0)
    )

    out = dict(row)
    out.update(
        {
            "objective": objective,
            "passed_balanced_gate": passed_balanced,
            "passed_frequency_gate": passed_frequency,
            "improves_train_and_oos_net": improves_both,
            "delta": {
                "train_net_return_pct": train_net_delta,
                "train_total_trades": train_trades - float(base_train.get("total_trades", 0.0)),
                "train_dollar_profit_factor": train_pf - base_train_pf,
                "train_max_dd_pct": train_dd - base_train_dd,
                "oos_net_return_pct": oos_net_delta,
                "oos_total_trades": oos_trades - float(base_oos.get("total_trades", 0.0)),
                "oos_avg_r": oos_avg_r - base_oos_avg_r,
                "oos_dollar_profit_factor": oos_pf - base_oos_pf,
                "oos_max_dd_pct": oos_dd - base_oos_dd,
            },
            "retention": {
                "train_net": train_net_retention,
                "train_trades": train_trade_retention,
                "oos_trades": oos_trade_retention,
            },
        }
    )
    return out


def flatten_oos_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat = []
    for row in rows:
        oos = row.get("oos", {}) or {}
        delta = row.get("oos_vs_round6", {}) or {}
        flat.append(
            {
                "name": row.get("name", ""),
                "stage": row.get("stage", ""),
                "objective": row.get("oos_stage_objective", 0.0),
                "oos_net_return_pct": oos.get("net_return_pct", ""),
                "oos_total_trades": oos.get("total_trades", ""),
                "oos_avg_r": oos.get("avg_r", ""),
                "oos_win_rate": oos.get("win_rate", ""),
                "oos_dollar_profit_factor": oos.get("dollar_profit_factor", ""),
                "oos_max_dd_pct": oos.get("max_dd_pct", ""),
                "delta_oos_net_return_pct": delta.get("net_return_pct", ""),
                "delta_oos_trades": delta.get("total_trades", ""),
                "delta_oos_avg_r": delta.get("avg_r", ""),
                "delta_oos_pf": delta.get("dollar_profit_factor", ""),
                "error": row.get("error", ""),
            }
        )
    return flat


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
                "objective": row.get("objective", 0.0),
                "passed_balanced_gate": row.get("passed_balanced_gate", False),
                "passed_frequency_gate": row.get("passed_frequency_gate", False),
                "improves_train_and_oos_net": row.get("improves_train_and_oos_net", False),
                "train_net_return_pct": train.get("net_return_pct", ""),
                "train_total_trades": train.get("total_trades", ""),
                "train_avg_r": train.get("avg_r", ""),
                "train_dollar_profit_factor": train.get("dollar_profit_factor", ""),
                "train_max_dd_pct": train.get("max_dd_pct", ""),
                "oos_net_return_pct": oos.get("net_return_pct", ""),
                "oos_total_trades": oos.get("total_trades", ""),
                "oos_avg_r": oos.get("avg_r", ""),
                "oos_win_rate": oos.get("win_rate", ""),
                "oos_dollar_profit_factor": oos.get("dollar_profit_factor", ""),
                "oos_max_dd_pct": oos.get("max_dd_pct", ""),
                "delta_train_net_return_pct": delta.get("train_net_return_pct", ""),
                "delta_train_trades": delta.get("train_total_trades", ""),
                "delta_oos_net_return_pct": delta.get("oos_net_return_pct", ""),
                "delta_oos_trades": delta.get("oos_total_trades", ""),
                "train_net_retention": retention.get("train_net", ""),
                "train_trade_retention": retention.get("train_trades", ""),
                "oos_trade_retention": retention.get("oos_trades", ""),
                "error": row.get("error", ""),
            }
        )
    return flat


def format_incremental_report(
    *,
    baseline: dict[str, Any],
    scored: list[dict[str, Any]],
    oos_rows: list[dict[str, Any]],
    recommendation: dict[str, Any] | None,
    elapsed_seconds: float,
) -> str:
    base_train = baseline.get("train", {}) or {}
    base_oos = baseline.get("oos", {}) or {}
    non_base = [row for row in scored if row.get("name") != "BASE_R6"]
    balanced = [row for row in non_base if row.get("passed_balanced_gate")]
    both = [row for row in non_base if row.get("improves_train_and_oos_net")]
    frequency = [row for row in non_base if row.get("passed_frequency_gate")]

    lines = [
        "# TPC Round 6 Incremental Mutation Search",
        "",
        f"Elapsed minutes: {elapsed_seconds / 60.0:.1f}",
        "",
        "## Baseline",
        f"Train: net {base_train.get('net_return_pct', 0.0):+.2f}%, trades {base_train.get('total_trades', 0.0):.0f}, avgR {base_train.get('avg_r', 0.0):+.3f}, $PF {base_train.get('dollar_profit_factor', 0.0):.2f}, DD {base_train.get('max_dd_pct', 0.0):.2f}%.",
        f"OOS: net {base_oos.get('net_return_pct', 0.0):+.2f}%, trades {base_oos.get('total_trades', 0.0):.0f}, avgR {base_oos.get('avg_r', 0.0):+.3f}, win {base_oos.get('win_rate', 0.0):.1%}, $PF {base_oos.get('dollar_profit_factor', 0.0):.2f}, DD {base_oos.get('max_dd_pct', 0.0):.2f}%.",
        "",
        "## OOS-Only Leaders",
        "| Candidate | Stage | OOS Net | Delta | Trades | AvgR | $PF | DD |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(oos_rows, key=lambda item: float((item.get("oos") or {}).get("net_return_pct", -999.0)), reverse=True)[:12]:
        oos = row.get("oos", {}) or {}
        delta = row.get("oos_vs_round6", {}) or {}
        lines.append(
            f"| {row.get('name', '')} | {row.get('stage', '')} | "
            f"{oos.get('net_return_pct', 0.0):+.2f}% | {delta.get('net_return_pct', 0.0):+.2f}% | "
            f"{oos.get('total_trades', 0.0):.0f} | {oos.get('avg_r', 0.0):+.3f} | "
            f"{oos.get('dollar_profit_factor', 0.0):.2f} | {oos.get('max_dd_pct', 0.0):.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Train+OOS Validation Leaders",
            "| Candidate | Balanced | Both Ret | Freq | Train Net | Train Trades | OOS Net | OOS Trades | OOS AvgR | OOS $PF |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in non_base[:15]:
        train = row.get("train", {}) or {}
        oos = row.get("oos", {}) or {}
        lines.append(
            f"| {row.get('name', '')} | {str(row.get('passed_balanced_gate', False))} | "
            f"{str(row.get('improves_train_and_oos_net', False))} | {str(row.get('passed_frequency_gate', False))} | "
            f"{train.get('net_return_pct', 0.0):+.2f}% | {train.get('total_trades', 0.0):.0f} | "
            f"{oos.get('net_return_pct', 0.0):+.2f}% | {oos.get('total_trades', 0.0):.0f} | "
            f"{oos.get('avg_r', 0.0):+.3f} | {oos.get('dollar_profit_factor', 0.0):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Gate Counts",
            f"- Balanced OOS uplift with material in-sample preservation: {len(balanced)}",
            f"- Improves both train and OOS headline return: {len(both)}",
            f"- Improves OOS while preserving frequency and at least 85% train net: {len(frequency)}",
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
                f"Train: net {train.get('net_return_pct', 0.0):+.2f}%, trades {train.get('total_trades', 0.0):.0f}, $PF {train.get('dollar_profit_factor', 0.0):.2f}, DD {train.get('max_dd_pct', 0.0):.2f}%.",
                f"OOS: net {oos.get('net_return_pct', 0.0):+.2f}%, trades {oos.get('total_trades', 0.0):.0f}, avgR {oos.get('avg_r', 0.0):+.3f}, $PF {oos.get('dollar_profit_factor', 0.0):.2f}, DD {oos.get('max_dd_pct', 0.0):.2f}%.",
                "Treat this as selection-OOS evidence. A fresh holdout or forward paper window is still required before promotion.",
            ]
        )
    return "\n".join(lines) + "\n"


def candidate_to_dict(candidate: Candidate) -> dict[str, Any]:
    return {
        "name": candidate.name,
        "stage": candidate.stage,
        "source": candidate.source,
        "intent": candidate.intent,
        "mutations": normalize_jsonable(candidate.mutations),
    }


def first_by_name(rows: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("name") == name:
            return row
    return None


def first_candidate(rows: list[dict[str, Any]], flag: str) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("name") != "BASE_R6" and row.get(flag)), None)


if __name__ == "__main__":
    main()
