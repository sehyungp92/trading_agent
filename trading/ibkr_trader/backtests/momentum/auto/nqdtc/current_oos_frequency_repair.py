"""Current NQDTC OOS frequency ablation and targeted repair diagnostic.

This runner evaluates the current round_4 incumbent, not the older round_3
baseline used by the generic OOS repair helper.  The OOS window is explicitly
used for diagnosis/selection here, so the output should be treated as a repair
study, not a fresh holdout validation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_root = Path(__file__).resolve().parents[4]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from backtests.swing.auto.incumbent_repair import (  # noqa: E402
    MISSING,
    RepairCandidate,
    score_candidate,
    serialize,
    short_key,
)
from backtests.swing.auto.oos_repair_diagnostics import (  # noqa: E402
    build_extra_historical_features,
    evaluate_strategy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CURRENT_CONFIG = PROJECT_ROOT / "backtests/output/momentum/nqdtc/round_4/optimized_config.json"
PRIOR_CONFIG = PROJECT_ROOT / "backtests/output/momentum/nqdtc/round_3/optimized_config.json"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "candidate_progress.jsonl"
    progress_path.write_text("", encoding="utf-8")

    current = read_json(Path(args.config_path))
    prior = read_json(Path(args.prior_config_path))
    spec = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": "nqdtc",
        "config_path": str(Path(args.config_path).resolve()),
        "prior_config_path": str(Path(args.prior_config_path).resolve()),
        "data_end": args.data_end,
        "max_workers": args.max_workers,
        "stages": args.stages,
        "selection_oos_note": (
            "The OOS window is used for diagnosis/selection in this run; "
            "it is no longer an untouched holdout."
        ),
    }
    write_json(output_dir / "run_spec.json", spec)

    print("[nqdtc-current] evaluating current incumbent", flush=True)
    started = time.time()
    baseline = evaluate_strategy("nqdtc", current, args.data_end)
    candidates = build_candidate_suite(current, prior, args.stages)
    print(
        f"[nqdtc-current] baseline IS={baseline.is_metrics.total_trades} "
        f"{baseline.is_metrics.net_r:+.2f}R; OOS={baseline.oos_metrics.total_trades} "
        f"{baseline.oos_metrics.net_r:+.2f}R; evaluating {len(candidates)} candidates",
        flush=True,
    )

    evaluations = evaluate_candidates(
        baseline=baseline,
        candidates=candidates,
        data_end=args.data_end,
        max_workers=args.max_workers,
        progress_path=progress_path,
    )
    top = sorted(evaluations, key=lambda item: item.objective_delta, reverse=True)
    passed = [item for item in top if item.passed]
    balanced = [
        item for item in evaluations
        if is_balanced_frequency_uplift(item, baseline)
    ]
    balanced = sorted(
        balanced,
        key=lambda item: (
            item.run.oos_metrics.net_r,
            item.run.oos_metrics.total_trades,
            item.run.is_metrics.net_r,
        ),
        reverse=True,
    )
    frequency_leaders = sorted(
        evaluations,
        key=lambda item: (
            item.run.oos_metrics.total_trades,
            item.run.oos_metrics.net_r,
            item.run.is_metrics.net_r,
        ),
        reverse=True,
    )
    oos_net_leaders = sorted(
        evaluations,
        key=lambda item: (
            item.run.oos_metrics.net_r,
            item.run.oos_metrics.total_trades,
            item.run.is_metrics.net_r,
        ),
        reverse=True,
    )
    by_stage = stage_leaders(evaluations, args.top_n)

    summary = {
        "run_spec": spec,
        "elapsed_seconds": round(time.time() - started, 2),
        "baseline": serialize(baseline),
        "candidate_count": len(candidates),
        "top": [serialize(item) for item in top[: args.top_n]],
        "passed": [serialize(item) for item in passed[: args.top_n]],
        "balanced_frequency_uplifts": [serialize(item) for item in balanced[: args.top_n]],
        "frequency_leaders": [serialize(item) for item in frequency_leaders[: args.top_n]],
        "oos_net_leaders": [serialize(item) for item in oos_net_leaders[: args.top_n]],
        "stage_leaders": by_stage,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.txt").write_text(format_report(summary), encoding="utf-8")
    print(f"[nqdtc-current] complete in {(time.time() - started) / 60.0:.1f} min", flush=True)
    print(f"Output: {output_dir.resolve()}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-end", default="2026-05-01")
    parser.add_argument("--config-path", default=str(CURRENT_CONFIG))
    parser.add_argument("--prior-config-path", default=str(PRIOR_CONFIG))
    parser.add_argument(
        "--output-dir",
        default="backtests/output/momentum/nqdtc/current_oos_frequency_repair_20260504",
    )
    parser.add_argument("--stages", default="ablation,perturbation,targeted")
    parser.add_argument("--max-workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--top-n", type=int, default=24)
    return parser.parse_args()


def build_candidate_suite(current: dict[str, Any], prior: dict[str, Any], stages_text: str) -> list[RepairCandidate]:
    stages = {item.strip() for item in stages_text.split(",") if item.strip()}
    candidates: list[RepairCandidate] = []
    if "ablation" in stages:
        candidates.extend(build_ablation_candidates(current, prior))
    if "perturbation" in stages:
        candidates.extend(build_perturbation_candidates(current))
    if "targeted" in stages:
        candidates.extend(build_targeted_candidates(current))
    if "refine" in stages:
        candidates.extend(build_refine_candidates(current))
    return dedupe(candidates, current)


def build_ablation_candidates(current: dict[str, Any], prior: dict[str, Any]) -> list[RepairCandidate]:
    candidates: list[RepairCandidate] = []
    previous_by_key: dict[str, Any] = {}

    for name, mutations, previous_values in build_extra_historical_features("nqdtc"):
        active_keys = [key for key, value in mutations.items() if current.get(key, MISSING) == value]
        if not active_keys:
            continue
        for key in active_keys:
            previous_by_key[key] = previous_values.get(key, MISSING)
        candidates.append(
            RepairCandidate(
                name=f"ablate_cluster_{name}",
                stage="ablation",
                mutations=reverted_config(current, active_keys, previous_values),
                intent="Remove one accepted historical mutation cluster.",
                source=name,
            )
        )
        for key in active_keys:
            candidates.append(
                RepairCandidate(
                    name=f"ablate_key_prior_{short_key(key)}",
                    stage="ablation",
                    mutations=reverted_config(current, [key], previous_values),
                    intent="Revert one accepted historical mutation key to its prior value.",
                    source=name,
                )
            )

    round4_previous = {
        key: prior.get(key, MISSING)
        for key, value in current.items()
        if prior.get(key, MISSING) != value
    }
    round4_keys = sorted(round4_previous)
    if round4_keys:
        candidates.append(
            RepairCandidate(
                name="ablate_cluster_round4_narrow_oos_repair",
                stage="ablation",
                mutations=reverted_config(current, round4_keys, round4_previous),
                intent="Revert the accepted round_4 narrow OOS repair bundle.",
                source="round4_narrow_oos_repair",
            )
        )
        for key in round4_keys:
            previous_by_key[key] = round4_previous[key]
            candidates.append(
                RepairCandidate(
                    name=f"ablate_round4_key_prior_{short_key(key)}",
                    stage="ablation",
                    mutations=reverted_config(current, [key], round4_previous),
                    intent="Revert one accepted round_4 mutation key.",
                    source="round4_narrow_oos_repair",
                )
            )

    for key in sorted(current):
        dropped = dict(current)
        dropped.pop(key, None)
        candidates.append(
            RepairCandidate(
                name=f"ablate_key_drop_{short_key(key)}",
                stage="ablation",
                mutations=dropped,
                intent="Drop one active incumbent mutation key.",
                source="active_key_inventory",
            )
        )
        previous = previous_by_key.get(key, prior.get(key, MISSING))
        if previous != MISSING and previous != current.get(key, MISSING):
            candidates.append(
                RepairCandidate(
                    name=f"ablate_key_prior_{short_key(key)}",
                    stage="ablation",
                    mutations=reverted_config(current, [key], {key: previous}),
                    intent="Revert one active incumbent mutation key to its nearest prior value.",
                    source="active_key_inventory",
                )
            )
    return candidates


def build_perturbation_candidates(current: dict[str, Any]) -> list[RepairCandidate]:
    candidates: list[RepairCandidate] = []
    for key in sorted(current):
        value = current[key]
        if isinstance(value, bool):
            add_patch(candidates, current, "perturbation", f"flip_{short_key(key)}", {key: not value}, key)
        elif isinstance(value, int) and not isinstance(value, bool):
            for new_value in int_variants(key, value):
                add_patch(
                    candidates,
                    current,
                    "perturbation",
                    f"perturb_{short_key(key)}_{fmt_value(new_value)}",
                    {key: new_value},
                    key,
                )
        elif isinstance(value, float):
            for new_value in float_variants(key, value):
                add_patch(
                    candidates,
                    current,
                    "perturbation",
                    f"perturb_{short_key(key)}_{fmt_value(new_value)}",
                    {key: new_value},
                    key,
                )

    for partial in [0.30, 0.35, 0.40, 0.45, 0.55]:
        add_patch(
            candidates,
            current,
            "perturbation",
            f"add_TP1_PARTIAL_PCT_{fmt_value(partial)}",
            {"param_overrides.TP1_PARTIAL_PCT": partial},
            "TP1_PARTIAL_PCT",
        )
    return candidates


def build_targeted_candidates(current: dict[str, Any]) -> list[RepairCandidate]:
    candidates: list[RepairCandidate] = []

    # OOS weakness: Aligned is the largest first-blocked regime and has positive
    # OOS shadow EV. Probe it with explicit non-Range score guards.
    for mult in [2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0]:
        add_patch(
            candidates,
            current,
            "targeted",
            f"aligned_open_nrm_{fmt_value(mult)}",
            {
                "param_overrides.BLOCK_ALIGNED_REGIME": False,
                "param_overrides.SCORE_NON_RANGE_MULT": mult,
            },
            "aligned_shadow_recovery",
        )
        for cooldown in [30, 45, 60]:
            add_patch(
                candidates,
                current,
                "targeted",
                f"aligned_open_nrm_{fmt_value(mult)}_cd{cooldown}",
                {
                    "param_overrides.BLOCK_ALIGNED_REGIME": False,
                    "param_overrides.SCORE_NON_RANGE_MULT": mult,
                    "param_overrides.MIN_INTER_TRADE_GAP_MINUTES": cooldown,
                },
                "aligned_shadow_recovery",
            )
        for score in [1.75, 2.0]:
            add_patch(
                candidates,
                current,
                "targeted",
                f"aligned_open_nrm_{fmt_value(mult)}_score_{fmt_value(score)}",
                {
                    "param_overrides.BLOCK_ALIGNED_REGIME": False,
                    "param_overrides.SCORE_NON_RANGE_MULT": mult,
                    "param_overrides.SCORE_NORMAL": score,
                },
                "aligned_shadow_recovery",
            )
        for max_stop in [175, 200, 225, 250]:
            add_patch(
                candidates,
                current,
                "targeted",
                f"aligned_open_nrm_{fmt_value(mult)}_maxstop{max_stop}",
                {
                    "param_overrides.BLOCK_ALIGNED_REGIME": False,
                    "param_overrides.SCORE_NON_RANGE_MULT": mult,
                    "flags.max_stop_width": True,
                    "param_overrides.MAX_STOP_WIDTH_PTS": max_stop,
                },
                "aligned_shadow_recovery",
            )

    # Score-gate shadow was slightly negative OOS, so pair any frequency looseners
    # with guards and separately test stricter score variants.
    for score in [1.65, 1.75, 2.0]:
        add_patch(
            candidates,
            current,
            "targeted",
            f"stricter_score_{fmt_value(score)}",
            {"param_overrides.SCORE_NORMAL": score},
            "score_gate_guard",
        )
    for rvol in [1.35, 1.75, 2.0]:
        add_patch(
            candidates,
            current,
            "targeted",
            f"rvol_{fmt_value(rvol)}",
            {"param_overrides.RVOL_SCORE_THRESH": rvol},
            "score_gate_guard",
        )

    # Entry/fill frequency controls not visible in first-block gate attribution.
    for cooldown in [15, 20, 30, 35, 40, 50, 60]:
        add_patch(
            candidates,
            current,
            "targeted",
            f"cooldown_{cooldown}",
            {"param_overrides.MIN_INTER_TRADE_GAP_MINUTES": cooldown},
            "frequency_control",
        )
    for hour_flag in ["flags.block_04_et", "flags.block_06_et", "flags.block_12_et"]:
        hour_name = short_key(hour_flag)
        add_patch(candidates, current, "targeted", f"open_{hour_name}", {hour_flag: False}, "hour_harvest")
        add_patch(
            candidates,
            current,
            "targeted",
            f"open_{hour_name}_score_1p75",
            {hour_flag: False, "param_overrides.SCORE_NORMAL": 1.75},
            "hour_harvest",
        )
        add_patch(
            candidates,
            current,
            "targeted",
            f"open_{hour_name}_rvol_1p75",
            {hour_flag: False, "param_overrides.RVOL_SCORE_THRESH": 1.75},
            "hour_harvest",
        )
    add_patch(
        candidates,
        current,
        "targeted",
        "open_04_06_12_score_1p75",
        {
            "flags.block_04_et": False,
            "flags.block_06_et": False,
            "flags.block_12_et": False,
            "param_overrides.SCORE_NORMAL": 1.75,
        },
        "hour_harvest",
    )

    # Conservative structural expansion probes.
    for patch_name, patch in [
        ("a_entry_enabled", {"param_overrides.A_ENTRY_ENABLED": True}),
        (
            "a_entry_enabled_maxstop175",
            {"param_overrides.A_ENTRY_ENABLED": True, "param_overrides.MAX_STOP_WIDTH_PTS": 175},
        ),
        (
            "c_continuation_enabled",
            {"param_overrides.C_CONT_ENTRY_ENABLED": True, "flags.entry_c_continuation": True},
        ),
        (
            "c_continuation_aligned_guarded",
            {
                "param_overrides.C_CONT_ENTRY_ENABLED": True,
                "flags.entry_c_continuation": True,
                "param_overrides.BLOCK_ALIGNED_REGIME": False,
                "param_overrides.SCORE_NON_RANGE_MULT": 3.0,
            },
        ),
        ("eth_shorts_open", {"flags.block_eth_shorts": False}),
        ("max_stop_cap_off", {"flags.max_stop_width": False}),
        ("max_stop_225", {"param_overrides.MAX_STOP_WIDTH_PTS": 225}),
        ("max_stop_250", {"param_overrides.MAX_STOP_WIDTH_PTS": 250}),
        ("min_box_125", {"param_overrides.MIN_BOX_WIDTH": 125}),
        ("min_box_100", {"param_overrides.MIN_BOX_WIDTH": 100}),
        ("loss_streak_3", {"param_overrides.LOSS_STREAK_THRESHOLD": 3}),
        ("loss_streak_4", {"param_overrides.LOSS_STREAK_THRESHOLD": 4}),
        ("caution_open_nrm3", {
            "param_overrides.BLOCK_CAUTION_REGIME": False,
            "param_overrides.SCORE_NON_RANGE_MULT": 3.0,
        }),
    ]:
        add_patch(candidates, current, "targeted", patch_name, patch, "structural_probe")

    return candidates


def build_refine_candidates(current: dict[str, Any]) -> list[RepairCandidate]:
    """Small combination pass around the strongest single-candidate levers."""
    candidates: list[RepairCandidate] = []

    ratchets = [0.45, 0.50, 0.55, 0.60]
    cooldowns = [50, 55, 60]
    for ratchet in ratchets:
        add_patch(
            candidates,
            current,
            "refine",
            f"ratchet_threshold_{fmt_value(ratchet)}",
            {"param_overrides.RATCHET_THRESHOLD_R": ratchet},
            "single_refine",
        )
        for cooldown in cooldowns:
            add_patch(
                candidates,
                current,
                "refine",
                f"ratchet_{fmt_value(ratchet)}_cooldown_{cooldown}",
                {
                    "param_overrides.RATCHET_THRESHOLD_R": ratchet,
                    "param_overrides.MIN_INTER_TRADE_GAP_MINUTES": cooldown,
                },
                "passed_combo_refine",
            )
        for score in [1.25, 1.35]:
            add_patch(
                candidates,
                current,
                "refine",
                f"ratchet_{fmt_value(ratchet)}_score_{fmt_value(score)}",
                {
                    "param_overrides.RATCHET_THRESHOLD_R": ratchet,
                    "param_overrides.SCORE_NORMAL": score,
                },
                "passed_combo_refine",
            )
        for patch_name, patch in [
            ("open12", {"flags.block_12_et": False}),
            ("minbox125", {"param_overrides.MIN_BOX_WIDTH": 125}),
            ("rvol175", {"param_overrides.RVOL_SCORE_THRESH": 1.75}),
            ("open04_rvol175", {"flags.block_04_et": False, "param_overrides.RVOL_SCORE_THRESH": 1.75}),
            ("eth_shorts_open", {"flags.block_eth_shorts": False}),
        ]:
            merged = {"param_overrides.RATCHET_THRESHOLD_R": ratchet, **patch}
            add_patch(
                candidates,
                current,
                "refine",
                f"ratchet_{fmt_value(ratchet)}_{patch_name}",
                merged,
                "passed_combo_refine",
            )
        for cooldown in [50, 55]:
            add_patch(
                candidates,
                current,
                "refine",
                f"ratchet_{fmt_value(ratchet)}_open12_cd{cooldown}",
                {
                    "param_overrides.RATCHET_THRESHOLD_R": ratchet,
                    "flags.block_12_et": False,
                    "param_overrides.MIN_INTER_TRADE_GAP_MINUTES": cooldown,
                },
                "passed_combo_refine",
            )
        add_patch(
            candidates,
            current,
            "refine",
            f"ratchet_{fmt_value(ratchet)}_open12_score125",
            {
                "param_overrides.RATCHET_THRESHOLD_R": ratchet,
                "flags.block_12_et": False,
                "param_overrides.SCORE_NORMAL": 1.25,
            },
            "passed_combo_refine",
        )
        add_patch(
            candidates,
            current,
            "refine",
            f"ratchet_{fmt_value(ratchet)}_open04_rvol175_open12",
            {
                "param_overrides.RATCHET_THRESHOLD_R": ratchet,
                "flags.block_04_et": False,
                "flags.block_12_et": False,
                "param_overrides.RVOL_SCORE_THRESH": 1.75,
            },
            "near_miss_combo_refine",
        )

    for patch_name, patch in [
        (
            "cooldown50_open12",
            {"param_overrides.MIN_INTER_TRADE_GAP_MINUTES": 50, "flags.block_12_et": False},
        ),
        (
            "cooldown55_open12",
            {"param_overrides.MIN_INTER_TRADE_GAP_MINUTES": 55, "flags.block_12_et": False},
        ),
        (
            "cooldown50_score125",
            {"param_overrides.MIN_INTER_TRADE_GAP_MINUTES": 50, "param_overrides.SCORE_NORMAL": 1.25},
        ),
        (
            "open12_score125",
            {"flags.block_12_et": False, "param_overrides.SCORE_NORMAL": 1.25},
        ),
        (
            "open12_minbox125",
            {"flags.block_12_et": False, "param_overrides.MIN_BOX_WIDTH": 125},
        ),
        (
            "open04_rvol175_open12",
            {"flags.block_04_et": False, "flags.block_12_et": False, "param_overrides.RVOL_SCORE_THRESH": 1.75},
        ),
        (
            "score125_rvol175",
            {"param_overrides.SCORE_NORMAL": 1.25, "param_overrides.RVOL_SCORE_THRESH": 1.75},
        ),
    ]:
        add_patch(candidates, current, "refine", patch_name, patch, "combo_refine")

    return candidates


def reverted_config(current: dict[str, Any], keys: list[str], previous_values: dict[str, Any]) -> dict[str, Any]:
    reverted = dict(current)
    for key in keys:
        previous = previous_values.get(key, MISSING)
        if previous == MISSING:
            reverted.pop(key, None)
        else:
            reverted[key] = previous
    return reverted


def int_variants(key: str, value: int) -> list[int]:
    if key.endswith("MIN_INTER_TRADE_GAP_MINUTES"):
        return [15, 20, 30, 35, 40, 45, 50, 60]
    if key.endswith("MIN_BOX_WIDTH"):
        return [100, 125, 150, 175, 200, 225]
    if key.endswith("LOSS_STREAK_THRESHOLD"):
        return [1, 2, 3, 4]
    return sorted({max(0, value + delta) for delta in [-2, -1, 1, 2]})


def float_variants(key: str, value: float) -> list[float]:
    explicit = {
        "param_overrides.SCORE_NORMAL": [1.25, 1.35, 1.5, 1.65, 1.75, 2.0],
        "param_overrides.RVOL_SCORE_THRESH": [1.25, 1.35, 1.5, 1.75, 2.0],
        "param_overrides.TP1_R": [1.20, 1.30, 1.35, 1.40, 1.45, 1.50, 1.60],
        "param_overrides.RATCHET_THRESHOLD_R": [0.50, 0.65, 0.75, 0.90, 1.0],
        "param_overrides.SCORE_NON_RANGE_MULT": [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5],
    }
    if key in explicit:
        return explicit[key]
    return sorted({round(value * factor, 6) for factor in [0.80, 0.90, 0.95, 1.05, 1.10, 1.20]})


def add_patch(
    candidates: list[RepairCandidate],
    current: dict[str, Any],
    stage: str,
    name: str,
    patch: dict[str, Any],
    source: str,
) -> None:
    merged = dict(current)
    changed = False
    for key, value in patch.items():
        if merged.get(key, MISSING) != value:
            merged[key] = value
            changed = True
    if changed:
        candidates.append(
            RepairCandidate(
                name=name,
                stage=stage,
                mutations=merged,
                intent="Current NQDTC OOS frequency diagnostic candidate.",
                source=source,
            )
        )


def dedupe(candidates: list[RepairCandidate], current: dict[str, Any]) -> list[RepairCandidate]:
    current_sig = signature(current)
    seen: set[str] = set()
    out: list[RepairCandidate] = []
    for item in candidates:
        sig = signature(item.mutations)
        if sig == current_sig or sig in seen:
            continue
        seen.add(sig)
        out.append(item)
    return out


def evaluate_candidates(
    *,
    baseline,
    candidates: list[RepairCandidate],
    data_end: str,
    max_workers: int,
    progress_path: Path,
):
    evaluations = []
    total = len(candidates)
    if max_workers <= 1:
        for idx, candidate in enumerate(candidates, start=1):
            evaluation = evaluate_one(candidate, serialize(baseline), data_end)
            evaluations.append(evaluation)
            append_progress(progress_path, idx, total, evaluation)
            print_progress(idx, total, evaluation)
        return evaluations

    with ProcessPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = [
            pool.submit(evaluate_one, candidate, serialize(baseline), data_end)
            for candidate in candidates
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            evaluation = future.result()
            evaluations.append(evaluation)
            append_progress(progress_path, completed, total, evaluation)
            print_progress(completed, total, evaluation)
    return evaluations


def evaluate_one(candidate: RepairCandidate, baseline_payload: dict[str, Any], data_end: str):
    from backtests.swing.auto.incumbent_repair import FoldMetrics, StrategyRun, WindowMetrics

    def wm(payload):
        return WindowMetrics(**payload)

    baseline = StrategyRun(
        strategy=baseline_payload["strategy"],
        mutations=baseline_payload["mutations"],
        is_metrics=wm(baseline_payload["is_metrics"]),
        oos_metrics=wm(baseline_payload["oos_metrics"]),
        fold_metrics=[
            FoldMetrics(name=item["name"], start=item["start"], end=item["end"], metrics=wm(item["metrics"]))
            for item in baseline_payload["fold_metrics"]
        ],
        assessment=baseline_payload["assessment"],
        action=baseline_payload["action"],
        error=baseline_payload.get("error", ""),
    )
    try:
        run = evaluate_strategy("nqdtc", candidate.mutations, data_end)
        return score_candidate(candidate, baseline, run)
    except Exception as exc:
        from backtests.swing.auto.incumbent_repair import CandidateEvaluation, StrategyRun, WindowMetrics

        run = StrategyRun(
            strategy="nqdtc",
            mutations=dict(candidate.mutations),
            is_metrics=WindowMetrics(),
            oos_metrics=WindowMetrics(),
            fold_metrics=[],
            assessment="ERROR",
            action="Error",
            error=str(exc),
        )
        return CandidateEvaluation(
            candidate=candidate,
            run=run,
            objective_delta=-999.0,
            passed=False,
            reasons=[f"error: {exc}"],
        )


def is_balanced_frequency_uplift(evaluation, baseline) -> bool:
    run = evaluation.run
    return (
        run.oos_metrics.total_trades > baseline.oos_metrics.total_trades
        and run.oos_metrics.net_r >= baseline.oos_metrics.net_r
        and run.is_metrics.total_trades >= int(baseline.is_metrics.total_trades * 0.90)
        and run.is_metrics.net_r >= baseline.is_metrics.net_r * 0.95
        and run.is_metrics.avg_r >= baseline.is_metrics.avg_r - 0.15
    )


def stage_leaders(evaluations, top_n: int) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for item in sorted(evaluations, key=lambda ev: ev.objective_delta, reverse=True):
        grouped.setdefault(item.candidate.stage, []).append(item)
    return {
        stage: [serialize(item) for item in items[:top_n]]
        for stage, items in grouped.items()
    }


def append_progress(path: Path, completed: int, total: int, evaluation) -> None:
    payload = {
        "completed": completed,
        "total": total,
        "candidate": evaluation.candidate.name,
        "stage": evaluation.candidate.stage,
        "source": evaluation.candidate.source,
        "objective_delta": evaluation.objective_delta,
        "passed": evaluation.passed,
        "reasons": evaluation.reasons,
        "is_metrics": serialize(evaluation.run.is_metrics),
        "oos_metrics": serialize(evaluation.run.oos_metrics),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def print_progress(completed: int, total: int, evaluation) -> None:
    oos = evaluation.run.oos_metrics
    is_m = evaluation.run.is_metrics
    print(
        f"[nqdtc-current] {completed}/{total} {evaluation.candidate.stage}/"
        f"{evaluation.candidate.name} obj={evaluation.objective_delta:+.3%} "
        f"OOS={oos.total_trades} {oos.net_r:+.2f}R "
        f"IS={is_m.total_trades} {is_m.net_r:+.1f}R passed={evaluation.passed}",
        flush=True,
    )


def format_report(summary: dict[str, Any]) -> str:
    lines = [
        "NQDTC Current OOS Frequency Repair Summary",
        "=" * 96,
        summary["run_spec"]["selection_oos_note"],
        f"Config: {summary['run_spec']['config_path']}",
        f"Data end: {summary['run_spec']['data_end']}",
        "",
    ]
    base = summary["baseline"]
    lines.append(
        "Baseline IS: "
        f"trades={base['is_metrics']['total_trades']} "
        f"PF={fmt(base['is_metrics']['profit_factor'])} "
        f"netR={base['is_metrics']['net_r']:.2f} "
        f"avgR={base['is_metrics']['avg_r']:.3f}"
    )
    lines.append(
        "Baseline OOS: "
        f"trades={base['oos_metrics']['total_trades']} "
        f"PF={fmt(base['oos_metrics']['profit_factor'])} "
        f"netR={base['oos_metrics']['net_r']:.2f} "
        f"avgR={base['oos_metrics']['avg_r']:.3f}"
    )
    lines.append(f"Candidates evaluated: {summary['candidate_count']}")

    for key, title in [
        ("balanced_frequency_uplifts", "Balanced frequency uplifts"),
        ("top", "Top objective candidates"),
        ("frequency_leaders", "Top frequency candidates"),
        ("oos_net_leaders", "Top OOS-net candidates"),
    ]:
        lines.append("")
        lines.append(title + ":")
        items = summary.get(key, [])[:12]
        if not items:
            lines.append("  None")
            continue
        for item in items:
            cand = item["candidate"]
            run = item["run"]
            oos = run["oos_metrics"]
            is_m = run["is_metrics"]
            lines.append(
                f"  {cand['stage']}/{cand['name']}: "
                f"obj={item['objective_delta']:+.2%}, passed={item['passed']}, "
                f"OOS trades={oos['total_trades']} netR={oos['net_r']:.2f} "
                f"avgR={oos['avg_r']:.3f}, IS trades={is_m['total_trades']} "
                f"netR={is_m['net_r']:.1f}, reasons={item.get('reasons', [])}"
            )
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    if isinstance(value, str):
        return value
    return f"{float(value):.2f}"


def fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}".replace(".", "p")
    return str(value).replace(".", "p")


def signature(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serialize(payload), indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
