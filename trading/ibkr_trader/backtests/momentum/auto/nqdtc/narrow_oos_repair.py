"""Narrow NQDTC OOS repair sweep.

This diagnostic follows the broad OOS repair run with a focused candidate set
around the only broad changes that showed positive OOS promise:
neutral/aligned regime recovery.  It deliberately keeps candidates inside
existing engine controls: non-range score multipliers, displacement, cooldown,
box width, and exit-management params.
"""
from __future__ import annotations

import argparse
import json
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
    RepairCandidate,
    score_candidate,
    serialize,
)
from backtests.swing.auto.oos_repair_diagnostics import (  # noqa: E402
    config_path_for,
    evaluate_strategy,
)


DROP = "__NQDTC_NARROW_DROP__"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "candidate_progress.jsonl"
    progress_path.write_text("", encoding="utf-8")

    base = read_json(config_path_for("nqdtc"))
    run_spec = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": "nqdtc",
        "data_end": args.data_end,
        "max_workers": args.max_workers,
        "profile": args.profile,
        "selection_oos_note": (
            "The OOS window is used for diagnosis/selection in this run; "
            "it is no longer an untouched holdout."
        ),
        "candidate_focus": (
            "narrow neutral/aligned recovery using existing score, displacement, "
            "cooldown, box-width, and exit controls"
        ),
    }
    write_json(output_dir / "run_spec.json", run_spec)

    print("[nqdtc] evaluating baseline", flush=True)
    baseline = evaluate_strategy("nqdtc", base, args.data_end)
    candidates = build_candidates(base, args.profile)
    print(
        f"[nqdtc] baseline OOS trades={baseline.oos_metrics.total_trades} "
        f"netR={baseline.oos_metrics.net_r:.3f}; evaluating {len(candidates)} candidates",
        flush=True,
    )

    started = time.time()
    evaluations = evaluate_candidates(
        baseline=baseline,
        candidates=candidates,
        data_end=args.data_end,
        max_workers=args.max_workers,
        progress_path=progress_path,
    )
    top = sorted(evaluations, key=lambda item: item.objective_delta, reverse=True)
    passed = [item for item in top if item.passed]
    positive_passed = [
        item for item in passed
        if item.run.oos_metrics.net_r > 0 and item.run.oos_metrics.total_trades >= baseline.oos_metrics.total_trades
    ]
    oos_leaders = sorted(
        evaluations,
        key=lambda item: (
            item.run.oos_metrics.net_r,
            item.run.oos_metrics.total_trades,
            item.run.is_metrics.net_r,
        ),
        reverse=True,
    )

    summary = {
        "run_spec": run_spec,
        "elapsed_seconds": round(time.time() - started, 2),
        "baseline": serialize(baseline),
        "candidate_count": len(candidates),
        "top": [serialize(item) for item in top[: args.top_n]],
        "passed": [serialize(item) for item in passed[: args.top_n]],
        "positive_passed": [serialize(item) for item in positive_passed[: args.top_n]],
        "oos_net_leaders": [serialize(item) for item in oos_leaders[: args.top_n]],
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.txt").write_text(format_report(summary), encoding="utf-8")
    print(f"[nqdtc] complete in {(time.time() - started) / 60.0:.1f} min", flush=True)
    print(f"Output: {output_dir.resolve()}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-end", default="2026-05-01")
    parser.add_argument(
        "--output-dir",
        default="backtests/output/momentum/nqdtc/narrow_oos_repair_20260504",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--profile", choices=["coarse", "refine"], default="coarse")
    return parser.parse_args()


def build_candidates(base: dict[str, Any], profile: str = "coarse") -> list[RepairCandidate]:
    if profile == "refine":
        return build_refine_candidates(base)

    candidates: list[RepairCandidate] = []

    neutral_mults = [1.5, 1.75, 2.0, 2.25, 2.5]
    aligned_mults = [1.5, 2.0, 2.5, 3.0]
    cooldowns = [None, 45, 60]
    max_boxes = [None, 450, 600]

    for mult in neutral_mults:
        for displacement_on in [False, True]:
            for cooldown in cooldowns:
                for max_box in max_boxes:
                    changes: dict[str, Any] = {
                        "param_overrides.BLOCK_NEUTRAL_REGIME": False,
                        "param_overrides.SCORE_NON_RANGE_MULT": mult,
                    }
                    if displacement_on:
                        changes["flags.displacement_threshold"] = True
                    if cooldown is not None:
                        changes["param_overrides.MIN_INTER_TRADE_GAP_MINUTES"] = cooldown
                    if max_box is not None:
                        changes["param_overrides.MAX_BOX_WIDTH"] = max_box
                    name = "neutral"
                    name += f"_nrm{label(mult)}"
                    if displacement_on:
                        name += "_disp"
                    if cooldown is not None:
                        name += f"_cd{cooldown}"
                    if max_box is not None:
                        name += f"_maxbox{max_box}"
                    candidates.append(candidate(base, name, changes))

    for mult in aligned_mults:
        for cooldown in [45, 60]:
            for max_box in [None, 600]:
                changes = {
                    "param_overrides.BLOCK_ALIGNED_REGIME": False,
                    "param_overrides.SCORE_NON_RANGE_MULT": mult,
                    "flags.displacement_threshold": True,
                    "param_overrides.MIN_INTER_TRADE_GAP_MINUTES": cooldown,
                }
                if max_box is not None:
                    changes["param_overrides.MAX_BOX_WIDTH"] = max_box
                name = f"aligned_nrm{label(mult)}_disp_cd{cooldown}"
                if max_box is not None:
                    name += f"_maxbox{max_box}"
                candidates.append(candidate(base, name, changes))

    for mult in [2.0, 2.5, 3.0]:
        for cooldown in [45, 60]:
            changes = {
                "param_overrides.BLOCK_NEUTRAL_REGIME": False,
                "param_overrides.BLOCK_ALIGNED_REGIME": False,
                "param_overrides.SCORE_NON_RANGE_MULT": mult,
                "flags.displacement_threshold": True,
                "param_overrides.MIN_INTER_TRADE_GAP_MINUTES": cooldown,
            }
            candidates.append(candidate(base, f"neutral_aligned_nrm{label(mult)}_disp_cd{cooldown}", changes))

    overlay_specs = [
        ("tp_partial_drop", {"param_overrides.TP1_PARTIAL_PCT": DROP}),
        ("tp_partial_045", {"param_overrides.TP1_PARTIAL_PCT": 0.45}),
        ("tp1_150", {"param_overrides.TP1_R": 1.50}),
        ("tp1_168", {"param_overrides.TP1_R": 1.68}),
        ("ratchet_lock_045", {"param_overrides.RATCHET_LOCK_PCT": 0.45}),
        ("ratchet_lock_055", {"param_overrides.RATCHET_LOCK_PCT": 0.55}),
    ]
    seed_changes = [
        (
            "neutral_nrm175_disp_cd45",
            {
                "param_overrides.BLOCK_NEUTRAL_REGIME": False,
                "param_overrides.SCORE_NON_RANGE_MULT": 1.75,
                "flags.displacement_threshold": True,
                "param_overrides.MIN_INTER_TRADE_GAP_MINUTES": 45,
            },
        ),
        (
            "neutral_nrm200_disp_cd45",
            {
                "param_overrides.BLOCK_NEUTRAL_REGIME": False,
                "param_overrides.SCORE_NON_RANGE_MULT": 2.0,
                "flags.displacement_threshold": True,
                "param_overrides.MIN_INTER_TRADE_GAP_MINUTES": 45,
            },
        ),
        (
            "neutral_nrm225_disp_cd45",
            {
                "param_overrides.BLOCK_NEUTRAL_REGIME": False,
                "param_overrides.SCORE_NON_RANGE_MULT": 2.25,
                "flags.displacement_threshold": True,
                "param_overrides.MIN_INTER_TRADE_GAP_MINUTES": 45,
            },
        ),
        (
            "neutral_nrm200_disp_cd45_maxbox600",
            {
                "param_overrides.BLOCK_NEUTRAL_REGIME": False,
                "param_overrides.SCORE_NON_RANGE_MULT": 2.0,
                "flags.displacement_threshold": True,
                "param_overrides.MIN_INTER_TRADE_GAP_MINUTES": 45,
                "param_overrides.MAX_BOX_WIDTH": 600,
            },
        ),
    ]
    for seed_name, seed in seed_changes:
        for overlay_name, overlay in overlay_specs:
            merged = dict(seed)
            merged.update(overlay)
            candidates.append(candidate(base, f"{seed_name}_{overlay_name}", merged))

    return dedupe(candidates)


def build_refine_candidates(base: dict[str, Any]) -> list[RepairCandidate]:
    candidates: list[RepairCandidate] = []

    for mult in [2.2, 2.25, 2.3, 2.35]:
        for cooldown in [40, 45, 50]:
            for partial_name, partial_value in [
                ("p025", DROP),
                ("p030", 0.30),
                ("p035", 0.35),
                ("p040", 0.40),
            ]:
                changes = neutral_refine_seed(mult, cooldown)
                if partial_value == DROP:
                    changes["param_overrides.TP1_PARTIAL_PCT"] = DROP
                else:
                    changes["param_overrides.TP1_PARTIAL_PCT"] = partial_value
                candidates.append(candidate(base, f"refine_nrm{label(mult)}_cd{cooldown}_{partial_name}", changes))

    base_seed = neutral_refine_seed(2.25, 45)
    base_seed["param_overrides.TP1_PARTIAL_PCT"] = DROP
    for tp1 in [1.35, 1.45, 1.50, 1.55, 1.60]:
        changes = dict(base_seed)
        changes["param_overrides.TP1_R"] = tp1
        candidates.append(candidate(base, f"refine_nrm2p25_cd45_p025_tp1_{label(tp1)}", changes))

    for lock in [0.40, 0.45, 0.50]:
        changes = dict(base_seed)
        changes["param_overrides.RATCHET_LOCK_PCT"] = lock
        candidates.append(candidate(base, f"refine_nrm2p25_cd45_p025_lock_{label(lock)}", changes))

    for max_box in [550, 600, 650, 700, 800]:
        changes = dict(base_seed)
        changes["param_overrides.MAX_BOX_WIDTH"] = max_box
        candidates.append(candidate(base, f"refine_nrm2p25_cd45_p025_maxbox{max_box}", changes))

    for min_box in [175, 200, 225]:
        changes = dict(base_seed)
        changes["param_overrides.MIN_BOX_WIDTH"] = min_box
        candidates.append(candidate(base, f"refine_nrm2p25_cd45_p025_minbox{min_box}", changes))

    for score_normal in [1.4, 1.55, 1.6]:
        changes = dict(base_seed)
        changes["param_overrides.SCORE_NORMAL"] = score_normal
        candidates.append(candidate(base, f"refine_nrm2p25_cd45_p025_score_{label(score_normal)}", changes))

    return dedupe(candidates)


def neutral_refine_seed(mult: float, cooldown: int) -> dict[str, Any]:
    return {
        "param_overrides.BLOCK_NEUTRAL_REGIME": False,
        "param_overrides.SCORE_NON_RANGE_MULT": mult,
        "flags.displacement_threshold": True,
        "param_overrides.MIN_INTER_TRADE_GAP_MINUTES": cooldown,
    }


def candidate(base: dict[str, Any], name: str, changes: dict[str, Any]) -> RepairCandidate:
    mutations = dict(base)
    for key, value in changes.items():
        if value == DROP:
            mutations.pop(key, None)
        else:
            mutations[key] = value
    return RepairCandidate(
        name=name,
        stage="narrow_targeted",
        mutations=mutations,
        intent="Narrow NQDTC OOS repair candidate.",
        source="nqdtc_narrow_oos_repair",
    )


def dedupe(candidates: list[RepairCandidate]) -> list[RepairCandidate]:
    seen: set[str] = set()
    out: list[RepairCandidate] = []
    for item in candidates:
        key = json.dumps(item.mutations, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
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
    with ProcessPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = [
            pool.submit(evaluate_one, candidate, serialize(baseline), data_end)
            for candidate in candidates
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            evaluation = future.result()
            evaluations.append(evaluation)
            append_progress(progress_path, completed, total, evaluation)
            print(
                f"[nqdtc] {completed}/{total} {evaluation.candidate.name} "
                f"obj={evaluation.objective_delta:+.3%} "
                f"oos={evaluation.run.oos_metrics.total_trades} "
                f"{evaluation.run.oos_metrics.net_r:+.2f}R "
                f"passed={evaluation.passed}",
                flush=True,
            )
    return evaluations


def evaluate_one(candidate: RepairCandidate, baseline_payload: dict[str, Any], data_end: str):
    from backtests.swing.auto.incumbent_repair import StrategyRun, WindowMetrics, FoldMetrics

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
        from backtests.swing.auto.incumbent_repair import CandidateEvaluation

        return CandidateEvaluation(
            candidate=candidate,
            run=run,
            objective_delta=-999.0,
            passed=False,
            reasons=[f"error: {exc}"],
        )


def append_progress(path: Path, completed: int, total: int, evaluation) -> None:
    payload = {
        "completed": completed,
        "total": total,
        "candidate": evaluation.candidate.name,
        "objective_delta": evaluation.objective_delta,
        "passed": evaluation.passed,
        "reasons": evaluation.reasons,
        "is_metrics": serialize(evaluation.run.is_metrics),
        "oos_metrics": serialize(evaluation.run.oos_metrics),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def format_report(summary: dict[str, Any]) -> str:
    lines = [
        "NQDTC Narrow OOS Repair Summary",
        "=" * 88,
        summary["run_spec"]["selection_oos_note"],
        f"Data end: {summary['run_spec']['data_end']}",
        "",
    ]
    base = summary["baseline"]
    lines.append(
        "Baseline IS: "
        f"trades={base['is_metrics']['total_trades']} "
        f"netR={base['is_metrics']['net_r']:.1f} "
        f"avgR={base['is_metrics']['avg_r']:.3f}"
    )
    lines.append(
        "Baseline OOS: "
        f"trades={base['oos_metrics']['total_trades']} "
        f"netR={base['oos_metrics']['net_r']:.1f} "
        f"avgR={base['oos_metrics']['avg_r']:.3f}"
    )
    lines.append(f"Candidates evaluated: {summary['candidate_count']}")
    for label, title in [
        ("positive_passed", "Positive-OOS passed candidates"),
        ("passed", "Top passed candidates"),
        ("oos_net_leaders", "Top OOS-net candidates"),
    ]:
        lines.append("")
        lines.append(title + ":")
        items = summary.get(label, [])[:10]
        if not items:
            lines.append("  None")
            continue
        for item in items:
            c = item["candidate"]
            run = item["run"]
            lines.append(
                f"  {c['name']}: obj={item['objective_delta']:+.2%}, "
                f"passed={item['passed']}, "
                f"OOS trades={run['oos_metrics']['total_trades']} "
                f"netR={run['oos_metrics']['net_r']:.2f} "
                f"avgR={run['oos_metrics']['avg_r']:.3f}, "
                f"IS trades={run['is_metrics']['total_trades']} "
                f"netR={run['is_metrics']['net_r']:.1f}, "
                f"reasons={item.get('reasons', [])}"
            )
    return "\n".join(lines) + "\n"


def label(value: float) -> str:
    return str(value).replace(".", "p")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
