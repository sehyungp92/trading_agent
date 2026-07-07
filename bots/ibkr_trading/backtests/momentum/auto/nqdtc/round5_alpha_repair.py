"""Round-5 NQDTC alpha repair search focused on PF and net return.

This is a diagnostic/repair runner, not a fresh holdout validation.  It uses
the current OOS window to understand and repair the round-5 PF/net dilution.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not locate repository root.")


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.shared.validation.oos_validation import (  # noqa: E402
    BACKTEST_START,
    BACKTEST_START_DATE,
    OOS_CUTOFF,
    OOS_CUTOFF_DATE,
    WindowMetrics,
    _compute_oos_months,
    _get_entry_time,
    _get_r_multiple,
    _window_months,
    compute_window_metrics,
)
from backtests.swing.auto.incumbent_repair import (  # noqa: E402
    FoldMetrics,
    MISSING,
    RepairCandidate,
    StrategyRun,
    build_fold_metrics,
    read_json,
    serialize,
    short_key,
    write_json,
)
from backtests.swing.auto.oos_repair_diagnostics import (  # noqa: E402
    build_extra_ablation_candidates,
    build_extra_historical_features,
    build_extra_targeted_candidates,
    evaluate_strategy,
)

ROUND4_CONFIG = ROOT / "backtests/output/momentum/nqdtc/round_4/optimized_config.json"
ROUND5_CONFIG = ROOT / "backtests/output/momentum/nqdtc/round_5/optimized_config.json"
DATA_DIR = ROOT / "backtests/momentum/data/raw"


@dataclass
class AlphaEvaluation:
    candidate: RepairCandidate
    run: StrategyRun
    score: float
    strict_pass: bool
    repair_pass: bool
    reasons: list[str] = field(default_factory=list)
    deltas: dict[str, Any] = field(default_factory=dict)


def main() -> None:
    args = parse_args()
    logging.getLogger("strategies.momentum.nqdtc.box").setLevel(logging.WARNING)
    logging.getLogger("backtests.momentum.engine.nqdtc_engine").setLevel(logging.WARNING)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "candidate_progress.jsonl"
    progress_path.write_text("", encoding="utf-8")

    started = time.time()
    round4 = read_json(Path(args.round4_config))
    round5 = read_json(Path(args.round5_config))

    spec = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_end": args.data_end,
        "oos_cutoff": OOS_CUTOFF_DATE.isoformat(),
        "round4_config": str(Path(args.round4_config).resolve()),
        "round5_config": str(Path(args.round5_config).resolve()),
        "max_workers": args.max_workers,
        "top_n": args.top_n,
        "selection_oos_note": (
            "The OOS window is used for diagnosis and repair selection here; "
            "any promotion still needs forward monitoring or a fresh holdout."
        ),
    }
    write_json(output_dir / "run_spec.json", spec)

    print("[alpha-repair] evaluating round4 and round5 baselines", flush=True)
    round4_run = evaluate_strategy("nqdtc", round4, args.data_end)
    round5_run = evaluate_strategy("nqdtc", round5, args.data_end)

    print("[alpha-repair] running full cohort diagnostics", flush=True)
    round4_full = run_full_replay(round4, args.data_end)
    round5_full = run_full_replay(round5, args.data_end)
    diagnostics = {
        "round4": diagnose_replay(round4_full, args.data_end),
        "round5": diagnose_replay(round5_full, args.data_end),
        "round5_vs_round4": compare_diagnostics(round4_full, round5_full, args.data_end),
    }
    write_json(output_dir / "round5_vs_round4_diagnostics.json", diagnostics)

    candidates = build_candidate_suite(round5, round4, args)
    print(
        f"[alpha-repair] round5 IS={round5_run.is_metrics.total_trades} "
        f"{round5_run.is_metrics.net_r:+.2f}R PF={fmt_pf(round5_run.is_metrics.profit_factor)}; "
        f"OOS={round5_run.oos_metrics.total_trades} {round5_run.oos_metrics.net_r:+.2f}R "
        f"PF={fmt_pf(round5_run.oos_metrics.profit_factor)}; evaluating {len(candidates)} candidates",
        flush=True,
    )

    evaluations = evaluate_candidates(
        candidates=candidates,
        round5_baseline=round5_run,
        round4_baseline=round4_run,
        data_end=args.data_end,
        max_workers=args.max_workers,
        progress_path=progress_path,
    )
    ranked = sorted(evaluations, key=rank_key, reverse=True)
    strict = [ev for ev in ranked if ev.strict_pass]
    repair = [ev for ev in ranked if ev.repair_pass]

    top_for_full = ranked[: args.full_metrics_top_n]
    print(f"[alpha-repair] computing full metrics for top {len(top_for_full)} candidates", flush=True)
    full_metrics = compute_full_metrics_table(top_for_full, round4, round5, args)

    summary = {
        "run_spec": {**spec, "candidate_count": len(candidates), "elapsed_seconds": round(time.time() - started, 2)},
        "round4_baseline": serialize(round4_run),
        "round5_baseline": serialize(round5_run),
        "round5_vs_round4_diagnostics": diagnostics["round5_vs_round4"],
        "candidate_count": len(candidates),
        "strict_uplifts": [serialize_alpha(ev) for ev in strict[: args.top_n]],
        "repair_uplifts": [serialize_alpha(ev) for ev in repair[: args.top_n]],
        "ranked": [serialize_alpha(ev) for ev in ranked[: args.top_n]],
        "is_pf_leaders": [serialize_alpha(ev) for ev in sorted(evaluations, key=is_pf_key, reverse=True)[: args.top_n]],
        "is_net_leaders": [serialize_alpha(ev) for ev in sorted(evaluations, key=is_net_key, reverse=True)[: args.top_n]],
        "oos_net_leaders": [serialize_alpha(ev) for ev in sorted(evaluations, key=oos_net_key, reverse=True)[: args.top_n]],
        "stage_leaders": stage_leaders(evaluations, args.top_n),
        "full_metrics": full_metrics,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.txt").write_text(format_report(summary), encoding="utf-8")
    print(f"[alpha-repair] complete in {(time.time() - started) / 60.0:.1f} min", flush=True)
    print(f"Output: {output_dir.resolve()}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round4-config", default=str(ROUND4_CONFIG))
    parser.add_argument("--round5-config", default=str(ROUND5_CONFIG))
    parser.add_argument("--data-end", default="2026-05-24")
    parser.add_argument(
        "--output-dir",
        default="backtests/output/momentum/nqdtc/round_5/alpha_repair_20260524",
    )
    parser.add_argument("--max-workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--full-metrics-top-n", type=int, default=18)
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--include-native-targets", action="store_true")
    return parser.parse_args()


def build_candidate_suite(current: dict[str, Any], round4: dict[str, Any], args: argparse.Namespace) -> list[RepairCandidate]:
    candidates: list[RepairCandidate] = []

    def add(name: str, patch: dict[str, Any], stage: str, source: str, intent: str = "") -> None:
        merged = dict(current)
        changed = False
        for key, value in patch.items():
            if value == "__DROP__":
                if key in merged:
                    merged.pop(key)
                    changed = True
            elif merged.get(key, MISSING) != value:
                merged[key] = value
                changed = True
        if changed:
            candidates.append(
                RepairCandidate(
                    name=name,
                    stage=stage,
                    mutations=merged,
                    intent=intent or "Round-5 alpha repair probe.",
                    source=source,
                )
            )

    def add_config(name: str, mutations: dict[str, Any], stage: str, source: str, intent: str = "") -> None:
        if mutations != current:
            candidates.append(
                RepairCandidate(
                    name=name,
                    stage=stage,
                    mutations=dict(mutations),
                    intent=intent or "Whole-config comparison candidate.",
                    source=source,
                )
            )

    add_config("round4_full_revert", round4, "baseline_compare", "round4")

    for cand in build_extra_ablation_candidates("nqdtc", current):
        candidates.append(
            RepairCandidate(
                name=cand.name,
                stage="ablation_all_rounds",
                mutations=cand.mutations,
                intent=cand.intent,
                source=cand.source,
            )
        )

    for key in sorted(set(current) | set(round4)):
        if current.get(key, MISSING) != round4.get(key, MISSING):
            if round4.get(key, MISSING) == MISSING:
                add(f"revert_round4_drop_{short_key(key)}", {key: "__DROP__"}, "round4_key_revert", "round4")
            else:
                add(f"revert_round4_{short_key(key)}", {key: round4[key]}, "round4_key_revert", "round4")

    for name, mutations, previous_values in build_extra_historical_features("nqdtc"):
        active = [key for key, value in mutations.items() if current.get(key, MISSING) == value]
        for key in active:
            previous = previous_values.get(key, MISSING)
            if previous == MISSING:
                add(f"historical_drop_{name}_{short_key(key)}", {key: "__DROP__"}, "historical_key_revert", name)
            else:
                add(f"historical_prior_{name}_{short_key(key)}", {key: previous}, "historical_key_revert", name)

    if args.include_native_targets:
        for cand in build_extra_targeted_candidates("nqdtc", current):
            candidates.append(
                RepairCandidate(
                    name=cand.name,
                    stage="native_targeted",
                    mutations=cand.mutations,
                    intent=cand.intent,
                    source=cand.source,
                )
            )

    # Neutral gate and core interaction grid.
    for block_neutral in [True, False]:
        for mult in [2.10, 2.25, 2.35, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00]:
            patch = {"param_overrides.BLOCK_NEUTRAL_REGIME": block_neutral}
            if not block_neutral:
                patch["param_overrides.SCORE_NON_RANGE_MULT"] = mult
            label = "blocked" if block_neutral else f"nrm_{fmt_value(mult)}"
            add(f"neutral_{label}", patch, "neutral_threshold", "neutral_gate")
            if block_neutral:
                continue
            for offset in [0.236, 0.248, 0.252, 0.264, 0.276, 0.288, 0.300, 0.320]:
                add(
                    f"neutral_nrm_{fmt_value(mult)}_c_{fmt_value(offset)}",
                    {**patch, "param_overrides.C_ENTRY_OFFSET_ATR_STANDARD": offset},
                    "neutral_c_offset",
                    "neutral_gate",
                )
                for tp1 in [1.45, 1.55, 1.60, 1.70, 1.80]:
                    add(
                        f"neutral_nrm_{fmt_value(mult)}_c_{fmt_value(offset)}_tp1_{fmt_value(tp1)}",
                        {
                            **patch,
                            "param_overrides.C_ENTRY_OFFSET_ATR_STANDARD": offset,
                            "param_overrides.TP1_R": tp1,
                        },
                        "neutral_exit_grid",
                        "neutral_gate_exit",
                    )

    # Exit and monetization probes centered on the current book.
    for tp1 in [1.35, 1.45, 1.50, 1.55, 1.60, 1.65, 1.70, 1.80, 1.90]:
        for pct in [0.35, 0.40, 0.45, 0.50, 0.55]:
            add(
                f"tp1_{fmt_value(tp1)}_pct_{fmt_value(pct)}",
                {"param_overrides.TP1_R": tp1, "param_overrides.TP1_PARTIAL_PCT": pct},
                "exit_shape",
                "tp1_grid",
            )
    for tp2 in [2.00, 2.125, 2.25, 2.375, 2.50, 2.75, 3.00]:
        for pct in [0.10, 0.15, 0.20, 0.25, 0.30]:
            add(
                f"tp2_{fmt_value(tp2)}_pct_{fmt_value(pct)}",
                {"param_overrides.TP2_R": tp2, "param_overrides.TP2_PARTIAL_PCT": pct},
                "exit_shape",
                "tp2_grid",
            )
    for mode in ["off", "range_only", "degraded_only", "range_degraded"]:
        add(f"tp1_cap_{mode}", {"param_overrides.TP1_ONLY_CAP_MODE": mode}, "exit_shape", "tp1_cap")

    ratchets = [
        ("mfe_balanced", 2.0, 0.80, 3.0, 1.35, 4.0, 2.00),
        ("mfe_fast", 1.75, 0.75, 2.75, 1.30, 3.75, 1.90),
        ("mfe_loose", 2.25, 0.75, 3.25, 1.25, 4.25, 1.80),
        ("mfe_quality", 2.0, 0.95, 3.0, 1.55, 4.0, 2.25),
    ]
    for name, t1, l1, t2, l2, t3, l3 in ratchets:
        add(
            name,
            {
                "param_overrides.MFE_RATCHET_TIERS_ENABLED": True,
                "param_overrides.MFE_RATCHET_T1_R": t1,
                "param_overrides.MFE_RATCHET_T1_LOCK_R": l1,
                "param_overrides.MFE_RATCHET_T2_R": t2,
                "param_overrides.MFE_RATCHET_T2_LOCK_R": l2,
                "param_overrides.MFE_RATCHET_T3_R": t3,
                "param_overrides.MFE_RATCHET_T3_LOCK_R": l3,
            },
            "exit_protection",
            "mfe_ratchet",
        )

    # Context filters should rescue weak-score or wide-box quality without a
    # blunt Neutral re-block.
    for max_box in [175.0, 200.0, 225.0, 250.0, 275.0]:
        for min_rvol in [1.50, 1.75, 2.00, 2.25]:
            add(
                f"weak_score_box_{fmt_value(max_box)}_rvol_{fmt_value(min_rvol)}",
                {
                    "param_overrides.WEAK_SCORE_BAND_FILTER_ENABLED": True,
                    "param_overrides.WEAK_SCORE_BAND_LOW": 2.5,
                    "param_overrides.WEAK_SCORE_BAND_HIGH": 3.0,
                    "param_overrides.WEAK_SCORE_BAND_MAX_BOX_WIDTH": max_box,
                    "param_overrides.WEAK_SCORE_BAND_MIN_RVOL": min_rvol,
                },
                "context_filter",
                "weak_score_band",
            )
    for width in [200.0, 225.0, 250.0, 275.0, 300.0, 325.0]:
        for score in [2.5, 3.0, 3.5, 4.0]:
            for min_rvol in [1.50, 1.75, 2.00]:
                add(
                    f"wide_box_{fmt_value(width)}_score_{fmt_value(score)}_rvol_{fmt_value(min_rvol)}",
                    {
                        "param_overrides.WIDE_BOX_SCORE_FILTER_ENABLED": True,
                        "param_overrides.WIDE_BOX_MIN_WIDTH": width,
                        "param_overrides.WIDE_BOX_MIN_SCORE": score,
                        "param_overrides.WIDE_BOX_MIN_RVOL": min_rvol,
                    },
                    "context_filter",
                    "wide_box_score",
                )

    # Entry and structure filters.
    for min_width in [50, 75, 100, 125, 150, 175]:
        add(f"min_box_{min_width}", {"param_overrides.MIN_BOX_WIDTH": min_width}, "box_filter", "box_width")
    for max_width in [175, 200, 225, 250, 275, 300, 350]:
        add(f"max_box_{max_width}", {"param_overrides.MAX_BOX_WIDTH": max_width}, "box_filter", "box_width")
    for gap in [30, 35, 40, 45, 50, 60, 75, 90]:
        add(f"cooldown_{gap}", {"param_overrides.MIN_INTER_TRADE_GAP_MINUTES": gap}, "timing_filter", "cooldown")
    for stop_width in [150, 175, 200, 225, 250, 275]:
        add(f"max_stop_width_{stop_width}", {"param_overrides.MAX_STOP_WIDTH_PTS": stop_width}, "risk_filter", "stop_width")
    for threshold in [2, 3, 4]:
        for skip in [6, 9, 12, 18]:
            add(
                f"loss_streak_{threshold}_skip_{skip}",
                {
                    "param_overrides.LOSS_STREAK_THRESHOLD": threshold,
                    "param_overrides.LOSS_STREAK_SKIP_BARS": skip,
                },
                "risk_filter",
                "loss_streak",
            )
    add("a_disabled", {"param_overrides.A_ENTRY_ENABLED": False}, "entry_filter", "a_entry")
    for a_score in [2.0, 2.5, 3.0, 3.5, 4.0]:
        add(f"a_min_score_{fmt_value(a_score)}", {"param_overrides.A_MIN_SCORE": a_score}, "entry_filter", "a_entry")
    for a_width in [150.0, 175.0, 200.0, 225.0, 250.0]:
        for ttl in [8, 10, 12, 14, 16]:
            add(
                f"a_box_{fmt_value(a_width)}_ttl_{ttl}",
                {"param_overrides.A_MAX_BOX_WIDTH": a_width, "param_overrides.A_TTL_5M_BARS": ttl},
                "entry_filter",
                "a_entry",
            )
    add(
        "a_block_weak_band",
        {"param_overrides.A_BLOCK_WEAK_SCORE_BAND": True},
        "entry_filter",
        "a_entry",
    )

    # Combined candidates: keep neutral open but insist on quality context plus
    # conservative exit/risk interactions.
    for mult in [2.25, 2.35, 2.50, 2.75, 3.00]:
        for offset in [0.252, 0.264, 0.276, 0.288]:
            for max_box in [200.0, 225.0, 250.0]:
                add(
                    f"combo_nrm_{fmt_value(mult)}_c_{fmt_value(offset)}_weak_box_{fmt_value(max_box)}",
                    {
                        "param_overrides.BLOCK_NEUTRAL_REGIME": False,
                        "param_overrides.SCORE_NON_RANGE_MULT": mult,
                        "param_overrides.C_ENTRY_OFFSET_ATR_STANDARD": offset,
                        "param_overrides.WEAK_SCORE_BAND_FILTER_ENABLED": True,
                        "param_overrides.WEAK_SCORE_BAND_MAX_BOX_WIDTH": max_box,
                        "param_overrides.WEAK_SCORE_BAND_MIN_RVOL": 1.75,
                    },
                    "combined_alpha",
                    "neutral_context",
                )
            for stop_width in [175, 200, 225]:
                add(
                    f"combo_nrm_{fmt_value(mult)}_c_{fmt_value(offset)}_stop_{stop_width}",
                    {
                        "param_overrides.BLOCK_NEUTRAL_REGIME": False,
                        "param_overrides.SCORE_NON_RANGE_MULT": mult,
                        "param_overrides.C_ENTRY_OFFSET_ATR_STANDARD": offset,
                        "param_overrides.MAX_STOP_WIDTH_PTS": stop_width,
                    },
                    "combined_alpha",
                    "neutral_risk",
                )
            for gap in [45, 60, 75]:
                add(
                    f"combo_nrm_{fmt_value(mult)}_c_{fmt_value(offset)}_gap_{gap}",
                    {
                        "param_overrides.BLOCK_NEUTRAL_REGIME": False,
                        "param_overrides.SCORE_NON_RANGE_MULT": mult,
                        "param_overrides.C_ENTRY_OFFSET_ATR_STANDARD": offset,
                        "param_overrides.MIN_INTER_TRADE_GAP_MINUTES": gap,
                    },
                    "combined_alpha",
                    "neutral_timing",
                )

    candidates = dedupe_candidates(candidates, current)
    if args.candidate_limit and args.candidate_limit > 0:
        return candidates[: args.candidate_limit]
    return candidates


def evaluate_candidates(
    *,
    candidates: list[RepairCandidate],
    round5_baseline: StrategyRun,
    round4_baseline: StrategyRun,
    data_end: str,
    max_workers: int,
    progress_path: Path,
) -> list[AlphaEvaluation]:
    baseline5 = serialize(round5_baseline)
    baseline4 = serialize(round4_baseline)
    evaluations: list[AlphaEvaluation] = []
    total = len(candidates)
    if max_workers <= 1:
        for idx, candidate in enumerate(candidates, start=1):
            ev = evaluate_one(candidate, baseline5, baseline4, data_end)
            evaluations.append(ev)
            append_progress(progress_path, idx, total, ev)
            print_progress(idx, total, ev)
        return evaluations

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(evaluate_one, c, baseline5, baseline4, data_end) for c in candidates]
        for idx, future in enumerate(as_completed(futures), start=1):
            ev = future.result()
            evaluations.append(ev)
            append_progress(progress_path, idx, total, ev)
            print_progress(idx, total, ev)
    return evaluations


def evaluate_one(
    candidate: RepairCandidate,
    baseline5_payload: dict[str, Any],
    baseline4_payload: dict[str, Any],
    data_end: str,
) -> AlphaEvaluation:
    baseline5 = strategy_run_from_payload(baseline5_payload)
    baseline4 = strategy_run_from_payload(baseline4_payload)
    try:
        run = evaluate_strategy("nqdtc", candidate.mutations, data_end)
        return score_alpha_candidate(candidate, run, baseline5, baseline4)
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
        return AlphaEvaluation(
            candidate=candidate,
            run=run,
            score=-999.0,
            strict_pass=False,
            repair_pass=False,
            reasons=[f"error: {exc}"],
        )


def score_alpha_candidate(
    candidate: RepairCandidate,
    run: StrategyRun,
    baseline5: StrategyRun,
    baseline4: StrategyRun,
) -> AlphaEvaluation:
    im = run.is_metrics
    om = run.oos_metrics
    b5i = baseline5.is_metrics
    b5o = baseline5.oos_metrics
    b4i = baseline4.is_metrics
    b4o = baseline4.oos_metrics

    deltas = {
        "is_net_r_delta_r5": im.net_r - b5i.net_r,
        "is_pf_delta_r5": pf_value(im.profit_factor) - pf_value(b5i.profit_factor),
        "is_avg_r_delta_r5": im.avg_r - b5i.avg_r,
        "is_trade_delta_r5": im.total_trades - b5i.total_trades,
        "oos_net_r_delta_r5": om.net_r - b5o.net_r,
        "oos_pf_delta_r5": pf_value(om.profit_factor) - pf_value(b5o.profit_factor),
        "oos_avg_r_delta_r5": om.avg_r - b5o.avg_r,
        "oos_trade_delta_r5": om.total_trades - b5o.total_trades,
        "is_net_r_delta_r4": im.net_r - b4i.net_r,
        "is_pf_delta_r4": pf_value(im.profit_factor) - pf_value(b4i.profit_factor),
        "oos_net_r_delta_r4": om.net_r - b4o.net_r,
        "oos_pf_delta_r4": pf_value(om.profit_factor) - pf_value(b4o.profit_factor),
    }

    fold_rs = [float(f.metrics.net_r) for f in run.fold_metrics]
    fold_pfs = [pf_value(f.metrics.profit_factor) for f in run.fold_metrics if f.metrics.total_trades > 0]
    positive_folds = sum(1 for value in fold_rs if value > 0)
    negative_folds = sum(1 for value in fold_rs if value < 0)
    deltas.update(
        {
            "positive_folds": positive_folds,
            "negative_folds": negative_folds,
            "min_fold_net_r": min(fold_rs) if fold_rs else 0.0,
            "median_fold_pf": median(fold_pfs) if fold_pfs else 0.0,
        }
    )

    reasons: list[str] = []
    if im.net_r < b5i.net_r:
        reasons.append("IS netR below round5")
    if pf_value(im.profit_factor) < pf_value(b5i.profit_factor):
        reasons.append("IS PF below round5")
    if om.net_r < b5o.net_r:
        reasons.append("OOS netR below round5")
    if pf_value(om.profit_factor) < pf_value(b5o.profit_factor):
        reasons.append("OOS PF below round5")
    if im.total_trades < max(100, int(b5i.total_trades * 0.72)):
        reasons.append("IS trade count too low")
    if om.total_trades < max(4, int(b5o.total_trades * 0.65)):
        reasons.append("OOS trade count too low")
    if positive_folds < max(4, len(run.fold_metrics) - 2):
        reasons.append("Fold robustness weak")
    material_uplift = (
        im.net_r > b5i.net_r + 0.25
        or pf_value(im.profit_factor) > pf_value(b5i.profit_factor) + 0.03
        or om.net_r > b5o.net_r + 0.25
        or pf_value(om.profit_factor) > pf_value(b5o.profit_factor) + 0.10
    )
    if not material_uplift:
        reasons.append("No material uplift versus round5")

    strict_pass = not reasons
    repair_pass = (
        material_uplift
        and
        im.net_r >= b4i.net_r
        and pf_value(im.profit_factor) >= pf_value(b5i.profit_factor)
        and om.net_r >= b5o.net_r * 0.85
        and om.net_r > b4o.net_r
        and im.total_trades >= int(b4i.total_trades * 0.90)
        and positive_folds >= max(4, len(run.fold_metrics) - 3)
    )

    score = (
        1.45 * norm(deltas["is_net_r_delta_r5"], 20.0)
        + 1.25 * norm(deltas["is_pf_delta_r5"], 0.50)
        + 0.75 * norm(deltas["is_avg_r_delta_r5"], 0.20)
        + 1.80 * norm(deltas["oos_net_r_delta_r5"], 4.0)
        + 1.15 * norm(deltas["oos_pf_delta_r5"], 1.00)
        + 0.50 * norm(deltas["oos_trade_delta_r5"], 4.0)
        + 0.30 * norm(deltas["is_trade_delta_r5"], 45.0)
        + 0.30 * norm(deltas["is_net_r_delta_r4"], 20.0)
        + 0.20 * norm(deltas["oos_net_r_delta_r4"], 4.0)
        + 0.08 * positive_folds
        - 0.18 * negative_folds
        - 0.50 * max(0.0, -float(deltas["min_fold_net_r"]) / 8.0)
    )
    if strict_pass:
        score += 2.0
    elif repair_pass:
        score += 0.75
    if candidate.stage.startswith("ablation") or "revert" in candidate.stage:
        score += 0.10

    return AlphaEvaluation(
        candidate=candidate,
        run=run,
        score=score,
        strict_pass=strict_pass,
        repair_pass=repair_pass,
        reasons=reasons,
        deltas=deltas,
    )


def run_full_replay(mutations: dict[str, Any], data_end: str) -> dict[str, Any]:
    from backtests.momentum.auto.config_mutator import mutate_nqdtc_config
    from backtests.momentum.auto.nqdtc.worker import load_worker_data
    from backtests.momentum.config_nqdtc import NQDTCBacktestConfig
    from backtests.momentum.data.replay_cache import replay_engine_kwargs
    from backtests.momentum.engine.nqdtc_engine import NQDTCEngine
    from backtests.swing.auto.incumbent_repair import finalize_config

    base_config = NQDTCBacktestConfig(
        initial_equity=10_000,
        data_dir=DATA_DIR,
        fixed_qty=10,
        track_signals=True,
        track_shadows=True,
        scoring_mode=False,
        max_dd_abort=0.0,
    )
    config = finalize_config(mutate_nqdtc_config(base_config, mutations), data_end)
    bundle = load_worker_data("NQ", DATA_DIR)
    engine = NQDTCEngine("MNQ", config)
    result = engine.run(**replay_engine_kwargs(bundle))
    return {
        "trades": list(result.trades),
        "signals": list(result.signal_events),
        "engine_counters": {
            "breakouts_evaluated": result.breakouts_evaluated,
            "breakouts_qualified": result.breakouts_qualified,
            "entries_placed": result.entries_placed,
            "entries_filled": result.entries_filled,
            "gates_blocked": result.gates_blocked,
        },
        "shadow_summary": result.shadow_summary,
    }


def diagnose_replay(full: dict[str, Any], data_end: str) -> dict[str, Any]:
    end = datetime.combine(date.fromisoformat(data_end) + timedelta(days=1), datetime.min.time())
    trades = [t for t in full["trades"] if BACKTEST_START <= _get_entry_time(t) < end]
    is_trades = [t for t in trades if _get_entry_time(t) < OOS_CUTOFF]
    oos_trades = [t for t in trades if _get_entry_time(t) >= OOS_CUTOFF]
    signals = [
        s for s in full["signals"]
        if getattr(s, "timestamp", None) is not None and BACKTEST_START <= normalize_dt(s.timestamp) < end
    ]
    is_signals = [s for s in signals if normalize_dt(s.timestamp) < OOS_CUTOFF]
    oos_signals = [s for s in signals if normalize_dt(s.timestamp) >= OOS_CUTOFF]
    return {
        "periods": {
            "is": metrics_for_trades(is_trades, _window_months(BACKTEST_START_DATE, OOS_CUTOFF_DATE)),
            "oos": metrics_for_trades(oos_trades, _compute_oos_months(data_end)),
            "full": summarize_rs([float(_get_r_multiple(t)) for t in trades]),
        },
        "engine_counters": full["engine_counters"],
        "loss_concentration": {
            "is": loss_concentration(is_trades),
            "oos": loss_concentration(oos_trades),
            "full": loss_concentration(trades),
        },
        "groups": {
            "is": group_summaries(is_trades),
            "oos": group_summaries(oos_trades),
            "full": group_summaries(trades),
        },
        "signals": {
            "is": signal_summary(is_signals),
            "oos": signal_summary(oos_signals),
            "full": signal_summary(signals),
        },
        "oos_trades": [trade_to_dict(t) for t in sorted(oos_trades, key=_get_entry_time)],
    }


def compare_diagnostics(round4_full: dict[str, Any], round5_full: dict[str, Any], data_end: str) -> dict[str, Any]:
    r4 = diagnose_replay(round4_full, data_end)
    r5 = diagnose_replay(round5_full, data_end)
    out = {
        "period_delta": {},
        "root_cause": [],
        "round4_full_regime": r4["groups"]["full"].get("regime", {}),
        "round5_full_regime": r5["groups"]["full"].get("regime", {}),
        "round5_neutral_full": r5["groups"]["full"].get("regime", {}).get("Neutral"),
        "round5_range_full": r5["groups"]["full"].get("regime", {}).get("Range"),
        "round5_neutral_is": r5["groups"]["is"].get("regime", {}).get("Neutral"),
        "round5_neutral_oos": r5["groups"]["oos"].get("regime", {}).get("Neutral"),
    }
    for period in ["is", "oos", "full"]:
        a = r4["periods"][period]
        b = r5["periods"][period]
        out["period_delta"][period] = {
            "trades_delta": trade_count(b) - trade_count(a),
            "net_r_delta": round(b["net_r"] - a["net_r"], 6),
            "avg_r_delta": round(b["avg_r"] - a["avg_r"], 6),
            "pf_delta": round(pf_value(b["profit_factor"]) - pf_value(a["profit_factor"]), 6),
        }
    neutral = out["round5_neutral_full"] or {}
    range_group = out["round5_range_full"] or {}
    if neutral and range_group and float(neutral.get("avg_r", 0.0)) < float(range_group.get("avg_r", 0.0)) * 0.35:
        out["root_cause"].append(
            "Neutral expansion is materially lower expectancy than Range and dilutes PF/net."
        )
    if r5["loss_concentration"]["full"].get("largest_loss_r", 0.0) > -1.6:
        out["root_cause"].append(
            "Losses are not concentrated in a few catastrophic tail events; the issue is ordinary low-expectancy flow."
        )
    return out


def compute_full_metrics_table(
    evaluations: list[AlphaEvaluation],
    round4: dict[str, Any],
    round5: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    from backtests.momentum.auto.nqdtc.plugin import NQDTCPlugin

    plugin = NQDTCPlugin(data_dir=DATA_DIR, initial_equity=10_000.0, max_workers=1, num_phases=5)
    rows = [
        {"candidate": "round4", "stage": "baseline", "metrics": plugin.compute_final_metrics(round4)},
        {"candidate": "round5", "stage": "baseline", "metrics": plugin.compute_final_metrics(round5)},
    ]
    seen = {json.dumps(round4, sort_keys=True), json.dumps(round5, sort_keys=True)}
    for ev in evaluations:
        sig = json.dumps(ev.candidate.mutations, sort_keys=True, default=str)
        if sig in seen:
            continue
        seen.add(sig)
        metrics = plugin.compute_final_metrics(ev.candidate.mutations)
        rows.append(
            {
                "candidate": ev.candidate.name,
                "stage": ev.candidate.stage,
                "source": ev.candidate.source,
                "score": ev.score,
                "strict_pass": ev.strict_pass,
                "repair_pass": ev.repair_pass,
                "reasons": ev.reasons,
                "split_metrics": serialize(ev.run),
                "metrics": metrics,
                "mutations": ev.candidate.mutations,
            }
        )
    return rows


def group_summaries(trades: list[Any]) -> dict[str, Any]:
    return {
        "regime": group_trade_summary(trades, lambda t: getattr(t, "composite_regime", "")),
        "entry_subtype": group_trade_summary(trades, lambda t: getattr(t, "entry_subtype", "")),
        "session_direction": group_trade_summary(trades, lambda t: f"{getattr(t, 'session', '')}_{dir_label(getattr(t, 'direction', 0))}"),
        "ny_hour": group_trade_summary(trades, lambda t: f"{ny_hour(_get_entry_time(t)):02d}"),
        "score_bin": group_trade_summary(trades, lambda t: value_bin(float(getattr(t, "score_at_entry", 0.0)), [1.5, 2.0, 2.5, 3.0, 3.5, 4.0])),
        "rvol_bin": group_trade_summary(trades, lambda t: value_bin(float(getattr(t, "rvol_at_entry", 0.0)), [1.0, 1.5, 2.0, 2.5, 3.0])),
        "box_width_bin": group_trade_summary(trades, lambda t: value_bin(float(getattr(t, "box_width", 0.0)), [50, 100, 150, 200, 250, 300, 400])),
        "disp_norm_bin": group_trade_summary(trades, lambda t: value_bin(float(getattr(t, "disp_norm_at_entry", 0.0)), [0.25, 0.50, 0.75, 1.0, 1.25, 1.5])),
        "regime_entry": group_trade_summary(trades, lambda t: f"{getattr(t, 'composite_regime', '')}_{getattr(t, 'entry_subtype', '')}"),
    }


def metrics_for_trades(trades: list[Any], months: float) -> dict[str, Any]:
    return serialize(compute_window_metrics([float(_get_r_multiple(t)) for t in trades], months))


def trade_to_dict(trade: Any) -> dict[str, Any]:
    entry = _get_entry_time(trade)
    exit_time = normalize_dt(getattr(trade, "exit_time", None))
    return {
        "entry_time": entry.isoformat(),
        "exit_time": exit_time.isoformat() if exit_time else None,
        "ny_hour": ny_hour(entry),
        "direction": dir_label(getattr(trade, "direction", 0)),
        "session": getattr(trade, "session", ""),
        "entry_subtype": getattr(trade, "entry_subtype", ""),
        "composite_regime": getattr(trade, "composite_regime", ""),
        "exit_reason": getattr(trade, "exit_reason", ""),
        "r_multiple": round(float(getattr(trade, "r_multiple", 0.0)), 6),
        "mfe_r": round(float(getattr(trade, "mfe_r", 0.0)), 6),
        "mae_r": round(float(getattr(trade, "mae_r", 0.0)), 6),
        "score_at_entry": round(float(getattr(trade, "score_at_entry", 0.0)), 6),
        "displacement_at_entry": round(float(getattr(trade, "displacement_at_entry", 0.0)), 6),
        "disp_norm_at_entry": round(float(getattr(trade, "disp_norm_at_entry", 0.0)), 6),
        "rvol_at_entry": round(float(getattr(trade, "rvol_at_entry", 0.0)), 6),
        "box_width": round(float(getattr(trade, "box_width", 0.0)), 6),
        "tp1_hit": bool(getattr(trade, "tp1_hit", False)),
        "tp2_hit": bool(getattr(trade, "tp2_hit", False)),
    }


def group_trade_summary(trades: list[Any], key_fn) -> dict[str, Any]:
    groups: dict[str, list[float]] = {}
    for trade in trades:
        groups.setdefault(str(key_fn(trade)), []).append(float(_get_r_multiple(trade)))
    return {
        key: summarize_rs(rs)
        for key, rs in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    }


def summarize_rs(rs: list[float]) -> dict[str, Any]:
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_win / gross_loss if gross_loss > 0 else ("inf" if gross_win > 0 else 0.0)
    return {
        "trades": len(rs),
        "wins": len(wins),
        "win_rate": len(wins) / len(rs) if rs else 0.0,
        "net_r": round(sum(rs), 6),
        "avg_r": round(sum(rs) / len(rs), 6) if rs else 0.0,
        "profit_factor": pf if isinstance(pf, str) else round(pf, 6),
        "min_r": round(min(rs), 6) if rs else 0.0,
        "max_r": round(max(rs), 6) if rs else 0.0,
    }


def loss_concentration(trades: list[Any]) -> dict[str, Any]:
    rs = [float(_get_r_multiple(t)) for t in trades]
    if not rs:
        return {}
    losses = sorted([r for r in rs if r < 0])
    winners = sorted([r for r in rs if r > 0], reverse=True)
    net = sum(rs)
    return {
        "trade_count": len(rs),
        "net_r": round(net, 6),
        "loss_count": len(losses),
        "winner_count": len(winners),
        "largest_loss_r": round(losses[0], 6) if losses else 0.0,
        "largest_win_r": round(winners[0], 6) if winners else 0.0,
        "net_ex_largest_loss_r": round(net - losses[0], 6) if losses else round(net, 6),
        "net_ex_two_largest_losses_r": round(net - sum(losses[:2]), 6) if len(losses) >= 2 else round(net, 6),
        "net_ex_largest_win_r": round(net - winners[0], 6) if winners else round(net, 6),
        "losses": [round(r, 6) for r in losses[:10]],
        "winners": [round(r, 6) for r in winners[:10]],
    }


def signal_summary(signals: list[Any]) -> dict[str, Any]:
    first_blocks: dict[str, int] = {}
    regimes: dict[str, int] = {}
    passed = 0
    for sig in signals:
        if getattr(sig, "passed_all", False):
            passed += 1
        reason = getattr(sig, "first_block_reason", "") or "passed_all"
        first_blocks[reason] = first_blocks.get(reason, 0) + 1
        regime = getattr(sig, "composite_regime", "")
        regimes[regime] = regimes.get(regime, 0) + 1
    return {
        "evaluated": len(signals),
        "passed_all": passed,
        "first_block_reason": dict(sorted(first_blocks.items(), key=lambda item: (-item[1], item[0]))),
        "composite_regime": dict(sorted(regimes.items(), key=lambda item: (-item[1], item[0]))),
    }


def strategy_run_from_payload(payload: dict[str, Any]) -> StrategyRun:
    def wm(item: dict[str, Any]) -> WindowMetrics:
        return WindowMetrics(**item)

    return StrategyRun(
        strategy=payload["strategy"],
        mutations=payload["mutations"],
        is_metrics=wm(payload["is_metrics"]),
        oos_metrics=wm(payload["oos_metrics"]),
        fold_metrics=[
            FoldMetrics(name=f["name"], start=f["start"], end=f["end"], metrics=wm(f["metrics"]))
            for f in payload["fold_metrics"]
        ],
        assessment=payload["assessment"],
        action=payload["action"],
        error=payload.get("error", ""),
    )


def append_progress(path: Path, completed: int, total: int, ev: AlphaEvaluation) -> None:
    payload = {
        "completed": completed,
        "total": total,
        "candidate": ev.candidate.name,
        "stage": ev.candidate.stage,
        "source": ev.candidate.source,
        "score": ev.score,
        "strict_pass": ev.strict_pass,
        "repair_pass": ev.repair_pass,
        "reasons": ev.reasons,
        "deltas": ev.deltas,
        "is_metrics": serialize(ev.run.is_metrics),
        "oos_metrics": serialize(ev.run.oos_metrics),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(serialize(payload), default=str) + "\n")


def print_progress(completed: int, total: int, ev: AlphaEvaluation) -> None:
    im = ev.run.is_metrics
    om = ev.run.oos_metrics
    print(
        f"[alpha-repair] {completed}/{total} {ev.candidate.stage}/{ev.candidate.name} "
        f"score={ev.score:+.3f} IS={im.total_trades} {im.net_r:+.1f}R PF={fmt_pf(im.profit_factor)} "
        f"OOS={om.total_trades} {om.net_r:+.1f}R PF={fmt_pf(om.profit_factor)} "
        f"strict={ev.strict_pass} repair={ev.repair_pass}",
        flush=True,
    )


def rank_key(ev: AlphaEvaluation) -> tuple[Any, ...]:
    im = ev.run.is_metrics
    om = ev.run.oos_metrics
    return (
        ev.strict_pass,
        ev.repair_pass,
        ev.score,
        om.net_r,
        pf_value(om.profit_factor),
        im.net_r,
        pf_value(im.profit_factor),
        im.total_trades,
    )


def is_pf_key(ev: AlphaEvaluation) -> tuple[Any, ...]:
    im = ev.run.is_metrics
    om = ev.run.oos_metrics
    return (pf_value(im.profit_factor), im.net_r, om.net_r, im.total_trades)


def is_net_key(ev: AlphaEvaluation) -> tuple[Any, ...]:
    im = ev.run.is_metrics
    om = ev.run.oos_metrics
    return (im.net_r, pf_value(im.profit_factor), om.net_r, im.total_trades)


def oos_net_key(ev: AlphaEvaluation) -> tuple[Any, ...]:
    im = ev.run.is_metrics
    om = ev.run.oos_metrics
    return (om.net_r, pf_value(om.profit_factor), om.total_trades, im.net_r)


def stage_leaders(evaluations: list[AlphaEvaluation], top_n: int) -> dict[str, list[Any]]:
    by_stage: dict[str, list[AlphaEvaluation]] = {}
    for ev in evaluations:
        by_stage.setdefault(ev.candidate.stage, []).append(ev)
    return {
        stage: [serialize_alpha(ev) for ev in sorted(items, key=rank_key, reverse=True)[:top_n]]
        for stage, items in sorted(by_stage.items())
    }


def serialize_alpha(ev: AlphaEvaluation) -> dict[str, Any]:
    return {
        "candidate": serialize(ev.candidate),
        "run": serialize(ev.run),
        "score": ev.score,
        "strict_pass": ev.strict_pass,
        "repair_pass": ev.repair_pass,
        "reasons": list(ev.reasons),
        "deltas": serialize(ev.deltas),
    }


def dedupe_candidates(candidates: list[RepairCandidate], current: dict[str, Any]) -> list[RepairCandidate]:
    seen = {json.dumps(current, sort_keys=True, default=str)}
    out: list[RepairCandidate] = []
    for candidate in candidates:
        sig = json.dumps(candidate.mutations, sort_keys=True, default=str)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(candidate)
    return out


def format_report(summary: dict[str, Any]) -> str:
    spec = summary["run_spec"]
    r4 = summary["round4_baseline"]
    r5 = summary["round5_baseline"]
    diag = summary["round5_vs_round4_diagnostics"]
    lines = [
        "NQDTC Round 5 Alpha Repair Search",
        "=" * 112,
        spec["selection_oos_note"],
        f"Data end: {spec['data_end']}; OOS starts {spec['oos_cutoff']}; candidates: {summary['candidate_count']}",
        "",
        "Baselines",
        fmt_run_line("Round4", r4),
        fmt_run_line("Round5", r5),
        "",
        "Round5 vs Round4 Deltas",
    ]
    for period, delta in diag["period_delta"].items():
        lines.append(
            f"  {period.upper()}: trades {delta['trades_delta']:+d}, "
            f"netR {delta['net_r_delta']:+.2f}, avgR {delta['avg_r_delta']:+.3f}, "
            f"PF {delta['pf_delta']:+.2f}"
        )
    lines.append("")
    lines.append("Root Cause Signals")
    for item in diag.get("root_cause") or ["No automatic root-cause note generated."]:
        lines.append(f"  - {item}")
    lines.append(f"  - Round5 Neutral full: {json.dumps(diag.get('round5_neutral_full'), default=str)}")
    lines.append(f"  - Round5 Range full:   {json.dumps(diag.get('round5_range_full'), default=str)}")

    lines.extend(["", "Strict Uplifts"])
    if summary["strict_uplifts"]:
        for ev in summary["strict_uplifts"][:10]:
            lines.append(fmt_eval_line(ev))
    else:
        lines.append("  None found that improved IS PF/net and OOS PF/net versus round5 simultaneously.")

    lines.extend(["", "Repair Uplifts"])
    for ev in summary["repair_uplifts"][:15]:
        lines.append(fmt_eval_line(ev))
    if not summary["repair_uplifts"]:
        lines.append("  None found under the relaxed repair criteria.")

    lines.extend(["", "Top Ranked"])
    for ev in summary["ranked"][:20]:
        lines.append(fmt_eval_line(ev))

    lines.extend(["", "Full Metrics For Top Candidates"])
    for row in summary["full_metrics"][:20]:
        metrics = row["metrics"]
        lines.append(
            f"  {row['candidate']}: trades={int(metrics.get('total_trades', 0))} "
            f"PF={float(metrics.get('profit_factor', 0.0)):.2f} "
            f"net={float(metrics.get('net_return_pct', 0.0)):+.1f}% "
            f"robust={float(metrics.get('robust_net_return_pct', 0.0)):+.1f}% "
            f"avgR={float(metrics.get('avg_r', 0.0)):+.3f} "
            f"DD={float(metrics.get('max_dd_pct', 0.0)):.1%}"
        )
    return "\n".join(lines) + "\n"


def fmt_run_line(label: str, run_payload: dict[str, Any]) -> str:
    im = run_payload["is_metrics"]
    om = run_payload["oos_metrics"]
    return (
        f"  {label}: IS n={im['total_trades']} netR={im['net_r']:+.2f} "
        f"avgR={im['avg_r']:+.3f} PF={fmt_pf(im['profit_factor'])}; "
        f"OOS n={om['total_trades']} netR={om['net_r']:+.2f} "
        f"avgR={om['avg_r']:+.3f} PF={fmt_pf(om['profit_factor'])}"
    )


def fmt_eval_line(ev_payload: dict[str, Any]) -> str:
    candidate = ev_payload["candidate"]
    im = ev_payload["run"]["is_metrics"]
    om = ev_payload["run"]["oos_metrics"]
    return (
        f"  {candidate['name']} [{candidate['stage']}]: score={ev_payload['score']:+.3f} "
        f"IS n={im['total_trades']} netR={im['net_r']:+.2f} PF={fmt_pf(im['profit_factor'])} "
        f"OOS n={om['total_trades']} netR={om['net_r']:+.2f} PF={fmt_pf(om['profit_factor'])} "
        f"reasons={ev_payload['reasons']}"
    )


def fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}".replace(".", "p")
    return str(value).replace(".", "p")


def fmt_pf(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        f = float(value)
    except Exception:
        return str(value)
    if math.isinf(f):
        return "inf"
    return f"{f:.2f}"


def pf_value(value: Any) -> float:
    if isinstance(value, str):
        if value.lower() == "inf":
            return 20.0
        try:
            return float(value)
        except ValueError:
            return 0.0
    value = float(value)
    if math.isinf(value):
        return 20.0
    if math.isnan(value):
        return 0.0
    return value


def trade_count(metrics: dict[str, Any]) -> int:
    return int(metrics.get("trades", metrics.get("total_trades", 0)) or 0)


def norm(value: float, scale: float) -> float:
    return max(-2.5, min(2.5, value / scale))


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def normalize_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value
    return datetime.fromisoformat(str(value)).replace(tzinfo=None)


def ny_hour(dt: datetime) -> int:
    from zoneinfo import ZoneInfo

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(ZoneInfo("America/New_York")).hour


def dir_label(direction: Any) -> str:
    try:
        val = int(direction)
    except Exception:
        val = int(getattr(direction, "value", 0) or 0)
    if val > 0:
        return "LONG"
    if val < 0:
        return "SHORT"
    return str(direction)


def value_bin(value: float, edges: list[float]) -> str:
    if math.isnan(value):
        return "nan"
    prev = "-inf"
    for edge in edges:
        if value < edge:
            return f"[{prev},{edge})"
        prev = str(edge)
    return f"[{prev},inf)"


if __name__ == "__main__":
    main()
