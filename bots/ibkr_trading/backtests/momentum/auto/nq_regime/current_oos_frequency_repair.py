"""Current NQ_REGIME OOS frequency ablation and targeted repair study.

The OOS window is used for diagnosis/selection in this runner. Treat the
outputs as a repair study, not as an untouched validation holdout.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backtests.momentum.auto.nq_regime.phase_candidates import get_phase_candidates  # noqa: E402
from backtests.momentum.auto.nq_regime.worker import mutate_config  # noqa: E402
from backtests.momentum.config_regime import NqRegimeBacktestConfig  # noqa: E402
from backtests.momentum.engine.regime_engine import load_nq_regime_data, run_nq_regime_backtest  # noqa: E402
from backtests.shared.validation.oos_validation import (  # noqa: E402
    BACKTEST_START,
    BACKTEST_START_DATE,
    OOS_CUTOFF,
    OOS_CUTOFF_DATE,
    WindowMetrics,
    _compute_oos_months,
    _window_months,
    compute_window_metrics,
)
from backtests.swing.auto.incumbent_repair import (  # noqa: E402
    MISSING,
    build_phase_state_features,
    short_key,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CURRENT_CONFIG = PROJECT_ROOT / "backtests/output/momentum/nq_regime/round_5/optimized_config.json"
ROUND_ROOT = PROJECT_ROOT / "backtests/output/momentum/nq_regime"
DATA_DIR = PROJECT_ROOT / "backtests/momentum/data/raw"

_WORKER_DATA = None
_WORKER_BASE_CONFIG: NqRegimeBacktestConfig | None = None
_WORKER_DATA_END = "2026-05-01"
_WORKER_MODE = "oos"


@dataclass(frozen=True)
class RepairCandidate:
    name: str
    stage: str
    mutations: dict[str, Any]
    intent: str
    source: str = ""


@dataclass
class RunSummary:
    mode: str
    mutations: dict[str, Any]
    is_metrics: WindowMetrics = field(default_factory=WindowMetrics)
    oos_metrics: WindowMetrics = field(default_factory=WindowMetrics)
    oos_modules: dict[str, dict[str, Any]] = field(default_factory=dict)
    oos_setups: dict[str, dict[str, Any]] = field(default_factory=dict)
    oos_trades: list[dict[str, Any]] = field(default_factory=list)
    oos_routing: dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass
class CandidateEvaluation:
    candidate: RepairCandidate
    run: RunSummary
    objective_delta: float
    passed: bool
    reasons: list[str] = field(default_factory=list)
    deltas: dict[str, float] = field(default_factory=dict)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "candidate_progress.jsonl"
    progress_path.write_text("", encoding="utf-8")

    current = read_json(Path(args.config_path))
    run_spec = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": "nq_regime",
        "config_path": str(Path(args.config_path).resolve()),
        "data_end": args.data_end,
        "stages": args.stages,
        "max_workers": args.max_workers,
        "full_replay_top_n": args.full_replay_top_n,
        "selection_oos_note": (
            "The OOS window is used for diagnosis/selection in this run; "
            "it is no longer an untouched holdout."
        ),
    }
    write_json(output_dir / "run_spec.json", run_spec)

    started = time.time()
    print("[nq-regime-oos] evaluating OOS baseline", flush=True)
    baseline_oos = evaluate_direct(current, args.data_end, mode="oos")
    baseline_full = full_baseline_from_report(current, baseline_oos)
    candidates = build_candidate_suite(current, args.stages)
    print(
        f"[nq-regime-oos] baseline OOS trades={baseline_oos.oos_metrics.total_trades} "
        f"netR={baseline_oos.oos_metrics.net_r:+.2f}; screening {len(candidates)} candidates",
        flush=True,
    )

    screened = evaluate_candidates(
        candidates=candidates,
        baseline=baseline_oos,
        data_end=args.data_end,
        mode="oos",
        max_workers=args.max_workers,
        progress_path=progress_path,
    )
    screened_sorted = sorted(screened, key=screening_rank, reverse=True)
    full_candidates = select_full_replay_candidates(screened_sorted, args.full_replay_top_n)
    print(f"[nq-regime-oos] full replay shortlist={len(full_candidates)}", flush=True)
    full = evaluate_candidates(
        candidates=full_candidates,
        baseline=baseline_full,
        data_end=args.data_end,
        mode="full",
        max_workers=args.max_workers,
        progress_path=progress_path,
    )
    full_sorted = sorted(full, key=full_rank, reverse=True)

    balanced = [item for item in full_sorted if is_balanced_uplift(item, baseline_full)]
    frequency_leaders = sorted(full, key=lambda item: (item.run.oos_metrics.total_trades, item.run.oos_metrics.net_r), reverse=True)
    oos_net_leaders = sorted(full, key=lambda item: (item.run.oos_metrics.net_r, item.run.oos_metrics.total_trades), reverse=True)
    stage_leaders = stage_leader_map(full_sorted, args.top_n)
    recommendation = balanced[0] if balanced else None
    if recommendation is not None:
        write_json(output_dir / "recommended_config.json", recommendation.candidate.mutations)

    summary = {
        "run_spec": run_spec,
        "elapsed_seconds": round(time.time() - started, 2),
        "baseline_oos": serialize(baseline_oos),
        "baseline_full": serialize(baseline_full),
        "candidate_count": len(candidates),
        "screened_count": len(screened),
        "full_replay_count": len(full),
        "screening_top": [serialize(item) for item in screened_sorted[: args.top_n]],
        "full_top": [serialize(item) for item in full_sorted[: args.top_n]],
        "balanced_frequency_uplifts": [serialize(item) for item in balanced[: args.top_n]],
        "frequency_leaders": [serialize(item) for item in frequency_leaders[: args.top_n]],
        "oos_net_leaders": [serialize(item) for item in oos_net_leaders[: args.top_n]],
        "stage_leaders": stage_leaders,
        "recommended": serialize(recommendation) if recommendation is not None else None,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.txt").write_text(format_report(summary), encoding="utf-8")
    print(f"[nq-regime-oos] complete in {(time.time() - started) / 60.0:.1f} min", flush=True)
    print(f"Output: {output_dir.resolve()}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", default=str(CURRENT_CONFIG))
    parser.add_argument("--data-end", default="2026-05-01")
    parser.add_argument("--output-dir", default="backtests/output/momentum/nq_regime/current_oos_frequency_repair_20260504")
    parser.add_argument("--stages", default="ablation,perturbation,targeted")
    parser.add_argument("--max-workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--full-replay-top-n", type=int, default=28)
    parser.add_argument("--top-n", type=int, default=24)
    return parser


def build_candidate_suite(current: dict[str, Any], stages_text: str) -> list[RepairCandidate]:
    stages = {item.strip() for item in stages_text.split(",") if item.strip()}
    candidates: list[RepairCandidate] = []
    if "ablation" in stages:
        candidates.extend(build_ablation_candidates(current))
    if "perturbation" in stages:
        candidates.extend(build_perturbation_candidates(current))
    if "targeted" in stages:
        candidates.extend(build_targeted_candidates(current))
    return dedupe(candidates, current)


def build_ablation_candidates(current: dict[str, Any]) -> list[RepairCandidate]:
    candidates: list[RepairCandidate] = []
    previous_by_key: dict[str, Any] = {}
    for name, mutations, previous_values in historical_features():
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
                    intent="Revert one accepted mutation key to its prior value.",
                    source=name,
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
        previous = previous_by_key.get(key, MISSING)
        if previous != MISSING and previous != current.get(key, MISSING):
            candidates.append(
                RepairCandidate(
                    name=f"ablate_key_prior_inventory_{short_key(key)}",
                    stage="ablation",
                    mutations=reverted_config(current, [key], {key: previous}),
                    intent="Revert one active incumbent key to its nearest historical prior.",
                    source="active_key_inventory",
                )
            )
    return candidates


def historical_features() -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    features: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for round_dir in sorted((item for item in ROUND_ROOT.glob("round_*") if item.is_dir()), key=lambda p: p.name):
        features.extend(build_phase_state_features(round_dir / "phase_state.json", round_dir.name))
    return features


def build_perturbation_candidates(current: dict[str, Any]) -> list[RepairCandidate]:
    candidates: list[RepairCandidate] = []
    for key in sorted(current):
        value = current[key]
        if isinstance(value, bool):
            add_patch(candidates, current, "perturbation", f"flip_{short_key(key)}", {key: not value}, key)
        elif isinstance(value, int) and not isinstance(value, bool):
            for variant in int_variants(key, value):
                add_patch(candidates, current, "perturbation", f"perturb_{short_key(key)}_{fmt_value(variant)}", {key: variant}, key)
        elif isinstance(value, float):
            for variant in float_variants(key, value):
                add_patch(candidates, current, "perturbation", f"perturb_{short_key(key)}_{fmt_value(variant)}", {key: variant}, key)
        elif isinstance(value, str):
            for variant in string_variants(key, value):
                add_patch(candidates, current, "perturbation", f"perturb_{short_key(key)}_{variant}", {key: variant}, key)
    return candidates


def build_targeted_candidates(current: dict[str, Any]) -> list[RepairCandidate]:
    candidates: list[RepairCandidate] = []
    for phase in range(1, 8):
        for name, mutations in get_phase_candidates(phase, current):
            add_patch(candidates, current, "targeted", f"native_{name}", mutations, name)

    targeted_patches: list[tuple[str, dict[str, Any], str]] = []
    for score in (8, 9, 10):
        targeted_patches.append((f"struct_short_score_{score}", {"param_overrides.STRUCTURAL_SHORT_MIN_SCORE": score}, "structural_zero_oos"))
    for body in (0.45, 0.50, 0.55):
        targeted_patches.append((f"struct_body_{fmt_value(body)}", {"param_overrides.STRUCTURAL_MIN_BODY_PCT": body}, "structural_zero_oos"))
    for loc in (0.55, 0.60, 0.65):
        targeted_patches.append((f"struct_close_loc_{fmt_value(loc)}", {"param_overrides.STRUCTURAL_MIN_CLOSE_LOCATION": loc}, "structural_zero_oos"))
    for max_stop in (45.0, 55.0, 60.0):
        targeted_patches.append(
            (
                f"struct_stop_{fmt_value(max_stop)}",
                {
                    "param_overrides.STRUCTURAL_MAX_STOP_PTS": max_stop,
                    "param_overrides.STRUCTURAL_HYBRID_CLOSE_MAX_STOP_PTS": max_stop,
                },
                "structural_zero_oos",
            )
        )
    targeted_patches.extend(
        [
            (
                "struct_unlock_soft_candle",
                {
                    "param_overrides.STRUCTURAL_MIN_BODY_PCT": 0.45,
                    "param_overrides.STRUCTURAL_MIN_CLOSE_LOCATION": 0.55,
                    "param_overrides.STRUCTURAL_SHORT_MIN_SCORE": 9,
                },
                "structural_zero_oos",
            ),
            (
                "struct_unlock_soft_candle_stop55",
                {
                    "param_overrides.STRUCTURAL_MIN_BODY_PCT": 0.45,
                    "param_overrides.STRUCTURAL_MIN_CLOSE_LOCATION": 0.55,
                    "param_overrides.STRUCTURAL_SHORT_MIN_SCORE": 9,
                    "param_overrides.STRUCTURAL_MAX_STOP_PTS": 55.0,
                    "param_overrides.STRUCTURAL_HYBRID_CLOSE_MAX_STOP_PTS": 55.0,
                },
                "structural_zero_oos",
            ),
            (
                "struct_adaptive_retest_soft",
                {
                    "param_overrides.STRUCTURAL_ENTRY_MODE": "adaptive_retest",
                    "param_overrides.STRUCTURAL_MIN_BODY_PCT": 0.50,
                    "param_overrides.STRUCTURAL_MIN_CLOSE_LOCATION": 0.60,
                    "param_overrides.STRUCTURAL_MAX_STOP_PTS": 45.0,
                },
                "structural_zero_oos",
            ),
            (
                "struct_pullback_room05",
                {"param_overrides.STRUCTURAL_PULLBACK_RECLAIM_MIN_ROOM_R": 0.50},
                "structural_zero_oos",
            ),
            (
                "struct_pullback_entry_pullback",
                {"param_overrides.STRUCTURAL_PULLBACK_RECLAIM_ENTRY_MODE": "pullback"},
                "structural_zero_oos",
            ),
            (
                "struct_continuation_guarded",
                {
                    "param_overrides.STRUCTURAL_CONTINUATION_ENABLED": True,
                    "param_overrides.STRUCTURAL_CONTINUATION_MIN_SCORE": 9,
                    "param_overrides.STRUCTURAL_CONTINUATION_MIN_ROOM_R": 1.0,
                    "param_overrides.STRUCTURAL_CONTINUATION_MIN_VOLUME_MULTIPLE": 0.8,
                },
                "structural_zero_oos",
            ),
        ]
    )

    for volume in (0.8, 0.9, 1.0, 1.1):
        targeted_patches.append((f"sw_volume_{fmt_value(volume)}", {"param_overrides.SECOND_WIND_MIN_VOLUME_MULTIPLE": volume}, "second_wind_weak_volume"))
    for stop in (35.0, 40.0, 45.0, 60.0):
        targeted_patches.append(
            (
                f"sw_stop_{fmt_value(stop)}",
                {"param_overrides.SECOND_WIND_STOP_CAP": stop, "param_overrides.SECOND_WIND_MAX_STOP_PTS": stop},
                "second_wind_stop_cap",
            )
        )
    targeted_patches.extend(
        [
            (
                "sw_room05_volume10",
                {
                    "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_ROOM_R": 0.5,
                    "param_overrides.SECOND_WIND_MIN_VOLUME_MULTIPLE": 1.0,
                },
                "second_wind_frequency",
            ),
            (
                "sw_stop40_volume10",
                {
                    "param_overrides.SECOND_WIND_STOP_CAP": 40.0,
                    "param_overrides.SECOND_WIND_MAX_STOP_PTS": 40.0,
                    "param_overrides.SECOND_WIND_MIN_VOLUME_MULTIPLE": 1.0,
                },
                "second_wind_frequency",
            ),
            (
                "sw_vwap_ema_off",
                {"param_overrides.SECOND_WIND_VWAP_RECLAIM_REQUIRE_EMA_ALIGNMENT": False},
                "second_wind_vwap_reclaim",
            ),
            (
                "sw_second_leg_pragmatic",
                {
                    "param_overrides.SECOND_WIND_SECOND_LEG_ENABLED": True,
                    "param_overrides.SECOND_WIND_SECOND_LEG_MIN_SCORE": 8,
                    "param_overrides.SECOND_WIND_SECOND_LEG_MIN_PM_SCORE": 0.55,
                    "param_overrides.SECOND_WIND_SECOND_LEG_MIN_VOLUME_MULTIPLE": 1.0,
                    "param_overrides.SECOND_WIND_SECOND_LEG_MIN_CLOSE_LOCATION": 0.60,
                    "param_overrides.SECOND_WIND_SECOND_LEG_REQUIRE_IMPULSE": False,
                },
                "second_wind_second_leg",
            ),
        ]
    )

    for max_pen in (14.0, 15.0, 18.0):
        targeted_patches.append((f"rev_max_pen_{fmt_value(max_pen)}", {"param_overrides.REVERSION_MAX_PENETRATION_PTS": max_pen}, "reversion_penetration"))
    for stop in (12.0, 15.0):
        targeted_patches.append(
            (
                f"rev_stop_{fmt_value(stop)}",
                {
                    "param_overrides.REVERSION_STANDARD_STOP_CAP": stop,
                    "param_overrides.REVERSION_A_PLUS_STOP_CAP": max(stop, 15.0),
                },
                "reversion_stop_cap",
            )
        )
    targeted_patches.extend(
        [
            (
                "rev_pen15_stop15",
                {
                    "param_overrides.REVERSION_MAX_PENETRATION_PTS": 15.0,
                    "param_overrides.REVERSION_STANDARD_STOP_CAP": 12.0,
                    "param_overrides.REVERSION_A_PLUS_STOP_CAP": 15.0,
                },
                "reversion_penetration_stop",
            ),
            (
                "rev_adaptive_market_oos",
                {
                    "param_overrides.REVERSION_ENTRY_MODEL": "adaptive_reclaim_retest",
                    "param_overrides.REVERSION_ADAPTIVE_MARKET_MIN_SCORE": 10,
                    "param_overrides.REVERSION_ADAPTIVE_MARKET_MAX_PENETRATION_PTS": 8.0,
                    "param_overrides.REVERSION_ADAPTIVE_MARKET_MIN_ROOM_R": 1.0,
                },
                "reversion_entry_model",
            ),
            (
                "capacity_5_guarded",
                {
                    "param_overrides.MAX_TRADES_PER_DAY": 5,
                    "param_overrides.MAX_FULL_RISK_TRADES": 3,
                    "param_overrides.MAX_DAILY_REALIZED_R_LOSS": -2.5,
                },
                "capacity",
            ),
            (
                "capacity_6_guarded",
                {
                    "param_overrides.MAX_TRADES_PER_DAY": 6,
                    "param_overrides.MAX_FULL_RISK_TRADES": 4,
                    "param_overrides.MAX_DAILY_REALIZED_R_LOSS": -3.0,
                },
                "capacity",
            ),
        ]
    )
    for name, patch, source in targeted_patches:
        add_patch(candidates, current, "targeted", name, patch, source)
    return candidates


def int_variants(key: str, value: int) -> list[int]:
    explicit = {
        "param_overrides.MAX_TRADES_PER_DAY": [3, 4, 5, 6],
        "param_overrides.MAX_FULL_RISK_TRADES": [2, 3, 4],
        "param_overrides.ENTRY_TTL_RETEST_MINUTES": [60, 90, 120, 150, 180],
        "param_overrides.ENTRY_TTL_MOMENTUM_MINUTES": [10, 15, 20, 30],
        "param_overrides.ROUTE_CANDIDATE_LED_MIN_SCORE": [7, 8, 9, 10],
        "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_SCORE": [7, 8, 9, 10],
        "param_overrides.REVERSION_SWING_LOOKBACK_BARS": [24, 36, 48, 60, 72],
        "param_overrides.REVERSION_SWING_MAX_LEVELS_PER_SIDE": [3, 4, 6, 8],
    }
    if key in explicit:
        return [item for item in explicit[key] if item != value]
    if any(token in key for token in ("_SCORE", "_A_SCORE", "_A_PLUS_SCORE")):
        return [item for item in range(max(0, value - 2), value + 3) if item != value]
    return sorted({max(0, value + delta) for delta in (-2, -1, 1, 2)})


def float_variants(key: str, value: float) -> list[float]:
    explicit = {
        "param_overrides.REGIME_MIN_CONFIDENCE": [0.55, 0.60, 0.65, 0.70],
        "param_overrides.REGIME_MIN_MARGIN": [0.05, 0.10, 0.15, 0.20],
        "param_overrides.ROUTE_CANDIDATE_LED_MIN_ROOM_R": [0.0, 0.25, 0.50, 0.75, 1.0],
        "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_ROOM_R": [0.5, 1.0, 1.5, 2.0, 3.0],
        "param_overrides.SECOND_WIND_CANDIDATE_LED_MIN_PM_SCORE": [0.45, 0.50, 0.55, 0.60],
        "param_overrides.SECOND_WIND_MIN_PM_SCORE": [0.45, 0.50, 0.55, 0.60],
        "param_overrides.SECOND_WIND_MIN_VOLUME_MULTIPLE": [0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
        "param_overrides.SECOND_WIND_STOP_CAP": [25.0, 30.0, 35.0, 40.0, 45.0],
        "param_overrides.SECOND_WIND_MAX_STOP_PTS": [25.0, 30.0, 35.0, 40.0, 45.0],
        "param_overrides.REVERSION_STANDARD_STOP_CAP": [8.0, 10.0, 12.0, 15.0],
        "param_overrides.REVERSION_A_PLUS_STOP_CAP": [10.0, 12.0, 15.0, 18.0],
        "param_overrides.REVERSION_MAX_PENETRATION_PTS": [10.0, 12.0, 14.0, 15.0, 18.0],
        "param_overrides.STRUCTURAL_MIN_BODY_PCT": [0.45, 0.50, 0.55, 0.60],
        "param_overrides.STRUCTURAL_MIN_CLOSE_LOCATION": [0.55, 0.60, 0.65, 0.70],
        "param_overrides.STRUCTURAL_MAX_STOP_PTS": [35.0, 45.0, 55.0, 60.0],
        "param_overrides.STRUCTURAL_HYBRID_CLOSE_MAX_STOP_PTS": [35.0, 45.0, 55.0, 60.0],
        "param_overrides.STRUCTURAL_PULLBACK_RECLAIM_MIN_ROOM_R": [0.5, 0.75, 1.0, 1.25],
        "param_overrides.PROFIT_FLOOR_TRIGGER_R": [0.25, 0.50, 0.75, 1.0],
        "param_overrides.PROFIT_FLOOR_LOCK_R": [0.10, 0.25, 0.35, 0.50],
        "param_overrides.MFE_RATCHET_FLOOR_PCT": [0.50, 0.60, 0.65, 0.70],
    }
    if key in explicit:
        return [item for item in explicit[key] if not math.isclose(float(item), float(value))]
    return sorted({round(value * factor, 6) for factor in (0.80, 0.90, 1.10, 1.20) if not math.isclose(value * factor, value)})


def string_variants(key: str, value: str) -> list[str]:
    variants = {
        "param_overrides.STRUCTURAL_ENTRY_MODE": ["adaptive_retest", "hybrid_close_adaptive", "breakout_close", "structure_shift"],
        "param_overrides.STRUCTURAL_PULLBACK_RECLAIM_ENTRY_MODE": ["close", "pullback", "breakout_stop"],
        "param_overrides.STRUCTURAL_STOP_MODEL": ["recent_5m", "ib_level"],
        "param_overrides.REVERSION_ENTRY_MODEL": ["swept_level_retest", "reclaim_close", "structure_shift", "adaptive_reclaim_retest"],
        "param_overrides.SECOND_WIND_ENTRY_MODEL": ["trigger_midpoint", "trigger_close", "breakout_stop", "ema_pullback"],
    }
    return [item for item in variants.get(key, []) if item != value]


def reverted_config(current: dict[str, Any], keys: list[str], previous_values: dict[str, Any]) -> dict[str, Any]:
    reverted = dict(current)
    for key in keys:
        previous = previous_values.get(key, MISSING)
        if previous == MISSING:
            reverted.pop(key, None)
        else:
            reverted[key] = previous
    return reverted


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
                intent="Current NQ_REGIME OOS frequency diagnostic candidate.",
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


def init_worker(mode: str, data_end: str) -> None:
    global _WORKER_DATA, _WORKER_BASE_CONFIG, _WORKER_DATA_END, _WORKER_MODE
    _WORKER_MODE = mode
    _WORKER_DATA_END = data_end
    start = BACKTEST_START if mode == "full" else OOS_CUTOFF
    _WORKER_BASE_CONFIG = NqRegimeBacktestConfig(
        start_date=start.replace(tzinfo=timezone.utc),
        end_date=datetime.combine(date.fromisoformat(data_end), datetime.min.time(), tzinfo=timezone.utc),
        initial_equity=10_000.0,
        data_dir=DATA_DIR,
        analysis_symbol="NQ",
        trade_symbol="MNQ",
        fixed_qty=10,
    )
    _WORKER_DATA = load_nq_regime_data(_WORKER_BASE_CONFIG)


def evaluate_candidates(
    *,
    candidates: list[RepairCandidate],
    baseline: RunSummary,
    data_end: str,
    mode: str,
    max_workers: int,
    progress_path: Path,
) -> list[CandidateEvaluation]:
    if max_workers <= 1:
        init_worker(mode, data_end)
        out = []
        for idx, candidate in enumerate(candidates, start=1):
            item = evaluate_one(candidate, baseline, data_end, mode)
            out.append(item)
            append_progress(progress_path, idx, len(candidates), item)
            print_progress(mode, idx, len(candidates), item)
        return out
    out: list[CandidateEvaluation] = []
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(mode, data_end)) as pool:
        futures = {pool.submit(evaluate_one, candidate, baseline, data_end, mode): candidate for candidate in candidates}
        for idx, future in enumerate(as_completed(futures), start=1):
            item = future.result()
            out.append(item)
            append_progress(progress_path, idx, len(candidates), item)
            print_progress(mode, idx, len(candidates), item)
    return out


def evaluate_one(candidate: RepairCandidate, baseline: RunSummary, data_end: str, mode: str) -> CandidateEvaluation:
    try:
        run = evaluate_worker(candidate.mutations, data_end, mode)
        return score_candidate(candidate, baseline, run)
    except Exception as exc:
        run = RunSummary(mode=mode, mutations=dict(candidate.mutations), error=str(exc))
        return CandidateEvaluation(candidate=candidate, run=run, objective_delta=-999.0, passed=False, reasons=[str(exc)])


def evaluate_direct(mutations: dict[str, Any], data_end: str, *, mode: str) -> RunSummary:
    init_worker(mode, data_end)
    return evaluate_worker(mutations, data_end, mode)


def evaluate_worker(mutations: dict[str, Any], data_end: str, mode: str) -> RunSummary:
    del data_end
    if _WORKER_BASE_CONFIG is None:
        init_worker(mode, _WORKER_DATA_END)
    assert _WORKER_BASE_CONFIG is not None
    config = mutate_config(_WORKER_BASE_CONFIG, mutations)
    result = run_nq_regime_backtest(_WORKER_DATA, config)
    return summarize_run(result.trades, result.signal_events, mutations, mode)


def full_baseline_from_report(current: dict[str, Any], baseline_oos: RunSummary) -> RunSummary:
    is_metrics = WindowMetrics(
        total_trades=473,
        winning_trades=342,
        win_rate=0.7230443974630021,
        profit_factor=6.794979656384601,
        net_r=386.5956654486337,
        avg_r=0.8173269882634963,
        max_drawdown_r=3.15107854553699,
        trades_per_month=17.775456790123457,
        months=_window_months(BACKTEST_START_DATE, OOS_CUTOFF_DATE),
    )
    return RunSummary(mode="full", mutations=dict(current), is_metrics=is_metrics, oos_metrics=baseline_oos.oos_metrics)


def summarize_run(trades: list[Any], events: list[Any], mutations: dict[str, Any], mode: str) -> RunSummary:
    if mode == "full":
        is_trades = []
        oos_trades = []
        for trade in trades:
            entry = utc_naive(trade.entry_time)
            if entry < BACKTEST_START:
                continue
            if entry < OOS_CUTOFF:
                is_trades.append(trade)
            else:
                oos_trades.append(trade)
        is_metrics = compute_window_metrics([float(trade.r_multiple) for trade in is_trades], _window_months(BACKTEST_START_DATE, OOS_CUTOFF_DATE))
        oos_metrics = compute_window_metrics([float(trade.r_multiple) for trade in oos_trades], _compute_oos_months(_WORKER_DATA_END))
    else:
        oos_trades = [trade for trade in trades if utc_naive(trade.entry_time) >= OOS_CUTOFF]
        is_metrics = WindowMetrics(months=_window_months(BACKTEST_START_DATE, OOS_CUTOFF_DATE))
        oos_metrics = compute_window_metrics([float(trade.r_multiple) for trade in oos_trades], _compute_oos_months(_WORKER_DATA_END))
    return RunSummary(
        mode=mode,
        mutations=dict(mutations),
        is_metrics=is_metrics,
        oos_metrics=oos_metrics,
        oos_modules=cohort_metrics(oos_trades, key_fn=lambda trade: str(trade.module or "")),
        oos_setups=cohort_metrics(oos_trades, key_fn=lambda trade: f"{trade.module}.{trade.setup_type}"),
        oos_trades=trade_rows(oos_trades),
        oos_routing=routing_summary(events),
    )


def cohort_metrics(trades: list[Any], *, key_fn) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Any]] = defaultdict(list)
    for trade in trades:
        groups[str(key_fn(trade) or "unknown")].append(trade)
    out: dict[str, dict[str, Any]] = {}
    for key, group in sorted(groups.items()):
        rs = [float(trade.r_multiple) for trade in group]
        wins = [r for r in rs if r > 0]
        losses = [r for r in rs if r < 0]
        gross_loss = abs(sum(losses))
        out[key] = {
            "trades": len(group),
            "net_r": sum(rs),
            "avg_r": sum(rs) / len(rs) if rs else 0.0,
            "win_rate": len(wins) / len(rs) if rs else 0.0,
            "profit_factor": sum(wins) / gross_loss if gross_loss else (10.0 if wins else 0.0),
        }
    return out


def trade_rows(trades: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for trade in trades:
        rows.append(
            {
                "entry_time": trade.entry_time.isoformat(),
                "module": trade.module,
                "setup_type": trade.setup_type,
                "side": trade.side,
                "grade": trade.grade,
                "score": trade.setup_score,
                "r_multiple": trade.r_multiple,
                "mfe_r": trade.mfe_r,
                "mae_r": trade.mae_r,
                "exit_reason": trade.exit_reason,
                "entry_model": trade.entry_model,
                "target_room_r": trade.target_room_r,
                "stop_distance_points": trade.stop_distance_points,
            }
        )
    return rows


def routing_summary(events: list[Any]) -> dict[str, Any]:
    routing = [event for event in events if event.code == "ROUTING_DECISION" and utc_naive(event.ts) >= OOS_CUTOFF]
    rows = {
        module: {
            "candidates": 0,
            "valid": 0,
            "selected": 0,
            "blocked": 0,
            "valid_blocked": 0,
            "vetoes": Counter(),
            "blocks": Counter(),
        }
        for module in ("structural_expansion", "liquidity_reversion", "second_wind")
    }
    reasons = Counter()
    for event in routing:
        details = event.details
        reasons[str(details.get("reason", ""))] += 1
        selected_module = str(details.get("selected_module", ""))
        if details.get("selected") and selected_module in rows:
            rows[selected_module]["selected"] += 1
        for item in details.get("candidate_inventory") or []:
            module = str(item.get("module", ""))
            if module not in rows:
                continue
            rows[module]["candidates"] += 1
            if item.get("valid"):
                rows[module]["valid"] += 1
            rows[module]["vetoes"].update(str(veto) for veto in item.get("vetoes") or [])
        for item in details.get("blocked_candidates") or []:
            module = str(item.get("module", ""))
            if module not in rows:
                continue
            rows[module]["blocked"] += 1
            if item.get("valid"):
                rows[module]["valid_blocked"] += 1
            rows[module]["blocks"][str(item.get("block_reason", ""))] += 1
    return {
        "routing_decisions": len(routing),
        "reasons": reasons.most_common(12),
        "entry_blocked_by_session": sum(1 for event in events if event.code == "ENTRY_BLOCKED_BY_SESSION" and utc_naive(event.ts) >= OOS_CUTOFF),
        "daily_lockout": sum(1 for event in events if event.code == "DAILY_LOCKOUT" and utc_naive(event.ts) >= OOS_CUTOFF),
        "entry_blocked_by_size": sum(1 for event in events if event.code == "ENTRY_BLOCKED_BY_SIZE" and utc_naive(event.ts) >= OOS_CUTOFF),
        "modules": {
            module: {
                key: value.most_common(8) if isinstance(value, Counter) else value
                for key, value in row.items()
            }
            for module, row in rows.items()
        },
    }


def score_candidate(candidate: RepairCandidate, baseline: RunSummary, run: RunSummary) -> CandidateEvaluation:
    deltas = {
        "oos_trade_delta": metric_delta(run.oos_metrics.total_trades, baseline.oos_metrics.total_trades, min_scale=3.0),
        "oos_net_r_delta": metric_delta(run.oos_metrics.net_r, baseline.oos_metrics.net_r, min_scale=1.0),
        "oos_avg_r_delta": run.oos_metrics.avg_r - baseline.oos_metrics.avg_r,
        "is_trade_delta": metric_delta(run.is_metrics.total_trades, baseline.is_metrics.total_trades, min_scale=10.0),
        "is_net_r_delta": metric_delta(run.is_metrics.net_r, baseline.is_metrics.net_r, min_scale=5.0),
        "is_avg_r_delta": run.is_metrics.avg_r - baseline.is_metrics.avg_r,
    }
    objective = (
        0.38 * deltas["oos_net_r_delta"]
        + 0.34 * deltas["oos_trade_delta"]
        + 0.10 * deltas["oos_avg_r_delta"]
        + 0.10 * deltas["is_net_r_delta"]
        + 0.08 * deltas["is_trade_delta"]
    )
    reasons = acceptance_reasons(baseline, run)
    return CandidateEvaluation(
        candidate=candidate,
        run=run,
        objective_delta=objective,
        passed=not reasons,
        reasons=reasons,
        deltas=deltas,
    )


def acceptance_reasons(baseline: RunSummary, run: RunSummary) -> list[str]:
    if run.error:
        return [run.error]
    reasons: list[str] = []
    if run.oos_metrics.total_trades < baseline.oos_metrics.total_trades:
        reasons.append("oos_trade_regression")
    if run.oos_metrics.net_r < baseline.oos_metrics.net_r - max(1.0, abs(baseline.oos_metrics.net_r) * 0.10):
        reasons.append("oos_net_r_regression")
    if baseline.mode == "full" and run.mode == "full":
        if run.is_metrics.total_trades < int(baseline.is_metrics.total_trades * 0.90):
            reasons.append("is_trade_floor")
        if run.is_metrics.net_r < baseline.is_metrics.net_r * 0.95:
            reasons.append("is_net_r_floor")
        if run.is_metrics.avg_r < baseline.is_metrics.avg_r - 0.15:
            reasons.append("is_avg_r_floor")
        if run.is_metrics.max_drawdown_r > max(baseline.is_metrics.max_drawdown_r * 1.5, baseline.is_metrics.max_drawdown_r + 2.0):
            reasons.append("is_dd_expansion")
    return reasons


def select_full_replay_candidates(screened: list[CandidateEvaluation], limit: int) -> list[RepairCandidate]:
    selected: list[RepairCandidate] = []
    seen: set[str] = set()

    def add(item: CandidateEvaluation) -> None:
        sig = signature(item.candidate.mutations)
        if sig in seen:
            return
        seen.add(sig)
        selected.append(item.candidate)

    buckets = [
        screened[: max(8, limit // 3)],
        sorted(screened, key=lambda item: (item.run.oos_metrics.total_trades, item.run.oos_metrics.net_r), reverse=True)[: max(8, limit // 3)],
        sorted(screened, key=lambda item: (item.run.oos_metrics.net_r, item.run.oos_metrics.total_trades), reverse=True)[: max(8, limit // 3)],
        [item for item in screened if item.passed][: max(8, limit // 3)],
    ]
    for bucket in buckets:
        for item in bucket:
            add(item)
            if len(selected) >= limit:
                return selected
    return selected


def is_balanced_uplift(item: CandidateEvaluation, baseline: RunSummary) -> bool:
    run = item.run
    return (
        item.passed
        and run.oos_metrics.total_trades > baseline.oos_metrics.total_trades
        and run.oos_metrics.net_r >= baseline.oos_metrics.net_r
        and run.is_metrics.total_trades >= int(baseline.is_metrics.total_trades * 0.90)
        and run.is_metrics.net_r >= baseline.is_metrics.net_r * 0.95
        and run.is_metrics.avg_r >= baseline.is_metrics.avg_r - 0.15
    )


def screening_rank(item: CandidateEvaluation) -> tuple[float, int, float]:
    return (item.objective_delta, item.run.oos_metrics.total_trades, item.run.oos_metrics.net_r)


def full_rank(item: CandidateEvaluation) -> tuple[bool, float, int, float, float]:
    return (item.passed, item.objective_delta, item.run.oos_metrics.total_trades, item.run.oos_metrics.net_r, item.run.is_metrics.net_r)


def stage_leader_map(items: list[CandidateEvaluation], top_n: int) -> dict[str, list[Any]]:
    grouped: dict[str, list[CandidateEvaluation]] = defaultdict(list)
    for item in items:
        grouped[item.candidate.stage].append(item)
    return {stage: [serialize(item) for item in group[:top_n]] for stage, group in grouped.items()}


def append_progress(path: Path, completed: int, total: int, item: CandidateEvaluation) -> None:
    payload = {
        "completed": completed,
        "total": total,
        "mode": item.run.mode,
        "candidate": item.candidate.name,
        "stage": item.candidate.stage,
        "source": item.candidate.source,
        "objective_delta": item.objective_delta,
        "passed": item.passed,
        "reasons": item.reasons,
        "is_metrics": serialize(item.run.is_metrics),
        "oos_metrics": serialize(item.run.oos_metrics),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(serialize(payload), default=str) + "\n")


def print_progress(mode: str, completed: int, total: int, item: CandidateEvaluation) -> None:
    oos = item.run.oos_metrics
    is_m = item.run.is_metrics
    print(
        f"[nq-regime-oos:{mode}] {completed}/{total} {item.candidate.stage}/{item.candidate.name} "
        f"obj={item.objective_delta:+.3%} OOS={oos.total_trades} {oos.net_r:+.2f}R "
        f"IS={is_m.total_trades} {is_m.net_r:+.1f}R passed={item.passed}",
        flush=True,
    )


def format_report(summary: dict[str, Any]) -> str:
    base = summary["baseline_full"]
    base_oos = base["oos_metrics"]
    base_is = base["is_metrics"]
    lines = [
        "NQ_REGIME Current OOS Frequency Repair Summary",
        "=" * 100,
        summary["run_spec"]["selection_oos_note"],
        f"Config: {summary['run_spec']['config_path']}",
        f"Data end: {summary['run_spec']['data_end']}",
        "",
        (
            f"Baseline IS: trades={base_is['total_trades']} PF={fmt(base_is['profit_factor'])} "
            f"netR={base_is['net_r']:.2f} avgR={base_is['avg_r']:.3f}"
        ),
        (
            f"Baseline OOS: trades={base_oos['total_trades']} PF={fmt(base_oos['profit_factor'])} "
            f"netR={base_oos['net_r']:.2f} avgR={base_oos['avg_r']:.3f}"
        ),
        f"Candidates screened: {summary['screened_count']}; full replays: {summary['full_replay_count']}",
        "",
        "Baseline OOS module mix:",
    ]
    for module, metrics in (summary["baseline_oos"].get("oos_modules") or {}).items():
        lines.append(
            f"  {module}: trades={metrics['trades']} netR={metrics['net_r']:.2f} "
            f"avgR={metrics['avg_r']:.3f}"
        )
    routing = summary["baseline_oos"].get("oos_routing") or {}
    if routing:
        lines.append("Baseline OOS routing bottlenecks:")
        modules = routing.get("modules", {})
        for module, row in modules.items():
            top_veto = row.get("vetoes", [["none", 0]])[0]
            top_block = row.get("blocks", [["none", 0]])[0]
            lines.append(
                f"  {module}: cand={row.get('candidates', 0)} valid={row.get('valid', 0)} "
                f"selected={row.get('selected', 0)} top_veto={top_veto} top_block={top_block}"
            )
    for key, title in [
        ("balanced_frequency_uplifts", "Balanced Full-Replay Frequency Uplifts"),
        ("full_top", "Top Full-Replay Objective Candidates"),
        ("frequency_leaders", "Top Full-Replay Frequency Candidates"),
        ("oos_net_leaders", "Top Full-Replay OOS-Net Candidates"),
        ("screening_top", "Top OOS-Screen Candidates"),
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
                f"  {cand['stage']}/{cand['name']}: obj={item['objective_delta']:+.2%}, "
                f"passed={item['passed']}, OOS trades={oos['total_trades']} netR={oos['net_r']:.2f} "
                f"avgR={oos['avg_r']:.3f}, IS trades={is_m['total_trades']} "
                f"netR={is_m['net_r']:.1f}, reasons={item.get('reasons', [])}"
            )
    recommendation = summary.get("recommended")
    lines.append("")
    if recommendation:
        cand = recommendation["candidate"]
        lines.append(f"Recommended repair candidate: {cand['stage']}/{cand['name']}")
        lines.append("Recommended config written to recommended_config.json")
    else:
        lines.append("Recommended repair candidate: None passed the balanced IS/OOS uplift gate.")
    return "\n".join(lines) + "\n"


def utc_naive(value: datetime) -> datetime:
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def metric_delta(candidate: float, baseline: float, *, min_scale: float) -> float:
    scale = max(abs(float(baseline)), min_scale)
    return (float(candidate) - float(baseline)) / scale


def signature(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def fmt(value: Any) -> str:
    if isinstance(value, str):
        return value
    value = float(value)
    if math.isinf(value):
        return "inf"
    return f"{value:.2f}"


def fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}".replace(".", "p")
    return str(value).replace(".", "p")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serialize(payload), indent=2, default=str), encoding="utf-8")


def serialize(value: Any) -> Any:
    if is_dataclass(value):
        return {key: serialize(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(item) for item in value]
    if isinstance(value, (Path, date, datetime)):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    return value


if __name__ == "__main__":
    main()
