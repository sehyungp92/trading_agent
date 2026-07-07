"""Run an extra Helix win-rate repair phase from a saved mutation seed.

This is intentionally a narrow post-round runner rather than a fifth built-in
phase. The goal is to test whether trade-management changes can lift win rate
without lowering trade count or net return from the supplied baseline.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

_root = Path(__file__).resolve().parents[4]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from backtests.shared.auto.phase_state import _atomic_write_json, _utc_now_iso
from backtests.shared.auto.plugin_utils import create_process_pool, pool_map_with_heartbeat, shutdown_process_pool
from backtests.shared.auto.types import Experiment, ScoredCandidate
from backtests.swing.auto.helix.plugin import HelixPlugin
from backtests.swing.auto.helix.worker import init_worker, score_candidate


WIN_RATE_TARGET = 50.0
TRADE_COUNT_TARGET = 300.0
WINNING_TRADES_TARGET = 180.0
MIN_TRADE_RETENTION_FLOOR = 270.0
MIN_TRADE_RETENTION_RATIO = 1.0
WIN_RATE_REGRESSION_TOLERANCE = 0.05
NET_RETURN_REGRESSION_TOLERANCE = 0.25
DEFAULT_MIN_DELTA = 0.001


def _load_mutations(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict) and isinstance(payload.get("mutations"), dict):
        return dict(payload["mutations"])
    if isinstance(payload, dict) and isinstance(payload.get("cumulative_mutations"), dict):
        return dict(payload["cumulative_mutations"])
    if isinstance(payload, dict):
        return dict(payload)
    raise TypeError(f"Unexpected mutation seed payload in {path}")


def _clip01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _scale(value: float, floor: float, target: float) -> float:
    if target <= floor:
        return 0.0
    return _clip01((value - floor) / (target - floor))


def _metric(metrics: dict[str, Any], name: str, default: float = 0.0) -> float:
    value = metrics.get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _winrate_score(metrics: dict[str, Any], baseline: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Seven-component frontier-scaled score for this repair phase.

    Components:
      1. win_rate
      2. net_return
      3. frequency
      4. winning_trades
      5. profit_factor
      6. exit_quality
      7. inv_drawdown
    """

    _ = baseline

    wr = _metric(metrics, "win_rate")
    net_return = _metric(metrics, "net_return_pct")
    trades = _metric(metrics, "total_trades")
    pf = _metric(metrics, "profit_factor")
    dd = _metric(metrics, "max_r_dd")
    exit_eff = _metric(metrics, "exit_efficiency")
    waste = _metric(metrics, "waste_ratio")
    tail = _metric(metrics, "tail_pct")
    winning_trades = trades * wr / 100.0

    wr_component = _scale(wr, 35.0, 55.0)
    net_component = _scale(net_return, 50.0, 250.0)
    frequency_component = _scale(trades, 270.0, 420.0)
    winning_trades_component = _scale(winning_trades, 100.0, WINNING_TRADES_TARGET)
    pf_component = _scale(pf, 1.20, 3.50)
    exit_component = _clip01(
        0.45 * _scale(exit_eff, 0.30, 0.65)
        + 0.30 * _scale(waste, 0.55, 0.85)
        + 0.25 * _scale(tail, 0.55, 0.85)
    )
    dd_component = _scale(14.0 - dd, 0.0, 10.0)

    components = {
        "win_rate": wr_component,
        "net_return": net_component,
        "frequency": frequency_component,
        "winning_trades": winning_trades_component,
        "profit_factor": pf_component,
        "exit_quality": exit_component,
        "inv_drawdown": dd_component,
    }
    total = (
        0.27 * wr_component
        + 0.18 * net_component
        + 0.13 * frequency_component
        + 0.15 * winning_trades_component
        + 0.12 * pf_component
        + 0.08 * exit_component
        + 0.07 * dd_component
    )
    return total, components


def _hard_reject_reason(metrics: dict[str, Any], baseline: dict[str, Any]) -> str:
    baseline_trades = _metric(baseline, "total_trades")
    baseline_return = _metric(baseline, "net_return_pct")
    baseline_pf = _metric(baseline, "profit_factor")
    baseline_tail = _metric(baseline, "tail_pct")
    baseline_dd = _metric(baseline, "max_r_dd")
    baseline_side = _metric(baseline, "min_side_pf")
    baseline_wr = _metric(baseline, "win_rate")
    baseline_winning_trades = baseline_trades * baseline_wr / 100.0

    trades = _metric(metrics, "total_trades")
    net_return = _metric(metrics, "net_return_pct")
    pf = _metric(metrics, "profit_factor")
    tail = _metric(metrics, "tail_pct")
    dd = _metric(metrics, "max_r_dd")
    side = _metric(metrics, "min_side_pf")
    wr = _metric(metrics, "win_rate")
    winning_trades = trades * wr / 100.0

    min_trades = max(MIN_TRADE_RETENTION_FLOOR, baseline_trades * MIN_TRADE_RETENTION_RATIO)
    if wr + WIN_RATE_REGRESSION_TOLERANCE < baseline_wr:
        return f"win_rate_regression {wr:.2f}% < {baseline_wr:.2f}%"
    if trades < min_trades:
        return f"trade_count_regression {trades:.0f} < {min_trades:.0f}"
    if winning_trades + 0.5 < baseline_winning_trades:
        return f"winning_trade_count_regression {winning_trades:.1f} < {baseline_winning_trades:.1f}"
    if net_return + NET_RETURN_REGRESSION_TOLERANCE < baseline_return:
        return f"net_return_regression {net_return:.2f}% < {baseline_return:.2f}%"
    if pf < max(1.20, baseline_pf * 0.95):
        return f"pf_regression {pf:.2f} < {max(1.20, baseline_pf * 0.95):.2f}"
    if tail < baseline_tail * 0.85:
        return f"tail_regression {tail:.3f} < {baseline_tail * 0.85:.3f}"
    if dd > max(baseline_dd * 1.15, baseline_dd + 1.0):
        return f"drawdown_regression {dd:.2f}R > {max(baseline_dd * 1.15, baseline_dd + 1.0):.2f}R"
    if side < baseline_side * 0.90:
        return f"side_quality_regression {side:.2f} < {baseline_side * 0.90:.2f}"
    return ""


def _candidate(name: str, **mutations: Any) -> Experiment:
    return Experiment(name=name, mutations={f"param_overrides.{key}": value for key, value in mutations.items()})


def build_candidates() -> list[Experiment]:
    """Real-alpha candidate families from final diagnostics.

    The focus is conversion plus coverage: structural Class D discrimination,
    Class B/A expansion probes, earlier positive stop floors, stale repair, and
    limited signal-restoration probes.
    """

    candidates: list[Experiment] = []

    # Pre-entry Class D discriminator probes. The smoke pass kept only the
    # mild structural/freshness gates that preserved tail and net return.
    for name, mutations in [
        ("d_sep_4", {"CLASS_D_MIN_PIVOT_SEP_BARS": 4}),
        ("d_sep_6", {"CLASS_D_MIN_PIVOT_SEP_BARS": 6}),
        ("d_p2_age_20", {"CLASS_D_MAX_PIVOT2_AGE_BARS": 20}),
        ("d_p2_age_24", {"CLASS_D_MAX_PIVOT2_AGE_BARS": 24}),
        ("d_p2_age_30", {"CLASS_D_MAX_PIVOT2_AGE_BARS": 30}),
        ("d_daily_ext_300", {"CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.00}),
        ("d_daily_ext_350", {"CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.50}),
        ("d_sep4_age20", {"CLASS_D_MIN_PIVOT_SEP_BARS": 4, "CLASS_D_MAX_PIVOT2_AGE_BARS": 20}),
        ("d_sep4_dailyext300", {"CLASS_D_MIN_PIVOT_SEP_BARS": 4, "CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.00}),
        ("d_short0_dailyext300", {"CLASS_D_SHORT_MIN_ADX": 0.0, "CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.00}),
        ("d_short0_sep4_dailyext300", {
            "CLASS_D_SHORT_MIN_ADX": 0.0,
            "CLASS_D_MIN_PIVOT_SEP_BARS": 4,
            "CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.00,
        }),
    ]:
        candidates.append(_candidate(name, **mutations))

    # Earlier BE/profit-floor variants. Negative BE_ATR1H_OFFSET moves the
    # stop slightly beyond entry after the R trigger, which can turn some
    # right-then-stopped losers into small winners without reducing entries.
    for r_be, offset in [
        (0.50, 0.00), (0.55, 0.00), (0.65, 0.00),
        (0.55, -0.02), (0.65, -0.02), (0.75, -0.02),
        (0.55, -0.05), (0.65, -0.05), (0.75, -0.05),
        (0.60, -0.08),
    ]:
        candidates.append(_candidate(f"be1h_{r_be:.2f}_offset_{offset:+.2f}", R_BE_1H=r_be, BE_ATR1H_OFFSET=offset))

    # RTS profit floor variants aimed directly at the 73 right-then-stopped
    # losers in the final diagnostics.
    for mfe, giveback, floor, min_bars, fade_bars, max_mfe in [
        (0.25, 0.05, 0.05, 4, 1, 1.75),
        (0.30, 0.10, 0.05, 4, 1, 1.75),
        (0.45, 0.20, 0.05, 3, 1, 1.75),
        (0.50, 0.25, 0.05, 3, 1, 1.75),
        (0.50, 0.25, 0.10, 3, 1, 1.75),
        (0.60, 0.30, 0.10, 4, 1, 1.75),
        (0.60, 0.30, 0.15, 4, 1, 1.75),
        (0.45, 0.20, 0.05, 2, 0, 1.50),
        (0.50, 0.20, 0.10, 3, 0, 1.50),
        (0.55, 0.25, 0.10, 3, 0, 1.25),
        (0.65, 0.30, 0.15, 4, 0, 1.50),
    ]:
        candidates.append(_candidate(
            f"rts_mfe{mfe:.2f}_gb{giveback:.2f}_floor{floor:.2f}_b{min_bars}_f{fade_bars}_max{max_mfe:.2f}",
            RTS_GUARD_MFE_R=mfe,
            RTS_GUARD_MIN_GIVEBACK_R=giveback,
            RTS_GUARD_FLOOR_R=floor,
            RTS_GUARD_MIN_BARS=min_bars,
            RTS_GUARD_FADE_BARS=fade_bars,
            RTS_GUARD_MAX_MFE_R=max_mfe,
        ))

    # Earlier partials can improve win classification if the partial is large
    # enough, but the hard rejects guard against destroying tail/net return.
    for r_partial, frac in [
        (1.00, 0.40), (1.00, 0.60),
        (1.20, 0.50), (1.20, 0.70),
        (1.50, 0.50), (1.50, 0.70),
        (1.65, 0.70), (1.80, 0.80),
    ]:
        candidates.append(_candidate(f"partial_{r_partial:.2f}_frac{frac:.2f}", R_PARTIAL_2P5=r_partial, PARTIAL_2P5_FRAC=frac))

    # Stale repair probes: some variants cut dead trades earlier, others let
    # mild stale trades breathe. These are structural and small in count.
    for name, mutations in [
        ("early_stale_16", {"EARLY_STALE_BARS": 16}),
        ("early_stale_20", {"EARLY_STALE_BARS": 20}),
        ("stale_24_thresh010", {"STALE_1H_BARS": 24, "STALE_R_THRESH": 0.10}),
        ("stale_36_thresh000", {"STALE_1H_BARS": 36, "STALE_R_THRESH": 0.00}),
        ("stale_48_thresh000", {"STALE_1H_BARS": 48, "STALE_R_THRESH": 0.00}),
        ("stale_36_thresh_neg010", {"STALE_1H_BARS": 36, "STALE_R_THRESH": -0.10}),
    ]:
        candidates.append(_candidate(name, **mutations))

    # Momentum bail probes reduce loss size; they only get accepted if the
    # win-rate/net-return/frequency score improves under hard guardrails.
    for bars, thresh in [(8, -0.25), (10, -0.25), (12, -0.10), (16, -0.10)]:
        candidates.append(_candidate(f"d_bail_{bars}_th{thresh:+.2f}", CLASS_D_BAIL_BARS=bars, CLASS_D_BAIL_R_THRESH=thresh))

    # Frequency restoration probes. These deliberately test whether Phase 1's
    # D-gates gave up too much frequency. They are not allowed through unless
    # they also improve net return and preserve quality.
    for name, mutations in [
        ("restore_d_streak1", {"CLASS_D_REGIME_STREAK_MIN": 1}),
        ("restore_d_streak0", {"CLASS_D_REGIME_STREAK_MIN": 0}),
        ("restore_d_short_adx20", {"CLASS_D_SHORT_MIN_ADX": 20.0}),
        ("restore_d_short_adx16", {"CLASS_D_SHORT_MIN_ADX": 16.0}),
        ("restore_d_short_adx0", {"CLASS_D_SHORT_MIN_ADX": 0.0}),
        ("restore_d_short20_streak1", {"CLASS_D_SHORT_MIN_ADX": 20.0, "CLASS_D_REGIME_STREAK_MIN": 1}),
        ("restore_d_short16_streak1", {"CLASS_D_SHORT_MIN_ADX": 16.0, "CLASS_D_REGIME_STREAK_MIN": 1}),
        ("restore_d_short0_streak1", {"CLASS_D_SHORT_MIN_ADX": 0.0, "CLASS_D_REGIME_STREAK_MIN": 1}),
        ("class_b_min_adx18", {"CLASS_B_MIN_ADX": 18.0}),
        ("enable_class_a", {}),
        ("enable_class_a_small", {
            "CLASS_A_SIZE_TREND": 0.35,
            "CLASS_A_SIZE_CHOP": 0.25,
            "CLASS_A_SIZE_COUNTER": 0.20,
            "DIV_MAG_FLOOR": 0.08,
        }),
        ("enable_class_c", {}),
        ("class_b_min_adx24", {"CLASS_B_MIN_ADX": 24.0}),
        ("class_b_min_adx20", {"CLASS_B_MIN_ADX": 20.0}),
        ("adx_upper_70", {"ADX_UPPER_GATE": 70.0}),
        ("adx_upper_off", {"ADX_UPPER_GATE": 999.0}),
        ("restore_d_short16_sep4", {"CLASS_D_SHORT_MIN_ADX": 16.0, "CLASS_D_MIN_PIVOT_SEP_BARS": 4}),
        ("restore_d_short0_sep4", {"CLASS_D_SHORT_MIN_ADX": 0.0, "CLASS_D_MIN_PIVOT_SEP_BARS": 4}),
        ("enable_class_c_sep4", {"CLASS_D_MIN_PIVOT_SEP_BARS": 4}),
        ("freq_d_short0_sep4_daily_rts025", {
            "CLASS_D_SHORT_MIN_ADX": 0.0,
            "CLASS_D_MIN_PIVOT_SEP_BARS": 4,
            "CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.00,
            "RTS_GUARD_MFE_R": 0.25,
            "RTS_GUARD_MIN_GIVEBACK_R": 0.05,
            "RTS_GUARD_FLOOR_R": 0.05,
        }),
        ("freq_d_short0_sep4_daily_rts030", {
            "CLASS_D_SHORT_MIN_ADX": 0.0,
            "CLASS_D_MIN_PIVOT_SEP_BARS": 4,
            "CLASS_D_MAX_DAILY_EXTENSION_ATR": 3.00,
            "RTS_GUARD_MFE_R": 0.30,
            "RTS_GUARD_MIN_GIVEBACK_R": 0.10,
            "RTS_GUARD_FLOOR_R": 0.05,
        }),
        ("freq_d_short0_sep4_classc_rts035", {
            "CLASS_D_SHORT_MIN_ADX": 0.0,
            "CLASS_D_MIN_PIVOT_SEP_BARS": 4,
            "RTS_GUARD_MFE_R": 0.35,
            "RTS_GUARD_MIN_GIVEBACK_R": 0.15,
            "RTS_GUARD_FLOOR_R": 0.10,
        }),
    ]:
        if name == "enable_class_a":
            candidates.append(Experiment(name=name, mutations={"flags.disable_class_a": False}))
        elif name == "enable_class_a_small":
            muts = {"flags.disable_class_a": False}
            muts.update({f"param_overrides.{key}": value for key, value in mutations.items()})
            candidates.append(Experiment(name=name, mutations=muts))
        elif name == "enable_class_c":
            candidates.append(Experiment(name=name, mutations={"flags.disable_class_c": False}))
        elif name in {"enable_class_c_sep4", "freq_d_short0_sep4_classc_rts035"}:
            muts = {"flags.disable_class_c": False}
            muts.update({f"param_overrides.{key}": value for key, value in mutations.items()})
            candidates.append(Experiment(name=name, mutations=muts))
        else:
            candidates.append(_candidate(name, **mutations))

    # A few paired conversion variants, deliberately low-dimensional.
    for name, mutations in [
        ("be055_rts_floor005", {
            "R_BE_1H": 0.55,
            "BE_ATR1H_OFFSET": 0.0,
            "RTS_GUARD_MFE_R": 0.50,
            "RTS_GUARD_MIN_GIVEBACK_R": 0.25,
            "RTS_GUARD_FLOOR_R": 0.05,
        }),
        ("be055_profit_rts_floor010", {
            "R_BE_1H": 0.55,
            "BE_ATR1H_OFFSET": -0.02,
            "RTS_GUARD_MFE_R": 0.50,
            "RTS_GUARD_MIN_GIVEBACK_R": 0.25,
            "RTS_GUARD_FLOOR_R": 0.10,
        }),
        ("partial120_be065", {
            "R_PARTIAL_2P5": 1.20,
            "PARTIAL_2P5_FRAC": 0.50,
            "R_BE_1H": 0.65,
            "BE_ATR1H_OFFSET": 0.0,
        }),
        ("partial150_be055_rts", {
            "R_PARTIAL_2P5": 1.50,
            "PARTIAL_2P5_FRAC": 0.50,
            "R_BE_1H": 0.55,
            "BE_ATR1H_OFFSET": 0.0,
            "RTS_GUARD_FLOOR_R": 0.05,
        }),
    ]:
        candidates.append(_candidate(name, **mutations))

    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique: list[Experiment] = []
    for candidate in candidates:
        if candidate.name in seen:
            continue
        seen.add(candidate.name)
        unique.append(candidate)
    return unique


def _evaluate_candidates(
    candidates: list[Experiment],
    current_mutations: dict[str, Any],
    *,
    data_dir: Path,
    equity: float,
    start_date: str | None,
    end_date: str | None,
    max_workers: int,
) -> list[ScoredCandidate]:
    pool = create_process_pool(
        max_workers,
        initializer=init_worker,
        initargs=(str(data_dir), equity, start_date, end_date),
        description="Helix win-rate repair",
    )
    try:
        args = [
            (
                candidate.name,
                candidate.mutations,
                current_mutations,
                4,
                None,
                {"min_trades": 1, "min_pf": 0.0, "max_r_dd": 999.0, "min_tail_pct": 0.0, "min_regime_pf": 0.0, "min_side_pf": 0.0},
            )
            for candidate in candidates
        ]
        return pool_map_with_heartbeat(
            pool,
            score_candidate,
            args,
            description="Helix win-rate repair",
            heartbeat_seconds=120.0,
            per_candidate_timeout_seconds=420.0,
            minimum_timeout_seconds=600.0,
        )
    finally:
        shutdown_process_pool(pool)


def run_repair(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_mutations = _load_mutations(Path(args.seed))
    plugin = HelixPlugin(
        data_dir=Path(args.data_dir),
        initial_equity=float(args.equity),
        max_workers=1,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    baseline_metrics = plugin.compute_final_metrics(base_mutations)
    base_score, base_components = _winrate_score(baseline_metrics, baseline_metrics)

    candidates = build_candidates()
    current_mutations = dict(base_mutations)
    current_metrics = dict(baseline_metrics)
    current_score = base_score
    remaining = list(candidates)
    rounds: list[dict[str, Any]] = []
    kept_features: list[str] = []
    accepted_mutations: dict[str, Any] = {}
    started = time.monotonic()

    _atomic_write_json(
        {
            "status": "started",
            "generated_at_utc": _utc_now_iso(),
            "seed": str(Path(args.seed).resolve()),
            "baseline_metrics": baseline_metrics,
            "baseline_score": base_score,
            "baseline_components": base_components,
            "candidate_count": len(candidates),
            "score_components": [
                "win_rate",
                "net_return",
                "frequency",
                "winning_trades",
                "profit_factor",
                "exit_quality",
                "inv_drawdown",
            ],
        },
        output_dir / "winrate_repair_progress.json",
    )

    for round_num in range(1, int(args.max_rounds) + 1):
        if not remaining:
            break
        results = _evaluate_candidates(
            remaining,
            current_mutations,
            data_dir=Path(args.data_dir),
            equity=float(args.equity),
            start_date=args.start_date,
            end_date=args.end_date,
            max_workers=int(args.max_workers),
        )

        scored_rows: list[dict[str, Any]] = []
        best_row: dict[str, Any] | None = None
        for candidate, result in zip(remaining, results):
            metrics = dict(result.metrics or {})
            if not metrics:
                row = {
                    "name": candidate.name,
                    "mutations": candidate.mutations,
                    "score": 0.0,
                    "delta_pct": -100.0,
                    "eligible": False,
                    "reject_reason": result.reject_reason or "missing_metrics",
                    "metrics": {},
                    "components": {},
                }
                scored_rows.append(row)
                continue

            custom_score, components = _winrate_score(metrics, baseline_metrics)
            reject_reason = _hard_reject_reason(metrics, current_metrics)
            eligible = not reject_reason
            row = {
                "name": candidate.name,
                "mutations": candidate.mutations,
                "score": custom_score if eligible else 0.0,
                "raw_score": custom_score,
                "delta_pct": ((custom_score - current_score) / max(abs(current_score), 1e-9)) * 100.0,
                "eligible": eligible,
                "reject_reason": reject_reason,
                "metrics": metrics,
                "components": components,
            }
            scored_rows.append(row)
            if eligible and (best_row is None or row["score"] > best_row["score"]):
                best_row = row

        scored_rows.sort(key=lambda item: (bool(item["eligible"]), float(item["score"])), reverse=True)
        round_payload = {
            "round_num": round_num,
            "current_score": current_score,
            "current_metrics": current_metrics,
            "candidates_tested": len(remaining),
            "best": best_row,
            "top_candidates": scored_rows[:15],
        }

        if best_row is None:
            round_payload["kept"] = False
            round_payload["stop_reason"] = "no_eligible_candidate"
            rounds.append(round_payload)
            break

        best_delta = (float(best_row["score"]) - current_score) / max(abs(current_score), 1e-9)
        if best_delta < float(args.min_delta):
            round_payload["kept"] = False
            round_payload["stop_reason"] = f"best_delta {best_delta:.4%} < min_delta {float(args.min_delta):.4%}"
            rounds.append(round_payload)
            break

        current_mutations.update(best_row["mutations"])
        accepted_mutations.update(best_row["mutations"])
        kept_features.append(str(best_row["name"]))
        current_metrics = dict(best_row["metrics"])
        current_score = float(best_row["score"])
        remaining = [candidate for candidate in remaining if candidate.name != best_row["name"]]
        round_payload["kept"] = True
        round_payload["accepted_name"] = best_row["name"]
        round_payload["accepted_delta_pct"] = best_delta * 100.0
        rounds.append(round_payload)

        _atomic_write_json(
            {
                "status": "running",
                "updated_at_utc": _utc_now_iso(),
                "current_score": current_score,
                "current_metrics": current_metrics,
                "kept_features": kept_features,
                "accepted_mutations": accepted_mutations,
                "rounds": rounds,
                "remaining_candidates": len(remaining),
            },
            output_dir / "winrate_repair_progress.json",
        )

    elapsed = time.monotonic() - started
    final_payload = {
        "status": "completed",
        "generated_at_utc": _utc_now_iso(),
        "elapsed_seconds": elapsed,
        "seed": str(Path(args.seed).resolve()),
        "score_definition": {
            "component_count": 7,
            "weights": {
                "win_rate": 0.27,
                "net_return": 0.18,
                "frequency": 0.13,
                "winning_trades": 0.15,
                "profit_factor": 0.12,
                "exit_quality": 0.08,
                "inv_drawdown": 0.07,
            },
            "targets": {
                "win_rate_pct": 55.0,
                "trade_count": 420.0,
                "winning_trades": WINNING_TRADES_TARGET,
                "net_return_pct": 250.0,
                "profit_factor": 3.5,
                "exit_efficiency": 0.65,
                "waste_ratio": 0.85,
                "tail_pct": 0.85,
            },
            "hard_rejects": [
                "win_rate does not materially regress vs current baseline",
                "total_trades >= max(270, current baseline)",
                "winning_trades >= current baseline",
                "net_return_pct does not materially regress vs current baseline",
                "profit_factor >= max(1.20, current baseline * 0.95)",
                "tail_pct >= current baseline * 0.85",
                "max_r_dd <= max(current baseline * 1.15, current baseline + 1R)",
                "min_side_pf >= current baseline * 0.90",
            ],
        },
        "baseline_score": base_score,
        "baseline_metrics": baseline_metrics,
        "final_score": current_score,
        "final_metrics": current_metrics,
        "kept_features": kept_features,
        "accepted_mutations": accepted_mutations,
        "final_mutations": current_mutations,
        "rounds": rounds,
        "candidate_count": len(candidates),
    }
    _atomic_write_json(final_payload, output_dir / "winrate_repair_result.json")
    _atomic_write_json(current_mutations, output_dir / "optimized_config_winrate_repair.json")

    if args.write_diagnostics:
        state_path = output_dir / "phase_state_winrate_repair.json"
        phase_state = {
            "current_phase": 5,
            "completed_phases": [1, 2, 3, 4, 5],
            "cumulative_mutations": current_mutations,
            "phase_results": {
                "5": {
                    "focus": "WIN_RATE_REPAIR",
                    "base_mutations": base_mutations,
                    "final_mutations": current_mutations,
                    "base_score": base_score,
                    "final_score": current_score,
                    "kept_features": kept_features,
                    "final_metrics": current_metrics,
                    "new_mutations": accepted_mutations,
                    "rounds": rounds,
                    "score_definition": final_payload["score_definition"],
                }
            },
            "phase_gate_results": {},
            "retry_count": {},
            "scoring_retries": {},
            "diagnostic_retries": {},
            "phase_timestamps": {"5": {"started": _utc_now_iso(), "completed": _utc_now_iso()}},
            "round_name": "Round 3 Phase 5 Helix win-rate repair",
        }
        _atomic_write_json(phase_state, state_path)
        diag_path = output_dir / "round_3_phase_5_winrate_repair_diagnostics.txt"
        summary_path = output_dir / "round_3_phase_5_winrate_repair_summary.json"
        cmd = [
            sys.executable,
            str(_root / "backtests" / "swing" / "analysis" / "helix_full_diagnostics.py"),
            "--state-path",
            str(state_path),
            "--phase-result",
            "current",
            "--output",
            str(diag_path),
            "--summary-json",
            str(summary_path),
            "--title",
            "HELIX ROUND 3 PHASE 5 WIN-RATE REPAIR DIAGNOSTICS (SYNCHRONIZED / FEE-NET)",
            "--lineage-label",
            "Helix Round 3 Phase 5",
            "--equity",
            str(float(args.equity)),
        ]
        if args.start_date:
            cmd.extend(["--start-date", str(args.start_date)])
        if args.end_date:
            cmd.extend(["--end-date", str(args.end_date)])
        completed = subprocess.run(cmd, cwd=_root, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
        final_payload["diagnostics_stdout"] = completed.stdout
        final_payload["diagnostics_path"] = str(diag_path.resolve())
        final_payload["diagnostics_summary_path"] = str(summary_path.resolve())
        _atomic_write_json(final_payload, output_dir / "winrate_repair_result.json")

    return final_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="backtests/swing/data/raw")
    parser.add_argument("--equity", type=float, default=25_000.0)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default="2026-03-20")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--write-diagnostics", action="store_true")
    args = parser.parse_args()

    result = run_repair(args)
    print(json.dumps({
        "status": result["status"],
        "baseline_win_rate": result["baseline_metrics"].get("win_rate"),
        "final_win_rate": result["final_metrics"].get("win_rate"),
        "baseline_trades": result["baseline_metrics"].get("total_trades"),
        "final_trades": result["final_metrics"].get("total_trades"),
        "baseline_net_return": result["baseline_metrics"].get("net_return_pct"),
        "final_net_return": result["final_metrics"].get("net_return_pct"),
        "kept_features": result["kept_features"],
        "diagnostics_path": result.get("diagnostics_path"),
    }, indent=2))


if __name__ == "__main__":
    main()
