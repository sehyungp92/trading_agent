from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
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

from backtests.momentum.auto.nqdtc.plugin import NQDTCPlugin, _metrics_from_dict, score_phase_metrics
from backtests.shared.auto.phase_gates import evaluate_gate
from backtests.shared.auto.phase_state import PhaseState, save_phase_state
from backtests.shared.auto.provenance import build_phase_auto_provenance
from backtests.shared.auto.round_manager import RoundManager, canonicalize_metrics
from backtests.shared.auto.types import GreedyRound
from strategies.momentum.nqdtc import config as LIVE_C


ROUND_NUM = 5
BASE_CONFIG = ROOT / "backtests/output/momentum/nqdtc/round_4/optimized_config.json"
DEEP_DIVE_SUMMARY = ROOT / "backtests/output/momentum/nqdtc/round_4/oos_deep_dive_20260524/summary.json"
DEEP_DIVE_TEXT = ROOT / "backtests/output/momentum/nqdtc/round_4/oos_deep_dive_20260524/summary.txt"
ABLATION_TEXT = ROOT / "backtests/output/momentum/nqdtc/round_4/oos_ablation_perturbation_repair_20260524/summary.txt"
ALPHA_REPAIR_SUMMARY = ROOT / "backtests/output/momentum/nqdtc/round_5/alpha_repair_20260524/summary.json"
ALPHA_REPAIR_TEXT = ROOT / "backtests/output/momentum/nqdtc/round_5/alpha_repair_20260524/summary.txt"
ALPHA_REFINE_SUMMARY = ROOT / "backtests/output/momentum/nqdtc/round_5/alpha_refine_20260524/summary.json"
ALPHA_REFINE_TEXT = ROOT / "backtests/output/momentum/nqdtc/round_5/alpha_refine_20260524/summary.txt"

PROMOTED_CANDIDATE = "min_box_100_c_0p248"
PROMOTED_DELTAS: dict[str, Any] = {
    "param_overrides.BLOCK_NEUTRAL_REGIME": False,
    "param_overrides.SCORE_NON_RANGE_MULT": 2.25,
    "param_overrides.C_ENTRY_OFFSET_ATR_STANDARD": 0.248,
    "param_overrides.TP1_R": 1.6,
    "param_overrides.MIN_BOX_WIDTH": 100,
}
PROMOTED_OOS_EVIDENCE = {
    "candidate": PROMOTED_CANDIDATE,
    "is_trades": 152,
    "is_net_r": 80.59,
    "is_avg_r": 0.530,
    "is_profit_factor": 2.25,
    "oos_trades": 7,
    "oos_net_r": 5.57,
    "oos_avg_r": 0.796,
    "oos_profit_factor": 2.85,
    "note": "Evidence from focused round 5 alpha repair refinement through 2026-05-24; OOS was used for selection diagnostics and is not a fresh holdout.",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _build_round5_mutations() -> dict[str, Any]:
    mutations = dict(_read_json(BASE_CONFIG))
    mutations.update(PROMOTED_DELTAS)
    return mutations


def _metrics_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "total_trades",
        "win_rate",
        "profit_factor",
        "net_return_pct",
        "robust_net_return_pct",
        "max_dd_pct",
        "calmar",
        "sharpe",
        "sortino",
        "avg_r",
        "capture_ratio",
        "range_regime_pct",
        "tp1_hit_rate",
        "tp2_hit_rate",
        "avg_hold_hours",
        "largest_win_pnl_share",
        "largest_winner_r",
    )
    return {key: metrics.get(key) for key in keys}


def _parity_snapshot(mutations: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "BLOCK_NEUTRAL_REGIME": mutations["param_overrides.BLOCK_NEUTRAL_REGIME"],
        "BLOCK_CAUTION_REGIME": mutations["param_overrides.BLOCK_CAUTION_REGIME"],
        "SCORE_NON_RANGE_MULT": mutations["param_overrides.SCORE_NON_RANGE_MULT"],
        "DISPLACEMENT_THRESHOLD_ENABLED": mutations["flags.displacement_threshold"],
        "BLOCK_ETH_SHORTS": mutations["flags.block_eth_shorts"],
        "A_ENTRY_ENABLED": mutations["param_overrides.A_ENTRY_ENABLED"],
        "A_ENTRY_RETEST_ENABLED": mutations["flags.entry_a_retest"],
        "A_ENTRY_LATCH_ENABLED": mutations["flags.entry_a_latch"],
        "A_TTL_5M_BARS": mutations["param_overrides.A_TTL_5M_BARS"],
        "A_MAX_BOX_WIDTH": mutations["param_overrides.A_MAX_BOX_WIDTH"],
        "C_ENTRY_OFFSET_ATR_STANDARD": mutations["param_overrides.C_ENTRY_OFFSET_ATR_STANDARD"],
        "TP1_R": mutations["param_overrides.TP1_R"],
        "TP1_PARTIAL_PCT": mutations["param_overrides.TP1_PARTIAL_PCT"],
        "TP2_R": mutations["param_overrides.TP2_R"],
        "TP2_PARTIAL_PCT": mutations["param_overrides.TP2_PARTIAL_PCT"],
        "TP1_ONLY_CAP_MODE": mutations["param_overrides.TP1_ONLY_CAP_MODE"],
        "MIN_INTER_TRADE_GAP_MINUTES": mutations["param_overrides.MIN_INTER_TRADE_GAP_MINUTES"],
        "MIN_BOX_WIDTH": mutations["param_overrides.MIN_BOX_WIDTH"],
        "LOSS_STREAK_THRESHOLD": mutations["param_overrides.LOSS_STREAK_THRESHOLD"],
    }
    actual = {
        "BLOCK_NEUTRAL_REGIME": LIVE_C.BLOCK_NEUTRAL_REGIME,
        "BLOCK_CAUTION_REGIME": LIVE_C.BLOCK_CAUTION_REGIME,
        "SCORE_NON_RANGE_MULT": LIVE_C.SCORE_NON_RANGE_MULT,
        "DISPLACEMENT_THRESHOLD_ENABLED": LIVE_C.DISPLACEMENT_THRESHOLD_ENABLED,
        "BLOCK_ETH_SHORTS": LIVE_C.BLOCK_ETH_SHORTS,
        "A_ENTRY_ENABLED": LIVE_C.A_ENTRY_ENABLED,
        "A_ENTRY_RETEST_ENABLED": LIVE_C.A_ENTRY_RETEST_ENABLED,
        "A_ENTRY_LATCH_ENABLED": LIVE_C.A_ENTRY_LATCH_ENABLED,
        "A_TTL_5M_BARS": LIVE_C.A_TTL_5M_BARS,
        "A_MAX_BOX_WIDTH": LIVE_C.A_MAX_BOX_WIDTH,
        "C_ENTRY_OFFSET_ATR_STANDARD": LIVE_C.C_ENTRY_OFFSET_ATR_STANDARD,
        "TP1_R": LIVE_C.TP1_R,
        "TP1_PARTIAL_PCT": LIVE_C.EXIT_TIERS["Neutral"][0][1],
        "TP2_R": LIVE_C.EXIT_TIERS["Neutral"][1][0],
        "TP2_PARTIAL_PCT": LIVE_C.EXIT_TIERS["Neutral"][1][1],
        "TP1_ONLY_CAP_MODE": LIVE_C.TP1_ONLY_CAP_MODE,
        "MIN_INTER_TRADE_GAP_MINUTES": LIVE_C.MIN_INTER_TRADE_GAP_MINUTES,
        "MIN_BOX_WIDTH": LIVE_C.MIN_BOX_WIDTH,
        "LOSS_STREAK_THRESHOLD": LIVE_C.LOSS_STREAK_THRESHOLD,
    }
    checks = {
        key: {
            "expected_from_optimized_config": expected[key],
            "live_constant": actual[key],
            "matches": actual[key] == expected[key],
        }
        for key in expected
    }
    return {
        "generated_at_utc": _utc_now(),
        "round": ROUND_NUM,
        "candidate": PROMOTED_CANDIDATE,
        "checks": checks,
        "all_checks_passed": all(item["matches"] for item in checks.values()),
        "notes": [
            "Backtest-only flags remain represented in optimized_config.json.",
            "Live engine now honors A retest/latch selection, displacement disablement, composite regime blocks, inter-trade gap, and loss-streak threshold.",
        ],
    }


def _evaluation_text(
    *,
    mutations: dict[str, Any],
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any] | None,
    parity: dict[str, Any],
) -> str:
    lines = [
        "NQDTC ROUND 5 PROMOTION EVALUATION",
        "=" * 70,
        f"Generated UTC: {_utc_now()}",
        f"Promoted candidate: {PROMOTED_CANDIDATE}",
        f"Baseline config: {BASE_CONFIG}",
        f"Targeted OOS evidence: IS {PROMOTED_OOS_EVIDENCE['is_trades']} trades, "
        f"netR {PROMOTED_OOS_EVIDENCE['is_net_r']:+.2f}; OOS {PROMOTED_OOS_EVIDENCE['oos_trades']} trades, "
        f"netR {PROMOTED_OOS_EVIDENCE['oos_net_r']:+.2f}.",
        "",
        "Promoted Deltas",
    ]
    for key, value in PROMOTED_DELTAS.items():
        lines.append(f"  {key}: {value}")

    lines.extend(
        [
            "",
            "Full Replay Metrics",
            f"  Trades:        {metrics['total_trades']}",
            f"  Win rate:      {metrics['win_rate']:.1%}",
            f"  PF:            {metrics['profit_factor']:.2f}",
            f"  Net return:    {metrics['net_return_pct']:+.1f}%",
            f"  Robust return: {metrics['robust_net_return_pct']:+.1f}%",
            f"  Max DD:        {metrics['max_dd_pct']:.2%}",
            f"  Avg R:         {metrics['avg_r']:+.3f}",
            f"  Capture:       {metrics['capture_ratio']:.3f}",
        ]
    )
    if baseline_metrics:
        lines.extend(
            [
                "",
                "Round 4 Full Replay Comparison",
                f"  Trades:     {baseline_metrics['total_trades']} -> {metrics['total_trades']} "
                f"({metrics['total_trades'] - baseline_metrics['total_trades']:+})",
                f"  Net return: {baseline_metrics['net_return_pct']:+.1f}% -> {metrics['net_return_pct']:+.1f}% "
                f"({metrics['net_return_pct'] - baseline_metrics['net_return_pct']:+.1f}pp)",
                f"  PF:         {baseline_metrics['profit_factor']:.2f} -> {metrics['profit_factor']:.2f}",
                f"  Avg R:      {baseline_metrics['avg_r']:+.3f} -> {metrics['avg_r']:+.3f}",
            ]
        )

    lines.extend(["", "Live/Backtest Parity"])
    lines.append(f"  Checks passed: {parity['all_checks_passed']}")
    for key, item in parity["checks"].items():
        status = "OK" if item["matches"] else "MISMATCH"
        lines.append(f"  {status:<8} {key}: live={item['live_constant']} expected={item['expected_from_optimized_config']}")

    lines.extend(["", "Mutation Count", f"  {len(mutations)}"])
    return "\n".join(lines) + "\n"


def promote(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    manager = RoundManager("momentum", "nqdtc")
    round_dir = manager.get_round_dir(ROUND_NUM)
    mutations = _build_round5_mutations()
    parity = _parity_snapshot(mutations)
    if args.require_parity and not parity["all_checks_passed"]:
        mismatches = [key for key, item in parity["checks"].items() if not item["matches"]]
        raise RuntimeError(f"Live/backtest parity checks failed: {', '.join(mismatches)}")

    plugin = NQDTCPlugin(data_dir=data_dir, initial_equity=float(args.equity), max_workers=args.max_workers, num_phases=5)
    try:
        metrics = plugin.compute_final_metrics(mutations)
        baseline_metrics = plugin.compute_final_metrics(_read_json(BASE_CONFIG)) if args.compare_round4 else None
        m = _metrics_from_dict(metrics)
        score = score_phase_metrics(5, m)
        state = PhaseState(
            current_phase=5,
            completed_phases=[5],
            cumulative_mutations=mutations,
            phase_results={
                5: {
                    "focus": "PF/net repair refinement adoption",
                    "base_mutations": _read_json(BASE_CONFIG),
                    "final_mutations": mutations,
                    "base_score": 0.0,
                    "final_score": score.total,
                    "kept_features": [PROMOTED_CANDIDATE],
                    "rounds": [
                        asdict(
                            GreedyRound(
                                round_num=1,
                                candidates_tested=1,
                                best_name=PROMOTED_CANDIDATE,
                                best_score=score.total,
                                best_delta_pct=0.0,
                                kept=True,
                                rejected_count=0,
                            )
                        )
                    ],
                    "final_metrics": metrics,
                    "total_candidates": 1,
                    "accepted_count": 1,
                    "promoted_deltas": PROMOTED_DELTAS,
                    "promoted_oos_evidence": PROMOTED_OOS_EVIDENCE,
                }
            },
            round_name="round_5_pf_net_repair_refinement",
        )
        gate = evaluate_gate(plugin._gate_criteria(5, metrics, state))
        state.phase_gate_results[5] = {
            "passed": gate.passed,
            "criteria": [asdict(item) for item in gate.criteria],
            "failure_category": gate.failure_category,
            "recommendations": list(gate.recommendations),
        }
        save_phase_state(state, manager.phase_state_path(round_dir))

        artifacts = plugin.build_end_of_round_artifacts(state)
        diagnostics_path = manager.diagnostics_path(round_dir)
        diagnostics_path.write_text(artifacts.final_diagnostics_text, encoding="utf-8")
    finally:
        plugin.close_pool()

    provenance = build_phase_auto_provenance(
        "nqdtc",
        repo_root=ROOT,
        code_dirs=(ROOT / "backtests/momentum/auto/nqdtc",),
        code_paths=(
            Path(__file__).resolve(),
            ROOT / "backtests/momentum/engine/nqdtc_engine.py",
            ROOT / "backtests/momentum/engine/sim_broker.py",
            ROOT / "backtests/momentum/config_nqdtc.py",
            ROOT / "backtests/momentum/auto/config_mutator.py",
            ROOT / "backtests/momentum/data/replay_cache.py",
            ROOT / "strategies/momentum/nqdtc/config.py",
            ROOT / "strategies/momentum/nqdtc/engine.py",
            ROOT / "strategies/momentum/nqdtc/signals.py",
            ROOT / "strategies/momentum/nqdtc/stops.py",
            ROOT / "strategies/momentum/nqdtc/sizing.py",
        ),
        data_dir=data_dir,
        source_artifacts={
            "round4_optimized_config": BASE_CONFIG,
            "deep_dive_summary": DEEP_DIVE_SUMMARY,
            "deep_dive_text": DEEP_DIVE_TEXT,
            "ablation_summary": ABLATION_TEXT,
            "alpha_repair_summary": ALPHA_REPAIR_SUMMARY,
            "alpha_repair_text": ALPHA_REPAIR_TEXT,
            "alpha_refine_summary": ALPHA_REFINE_SUMMARY,
            "alpha_refine_text": ALPHA_REFINE_TEXT,
        },
        diagnostics_paths={"round5_final": manager.diagnostics_path(round_dir)},
        selection_context={
            "round": ROUND_NUM,
            "promoted_candidate": PROMOTED_CANDIDATE,
            "promoted_deltas": PROMOTED_DELTAS,
            "promoted_oos_evidence": PROMOTED_OOS_EVIDENCE,
            "promotion_basis": "focused alpha repair refinement plus live/backtest parity update",
            "initial_equity": float(args.equity),
        },
    )

    manager.write_run_spec(
        round_dir,
        ROUND_NUM,
        strategy_name="nqdtc",
        description=f"Round 5 adoption of {PROMOTED_CANDIDATE} from PF/net alpha repair refinement.",
        scoring_weights={
            "returns": 0.22,
            "pf": 0.12,
            "expectancy": 0.14,
            "frequency": 0.18,
            "risk": 0.10,
            "exit_capture": 0.16,
            "stability": 0.08,
        },
        baseline_mutations=_read_json(BASE_CONFIG),
        baseline_source=BASE_CONFIG,
        execution_context={
            "candidate": PROMOTED_CANDIDATE,
            "candidate_deltas": PROMOTED_DELTAS,
            "oos_evidence": PROMOTED_OOS_EVIDENCE,
            "parity_checks_passed": parity["all_checks_passed"],
            "data_dir": str(data_dir),
        },
        provenance=provenance,
        provenance_status="complete",
        overwrite=True,
    )
    manager.write_optimized_config(round_dir, mutations)
    manager.write_run_summary(
        round_dir,
        mutations,
        metrics,
        [5],
        round_num=ROUND_NUM,
        source_diagnostics=manager.diagnostics_path(round_dir),
        source_phase_state=manager.phase_state_path(round_dir),
        provenance=provenance,
        provenance_status="complete",
    )
    manager.append_to_manifest(ROUND_NUM, mutations, metrics, provenance=provenance, provenance_status="complete")

    _write_json(round_dir / "live_backtest_parity.json", parity)
    _write_json(
        manager.diagnostics_summary_path(round_dir),
        {
            "family": "momentum",
            "strategy": "nqdtc",
            "round": ROUND_NUM,
            "generated_at_utc": _utc_now(),
            "candidate": PROMOTED_CANDIDATE,
            "promoted_deltas": PROMOTED_DELTAS,
            "promoted_oos_evidence": PROMOTED_OOS_EVIDENCE,
            "headline_metrics": canonicalize_metrics(metrics),
            "final_metrics": metrics,
            "round4_full_replay_metrics": baseline_metrics,
            "metrics_subset": _metrics_subset(metrics),
            "live_backtest_parity": parity,
            "provenance": provenance.to_dict(),
            "provenance_status": "complete",
        },
    )
    manager.evaluation_path(round_dir).write_text(
        _evaluation_text(
            mutations=mutations,
            metrics=metrics,
            baseline_metrics=baseline_metrics,
            parity=parity,
        ),
        encoding="utf-8",
    )

    return {
        "round": ROUND_NUM,
        "candidate": PROMOTED_CANDIDATE,
        "metrics": _metrics_subset(metrics),
        "round4_full_replay_metrics": _metrics_subset(baseline_metrics or {}),
        "parity_checks_passed": parity["all_checks_passed"],
        "paths": {
            "round_dir": str(round_dir.resolve()),
            "optimized_config": str(manager.optimized_config_path(round_dir).resolve()),
            "diagnostics": str(manager.diagnostics_path(round_dir).resolve()),
            "run_summary": str(manager.run_summary_path(round_dir).resolve()),
            "manifest": str(manager.manifest_path.resolve()),
            "parity": str((round_dir / "live_backtest_parity.json").resolve()),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote NQDTC OOS repair candidate to round 5.")
    parser.add_argument("--data-dir", default="backtests/momentum/data/raw")
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--compare-round4", action="store_true", default=True)
    parser.add_argument("--require-parity", action="store_true", default=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    result = promote(parse_args(argv))
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
