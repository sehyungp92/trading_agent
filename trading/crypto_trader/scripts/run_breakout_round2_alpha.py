"""Run breakout round 2 alpha-focused phased auto optimization."""

from __future__ import annotations

import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import structlog

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_trader.cli import _configure_logging, _update_rounds_manifest
from crypto_trader.optimize.breakout_round2_alpha import (
    OPTIMIZATION_END_DATE,
    PHASE_NAMES,
    ROUND2_ALPHA_HARD_REJECTS,
    ROUND2_ALPHA_PHASE_GATE_CRITERIA,
    ROUND2_ALPHA_SCORING_CEILINGS,
    ROUND2_ALPHA_SCORING_WEIGHTS,
    BreakoutRound2AlphaPlugin,
    build_backtest_config,
    load_breakout_strategy,
)
from crypto_trader.optimize.breakout_round4_trade_frequency import run_greedy_without_pruning
import crypto_trader.optimize.phase_runner as phase_runner_module
from crypto_trader.optimize.phase_runner import PhaseRunner
from crypto_trader.optimize.phase_state import PhaseState


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _serialise_gate_criteria() -> dict[str, list[dict[str, float | str]]]:
    return {
        str(phase): [
            {
                "metric": criterion.metric,
                "operator": criterion.operator,
                "threshold": criterion.threshold,
            }
            for criterion in criteria
        ]
        for phase, criteria in ROUND2_ALPHA_PHASE_GATE_CRITERIA.items()
    }


def _select_promoted_phase(state: PhaseState) -> int | None:
    """Return the latest completed phase that passed its gate."""
    completed = sorted(state.completed_phases)
    for phase in reversed(completed):
        if state.phase_gate_results.get(phase, {}).get("passed") is True:
            return phase
    return completed[-1] if completed else None


def _phase_final_mutations(state: PhaseState, phase: int) -> dict[str, Any]:
    result = state.phase_results.get(phase, {})
    mutations = result.get("final_mutations")
    if isinstance(mutations, dict):
        return dict(mutations)

    cumulative: dict[str, Any] = {}
    for completed_phase in sorted(p for p in state.completed_phases if p <= phase):
        phase_result = state.phase_results.get(completed_phase, {})
        phase_mutations = phase_result.get("final_mutations")
        if isinstance(phase_mutations, dict):
            cumulative.update(phase_mutations)
    return cumulative


def _copy_raw_final_artifacts(round_dir: Path, raw_final_phase: int) -> None:
    suffix = f"phase_{raw_final_phase}"
    for source_name, target_name in {
        "round_final_diagnostics.txt": f"round_exploratory_{suffix}_diagnostics.txt",
        "round_evaluation.txt": f"round_exploratory_{suffix}_evaluation.txt",
        "optimized_config.json": f"exploratory_{suffix}_config.json",
    }.items():
        source = round_dir / source_name
        if source.exists():
            shutil.copy2(source, round_dir / target_name)


def _build_promoted_state(state: PhaseState, promoted_phase: int) -> PhaseState:
    keep = [phase for phase in sorted(state.completed_phases) if phase <= promoted_phase]
    return PhaseState(
        current_phase=promoted_phase + 1,
        completed_phases=keep,
        cumulative_mutations=_phase_final_mutations(state, promoted_phase),
        phase_metrics={
            phase: deepcopy(state.phase_metrics[phase])
            for phase in keep
            if phase in state.phase_metrics
        },
        round_name=state.round_name,
        scoring_retries={
            phase: count for phase, count in state.scoring_retries.items() if phase <= promoted_phase
        },
        diagnostic_retries={
            phase: count
            for phase, count in state.diagnostic_retries.items()
            if phase <= promoted_phase
        },
        retry_count={
            phase: count for phase, count in state.retry_count.items() if phase <= promoted_phase
        },
        phase_results={
            phase: deepcopy(state.phase_results[phase])
            for phase in keep
            if phase in state.phase_results
        },
        phase_gate_results={
            phase: deepcopy(state.phase_gate_results[phase])
            for phase in keep
            if phase in state.phase_gate_results
        },
        phase_timestamps={
            phase: deepcopy(state.phase_timestamps[phase])
            for phase in keep
            if phase in state.phase_timestamps
        },
        contract_hash=state.contract_hash,
        contract=deepcopy(state.contract),
        invalid_phases={
            phase: deepcopy(payload)
            for phase, payload in state.invalid_phases.items()
            if phase <= promoted_phase
        },
    )


def _metric_subset(metrics: dict[str, Any] | None) -> dict[str, Any]:
    metrics = metrics or {}
    keys = (
        "total_trades",
        "net_return_pct",
        "profit_factor",
        "expectancy_r",
        "max_drawdown_pct",
        "sharpe_ratio",
        "exit_efficiency",
    )
    return {key: metrics.get(key) for key in keys if key in metrics}


def _write_promotion_artifacts(
    round_dir: Path,
    state: PhaseState,
    promoted_state: PhaseState,
    raw_final_phase: int | None,
    promoted_phase: int | None,
) -> dict[str, Any]:
    raw_gate = state.phase_gate_results.get(raw_final_phase, {}) if raw_final_phase else {}
    promoted_gate = (
        promoted_state.phase_gate_results.get(promoted_phase, {}) if promoted_phase else {}
    )
    raw_metrics = state.phase_metrics.get(raw_final_phase) if raw_final_phase else None
    promoted_metrics = (
        promoted_state.phase_metrics.get(promoted_phase) if promoted_phase else None
    )
    summary = {
        "promoted_phase": promoted_phase,
        "raw_final_phase": raw_final_phase,
        "promoted_differs_from_raw_final": promoted_phase != raw_final_phase,
        "promotion_reason": (
            "latest gate-passing phase promoted; later exploratory phases failed phase gates"
            if promoted_phase != raw_final_phase
            else "raw final phase passed its gate"
        ),
        "promoted_gate_passed": promoted_gate.get("passed"),
        "promoted_gate_failures": promoted_gate.get("failure_reasons", []),
        "raw_final_gate_passed": raw_gate.get("passed"),
        "raw_final_gate_failures": raw_gate.get("failure_reasons", []),
        "promoted_metrics": _metric_subset(promoted_metrics),
        "raw_final_metrics": _metric_subset(raw_metrics),
        "promoted_mutations": dict(promoted_state.cumulative_mutations),
        "raw_final_mutations": dict(state.cumulative_mutations),
    }
    _write_json(round_dir / "round_promotion.json", summary)

    optimized_path = round_dir / "optimized_config.json"
    if optimized_path.exists():
        payload = json.loads(optimized_path.read_text(encoding="utf-8"))
        payload.setdefault("metadata", {})["promotion"] = summary
        _write_json(optimized_path, payload)

    return summary


def main() -> None:
    _configure_logging()
    log = structlog.get_logger("scripts.breakout_round2_alpha")

    data_dir = ROOT / "data"
    output_dir = ROOT / "output" / "breakout"
    round_dir = output_dir / "round_2"
    round_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = output_dir / "round_1" / "optimized_config.json"
    base_config = load_breakout_strategy(baseline_path)
    bt_cfg, window_meta = build_backtest_config(data_dir)

    plugin = BreakoutRound2AlphaPlugin(
        backtest_config=bt_cfg,
        base_config=base_config,
        data_dir=data_dir,
        max_workers=2,
    )
    phase_runner_module.run_greedy = run_greedy_without_pruning
    runner = PhaseRunner(
        plugin,
        round_dir,
        round_name="breakout_round2_alpha_holdout_capped",
        validation_mode="strict",
    )
    state = runner.load_state()

    run_spec = {
        "description": (
            "Breakout round 2 phased auto run seeded from output/breakout/round_1/"
            "optimized_config.json. The search is holdout-capped at 2026-04-20, "
            "keeps live/backtest parity by mutating only shared BreakoutConfig "
            "fields, and targets signal discrimination, selective entry quality, "
            "trade capture, and structural frequency before narrow finetuning."
        ),
        "symbols": list(bt_cfg.symbols),
        "max_workers": 2,
        "validation_mode": "strict",
        "contract_hash": runner.contract.get("contract_hash", ""),
        "profile_hash": runner.contract.get("profile_hash", ""),
        "contract": runner.contract,
        "window": window_meta,
        "holdout_policy": {
            "excluded_after": OPTIMIZATION_END_DATE.isoformat(),
            "note": "Bars after 2026-04-20 are reserved as holdout and not used by this round.",
        },
        "immutable_score": {
            "component_count": len(ROUND2_ALPHA_SCORING_WEIGHTS),
            "weights": ROUND2_ALPHA_SCORING_WEIGHTS,
            "ceilings": ROUND2_ALPHA_SCORING_CEILINGS,
            "hard_rejects": {
                key: {"operator": op, "threshold": threshold}
                for key, (op, threshold) in ROUND2_ALPHA_HARD_REJECTS.items()
            },
            "notes": [
                "The score uses exactly seven components: returns, coverage, expectancy, edge, capture, Sharpe, and risk.",
                "Ceilings are set above the strong round_1 baseline to avoid rewarding saturated PF or return spikes.",
                "Hard rejects enforce minimum sample, edge, positive expectancy, capture, and an aggressive-but-capped drawdown stance.",
            ],
        },
        "phase_gate_criteria": _serialise_gate_criteria(),
        "baseline_strategy": base_config.to_dict(),
        "baseline_source": str(baseline_path),
        "phase_objectives": {
            "1": "Tighten signal quality and block low-confluence, weak-body, low-volume, or poor-room trades.",
            "2": "Audit relaxed-body and symbol/direction pockets so negative or low-quality signal branches are rejected.",
            "3": "Test entry architecture changes that distinguish clean continuation from weak retests.",
            "4": "Improve capture and failure handling without adding backtest-only exits.",
            "5": "Expand trading frequency only through structural zone/profile/reentry changes that remain quality-gated.",
            "6": "Finetune accepted non-risk numeric parameters by +/-5% while excluding risk and risk-scale knobs.",
        },
        "parity_notes": [
            "No strategy-engine or backtest-only logic is introduced by this round.",
            "All experiments are normal BreakoutConfig mutations consumed by the shared live/backtest strategy path.",
            "The optimizer uses the live parity economic profile through the standard optimization contract preflight.",
        ],
        "excluded_search_space": [
            "Risk percentage, leverage, and risk-scale changes.",
            "Session, funding, or hour filters that could fit the small sample without a robust causal reason.",
            "Any same-bar fill, hindsight, or diagnostics-only behavior.",
        ],
        "promotion_policy": {
            "official_config": "latest completed phase whose phase gate passed",
            "reason": (
                "Later phases may be useful exploratory evidence, but gate-failed states "
                "are not promoted to optimized_config.json or the rounds manifest."
            ),
        },
        "phase_names": PHASE_NAMES,
    }
    _write_json(round_dir / "run_spec.json", run_spec)

    log.info(
        "breakout.round2_alpha.start",
        output_dir=str(round_dir),
        baseline=str(baseline_path),
        start_date=window_meta["start_date"],
        end_date=window_meta["end_date"],
        holdout_excluded_after=window_meta["holdout_excluded_after"],
        workers=2,
        score_components=len(ROUND2_ALPHA_SCORING_WEIGHTS),
    )

    state = runner.run_all_phases(state)

    raw_final_phase = max(state.phase_metrics) if state.phase_metrics else None
    promoted_phase = _select_promoted_phase(state)
    report_state = state
    if promoted_phase is not None and raw_final_phase is not None and promoted_phase != raw_final_phase:
        _copy_raw_final_artifacts(round_dir, raw_final_phase)
        report_state = _build_promoted_state(state, promoted_phase)
        if hasattr(plugin, "_last_result"):
            plugin._last_result = None
        runner.run_end_of_round(report_state)

    promotion = _write_promotion_artifacts(
        round_dir,
        state,
        report_state,
        raw_final_phase,
        promoted_phase,
    )

    final_phase = promoted_phase
    final_metrics = report_state.phase_metrics.get(final_phase) if final_phase else None
    final_mutations = dict(report_state.cumulative_mutations) if final_phase else {}

    phase_result = report_state.phase_results.get(final_phase, {}) if final_phase else {}
    gate_result = report_state.phase_gate_results.get(final_phase, {}) if final_phase else {}
    _update_rounds_manifest(
        output_dir,
        2,
        final_mutations,
        final_metrics,
        contract=runner.contract,
        phase_result=phase_result,
        gate_result=gate_result,
    )

    log.info(
        "breakout.round2_alpha.complete",
        output_dir=str(round_dir),
        final_phase=final_phase,
        raw_final_phase=raw_final_phase,
        promoted_differs_from_raw_final=promotion["promoted_differs_from_raw_final"],
        final_mutations=len(final_mutations),
        total_trades=(final_metrics or {}).get("total_trades"),
        net_return_pct=(final_metrics or {}).get("net_return_pct"),
        profit_factor=(final_metrics or {}).get("profit_factor"),
        expectancy_r=(final_metrics or {}).get("expectancy_r"),
        max_drawdown_pct=(final_metrics or {}).get("max_drawdown_pct"),
        exit_efficiency=(final_metrics or {}).get("exit_efficiency"),
    )


if __name__ == "__main__":
    main()
