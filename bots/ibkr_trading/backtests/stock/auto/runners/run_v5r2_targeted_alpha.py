"""IARIC Pullback V5R2 -- targeted residual-alpha optimization.

Starts from the latest corrected IARIC round, normally
``backtests/output/stock/iaric/round_2/optimized_config.json``, and only
searches existing pullback configuration parameters.  The round keeps the
headline score to seven scaled components:

  net_profit, expected_total_r, profit_factor, sharpe, inv_dd, total_trades,
  residual_alpha_quality.

Usage::

    python -m backtests.stock.auto.runners.run_v5r2_targeted_alpha --max-workers 2
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtests.shared.auto.phase_runner import PhaseRunner
from backtests.shared.auto.round_manager import RoundManager
from backtests.stock.auto.iaric.phase_candidates import (
    V5R2_BASE_MUTATIONS,
    V5R2_PHASE_CANDIDATES,
    V5R2_PHASE_FOCUS,
)
from backtests.stock.auto.iaric.phase_scoring import V5R2_PHASE_SCORING_WEIGHTS
from backtests.stock.auto.iaric.plugin import IARICPullbackPlugin

DATA_DIR = Path("backtests/stock/data/raw")
START_DATE = "2024-01-01"
END_DATE = "2026-03-01"
INITIAL_EQUITY = 10_000.0
PROFILE = "mainline"
NUM_PHASES = 4
MAX_WORKERS = 2
MAX_ROUNDS = 4
MIN_DELTA = 0.0015


def _previous_round_lineage(round_num: int) -> tuple[str, int]:
    if round_num == 3:
        return "v5r1", 5
    return "v5r2", NUM_PHASES


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--profile", default=PROFILE, choices=["mainline", "aggressive"])
    parser.add_argument("--num-phases", type=int, default=NUM_PHASES, choices=range(1, NUM_PHASES + 1))
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument("--start-phase", type=int, default=None, choices=range(1, NUM_PHASES + 1))
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    parser.add_argument("--min-delta", type=float, default=MIN_DELTA)
    return parser.parse_args()


def _score_components() -> list[str]:
    components = list(next(iter(V5R2_PHASE_SCORING_WEIGHTS.values())).keys())
    if len(components) > 7:
        raise ValueError(f"V5R2 score has {len(components)} components; expected at most 7.")
    return components


def _metric_summary(metrics: dict[str, Any]) -> dict[str, float]:
    keys = [
        "net_profit",
        "total_trades",
        "avg_r",
        "expected_total_r",
        "profit_factor",
        "sharpe",
        "max_drawdown_pct",
        "carry_trade_share",
        "carry_avg_r",
        "gap_selectivity_edge",
        "crowded_day_discrimination",
    ]
    return {key: float(metrics.get(key, 0.0) or 0.0) for key in keys}


def _baseline_signature(base_mutations: dict[str, Any]) -> str:
    payload = {
        "base_mutations": dict(sorted(base_mutations.items())),
        "score_weights": V5R2_PHASE_SCORING_WEIGHTS,
        "candidate_queues": V5R2_PHASE_CANDIDATES,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_phase0_baseline(
    plugin: IARICPullbackPlugin,
    output_dir: Path,
    base_mutations: dict[str, Any],
) -> dict[str, Any]:
    signature = _baseline_signature(base_mutations)
    path = output_dir / "phase_0_baseline.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("signature") == signature:
                return existing
        except (json.JSONDecodeError, OSError):
            pass

    metrics = plugin.compute_final_metrics(base_mutations)
    hard_rejects = dict(plugin._phase_hard_rejects.get(1, {}))
    reject_reason = plugin._phase_reject_reason(1, metrics, hard_rejects)
    score = 0.0 if reject_reason else plugin._score_phase_metrics(
        1,
        metrics,
        dict(plugin._phase_scoring_weights.get(1, {})),
    )
    payload = {
        "purpose": (
            "Phase-0 freeze of the corrected round-2 baseline before V5R2 residual-alpha search."
        ),
        "signature": signature,
        "score": score,
        "rejected": bool(reject_reason),
        "reject_reason": reject_reason,
        "score_component_count": len(_score_components()),
        "score_components": _score_components(),
        "metrics": _metric_summary(metrics),
        "mutations": dict(sorted(base_mutations.items())),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

    lines = [
        "=" * 72,
        "IARIC V5R2 PHASE 0 BASELINE",
        "=" * 72,
        f"Score: {score:.6f}",
        f"Rejected: {payload['rejected']}",
    ]
    if reject_reason:
        lines.append(f"Reject reason: {reject_reason}")
    metric_parts = payload["metrics"]
    lines.extend(
        [
            "",
            (
                f"Trades={metric_parts['total_trades']:.0f} "
                f"Net=${metric_parts['net_profit']:,.2f} "
                f"ExpR={metric_parts['expected_total_r']:.2f} "
                f"AvgR={metric_parts['avg_r']:.4f}"
            ),
            (
                f"PF={metric_parts['profit_factor']:.3f} "
                f"Sharpe={metric_parts['sharpe']:.3f} "
                f"DD={metric_parts['max_drawdown_pct']:.2%}"
            ),
            (
                f"Carry share={metric_parts['carry_trade_share']:.2%} "
                f"Carry avgR={metric_parts['carry_avg_r']:.4f} "
                f"Gap edge={metric_parts['gap_selectivity_edge']:.4f} "
                f"Crowded edge={metric_parts['crowded_day_discrimination']:.4f}"
            ),
        ]
    )
    (output_dir / "phase_0_baseline.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if reject_reason:
        raise RuntimeError(f"V5R2 phase-0 baseline failed hard gates: {reject_reason}")
    return payload


def _write_manifest(
    output_dir: Path,
    *,
    start_date: str,
    end_date: str,
    max_workers: int,
    profile: str,
    num_phases: int,
    max_rounds: int,
    min_delta: float,
    base_mutations: dict[str, Any],
    baseline_source: str,
    data_fingerprint: str,
) -> None:
    components = _score_components()
    phase_counts = {
        str(phase): {
            "focus": V5R2_PHASE_FOCUS[phase][0],
            "count": len(V5R2_PHASE_CANDIDATES.get(phase, [])),
            "queue_ids": [name for name, _ in V5R2_PHASE_CANDIDATES.get(phase, [])],
        }
        for phase in sorted(V5R2_PHASE_FOCUS)
        if phase <= num_phases
    }
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "framework": "backtests/shared/auto",
        "plugin": "IARICPullbackPlugin",
        "round": "V5R2",
        "profile": profile,
        "start_date": start_date,
        "end_date": end_date,
        "initial_equity": INITIAL_EQUITY,
        "max_workers": max_workers,
        "num_phases": num_phases,
        "max_rounds": max_rounds,
        "min_delta": min_delta,
        "baseline_source": baseline_source,
        "data_fingerprint": data_fingerprint,
        "base_mutations": dict(sorted(base_mutations.items())),
        "score_component_count": len(components),
        "score_components": components,
        "phase_count": len(phase_counts),
        "queue_count": sum(item["count"] for item in phase_counts.values()),
        "phases": phase_counts,
        "implementation_guardrail": (
            "Config-surface optimization only; no execution, fill, timestamp, "
            "or diagnostic denominator semantics changed. Canonical fill timing "
            "remains pb_open_scored_fill_timing=next_5m_open."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    round_manager: RoundManager | None = None
    round_num: int | None = None
    previous_round_provenance = None
    baseline_source = "V5R2_BASE_MUTATIONS"
    base_mutations = dict(V5R2_BASE_MUTATIONS)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        round_manager = RoundManager("stock", "iaric")
        refresh_existing = args.start_phase is not None and args.round is not None
        round_num, output_dir = round_manager.resolve_round(
            args.round,
            for_write=not refresh_existing,
            expected_phases=args.num_phases,
        )
        if round_num and round_num > 1:
            previous_round_name, previous_num_phases = _previous_round_lineage(round_num)
            provenance_probe = IARICPullbackPlugin(
                DATA_DIR,
                start_date=args.start_date,
                end_date=args.end_date,
                initial_equity=INITIAL_EQUITY,
                max_workers=args.max_workers,
                num_phases=previous_num_phases,
                profile=args.profile,
                round_name=previous_round_name,
            ).build_provenance()
            previous_round_provenance = provenance_probe
            base_mutations = round_manager.get_previous_mutations(
                round_num,
                current_provenance=provenance_probe,
            )
            baseline_source = str(round_manager.optimized_config_path(round_manager.round_path(round_num - 1)).resolve())
    base_mutations["param_overrides.pb_open_scored_fill_timing"] = "next_5m_open"

    print("=" * 72)
    print("IARIC Pullback V5R2 -- Targeted Residual-Alpha Auto-Optimization")
    print("=" * 72)
    print(f"Output dir: {output_dir}")
    print(f"Baseline: {baseline_source}")
    print(f"Date range: {args.start_date} -> {args.end_date}")
    print(f"Profile: {args.profile}")
    print(f"Phases: {args.num_phases}")
    print(f"Max rounds per phase: {args.max_rounds}")
    print(f"Min score delta: {args.min_delta:.4f}")
    print(f"Max workers: {args.max_workers}")
    total_candidates = 0
    for phase in sorted(V5R2_PHASE_FOCUS):
        if phase > args.num_phases:
            break
        candidates = V5R2_PHASE_CANDIDATES.get(phase, [])
        total_candidates += len(candidates)
        print(f"Phase {phase}: {len(candidates):>3} candidates | {V5R2_PHASE_FOCUS[phase][0]}")
    print(f"Total: {total_candidates} candidates across {args.num_phases} phases")
    print(f"Score components ({len(_score_components())}): {', '.join(_score_components())}")
    print(flush=True)

    plugin = IARICPullbackPlugin(
        DATA_DIR,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_equity=INITIAL_EQUITY,
        max_workers=args.max_workers,
        num_phases=args.num_phases,
        profile=args.profile,
        round_name="v5r2",
    )
    plugin.initial_mutations = base_mutations
    plugin.previous_round_provenance = previous_round_provenance

    phase0 = _write_phase0_baseline(plugin, output_dir, base_mutations)
    print(
        "Phase 0 baseline: "
        f"score={float(phase0['score']):.6f} "
        f"net=${float(phase0['metrics']['net_profit']):,.2f} "
        f"PF={float(phase0['metrics']['profit_factor']):.3f} "
        f"DD={float(phase0['metrics']['max_drawdown_pct']):.2%}",
        flush=True,
    )

    _write_manifest(
        output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=args.max_workers,
        profile=args.profile,
        num_phases=args.num_phases,
        max_rounds=args.max_rounds,
        min_delta=args.min_delta,
        base_mutations=base_mutations,
        baseline_source=baseline_source,
        data_fingerprint=plugin._replay_data_fingerprint(),
    )

    runner = PhaseRunner(
        plugin=plugin,
        output_dir=output_dir,
        round_name="v5r2_targeted_alpha",
        max_rounds=args.max_rounds,
        min_delta=args.min_delta,
        max_retries=0,
        max_diagnostic_retries=0,
        round_manager=round_manager,
        round_num=round_num,
    )
    runner.run_all_phases(start_phase=args.start_phase)


if __name__ == "__main__":
    main()
