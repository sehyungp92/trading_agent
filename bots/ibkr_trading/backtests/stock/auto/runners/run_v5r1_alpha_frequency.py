"""IARIC Pullback V5R1 -- alpha extraction and frequency expansion.

Starts from the latest optimized centralized round config, normally
``backtests/output/stock/iaric/round_1/optimized_config.json`` for this run,
and keeps all experiments inside the existing pullback config surface.

The immutable V5R1 score has seven components:
  expected_total_r, total_trades, avg_r, profit_factor, sharpe, inv_dd,
  alpha_discrimination.

Usage::

    python -m backtests.stock.auto.runners.run_v5r1_alpha_frequency --max-workers 2
    python -m backtests.stock.auto.runners.run_v5r1_alpha_frequency --round 2 --max-workers 2
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
    V5R1_BASE_MUTATIONS,
    V5R1_PHASE_CANDIDATES,
    V5R1_PHASE_FOCUS,
)
from backtests.stock.auto.iaric.phase_scoring import V5R1_PHASE_SCORING_WEIGHTS
from backtests.stock.auto.iaric.plugin import IARICPullbackPlugin

DATA_DIR = Path("backtests/stock/data/raw")
START_DATE = "2024-01-01"
END_DATE = "2026-03-01"
INITIAL_EQUITY = 10_000.0
PROFILE = "mainline"
NUM_PHASES = 5


def _previous_round_lineage(round_num: int) -> tuple[str, int]:
    if round_num == 2:
        return "v4r1", 5
    return "v5r1", NUM_PHASES


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--profile", default=PROFILE, choices=["mainline", "aggressive"])
    parser.add_argument("--num-phases", type=int, default=NUM_PHASES)
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument("--start-phase", type=int, default=None, choices=range(1, NUM_PHASES + 1))
    parser.add_argument("--phase0-ablation", action="store_true")
    return parser.parse_args()


def _write_manifest(
    output_dir: Path,
    *,
    start_date: str,
    end_date: str,
    max_workers: int,
    profile: str,
    num_phases: int,
    base_mutations: dict[str, Any],
    baseline_source: str,
    phase0_ablation_enabled: bool,
) -> None:
    phase_counts = {
        str(phase): {
            "focus": V5R1_PHASE_FOCUS[phase][0],
            "count": len(V5R1_PHASE_CANDIDATES.get(phase, [])),
            "queue_ids": [name for name, _ in V5R1_PHASE_CANDIDATES.get(phase, [])],
        }
        for phase in sorted(V5R1_PHASE_FOCUS)
        if phase <= num_phases
    }
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "framework": "backtests/shared/auto",
        "plugin": "IARICPullbackPlugin",
        "round": "V5R1",
        "profile": profile,
        "start_date": start_date,
        "end_date": end_date,
        "initial_equity": INITIAL_EQUITY,
        "max_workers": max_workers,
        "num_phases": num_phases,
        "baseline_source": baseline_source,
        "phase0_ablation_enabled": phase0_ablation_enabled,
        "base_mutations": dict(sorted(base_mutations.items())),
        "score_component_count": len(next(iter(V5R1_PHASE_SCORING_WEIGHTS.values()))),
        "score_components": list(next(iter(V5R1_PHASE_SCORING_WEIGHTS.values())).keys()),
        "phase_count": len(phase_counts),
        "queue_count": sum(item["count"] for item in phase_counts.values()),
        "phases": phase_counts,
        "implementation_guardrail": (
            "Config-surface optimization only; no execution, fill, timestamp, or "
            "diagnostic denominator semantics changed."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _phase0_ablation_candidates() -> list[tuple[str, dict[str, Any]]]:
    return [
        (
            "open_scored_slots_2",
            {
                "param_overrides.pb_v2_open_scored_max_slots": 2,
            },
        ),
        (
            "open_scored_min_55",
            {
                "param_overrides.pb_v2_open_scored_min_score": 55.0,
            },
        ),
        (
            "open_scored_rank_75",
            {
                "param_overrides.pb_v2_open_scored_rank_pct_max": 75.0,
            },
        ),
        (
            "open_scored_off",
            {
                "param_overrides.pb_v2_open_scored_enabled": False,
                "param_overrides.pb_open_scored_enabled": False,
            },
        ),
    ]


def _phase0_metrics_summary(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "total_trades": float(metrics.get("total_trades", 0.0)),
        "avg_r": float(metrics.get("avg_r", 0.0)),
        "expected_total_r": float(metrics.get("expected_total_r", 0.0)),
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "sharpe": float(metrics.get("sharpe", 0.0)),
        "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0.0)),
        "open_scored_share": float(metrics.get("open_scored_share", 0.0)),
    }


def _phase0_signature(base_mutations: dict[str, Any]) -> str:
    payload = {
        "base_mutations": dict(sorted(base_mutations.items())),
        "candidates": [
            (name, dict(sorted(mutations.items())))
            for name, mutations in _phase0_ablation_candidates()
        ],
        "score_weights": V5R1_PHASE_SCORING_WEIGHTS.get(1, {}),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_phase0_ablation(
    plugin: IARICPullbackPlugin,
    base_mutations: dict[str, Any],
    output_dir: Path,
    *,
    min_delta: float = 0.003,
) -> dict[str, Any]:
    signature = _phase0_signature(base_mutations)
    existing_path = output_dir / "phase_0_ablation.json"
    if existing_path.exists():
        try:
            payload = json.loads(existing_path.read_text(encoding="utf-8"))
            if payload.get("signature") == signature:
                adopted_name = str(payload.get("adopted_name", "__baseline__"))
                candidate_lookup = dict(_phase0_ablation_candidates())
                if adopted_name in candidate_lookup:
                    mutations = dict(base_mutations)
                    mutations.update(candidate_lookup[adopted_name])
                    return mutations
                if adopted_name == "__baseline__":
                    return dict(base_mutations)
        except (json.JSONDecodeError, OSError):
            pass

    hard_rejects = dict(plugin._phase_hard_rejects.get(1, {}))
    scoring_weights = dict(plugin._phase_scoring_weights.get(1, {}))

    baseline_metrics = plugin._run_config(base_mutations, store_context=False)["metrics"]
    baseline_reject = plugin._phase_reject_reason(1, baseline_metrics, hard_rejects)
    baseline_score = 0.0 if baseline_reject else plugin._score_phase_metrics(1, baseline_metrics, scoring_weights)
    records: list[dict[str, Any]] = [
        {
            "name": "__baseline__",
            "score": baseline_score,
            "rejected": bool(baseline_reject),
            "reject_reason": baseline_reject,
            "mutations": {},
            "metrics": _phase0_metrics_summary(baseline_metrics),
        }
    ]

    best_name = "__baseline__"
    best_score = baseline_score
    best_mutations = dict(base_mutations)
    for name, candidate_mutations in _phase0_ablation_candidates():
        mutations = dict(base_mutations)
        mutations.update(candidate_mutations)
        metrics = plugin._run_config(mutations, store_context=False)["metrics"]
        reject_reason = plugin._phase_reject_reason(1, metrics, hard_rejects)
        score = 0.0 if reject_reason else plugin._score_phase_metrics(1, metrics, scoring_weights)
        record = {
            "name": name,
            "score": score,
            "delta_vs_baseline": score - baseline_score,
            "rejected": bool(reject_reason),
            "reject_reason": reject_reason,
            "mutations": candidate_mutations,
            "metrics": _phase0_metrics_summary(metrics),
        }
        records.append(record)
        if not reject_reason and score > best_score + min_delta:
            best_name = name
            best_score = score
            best_mutations = mutations

    payload = {
        "purpose": "Conservative open-scored timing phase-0 ablation of inherited V5R1 baseline settings.",
        "adopted": best_name != "__baseline__",
        "adopted_name": best_name,
        "baseline_score": baseline_score,
        "best_score": best_score,
        "min_delta": min_delta,
        "signature": signature,
        "records": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase_0_ablation.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    lines = [
        "=" * 72,
        "IARIC V5R1 PHASE 0 ABLATION",
        "=" * 72,
        f"Adopted: {payload['adopted']} ({best_name})",
        f"Score: {baseline_score:.4f} -> {best_score:.4f}",
        "",
    ]
    for record in records:
        metrics = record["metrics"]
        lines.append(
            f"{record['name']}: score={record['score']:.4f} "
            f"trades={metrics['total_trades']:.0f} avg_r={metrics['avg_r']:+.3f} "
            f"expR={metrics['expected_total_r']:+.2f} PF={metrics['profit_factor']:.2f} "
            f"Sharpe={metrics['sharpe']:.2f} DD={metrics['max_drawdown_pct']:.1%} "
            f"open_scored={metrics['open_scored_share']:.1%}"
        )
        if record.get("reject_reason"):
            lines.append(f"  rejected: {record['reject_reason']}")
    (output_dir / "phase_0_ablation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return best_mutations


def main() -> None:
    args = _parse_args()
    round_manager = None
    round_num = None
    previous_round_provenance = None
    baseline_source = "V5R1_BASE_MUTATIONS"
    base_mutations = dict(V5R1_BASE_MUTATIONS)
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
    base_mutations.setdefault("param_overrides.pb_open_scored_fill_timing", "next_5m_open")

    print("=" * 72)
    print("IARIC Pullback V5R1 -- Alpha/Frequency Auto-Optimization")
    print("=" * 72)
    print(f"Output dir: {output_dir}")
    print(f"Baseline: {baseline_source}")
    print(f"Date range: {args.start_date} -> {args.end_date}")
    print(f"Profile: {args.profile}")
    print(f"Phases: {args.num_phases}")
    print(f"Max workers: {args.max_workers}")
    total_candidates = 0
    for phase in sorted(V5R1_PHASE_FOCUS):
        if phase > args.num_phases:
            break
        candidates = V5R1_PHASE_CANDIDATES.get(phase, [])
        total_candidates += len(candidates)
        print(f"Phase {phase}: {len(candidates):>3} candidates | {V5R1_PHASE_FOCUS[phase][0]}")
    print(f"Total: {total_candidates} candidates across {args.num_phases} phases")
    print(flush=True)

    plugin = IARICPullbackPlugin(
        DATA_DIR,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_equity=INITIAL_EQUITY,
        max_workers=args.max_workers,
        num_phases=args.num_phases,
        profile=args.profile,
        round_name="v5r1",
    )
    plugin.initial_mutations = base_mutations
    plugin.previous_round_provenance = previous_round_provenance
    if args.phase0_ablation:
        base_mutations = _run_phase0_ablation(plugin, base_mutations, output_dir)
        plugin.initial_mutations = base_mutations

    _write_manifest(
        output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        max_workers=args.max_workers,
        profile=args.profile,
        num_phases=args.num_phases,
        base_mutations=base_mutations,
        baseline_source=baseline_source,
        phase0_ablation_enabled=args.phase0_ablation,
    )

    runner = PhaseRunner(
        plugin=plugin,
        output_dir=output_dir,
        round_name="v5r1",
        max_retries=0,
        max_diagnostic_retries=0,
        round_manager=round_manager,
        round_num=round_num,
    )
    runner.run_all_phases(start_phase=args.start_phase)


if __name__ == "__main__":
    main()
