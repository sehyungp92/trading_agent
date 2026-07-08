"""Combination probe for the strongest NQ_REGIME OOS repair candidates."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backtests.momentum.auto.nq_regime.current_oos_frequency_repair import (  # noqa: E402
    CURRENT_CONFIG,
    RepairCandidate,
    evaluate_candidates,
    evaluate_direct,
    full_baseline_from_report,
    serialize,
    signature,
    write_json,
)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "candidate_progress.jsonl"
    progress_path.write_text("", encoding="utf-8")
    current = json.loads(Path(args.config_path).read_text(encoding="utf-8"))

    started = time.time()
    baseline_oos = evaluate_direct(current, args.data_end, mode="oos")
    baseline_full = full_baseline_from_report(current, baseline_oos)
    candidates = combo_candidates(current)
    print(f"[nq-regime-combo] evaluating {len(candidates)} combo candidates", flush=True)
    full = evaluate_candidates(
        candidates=candidates,
        baseline=baseline_full,
        data_end=args.data_end,
        mode="full",
        max_workers=args.max_workers,
        progress_path=progress_path,
    )
    ranked = sorted(
        full,
        key=lambda item: (
            item.passed,
            item.run.oos_metrics.net_r,
            item.run.oos_metrics.total_trades,
            item.run.is_metrics.net_r,
        ),
        reverse=True,
    )
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_end": args.data_end,
        "elapsed_seconds": round(time.time() - started, 2),
        "baseline_full": serialize(baseline_full),
        "candidates": [serialize(item) for item in ranked],
    }
    write_json(output_dir / "summary.json", summary)
    if ranked:
        recommended = dict(current)
        recommended.update(ranked[0].candidate.mutations)
        write_json(output_dir / "recommended_combo_config.json", recommended)
    (output_dir / "summary.txt").write_text(format_report(summary), encoding="utf-8")
    print(f"[nq-regime-combo] complete in {(time.time() - started) / 60.0:.1f} min", flush=True)
    print(f"Output: {output_dir.resolve()}", flush=True)


def combo_candidates(current: dict[str, Any]) -> list[RepairCandidate]:
    patches: list[tuple[str, dict[str, Any], str]] = [
        ("combo_radius1_sw_volume08", {
            "param_overrides.REVERSION_SWING_RADIUS": 1,
            "param_overrides.SECOND_WIND_MIN_VOLUME_MULTIPLE": 0.8,
        }, "top_pair"),
        ("combo_radius1_struct_structure_shift", {
            "param_overrides.REVERSION_SWING_RADIUS": 1,
            "param_overrides.STRUCTURAL_ENTRY_MODE": "structure_shift",
        }, "top_pair"),
        ("combo_radius1_reversion_lookback36", {
            "param_overrides.REVERSION_SWING_RADIUS": 1,
            "param_overrides.REVERSION_SWING_LOOKBACK_BARS": 36,
        }, "top_pair"),
        ("combo_radius1_reversion_round5_ablation", {
            "param_overrides.REVERSION_SWING_RADIUS": 1,
            "param_overrides.REVERSION_SWING_LOOKBACK_BARS": 36,
            "param_overrides.REVERSION_SWING_MAX_LEVELS_PER_SIDE": 4,
        }, "top_pair"),
        ("combo_radius1_struct_short8", {
            "param_overrides.REVERSION_SWING_RADIUS": 1,
            "param_overrides.STRUCTURAL_SHORT_MIN_SCORE": 8,
        }, "top_pair"),
        ("combo_radius1_struct_continuation_guarded", {
            "param_overrides.REVERSION_SWING_RADIUS": 1,
            "param_overrides.STRUCTURAL_CONTINUATION_ENABLED": True,
            "param_overrides.STRUCTURAL_CONTINUATION_MIN_SCORE": 9,
            "param_overrides.STRUCTURAL_CONTINUATION_MIN_ROOM_R": 1.0,
            "param_overrides.STRUCTURAL_CONTINUATION_MIN_VOLUME_MULTIPLE": 0.8,
        }, "top_pair"),
        ("combo_radius1_sw_volume08_struct_structure_shift", {
            "param_overrides.REVERSION_SWING_RADIUS": 1,
            "param_overrides.SECOND_WIND_MIN_VOLUME_MULTIPLE": 0.8,
            "param_overrides.STRUCTURAL_ENTRY_MODE": "structure_shift",
        }, "three_way"),
        ("combo_radius1_sw_volume08_rev_lookback36", {
            "param_overrides.REVERSION_SWING_RADIUS": 1,
            "param_overrides.SECOND_WIND_MIN_VOLUME_MULTIPLE": 0.8,
            "param_overrides.REVERSION_SWING_LOOKBACK_BARS": 36,
        }, "three_way"),
        ("combo_radius1_sw_volume08_struct_short8", {
            "param_overrides.REVERSION_SWING_RADIUS": 1,
            "param_overrides.SECOND_WIND_MIN_VOLUME_MULTIPLE": 0.8,
            "param_overrides.STRUCTURAL_SHORT_MIN_SCORE": 8,
        }, "three_way"),
        ("combo_radius1_rev_pen15_stop15", {
            "param_overrides.REVERSION_SWING_RADIUS": 1,
            "param_overrides.REVERSION_MAX_PENETRATION_PTS": 15.0,
            "param_overrides.REVERSION_STANDARD_STOP_CAP": 12.0,
            "param_overrides.REVERSION_A_PLUS_STOP_CAP": 15.0,
        }, "reversion_stack"),
    ]
    current_sig = signature(current)
    seen: set[str] = set()
    out: list[RepairCandidate] = []
    for name, patch, source in patches:
        merged = dict(current)
        merged.update(patch)
        sig = signature(merged)
        if sig == current_sig or sig in seen:
            continue
        seen.add(sig)
        out.append(
            RepairCandidate(
                name=name,
                stage="combo",
                mutations=merged,
                intent="Stack strongest single-mutation OOS repair ideas.",
                source=source,
            )
        )
    return out


def format_report(summary: dict[str, Any]) -> str:
    base = summary["baseline_full"]
    lines = [
        "NQ_REGIME Combo Probe",
        "=" * 88,
        (
            f"Baseline IS trades={base['is_metrics']['total_trades']} "
            f"netR={base['is_metrics']['net_r']:.2f}; "
            f"OOS trades={base['oos_metrics']['total_trades']} "
            f"netR={base['oos_metrics']['net_r']:.2f}"
        ),
        "",
    ]
    for item in summary["candidates"]:
        cand = item["candidate"]
        run = item["run"]
        lines.append(
            f"{cand['name']}: passed={item['passed']} obj={item['objective_delta']:+.2%} "
            f"OOS trades={run['oos_metrics']['total_trades']} netR={run['oos_metrics']['net_r']:.2f} "
            f"avgR={run['oos_metrics']['avg_r']:.3f}; "
            f"IS trades={run['is_metrics']['total_trades']} netR={run['is_metrics']['net_r']:.1f} "
            f"avgR={run['is_metrics']['avg_r']:.3f}; reasons={item.get('reasons', [])}"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", default=str(CURRENT_CONFIG))
    parser.add_argument("--data-end", default="2026-05-01")
    parser.add_argument("--output-dir", default="backtests/output/momentum/nq_regime/current_oos_combo_probe_20260504")
    parser.add_argument("--max-workers", type=int, default=4)
    return parser


if __name__ == "__main__":
    main()
