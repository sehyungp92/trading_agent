"""Focused refinement around the round-5 NQDTC alpha repair leaders."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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

from backtests.swing.auto.incumbent_repair import MISSING, RepairCandidate, read_json, serialize, write_json  # noqa: E402
from backtests.swing.auto.oos_repair_diagnostics import evaluate_strategy  # noqa: E402
from backtests.momentum.auto.nqdtc.round5_alpha_repair import (  # noqa: E402
    ROUND4_CONFIG,
    ROUND5_CONFIG,
    compute_full_metrics_table,
    evaluate_candidates,
    fmt_pf,
    fmt_value,
    rank_key,
    serialize_alpha,
    stage_leaders,
)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "candidate_progress.jsonl"
    progress_path.write_text("", encoding="utf-8")
    started = time.time()

    round4 = read_json(Path(args.round4_config))
    round5 = read_json(Path(args.round5_config))
    round4_run = evaluate_strategy("nqdtc", round4, args.data_end)
    round5_run = evaluate_strategy("nqdtc", round5, args.data_end)
    candidates = build_refine_candidates(round5)
    print(
        f"[alpha-refine] round5 IS={round5_run.is_metrics.net_r:+.2f}R PF={fmt_pf(round5_run.is_metrics.profit_factor)} "
        f"OOS={round5_run.oos_metrics.net_r:+.2f}R PF={fmt_pf(round5_run.oos_metrics.profit_factor)}; "
        f"evaluating {len(candidates)} candidates",
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
    full_metrics = compute_full_metrics_table(ranked[: args.full_metrics_top_n], round4, round5, args)
    summary = {
        "run_spec": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_end": args.data_end,
            "round4_config": str(Path(args.round4_config).resolve()),
            "round5_config": str(Path(args.round5_config).resolve()),
            "candidate_count": len(candidates),
            "max_workers": args.max_workers,
            "elapsed_seconds": round(time.time() - started, 2),
            "selection_oos_note": (
                "Focused repair refinement; OOS is used for selection diagnostics and is not a fresh holdout."
            ),
        },
        "round4_baseline": serialize(round4_run),
        "round5_baseline": serialize(round5_run),
        "ranked": [serialize_alpha(ev) for ev in ranked[: args.top_n]],
        "repair_uplifts": [serialize_alpha(ev) for ev in ranked if ev.repair_pass][: args.top_n],
        "strict_uplifts": [serialize_alpha(ev) for ev in ranked if ev.strict_pass][: args.top_n],
        "stage_leaders": stage_leaders(evaluations, args.top_n),
        "full_metrics": full_metrics,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.txt").write_text(format_report(summary), encoding="utf-8")
    print(f"[alpha-refine] complete in {(time.time() - started) / 60.0:.1f} min", flush=True)
    print(f"Output: {output_dir.resolve()}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round4-config", default=str(ROUND4_CONFIG))
    parser.add_argument("--round5-config", default=str(ROUND5_CONFIG))
    parser.add_argument("--data-end", default="2026-05-24")
    parser.add_argument(
        "--output-dir",
        default="backtests/output/momentum/nqdtc/round_5/alpha_refine_20260524",
    )
    parser.add_argument("--max-workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--full-metrics-top-n", type=int, default=32)
    return parser.parse_args()


def build_refine_candidates(current: dict[str, Any]) -> list[RepairCandidate]:
    candidates: list[RepairCandidate] = []
    widths = [90, 100, 110, 125, 140, 150]

    def add(name: str, patch: dict[str, Any], stage: str, source: str) -> None:
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
                    intent="Focused round-5 alpha repair refinement.",
                    source=source,
                )
            )

    mfe_variants = {
        "mfe_fast": {
            "param_overrides.MFE_RATCHET_TIERS_ENABLED": True,
            "param_overrides.MFE_RATCHET_T1_R": 1.75,
            "param_overrides.MFE_RATCHET_T1_LOCK_R": 0.75,
            "param_overrides.MFE_RATCHET_T2_R": 2.75,
            "param_overrides.MFE_RATCHET_T2_LOCK_R": 1.30,
            "param_overrides.MFE_RATCHET_T3_R": 3.75,
            "param_overrides.MFE_RATCHET_T3_LOCK_R": 1.90,
        },
        "mfe_quality": {
            "param_overrides.MFE_RATCHET_TIERS_ENABLED": True,
            "param_overrides.MFE_RATCHET_T1_R": 2.0,
            "param_overrides.MFE_RATCHET_T1_LOCK_R": 0.95,
            "param_overrides.MFE_RATCHET_T2_R": 3.0,
            "param_overrides.MFE_RATCHET_T2_LOCK_R": 1.55,
            "param_overrides.MFE_RATCHET_T3_R": 4.0,
            "param_overrides.MFE_RATCHET_T3_LOCK_R": 2.25,
        },
    }

    for width in widths:
        base = {"param_overrides.MIN_BOX_WIDTH": width}
        add(f"min_box_{width}", base, "min_box", "box_width")
        for tp2 in [2.25, 2.375, 2.50, 2.75]:
            for pct in [0.15, 0.20, 0.25]:
                add(
                    f"min_box_{width}_tp2_{fmt_value(tp2)}_pct_{fmt_value(pct)}",
                    {**base, "param_overrides.TP2_R": tp2, "param_overrides.TP2_PARTIAL_PCT": pct},
                    "min_box_tp2",
                    "box_exit",
                )
        for name, patch in mfe_variants.items():
            add(f"min_box_{width}_{name}", {**base, **patch}, "min_box_mfe", "box_mfe")
            add(
                f"min_box_{width}_{name}_tp2_2p5",
                {**base, **patch, "param_overrides.TP2_R": 2.50, "param_overrides.TP2_PARTIAL_PCT": 0.20},
                "min_box_mfe_tp2",
                "box_mfe_exit",
            )
        for tp1 in [1.55, 1.60, 1.65, 1.70]:
            for pct in [0.35, 0.40, 0.45]:
                add(
                    f"min_box_{width}_tp1_{fmt_value(tp1)}_pct_{fmt_value(pct)}",
                    {**base, "param_overrides.TP1_R": tp1, "param_overrides.TP1_PARTIAL_PCT": pct},
                    "min_box_tp1",
                    "box_exit",
                )
        for offset in [0.236, 0.248, 0.252, 0.264, 0.276]:
            add(
                f"min_box_{width}_c_{fmt_value(offset)}",
                {**base, "param_overrides.C_ENTRY_OFFSET_ATR_STANDARD": offset},
                "min_box_offset",
                "box_entry",
            )
        for mult in [2.10, 2.25, 2.35]:
            for offset in [0.236, 0.252, 0.264]:
                add(
                    f"min_box_{width}_nrm_{fmt_value(mult)}_c_{fmt_value(offset)}",
                    {
                        **base,
                        "param_overrides.BLOCK_NEUTRAL_REGIME": False,
                        "param_overrides.SCORE_NON_RANGE_MULT": mult,
                        "param_overrides.C_ENTRY_OFFSET_ATR_STANDARD": offset,
                    },
                    "min_box_neutral",
                    "box_neutral",
                )
        for gap in [35, 45, 60]:
            add(
                f"min_box_{width}_gap_{gap}",
                {**base, "param_overrides.MIN_INTER_TRADE_GAP_MINUTES": gap},
                "min_box_timing",
                "box_timing",
            )
        for tp2_pct in [0.15, 0.20]:
            add(
                f"min_box_{width}_tp2_2p5_pct_{fmt_value(tp2_pct)}_mfe_quality",
                {
                    **base,
                    **mfe_variants["mfe_quality"],
                    "param_overrides.TP2_R": 2.50,
                    "param_overrides.TP2_PARTIAL_PCT": tp2_pct,
                },
                "min_box_exit_stack",
                "box_exit_stack",
            )

    return dedupe(candidates, current)


def dedupe(candidates: list[RepairCandidate], current: dict[str, Any]) -> list[RepairCandidate]:
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
    lines = [
        "NQDTC Round 5 Alpha Repair Refinement",
        "=" * 100,
        summary["run_spec"]["selection_oos_note"],
        f"Data end: {summary['run_spec']['data_end']}; candidates: {summary['run_spec']['candidate_count']}",
        "",
        "Top Split Rankings",
    ]
    for ev in summary["ranked"][:20]:
        im = ev["run"]["is_metrics"]
        om = ev["run"]["oos_metrics"]
        lines.append(
            f"  {ev['candidate']['name']}: score={ev['score']:+.3f} "
            f"IS n={im['total_trades']} netR={im['net_r']:+.2f} PF={fmt_pf(im['profit_factor'])} "
            f"OOS n={om['total_trades']} netR={om['net_r']:+.2f} PF={fmt_pf(om['profit_factor'])} "
            f"reasons={ev['reasons']}"
        )
    lines.extend(["", "Full Metrics"])
    for row in summary["full_metrics"][:30]:
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


if __name__ == "__main__":
    main()
