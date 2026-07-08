"""Replay an existing swing portfolio-synergy round under current code."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from backtests.shared.auto.round_manager import RoundManager
from backtests.swing.auto.portfolio_synergy.run_latest_two_rounds import _json_default
from backtests.swing.auto.portfolio_synergy.run_phase_auto_from_latest import (
    PortfolioSynergyPhasePlugin,
    _format_final_diagnostics,
    _source_artifact_records,
)


MATERIAL_DRIFT_THRESHOLDS = {
    "profit_factor_rel": 0.005,
    "net_return_pct_abs": 0.5,
    "max_drawdown_pct_abs": 0.25,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="backtests/output/swing/portfolio_synergy/round_3")
    parser.add_argument("--data-dir", default="backtests/swing/data/raw")
    parser.add_argument("--equity", type=float, default=50_000.0)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Persist current recompute metrics even when material drift is detected.",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    round_num = _round_num(run_dir)
    manager = RoundManager("swing", "portfolio_synergy")
    mutations = _load_mutations(run_dir / "optimized_config.json")
    saved_summary = _load_json(run_dir / "run_summary.json")
    saved_metrics = dict(saved_summary.get("final_metrics") or {})
    run_spec = _load_json(run_dir / "run_spec.json") if (run_dir / "run_spec.json").exists() else {}

    plugin = PortfolioSynergyPhasePlugin(
        Path(args.data_dir),
        initial_equity=float(args.equity),
        max_workers=int(args.max_workers),
        initial_mutations=mutations,
        base_source=str(run_spec.get("baseline_source") or saved_summary.get("config_source") or run_dir),
    )
    current_metrics = plugin.compute_final_metrics(mutations)
    provenance = plugin.build_provenance()
    drift = _classify_material_drift(saved_metrics, current_metrics)
    should_update_metrics = drift["status"] == "no_material_drift" or args.refresh

    diagnostics_path = manager.diagnostics_path(run_dir)
    diagnostics_path.write_text(_format_final_diagnostics(current_metrics), encoding="utf-8")

    summary_metrics = current_metrics if should_update_metrics else saved_metrics
    manager.write_run_summary(
        run_dir,
        mutations,
        summary_metrics,
        list(saved_summary.get("completed_phases") or []),
        round_num=round_num,
        source_diagnostics=diagnostics_path,
        provenance=provenance,
        provenance_status="complete",
    )
    summary = _load_json(manager.run_summary_path(run_dir))
    summary.update(
        {
            "diagnostics_refresh": {
                "status": drift["status"],
                "updated_saved_metrics": should_update_metrics,
                "material_thresholds": MATERIAL_DRIFT_THRESHOLDS,
                "deltas": drift["deltas"],
            },
            "selection_status": (
                "current_code_replayed_without_material_drift"
                if drift["status"] == "no_material_drift"
                else "stale_selected_after_current_code_replay"
            ),
        }
    )
    _write_json(manager.run_summary_path(run_dir), summary)
    diagnostics_summary_path = manager.diagnostics_summary_path(run_dir)
    diagnostics_summary = _load_json(diagnostics_summary_path) if diagnostics_summary_path.exists() else {}
    diagnostics_summary.update(
        {
            "family": "swing",
            "strategy": "portfolio_synergy",
            "round": round_num,
            "generated_at_utc": summary.get("generated_at_utc"),
            "initial_equity": float(args.equity),
            "data_dir": str(Path(args.data_dir)),
            "headline_metrics": summary.get("headline_metrics", {}),
            "final_metrics": summary.get("final_metrics", {}),
            "provenance": provenance.to_dict(),
            "provenance_status": "complete",
            "selection_status": summary["selection_status"],
            "diagnostics_refresh": summary["diagnostics_refresh"],
            "source_strategy_artifacts": _source_artifact_records(),
        }
    )
    _write_json(diagnostics_summary_path, diagnostics_summary)
    manager.append_to_manifest(round_num, mutations, summary_metrics, provenance=provenance, provenance_status="complete")

    spec = dict(run_spec)
    spec["provenance"] = provenance.to_dict()
    spec["provenance_status"] = "complete"
    spec["diagnostics_refresh"] = summary["diagnostics_refresh"]
    spec["selection_status"] = summary["selection_status"]
    spec["source_strategy_artifacts"] = diagnostics_summary["source_strategy_artifacts"]
    _write_json(manager.run_spec_path(run_dir), spec)

    print(f"Swing portfolio diagnostics replay complete: {run_dir}")
    print(f"Drift status: {drift['status']}")
    print(
        "Current metrics: "
        f"trades={current_metrics.get('total_trades')}, "
        f"return={current_metrics.get('net_return_pct', 0):+.4f}%, "
        f"pf={current_metrics.get('profit_factor', 0):.6f}, "
        f"dd={current_metrics.get('max_drawdown_pct', 0):.4f}%"
    )


def _classify_material_drift(saved: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    deltas: dict[str, dict[str, Any]] = {}
    count_keys = ("total_trades", "portfolio_rule_block_events", "portfolio_rule_sizing_events")
    for key in count_keys:
        saved_value = saved.get(key)
        current_value = current.get(key)
        if saved_value is None or current_value is None:
            continue
        if int(saved_value) != int(current_value):
            deltas[key] = {"saved": saved_value, "current": current_value}

    _add_abs_delta(deltas, saved, current, "net_return_pct", MATERIAL_DRIFT_THRESHOLDS["net_return_pct_abs"])
    _add_abs_delta(deltas, saved, current, "max_drawdown_pct", MATERIAL_DRIFT_THRESHOLDS["max_drawdown_pct_abs"])
    _add_rel_delta(deltas, saved, current, "profit_factor", MATERIAL_DRIFT_THRESHOLDS["profit_factor_rel"])
    return {"status": "selection_drift" if deltas else "no_material_drift", "deltas": deltas}


def _add_abs_delta(
    deltas: dict[str, dict[str, Any]],
    saved: dict[str, Any],
    current: dict[str, Any],
    key: str,
    tolerance: float,
) -> None:
    saved_value = saved.get(key)
    current_value = current.get(key)
    if saved_value is None or current_value is None:
        return
    if abs(float(saved_value) - float(current_value)) > tolerance:
        deltas[key] = {"saved": saved_value, "current": current_value}


def _add_rel_delta(
    deltas: dict[str, dict[str, Any]],
    saved: dict[str, Any],
    current: dict[str, Any],
    key: str,
    tolerance: float,
) -> None:
    saved_value = saved.get(key)
    current_value = current.get(key)
    if saved_value is None or current_value is None:
        return
    base = max(abs(float(saved_value)), 1e-12)
    if abs(float(saved_value) - float(current_value)) / base > tolerance:
        deltas[key] = {"saved": saved_value, "current": current_value}


def _load_mutations(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if isinstance(payload.get("mutations"), dict):
        return dict(payload["mutations"])
    if isinstance(payload.get("cumulative_mutations"), dict):
        return dict(payload["cumulative_mutations"])
    return dict(payload)


def _round_num(run_dir: Path) -> int:
    stem = run_dir.name
    if stem.startswith("round_"):
        return int(stem.split("_", 1)[1])
    raise ValueError(f"Cannot infer round number from {run_dir}")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
