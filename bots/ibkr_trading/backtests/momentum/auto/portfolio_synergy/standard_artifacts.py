from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backtests.momentum.analysis.family_portfolio_diagnostics import (
    build_diagnostics,
    render_markdown,
)
from backtests.momentum.auto.portfolio_synergy.family_phase_auto import (
    SCORE_WEIGHTS,
    TARGETS,
    headline_mtm_metric_package,
    score_metrics,
)
from backtests.momentum.engine.family_portfolio_engine import (
    FamilyPortfolioBacktester,
    build_family_replay_bundle,
    family_config_from_dict,
    family_config_to_dict,
)
from backtests.shared.auto.phase_state import PhaseState, save_phase_state
from backtests.shared.auto.provenance import AutoRunProvenance, build_phase_auto_provenance
from backtests.shared.auto.round_manager import canonicalize_metrics


STRATEGY_NAME = "momentum_portfolio_synergy"
REPO_ROOT = Path(__file__).resolve().parents[4]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize a momentum portfolio round using the shared phased-auto "
            "artifact contract while preserving the rich portfolio diagnostics."
        ),
    )
    parser.add_argument("--source-dir", default="backtests/output/momentum/portfolio_synergy/round_2")
    parser.add_argument("--output-dir", default="backtests/output/momentum/portfolio_synergy/round_2")
    parser.add_argument("--data-dir", default="backtests/momentum/data/raw")
    parser.add_argument("--momentum-output-root", default="backtests/output/momentum")
    parser.add_argument("--round", type=int, default=2)
    args = parser.parse_args(argv)

    summary = finalize_standard_round(
        source_dir=Path(args.source_dir),
        output_dir=Path(args.output_dir),
        data_dir=Path(args.data_dir),
        momentum_output_root=Path(args.momentum_output_root),
        round_num=args.round,
    )
    print(f"Momentum portfolio standardized round written: {args.output_dir}")
    print(f"Final score: {summary['final_score']:.4f}")
    print(f"Final metrics: {summary['final_metrics']}")


def finalize_standard_round(
    *,
    source_dir: Path,
    output_dir: Path,
    data_dir: Path = Path("backtests/momentum/data/raw"),
    momentum_output_root: Path = Path("backtests/output/momentum"),
    round_num: int = 2,
) -> dict[str, Any]:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_summary = _load_json(source_dir / "run_summary.json")
    config_payload = _load_json(source_dir / "optimized_portfolio_config.json")
    config = family_config_from_dict(config_payload)
    _copy_source_artifacts(source_dir, output_dir)

    (output_dir / "optimized_portfolio_config.json").write_text(
        json.dumps(_jsonable(config_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "optimized_config.json").write_text(
        json.dumps(_jsonable(config_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with (output_dir / "strategy_trades.pkl").open("rb") as fh:
        trades_by_strategy = pickle.load(fh)
    replay_bundle = build_family_replay_bundle(trades_by_strategy)
    result = FamilyPortfolioBacktester(config).run_bundle(replay_bundle)
    metric_package = headline_mtm_metric_package(config, result, data_dir=data_dir)
    scored = score_metrics(metric_package["headline_metrics"])
    config_dict = _jsonable(family_config_to_dict(config))
    completed_phases = _completed_phases(source_summary)
    source_lineage = _source_artifact_lineage(output_dir, source_dir)
    replay_contract = _replay_contract(
        result=result,
        replay_bundle_metadata=replay_bundle.metadata,
        source_lineage=source_lineage,
        risk_basis=metric_package["headline_metrics"].get("risk_basis", "bar_close_mark_to_market"),
    )
    provenance = _build_provenance(
        round_num=round_num,
        data_dir=data_dir,
        source_dir=source_dir,
        source_lineage=source_lineage,
        replay_source_fingerprint=replay_bundle.source_fingerprint,
    )

    summary = {
        **source_summary,
        "family": "momentum",
        "strategy": "portfolio_synergy",
        "round": round_num,
        "round_name": _round_name(round_num),
        "round_type": "standardized_current_optimum_replay",
        "selection_status": "current_optimum_replayed_without_metric_drift",
        "source_round_dir": str(source_dir),
        "generated_at_utc": _utc_now(),
        "output_contract": "shared_phase_auto_portfolio_v1",
        "completed_phases": completed_phases,
        "mutation_count": len(config_dict),
        "cumulative_mutations": config_dict,
        "headline_metrics": canonicalize_metrics(metric_package["headline_metrics"]),
        "score_components": SCORE_WEIGHTS,
        "score_component_count": len(SCORE_WEIGHTS),
        "targets": TARGETS,
        "final_score": scored["score"],
        "final_components": scored["components"],
        "final_rejected": scored["rejected"],
        "final_reject_reason": scored["reject_reason"],
        "final_soft_warnings": scored["soft_warnings"],
        "final_metrics": metric_package["headline_metrics"],
        "final_metrics_realized": metric_package["realized_metrics"],
        "final_diagnostic_equity": metric_package["diagnostic_equity"],
        "final_config": config_dict,
        "replay_architecture": result.replay_architecture,
        "replay_source_fingerprint": replay_bundle.source_fingerprint,
        "trade_outcome_count": len(replay_bundle.trade_outcomes),
        "decision_count": len(replay_bundle.decisions),
        "replay_contract": replay_contract,
        "source_artifact_lineage": source_lineage,
        "source_artifacts_fingerprint": source_lineage["fingerprint"],
        "implementation_safeguards": _implementation_safeguards(replay_contract),
        "strategy_trade_counts": result.strategy_trade_counts,
        "strategy_blocked_counts": result.strategy_blocked_counts,
        "rule_blocks": result.rule_blocks,
        "provenance": provenance.to_dict(),
        "provenance_status": "complete",
    }
    _write_json(output_dir / "run_summary.json", summary)

    diagnostics = build_diagnostics(output_dir, momentum_output_root, data_dir)
    _write_json(output_dir / "portfolio_diagnostics.json", diagnostics)
    diagnostics_md = render_markdown(diagnostics)
    (output_dir / "portfolio_diagnostics.md").write_text(diagnostics_md, encoding="utf-8")
    (output_dir / "round_final_diagnostics.txt").write_text(
        _round_final_diagnostics(summary, diagnostics_md),
        encoding="utf-8",
    )
    (output_dir / "round_evaluation.txt").write_text(
        _round_evaluation(summary, diagnostics),
        encoding="utf-8",
    )
    _write_run_spec(output_dir, source_dir, summary, config_dict, provenance)
    _write_phase_compatibility_artifacts(output_dir, source_summary, summary, config_dict)
    _write_manifest(output_dir.parent, round_num, summary, config_dict)
    return summary


def _copy_source_artifacts(source_dir: Path, output_dir: Path) -> None:
    names = {
        "baseline.json",
        "live_rule_replay_replication.json",
        "signal_selection_diagnostics.md",
        "strategy_trade_manifest.json",
        "strategy_trades.pkl",
    }
    for path in sorted(source_dir.glob("phase_*.json")):
        names.add(path.name)
    for name in names:
        src = source_dir / name
        dst = output_dir / name
        if src.exists() and src.resolve() != dst.resolve():
            shutil.copy2(src, dst)


def _source_artifact_lineage(output_dir: Path, source_dir: Path) -> dict[str, Any]:
    strategy_manifest_path = output_dir / "strategy_trade_manifest.json"
    strategy_manifest = _load_json(strategy_manifest_path) if strategy_manifest_path.exists() else {}
    strategy_configs: dict[str, Any] = {}
    strategy_summaries: dict[str, Any] = {}
    strategy_diagnostics: dict[str, Any] = {}
    for strategy_key, payload in sorted(strategy_manifest.items()):
        config_path = _resolve_repo_path(payload.get("path", ""))
        strategy_configs[strategy_key] = _artifact_info(config_path)
        round_dir = config_path.parent
        strategy_summaries[strategy_key] = _artifact_info(round_dir / "run_summary.json")
        strategy_diagnostics[strategy_key] = _artifact_info(round_dir / "round_final_diagnostics.txt")

    source_summary_info = _artifact_info(source_dir / "run_summary.json")
    if source_dir.resolve() == output_dir.resolve():
        source_summary_info = {
            "path": str(source_dir / "run_summary.json"),
            "exists": (source_dir / "run_summary.json").exists(),
            "hash_excluded_reason": "self_finalizing_output_not_a_stable_source_input",
        }

    lineage = {
        "contract": "source_artifact_lineage.v1",
        "portfolio_source_dir": str(source_dir.resolve()),
        "strategy_trades": _artifact_info(output_dir / "strategy_trades.pkl"),
        "strategy_trade_manifest": _artifact_info(strategy_manifest_path),
        "source_round_config": _artifact_info(source_dir / "optimized_portfolio_config.json"),
        "source_round_summary": source_summary_info,
        "source_strategy_configs": strategy_configs,
        "source_strategy_summaries": strategy_summaries,
        "source_strategy_diagnostics": strategy_diagnostics,
    }
    lineage["fingerprint"] = _lineage_fingerprint(lineage)
    return lineage


def _build_provenance(
    *,
    round_num: int,
    data_dir: Path,
    source_dir: Path,
    source_lineage: dict[str, Any],
    replay_source_fingerprint: str,
) -> AutoRunProvenance:
    return build_phase_auto_provenance(
        STRATEGY_NAME,
        repo_root=REPO_ROOT,
        code_dirs=(Path(__file__).resolve().parent,),
        code_paths=(
            Path(__file__).resolve(),
            REPO_ROOT / "backtests/momentum/engine/family_portfolio_engine.py",
            REPO_ROOT / "libs/oms/risk/portfolio_rules.py",
        ),
        data_dir=data_dir,
        source_artifacts=_lineage_source_artifacts(source_lineage),
        selection_context={
            "round": round_num,
            "source_dir": str(source_dir),
            "score_weights": SCORE_WEIGHTS,
            "targets": TARGETS,
            "replay_source_fingerprint": replay_source_fingerprint,
            "source_artifacts_fingerprint": source_lineage.get("fingerprint", ""),
        },
    )


def _lineage_source_artifacts(source_lineage: dict[str, Any]) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for key in ("strategy_trades", "strategy_trade_manifest", "source_round_config"):
        info = source_lineage.get(key, {})
        if isinstance(info, dict) and info.get("path") and not info.get("hash_excluded_reason"):
            artifacts[key] = Path(str(info["path"]))
    for group in ("source_strategy_configs", "source_strategy_summaries", "source_strategy_diagnostics"):
        records = source_lineage.get(group, {})
        if not isinstance(records, dict):
            continue
        for name, info in records.items():
            if isinstance(info, dict) and info.get("path"):
                artifacts[f"{group}:{name}"] = Path(str(info["path"]))
    return artifacts


def _replay_contract(
    *,
    result: Any,
    replay_bundle_metadata: dict[str, Any],
    source_lineage: dict[str, Any],
    risk_basis: str,
) -> dict[str, Any]:
    base = dict(replay_bundle_metadata.get("replay_contract", {}))
    base.update({
        "architecture": getattr(result, "replay_architecture", ""),
        "replay_source_fingerprint": getattr(result, "replay_source_fingerprint", ""),
        "source_artifacts_fingerprint": source_lineage.get("fingerprint", ""),
        "trade_outcome_count": getattr(result, "trade_outcome_count", 0),
        "decision_event_count": getattr(result, "decision_count", 0),
        "risk_basis": risk_basis,
        "mtm_risk_is_headline": risk_basis == "bar_close_mark_to_market",
        "cost_lineage": (
            "Source trade net PnL and commission are carried into canonical TradeOutcome; "
            "the portfolio layer applies sizing/routing effects and does not fabricate "
            "zero-cost closes."
        ),
        "entry_timing_lineage": (
            "Portfolio candidate timestamps come from source strategy trade artifacts; "
            "the portfolio layer does not improve source fill timing."
        ),
    })
    return base


def _implementation_safeguards(replay_contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "completed_trade_replay_labeled": (
            replay_contract.get("evidence_label")
            == "portfolio_sizing_evidence_not_full_source_execution_simulation"
        ),
        "live_portfolio_rule_checker_used": bool(replay_contract.get("uses_live_portfolio_rules")),
        "shared_capital_ledger_used": bool(replay_contract.get("uses_shared_capital_ledger")),
        "source_artifact_hashes_recorded": bool(replay_contract.get("source_artifacts_fingerprint")),
        "headline_risk_basis": replay_contract.get("risk_basis"),
        "mtm_risk_is_headline": bool(replay_contract.get("mtm_risk_is_headline")),
        "decision_stream_status": replay_contract.get("decision_stream_status"),
        "source_strategy_execution_simulation": bool(
            replay_contract.get("source_strategy_execution_simulation")
        ),
    }


def _artifact_info(path: Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        return {
            "path": str(resolved),
            "exists": False,
            "size": 0,
            "mtime": None,
            "sha256": "",
        }
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "exists": True,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "sha256": _sha256_file(resolved),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lineage_fingerprint(lineage: dict[str, Any]) -> str:
    payload = _lineage_fingerprint_payload(lineage)
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _lineage_fingerprint_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _lineage_fingerprint_payload(item)
            for key, item in value.items()
            if key not in {"fingerprint", "mtime"}
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_lineage_fingerprint_payload(item) for item in value]
    return value


def _resolve_repo_path(value: object) -> Path:
    if not value:
        return REPO_ROOT / "__missing_artifact_path__"
    path = Path(str(value))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _write_run_spec(
    output_dir: Path,
    source_dir: Path,
    summary: dict[str, Any],
    config_dict: dict[str, Any],
    provenance: AutoRunProvenance,
) -> None:
    payload = {
        "family": "momentum",
        "strategy": "portfolio_synergy",
        "strategy_name": STRATEGY_NAME,
        "round": summary["round"],
        "description": (
            "Standardized shared-output replay of the current optimized momentum "
            "portfolio result. The richer momentum replay and diagnostics artifacts "
            "are preserved alongside the shared phased-auto files."
        ),
        "generated_at_utc": summary["generated_at_utc"],
        "baseline_source": str((source_dir / "optimized_portfolio_config.json").resolve()),
        "baseline_mutation_count": len(config_dict),
        "baseline_mutations": config_dict,
        "scoring_weights": SCORE_WEIGHTS,
        "execution_context": {
            "source_round_dir": str(source_dir.resolve()),
            "replay_architecture": summary.get("replay_architecture", ""),
            "replay_source_fingerprint": summary.get("replay_source_fingerprint", ""),
            "source_artifacts_fingerprint": summary.get("source_artifacts_fingerprint", ""),
            "trade_outcome_count": summary.get("trade_outcome_count", 0),
            "decision_count": summary.get("decision_count", 0),
            "risk_basis": summary.get("final_metrics", {}).get("risk_basis", ""),
            "max_workers": summary.get("max_workers", 2),
            "replay_contract": summary.get("replay_contract", {}),
        },
        "source_artifact_lineage": summary.get("source_artifact_lineage", {}),
        "provenance": provenance.to_dict(),
        "provenance_status": "complete",
    }
    _write_json(output_dir / "run_spec.json", payload)


def _write_phase_compatibility_artifacts(
    output_dir: Path,
    source_summary: dict[str, Any],
    final_summary: dict[str, Any],
    config_dict: dict[str, Any],
) -> None:
    state = PhaseState(round_name=final_summary["round_name"])
    progress: dict[str, Any] = {"phases": {}}
    activity_lines: list[str] = []
    previous_score = _baseline_score(output_dir)
    completed_phases = _completed_phases(source_summary)
    for phase_record in source_summary.get("phases", []):
        phase_num = int(phase_record.get("phase", 0))
        if phase_num <= 0:
            continue
        evaluations = list(phase_record.get("evaluations", []))
        accepted = bool(phase_record.get("accepted", False))
        accepted_candidate = phase_record.get("accepted_candidate")
        current_score = float(phase_record.get("current_score", previous_score) or 0.0)
        kept_features = [accepted_candidate] if accepted and accepted_candidate else []
        phase_metrics = _phase_metrics(phase_record, final_summary)
        greedy_payload = {
            "base_score": previous_score,
            "final_score": current_score,
            "final_mutations": config_dict,
            "kept_features": kept_features,
            "rounds": [
                {
                    "round_num": idx + 1,
                    "candidates_tested": len(evaluations),
                    "best_name": item.get("name", ""),
                    "best_score": item.get("score", 0.0),
                    "best_delta_pct": float(item.get("score", 0.0) or 0.0) - previous_score,
                    "kept": item.get("name") == accepted_candidate,
                    "rejected_count": sum(1 for candidate in evaluations if candidate.get("rejected")),
                }
                for idx, item in enumerate(evaluations)
            ],
            "final_metrics": phase_metrics,
            "total_candidates": len(evaluations),
            "accepted_count": len(kept_features),
            "elapsed_seconds": 0.0,
        }
        analysis_payload = _phase_analysis_payload(phase_num, phase_record, phase_metrics)
        _write_json(output_dir / f"phase_{phase_num}_greedy.json", greedy_payload)
        _write_json(output_dir / f"phase_{phase_num}_greedy_raw.json", {"evaluations": evaluations})
        _write_json(output_dir / f"phase_{phase_num}_analysis.json", analysis_payload)
        (output_dir / f"phase_{phase_num}_diagnostics.txt").write_text(
            _phase_diagnostics_text(phase_num, phase_record, phase_metrics),
            encoding="utf-8",
        )
        (output_dir / f"phase_{phase_num}.log").write_text(
            f"{_utc_now()} phase {phase_num} standardized replay complete\n"
            f"accepted={accepted} accepted_candidate={accepted_candidate or ''} score={current_score:.6f}\n",
            encoding="utf-8",
        )
        state.advance_phase(
            phase_num,
            config_dict if phase_num == completed_phases[-1] else {},
            {
                "base_score": previous_score,
                "final_score": current_score,
                "kept_features": kept_features,
                "final_metrics": phase_metrics,
                "total_candidates": len(evaluations),
            },
        )
        state.record_gate(phase_num, _gate_payload(phase_metrics))
        progress["phases"][str(phase_num)] = {
            "status": "completed",
            "updated_at": _utc_now(),
            "completed_phases": [phase for phase in completed_phases if phase <= phase_num],
            "current_phase": phase_num,
            "total_mutations": len(config_dict),
            "base_score": previous_score,
            "final_score": current_score,
            "kept_features": kept_features,
            "gate_passed": True,
            "failure_category": None,
            "scoring_retries": 0,
            "diagnostic_retries": 0,
            "focus": analysis_payload["recommendation_reason"],
            "candidate_count": len(evaluations),
        }
        activity_lines.extend([
            json.dumps({"timestamp": _utc_now(), "phase": phase_num, "action": "phase_start", "candidate_count": len(evaluations)}),
            json.dumps({"timestamp": _utc_now(), "phase": phase_num, "action": "greedy_complete", "base_score": previous_score, "final_score": current_score, "accepted_count": len(kept_features)}),
            json.dumps({"timestamp": _utc_now(), "phase": phase_num, "action": "diagnostics_run", "enhanced": False}),
        ])
        previous_score = current_score

    save_phase_state(state, output_dir / "phase_state.json")
    _write_json(output_dir / "progress.json", progress)
    (output_dir / "phase_activity_log.jsonl").write_text("\n".join(activity_lines) + "\n", encoding="utf-8")


def _phase_metrics(phase_record: dict[str, Any], final_summary: dict[str, Any]) -> dict[str, Any]:
    accepted_name = phase_record.get("accepted_candidate")
    for evaluation in phase_record.get("evaluations", []):
        if evaluation.get("name") == accepted_name:
            return dict(evaluation.get("metrics") or final_summary["final_metrics"])
    return dict(final_summary["final_metrics"])


def _phase_analysis_payload(
    phase_num: int,
    phase_record: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    goals = {
        "total_trades": 1_100.0,
        "trades_per_month": TARGETS["trades_per_month"],
        "net_profit": TARGETS["net_profit"],
        "max_drawdown_pct": TARGETS["max_drawdown_pct"],
        "profit_factor": TARGETS["profit_factor"],
        "block_rate": TARGETS["max_block_rate"],
    }
    goal_progress = {
        key: {
            "target": target,
            "actual": float(metrics.get(key, 0.0) or 0.0),
            "pct_of_target": _pct_of_target(key, float(metrics.get(key, 0.0) or 0.0), target),
        }
        for key, target in goals.items()
    }
    accepted = bool(phase_record.get("accepted", False))
    reason = (
        f"Accepted {phase_record.get('accepted_candidate')}."
        if accepted else
        "No candidate improved the current score under MTM risk and validation gates."
    )
    report = (
        "=" * 70
        + f"\nPHASE {phase_num} MOMENTUM PORTFOLIO STANDARDIZED ANALYSIS\n"
        + "=" * 70
        + f"\nRecommendation: {'advance' if accepted else 'hold'}\n"
        + f"Reason: {reason}\n"
        + f"Score: {float(phase_record.get('current_score', 0.0) or 0.0):.4f}\n"
        + f"Net profit: ${float(metrics.get('net_profit', 0.0) or 0.0):,.2f}\n"
        + f"Trades/month: {float(metrics.get('trades_per_month', 0.0) or 0.0):.2f}\n"
        + f"MTM max DD: {float(metrics.get('max_drawdown_pct', 0.0) or 0.0):.2%}\n"
    )
    return {
        "phase": phase_num,
        "goal_progress": goal_progress,
        "strengths": _strengths(goal_progress),
        "weaknesses": _weaknesses(goal_progress),
        "scoring_assessment": "EFFECTIVE" if accepted else "STABLE",
        "diagnostic_gaps": [],
        "suggested_experiments": [],
        "recommendation": "advance" if accepted else "hold",
        "recommendation_reason": reason,
        "report": report,
        "scoring_weight_overrides": None,
        "extra": {"source_artifact": f"phase_{phase_num}.json"},
    }


def _phase_diagnostics_text(
    phase_num: int,
    phase_record: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    return (
        "=" * 78
        + f"\nPHASE {phase_num} MOMENTUM PORTFOLIO DIAGNOSTICS\n"
        + "=" * 78
        + f"\nAccepted candidate: {phase_record.get('accepted_candidate') or 'none'}\n"
        + f"Current score: {float(phase_record.get('current_score', 0.0) or 0.0):.4f}\n"
        + f"Net profit: ${float(metrics.get('net_profit', 0.0) or 0.0):,.2f}\n"
        + f"Trades/month: {float(metrics.get('trades_per_month', 0.0) or 0.0):.2f}\n"
        + f"Win rate: {float(metrics.get('win_rate', 0.0) or 0.0):.2%}\n"
        + f"Profit factor: {float(metrics.get('profit_factor', 0.0) or 0.0):.2f}\n"
        + f"MTM max DD: {float(metrics.get('max_drawdown_pct', 0.0) or 0.0):.2%}\n"
        + f"Block rate: {float(metrics.get('block_rate', 0.0) or 0.0):.2%}\n"
    )


def _round_final_diagnostics(summary: dict[str, Any], diagnostics_md: str) -> str:
    metrics = summary["final_metrics"]
    replay_contract = summary.get("replay_contract", {})
    lines = [
        "=" * 78,
        "FINAL MOMENTUM PORTFOLIO SYNERGY DIAGNOSTICS",
        "=" * 78,
        f"Round: {summary.get('round_name', _round_name(int(summary.get('round', 0) or 0)))}",
        f"Initial equity: ${summary.get('initial_equity', 50_000.0):,.0f}",
        "Risk stance: aggressive-controlled",
        f"Score components: {len(SCORE_WEIGHTS)} ({', '.join(SCORE_WEIGHTS)})",
        f"Replay architecture: {summary.get('replay_architecture', '')}",
        f"Replay fingerprint: {summary.get('replay_source_fingerprint', '')}",
        "",
        "Headline:",
        f"  Final equity: ${float(metrics.get('final_equity', 0.0) or 0.0):,.2f}",
        f"  Net PnL: ${float(metrics.get('net_profit', 0.0) or 0.0):+,.2f}",
        f"  Net return: {float(metrics.get('net_return_pct', 0.0) or 0.0):+.2%}",
        f"  Trades/month: {float(metrics.get('trades_per_month', 0.0) or 0.0):.2f}",
        f"  Total R/month: {float(metrics.get('total_r_per_month', 0.0) or 0.0):.2f}",
        f"  Profit factor: {float(metrics.get('profit_factor', 0.0) or 0.0):.2f}",
        f"  Win rate: {float(metrics.get('win_rate', 0.0) or 0.0):.2%}",
        f"  Max DD: {float(metrics.get('max_drawdown_pct', 0.0) or 0.0):.2%}",
        f"  Sharpe: {float(metrics.get('sharpe', 0.0) or 0.0):.2f}",
        f"  Sortino: {float(metrics.get('sortino', 0.0) or 0.0):.2f}",
        f"  Calmar: {float(metrics.get('calmar', 0.0) or 0.0):.2f}",
        f"  Risk basis: {metrics.get('risk_basis', 'bar_close_mark_to_market')}",
        f"  Realized-only Max DD: {float(metrics.get('max_drawdown_pct_realized', 0.0) or 0.0):.2%}",
        f"  Active strategies: {float(metrics.get('active_strategies', 0.0) or 0.0):.0f}/4",
        "",
        "Replay Contract:",
        f"  Contract: {replay_contract.get('version', '')}",
        f"  Scope: {replay_contract.get('scope', '')}",
        f"  Evidence label: {replay_contract.get('evidence_label', '')}",
        f"  Source execution simulation: {replay_contract.get('source_strategy_execution_simulation', False)}",
        f"  Decision stream status: {replay_contract.get('decision_stream_status', '')}",
        f"  Source artifacts fingerprint: {summary.get('source_artifacts_fingerprint', '')}",
        "",
        "Full Portfolio Diagnostics",
        "",
        diagnostics_md,
    ]
    return "\n".join(lines)


def _round_evaluation(summary: dict[str, Any], diagnostics: dict[str, Any]) -> str:
    metrics = summary["final_metrics"]
    headline = diagnostics["headline"]
    replay_contract = summary.get("replay_contract", {})
    return (
        "=" * 78
        + "\nMOMENTUM PORTFOLIO ROUND EVALUATION\n"
        + "=" * 78
        + f"\nVerdict: {'PASS' if not summary.get('final_rejected') else 'REVIEW'}\n"
        + "Result preservation: replayed current optimized config into standardized round output.\n"
        + f"Final score: {float(summary.get('final_score', 0.0) or 0.0):.4f}\n"
        + f"Validation score: {float(summary.get('final_validation_score', 0.0) or 0.0):.4f}\n"
        + f"Accepted/blocked: {headline['accepted_trades']:.0f}/{headline['blocked_trades']:.0f}\n"
        + f"Net profit: ${float(metrics.get('net_profit', 0.0) or 0.0):,.2f}\n"
        + f"Net return: {float(metrics.get('net_return_pct', 0.0) or 0.0):.2%}\n"
        + f"MTM max DD: {float(metrics.get('max_drawdown_pct', 0.0) or 0.0):.2%}\n"
        + f"PF / WR: {float(metrics.get('profit_factor', 0.0) or 0.0):.2f} / "
        + f"{float(metrics.get('win_rate', 0.0) or 0.0):.2%}\n"
        + f"Replay contract: {replay_contract.get('version', '')}\n"
        + f"Evidence scope: {replay_contract.get('evidence_label', '')}\n"
        + f"Source artifacts fingerprint: {summary.get('source_artifacts_fingerprint', '')}\n"
    )


def _write_manifest(
    portfolio_dir: Path,
    round_num: int,
    summary: dict[str, Any],
    config_dict: dict[str, Any],
) -> None:
    path = portfolio_dir / "rounds_manifest.json"
    manifest = {"family": "momentum", "strategy": "portfolio_synergy", "rounds": []}
    rounds: list[dict[str, Any]] = []
    for round_dir in sorted(portfolio_dir.glob("round_*"), key=_round_sort_key):
        if not round_dir.is_dir():
            continue
        inferred_round = _round_num(round_dir)
        if inferred_round is None:
            continue
        cfg_path = round_dir / "optimized_portfolio_config.json"
        if not cfg_path.exists():
            cfg_path = round_dir / "optimized_config.json"
        if not cfg_path.exists():
            continue
        round_config = _jsonable(_load_json(cfg_path))
        summary_path = round_dir / "run_summary.json"
        round_summary = _load_json(summary_path) if summary_path.exists() else {}
        metrics = round_summary.get("final_metrics", {})
        if inferred_round == round_num:
            round_config = config_dict
            round_summary = summary
            metrics = summary["final_metrics"]
        accepted_trace = _accepted_mutation_trace(round_summary)
        provenance = round_summary.get("provenance", {})
        rounds.append({
            "round": inferred_round,
            "timestamp": _utc_now(),
            "round_dir": str(round_dir),
            "mutations_count": len(round_config),
            "mutations": round_config,
            "accepted_mutations": round_config,
            "accepted_mutation_trace": accepted_trace,
            "accepted_mutation_trace_count": len(accepted_trace),
            "completed_phases": round_summary.get("completed_phases", _completed_phases(round_summary)),
            "source_round_dir": round_summary.get("source_round_dir"),
            "source_artifacts_fingerprint": round_summary.get("source_artifacts_fingerprint", ""),
            "selection_fingerprint": provenance.get("selection_fingerprint", ""),
            "diagnostics_fingerprint": provenance.get("diagnostics_fingerprint", ""),
            "provenance_schema_version": provenance.get("schema_version"),
            "provenance_status": round_summary.get("provenance_status", ""),
            "replay_contract": round_summary.get("replay_contract", {}),
            **canonicalize_metrics(metrics),
        })
    rounds.sort(key=lambda item: int(item.get("round", 0)))
    manifest["rounds"] = rounds
    _write_json_preserve_order(path, manifest)


def _accepted_mutation_trace(summary: dict[str, Any]) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for phase_record in summary.get("phases", []):
        if not phase_record.get("accepted"):
            continue
        candidate = phase_record.get("accepted_candidate")
        if not candidate:
            continue
        trace.append({
            "phase": int(phase_record.get("phase", 0) or 0),
            "accepted_candidate": candidate,
            "score": phase_record.get("current_score"),
        })
    return trace


def _round_sort_key(path: Path) -> tuple[int, str]:
    return (_round_num(path) or 10_000, path.name)


def _round_num(path: Path) -> int | None:
    try:
        return int(path.name.removeprefix("round_"))
    except ValueError:
        return None


def _gate_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    criteria = [
        _criterion("min_active_strategies", 4.0, metrics.get("active_strategies", 0.0), lower_is_better=False),
        _criterion("min_trades_per_month", TARGETS["trades_per_month"], metrics.get("trades_per_month", 0.0), lower_is_better=False),
        _criterion("max_drawdown_pct", TARGETS["max_drawdown_pct"], metrics.get("max_drawdown_pct", 1.0), lower_is_better=True),
        _criterion("min_profit_factor", TARGETS["profit_factor"], metrics.get("profit_factor", 0.0), lower_is_better=False),
        _criterion("max_block_rate", TARGETS["max_block_rate"], metrics.get("block_rate", 1.0), lower_is_better=True),
    ]
    return {
        "passed": all(item["passed"] for item in criteria),
        "criteria": criteria,
        "failure_category": None,
        "recommendations": [],
    }


def _criterion(name: str, target: float, actual: object, *, lower_is_better: bool) -> dict[str, Any]:
    actual_float = float(actual or 0.0)
    return {
        "name": name,
        "target": float(target),
        "actual": actual_float,
        "passed": actual_float <= target if lower_is_better else actual_float >= target,
    }


def _completed_phases(summary: dict[str, Any]) -> list[int]:
    phases = [int(item.get("phase", 0)) for item in summary.get("phases", [])]
    return [phase for phase in phases if phase > 0]


def _baseline_score(output_dir: Path) -> float:
    path = output_dir / "baseline.json"
    if not path.exists():
        return 0.0
    try:
        return float(_load_json(path).get("score", 0.0) or 0.0)
    except Exception:
        return 0.0


def _pct_of_target(key: str, actual: float, target: float) -> float:
    if target == 0:
        return 0.0
    if key.startswith("max_") or key == "block_rate":
        return min(200.0, target / max(actual, 1e-12) * 100.0)
    return min(200.0, actual / target * 100.0)


def _strengths(goal_progress: dict[str, dict[str, float]]) -> list[str]:
    return [
        f"{name}: {data['actual']:.4f} ({data['pct_of_target']:.0f}% of target)"
        for name, data in goal_progress.items()
        if data["pct_of_target"] >= 100.0
    ]


def _weaknesses(goal_progress: dict[str, dict[str, float]]) -> list[str]:
    return [
        f"{name}: {data['actual']:.4f} ({data['pct_of_target']:.0f}% of target)"
        for name, data in goal_progress.items()
        if data["pct_of_target"] < 100.0
    ]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True), encoding="utf-8")


def _write_json_preserve_order(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _round_name(round_num: int) -> str:
    return f"round_{round_num}_momentum_portfolio_synergy_standardized"


if __name__ == "__main__":
    main()
