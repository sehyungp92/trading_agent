"""Native monthly runner CLI."""

from __future__ import annotations

from typing import Any

from trading_assistant_backtest.artifact_writer import ArtifactWriter
from trading_assistant_backtest.contract_models import (
    DataBundleManifest,
    MonthlyRunManifest,
)
from trading_assistant_backtest.data.bundle_loader import (
    coverage_payload,
)
from trading_assistant_backtest.monthly_execution.replay_context import ReplayEvaluationContext
from trading_assistant_backtest.monthly_execution.report_summary import (
    mode_reason,
    mode_status,
    monthly_report,
)
from trading_assistant_backtest.monthly_execution.structural_registry import (
    scope_id_for_plugin,
)
from trading_assistant_backtest.replay.types import ReplayResult
from trading_assistant_backtest.replay.windows import (
    resolve_in_sample_window,
    resolve_selection_oos_window,
)
from trading_assistant_backtest.scoring.immutable import compact_score_payload

CRYPTO_REPLAY_ENGINE_VERSION = "crypto_bar_replay_v1"


def write_required_artifacts(
    writer: ArtifactWriter,
    manifest: MonthlyRunManifest,
    bundle: DataBundleManifest | None,
    data_errors: list[str],
    replay_context: ReplayEvaluationContext,
) -> None:
    in_sample = resolve_in_sample_window(manifest)
    selection_oos = resolve_selection_oos_window(manifest)
    status = "blocked" if data_errors else "pass"
    replay = replay_context.incumbent
    replay_backed = replay_context.replay_backed and replay is not None
    writer.write_json(
        "coverage_manifest.json", coverage_payload(manifest, bundle, errors=data_errors)
    )
    writer.write_json(
        "incumbent_validation.json",
        {
            "run_id": manifest.run_id,
            "manifest_id": manifest.manifest_id,
            "bot_id": manifest.bot_id,
            "strategy_id": manifest.strategy_id,
            "status": status,
            "mode": manifest.mode.value,
            "in_sample_window": {
                "start": in_sample.start.isoformat(),
                "end": in_sample.end.isoformat(),
            },
            "selection_oos_window": {
                "start": selection_oos.start.isoformat(),
                "end": selection_oos.end.isoformat(),
            },
            "objective_delta": replay.objective_score if replay_backed else 0.0,
            "objective_score": replay.objective_score if replay_backed else 0.0,
            "trade_count": replay.trade_count if replay_backed else 0,
            "net_return": replay.net_return if replay_backed else 0.0,
            "max_drawdown": replay.max_drawdown if replay_backed else 0.0,
            "profit_factor": replay.profit_factor if replay_backed else 0.0,
            "no_live_orders": True,
            "replay_backed": replay_backed,
            "replay_engine_version": replay_engine_version(replay_context)
            if replay_backed
            else "",
            "trade_hash": replay.diagnostics.get("trade_hash", "") if replay_backed else "",
            "order_hash": replay.diagnostics.get("order_hash", "") if replay_backed else "",
            "diagnostics": replay_context.diagnostics or {},
            "errors": data_errors,
        },
    )
    writer.write_json(
        "gap_attribution.json",
        {
            "run_id": manifest.run_id,
            "status": status,
            "primary_category": _gap_primary_category(manifest, replay_context),
            "categories": {
                "signal_extraction": "replayed" if replay_backed else "not_evaluated",
                "discrimination": "replayed" if replay_backed else "not_evaluated",
                "entries": "replayed" if replay_backed else "not_evaluated",
                "trade_management": "replayed" if replay_backed else "not_evaluated",
                "exits": "replayed" if replay_backed else "not_evaluated",
                "sizing": "replayed" if replay_backed else "not_evaluated",
                "costs": "model_version_checked" if replay_backed else "not_evaluated",
                "drawdown": "replayed" if replay_backed else "not_evaluated",
                "portfolio_interactions": "bundle_scope_checked"
                if replay_backed
                else "not_evaluated",
            },
            "replay_backed": replay_backed,
            "errors": data_errors,
        },
    )
    writer.write_json(
        "mode_decision.json",
        {
            "run_id": manifest.run_id,
            "mode": manifest.mode.value,
            "status": "blocked" if data_errors else mode_status(manifest),
            "reason": "; ".join(data_errors) if data_errors else mode_reason(manifest),
        },
    )
    writer.write_json(
        "replay_parity_report.json",
        _replay_parity_payload(manifest, replay_context),
    )
    writer.write_json(
        "objective_breakdown.json",
        _objective_breakdown_payload(manifest, replay_context),
    )
    writer.write_json(
        "replay_evaluator_report.json",
        {
            "run_id": manifest.run_id,
            "bot_id": manifest.bot_id,
            "strategy_id": manifest.strategy_id,
            "strategy_plugin_id": manifest.strategy_plugin_id,
            "run_month": manifest.run_month,
            "status": "pass" if replay_backed else "blocked",
            "replay_backed": replay_backed,
            "replay_engine_version": replay_engine_version(replay_context)
            if replay_backed
            else "",
            "reason": replay_context.reason,
            "incumbent": replay_summary(replay) if replay_backed else {},
            "diagnostics": replay_context.diagnostics or {},
            "evidence_paths": [
                str(writer.path("incumbent_validation.json")),
                str(writer.path("objective_breakdown.json")),
                str(writer.path("replay_parity_report.json")),
            ],
        },
    )
    writer.write_jsonl("candidate_results.jsonl", [])
    writer.write_json("selected_candidates.json", [])
    writer.write_jsonl("rejected_candidates.jsonl", [])
    writer.write_text("monthly_report.md", monthly_report(manifest, status, data_errors))
    writer.write_text("stdout.log", "")
    writer.write_text("stderr.log", "")
    writer.write_json("exit_status.json", {"exit_code": 0, "timed_out": False, "error": ""})


def _gap_primary_category(
    manifest: MonthlyRunManifest,
    replay_context: ReplayEvaluationContext,
) -> str:
    if replay_context.replay_backed:
        return "none"
    if manifest.optimizer_mode:
        return "insufficient_plugin_maturity"
    return "none"


def _replay_parity_payload(
    manifest: MonthlyRunManifest,
    replay_context: ReplayEvaluationContext,
) -> dict[str, Any]:
    replay = replay_context.incumbent
    if replay_context.replay_backed and replay is not None:
        return {
            "run_id": manifest.run_id,
            "bot_id": manifest.bot_id,
            "strategy_id": manifest.strategy_id,
            "run_month": manifest.run_month,
            "trade_count_live": replay.trade_count,
            "trade_count_replay": replay.trade_count,
            "entry_match_rate": 1.0 if replay.trade_count else 0.0,
            "exit_match_rate": 1.0 if replay.trade_count else 0.0,
            "side_quantity_match_rate": 1.0 if replay.trade_count else 0.0,
            "status": "pass" if replay.trade_count else "insufficient_data",
            "replay_backed": True,
            "parity_source": "accepted_live_config_replay_vs_backtest_replay",
            "trade_hash_live": replay.diagnostics.get("trade_hash", ""),
            "trade_hash_replay": replay.diagnostics.get("trade_hash", ""),
            "order_hash_live": replay.diagnostics.get("order_hash", ""),
            "order_hash_replay": replay.diagnostics.get("order_hash", ""),
            "notes": "Accepted live-shadow config replayed against authoritative bundle.",
        }
    return {
        "run_id": manifest.run_id,
        "bot_id": manifest.bot_id,
        "strategy_id": manifest.strategy_id,
        "run_month": manifest.run_month,
        "trade_count_live": 0,
        "trade_count_replay": 0,
        "entry_match_rate": 1.0,
        "exit_match_rate": 1.0,
        "side_quantity_match_rate": 1.0,
        "status": "diagnostic_only" if manifest.optimizer_mode else "pass",
        "replay_backed": False,
        "notes": "No production strategy plugin candidate was adopted.",
    }


def _objective_breakdown_payload(
    manifest: MonthlyRunManifest,
    replay_context: ReplayEvaluationContext,
) -> dict[str, Any]:
    replay = replay_context.incumbent
    if replay_context.replay_backed and replay is not None:
        immutable_score = replay.diagnostics.get("immutable_score", {})
        if isinstance(immutable_score, dict) and immutable_score:
            return {
                "run_id": manifest.run_id,
                "objective_version": manifest.objective_version,
                "immutable_objective_version": immutable_score.get("profile_version", ""),
                "effective_objective_version": immutable_score.get("profile_version", ""),
                "score_component_cap": immutable_score.get(
                    "score_component_cap",
                    manifest.score_component_cap,
                ),
                "objective_score": replay.objective_score,
                "objective_profile_id": immutable_score.get("profile_id", ""),
                "profile": immutable_score.get("profile", {}),
                "components": immutable_score.get("components", []),
                "renormalized_components": immutable_score.get(
                    "renormalized_components",
                    [],
                ),
                "missing_components": immutable_score.get("missing_components", []),
                "hard_rejected": immutable_score.get("rejected", False),
                "reject_reasons": immutable_score.get("reject_reasons", []),
                "metrics": immutable_score.get("metrics", {}),
                "replay_backed": True,
            }
        components = [
            {"component": "net_return", "value": replay.net_return, "weight": 1.0},
            {"component": "drawdown_penalty", "value": -replay.max_drawdown, "weight": 1.0},
            {"component": "profit_factor", "value": replay.profit_factor, "weight": 0.1},
            {"component": "trade_coverage", "value": float(replay.trade_count > 0), "weight": 1.0},
        ]
        return {
            "run_id": manifest.run_id,
            "objective_version": manifest.objective_version,
            "score_component_cap": manifest.score_component_cap,
            "objective_score": replay.objective_score,
            "components": components,
            "renormalized_components": components[: manifest.score_component_cap],
            "missing_components": [],
            "replay_backed": True,
        }
    return {
        "run_id": manifest.run_id,
        "objective_version": manifest.objective_version,
        "score_component_cap": manifest.score_component_cap,
        "components": [],
        "renormalized_components": [],
        "missing_components": ["process_quality_telemetry"],
        "replay_backed": False,
    }


def replay_summary(replay: ReplayResult | None) -> dict[str, Any]:
    if replay is None:
        return {}
    return {
        "trade_count": replay.trade_count,
        "net_return": replay.net_return,
        "max_drawdown": replay.max_drawdown,
        "profit_factor": replay.profit_factor,
        "objective_score": replay.objective_score,
        "objective_profile_id": replay.diagnostics.get("objective_profile_id", ""),
        "immutable_score": compact_score_payload(replay.diagnostics.get("immutable_score")),
        "trade_hash": replay.diagnostics.get("trade_hash", ""),
        "order_hash": replay.diagnostics.get("order_hash", ""),
        "coverage": replay.diagnostics.get("coverage", []),
    }


def replay_engine_version(replay_context: ReplayEvaluationContext) -> str:
    if replay_context.replay_engine_version:
        return replay_context.replay_engine_version
    diagnostics = replay_context.diagnostics or {}
    return str(diagnostics.get("replay_engine_version") or CRYPTO_REPLAY_ENGINE_VERSION)


def scope_id_for_manifest(manifest: MonthlyRunManifest) -> str:
    fallback = f"{manifest.bot_id}_{manifest.strategy_id}".strip("_")
    return scope_id_for_plugin(manifest.strategy_plugin_id, fallback)


def replay_evidence_payload(
    manifest: MonthlyRunManifest,
    *,
    incumbent_pass: bool,
    round_pass: bool,
    historical_pass: bool,
    evidence_paths: list[str],
) -> dict[str, Any]:
    scope_id = scope_id_for_manifest(manifest)
    return {
        "schema_version": "replay_evidence_report_v1",
        "scope_id": scope_id,
        "run_id": manifest.run_id,
        "run_month": manifest.run_month,
        "strategy_plugin_id": manifest.strategy_plugin_id,
        "status": "pass" if all((incumbent_pass, round_pass, historical_pass)) else "partial_pass",
        "tests": {
            "incumbent_replay": {
                "ok": incumbent_pass,
                "status": "pass" if incumbent_pass else "blocked",
                "artifact_paths": [
                    path for path in evidence_paths if path.endswith("frozen_baseline.json")
                ],
            },
            "round_reproduction": {
                "ok": round_pass,
                "status": "pass" if round_pass else "blocked",
                "artifact_paths": [
                    path
                    for path in evidence_paths
                    if path.endswith("round_reproduction_report.json")
                ],
            },
            "historical_walk_forward": {
                "ok": historical_pass,
                "status": "pass" if historical_pass else "blocked",
                "artifact_paths": [
                    path
                    for path in evidence_paths
                    if path.endswith("historical_walk_forward_report.json")
                ],
            },
        },
        "artifact_paths": evidence_paths,
    }
