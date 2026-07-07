"""Shared AM-14 performance-learning acceptance source fixtures."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def write_performance_learning_sources(findings: Path) -> None:
    """Write a minimal source-backed strategy/portfolio learning graph."""

    proposed_at = datetime(2026, 6, 21, 8, 0, tzinfo=timezone.utc)
    artifacts = findings / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    run_manifest = artifacts / "run_manifest.json"
    artifact_index = artifacts / "artifact_index.json"
    search_brief = artifacts / "monthly_search_brief.json"
    priors = artifacts / "outcome_priors_snapshot.json"
    verifier = artifacts / "monthly_evidence_verification.json"
    portfolio_metrics = artifacts / "portfolio_rolling_metrics.json"
    sufficiency_manifest = artifacts / "learning_sufficiency_manifest.json"

    _write_json(run_manifest, {
        "run_id": "monthly-run-1",
        "run_month": "2026-06",
        "bot_id": "bot1",
        "strategy_id": "strat1",
        "objective_version": "objective_weights_v1",
        "data_bundle_checksum": "bundle-source-1",
        "workflow_contract_version": "monthly_optimizer_v1",
        "monthly_search_brief_path": str(search_brief),
        "monthly_search_brief_id": "brief-2026-06-breakout",
        "source_weekly_signal_ids": ["weekly-signal-breakout"],
        "outcome_prior_snapshot_path": str(priors),
        "learning_sufficiency_manifest_path": str(sufficiency_manifest),
        "learning_sufficiency_status": "learning_authoritative",
        "supported_learning_capabilities": ["approval_grade_strategy_change"],
        "blocked_learning_capabilities": [],
    })
    _write_json(artifact_index, {
        "run_id": "monthly-run-1",
        "artifact_root": str(artifacts),
        "index_version": "backtest_artifact_index_v1",
        "artifacts": {
            "monthly_search_brief.json": str(search_brief),
            "outcome_priors_snapshot.json": str(priors),
            "monthly_evidence_verification.json": str(verifier),
            "learning_sufficiency_manifest.json": str(sufficiency_manifest),
            "portfolio_synergy.json": str(portfolio_metrics),
        },
    })
    _write_json(search_brief, {
        "run_month": "2026-06",
        "monthly_search_brief_id": "brief-2026-06-breakout",
        "source_weekly_signal_ids": ["weekly-signal-breakout"],
        "attribution": {"brief-2026-06-breakout": ["weekly-signal-breakout"]},
    })
    _write_json(priors, {
        "prior_id": "prior-breakout",
        "source_outcome_ids": ["outcome-breakout"],
        "allocation_multiplier": 1.1,
        "gate_strictness": "normal",
        "rollback_priority": "none",
        "evidence_paths": [str(run_manifest)],
    })
    _write_json(verifier, {"verifier_version": "monthly_evidence_verifier_v1"})
    _write_json(sufficiency_manifest, {
        "manifest_id": "learning-sufficiency-1",
        "eligibility": "learning_authoritative",
        "supported_learning_capabilities": ["approval_grade_strategy_change"],
        "blocked_learning_capabilities": [],
        "capability_status": {
            "approval_grade_strategy_change": {"status": "learning_authoritative"},
        },
        "runtime_evidence_support": {
            event_class: {
                "support_state": "observed",
                "configured_event_types": [event_class],
                "observed_event_count": 1,
                "event_value_classifications": {event_class: "learning_authority"},
            }
            for event_class in (
                "trade",
                "missed_opportunity",
                "filter_decision",
                "portfolio_rule",
                "order",
                "fill",
                "pipeline_funnel",
                "deployment_metadata",
            )
        },
    })
    _write_json(portfolio_metrics, {
        "portfolio_context": {
            "allocation_weights": {"strat1": 0.55, "mean_reversion": 0.45},
            "correlation": {"strat1:mean_reversion": 0.21},
            "marginal_contribution": {"strat1": 0.018},
        }
    })

    _write_jsonl(findings / "proposal_ledger.jsonl", [
        {
            "type": "candidate",
            "payload": {
                "proposal_id": "proposal-strat-1",
                "source": "monthly_model_review",
                "kind": "parameter_change",
                "bot_id": "bot1",
                "strategy_id": "strat1",
                "lifecycle_stage": "entry",
                "hypothesis_id": "hyp-breakout-quality",
                "title": "Raise breakout quality floor",
                "affected_parameters": ["entry.quality_min"],
                "affected_files": ["strategies/strat1/config.yaml"],
                "evaluation_method": "monthly_replay",
                "linked_diagnostics": [
                    "weekly-signal-breakout",
                    str(run_manifest),
                    str(artifact_index),
                    str(search_brief),
                ],
                "linked_run_id": "monthly-run-1",
                "proposed_at": proposed_at.isoformat(),
            },
        },
        {
            "type": "evaluation",
            "payload": {
                "proposal_id": "proposal-strat-1",
                "method": "monthly_replay",
                "summary": "Replay gates passed after costs",
                "objective_score": 0.143,
                "confidence": 0.78,
                "decision": "approve",
                "decision_reason": "Full-fidelity monthly replay improved objective after costs.",
                "evidence_paths": [str(artifact_index), str(verifier)],
                "evaluated_at": datetime(2026, 6, 21, 9, 0, tzinfo=timezone.utc).isoformat(),
            },
        },
    ])

    _write_jsonl(findings / "strategy_change_ledger.jsonl", [
        {
            "type": "record",
            "payload": {
                "record_id": "change-1",
                "bot_id": "bot1",
                "strategy_id": "strat1",
                "record_type": "monthly_review",
                "mutation_diff": {
                    "entry.quality_min": {"old": 0.62, "new": 0.68},
                    "source_weekly_signal_ids": ["weekly-signal-breakout"],
                    "brief_attribution_ids": ["brief-2026-06-breakout"],
                    "strategy_slice": {
                        "regime": "risk_on",
                        "symbol": "BTC",
                        "session": "us",
                        "side": "long",
                        "liquidity": "high",
                        "sample_size": 240,
                        "trade_count": 132,
                        "cost_bps": 6.4,
                        "failure_mode": "late_breakout_noise",
                    },
                },
                "source_proposal_ids": ["proposal-strat-1"],
                "approval_request_id": "approval-1",
                "deployment_id": "deploy-1",
                "evidence_paths": [str(run_manifest), str(artifact_index), str(search_brief), str(verifier)],
                "objective_deltas": {
                    "objective_delta": 0.143,
                    "return_delta": 0.041,
                    "drawdown_delta": -0.018,
                    "cost_delta": -0.006,
                    "slippage_delta": -0.002,
                    "confidence": 0.78,
                },
                "monthly_verdict": {
                    "verdict": "improved",
                    "objective_delta": 0.091,
                    "return_delta": 0.028,
                    "drawdown_delta": -0.011,
                    "cost_delta": -0.004,
                    "evidence_paths": ["artifacts/monthly_outcome.json"],
                    "strategy_slice": {
                        "regime": "risk_on",
                        "symbol": "BTC",
                        "session": "us",
                        "side": "long",
                        "liquidity": "high",
                        "sample_size": 240,
                        "trade_count": 132,
                        "cost_bps": 6.4,
                        "failure_mode": "late_breakout_noise",
                    },
                },
                "decision_reason": "Approval packet passed verifier.",
                "monthly_status": "approved",
                "run_id": "monthly-run-1",
                "run_month": "2026-06",
                "created_at": proposed_at.isoformat(),
                "updated_at": datetime(2026, 6, 21, 10, 0, tzinfo=timezone.utc).isoformat(),
            },
        }
    ])

    _write_jsonl(findings / "portfolio_outcomes.jsonl", [
        {
            "outcome_id": "portfolio-outcome-1",
            "proposal_id": "proposal-portfolio-1",
            "bot_id": "PORTFOLIO",
            "portfolio_id": "core_portfolio",
            "deployment_id": "portfolio-deploy-1",
            "outcome_source": "follow_up_persistence",
            "measured_at": "2026-06-22T12:00:00+00:00",
            "composite_delta": 0.034,
            "return_delta": 0.019,
            "drawdown_delta": -0.007,
            "cost_delta": -0.001,
            "verdict": "positive",
            "source_weekly_signal_ids": ["weekly-signal-allocation"],
            "brief_attribution_ids": ["brief-2026-06-allocation"],
            "monthly_search_brief_path": str(search_brief),
            "evidence_paths": [str(portfolio_metrics)],
            "portfolio_metrics_path": str(portfolio_metrics),
            "portfolio_context": {
                "allocation_weights": {"strat1": 0.55, "mean_reversion": 0.45},
                "risk_budgets": {"strat1": 0.6, "mean_reversion": 0.4},
                "exposure": {"net": 0.72},
                "correlation": {"strat1:mean_reversion": 0.21},
                "drawdown_overlap": {"strat1:mean_reversion": 0.08},
                "crowding": "low",
                "cannibalization": "none",
                "marginal_contribution": {"strat1": 0.018},
                "concentration": "within_limit",
                "liquidity_constraints": ["btc depth ok"],
            },
        },
    ])

    _write_jsonl(findings / "loop_run_ledger.jsonl", [
        {
            "loop_run_id": "loop-1",
            "loop_id": "monthly_validation",
            "job_key": "monthly_validation",
            "scope_key": "bot:bot1",
            "bot_id": "bot1",
            "strategy_id": "strat1",
            "status": "completed",
            "task_id": "task-1",
            "agent_run_id": "agent-run-1",
            "proposal_ids": ["proposal-strat-1"],
        },
    ])


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )
