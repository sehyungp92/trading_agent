from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from trading_assistant.analysis.monthly_model_response_parser import parse_monthly_model_review
from trading_assistant.analysis.monthly_model_response_validator import MonthlyModelResponseValidator
from trading_assistant.schemas.market_data_manifest import MarketDataManifest
from trading_assistant.schemas.monthly_candidates import MonthlyImprovementCandidate
from trading_assistant.schemas.monthly_validation import MonthlyValidationStatus
from trading_assistant.schemas.strategy_change_ledger import StrategyChangeRecordType
from trading_assistant.schemas.events import TradeEvent
from trading_assistant.schemas.monthly_evidence_verification import (
    MonthlyEvidenceFinding,
    MonthlyEvidenceRecommendedAction,
    MonthlyEvidenceVerdict,
    MonthlyEvidenceVerification,
)
from trading_assistant.skills.approval_tracker import ApprovalTracker
from trading_assistant.skills.monthly_validation_orchestrator import MonthlyValidationOrchestrator, MonthlyValidationRequest
from trading_assistant.skills.strategy_change_ledger import StrategyChangeLedger


def _trade(**overrides) -> TradeEvent:
    data = {
        "trade_id": "t1",
        "bot_id": "bot1",
        "strategy_id": "strat1",
        "pair": "AAPL",
        "side": "LONG",
        "entry_time": "2026-04-02T10:00:00Z",
        "exit_time": "2026-04-02T11:00:00Z",
        "entry_price": 100.0,
        "exit_price": 101.0,
        "position_size": 1.0,
        "pnl": 1.0,
        "pnl_pct": 1.0,
        "strategy_version": "sv1",
        "config_version": "cv1",
        "deployment_id": "dep1",
        "parameter_set_id": "ps1",
        "experiment_id": "exp1",
    }
    data.update(overrides)
    return TradeEvent.model_validate(data)


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    curated = tmp_path / "curated"
    findings = tmp_path / "memory" / "findings"
    bot_dir = curated / "2026-04-02" / "bot1"
    bot_dir.mkdir(parents=True)
    (bot_dir / "trades.jsonl").write_text(_trade().model_dump_json() + "\n", encoding="utf-8")
    (bot_dir / "missed.jsonl").write_text(
        "\n".join([
            json.dumps({
                "bot_id": "bot1",
                "strategy_id": "strat1",
                "opportunity_id": "miss-a",
                "pair": "AAPL",
                "signal": "opening_breakout",
                "would_have_pnl": 12.0,
                "strategy_version": "sv1",
                "config_version": "cv1",
                "deployment_id": "dep1",
            }),
            json.dumps({
                "bot_id": "bot1",
                "strategy_id": "strat1",
                "opportunity_id": "miss-b",
                "pair": "AAPL",
                "signal": "opening_breakout",
                "would_have_pnl": 8.0,
                "strategy_version": "sv1",
                "config_version": "cv1",
                "deployment_id": "dep1",
            }),
        ])
        + "\n",
        encoding="utf-8",
    )

    market_root = tmp_path / "market_data"
    manifest_path = market_root / "manifests" / "bot1" / "strat1" / "2026-04.coverage_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(MarketDataManifest(
        source="fixture",
        market="equity",
        symbol="AAPL",
        timeframe="1m",
        start_ts=datetime(2026, 4, 1, tzinfo=timezone.utc),
        end_ts=datetime(2026, 4, 30, tzinfo=timezone.utc),
        expected_bars=10,
        actual_bars=10,
        usable_for_authoritative_validation=True,
    ).model_dump_json(indent=2), encoding="utf-8")

    repo = tmp_path / "backtests"
    repo.mkdir()
    return curated, findings, market_root, repo


def _write_fixture_runner(
    repo: Path,
    *,
    valid_candidate: bool = True,
    omit_source: bool = False,
    write_model_review: bool = True,
    model_review_known_evidence: bool = True,
    outside_evidence_path: bool = False,
) -> None:
    evidence_setup = ""
    if outside_evidence_path:
        evidence_setup = '(root.parent / "outside_candidate_evidence.json").write_text("outside evidence")'
        evidence_paths_expr = '[str(root.parent / "outside_candidate_evidence.json")]'
    else:
        evidence_paths_expr = '[str(root / "candidate_results.jsonl")]' if valid_candidate else "[]"
    source_line = "" if omit_source else '"source": "smoke_repair",'
    if outside_evidence_path and model_review_known_evidence:
        model_review_path_expr = 'str(root.parent / "outside_candidate_evidence.json")'
    else:
        model_review_path_expr = 'str(root / "candidate_results.jsonl")' if model_review_known_evidence else '"invented.json"'
    model_review_block = f"""
(root / "model_review.json").write_text(json.dumps({{
    "run_id": manifest["run_id"],
    "bot_id": manifest["bot_id"],
    "strategy_id": manifest["strategy_id"],
    "candidate_reviews": [{{
        "candidate_id": "cand-smoke-1",
        "recommendation": "approval packet is coherent",
        "evidence_paths": [{model_review_path_expr}],
        "capability_labels": ["filter_threshold_learning"],
        "evidence_authority": "learning_authoritative",
        "expected_objective_impact": {{"latest_month_oos": 0.18}},
        "risk_classification": "medium",
        "replay_or_experiment_plan": "Measure next completed month.",
        "acceptance_criteria": ["positive latest OOS"],
        "rollback_plan": "Restore config version cv1.",
        "routing": "experiment"
    }}]
}}))
""" if write_model_review else ""
    repo.joinpath("fixture_runner.py").write_text(
        f"""
import json
import hashlib
from pathlib import Path
import sys

manifest = json.loads(Path(sys.argv[1]).read_text())
manifest_id = hashlib.sha256("|".join([
    manifest["run_id"],
    manifest["run_month"],
    manifest["bot_id"],
    manifest["strategy_id"],
    manifest["mode"],
    manifest["latest_month_start"],
    manifest["latest_month_end"],
]).encode("utf-8")).hexdigest()[:16]
root = Path(manifest["artifact_root"])
root.mkdir(parents=True, exist_ok=True)
{evidence_setup}

candidate = {{
    "candidate_id": "cand-smoke-1",
    {source_line}
    "run_id": manifest["run_id"],
    "manifest_id": manifest_id,
    "round_id": manifest.get("round_id") or "round_1",
    "prior_round_id": manifest.get("prior_round_id") or "round_0",
    "next_round_id": manifest.get("next_round_id") or "round_1",
    "backtest_repo_commit_sha": manifest.get("backtest_repo_commit_sha") or "fixture-backtest-sha",
    "live_trading_repo_commit_sha": manifest.get("trading_repo_commit_sha") or "fixture-live-sha",
    "control_plane_commit_sha": manifest.get("control_plane_commit_sha") or "fixture-control-sha",
    "family": "accepted_prefix_rollback",
    "title": "Rollback harmful filter overreach",
    "description": "Undo the latest filter mutation after replay attribution.",
    "change_kind": "rollback",
    "risk_classification": "medium",
    "objective_score": 1.18,
    "baseline_score": 1.0,
    "objective_delta": 0.18,
    "objective_version": "objective_weights_v1",
    "effective_objective_version": "immutable_score_profiles_v1",
    "immutable_objective_version": "immutable_score_profiles_v1",
    "objective_profile_id": "trading.momentum.nqdtc",
    "objective_profile_family": "trading_momentum",
    "objective_profile_scope": "strategy",
    "score_component_cap": 5,
    "immutable_score": {{
        "profile_id": "trading.momentum.nqdtc",
        "profile_version": "immutable_score_profiles_v1",
        "objective_score": 1.18,
        "score_component_cap": 5,
        "renormalized_components": []
    }},
    "objective_deltas": {{"latest_month_oos": 0.18, "calibration": 0.05}},
    "latest_month_oos_delta": 0.18,
    "calibration_objective_delta": 0.05,
    "trade_count": 44,
    "evidence_paths": {evidence_paths_expr},
    "planned_files": ["config/strategy.yaml"],
    "param_changes": [{{"param_name": "entry_filter_enabled", "current": True, "proposed": False}}],
    "acceptance_criteria": ["Latest OOS objective remains positive after costs."],
    "replay_or_experiment_plan": "Apply rollback after approval and measure next completed month.",
    "rollback_plan": "Restore config version cv1.",
    "deterministic_gate_inputs": {{
        "latest_month_oos_improvement": True,
        "calibration_support": True,
        "leakage_passed": True,
        "sufficient_trade_count": True,
        "cost_gate_passed": True,
        "drawdown_gate_passed": True,
        "outlier_concentration_passed": True,
        "risk_constraints_passed": True,
        "runner_contract_version": "smoke_repair_runner_contract_v1"
    }}
}}

(root / "coverage_manifest.json").write_text("{{}}")
(root / "incumbent_validation.json").write_text(json.dumps({{"objective_delta": -0.14}}))
(root / "gap_attribution.json").write_text(json.dumps({{"primary_category": "filter_overreach"}}))
(root / "mode_decision.json").write_text(json.dumps({{"status": "repair", "mode": "smoke_repair"}}))
(root / "objective_breakdown.json").write_text(json.dumps({{
    "objective_version": "objective_weights_v1",
    "effective_objective_version": "immutable_score_profiles_v1",
    "immutable_objective_version": "immutable_score_profiles_v1",
    "objective_profile_id": "trading.momentum.nqdtc",
    "score_component_cap": 5
}}))
(root / "candidate_results.jsonl").write_text(json.dumps(candidate) + "\\n")
(root / "selected_candidates.json").write_text(json.dumps([candidate]))
(root / "rejected_candidates.jsonl").write_text(json.dumps({{"candidate_id": "weak-1", "reason": "failed drawdown gate"}}) + "\\n")
(root / "monthly_report.md").write_text("candidate fixture")
{model_review_block}
(root / "replay_parity_report.json").write_text(json.dumps({{
    "bot_id": manifest["bot_id"],
    "strategy_id": manifest["strategy_id"],
    "run_month": manifest["run_month"],
    "trade_count_live": 1,
    "trade_count_replay": 1,
    "entry_match_rate": 1.0,
    "exit_match_rate": 1.0,
    "side_quantity_match_rate": 1.0,
    "status": "pass"
}}))

required = [
    "coverage_manifest.json",
    "incumbent_validation.json",
    "gap_attribution.json",
    "mode_decision.json",
    "replay_parity_report.json",
    "objective_breakdown.json",
    "candidate_results.jsonl",
    "selected_candidates.json",
    "rejected_candidates.jsonl",
    "monthly_report.md",
    "stdout.log",
    "stderr.log",
    "exit_status.json",
]
(root / "artifact_index.json").write_text(json.dumps({{
    "run_id": manifest["run_id"],
    "manifest_id": manifest_id,
    "artifact_root": str(root),
    "artifacts": {{name: str(root / name) for name in required}},
}}))
""",
        encoding="utf-8",
    )


def test_approval_gated_monthly_failure_creates_evidence_backed_approval(tmp_path: Path) -> None:
    curated, findings, market_root, repo = _write_inputs(tmp_path)
    _write_fixture_runner(repo, valid_candidate=True)
    approval_tracker = ApprovalTracker(findings / "approvals.jsonl")

    orchestrator = MonthlyValidationOrchestrator(
        curated_dir=curated,
        findings_dir=findings,
        market_data_root=market_root,
        backtest_repo_path=repo,
        backtest_artifact_root=tmp_path / "artifacts",
        approval_tracker=approval_tracker,
    )
    result = orchestrator.run(MonthlyValidationRequest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-04",
        shadow=False,
        backtest_command=["python", "fixture_runner.py", "{manifest}"],
        optimizer_sequence_enabled=False,
    ))

    assert result.status == MonthlyValidationStatus.REPAIR
    assert result.selected_candidate_count == 1
    assert result.rejected_candidate_count == 1
    assert result.approval_ready_candidate_count == 1
    assert len(result.approval_request_ids) == 1
    assert Path(result.candidate_gate_report_path).exists()
    assert Path(result.approval_packet_paths[0]).exists()
    assert result.monthly_evidence_verification_paths
    assert Path(result.strategy_discovery_packet_path).exists()
    assert result.strategy_discovery_packet_path in result.evidence_paths
    discovery_packet = json.loads(Path(result.strategy_discovery_packet_path).read_text(encoding="utf-8"))
    assert discovery_packet["authority"] == "diagnostics_only"
    assert discovery_packet["approval_gate_eligible"] is False
    packet = json.loads(Path(result.approval_packet_paths[0]).read_text(encoding="utf-8"))
    assert result.proposal_ids == [packet["proposal_id"]]
    assert result.monthly_evidence_verification_paths[0] in packet["artifact_paths"]
    assert (
        result.monthly_evidence_verification_paths[0]
        in packet["machine_readable_payload"]["approval_gate_evidence"]
    )

    request = approval_tracker.get_by_id(result.approval_request_ids[0])
    assert request is not None
    assert request.monthly_run_id == result.run_id
    assert request.strategy_id == "strat1"
    assert request.evidence_paths
    assert request.rollback_plan == "Restore config version cv1."
    assert request.risk_tier.value == "requires_approval"

    proposed = [
        record for record in StrategyChangeLedger(findings).get_for_strategy("bot1", "strat1")
        if record.record_type == StrategyChangeRecordType.PROPOSED_CHANGE
    ]
    assert proposed
    assert proposed[0].approval_request_id == request.request_id
    monthly_reviews = [
        record for record in StrategyChangeLedger(findings).get_for_strategy("bot1", "strat1")
        if record.record_type == StrategyChangeRecordType.MONTHLY_REVIEW
    ]
    assert monthly_reviews
    assert result.approval_packet_paths[0] in monthly_reviews[0].evidence_paths
    assert result.candidate_gate_report_path in monthly_reviews[0].evidence_paths
    assert result.model_review_validation_path in monthly_reviews[0].evidence_paths
    assert result.monthly_report_path in monthly_reviews[0].evidence_paths
    assert str(Path(result.monthly_report_path).parent / "monthly_validation_result.json") in monthly_reviews[0].evidence_paths
    packet = json.loads(Path(result.approval_packet_paths[0]).read_text(encoding="utf-8"))
    assert packet["effective_objective_version"] == "immutable_score_profiles_v1"
    assert packet["objective_profile_id"] == "trading.momentum.nqdtc"
    assert (
        packet["machine_readable_payload"]["objective"]["immutable_score"]["profile_id"]
        == "trading.momentum.nqdtc"
    )
    for evidence_path in monthly_reviews[0].evidence_paths:
        assert Path(evidence_path).exists(), evidence_path
    ledger_events = [
        json.loads(line)
        for line in (findings / "strategy_change_ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    monthly_review_events = [
        event for event in ledger_events
        if event.get("payload", {}).get("record_id") == monthly_reviews[0].record_id
    ]
    assert [event["type"] for event in monthly_review_events] == ["record"]


def test_monthly_run_wires_expected_sessions_and_runtime_support(tmp_path: Path) -> None:
    curated, findings, market_root, repo = _write_inputs(tmp_path)
    _write_fixture_runner(repo, valid_candidate=True)
    expected_sessions = tmp_path / "expected_active_sessions.json"
    expected_sessions.write_text(json.dumps({"expected_session_days": ["2026-04-02"]}), encoding="utf-8")
    runtime_support = tmp_path / "runtime_evidence_support.json"
    runtime_support.write_text(json.dumps({
        "support_source_paths": ["sidecar.py"],
        "event_value_classifications": {
            "trade": "learning_authority",
            "missed_opportunity": "learning_authority",
            "pipeline_funnel": "learning_authority",
        },
        "evidence_classes": {
            "trade": {"supported": True, "configured_event_types": ["trade"]},
            "missed_opportunity": {"supported": True, "configured_event_types": ["missed_opportunity"]},
            "pipeline_funnel": {"supported": True, "configured_event_types": ["pipeline_funnel"]},
        },
    }), encoding="utf-8")

    result = MonthlyValidationOrchestrator(
        curated_dir=curated,
        findings_dir=findings,
        market_data_root=market_root,
        backtest_repo_path=repo,
        backtest_artifact_root=tmp_path / "artifacts",
    ).run(MonthlyValidationRequest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-04",
        backtest_command=["python", "fixture_runner.py", "{manifest}"],
        optimizer_sequence_enabled=False,
        expected_session_paths=[expected_sessions],
        runtime_support_paths=[runtime_support],
    ))

    run_manifest = json.loads(Path(result.run_manifest_path).read_text(encoding="utf-8"))
    learning_manifest = json.loads(Path(result.learning_sufficiency_manifest_path).read_text(encoding="utf-8"))
    assert run_manifest["expected_session_paths"] == [str(expected_sessions)]
    assert run_manifest["runtime_support_paths"] == [str(runtime_support)]
    assert str(expected_sessions) in result.evidence_paths
    assert str(runtime_support) in result.evidence_paths
    assert (
        learning_manifest["denominator_coverage"]["denominator_coverage"]["details"]["expected_session_source_paths"]
        == [str(expected_sessions).replace("\\", "/")]
    )
    assert learning_manifest["runtime_evidence_support"]["trade"]["support_source_paths"] == ["sidecar.py"]


def test_shadow_monthly_candidate_never_creates_approval_request(tmp_path: Path) -> None:
    curated, findings, market_root, repo = _write_inputs(tmp_path)
    _write_fixture_runner(repo, valid_candidate=True)
    approval_tracker = ApprovalTracker(findings / "approvals.jsonl")

    orchestrator = MonthlyValidationOrchestrator(
        curated_dir=curated,
        findings_dir=findings,
        market_data_root=market_root,
        backtest_repo_path=repo,
        backtest_artifact_root=tmp_path / "artifacts",
        approval_tracker=approval_tracker,
    )
    result = orchestrator.run(MonthlyValidationRequest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-04",
        shadow=True,
        backtest_command=["python", "fixture_runner.py", "{manifest}"],
        optimizer_sequence_enabled=False,
    ))

    assert result.selected_candidate_count == 1
    assert result.approval_ready_candidate_count == 0
    assert result.approval_request_ids == []
    assert approval_tracker.get_pending() == []
    packet = json.loads(Path(result.approval_packet_paths[0]).read_text(encoding="utf-8"))
    assert "monthly validation is running in shadow mode" in packet["approval_suppressed_reasons"]


def test_candidate_without_evidence_fails_closed(tmp_path: Path) -> None:
    curated, findings, market_root, repo = _write_inputs(tmp_path)
    _write_fixture_runner(repo, valid_candidate=False)
    approval_tracker = ApprovalTracker(findings / "approvals.jsonl")

    orchestrator = MonthlyValidationOrchestrator(
        curated_dir=curated,
        findings_dir=findings,
        market_data_root=market_root,
        backtest_repo_path=repo,
        backtest_artifact_root=tmp_path / "artifacts",
        approval_tracker=approval_tracker,
    )
    result = orchestrator.run(MonthlyValidationRequest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-04",
        shadow=False,
        backtest_command=["python", "fixture_runner.py", "{manifest}"],
        optimizer_sequence_enabled=False,
    ))

    assert result.approval_ready_candidate_count == 0
    assert approval_tracker.get_pending() == []
    gates = json.loads(Path(result.candidate_gate_report_path).read_text(encoding="utf-8"))
    failed = [check["name"] for check in gates[0]["checks"] if not check["passed"]]
    assert "candidate_evidence_paths" in failed


def test_candidate_evidence_outside_artifact_root_fails_closed(tmp_path: Path) -> None:
    curated, findings, market_root, repo = _write_inputs(tmp_path)
    _write_fixture_runner(repo, outside_evidence_path=True)
    approval_tracker = ApprovalTracker(findings / "approvals.jsonl")

    orchestrator = MonthlyValidationOrchestrator(
        curated_dir=curated,
        findings_dir=findings,
        market_data_root=market_root,
        backtest_repo_path=repo,
        backtest_artifact_root=tmp_path / "artifacts",
        approval_tracker=approval_tracker,
    )
    result = orchestrator.run(MonthlyValidationRequest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-04",
        shadow=False,
        backtest_command=["python", "fixture_runner.py", "{manifest}"],
        optimizer_sequence_enabled=False,
    ))

    assert result.approval_ready_candidate_count == 0
    assert approval_tracker.get_pending() == []
    gates = json.loads(Path(result.candidate_gate_report_path).read_text(encoding="utf-8"))
    failed = [check["name"] for check in gates[0]["checks"] if not check["passed"]]
    assert "candidate_artifact_containment" in failed


def test_verifier_suppression_does_not_record_proposal_approve(tmp_path: Path) -> None:
    curated, findings, market_root, repo = _write_inputs(tmp_path)
    _write_fixture_runner(repo, valid_candidate=True)
    approval_tracker = ApprovalTracker(findings / "approvals.jsonl")

    class FailingVerifier:
        saw_persisted_packet = False
        saw_persisted_gate_report = False

        def verify(self, **kwargs):
            packet = kwargs["approval_packet"]
            self.saw_persisted_packet = Path(packet.approval_packet_path).exists()
            gate_paths = packet.machine_readable_payload.get("approval_gate_evidence", [])
            self.saw_persisted_gate_report = any(
                Path(path).name == "candidate_gate_report.json" and Path(path).exists()
                for path in gate_paths
            )
            return MonthlyEvidenceVerification(
                run_id=kwargs["monthly_result"].run_id,
                candidate_id=packet.candidate_id,
                verdict=MonthlyEvidenceVerdict.FAIL,
                recommended_action=MonthlyEvidenceRecommendedAction.SUPPRESS_APPROVAL,
                blocking_findings=[
                    MonthlyEvidenceFinding(
                        code="forced_verifier_failure",
                        message="fixture verifier failure",
                    )
                ],
            )

        def write(self, verification, artifact_root: Path, *, candidate_id: str) -> Path:
            path = artifact_root / f"monthly_evidence_verification_{candidate_id}.json"
            verification.verifier_artifact_path = str(path)
            path.write_text(verification.model_dump_json(indent=2), encoding="utf-8")
            return path

    verifier = FailingVerifier()
    orchestrator = MonthlyValidationOrchestrator(
        curated_dir=curated,
        findings_dir=findings,
        market_data_root=market_root,
        backtest_repo_path=repo,
        backtest_artifact_root=tmp_path / "artifacts",
        approval_tracker=approval_tracker,
    )
    orchestrator.candidate_pipeline.evidence_verifier = verifier
    result = orchestrator.run(MonthlyValidationRequest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-04",
        shadow=False,
        backtest_command=["python", "fixture_runner.py", "{manifest}"],
        optimizer_sequence_enabled=False,
    ))

    assert verifier.saw_persisted_packet is True
    assert verifier.saw_persisted_gate_report is True
    assert result.approval_ready_candidate_count == 0
    assert approval_tracker.get_pending() == []
    packet = json.loads(Path(result.approval_packet_paths[0]).read_text(encoding="utf-8"))
    assert "monthly evidence verifier verdict is fail" in packet["approval_suppressed_reasons"]
    proposals = orchestrator.proposal_ledger.list_all()
    assert proposals
    assert proposals[-1].evaluations[-1].decision == "reject"
    assert proposals[-1].evaluations[-1].decision != "approve"
    assert proposals[-1].evaluations[-1].method == "monthly_candidate_gates_and_verifier"


def test_final_packet_registry_validation_blocks_approval_after_verifier_pass(tmp_path: Path) -> None:
    curated, findings, market_root, repo = _write_inputs(tmp_path)
    _write_fixture_runner(repo, valid_candidate=True)
    approval_tracker = ApprovalTracker(findings / "approvals.jsonl")

    class PassingVerifierWithWrongArtifactName:
        def verify(self, **kwargs):
            packet = kwargs["approval_packet"]
            return MonthlyEvidenceVerification(
                run_id=kwargs["monthly_result"].run_id,
                candidate_id=packet.candidate_id,
                verdict=MonthlyEvidenceVerdict.PASS,
                recommended_action=MonthlyEvidenceRecommendedAction.ROUTE_APPROVAL,
            )

        def write(self, verification, artifact_root: Path, *, candidate_id: str) -> Path:
            path = artifact_root / f"verifier_output_{candidate_id}.json"
            verification.verifier_artifact_path = str(path)
            path.write_text(verification.model_dump_json(indent=2), encoding="utf-8")
            return path

    orchestrator = MonthlyValidationOrchestrator(
        curated_dir=curated,
        findings_dir=findings,
        market_data_root=market_root,
        backtest_repo_path=repo,
        backtest_artifact_root=tmp_path / "artifacts",
        approval_tracker=approval_tracker,
    )
    orchestrator.candidate_pipeline.evidence_verifier = PassingVerifierWithWrongArtifactName()
    result = orchestrator.run(MonthlyValidationRequest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-04",
        shadow=False,
        backtest_command=["python", "fixture_runner.py", "{manifest}"],
        optimizer_sequence_enabled=False,
    ))

    assert result.approval_ready_candidate_count == 0
    assert approval_tracker.get_pending() == []
    packet = json.loads(Path(result.approval_packet_paths[0]).read_text(encoding="utf-8"))
    assert packet["evidence_verification_verdict"] == "pass"
    assert any(
        "monthly_evidence_verification" in reason
        for reason in packet["approval_suppressed_reasons"]
    )


def test_candidate_source_can_be_inferred_from_mode_decision(tmp_path: Path) -> None:
    curated, findings, market_root, repo = _write_inputs(tmp_path)
    _write_fixture_runner(repo, valid_candidate=True, omit_source=True)
    approval_tracker = ApprovalTracker(findings / "approvals.jsonl")

    orchestrator = MonthlyValidationOrchestrator(
        curated_dir=curated,
        findings_dir=findings,
        market_data_root=market_root,
        backtest_repo_path=repo,
        backtest_artifact_root=tmp_path / "artifacts",
        approval_tracker=approval_tracker,
    )
    result = orchestrator.run(MonthlyValidationRequest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-04",
        shadow=False,
        backtest_command=["python", "fixture_runner.py", "{manifest}"],
        optimizer_sequence_enabled=False,
    ))

    assert result.gate_passed_candidate_count == 1
    assert result.approval_ready_candidate_count == 1
    gates = json.loads(Path(result.candidate_gate_report_path).read_text(encoding="utf-8"))
    source_gate = next(check for check in gates[0]["checks"] if check["name"] == "supported_candidate_source")
    assert source_gate["passed"] is True


def test_existing_model_review_is_validated_and_linked(tmp_path: Path) -> None:
    curated, findings, market_root, repo = _write_inputs(tmp_path)
    _write_fixture_runner(repo, write_model_review=True, model_review_known_evidence=False)
    approval_tracker = ApprovalTracker(findings / "approvals.jsonl")

    orchestrator = MonthlyValidationOrchestrator(
        curated_dir=curated,
        findings_dir=findings,
        market_data_root=market_root,
        backtest_repo_path=repo,
        backtest_artifact_root=tmp_path / "artifacts",
        approval_tracker=approval_tracker,
    )
    result = orchestrator.run(MonthlyValidationRequest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-04",
        shadow=False,
        backtest_command=["python", "fixture_runner.py", "{manifest}"],
        optimizer_sequence_enabled=False,
    ))

    assert result.model_review_path.endswith("model_review.json")
    assert result.model_review_valid is False
    assert result.model_review_validation_path
    assert any("deterministic evidence set" in issue for issue in result.model_review_issues)
    packet = json.loads(Path(result.approval_packet_paths[0]).read_text(encoding="utf-8"))
    assert packet["machine_readable_payload"]["model_review_validation"]["valid"] is False


def test_missing_monthly_model_review_blocks_approval(tmp_path: Path) -> None:
    curated, findings, market_root, repo = _write_inputs(tmp_path)
    _write_fixture_runner(repo, write_model_review=False)
    approval_tracker = ApprovalTracker(findings / "approvals.jsonl")

    orchestrator = MonthlyValidationOrchestrator(
        curated_dir=curated,
        findings_dir=findings,
        market_data_root=market_root,
        backtest_repo_path=repo,
        backtest_artifact_root=tmp_path / "artifacts",
        approval_tracker=approval_tracker,
    )
    result = orchestrator.run(MonthlyValidationRequest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-04",
        shadow=False,
        backtest_command=["python", "fixture_runner.py", "{manifest}"],
        optimizer_sequence_enabled=False,
    ))

    assert result.approval_ready_candidate_count == 0
    assert result.repair_required is True
    assert result.approval_packet_paths == []
    assert result.candidate_summary_path == ""
    assert result.candidate_gate_report_path == ""
    artifact_root = Path(result.monthly_report_path).parent
    assert (artifact_root / "model_review_request.json").exists()
    assert not list(artifact_root.glob("approval_packet_*.json"))
    repair = json.loads(Path(result.repair_request_path).read_text(encoding="utf-8"))
    assert repair["classification"] == "model_review"
    monthly_reviews = [
        record for record in StrategyChangeLedger(findings).get_for_strategy("bot1", "strat1")
        if record.record_type == StrategyChangeRecordType.MONTHLY_REVIEW
    ]
    assert monthly_reviews
    assert result.repair_request_path in monthly_reviews[0].evidence_paths
    assert "requires repair" in monthly_reviews[0].decision_reason


def test_monthly_model_review_invoker_creates_valid_review(tmp_path: Path) -> None:
    curated, findings, market_root, repo = _write_inputs(tmp_path)
    _write_fixture_runner(repo, write_model_review=False)
    approval_tracker = ApprovalTracker(findings / "approvals.jsonl")

    def invoker(prompt_package, run_id: str) -> str:
        assert prompt_package.metadata["workflow"] == "monthly_model_review"
        request = prompt_package.data
        evidence_path = request["selected_candidates"][0]["evidence_paths"][0]
        return json.dumps({
            "run_id": request["run_id"],
            "bot_id": request["bot_id"],
            "strategy_id": request["strategy_id"],
            "candidate_reviews": [{
                "candidate_id": "cand-smoke-1",
                "recommendation": "approval packet is coherent",
                "evidence_paths": [evidence_path],
                "capability_labels": ["filter_threshold_learning"],
                "evidence_authority": "learning_authoritative",
                "expected_objective_impact": {"latest_month_oos": 0.18},
                "risk_classification": "medium",
                "replay_or_experiment_plan": "Measure next completed month.",
                "acceptance_criteria": ["positive latest OOS"],
                "rollback_plan": "Restore config version cv1.",
                "routing": "experiment",
            }],
        })

    orchestrator = MonthlyValidationOrchestrator(
        curated_dir=curated,
        findings_dir=findings,
        market_data_root=market_root,
        backtest_repo_path=repo,
        backtest_artifact_root=tmp_path / "artifacts",
        approval_tracker=approval_tracker,
        model_review_invoker=invoker,
    )
    result = orchestrator.run(MonthlyValidationRequest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-04",
        shadow=False,
        backtest_command=["python", "fixture_runner.py", "{manifest}"],
        optimizer_sequence_enabled=False,
    ))

    assert result.model_review_valid is True
    assert result.approval_ready_candidate_count == 1
    assert Path(result.model_review_path).exists()


def test_monthly_model_review_requires_evidence_for_actionable_output() -> None:
    response = """
<!-- MONTHLY_MODEL_REVIEW
{
  "run_id": "monthly-bot1-strat1-2026-04",
  "bot_id": "bot1",
  "strategy_id": "strat1",
  "candidate_reviews": [
    {
      "candidate_id": "cand1",
      "recommendation": "looks good",
      "routing": "experiment",
      "risk_classification": "high",
      "expected_objective_impact": {"latest_month_oos": 0.1},
      "replay_or_experiment_plan": "paper deploy",
      "acceptance_criteria": ["passes replay"],
      "rollback_plan": "restore incumbent"
    }
  ],
  "structural_proposals": [
    {
      "bot_id": "bot1",
      "title": "Split filter by regime",
      "routing": "hypothesis_only"
    }
  ]
}
-->
"""
    review = parse_monthly_model_review(response)
    validation = MonthlyModelResponseValidator().validate(review)

    assert review.parse_success is True
    assert validation.valid is False
    assert any(issue.message == "evidence_paths are required" for issue in validation.issues)
    assert validation.approval_tiers["cand1"] == "requires_double_approval"
    assert "Split filter by regime" in validation.hypothesis_only_ids


def test_monthly_model_review_accepts_authoritative_capability_labeled_output(tmp_path: Path) -> None:
    evidence = tmp_path / "candidate_gate_report.json"
    evidence.write_text("{}", encoding="utf-8")
    review = parse_monthly_model_review(f"""
<!-- MONTHLY_MODEL_REVIEW
{{
  "run_id": "monthly-bot1-strat1-2026-04",
  "bot_id": "bot1",
  "strategy_id": "strat1",
  "candidate_reviews": [
    {{
      "candidate_id": "cand1",
      "recommendation": "looks good",
      "routing": "experiment",
      "risk_classification": "medium",
      "evidence_paths": [{json.dumps(str(evidence))}],
      "capability_labels": ["execution_learning"],
      "evidence_authority": "learning_authoritative",
      "expected_objective_impact": {{"latest_month_oos": 0.1}},
      "replay_or_experiment_plan": "paper deploy",
      "acceptance_criteria": ["passes replay"],
      "rollback_plan": "restore incumbent"
    }}
  ]
}}
-->
""")

    validation = MonthlyModelResponseValidator().validate(
        review,
        allowed_evidence_paths=[str(evidence)],
    )

    assert validation.valid is True
    assert validation.actionable_candidate_ids == ["cand1"]


def test_monthly_model_review_rejects_diagnostics_only_as_approval_grade(tmp_path: Path) -> None:
    evidence = tmp_path / "learning_sufficiency_manifest.json"
    evidence.write_text("{}", encoding="utf-8")
    review = parse_monthly_model_review(f"""
<!-- MONTHLY_MODEL_REVIEW
{{
  "run_id": "monthly-bot1-strat1-2026-04",
  "bot_id": "bot1",
  "strategy_id": "strat1",
  "candidate_reviews": [
    {{
      "candidate_id": "cand1",
      "recommendation": "looks good",
      "routing": "experiment",
      "risk_classification": "medium",
      "evidence_paths": [{json.dumps(str(evidence))}],
      "capability_labels": ["execution_learning"],
      "evidence_authority": "diagnostics_only",
      "expected_objective_impact": {{"latest_month_oos": 0.1}},
      "replay_or_experiment_plan": "paper deploy",
      "acceptance_criteria": ["passes replay"],
      "rollback_plan": "restore incumbent"
    }}
  ]
}}
-->
""")

    validation = MonthlyModelResponseValidator().validate(
        review,
        allowed_evidence_paths=[str(evidence)],
    )

    assert validation.valid is False
    assert any(
        issue.message == "diagnostics-only evidence cannot be presented as approval-grade"
        for issue in validation.issues
    )


def test_monthly_model_review_rejects_existing_evidence_outside_allowlist(tmp_path: Path) -> None:
    unallowed = tmp_path / "approval_packet_cand1.json"
    allowed = tmp_path / "candidate_gate_report.json"
    unallowed.write_text("{}", encoding="utf-8")
    allowed.write_text("{}", encoding="utf-8")
    review = parse_monthly_model_review(f"""
<!-- MONTHLY_MODEL_REVIEW
{{
  "run_id": "monthly-bot1-strat1-2026-04",
  "bot_id": "bot1",
  "strategy_id": "strat1",
  "candidate_reviews": [
    {{
      "candidate_id": "cand1",
      "recommendation": "looks good",
      "routing": "experiment",
      "risk_classification": "medium",
      "evidence_paths": [{json.dumps(str(unallowed))}],
      "capability_labels": ["filter_threshold_learning"],
      "evidence_authority": "learning_authoritative",
      "expected_objective_impact": {{"latest_month_oos": 0.1}},
      "replay_or_experiment_plan": "paper deploy",
      "acceptance_criteria": ["passes replay"],
      "rollback_plan": "restore incumbent"
    }}
  ]
}}
-->
""")

    validation = MonthlyModelResponseValidator().validate(
        review,
        allowed_evidence_paths=[str(allowed)],
    )

    assert validation.valid is False
    assert any("not in deterministic evidence set" in issue.message for issue in validation.issues)


def test_monthly_model_review_parser_drops_non_object_items() -> None:
    review = parse_monthly_model_review("""
<!-- MONTHLY_MODEL_REVIEW
{
  "run_id": "monthly-bot1-strat1-2026-04",
  "candidate_reviews": ["bad", {"candidate_id": "cand1", "routing": "hypothesis_only"}],
  "structural_proposals": [42]
}
-->
""")

    assert review.parse_success is True
    assert [item.candidate_id for item in review.candidate_reviews] == ["cand1"]
    assert review.dropped_counts == {"candidate_reviews": 1, "structural_proposals": 1}


def test_monthly_candidate_preserves_workspace_attempt_contract_fields() -> None:
    candidate = MonthlyImprovementCandidate.from_raw({
        "candidate_id": "cand-workspace",
        "source": "phased_auto",
        "candidate_workspace_key": "round_1_cand_a",
        "candidate_workspace_path": "workspaces/round_1_cand_a",
        "candidate_attempt_id": "attempt-1",
        "candidate_attempt_status": "completed",
        "retry_attempt": 1,
        "retry_reason": "stall_retry",
        "stall_timeout_seconds": 600,
        "workflow_contract_version": "phased_auto_runner_contract_v1",
        "runner_contract_version": "phased_auto_runner_contract_v1",
    })

    assert candidate.candidate_workspace_key == "round_1_cand_a"
    assert candidate.candidate_attempt_status == "completed"
    assert candidate.retry_attempt == 1
    assert candidate.stall_timeout_seconds == 600
    assert candidate.deterministic_gate_inputs["runner_contract_version"] == "phased_auto_runner_contract_v1"
