from __future__ import annotations

from datetime import date
from pathlib import Path

from trading_assistant.schemas.backtest_artifacts import (
    REQUIRED_BACKTEST_ARTIFACTS,
    BacktestArtifactIndex,
)
from trading_assistant.schemas.loop_contracts import LoopAuthority, LoopContract
from trading_assistant.schemas.monthly_artifact_contract import MonthlyArtifactStatus
from trading_assistant.schemas.monthly_candidates import (
    MonthlyApprovalEvidencePacket,
    MonthlyCandidateGateReport,
    MonthlyGateCheck,
    MonthlyImprovementCandidate,
)
from trading_assistant.schemas.monthly_evidence_verification import MonthlyEvidenceVerdict
from trading_assistant.schemas.monthly_model_review import MonthlyModelValidationResult
from trading_assistant.schemas.monthly_run_manifest import MonthlyRunManifest
from trading_assistant.schemas.monthly_validation import MonthlyValidationResult, MonthlyValidationStatus
from trading_assistant.skills.monthly_artifact_contract import MonthlyArtifactContract
from trading_assistant.skills.monthly_evidence_verifier import MonthlyEvidenceVerifier


def _write_required_artifacts(root: Path) -> None:
    for name in REQUIRED_BACKTEST_ARTIFACTS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text("{}", encoding="utf-8")
        elif path.suffix == ".jsonl":
            path.write_text("", encoding="utf-8")
        else:
            path.write_text("artifact\n", encoding="utf-8")


def _manifest(root: Path) -> MonthlyRunManifest:
    return MonthlyRunManifest(
        run_id="monthly-bot1-strat1-2026-05",
        run_month="2026-05",
        bot_id="bot1",
        strategy_id="strat1",
        latest_month_start=date(2026, 5, 1),
        latest_month_end=date(2026, 5, 31),
        market_data_manifest_path=str(root / "market_data_manifest.json"),
        telemetry_manifest_path=str(root / "telemetry_manifest.json"),
        artifact_root=str(root),
    )


def _index(root: Path, manifest: MonthlyRunManifest) -> BacktestArtifactIndex:
    return BacktestArtifactIndex(
        run_id=manifest.run_id,
        manifest_id=manifest.manifest_id,
        artifact_root=str(root),
        artifacts={name: str(root / name) for name in REQUIRED_BACKTEST_ARTIFACTS},
    )


def _loop_contract() -> LoopContract:
    sections = {
        "Purpose": "Test loop.",
        "Current focus": "Test.",
        "Authority boundary": "No live mutation.",
        "Inputs": "Test input.",
        "Outputs": "Test output.",
        "Required checks": "Test check.",
        "Failure modes": "Test failure.",
        "Escalation path": "Test escalation.",
        "Backlog": "None.",
        "Timeline": "Now.",
    }
    return LoopContract(
        loop_id="monthly_validation",
        job_key="monthly_validation",
        authority=LoopAuthority(
            negative_authority=[
                "no_live_bot_mutation",
                "no_autonomous_policy_memory_write",
            ],
        ),
        stopping_criteria=["test complete"],
        body_sections=sections,
    )


def test_contract_reports_complete_index_and_wraps_backtest_index(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    _write_required_artifacts(root)
    manifest = _manifest(root)
    index = _index(root, manifest)

    contract = MonthlyArtifactContract.from_run(
        manifest=manifest,
        artifact_index=index,
        artifact_root=root,
    )

    assert contract.path("coverage_manifest.json") == index.artifact_path("coverage_manifest.json")
    assert contract.issues() == []
    assert contract.view("coverage_manifest.json").required is True


def test_contract_surfaces_missing_malformed_out_of_root_and_scope_issues(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    outside_root = tmp_path / "outside"
    root.mkdir()
    outside_root.mkdir()
    _write_required_artifacts(root)
    (root / "objective_breakdown.json").unlink()
    (root / "selected_candidates.json").write_text("{not-json", encoding="utf-8")
    outside_coverage = outside_root / "coverage_manifest.json"
    outside_coverage.write_text("{}", encoding="utf-8")
    manifest = _manifest(root)
    index = _index(root, manifest)
    index.run_id = "monthly-other"
    index.manifest_id = "manifest-other"
    index.artifacts["coverage_manifest.json"] = str(outside_coverage)

    issues = MonthlyArtifactContract.from_run(
        manifest=manifest,
        artifact_index=index,
        artifact_root=root,
    ).issues()
    statuses = {issue.status for issue in issues}

    assert MonthlyArtifactStatus.SCOPE_MISMATCH in statuses
    assert MonthlyArtifactStatus.MISSING_REQUIRED in statuses
    assert MonthlyArtifactStatus.MALFORMED in statuses
    assert MonthlyArtifactStatus.OUTSIDE_ROOT in statuses


def test_contract_composes_authority_registry_for_gate_evidence(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    _write_required_artifacts(root)
    monthly_result_path = root / "monthly_validation_result.json"
    gate_path = root / "candidate_gate_report.json"
    validation_path = root / "model_review_validation.json"
    advisory_path = root / "monthly_search_brief.json"
    diagnostics_path = root / "runner_observability.json"
    for path in [monthly_result_path, gate_path, validation_path, advisory_path, diagnostics_path]:
        path.write_text("{}", encoding="utf-8")
    manifest = _manifest(root)
    contract = MonthlyArtifactContract.from_run(
        manifest=manifest,
        artifact_index=_index(root, manifest),
        artifact_root=root,
    )
    candidate = MonthlyImprovementCandidate.from_raw({
        "candidate_id": "cand1",
        "evidence_paths": [str(root / "candidate_results.jsonl"), str(advisory_path), str(diagnostics_path)],
    })

    view = contract.approval_gate_evidence(
        "cand1",
        candidate=candidate,
        monthly_result_path=monthly_result_path,
        replay_parity_path=str(root / "replay_parity_report.json"),
        candidate_gate_report_path=str(gate_path),
        model_review_validation_path=str(validation_path),
    )

    assert str(monthly_result_path) in view.approval_gate_evidence
    assert str(root / "replay_parity_report.json") in view.approval_gate_evidence
    assert str(advisory_path) not in view.approval_gate_evidence
    assert str(diagnostics_path) not in view.approval_gate_evidence


def test_contract_loads_selected_and_rejected_candidates(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    _write_required_artifacts(root)
    (root / "mode_decision.json").write_text('{"mode": "repair"}', encoding="utf-8")
    (root / "objective_breakdown.json").write_text(
        '{"objective_version": "objective-test", "profile": {"family": "core"}}',
        encoding="utf-8",
    )
    (root / "selected_candidates.json").write_text(
        '{"selected_candidates": [{"candidate_id": "cand1", "evidence_paths": ["candidate_results.jsonl"]}]}',
        encoding="utf-8",
    )
    (root / "rejected_candidates.jsonl").write_text('{"candidate_id": "cand2"}\n', encoding="utf-8")
    manifest = _manifest(root)
    contract = MonthlyArtifactContract.from_run(
        manifest=manifest,
        artifact_index=_index(root, manifest),
        artifact_root=root,
    )

    selected = contract.load_selected_candidates(bot_id="bot1", strategy_id="strat1")
    contract.normalize_candidate_paths(selected[0])

    assert selected[0].candidate_id == "cand1"
    assert selected[0].source.value == "smoke_repair"
    assert selected[0].evidence_paths == [str(root / "candidate_results.jsonl")]
    assert contract.load_rejected_candidates() == [{"candidate_id": "cand2"}]


def test_verifier_runs_from_contract_built_input(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    _write_required_artifacts(root)
    for name in [
        "monthly_validation_result.json",
        "candidate_gate_report.json",
        "model_review_validation.json",
        "approval_packet_cand1.json",
        "model_review.json",
    ]:
        (root / name).write_text("{}", encoding="utf-8")
    manifest = _manifest(root)
    monthly_result = MonthlyValidationResult(
        run_id=manifest.run_id,
        run_month=manifest.run_month,
        bot_id=manifest.bot_id,
        strategy_id=manifest.strategy_id,
        status=MonthlyValidationStatus.EXPERIMENT,
        run_manifest_path=str(root / "run_manifest.json"),
        artifact_index_path=str(root / "artifact_index.json"),
        replay_parity_path=str(root / "replay_parity_report.json"),
        evidence_paths=[str(root / "replay_parity_report.json")],
    )
    candidate = MonthlyImprovementCandidate.from_raw({
        "candidate_id": "cand1",
        "bot_id": manifest.bot_id,
        "strategy_id": manifest.strategy_id,
        "run_id": manifest.run_id,
        "evidence_paths": [str(root / "candidate_results.jsonl")],
        "rollback_plan": "restore incumbent",
        "acceptance_criteria": ["positive OOS"],
    })
    gate = MonthlyCandidateGateReport(
        candidate_id="cand1",
        checks=[MonthlyGateCheck(name="all_gates", passed=True)],
    )
    validation = MonthlyModelValidationResult(
        valid=True,
        actionable_candidate_ids=["cand1"],
        approval_tiers={"cand1": "requires_approval"},
    )
    packet = MonthlyApprovalEvidencePacket(
        candidate_id="cand1",
        run_id=manifest.run_id,
        run_month=manifest.run_month,
        bot_id=manifest.bot_id,
        strategy_id=manifest.strategy_id,
        title="Packet",
        rollback_plan="restore incumbent",
        objective_deltas={"latest_month_oos": 0.1},
        data_coverage_status="authoritative",
        replay_parity_status="pass",
        artifact_paths=[
            str(root / "monthly_validation_result.json"),
            str(root / "replay_parity_report.json"),
            str(root / "candidate_gate_report.json"),
            str(root / "model_review_validation.json"),
            str(root / "candidate_results.jsonl"),
        ],
        model_review_path=str(root / "model_review.json"),
        model_review_validation_path=str(root / "model_review_validation.json"),
        approval_packet_path=str(root / "approval_packet_cand1.json"),
        machine_readable_payload={
            "approval_gate_evidence": [
                str(root / "monthly_validation_result.json"),
                str(root / "replay_parity_report.json"),
                str(root / "candidate_gate_report.json"),
                str(root / "model_review_validation.json"),
            ],
        },
    )
    contract = MonthlyArtifactContract.from_run(
        manifest=manifest,
        artifact_index=_index(root, manifest),
        artifact_root=root,
    )

    verifier_input = contract.verifier_input(
        "cand1",
        monthly_result=monthly_result,
        selected_candidates=[candidate],
        gate_reports=[gate],
        approval_packet=packet,
        run_manifest=manifest,
        model_validation=validation,
        model_review_validation_path=str(root / "model_review_validation.json"),
    )
    result = MonthlyEvidenceVerifier().verify_input(verifier_input)

    assert result.verdict == MonthlyEvidenceVerdict.PASS


def test_verifier_input_preserves_loop_contract(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    _write_required_artifacts(root)
    manifest = _manifest(root)
    loop_contract = _loop_contract()
    monthly_result = MonthlyValidationResult(
        run_id=manifest.run_id,
        run_month=manifest.run_month,
        bot_id=manifest.bot_id,
        strategy_id=manifest.strategy_id,
        status=MonthlyValidationStatus.EXPERIMENT,
    )
    contract = MonthlyArtifactContract.from_run(
        manifest=manifest,
        artifact_index=_index(root, manifest),
        artifact_root=root,
    )

    verifier_input = contract.verifier_input(
        "cand1",
        monthly_result=monthly_result,
        loop_contract=loop_contract,
    )

    assert verifier_input.loop_contract is loop_contract
