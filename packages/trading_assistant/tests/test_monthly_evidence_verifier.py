from __future__ import annotations

from datetime import date
from pathlib import Path

from trading_assistant.schemas.backtest_artifacts import BacktestArtifactIndex, REQUIRED_BACKTEST_ARTIFACTS
from trading_assistant.schemas.monthly_candidates import (
    MonthlyApprovalEvidencePacket,
    MonthlyCandidateGateReport,
    MonthlyGateCheck,
    MonthlyGateSeverity,
    MonthlyImprovementCandidate,
)
from trading_assistant.schemas.monthly_evidence_verification import MonthlyEvidenceVerdict
from trading_assistant.schemas.monthly_model_review import (
    MonthlyModelCandidateReview,
    MonthlyModelReview,
    MonthlyModelRouting,
    MonthlyModelValidationResult,
)
from trading_assistant.schemas.monthly_run_manifest import MonthlyRunManifest
from trading_assistant.schemas.monthly_validation import MonthlyValidationResult, MonthlyValidationStatus
from trading_assistant.skills.monthly_evidence_verifier import MonthlyEvidenceVerifier


def _fixture(tmp_path: Path):
    root = tmp_path / "artifacts"
    root.mkdir()
    paths = {
        "monthly_result": root / "monthly_validation_result.json",
        "parity": root / "replay_parity_report.json",
        "gate": root / "candidate_gate_report.json",
        "model_validation": root / "model_review_validation.json",
        "candidate": root / "candidate_results.jsonl",
        "packet": root / "approval_packet_cand1.json",
        "model_review": root / "model_review.json",
    }
    for path in paths.values():
        path.write_text("{}" if path.suffix == ".json" else "", encoding="utf-8")
    for name in REQUIRED_BACKTEST_ARTIFACTS:
        path = root / name
        if path.exists():
            continue
        if path.suffix == ".json":
            path.write_text("{}", encoding="utf-8")
        elif path.suffix == ".jsonl":
            path.write_text("", encoding="utf-8")
        else:
            path.write_text("fixture artifact\n", encoding="utf-8")
    monthly_result = MonthlyValidationResult(
        run_id="monthly-bot1-strat1-2026-05",
        run_month="2026-05",
        bot_id="bot1",
        strategy_id="strat1",
        status=MonthlyValidationStatus.REPAIR,
        run_manifest_path=str(root / "run_manifest.json"),
        artifact_index_path=str(root / "artifact_index.json"),
        replay_parity_path=str(paths["parity"]),
        evidence_paths=[str(paths["parity"])],
    )
    manifest = MonthlyRunManifest(
        run_id=monthly_result.run_id,
        run_month=monthly_result.run_month,
        bot_id="bot1",
        strategy_id="strat1",
        latest_month_start=date(2026, 5, 1),
        latest_month_end=date(2026, 5, 31),
        market_data_manifest_path=str(root / "market_data_manifest.json"),
        telemetry_manifest_path=str(root / "telemetry_manifest.json"),
        artifact_root=str(root),
    )
    candidate = MonthlyImprovementCandidate.from_raw({
        "candidate_id": "cand1",
        "source": "smoke_repair",
        "bot_id": "bot1",
        "strategy_id": "strat1",
        "run_id": monthly_result.run_id,
        "evidence_paths": [str(paths["candidate"])],
        "rollback_plan": "restore incumbent",
        "replay_or_experiment_plan": "measure next month",
        "acceptance_criteria": ["positive OOS"],
        "param_changes": [{"param_name": "x", "current": 1, "proposed": 2}],
    })
    gate = MonthlyCandidateGateReport(
        candidate_id="cand1",
        checks=[MonthlyGateCheck(name="all_gates", passed=True)],
    )
    review = MonthlyModelReview(
        run_id=monthly_result.run_id,
        bot_id="bot1",
        strategy_id="strat1",
        candidate_reviews=[
            MonthlyModelCandidateReview(
                candidate_id="cand1",
                evidence_paths=[str(paths["candidate"])],
                expected_objective_impact={"latest_month_oos": 0.1},
                replay_or_experiment_plan="measure next month",
                acceptance_criteria=["positive OOS"],
                rollback_plan="restore incumbent",
                routing=MonthlyModelRouting.EXPERIMENT,
            )
        ],
    )
    validation = MonthlyModelValidationResult(
        valid=True,
        actionable_candidate_ids=["cand1"],
        approval_tiers={"cand1": "requires_approval"},
    )
    packet = MonthlyApprovalEvidencePacket(
        candidate_id="cand1",
        run_id=monthly_result.run_id,
        run_month=monthly_result.run_month,
        bot_id="bot1",
        strategy_id="strat1",
        title="Clean packet",
        rollback_plan="restore incumbent",
        objective_deltas={"latest_month_oos": 0.1},
        data_coverage_status="authoritative",
        replay_parity_status="pass",
        artifact_paths=[
            str(paths["monthly_result"]),
            str(paths["parity"]),
            str(paths["gate"]),
            str(paths["model_validation"]),
            str(paths["candidate"]),
        ],
        model_review_path=str(paths["model_review"]),
        model_review_validation_path=str(paths["model_validation"]),
        approval_packet_path=str(paths["packet"]),
        machine_readable_payload={
            "approval_gate_evidence": [
                str(paths["monthly_result"]),
                str(paths["parity"]),
                str(paths["gate"]),
                str(paths["model_validation"]),
            ],
        },
    )
    index = BacktestArtifactIndex(
        run_id=monthly_result.run_id,
        manifest_id=manifest.manifest_id,
        artifact_root=str(root),
        artifacts={name: str(root / name) for name in REQUIRED_BACKTEST_ARTIFACTS},
    )
    return monthly_result, manifest, index, candidate, gate, review, validation, packet, paths


def test_clean_monthly_evidence_passes_and_writes_artifact(tmp_path: Path) -> None:
    monthly_result, manifest, index, candidate, gate, review, validation, packet, paths = _fixture(tmp_path)
    verifier = MonthlyEvidenceVerifier()

    result = verifier.verify(
        monthly_result=monthly_result,
        artifact_index=index,
        selected_candidates=[candidate],
        gate_reports=[gate],
        approval_packet=packet,
        run_manifest=manifest,
        model_review=review,
        model_validation=validation,
        model_review_validation_path=str(paths["model_validation"]),
    )
    out = verifier.write(result, Path(manifest.artifact_root), candidate_id="cand1")

    assert result.verdict == MonthlyEvidenceVerdict.PASS
    assert result.recommended_action.value == "route_approval"
    assert out.exists()


def test_missing_required_gate_evidence_fails_verifier(tmp_path: Path) -> None:
    monthly_result, manifest, index, candidate, gate, review, validation, packet, paths = _fixture(tmp_path)
    packet.machine_readable_payload["approval_gate_evidence"] = [str(paths["parity"])]

    result = MonthlyEvidenceVerifier().verify(
        monthly_result=monthly_result,
        artifact_index=index,
        selected_candidates=[candidate],
        gate_reports=[gate],
        approval_packet=packet,
        run_manifest=manifest,
        model_review=review,
        model_validation=validation,
        model_review_validation_path=str(paths["model_validation"]),
    )

    assert result.verdict == MonthlyEvidenceVerdict.FAIL
    assert {
        "monthly_validation_result",
        "candidate_gate_report",
        "model_review_validation",
    } <= {
        check.message.split(":", 1)[0]
        for check in result.authority_checks
        if check.name == "required_approval_gate_artifact_present"
    }


def test_invalid_artifact_index_fails_closed(tmp_path: Path) -> None:
    monthly_result, manifest, index, candidate, gate, review, validation, packet, paths = _fixture(tmp_path)
    Path(index.artifacts["objective_breakdown.json"]).unlink()

    result = MonthlyEvidenceVerifier().verify(
        monthly_result=monthly_result,
        artifact_index=index,
        selected_candidates=[candidate],
        gate_reports=[gate],
        approval_packet=packet,
        run_manifest=manifest,
        model_review=review,
        model_validation=validation,
        model_review_validation_path=str(paths["model_validation"]),
    )

    assert result.verdict == MonthlyEvidenceVerdict.FAIL
    assert any(
        finding.code == "artifact_index_invalid"
        and "missing required artifacts" in finding.message
        for finding in result.blocking_findings
    )


def test_nonexistent_model_review_evidence_fails_verification(tmp_path: Path) -> None:
    monthly_result, manifest, index, candidate, gate, review, validation, packet, paths = _fixture(tmp_path)
    review.candidate_reviews[0].evidence_paths = [str(tmp_path / "invented_evidence.json")]

    result = MonthlyEvidenceVerifier().verify(
        monthly_result=monthly_result,
        artifact_index=index,
        selected_candidates=[candidate],
        gate_reports=[gate],
        approval_packet=packet,
        run_manifest=manifest,
        model_review=review,
        model_validation=validation,
        model_review_validation_path=str(paths["model_validation"]),
    )

    assert result.verdict == MonthlyEvidenceVerdict.FAIL
    assert any(
        finding.code in {"unknown_evidence_path", "missing_evidence_path"}
        for finding in result.blocking_findings
    )


def test_advisory_model_review_evidence_fails_verification(tmp_path: Path) -> None:
    monthly_result, manifest, index, candidate, gate, review, validation, packet, paths = _fixture(tmp_path)
    brief_path = Path(manifest.artifact_root) / "monthly_search_brief.json"
    brief_path.write_text("{}", encoding="utf-8")
    review.candidate_reviews[0].evidence_paths = [str(brief_path)]

    result = MonthlyEvidenceVerifier().verify(
        monthly_result=monthly_result,
        artifact_index=index,
        selected_candidates=[candidate],
        gate_reports=[gate],
        approval_packet=packet,
        run_manifest=manifest,
        model_review=review,
        model_validation=validation,
        model_review_validation_path=str(paths["model_validation"]),
    )

    assert result.verdict == MonthlyEvidenceVerdict.FAIL
    assert any(
        finding.code == "model_review_evidence_authority_violation"
        and "advisory evidence" in finding.message
        for finding in result.blocking_findings
    )


def test_diagnostics_only_model_review_evidence_fails_verification(tmp_path: Path) -> None:
    monthly_result, manifest, index, candidate, gate, review, validation, packet, paths = _fixture(tmp_path)
    observability_path = Path(manifest.artifact_root) / "runner_observability.json"
    observability_path.write_text("{}", encoding="utf-8")
    review.candidate_reviews[0].evidence_paths = [str(observability_path)]

    result = MonthlyEvidenceVerifier().verify(
        monthly_result=monthly_result,
        artifact_index=index,
        selected_candidates=[candidate],
        gate_reports=[gate],
        approval_packet=packet,
        run_manifest=manifest,
        model_review=review,
        model_validation=validation,
        model_review_validation_path=str(paths["model_validation"]),
    )

    assert result.verdict == MonthlyEvidenceVerdict.FAIL
    assert any(
        finding.code == "model_review_evidence_authority_violation"
        and "diagnostics_only evidence" in finding.message
        for finding in result.blocking_findings
    )


def test_hypothesis_only_advisory_evidence_does_not_suppress_clean_packet(tmp_path: Path) -> None:
    monthly_result, manifest, index, candidate, gate, review, validation, packet, paths = _fixture(tmp_path)
    brief_path = Path(manifest.artifact_root) / "monthly_search_brief.json"
    brief_path.write_text("{}", encoding="utf-8")
    review.candidate_reviews.append(
        MonthlyModelCandidateReview(
            candidate_id="cand2",
            evidence_paths=[str(brief_path)],
            routing=MonthlyModelRouting.HYPOTHESIS_ONLY,
            recommendation="Keep as search-order context only.",
        )
    )
    validation.hypothesis_only_ids = ["cand2"]

    result = MonthlyEvidenceVerifier().verify(
        monthly_result=monthly_result,
        artifact_index=index,
        selected_candidates=[candidate],
        gate_reports=[gate],
        approval_packet=packet,
        run_manifest=manifest,
        model_review=review,
        model_validation=validation,
        model_review_validation_path=str(paths["model_validation"]),
    )

    assert result.verdict == MonthlyEvidenceVerdict.PASS
    assert not any(
        finding.code == "model_review_evidence_authority_violation"
        for finding in result.blocking_findings
    )
    assert str(brief_path) not in {check.path for check in result.evidence_path_checks}


def test_shadow_metadata_overclaim_fails_verification(tmp_path: Path) -> None:
    monthly_result, manifest, index, candidate, gate, review, validation, packet, paths = _fixture(tmp_path)
    packet.human_summary = "This has live approval evidence from a shadow run."
    packet.machine_readable_payload["deployment_metadata"] = {
        "emission_environment": "shadow",
        "metadata_source": "local_fixture",
    }

    result = MonthlyEvidenceVerifier().verify(
        monthly_result=monthly_result,
        artifact_index=index,
        selected_candidates=[candidate],
        gate_reports=[gate],
        approval_packet=packet,
        run_manifest=manifest,
        model_review=review,
        model_validation=validation,
        model_review_validation_path=str(paths["model_validation"]),
        deployment_metadata_blockers=["metadata_source contains non-approval token(s): local"],
    )

    assert result.verdict == MonthlyEvidenceVerdict.FAIL
    assert any(finding.code == "deployment_metadata_overstated" for finding in result.blocking_findings)


def test_deployment_blocker_overclaim_fails_without_shadow_marker(tmp_path: Path) -> None:
    monthly_result, manifest, index, candidate, gate, review, validation, packet, paths = _fixture(tmp_path)
    packet.human_summary = "This has live approval evidence."

    result = MonthlyEvidenceVerifier().verify(
        monthly_result=monthly_result,
        artifact_index=index,
        selected_candidates=[candidate],
        gate_reports=[gate],
        approval_packet=packet,
        run_manifest=manifest,
        model_review=review,
        model_validation=validation,
        model_review_validation_path=str(paths["model_validation"]),
        deployment_metadata_blockers=["metadata_source contains non-approval token(s): local"],
    )

    assert result.verdict == MonthlyEvidenceVerdict.FAIL
    assert any(finding.code == "deployment_metadata_overstated" for finding in result.blocking_findings)


def test_local_artifact_path_does_not_create_deployment_false_positive(tmp_path: Path) -> None:
    monthly_result, manifest, index, candidate, gate, review, validation, packet, paths = _fixture(tmp_path)
    local_dir = Path(manifest.artifact_root) / "local-cache"
    local_dir.mkdir()
    local_parity = local_dir / "replay_parity_report.json"
    local_parity.write_text("{}", encoding="utf-8")
    packet.human_summary = "This has live approval evidence."
    packet.artifact_paths.append(str(local_parity))
    packet.machine_readable_payload["approval_gate_evidence"].append(str(local_parity))

    result = MonthlyEvidenceVerifier().verify(
        monthly_result=monthly_result,
        artifact_index=index,
        selected_candidates=[candidate],
        gate_reports=[gate],
        approval_packet=packet,
        run_manifest=manifest,
        model_review=review,
        model_validation=validation,
        model_review_validation_path=str(paths["model_validation"]),
        deployment_metadata_blockers=[],
    )

    assert result.verdict == MonthlyEvidenceVerdict.PASS
    assert not any(finding.code == "deployment_metadata_overstated" for finding in result.blocking_findings)


def test_mismatched_scope_fails_verification(tmp_path: Path) -> None:
    monthly_result, manifest, index, candidate, gate, review, validation, packet, paths = _fixture(tmp_path)
    packet.run_id = "monthly-other"

    result = MonthlyEvidenceVerifier().verify(
        monthly_result=monthly_result,
        artifact_index=index,
        selected_candidates=[candidate],
        gate_reports=[gate],
        approval_packet=packet,
        run_manifest=manifest,
        model_review=review,
        model_validation=validation,
        model_review_validation_path=str(paths["model_validation"]),
    )

    assert result.verdict == MonthlyEvidenceVerdict.FAIL
    assert any(finding.code == "packet_scope_matches_monthly_result" for finding in result.blocking_findings)


def test_no_selected_candidates_do_not_require_verifier_success(tmp_path: Path) -> None:
    monthly_result, *_rest = _fixture(tmp_path)

    result = MonthlyEvidenceVerifier().verify(
        monthly_result=monthly_result,
        artifact_index=None,
        selected_candidates=[],
        gate_reports=[],
        approval_packet=None,
    )

    assert result.verdict == MonthlyEvidenceVerdict.PASS
    assert result.recommended_action.value == "no_action"


def test_soft_gate_gap_routes_to_human_review(tmp_path: Path) -> None:
    monthly_result, manifest, index, candidate, _gate, review, validation, packet, paths = _fixture(tmp_path)
    gate = MonthlyCandidateGateReport(
        candidate_id="cand1",
        checks=[
            MonthlyGateCheck(name="all_hard_gates", passed=True),
            MonthlyGateCheck(
                name="candidate_workspace_attempt",
                passed=False,
                severity=MonthlyGateSeverity.SOFT,
                reason="workspace metadata not supplied",
            ),
        ],
    )

    result = MonthlyEvidenceVerifier().verify(
        monthly_result=monthly_result,
        artifact_index=index,
        selected_candidates=[candidate],
        gate_reports=[gate],
        approval_packet=packet,
        run_manifest=manifest,
        model_review=review,
        model_validation=validation,
        model_review_validation_path=str(paths["model_validation"]),
    )

    assert result.verdict == MonthlyEvidenceVerdict.NEEDS_HUMAN_REVIEW
    assert result.recommended_action.value == "human_review"
    assert any(finding.code == "soft_gate_requires_human_review" for finding in result.non_blocking_findings)


def test_lightweight_outcome_cannot_be_final_material_gate_evidence(tmp_path: Path) -> None:
    monthly_result, manifest, index, candidate, gate, review, validation, packet, paths = _fixture(tmp_path)
    packet.machine_readable_payload["approval_gate_evidence"].append(
        str(tmp_path / "early_warning_outcome.json")
    )
    packet.machine_readable_payload["early_warning_outcome"] = {"source": "auto_outcome"}

    result = MonthlyEvidenceVerifier().verify(
        monthly_result=monthly_result,
        artifact_index=index,
        selected_candidates=[candidate],
        gate_reports=[gate],
        approval_packet=packet,
        run_manifest=manifest,
        model_review=review,
        model_validation=validation,
        model_review_validation_path=str(paths["model_validation"]),
    )

    assert result.verdict == MonthlyEvidenceVerdict.FAIL
    assert any(
        finding.code == "lightweight_outcome_not_final_material_evidence"
        for finding in result.blocking_findings
    )
