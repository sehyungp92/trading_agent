from __future__ import annotations

import json
from pathlib import Path

from trading_assistant.schemas.backtest_artifacts import (
    REQUIRED_BACKTEST_ARTIFACTS,
    BacktestArtifactIndex,
)
from trading_assistant.schemas.artifact_authority import ArtifactAuthority, artifact_type_from_path
from trading_assistant.schemas.monthly_candidates import MonthlyApprovalEvidencePacket
from trading_assistant.skills.artifact_authority_registry import ArtifactAuthorityRegistry


def test_registry_answers_approval_gate_eligibility() -> None:
    registry = ArtifactAuthorityRegistry.load()

    assert registry.may_satisfy_approval_gate("monthly_search_brief") is False
    assert registry.may_satisfy_approval_gate("memory_policies") is False
    assert registry.may_satisfy_approval_gate("replay_parity_report") is True
    assert registry.may_satisfy_approval_gate("approval_packet") is True
    assert registry.may_satisfy_approval_gate("/tmp/monthly_evidence_verification_cand1.json") is True
    assert registry.may_satisfy_approval_gate("runner_observability") is False


def test_monthly_search_brief_cannot_satisfy_approval_gate() -> None:
    registry = ArtifactAuthorityRegistry.load()
    packet = MonthlyApprovalEvidencePacket(
        candidate_id="cand1",
        run_id="monthly-bot1-strat1-2026-05",
        run_month="2026-05",
        bot_id="bot1",
        strategy_id="strat1",
        title="Misuse search brief",
        artifact_paths=[
            "/tmp/monthly_search_brief.json",
            "/tmp/monthly_validation_result.json",
            "/tmp/replay_parity_report.json",
            "/tmp/candidate_gate_report.json",
            "/tmp/model_review_validation.json",
        ],
        rollback_plan="restore incumbent",
        objective_deltas={"latest_month_oos": 0.1},
        data_coverage_status="authoritative",
        replay_parity_status="pass",
        machine_readable_payload={
            "approval_gate_evidence": ["/tmp/monthly_search_brief.json"],
        },
    )

    issues = registry.validate_approval_packet(packet)

    assert any(issue.am_row == "AM-06" for issue in issues)
    assert any("not eligible" in issue.message for issue in issues)


def test_registry_is_seeded_from_required_backtest_artifacts(tmp_path: Path) -> None:
    registry = ArtifactAuthorityRegistry.load()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    artifacts = {}
    for name in REQUIRED_BACKTEST_ARTIFACTS:
        path = artifact_root / name
        if name.endswith(".json"):
            path.write_text("{}", encoding="utf-8")
        else:
            path.write_text("", encoding="utf-8")
        artifacts[name] = str(path)
    index = BacktestArtifactIndex(
        run_id="monthly-bot1-strat1-2026-05",
        artifact_root=str(artifact_root),
        artifacts=artifacts,
    )

    required_types = {
        registry.get(name).source_contract
        for name in REQUIRED_BACKTEST_ARTIFACTS
        if registry.get(name) is not None
    }

    assert required_types == {"REQUIRED_BACKTEST_ARTIFACTS"}
    assert registry.validate_backtest_artifact_index(index) == []


def test_missing_required_backtest_artifacts_are_reported(tmp_path: Path) -> None:
    registry = ArtifactAuthorityRegistry.load()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    index = BacktestArtifactIndex(
        run_id="monthly-bot1-strat1-2026-05",
        artifact_root=str(artifact_root),
        artifacts={"runner_observability.json": str(artifact_root / "runner_observability.json")},
    )

    issues = registry.validate_backtest_artifact_index(index)

    assert issues
    assert any(issue.am_row == "AM-06" for issue in issues)
    assert all(issue.remediation for issue in issues)


def test_unknown_approval_packet_artifact_fails_validation() -> None:
    registry = ArtifactAuthorityRegistry.load()
    packet = {
        "candidate_id": "cand1",
        "run_id": "monthly-bot1-strat1-2026-05",
        "run_month": "2026-05",
        "bot_id": "bot1",
        "strategy_id": "strat1",
        "title": "Unknown artifact",
        "artifact_paths": [
            "/tmp/not_registered.foo",
            "/tmp/monthly_validation_result.json",
            "/tmp/replay_parity_report.json",
            "/tmp/candidate_gate_report.json",
            "/tmp/model_review_validation.json",
        ],
        "machine_readable_payload": {},
    }

    issues = registry.validate_approval_packet(packet)

    assert any(issue.artifact.endswith("not_registered.foo") for issue in issues)
    assert all(json.loads(json.dumps(issue.model_dump(mode="json"))) for issue in issues)


def test_required_gate_evidence_must_be_in_gate_paths() -> None:
    registry = ArtifactAuthorityRegistry.load()
    packet = {
        "candidate_id": "cand1",
        "run_id": "monthly-bot1-strat1-2026-05",
        "run_month": "2026-05",
        "bot_id": "bot1",
        "strategy_id": "strat1",
        "title": "Context-only gate artifacts",
        "artifact_paths": [
            "/tmp/monthly_validation_result.json",
            "/tmp/replay_parity_report.json",
            "/tmp/candidate_gate_report.json",
            "/tmp/model_review_validation.json",
        ],
        "machine_readable_payload": {
            "approval_gate_evidence": ["/tmp/monthly_evidence_verification_cand1.json"],
        },
    }

    issues = registry.validate_approval_packet(packet)

    missing = {issue.artifact for issue in issues if issue.am_row == "AM-09"}
    assert {
        "monthly_validation_result",
        "replay_parity_report",
        "candidate_gate_report",
        "model_review_validation",
    }.issubset(missing)
    assert "monthly_evidence_verification" not in missing
    assert "monthly_evidence_verification" not in registry.required_approval_gate_types(stage="pre_verifier")
    assert "monthly_evidence_verification" in registry.required_approval_gate_types(stage="final")
    assert registry.missing_required_approval_gate_types(packet, stage="pre_verifier") == [
        "candidate_gate_report",
        "model_review_validation",
        "monthly_validation_result",
        "replay_parity_report",
    ]


def test_exact_gate_artifacts_do_not_match_draft_or_backup_names() -> None:
    registry = ArtifactAuthorityRegistry.load()

    assert registry.get("/tmp/monthly_validation_result_old.json") is None
    assert registry.get("/tmp/replay_parity_report_backup.json") is None
    assert registry.get("/tmp/model_review_validation_draft.json") is None
    assert registry.get("/tmp/2026-04.coverage_manifest.json") is not None
    assert registry.get("/tmp/model_review_request.json") is not None
    assert registry.may_satisfy_approval_gate("/tmp/model_review_prompt.md") is False
    assert registry.may_satisfy_approval_gate("/tmp/monthly_evidence_verification_cand1.json") is True
    assert registry.may_satisfy_approval_gate("/tmp/approval_packet_cand1.json") is True


def test_learning_instrumentation_artifacts_are_registered_but_not_gate_evidence() -> None:
    registry = ArtifactAuthorityRegistry.load()

    sufficiency = registry.get("/tmp/learning_sufficiency_manifest.json")
    discovery = registry.get("/tmp/strategy_discovery_packet.json")

    assert artifact_type_from_path("/tmp/learning_sufficiency_manifest.json") == "learning_sufficiency_manifest"
    assert artifact_type_from_path("/tmp/strategy_discovery_packet.json") == "strategy_discovery_packet"
    assert sufficiency is not None
    assert sufficiency.authority == ArtifactAuthority.BINDING
    assert sufficiency.may_satisfy_approval_gate is False
    assert discovery is not None
    assert discovery.authority == ArtifactAuthority.DIAGNOSTICS_ONLY
    assert discovery.may_satisfy_approval_gate is False
    assert registry.may_satisfy_approval_gate("/tmp/learning_sufficiency_manifest.json") is False
    assert registry.may_satisfy_approval_gate("/tmp/strategy_discovery_packet.json") is False


def test_learning_instrumentation_artifacts_are_rejected_as_direct_approval_gates() -> None:
    registry = ArtifactAuthorityRegistry.load()
    packet = {
        "candidate_id": "cand1",
        "run_id": "monthly-bot1-strat1-2026-05",
        "run_month": "2026-05",
        "bot_id": "bot1",
        "strategy_id": "strat1",
        "title": "Sufficiency as context",
        "artifact_paths": [
            "/tmp/learning_sufficiency_manifest.json",
            "/tmp/strategy_discovery_packet.json",
            "/tmp/monthly_validation_result.json",
            "/tmp/replay_parity_report.json",
            "/tmp/candidate_gate_report.json",
            "/tmp/model_review_validation.json",
        ],
        "machine_readable_payload": {
            "approval_gate_evidence": [
                "/tmp/learning_sufficiency_manifest.json",
                "/tmp/strategy_discovery_packet.json",
            ],
        },
    }

    issues = registry.validate_approval_packet(packet)

    assert any(issue.artifact.endswith("learning_sufficiency_manifest.json") for issue in issues)
    assert any(issue.artifact.endswith("strategy_discovery_packet.json") for issue in issues)
    assert all("not eligible" in issue.message for issue in issues if issue.artifact.endswith(".json"))
