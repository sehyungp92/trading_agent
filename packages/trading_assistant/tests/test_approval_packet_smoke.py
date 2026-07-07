from __future__ import annotations

from pathlib import Path

from tests.test_monthly_evidence_verifier import _fixture
from trading_assistant.skills.artifact_authority_registry import ArtifactAuthorityRegistry
from trading_assistant.skills.monthly_evidence_verifier import MonthlyEvidenceVerifier


def test_fixture_approval_ready_routes_to_approval_but_never_deploys(tmp_path: Path) -> None:
    monthly_result, manifest, index, candidate, gate, review, validation, packet, paths = _fixture(tmp_path)

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

    assert result.recommended_action.value == "route_approval"
    assert "deploy" not in result.model_dump_json().lower()
    assert packet.approval_ready is False


def test_monthly_search_brief_misuse_smoke_fails_authority_validation(tmp_path: Path) -> None:
    *_fixture_values, packet, _paths = _fixture(tmp_path)
    packet.machine_readable_payload["approval_gate_evidence"] = [
        str(tmp_path / "monthly_search_brief.json")
    ]

    issues = ArtifactAuthorityRegistry.load().validate_approval_packet(packet)

    assert any(issue.am_row in {"AM-06", "AM-07"} for issue in issues)
