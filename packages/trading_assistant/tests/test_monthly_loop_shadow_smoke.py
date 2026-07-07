from __future__ import annotations

from pathlib import Path

from tests.test_monthly_evidence_verifier import _fixture
from trading_assistant.schemas.monthly_evidence_verification import MonthlyEvidenceVerdict
from trading_assistant.skills.monthly_evidence_verifier import MonthlyEvidenceVerifier


def test_shadow_metadata_smoke_suppresses_approval_without_live_access(tmp_path: Path) -> None:
    monthly_result, manifest, index, candidate, gate, review, validation, packet, paths = _fixture(tmp_path)
    packet.human_summary = "live approval evidence from shadow metadata"
    packet.machine_readable_payload["deployment_metadata"] = {"emission_environment": "shadow"}

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
        deployment_metadata_blockers=["emission_environment contains non-approval token(s): shadow"],
    )

    assert result.verdict == MonthlyEvidenceVerdict.FAIL
    assert result.recommended_action.value == "suppress_approval"
