"""Independent deterministic verifier for monthly approval evidence."""

from __future__ import annotations

import json
from pathlib import Path

from trading_assistant.schemas.artifact_authority import artifact_type_from_path
from trading_assistant.schemas.backtest_artifacts import BacktestArtifactIndex
from trading_assistant.schemas.loop_contracts import LoopContract
from trading_assistant.schemas.monthly_artifact_contract import MonthlyVerifierInput
from trading_assistant.schemas.monthly_candidates import (
    MonthlyApprovalEvidencePacket,
    MonthlyCandidateGateReport,
    MonthlyImprovementCandidate,
)
from trading_assistant.schemas.monthly_evidence_verification import (
    ApprovalPacketCheck,
    AuthorityCheck,
    EvidencePathCheck,
    MonthlyEvidenceFinding,
    MonthlyEvidenceRecommendedAction,
    MonthlyEvidenceVerdict,
    MonthlyEvidenceVerification,
)
from trading_assistant.schemas.monthly_model_review import (
    MonthlyModelReview,
    MonthlyModelRouting,
    MonthlyModelValidationResult,
)
from trading_assistant.schemas.monthly_run_manifest import MonthlyRunManifest
from trading_assistant.schemas.monthly_validation import MonthlyValidationResult
from trading_assistant.skills.artifact_authority_registry import ArtifactAuthorityRegistry


class MonthlyEvidenceVerifier:
    """Read-only honesty and consistency verifier for approval-facing packets."""

    def __init__(self, registry: ArtifactAuthorityRegistry | None = None) -> None:
        self.registry = registry or ArtifactAuthorityRegistry.load()

    def verify_input(self, verifier_input: MonthlyVerifierInput) -> MonthlyEvidenceVerification:
        """Verify a contract-built monthly evidence view.

        The public ``verify(...)`` signature remains available while callers
        migrate to the monthly artifact contract.
        """
        return self.verify(**verifier_input.to_verify_kwargs())

    def verify(
        self,
        *,
        monthly_result: MonthlyValidationResult,
        artifact_index: BacktestArtifactIndex | None,
        selected_candidates: list[MonthlyImprovementCandidate],
        gate_reports: list[MonthlyCandidateGateReport],
        approval_packet: MonthlyApprovalEvidencePacket | None,
        run_manifest: MonthlyRunManifest | None = None,
        model_review: MonthlyModelReview | None = None,
        model_validation: MonthlyModelValidationResult | None = None,
        model_review_validation_path: str = "",
        deployment_metadata_blockers: list[str] | None = None,
        loop_contract: LoopContract | None = None,
    ) -> MonthlyEvidenceVerification:
        if approval_packet is None or not selected_candidates:
            return MonthlyEvidenceVerification(
                run_id=monthly_result.run_id,
                verdict=MonthlyEvidenceVerdict.PASS,
                recommended_action=MonthlyEvidenceRecommendedAction.NO_ACTION,
                non_blocking_findings=[
                    MonthlyEvidenceFinding(
                        code="no_selected_candidates",
                        severity="info",
                        message="No selected candidates require approval-facing verification.",
                    )
                ],
            )

        blocking: list[MonthlyEvidenceFinding] = []
        non_blocking: list[MonthlyEvidenceFinding] = []
        authority_checks: list[AuthorityCheck] = []
        packet_checks: list[ApprovalPacketCheck] = []

        candidate_ids = {candidate.candidate_id for candidate in selected_candidates}
        gate_by_candidate = {report.candidate_id: report for report in gate_reports}
        candidate_id = approval_packet.candidate_id
        candidate = next((item for item in selected_candidates if item.candidate_id == candidate_id), None)
        gate_report = gate_by_candidate.get(candidate_id)

        def block(code: str, message: str, paths: list[str] | None = None, remediation: str = "") -> None:
            blocking.append(MonthlyEvidenceFinding(
                code=code,
                message=message,
                evidence_paths=paths or [],
                remediation=remediation,
            ))

        def check_packet(name: str, passed: bool, message: str = "") -> None:
            packet_checks.append(ApprovalPacketCheck(name=name, passed=passed, message=message))
            if not passed:
                block(name, message)

        ids_match = (
            approval_packet.run_id == monthly_result.run_id
            and approval_packet.bot_id == monthly_result.bot_id
            and approval_packet.strategy_id == monthly_result.strategy_id
        )
        check_packet(
            "packet_scope_matches_monthly_result",
            ids_match,
            "approval packet run/bot/strategy IDs must match monthly result",
        )
        if run_manifest is not None:
            manifest_match = (
                run_manifest.run_id == monthly_result.run_id
                and run_manifest.bot_id == monthly_result.bot_id
                and run_manifest.strategy_id == monthly_result.strategy_id
            )
            check_packet(
                "run_manifest_scope_matches",
                manifest_match,
                "run manifest scope must match monthly result and packet",
            )
        if artifact_index is not None:
            expected_manifest_id = run_manifest.manifest_id if run_manifest is not None else ""
            for issue in self.registry.validate_backtest_artifact_index(
                artifact_index,
                expected_run_id=monthly_result.run_id,
                expected_manifest_id=expected_manifest_id,
                require_manifest_id=run_manifest is not None,
            ):
                block(
                    "artifact_index_invalid",
                    issue.message,
                    [issue.artifact],
                    issue.remediation,
                )

        if model_validation is None:
            block(
                "missing_model_review_validation",
                "valid monthly model-review validation is required before approval-facing routing",
                remediation="Write model_review_validation.json with MonthlyModelResponseValidator output.",
            )
        elif not model_validation.valid:
            block(
                "invalid_model_review_validation",
                "monthly model-review validation failed: "
                + "; ".join(issue.message for issue in model_validation.issues[:3]),
                [model_review_validation_path] if model_review_validation_path else [],
            )
        else:
            authority_checks.append(AuthorityCheck(
                name="model_review_validation_valid",
                passed=True,
                message="MonthlyModelResponseValidator passed before verifier.",
            ))

        if model_validation is not None:
            for actionable_id in model_validation.actionable_candidate_ids:
                if actionable_id not in candidate_ids:
                    block(
                        "actionable_model_review_without_selected_candidate",
                        f"actionable model-review item {actionable_id!r} is not a deterministic selected candidate",
                    )
            if candidate_id not in set(model_validation.actionable_candidate_ids):
                block(
                    "packet_candidate_not_actionable_in_model_review",
                    f"candidate {candidate_id!r} is not actionable in model-review validation",
                )
        if model_review is not None:
            for issue in self.registry.validate_model_review_evidence(
                _actionable_model_review_evidence_paths(
                    model_review,
                    model_validation=model_validation,
                    packet_candidate_id=candidate_id,
                )
            ):
                block(
                    "model_review_evidence_authority_violation",
                    issue.message,
                    [issue.artifact],
                    issue.remediation,
                )

        if gate_report is None:
            block("missing_candidate_gate_report", f"candidate {candidate_id!r} has no gate report")
        elif not gate_report.passed:
            block(
                "candidate_gates_failed",
                "verifier cannot route approval when deterministic candidate gates failed: "
                + "; ".join(gate_report.blocking_reasons[:3]),
            )
        else:
            for check in gate_report.checks:
                if check.passed or check.severity.value == "hard":
                    continue
                non_blocking.append(MonthlyEvidenceFinding(
                    code="soft_gate_requires_human_review",
                    severity="warning",
                    message=(
                        f"soft candidate gate {check.name!r} did not pass: "
                        f"{check.reason or 'manual review required'}"
                    ),
                    evidence_paths=check.evidence_paths,
                    remediation="Resolve the soft gate or review it explicitly before automated approval routing.",
                ))

        if candidate is None:
            block("packet_candidate_not_selected", f"packet candidate {candidate_id!r} is not selected")

        required_packet_fields = {
            "rollback_plan": approval_packet.rollback_plan,
            "objective_deltas": approval_packet.objective_deltas,
            "data_coverage_status": approval_packet.data_coverage_status,
            "replay_parity_status": approval_packet.replay_parity_status,
            "artifact_paths": approval_packet.artifact_paths,
            "model_review_validation_path": model_review_validation_path,
        }
        for field, value in required_packet_fields.items():
            passed = bool(value)
            check_packet(
                f"packet_has_{field}",
                passed,
                f"approval packet is missing {field}",
            )

        tiers = model_validation.approval_tiers if model_validation is not None else {}
        check_packet(
            "packet_has_human_approval_tier",
            bool(tiers.get(candidate_id)),
            "model-review validation must provide a human approval tier for the packet candidate",
        )

        evidence_checks = self._check_evidence_paths(
            approval_packet=approval_packet,
            model_review_evidence_paths=_actionable_model_review_evidence_paths(
                model_review,
                model_validation=model_validation,
                packet_candidate_id=candidate_id,
            ),
            model_review_validation_path=model_review_validation_path,
        )
        for check in evidence_checks:
            if not check.known:
                block(
                    "unknown_evidence_path",
                    f"unregistered evidence path: {check.path}",
                    [check.path],
                    "Register the artifact type or remove it from approval-facing evidence.",
                )
            elif not check.exists:
                block(
                    "missing_evidence_path",
                    f"evidence path does not exist: {check.path}",
                    [check.path],
                )

        authority_checks.extend(self._authority_checks(approval_packet))
        for check in authority_checks:
            if not check.passed:
                block(check.name, check.message)

        blockers = deployment_metadata_blockers or []
        if blockers and _packet_claims_live_deployment_evidence(approval_packet):
            block(
                "deployment_metadata_overstated",
                "approval packet overstates blocked deployment metadata as live approval evidence",
                remediation=(
                    "Keep deployment metadata blockers visible and suppress approval "
                    "until live metadata is authoritative."
                ),
            )

        if loop_contract is not None and loop_contract.authority.may_modify_live_bot_state:
            block(
                "loop_contract_grants_live_mutation",
                "monthly loop contract grants live bot mutation authority",
            )

        if blocking:
            verdict = MonthlyEvidenceVerdict.FAIL
            recommended = MonthlyEvidenceRecommendedAction.SUPPRESS_APPROVAL
        elif non_blocking:
            verdict = MonthlyEvidenceVerdict.NEEDS_HUMAN_REVIEW
            recommended = MonthlyEvidenceRecommendedAction.HUMAN_REVIEW
        else:
            verdict = MonthlyEvidenceVerdict.PASS
            recommended = MonthlyEvidenceRecommendedAction.ROUTE_APPROVAL
        return MonthlyEvidenceVerification(
            run_id=monthly_result.run_id,
            candidate_id=candidate_id,
            verdict=verdict,
            blocking_findings=blocking,
            non_blocking_findings=non_blocking,
            evidence_path_checks=evidence_checks,
            authority_checks=authority_checks,
            approval_packet_checks=packet_checks,
            recommended_action=recommended,
            source_payload={
                "artifact_index_path": monthly_result.artifact_index_path,
                "model_review_validation_path": model_review_validation_path,
                "selected_candidate_count": len(selected_candidates),
                "gate_report_count": len(gate_reports),
            },
        )

    def write(
        self,
        verification: MonthlyEvidenceVerification,
        artifact_root: Path,
        *,
        candidate_id: str,
    ) -> Path:
        artifact_root.mkdir(parents=True, exist_ok=True)
        safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in candidate_id)[:64] or "candidate"
        path = artifact_root / f"monthly_evidence_verification_{safe_id}.json"
        verification.verifier_artifact_path = str(path)
        path.write_text(verification.model_dump_json(indent=2), encoding="utf-8")
        return path

    def _check_evidence_paths(
        self,
        *,
        approval_packet: MonthlyApprovalEvidencePacket,
        model_review_evidence_paths: list[str],
        model_review_validation_path: str,
    ) -> list[EvidencePathCheck]:
        paths = [
            *approval_packet.artifact_paths,
            approval_packet.model_review_path,
            model_review_validation_path,
            approval_packet.approval_packet_path,
            *model_review_evidence_paths,
        ]
        checks: list[EvidencePathCheck] = []
        seen: set[str] = set()
        for raw in paths:
            path = str(raw or "").strip()
            if not path or path in seen:
                continue
            seen.add(path)
            entry = self.registry.get(path)
            checks.append(EvidencePathCheck(
                path=path,
                known=entry is not None,
                exists=Path(path).exists(),
                eligible_for_approval_gate=bool(entry and entry.may_satisfy_approval_gate),
                message=artifact_type_from_path(path),
            ))
        return checks

    def _authority_checks(self, approval_packet: MonthlyApprovalEvidencePacket) -> list[AuthorityCheck]:
        checks: list[AuthorityCheck] = []
        machine_payload = approval_packet.machine_readable_payload
        gate_paths = machine_payload.get("approval_gate_evidence", [])
        if isinstance(gate_paths, str):
            gate_paths = [gate_paths]
        if not isinstance(gate_paths, list):
            gate_paths = []
        for path in gate_paths:
            eligible = self.registry.may_satisfy_approval_gate(str(path))
            checks.append(AuthorityCheck(
                name="approval_gate_artifact_authority",
                passed=eligible,
                message=(
                    "" if eligible
                    else f"{path} cannot satisfy approval gates"
                ),
            ))
        for artifact_type in self.registry.missing_required_approval_gate_types(
            approval_packet,
            stage="pre_verifier",
        ):
            checks.append(AuthorityCheck(
                name="required_approval_gate_artifact_present",
                passed=False,
                message=(
                    f"{artifact_type}: approval packet is missing required "
                    "pre-verifier approval-gate artifact type"
                ),
            ))
        for path in approval_packet.artifact_paths:
            if artifact_type_from_path(path) == "monthly_search_brief":
                checks.append(AuthorityCheck(
                    name="monthly_search_brief_advisory_only",
                    passed=True,
                    message="monthly search brief may be context only, not gate evidence",
                ))
        serialized = json.dumps(machine_payload, default=str).lower()
        lightweight_markers = ("early_warning", "lightweight_outcome", "auto_outcome")
        if any(marker in serialized for marker in lightweight_markers):
            gate_serialized = json.dumps(gate_paths, default=str).lower()
            checks.append(AuthorityCheck(
                name="lightweight_outcome_not_final_material_evidence",
                passed=not any(marker in gate_serialized for marker in lightweight_markers),
                message="lightweight/early outcome evidence cannot finalize material strategy approval",
            ))
        return checks


def _packet_claims_live_deployment_evidence(packet: MonthlyApprovalEvidencePacket) -> bool:
    text = " ".join([
        packet.human_summary,
        packet.incumbent_validation_summary,
        packet.smoke_or_phased_evidence,
        json.dumps(packet.machine_readable_payload, default=str),
    ]).lower()
    overclaim_markers = (
        "live approval evidence",
        "live deployment evidence",
        "live_validated",
        "approval_ready deployment",
    )
    return any(marker in text for marker in overclaim_markers)


_ACTIONABLE_MODEL_REVIEW_ROUTES = {
    MonthlyModelRouting.SMOKE_REPAIR.value,
    MonthlyModelRouting.PHASED_AUTO.value,
    MonthlyModelRouting.EXPERIMENT.value,
    MonthlyModelRouting.MANUAL_DESIGN_REVIEW.value,
}


def _actionable_model_review_evidence_paths(
    model_review: MonthlyModelReview | None,
    *,
    model_validation: MonthlyModelValidationResult | None,
    packet_candidate_id: str,
) -> list[str]:
    if model_review is None:
        return []
    actionable_ids = {
        str(item)
        for item in (
            model_validation.actionable_candidate_ids
            if model_validation is not None else []
        )
        if str(item)
    }
    if packet_candidate_id:
        actionable_ids.add(packet_candidate_id)
    paths: list[str] = []
    for candidate in model_review.candidate_reviews:
        if candidate.candidate_id in actionable_ids or (
            model_validation is None
            and candidate.routing.value in _ACTIONABLE_MODEL_REVIEW_ROUTES
        ):
            paths.extend(candidate.evidence_paths)
    for proposal in model_review.structural_proposals:
        proposal_id = _structural_proposal_id(proposal)
        route = str(getattr(proposal, "routing", "") or "").strip().lower()
        if proposal_id in actionable_ids or (
            model_validation is None and route in _ACTIONABLE_MODEL_REVIEW_ROUTES
        ):
            paths.extend(getattr(proposal, "evidence_paths", []) or [])
    return [path for path in paths if str(path)]


def _structural_proposal_id(proposal: object) -> str:
    return str(
        getattr(proposal, "hypothesis_id", None)
        or getattr(proposal, "linked_suggestion_id", None)
        or getattr(proposal, "title", "")
        or ""
    )
