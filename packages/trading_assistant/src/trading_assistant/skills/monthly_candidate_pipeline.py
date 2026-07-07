"""Monthly candidate ingestion, deterministic gates, and approval packets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from trading_assistant.analysis.monthly_model_response_parser import parse_monthly_model_review
from trading_assistant.analysis.monthly_model_response_validator import MonthlyModelResponseValidator
from trading_assistant.orchestrator.learning_sufficiency_audit import (
    CAPABILITY_REQUIREMENTS,
    runtime_source_authority_for_checks,
)
from trading_assistant.schemas.approval import ApprovalRequest, RepoRiskTier
from trading_assistant.schemas.backtest_artifacts import BacktestArtifactIndex
from trading_assistant.schemas.decision_parity import DecisionParityReport
from trading_assistant.schemas.discovery import StrategyDiscoveryPacket
from trading_assistant.schemas.market_data_manifest import MarketDataManifest
from trading_assistant.schemas.learning_sufficiency import (
    CoverageCheck,
    LearningCapabilityAuthority,
    LearningSufficiencyManifest,
)
from trading_assistant.schemas.monthly_artifact_contract import MonthlyVerifierInput
from trading_assistant.schemas.monthly_candidates import (
    MonthlyApprovalEvidencePacket,
    MonthlyCandidateGateReport,
    MonthlyCandidateProcessingResult,
    MonthlyCandidateSource,
    MonthlyGateCheck,
    MonthlyGateSeverity,
    MonthlyImprovementCandidate,
    MonthlyRiskClassification,
)
from trading_assistant.schemas.monthly_model_review import MonthlyModelReview, MonthlyModelValidationResult
from trading_assistant.schemas.monthly_run_manifest import MonthlyRunManifest
from trading_assistant.schemas.monthly_validation import MonthlyValidationResult, MonthlyValidationStatus
from trading_assistant.schemas.proposal_ledger import (
    ProposalCandidate,
    ProposalEvaluation,
    ProposalKind,
    ProposalSource,
)
from trading_assistant.schemas.repo_changes import ChangeKind, FileChange
from trading_assistant.schemas.replay_parity import ReplayParityReport
from trading_assistant.schemas.strategy_plugin_contract import StrategyPluginContract
from trading_assistant.schemas.strategy_change_ledger import StrategyChangeRecord, StrategyChangeRecordType
from trading_assistant.schemas.telemetry_manifest import TelemetryEligibility, TelemetryManifest
from trading_assistant.skills.monthly_artifact_contract import MonthlyArtifactContract
from trading_assistant.skills.monthly_deployment_metadata import deployment_metadata_errors
from trading_assistant.skills.monthly_evidence_verifier import MonthlyEvidenceVerifier
from trading_assistant.skills.outcome_prior_store import OutcomePriorStore
from trading_assistant.skills.proposal_ledger import make_proposal_id
from trading_assistant.skills.search_allocation_policy import SearchAllocationPolicy
from trading_assistant.skills.strategy_discovery_packet_builder import (
    MIN_DISCOVERY_AFTER_COST_ESTIMATE,
    MIN_DISCOVERY_CLUSTER_COUNT,
)

_APPROVAL_STATUSES = {
    MonthlyValidationStatus.REPAIR,
    MonthlyValidationStatus.ROLLBACK,
    MonthlyValidationStatus.EXPERIMENT,
    MonthlyValidationStatus.QUARANTINE,
}

class MonthlyCandidatePipeline:
    """Consumes backtest candidate artifacts and records approval-ready packets."""

    def __init__(
        self,
        *,
        approval_tracker: object | None = None,
        proposal_ledger: object | None = None,
        strategy_change_ledger: object | None = None,
        outcome_prior_store: OutcomePriorStore | None = None,
        evidence_verifier: MonthlyEvidenceVerifier | None = None,
        require_evidence_verifier: bool = True,
        min_trade_count: int = 10,
        max_outlier_win_concentration: float = 0.40,
    ) -> None:
        self.approval_tracker = approval_tracker
        self.proposal_ledger = proposal_ledger
        self.strategy_change_ledger = strategy_change_ledger
        self.outcome_prior_store = outcome_prior_store
        self.search_allocation_policy = (
            SearchAllocationPolicy(outcome_prior_store)
            if outcome_prior_store is not None else None
        )
        self.evidence_verifier = evidence_verifier or MonthlyEvidenceVerifier()
        self.require_evidence_verifier = require_evidence_verifier
        self.min_trade_count = min_trade_count
        self.max_outlier_win_concentration = max_outlier_win_concentration

    def process(
        self,
        *,
        monthly_result: MonthlyValidationResult,
        artifact_index: BacktestArtifactIndex,
        coverage: MarketDataManifest | None,
        telemetry: TelemetryManifest | None = None,
        parity_report: ReplayParityReport | None,
        artifact_root: Path,
        monthly_result_path: Path,
        shadow: bool,
        model_review_path: str = "",
    ) -> MonthlyCandidateProcessingResult:
        artifact_root = Path(artifact_root)
        artifact_root.mkdir(parents=True, exist_ok=True)
        if monthly_result.repair_required:
            summary_path = artifact_root / "candidate_generation_summary.json"
            result = MonthlyCandidateProcessingResult(
                run_id=monthly_result.run_id,
                run_month=monthly_result.run_month,
                bot_id=monthly_result.bot_id,
                strategy_id=monthly_result.strategy_id,
                selected_candidates=[],
                rejected_candidates=[],
                gate_reports=[],
                approval_packets=[],
                approval_request_ids=[],
                gate_passed_candidate_count=0,
                approval_ready_candidate_count=0,
                candidate_summary_path=str(summary_path),
                gate_report_path="",
                approval_packet_paths=[],
                model_review_path=model_review_path,
            )
            summary_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            return result
        run_manifest = _load_run_manifest(monthly_result.run_manifest_path)
        artifact_contract = MonthlyArtifactContract.from_index(
            artifact_index,
            manifest=run_manifest,
        )
        selected = artifact_contract.load_selected_candidates(
            bot_id=monthly_result.bot_id,
            strategy_id=monthly_result.strategy_id,
        )
        for candidate in selected:
            artifact_contract.normalize_candidate_paths(candidate)
        if self.search_allocation_policy is not None:
            selected = self.search_allocation_policy.order_candidates(selected)
        rejected = artifact_contract.load_rejected_candidates()
        model_validation_path = ""
        model_review: MonthlyModelReview | None = None
        model_validation: MonthlyModelValidationResult | None = None
        if model_review_path:
            allowed_evidence = _dedupe([
                *monthly_result.evidence_paths,
                *[
                    path
                    for candidate in selected
                    for path in [*candidate.evidence_paths, *candidate.artifact_paths]
                ],
            ])
            model_review, model_validation, model_validation_path = self._load_and_validate_model_review(
                model_review_path=model_review_path,
                monthly_result=monthly_result,
                allowed_evidence_paths=allowed_evidence,
            )
        gate_reports: list[MonthlyCandidateGateReport] = []
        packets: list[MonthlyApprovalEvidencePacket] = []
        request_ids: list[str] = []
        packet_paths: list[str] = []
        proposal_ids: list[str] = []
        evidence_verification_paths: list[str] = []
        evidence_verification_verdicts: dict[str, str] = {}
        gate_report_path = artifact_root / "candidate_gate_report.json"
        candidate_summary_path = artifact_root / "candidate_generation_summary.json"
        deployment_metadata_blockers = deployment_metadata_errors(
            run_manifest,
            missing_reason="deployment metadata missing for approval evidence",
        )

        for candidate in selected:
            gate_report = self.evaluate_candidate(
                candidate=candidate,
                monthly_result=monthly_result,
                artifact_index=artifact_index,
                artifact_contract=artifact_contract,
                coverage=coverage,
                telemetry=telemetry,
                parity_report=parity_report,
                model_validation=model_validation,
            )
            gate_reports.append(gate_report)
            packet = self.build_packet(
                candidate=candidate,
                gate_report=gate_report,
                monthly_result=monthly_result,
                coverage=coverage,
                telemetry=telemetry,
                parity_report=parity_report,
                monthly_result_path=monthly_result_path,
                model_review_path=model_review_path,
                model_validation=model_validation,
                model_review_validation_path=model_validation_path,
                candidate_gate_report_path=str(gate_report_path),
                artifact_contract=artifact_contract,
            )
            packet_path = artifact_root / f"approval_packet_{_safe_id(candidate.candidate_id)}.json"
            packet.approval_packet_path = str(packet_path)
            proposal_id = self._proposal_id(candidate, monthly_result)
            packet.proposal_id = proposal_id
            packet.suggestion_id = proposal_id
            proposal_ids = _dedupe([*proposal_ids, proposal_id])
            gate_report_path.write_text(
                json.dumps([report.model_dump(mode="json") for report in gate_reports], indent=2),
                encoding="utf-8",
            )
            packet_path.write_text(packet.model_dump_json(indent=2), encoding="utf-8")
            verifier_input = artifact_contract.verifier_input(
                candidate.candidate_id,
                monthly_result=monthly_result,
                selected_candidates=selected,
                gate_reports=[*gate_reports],
                approval_packet=packet,
                run_manifest=run_manifest,
                model_review=model_review,
                model_validation=model_validation,
                model_review_validation_path=model_validation_path,
                deployment_metadata_blockers=deployment_metadata_blockers,
            )
            verification = self._verify_evidence(verifier_input)
            verification_path = self.evidence_verifier.write(
                verification,
                artifact_root,
                candidate_id=candidate.candidate_id,
            )
            packet.evidence_verification_path = str(verification_path)
            packet.evidence_verification_id = verification.verification_id
            packet.evidence_verification_verdict = verification.verdict.value
            packet.artifact_paths = _dedupe([*packet.artifact_paths, str(verification_path)])
            gate_evidence = packet.machine_readable_payload.get("approval_gate_evidence", [])
            if not isinstance(gate_evidence, list):
                gate_evidence = [str(gate_evidence)] if gate_evidence else []
            packet.machine_readable_payload["approval_gate_evidence"] = _dedupe([
                *[str(path) for path in gate_evidence],
                str(verification_path),
            ])
            packet.machine_readable_payload["monthly_evidence_verification"] = verification.model_dump(mode="json")
            evidence_verification_paths.append(str(verification_path))
            evidence_verification_verdicts[candidate.candidate_id] = verification.verdict.value
            verifier_passed = verification.verdict.value == "pass"
            verifier_review_required = verification.verdict.value == "needs_human_review"
            verifier_allows_approval = verifier_passed or not self.require_evidence_verifier
            final_packet_issues = artifact_contract.registry.validate_approval_packet(packet)
            final_packet_issue_reasons = [
                f"{issue.artifact}: {issue.message}"
                for issue in final_packet_issues
            ]

            if (
                gate_report.passed
                and verifier_allows_approval
                and not final_packet_issues
                and not shadow
                and self.approval_tracker is not None
            ):
                request = self._build_approval_request(packet, candidate)
                strategy_record_id = self._record_strategy_change(candidate, packet, request)
                packet.request_id = request.request_id
                packet.strategy_change_record_id = strategy_record_id
                request.strategy_change_record_id = strategy_record_id
                self.approval_tracker.create_request(request)
                request_ids.append(request.request_id)
                packet.approval_ready = True
            else:
                packet.approval_ready = False
                packet.approval_suppressed_reasons = self._suppression_reasons(
                    gate_report,
                    shadow=shadow,
                    approval_tracker_present=self.approval_tracker is not None,
                    verifier_verdict=verification.verdict.value if self.require_evidence_verifier else "",
                    verifier_review_required=verifier_review_required,
                    final_packet_issues=final_packet_issue_reasons,
                )

            self._record_proposal(
                candidate,
                packet,
                gate_report,
                monthly_result,
                verification=verification,
            )
            packet_path.write_text(packet.model_dump_json(indent=2), encoding="utf-8")
            packet_paths.append(str(packet_path))
            packets.append(packet)

        gate_report_path.write_text(
            json.dumps([report.model_dump(mode="json") for report in gate_reports], indent=2),
            encoding="utf-8",
        )
        result = MonthlyCandidateProcessingResult(
            run_id=monthly_result.run_id,
            run_month=monthly_result.run_month,
            bot_id=monthly_result.bot_id,
            strategy_id=monthly_result.strategy_id,
            selected_candidates=selected,
            rejected_candidates=rejected,
            gate_reports=gate_reports,
            approval_packets=packets,
            approval_request_ids=request_ids,
            gate_passed_candidate_count=sum(1 for report in gate_reports if report.passed),
            approval_ready_candidate_count=len(request_ids),
            candidate_summary_path=str(candidate_summary_path),
            gate_report_path=str(gate_report_path),
            approval_packet_paths=packet_paths,
            proposal_ids=proposal_ids,
            model_review_path=model_review_path,
            model_review_validation_path=model_validation_path,
            evidence_verification_paths=evidence_verification_paths,
            evidence_verification_verdicts=evidence_verification_verdicts,
            model_review_valid=model_validation.valid if model_validation is not None else None,
            model_review_issues=(
                [issue.message for issue in model_validation.issues]
                if model_validation is not None else []
            ),
        )
        candidate_summary_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result

    def _verify_evidence(self, verifier_input: MonthlyVerifierInput) -> Any:
        verify_input = getattr(self.evidence_verifier, "verify_input", None)
        if callable(verify_input):
            return verify_input(verifier_input)
        return self.evidence_verifier.verify(**verifier_input.to_verify_kwargs())

    def evaluate_candidate(
        self,
        *,
        candidate: MonthlyImprovementCandidate,
        monthly_result: MonthlyValidationResult,
        artifact_index: BacktestArtifactIndex | None = None,
        artifact_contract: MonthlyArtifactContract | None = None,
        coverage: MarketDataManifest | None = None,
        telemetry: TelemetryManifest,
        parity_report: ReplayParityReport | None,
        model_validation: MonthlyModelValidationResult | None = None,
    ) -> MonthlyCandidateGateReport:
        contract = artifact_contract or (
            MonthlyArtifactContract.from_index(artifact_index)
            if artifact_index is not None else None
        )
        candidate_paths = [*candidate.evidence_paths, *candidate.artifact_paths]
        path_contract = contract or MonthlyArtifactContract(artifact_root=Path.cwd())
        existing_candidate_paths = path_contract.existing_paths(candidate_paths)
        checks = [
            MonthlyGateCheck(
                name="monthly_status_allows_candidate",
                passed=monthly_result.status in _APPROVAL_STATUSES and not monthly_result.blocking_reasons,
                reason=(
                    f"monthly status {monthly_result.status.value} does not allow approval-ready candidates"
                    if monthly_result.status not in _APPROVAL_STATUSES
                    else "; ".join(monthly_result.blocking_reasons)
                ),
                evidence_paths=[monthly_result.run_manifest_path],
            ),
            MonthlyGateCheck(
                name="candidate_decision_allows_action",
                passed=candidate.decision.value in {"repair", "rollback", "experiment"},
                reason=(
                    ""
                    if candidate.decision.value in {"repair", "rollback", "experiment"}
                    else f"candidate decision is {candidate.decision.value}"
                ),
                evidence_paths=candidate.evidence_paths,
            ),
            MonthlyGateCheck(
                name="supported_candidate_source",
                passed=candidate.source in {
                    MonthlyCandidateSource.SMOKE_REPAIR,
                    MonthlyCandidateSource.PHASED_AUTO,
                },
                reason=(
                    ""
                    if candidate.source in {
                        MonthlyCandidateSource.SMOKE_REPAIR,
                        MonthlyCandidateSource.PHASED_AUTO,
                    }
                    else f"unsupported candidate source: {candidate.source.value}"
                ),
                evidence_paths=candidate.evidence_paths,
            ),
            self._runner_contract_gate(candidate),
            self._candidate_lineage_gate(candidate, monthly_result, artifact_index),
            self._candidate_workspace_gate(candidate),
            MonthlyGateCheck(
                name="candidate_evidence_paths",
                passed=bool(existing_candidate_paths),
                reason=(
                    ""
                    if existing_candidate_paths
                    else "candidate lacks existing replay evidence paths"
                ),
                evidence_paths=candidate_paths,
            ),
            self._candidate_artifact_containment_gate(candidate, artifact_contract=contract),
            MonthlyGateCheck(
                name="market_data_coverage",
                passed=coverage is not None and coverage.usable_for_authoritative_validation,
                reason="" if coverage is not None and coverage.usable_for_authoritative_validation else "market data is not authoritative",
                evidence_paths=[monthly_result.market_data_manifest_path],
            ),
            MonthlyGateCheck(
                name="telemetry_lineage",
                passed=(
                    telemetry is not None
                    and telemetry.authoritative_eligibility == TelemetryEligibility.AUTHORITATIVE
                ),
                reason=(
                    ""
                    if telemetry is not None
                    and telemetry.authoritative_eligibility == TelemetryEligibility.AUTHORITATIVE
                    else (
                        f"telemetry eligibility is {telemetry.authoritative_eligibility.value}"
                        if telemetry is not None else "telemetry manifest unavailable"
                    )
                ),
                evidence_paths=[monthly_result.telemetry_manifest_path],
            ),
            *self._learning_sufficiency_gates(candidate, monthly_result),
            self._new_strategy_discovery_gate(candidate, monthly_result),
            MonthlyGateCheck(
                name="replay_parity",
                passed=parity_report is not None and parity_report.eligible_for_authoritative_validation,
                reason=(
                    ""
                    if parity_report is not None and parity_report.eligible_for_authoritative_validation
                    else "replay parity is missing or not authoritative"
                ),
                evidence_paths=[monthly_result.replay_parity_path],
            ),
            self._approval_payload_gate(candidate),
            self._bool_gate(
                candidate,
                name="no_leakage",
                keys=("leakage_passed", "no_leakage", "fold_leakage_passed"),
                missing_reason="leakage check evidence is missing",
            ),
            self._improvement_gate(candidate),
            self._calibration_gate(candidate),
            self._trade_count_gate(candidate),
            self._bool_gate(
                candidate,
                name="realistic_costs",
                keys=("cost_gate_passed", "realistic_costs_passed", "slippage_cost_gate_passed"),
                missing_reason="realistic fee/slippage/funding evidence is missing",
            ),
            self._drawdown_gate(candidate),
            self._outlier_gate(candidate),
            self._bool_gate(
                candidate,
                name="risk_constraints",
                keys=("risk_constraints_passed", "portfolio_risk_constraints_passed"),
                missing_reason="portfolio/risk-constraint evidence is missing",
            ),
            self._phase_support_gate(candidate),
            self._decision_parity_gate(candidate, monthly_result, artifact_contract=contract),
            self._strategy_plugin_contract_gate(candidate, monthly_result),
            self._outcome_prior_gate(candidate),
            self._model_review_gate(candidate, model_validation),
        ]
        return MonthlyCandidateGateReport(
            candidate_id=candidate.candidate_id,
            source=candidate.source,
            checks=checks,
            objective_version=candidate.objective_version,
            effective_objective_version=candidate.effective_objective_version,
            objective_profile_id=candidate.objective_profile_id,
        )

    def build_packet(
        self,
        *,
        candidate: MonthlyImprovementCandidate,
        gate_report: MonthlyCandidateGateReport,
        monthly_result: MonthlyValidationResult,
        coverage: MarketDataManifest | None,
        telemetry: TelemetryManifest,
        parity_report: ReplayParityReport | None,
        monthly_result_path: Path,
        model_review_path: str = "",
        model_validation: MonthlyModelValidationResult | None = None,
        model_review_validation_path: str = "",
        candidate_gate_report_path: str = "",
        artifact_contract: MonthlyArtifactContract | None = None,
    ) -> MonthlyApprovalEvidencePacket:
        artifact_paths = (
            artifact_contract.approval_packet_artifact_paths(
                monthly_result=monthly_result,
                candidate=candidate,
                monthly_result_path=monthly_result_path,
                candidate_gate_report_path=candidate_gate_report_path,
                model_review_path=model_review_path,
                model_review_validation_path=model_review_validation_path,
            )
            if artifact_contract is not None else _dedupe([
                *monthly_result.evidence_paths,
                str(monthly_result_path),
                monthly_result.artifact_index_path,
                monthly_result.replay_parity_path,
                candidate_gate_report_path,
                model_review_path,
                model_review_validation_path,
                *candidate.evidence_paths,
                *candidate.artifact_paths,
            ])
        )
        approval_evidence = (
            artifact_contract.approval_gate_evidence(
                candidate.candidate_id,
                candidate=candidate,
                monthly_result_path=monthly_result_path,
                replay_parity_path=monthly_result.replay_parity_path,
                candidate_gate_report_path=candidate_gate_report_path,
                model_review_validation_path=model_review_validation_path,
            ).approval_gate_evidence
            if artifact_contract is not None else _dedupe([
                str(monthly_result_path),
                monthly_result.replay_parity_path,
                candidate_gate_report_path,
                model_review_validation_path,
                *candidate.evidence_paths,
            ])
        )
        latest_delta = _candidate_float(candidate, "latest_month_oos_delta", "latest_month_objective_delta")
        calibration_delta = _candidate_float(candidate, "calibration_objective_delta", "calibration_delta")
        objective_ref = candidate.effective_objective_version or candidate.objective_version
        profile_ref = (
            f", profile={candidate.objective_profile_id}"
            if candidate.objective_profile_id else ""
        )
        human_summary = (
            f"{candidate.source.value} candidate {candidate.candidate_id} for "
            f"{monthly_result.bot_id}/{monthly_result.strategy_id}: "
            f"objective={objective_ref}{profile_ref}, "
            f"objective_delta={candidate.objective_delta:+.4f}, "
            f"gates={'pass' if gate_report.passed else 'fail'}."
        )
        return MonthlyApprovalEvidencePacket(
            candidate_id=candidate.candidate_id,
            run_id=monthly_result.run_id,
            run_month=monthly_result.run_month,
            bot_id=monthly_result.bot_id,
            strategy_id=monthly_result.strategy_id,
            strategy_change_record_id=monthly_result.strategy_change_record_id,
            title=candidate.title,
            reason_for_change=monthly_result.gap_attribution.summary or candidate.description,
            incumbent_validation_summary=f"Monthly status: {monthly_result.status.value}",
            smoke_or_phased_evidence=f"{candidate.source.value}:{candidate.family}",
            objective_deltas={
                **candidate.objective_deltas,
                "candidate_objective_delta": candidate.objective_delta,
            },
            objective_version=candidate.objective_version,
            effective_objective_version=candidate.effective_objective_version,
            objective_profile_id=candidate.objective_profile_id,
            score_component_cap=candidate.score_component_cap,
            latest_month_behavior=(
                f"latest OOS delta {latest_delta:+.4f}"
                if latest_delta is not None else "latest OOS improvement evidence supplied by gate inputs"
            ),
            calibration_support=(
                f"calibration delta {calibration_delta:+.4f}"
                if calibration_delta is not None else "calibration support evidence supplied by gate inputs"
            ),
            data_coverage_status=(
                f"coverage={coverage.coverage_ratio:.3f}, usable={coverage.usable_for_authoritative_validation}"
                if coverage else "market data manifest unavailable"
            ),
            replay_parity_status=parity_report.status.value if parity_report else "missing",
            risk_classification=candidate.risk_classification,
            rollback_plan=candidate.rollback_plan,
            artifact_paths=artifact_paths,
            model_review_path=model_review_path,
            model_review_validation_path=model_review_validation_path,
            human_summary=human_summary,
            machine_readable_payload={
                "objective": {
                    "objective_version": candidate.objective_version,
                    "effective_objective_version": candidate.effective_objective_version,
                    "immutable_objective_version": candidate.immutable_objective_version,
                    "objective_profile_id": candidate.objective_profile_id,
                    "objective_profile_family": candidate.objective_profile_family,
                    "objective_profile_scope": candidate.objective_profile_scope,
                    "score_component_cap": candidate.score_component_cap,
                    "immutable_score": candidate.immutable_score,
                },
                "candidate": candidate.model_dump(mode="json"),
                "gate_report": gate_report.model_dump(mode="json"),
                "monthly_validation": monthly_result.model_dump(mode="json"),
                "telemetry_eligibility": (
                    telemetry.authoritative_eligibility.value
                    if telemetry is not None else "unavailable"
                ),
                "model_review_validation": (
                    model_validation.model_dump(mode="json")
                    if model_validation is not None else None
                ),
                "model_review_validation_path": model_review_validation_path,
                "approval_gate_evidence": approval_evidence,
            },
            approval_ready=False,
        )

    def _load_and_validate_model_review(
        self,
        *,
        model_review_path: str,
        monthly_result: MonthlyValidationResult,
        allowed_evidence_paths: list[str],
    ) -> tuple[MonthlyModelReview | None, MonthlyModelValidationResult | None, str]:
        path = Path(model_review_path)
        if not path.exists():
            return None, None, ""
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None, None, ""
        review = _parse_model_review_text(text)
        validation = MonthlyModelResponseValidator().validate(
            review,
            allowed_evidence_paths=[*allowed_evidence_paths, str(path)],
            expected_run_id=monthly_result.run_id,
            expected_bot_id=monthly_result.bot_id,
            expected_strategy_id=monthly_result.strategy_id,
        )
        validation_path = path.with_name("model_review_validation.json")
        validation_path.write_text(validation.model_dump_json(indent=2), encoding="utf-8")
        return review, validation, str(validation_path)

    def _learning_sufficiency_gate(
        self,
        candidate: MonthlyImprovementCandidate,
        monthly_result: MonthlyValidationResult,
    ) -> MonthlyGateCheck:
        gates = self._learning_sufficiency_gates(candidate, monthly_result)
        for gate in gates:
            if gate.name == "learning_capability_authority":
                return gate
        return gates[0]

    def _learning_sufficiency_gates(
        self,
        candidate: MonthlyImprovementCandidate,
        monthly_result: MonthlyValidationResult,
    ) -> list[MonthlyGateCheck]:
        required_capabilities = _required_learning_capabilities(candidate.change_kind)
        manifest_path = str(monthly_result.learning_sufficiency_manifest_path or "").strip()
        evidence_path = [manifest_path] if manifest_path else []
        gate_names = [
            "learning_sufficiency_manifest_present",
            "learning_capability_authority",
            "causal_join_completeness",
            "denominator_coverage",
            "after_cost_outcome_coverage",
            "proposal_trace_coverage",
            "counterfactual_backfill_coverage",
            "runtime_evidence_coverage",
            "instrumentation_gap_impact",
        ]
        if not required_capabilities:
            reason = "candidate change kind does not require capability-specific learning authority"
            return [
                MonthlyGateCheck(name=name, passed=True, reason=reason, evidence_paths=evidence_path)
                for name in gate_names
            ]

        manifest, manifest_error = _load_learning_sufficiency_manifest(manifest_path)
        present_gate = MonthlyGateCheck(
            name="learning_sufficiency_manifest_present",
            passed=manifest is not None,
            reason=manifest_error,
            evidence_paths=evidence_path,
        )
        if manifest is None:
            missing_reason = manifest_error or "learning sufficiency manifest is unavailable"
            return [
                present_gate,
                *[
                    MonthlyGateCheck(name=name, passed=False, reason=missing_reason, evidence_paths=evidence_path)
                    for name in gate_names[1:]
                ],
            ]

        required_checks = _required_learning_checks(required_capabilities)
        all_checks = _manifest_checks(manifest)
        capability_blockers: list[str] = []
        capability_evidence = [manifest_path]
        for capability in required_capabilities:
            status = manifest.capability_status.get(capability)
            if status is None:
                capability_blockers.append(f"{capability}:missing")
                continue
            capability_evidence.extend(status.evidence_paths)
            if status.status != LearningCapabilityAuthority.LEARNING_AUTHORITATIVE:
                details = ", ".join(status.blocking_reasons or status.blocking_checks)
                suffix = f" ({details})" if details else ""
                capability_blockers.append(f"{capability}:{status.status.value}{suffix}")
        gates: list[MonthlyGateCheck] = [
            present_gate,
            MonthlyGateCheck(
                name="learning_capability_authority",
                passed=not capability_blockers,
                reason=(
                    ""
                    if not capability_blockers
                    else "required learning capability is not authoritative: "
                    + "; ".join(capability_blockers)
                ),
                evidence_paths=_dedupe(capability_evidence),
            ),
            _coverage_gate(
                "causal_join_completeness",
                required_checks,
                all_checks,
                {
                    "decision_to_trade_join",
                    "decision_to_order_join",
                    "order_to_fill_join",
                    "risk_portfolio_join",
                },
                manifest_path,
            ),
            _coverage_gate(
                "denominator_coverage",
                required_checks,
                all_checks,
                {"denominator_coverage"},
                manifest_path,
            ),
            _coverage_gate(
                "after_cost_outcome_coverage",
                required_checks,
                all_checks,
                {"after_cost_coverage"},
                manifest_path,
            ),
            _coverage_gate(
                "proposal_trace_coverage",
                required_checks,
                all_checks,
                {"proposal_trace_coverage"},
                manifest_path,
            ),
            _coverage_gate(
                "counterfactual_backfill_coverage",
                required_checks,
                all_checks,
                {"counterfactual_coverage"},
                manifest_path,
            ),
            _coverage_gate(
                "runtime_evidence_coverage",
                required_checks,
                all_checks,
                {
                    "trade_outcome_lineage",
                    "missed_opportunity_lineage",
                    "filter_decision_coverage",
                    "orderbook_context_coverage",
                    "portfolio_rule_coverage",
                    "deployment_metadata_coverage",
                },
                manifest_path,
                manifest=manifest,
            ),
            _instrumentation_gap_impact_gate(required_capabilities, manifest, manifest_path),
        ]
        return gates

    @staticmethod
    def _new_strategy_discovery_gate(
        candidate: MonthlyImprovementCandidate,
        monthly_result: MonthlyValidationResult,
    ) -> MonthlyGateCheck:
        if not _is_new_strategy_candidate(candidate):
            return MonthlyGateCheck(
                name="new_strategy_discovery_packet",
                passed=True,
                severity=MonthlyGateSeverity.SOFT,
                reason="not required for non-new-strategy candidate",
                evidence_paths=candidate.evidence_paths,
            )

        packet_path = str(monthly_result.strategy_discovery_packet_path or "").strip()
        if not packet_path or not Path(packet_path).exists():
            return MonthlyGateCheck(
                name="new_strategy_discovery_packet",
                passed=False,
                reason="new-strategy candidate requires strategy_discovery_packet.json",
                evidence_paths=[packet_path] if packet_path else candidate.evidence_paths,
            )
        try:
            packet = StrategyDiscoveryPacket.model_validate(
                json.loads(Path(packet_path).read_text(encoding="utf-8"))
            )
        except Exception as exc:
            return MonthlyGateCheck(
                name="new_strategy_discovery_packet",
                passed=False,
                reason=f"strategy discovery packet is invalid: {exc}",
                evidence_paths=[packet_path],
            )

        clusters = [*packet.missed_opportunity_clusters, *packet.denominator_clusters]
        material_cluster_ids = {
            cluster.cluster_id
            for cluster in clusters
            if cluster.support_count >= MIN_DISCOVERY_CLUSTER_COUNT
            and cluster.control_count > 0
            and cluster.estimated_after_cost_pnl >= MIN_DISCOVERY_AFTER_COST_ESTIMATE
        }
        cited_cluster_ids = set(_string_list(
            candidate.raw_payload.get("strategy_discovery_cluster_ids")
            or candidate.raw_payload.get("source_discovery_cluster_ids")
            or candidate.raw_payload.get("source_strategy_discovery_cluster_ids")
        ))
        reasons: list[str] = []
        if packet.authority != "diagnostics_only" or packet.evidence_authority != "diagnostics_only":
            reasons.append("strategy discovery packet must remain diagnostics_only")
        if packet.approval_gate_eligible:
            reasons.append("strategy discovery packet cannot satisfy approval gates")
        if not clusters:
            reasons.append("strategy discovery packet has no recurring opportunity clusters")
        elif not material_cluster_ids:
            reasons.append("strategy discovery packet has no material after-cost clusters with controls")
        if not packet.control_slices:
            reasons.append("new-strategy discovery requires control slices")
        elif not any(int(slice_.get("control_count") or 0) > 0 for slice_ in packet.control_slices):
            reasons.append("new-strategy discovery requires non-empty control slices")
        if not packet.after_cost_estimates:
            reasons.append("new-strategy discovery requires after-cost estimates")
        elif not any(
            _number(estimate.get("estimated_after_cost_pnl")) >= MIN_DISCOVERY_AFTER_COST_ESTIMATE
            for estimate in packet.after_cost_estimates
        ):
            reasons.append("new-strategy discovery requires material after-cost estimates")
        if not packet.replay_or_shadow_plan:
            reasons.append("new-strategy discovery requires replay_or_shadow_plan")
        if not candidate.replay_or_experiment_plan:
            reasons.append("candidate requires replay_or_experiment_plan")
        if not cited_cluster_ids:
            reasons.append("candidate must cite strategy_discovery_cluster_ids")
        elif not cited_cluster_ids & material_cluster_ids:
            reasons.append("candidate strategy_discovery_cluster_ids do not match material packet clusters")

        return MonthlyGateCheck(
            name="new_strategy_discovery_packet",
            passed=not reasons,
            reason="; ".join(reasons),
            evidence_paths=_dedupe([packet_path, *packet.evidence_paths]),
        )

    def _record_proposal(
        self,
        candidate: MonthlyImprovementCandidate,
        packet: MonthlyApprovalEvidencePacket,
        gate_report: MonthlyCandidateGateReport,
        monthly_result: MonthlyValidationResult,
        *,
        verification: object | None = None,
    ) -> str:
        proposal_id = packet.proposal_id or self._proposal_id(candidate, monthly_result)
        if self.proposal_ledger is None:
            return proposal_id

        affected_parameters = [
            str(change.get("param_name") or change.get("parameter") or "")
            for change in candidate.param_changes
            if str(change.get("param_name") or change.get("parameter") or "")
        ]
        affected_files = _dedupe([
            *candidate.planned_files,
            *[
                str(change.get("file_path") or "")
                for change in candidate.file_changes
                if str(change.get("file_path") or "")
            ],
        ])
        candidate_record = ProposalCandidate(
            proposal_id=proposal_id,
            source=_proposal_source(candidate.source),
            kind=_proposal_kind(candidate.change_kind),
            bot_id=monthly_result.bot_id,
            strategy_id=monthly_result.strategy_id,
            lifecycle_stage=candidate.family,
            title=candidate.title,
            description=candidate.description or packet.reason_for_change,
            expected_mechanism=candidate.raw_payload.get("expected_mechanism", ""),
            affected_parameters=affected_parameters,
            affected_files=affected_files,
            acceptance_criteria=candidate.acceptance_criteria,
            evaluation_method=candidate.source.value,
            linked_diagnostics=[monthly_result.gap_attribution.primary_category.value],
            linked_run_id=monthly_result.run_id,
            suggestion_id=proposal_id,
        )
        self.proposal_ledger.record_candidate(candidate_record)
        verifier_verdict = str(getattr(getattr(verification, "verdict", ""), "value", "") or "")
        decision, decision_reason, confidence = _proposal_decision(
            gate_report=gate_report,
            packet=packet,
            verifier_verdict=verifier_verdict,
        )
        self.proposal_ledger.record_evaluation(
            proposal_id,
            ProposalEvaluation(
                proposal_id=proposal_id,
                method="monthly_candidate_gates_and_verifier",
                summary=packet.human_summary,
                objective_score=candidate.objective_score or candidate.objective_delta,
                confidence=confidence,
                decision=decision,
                decision_reason=decision_reason,
                evidence_paths=packet.artifact_paths,
            ),
        )
        return proposal_id

    @staticmethod
    def _proposal_id(
        candidate: MonthlyImprovementCandidate,
        monthly_result: MonthlyValidationResult,
    ) -> str:
        return make_proposal_id(
            _proposal_source(candidate.source),
            monthly_result.bot_id,
            _proposal_kind(candidate.change_kind),
            candidate.title,
            strategy_id=monthly_result.strategy_id,
            link_key=f"{monthly_result.run_id}:{candidate.candidate_id}",
        )

    def _record_strategy_change(
        self,
        candidate: MonthlyImprovementCandidate,
        packet: MonthlyApprovalEvidencePacket,
        request: ApprovalRequest,
    ) -> str:
        if self.strategy_change_ledger is None:
            return ""
        record = StrategyChangeRecord(
            bot_id=packet.bot_id,
            strategy_id=packet.strategy_id,
            record_type=StrategyChangeRecordType.PROPOSED_CHANGE,
            prior_config_version=str(candidate.raw_payload.get("prior_config_version") or ""),
            new_config_version=str(candidate.raw_payload.get("new_config_version") or candidate.raw_payload.get("proposed_config_version") or ""),
            mutation_diff={
                "candidate_id": candidate.candidate_id,
                "source": candidate.source.value,
                "family": candidate.family,
                "change_kind": candidate.change_kind,
                "param_changes": candidate.param_changes,
                "file_changes": candidate.file_changes,
                "proposed_changes": candidate.proposed_changes,
            },
            source_proposal_ids=[packet.proposal_id] if packet.proposal_id else [],
            source_suggestion_ids=[packet.suggestion_id] if packet.suggestion_id else [],
            approval_request_id=request.request_id,
            evidence_paths=packet.artifact_paths,
            objective_deltas=packet.objective_deltas,
            decision_reason=packet.human_summary,
            monthly_status="approval_ready",
            run_id=packet.run_id,
            run_month=packet.run_month,
        )
        self.strategy_change_ledger.record_proposed_change(record)
        return record.record_id

    def _build_approval_request(
        self,
        packet: MonthlyApprovalEvidencePacket,
        candidate: MonthlyImprovementCandidate,
    ) -> ApprovalRequest:
        request_id = hashlib.sha256(
            f"{packet.run_id}:{packet.candidate_id}:{packet.proposal_id}".encode(),
        ).hexdigest()[:12]
        file_changes = [
            FileChange.model_validate(change)
            for change in candidate.file_changes
            if isinstance(change, dict) and str(change.get("file_path") or "")
        ]
        planned_files = _dedupe([
            *candidate.planned_files,
            *[change.file_path for change in file_changes],
        ])
        return ApprovalRequest(
            request_id=request_id,
            suggestion_id=packet.proposal_id or packet.candidate_id,
            bot_id=packet.bot_id,
            strategy_id=packet.strategy_id,
            change_kind=_change_kind(candidate.change_kind, has_file_changes=bool(file_changes)),
            title=packet.title,
            summary=packet.human_summary,
            param_changes=candidate.param_changes,
            file_changes=file_changes,
            planned_files=planned_files,
            verification_commands=_string_list(candidate.raw_payload.get("verification_commands")),
            risk_tier=_risk_tier(packet.risk_classification),
            draft_pr=False,
            implementation_notes=json.dumps(packet.machine_readable_payload, indent=2, default=str),
            monthly_run_id=packet.run_id,
            monthly_run_month=packet.run_month,
            proposal_id=packet.proposal_id,
            evidence_paths=packet.artifact_paths,
            objective_deltas=packet.objective_deltas,
            risk_classification=packet.risk_classification.value,
            rollback_plan=packet.rollback_plan,
            approval_packet_path=packet.approval_packet_path,
            machine_readable_payload=packet.machine_readable_payload,
        )

    @staticmethod
    def _suppression_reasons(
        gate_report: MonthlyCandidateGateReport,
        *,
        shadow: bool,
        approval_tracker_present: bool,
        verifier_verdict: str = "",
        verifier_review_required: bool = False,
        final_packet_issues: list[str] | None = None,
    ) -> list[str]:
        reasons = list(gate_report.blocking_reasons)
        reasons.extend(final_packet_issues or [])
        if shadow:
            reasons.append("monthly validation is running in shadow mode")
        if gate_report.passed and not approval_tracker_present:
            reasons.append("approval tracker is unavailable")
        if verifier_verdict and verifier_verdict != "pass":
            if verifier_review_required:
                reasons.append("monthly evidence verifier requires human review")
            else:
                reasons.append(f"monthly evidence verifier verdict is {verifier_verdict}")
        return _dedupe(reasons)

    def _bool_gate(
        self,
        candidate: MonthlyImprovementCandidate,
        *,
        name: str,
        keys: tuple[str, ...],
        missing_reason: str,
    ) -> MonthlyGateCheck:
        value = _candidate_bool(candidate, *keys)
        return MonthlyGateCheck(
            name=name,
            passed=value is True,
            reason="" if value is True else (f"{name} did not pass" if value is False else missing_reason),
            evidence_paths=candidate.evidence_paths,
        )

    @staticmethod
    def _approval_payload_gate(candidate: MonthlyImprovementCandidate) -> MonthlyGateCheck:
        missing: list[str] = []
        if not candidate.rollback_plan:
            missing.append("rollback_plan")
        if not candidate.replay_or_experiment_plan:
            missing.append("replay_or_experiment_plan")
        if not candidate.acceptance_criteria:
            missing.append("acceptance_criteria")
        if not any([
            candidate.param_changes,
            candidate.file_changes,
            candidate.proposed_changes,
            candidate.planned_files,
        ]):
            missing.append("change_payload")
        return MonthlyGateCheck(
            name="approval_packet_payload",
            passed=not missing,
            reason="" if not missing else f"candidate is missing approval payload fields: {', '.join(missing)}",
            evidence_paths=candidate.evidence_paths,
        )

    @staticmethod
    def _phase_support_gate(candidate: MonthlyImprovementCandidate) -> MonthlyGateCheck:
        if candidate.source != MonthlyCandidateSource.PHASED_AUTO:
            return MonthlyGateCheck(
                name="purged_fold_support",
                passed=True,
                severity=MonthlyGateSeverity.SOFT,
                reason="not required for non-phased-auto candidate",
                evidence_paths=candidate.evidence_paths,
            )
        explicit = _candidate_bool(
            candidate,
            "fold_support_passed",
            "purged_fold_support",
            "positive_fold_support",
            "folds_positive",
        )
        return MonthlyGateCheck(
            name="purged_fold_support",
            passed=explicit is True,
            reason="" if explicit is True else "phased-auto candidate lacks positive purged-fold support",
            evidence_paths=candidate.evidence_paths,
        )

    @staticmethod
    def _decision_parity_gate(
        candidate: MonthlyImprovementCandidate,
        monthly_result: MonthlyValidationResult,
        *,
        artifact_contract: MonthlyArtifactContract | None = None,
    ) -> MonthlyGateCheck:
        if not _is_structural_candidate(candidate):
            return MonthlyGateCheck(
                name="decision_parity_report",
                passed=True,
                severity=MonthlyGateSeverity.SOFT,
                reason="not required for non-structural candidate",
                evidence_paths=candidate.evidence_paths,
            )
        path = Path(candidate.decision_parity_report_path)
        if not candidate.decision_parity_report_path or not path.exists():
            return MonthlyGateCheck(
                name="decision_parity_report",
                passed=False,
                reason="structural candidate requires decision_parity_report_path",
                evidence_paths=candidate.evidence_paths,
            )
        try:
            report = DecisionParityReport.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            return MonthlyGateCheck(
                name="decision_parity_report",
                passed=False,
                reason=f"decision parity report is invalid: {exc}",
                evidence_paths=[str(path)],
            )
        reasons: list[str] = []
        manifest = _load_run_manifest(monthly_result.run_manifest_path)
        contract = _load_strategy_plugin_contract(manifest)
        if report.run_id != monthly_result.run_id:
            reasons.append("decision parity run_id does not match monthly run")
        if report.candidate_id != candidate.candidate_id:
            reasons.append("decision parity candidate_id does not match candidate")
        if not report.eligible_for_structural_approval:
            reasons.append("decision parity report is not pass")
        evidence_paths = _dedupe([
            *report.evidence_paths,
            *[
                evidence_path
                for check in report.checks
                for evidence_path in check.evidence_paths
            ],
        ])
        missing_evidence = [evidence_path for evidence_path in evidence_paths if not Path(evidence_path).exists()]
        if missing_evidence:
            reasons.append(
                "decision parity evidence paths do not exist: "
                + ", ".join(missing_evidence[:5])
            )
        root = (
            Path(monthly_result.artifact_index_path).parent
            if monthly_result.artifact_index_path else None
        )
        scope_contract = artifact_contract or (
            MonthlyArtifactContract(artifact_root=root) if root is not None else None
        )
        outside_evidence = scope_contract.paths_outside_root(evidence_paths) if scope_contract else []
        if outside_evidence:
            reasons.append(
                "decision parity evidence paths outside artifact_root: "
                + ", ".join(outside_evidence[:5])
            )
        if manifest and manifest.strategy_plugin_id and report.strategy_plugin_id != manifest.strategy_plugin_id:
            reasons.append("decision parity strategy_plugin_id does not match run manifest")
        if contract:
            if report.strategy_plugin_id != contract.plugin_id:
                reasons.append("decision parity strategy_plugin_id does not match plugin contract")
            if report.live_repo_commit_sha != contract.live_repo_commit_sha:
                reasons.append("decision parity live_repo_commit_sha does not match plugin contract")
            if report.backtest_adapter_commit_sha != contract.backtest_adapter_commit_sha:
                reasons.append("decision parity backtest_adapter_commit_sha does not match plugin contract")
        return MonthlyGateCheck(
            name="decision_parity_report",
            passed=not reasons,
            reason="; ".join(reasons),
            evidence_paths=[str(path), *evidence_paths],
        )

    @staticmethod
    def _strategy_plugin_contract_gate(
        candidate: MonthlyImprovementCandidate,
        monthly_result: MonthlyValidationResult,
    ) -> MonthlyGateCheck:
        if not _is_structural_candidate(candidate):
            return MonthlyGateCheck(
                name="strategy_plugin_contract_maturity",
                passed=True,
                severity=MonthlyGateSeverity.SOFT,
                reason="not required for non-structural candidate",
                evidence_paths=candidate.evidence_paths,
            )
        manifest = _load_run_manifest(monthly_result.run_manifest_path)
        contract_path = Path(manifest.strategy_plugin_contract_path) if manifest and manifest.strategy_plugin_contract_path else None
        if contract_path is None or not contract_path.exists():
            return MonthlyGateCheck(
                name="strategy_plugin_contract_maturity",
                passed=False,
                reason="structural candidate requires strategy plugin contract evidence",
                evidence_paths=[monthly_result.run_manifest_path],
            )
        try:
            contract = StrategyPluginContract.model_validate(
                json.loads(contract_path.read_text(encoding="utf-8"))
            )
        except Exception as exc:
            return MonthlyGateCheck(
                name="strategy_plugin_contract_maturity",
                passed=False,
                reason=f"strategy plugin contract is invalid: {exc}",
                evidence_paths=[str(contract_path)],
            )
        reasons: list[str] = []
        if not contract.eligible_for_approval:
            reasons.append(f"strategy plugin contract maturity is {contract.maturity.value}")
        reasons.extend(
            deployment_metadata_errors(
                manifest,
                missing_reason=(
                    "approval-ready strategy plugin requires deployment metadata evidence"
                ),
            )
        )
        if manifest and manifest.strategy_plugin_id and contract.plugin_id != manifest.strategy_plugin_id:
            reasons.append("strategy plugin contract plugin_id does not match run manifest")
        return MonthlyGateCheck(
            name="strategy_plugin_contract_maturity",
            passed=not reasons,
            reason="; ".join(reasons),
            evidence_paths=[str(contract_path), *contract.parity_fixture_set],
        )

    @staticmethod
    def _runner_contract_gate(candidate: MonthlyImprovementCandidate) -> MonthlyGateCheck:
        version = str(
            candidate.deterministic_gate_inputs.get("runner_contract_version")
            or candidate.deterministic_gate_inputs.get("source_runner_contract_version")
            or candidate.workflow_contract_version
            or ""
        ).strip()
        expected = _runner_contract_version(candidate.source)
        passed = bool(expected and version == expected)
        return MonthlyGateCheck(
            name="source_runner_contract",
            passed=passed,
            reason="" if passed else f"candidate missing {expected or 'known'} runner contract",
            evidence_paths=[*candidate.evidence_paths, *candidate.artifact_paths],
        )

    @staticmethod
    def _candidate_lineage_gate(
        candidate: MonthlyImprovementCandidate,
        monthly_result: MonthlyValidationResult,
        artifact_index: BacktestArtifactIndex | None,
    ) -> MonthlyGateCheck:
        reasons: list[str] = []
        if candidate.run_id != monthly_result.run_id:
            reasons.append("candidate run_id does not match monthly run")
        run_manifest = _load_run_manifest(monthly_result.run_manifest_path)
        manifest_id = (
            artifact_index.manifest_id
            if artifact_index is not None and artifact_index.manifest_id
            else (run_manifest.manifest_id if run_manifest is not None else "")
        )
        if manifest_id and candidate.manifest_id != manifest_id:
            reasons.append("candidate manifest_id does not match artifact index")
        for field_name in ("round_id", "prior_round_id", "next_round_id"):
            if not str(getattr(candidate, field_name) or "").strip():
                reasons.append(f"candidate missing {field_name}")
        if not candidate.backtest_repo_commit_sha:
            reasons.append("candidate missing backtest_repo_commit_sha")
        if not (candidate.live_trading_repo_commit_sha or candidate.code_sha):
            reasons.append("candidate missing live_trading_repo_commit_sha")
        if not candidate.control_plane_commit_sha:
            reasons.append("candidate missing control_plane_commit_sha")
        return MonthlyGateCheck(
            name="candidate_lineage_contract",
            passed=not reasons,
            reason="; ".join(reasons),
            evidence_paths=[
                monthly_result.run_manifest_path,
                *candidate.evidence_paths,
                *candidate.artifact_paths,
            ],
        )

    @staticmethod
    def _candidate_workspace_gate(candidate: MonthlyImprovementCandidate) -> MonthlyGateCheck:
        supplied = any([
            candidate.candidate_workspace_key,
            candidate.candidate_workspace_path,
            candidate.candidate_attempt_id,
            candidate.candidate_attempt_status,
        ])
        if not supplied:
            return MonthlyGateCheck(
                name="candidate_workspace_attempt",
                passed=True,
                severity=MonthlyGateSeverity.SOFT,
                reason="candidate workspace/attempt metadata not supplied",
                evidence_paths=candidate.evidence_paths,
            )

        reasons: list[str] = []
        key = candidate.candidate_workspace_key
        if not key or key != _safe_id(key):
            reasons.append("candidate_workspace_key is missing or unsafe")
        status = candidate.candidate_attempt_status.strip().lower()
        if status not in {"completed", "succeeded", "success"}:
            reasons.append("candidate attempt did not complete successfully")
        path = Path(candidate.candidate_workspace_path)
        root = Path(str(candidate.deterministic_gate_inputs.get("artifact_root") or ""))
        if not candidate.candidate_workspace_path or not path.exists():
            reasons.append("candidate_workspace_path is missing or does not exist")
        elif root:
            try:
                path.resolve().relative_to(root.resolve())
            except (OSError, ValueError):
                reasons.append("candidate_workspace_path is outside artifact_root")
        if candidate.stall_timeout_seconds < 0:
            reasons.append("stall_timeout_seconds cannot be negative")

        return MonthlyGateCheck(
            name="candidate_workspace_attempt",
            passed=not reasons,
            reason="; ".join(reasons),
            evidence_paths=[candidate.candidate_workspace_path, *candidate.evidence_paths],
        )

    @staticmethod
    def _candidate_artifact_containment_gate(
        candidate: MonthlyImprovementCandidate,
        *,
        artifact_contract: MonthlyArtifactContract | None = None,
    ) -> MonthlyGateCheck:
        paths = _dedupe([*candidate.evidence_paths, *candidate.artifact_paths])
        if not paths:
            return MonthlyGateCheck(
                name="candidate_artifact_containment",
                passed=True,
                severity=MonthlyGateSeverity.SOFT,
                reason="candidate has no evidence/artifact paths to contain",
                evidence_paths=[],
            )
        root = Path(str(candidate.deterministic_gate_inputs.get("artifact_root") or ""))
        scope_contract = artifact_contract or MonthlyArtifactContract(artifact_root=root)
        outside = scope_contract.paths_outside_root(paths)
        return MonthlyGateCheck(
            name="candidate_artifact_containment",
            passed=not outside,
            reason=(
                ""
                if not outside
                else "candidate evidence/artifact paths outside artifact_root: "
                + ", ".join(outside[:5])
            ),
            evidence_paths=paths,
        )

    @staticmethod
    def _model_review_gate(
        candidate: MonthlyImprovementCandidate,
        validation: MonthlyModelValidationResult | None,
    ) -> MonthlyGateCheck:
        if validation is None:
            return MonthlyGateCheck(
                name="monthly_model_review",
                passed=False,
                reason="valid monthly model review is required before approval-ready candidates",
                evidence_paths=candidate.evidence_paths,
            )
        if not validation.valid:
            reasons = [issue.message for issue in validation.issues[:3]]
            return MonthlyGateCheck(
                name="monthly_model_review",
                passed=False,
                reason="monthly model review failed validation: " + "; ".join(reasons),
                evidence_paths=candidate.evidence_paths,
            )
        passed = candidate.candidate_id in set(validation.actionable_candidate_ids)
        return MonthlyGateCheck(
            name="monthly_model_review",
            passed=passed,
            reason="" if passed else "model review did not route this candidate as actionable",
            evidence_paths=candidate.evidence_paths,
        )

    @staticmethod
    def _improvement_gate(candidate: MonthlyImprovementCandidate) -> MonthlyGateCheck:
        explicit = _candidate_bool(candidate, "latest_month_oos_improvement", "latest_oos_improvement")
        delta = _candidate_float(candidate, "latest_month_oos_delta", "latest_month_objective_delta")
        passed = explicit if explicit is not None else (delta is not None and delta > 0)
        return MonthlyGateCheck(
            name="latest_month_oos_improvement",
            passed=passed is True,
            reason="" if passed is True else "latest-month selection-OOS improvement is missing or non-positive",
            evidence_paths=candidate.evidence_paths,
        )

    @staticmethod
    def _calibration_gate(candidate: MonthlyImprovementCandidate) -> MonthlyGateCheck:
        explicit = _candidate_bool(candidate, "calibration_support", "calibration_supported")
        delta = _candidate_float(candidate, "calibration_objective_delta", "calibration_delta")
        passed = explicit if explicit is not None else (delta is not None and delta > 0)
        return MonthlyGateCheck(
            name="calibration_support",
            passed=passed is True,
            reason="" if passed is True else "calibration support is missing or non-positive",
            evidence_paths=candidate.evidence_paths,
        )

    def _trade_count_gate(self, candidate: MonthlyImprovementCandidate) -> MonthlyGateCheck:
        explicit = _candidate_bool(candidate, "sufficient_trade_count")
        sparse = _candidate_value(candidate, "sparse_sample_classification", "sample_classification")
        trade_count = _candidate_float(candidate, "trade_count", "selection_trade_count")
        passed = explicit if explicit is not None else False
        if explicit is None and trade_count is not None:
            passed = trade_count >= self.min_trade_count
        if not passed and str(sparse or "").strip():
            passed = True
        return MonthlyGateCheck(
            name="trade_count_or_sparse_classification",
            passed=passed,
            reason="" if passed else "trade count is insufficient and no sparse-sample classification was supplied",
            evidence_paths=candidate.evidence_paths,
        )

    @staticmethod
    def _drawdown_gate(candidate: MonthlyImprovementCandidate) -> MonthlyGateCheck:
        explicit = _candidate_bool(candidate, "drawdown_gate_passed", "no_material_drawdown_increase")
        delta = _candidate_float(candidate, "max_drawdown_delta_pct", "drawdown_delta_pct")
        passed = explicit if explicit is not None else (delta is not None and delta <= 0)
        return MonthlyGateCheck(
            name="drawdown_gate",
            passed=passed is True,
            reason="" if passed is True else "candidate materially increases max drawdown or lacks drawdown evidence",
            evidence_paths=candidate.evidence_paths,
        )

    def _outlier_gate(self, candidate: MonthlyImprovementCandidate) -> MonthlyGateCheck:
        explicit = _candidate_bool(candidate, "outlier_concentration_passed", "no_outlier_dependency")
        concentration = _candidate_float(candidate, "outlier_win_concentration", "top_win_concentration")
        passed = explicit if explicit is not None else (
            concentration is not None and concentration <= self.max_outlier_win_concentration
        )
        return MonthlyGateCheck(
            name="outlier_concentration",
            passed=passed is True,
            reason="" if passed is True else "candidate depends on too few outlier wins or lacks outlier evidence",
            evidence_paths=candidate.evidence_paths,
        )

    def _outcome_prior_gate(self, candidate: MonthlyImprovementCandidate) -> MonthlyGateCheck:
        if self.search_allocation_policy is None:
            return MonthlyGateCheck(
                name="outcome_prior_controls",
                passed=True,
                severity=MonthlyGateSeverity.SOFT,
                reason="no outcome priors available",
                evidence_paths=candidate.evidence_paths,
            )
        requires_stronger = self.search_allocation_policy.requires_stronger_evidence(
            bot_id=candidate.bot_id,
            strategy_id=candidate.strategy_id,
            mutation_family=candidate.family,
            category=candidate.change_kind,
        )
        if not requires_stronger:
            return MonthlyGateCheck(
                name="outcome_prior_controls",
                passed=True,
                severity=MonthlyGateSeverity.SOFT,
                reason="no negative monthly prior for this family",
                evidence_paths=candidate.evidence_paths,
            )
        explicit = _candidate_bool(
            candidate,
            "stronger_evidence_passed",
            "negative_prior_override",
            "authoritative_prior_override",
        )
        return MonthlyGateCheck(
            name="outcome_prior_controls",
            passed=explicit is True,
            reason=(
                ""
                if explicit is True
                else "negative monthly priors require stronger validation evidence"
            ),
            evidence_paths=candidate.evidence_paths,
        )


def _load_run_manifest(path: str) -> MonthlyRunManifest | None:
    if not path:
        return None
    try:
        return MonthlyRunManifest.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
    except Exception:
        return None


def _load_strategy_plugin_contract(manifest: MonthlyRunManifest | None) -> StrategyPluginContract | None:
    if not manifest or not manifest.strategy_plugin_contract_path:
        return None
    path = Path(manifest.strategy_plugin_contract_path)
    if not path.exists() or not path.is_file():
        return None
    try:
        return StrategyPluginContract.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


def _load_learning_sufficiency_manifest(
    manifest_path: str,
) -> tuple[LearningSufficiencyManifest | None, str]:
    if not manifest_path:
        return None, "learning sufficiency manifest path is missing"
    path = Path(manifest_path)
    if not path.exists() or not path.is_file():
        return None, "learning sufficiency manifest is missing"
    try:
        return LearningSufficiencyManifest.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        ), ""
    except Exception as exc:
        return None, f"learning sufficiency manifest is malformed: {exc}"


def _required_learning_checks(required_capabilities: list[str]) -> set[str]:
    checks: set[str] = set()
    for capability in required_capabilities:
        checks.update(CAPABILITY_REQUIREMENTS.get(capability, ()))
    return checks


def _manifest_checks(manifest: LearningSufficiencyManifest) -> dict[str, CoverageCheck]:
    return {
        **manifest.required_event_coverage,
        **manifest.join_coverage,
        **manifest.denominator_coverage,
        "lineage_coverage": manifest.lineage_coverage,
        "after_cost_coverage": manifest.after_cost_coverage,
        "counterfactual_coverage": manifest.counterfactual_coverage,
        "proposal_trace_coverage": manifest.proposal_trace_coverage,
        "deployment_metadata_coverage": manifest.deployment_metadata_coverage,
    }


def _coverage_gate(
    name: str,
    required_checks: set[str],
    all_checks: dict[str, CoverageCheck],
    relevant_checks: set[str],
    manifest_path: str,
    *,
    manifest: LearningSufficiencyManifest | None = None,
) -> MonthlyGateCheck:
    check_ids = sorted(required_checks & relevant_checks)
    runtime_check_ids = (
        sorted(
            check_id for check_id in required_checks
            if all_checks.get(check_id) is not None
            and all_checks[check_id].satisfies_learning_authority
        )
        if name == "runtime_evidence_coverage" else check_ids
    )
    if not check_ids:
        return MonthlyGateCheck(
            name=name,
            passed=True,
            reason="not required for candidate learning capability",
            evidence_paths=[manifest_path] if manifest_path else [],
        )
    blockers: list[str] = []
    evidence_paths = [manifest_path] if manifest_path else []
    for check_id in check_ids:
        check = all_checks.get(check_id)
        if check is None:
            blockers.append(f"{check_id}:missing")
            continue
        evidence_paths.extend(check.evidence_paths)
        if not check.satisfies_learning_authority:
            blockers.append(_coverage_blocker(check_id, check))
    if name == "runtime_evidence_coverage" and manifest is not None:
        runtime_blockers, runtime_evidence_paths = runtime_source_authority_for_checks(
            runtime_check_ids,
            manifest.runtime_evidence_support,
            all_checks,
        )
        blockers.extend(runtime_blockers)
        evidence_paths.extend(runtime_evidence_paths)
    blockers = _dedupe(blockers)
    return MonthlyGateCheck(
        name=name,
        passed=not blockers,
        reason="" if not blockers else "; ".join(blockers),
        evidence_paths=_dedupe(evidence_paths),
    )


def _coverage_blocker(check_id: str, check: CoverageCheck) -> str:
    details: list[str] = []
    if check.reason:
        details.append(check.reason)
    if check.missing_fields:
        details.append("missing_fields=" + ",".join(check.missing_fields[:5]))
    if check.missing_event_types:
        details.append("missing_event_types=" + ",".join(check.missing_event_types[:5]))
    details.append(f"observed={check.observed_count}")
    details.append(f"required={check.required_count}")
    return f"{check_id}:{check.status.value} ({'; '.join(details)})"


def _instrumentation_gap_impact_gate(
    required_capabilities: list[str],
    manifest: LearningSufficiencyManifest,
    manifest_path: str,
) -> MonthlyGateCheck:
    relevant_gaps = [
        gap
        for gap in manifest.known_gaps
        if gap.blocked_learning_capability in set(required_capabilities)
    ]
    if not relevant_gaps:
        return MonthlyGateCheck(
            name="instrumentation_gap_impact",
            passed=True,
            reason="no instrumentation gaps block the requested learning capability",
            evidence_paths=[manifest_path] if manifest_path else [],
        )
    evidence_paths = _dedupe([
        manifest_path,
        *[
            path
            for gap in relevant_gaps
            for path in gap.evidence_paths
        ],
    ])
    top_gaps = [
        (
            f"{gap.blocked_learning_capability}:"
            f"{gap.event_type or 'evidence'}:"
            f"{gap.missing_field or 'missing'}:"
            f"{gap.expected_learning_value.value}"
        )
        for gap in relevant_gaps[:5]
    ]
    return MonthlyGateCheck(
        name="instrumentation_gap_impact",
        passed=False,
        reason="instrumentation gaps block requested learning capability: " + "; ".join(top_gaps),
        evidence_paths=evidence_paths,
    )


def _is_structural_candidate(candidate: MonthlyImprovementCandidate) -> bool:
    return (
        candidate.change_kind == "structural_change"
        or bool(candidate.file_changes)
        or bool(candidate.live_repo_patch_path)
        or bool(candidate.backtest_adapter_patch_path)
    )


def _is_new_strategy_candidate(candidate: MonthlyImprovementCandidate) -> bool:
    normalized = (candidate.change_kind or "").strip().lower().replace("-", "_")
    return normalized in {"new_strategy", "new_strategy_discovery", "structural_change"}


def _parse_model_review_text(text: str) -> MonthlyModelReview:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return parse_monthly_model_review(text)
    if isinstance(raw, dict):
        try:
            return MonthlyModelReview.model_validate(raw)
        except Exception:
            return parse_monthly_model_review(text)
    return parse_monthly_model_review(text)


def _runner_contract_version(source: MonthlyCandidateSource) -> str:
    if source == MonthlyCandidateSource.SMOKE_REPAIR:
        return "smoke_repair_runner_contract_v1"
    if source == MonthlyCandidateSource.PHASED_AUTO:
        return "phased_auto_runner_contract_v1"
    return ""


def _candidate_value(candidate: MonthlyImprovementCandidate, *keys: str) -> Any:
    for key in keys:
        if key in candidate.deterministic_gate_inputs:
            return candidate.deterministic_gate_inputs[key]
        if key in candidate.raw_payload:
            return candidate.raw_payload[key]
        if key in candidate.objective_deltas:
            return candidate.objective_deltas[key]
    return None


def _candidate_bool(candidate: MonthlyImprovementCandidate, *keys: str) -> bool | None:
    value = _candidate_value(candidate, *keys)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "pass", "passed", "yes", "y"}:
            return True
        if lowered in {"false", "fail", "failed", "no", "n"}:
            return False
    return None


def _candidate_float(candidate: MonthlyImprovementCandidate, *keys: str) -> float | None:
    value = _candidate_value(candidate, *keys)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _proposal_source(source: MonthlyCandidateSource) -> ProposalSource:
    if source == MonthlyCandidateSource.SMOKE_REPAIR:
        return ProposalSource.MONTHLY_SMOKE_REPAIR
    if source == MonthlyCandidateSource.PHASED_AUTO:
        return ProposalSource.MONTHLY_PHASED_AUTO
    if source == MonthlyCandidateSource.MODEL_REVIEW:
        return ProposalSource.MONTHLY_MODEL_REVIEW
    return ProposalSource.DETERMINISTIC


def _proposal_kind(change_kind: str) -> ProposalKind:
    try:
        return ProposalKind(change_kind)
    except ValueError:
        if change_kind == "rollback":
            return ProposalKind.ROLLBACK
        return ProposalKind.STRUCTURAL_CHANGE


def _proposal_decision(
    *,
    gate_report: MonthlyCandidateGateReport,
    packet: MonthlyApprovalEvidencePacket,
    verifier_verdict: str,
) -> tuple[str, str, float]:
    if not gate_report.passed:
        return "reject", "; ".join(gate_report.blocking_reasons), 0.0
    if verifier_verdict == "fail":
        return "reject", "monthly evidence verifier verdict is fail", 0.0
    if verifier_verdict == "needs_human_review":
        return "defer", "monthly evidence verifier requires human review", 0.5
    if not packet.approval_ready:
        return "defer", "; ".join(packet.approval_suppressed_reasons), 0.5
    return "approve", "candidate gates and monthly evidence verifier passed", 1.0


def _change_kind(change_kind: str, *, has_file_changes: bool) -> ChangeKind:
    raw = (change_kind or "").strip().lower()
    if raw == ChangeKind.ROLLBACK.value:
        return ChangeKind.ROLLBACK
    if raw == ChangeKind.BUG_FIX.value:
        return ChangeKind.BUG_FIX
    if raw == ChangeKind.STRUCTURAL_CHANGE.value or has_file_changes:
        return ChangeKind.STRUCTURAL_CHANGE
    return ChangeKind.PARAMETER_CHANGE


def _required_learning_capabilities(change_kind: str) -> list[str]:
    normalized = (change_kind or "").strip().lower().replace("-", "_")
    if normalized in {"filter_threshold_change", "filter_change", "threshold_change"}:
        return ["filter_threshold_learning"]
    if normalized in {"execution_change", "order_routing_change", "fill_quality_change"}:
        return ["execution_learning"]
    if normalized in {"sizing_change", "position_sizing_change", "risk_sizing_change"}:
        return ["sizing_learning"]
    if normalized in {"portfolio_rule_change", "allocation_change", "correlation_rule_change"}:
        return ["portfolio_interaction_learning"]
    if normalized in {"structural_change", "new_strategy", "new_strategy_discovery"}:
        return ["new_strategy_discovery"]
    if normalized in {"approval_grade_strategy_change", "approval_grade_change"}:
        return ["approval_grade_strategy_change"]
    return []


def _risk_tier(risk: MonthlyRiskClassification) -> RepoRiskTier:
    if risk in {MonthlyRiskClassification.HIGH, MonthlyRiskClassification.CRITICAL}:
        return RepoRiskTier.REQUIRES_DOUBLE_APPROVAL
    return RepoRiskTier.REQUIRES_APPROVAL


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:64] or "candidate"


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list | tuple):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str) and value:
        return [value]
    return []


def _number(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out
