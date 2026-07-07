"""Artifact authority registry seeded from existing contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from trading_assistant.paths import package_root
from trading_assistant.schemas.artifact_authority import (
    ArtifactAuthority,
    ArtifactAuthorityIssue,
    ArtifactRegistryEntry,
    artifact_type_from_path,
    payload_paths,
)
from trading_assistant.schemas.backtest_artifacts import (
    OPTIONAL_BACKTEST_ARTIFACTS,
    REQUIRED_BACKTEST_ARTIFACTS,
    BacktestArtifactIndex,
)
from trading_assistant.schemas.monthly_candidates import MonthlyApprovalEvidencePacket


_APPROVAL_GATE_REQUIRED_TYPES = {
    "monthly_validation_result",
    "replay_parity_report",
    "candidate_gate_report",
    "model_review_validation",
    "monthly_evidence_verification",
}

_PRE_VERIFIER_EXCLUDED_TYPES = {"monthly_evidence_verification"}

_ACTIONABLE_MODEL_REVIEW_AUTHORITIES = {
    ArtifactAuthority.APPROVAL_GATE,
    ArtifactAuthority.BINDING,
}


class ArtifactAuthorityRegistry:
    def __init__(self, entries: dict[str, ArtifactRegistryEntry]) -> None:
        self.entries = entries

    @classmethod
    def load(cls, memory_dir: Path | None = None) -> "ArtifactAuthorityRegistry":
        entries = _seed_entries()
        path = (memory_dir or package_root() / "memory") / "artifacts" / "registry.yaml"
        if path.exists():
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            for item in raw.get("artifacts", []):
                entry = ArtifactRegistryEntry.model_validate(item)
                entries[entry.artifact_type] = entry
        return cls(entries)

    def get(self, artifact_type_or_path: str) -> ArtifactRegistryEntry | None:
        artifact_type = artifact_type_or_path
        if "/" in artifact_type_or_path or "\\" in artifact_type_or_path or "." in Path(artifact_type_or_path).name:
            artifact_type = artifact_type_from_path(artifact_type_or_path)
        return self.entries.get(artifact_type)

    def may_satisfy_approval_gate(self, artifact_type_or_path: str) -> bool:
        entry = self.get(artifact_type_or_path)
        return bool(entry and entry.may_satisfy_approval_gate)

    def required_approval_gate_types(self, *, stage: str = "final") -> set[str]:
        if stage == "final":
            return set(_APPROVAL_GATE_REQUIRED_TYPES)
        if stage == "pre_verifier":
            return set(_APPROVAL_GATE_REQUIRED_TYPES - _PRE_VERIFIER_EXCLUDED_TYPES)
        raise ValueError(f"unknown approval gate stage: {stage}")

    def missing_required_approval_gate_types(
        self,
        packet: MonthlyApprovalEvidencePacket | dict[str, Any],
        *,
        stage: str = "final",
    ) -> list[str]:
        payload = packet.model_dump(mode="json") if isinstance(packet, MonthlyApprovalEvidencePacket) else dict(packet)
        machine_payload = payload.get("machine_readable_payload")
        if not isinstance(machine_payload, dict):
            machine_payload = {}
        present_types = {
            artifact_type_from_path(path)
            for path in payload_paths(machine_payload, "approval_gate_evidence")
        }
        return sorted(self.required_approval_gate_types(stage=stage) - present_types)

    def validate_approval_packet(
        self,
        packet: MonthlyApprovalEvidencePacket | dict[str, Any],
        *,
        require_gate_artifacts: bool = True,
    ) -> list[ArtifactAuthorityIssue]:
        payload = packet.model_dump(mode="json") if isinstance(packet, MonthlyApprovalEvidencePacket) else dict(packet)
        issues: list[ArtifactAuthorityIssue] = []
        artifact_paths = payload_paths(payload, "artifact_paths")
        machine_payload = payload.get("machine_readable_payload")
        if not isinstance(machine_payload, dict):
            machine_payload = {}
        gate_paths = payload_paths(machine_payload, "approval_gate_evidence")
        for path in gate_paths:
            if not self.may_satisfy_approval_gate(path):
                issues.append(ArtifactAuthorityIssue(
                    artifact=path,
                    message="artifact is not eligible to satisfy approval gates",
                    am_row="AM-06",
                    remediation="Use replay/parity/objective/model-validation evidence for gates; keep advisory artifacts as context.",
                ))
        for path in artifact_paths:
            if self.get(path) is None:
                issues.append(ArtifactAuthorityIssue(
                    artifact=path,
                    message="approval packet references an unregistered artifact type",
                    am_row="AM-06",
                    remediation="Register the artifact authority or remove it from approval packet evidence.",
                ))
        if require_gate_artifacts:
            for artifact_type in self.missing_required_approval_gate_types(payload, stage="final"):
                issues.append(ArtifactAuthorityIssue(
                    artifact=artifact_type,
                    message="approval packet is missing required approval-gate artifact type",
                    am_row="AM-09",
                    remediation=f"Attach {artifact_type} before routing approval.",
                ))
        return issues

    def validate_model_review_evidence(self, paths: list[str]) -> list[ArtifactAuthorityIssue]:
        issues: list[ArtifactAuthorityIssue] = []
        for path in paths:
            entry = self.get(path)
            if entry is None:
                issues.append(ArtifactAuthorityIssue(
                    artifact=path,
                    message="model review cites an unregistered artifact type",
                    am_row="AM-08",
                    remediation="Add deterministic evidence to the registry or keep the item hypothesis_only.",
                ))
            elif entry.authority not in _ACTIONABLE_MODEL_REVIEW_AUTHORITIES:
                issues.append(ArtifactAuthorityIssue(
                    artifact=path,
                    message=(
                        f"{entry.authority.value} evidence cannot support "
                        "actionable monthly model-review routing"
                    ),
                    am_row="AM-08",
                    remediation=(
                        "Use deterministic binding or approval-gate evidence for "
                        "actionable routing; keep advisory, diagnostics, generated, "
                        "and human-owned artifacts as context only."
                    ),
                ))
        return issues

    def validate_backtest_artifact_index(
        self,
        index: BacktestArtifactIndex,
        *,
        expected_run_id: str = "",
        expected_manifest_id: str = "",
        require_manifest_id: bool = False,
    ) -> list[ArtifactAuthorityIssue]:
        issues: list[ArtifactAuthorityIssue] = []
        expected = set(REQUIRED_BACKTEST_ARTIFACTS)
        registered_required = {
            entry.artifact_type
            for entry in self.entries.values()
            if entry.source_contract == "REQUIRED_BACKTEST_ARTIFACTS"
        }
        missing_registry = sorted(
            artifact_type_from_path(name)
            for name in expected
            if artifact_type_from_path(name) not in registered_required
        )
        for artifact_type in missing_registry:
            issues.append(ArtifactAuthorityIssue(
                artifact=artifact_type,
                message="required backtest artifact is not represented in authority registry",
                am_row="AM-06",
                remediation="Seed the registry from REQUIRED_BACKTEST_ARTIFACTS.",
            ))
        for error in index.validation_errors(
            expected_run_id=expected_run_id,
            expected_manifest_id=expected_manifest_id,
            require_manifest_id=require_manifest_id,
        ):
            issues.append(ArtifactAuthorityIssue(
                artifact="artifact_index",
                message=error,
                am_row="AM-06",
                remediation="Regenerate artifact_index.json from the monthly runner contract.",
            ))
        return issues


def _seed_entries() -> dict[str, ArtifactRegistryEntry]:
    entries: dict[str, ArtifactRegistryEntry] = {}
    approval_gate_required = {
        "replay_parity_report",
        "objective_breakdown",
        "selected_candidates",
        "candidate_results",
        "coverage_manifest",
    }
    for name in REQUIRED_BACKTEST_ARTIFACTS:
        artifact_type = artifact_type_from_path(name)
        entries[artifact_type] = ArtifactRegistryEntry(
            artifact_type=artifact_type,
            authority=(
                ArtifactAuthority.APPROVAL_GATE
                if artifact_type in approval_gate_required
                else ArtifactAuthority.BINDING
            ),
            owner_package="trading_assistant_backtest",
            source_contract="REQUIRED_BACKTEST_ARTIFACTS",
            path_patterns=[name],
            may_satisfy_approval_gate=artifact_type in approval_gate_required,
            readers=["monthly_candidate_pipeline", "monthly_evidence_verifier"],
            writers=["monthly_runner"],
        )
    for name in OPTIONAL_BACKTEST_ARTIFACTS:
        artifact_type = artifact_type_from_path(name)
        entries.setdefault(
            artifact_type,
            ArtifactRegistryEntry(
                artifact_type=artifact_type,
                authority=(
                    ArtifactAuthority.APPROVAL_GATE
                    if artifact_type in {"decision_parity_report", "fold_manifest", "rounds_manifest"}
                    else ArtifactAuthority.DIAGNOSTICS_ONLY
                ),
                owner_package="trading_assistant_backtest",
                source_contract="OPTIONAL_BACKTEST_ARTIFACTS",
                path_patterns=[name],
                may_satisfy_approval_gate=artifact_type in {
                    "decision_parity_report",
                    "fold_manifest",
                    "rounds_manifest",
                },
                readers=["monthly_candidate_pipeline", "monthly_evidence_verifier"],
                writers=["monthly_runner"],
            ),
        )
    manual_entries = [
        ArtifactRegistryEntry(
            artifact_type="monthly_search_brief",
            authority=ArtifactAuthority.ADVISORY,
            owner_package="trading_assistant",
            schema_ref="trading_assistant.schemas.monthly_search_brief.MonthlySearchBrief",
            path_patterns=["monthly_search_brief.json"],
            may_satisfy_approval_gate=False,
            readers=["monthly_optimizer_runner"],
            writers=["monthly_search_brief_builder"],
            notes="Search ordering and priors only; never proof of monthly gate passage.",
        ),
        ArtifactRegistryEntry(
            artifact_type="memory_policies",
            authority=ArtifactAuthority.HUMAN_OWNED,
            owner_package="trading_assistant",
            path_patterns=["memory/policies/**"],
            may_satisfy_approval_gate=False,
            readers=["context_builder"],
            notes="Human-owned policy memory; loops cannot write autonomously.",
        ),
        ArtifactRegistryEntry(
            artifact_type="approval_packet",
            authority=ArtifactAuthority.APPROVAL_GATE,
            owner_package="trading_assistant",
            schema_ref="trading_assistant.schemas.monthly_candidates.MonthlyApprovalEvidencePacket",
            path_patterns=["approval_packet_*.json"],
            may_satisfy_approval_gate=True,
            readers=["approval_tracker", "monthly_evidence_verifier"],
            writers=["monthly_candidate_pipeline"],
        ),
        ArtifactRegistryEntry(
            artifact_type="monthly_validation_result",
            authority=ArtifactAuthority.APPROVAL_GATE,
            owner_package="trading_assistant",
            schema_ref="trading_assistant.schemas.monthly_validation.MonthlyValidationResult",
            path_patterns=["monthly_validation_result.json"],
            may_satisfy_approval_gate=True,
            readers=["monthly_candidate_pipeline", "monthly_evidence_verifier"],
            writers=["monthly_validation_orchestrator"],
        ),
        ArtifactRegistryEntry(
            artifact_type="monthly_run_manifest",
            authority=ArtifactAuthority.BINDING,
            owner_package="trading_assistant",
            schema_ref="trading_assistant.schemas.monthly_run_manifest.MonthlyRunManifest",
            path_patterns=["run_manifest.json"],
            may_satisfy_approval_gate=False,
            readers=["monthly_validation_orchestrator", "monthly_evidence_verifier"],
            writers=["monthly_validation_orchestrator"],
        ),
        ArtifactRegistryEntry(
            artifact_type="telemetry_manifest",
            authority=ArtifactAuthority.BINDING,
            owner_package="trading_assistant",
            schema_ref="trading_assistant.schemas.telemetry_manifest.TelemetryManifest",
            path_patterns=["telemetry_manifest.json"],
            may_satisfy_approval_gate=False,
            readers=["monthly_candidate_pipeline", "monthly_evidence_verifier"],
            writers=["monthly_validation_orchestrator"],
        ),
        ArtifactRegistryEntry(
            artifact_type="learning_sufficiency_manifest",
            authority=ArtifactAuthority.BINDING,
            owner_package="trading_assistant",
            schema_ref="trading_assistant.schemas.learning_sufficiency.LearningSufficiencyManifest",
            path_patterns=["learning_sufficiency_manifest.json"],
            may_satisfy_approval_gate=False,
            readers=["monthly_candidate_pipeline", "monthly_evidence_verifier", "weekly_prompt_assembler"],
            writers=["learning_sufficiency_auditor", "monthly_validation_orchestrator"],
            notes=(
                "Binding learning authority consumed by monthly gates; not a standalone "
                "approval-gate artifact."
            ),
        ),
        ArtifactRegistryEntry(
            artifact_type="strategy_discovery_packet",
            authority=ArtifactAuthority.DIAGNOSTICS_ONLY,
            owner_package="trading_assistant",
            path_patterns=["strategy_discovery_packet.json"],
            may_satisfy_approval_gate=False,
            readers=["monthly_candidate_pipeline", "weekly_prompt_assembler"],
            writers=["strategy_discovery_builder"],
            notes=(
                "Diagnostics-only discovery context until replay, bridge maturity, "
                "and sufficiency gates promote a proposal."
            ),
        ),
        ArtifactRegistryEntry(
            artifact_type="artifact_index",
            authority=ArtifactAuthority.BINDING,
            owner_package="trading_assistant_backtest",
            schema_ref="trading_assistant.schemas.backtest_artifacts.BacktestArtifactIndex",
            path_patterns=["artifact_index.json"],
            may_satisfy_approval_gate=False,
            readers=["monthly_candidate_pipeline", "monthly_evidence_verifier"],
            writers=["monthly_runner"],
        ),
        ArtifactRegistryEntry(
            artifact_type="outcome_priors_snapshot",
            authority=ArtifactAuthority.ADVISORY,
            owner_package="trading_assistant",
            path_patterns=["outcome_priors_snapshot.json"],
            may_satisfy_approval_gate=False,
            readers=["monthly_validation_orchestrator", "monthly_evidence_verifier"],
            writers=["outcome_prior_store"],
            notes="Advisory prior snapshot; cannot satisfy approval gates.",
        ),
        ArtifactRegistryEntry(
            artifact_type="candidate_gate_report",
            authority=ArtifactAuthority.APPROVAL_GATE,
            owner_package="trading_assistant",
            schema_ref="trading_assistant.schemas.monthly_candidates.MonthlyCandidateGateReport",
            path_patterns=["candidate_gate_report.json"],
            may_satisfy_approval_gate=True,
            readers=["approval_tracker", "monthly_evidence_verifier"],
            writers=["monthly_candidate_pipeline"],
        ),
        ArtifactRegistryEntry(
            artifact_type="model_review_validation",
            authority=ArtifactAuthority.APPROVAL_GATE,
            owner_package="trading_assistant",
            schema_ref="trading_assistant.schemas.monthly_model_review.MonthlyModelValidationResult",
            path_patterns=["model_review_validation.json"],
            may_satisfy_approval_gate=True,
            readers=["monthly_candidate_pipeline", "monthly_evidence_verifier"],
            writers=["monthly_candidate_pipeline"],
        ),
        ArtifactRegistryEntry(
            artifact_type="monthly_evidence_verification",
            authority=ArtifactAuthority.APPROVAL_GATE,
            owner_package="trading_assistant",
            schema_ref="trading_assistant.schemas.monthly_evidence_verification.MonthlyEvidenceVerification",
            path_patterns=["monthly_evidence_verification_*.json"],
            may_satisfy_approval_gate=True,
            readers=["approval_tracker", "monthly_candidate_pipeline"],
            writers=["monthly_evidence_verifier"],
            notes="Independent read-only verifier output required before approval-ready monthly routing.",
        ),
        ArtifactRegistryEntry(
            artifact_type="monthly_model_review",
            authority=ArtifactAuthority.GENERATED,
            owner_package="trading_assistant",
            schema_ref="trading_assistant.schemas.monthly_model_review.MonthlyModelReview",
            path_patterns=["model_review.json"],
            may_satisfy_approval_gate=False,
            readers=["monthly_candidate_pipeline", "monthly_evidence_verifier"],
            writers=["monthly_model_review_runner"],
        ),
        ArtifactRegistryEntry(
            artifact_type="model_review_request",
            authority=ArtifactAuthority.GENERATED,
            owner_package="trading_assistant",
            path_patterns=["model_review_request.json"],
            may_satisfy_approval_gate=False,
            readers=["monthly_model_review_runner", "monthly_evidence_verifier"],
            writers=["monthly_model_review_runner"],
            notes="Frozen model-review prompt input; audit context only, not approval proof.",
        ),
        ArtifactRegistryEntry(
            artifact_type="model_review_prompt",
            authority=ArtifactAuthority.GENERATED,
            owner_package="trading_assistant",
            path_patterns=["model_review_prompt.md"],
            may_satisfy_approval_gate=False,
            readers=["monthly_model_review_runner", "monthly_evidence_verifier"],
            writers=["monthly_model_review_runner"],
            notes="Prompt rendering for audit context only; not approval proof.",
        ),
    ]
    for entry in manual_entries:
        entries[entry.artifact_type] = entry
    entries["runner_observability"] = entries["runner_observability"].model_copy(
        update={
            "authority": ArtifactAuthority.DIAGNOSTICS_ONLY,
            "may_satisfy_approval_gate": False,
            "notes": "Diagnostics only; cannot satisfy approval gates.",
        }
    )
    return entries
