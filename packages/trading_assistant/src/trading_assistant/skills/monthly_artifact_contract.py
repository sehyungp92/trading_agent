"""Deep monthly artifact contract for approval-facing evidence.

This module is the control-plane adapter over the backtest artifact index. It
intentionally composes ``BacktestArtifactIndex`` and ``ArtifactAuthorityRegistry``
instead of owning artifact names or authority seed data.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from trading_assistant.schemas.artifact_authority import (
    ArtifactRegistryEntry,
    artifact_type_from_path,
)
from trading_assistant.schemas.backtest_artifacts import (
    OPTIONAL_BACKTEST_ARTIFACTS,
    REQUIRED_BACKTEST_ARTIFACTS,
    BacktestArtifactIndex,
)
from trading_assistant.schemas.loop_contracts import LoopContract
from trading_assistant.schemas.monthly_artifact_contract import (
    MonthlyApprovalEvidenceView,
    MonthlyArtifactIssue,
    MonthlyArtifactStatus,
    MonthlyArtifactView,
    MonthlyVerifierInput,
)
from trading_assistant.schemas.monthly_candidates import (
    MonthlyApprovalEvidencePacket,
    MonthlyCandidateGateReport,
    MonthlyCandidateSource,
    MonthlyImprovementCandidate,
)
from trading_assistant.schemas.monthly_model_review import (
    MonthlyModelReview,
    MonthlyModelValidationResult,
)
from trading_assistant.schemas.monthly_run_manifest import MonthlyRunManifest
from trading_assistant.schemas.monthly_validation import MonthlyValidationResult
from trading_assistant.skills.artifact_authority_registry import ArtifactAuthorityRegistry


ModelT = TypeVar("ModelT", bound=BaseModel)


class MonthlyArtifactContract:
    """Contract-level facade over monthly artifact paths, scope, and authority."""

    def __init__(
        self,
        *,
        manifest: MonthlyRunManifest | None = None,
        artifact_index: BacktestArtifactIndex | None = None,
        artifact_root: Path | str,
        registry: ArtifactAuthorityRegistry | None = None,
    ) -> None:
        self.manifest = manifest
        self.artifact_index = artifact_index
        self.artifact_root = Path(artifact_root)
        self._registry = registry

    @property
    def registry(self) -> ArtifactAuthorityRegistry:
        if self._registry is None:
            self._registry = ArtifactAuthorityRegistry.load()
        return self._registry

    @classmethod
    def from_run(
        cls,
        *,
        manifest: MonthlyRunManifest,
        artifact_index: BacktestArtifactIndex | None,
        artifact_root: Path,
        registry: ArtifactAuthorityRegistry | None = None,
    ) -> "MonthlyArtifactContract":
        return cls(
            manifest=manifest,
            artifact_index=artifact_index,
            artifact_root=artifact_root,
            registry=registry,
        )

    @classmethod
    def from_index(
        cls,
        artifact_index: BacktestArtifactIndex,
        *,
        manifest: MonthlyRunManifest | None = None,
        registry: ArtifactAuthorityRegistry | None = None,
    ) -> "MonthlyArtifactContract":
        return cls(
            manifest=manifest,
            artifact_index=artifact_index,
            artifact_root=Path(artifact_index.artifact_root),
            registry=registry,
        )

    def path(self, artifact_name: str) -> Path | None:
        if self.artifact_index is not None:
            return self.artifact_index.artifact_path(artifact_name)
        raw = str(artifact_name or "").strip()
        if not raw:
            return None
        path = Path(raw)
        return path if path.is_absolute() else self.artifact_root / path

    def path_str(self, artifact_name: str, *, require_exists: bool = True) -> str:
        path = self.path(artifact_name)
        if path is None:
            return ""
        if require_exists and not path.exists():
            return ""
        return str(path)

    def load_json(self, artifact_name: str, errors: list[str] | None = None) -> Any:
        path = self.path(artifact_name)
        if path is None or not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            if errors is not None:
                errors.append(f"invalid {artifact_name}: {exc}")
            return None

    def load_json_object(self, artifact_name: str, errors: list[str] | None = None) -> dict[str, Any]:
        path = self.path(artifact_name)
        if path is None or not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            if errors is not None:
                errors.append(f"invalid {artifact_name}: {exc}")
            return {}
        if not isinstance(payload, dict):
            if errors is not None:
                errors.append(f"invalid {artifact_name}: expected JSON object")
            return {}
        return payload

    def load_jsonl(self, artifact_name: str, errors: list[str] | None = None) -> list[dict[str, Any]]:
        path = self.path(artifact_name)
        if path is None or not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
        except Exception as exc:
            if errors is not None:
                errors.append(f"invalid {artifact_name}: {exc}")
        return rows

    def load_model(
        self,
        artifact_name: str,
        model_type: type[ModelT],
        errors: list[str] | None = None,
    ) -> ModelT | None:
        path = self.path(artifact_name)
        if path is None or not path.exists():
            return None
        try:
            return model_type.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            if errors is not None:
                errors.append(f"invalid {artifact_name}: {exc}")
            return None

    def validate_scope(self) -> list[MonthlyArtifactIssue]:
        if self.artifact_index is None:
            return []
        issues: list[MonthlyArtifactIssue] = []
        expected_run_id = self.manifest.run_id if self.manifest is not None else ""
        expected_manifest_id = self.manifest.manifest_id if self.manifest is not None else ""
        if expected_run_id and self.artifact_index.run_id != expected_run_id:
            issues.append(MonthlyArtifactIssue(
                code="artifact_index_run_id_mismatch",
                message=(
                    f"artifact index run_id mismatch: {self.artifact_index.run_id!r} "
                    f"!= {expected_run_id!r}"
                ),
                artifact_name="artifact_index",
                path=str(self.artifact_root / "artifact_index.json"),
                status=MonthlyArtifactStatus.SCOPE_MISMATCH,
            ))
        if expected_manifest_id:
            if not self.artifact_index.manifest_id:
                issues.append(MonthlyArtifactIssue(
                    code="artifact_index_manifest_id_missing",
                    message="artifact index manifest_id is required for optimizer runs",
                    artifact_name="artifact_index",
                    path=str(self.artifact_root / "artifact_index.json"),
                    status=MonthlyArtifactStatus.SCOPE_MISMATCH,
                ))
            elif self.artifact_index.manifest_id != expected_manifest_id:
                issues.append(MonthlyArtifactIssue(
                    code="artifact_index_manifest_id_mismatch",
                    message=(
                        "artifact index manifest_id mismatch: "
                        f"{self.artifact_index.manifest_id!r} != {expected_manifest_id!r}"
                    ),
                    artifact_name="artifact_index",
                    path=str(self.artifact_root / "artifact_index.json"),
                    status=MonthlyArtifactStatus.SCOPE_MISMATCH,
                ))
        return issues

    def validate_containment(self) -> list[MonthlyArtifactIssue]:
        if self.artifact_index is None:
            return []
        return [
            MonthlyArtifactIssue(
                code="artifact_path_outside_root",
                message=f"artifact path is outside artifact_root: {name}",
                artifact_name=name,
                path=self.path_str(name, require_exists=False),
                status=MonthlyArtifactStatus.OUTSIDE_ROOT,
            )
            for name in self.artifact_index.paths_outside_root()
        ]

    def validate_required(self) -> list[MonthlyArtifactIssue]:
        if self.artifact_index is None:
            return []
        issues = [
            MonthlyArtifactIssue(
                code="missing_required_artifact",
                message=f"missing required artifact: {name}",
                artifact_name=name,
                path=self.path_str(name, require_exists=False),
                status=MonthlyArtifactStatus.MISSING_REQUIRED,
            )
            for name in self.artifact_index.missing_required()
        ]
        issues.extend(
            MonthlyArtifactIssue(
                code="malformed_artifact",
                message=f"malformed required artifact: {name}",
                artifact_name=name,
                path=self.path_str(name, require_exists=False),
                status=MonthlyArtifactStatus.MALFORMED,
            )
            for name in self.artifact_index.malformed_required()
        )
        return issues

    def issues(self) -> list[MonthlyArtifactIssue]:
        return [
            *self.validate_scope(),
            *self.validate_required(),
            *self.validate_containment(),
        ]

    def validation_errors(self) -> list[str]:
        if self.artifact_index is None:
            return []
        expected_run_id = self.manifest.run_id if self.manifest is not None else ""
        expected_manifest_id = self.manifest.manifest_id if self.manifest is not None else ""
        return self.artifact_index.validation_errors(
            expected_run_id=expected_run_id,
            expected_manifest_id=expected_manifest_id,
            require_manifest_id=self.manifest is not None,
        )

    def view(self, artifact_name: str) -> MonthlyArtifactView:
        path = self.path(artifact_name)
        entry = self.authority_for(artifact_name)
        return MonthlyArtifactView(
            name=artifact_name,
            path=str(path) if path is not None else "",
            exists=bool(path and path.exists()),
            required=artifact_name in REQUIRED_BACKTEST_ARTIFACTS,
            optional=artifact_name in OPTIONAL_BACKTEST_ARTIFACTS,
            artifact_type=artifact_type_from_path(artifact_name),
            authority=entry.authority if entry is not None else None,
            may_satisfy_approval_gate=bool(entry and entry.may_satisfy_approval_gate),
        )

    def artifact_views(self) -> list[MonthlyArtifactView]:
        return [
            self.view(name)
            for name in [*REQUIRED_BACKTEST_ARTIFACTS, *OPTIONAL_BACKTEST_ARTIFACTS]
        ]

    def authority_for(self, artifact_name_or_path: str) -> ArtifactRegistryEntry | None:
        return self.registry.get(artifact_name_or_path)

    def missing_named_artifacts(self, names: Iterable[str]) -> list[str]:
        return [
            name for name in names
            if not self.path_str(name)
        ]

    def existing_paths(self, paths: Iterable[str]) -> list[str]:
        return [path for path in paths if path and Path(path).exists()]

    def paths_outside_root(self, paths: Iterable[str]) -> list[str]:
        try:
            root = self.artifact_root.resolve()
        except OSError:
            return [str(raw) for raw in paths if raw]
        outside: list[str] = []
        for raw in paths:
            if not raw:
                continue
            try:
                Path(raw).resolve().relative_to(root)
            except (OSError, ValueError):
                outside.append(str(raw))
        return outside

    def load_selected_candidates(
        self,
        *,
        bot_id: str = "",
        strategy_id: str = "",
    ) -> list[MonthlyImprovementCandidate]:
        default_source = self.source_from_mode_decision()
        rows = self.load_candidate_rows("selected_candidates.json")
        objective_context = _candidate_objective_context(
            self.load_json("objective_breakdown.json")
        )
        return [
            MonthlyImprovementCandidate.from_raw(
                {**objective_context, **row},
                bot_id=bot_id,
                strategy_id=strategy_id,
                default_source=default_source,
            )
            for row in rows
        ]

    def normalize_candidate_paths(self, candidate: MonthlyImprovementCandidate) -> MonthlyImprovementCandidate:
        root = self.artifact_root
        candidate.evidence_paths = _dedupe([
            _resolve_artifact_path(path, root)
            for path in candidate.evidence_paths
        ])
        candidate.artifact_paths = _dedupe([
            _resolve_artifact_path(path, root)
            for path in candidate.artifact_paths
        ])
        candidate.candidate_workspace_path = _resolve_artifact_path(
            candidate.candidate_workspace_path,
            root,
        )
        for attr in (
            "workflow_contract_path",
            "live_repo_patch_path",
            "backtest_adapter_patch_path",
            "config_patch_path",
            "decision_parity_report_path",
            "fold_manifest_path",
            "rounds_manifest_path",
            "end_of_round_diagnostics_path",
            "confirmatory_rerank_path",
            "checkpoint_path",
        ):
            resolved = _resolve_artifact_path(getattr(candidate, attr), root)
            setattr(candidate, attr, resolved)
            if resolved:
                candidate.artifact_paths.append(resolved)
        candidate.artifact_paths = _dedupe(candidate.artifact_paths)
        candidate.deterministic_gate_inputs.setdefault("artifact_root", str(root))
        return candidate

    def load_rejected_candidates(self) -> list[dict[str, Any]]:
        return self.load_jsonl("rejected_candidates.jsonl")

    def load_candidate_rows(
        self,
        artifact_name: str,
        errors: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        path = self.path(artifact_name)
        if path is None or not path.exists():
            return []
        raw = self.load_json(artifact_name, errors)
        if raw is None:
            return []
        rows = _candidate_rows(raw)
        if rows is None:
            if errors is not None:
                errors.append(f"invalid {artifact_name}: expected list of candidates")
            return []
        return rows

    def source_from_mode_decision(self) -> MonthlyCandidateSource:
        data = self.load_json("mode_decision.json")
        if not isinstance(data, dict):
            return MonthlyCandidateSource.UNKNOWN
        raw = (
            data.get("candidate_source")
            or data.get("mode")
            or data.get("routing")
            or data.get("decision")
            or data.get("status")
        )
        if not raw:
            return MonthlyCandidateSource.UNKNOWN
        normalized = str(raw).strip().lower().replace("-", "_")
        if normalized in {"smoke", "smoke_repair", "repair", "rollback"}:
            return MonthlyCandidateSource.SMOKE_REPAIR
        if normalized in {"phased", "phased_auto", "auto", "experiment"}:
            return MonthlyCandidateSource.PHASED_AUTO
        return MonthlyCandidateSource.UNKNOWN

    def approval_packet_artifact_paths(
        self,
        *,
        monthly_result: MonthlyValidationResult,
        candidate: MonthlyImprovementCandidate,
        monthly_result_path: Path | str = "",
        candidate_gate_report_path: str = "",
        model_review_path: str = "",
        model_review_validation_path: str = "",
    ) -> list[str]:
        return _dedupe([
            *monthly_result.evidence_paths,
            str(monthly_result_path) if monthly_result_path else "",
            monthly_result.artifact_index_path,
            monthly_result.replay_parity_path,
            candidate_gate_report_path,
            model_review_path,
            model_review_validation_path,
            *candidate.evidence_paths,
            *candidate.artifact_paths,
        ])

    def approval_gate_evidence(
        self,
        candidate_id: str,
        *,
        candidate: MonthlyImprovementCandidate | None = None,
        monthly_result_path: Path | str = "",
        replay_parity_path: str = "",
        candidate_gate_report_path: str = "",
        model_review_validation_path: str = "",
        extra_paths: Iterable[str] = (),
    ) -> MonthlyApprovalEvidenceView:
        gate_paths = [
            str(monthly_result_path) if monthly_result_path else "",
            replay_parity_path,
            candidate_gate_report_path,
            model_review_validation_path,
            *list(extra_paths),
        ]
        if candidate is not None:
            gate_paths.extend(candidate.evidence_paths)
        selected = self.load_selected_candidates()
        rejected = self.load_rejected_candidates()
        selected_path = self.path_str("selected_candidates.json")
        rejected_path = self.path_str("rejected_candidates.jsonl")
        return MonthlyApprovalEvidenceView(
            candidate_id=candidate_id,
            artifact_paths=_dedupe([
                selected_path,
                rejected_path,
                *gate_paths,
                *([] if candidate is None else candidate.artifact_paths),
            ]),
            approval_gate_evidence=_dedupe([
                path for path in gate_paths
                if path and self.registry.may_satisfy_approval_gate(path)
            ]),
            selected_candidate_count=len(selected),
            rejected_candidate_count=len(rejected),
            selected_candidate_path=selected_path,
            rejected_candidates_path=rejected_path,
        )

    def verifier_input(
        self,
        candidate_id: str,
        *,
        monthly_result: MonthlyValidationResult,
        selected_candidates: list[MonthlyImprovementCandidate] | None = None,
        gate_reports: list[MonthlyCandidateGateReport] | None = None,
        approval_packet: MonthlyApprovalEvidencePacket | None = None,
        run_manifest: MonthlyRunManifest | None = None,
        model_review: MonthlyModelReview | None = None,
        model_validation: MonthlyModelValidationResult | None = None,
        model_review_validation_path: str = "",
        deployment_metadata_blockers: list[str] | None = None,
        loop_contract: LoopContract | None = None,
    ) -> MonthlyVerifierInput:
        selected = selected_candidates
        if selected is None:
            selected = self.load_selected_candidates(
                bot_id=monthly_result.bot_id,
                strategy_id=monthly_result.strategy_id,
            )
        packet = approval_packet
        if packet is not None and packet.candidate_id != candidate_id:
            packet = None
        return MonthlyVerifierInput(
            monthly_result=monthly_result,
            artifact_index=self.artifact_index,
            selected_candidates=selected,
            gate_reports=gate_reports or [],
            approval_packet=packet,
            run_manifest=run_manifest or self.manifest,
            model_review=model_review,
            model_validation=model_validation,
            model_review_validation_path=model_review_validation_path,
            deployment_metadata_blockers=deployment_metadata_blockers or [],
            loop_contract=loop_contract,
        )


def _candidate_rows(raw: Any) -> list[dict[str, Any]] | None:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        candidates = (
            raw.get("candidates")
            or raw.get("selected_candidates")
            or raw.get("selected")
            or raw.get("shortlist")
            or []
        )
        return [item for item in candidates if isinstance(item, dict)]
    return None


def _candidate_objective_context(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return {
        key: value
        for key, value in {
            "objective_version": raw.get("objective_version"),
            "effective_objective_version": raw.get("effective_objective_version")
            or raw.get("immutable_objective_version"),
            "immutable_objective_version": raw.get("immutable_objective_version"),
            "objective_profile_id": raw.get("objective_profile_id"),
            "objective_profile_family": (raw.get("profile") or {}).get("family")
            if isinstance(raw.get("profile"), dict) else "",
            "objective_profile_scope": (raw.get("profile") or {}).get("scope")
            if isinstance(raw.get("profile"), dict) else "",
            "score_component_cap": raw.get("score_component_cap"),
        }.items()
        if value not in (None, "")
    }


def _resolve_artifact_path(path: str, root: Path) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    return str(candidate)


def _dedupe(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out
