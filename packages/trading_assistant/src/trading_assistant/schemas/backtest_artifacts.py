"""Backtest artifact contract schemas."""
from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


REQUIRED_BACKTEST_ARTIFACTS = [
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

PHASE4_OPTIMIZER_ARTIFACTS = [
    "optimizer_run_manifest.json",
    "leakage_report.json",
    "cost_sensitivity.json",
    "fold_validation.json",
    "outlier_sensitivity.json",
    "portfolio_synergy.json",
    "fold_manifest.json",
    "rounds_manifest.json",
    "end_of_round_diagnostics.json",
    "llm_experiment_plan.json",
    "candidate_workspace_manifest.json",
    "candidate_attempts.jsonl",
    "runner_observability.json",
    "confirmatory_rerank.json",
    "fold_candidate_results.jsonl",
    "fold_score_matrix.json",
    "selection_oos_evaluation.json",
    "selection_oos_repair_trigger.json",
    "repair_failure_attribution.json",
    "accepted_mutation_chain.json",
    "repair_candidate_results.jsonl",
    "repair_checkpoint.json",
    "round_n_plus_1_recommendation.json",
]

PHASE4_OOS_REPAIR_ARTIFACTS = [
    "repair_ablation_matrix.jsonl",
]

PHASE4_STRUCTURAL_CANDIDATE_ARTIFACTS = [
    "structural_candidate_plan.json",
    "structural_selection_gate.json",
    "live_repo_patch.diff",
    "backtest_adapter_patch.diff",
    "config_patch.diff",
    "decision_parity_report.json",
]

LEARNING_INSTRUMENTATION_ARTIFACTS = [
    "learning_sufficiency_manifest.json",
    "strategy_discovery_packet.json",
]

OPTIONAL_BACKTEST_ARTIFACTS: list[str] = [
    *PHASE4_OPTIMIZER_ARTIFACTS,
    *PHASE4_OOS_REPAIR_ARTIFACTS,
    *PHASE4_STRUCTURAL_CANDIDATE_ARTIFACTS,
    *LEARNING_INSTRUMENTATION_ARTIFACTS,
]

JSON_ARTIFACTS = {
    name for name in [*REQUIRED_BACKTEST_ARTIFACTS, *OPTIONAL_BACKTEST_ARTIFACTS]
    if name.endswith(".json")
}
JSONL_ARTIFACTS = {
    name for name in [*REQUIRED_BACKTEST_ARTIFACTS, *OPTIONAL_BACKTEST_ARTIFACTS]
    if name.endswith(".jsonl")
}


class ArtifactStatus(str, Enum):
    COMPLETE = "complete"
    MISSING_REQUIRED = "missing_required"
    MALFORMED = "malformed"


class BacktestArtifactIndex(BaseModel):
    """Machine-readable index returned by the backtest repo."""

    run_id: str
    manifest_id: str = ""
    artifact_root: str
    artifacts: dict[str, str] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    index_version: str = "backtest_artifact_index_v1"

    @model_validator(mode="after")
    def _fill_default_paths(self) -> BacktestArtifactIndex:
        for name in REQUIRED_BACKTEST_ARTIFACTS + OPTIONAL_BACKTEST_ARTIFACTS:
            self.artifacts.setdefault(name, str(Path(self.artifact_root) / name))
        return self

    def artifact_path(self, name: str) -> Path | None:
        raw = str(self.artifacts.get(name, "") or "").strip()
        if not raw:
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = Path(self.artifact_root) / path
        return path

    def missing_required(self) -> list[str]:
        missing: list[str] = []
        for name in REQUIRED_BACKTEST_ARTIFACTS:
            path = self.artifact_path(name)
            if path is None or not path.exists():
                missing.append(name)
        return missing

    def paths_outside_root(self) -> list[str]:
        root = Path(self.artifact_root).resolve()
        outside: list[str] = []
        for name in [*REQUIRED_BACKTEST_ARTIFACTS, *self.present_optional_artifacts()]:
            path = self.artifact_path(name)
            if path is None:
                continue
            try:
                path.resolve().relative_to(root)
            except (OSError, ValueError):
                outside.append(name)
        return outside

    def malformed_required(self) -> list[str]:
        malformed: list[str] = []
        for name in [*REQUIRED_BACKTEST_ARTIFACTS, *self.present_optional_artifacts()]:
            path = self.artifact_path(name)
            if path is None or not path.exists():
                continue
            if name in JSON_ARTIFACTS:
                try:
                    json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    malformed.append(name)
            elif name in JSONL_ARTIFACTS:
                try:
                    for line in path.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            json.loads(line)
                except Exception:
                    malformed.append(name)
        return malformed

    def present_optional_artifacts(self) -> list[str]:
        present: list[str] = []
        for name in OPTIONAL_BACKTEST_ARTIFACTS:
            path = self.artifact_path(name)
            if path is not None and path.exists():
                present.append(name)
        return present

    def validation_errors(
        self,
        *,
        expected_run_id: str = "",
        expected_manifest_id: str = "",
        require_manifest_id: bool = False,
    ) -> list[str]:
        errors: list[str] = []
        if expected_run_id and self.run_id != expected_run_id:
            errors.append(f"artifact index run_id mismatch: {self.run_id!r} != {expected_run_id!r}")
        if expected_manifest_id:
            if require_manifest_id and not self.manifest_id:
                errors.append("artifact index manifest_id is required for optimizer runs")
            elif self.manifest_id and self.manifest_id != expected_manifest_id:
                errors.append(
                    f"artifact index manifest_id mismatch: {self.manifest_id!r} != {expected_manifest_id!r}"
                )
        missing = self.missing_required()
        if missing:
            errors.append(f"missing required artifacts: {', '.join(missing)}")
        outside = self.paths_outside_root()
        if outside:
            errors.append(f"artifact paths outside artifact_root: {', '.join(outside)}")
        malformed = self.malformed_required()
        if malformed:
            errors.append(f"malformed required artifacts: {', '.join(malformed)}")
        return errors

    def status(self) -> ArtifactStatus:
        if self.missing_required():
            return ArtifactStatus.MISSING_REQUIRED
        if self.paths_outside_root() or self.malformed_required():
            return ArtifactStatus.MALFORMED
        return ArtifactStatus.COMPLETE


class BacktestExitStatus(BaseModel):
    exit_code: int = 0
    timed_out: bool = False
    error: str = ""
    started_at: datetime | None = None
    finished_at: datetime | None = None


def missing_required_artifact_keys(raw_index: Mapping[str, object]) -> list[str]:
    artifacts = raw_index.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return list(REQUIRED_BACKTEST_ARTIFACTS)
    return [
        name for name in REQUIRED_BACKTEST_ARTIFACTS
        if not str(artifacts.get(name, "") or "").strip()
    ]
