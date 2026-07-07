"""Local contract validation helpers.

This is intentionally lighter than the upstream `trading_assistant` validator; it catches
runner-owned artifact failures before the control plane performs the final check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from trading_assistant_backtest.contract_models import BacktestArtifactIndex, MonthlyRunManifest
from trading_assistant_backtest.contract_models import DecisionParityReport
from trading_assistant_backtest.contract_models import DecisionParityStatus
from trading_assistant_backtest.contract_models import MonthlyRunMode
from trading_assistant_backtest.manifest_loader import load_manifest
from trading_assistant_backtest.paths import monorepo_root, normalize_workspace_path


@dataclass(frozen=True)
class LocalContractValidation:
    manifest_path: str
    artifact_index_path: str
    valid: bool
    errors: list[str] = field(default_factory=list)


def validate_manifest_file(manifest_path: str | Path) -> LocalContractValidation:
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    index_path = Path(manifest.artifact_root) / "artifact_index.json"
    if not index_path.exists():
        return LocalContractValidation(
            manifest_path=str(manifest_path),
            artifact_index_path=str(index_path),
            valid=False,
            errors=["missing artifact_index.json"],
        )
    try:
        index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        index = BacktestArtifactIndex.model_validate(
            _normalize_artifact_index_paths(index_payload)
        )
    except Exception as exc:
        return LocalContractValidation(
            manifest_path=str(manifest_path),
            artifact_index_path=str(index_path),
            valid=False,
            errors=[f"malformed artifact_index.json: {exc}"],
        )
    errors = index.validation_errors(
        expected_run_id=manifest.run_id,
        expected_manifest_id=manifest.manifest_id,
        require_manifest_id=manifest.optimizer_mode,
    )
    errors.extend(_linkage_errors(manifest, index, manifest_path))
    errors.extend(_structural_selection_gate_errors(manifest, index))
    return LocalContractValidation(
        manifest_path=str(manifest_path),
        artifact_index_path=str(index_path),
        valid=not errors,
        errors=errors,
    )


def _linkage_errors(
    manifest: MonthlyRunManifest,
    index: BacktestArtifactIndex,
    manifest_path: Path,
) -> list[str]:
    errors: list[str] = []
    try:
        if Path(index.artifact_root).resolve() != Path(manifest.artifact_root).resolve():
            errors.append("artifact index artifact_root does not match run manifest")
    except Exception:
        errors.append("artifact index artifact_root is not resolvable")
    manifest_mtime = manifest_path.stat().st_mtime if manifest_path.exists() else 0.0
    stale: list[str] = []
    for name in index.artifacts:
        path = index.artifact_path(name)
        if path is not None and path.exists() and path.stat().st_mtime < manifest_mtime - 1.0:
            stale.append(name)
    if stale:
        errors.append(f"stale artifacts older than run manifest: {', '.join(sorted(stale))}")
    return errors


def _structural_selection_gate_errors(
    manifest: MonthlyRunManifest,
    index: BacktestArtifactIndex,
) -> list[str]:
    if manifest.mode != MonthlyRunMode.STRUCTURAL_REVIEW:
        return []
    gate_path = index.artifact_path("structural_selection_gate.json")
    if gate_path is None or not gate_path.exists():
        return ["structural review missing structural_selection_gate.json"]
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"malformed structural_selection_gate.json: {exc}"]
    errors: list[str] = []
    if gate.get("run_id") != manifest.run_id:
        errors.append("structural selection gate run_id does not match manifest")
    selection_allowed = bool(gate.get("selection_allowed"))
    patch_checks = gate.get("patch_checks")
    if not isinstance(patch_checks, list):
        errors.append("structural selection gate missing patch_checks")
        patch_checks = []
    if selection_allowed:
        unusable = [
            str(item.get("artifact_name") or "unknown")
            for item in patch_checks
            if not isinstance(item, dict)
            or not bool(item.get("usable_for_structural_selection"))
        ]
        if unusable:
            errors.append(
                "structural selection gate allowed selection with unusable patch artifacts: "
                + ", ".join(unusable)
            )
        parity = gate.get("decision_parity")
        if not isinstance(parity, dict):
            errors.append("structural selection gate missing decision_parity")
        else:
            if parity.get("status") != DecisionParityStatus.PASS.value:
                errors.append("structural selection gate allowed selection without passing parity")
            report_path_text = str(parity.get("report_path") or "").strip()
            if not report_path_text:
                errors.append("structural selection gate decision parity report does not exist")
            else:
                report_path = normalize_workspace_path(monorepo_root(), report_path_text)
                if not report_path.exists():
                    errors.append("structural selection gate decision parity report does not exist")
                else:
                    try:
                        report = DecisionParityReport.model_validate(
                            json.loads(report_path.read_text(encoding="utf-8"))
                        )
                    except Exception as exc:
                        errors.append(f"structural selection gate decision parity is invalid: {exc}")
                    else:
                        if not report.eligible_for_structural_approval:
                            errors.append(
                                "structural selection gate decision parity is not eligible for "
                                "approval"
                            )
    return errors


def _normalize_artifact_index_paths(payload: object) -> object:
    if not isinstance(payload, dict):
        return payload
    root = monorepo_root()
    normalized = dict(payload)
    artifact_root = normalized.get("artifact_root")
    if isinstance(artifact_root, str) and artifact_root.strip():
        normalized["artifact_root"] = str(normalize_workspace_path(root, artifact_root))
    artifacts = normalized.get("artifacts")
    if isinstance(artifacts, dict):
        normalized["artifacts"] = {
            str(name): str(normalize_workspace_path(root, path))
            for name, path in artifacts.items()
            if str(name).strip() and str(path).strip()
        }
    return normalized
