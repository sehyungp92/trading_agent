"""Subprocess client for the sibling full-fidelity backtest repo."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from trading_assistant.orchestrator.backtest_invocation import (
    backtest_repo_commit_sha,
    build_backtest_command,
    validate_backtest_repo,
)
from trading_assistant.schemas.backtest_artifacts import (
    BacktestArtifactIndex,
    BacktestExitStatus,
    missing_required_artifact_keys,
)
from trading_assistant.schemas.data_bundle_manifest import DataBundleManifest, DataBundleStatus
from trading_assistant.schemas.monthly_run_manifest import MonthlyRunManifest, MonthlyRunMode


@dataclass
class BacktestRunnerResult:
    success: bool
    artifact_index: BacktestArtifactIndex | None = None
    commit_sha: str = ""
    stdout: str = ""
    stderr: str = ""
    exit_status: BacktestExitStatus = field(default_factory=BacktestExitStatus)
    error: str = ""


class BacktestRunnerClient:
    """Runs a manifest-backed backtest command and validates artifacts."""

    def __init__(self, timeout_seconds: int = 3600) -> None:
        self.timeout_seconds = timeout_seconds

    def run(self, manifest: MonthlyRunManifest, manifest_path: Path) -> BacktestRunnerResult:
        ok, reason = validate_backtest_repo(manifest.backtest_repo_path)
        if not ok:
            return BacktestRunnerResult(success=False, error=reason)

        repo_path = Path(manifest.backtest_repo_path)
        commit_sha = backtest_repo_commit_sha(repo_path)
        command = build_backtest_command(manifest, manifest_path)
        started_at = datetime.now(timezone.utc)
        try:
            completed = subprocess.run(
                command,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            return BacktestRunnerResult(
                success=False,
                commit_sha=commit_sha,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                exit_status=BacktestExitStatus(
                    exit_code=-1,
                    timed_out=True,
                    started_at=started_at,
                    finished_at=datetime.now(timezone.utc),
                    error="backtest command timed out",
                ),
                error="backtest command timed out",
            )

        exit_status = BacktestExitStatus(
            exit_code=completed.returncode,
            timed_out=timed_out,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )
        artifact_root = Path(manifest.artifact_root)
        artifact_root.mkdir(parents=True, exist_ok=True)
        (artifact_root / "stdout.log").write_text(completed.stdout or "", encoding="utf-8")
        (artifact_root / "stderr.log").write_text(completed.stderr or "", encoding="utf-8")
        (artifact_root / "exit_status.json").write_text(
            exit_status.model_dump_json(indent=2),
            encoding="utf-8",
        )

        if completed.returncode != 0:
            return BacktestRunnerResult(
                success=False,
                commit_sha=commit_sha,
                stdout=completed.stdout,
                stderr=completed.stderr,
                exit_status=exit_status,
                error=f"backtest command failed with exit code {completed.returncode}",
            )

        index_path = artifact_root / "artifact_index.json"
        if not index_path.exists():
            return BacktestRunnerResult(
                success=False,
                commit_sha=commit_sha,
                stdout=completed.stdout,
                stderr=completed.stderr,
                exit_status=exit_status,
                error="missing artifact_index.json",
            )
        try:
            raw = json.loads(index_path.read_text(encoding="utf-8"))
            missing_keys = missing_required_artifact_keys(raw)
            if missing_keys:
                raise ValueError(f"artifact index missing required keys: {', '.join(missing_keys)}")
            index = BacktestArtifactIndex.model_validate(raw)
        except Exception as exc:
            return BacktestRunnerResult(
                success=False,
                commit_sha=commit_sha,
                stdout=completed.stdout,
                stderr=completed.stderr,
                exit_status=exit_status,
                error=f"malformed artifact_index.json: {exc}",
            )

        validation_errors = self.validate_artifact_contract(
            manifest,
            index,
            manifest_path=manifest_path,
        )
        if validation_errors:
            return BacktestRunnerResult(
                success=False,
                artifact_index=index,
                commit_sha=commit_sha,
                stdout=completed.stdout,
                stderr=completed.stderr,
                exit_status=exit_status,
                error="; ".join(validation_errors),
            )
        return BacktestRunnerResult(
            success=True,
            artifact_index=index,
            commit_sha=commit_sha,
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_status=exit_status,
        )

    @staticmethod
    def validate_artifact_contract(
        manifest: MonthlyRunManifest,
        index: BacktestArtifactIndex,
        *,
        manifest_path: Path,
    ) -> list[str]:
        return [
            *index.validation_errors(
                expected_run_id=manifest.run_id,
                expected_manifest_id=manifest.manifest_id,
                require_manifest_id=_requires_manifest_id(manifest),
            ),
            *BacktestRunnerClient._artifact_linkage_errors(
                index,
                manifest_path,
                Path(manifest.artifact_root),
            ),
            *BacktestRunnerClient._data_bundle_errors(index, manifest),
        ]

    @staticmethod
    def _artifact_linkage_errors(
        index: BacktestArtifactIndex,
        manifest_path: Path,
        artifact_root: Path,
    ) -> list[str]:
        errors: list[str] = []
        try:
            if Path(index.artifact_root).resolve() != artifact_root.resolve():
                errors.append("artifact index artifact_root does not match run manifest")
        except Exception:
            errors.append("artifact index artifact_root is not resolvable")

        exit_status_path = index.artifact_path("exit_status.json")
        if exit_status_path is not None and exit_status_path.exists():
            try:
                BacktestExitStatus.model_validate(json.loads(exit_status_path.read_text(encoding="utf-8")))
            except Exception as exc:
                errors.append(f"malformed exit_status.json: {exc}")

        try:
            manifest_mtime = manifest_path.stat().st_mtime
        except OSError:
            manifest_mtime = 0.0
        stale: list[str] = []
        for name, raw_path in index.artifacts.items():
            path = index.artifact_path(name)
            if path is None or not path.exists() or raw_path == "":
                continue
            try:
                if path.stat().st_mtime < manifest_mtime - 1.0:
                    stale.append(name)
            except OSError:
                continue
        if stale:
            errors.append(f"stale artifacts older than run manifest: {', '.join(sorted(stale))}")
        return errors

    @staticmethod
    def _data_bundle_errors(
        index: BacktestArtifactIndex,
        manifest: MonthlyRunManifest,
    ) -> list[str]:
        if not _requires_manifest_id(manifest):
            return []
        errors: list[str] = []
        bundle_path = Path(manifest.data_bundle_manifest_path or manifest.market_data_manifest_path)
        if not bundle_path.exists():
            return [f"data bundle manifest missing: {bundle_path}"]
        try:
            bundle = DataBundleManifest.model_validate(
                json.loads(bundle_path.read_text(encoding="utf-8"))
            )
        except Exception as exc:
            return [f"malformed data bundle manifest: {exc}"]
        if bundle.status != DataBundleStatus.AUTHORITATIVE:
            errors.append(f"data bundle is not authoritative: {bundle.status.value}")
        expected_checksum = manifest.data_bundle_checksum or manifest.data_manifest_checksum
        if expected_checksum and bundle.bundle_checksum != expected_checksum:
            errors.append("data bundle checksum does not match run manifest")

        emitted_path = index.artifact_path("coverage_manifest.json")
        if emitted_path is None or not emitted_path.exists():
            return errors
        try:
            emitted = json.loads(emitted_path.read_text(encoding="utf-8"))
        except Exception:
            return errors
        emitted_checksum = _find_checksum(emitted)
        if not emitted_checksum:
            errors.append("coverage_manifest.json must include data bundle checksum for optimizer runs")
        elif emitted_checksum != bundle.bundle_checksum:
            errors.append("coverage_manifest.json data bundle checksum does not match run manifest")
        return errors


def _requires_manifest_id(manifest: MonthlyRunManifest) -> bool:
    return manifest.mode in {
        MonthlyRunMode.PHASED_AUTO,
        MonthlyRunMode.SMOKE_REPAIR,
        MonthlyRunMode.STRUCTURAL_REVIEW,
    }


def _find_checksum(value: object) -> str:
    if isinstance(value, dict):
        for key in ("bundle_checksum", "data_bundle_checksum", "checksum"):
            raw = value.get(key)
            if raw:
                return str(raw)
        for nested in value.values():
            found = _find_checksum(nested)
            if found:
                return found
    if isinstance(value, list):
        for item in value:
            found = _find_checksum(item)
            if found:
                return found
    return ""
