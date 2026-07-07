"""Backtest invocation boundary helpers."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from trading_assistant.schemas.monthly_run_manifest import MonthlyRunManifest


def validate_backtest_repo(path: str | Path) -> tuple[bool, str]:
    if str(path).strip() == "":
        return False, "BACKTEST_REPO_PATH is empty"
    repo = Path(path)
    if not repo.exists():
        return False, f"BACKTEST_REPO_PATH does not exist: {repo}"
    if not repo.is_dir():
        return False, f"BACKTEST_REPO_PATH is not a directory: {repo}"
    return True, ""


def backtest_repo_commit_sha(path: str | Path) -> str:
    repo = Path(path)
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def build_backtest_command(manifest: MonthlyRunManifest, manifest_path: Path) -> list[str]:
    if manifest.backtest_command:
        replacements = {
            "{manifest}": str(manifest_path),
            "{artifact_root}": manifest.artifact_root,
            "{mode}": manifest.mode.value,
            "{run_id}": manifest.run_id,
            "{run_month}": manifest.run_month,
            "{workflow_contract_version}": manifest.workflow_contract_version,
        }
        command: list[str] = []
        for part in manifest.backtest_command:
            rendered = part
            for needle, value in replacements.items():
                rendered = rendered.replace(needle, value)
            command.append(rendered)
        return command
    return [
        sys.executable,
        "-m",
        "backtests.shared.monthly_repair",
        "--manifest",
        str(manifest_path),
    ]
