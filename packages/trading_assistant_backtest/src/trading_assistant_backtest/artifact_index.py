"""Artifact index construction."""

from __future__ import annotations

from pathlib import Path

from trading_assistant_backtest.contract_models import (
    OPTIONAL_BACKTEST_ARTIFACTS,
    REQUIRED_BACKTEST_ARTIFACTS,
    BacktestArtifactIndex,
    MonthlyRunManifest,
)


def build_artifact_index(
    manifest: MonthlyRunManifest, artifact_root: Path
) -> BacktestArtifactIndex:
    artifacts: dict[str, str] = {}
    for name in [*REQUIRED_BACKTEST_ARTIFACTS, *OPTIONAL_BACKTEST_ARTIFACTS]:
        path = artifact_root / name
        if name in REQUIRED_BACKTEST_ARTIFACTS or path.exists():
            artifacts[name] = str(path)
    return BacktestArtifactIndex(
        run_id=manifest.run_id,
        manifest_id=manifest.manifest_id if manifest.optimizer_mode else "",
        artifact_root=str(artifact_root),
        artifacts=artifacts,
    )
