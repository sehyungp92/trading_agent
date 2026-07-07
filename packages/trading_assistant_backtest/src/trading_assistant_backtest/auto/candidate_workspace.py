"""Candidate workspace manager."""

from __future__ import annotations

from pathlib import Path

from trading_assistant_backtest.contract_models import (
    CandidateWorkspaceManifest,
    sanitize_workspace_key,
)


class CandidateWorkspaceManager:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def prepare(
        self,
        *,
        run_id: str,
        candidate_id: str,
        workspace_key: str = "",
        structural: bool = False,
    ) -> CandidateWorkspaceManifest:
        safe_key = sanitize_workspace_key(workspace_key or f"{run_id}-{candidate_id}")
        workspace_path = (self.workspace_root / safe_key).resolve()
        workspace_path.relative_to(self.workspace_root)
        workspace_path.mkdir(parents=True, exist_ok=True)
        manifest_path = workspace_path / "candidate_workspace_manifest.json"
        manifest = CandidateWorkspaceManifest(
            run_id=run_id,
            candidate_id=candidate_id,
            workspace_key=safe_key,
            workspace_root=str(self.workspace_root),
            workspace_path=str(workspace_path),
            manifest_path=str(manifest_path),
            structural=structural,
        )
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return manifest
