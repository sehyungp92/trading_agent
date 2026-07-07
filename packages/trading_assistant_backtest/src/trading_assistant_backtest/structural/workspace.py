"""Structural workspace helpers."""

from __future__ import annotations

from pathlib import Path

from trading_assistant_backtest.auto.candidate_workspace import CandidateWorkspaceManager


def prepare_structural_workspace(root: Path, run_id: str, candidate_id: str):
    return CandidateWorkspaceManager(root).prepare(
        run_id=run_id,
        candidate_id=candidate_id,
        workspace_key=candidate_id,
        structural=True,
    )
